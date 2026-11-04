"""Quadrature rules for the path integral of Integrated Gradients, fixed and adaptive.

Every rule here approximates the same quantity as
:func:`evd.explain.attribution.integrated_gradients_fixed`: the straight path from a baseline
``b`` to the clip features ``x``, ``delta = x - b``. A rule is a set of points ``alpha_j`` in
``[0, 1]`` with weights ``w_j``, and the token map is

    ``sum_j w_j * sum_d delta_d * dF/dx_d (b + alpha_j * delta)``,

summed over the feature axis as the localization reads it (``exp/localization_class.integrate``).
The maps are accumulated per node, so one evaluation of the path serves any rule; the campaign
path in ``attribution.py`` is left as it is, since its float order defines records already
written. Two quantities come for free with every node and drive the adaptive rules: the logit
``F(b + alpha delta)`` and the directional derivative ``g(alpha) = sum of the node's map``,
whose integral over a panel ``[a, c]`` is exactly ``F(c) - F(a)``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

#: A point of a path: the baseline name and the position ``alpha`` in ``[0, 1]``.
Row = tuple[str, float]


def gauss_legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    """:return: points and weights of the ``n``-point Gauss-Legendre rule on ``[0, 1]``.

    :raises ValueError: if ``n`` is not positive.
    """
    if n < 1:
        raise ValueError(f"Gauss-Legendre needs at least one node, got {n}")
    points, weights = np.polynomial.legendre.leggauss(n)
    return (points + 1) / 2, weights / 2


def riemann_right(n: int) -> tuple[np.ndarray, np.ndarray]:
    """:return: the right Riemann sum of the original IG paper (Sundararajan et al. 2017, eq. 3).

    :raises ValueError: if ``n`` is not positive.
    """
    if n < 1:
        raise ValueError(f"A Riemann sum needs at least one node, got {n}")
    return np.arange(1, n + 1) / n, np.full(n, 1.0 / n)


def midpoint(n: int) -> tuple[np.ndarray, np.ndarray]:
    """:return: the composite midpoint rule with ``n`` equal cells.

    :raises ValueError: if ``n`` is not positive.
    """
    if n < 1:
        raise ValueError(f"The midpoint rule needs at least one node, got {n}")
    return (np.arange(n) + 0.5) / n, np.full(n, 1.0 / n)


def composite_gauss(panels: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    """:return: ``order``-point Gauss-Legendre on each of ``panels`` equal panels of ``[0, 1]``.

    A sharp feature costs a global rule its convergence everywhere; a composite rule confines
    it to one panel.

    :raises ValueError: if ``panels`` or ``order`` is not positive.
    """
    if panels < 1:
        raise ValueError(f"A composite rule needs at least one panel, got {panels}")
    points, weights = gauss_legendre(order)
    starts = np.arange(panels) / panels
    return (
        (starts[:, None] + points[None, :] / panels).ravel(),
        np.tile(weights / panels, panels),
    )


def guided_midpoint(
    probe_alphas: Sequence[float], probe_logits: Sequence[float], n: int, floor: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Midpoint rule on ``n`` cells of equal mass under a density that follows the logit.

    The density of each probe cell is ``(1 - floor)`` times its share of ``|Delta logit|`` plus
    ``floor`` times its share of length, so the nodes gather where the output changes along the
    path and no cell is left empty. The rule is exact for constant integrands.

    :param probe_alphas: increasing probe positions, from 0 to 1 included.
    :param probe_logits: logit at each probe position (forward only).
    :param n: number of nodes.
    :param floor: share of the density spread uniformly, in ``(0, 1]``.
    :return: points (the midpoint in ``alpha`` of each cell) and weights (the cell widths).
    :raises ValueError: on a probe grid that does not span ``[0, 1]`` increasingly, mismatching
        lengths, a non-positive ``n`` or a ``floor`` outside ``(0, 1]``.
    """
    a = np.asarray(probe_alphas, dtype=np.float64)
    logits = np.asarray(probe_logits, dtype=np.float64)
    if a.ndim != 1 or len(a) < 2 or len(a) != len(logits):
        raise ValueError("Guided rule: probes and logits must be two vectors of equal length >= 2")
    if a[0] != 0.0 or a[-1] != 1.0 or np.any(np.diff(a) <= 0):
        raise ValueError("Guided rule: probe positions must increase from 0 to 1")
    if n < 1 or not 0.0 < floor <= 1.0:
        raise ValueError(f"Guided rule: n must be positive and floor in (0, 1], got {n}, {floor}")
    change = np.abs(np.diff(logits))
    width = np.diff(a)
    mass = floor * width
    if np.isfinite(change).all() and change.sum() > 0:
        mass = mass + (1 - floor) * change / change.sum()
    else:
        mass = width.copy()
    cdf = np.concatenate([[0.0], np.cumsum(mass)])
    cdf /= cdf[-1]
    edges = np.interp(np.linspace(0.0, 1.0, n + 1), cdf, a)
    edges[0], edges[-1] = 0.0, 1.0
    return (edges[:-1] + edges[1:]) / 2, np.diff(edges)


def rescaled(token_map: np.ndarray, difference: float, eps: float = 1e-12) -> np.ndarray:
    """:return: the map scaled so that its sum equals ``difference`` (completeness enforced);
    unchanged when its sum is below ``eps`` in magnitude."""
    total = float(np.sum(token_map))
    return token_map if abs(total) < eps else token_map * (difference / total)


def path_maps(
    forward: Callable[[Tensor], Tensor],
    original: Tensor,
    baselines: Mapping[str, Tensor],
    rows: Sequence[Row],
    *,
    batch_size: int,
    gradients: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Logits, and token maps of the integrand, at points of the straight paths.

    :param forward: the head on a ``[B, N, D]`` batch, one logit per row (``[B]`` or
        ``[B, 1]``); rows must be independent. Precision is the caller's (autocast inside).
    :param original: ``[1, N, D]`` clip features.
    :param baselines: name to ``[1, N, D]`` baseline features, dtype and device of ``original``.
    :param rows: the ``(baseline, alpha)`` points, evaluated in this order.
    :param batch_size: rows per head call.
    :param gradients: ``False`` computes the logits only, under ``no_grad``.
    :return: ``(logits [R], maps [R, N] or None)`` in float64/float32 numpy; a map is
        ``sum_d delta_d * dF/dx_d`` at the point, so its sum is the directional derivative.
    :raises ValueError: on a batch dimension other than one, mismatching shapes, an unknown
        baseline or a non-positive ``batch_size``.
    """
    if original.ndim < 2 or original.shape[0] != 1:
        raise ValueError(f"IG: the original needs a batch dimension of one, got {original.shape}")
    if any(b.shape != original.shape for b in baselines.values()):
        raise ValueError("IG: every baseline must have the shape of the original")
    if batch_size < 1:
        raise ValueError(f"IG: batch_size must be positive, got {batch_size}")
    unknown = {name for name, _ in rows} - set(baselines)
    if unknown:
        raise ValueError(f"IG: unknown baselines {sorted(unknown)}")
    original = original.detach()
    deltas = {name: original - b.detach() for name, b in baselines.items()}
    logits, maps = [], []
    for begin in range(0, len(rows), batch_size):
        chunk = rows[begin : begin + batch_size]
        x = torch.cat([baselines[n].detach() + float(a) * deltas[n] for n, a in chunk])
        if gradients:
            with torch.enable_grad():
                x.requires_grad_(True)
                out = forward(x).reshape(len(x), -1)[:, 0]
                gradient = torch.autograd.grad(out.sum(), x)[0]
            delta = torch.cat([deltas[n] for n, _ in chunk])
            maps.append((gradient.float() * delta.float()).sum(-1).cpu().numpy())
        else:
            with torch.no_grad():
                out = forward(x).reshape(len(x), -1)[:, 0]
        logits.append(out.detach().double().cpu().numpy())
    if not rows:
        return np.zeros(0), (np.zeros((0, original.shape[1]), np.float32) if gradients else None)
    return np.concatenate(logits), (np.concatenate(maps) if gradients else None)


def apply_rule(alphas: np.ndarray, weights: np.ndarray, maps: np.ndarray) -> np.ndarray:
    """:return: ``sum_j weights[j] * maps[j]`` in float64, the rule's token map."""
    return np.einsum("j,jn->n", np.asarray(weights, np.float64), maps.astype(np.float64))


@dataclass
class _Panel:
    a: float
    c: float
    error: float = np.inf


def adaptive_simpson(
    evaluate: Callable[[list[Row]], tuple[np.ndarray, np.ndarray]],
    names: Sequence[str],
    budget: int,
    *,
    initial_panels: int = 4,
    per_round: int = 4,
) -> dict[str, dict]:
    """Adaptive Simpson rule per baseline, refined where completeness fails on a panel.

    Each panel ``[a, c]`` holds the Simpson rule on ``a``, its midpoint and ``c``. Its error is
    exact up to the quadrature of the scalar projection: ``|Simpson(g) - (F(c) - F(a))|``, with
    ``g`` the directional derivative and ``F`` the logit, both free at every node. Every round
    bisects, in every baseline at once, the ``per_round`` panels with the largest error that the
    budget still allows (two new nodes each, the parent's nodes kept), so the head calls stay
    batched.

    :param evaluate: ``rows -> (logits [R], maps [R, N])`` at the given points.
    :param names: baselines to integrate.
    :param budget: maximum gradient evaluations per baseline (``2 * initial_panels + 1`` at
        least).
    :param initial_panels: equal panels of the first round.
    :param per_round: bisections per baseline and round.
    :return: per baseline ``{"alphas", "weights", "map", "nodes", "panels", "rounds",
        "estimated_error"}``, the map in float64.
    :raises ValueError: if the budget does not cover the first round or a count is not positive.
    """
    if initial_panels < 1 or per_round < 1:
        raise ValueError("Adaptive Simpson: panels and bisections per round must be positive")
    if budget < 2 * initial_panels + 1:
        raise ValueError(f"Adaptive Simpson: budget {budget} below the first round")
    nodes: dict[str, dict[float, tuple[float, np.ndarray]]] = {n: {} for n in names}
    panels = {
        n: [_Panel(k / initial_panels, (k + 1) / initial_panels) for k in range(initial_panels)]
        for n in names
    }

    def run(rows: list[Row]) -> None:
        if not rows:
            return
        logits, maps = evaluate(rows)
        for (name, alpha), logit, token_map in zip(rows, logits, maps, strict=True):
            nodes[name][alpha] = (float(logit), token_map)

    def score(name: str, panel: _Panel) -> None:
        m = (panel.a + panel.c) / 2
        g = {x: float(nodes[name][x][1].sum()) for x in (panel.a, m, panel.c)}
        simpson = (panel.c - panel.a) / 6 * (g[panel.a] + 4 * g[m] + g[panel.c])
        panel.error = abs(simpson - (nodes[name][panel.c][0] - nodes[name][panel.a][0]))

    first = sorted({k / (2 * initial_panels) for k in range(2 * initial_panels + 1)})
    run([(n, a) for n in names for a in first])
    rounds = 1
    for n in names:
        for p in panels[n]:
            score(n, p)
    while True:
        rows, split = [], {}
        for n in names:
            room = (budget - len(nodes[n])) // 2
            chosen = sorted(panels[n], key=lambda p: -p.error)[: min(per_round, room)]
            split[n] = chosen
            for p in chosen:
                m = (p.a + p.c) / 2
                rows += [(n, (p.a + m) / 2), (n, (m + p.c) / 2)]
        if not rows:
            break
        run(rows)
        rounds += 1
        for n in names:
            for p in split[n]:
                m = (p.a + p.c) / 2
                panels[n].remove(p)
                for child in (_Panel(p.a, m), _Panel(m, p.c)):
                    score(n, child)
                    panels[n].append(child)
    out = {}
    for n in names:
        weight: dict[float, float] = {}
        for p in panels[n]:
            m, h = (p.a + p.c) / 2, p.c - p.a
            for x, w in ((p.a, h / 6), (m, 4 * h / 6), (p.c, h / 6)):
                weight[x] = weight.get(x, 0.0) + w
        alphas = np.array(sorted(weight))
        weights = np.array([weight[x] for x in alphas])
        maps = np.stack([nodes[n][x][1] for x in alphas])
        out[n] = {
            "alphas": alphas,
            "weights": weights,
            "map": apply_rule(alphas, weights, maps),
            "nodes": len(nodes[n]),
            "panels": len(panels[n]),
            "rounds": rounds,
            "estimated_error": float(sum(p.error for p in panels[n])),
        }
    return out
