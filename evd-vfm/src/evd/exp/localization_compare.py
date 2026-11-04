"""Boxes of the localization comparison methods, as material of the class-routing AP stage.

The methods of other paradigms, evaluated with the official AP of
``scripts/localization_class_ap.py`` on the same detector and chunks as ours:

- **IG alone** (:func:`ig_tree_boxes`): no concept. On every plane, every distinct
  8-connected component of ``{IG ≥ t}`` over the positive values ``t`` of the map (the
  component tree: at most one node per token) is a box; greedy NMS at IoU 0.5 per plane
  removes the nested ones; the confidence is the component's positive IG mass. The same
  principle as the threshold-free evaluation of saliency maps in weakly supervised
  localization (MaxBoxAcc, Choe et al., CVPR 2020), without choosing a best threshold: the AP
  ranks every box by its confidence. Ablations: other confidences (density, peak, birth level,
  persistence), 4-connectivity, other suppressions, other threshold sets, a minimum mass, the
  single baselines and the absolute value of the map.
- **VTCD concepts without IG** (:func:`concept_planes`): each concept of the chunk's dictionary
  present on a plane gives one box (or one per component), scored by a chunk-independent
  concept score: the enrichment of the concept in the positive training chunks (P05 of
  16 October 2026), or the CRIS importance of the VTCD paper; a small token tie-break keeps
  the order of equal scores reproducible.
- **Grad-CAM** (:func:`mask_components`): the 70% positive-mass mask of 16 October, one box
  per 8-connected component, scored by its positive mass.

The material of one chunk is a "partition" in the format of the AP stage: ``masses`` holds a
single value, the chunk's total (the ``share`` score divides by it), and ``planes`` the final
boxes of each plane as components of concept 0 whose ``mass`` is the box confidence, so the
AP variant ``component`` reads them unchanged. Suppression and fusion are applied here, with
the AP module's own code (:func:`evd.exp.localization_ap.detections`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from evd.exp.localization_ap import detections

#: Confidences of a component-tree node.
TREE_SCORES = ("mass", "density", "peak", "level", "persistence")
#: Box rules applied to the component-tree nodes of a plane (variants of the AP module, or
#: ``largest``: the largest node alive at each level, without suppression).
TREE_RULES = (
    "nms50-component",
    "component",
    "nms30-component",
    "nms70-component",
    "softnms-component",
    "wbf-component",
    "largest",
)
#: Token fractions of the nested thresholds (the counting cut of ADR-072, at several shares).
NESTED_FRACTIONS = (0.01, 0.02, 0.03, 0.05, 0.10, 0.20)
#: Weight of the token tie-break of chunk-independent concept scores: far below any score gap.
TIE_BREAK = 1e-9


def _neighbours(connectivity: int) -> tuple[tuple[int, int], ...]:
    if connectivity == 8:
        return ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    if connectivity == 4:
        return ((-1, 0), (0, -1), (0, 1), (1, 0))
    raise ValueError(f"Connectivity must be 4 or 8, not {connectivity}")


def component_tree(values: np.ndarray, connectivity: int = 8) -> list[dict]:
    """Every distinct connected component of ``{values ≥ t}`` over the positive values ``t``.

    Tokens enter by decreasing value (all tokens of one value together) and join their
    neighbours with a union-find; after each value, every component a token joined is a new
    node. A node is a token set: it is born at the value ``t`` that made it (``level``) and
    lives until the next value that grows it or merges it with another (``death``, ``0`` for
    the components still unchanged at the smallest positive value).

    :param values: ``H×W`` map of one plane; only its positive values make thresholds.
    :param connectivity: 8 or 4.
    :return: per node ``{"cells", "tokens", "mass", "peak", "level", "death"}``, ``cells``
        the token box ``(y0, x0, y1, x1)`` with ``y1``, ``x1`` excluded and ``mass`` the sum of
        its tokens' values; in birth order (decreasing level).
    :raises ValueError: on a map that is not 2D or not finite, or a connectivity other than 4
        or 8.
    """
    grid = np.asarray(values, dtype=np.float64)
    if grid.ndim != 2 or not np.isfinite(grid).all():
        raise ValueError("The component tree needs a finite 2D map")
    steps = _neighbours(connectivity)
    rows, cols = grid.shape
    flat = grid.ravel()
    order = np.argsort(-flat, kind="stable")
    order = order[flat[order] > 0]
    parent = np.full(flat.size, -1, dtype=np.int64)
    stats: dict[int, list] = {}
    current: dict[int, int] = {}
    nodes: list[dict] = []

    def find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = int(parent[root])
        while parent[i] != root:
            parent[i], i = root, int(parent[i])
        return root

    start = 0
    while start < len(order):
        level = flat[order[start]]
        stop = start
        while stop < len(order) and flat[order[stop]] == level:
            stop += 1
        joined: set[int] = set()
        for index in order[start:stop]:
            index = int(index)
            y, x = divmod(index, cols)
            parent[index] = index
            stats[index] = [y, x, y + 1, x + 1, 1, flat[index], flat[index]]
            for dy, dx in steps:
                ny, nx = y + dy, x + dx
                if 0 <= ny < rows and 0 <= nx < cols and parent[ny * cols + nx] != -1:
                    a, b = find(index), find(ny * cols + nx)
                    if a == b:
                        continue
                    for root in (a, b):
                        if root in current:
                            nodes[current.pop(root)]["death"] = float(level)
                    sa, sb = stats.pop(a), stats.pop(b)
                    parent[b] = a
                    stats[a] = [
                        min(sa[0], sb[0]),
                        min(sa[1], sb[1]),
                        max(sa[2], sb[2]),
                        max(sa[3], sb[3]),
                        sa[4] + sb[4],
                        sa[5] + sb[5],
                        max(sa[6], sb[6]),
                    ]
            joined.add(index)
        for root in {find(i) for i in joined}:
            y0, x0, y1, x1, tokens, mass, peak = stats[root]
            current[root] = len(nodes)
            nodes.append(
                {
                    "cells": (y0, x0, y1, x1),
                    "tokens": int(tokens),
                    "mass": float(mass),
                    "peak": float(peak),
                    "level": float(level),
                    "death": 0.0,
                }
            )
        start = stop
    return nodes


def node_score(node: Mapping, score: str) -> float:
    """:return: the confidence of a tree node: positive mass, mean value (``density``), peak,
    birth level, or ``persistence`` (birth level minus the level of its next change).

    :raises ValueError: on an unknown score.
    """
    if score == "mass":
        return float(node["mass"])
    if score == "density":
        return float(node["mass"]) / node["tokens"]
    if score == "peak":
        return float(node["peak"])
    if score == "level":
        return float(node["level"])
    if score == "persistence":
        return float(node["level"]) - float(node["death"])
    raise ValueError(f"Unknown tree score {score!r}; known: {TREE_SCORES}")


def alive_at(nodes: Sequence[Mapping], thresholds: Sequence[float]) -> list[Mapping]:
    """:return: the nodes that are a component of ``{v ≥ t}`` for some ``t`` of
    ``thresholds`` (``death < t ≤ level``), each once, in birth order."""
    cuts = np.asarray(sorted(float(t) for t in thresholds if t > 0), dtype=np.float64)
    if not len(cuts):
        return []
    kept = []
    for node in nodes:
        first = int(np.searchsorted(cuts, node["death"], side="right"))
        if first < len(cuts) and cuts[first] <= node["level"]:
            kept.append(node)
    return kept


def largest_per_level(nodes: Sequence[Mapping]) -> list[Mapping]:
    """:return: for every level of the tree the largest node alive there (most tokens, then
    more mass, then earlier birth), each node once, in birth order."""
    if not nodes:
        return []
    born = np.array([n["level"] for n in nodes])
    death = np.array([n["death"] for n in nodes])
    # Lexicographic key: tokens, then mass, then earlier birth (a smaller index).
    order = np.lexsort(
        (-np.arange(len(nodes)), [n["mass"] for n in nodes], [n["tokens"] for n in nodes])
    )
    rank = np.empty(len(nodes), dtype=np.int64)
    rank[order] = np.arange(len(nodes))
    chosen = set()
    for level in np.unique(born):
        alive = np.flatnonzero((death < level) & (level <= born))
        chosen.add(int(alive[np.argmax(rank[alive])]))
    return [nodes[i] for i in sorted(chosen)]


def chunk_thresholds(values: np.ndarray, rule: str) -> np.ndarray | None:
    """Chunk-level thresholds of a threshold-set ablation of IG alone.

    :param values: ``T×H×W`` map (only positive values count).
    :param rule: ``all`` (every positive value: ``None``, no restriction), ``q100`` (the
        quantiles 0.00:0.01:0.99 of the positive values), ``nested`` (the values keeping the
        top :data:`NESTED_FRACTIONS` of the chunk's tokens) or ``top5`` (the top 5% only).
    :return: the thresholds, or ``None`` for every positive value.
    :raises ValueError: on an unknown rule.
    """
    if rule == "all":
        return None
    flat = np.sort(np.asarray(values, dtype=np.float64).ravel())[::-1]
    positive = flat[flat > 0]
    if not len(positive):
        return np.zeros(0)
    if rule == "q100":
        return np.unique(np.quantile(positive, np.arange(100) / 100))
    if rule in ("nested", "top5"):
        fractions = NESTED_FRACTIONS if rule == "nested" else (0.05,)
        ranks = [min(len(positive), max(1, int(np.ceil(f * flat.size)))) for f in fractions]
        return np.unique(positive[[r - 1 for r in ranks]])
    raise ValueError(f"Unknown threshold rule {rule!r}")


def _native(cells: Sequence[int], rows: int, cols: int, width: int, height: int) -> list[int]:
    from evd.explain.boxes import Box, in_pixel

    y0, x0, y1, x1 = cells
    box = Box((x0, y0, x1, y1), (x0 / cols, y0 / rows, x1 / cols, y1 / rows), 0)
    return list(in_pixel(box, width, height))


def inclusive_nms(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> list[int]:
    """The authors' greedy NMS (``evd.vendor.soft_nms.cpu_nms``), vectorized over the boxes.

    Same rule as the vendored loop: inclusive boxes, areas ``(x2 - x1 + 1)·(y2 - y1 + 1)``, a
    lower-scoring box dropped when its IoU with a kept one is ``≥ thresh``, ties in index order.
    The loop costs seconds per chunk on the hundreds of nested boxes of a component tree; the
    ``≥`` matters there, since boxes on the token grid often overlap by exactly one half.

    :param boxes: ``N×4`` inclusive ``(x1, y1, x2, y2)``.
    :param scores: ``N`` scores.
    :param thresh: IoU at which a box is suppressed.
    :return: indices of the kept boxes, by decreasing score.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    keep = []
    while len(order):
        i, rest = order[0], order[1:]
        keep.append(int(i))
        w = np.minimum(boxes[i, 2], boxes[rest, 2]) - np.maximum(boxes[i, 0], boxes[rest, 0]) + 1
        h = np.minimum(boxes[i, 3], boxes[rest, 3]) - np.maximum(boxes[i, 1], boxes[rest, 1]) + 1
        inter = np.clip(w, 0, None) * np.clip(h, 0, None)
        order = rest[inter / (areas[i] + areas[rest] - inter) < thresh]
    return keep


def _final(components: list[dict], rule: str, total: float, width: int, height: int) -> list[dict]:
    """:return: the boxes ``rule`` keeps of scored components, as concept-0 components."""
    if rule in ("component", "largest") or not components:
        return components
    if rule.startswith("nms"):
        boxes = np.array([c["box"] for c in components], dtype=np.float64)
        scores = np.array([c["mass"] for c in components], dtype=np.float64)
        return [components[i] for i in inclusive_nms(boxes, scores, int(rule[3:5]) / 100)]
    out = []
    for made in detections(rule, components, [total], "mass", width, height):
        x0, y0, x1, y1 = made.rects[0]
        box = [x0, y0, x1 - 1.0, y1 - 1.0]
        if all(float(v).is_integer() for v in box):
            box = [int(v) for v in box]
        out.append({"concept": 0, "box": box, "tokens": 0, "mass": float(made.score)})
    return out


def ig_tree_boxes(
    values: np.ndarray,
    width: int,
    height: int,
    *,
    connectivity: int = 8,
    score: str = "mass",
    rule: str = "nms50-component",
    thresholds: str = "all",
    min_share: float = 0.0,
) -> dict:
    """The boxes of IG alone on one chunk, as a partition of the AP stage.

    :param values: ``T×H×W`` attribution map; its positive part makes the tree.
    :param width: native frame width.
    :param height: native frame height.
    :param connectivity: 8 (default) or 4.
    :param score: one of :data:`TREE_SCORES`, the box confidence.
    :param rule: one of :data:`TREE_RULES`.
    :param thresholds: a rule of :func:`chunk_thresholds`.
    :param min_share: nodes with less than this share of the chunk's positive mass are dropped
        before the box rule.
    :return: ``{"masses": [total], "planes": [...]}``, each plane a list of concept-0
        components scored by ``score``.
    :raises ValueError: on an unknown rule or score, or a map that is not 3D.
    """
    if rule not in TREE_RULES:
        raise ValueError(f"Unknown tree rule {rule!r}; known: {TREE_RULES}")
    grid = np.asarray(values, dtype=np.float64)
    if grid.ndim != 3:
        raise ValueError("IG alone needs a T×H×W map")
    total = float(np.maximum(grid, 0).sum())
    cuts = chunk_thresholds(grid, thresholds)
    _, rows, cols = grid.shape
    planes = []
    for plane in grid:
        nodes = component_tree(plane, connectivity)
        if cuts is not None:
            nodes = alive_at(nodes, cuts)
        if min_share > 0:
            nodes = [n for n in nodes if n["mass"] >= min_share * total]
        if rule == "largest":
            nodes = largest_per_level(nodes)
        components = [
            {
                "concept": 0,
                "box": _native(n["cells"], rows, cols, width, height),
                "tokens": n["tokens"],
                "mass": node_score(n, score),
            }
            for n in nodes
        ]
        planes.append(_final(components, rule, total, width, height))
    return {"masses": [total], "planes": planes}


def concept_planes(
    token_concepts: np.ndarray,
    scores: Mapping[int, float],
    width: int,
    height: int,
    *,
    per_component: bool = False,
    total: float = 1.0,
) -> dict:
    """The boxes of concepts ordered by a score, without attribution, as a partition.

    Every concept with a positive score present on a plane gives one box enclosing its tokens
    there (``per_component``: one per 8-connected component), scored by the concept's score
    plus :data:`TIE_BREAK` times the box's tokens, so that equal scores rank larger boxes
    first, reproducibly.

    :param token_concepts: ``T×H×W`` concept of each token.
    :param scores: concept → score; concepts missing or with a score ``≤ 0`` give no box.
    :param width: native frame width.
    :param height: native frame height.
    :param per_component: one box per component instead of per concept and plane.
    :param total: the value stored as the chunk total (the ``share`` score divides by it).
    :return: ``{"masses": [total], "planes": [...]}``.
    """
    import cv2

    ids = np.asarray(token_concepts)
    _, rows, cols = ids.shape
    kept = sorted(c for c, s in scores.items() if s > 0)
    planes = []
    for plane in ids:
        boxes = []
        for concept in kept:
            mask = (plane == concept).astype(np.uint8)
            if not mask.any():
                continue
            if per_component:
                number, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
                parts = [
                    (
                        int(stats[i, cv2.CC_STAT_TOP]),
                        int(stats[i, cv2.CC_STAT_LEFT]),
                        int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]),
                        int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]),
                        int(stats[i, cv2.CC_STAT_AREA]),
                    )
                    for i in range(1, number)
                ]
            else:
                ys, xs = np.nonzero(mask)
                parts = [
                    (int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1, len(ys))
                ]
            for y0, x0, y1, x1, tokens in parts:
                boxes.append(
                    {
                        "concept": 0,
                        "box": _native((y0, x0, y1, x1), rows, cols, width, height),
                        "tokens": tokens,
                        "mass": float(scores[concept]) + TIE_BREAK * tokens,
                    }
                )
        planes.append(boxes)
    return {"masses": [total], "planes": planes}


def token_share_scores(token_concepts: np.ndarray, concepts: int) -> dict[int, float]:
    """:return: concept → share of the chunk's tokens assigned to it (the alternative
    ordering without attribution)."""
    counts = np.bincount(np.asarray(token_concepts).ravel(), minlength=concepts)
    return {int(c): float(counts[c] / counts.sum()) for c in range(concepts)}


def mask_components(values: np.ndarray, mask: np.ndarray, width: int, height: int) -> dict:
    """The boxes of a thresholded map: one per 8-connected component of ``mask`` on each
    plane, scored by the component's positive mass, as a partition.

    :param values: ``T×H×W`` map.
    :param mask: ``T×H×W`` boolean mask of the map's cut.
    :return: ``{"masses": [total positive mass], "planes": [...]}``.
    """
    import cv2

    grid = np.maximum(np.asarray(values, dtype=np.float64), 0)
    _, rows, cols = grid.shape
    planes = []
    for plane, cut in zip(grid, np.asarray(mask, dtype=bool), strict=True):
        number, labels, stats, _ = cv2.connectedComponentsWithStats(
            cut.astype(np.uint8), connectivity=8
        )
        masses = np.bincount(labels.ravel(), weights=plane.ravel(), minlength=number)
        boxes = []
        for i in range(1, number):
            y0, x0 = int(stats[i, cv2.CC_STAT_TOP]), int(stats[i, cv2.CC_STAT_LEFT])
            y1 = y0 + int(stats[i, cv2.CC_STAT_HEIGHT])
            x1 = x0 + int(stats[i, cv2.CC_STAT_WIDTH])
            boxes.append(
                {
                    "concept": 0,
                    "box": _native((y0, x0, y1, x1), rows, cols, width, height),
                    "tokens": int(stats[i, cv2.CC_STAT_AREA]),
                    "mass": float(masses[i]),
                }
            )
        planes.append(boxes)
    return {"masses": [float(grid.sum())], "planes": planes}
