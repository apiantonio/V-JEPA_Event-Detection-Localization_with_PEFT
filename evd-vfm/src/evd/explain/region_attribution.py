"""Region-perturbation estimators of the automatic-localization campaign.

RISE, occlusion, LIME, KernelSHAP, HSIC), Score-CAM, Ablation-CAM.
Everything here is pure numpy on logits and masks, so the
estimators are verified on CPU; the stage (``evd.exp.localization_perturb``) only produces
the perturbed logits.

Regions: 2 temporal halves × 4 × 4 cells of the token grid ``(16, 24, 24)``, i.e. 32
regions of 8 × 6 × 6 tokens, aligned with tubelets and patches. A region value is spread
uniformly over its tokens (value / token count), keeping its sign.
"""

from __future__ import annotations

import math

import numpy as np

REGION_GRID = (2, 4, 4)


def region_index(grid: tuple[int, int, int]) -> np.ndarray:
    """:return: ``(T,H,W)`` map of the region id (0..31) of every token."""
    t, h, w = grid
    rt, rh, rw = REGION_GRID
    if t % rt or h % rh or w % rw:
        raise ValueError(f"Token grid {grid} is not divisible by the region grid {REGION_GRID}")
    tt = np.arange(t)[:, None, None] // (t // rt)
    hh = np.arange(h)[None, :, None] // (h // rh)
    ww = np.arange(w)[None, None, :] // (w // rw)
    return (tt * rh + hh) * rw + ww


def regions_to_tokens(values: np.ndarray, grid: tuple[int, int, int]) -> np.ndarray:
    """:return: token map with each region value divided by its token count."""
    index = region_index(grid)
    counts = np.bincount(index.ravel(), minlength=values.size)
    return (np.asarray(values, dtype=np.float64) / counts)[index]


def bernoulli_presence(n: int, regions: int, seed: int) -> np.ndarray:
    """:return: ``n × regions`` presence matrix, Bernoulli(0.5), from a fixed seed."""
    return np.random.default_rng(seed).random((n, regions)) < 0.5


def rise_map(
    presence: np.ndarray, logits: np.ndarray, grid, probability: float = 0.5
) -> np.ndarray:
    """RISE: per token, the logit averaged over masks keeping it, over the keep probability.

    :param presence: ``N × 32`` boolean, region kept.
    :param logits: ``N`` anomaly logits of the masked inputs.
    :return: signed ``(T,H,W)`` map.
    """
    region = presence.astype(np.float64).T @ np.asarray(logits, dtype=np.float64)
    return (region / (len(logits) * probability))[region_index(grid)]


def lime_coefficients(
    presence: np.ndarray, logits: np.ndarray, *, width: float = 0.25, ridge: float = 1.0
) -> np.ndarray:
    """LIME surrogate: weighted ridge of the logit on region presence, free intercept.

    Kernel ``exp(-d²/(2·width²))`` with ``d`` the cosine distance from the full mask (1 for
    the empty mask); weights normalized to sum ``N``.

    :return: ``32`` signed coefficients.
    """
    z = presence.astype(np.float64)
    n, d = z.shape
    kept = z.sum(1)
    distance = np.where(kept > 0, 1 - np.sqrt(kept / d), 1.0)
    weights = np.exp(-(distance**2) / (2 * width**2))
    weights *= n / weights.sum()
    x = np.hstack([np.ones((n, 1)), z])
    penalty = np.eye(d + 1) * ridge
    penalty[0, 0] = 0.0
    xtw = x.T * weights
    solution = np.linalg.solve(xtw @ x + penalty, xtw @ np.asarray(logits, dtype=np.float64))
    return solution[1:]


def hsic_scores(presence: np.ndarray, logits: np.ndarray) -> np.ndarray:
    """HSIC between each region's presence and the logit (unsigned dependence).

    ``h_i = trace(K_i H L H) / (N-1)²``, ``K_i`` equality of presence states, ``L`` Gaussian
    on logits with bandwidth the median non-zero pairwise distance.

    :return: ``32`` non-negative scores; all zero for a constant output.
    """
    y = np.asarray(logits, dtype=np.float64)
    n = len(y)
    distances = np.abs(y[:, None] - y[None, :])
    nonzero = distances[distances > 0]
    if not nonzero.size:
        return np.zeros(presence.shape[1])
    sigma = np.median(nonzero)
    big_l = np.exp(-(distances**2) / (2 * sigma**2))
    centering = np.eye(n) - 1.0 / n
    hlh = centering @ big_l @ centering
    scores = []
    for i in range(presence.shape[1]):
        k = (presence[:, i][:, None] == presence[:, i][None, :]).astype(np.float64)
        scores.append(float(np.sum(k * hlh)) / (n - 1) ** 2)  # trace(K HLH) for symmetric K
    return np.asarray(scores)


def kernel_shap_coalitions(n: int, players: int, seed: int) -> np.ndarray:
    """Sample coalitions with size probability proportional to the Shapley-kernel mass.

    The kernel weight of a coalition of size ``s`` is ``(d-1)/(C(d,s) s (d-s))``; summed over
    the ``C(d,s)`` coalitions of that size it is ``(d-1)/(s(d-s))``. Sizes are drawn with that
    mass and subsets uniformly, so the sample is distributed as the kernel and the regression
    weights are equal.

    :return: ``n × players`` boolean presence, never empty nor full.
    """
    rng = np.random.default_rng(seed)
    sizes = np.arange(1, players)
    mass = (players - 1) / (sizes * (players - sizes))
    drawn = rng.choice(sizes, size=n, p=mass / mass.sum())
    coalitions = np.zeros((n, players), dtype=bool)
    for row, size in enumerate(drawn):
        coalitions[row, rng.choice(players, size=size, replace=False)] = True
    return coalitions


def kernel_shap(
    coalitions: np.ndarray, logits: np.ndarray, empty: float, full: float
) -> np.ndarray:
    """KernelSHAP values with the efficiency constraint ``sum(phi) = full - empty``.

    Least squares on kernel-distributed coalitions (equal weights), eliminating the last
    player through the constraint.

    :return: ``players`` signed attributions.
    """
    z = coalitions.astype(np.float64)
    y = np.asarray(logits, dtype=np.float64) - empty
    total = full - empty
    x = z[:, :-1] - z[:, -1:]
    target = y - z[:, -1] * total
    phi, *_ = np.linalg.lstsq(x, target, rcond=None)
    return np.append(phi, total - phi.sum())


def occlusion_windows(cells: int = 6, size: int = 2) -> list[tuple[int, int]]:
    """:return: top-left cells of the sliding ``size × size`` windows on a ``cells²`` grid."""
    return [(i, j) for i in range(cells - size + 1) for j in range(cells - size + 1)]


def occlusion_token_masks(grid, cells: int = 6, size: int = 2) -> np.ndarray:
    """:return: ``W × T × H × W`` boolean masks of the occluded tokens, full duration."""
    t, h, w = grid
    ch, cw = h // cells, w // cells
    masks = []
    for i, j in occlusion_windows(cells, size):
        mask = np.zeros(grid, dtype=bool)
        mask[:, i * ch : (i + size) * ch, j * cw : (j + size) * cw] = True
        masks.append(mask)
    return np.stack(masks)


def occlusion_map(masks: np.ndarray, drops: np.ndarray) -> np.ndarray:
    """:return: per token, the mean signed logit drop of the windows covering it."""
    covered = masks.sum(0)
    total = np.tensordot(np.asarray(drops, dtype=np.float64), masks.astype(np.float64), axes=1)
    return np.divide(total, covered, out=np.zeros(covered.shape), where=covered > 0)


def top_variance_channels(variances: np.ndarray, count: int = 32) -> np.ndarray:
    """:return: channels with the largest mean spatial variance, ties by channel index."""
    mean = np.asarray(variances, dtype=np.float64).mean(0)
    return np.lexsort((np.arange(mean.size), -mean))[:count]


def score_cam(activations: np.ndarray, scores: np.ndarray, logit: float) -> np.ndarray | None:
    """Score-CAM combination with softmax weights of ``c_k = logit(x·mask_k) - logit(black)``.

    :param activations: ``K × T × H × W`` signed channel maps of the non-constant channels.
    :param scores: ``K`` increments.
    :return: the combined map, or ``None`` when no increment exceeds the tolerance.
    """
    if not scores.size or scores.max() <= 1e-6 * max(1.0, abs(logit)):
        return None
    weights = np.exp(scores - scores.max())
    weights /= weights.sum()
    return np.tensordot(weights, activations, axes=1)


def ablation_cam(activations: np.ndarray, ablated: np.ndarray, logit: float) -> np.ndarray | None:
    """Ablation-CAM with weights ``(logit - logit_k)/(|logit| + 1e-6)``.

    :return: the combined map, or ``None`` when no ablation lowers the logit beyond tolerance.
    """
    drops = logit - np.asarray(ablated, dtype=np.float64)
    if drops.max() <= 1e-6 * max(1.0, abs(logit)):
        return None
    return np.tensordot(drops / (abs(logit) + 1e-6), activations, axes=1)


def project_mean_box(values: np.ndarray, mean: float, iterations: int = 60) -> np.ndarray:
    """Euclidean projection on ``{m in [0,1]^P : mean(m) = mean}`` by bisection on a shift.

    :return: projected values with the requested mean (to float precision).
    """
    v = np.asarray(values, dtype=np.float64)
    low, high = v.min() - 1.0, v.max()
    for _ in range(iterations):
        shift = (low + high) / 2
        if np.clip(v - shift, 0, 1).mean() > mean:
            low = shift
        else:
            high = shift
    return np.clip(v - (low + high) / 2, 0, 1)


def top_fraction_mask(values: np.ndarray, fraction: float) -> np.ndarray:
    """:return: the first ``ceil(fraction·P)`` positive values, stable time/row/column ties."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    if np.ptp(flat) <= 1e-8:
        return np.zeros(values.shape, dtype=bool)
    order = np.argsort(-flat, kind="stable")[: math.ceil(fraction * flat.size)]
    mask = np.zeros(flat.size, dtype=bool)
    mask[order] = True
    return (mask & (flat > 0)).reshape(values.shape)
