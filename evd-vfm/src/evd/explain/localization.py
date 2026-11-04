"""Deterministic positive-map masks for the automatic-localization campaign.

All thresholds act on raw values. No function accepts labels, event intervals,
class names or rendered colours. Box extraction reuses ``explain.boxes``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label

from evd.explain.boxes import boxes_from_mask, in_pixel


@dataclass(frozen=True)
class MaskRule:
    """Frozen extraction rule; ``absolute`` values must be learned outside test."""

    kind: str = "mass"
    value: float = 0.7
    low: float = 0.4
    bins: int = 256
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.kind not in {"mass", "range", "count", "absolute", "otsu", "hysteresis"}:
            raise ValueError(f"Unknown mask rule: {self.kind}")
        if not np.isfinite([self.value, self.low, self.epsilon]).all() or self.epsilon < 0:
            raise ValueError("Mask parameters must be finite and epsilon nonnegative")
        if self.kind in {"mass", "range", "count", "hysteresis"} and not 0 < self.value <= 1:
            raise ValueError("Relative threshold must be in (0,1]")
        if self.kind == "hysteresis" and not 0 <= self.low <= self.value:
            raise ValueError("Hysteresis low threshold must be between zero and high")
        if self.bins < 2:
            raise ValueError("Otsu requires at least two bins")


def map_mask(values: np.ndarray, rule: MaskRule) -> np.ndarray:
    """Extract a mask with stable time/row/column ties and positive evidence only.

    :param values: finite signed ``T,H,W`` map.
    :param rule: previously frozen dynamic rule or validation-learned absolute threshold.
    :return: Boolean mask; flat relative maps abstain, absolute maps may be full.
    :raises ValueError: on invalid maps rather than silently publishing empty outputs.
    """
    signed = np.asarray(values, dtype=np.float64)
    if signed.ndim != 3 or not signed.size or not np.isfinite(signed).all():
        raise ValueError("Localization requires a nonempty finite T,H,W map")
    positive = np.maximum(signed, 0)
    if rule.kind == "absolute":
        return (positive > rule.value) & (signed > 0)
    if positive.sum() == 0 or np.ptp(positive) <= rule.epsilon:
        return np.zeros_like(positive, dtype=bool)
    if rule.kind in {"mass", "count"}:
        flat = positive.ravel()
        order = np.argsort(-flat, kind="stable")
        if rule.kind == "mass":
            cumulative = np.cumsum(flat[order])
            count = int(np.searchsorted(cumulative, rule.value * cumulative[-1], side="left")) + 1
        else:
            count = math.ceil(rule.value * flat.size)
        mask = np.zeros(flat.size, dtype=bool)
        mask[order[:count]] = True
        return (mask & (flat > 0)).reshape(positive.shape)
    minimum, maximum = positive.min(), positive.max()
    if rule.kind == "range":
        return positive > minimum + rule.value * (maximum - minimum)
    if rule.kind == "hysteresis":
        seeds = positive > minimum + rule.value * (maximum - minimum)
        weak = (positive > minimum + rule.low * (maximum - minimum)) & (positive > 0)
        result = np.zeros_like(weak)
        for t in range(len(weak)):
            components, _ = label(weak[t], structure=np.ones((3, 3), dtype=int))
            selected = np.unique(components[seeds[t]])
            result[t] = np.isin(components, selected[selected != 0])
        return result
    counts, edges = np.histogram(positive, bins=rule.bins, range=(minimum, maximum))
    centers = (edges[:-1] + edges[1:]) / 2
    weights = np.cumsum(counts, dtype=np.float64)
    sums = np.cumsum(counts * centers)
    other = weights[-1] - weights
    denominator = weights * other
    between = np.divide(
        (sums[-1] * weights - sums * weights[-1]) ** 2,
        denominator,
        out=np.full(len(weights), -np.inf),
        where=denominator > 0,
    )
    threshold = centers[int(np.argmax(between))]
    return (positive > threshold) & (positive > 0)


def native_boxes(mask: np.ndarray, width: int, height: int) -> list[list[list[int]]]:
    """Convert each temporal plane to native boxes without cross-plane merging.

    :param mask: Boolean T,H,W tensor; a full mask is valid, an empty mask abstains.
    :return: boxes per plane with inclusive integer native coordinates.
    """
    boolean = np.asarray(mask)
    if boolean.dtype != np.bool_ or boolean.ndim != 3:
        raise ValueError("Box extraction requires a Boolean T,H,W mask")
    components = boxes_from_mask(boolean, connectivity=8, min_area=1, remove_contained=True)
    return [[list(in_pixel(box, width, height)) for box in plane] for plane in components]
