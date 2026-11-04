"""From an explanation map to the rectangles surrounding it, without torch.

A rectangular bounding box around any area highlighted by VTCD concept 0, several boxes
for disconnected areas, as a possible automatic annotation technique for spatial event
localization; the same is applied to the highest-attribution tokens of
``ig_ensemble``.

Native coordinates are not computed here: a box is born on the token grid and is
expressed in frame fractions; mapping it to pixels is valid only because the
``edp_v2`` preprocessing is ``squash`` (a pure resize without crop).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = [
    "Box",
    "MergeRule",
    "boxes_from_mask",
    "in_pixel",
    "concept_mask",
    "top_fraction_mask",
    "without_contained",
]


@dataclass(frozen=True)
class Box:
    """A rectangle on the token grid of one temporal plane.

    :param token: ``(x0, y0, x1, y1)`` on the ``H'xW'`` grid, half-open bounds (the box
        covers columns ``x0 .. x1-1``).
    :param norm: the same bounds as frame fractions in ``[0, 1]``, on the outer border of
        the tokens.
    :param area_token: number of tokens in the connected component (not the rectangle
        area; their ratio is the fill).
    :param componenti: number of 8-connected components merged into the box; 1 without
        :class:`MergeRule`.
    """

    token: tuple[int, int, int, int]
    norm: tuple[float, float, float, float]
    area_token: int
    componenti: int = 1

    @property
    def riempimento(self) -> float:
        """:return: fraction of the rectangle actually covered, in ``(0, 1]``."""
        x0, y0, x1, y1 = self.token
        return self.area_token / max((x1 - x0) * (y1 - y0), 1)


@dataclass(frozen=True)
class MergeRule:
    """How the components of a plane are merged into fewer boxes (30 September 2026).

    Three criteria, applied pairwise until no pair satisfies them:

    1. Containment (IoS, intersection over the smaller box area, as in SAHI Greedy NMM):
       with ``IoS >= ios_min`` the two boxes merge (IoU would never merge a small box
       inside a large one). Boxes are merged, not suppressed.
    2. Proximity (single-linkage clustering on tokens): only components within Chebyshev
       distance ``<= distanza`` are candidates (1 is 8-connectivity).
    3. Fill (Selective Search fill similarity, Uijlings et al. 2013): a nearby pair merges
       only if ``(area_a + area_b) / area(union box) >= riempimento_min``, which prevents
       chaining into a frame-wide rectangle.

    The mask does not change: a box stays the tight rectangle of the group's tokens
    (ADR-072); no morphological closing.

    :param distanza: maximum Chebyshev distance between candidate components, in tokens.
    :param riempimento_min: minimum fill of the union box of two nearby components, in
        ``(0, 1]``.
    :param ios_min: IoS above which two boxes always merge, in ``(0, 1]``.
    :raises ValueError: on out-of-range parameters.
    """

    distanza: int = 2
    riempimento_min: float = 0.5
    ios_min: float = 0.8

    def __post_init__(self) -> None:
        if self.distanza < 1:
            raise ValueError(
                f"The distance is measured in tokens and is at least 1: {self.distanza}"
            )
        for name in ("riempimento_min", "ios_min"):
            value = getattr(self, name)
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1], got {value}")


def _merge_plane(
    etichette: np.ndarray, numero: int, regola: MergeRule
) -> list[tuple[tuple[int, int, int, int], int, int]]:
    """Merge the components of a plane according to the rule.

    One merge per step until no pair qualifies: containment with the highest IoS first,
    then the nearby pair with the highest union fill. Ties go to the lowest indices, so
    results are deterministic.

    :param etichette: labels from ``cv2.connectedComponentsWithStats``, 0 = background.
    :param numero: number of labels, background included.
    :param regola: the three criteria.
    :return: ``(half-open box, tokens, components)`` per group.
    """
    n = numero - 1
    if n <= 0:
        return []
    box = np.zeros((n, 4), dtype=np.int64)
    area = np.zeros(n, dtype=np.int64)
    for i in range(n):
        ys, xs = np.nonzero(etichette == i + 1)
        box[i] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        area[i] = xs.size
    # Proximity: the dilation of radius `distanza` of a component touches the other one iff
    # their Chebyshev distance is at most `distanza`.
    side = 2 * regola.distanza + 1
    kernel = np.ones((side, side), dtype=np.uint8)
    nearby = np.zeros((n, n), dtype=bool)
    for i in range(n):
        dilated = cv2.dilate((etichette == i + 1).astype(np.uint8), kernel)
        touched = np.unique(etichette[dilated.astype(bool)])
        nearby[i, touched[touched > 0] - 1] = True
    np.fill_diagonal(nearby, False)
    nearby |= nearby.T
    components = np.ones(n, dtype=np.int64)
    alive = np.ones(n, dtype=bool)
    while alive.sum() > 1:
        idx = np.flatnonzero(alive)
        b = box[idx]
        ux0 = np.minimum.outer(b[:, 0], b[:, 0])
        uy0 = np.minimum.outer(b[:, 1], b[:, 1])
        ux1 = np.maximum.outer(b[:, 2], b[:, 2])
        uy1 = np.maximum.outer(b[:, 3], b[:, 3])
        ix = np.clip(
            np.minimum.outer(b[:, 2], b[:, 2]) - np.maximum.outer(b[:, 0], b[:, 0]), 0, None
        )
        iy = np.clip(
            np.minimum.outer(b[:, 3], b[:, 3]) - np.maximum.outer(b[:, 1], b[:, 1]), 0, None
        )
        box_areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        ios = ix * iy / np.minimum.outer(box_areas, box_areas)
        fill = np.add.outer(area[idx], area[idx]) / ((ux1 - ux0) * (uy1 - uy0))
        pairs = np.triu(np.ones_like(ios, dtype=bool), k=1)
        contained = pairs & (ios >= regola.ios_min)
        full = pairs & nearby[np.ix_(idx, idx)] & (fill >= regola.riempimento_min)
        if contained.any():
            score = np.where(contained, ios, -1.0)
        elif full.any():
            score = np.where(full, fill, -1.0)
        else:
            break
        a, c = np.unravel_index(int(np.argmax(score)), score.shape)
        i, j = int(idx[a]), int(idx[c])
        box[i] = (
            min(box[i, 0], box[j, 0]),
            min(box[i, 1], box[j, 1]),
            max(box[i, 2], box[j, 2]),
            max(box[i, 3], box[j, 3]),
        )
        area[i] += area[j]
        components[i] += components[j]
        nearby[i] |= nearby[j]
        nearby[:, i] = nearby[i]
        nearby[i, i] = False
        alive[j] = False
    return [
        (tuple(int(v) for v in box[i]), int(area[i]), int(components[i]))
        for i in np.flatnonzero(alive)
    ]


def concept_mask(ids: np.ndarray, *, concept_id: int | tuple[int, ...]) -> np.ndarray:
    """Return the tokens assigned to a global VTCD concept, or to a union of concepts.

    ``vtcd_ids`` is already discrete, so no threshold is chosen; the same condition used by
    :func:`evd.explain.perturbed.vtcd_concept_view`. A tuple of IDs treats several concepts
    as one (e.g. Shoplifting 5+7, 3 October 2026).

    :param ids: ``[T', H', W']`` grid with the concept ID of each token.
    :param concept_id: concept to isolate, or the tuple of concepts to merge.
    :return: boolean mask with the same shape.
    :raises ValueError: on a non-3D grid, a negative ``concept_id`` or an empty tuple.
    """
    grid = np.asarray(ids)
    if grid.ndim != 3:
        raise ValueError(f"VTCD requires a [T,H,W] grid, got {grid.shape}")
    concepts = (concept_id,) if isinstance(concept_id, (int, np.integer)) else tuple(concept_id)
    if not concepts or min(concepts) < 0:
        raise ValueError("concept_id must be non-negative, and a union must not be empty")
    return np.isin(grid, concepts)


def top_fraction_mask(
    map_data: np.ndarray, *, frazione: float, absolute: bool = False
) -> tuple[np.ndarray, dict]:
    """Return the highest-attribution tokens of the window, cut by count.

    Sorting and taking the first ``k`` (instead of comparing with a quantile) keeps the
    coverage constant across windows despite repeated values. Values are signed by default
    (tokens pushing the logit towards the anomaly; ``ig_ensemble`` mixes components of
    opposite sign, ADR-073); ``assoluto=True`` ranks by magnitude instead.

    :param mappa: ``[T', H', W']`` attribution grid.
    :param frazione: fraction of tokens to select, in ``(0, 1]``.
    :param assoluto: rank by ``|value|`` instead of value.
    :return: ``(boolean mask, receipt)``; the receipt carries the effective threshold, the
        matching quantile and the positive mass.
    :raises ValueError: on a non-3D grid, a fraction outside ``(0, 1]`` or non-finite
        values.
    """
    grid = np.asarray(map_data, dtype=np.float64)
    if grid.ndim != 3:
        raise ValueError(f"A [T,H,W] grid is required, got {grid.shape}")
    if not 0 < frazione <= 1:
        raise ValueError(f"The fraction must be in (0, 1], got {frazione}")
    if not np.isfinite(grid).all():
        raise ValueError("The map contains non-finite values: inconsistent attribution")
    values = grid.ravel()
    total = values.size
    k = min(max(int(round(frazione * total)), 1), total)
    key = np.abs(values) if absolute else values
    # `stable` decides which tied tokens enter, so repeated runs give the same mask.
    order = np.argsort(-key, kind="stable")
    flat = np.zeros(total, dtype=bool)
    flat[order[:k]] = True
    mask = flat.reshape(grid.shape)
    threshold = float(key[order[k - 1]])
    selected = values[flat]
    receipt = {
        "regola": "top-k by count over the whole window",
        "valore": "assoluto" if absolute else "firmato",
        "frazione": float(frazione),
        "token_selezionati": int(k),
        "total_tokens": int(total),
        "coverage": float(k / total),
        "soglia": threshold,
        # Tokens strictly below the threshold: with ties at the cut it is below 1 - fraction,
        # and the gap counts ties broken by token order.
        "quantile_effettivo": float(np.mean(key < threshold)),
        "pareggi_sul_taglio": int(np.sum(key == threshold) - 1),
        "massa_positiva": float(np.mean(selected > 0)),
        "somma_selezionata": float(selected.sum()),
        "somma_totale": float(values.sum()),
    }
    return mask, receipt


def without_contained(caselle: list[Box]) -> list[Box]:
    """Remove the boxes fully contained in a larger one of the same plane (3 October 2026).

    Containment is on token bounds (borders included), "larger" is the rectangle area. Two
    identical rectangles are duplicates: the first in ``(-area, y0, x0)`` order is kept.
    Unlike :class:`MergeRule`, a box sticking out by one token stays.

    :param caselle: boxes of a plane, already sorted.
    :return: the remaining boxes, in the same order.
    """

    def area(c: Box) -> int:
        return (c.token[2] - c.token[0]) * (c.token[3] - c.token[1])

    def inside(a: Box, b: Box) -> bool:
        return (
            b.token[0] <= a.token[0]
            and b.token[1] <= a.token[1]
            and a.token[2] <= b.token[2]
            and a.token[3] <= b.token[3]
        )

    kept = []
    for i, box in enumerate(caselle):
        covered = any(
            inside(box, other) and (area(other) > area(box) or j < i)
            for j, other in enumerate(caselle)
            if j != i
        )
        if not covered:
            kept.append(box)
    return kept


def boxes_from_mask(
    mask: np.ndarray,
    *,
    connectivity: int = 8,
    min_area: int = 1,
    fusione: MergeRule | None = None,
    remove_contained: bool = False,
) -> list[list[Box]]:
    """Return one rectangle per connected area, plane by plane.

    8-connectivity and minimum area of 1 token (decision of 22 September 2026): diagonal
    neighbours are the same area and no area is discarded. Temporal planes are not merged
    and boxes are not tracked (interpolating would invent motion, as in
    :func:`evd.explain.project.offline_plane`).

    :param maschera: boolean ``[T', H', W']``.
    :param connettivita: ``4`` (sides only) or ``8`` (diagonals too).
    :param min_area: components with fewer tokens are discarded.
    :param fusione: merges components after finding them (:class:`MergeRule`); ``None``
        keeps one box per component. The minimum area applies to the groups.
    :param remove_contained: finally remove boxes contained in a larger one
        (:func:`without_contained`).
    :return: one list of boxes per plane, sorted by ``(-area, y0, x0)`` so box indices and
        colours are stable across renderings.
    :raises ValueError: on a non-3D mask or a connectivity other than 4/8.
    """
    grid = np.asarray(mask)
    if grid.ndim != 3:
        raise ValueError(f"A [T,H,W] mask is required, got {grid.shape}")
    if connectivity not in (4, 8):
        raise ValueError(f"Allowed connectivity: 4 or 8, got {connectivity}")
    if min_area < 1:
        raise ValueError("The minimum area is measured in tokens and is at least 1")
    _, height, width = grid.shape
    planes: list[list[Box]] = []
    for plane in grid:
        binary = np.ascontiguousarray(plane.astype(np.uint8))
        number, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=connectivity
        )
        boxes = []
        if fusione is not None:
            for (x0, y0, x1, y1), area, groups in _merge_plane(labels, number, fusione):
                if area < min_area:
                    continue
                boxes.append(
                    Box(
                        token=(x0, y0, x1, y1),
                        norm=(x0 / width, y0 / height, x1 / width, y1 / height),
                        area_token=area,
                        componenti=groups,
                    )
                )
            boxes.sort(key=lambda c: (-c.area_token, c.token[1], c.token[0]))
            planes.append(without_contained(boxes) if remove_contained else boxes)
            continue
        for component in range(1, number):
            area = int(stats[component, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x0 = int(stats[component, cv2.CC_STAT_LEFT])
            y0 = int(stats[component, cv2.CC_STAT_TOP])
            x1 = x0 + int(stats[component, cv2.CC_STAT_WIDTH])
            y1 = y0 + int(stats[component, cv2.CC_STAT_HEIGHT])
            boxes.append(
                Box(
                    token=(x0, y0, x1, y1),
                    norm=(x0 / width, y0 / height, x1 / width, y1 / height),
                    area_token=area,
                )
            )
        boxes.sort(key=lambda c: (-c.area_token, c.token[1], c.token[0]))
        planes.append(without_contained(boxes) if remove_contained else boxes)
    return planes


def in_pixel(box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    """Return the rectangle at the output resolution, with inclusive bounds.

    ``cv2.rectangle`` draws up to the last given pixel while :attr:`Box.norm` is
    half-open, so the last pixel is ``round(x1 * width) - 1``, clamped to the frame.

    :param box: box to convert.
    :param larghezza: output frame width.
    :param height: output frame height.
    :return: ``(x0, y0, x1, y1)`` in pixels, ``x1``/``y1`` included.
    """
    x0n, y0n, x1n, y1n = box.norm
    x0 = int(np.clip(round(x0n * width), 0, width - 1))
    y0 = int(np.clip(round(y0n * height), 0, height - 1))
    x1 = int(np.clip(round(x1n * width) - 1, x0, width - 1))
    y1 = int(np.clip(round(y1n * height) - 1, y0, height - 1))
    return x0, y0, x1, y1
