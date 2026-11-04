"""Detections of one image for the localization AP, under every box variant.

Plan: ``docs/varie/campagna_ap_localizzazione_20261027.md``. An image is one temporal plane
of one chunk; its raw material are the 8-connected components of the chunk's concepts on that
plane (:func:`evd.exp.localization_class.concept_components`) and the positive IG masses of the
concepts over the whole chunk. A variant turns them into scored detections; no variant reads
the ground truth, so the AP evaluator can derive all of them from one sealed stage. From 28
October 2026 a stage also holds the 4-connected components and the box of every SLIC region on
every plane (:func:`evd.exp.localization_class.region_boxes`), the material of the variants
that never merge a concept's regions.

Rectangles leave this module half-open, ``(x0, y0, x1, y1)`` with ``x1`` excluded, in native
pixels: an inclusive box ``[x0, y0, x1, y1]`` becomes ``(x0, y0, x1 + 1, y1 + 1)``, the COCO
convention (``w = x1 - x0 + 1``). A detection is a set of rectangles: one for a box, several
for a union of areas.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from evd.vendor.soft_nms import cpu_nms, cpu_soft_nms
from evd.vendor.wbf import weighted_boxes_fusion

#: Score types: the concept (or component) positive IG mass, or that mass over the chunk total.
SCORES = ("mass", "share")
#: IoU thresholds of the NMS variants (decision of 27 October 2026).
NMS_THRESHOLDS = (0.3, 0.5, 0.7)
#: Soft-NMS parameters of the authors' code: Gaussian decay, ``sigma`` and score cut.
SOFT_NMS = {"sigma": 0.5, "Nt": 0.3, "threshold": 0.001, "method": 2}
#: WBF parameters of the authors' library defaults.
WBF = {"iou_thr": 0.55, "skip_box_thr": 0.0, "conf_type": "avg"}
#: The variants of 27 October 2026, on the 8-connected components; their stages lack the
#: material of the later variants.
LEGACY_BOX_VARIANTS = (
    "component",
    "concept-union",
    "concept-box",
    "component-concept-score",
    "largest-component",
    *(
        f"nms{round(t * 100):02d}-{base}"
        for base in ("component", "concept-box")
        for t in NMS_THRESHOLDS
    ),
    "softnms-component",
    "softnms-concept-box",
    "wbf-component",
    "wbf-concept-box",
)
#: Variants of 28 October 2026 on the 4-connected components: the 8-connected variant of the
#: same name without the ``4``, on components split where they touch only at a corner.
COMPONENT4_VARIANTS = {
    "component4": "component",
    "concept-union4": "concept-union",
    "component4-concept-score": "component-concept-score",
    "largest-component4": "largest-component",
}
#: Variants of 28 October 2026 on the SLIC regions of each concept (``region_boxes``).
REGION_VARIANTS = (
    "region",
    "region-concept-score",
    "region-combos-nms50",
    "region-combos-nms50-concept",
)
#: Proposal filter and NMS of the region combinations, after YOLO's ``non_max_suppression``:
#: confidence above ``conf_thres`` (YOLO validation's 0.001, here on the share of the chunk's
#: positive mass), then greedy NMS dropping an IoU above ``iou_thres`` (decision of 28 October
#: 2026), at most ``max_det`` boxes per image.
YOLO_NMS = {"conf_thres": 0.001, "iou_thres": 0.5, "max_det": 300}
#: At most this many regions form one proposal: the box of any set of regions is the box of
#: the at most four regions holding its extremes, so larger sets add no new box.
COMBINATION_SIZE = 4
#: Variants whose detections are fixed boxes, matched one by one (COCO).
BOX_VARIANTS = (*LEGACY_BOX_VARIANTS, *COMPONENT4_VARIANTS, *REGION_VARIANTS)
#: The cumulative union: one detection per image and threshold, built at evaluation time.
UNION = "union"
VARIANTS = (UNION, *BOX_VARIANTS)
#: The variants a stage of 27 October 2026 (8-connected components only) can evaluate.
LEGACY_VARIANTS = (UNION, *LEGACY_BOX_VARIANTS)

Rect = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    """A scored detection of one image.

    :param rects: half-open native rectangles whose union is the detected area.
    :param score: confidence, the higher the earlier it enters the precision–recall curve.
    """

    rects: tuple[Rect, ...]
    score: float


def half_open(box: Sequence[float]) -> Rect:
    """:return: the half-open rectangle of an inclusive native box."""
    x0, y0, x1, y1 = box
    return (float(x0), float(y0), float(x1) + 1.0, float(y1) + 1.0)


def concept_order(masses: Sequence[float]) -> list[int]:
    """:return: concepts with positive mass, by decreasing mass and then id (as P07)."""
    values = np.asarray(masses, dtype=np.float64)
    order = np.lexsort((np.arange(len(values)), -values))
    return [int(k) for k in order if values[k] > 0]


def union_concepts(
    components: Sequence[Mapping], masses: Sequence[float], score: str
) -> list[tuple[float, tuple[Rect, ...]]]:
    """Material of the cumulative union of one image: each scored concept with its rectangles.

    :param components: the image's components (``concept``, ``box``, ``tokens``, ``mass``).
    :param masses: positive IG mass of each concept over the whole chunk.
    :param score: one of :data:`SCORES`.
    :return: ``(score, rectangles on this plane)`` for every concept with positive mass, in
        :func:`concept_order`; a concept absent from the plane has no rectangle, but its
        score is still a threshold of the curve.
    """
    total = float(np.sum(masses))
    out = []
    for concept in concept_order(masses):
        rects = tuple(half_open(c["box"]) for c in components if c["concept"] == concept)
        out.append((_score(float(masses[concept]), total, score), rects))
    return out


def _score(value: float, total: float, score: str) -> float:
    if score == "mass":
        return value
    if score == "share":
        return value / total
    raise ValueError(f"Unknown score {score!r}; known: {SCORES}")


def _components(components: Sequence[Mapping], masses: Sequence[float]) -> list[Mapping]:
    """:return: components of positive-mass concepts, concepts in :func:`concept_order`."""
    order = concept_order(masses)
    rank = {concept: i for i, concept in enumerate(order)}
    kept = [c for c in components if c["concept"] in rank]
    return sorted(kept, key=lambda c: rank[c["concept"]])


def _by_concept(components: Sequence[Mapping], masses: Sequence[float]) -> list[list[Mapping]]:
    groups: dict[int, list[Mapping]] = {}
    for c in _components(components, masses):
        groups.setdefault(c["concept"], []).append(c)
    return list(groups.values())


def _sorted(detections: list[Detection]) -> list[Detection]:
    """Decreasing score, stable: ties keep the canonical order (concept, then component)."""
    order = np.argsort([-d.score for d in detections], kind="mergesort")
    return [detections[i] for i in order]


def _inclusive_rows(boxes: list[Sequence[int]], scores: list[float]) -> np.ndarray:
    return np.array([[*b, s] for b, s in zip(boxes, scores, strict=True)], dtype=np.float64)


def _base(
    name: str, components: Sequence[Mapping], masses: Sequence[float], total: float, score: str
) -> tuple[list[list[int]], list[float]]:
    """Inclusive boxes and scores of the two bases of NMS, Soft-NMS and WBF."""
    if name == "component":
        kept = [c for c in _components(components, masses) if c["mass"] > 0]
        return [c["box"] for c in kept], [_score(c["mass"], total, score) for c in kept]
    if name == "concept-box":
        groups = _by_concept(components, masses)
        boxes = [
            [
                min(c["box"][0] for c in g),
                min(c["box"][1] for c in g),
                max(c["box"][2] for c in g),
                max(c["box"][3] for c in g),
            ]
            for g in groups
        ]
        return boxes, [_score(float(masses[g[0]["concept"]]), total, score) for g in groups]
    raise ValueError(f"Unknown base {name!r}")


def greedy_nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    """Greedy NMS on half-open boxes, as ``torchvision.ops.nms`` (the NMS of YOLO).

    A box is dropped when its IoU with a kept higher-scoring box is above ``iou``; areas are
    ``(x1 - x0)·(y1 - y0)``, exact on half-open rectangles. Unlike ``torchvision`` the order
    is stable on ties (earlier index first), so the result does not depend on the sort.

    :param boxes: ``N×4`` half-open ``(x0, y0, x1, y1)``.
    :param scores: ``N`` scores.
    :param iou: suppression threshold.
    :return: indices of the kept boxes, by decreasing score.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while len(order):
        i, rest = order[0], order[1:]
        keep.append(int(i))
        w = np.minimum(boxes[i, 2], boxes[rest, 2]) - np.maximum(boxes[i, 0], boxes[rest, 0])
        h = np.minimum(boxes[i, 3], boxes[rest, 3]) - np.maximum(boxes[i, 1], boxes[rest, 1])
        inter = np.clip(w, 0, None) * np.clip(h, 0, None)
        union = areas[i] + areas[rest] - inter
        overlap = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[overlap <= iou]
    return keep


def region_proposals(boxes: np.ndarray, masses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every box made by merging any non-empty set of one concept's regions on a plane.

    The box of a set of regions is the box enclosing them; its confidence is the positive IG
    mass of the set. Among the sets with the same box the heaviest is the one holding every
    region whose box lies inside it, so each distinct box comes once with that mass, the one
    YOLO's NMS would keep (an identical lower-scoring copy has IoU 1). Since the extremes of a
    set are held by at most four regions, the sets of at most :data:`COMBINATION_SIZE` regions
    already give every box: the enumeration is exact, not truncated.

    :param boxes: ``N×4`` half-open boxes of the concept's regions on the plane.
    :param masses: ``N`` positive IG masses of the regions on the plane.
    :return: ``(proposals, scores)``: distinct ``P×4`` boxes, in lexicographic order, and their
        masses.
    """
    from itertools import combinations

    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    masses = np.asarray(masses, dtype=np.float64)
    if not len(boxes):
        return np.zeros((0, 4)), np.zeros(0)
    found = []
    for size in range(1, min(COMBINATION_SIZE, len(boxes)) + 1):
        index = np.array(list(combinations(range(len(boxes)), size)), dtype=np.int64)
        chosen = boxes[index]
        found.append(
            np.stack(
                [
                    chosen[..., 0].min(1),
                    chosen[..., 1].min(1),
                    chosen[..., 2].max(1),
                    chosen[..., 3].max(1),
                ],
                axis=1,
            )
        )
    proposals = np.unique(np.concatenate(found), axis=0)
    inside = (
        (boxes[None, :, 0] >= proposals[:, None, 0])
        & (boxes[None, :, 1] >= proposals[:, None, 1])
        & (boxes[None, :, 2] <= proposals[:, None, 2])
        & (boxes[None, :, 3] <= proposals[:, None, 3])
    )
    return proposals, inside.astype(np.float64) @ masses


def _regions(regions: Sequence[Mapping], masses: Sequence[float]) -> list[Mapping]:
    """:return: regions of positive-mass concepts, concepts in :func:`concept_order`, then by
    region id."""
    rank = {concept: i for i, concept in enumerate(concept_order(masses))}
    kept = [r for r in regions if r["concept"] in rank]
    return sorted(kept, key=lambda r: (rank[r["concept"]], r["region"]))


def _region_detections(
    variant: str, regions: Sequence[Mapping], masses: Sequence[float], total: float, score: str
) -> list[Detection]:
    kept = _regions(regions, masses)
    if variant == "region":
        return _sorted(
            [
                Detection((half_open(r["box"]),), _score(r["mass"], total, score))
                for r in kept
                if r["mass"] > 0
            ]
        )
    if variant == "region-concept-score":
        ordered = sorted(
            kept, key=lambda r: (-float(masses[r["concept"]]), r["concept"], -r["mass"])
        )
        return _sorted(
            [
                Detection((half_open(r["box"]),), _score(float(masses[r["concept"]]), total, score))
                for r in ordered
            ]
        )
    groups: dict[int, list[Mapping]] = {}
    for r in kept:
        groups.setdefault(r["concept"], []).append(r)
    pools = []
    for group in groups.values():
        boxes, weights = region_proposals(
            np.array([half_open(r["box"]) for r in group]), np.array([r["mass"] for r in group])
        )
        confident = weights / total > YOLO_NMS["conf_thres"]
        pools.append((boxes[confident], weights[confident]))
    if variant == "region-combos-nms50-concept":
        found = []
        for boxes, weights in pools:
            found.extend(
                (boxes[i], weights[i]) for i in greedy_nms(boxes, weights, YOLO_NMS["iou_thres"])
            )
    elif variant == "region-combos-nms50":
        boxes = np.concatenate([b for b, _ in pools]) if pools else np.zeros((0, 4))
        weights = np.concatenate([w for _, w in pools]) if pools else np.zeros(0)
        found = [(boxes[i], weights[i]) for i in greedy_nms(boxes, weights, YOLO_NMS["iou_thres"])]
    else:
        raise ValueError(f"Unknown variant {variant!r}; known: {VARIANTS}")
    out = _sorted(
        [Detection((tuple(float(v) for v in b),), _score(float(w), total, score)) for b, w in found]
    )
    return out[: YOLO_NMS["max_det"]]


def detections(
    variant: str,
    components: Sequence[Mapping],
    masses: Sequence[float],
    score: str,
    width: int,
    height: int,
    *,
    components4: Sequence[Mapping] | None = None,
    regions: Sequence[Mapping] | None = None,
) -> list[Detection]:
    """Scored detections of one image under one box variant.

    - ``component``: every component of a positive-mass concept is a box, scored by its own
      positive mass; components with zero positive mass are not emitted.
    - ``concept-union``: one detection per concept, the union of its components' areas,
      scored by the concept mass.
    - ``concept-box``: one box per concept enclosing all its components (concept mass).
    - ``component-concept-score``: every component is a box scored by its concept's mass;
      ties between components of one concept are broken by the component mass.
    - ``largest-component``: per concept its component with most tokens (more mass, then
      the ``(-tokens, y0, x0)`` order, on ties), scored by the concept mass.
    - ``nmsNN-<base>``: the authors' greedy NMS on ``component`` or ``concept-box`` at IoU
      ``0.NN``; ``softnms-<base>``: their Gaussian Soft-NMS (:data:`SOFT_NMS`);
      ``wbf-<base>``: their Weighted Boxes Fusion (:data:`WBF`), on boxes normalized by the
      frame size and mapped back.
    - ``component4``, ``concept-union4``, ``component4-concept-score``, ``largest-component4``:
      the variant without ``4`` on the 4-connected components (``components4``).
    - ``region``: every SLIC region of a positive-mass concept is a box, scored by its own
      positive mass on the plane (zero-mass regions are not emitted); two touching regions of
      one concept stay two boxes. ``region-concept-score``: the same boxes scored by the concept
      mass, ties broken by the region mass.
    - ``region-combos-nms50``: every box of any set of one concept's regions
      (:func:`region_proposals`), scored by the set's mass, filtered and suppressed as YOLO does
      (:data:`YOLO_NMS`) over all concepts at once; ``region-combos-nms50-concept`` suppresses
      within each concept only, as YOLO does per class.

    :param variant: one of :data:`BOX_VARIANTS`.
    :param components: the image's 8-connected components (``concept``, ``box``, ``tokens``,
        ``mass``).
    :param masses: positive IG mass of each concept over the whole chunk.
    :param score: one of :data:`SCORES`.
    :param width: native frame width.
    :param height: native frame height.
    :param components4: the image's 4-connected components, for the ``…4`` variants.
    :param regions: the image's regions (``region``, ``concept``, ``box``, ``tokens``,
        ``mass``), for the region variants.
    :return: detections by decreasing score (stable on ties).
    :raises ValueError: on an unknown variant or score, or without the material it needs.
    """
    if score not in SCORES:
        raise ValueError(f"Unknown score {score!r}; known: {SCORES}")
    total = float(np.sum(masses))
    if total <= 0:
        return []
    if variant in COMPONENT4_VARIANTS:
        if components4 is None:
            raise ValueError(f"{variant} needs the 4-connected components of the image")
        return detections(COMPONENT4_VARIANTS[variant], components4, masses, score, width, height)
    if variant in REGION_VARIANTS:
        if regions is None:
            raise ValueError(f"{variant} needs the regions of the image")
        return _region_detections(variant, regions, masses, total, score)
    if variant == "component":
        boxes, scores = _base("component", components, masses, total, score)
        return _sorted([Detection((half_open(b),), s) for b, s in zip(boxes, scores, strict=True)])
    if variant == "concept-box":
        boxes, scores = _base("concept-box", components, masses, total, score)
        return _sorted([Detection((half_open(b),), s) for b, s in zip(boxes, scores, strict=True)])
    if variant == "concept-union":
        return _sorted(
            [
                Detection(
                    tuple(half_open(c["box"]) for c in g),
                    _score(float(masses[g[0]["concept"]]), total, score),
                )
                for g in _by_concept(components, masses)
            ]
        )
    if variant == "component-concept-score":
        ordered = sorted(
            _components(components, masses),
            key=lambda c: (-float(masses[c["concept"]]), c["concept"], -c["mass"]),
        )
        return _sorted(
            [
                Detection((half_open(c["box"]),), _score(float(masses[c["concept"]]), total, score))
                for c in ordered
            ]
        )
    if variant == "largest-component":
        out = []
        for g in _by_concept(components, masses):
            best = g[0]
            for c in g[1:]:
                if (c["tokens"], c["mass"]) > (best["tokens"], best["mass"]):
                    best = c
            out.append(
                Detection(
                    (half_open(best["box"]),), _score(float(masses[best["concept"]]), total, score)
                )
            )
        return _sorted(out)
    method, _, base = variant.partition("-")
    if base not in ("component", "concept-box") or variant not in BOX_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; known: {VARIANTS}")
    boxes, scores = _base(base, components, masses, total, score)
    if not boxes:
        return []
    if method.startswith("nms"):
        keep = cpu_nms(_inclusive_rows(boxes, scores), int(method[3:]) / 100)
        return _sorted([Detection((half_open(boxes[i]),), scores[i]) for i in keep])
    if method == "softnms":
        kept = cpu_soft_nms(_inclusive_rows(boxes, scores), **SOFT_NMS)
        return _sorted([Detection((half_open(row[:4]),), float(row[4])) for row in kept])
    if method == "wbf":
        scale = np.array([width, height, width, height], dtype=np.float64)
        normalized = (np.array([half_open(b) for b in boxes]) / scale).tolist()
        fused, fused_scores, _ = weighted_boxes_fusion(
            [normalized], [scores], [[0] * len(boxes)], weights=None, **WBF
        )
        return _sorted(
            [
                Detection(
                    (tuple(float(v) for v in np.asarray(b, dtype=np.float64) * scale),), float(s)
                )
                for b, s in zip(fused, fused_scores, strict=True)
            ]
        )
    raise ValueError(f"Unknown variant {variant!r}; known: {VARIANTS}")
