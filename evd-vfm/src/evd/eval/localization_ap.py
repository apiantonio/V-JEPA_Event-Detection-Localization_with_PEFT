"""Per-event COCO AP of automatic localization, with unions of boxes.

The unit is the event, as in :func:`evd.eval.localization.aggregate_localization`:
the AP of each event over its images, then the mean over the events of a class, 
then the mean over the thirteen classes. An image is one distinct pair 
(annotated plane, predicted chunk plane) of the event.

Three readings of the ground truth, never compared with each other: ``union`` (one object per
image, the union of its boxes, as the IoU metric), ``instances`` (every box an object) and
``touching`` (boxes whose pixels are 8-connected joined into one object, transitively).
Matching and accumulation are the COCO ones (:mod:`evd.vendor.cocoeval_ap`); IoUs are between
unions of rectangles. The cumulative union has its own curve (:func:`union_event_precision`):
at a threshold each image has one detection, the union of the concepts at or above it.

RBDC and RBDR (Ramachandra & Jones, WACV 2020) come from :mod:`evd.vendor.rbdc` for the box
variants, imported only when asked: that port of a CC BY-NC-ND script is not distributed, and
the AP does not need it. Per event the frames are the event's images only (decision of 27 October 2026): RBDR,
the rate of ground-truth regions reached by any detection, needs nothing else; RBDC counts false
positives per frame and misses those of normal frames, so it reads optimistic. In real use (28
October 2026) the pooled metrics add the dataset's negative images, the chunk planes of the
frames outside every temporal event: they have no object, every detection on them is false, and
they count in the frames of RBDC (``frames``) and in the false positives of the pooled AP.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from evd.eval.localization import PlaneIoU, aggregate_localization
from evd.exp.localization_ap import Detection, Rect
from evd.vendor.cocoeval_ap import IOU_THRESHOLDS, RECALL_THRESHOLDS, accumulate, evaluate_image

#: Readings of the ground truth of an image (``touching``: 27 October 2026).
GT_MODES = ("union", "instances", "touching")
#: Reported summaries: the mean over the ten IoU thresholds, and at 0.50 and 0.75.
METRICS = ("AP", "AP50", "AP75")
#: IoU thresholds 0.05:0.95 by 0.05 (decision of 27 October 2026), for intervals chosen later.
#: The upper ten are COCO's array itself, so the AP at 0.50:0.95 is reproduced bit for bit.
IOU_THRESHOLDS_FINE = np.concatenate([np.linspace(0.05, 0.45, 9), IOU_THRESHOLDS])


def _interval(start: float, step: float, end: float) -> tuple[float, ...]:
    count = int(round((end - start) / step)) + 1
    return tuple(round(start + i * step, 2) for i in range(count))


#: Means over IoU thresholds (decision of 27 October 2026), name → thresholds. Intervals are
#: ``[start:step:end]``, at step 0.1 (the primary ones) and 0.05; sets of single values are in
#: braces; ``[0.5:0.05:0.95]`` is COCO's AP.
IOU_SUMMARIES: dict[str, tuple[float, ...]] = {
    **{
        f"[{a}:{s}:{b}]": _interval(a, s, b)
        for s in (0.1, 0.05)
        for a, b in ((0.1, 0.5), (0.1, 0.3), (0.3, 0.5), (0.1, 0.7), (0.3, 0.7), (0.5, 0.7))
    },
    "[0.5:0.05:0.95]": _interval(0.5, 0.05, 0.95),
    **{
        "{" + ",".join(str(t) for t in values) + "}": values
        for values in ((0.3, 0.5, 0.7), (0.1, 0.3, 0.5), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7))
    },
}


def summary_columns(thresholds: Sequence[float], values: Sequence[float]) -> list[int]:
    """:return: the indices of ``values`` (IoU thresholds) in ``thresholds``, to two decimals.

    :raises ValueError: if a value is not among the thresholds.
    """
    index = {round(float(t), 2): i for i, t in enumerate(thresholds)}
    missing = [v for v in values if round(v, 2) not in index]
    if missing:
        raise ValueError(f"IoU thresholds {missing} were not evaluated")
    return [index[round(v, 2)] for v in values]


def touching_groups(boxes: Sequence[Sequence[int]]) -> list[list[int]]:
    """Group inclusive boxes whose pixels are 8-connected, transitively.

    Two boxes touch when they overlap, share a side or only a corner pixel, the rule of the
    predicted components (``evd.exp.localization_class.concept_components``).

    :return: groups of box indices, each in index order, by their first index.
    """
    parent = list(range(len(boxes)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, (ax0, ay0, ax1, ay1) in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            bx0, by0, bx1, by1 = boxes[j]
            if bx0 <= ax1 + 1 and ax0 <= bx1 + 1 and by0 <= ay1 + 1 and ay0 <= by1 + 1:
                ri, rj = root(i), root(j)
                parent[max(ri, rj)] = min(ri, rj)
    groups: dict[int, list[int]] = {}
    for i in range(len(boxes)):
        groups.setdefault(root(i), []).append(i)
    return list(groups.values())


def ground_truth(boxes: Sequence[Sequence[int]], mode: str) -> list[tuple[Rect, ...]]:
    """Objects of one image from its inclusive native GT boxes.

    :param boxes: the image's ground-truth boxes, inclusive ``[x0, y0, x1, y1]``.
    :param mode: one of :data:`GT_MODES`.
    :return: objects, each a tuple of half-open rectangles.
    :raises ValueError: without boxes, or on an unknown mode.
    """
    if not boxes:
        raise ValueError("An image of the AP needs ground truth")
    rects = [(float(x0), float(y0), float(x1) + 1, float(y1) + 1) for x0, y0, x1, y1 in boxes]
    if mode == "union":
        return [tuple(rects)]
    if mode == "instances":
        return [(r,) for r in rects]
    if mode == "touching":
        return [tuple(rects[i] for i in group) for group in touching_groups(boxes)]
    raise ValueError(f"Unknown ground-truth mode {mode!r}; known: {GT_MODES}")


def region_ious(detected: Sequence[Sequence[Rect]], truths: Sequence[Sequence[Rect]]) -> np.ndarray:
    """Exact IoU between unions of half-open rectangles, by coordinate compression.

    ``evd.eval.localization.union_box_iou`` does the same on inclusive integer boxes and
    rejects fractional coordinates, which WBF produces; on integer boxes the two agree.

    :param detected: ``D`` regions, each a sequence of rectangles (empty: an empty region).
    :param truths: ``G`` regions.
    :return: ``D×G`` IoU; zero where a region is empty.
    """
    regions = [*detected, *truths]
    rects = [r for region in regions for r in region]
    out = np.zeros((len(detected), len(truths)))
    if not rects or not len(detected) or not len(truths):
        return out
    xs = np.unique([v for r in rects for v in (r[0], r[2])])
    ys = np.unique([v for r in rects for v in (r[1], r[3])])
    area = np.outer(np.diff(ys), np.diff(xs))
    masks = np.zeros((len(regions), len(ys) - 1, len(xs) - 1), dtype=bool)
    for index, region in enumerate(regions):
        for x0, y0, x1, y1 in region:
            i0, i1 = np.searchsorted(ys, [y0, y1])
            j0, j1 = np.searchsorted(xs, [x0, x1])
            masks[index, i0:i1, j0:j1] = True
    flat = masks.reshape(len(regions), -1).astype(np.float64)
    weights = area.ravel()
    det, gt = flat[: len(detected)], flat[len(detected) :]
    inter = (det * weights) @ gt.T
    sizes_d, sizes_g = det @ weights, gt @ weights
    union = sizes_d[:, None] + sizes_g[None, :] - inter
    np.divide(inter, union, out=out, where=union > 0)
    return out


@dataclass(frozen=True)
class Image:
    """One image of an event under one variant.

    :param truths: ground-truth objects (:func:`ground_truth`).
    :param detections: box-variant detections by decreasing score (stable), or ``None``.
    :param concepts: material of the cumulative union, ``(score, rectangles)`` per concept
        in decreasing score (``evd.exp.localization_ap.union_concepts``), or ``None``.
    """

    truths: list[tuple[Rect, ...]]
    detections: list[Detection] | None = None
    concepts: list[tuple[float, tuple[Rect, ...]]] | None = None


def prepare_boxes(images: Sequence[Image]) -> list[tuple[np.ndarray, np.ndarray]]:
    """:return: per image its detection scores by decreasing score (stable) and their ``D×G``
    IoU with the image's objects, shared by the AP and RBDC."""
    prepared = []
    for image in images:
        dets = image.detections or []
        order = np.argsort([-d.score for d in dets], kind="mergesort")
        dets = [dets[i] for i in order]
        ious = region_ious([d.rects for d in dets], image.truths)
        prepared.append((np.array([d.score for d in dets], dtype=np.float64), ious))
    return prepared


def box_matches(
    prepared: Sequence[tuple[np.ndarray, np.ndarray]], iou_thresholds: Sequence[float]
) -> list[np.ndarray]:
    """:return: per image the ``T×D`` COCO matches of its detections
    (:func:`evd.vendor.cocoeval_ap.evaluate_image`); all false on an image without objects."""
    return [evaluate_image(ious, iou_thresholds) for _, ious in prepared]


def box_event_precision(
    images: Sequence[Image],
    iou_thresholds: Sequence[float] = IOU_THRESHOLDS,
    prepared: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    matches: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """``T×R`` COCO interpolated precision of one event, detections matched one by one.

    :param images: the event's images with ``detections``.
    :param iou_thresholds: the ``T`` IoU thresholds.
    :param prepared: :func:`prepare_boxes` of ``images``, if already computed.
    :param matches: :func:`box_matches` of ``prepared``, if already computed.
    :return: the precision of :func:`evd.vendor.cocoeval_ap.accumulate`.
    """
    prepared = prepare_boxes(images) if prepared is None else prepared
    matches = box_matches(prepared, iou_thresholds) if matches is None else matches
    scores = [scores for scores, _ in prepared]
    return accumulate(scores, matches, sum(len(i.truths) for i in images))


def box_rbdc(
    prepared: Sequence[tuple[np.ndarray, np.ndarray]],
    iou_thresholds: Sequence[float],
    frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """RBDC and RBDR of box detections over a set of images (the frames).

    :param prepared: :func:`prepare_boxes` of the images.
    :param iou_thresholds: the ``T`` values of the overlap threshold ``beta``.
    :param frames: the frames of the FPR denominator; ``None`` for ``len(prepared)``. More than
        the prepared images when some images have no detection and were not prepared (the
        negative images of real use without a box).
    :return: ``(rbdc, rbdr)`` of :func:`evd.vendor.rbdc.rbdc`, each of length ``T``, the area
        brought back into ``[0, 1]``: the trapezoids of a full curve can sum to one ULP above 1
        (as in the original script), which the aggregation rejects.
    :raises ValueError: if ``frames`` is below the prepared images.
    :raises ImportError: if :mod:`evd.vendor.rbdc` is not installed (it is not distributed).
    """
    frames = len(prepared) if frames is None else frames
    if frames < len(prepared):
        raise ValueError(f"{frames} frames for {len(prepared)} prepared images")
    try:
        from evd.vendor.rbdc import rbdc
    except ImportError as error:
        raise ImportError(
            "RBDC and RBDR need evd/vendor/rbdc.py, a port of evaluation/compute_tbdc_rbdc.py "
            "of https://github.com/lilygeorgescu/AED (CC BY-NC-ND 4.0) that is not distributed "
            "with this code; evaluate with --ap-only, which needs neither"
        ) from error
    areas, rates = rbdc([s for s, _ in prepared], [i for _, i in prepared], frames, iou_thresholds)
    return np.clip(areas, 0.0, 1.0), rates


def prepare_union(images: Sequence[Image]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """:return: per image the scores of its concepts (decreasing), the ``P×G`` IoU of each
    prefix union with the image's objects and whether that union is non-empty."""
    prepared = []
    for image in images:
        concepts = image.concepts or []
        regions, union = [], ()
        for _, rects in concepts:
            union = union + tuple(rects)
            regions.append(union)
        ious = region_ious(regions, image.truths) if regions else np.zeros((0, len(image.truths)))
        prepared.append(
            (
                np.array([s for s, _ in concepts], dtype=np.float64),
                ious,
                np.array([len(r) > 0 for r in regions], dtype=bool),
            )
        )
    return prepared


def union_rbdc(
    prepared: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    iou_thresholds: Sequence[float],
    frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """RBDC and RBDR of the cumulative union over a set of images (the frames).

    At threshold ``c`` an image has one detection, the union of its concepts scoring ``>= c``
    (none when it is empty), counted as :mod:`evd.vendor.rbdc` counts one detection: it detects
    every object it reaches with IoU ``>= beta``, and is a false positive if it reaches none.
    Lowering ``c`` can turn a true positive into a false one, so the curve is not monotone:
    the RBDR at false-positive rate ``f`` is the highest RBDR of a threshold with FPR ``<= f``,
    as :func:`union_event_precision` takes the precision envelope; RBDC is its area over FPR in
    ``[0, 1]``. RBDR is the point of the lowest threshold, every concept in the union.

    :param prepared: :func:`prepare_union` of the images.
    :param iou_thresholds: the ``T`` values of ``beta``.
    :param frames: the frames of the FPR denominator, as :func:`box_rbdc`.
    :return: ``(rbdc, rbdr)``, each of length ``T``.
    :raises ValueError: without ground-truth regions, or if ``frames`` is below the images.
    """
    betas = np.asarray(iou_thresholds, dtype=np.float64)
    regions = sum(ious.shape[1] for _, ious, _ in prepared)
    if regions <= 0 or not prepared:
        raise ValueError("RBDC needs ground-truth regions and frames")
    frames = len(prepared) if frames is None else frames
    if frames < len(prepared):
        raise ValueError(f"{frames} frames for {len(prepared)} prepared images")
    # Each concept moves its image from the state of the previous prefix to its own.
    scores, delta_tp, delta_fp = [], [], []
    for concept_scores, ious, nonempty in prepared:
        if not len(concept_scores):
            continue
        reach = ious[None, :, :] >= betas[:, None, None]  # T×P×G
        tp = np.where(nonempty[None, :], reach.sum(axis=2), 0)
        fp = np.where(nonempty[None, :], ~reach.any(axis=2), False).astype(np.float64)
        zero = np.zeros((len(betas), 1))
        delta_tp.append(np.diff(np.concatenate([zero, tp], axis=1), axis=1))
        delta_fp.append(np.diff(np.concatenate([zero, fp], axis=1), axis=1))
        scores.append(concept_scores)
    rbdc_values, rbdr_values = np.zeros(len(betas)), np.zeros(len(betas))
    if not scores:
        return rbdc_values, rbdr_values
    flat = np.concatenate(scores)
    order = np.argsort(-flat, kind="mergesort")
    cum_tp = np.cumsum(np.concatenate(delta_tp, axis=1)[:, order], axis=1)
    cum_fp = np.cumsum(np.concatenate(delta_fp, axis=1)[:, order], axis=1)
    # A threshold includes every concept at or above it: the state after its last tie.
    ends = np.flatnonzero(np.append(flat[order][1:] != flat[order][:-1], True))
    fpr = np.concatenate([np.zeros((len(betas), 1)), cum_fp[:, ends] / frames], axis=1)
    rbdr = np.concatenate([np.zeros((len(betas), 1)), cum_tp[:, ends] / regions], axis=1)
    for t in range(len(betas)):
        keep = fpr[t] <= 1
        f, r = fpr[t][keep], rbdr[t][keep]
        by_fpr = np.argsort(f, kind="mergesort")
        f, r = f[by_fpr], np.maximum.accumulate(r[by_fpr])
        last = np.append(f[1:] != f[:-1], True)
        f, r = f[last], r[last]
        rbdc_values[t] = float(np.sum(r * np.diff(np.append(f, 1.0))))
    rbdr_values[:] = rbdr[:, -1]
    # As in box_rbdc: the steps of a full curve can sum to one ULP above 1.
    return np.clip(rbdc_values, 0.0, 1.0), rbdr_values


def union_event_precision(
    images: Sequence[Image],
    iou_thresholds: Sequence[float] = IOU_THRESHOLDS,
    recall_thresholds: Sequence[float] = RECALL_THRESHOLDS,
    prepared: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> np.ndarray:
    """``T×R`` interpolated precision of the cumulative union over one event.

    At threshold ``c`` an image detects the union of its concepts scoring ``>= c`` (on its
    plane; nothing if none of them is there) and is matched as one COCO detection: a true
    positive if some object reaches the IoU threshold, a false positive otherwise. The
    thresholds are every distinct concept score of the event. Lowering ``c`` can turn a true
    positive into a false one, so recall is not monotone: the interpolated precision at
    recall ``r`` is the highest precision of a threshold whose recall is ``>= r``, which is
    COCO's envelope when recall is monotone. Precision is ``tp / (tp + fp + eps)``, as COCO.

    :param images: the event's images with ``concepts``.
    :param iou_thresholds: the ``T`` IoU thresholds.
    :param recall_thresholds: the ``R`` recall thresholds.
    :param prepared: :func:`prepare_union` of ``images``, if already computed.
    :return: interpolated precision, zero where a recall is never reached.
    :raises ValueError: without ground truth.
    """
    npig = sum(len(i.truths) for i in images)
    if npig <= 0:
        raise ValueError("A category without ground truth has no AP")
    T, R = len(iou_thresholds), len(recall_thresholds)
    thresholds = sorted({s for i in images for s, _ in (i.concepts or [])}, reverse=True)
    # Per image: best IoU and non-emptiness of each prefix of its concepts.
    prepared = prepare_union(images) if prepared is None else prepared
    prefixes = [(scores, ious.max(1), nonempty) for scores, ious, nonempty in prepared]
    points = [[] for _ in range(T)]
    for c in thresholds:
        tp, nd = np.zeros(T), 0
        for scores, best, nonempty in prefixes:
            k = int(np.sum(scores >= c))
            if k == 0 or not nonempty[k - 1]:
                continue
            nd += 1
            tp += np.array([best[k - 1] >= min(t, 1 - 1e-10) for t in iou_thresholds])
        if nd == 0:
            continue
        for t in range(T):
            points[t].append((tp[t] / npig, tp[t] / (nd + np.spacing(1))))
    precision = np.zeros((T, R))
    for t in range(T):
        if not points[t]:
            continue
        rc = np.array([p[0] for p in points[t]])
        pr = np.array([p[1] for p in points[t]])
        for r, threshold in enumerate(recall_thresholds):
            reached = rc >= threshold
            if reached.any():
                precision[t, r] = pr[reached].max()
    return precision


def union_precision(
    prepared: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    iou_thresholds: Sequence[float] = IOU_THRESHOLDS,
    recall_thresholds: Sequence[float] = RECALL_THRESHOLDS,
) -> np.ndarray:
    """``T×R`` interpolated precision of the cumulative union, in one sweep over the concepts.

    The curve of :func:`union_event_precision` (equal to it on every set of images, a test
    checks it), computed by differences so that it scales to the images of a whole dataset:
    each concept moves its image from the state of the previous prefix to its own (detected or
    not, true positive or not), the concepts are taken by decreasing score and a threshold is
    the state after its last tie. Images without objects contribute false positives only.

    :param prepared: :func:`prepare_union` of the images.
    :param iou_thresholds: the ``T`` IoU thresholds.
    :param recall_thresholds: the ``R`` recall thresholds.
    :return: interpolated precision, zero where a recall is never reached.
    :raises ValueError: without ground truth.
    """
    objects = sum(ious.shape[1] for _, ious, _ in prepared)
    if objects <= 0:
        raise ValueError("A category without ground truth has no AP")
    limits = np.array([min(t, 1 - 1e-10) for t in iou_thresholds], dtype=np.float64)
    T, R = len(limits), len(recall_thresholds)
    scores, delta_tp, delta_nd = [], [], []
    for concept_scores, ious, nonempty in prepared:
        if not len(concept_scores):
            continue
        best = ious.max(1) if ious.shape[1] else np.zeros(len(concept_scores))
        hit = ((best[None, :] >= limits[:, None]) & nonempty[None, :]).astype(np.float64)
        delta_tp.append(np.diff(np.concatenate([np.zeros((T, 1)), hit], axis=1), axis=1))
        delta_nd.append(np.diff(np.concatenate([[0.0], nonempty.astype(np.float64)])))
        scores.append(concept_scores)
    precision = np.zeros((T, R))
    if not scores:
        return precision
    flat = np.concatenate(scores)
    order = np.argsort(-flat, kind="mergesort")
    ends = np.flatnonzero(np.append(flat[order][1:] != flat[order][:-1], True))
    tp = np.cumsum(np.concatenate(delta_tp, axis=1)[:, order], axis=1)[:, ends]
    nd = np.cumsum(np.concatenate(delta_nd)[order])[ends]
    points = nd > 0
    tp, nd = tp[:, points], nd[points]
    if not len(nd):
        return precision
    recall_values = np.asarray(recall_thresholds, dtype=np.float64)
    for t in range(T):
        rc, pr = tp[t] / objects, tp[t] / (nd + np.spacing(1))
        by_recall = np.argsort(rc, kind="mergesort")
        rc, best_after = rc[by_recall], np.maximum.accumulate(pr[by_recall][::-1])[::-1]
        first = np.searchsorted(rc, recall_values, side="left")
        reached = first < len(rc)
        precision[t, reached] = best_after[first[reached]]
    return precision


def summarize(precision: np.ndarray) -> dict[str, float]:
    """:return: AP (mean over IoU and recall thresholds), AP50 and AP75 of a ``T×R`` precision
    on the COCO IoU thresholds."""
    index = {round(float(t), 2): i for i, t in enumerate(IOU_THRESHOLDS)}
    return {
        "AP": float(precision.mean()),
        "AP50": float(precision[index[0.5]].mean()),
        "AP75": float(precision[index[0.75]].mean()),
    }


def ap_by_threshold(precision: np.ndarray) -> np.ndarray:
    """:return: the ``T`` APs of a ``T×R`` precision, one per IoU threshold (mean over recall)."""
    return precision.mean(axis=1)


def aggregate_by_threshold(
    events: Sequence[dict], values: np.ndarray, classes: Sequence[str]
) -> dict:
    """Event → class → all-class means at every IoU threshold, as :func:`aggregate_ap`.

    :param events: ``{"event_id", "class"}`` per event, in the rows' order.
    :param values: ``E×T`` per-event AP at each threshold.
    :param classes: the thirteen event classes.
    :return: ``{"macro": [T], "per_class": {class: [T]}}``.
    :raises ValueError: as :func:`evd.eval.localization.aggregate_localization`.
    """
    values = np.asarray(values, dtype=np.float64)
    macro, per_class = [], {c: [] for c in classes}
    for t in range(values.shape[1]):
        rows = [
            PlaneIoU(e["event_id"], e["class"], "event", values[i, t]) for i, e in enumerate(events)
        ]
        result = aggregate_localization(rows, classes)
        macro.append(result["value"])
        for row in result["per_class"]:
            per_class[row["class"]].append(row["iou"])
    return {"macro": macro, "per_class": per_class}


def aggregate_ap(events: Sequence[dict], classes: Sequence[str]) -> dict:
    """Event → class → all-class means of each metric, as the IoU metric aggregates.

    :param events: ``{"event_id", "class", "images", "AP", "AP50", "AP75"}`` per event.
    :param classes: the thirteen event classes.
    :return: per metric the macro value and the per-class means, plus the per-event rows.
    :raises ValueError: as :func:`evd.eval.localization.aggregate_localization`.
    """
    out: dict = {"per_event": list(events)}
    for metric in METRICS:
        # One row per event: aggregate_localization's plane level is then the identity.
        rows = [PlaneIoU(e["event_id"], e["class"], "event", e[metric]) for e in events]
        result = aggregate_localization(rows, classes)
        out[metric] = {
            "value": result["value"],
            "per_class": [
                {"class": r["class"], "value": r["iou"], "events": r["events"]}
                for r in result["per_class"]
            ],
        }
    return out
