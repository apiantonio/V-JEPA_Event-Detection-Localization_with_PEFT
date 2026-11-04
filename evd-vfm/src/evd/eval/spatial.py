"""Spatial counts on rectangular ground truth and on the token grid (VTCD campaign).

Pixels are inclusive and tokens half-open; planes are weighted once and the temporal
micro average is kept separate. Reuses the canonical boxes of ``explain.boxes`` and
builds no model.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment

from evd.explain.boxes import boxes_from_mask, in_pixel

SHAPES = ("mask", "boxes", "largest")
SPACES = ("pixel", "token")


def candidates(k: int) -> list[tuple[int, ...]]:
    """:return: empty set, singletons and pairs, sorted by complexity and ID."""
    return [(), *((i,) for i in range(k)), *combinations(range(k), 2)]


def truth(boxes: list[dict], grid: dict, category: str = "__all__") -> dict:
    """Rasterize the native ground truth and summarize its intersection with each cell.

    :param boxes: inclusive coordinates and half-open tokens, already validated.
    :param grid: native integer edges of the shape grid.
    :param category: event class or global union.
    :return: ground-truth areas and pixel/token occupancy, without double-counting boxes.
    """
    boxes = [b for b in boxes if category == "__all__" or b["class"] == category]
    x, y = grid["x_edges"], grid["y_edges"]
    px = np.zeros((y[-1], x[-1]), dtype=bool)
    tok = np.zeros((grid["rows"], grid["columns"]), dtype=bool)
    for b in boxes:
        a, c, e, f = b["xyxy"]
        px[c : f + 1, a : e + 1] = True
        a, c, e, f = b["token_xyxy"]
        tok[c:f, a:e] = True
    # An integral image avoids 576 slice-and-sum operations per ground truth.
    integral = np.pad(px.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    yy, xx = np.asarray(y), np.asarray(x)
    occupied = (
        integral[np.ix_(yy[1:], xx[1:])]
        - integral[np.ix_(yy[:-1], xx[1:])]
        - integral[np.ix_(yy[1:], xx[:-1])]
        + integral[np.ix_(yy[:-1], xx[:-1])]
    )
    area = np.outer(np.diff(yy), np.diff(xx))
    return {
        "occupation": np.stack((occupied.ravel(), tok.ravel())).astype(np.float64),
        "gt_area": np.array([px.sum(), tok.sum()], dtype=np.float64),
        "cell_area": np.stack((area.ravel(), np.ones(tok.size))),
        "boxes": boxes,
    }


def masks(
    ids: np.ndarray, subset: tuple[int, ...], width: int, height: int
) -> tuple[np.ndarray, list]:
    """Build the mask, the union of boxes and the box of the largest component.

    :return: three token masks and their inclusive pixel boxes.
    """
    mask = np.isin(ids, subset)
    cs = boxes_from_mask(mask[None], connectivity=8, min_area=1, remove_contained=True)[0]
    bb = np.zeros_like(mask)
    largest = np.zeros_like(mask)
    for c in cs:
        x0, y0, x1, y1 = c.token
        bb[y0:y1, x0:x1] = True
    if cs:
        x0, y0, x1, y1 = cs[0].token
        largest[y0:y1, x0:x1] = True
    return np.stack((mask, bb, largest)), [in_pixel(c, width, height) for c in cs]


def counts(predictions: np.ndarray, target: dict) -> np.ndarray:
    """Compute TP, FP and FN per shape and space, with exact native areas.

    :return: array ``[..., 2 spaces, 3 counts]``, pixels before tokens.
    """
    flat = predictions.reshape(*predictions.shape[:-2], -1).astype(np.float64)
    tp = flat @ target["occupation"].T
    predicted = flat @ target["cell_area"].T
    return np.stack((tp, predicted - tp, target["gt_area"] - tp), axis=-1)


def iou(stats: np.ndarray) -> np.ndarray:
    """:return: IoU from the counts, NaN if both sets are empty."""
    denom = stats.sum(-1)
    return np.divide(stats[..., 0], denom, out=np.full(denom.shape, np.nan), where=denom != 0)


def instance_counts(predicted: list, actual: list, threshold: float) -> tuple[int, int, int]:
    """One-to-one matching that first maximizes the pairs above threshold.

    :param predicted: inclusive pixel rectangles.
    :param actual: inclusive pixel rectangles.
    :param threshold: IoU threshold fixed before the evaluation.
    :return: instance-level TP, FP, FN.
    """
    if not predicted or not actual:
        return 0, len(predicted), len(actual)
    p, g = np.asarray(predicted), np.asarray(actual)
    start = np.maximum(p[:, None, :2], g[None, :, :2])
    end = np.minimum(p[:, None, 2:], g[None, :, 2:])
    intersect = np.maximum(end - start + 1, 0).prod(-1)
    areas_p = (p[:, 2:] - p[:, :2] + 1).prod(-1)
    areas_g = (g[:, 2:] - g[:, :2] + 1).prod(-1)
    overlap = intersect / (areas_p[:, None] + areas_g - intersect)
    # A good pair must outweigh any sum of below-threshold pairs.
    reward = (overlap >= threshold) * (min(len(p), len(g)) + 1) + overlap
    a, b = linear_sum_assignment(-reward)
    tp = int((overlap[a, b] >= threshold).sum())
    return tp, len(predicted) - tp, len(actual) - tp


def select(scores: np.ndarray, subsets: list[tuple[int, ...]], *, pair: bool) -> int:
    """Select a candidate from calibration scores only, breaking ties towards fewer elements.

    :return: index of the selected candidate.
    :raises ValueError: if all scores are undefined or unexpectedly non-finite.
    """
    allowed = np.array([len(s) <= (2 if pair else 1) for s in subsets])
    values = np.where(allowed, scores, -np.inf)
    if not np.isfinite(values).any():
        raise ValueError("calibration without positive planes")
    best = np.nanmax(values)
    return next(
        i
        for i, s in enumerate(subsets)
        if allowed[i] and np.isfinite(values[i]) and values[i] >= best - 1e-6
    )


def aggregate(rows: list[dict]) -> dict:
    """Aggregate planes already separated by frame and class, without averaging the background.

    :return: per-video macro, per-class macro, micro average and support of empty planes.
    """
    if not rows:
        return {"status": "no_support"}
    by_video, by_class = {}, {}
    tp = fp = fn = ftp = ffp = ffn = empty = empty_fp = 0.0
    hits = {0.3: {}, 0.5: {}}
    for row in rows:
        tp += row["tp"]
        fp += row["fp"]
        fn += row["fn"]
        ftp += row["frame_tp"]
        ffp += row["frame_fp"]
        ffn += row["frame_fn"]
        if row["gt_positive"]:
            by_video.setdefault(row["video"], []).append(row["iou"])
            by_class.setdefault(row["class"], {}).setdefault(row["video"], []).append(row["iou"])
            for t in hits:
                hits[t].setdefault(row["class"], {}).setdefault(row["video"], []).append(
                    row[f"hit_{t}"]
                )
        else:
            empty += 1
            empty_fp += row["fp_present"]

    def div(a, b):
        return float(a / b) if b else None

    per_class = {
        c: float(np.mean([np.mean(v) for v in groups.values()])) for c, groups in by_class.items()
    }
    return {
        "macro_plane_iou": div(
            sum(sum(v) for v in by_video.values()), sum(len(v) for v in by_video.values())
        ),
        "macro_video_iou": float(np.mean([np.mean(v) for v in by_video.values()]))
        if by_video
        else None,
        "macro_class_iou": float(np.mean(list(per_class.values()))) if per_class else None,
        "micro_plane_iou": div(tp, tp + fp + fn),
        "micro_frame_iou": div(ftp, ftp + ffp + ffn),
        "spatial_precision": div(tp, tp + fp),
        "spatial_recall": div(tp, tp + fn),
        "spatial_f1": div(2 * tp, 2 * tp + fp + fn),
        "spatial_f1_frame": div(2 * ftp, 2 * ftp + ffp + ffn),
        "hit_0.3": float(
            np.mean([np.mean([np.mean(v) for v in g.values()]) for g in hits[0.3].values()])
        )
        if by_class
        else None,
        "hit_0.5": float(
            np.mean([np.mean([np.mean(v) for v in g.values()]) for g in hits[0.5].values()])
        )
        if by_class
        else None,
        "planes": len(rows),
        "positive_planes": sum(r["gt_positive"] for r in rows),
        "empty_planes": int(empty),
        "empty_fp_rate": div(empty_fp, empty),
        "empty_predicted_fraction": float(
            np.mean([r["predicted_fraction"] for r in rows if not r["gt_positive"]])
        )
        if empty
        else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "frame_tp": ftp,
        "frame_fp": ffp,
        "frame_fn": ffn,
        "videos": len({r["video"] for r in rows}),
        "per_class": per_class,
    }
