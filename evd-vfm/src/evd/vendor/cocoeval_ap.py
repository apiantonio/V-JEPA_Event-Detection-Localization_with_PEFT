"""COCO detection matching and AP accumulation, copied from ``pycocotools``.

Provenance: ``pycocotools/cocoeval.py`` of the PyPI package ``pycocotools==2.0.11``
(sha256 ``84b0e75c6e972ca8b6fa3219fb03c1993dc9a3509df28447ca112c3f91828d61``),
``COCOeval.evaluateImg`` (lines 235-313), ``COCOeval.accumulate`` (lines 315-410) and
``Params.setDetParams`` (lines 502-511). BSD-2-Clause, Copyright (c) 2014, Piotr Dollar
and Tsung-Yi Lin. Copied rather than called because ``COCOeval`` computes IoUs only between
single rectangles or RLE masks, while the localization AP (plan
``docs/varie/campagna_ap_localizzazione_20261027.md``) matches unions of rectangles.

Changes with respect to the original:

1. Free functions on arrays instead of methods on ``COCO`` objects: the caller passes the
   IoU matrix of one image, computed by ``evd.eval.localization_ap.region_ious``.
2. No crowd, no ignore flag, a single area range (``all``) and no ``maxDets`` cut: the
   localization ground truth has none of them, and every detection of an image is kept.
   The branches that only these features reach are removed, the rest is line by line.
3. ``accumulate`` handles one category: the AP of one event (the event is the unit of
   the per-event AP) and returns its ``T×R`` interpolated precision.
4. ``try: ... except: pass`` around the recall lookup catches ``IndexError`` only, the
   only exception it guards against.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: ``Params.setDetParams``: np.arange causes trouble, the data point on arange is slightly
#: larger than the true value.
IOU_THRESHOLDS = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
RECALL_THRESHOLDS = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)


def evaluate_image(ious: np.ndarray, iou_thresholds: Sequence[float] = IOU_THRESHOLDS) -> np.ndarray:
    """Greedy COCO matching of one image (``COCOeval.evaluateImg``).

    :param ious: ``D×G`` IoU of each detection with each ground-truth object, detections
        already sorted by decreasing score with ``mergesort`` (as ``computeIoU`` does).
    :param iou_thresholds: the ``T`` IoU thresholds.
    :return: ``T×D`` boolean, whether each detection matched an object at each threshold.
    """
    ious = np.asarray(ious, dtype=np.float64)
    T = len(iou_thresholds)
    D, G = ious.shape
    gtm = np.zeros((T, G))
    dtm = np.zeros((T, D))
    if not ious.size == 0:
        for tind, t in enumerate(iou_thresholds):
            for dind in range(D):
                # information about best match so far (m=-1 -> unmatched)
                iou = min([t, 1 - 1e-10])
                m = -1
                for gind in range(G):
                    # if this gt already matched, continue
                    if gtm[tind, gind] > 0:
                        continue
                    # continue to next gt unless better match made
                    if ious[dind, gind] < iou:
                        continue
                    # if match successful and best so far, store appropriately
                    iou = ious[dind, gind]
                    m = gind
                # if match made store id of match for both dt and gt
                if m == -1:
                    continue
                dtm[tind, dind] = 1
                gtm[tind, m] = 1
    return dtm > 0


def accumulate(
    dt_scores: Sequence[np.ndarray],
    dt_matches: Sequence[np.ndarray],
    npig: int,
    recall_thresholds: Sequence[float] = RECALL_THRESHOLDS,
) -> np.ndarray:
    """Interpolated precision of one category over its images (``COCOeval.accumulate``).

    :param dt_scores: per image, the ``D_i`` detection scores in the order of
        :func:`evaluate_image` (decreasing, ``mergesort``).
    :param dt_matches: per image, the ``T×D_i`` matches of :func:`evaluate_image`.
    :param npig: number of ground-truth objects of the category, ``> 0``.
    :param recall_thresholds: the ``R`` recall thresholds.
    :return: ``T×R`` interpolated precision; zero where a recall is never reached.
    :raises ValueError: without ground truth (COCO marks the category absent, -1).
    """
    if npig <= 0:
        raise ValueError("A category without ground truth has no AP")
    T = dt_matches[0].shape[0] if len(dt_matches) else len(IOU_THRESHOLDS)
    R = len(recall_thresholds)
    precision = np.zeros((T, R))
    dtScores = np.concatenate([np.asarray(s, dtype=np.float64) for s in dt_scores] or [np.zeros(0)])

    # different sorting method generates slightly different results.
    # mergesort is used to be consistent as Matlab implementation.
    inds = np.argsort(-dtScores, kind='mergesort')

    dtm = np.concatenate(list(dt_matches) or [np.zeros((T, 0), dtype=bool)], axis=1)[:, inds]
    tps = dtm
    fps = np.logical_not(dtm)

    tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
    fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)
    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
        tp = np.array(tp)
        fp = np.array(fp)
        nd = len(tp)
        rc = tp / npig
        pr = tp / (fp + tp + np.spacing(1))
        q = np.zeros((R,))

        # numpy is slow without cython optimization for accessing elements
        # use python array gets significant speed improvement
        pr = pr.tolist(); q = q.tolist()

        for i in range(nd - 1, 0, -1):
            if pr[i] > pr[i - 1]:
                pr[i - 1] = pr[i]

        inds = np.searchsorted(rc, recall_thresholds, side='left')
        try:
            for ri, pi in enumerate(inds):
                q[ri] = pr[pi]
        except IndexError:
            pass
        precision[t, :] = np.array(q)
    return precision
