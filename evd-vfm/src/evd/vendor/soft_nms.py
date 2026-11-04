"""NMS and Soft-NMS of Bodla et al. (ICCV 2017), copied from the authors' code.

Provenance: ``bharatsingh430/soft-nms``, ``lib/nms/cpu_nms.pyx`` (sha256
``8f48a6491bbea14ee71509badf309b66a78766edf2373095ad138c0534804ae4``), commit
``b8e69bdf8df2ad53025c9d198ded909b50471d4f``: ``cpu_soft_nms`` (lines 17-109) and
``cpu_nms`` (lines 111-164, the py-faster-rcnn NMS shipped in the same file). MIT License,
Copyright (c) University of Maryland, College Park; written by Navaneeth Bodla and Bharat
Singh. Copied rather than imported because the original is Cython built inside a Faster
R-CNN tree, not a package.

Changes with respect to the original:

1. Cython to Python: the ``cdef`` declarations go, the loops stay line by line. Arithmetic
   is in float64 instead of the original float32.
2. ``cpu_nms`` orders the scores with a stable sort (``np.argsort(-scores, kind="stable")``)
   instead of ``scores.argsort()[::-1]``, so tied boxes keep their input order and the
   output does not depend on the sort implementation.
3. Both functions work on a copy and return kept boxes as well as indices: ``cpu_soft_nms``
   reorders and rescores its input in place, and its ``keep`` indexes the reordered array.

Both use the pixel convention of the original: inclusive coordinates, so a box is
``x2 - x1 + 1`` pixels wide.
"""

from __future__ import annotations

import numpy as np


def cpu_soft_nms(boxes, sigma=0.5, Nt=0.3, threshold=0.001, method=0):
    """Soft-NMS: decay the scores of the boxes overlapping a higher-scoring one.

    :param boxes: ``N×5`` array ``x1, y1, x2, y2, score``, inclusive pixel coordinates.
    :param sigma: Gaussian width (``method=2``).
    :param Nt: overlap threshold of the linear (``1``) and hard (``0``) methods.
    :param threshold: boxes whose decayed score falls below it are discarded.
    :param method: ``0`` hard NMS, ``1`` linear, ``2`` Gaussian.
    :return: the kept ``K×5`` boxes with their decayed scores, in selection order.
    """
    boxes = np.array(boxes, dtype=np.float64).reshape(-1, 5)
    N = boxes.shape[0]
    pos = 0
    maxscore = 0
    maxpos = 0

    for i in range(N):
        maxscore = boxes[i, 4]
        maxpos = i

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = boxes[i, 4]

        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1

        # add max box as a detection
        boxes[i, 0] = boxes[maxpos, 0]
        boxes[i, 1] = boxes[maxpos, 1]
        boxes[i, 2] = boxes[maxpos, 2]
        boxes[i, 3] = boxes[maxpos, 3]
        boxes[i, 4] = boxes[maxpos, 4]

        # swap ith box with position of max box
        boxes[maxpos, 0] = tx1
        boxes[maxpos, 1] = ty1
        boxes[maxpos, 2] = tx2
        boxes[maxpos, 3] = ty2
        boxes[maxpos, 4] = ts

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = boxes[i, 4]

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua  # iou between max box and detection box

                    if method == 1:  # linear
                        if ov > Nt:
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2:  # gaussian
                        weight = np.exp(-(ov * ov) / sigma)
                    else:  # original NMS
                        if ov > Nt:
                            weight = 0
                        else:
                            weight = 1

                    boxes[pos, 4] = weight * boxes[pos, 4]

                    # if box score falls below threshold, discard the box by swapping with last box
                    # update N
                    if boxes[pos, 4] < threshold:
                        boxes[pos, 0] = boxes[N - 1, 0]
                        boxes[pos, 1] = boxes[N - 1, 1]
                        boxes[pos, 2] = boxes[N - 1, 2]
                        boxes[pos, 3] = boxes[N - 1, 3]
                        boxes[pos, 4] = boxes[N - 1, 4]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

    keep = [i for i in range(N)]
    return boxes[keep]


def cpu_nms(dets, thresh):
    """Greedy NMS: drop every box overlapping a kept higher-scoring one by ``>= thresh``.

    :param dets: ``N×5`` array ``x1, y1, x2, y2, score``, inclusive pixel coordinates.
    :param thresh: IoU at which a lower-scoring box is suppressed.
    :return: indices of the kept boxes into ``dets``, by decreasing score.
    """
    dets = np.asarray(dets, dtype=np.float64).reshape(-1, 5)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(-scores, kind="stable")

    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=int)

    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(int(i))
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1

    return keep
