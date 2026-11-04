"""Alternative box algorithms of the VTCD campaign, isolated from the historical behaviour.

Contract: ``docs/varie/estensione_vtcd_box_mivia_discussione.md``. The original VTCD
mask stays distinct from the grouping mask.
"""

from __future__ import annotations

import cv2
import numpy as np

from evd.explain.boxes import Box, MergeRule, boxes_from_mask, in_pixel, without_contained

METHODS = ("B0", "B1_conn4", "B2_fill03", "B2_fill05", "B2_fill07", "B3_complete", "B4_close3")
METHODS += (
    "B2_d3_fill05",
    "B2_ios06",
    "B2_ios09",
    "B2_contain08",
    "B3_d3_complete",
    "B4_cross3",
    "F_area2",
    "F_area4",
    "F_area8",
    "M_open3",
)
METHOD_LABELS = {
    "B0": "8-connected components, minimum area 1 token, contained boxes removed",
    "B1_conn4": "4-connected components: side contact only",
    "B2_fill03": "Merge at IoS 0.8, or distance 2 with fill 0.3",
    "B2_fill05": "Merge at IoS 0.8, or distance 2 with fill 0.5",
    "B2_fill07": "Merge at IoS 0.8, or distance 2 with fill 0.7",
    "B3_complete": "Merge with maximum distance 2 between all members, fill 0.5",
    "B4_close3": "Morphological closing, 3x3 square, zero border",
    "B2_d3_fill05": "Merge at IoS 0.8, or distance 3 with fill 0.5",
    "B2_ios06": "Merge at IoS 0.6, distance 2, fill 0.5",
    "B2_ios09": "Merge at IoS 0.9, distance 2, fill 0.5",
    "B2_contain08": "Merge at IoS 0.8 only; no proximity merge between 8-connected components",
    "B3_d3_complete": "Merge with maximum distance 3 between all members, fill 0.5",
    "B4_cross3": "Morphological closing, 3x3 cross, zero border",
    "F_area2": "8-connected components of at least 2 tokens",
    "F_area4": "8-connected components of at least 4 tokens",
    "F_area8": "8-connected components of at least 8 tokens",
    "M_open3": "Morphological opening, 3x3 square, zero border",
}


def _complete(mask: np.ndarray, distance: int = 2) -> list[Box]:
    """Merge only groups whose original components are all close to each other.

    :return: rectangles with complete-linkage distance <= ``distance`` and fill >= 0.5,
        or IoS >= 0.8; the complete-linkage constraint is never bypassed.
    """
    number, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    n = number - 1
    if n == 0:
        return []
    boxes = [tuple(int(x) for x in (s[0], s[1], s[0] + s[2], s[1] + s[3])) for s in stats[1:]]
    areas = [int(s[4]) for s in stats[1:]]
    near = np.zeros((n, n), bool)
    for i in range(n):
        dilated = cv2.dilate(
            (labels == i + 1).astype(np.uint8),
            np.ones((2 * distance + 1, 2 * distance + 1), np.uint8),
        )
        ids = np.unique(labels[dilated.astype(bool)])
        near[i, ids[ids > 0] - 1] = True
    near |= near.T
    groups = [{i} for i in range(n)]

    def box(group):
        values = [boxes[i] for i in group]
        return (
            min(b[0] for b in values),
            min(b[1] for b in values),
            max(b[2] for b in values),
            max(b[3] for b in values),
        )

    while True:
        options = []
        for i, ga in enumerate(groups):
            for j in range(i + 1, len(groups)):
                gb = groups[j]
                if not near[np.ix_(sorted(ga), sorted(gb))].all():
                    continue
                a, b, u = box(ga), box(gb), box(ga | gb)
                intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                    0, min(a[3], b[3]) - max(a[1], b[1])
                )
                ios = intersection / min(
                    (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
                )
                fill = sum(areas[k] for k in ga | gb) / ((u[2] - u[0]) * (u[3] - u[1]))
                if ios >= 0.8 or fill >= 0.5:
                    options.append((ios >= 0.8, ios if ios >= 0.8 else fill, -i, -j, i, j))
        if not options:
            break
        *_, i, j = max(options)
        groups[i] |= groups[j]
        groups.pop(j)
    h, w = mask.shape
    result = []
    for group in groups:
        b = box(group)
        result.append(
            Box(
                b,
                (b[0] / w, b[1] / h, b[2] / w, b[3] / h),
                sum(areas[k] for k in group),
                len(group),
            )
        )
    return without_contained(result)


def masks(
    ids: np.ndarray, subset: tuple[int, ...], width: int, height: int, method: str = "B0"
) -> tuple[np.ndarray, list]:
    """Compute the three canonical shapes with a declared box algorithm.

    :return: original mask, union of rectangles, largest one; inclusive pixel boxes.
    :raises ValueError: on an algorithm outside the matrix.
    """
    if method not in METHODS:
        raise ValueError(f"unsupported bbox algorithm: {method}")
    raw = np.isin(ids, subset)
    grouping = raw
    rule = None
    conn = 4 if method == "B1_conn4" else 8
    if method in ("B2_fill03", "B2_fill05", "B2_fill07"):
        threshold = {"B2_fill03": 0.3, "B2_fill05": 0.5, "B2_fill07": 0.7}[method]
        rule = MergeRule(distanza=2, riempimento_min=threshold, ios_min=0.8)
    elif method in ("B2_d3_fill05", "B2_ios06", "B2_ios09", "B2_contain08"):
        distance, fill, ios = {
            "B2_d3_fill05": (3, 0.5, 0.8),
            "B2_ios06": (2, 0.5, 0.6),
            "B2_ios09": (2, 0.5, 0.9),
            "B2_contain08": (1, 1.0, 0.8),
        }[method]
        rule = MergeRule(distanza=distance, riempimento_min=fill, ios_min=ios)
    if method == "B4_close3":
        grouping = cv2.morphologyEx(
            raw.astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((3, 3), np.uint8),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
    elif method in ("B4_cross3", "M_open3"):
        kernel = (
            np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
            if method == "B4_cross3"
            else np.ones((3, 3), np.uint8)
        )
        grouping = cv2.morphologyEx(
            raw.astype(np.uint8),
            cv2.MORPH_CLOSE if method == "B4_cross3" else cv2.MORPH_OPEN,
            kernel,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
    if method in ("B3_complete", "B3_d3_complete"):
        cs = _complete(grouping, distance=3 if method == "B3_d3_complete" else 2)
    else:
        cs = boxes_from_mask(
            grouping[None],
            connectivity=conn,
            min_area=int(method.removeprefix("F_area")) if method.startswith("F_area") else 1,
            remove_contained=True,
            fusione=rule,
        )[0]
    cs = sorted(cs, key=lambda c: (-c.area_token, c.token[1], c.token[0]))
    union, largest = np.zeros_like(raw), np.zeros_like(raw)
    for c in cs:
        x0, y0, x1, y1 = c.token
        union[y0:y1, x0:x1] = True
    if cs:
        x0, y0, x1, y1 = cs[0].token
        largest[y0:y1, x0:x1] = True
    return np.stack((raw, union, largest)), [in_pixel(c, width, height) for c in cs]
