"""Equal-plane, equal-event, equal-class IoU for automatic localization.

Only the evaluator consumes event identities. Prediction workers must not import
ground-truth snapshots or use these identities to choose their output.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def union_box_iou(
    predicted: Sequence[Sequence[int]], actual: Sequence[Sequence[int]]
) -> float | None:
    """Measure exact native-pixel IoU between two unions of inclusive boxes.

    Unlike ``spatial.counts``, this accepts arbitrary native rectangles without
    projecting predictions onto a fixed token grid. Coordinate compression avoids
    allocating a full 4K image and counts overlapping rectangles only once.

    :param predicted: native inclusive ``(x0,y0,x1,y1)`` rectangles.
    :param actual: ground-truth rectangles under the same convention.
    :return: IoU, or ``None`` if both unions are empty.
    :raises ValueError: on fractional, negative or reversed coordinates.
    """
    groups = []
    for boxes in (predicted, actual):
        normalized = []
        for box in boxes:
            if len(box) != 4 or any(not math.isfinite(v) or int(v) != v for v in box):
                raise ValueError("Native boxes require four finite integer coordinates")
            x0, y0, x1, y1 = map(int, box)
            if min(x0, y0) < 0 or x1 < x0 or y1 < y0:
                raise ValueError("Native boxes must be nonnegative and ordered")
            normalized.append((x0, y0, x1 + 1, y1 + 1))
        groups.append(normalized)
    boxes = groups[0] + groups[1]
    if not boxes:
        return None
    xs = sorted({x for box in boxes for x in (box[0], box[2])})
    ys = sorted({y for box in boxes for y in (box[1], box[3])})
    xi, yi = {v: i for i, v in enumerate(xs)}, {v: i for i, v in enumerate(ys)}
    masks = np.zeros((2, len(ys) - 1, len(xs) - 1), dtype=bool)
    for index, group in enumerate(groups):
        for x0, y0, x1, y1 in group:
            masks[index, yi[y0] : yi[y1], xi[x0] : xi[x1]] = True
    area = np.outer(np.diff(ys).astype(np.float64), np.diff(xs).astype(np.float64))
    intersection = float(area[masks[0] & masks[1]].sum())
    union = float(area[masks[0] | masks[1]].sum())
    return intersection / union


@dataclass(frozen=True)
class PlaneIoU:
    """One already averaged annotated plane of one event.

    :param event_id: stable video/interval identity, never a nominal video class.
    :param category: actual annotated event class, used only by the evaluator.
    :param plane_id: frozen annotation unit identity within the event.
    :param iou: mean native IoU within this eligible plane; empty prediction is zero.
    """

    event_id: str
    category: str
    plane_id: str
    iou: float


def aggregate_localization(rows: Sequence[PlaneIoU], classes: Sequence[str]) -> dict:
    """Apply exactly plane → event → class → all-class equal-weight averages.

    :param rows: one row per eligible annotated plane/event pair.
    :param classes: the complete ordered set of thirteen event classes.
    :return: intermediate event records, thirteen class records and one scalar metric.
    :raises ValueError: on duplicates, nonfinite IoU, inconsistent classes or missing
        class support. Missing predictions must be handled before aggregation.
    """
    if len(classes) != 13 or len(set(classes)) != 13:
        raise ValueError("Localization requires exactly thirteen distinct event classes")
    expected = set(classes)
    events: dict[str, list[float]] = defaultdict(list)
    categories: dict[str, str] = {}
    seen = set()
    for row in rows:
        if row.category not in expected:
            raise ValueError(f"Unexpected event class: {row.category}")
        if not math.isfinite(row.iou) or not 0 <= row.iou <= 1:
            raise ValueError(f"Invalid IoU for {row.event_id}/{row.plane_id}: {row.iou}")
        key = row.event_id, row.plane_id
        if key in seen:
            raise ValueError(f"Duplicate event plane: {key}")
        seen.add(key)
        if row.event_id in categories and categories[row.event_id] != row.category:
            raise ValueError(f"Event changes class without changing identity: {row.event_id}")
        categories[row.event_id] = row.category
        events[row.event_id].append(row.iou)
    per_event = [
        {
            "event_id": event,
            "class": categories[event],
            "iou": math.fsum(values) / len(values),
            "planes": len(values),
        }
        for event, values in sorted(events.items())
    ]
    class_values: dict[str, list[float]] = defaultdict(list)
    for event in per_event:
        class_values[str(event["class"])].append(float(event["iou"]))
    missing = expected - class_values.keys()
    if missing:
        raise ValueError(f"Cannot publish a thirteen-class mean: no support for {sorted(missing)}")
    per_class = [
        {
            "class": category,
            "iou": math.fsum(class_values[category]) / len(class_values[category]),
            "events": len(class_values[category]),
        }
        for category in classes
    ]
    return {
        "metric": "macro_class_event_plane_iou",
        "value": math.fsum(float(row["iou"]) for row in per_class) / 13,
        "per_class": per_class,
        "per_event": per_event,
    }
