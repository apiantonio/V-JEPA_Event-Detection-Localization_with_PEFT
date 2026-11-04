"""Progress statistics of the spatial annotation, per video, class and event.

Counting units: event frames, base VTCD planes (``planes.protocol_units``), events and
videos. A unit is completed when all its frames are reviewed or excluded, in progress
when at least one frame has a saved state, to do otherwise. Contract in
``docs/20_annotatore_spaziale.md``.
"""

from __future__ import annotations

from collections import Counter

#: States that close a frame: reviewed (even without boxes) or excluded with a reason.
DONE = frozenset({"reviewed", "excluded"})


def _state(statuses: list[str | None]) -> str:
    """:return: ``done``, ``progress`` or ``todo`` for a set of frames."""
    if statuses and all(s in DONE for s in statuses):
        return "done"
    return "progress" if any(s is not None for s in statuses) else "todo"


def _tally(target: dict, key: str, state: str) -> None:
    row = target.setdefault(key, {"total": 0, "done": 0, "progress": 0})
    row["total"] += 1
    if state != "todo":
        row[state] += 1


def video_stats(video: dict, state: dict, units: list[dict]) -> dict:
    """Summarize the progress of a video with events.

    :param video: ``project.json`` entry with non-empty ``events``.
    :param state: saved state of the video (``frames`` keyed by string index).
    :param units: base planes of the video, from ``protocol_units``.
    :return: counts per frame, event, plane and class; a histogram of boxes per reviewed
        plane and completed planes per (day, reviewer), both with the plane classes so a
        class filter stays exact.
    """
    frames = state.get("frames", {})

    def status(n: int) -> str | None:
        return frames.get(str(n), {}).get("status")

    last = video["num_frames"] - 1
    by_class: dict[str, set[int]] = {}
    for e in video["events"]:
        by_class.setdefault(e["class_name"], set()).update(
            range(max(0, e["start"]), min(last, e["end"]) + 1)
        )
    event_frames = sorted(set().union(*by_class.values()))

    def frame_counts(indices) -> dict:
        counts = Counter(status(n) for n in indices)
        return {
            "total": len(indices),
            "reviewed": counts["reviewed"],
            "excluded": counts["excluded"],
            "draft": counts["draft"],
        }

    events, events_by_class = [], {}
    for e in video["events"]:
        start, end = max(0, e["start"]), min(last, e["end"])
        span = _state([status(n) for n in range(start, end + 1)])
        events.append({"class": e["class_name"], "frames": end - start + 1, "state": span})
        _tally(events_by_class, e["class_name"], span)

    planes, planes_by_class, boxes, completed = {}, {}, Counter(), Counter()
    for unit in units:
        span = _state([status(n) for n in range(unit["start"], unit["end"] + 1)])
        _tally(planes, "all", span)
        for name in unit["classes"]:
            _tally(planes_by_class, name, span)
        if span != "done":
            continue
        first = frames[str(unit["start"])]
        if first["status"] == "reviewed":
            boxes[(len(first.get("boxes", [])), tuple(unit["classes"]))] += 1
        # Completion = last modification of its frames: a later correction moves it.
        stamps = [
            frames[str(n)].get("updated_at", "") for n in range(unit["start"], unit["end"] + 1)
        ]
        completed[(max(stamps)[:10], first.get("reviewer", ""), tuple(unit["classes"]))] += 1

    return {
        "id": video["id"],
        "video_id": video["video_id"],
        "dataset": video["dataset"],
        "fps": video["fps"],
        "classes": sorted(by_class),
        "state": _state([status(n) for n in event_frames]),
        "frames": frame_counts(event_frames),
        "frames_by_class": {k: frame_counts(sorted(v)) for k, v in sorted(by_class.items())},
        "events": events,
        "events_by_class": events_by_class,
        "planes": planes.get("all", {"total": 0, "done": 0, "progress": 0}),
        "planes_by_class": planes_by_class,
        "boxes": [[n, list(classes), count] for (n, classes), count in sorted(boxes.items())],
        "completed_planes": [
            [day, who, list(classes), n] for (day, who, classes), n in sorted(completed.items())
        ],
        "revision": state.get("revision", 0),
        "updated_at": state.get("updated_at", ""),
    }
