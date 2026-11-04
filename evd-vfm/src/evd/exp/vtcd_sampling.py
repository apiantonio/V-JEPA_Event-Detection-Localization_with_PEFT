"""Selection of VTCD references without inference or spatial ground truth.

Contract: docs/varie/vtcd_sottoinsieme_significativo_20261010.md. Reuses the repository
EDP grid; order, events and quotas are traceable.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

Row = dict[str, Any]


def stable(value: str, seed: int = 42) -> bytes:
    """:return: a Python-independent sort key for the reference seed."""
    return hashlib.sha256((str(seed) + ":" + value).encode()).digest()


def round_robin(groups: dict[str, list], seed: int = 42) -> list:
    """:return: a prefix balanced over the groups, without added repetitions."""
    keys = sorted(groups, key=lambda k: stable(k, seed))
    return [
        groups[k][j]
        for j in range(max(map(len, groups.values()), default=0))
        for k in keys
        if j < len(groups[k])
    ]


def order_rows(rows: list[Row], annotations: dict, class_idx: int, seed: int = 42) -> list[Row]:
    """Sort by video/event diversity, then by chunk hash.

    :return: all chunks, each once, with nested prefixes.
    """
    by_video = defaultdict(list)
    for row in rows:
        by_video[row["video_id"]].append(row)
    arranged = {}
    for video, values in by_video.items():
        events = [
            (j, a, b) for j, (a, b, c) in enumerate(annotations[video].intervals) if c == class_idx
        ]
        groups = defaultdict(list)
        for row in values:
            event = max(
                events,
                key=lambda e: (
                    max(0, min(row["end"], e[2]) - max(row["start"], e[1]) + 1),
                    -abs(row["start"] + row["end"] - e[1] - e[2]),
                    -e[0],
                ),
            )
            groups[str(event[0])].append(row)
        arranged[video] = round_robin(
            {k: sorted(v, key=lambda r: stable(r["window_id"], seed)) for k, v in groups.items()},
            seed,
        )
    return round_robin(arranged, seed)


def event_rows(
    rows: list[Row], annotations: dict, class_idx: int, seed: int = 42
) -> tuple[list[Row], list[dict]]:
    """Match the maximum number of events to distinct chunks.

    Priority: event coverage within the window, distance from the centre, seeded hash.
    An augmenting path resolves collisions without duplicating a chunk; tie-breaks are
    deterministic and independent of the spatial ground truth.

    :return: distinct chunks and event/chunk matches, absences included.
    """
    by_video = defaultdict(list)
    for row in rows:
        by_video[row["video_id"]].append(row)
    result, assignments = [], []
    for video in sorted(by_video):
        candidates = {}
        lookup = {r["window_id"]: r for r in by_video[video]}
        events = [
            (j, a, b) for j, (a, b, c) in enumerate(annotations[video].intervals) if c == class_idx
        ]
        for j, a, b in events:
            eligible = [r for r in by_video[video] if min(r["end"], b) >= max(r["start"], a)]
            candidates[j] = [
                r["window_id"]
                for r in sorted(
                    eligible,
                    key=lambda r: (
                        -max(0, min(r["end"], b) - max(r["start"], a) + 1)
                        / (r["end"] - r["start"] + 1),
                        abs(r["start"] + r["end"] - a - b),
                        stable(r["window_id"], seed),
                    ),
                )
            ]
        owner = {}

        def assign(event, visited, links=candidates, assignment=owner):
            for wid in links[event]:
                if wid in visited:
                    continue
                visited.add(wid)
                if wid not in assignment or assign(assignment[wid], visited):
                    assignment[wid] = event
                    return True
            return False

        for event in sorted(candidates, key=lambda j: (len(candidates[j]), j)):
            assign(event, set())
        chosen = {event: wid for wid, event in owner.items()}
        for j, a, b in events:
            wid = chosen.get(j)
            assignments.append(
                {
                    "video_id": video,
                    "event_index": j,
                    "start": a,
                    "end": b,
                    "window_id": wid,
                    "candidates": len(candidates[j]),
                }
            )
            if wid is not None:
                result.append(lookup[wid])
    assert len({r["window_id"] for r in result}) == len(result)
    return result, assignments


def matched_events(
    full: list[Row],
    partial: list[Row],
    annotations: dict,
    class_idx: int,
    quantity: int,
    seed: int = 42,
) -> tuple[list[Row], list[Row], list[dict]]:
    """Freeze the same events with one full and one partial chunk each.

    :return: two lists of equal length and the provenance of the matches.
    :raises ValueError: if unique support is insufficient for the requested quantity.
    """
    by_type = []
    for values in (full, partial):
        events = defaultdict(list)
        for r in values:
            for j, (a, b, c) in enumerate(annotations[r["video_id"]].intervals):
                if c == class_idx and min(b, r["end"]) >= max(a, r["start"]):
                    events[(r["video_id"], j)].append(r)
        by_type.append(events)
    common = set(by_type[0]) & set(by_type[1])
    groups = defaultdict(list)
    for vid, event in common:
        groups[vid].append((vid, event))
    ordered = round_robin(
        {v: sorted(e, key=lambda t: stable(str(t), seed)) for v, e in groups.items()}, seed
    )
    selected = [[], []]
    used = [set(), set()]
    metadata = []
    for key in ordered:
        choices = []
        for typ in range(2):
            ranked = sorted(
                by_type[typ][key],
                key=lambda r: (
                    abs(r["class_fraction"] - (1.0 if typ == 0 else 0.5)),
                    stable(r["window_id"], seed),
                ),
            )
            choices.append(next((r for r in ranked if r["window_id"] not in used[typ]), None))
        if any(r is None for r in choices):
            continue
        for typ, row in enumerate(choices):
            used[typ].add(row["window_id"])
            selected[typ].append(row)
        metadata.append(
            {
                "video_id": key[0],
                "event_index": key[1],
                "full": choices[0]["window_id"],
                "partial": choices[1]["window_id"],
                "partial_fraction": choices[1]["class_fraction"],
            }
        )
        if len(metadata) == quantity:
            break
    if len(metadata) != quantity:
        raise ValueError(f"Only {len(metadata)} unique full/partial events, {quantity} requested")
    return selected[0], selected[1], metadata


def proportional(caps: list[int], total: int) -> list[int]:
    """:return: monotone proportional quotas, minimum 2, exact sum.
    :raises ValueError: if not all classes have support.
    """
    if total < 2 * len(caps) or total > sum(caps) or min(caps) < 2:
        raise ValueError("Proportional budget without support for all classes")
    counts = [2] * len(caps)
    for _ in range(total - sum(counts)):
        i = min(
            (i for i, m in enumerate(caps) if counts[i] < m),
            key=lambda i: ((counts[i] + 1) / caps[i], i),
        )
        counts[i] += 1
    return counts
