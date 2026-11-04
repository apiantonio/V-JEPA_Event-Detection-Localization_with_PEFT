"""Saved localization boxes of a run, as the demonstrator draws them.

``scripts/export_demo_localization.py`` writes ``runs/<run_id>/localization/``: an index
with the mean and per-class IoU of every evaluated method and, per video, the boxes already
projected onto native frames (``fine``, as in the evaluator). Here they are only read,
checked against the video and cut to the annotated intervals, the frames on which the
campaign measures them and on which the proposal renders draw them. No model is involved
(ADR-015).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

#: Folder of the run holding the export.
FOLDER = "localization"

#: Video fields that must match the opened file, or the boxes describe another video.
GEOMETRY = ("num_frames", "width", "height")


def export_dir(run_dir: str | Path, metrics: dict[str, Any]) -> Path | None:
    """Return the run folder holding the localization export that applies to a run.

    An external evaluation (``<run_id>__mivia``, ADR-066) reuses the weights of its source
    run, named by ``threshold.trasferita_da``: the boxes were exported there, once per model.

    :param run_dir: the opened run.
    :param metrics: its ``metrics.json``.
    :return: the folder containing ``localization/index.json``, or ``None``.
    """
    run_dir = Path(run_dir)
    candidates = [run_dir]
    source = (metrics.get("threshold") or {}).get("trasferita_da")
    if source:
        if run_dir.parent.name == source:
            candidates.append(run_dir.parent)
        candidates.append(run_dir.parent / str(source))
    for candidate in candidates:
        if (candidate / FOLDER / "index.json").is_file():
            return candidate
    return None


def exported_videos(run_dir: str | Path, metrics: dict[str, Any]) -> frozenset[str]:
    """:return: the videos with saved localization boxes for this run (empty if none)."""
    base = export_dir(run_dir, metrics)
    if base is None:
        return frozenset()
    index = json.loads((base / FOLDER / "index.json").read_text(encoding="utf-8"))
    return frozenset(index["videos"])


def for_video(
    run_dir: str | Path,
    video_id: str,
    *,
    num_frames: int,
    width: int,
    height: int,
    intervals: Sequence[tuple[int, int]],
) -> dict[str, Any] | None:
    """Return the localization block of the payload for one video.

    :param run_dir: the run folder.
    :param video_id: the opened video.
    :param num_frames: native frames of the opened file.
    :param width: native width of the opened file, in pixels.
    :param height: native height of the opened file, in pixels.
    :param intervals: annotated intervals, inclusive native frames; boxes outside them are
        dropped.
    :return: methods with their IoU (and, when exported, the readable ``label``, the
        ``campaign`` and the IoU of this video's events), the default method, a table of box
        lists in native inclusive ``[x0, y0, x1, y1]`` pixels, per method the
        ``[start, end, index]`` segments inside the intervals and, under ``truth``, the
        spatial ground truth the IoU was computed on (``None`` without one); ``None`` if the
        run has no export or the video has no localized chunk.
    :raises ValueError: if the exported video has a different geometry from the opened file.
    """
    root = Path(run_dir) / FOLDER
    index_path = root / "index.json"
    if not index_path.is_file():
        return None
    index = json.loads(index_path.read_text(encoding="utf-8"))
    relative = index["videos"].get(video_id)
    if relative is None:
        return None
    data = json.loads((root / relative).read_text(encoding="utf-8"))
    opened = {"num_frames": num_frames, "width": width, "height": height}
    for name in GEOMETRY:
        if int(data["video"][name]) != int(opened[name]):
            raise ValueError(
                f"The saved boxes of {video_id} describe a video with {name}="
                f"{data['video'][name]}, the opened file has {opened[name]}: they belong to "
                "another copy of the video and are not shown"
            )
    table, segments = clip_segments(data["boxes"], data["segments"], intervals)
    return {
        "truth": _truth(root, index, video_id, data["video"], intervals),
        "run_id": index["run_id"],
        "campaign": index["campaign"],
        "dataset": data["dataset"],
        "metric": index["metric"],
        "selection": index["selection"],
        "default": index["default"],
        "classes": index["classes"],
        "coordinate_space": index["coordinate_space"],
        "projection": index["projection"],
        "scope": "annotated intervals only",
        "methods": [_method_entry(method, video_id) for method in index["methods"]],
        "boxes": table,
        "segments": segments,
    }


def _method_entry(method: dict[str, Any], video_id: str) -> dict[str, Any]:
    """:return: what the page needs of a method, its event IoU cut to the opened video.

    The per-event IoU covers all evaluated videos; the page embeds one video only.
    """
    entry = {
        key: method[key]
        for key in ("id", "label", "campaign", "family", "family_label")
        if key in method
    }
    for dataset in ("ucf_crime", "mivia_crime"):
        block = {k: v for k, v in method[dataset].items() if k in ("rank", "mean", "classes")}
        events = method[dataset].get("events")
        if events is not None:
            block["events"] = {k: v for k, v in events.items() if k.split("/")[-2] == video_id}
        entry[dataset] = block
    return entry


def _truth(
    root: Path,
    index: dict[str, Any],
    video_id: str,
    video: dict[str, Any],
    intervals: Sequence[tuple[int, int]],
) -> dict[str, Any] | None:
    """Return the spatial ground truth of the video, the one the IoU was computed on.

    :return: box table, ``[start, end, index]`` segments inside the intervals, the evaluated
        events with their annotated interval (empty in older exports) and the snapshot
        digest, or ``None`` if the export carries no ground truth for the video.
    :raises ValueError: if its geometry differs from the exported boxes.
    """
    relative = (index.get("ground_truth") or {}).get("videos", {}).get(video_id)
    if relative is None:
        return None
    data = json.loads((root / relative).read_text(encoding="utf-8"))
    for name in GEOMETRY:
        if int(data["video"][name]) != int(video[name]):
            raise ValueError(f"The ground-truth boxes of {video_id} disagree on {name}")
    table, clipped = clip_segments(data["boxes"], {"truth": data["segments"]}, intervals)
    return {
        "boxes": table,
        "segments": clipped["truth"],
        "events": data.get("events", []),
        "frames": index["ground_truth"]["frames"],
        "snapshot_sha256": index["ground_truth"]["snapshot_sha256"],
    }


def clip_segments(
    boxes: Sequence[Sequence[Sequence[int]]],
    segments: dict[str, Sequence[Sequence[int]]],
    intervals: Sequence[tuple[int, int]],
) -> tuple[list, dict[str, list[list[int]]]]:
    """Keep only the frames inside the annotated intervals, re-indexing the box table.

    :param boxes: the box-list table the segments point into.
    :param segments: per method, inclusive ``[start, end, index]`` segments in frame order.
    :param intervals: inclusive annotated intervals; they may overlap.
    :return: the reduced table and the clipped segments, still in frame order.
    """
    merged: list[list[int]] = []
    for start, end in sorted((int(a), int(b)) for a, b in intervals):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    table: list = []
    refs: dict[int, int] = {}
    clipped: dict[str, list[list[int]]] = {}
    for method, rows in segments.items():
        out: list[list[int]] = []
        for start, end, ref in rows:
            for a, b in merged:
                lo, hi = max(start, a), min(end, b)
                if lo > hi:
                    continue
                if ref not in refs:
                    refs[ref] = len(table)
                    table.append(boxes[ref])
                out.append([lo, hi, refs[ref]])
        clipped[method] = out
    return table, clipped
