"""Freeze reviewed annotation planes and evaluate sealed, GT-free predictions.

The snapshot stays local and never enters a prediction manifest. Exactly one
quality metric is produced: equal planes, then equal events, then equal classes.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402

from evd.eval.localization import PlaneIoU, aggregate_localization, union_box_iou  # noqa: E402
from evd.explain.project import offline_assignment, offline_plane  # noqa: E402
from evd.types import WindowSpec  # noqa: E402
from evd.utils.hashing import file_sha256  # noqa: E402
from localization_auto_preflight import write  # noqa: E402


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot(source: Path, destination: Path) -> None:
    """Freeze one read transaction, without generating folds or candidate rankings."""
    destination.mkdir(parents=True, exist_ok=False)
    project_bytes = (source / "project.json").read_bytes()
    project = json.loads(project_bytes)
    (destination / "project.json").write_bytes(project_bytes)
    connection = sqlite3.connect(
        (source / "annotations.sqlite3").resolve().as_uri() + "?mode=ro", uri=True
    )
    db = sqlite3.connect(destination / "annotations.sqlite3")
    connection.backup(db)
    connection.close()
    if (source / "project.json").read_bytes() != project_bytes:
        raise RuntimeError("The project changed during the snapshot")
    counts = Counter()
    support = {}
    excluded = []
    for video in project["videos"]:
        if not video["events"]:
            continue
        state_row = db.execute("SELECT body FROM states WHERE video=?", (video["id"],)).fetchone()
        unit_row = db.execute(
            "SELECT body FROM protocol_units WHERE video=?", (video["id"],)
        ).fetchone()
        frames = json.loads(state_row[0])["frames"] if state_row else {}
        units = json.loads(unit_row[0])["units"] if unit_row else []
        events = []
        for event_index, event in enumerate(video["events"]):
            category = event["class_name"]
            event_id = f"{video['dataset']}/{video['video_id']}/event{event_index:03d}"
            planes = []
            visited = set()
            for unit_index, unit in enumerate(units):
                begin = max(0, int(event["start"]), int(unit["start"]))
                end = min(video["num_frames"] - 1, int(event["end"]), int(unit["end"]))
                segments = []
                for frame in range(begin, end + 1):
                    if frame in visited:
                        raise ValueError("An event frame belongs to two annotation units")
                    visited.add(frame)
                    state = frames.get(str(frame), {})
                    if state.get("status") != "reviewed":
                        continue
                    boxes = sorted(
                        [
                            list(box["xyxy"])
                            for box in state.get("boxes", [])
                            if box["class_name"] == category
                        ]
                    )
                    if not boxes:
                        continue
                    for x0, y0, x1, y1 in boxes:
                        if not (0 <= x0 <= x1 < video["width"] and 0 <= y0 <= y1 < video["height"]):
                            raise ValueError("Ground-truth box outside native image")
                    if (
                        segments
                        and segments[-1]["end"] == frame - 1
                        and segments[-1]["boxes"] == boxes
                    ):
                        segments[-1]["end"] = frame
                    else:
                        segments.append({"start": frame, "end": frame, "boxes": boxes})
                if segments:
                    planes.append(
                        {"plane_id": f"annotation-unit{unit_index:06d}", "segments": segments}
                    )
            if planes:
                events.append({"event_id": event_id, "class": category, "planes": planes})
                counts[f"{video['dataset']}_events"] += 1
                counts[f"{video['dataset']}_planes"] += len(planes)
                support.setdefault(video["dataset"], Counter())[category] += 1
            else:
                excluded.append(
                    {
                        "event_id": event_id,
                        "class": category,
                        "reason": "no_reviewed_positive_spatial_plane",
                    }
                )
        if events:
            write(
                destination / video["dataset"] / f"{video['video_id']}.json",
                {
                    "video_id": video["video_id"],
                    "dataset": video["dataset"],
                    "width": video["width"],
                    "height": video["height"],
                    "num_frames": video["num_frames"],
                    "events": events,
                },
            )
    db.close()
    for dataset, classes in support.items():
        if len(classes) != 13:
            raise ValueError(f"Cannot evaluate thirteen classes on {dataset}: {classes}")
    write(
        destination / "snapshot.json",
        {
            "status": "complete",
            "counts": dict(counts),
            "class_event_support": support,
            "excluded_events": excluded,
            "quality_evaluated": False,
            "project_sha256": file_sha256(destination / "project.json"),
            "database_sha256": file_sha256(destination / "annotations.sqlite3"),
            "unit": "frozen_annotation_plane_intersected_with_event",
            "within_plane": "mean_IoU_over_reviewed_frames_with_positive_class_GT",
            "missing_GT_is_negative": False,
        },
    )


def chunk_windows(chunks: list[dict]) -> list[WindowSpec]:
    """:return: the protocol window of every chunk record of a sealed video, in order."""
    return [
        WindowSpec(
            **{
                **row["window"],
                "explicit_frame_indices": tuple(row["window"]["explicit_frame_indices"]),
            }
        )
        for row in chunks
    ]


def negative_planes(complete: dict, chunks: list[dict], intervals) -> dict[tuple[int, int], int]:
    """Chunk planes of the frames outside every temporal event: the negative images of real use.

    A frame goes to one chunk and plane as in :func:`assigned_frames` (policy ``fine``, the
    nearest sample's plane, ties to the previous sample as ``offline_plane``); the frames on the
    same chunk plane are one image, as the GT frames with the same boxes are (decision of 28
    October 2026). A frame no window covers is not an image.

    :param complete: the video's ``complete.json``.
    :param chunks: its chunk records, in protocol order.
    :param intervals: ``(start, end, ...)`` of every temporal event, native frames, ends
        included, any class.
    :return: ``{(chunk index, temporal plane): native frames on it}``.
    :raises ValueError: if the samples of a window do not split into its planes.
    """
    num_frames = int(complete["video"]["num_frames"])
    windows = chunk_windows(chunks)
    assignment = offline_assignment(windows, num_frames, politica="fine")
    outside = np.ones(num_frames, dtype=bool)
    for start, end, *_ in intervals:
        outside[max(0, int(start)) : min(num_frames - 1, int(end)) + 1] = False
    eligible = outside & (assignment >= 0)
    out: dict[tuple[int, int], int] = {}
    for index in np.unique(assignment[eligible]):
        window, planes = windows[index], int(chunks[index]["grid"][0])
        if planes < 1 or window.num_frames % planes:
            raise ValueError("The number of samples must be divisible by the temporal planes")
        frames = np.flatnonzero(eligible & (assignment == index))
        samples = np.asarray(window.frame_indices)
        nearest = np.argmin(np.abs(samples[None, :] - frames[:, None]), axis=1)
        temporal, counts = np.unique(nearest // (window.num_frames // planes), return_counts=True)
        out.update({(int(index), int(t)): int(n) for t, n in zip(temporal, counts, strict=True)})
    return out


def assigned_frames(truth: dict, complete: dict, chunks: list[dict]):
    """Yield every reviewed GT frame with the chunk plane that predicts it.

    Each frame goes to one chunk (``offline_assignment``, policy ``fine``) and to the
    temporal plane of that chunk nearest to it (``offline_plane``); a frame no window covers
    gets index and plane ``-1``.

    :return: ``(event, plane, frame, index, temporal, boxes)`` per frame, in GT order.
    :raises ValueError: if prediction and GT disagree on the video geometry.
    """
    for name in ("video_id", "width", "height", "num_frames"):
        if complete["video"][name] != truth[name]:
            raise ValueError(f"Video identity/geometry mismatch: {name}")
    windows = chunk_windows(chunks)
    assignment = offline_assignment(windows, truth["num_frames"], politica="fine")
    for event in truth["events"]:
        for plane in event["planes"]:
            for segment in plane["segments"]:
                for frame in range(segment["start"], segment["end"] + 1):
                    index = int(assignment[frame])
                    temporal = (
                        offline_plane(windows[index], frame, chunks[index]["grid"][0])
                        if index >= 0
                        else -1
                    )
                    yield event, plane, frame, index, temporal, segment["boxes"]


def evaluate_video(
    truth: dict, complete: dict, chunks: list[dict], experiment: str
) -> list[PlaneIoU]:
    """Project predictions independently, then measure each fixed annotated plane."""
    result = []
    cache = {}
    sums: dict[tuple[str, str], list[float]] = {}
    for event, plane, _, index, temporal, actual in assigned_frames(truth, complete, chunks):
        signature = index, temporal, tuple(tuple(box) for box in actual)
        if signature not in cache:
            predicted = (
                chunks[index]["methods"][experiment]["boxes"][temporal] if index >= 0 else []
            )
            cache[signature] = union_box_iou(predicted, actual)
        value = cache[signature]
        if value is None:
            raise ValueError("Eligible ground truth must be nonempty")
        sums.setdefault((event["event_id"], plane["plane_id"]), []).append(value)
    for event in truth["events"]:
        for plane in event["planes"]:
            values = sums.get((event["event_id"], plane["plane_id"]))
            if not values:
                raise ValueError("Frozen plane has no eligible frame")
            result.append(
                PlaneIoU(
                    event["event_id"], event["class"], plane["plane_id"], sum(values) / len(values)
                )
            )
    return result


def sealed_videos(predictions: Path, seal: dict, wanted=lambda entry: True):
    """Yield every sealed video, checking its checksums before reading it.

    :param wanted: ``entry -> bool``; the chunks of an unwanted video are not read (its
        ``complete.json`` is still checked) and its ``chunks`` is ``None``.
    :return: ``(entry, complete, chunks)`` per video of the seal.
    :raises ValueError: if a prediction changed after sealing.
    """
    for entry in seal["videos"]:
        target = predictions / entry["directory"]
        if file_sha256(target / "complete.json") != entry["complete_sha256"]:
            raise ValueError("Prediction changed after sealing")
        complete = read(target / "complete.json")
        if not wanted(entry):
            yield entry, complete, None
            continue
        chunks = []
        for chunk in complete["chunks"]:
            path = target / chunk["path"]
            if file_sha256(path) != chunk["sha256"]:
                raise ValueError("Prediction chunk changed after sealing")
            chunks.append(read(path))
        yield entry, complete, chunks


def evaluate(
    snapshot_path: Path,
    predictions: Path,
    seal_path: Path,
    dataset: str,
    experiment: str,
    output: Path,
) -> None:
    """Require a full-dataset prediction seal before accessing spatial quality."""
    seal = read(seal_path)
    if seal.get("status") != "complete" or seal["dataset"] != dataset:
        raise ValueError("Predictions must be complete and sealed for the whole dataset")
    if experiment not in seal["methods"]:
        raise ValueError("Experiment absent from prediction seal")
    rows, fit_times, inference = [], set(), 0.0
    expected = {row["video_id"] for row in seal["videos"]}

    def truth_of(entry: dict) -> Path:
        return snapshot_path / dataset / f"{entry['video_id']}.json"

    for entry, complete, chunks in sealed_videos(predictions, seal, lambda e: truth_of(e).exists()):
        timing = complete["timing"][experiment]
        inference += timing["inference_compute_seconds"]
        fit_times.add(timing["fit_seconds"])
        if chunks is None:
            continue
        rows.extend(evaluate_video(read(truth_of(entry)), complete, chunks, experiment))
    if {path.stem for path in (snapshot_path / dataset).glob("*.json")} - expected:
        raise ValueError("A ground-truth video has no prediction")
    if len(fit_times) != 1:
        raise ValueError("Inconsistent shared fit timing")
    classes = sorted(read(snapshot_path / "snapshot.json")["class_event_support"][dataset])
    result = aggregate_localization(rows, classes)
    result.update(
        experiment=experiment,
        dataset=dataset,
        timing={"fit_seconds": fit_times.pop(), "inference_compute_seconds": inference},
        prediction_seal_sha256=file_sha256(seal_path),
        gt_snapshot_sha256=file_sha256(snapshot_path / "snapshot.json"),
    )
    write(output, result)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    freeze = subs.add_parser("snapshot")
    freeze.add_argument("--source", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    scoring = subs.add_parser("evaluate")
    scoring.add_argument("--snapshot", type=Path, required=True)
    scoring.add_argument("--predictions", type=Path, required=True)
    scoring.add_argument("--seal", type=Path, required=True)
    scoring.add_argument("--dataset", choices=("ucf_crime", "mivia_crime"), required=True)
    scoring.add_argument("--experiment", required=True)
    scoring.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "snapshot":
        snapshot(args.source, args.output)
    else:
        evaluate(
            args.snapshot, args.predictions, args.seal, args.dataset, args.experiment, args.output
        )


if __name__ == "__main__":
    main()
