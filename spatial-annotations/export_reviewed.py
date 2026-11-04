"""Export reviewed Forma annotations without the working database, drafts or machine paths."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import tempfile
import zipfile
from collections import Counter
from pathlib import Path


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode())


def normalized(frame: dict, width: int, height: int) -> list[dict]:
    boxes = []
    for box in frame.get("boxes", []):
        x0, y0, x1, y1 = box["xyxy"]
        if not (0 <= x0 <= x1 < width and 0 <= y0 <= y1 < height):
            raise ValueError("Box outside native image")
        boxes.append(
            {
                "class_name": box["class_name"],
                "xyxy": [x0, y0, x1, y1],
                "origin": box.get("origin", "unspecified"),
            }
        )
    return sorted(boxes, key=lambda b: (b["class_name"], b["xyxy"], b["origin"]))


def export(project: Path, output: Path, archives: Path) -> dict:
    project_bytes = (project / "project.json").read_bytes()
    manifest = json.loads(project_bytes)
    counts: dict[str, Counter] = {}
    files = []
    rows = []
    with tempfile.TemporaryDirectory(prefix="forma-reviewed-release-") as temporary:
        snapshot = Path(temporary) / "annotations.sqlite3"
        source = sqlite3.connect(
            (project / "annotations.sqlite3").resolve().as_uri() + "?mode=ro", uri=True
        )
        database = sqlite3.connect(snapshot)
        source.backup(database)
        source.close()
        database_digest = sha(snapshot)
        for video in manifest["videos"]:
            if video["split"] != "test":
                continue
            row = database.execute(
                "SELECT body FROM states WHERE video=?", (video["id"],)
            ).fetchone()
            frames = json.loads(row[0])["frames"] if row else {}
            reviewed = {int(k): v for k, v in frames.items() if v["status"] == "reviewed"}
            if not reviewed:
                continue
            dataset = video["dataset"]
            count = counts.setdefault(dataset, Counter())
            count["videos"] += 1
            count["reviewed_frames"] += len(reviewed)
            count["reviewed_empty_frames"] += sum(not f["boxes"] for f in reviewed.values())
            count["box_frame_instances"] += sum(len(f["boxes"]) for f in reviewed.values())
            events = video["events"]
            completed = sum(
                all(i in reviewed for i in range(e["start"], e["end"] + 1)) for e in events
            )
            count["completed_events"] += completed
            count["temporal_events_in_included_videos"] += len(events)
            segments = []
            for index, frame in sorted(reviewed.items()):
                if not 0 <= index < video["num_frames"]:
                    raise ValueError("Frame outside video")
                boxes = normalized(frame, video["width"], video["height"])
                unit = {
                    k: v
                    for k, v in frame.get("review_unit", {}).items()
                    if k in {"id", "start", "end", "mode"}
                }
                if (
                    segments
                    and segments[-1]["end"] == index - 1
                    and segments[-1]["boxes"] == boxes
                    and segments[-1]["review_unit"] == unit
                ):
                    segments[-1]["end"] = index
                else:
                    segments.append(
                        {"start": index, "end": index, "boxes": boxes, "review_unit": unit}
                    )
            data = {
                "schema": 1,
                "coordinate_convention": "native_pixels_xyxy_inclusive_zero_based",
                "frame_intervals": "inclusive",
                "missing_frames": "unannotated, not negative",
                **{
                    k: video[k]
                    for k in (
                        "dataset",
                        "video_id",
                        "split",
                        "width",
                        "height",
                        "num_frames",
                        "fps",
                        "sha256",
                    )
                },
                "events": events,
                "reviewed_segments": segments,
            }
            target = output / dataset / f"{video['video_id']}.json"
            write(target, data)
            # Independent on-disk expansion: verify every source reviewed frame and rectangle.
            loaded = json.loads(target.read_bytes())
            expanded = {}
            for segment in loaded["reviewed_segments"]:
                for index in range(segment["start"], segment["end"] + 1):
                    if index in expanded:
                        raise ValueError("Overlapping exported segments")
                    expanded[index] = segment["boxes"]
            assert expanded.keys() == reviewed.keys()
            assert all(
                expanded[i] == normalized(f, video["width"], video["height"])
                for i, f in reviewed.items()
            )
            count["segments"] += len(segments)
            files.append(
                {
                    "path": target.relative_to(output).as_posix(),
                    "sha256": sha(target),
                    "bytes": target.stat().st_size,
                }
            )
            rows.append(
                {
                    "dataset": dataset,
                    "video_id": video["video_id"],
                    "reviewed_frames": len(reviewed),
                    "reviewed_empty_frames": sum(not f["boxes"] for f in reviewed.values()),
                    "box_frame_instances": sum(len(f["boxes"]) for f in reviewed.values()),
                    "completed_events": completed,
                    "temporal_events": len(events),
                    "segments": len(segments),
                }
            )
        database.close()
    if (project / "project.json").read_bytes() != project_bytes:
        raise RuntimeError("Project manifest changed during export")
    summary = {
        "schema": 1,
        "source_project_sha256": hashlib.sha256(project_bytes).hexdigest(),
        "database_snapshot_sha256": database_digest,
        "status_filter": "reviewed only",
        "verification": "Every exported frame and rectangle expanded and compared against the consistent source snapshot",
        "datasets": {k: dict(v) for k, v in counts.items()},
        "files": files,
    }
    write(output / "manifest.json", summary)
    with (output / "videos.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    archives.mkdir(parents=True, exist_ok=True)
    for dataset in counts:
        with zipfile.ZipFile(
            archives / f"{dataset}-reviewed-spatial-annotations.zip", "w", zipfile.ZIP_DEFLATED
        ) as archive:
            for path in sorted((output / dataset).glob("*.json")):
                archive.write(path, f"spatial-annotations/{dataset}/{path.name}")
            for filename in ["manifest.json", "videos.csv", "README.md", "export_reviewed.py"]:
                if (output / filename).exists():
                    archive.write(output / filename, f"spatial-annotations/{filename}")
            if (output.parent / "LICENSE").exists():
                archive.write(output.parent / "LICENSE", "LICENSE")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--archives", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.project, args.output, args.archives)
    print(json.dumps(result["datasets"], indent=2))
