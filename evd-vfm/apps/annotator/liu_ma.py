"""Adapt the public Liu & Ma keyframes to the Forma import format.

Source: ``xuzero/UCFCrime_BoundingBox_Annotation``, ``Test_annotation.pkl``. No
interpolation and no assignment to VTCD planes: every exchange window contains only the
original frame. Undocumented conventions stay explicit in the provenance, together with
the original rows.
"""

from __future__ import annotations

import csv
import io
import json
import pickle
import re
from pathlib import Path

from apps.annotator.project import event_classes, import_proposals, sha

COMMIT = "f8caf93ee8632dfb2c04a7221d1e79fca4bea4c8"
TEST_SHA256 = "0339095e546514c9bac5584b1383919eca942ec198c203d7ce1f488cff9d5517"
REPOSITORY = "https://github.com/xuzero/UCFCrime_BoundingBox_Annotation"
DOWNLOAD = f"https://raw.githubusercontent.com/xuzero/UCFCrime_BoundingBox_Annotation/{COMMIT}/Test_annotation.pkl"
LABEL = "Liu & Ma 2019 · original samples"


class _DataOnly(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> object:
        raise ValueError("The pickle must contain data only, without classes or functions.")


def read_test(data: bytes) -> dict:
    """Read only the verified public version of the test set.

    :return: dictionary video -> original rows.
    :raises ValueError: on a different hash, or unexpected types or paths.
    """
    if sha(data) != TEST_SHA256:
        raise ValueError(
            "Test_annotation.pkl differs from the verified version: check the release."
        )
    raw = _DataOnly(io.BytesIO(data), encoding="latin1").load()
    if not isinstance(raw, dict):
        raise ValueError("Expected a dictionary of videos.")
    for name, rows in raw.items():
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"\w+", name)
            or not isinstance(rows, list)
        ):
            raise ValueError("Invalid identity or annotation list.")
        for row in rows:
            if (
                not isinstance(row, list)
                or len(row) != 5
                or not isinstance(row[0], str)
                or not re.search(rf"/Anomaly_Test/\w+/{re.escape(name)}/image_\d+\.jpg$", row[0])
                or any(type(n) is not int for n in row[1:])
            ):
                raise ValueError(f"Unrecognised row in {name}.")
    return raw


def convert_video(video: dict, rows: list, *, frame_offset: int = 0) -> tuple[dict, str, dict]:
    """Convert sparse boxes compatible with the project events.

    The number in the image name is kept (default offset zero). Coordinates are kept, with
    only the extremes clipped to the 320/240 border: the source does not state whether the
    extremes are inclusive, exclusive or one-based. This adaptation is meant for review, not
    to certify a ground truth.

    :param frame_offset: explicit index correction, recorded in the metadata.
    :return: metadata, a CSV compatible with ``import_proposals``, and a discard report.
    :raises ValueError: if the video is not a 320x240 UCF test video, or on duplicates or
        invalid geometry.
    """
    if video["dataset"] != "ucf_crime" or video["split"] != "test":
        raise ValueError("Liu & Ma Test can only be imported into the UCF-Crime test set.")
    if (video["width"], video["height"]) != (320, 240):
        raise ValueError("The verified version requires native 320x240 videos; no implicit resize.")
    report = {
        "video": video["video_id"],
        "total": len(rows),
        "accepted": 0,
        "border_clipped": 0,
        "excluded": [],
        "frame_offset": frame_offset,
    }
    metadata = {
        "schema": 1,
        "source_type": "external_keyframes",
        "sorgente": "Liu & Ma 2019",
        "variante": "annotati",
        "geometry": "squash",
        "size": 384,
        "grid": [1, 24, 24],
        "nativo": {k: video[k] for k in ("width", "height", "num_frames", "fps")},
        "video_id": video["video_id"],
        "finestre": [],
        "provenance": {
            "repository": REPOSITORY,
            "commit": COMMIT,
            "file": "Test_annotation.pkl",
            "sha256": TEST_SHA256,
            "video_sha256": video["sha256"],
            "frame_offset": frame_offset,
            "frame_rule": "filename_number_plus_offset",
            "coordinate_rule": "preserve_values_clip_image_boundary",
            "conventions_confirmed_by_authors": False,
            "temporal_policy": "observed_frames_only",
            "token_grid": "derived_from_edp_v2_not_upstream",
            "original_rows": rows,
        },
    }
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["video_id", "frame", "x0", "y0", "x1", "y1", "window_id", "piano", "box"])
    seen = set()
    for row in sorted(rows, key=lambda r: int(re.search(r"image_(\d+)\.jpg$", r[0])[1])):
        frame = int(re.search(r"image_(\d+)\.jpg$", row[0])[1]) + frame_offset
        x0, y0, x1, y1 = row[1:]
        if frame in seen:
            raise ValueError(f"Repeated frame: {video['video_id']}:{frame}")
        seen.add(frame)
        if not 0 <= frame < video["num_frames"] or not (
            0 <= x0 < x1 <= 320 and 0 <= y0 < y1 <= 240
        ):
            raise ValueError(f"Invalid geometry: {video['video_id']}:{frame}")
        if not event_classes(video, frame):
            report["excluded"].append(
                {"frame": frame, "reason": "outside_lab_events", "original_row": row}
            )
            continue
        coords = [x0, y0, min(x1, 319), min(y1, 239)]
        report["border_clipped"] += coords != row[1:]
        window = f"{video['video_id']}@liu-ma-frame-{frame}"
        metadata["finestre"].append(
            {"window_id": window, "piani": [{"t": 0, "box": [{"nativo": coords}]}]}
        )
        writer.writerow([video["video_id"], frame, *coords, window, 0, 1])
        report["accepted"] += 1
    metadata["audit"] = {k: v for k, v in report.items() if k != "excluded"}
    csv_text = buffer.getvalue()
    import_proposals(video, metadata, csv_text, "liu-ma-conversion-check")
    return metadata, csv_text, report


def prepare_collection(project: dict, data: bytes, output: Path, *, frame_offset: int = 0) -> dict:
    """Prepare importable pairs and an audit, without modifying the project.

    :return: per-video report and counts of the actual coverage.
    :raises ValueError: on an incompatible release, video or geometry.
    """
    raw = read_test(data)
    videos = {
        v["video_id"]: v
        for v in project["videos"]
        if v["dataset"] == "ucf_crime" and v["split"] == "test"
    }
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for name, rows in sorted(raw.items()):
        if name not in videos:
            reports.append(
                {"video": name, "total": len(rows), "accepted": 0, "reason": "not_in_project"}
            )
            continue
        meta, text, report = convert_video(videos[name], rows, frame_offset=frame_offset)
        reports.append(report)
        if not report["accepted"]:
            continue
        folder = output / name / "liu_ma_2019"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "boxes.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (folder / "boxes_per_frame.csv").write_text(text, encoding="utf-8")
    result = {
        "repository": REPOSITORY,
        "commit": COMMIT,
        "sha256": TEST_SHA256,
        "source_videos": len(raw),
        "source_boxes": sum(map(len, raw.values())),
        "importable_videos": sum(r["accepted"] > 0 for r in reports),
        "importable_boxes": sum(r["accepted"] for r in reports),
        "videos": reports,
        "test_videos_without_source": sorted(set(videos) - set(raw)),
        "conventions": "Filename index unchanged plus explicit offset; coordinates preserved with image-border clipping. Author conventions undocumented; review required.",
    }
    (output / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "Test_annotation.pkl").write_bytes(data)
    return result
