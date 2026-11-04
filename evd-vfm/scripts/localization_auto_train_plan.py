"""Freeze the train sample of the automatic-localization campaign (decisions A2, V3, M4).

No decoding and no spatial ground truth: only the inventory, the temporal annotations of
the train split and the protocol grid. Train videos byte-identical to a validation or test
video, and duplicate train videos, are excluded as in the VTCD reference plans
(``scripts/vtcd_spatial_selected_plan.py``), so no fit sees a test video.

- ``positive``: one chunk per train event, with the existing ``event_rows`` rule
  (coverage of the event, distance from its centre, hash at seed 42), every class.
- ``normal``: the same number of chunks from fully normal train videos, balanced per video
  (round robin over videos in hash order).
- ``vtcd_reference``: the first ``q`` one-per-event chunks of each class in ``order_rows``
  order, ``q`` = the smallest per-class count (equal per class, V3).
- ``bag``: up to 16 evenly spaced chunks of every train video, label = the video contains
  an event (P32, M4).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from evd.annotations.schema import window_overlap  # noqa: E402
from evd.annotations.ucf_classes import ucf_classes  # noqa: E402
from evd.annotations.ucf_lab import UcfLabAnnotations  # noqa: E402
from evd.data.inventory import read_records  # noqa: E402
from evd.exp.vtcd_sampling import event_rows, order_rows, round_robin, stable  # noqa: E402
from evd.protocol.chunks import sampler_for_protocol  # noqa: E402
from evd.protocol.split import load_split  # noqa: E402
from evd.types import VideoMeta  # noqa: E402

BAG_SIZE = 16


def eligible_train_videos(inventory: dict, split) -> tuple[list[str], list[dict]]:
    """Train videos a fit may see: not byte-identical to a val/test video, one per content.

    :param inventory: video id to inventory record (with ``sha256``).
    :param split: the sealed ``edp_v2`` split.
    :return: the eligible train video ids and the excluded ones with their reason.
    """
    held_out = {inventory[v].sha256 for v in (*split.val, *split.test)}
    excluded, by_hash = [], collections.defaultdict(list)
    for vid in sorted(split.train):
        if inventory[vid].sha256 in held_out:
            excluded.append({"video_id": vid, "reason": "byte-identical to a val/test video"})
        else:
            by_hash[inventory[vid].sha256].append(vid)
    train = []
    for vids in by_hash.values():
        train.append(vids[0])
        excluded += [{"video_id": v, "reason": f"duplicate of {vids[0]}"} for v in vids[1:]]
    return train, excluded


def build(data_root: Path) -> dict:
    """:return: the frozen train sample, without hashes."""
    records = read_records(data_root / "ucf_crime_inventory/inventory.csv")
    inventory = {r.video_id: r for r in records}
    annotations = UcfLabAnnotations.from_dir(
        data_root / "annotations/fs_ucf_crime",
        num_frames={r.video_id: r.num_frames for r in records},
    ).load()
    split = load_split(ROOT / "manifests/edp_v2/ucf_crime/split.json")
    train, excluded = eligible_train_videos(inventory, split)
    protocol = yaml.safe_load((ROOT / "configs/protocol/edp_v2.yaml").read_text(encoding="utf-8"))
    sampler = sampler_for_protocol(None, protocol, validazione=False)
    names = [c.name for c in ucf_classes() if c.name != "Normal"]
    positives = {c: [] for c in names}
    normals, bags = collections.defaultdict(list), []
    for vid in sorted(train):
        r, ann = inventory[vid], annotations[vid]
        meta = VideoMeta(vid, r.relative_path, r.num_frames, r.fps, r.width, r.height)
        windows = sampler.windows_for_video(meta)
        rows = []
        for w in windows:
            a, b = w.frame_span
            count, dominant = window_overlap(ann, a, b)
            row = {
                "video_id": vid,
                "window_id": w.window_id,
                "window": asdict(w),
                "start": a,
                "end": b,
                "label": int(count > 0),
                "class_name": names[dominant] if count else "Normal",
            }
            rows.append(row)
            if count:
                positives[names[dominant]].append(row)
            elif ann.is_normal:
                normals[vid].append(row)
        picks = np.unique(
            np.linspace(0, len(rows) - 1, min(BAG_SIZE, len(rows))).round().astype(int)
        )
        bags += [{**rows[i], "bag_label": int(not ann.is_normal)} for i in picks]
    one_per_event, traces = {}, {}
    idx = {c: j for j, c in enumerate(names)}
    for c in names:
        chosen, trace = event_rows(positives[c], annotations, idx[c])
        one_per_event[c] = order_rows(chosen, annotations, idx[c])
        traces[c] = {
            "events": len(trace),
            "matched": sum(t["window_id"] is not None for t in trace),
        }
    positive = [row for c in names for row in one_per_event[c]]
    ordered = round_robin(
        {v: sorted(rows, key=lambda r: stable(r["window_id"])) for v, rows in normals.items()}
    )
    if len(ordered) < len(positive):
        raise ValueError("Not enough normal train chunks")
    quota = min(len(rows) for rows in one_per_event.values())
    return {
        "positive": positive,
        "normal": ordered[: len(positive)],
        "vtcd_reference": [row for c in names for row in one_per_event[c][:quota]],
        "bag": bags,
        "vtcd_quota_per_class": quota,
        "events_per_class": traces,
        "excluded_train_videos": excluded,
        "train_videos": len(train),
    }


def build_validation(data_root: Path) -> dict:
    """:return: every chunk of fully normal validation videos and validation bags.

    The validation grid is contiguous (``val_hop_s``). Normal chunks feed the A1
    calibrations and the P29/P30 checkpoint choice; bags (up to 16 evenly spaced chunks per
    validation video, label = the video has an event) feed the P32 checkpoint choice.
    """
    records = read_records(data_root / "ucf_crime_inventory/inventory.csv")
    inventory = {r.video_id: r for r in records}
    annotations = UcfLabAnnotations.from_dir(
        data_root / "annotations/fs_ucf_crime",
        num_frames={r.video_id: r.num_frames for r in records},
    ).load()
    split = load_split(ROOT / "manifests/edp_v2/ucf_crime/split.json")
    protocol = yaml.safe_load((ROOT / "configs/protocol/edp_v2.yaml").read_text(encoding="utf-8"))
    sampler = sampler_for_protocol(None, protocol, validazione=True)
    normal, bags = [], []
    for vid in sorted(split.val):
        r, ann = inventory[vid], annotations[vid]
        meta = VideoMeta(vid, r.relative_path, r.num_frames, r.fps, r.width, r.height)
        rows = [
            {"video_id": vid, "window_id": w.window_id, "window": asdict(w)}
            for w in sampler.windows_for_video(meta)
        ]
        if ann.is_normal:
            normal += rows
        picks = np.unique(
            np.linspace(0, len(rows) - 1, min(BAG_SIZE, len(rows))).round().astype(int)
        )
        bags += [{**rows[i], "bag_label": int(not ann.is_normal)} for i in picks]
    return {"normal": normal, "bag": bags}


CRIS_VIDEOS = 30


def balanced_videos(chunks_by_class: dict[str, list[dict]], total: int) -> list[dict]:
    """One chunk per video, ``total`` videos balanced over classes (round robin).

    Classes are visited in hash order (seed 42); within a class, chunks keep their
    ``order_rows`` order and a video already taken is skipped.

    :raises ValueError: if the classes cannot supply ``total`` distinct videos.
    """
    queues = {c: list(rows) for c, rows in chunks_by_class.items()}
    taken, chosen = set(), []
    while len(chosen) < total and any(queues.values()):
        for c in sorted(queues, key=stable):
            while queues[c] and queues[c][0]["video_id"] in taken:
                queues[c].pop(0)
            if queues[c] and len(chosen) < total:
                row = queues[c].pop(0)
                taken.add(row["video_id"])
                chosen.append(row)
    if len(chosen) < total:
        raise ValueError(f"Only {len(chosen)} distinct videos for CRIS")
    return chosen


def validation_positives(data_root: Path) -> tuple[dict[str, list[dict]], dict]:
    """Every temporally positive chunk of the anomalous validation videos, by dominant class.

    Validation grid (``val_hop_s``); a chunk is positive when it intersects an event (the
    protocol rule, ``window_overlap``).

    :return: rows per class name (13 anomaly classes) and the loaded annotations.
    """
    records = read_records(data_root / "ucf_crime_inventory/inventory.csv")
    inventory = {r.video_id: r for r in records}
    annotations = UcfLabAnnotations.from_dir(
        data_root / "annotations/fs_ucf_crime",
        num_frames={r.video_id: r.num_frames for r in records},
    ).load()
    split = load_split(ROOT / "manifests/edp_v2/ucf_crime/split.json")
    protocol = yaml.safe_load((ROOT / "configs/protocol/edp_v2.yaml").read_text(encoding="utf-8"))
    sampler = sampler_for_protocol(None, protocol, validazione=True)
    names = [c.name for c in ucf_classes() if c.name != "Normal"]
    positives = {c: [] for c in names}
    for vid in sorted(split.val):
        r, ann = inventory[vid], annotations[vid]
        if ann.is_normal:
            continue
        meta = VideoMeta(vid, r.relative_path, r.num_frames, r.fps, r.width, r.height)
        for w in sampler.windows_for_video(meta):
            a, b = w.frame_span
            count, dominant = window_overlap(ann, a, b)
            if count:
                positives[names[dominant]].append(
                    {
                        "video_id": vid,
                        "window_id": w.window_id,
                        "window": asdict(w),
                        "start": a,
                        "end": b,
                        "label": 1,
                        "class_name": names[dominant],
                    }
                )
    return positives, annotations


def build_cris(data_root: Path, train_plan: dict) -> dict:
    """:return: 30 train discovery chunks and 30 validation ranking chunks, class-balanced.

    Train: from the one-per-event positives of the frozen train sample. Validation: one
    chunk per validation event on the validation grid (``event_rows``), then one chunk per
    video. Decision of 16 October 2026: token grid, balanced validation videos.
    """
    by_class = defaultdict(list)
    for row in train_plan["positive"]:
        by_class[row["class_name"]].append(row)
    discovery = balanced_videos(by_class, CRIS_VIDEOS)
    positives, annotations = validation_positives(data_root)
    names = list(positives)
    idx = {c: j for j, c in enumerate(names)}
    ranking_pool = {}
    for c in names:
        chosen, _ = event_rows(positives[c], annotations, idx[c])
        ranking_pool[c] = order_rows(chosen, annotations, idx[c])
    ranking = balanced_videos(ranking_pool, CRIS_VIDEOS)
    return {"discovery": discovery, "ranking": ranking}


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation", action="store_true", help="build the validation plan")
    parser.add_argument("--cris-from", type=Path, help="train plan: build the CRIS plan")
    args = parser.parse_args()
    if args.cris_from is not None:
        plan = build_cris(args.data_root, json.loads(args.cris_from.read_text(encoding="utf-8")))
    else:
        plan = build_validation(args.data_root) if args.validation else build(args.data_root)
    body = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    plan["sha256"] = hashlib.sha256(body.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
    counts = {k: len(v) for k, v in plan.items() if isinstance(v, list)}
    print(json.dumps({"counts": counts, "quota": plan.get("vtcd_quota_per_class")}))


if __name__ == "__main__":
    main()
