"""Freeze the chunk lists of the class-routing localization campaign (CPU, no decoding).

Plan: ``docs/varie/piano_localizzazione_per_classe_20261021.md`` §6.2, §7.1, §7.2. Reads the
inventory, the temporal annotations, the sealed split, the annotated-chunk targets of
17 October (indices and window ids only) and the frozen train plan; no spatial ground truth.
Writes in ``--output``:

- ``test.json``: the annotated test chunks of FS-UCF-Crime and MIVIA-Crime, with window,
  path and dominant temporal class (``None`` if the chunk intersects no temporal event);
- ``val.json``: every temporally positive chunk of the anomalous validation videos;
- ``references.json``: the A1 references (``vtcd_reference`` of the train plan, 16 per class);
- ``train_pure.json``: per class, the train chunks fully covered by events, on the eligible
  train videos of the train plan (no copy of a val/test video); a class with fewer than 16
  uses its A1 references instead (decision of 21 October);
- ``pilot.json``: 5 annotated test chunks per class (both test sets), in seed-42 hash order.

Every file carries the sha256 of its body.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import yaml  # noqa: E402

from evd.annotations.schema import window_overlap  # noqa: E402
from evd.annotations.ucf_classes import ucf_classes  # noqa: E402
from evd.annotations.ucf_lab import MIVIA_CRIME, UcfLabAnnotations  # noqa: E402
from evd.data.inventory import read_records  # noqa: E402
from evd.exp.localization_auto import read_json, write_json  # noqa: E402
from evd.exp.localization_class import MIN_CHUNKS  # noqa: E402
from evd.exp.vtcd_sampling import stable  # noqa: E402
from evd.protocol.chunks import sampler_for_protocol  # noqa: E402
from evd.protocol.split import load_split  # noqa: E402
from evd.types import VideoMeta  # noqa: E402
from localization_auto_train_plan import eligible_train_videos, validation_positives  # noqa: E402

PILOT_PER_CLASS = 5
CLASSES = [c.name for c in ucf_classes() if c.name != "Normal"]


def sealed(body: dict) -> dict:
    """:return: ``body`` with the sha256 of its canonical JSON."""
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return {**body, "sha256": hashlib.sha256(text.encode()).hexdigest()}


def row(dataset: str, meta: VideoMeta, window, count: int, dominant) -> dict:
    """:return: the record of one chunk, as ``window_from_record`` reads it."""
    a, b = window.frame_span
    return {
        "dataset": dataset,
        "video_id": meta.video_id,
        "window_id": window.window_id,
        "window": asdict(window),
        "start": a,
        "end": b,
        "class_name": CLASSES[dominant] if count else None,
        "coverage": count / (b - a + 1),
    }


def test_rows(targets: dict, manifests: dict, annotations: dict, protocol: dict) -> list[dict]:
    """:return: the annotated test chunks with their window, path and temporal class.

    :raises ValueError: if a target window id differs from the protocol grid.
    """
    sampler = sampler_for_protocol(None, protocol, validazione=False)
    rows = []
    for dataset, videos in targets.items():
        metas = {m["video_id"]: m for m in manifests[dataset]}
        for video_id, entry in sorted(videos.items()):
            m = metas[video_id]
            meta = VideoMeta(
                video_id, m["path"], m["num_frames"], m["fps"], m["width"], m["height"]
            )
            windows = sampler.windows_for_video(meta)
            for index, window_id in zip(entry["indices"], entry["window_ids"], strict=True):
                window = windows[index]
                if window.window_id != window_id:
                    raise ValueError(f"Target window differs from the grid: {video_id} {index}")
                count, dominant = window_overlap(annotations[dataset][video_id], *window.frame_span)
                rows.append(
                    {
                        **row(dataset, meta, window, count, dominant),
                        "index": index,
                        "path": m["path"],
                        "width": m["width"],
                        "height": m["height"],
                    }
                )
    return rows


def pure_rows(data_root: Path, protocol: dict) -> tuple[dict[str, list[dict]], list[dict]]:
    """:return: per class the fully covered train chunks of eligible videos, and the
    excluded train videos."""
    records = read_records(data_root / "ucf_crime_inventory/inventory.csv")
    inventory = {r.video_id: r for r in records}
    annotations = UcfLabAnnotations.from_dir(
        data_root / "annotations/fs_ucf_crime",
        num_frames={r.video_id: r.num_frames for r in records},
    ).load()
    split = load_split(ROOT / "manifests/edp_v2/ucf_crime/split.json")
    train, excluded = eligible_train_videos(inventory, split)
    sampler = sampler_for_protocol(None, protocol, validazione=False)
    pure = {c: [] for c in CLASSES}
    for vid in sorted(train):
        r = inventory[vid]
        meta = VideoMeta(vid, r.relative_path, r.num_frames, r.fps, r.width, r.height)
        for window in sampler.windows_for_video(meta):
            a, b = window.frame_span
            count, dominant = window_overlap(annotations[vid], a, b)
            if count >= b - a + 1:
                pure[CLASSES[dominant]].append(row("ucf_train", meta, window, count, dominant))
    return pure, excluded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="UCF inventory/annotations")
    parser.add_argument("--mivia-annotations", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True, help="annotated chunks of 17/10")
    parser.add_argument("--manifest-ucf", type=Path, required=True, help="VideoMeta rows")
    parser.add_argument("--manifest-mivia", type=Path, required=True, help="VideoMeta rows")
    parser.add_argument("--train-plan", type=Path, required=True, help="frozen train-plan.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = yaml.safe_load((ROOT / "configs/protocol/edp_v2.yaml").read_text(encoding="utf-8"))
    manifests = {
        "ucf_test": read_json(args.manifest_ucf),
        "mivia_test": read_json(args.manifest_mivia),
    }
    records = read_records(args.data_root / "ucf_crime_inventory/inventory.csv")
    annotations = {
        "ucf_test": UcfLabAnnotations.from_dir(
            args.data_root / "annotations/fs_ucf_crime",
            num_frames={r.video_id: r.num_frames for r in records},
        ).load(),
        "mivia_test": UcfLabAnnotations.from_dir(
            args.mivia_annotations,
            delivery=MIVIA_CRIME,
            num_frames={m["video_id"]: m["num_frames"] for m in manifests["mivia_test"]},
        ).load(),
    }
    test = test_rows(read_json(args.targets), manifests, annotations, protocol)
    by_class, _ = validation_positives(args.data_root)
    val = sorted(
        ({**r, "dataset": "ucf_val"} for rows in by_class.values() for r in rows),
        key=lambda r: (r["video_id"], r["window"]["end_frame"]),
    )
    train_plan = read_json(args.train_plan)
    references = [{**r, "dataset": "ucf_train"} for r in train_plan["vtcd_reference"]]
    pure, excluded = pure_rows(args.data_root, protocol)
    sources = {}
    for name in CLASSES:
        if len(pure[name]) < MIN_CHUNKS:
            sources[name] = {"source": "a1_references", "pure_chunks": len(pure[name])}
            pure[name] = [r for r in references if r["class_name"] == name]
        else:
            sources[name] = {"source": "pure", "pure_chunks": len(pure[name])}
    pilot = []
    for name in CLASSES:
        # Both test sets: FS-UCF-Crime alone has 4 annotated Abuse chunks.
        pool = sorted(
            (r for r in test if r["class_name"] == name), key=lambda r: stable(r["window_id"])
        )
        if len(pool) < PILOT_PER_CLASS:
            raise ValueError(f"Only {len(pool)} annotated test chunks of class {name}")
        pilot += pool[:PILOT_PER_CLASS]
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "test.json": {
            "rows": test,
            "targets_sha256": hashlib.sha256(args.targets.read_bytes()).hexdigest(),
        },
        "val.json": {"rows": val},
        "references.json": {"rows": references, "train_plan_sha256": train_plan["sha256"]},
        "train_pure.json": {"by_class": pure, "sources": sources, "excluded_videos": excluded},
        "pilot.json": {"rows": pilot, "per_class": PILOT_PER_CLASS, "seed": 42},
    }
    for name, body in outputs.items():
        write_json(args.output / name, sealed(body))
    print(
        json.dumps(
            {
                "test": len(test),
                "test_without_temporal_class": sum(r["class_name"] is None for r in test),
                "val": len(val),
                "references": len(references),
                "train_pure": {k: len(v) for k, v in pure.items()},
                "pilot": len(pilot),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
