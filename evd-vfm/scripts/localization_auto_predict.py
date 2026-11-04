"""Resumable, ground-truth-free predictions for preregistered localization readouts.

Inputs are video metadata, a validation-derived gate and the frozen detector.
No prediction path reads test annotations, event classes or evaluation results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from evd.exp.demo import live_engine, open_run  # noqa: E402
from evd.exp.explanation_advanced import extract_features, prepare_clip  # noqa: E402
from evd.explain.attribution import explain_head  # noqa: E402
from evd.explain.localization import MaskRule, map_mask, native_boxes  # noqa: E402
from evd.protocol.chunks import sampler_for_protocol  # noqa: E402
from evd.types import VideoMeta  # noqa: E402
from evd.utils.hashing import file_sha256  # noqa: E402
from localization_auto_preflight import CHECKPOINT_SHA, READOUTS, RUN, write  # noqa: E402


def read(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def key(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def sync_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def validate_inputs(rows: list[dict]) -> None:
    """Reject annotation-bearing manifests instead of ignoring their extra fields."""
    allowed = {field.name for field in fields(VideoMeta)}
    if not rows or len({row["video_id"] for row in rows}) != len(rows):
        raise ValueError("Empty manifest or duplicate videos")
    for row in rows:
        if set(row) != allowed:
            raise ValueError(f"Prediction input must contain only VideoMeta fields: {set(row)}")
        if min(row["width"], row["height"], row["fps"], row["num_frames"]) <= 0:
            raise ValueError("Invalid video geometry")


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", choices=("ucf_test", "mivia_test"), required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        raise ValueError("Invalid shard")
    rows = read(args.manifest)
    validate_inputs(rows)
    preflight = read(args.preflight / "gpu-preflight.json")
    gate = read(args.preflight / "gate.json")
    if preflight["status"] != "complete" or gate["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise ValueError("Missing faithful validation preflight")
    plan_path = ROOT / "configs/localization_auto_20261016.json"
    plan = read(plan_path)
    if not set(READOUTS) <= {row["id"] for row in plan["experiments"]}:
        raise ValueError("An implementation is absent from the frozen experiment plan")
    identity = {
        "schema": 1,
        "dataset": args.dataset,
        "source_commit": args.source_commit,
        "plan_sha256": file_sha256(plan_path),
        "input_manifest_sha256": file_sha256(args.manifest),
        "gate_sha256": file_sha256(args.preflight / "gate.json"),
        "checkpoint_sha256": CHECKPOINT_SHA,
        "methods": READOUTS,
        "mask_rule": asdict(MaskRule()),
        "projection": "fine",
        "test_labels_available": False,
    }
    # Hash distribution is reproducible and does not infer a class from a video ID.
    selected = [row for row in rows if int(key(row["video_id"]), 16) % args.shards == args.shard]
    selected.sort(key=lambda row: row["video_id"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no silent CPU fallback")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run = open_run(Path(os.environ["EVD_RUNS_DIR"]) / RUN)
    if file_sha256(run.checkpoint_path) != CHECKPOINT_SHA:
        raise ValueError("Selected checkpoint identity mismatch")
    before = sync_time()
    engine, spec = live_engine(run, device="cuda", batch_size=1, num_workers=0)
    engine.sampler = sampler_for_protocol(spec, run.protocol, validazione=False)
    setup_seconds = sync_time() - before
    runtime = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "precision": engine.scorer.precision,
        "tf32": False,
        "batch_size": 1,
        "threads": 1,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
    }
    for number, row in enumerate(selected):
        meta = VideoMeta(**row)
        target = args.output / args.dataset / key(meta.video_id)
        target.mkdir(parents=True, exist_ok=True)
        if (target / "complete.json").exists():
            complete = read(target / "complete.json")
            if complete["identity"] != identity or complete["video"] != row:
                raise ValueError("Existing video has incompatible provenance")
            for item in complete["chunks"]:
                if file_sha256(target / item["path"]) != item["sha256"]:
                    raise ValueError("Corrupt saved prediction")
            continue
        items = engine.finestre(Path(meta.path), meta)
        chunks = []
        for index, item in enumerate(items):
            destination = target / "chunks" / f"{index:06d}.json"
            if destination.exists():
                record = read(destination)
                if record["identity"] != identity or record["window"] != json.loads(
                    json.dumps(asdict(item.window))
                ):
                    raise ValueError("Existing chunk has incompatible provenance")
            else:
                before = sync_time()
                clip = prepare_clip(engine, item)
                decoded = sync_time()
                features, logit = extract_features(engine, clip)
                encoded = sync_time()
                probability = float(torch.sigmoid(torch.tensor(logit)).item())
                accepted = probability > gate["threshold"]
                methods = {}
                raw = {}
                for experiment, readout in READOUTS.items():
                    before_method = sync_time()
                    if accepted:
                        with engine.scorer.autocast():
                            attribution = explain_head(
                                engine.scorer.head, features, extra_cams=True
                            )
                        if abs(attribution.logit - logit) > 1e-5:
                            raise RuntimeError("Attribution altered the detector logit")
                        values = attribution.maps[readout]
                        raw[experiment] = values
                        mask = map_mask(values, MaskRule())
                        boxes = native_boxes(mask, meta.width, meta.height)
                    else:
                        boxes = [[] for _ in range(features.grid[0])]
                    method_seconds = sync_time() - before_method
                    methods[experiment] = {
                        "boxes": boxes,
                        "attribution_and_boxes_seconds": method_seconds,
                        "inference_compute_seconds": encoded - before + method_seconds,
                    }
                before_io = time.perf_counter()
                map_path = target / "maps" / f"{index:06d}.npz"
                map_path.parent.mkdir(exist_ok=True)
                with map_path.with_suffix(".npz.tmp").open("wb") as stream:
                    np.savez_compressed(stream, **raw)
                map_path.with_suffix(".npz.tmp").replace(map_path)
                record = {
                    "identity": identity,
                    "window": asdict(item.window),
                    "window_id": item.window.window_id,
                    "grid": features.grid,
                    "logit": logit,
                    "score": probability,
                    "gate_passed": accepted,
                    "methods": methods,
                    "timing": {
                        "decode_preprocess_seconds": decoded - before,
                        "encode_score_seconds": encoded - decoded,
                        "map_write_seconds": time.perf_counter() - before_io,
                        "shared_compute_charged_in_full_to_each_method": True,
                        "readouts_timed_separately": True,
                    },
                    "runtime": runtime,
                    "maps_sha256": file_sha256(map_path),
                }
                write(destination, record)
                del clip, features
            chunks.append(
                {"path": str(destination.relative_to(target)), "sha256": file_sha256(destination)}
            )
        timing = {}
        records = [read(target / item["path"]) for item in chunks]
        for experiment in READOUTS:
            timing[experiment] = {
                "fit_seconds": gate["fit_seconds"],
                "fit_scope": "shared_validation_gate_only; no learned attribution parameters",
                "inference_compute_seconds": sum(
                    record["methods"][experiment]["inference_compute_seconds"] for record in records
                ),
                "inference_chunks": len(records),
                "fit_shared_cost_do_not_sum_over_videos": True,
            }
        write(
            target / "complete.json",
            {
                "status": "complete",
                "identity": identity,
                "video": row,
                "chunks": chunks,
                "timing": timing,
                "prediction_scope": "entire_video_all_protocol_windows",
            },
        )
        print(
            json.dumps(
                {
                    "video": meta.video_id,
                    "status": "complete",
                    "chunks": len(chunks),
                    "progress": [number + 1, len(selected)],
                }
            ),
            flush=True,
        )
    write(
        args.output / "operations" / f"{args.dataset}-readouts-shard{args.shard:02d}.json",
        {
            "status": "complete",
            "identity": identity,
            "shard": args.shard,
            "shards": args.shards,
            "videos": [row["video_id"] for row in selected],
            "setup_seconds": setup_seconds,
            "setup_scope": "one_shared_model_load_per_job; separate_from_steady_state_inference",
            "runtime": runtime,
        },
    )


if __name__ == "__main__":
    main()
