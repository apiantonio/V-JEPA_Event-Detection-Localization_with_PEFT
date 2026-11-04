"""Validation calibrations of the automatic-localization campaign (decision A1).

``ig-normals`` explains every chunk of every fully normal validation video (validation
grid, no gate: decision A1 of 16 October 2026) with the IG of the test stage and records
the maximum of the positive ensemble map; ``ig-threshold`` turns those maxima into the P16
threshold, the 0.99 quantile with equal weight per video. Validation labels are used only
to list the normal videos, as for the gate. No test file is read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402

from evd.exp.localization_auto import (  # noqa: E402
    in_shard,
    read_json,
    video_key,
    weighted_quantile,
    write_json,
)
from evd.utils.hashing import file_sha256  # noqa: E402

QUANTILE = 0.99


def normal_validation_videos(annotation_file: Path, validation: set[str]) -> set[str]:
    """:return: validation videos whose annotation line carries no event."""
    normal = set()
    for line in annotation_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts and Path(parts[0]).stem in validation and len(parts) == 1:
            normal.add(Path(parts[0]).stem)
    return normal


def ig_normals(args) -> None:
    import torch

    from evd.exp.demo import live_engine, open_run
    from evd.exp.explanation_advanced import prepare_clip
    from evd.exp.localization_ig import explain_ig
    from evd.protocol.chunks import sampler_for_protocol
    from evd.types import VideoMeta
    from localization_auto_preflight import CHECKPOINT_SHA, RUN

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no silent CPU fallback")
    torch.manual_seed(42)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run = open_run(Path(os.environ["EVD_RUNS_DIR"]) / RUN)
    if file_sha256(run.checkpoint_path) != CHECKPOINT_SHA:
        raise ValueError("Selected checkpoint identity mismatch")
    split = read_json(ROOT / "manifests/edp_v2/ucf_crime/split.json")
    annotations = (
        Path(run.cfg["data"]["annotations"]["path"]) / "FS-UCF-Crime_official_annotations.txt"
    )
    normal = normal_validation_videos(annotations, set(split["val"]))
    gate = read_json(args.preflight / "gate.json")
    if len(normal) != gate["normal_videos"]:
        raise ValueError("Normal validation videos differ from the gate population")
    rows = [
        row
        for row in read_json(args.preflight / "inputs-ucf-val.json")
        if row["video_id"] in normal
    ]
    engine, spec = live_engine(run, device="cuda", batch_size=1, num_workers=0)
    engine.sampler = sampler_for_protocol(spec, run.protocol, validazione=True)
    identity = {
        "calibration": "p16_ig_normals",
        "source_commit": args.source_commit,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "gate_sha256": file_sha256(args.preflight / "gate.json"),
    }
    selected = sorted(
        (row for row in rows if in_shard(row["video_id"], args.shard, args.shards)),
        key=lambda row: row["video_id"],
    )
    for row in selected:
        target = args.output / "videos" / f"{video_key(row['video_id'])}.json"
        if target.exists():
            if read_json(target)["identity"] != identity:
                raise ValueError("Existing calibration with another identity")
            continue
        meta = VideoMeta(**row)
        # One line per finished chunk, so a long normal video resumes after a time limit.
        partial = target.with_suffix(".partial.jsonl")
        done = {}
        if partial.exists():
            lines = partial.read_text(encoding="utf-8").splitlines()
            if json.loads(lines[0]) != identity:
                raise ValueError("Partial calibration with another identity")
            done = {r["window_id"]: r for r in map(json.loads, lines[1:]) if r}
        else:
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_text(json.dumps(identity) + "\n", encoding="utf-8")
        chunks = []
        for item in engine.finestre(Path(meta.path), meta):
            if item.window.window_id in done:
                chunks.append(done[item.window.window_id])
                continue
            torch.cuda.synchronize()
            start = time.perf_counter()
            clip = prepare_clip(engine, item)
            maps, checks, _, logit, _ = explain_ig(engine.scorer, spec, clip, item.window.window_id)
            torch.cuda.synchronize()
            record = {
                "window_id": item.window.window_id,
                "max_positive_ensemble": float(max(0.0, np.max(maps["ig_ensemble"]))),
                "logit": logit,
                "converged": {
                    k: checks[k]["converged"]
                    for k in checks
                    if k.startswith("ig_") and k != "ig_ensemble"
                },
                "steps": {
                    k: checks[k]["steps"]
                    for k in checks
                    if k.startswith("ig_") and k != "ig_ensemble"
                },
                "seconds": time.perf_counter() - start,
                "gpu": torch.cuda.get_device_name(),
            }
            with partial.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            chunks.append(record)
            del clip
        write_json(
            target,
            {
                "identity": identity,
                "video_id": meta.video_id,
                "chunks": chunks,
            },
        )
        partial.unlink()
        print(f"{meta.video_id}: {len(chunks)} chunks", flush=True)
    write_json(
        args.output / "operations" / f"shard{args.shard:02d}.json",
        {"status": "complete", "identity": identity},
    )


def ig_threshold(args) -> None:
    gate = read_json(args.preflight / "gate.json")
    documents = [read_json(path) for path in sorted((args.output / "videos").glob("*.json"))]
    identities = {str(sorted(d["identity"].items())) for d in documents}
    chunks = sum(len(d["chunks"]) for d in documents)
    if (
        len(documents) != gate["normal_videos"]
        or chunks != gate["normal_chunks"]
        or len(identities) != 1
    ):
        raise ValueError(
            f"Incomplete calibration: {len(documents)} videos, {chunks} chunks, "
            f"{len(identities)} identities"
        )
    values = {d["video_id"]: [c["max_positive_ensemble"] for c in d["chunks"]] for d in documents}
    start = time.perf_counter()
    threshold = weighted_quantile(values, QUANTILE)
    seconds = sum(c["seconds"] for d in documents for c in d["chunks"])
    converged = [all(c["converged"].values()) for d in documents for c in d["chunks"]]
    write_json(
        args.output / "p16.json",
        {
            "experiment": "P16-normal_q99",
            "threshold": threshold,
            "quantile": QUANTILE,
            "statistic": "max of the positive IG ensemble map per chunk",
            "weights": "equal_video_then_equal_chunk",
            "normal_videos": len(documents),
            "normal_chunks": chunks,
            "chunks_all_baselines_converged": int(sum(converged)),
            "fit_seconds": seconds + time.perf_counter() - start,
            "fit_scope": "decode, encode and IG of every normal validation chunk, plus the quantile",
            "identity": documents[0]["identity"],
        },
    )
    print(f"P16 threshold {threshold:.6g} on {chunks} chunks", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=("ig-normals", "ig-threshold"))
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--source-commit", default="")
    args = parser.parse_args()
    (ig_normals if args.action == "ig-normals" else ig_threshold)(args)


if __name__ == "__main__":
    main()
