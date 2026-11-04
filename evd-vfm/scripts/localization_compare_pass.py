"""GPU pass of the localization comparisons: Grad-CAM and the CRIS concepts of every chunk.

Comparison methods approved on 29 October 2026. For every chunk of a plan file of the class-routing
campaign (``plan/test.json``, the annotated test chunks, or ``plan/real_use/campaign.json``,
the flagged chunks without a test record) and the campaign detector, this script saves:

- the Grad-CAM map of the binary logit on the final post-norm features, the P38 of
  16 October (``explain_head(..., extra_cams=True)["gradcam"]``, the computation of
  ``evd.exp.localization_readouts``);
- for every head holding one of the ``--cris-top`` concepts of highest CRIS importance
  (16 October fits: ``cris_top.npz`` scores over ``cris_dictionaries.npz``, in the global id
  order of ``localization_auto_cris.offsets``), the head's keys segmented and assigned to its
  concepts (``localization_cris.assign_head``, the test-time rule of P01): the concept of
  every token.

One ``.json`` and one ``.npz`` per chunk under ``--output/<set>/``, named as the class-routing
pass; resumes per chunk; shards on the window id. No spatial ground truth is read.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402

from evd.exp.localization_auto import in_shard, read_json, write_json  # noqa: E402
from localization_class_pass import CONFIG, _key, rows_of  # noqa: E402


def cris_concepts(fits: Path, top: int) -> list[dict]:
    """:return: the ``top`` CRIS concepts by decreasing importance, each with its head.

    :raises ValueError: if the scores do not cover the concepts of the dictionaries.
    """
    from localization_auto_cris import offsets

    dictionaries = np.load(fits / "cris_dictionaries.npz")
    scores = np.load(fits / "cris_top.npz")["scores"]
    table, total = offsets(dictionaries)
    if len(scores) != total:
        raise ValueError(f"{len(scores)} CRIS scores for {total} concepts")
    out = []
    for rank, index in enumerate(np.argsort(-scores, kind="stable")[:top]):
        head = max((n for n in table if table[n] <= index), key=lambda n: table[n])
        out.append(
            {
                "rank": rank,
                "id": int(index),
                "head": head,
                "layer": int(head[1:3]),
                "head_index": int(head[4:6]),
                "concept": int(index - table[head]),
                "score": float(scores[index]),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="a plan file with rows")
    parser.add_argument("--set", required=True, help="name of the output folder (test, real)")
    parser.add_argument("--inputs", type=Path, nargs="*", default=[], help="rows with video paths")
    parser.add_argument("--cris-fits", type=Path, required=True, help="16 October fits/cris")
    parser.add_argument("--cris-top", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, help="only the first chunks (rehearsal)")
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    parser.add_argument("--workers", type=int, default=max(1, (cpus or 2) - 2))
    parser.add_argument("--source-commit", default="")
    args = parser.parse_args()

    import torch

    from evd.data.dataset import WindowItem
    from evd.exp.demo import live_engine, open_run
    from evd.exp.explanation_advanced import extract_features, prepare_clip, window_from_record
    from evd.exp.localization_cris import KeyHooks, assign_head, qkv_modules
    from evd.explain.attribution import explain_head
    from evd.utils.hashing import file_sha256
    from localization_auto_preflight import CHECKPOINT_SHA, RUN

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no silent CPU fallback")
    torch.manual_seed(42)
    torch.set_num_threads(1)
    config = read_json(CONFIG)
    runs = Path(os.environ["EVD_RUNS_DIR"])
    run_id = config["detectors"]["campaign"]
    run = open_run(runs / run_id)
    if run_id != RUN or file_sha256(run.checkpoint_path) != CHECKPOINT_SHA:
        raise ValueError("Campaign detector identity mismatch")
    engine, _ = live_engine(run, device="cuda", batch_size=1, num_workers=0)
    scorer = engine.scorer
    concepts = cris_concepts(args.cris_fits, args.cris_top)
    dictionaries = np.load(args.cris_fits / "cris_dictionaries.npz")
    heads = sorted({c["head"] for c in concepts})
    backbone = scorer.backbone.module()
    modules = qkv_modules(backbone)
    hooks = KeyHooks(modules, 16 * 24 * 24, modules[0].out_features // 3 // 64)
    hooks.capture_layers = {int(h[1:3]) for h in heads}
    paths = {}
    for file in args.inputs:
        paths.update({r["video_id"]: r["path"] for r in read_json(file)})
    plan = read_json(args.plan)
    rows = [r for r in rows_of(plan) if in_shard(r["window_id"], args.shard, args.shards)]
    if args.limit is not None:
        rows = rows[: args.limit]
    identity = {
        "purpose": "localization_compare_pass",
        "plan_sha256": plan["sha256"],
        "set": args.set,
        "detector": run_id,
        "checkpoint_sha256": file_sha256(run.checkpoint_path),
        "gradcam": "explain_head extra_cams gradcam (P38-postnorm)",
        "cris_top_sha256": file_sha256(args.cris_fits / "cris_top.npz"),
        "cris_dictionaries_sha256": file_sha256(args.cris_fits / "cris_dictionaries.npz"),
        "cris_concepts": concepts,
        "source_commit": args.source_commit,
    }
    out = args.output / args.set
    out.mkdir(parents=True, exist_ok=True)
    pool = ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    )

    def finish(row: dict, record: dict, futures: dict) -> None:
        stem = out / _key(row)
        arrays = {"gradcam": record.pop("gradcam")}
        for head, future in futures.items():
            ids, seconds = future.result()
            arrays[f"cris_{head}"] = ids.astype(np.int16)
            record["seconds"][f"cris_cpu_{head}"] = seconds
        with stem.with_suffix(".tmp").open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        stem.with_suffix(".tmp").replace(stem.with_suffix(".npz"))
        write_json(stem.with_suffix(".json"), {"identity": identity, **record})

    pending = []
    for row in rows:
        stem = out / _key(row)
        if stem.with_suffix(".json").exists():
            if read_json(stem.with_suffix(".json"))["identity"] != identity:
                raise ValueError(f"Existing record with another identity: {stem}")
            continue
        torch.cuda.synchronize()
        start = time.perf_counter()
        clip = prepare_clip(
            engine, WindowItem(window_from_record(row), row.get("path") or paths[row["video_id"]])
        )
        torch.cuda.synchronize()
        decoded = time.perf_counter()
        hooks.keys.clear()
        features, logit = extract_features(engine, clip)
        torch.cuda.synchronize()
        encoded = time.perf_counter()
        with scorer.autocast():
            attribution = explain_head(scorer.head, features, extra_cams=True)
        torch.cuda.synchronize()
        cam = time.perf_counter()
        if abs(attribution.logit - logit) > 1e-5:
            raise RuntimeError("Attribution altered the detector logit")
        grid = tuple(features.grid)
        futures = {}
        for head in heads:
            keys = hooks.keys[int(head[1:3])][0, :, int(head[4:6])].float().cpu().numpy()
            futures[head] = pool.submit(assign_head, keys.reshape(*grid, -1), dictionaries[head])
        record = {
            "dataset": row["dataset"],
            "video_id": row["video_id"],
            "window_id": row["window_id"],
            "class_name": row.get("class_name"),
            "index": row.get("index"),
            "grid": list(grid),
            "logit": logit,
            "seconds": {
                "decode": decoded - start,
                "forward": encoded - decoded,
                "gradcam": cam - encoded,
            },
            "gradcam": attribution.maps["gradcam"].astype(np.float32),
            "gpu": torch.cuda.get_device_name(),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
        }
        pending.append((row, record, futures))
        del clip, features, attribution
        hooks.keys.clear()
        while len(pending) > 2 * args.workers or (
            pending and all(f.done() for f in pending[0][2].values())
        ):
            finish(*pending.pop(0))
    for item in pending:
        finish(*item)
    pool.shutdown()
    hooks.close()
    if args.limit is None:
        write_json(
            args.output / "operations" / f"{args.set}-shard{args.shard:02d}.json",
            {"status": "complete", "identity": identity, "chunks": len(rows)},
        )
    print(json.dumps({"set": args.set, "chunks": len(rows), "heads": heads}))


if __name__ == "__main__":
    main()
