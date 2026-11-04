"""CRIS fits of the localization campaign (P01): discovery, ranking shards, top concept.

``discover`` captures the keys of every head on the 30 train discovery chunks, segments
them in a spawn pool and fits one concept dictionary per head (elbow + Convex NMF).
``rank`` assigns the validation ranking chunks of one shard to those concepts and runs the
4,000 masked forwards per chunk (resumable per chunk). ``top`` averages the Eq. 6 scores
over the 30 ranking chunks and freezes the most important positive concept with its head's
dictionary and the total fit time. See :mod:`evd.exp.localization_cris`.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402

from evd.exp import localization_cris as lc  # noqa: E402
from evd.exp.localization_auto import read_json, write_json  # noqa: E402


def engine_and_hooks():
    import torch

    from evd.exp.demo import live_engine, open_run
    from evd.utils.hashing import file_sha256
    from localization_auto_preflight import CHECKPOINT_SHA, RUN

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no silent CPU fallback")
    torch.manual_seed(lc.SEED)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    run = open_run(Path(os.environ["EVD_RUNS_DIR"]) / RUN)
    if file_sha256(run.checkpoint_path) != CHECKPOINT_SHA:
        raise ValueError("Selected checkpoint identity mismatch")
    engine, _ = live_engine(run, device="cuda", batch_size=1, num_workers=0)
    backbone = engine.scorer.backbone.module()
    modules = lc.qkv_modules(backbone)
    heads = modules[0].out_features // 3 // 64
    if len(modules) != 24 or heads != 16:
        raise ValueError(f"Unexpected attention layout: {len(modules)} blocks, {heads} heads")
    hooks = lc.KeyHooks(modules, 16 * 24 * 24, heads)
    return engine, hooks


def capture(engine, hooks, row: dict, paths: dict):
    """:return: per-block keys ``N × heads × d`` (float32 numpy), grid and seconds."""
    import torch

    from evd.data.dataset import WindowItem
    from evd.exp.explanation_advanced import extract_features, prepare_clip, window_from_record

    start = time.perf_counter()
    hooks.capture_layers = set(range(len(hooks.modules)))
    clip = prepare_clip(engine, WindowItem(window_from_record(row), paths[row["video_id"]]))
    feats, _ = extract_features(engine, clip)
    torch.cuda.synchronize()
    keys = {layer: hooks.keys[layer][0].float().cpu().numpy() for layer in hooks.capture_layers}
    hooks.keys.clear()
    hooks.capture_layers = set()
    return clip, keys, tuple(feats.grid), time.perf_counter() - start


def paths_of(args) -> dict:
    paths = {}
    for manifest in (args.train_manifest, args.val_manifest):
        paths.update({row["video_id"]: row["path"] for row in read_json(manifest)})
    return paths


def discover(args) -> None:
    plan = read_json(args.plan)
    engine, hooks = engine_and_hooks()
    paths = paths_of(args)
    pool = lc.make_pool(args.workers)
    descriptors = {
        (layer, h): [] for layer in range(len(hooks.modules)) for h in range(hooks.heads)
    }
    seconds = {"capture": 0.0, "slic_cpu": 0.0}
    for number, row in enumerate(plan["discovery"]):
        clip, keys, grid, spent = capture(engine, hooks, row, paths)
        seconds["capture"] += spent
        futures = {
            key: pool.submit(lc.segment_head, keys[key[0]][:, key[1]].reshape(*grid, -1))
            for key in descriptors
        }
        for key, future in futures.items():
            _, values, cpu = future.result()
            descriptors[key].append(values)
            seconds["slic_cpu"] += cpu
        del clip, keys
        print(f"discovery {number + 1}/{len(plan['discovery'])}", flush=True)
    keys = sorted(descriptors)
    fitted = list(pool.map(lc.fit_head, [np.concatenate(descriptors[k]) for k in keys]))
    pool.shutdown()
    seconds["fit_cpu"] = sum(receipt["cpu_seconds"] for _, receipt in fitted)
    dictionaries = {
        f"L{layer:02d}H{h:02d}": d for (layer, h), (d, _) in zip(keys, fitted, strict=True)
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / "cris_dictionaries.npz", **dictionaries)
    write_json(
        args.output / "cris_discovery.json",
        {
            "seconds": seconds,
            "heads": {
                f"L{layer:02d}H{h:02d}": r for (layer, h), (_, r) in zip(keys, fitted, strict=True)
            },
            "concepts": int(sum(d.shape[0] for d in dictionaries.values())),
            "nmf_converged": int(sum(r["converged"] for _, r in fitted)),
            "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        },
    )
    print(f"concepts {sum(d.shape[0] for d in dictionaries.values())}", flush=True)


def offsets(dictionaries) -> tuple[dict, int]:
    """:return: the first global concept id of every head (``L..H..`` order) and the total."""
    start, table = 0, {}
    for name in sorted(dictionaries.files):
        table[name] = start
        start += dictionaries[name].shape[0]
    return table, start


def rank(args) -> None:
    plan = read_json(args.plan)
    dictionaries = np.load(args.output / "cris_dictionaries.npz")
    table, total = offsets(dictionaries)
    engine, hooks = engine_and_hooks()
    paths = paths_of(args)
    pool = lc.make_pool(args.workers)
    rows = [r for i, r in enumerate(plan["ranking"]) if i % args.shards == args.shard]
    for row in rows:
        key = hashlib.sha256(row["window_id"].encode()).hexdigest()[:24]
        target = args.output / "ranking" / f"{key}.npz"
        if target.exists():
            continue
        clip, keys, grid, capture_seconds = capture(engine, hooks, row, paths)
        futures = {
            (layer, h): pool.submit(
                lc.assign_head,
                keys[layer][:, h].reshape(*grid, -1),
                dictionaries[f"L{layer:02d}H{h:02d}"],
            )
            for layer in range(len(hooks.modules))
            for h in range(hooks.heads)
        }
        assign_cpu, ids = 0.0, []
        for layer in range(len(hooks.modules)):
            heads = []
            for head in range(hooks.heads):
                local, cpu = futures[(layer, head)].result()
                assign_cpu += cpu
                heads.append(local.reshape(-1).astype(np.int64) + table[f"L{layer:02d}H{head:02d}"])
            ids.append(np.stack(heads, axis=1))
        del keys
        seed = lc.SEED + int(key[:8], 16) % 100_000
        scores, receipt = lc.rank_video(engine.scorer, hooks, clip, ids, total, seed)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target, scores=scores)
        write_json(
            target.with_suffix(".json"),
            {
                "row": row,
                "seed": seed,
                "capture_seconds": capture_seconds,
                "assign_cpu_seconds": assign_cpu,
                **receipt,
            },
        )
        print(f"ranked {row['video_id']} in {receipt['forward_seconds']:.0f} s", flush=True)
    pool.shutdown()
    write_json(
        args.output / "operations" / f"rank-shard{args.shard:02d}.json",
        {"status": "complete", "chunks": len(rows)},
    )


def top(args) -> None:
    plan = read_json(args.plan)
    dictionaries = np.load(args.output / "cris_dictionaries.npz")
    table, total = offsets(dictionaries)
    keys = [hashlib.sha256(r["window_id"].encode()).hexdigest()[:24] for r in plan["ranking"]]
    scores = np.stack([np.load(args.output / "ranking" / f"{k}.npz")["scores"] for k in keys])
    receipts = [read_json(args.output / "ranking" / f"{k}.json") for k in keys]
    if scores.shape != (len(plan["ranking"]), total) or not all(r["finite"] for r in receipts):
        raise ValueError("Incomplete or non-finite CRIS ranking")
    mean = scores.mean(0)
    best = int(np.argmax(mean))
    discovery = read_json(args.output / "cris_discovery.json")
    fit = (
        discovery["seconds"]["capture"]
        + discovery["seconds"]["slic_cpu"]
        + discovery["seconds"]["fit_cpu"]
        + sum(
            r["capture_seconds"] + r["assign_cpu_seconds"] + r["forward_seconds"] for r in receipts
        )
    )
    head_name = max((n for n in table if table[n] <= best), key=lambda n: table[n])
    concept = best - table[head_name]
    if mean[best] <= 0:
        concept = -1
    np.savez(
        args.output / "cris_top.npz",
        layer=np.array(int(head_name[1:3])),
        head=np.array(int(head_name[4:6])),
        concept=np.array(concept),
        dictionary=dictionaries[head_name],
        fit_seconds=np.array(fit),
        heads=np.array(16),
        tokens=np.array(16 * 24 * 24),
        scores=mean,
    )
    order = np.argsort(-mean)[:10]
    write_json(
        args.output / "cris_top.json",
        {
            "layer_head": head_name,
            "concept": concept,
            "score": float(mean[best]),
            "top10": [{"id": int(i), "score": float(mean[i])} for i in order],
            "fit_seconds": fit,
            "concepts": total,
        },
    )
    print(f"top concept {head_name}#{concept} score {mean[best]:.4g}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=("discover", "rank", "top"))
    parser.add_argument("--plan", type=Path, required=True, help="cris-plan.json")
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="fits/cris")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    {"discover": discover, "rank": rank, "top": top}[args.action](args)


if __name__ == "__main__":
    main()
