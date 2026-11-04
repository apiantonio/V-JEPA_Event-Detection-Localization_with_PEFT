"""Extract train features of the frozen sample for the fits of the localization campaign.

``a2`` (first pass) encodes the positive, normal and VTCD-reference chunks of
``train-plan.json`` once each and saves, per chunk: the mean token (1024), the per-channel
spatial variance averaged over planes (P45/P46 channel choice), 32 token pairs at the same
spatial position on consecutive planes (64 tokens, positions seeded by the window id: PCA,
banks, Gaussian, P28 transitions, P29/P30 training) and the logit; VTCD references also keep
all their tokens in fp16 for the dictionary fit on CPU.

``project`` (second pass) encodes the train P32 bags and normal chunks, and the validation
normal chunks and bags of ``val-plan.json``, saving all their tokens projected on the train
standardization + PCA-128 (fp16) and their mean token: the representation of the memory
methods, of their A1 calibration and of the P29/P30/P32 checkpoint choice. Both passes
resume per chunk. Test files are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402

from evd.exp.localization_auto import in_shard, read_json, write_json  # noqa: E402

PAIRS = 32


def pair_positions(window_id: str, grid: tuple[int, int, int]) -> np.ndarray:
    """:return: ``2*PAIRS`` flat token indices, pairs ``(t,h,w)`` and ``(t+1,h,w)``."""
    t, h, w = grid
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(window_id.encode()).digest()[:4], "big")
    )
    anchors = rng.choice((t - 1) * h * w, size=PAIRS, replace=False)
    first = anchors  # flat index with t in [0, t-2] since the plane-major layout is t*h*w
    return np.stack([first, first + h * w], axis=1).reshape(-1)


def channel_spatial_variance(tokens: np.ndarray, grid: tuple[int, int, int]) -> np.ndarray:
    """:return: per-channel variance over spatial positions, averaged over planes."""
    t, h, w = grid
    return tokens.reshape(t, h * w, -1).var(axis=1).mean(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=("a2", "project"))
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path, help="inputs-ucf-train.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pca", type=Path, help="pca.npz of the fits (project mode)")
    parser.add_argument("--val-plan", type=Path, help="validation plan (project mode)")
    parser.add_argument("--val-manifest", type=Path, help="inputs-ucf-val.json (project mode)")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    import torch

    from evd.data.dataset import WindowItem
    from evd.exp.demo import live_engine, open_run
    from evd.exp.explanation_advanced import extract_features, prepare_clip, window_from_record
    from evd.utils.hashing import file_sha256
    from localization_auto_preflight import CHECKPOINT_SHA, RUN

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no silent CPU fallback")
    torch.manual_seed(42)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan = read_json(args.plan)
    paths = {row["video_id"]: row["path"] for row in read_json(args.manifest)}
    if args.val_manifest is not None:
        paths.update({row["video_id"]: row["path"] for row in read_json(args.val_manifest)})
    if args.mode == "a2":
        roles = {}
        for role in ("positive", "normal", "vtcd_reference"):
            for row in plan[role]:
                roles.setdefault(row["window_id"], (row, set()))[1].add(role)
        jobs = [(row, sorted(r)) for row, r in roles.values()]
        pca = None
    else:
        jobs, seen = [], {}
        sources = [("train_" + role, plan[role]) for role in ("bag", "normal")]
        if args.val_plan is not None:
            validation = read_json(args.val_plan)
            sources += [("val_" + role, validation[role]) for role in ("normal", "bag")]
        for role, rows in sources:
            for row in rows:
                if row["window_id"] in seen:
                    seen[row["window_id"]].append(role)
                else:
                    seen[row["window_id"]] = [role]
                    jobs.append((row, seen[row["window_id"]]))
        pca = np.load(args.pca)
    run = open_run(Path(os.environ["EVD_RUNS_DIR"]) / RUN)
    if file_sha256(run.checkpoint_path) != CHECKPOINT_SHA:
        raise ValueError("Selected checkpoint identity mismatch")
    engine, _ = live_engine(run, device="cuda", batch_size=1, num_workers=0)
    identity = {
        "mode": args.mode,
        "plan_sha256": plan["sha256"],
        "checkpoint_sha256": CHECKPOINT_SHA,
        "source_commit": args.source_commit,
        "pca_sha256": file_sha256(args.pca) if pca is not None else None,
        "val_plan_sha256": file_sha256(args.val_plan) if args.val_plan else None,
    }
    out = args.output / args.mode
    mine = [
        job
        for job in jobs
        if in_shard(job[0]["video_id"] + job[0]["window_id"], args.shard, args.shards)
    ]
    for number, (row, roles) in enumerate(sorted(mine, key=lambda j: j[0]["window_id"])):
        key = hashlib.sha256(row["window_id"].encode()).hexdigest()[:24]
        target = out / "chunks" / f"{key}.npz"
        receipt = out / "receipts" / f"{key}.json"
        if target.exists() and receipt.exists():
            if read_json(receipt)["identity"] != identity:
                raise ValueError("Existing feature with another identity")
            continue
        start = time.perf_counter()
        item = WindowItem(window_from_record(row), paths[row["video_id"]])
        clip = prepare_clip(engine, item)
        feats, logit = extract_features(engine, clip)
        grid = tuple(feats.grid)
        tokens = feats.tokens[0].float().cpu().numpy()
        arrays = {"logit": np.float32(logit), "grid": np.asarray(grid)}
        if args.mode == "a2":
            positions = pair_positions(row["window_id"], grid)
            arrays.update(
                mean=tokens.mean(0),
                channel_variance=channel_spatial_variance(tokens, grid),
                pair_positions=positions,
                pair_tokens=tokens[positions].astype(np.float16),
            )
            if "vtcd_reference" in roles:
                arrays["tokens"] = tokens.astype(np.float16)
        else:
            projected = ((tokens - pca["mean"]) / pca["scale"]) @ pca["components"].T
            arrays.update(tokens_pca=projected.astype(np.float16), mean=tokens.mean(0))
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.with_suffix(".tmp").open("wb") as stream:
            np.savez(stream, **arrays)
        target.with_suffix(".tmp").replace(target)
        write_json(
            receipt,
            {
                "identity": identity,
                "row": row,
                "roles": roles,
                "seconds": time.perf_counter() - start,
                "gpu": torch.cuda.get_device_name(),
            },
        )
        del clip, feats
        if number % 50 == 0:
            print(json.dumps({"done": number + 1, "of": len(mine)}), flush=True)
    write_json(
        out / "operations" / f"shard{args.shard:02d}.json",
        {"status": "complete", "identity": identity, "chunks": len(mine)},
    )


if __name__ == "__main__":
    main()
