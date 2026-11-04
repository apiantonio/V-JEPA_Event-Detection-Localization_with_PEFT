#!/usr/bin/env python
"""Thin CLI for the sharded feature store (F4)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evd.exp.feature_jobs import (  # noqa: E402
    ExtractConfig,
    merge_feature_shards,
    run_extract_job,
)
from evd.models.backbones.vjepa21_source import ARCHS  # noqa: E402
from evd.utils import env  # noqa: E402


def _layers(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"non-integer layer: {value!r}") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("the layer list is empty")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    extract = commands.add_parser("extract", help="estimate or extract a shard of videos")
    extract.add_argument("--videos-from", type=Path, required=True)
    extract.add_argument("--inventory", type=Path)
    extract.add_argument("--dataset-root", type=Path)
    extract.add_argument("--features-root", type=Path)
    extract.add_argument("--arch", choices=sorted(ARCHS), default="vit_large")
    extract.add_argument("--checkpoint", type=Path)
    extract.add_argument("--repo", type=Path)
    extract.add_argument("--size", type=int, required=True)
    extract.add_argument(
        "--num-frames",
        type=int,
        required=True,
        help="samples per window. MUST match the run that will read the "
        "store (protocol.temporal.num_frames): it enters the preprocess_hash, so "
        "a different value writes the store under a hash no run looks up. "
        "edp_v1: 16; edp_v2: 32 (ADR-034). No default, because a "
        "wrong default is silent (ADR-049)",
    )
    extract.add_argument(
        "--preprocess-variant",
        choices=("eval_three_views", "eval_center", "demo_center", "squash"),
        required=True,
    )
    extract.add_argument("--preprocess-fps", type=float, required=True)
    extract.add_argument("--preprocess-dtype", choices=("fp16", "fp32"), required=True)
    extract.add_argument("--storage-dtype", choices=("fp16", "fp32"), required=True)
    extract.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        required=True,
        help="compute regime; bf16 is AMP/autocast with fp32 weights and inputs",
    )
    extract.add_argument(
        "--layers",
        type=_layers,
        required=True,
        help="layers to extract, comma-separated list. No default: the "
        "plausible default (the four ViT-L hierarchical_layers) costs four "
        "times the disk (1.32 TB instead of 330 GB) and three quarters would "
        "never be read, since E2c/E2d read layer 23 only (ADR-033 "
        "par. 2, ADR-049). For E2: 23",
    )
    extract.add_argument("--hop-s", type=float, required=True)
    extract.add_argument(
        "--grid",
        choices=("endpoint", "fixed"),
        default="endpoint",
        help="the window grid. MUST match the run that will read "
        "the store (protocol.temporal.window_grid): the key is the window, "
        "so a different grid is not a half-valid cache, it is zero hits. "
        "endpoint = edp_v1 (ADR-024); fixed = edp_v2, the "
        "laboratory protocol (ADR-047)",
    )
    extract.add_argument(
        "--window-s",
        type=float,
        help="nominal window duration; required with --grid fixed (edp_v2: 8.0)",
    )
    extract.add_argument("--batch-size", type=int)
    extract.add_argument("--shard-index", type=int)
    extract.add_argument("--num-shards", type=int)
    extract.add_argument(
        "--store-shard",
        type=int,
        help="which index-N.sqlite to write; default: --shard-index. Used to "
        "APPEND to an already populated store: the slice of the video list stays "
        "0..num_shards-1 (otherwise it is not fully covered) while the index "
        "files start from an offset that does not collide with the existing ones",
    )
    extract.add_argument("--max-gb", type=float, default=500.0)
    extract.add_argument("--device")
    extract.add_argument("--dry-run", action="store_true")

    merge = commands.add_parser("merge", help="merge the shard indexes")
    merge.add_argument("--store-dir", type=Path, required=True)
    merge.add_argument("--num-shards", type=int, required=True)
    merge.add_argument(
        "--run-id",
        default="e0d_ucf_vjepa21l_frozen-none_kna_s1",
    )
    merge.add_argument("--expected-count", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "merge":
        report = merge_feature_shards(
            store_dir=args.store_dir,
            num_shards=args.num_shards,
            run_id=args.run_id,
            expected_count=args.expected_count,
        )
    else:
        default_shard, default_count = env.shard()
        report = run_extract_job(
            ExtractConfig(
                videos_from=args.videos_from,
                inventory_csv=args.inventory
                or env.data_dir() / "ucf_crime_inventory" / "inventory.csv",
                dataset_root=args.dataset_root or env.ucf_dir(),
                features_root=args.features_root or env.features_dir(),
                arch=args.arch,
                checkpoint=args.checkpoint,
                repo=args.repo,
                size=args.size,
                num_frames=args.num_frames,
                preprocess_variant=args.preprocess_variant,
                preprocess_fps=args.preprocess_fps,
                preprocess_dtype=args.preprocess_dtype,
                storage_dtype=args.storage_dtype,
                precision=args.precision,
                layers=args.layers,
                hop_s=args.hop_s,
                grid=args.grid,
                window_s=args.window_s,
                batch_size=args.batch_size,
                shard_index=(default_shard if args.shard_index is None else args.shard_index),
                num_shards=(default_count if args.num_shards is None else args.num_shards),
                store_shard=args.store_shard,
                max_gb=args.max_gb,
                device=args.device or env.device(),
                dry_run=args.dry_run,
            )
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
