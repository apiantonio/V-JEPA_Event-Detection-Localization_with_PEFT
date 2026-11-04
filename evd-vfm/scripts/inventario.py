"""Thin CLI for the shardable video inventory (F2)."""

from __future__ import annotations

import argparse
from pathlib import Path

from evd.data.inventory import (
    discover_videos,
    load_split_paths,
    merge_shards,
    scan_shard,
    shard_paths,
    write_shard,
)
from evd.utils import env


def _defaults() -> tuple[Path, Path]:
    return env.ucf_dir(), env.data_dir() / "ucf_crime_inventory"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="scan a deterministic slice of the videos")
    scan.add_argument("--root", type=Path, help="UCF_Crimes root (contains Videos/)")
    scan.add_argument("--output", type=Path, help="folder of the parts and of the final artifacts")
    scan.add_argument("--videos-subdir", default="Videos")
    scan.add_argument("--shard-index", type=int)
    scan.add_argument("--num-shards", type=int)
    scan.add_argument(
        "--metadata-only",
        action="store_true",
        help="do not count frames by decoding; only for quick trials, does not close F2",
    )

    merge = sub.add_parser("merge", help="merge all parts and generate CSV/manifest/report")
    merge.add_argument("--output", type=Path)
    merge.add_argument("--num-shards", type=int, required=True)
    merge.add_argument("--expected-count", type=int)
    merge.add_argument(
        "--split-file",
        action="append",
        type=Path,
        default=[],
        help="official split; repeatable, selects the canonical paths for the manifest",
    )
    merge.add_argument(
        "--split-prefix",
        default="Videos",
        help="prefix of the split paths relative to the scanned root",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    default_root, default_output = _defaults()
    output = args.output or default_output

    if args.command == "scan":
        root = args.root or default_root
        shard_index, num_shards = env.shard()
        if args.shard_index is not None:
            shard_index = args.shard_index
        if args.num_shards is not None:
            num_shards = args.num_shards
        all_paths = discover_videos(root / args.videos_subdir)
        selected = shard_paths(all_paths, shard_index, num_shards)
        print(
            f"inventory shard {shard_index}/{num_shards}: "
            f"{len(selected)} of {len(all_paths)} videos"
        )
        records = scan_shard(
            selected,
            dataset_root=root,
            verify_decode=not args.metadata_only,
        )
        part = write_shard(
            output,
            records,
            shard_index=shard_index,
            num_shards=num_shards,
        )
        print(f"written {part}")
        return 0

    canonical_paths = (
        load_split_paths(args.split_file, prefix=args.split_prefix) if args.split_file else None
    )
    summary = merge_shards(
        output,
        num_shards=args.num_shards,
        expected_count=args.expected_count,
        canonical_paths=canonical_paths,
    )
    print(
        f"inventory complete: {summary['n_videos']} videos, "
        f"{summary['n_pathological']} pathological, "
        f"{summary['total_duration_h']:.2f} hours"
    )
    print(f"artifacts in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
