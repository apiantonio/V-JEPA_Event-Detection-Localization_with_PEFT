#!/usr/bin/env python
"""Turn the sealed partition into the video lists consumed by the extraction.

    python scripts/lista_video_split.py --split val test --out-dir "$EVD_DATA_DIR/edp_v2_liste"

The list is not the partition but a projection of it, and this script is the only place
producing it. The file read by ``--videos-from`` wants one identifier per line
(:func:`~evd.data.feature_extraction.read_video_ids` takes its stem); empty lines and
``#`` are allowed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evd.protocol.split import load_split  # noqa: E402

PARTITIONS = ("train", "val", "test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-json",
        type=Path,
        default=Path("manifests/edp_v2/ucf_crime/split.json"),
        help="the sealed partition; default: the edp_v2 one",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        choices=PARTITIONS,
        required=True,
        help="which partitions to project",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # `load_split` rather than `json.loads`: it recomputes the payload sha256 and compares it
    # with the declared field, checks the schema and rejects a few-shot protocol.
    split = load_split(args.split_json)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for partition in args.split:
        video = getattr(split, partition)
        path = args.out_dir / f"{partition}.txt"
        # The header carries the VERIFIED partition sha: a list without its origin cannot be
        # checked.
        header_spec = (
            f"# {args.split_json} :: {partition} — {len(video)} video\n"
            f"# split sha256 {split.sha256}\n"
            f"# protocol {split.protocol_version}, builder {split.builder_commit}\n"
        )
        path.write_text(header_spec + "\n".join(video) + "\n", encoding="utf-8")
        print(f"{path}: {len(video)} video")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
