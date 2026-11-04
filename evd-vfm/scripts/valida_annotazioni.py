"""Validator CLI: annotations × F2 inventory -> ``report.json`` + ``report.md``.

``report.md`` is the artifact sent to the laboratory. Exit code 1 if there are
``error``-severity findings, so the script can be used as an automatic check.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evd.annotations.ucf_classes import class_from_relative_path, ucf_classes
from evd.annotations.ucf_lab import UcfLabAnnotations, delivery_of
from evd.annotations.ucf_sultani import UcfSultaniAnnotations
from evd.annotations.validate import validate
from evd.data.inventory import index_by_video_id, load_split_paths, read_records
from evd.utils.hashing import file_sha256
from evd.utils.io import write_json, write_text_atomic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("sultani", "lab"), required=True)
    parser.add_argument("--path", type=Path, required=True, help="file (sultani) or folder (lab)")
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("data/ucf_crime_inventory"),
        help="folder with inventory.csv and the official splits",
    )
    parser.add_argument("--out", type=Path, default=Path("data/annotations_report"))
    parser.add_argument(
        "--expect-split",
        action="append",
        default=[],
        help="official split whose videos must be annotated (repeatable)",
    )
    parser.add_argument("--index-base", type=int, default=0, help="sultani only")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    records = read_records(args.inventory / "inventory.csv")
    by_id = index_by_video_id(records)
    inventory = {video_id: record.to_meta() for video_id, record in by_id.items()}
    declared = {
        video_id: class_from_relative_path(record.relative_path)
        for video_id, record in by_id.items()
    }

    expect: list[str] = []
    for name in args.expect_split:
        split_file = name if Path(name).is_file() else args.inventory / name
        expect += [Path(p).stem for p in load_split_paths([split_file], prefix="")]

    if args.source == "sultani":
        source = UcfSultaniAnnotations(path=args.path, index_base=args.index_base)
        digest = file_sha256(args.path)
        name = args.path.name
    else:
        # `num_frames`, not `fps`: the laboratory file speaks in frames, and the Q20 clamp is what
        # matters (8 intervals end beyond the last frame). The validator reads `parse()`, which
        # does not clamp. The delivery in the folder decides the file name (FS-UCF-Crime and
        # MIVIA-Crime share the format with different file names).
        source = UcfLabAnnotations.from_dir(
            args.path,
            delivery=delivery_of(args.path),
            num_frames={k: m.num_frames for k, m in inventory.items()},
        )
        digest = file_sha256(source.path)
        name = f"{args.path.name}/{source.path.name}"

    report = validate(
        source.parse(),
        inventory=inventory,
        classes=ucf_classes(),
        expect_videos=expect or None,
        declared_class=declared,
        source_name=name,
        source_sha256=digest,
    )

    out = args.out
    write_json(out / "report.json", report.as_dict())
    write_text_atomic(out / "report.md", report.to_markdown())

    print(f"{report.n_rows} rows over {report.n_videos} videos · {name}")
    for code, n in report.counts.items():
        print(f"  {code:<36} {n}")
    print(f"report in {out}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
