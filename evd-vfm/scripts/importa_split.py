#!/usr/bin/env python
r"""Import and seal the split delivered by the laboratory (ADR-047).

Nothing is generated: the partition comes from outside (``split-FS-UCF-Crime.csv``, 31
August 2026) and this script transcribes it into the sealed repository format after
testing it. Unlike ``build_split.py``, which applies a rule and can regenerate its output,
there is no rule here, so five checks take its place:

1. coverage: every CSV video is in the F2 inventory, and the canonical videos left out by
   the laboratory are reported;
2. geometry: CSV ``total_frames`` and ``fps`` match those measured by our probe;
3. labels: ``label`` (anomalous/normal) matches the FS-UCF-Crime annotations, video by
   video;
4. grid: our implementation of ``docs/varie/protocollo-laboratorio`` §1 reproduces
   ``finestre_valutazione`` for every video, with the partition hop (4 s on test, 8 s on
   validation);
5. duplicates: byte-identical pairs crossing partitions are counted (not excluded: the
   split is adopted verbatim, ADR-047, and the leakage is declared).

Typical use::

    python scripts/importa_split.py --protocol configs/protocol/edp_v2.yaml \
        --csv manifests/edp_v2/ucf_crime/split-FS-UCF-Crime.csv \
        --set root=../data/ucf_crime_mirror --dry-run

No protocol value comes from the command line: duration, hop and validation hop are read
from ``configs/protocol/edp_v2.yaml``; only file locations are passed here.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evd.data.windows import CausalWindowSampler  # noqa: E402
from evd.exp.sources import apply_overrides, build_dataset  # noqa: E402
from evd.protocol.split import SplitSpec, load_split, save_split  # noqa: E402
from evd.types import PreprocessSpec  # noqa: E402

#: The three CSV partitions, in the order they are written in the split.
PARTITIONS = ("train", "val", "test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True, help="configs/protocol/edp_v2.yaml")
    parser.add_argument("--csv", type=Path, required=True, help="lo split consegnato")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/data/ucf_crime.yaml"),
        help="the `data` group: WHERE the videos are and which annotations to read",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="overrides",
        help="override on the `data` group (paths only), for the local mirror",
    )
    parser.add_argument("--out", type=Path, help="where to write; default: protocol.split_file")
    parser.add_argument("--dry-run", action="store_true", help="check without writing")
    parser.add_argument(
        "--verifica",
        action="store_true",
        help="re-read the existing file and check that it describes the same CSV",
    )
    return parser


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return ""


def read_csv(path: Path) -> list[dict[str, str]]:
    """:return: the rows of the delivered CSV.

    :raises SystemExit: if an expected column is missing or an unknown partition appears in
        the ``split`` column (better to stop than to guess where a video goes).
    """
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    expected = {"video_id", "split", "label", "total_frames", "fps", "finestre_valutazione"}
    missing = sorted(expected - set(rows[0]))
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    unknown = sorted({r["split"] for r in rows} - set(PARTITIONS))
    if unknown:
        raise SystemExit(f"{path}: unknown partitions in the 'split' column: {unknown}")
    return rows


def sampler_of(protocollo: Any, *, validazione: bool) -> CausalWindowSampler:
    """Return the sampler of the partition, with values read from the protocol."""
    from omegaconf import OmegaConf

    def field(percorso: str) -> Any:
        value = OmegaConf.select(protocollo, percorso, default=None)
        if value is None:
            raise SystemExit(f"the protocol does not declare {percorso!r}")
        return value

    hop = field("temporal.val_hop_s") if validazione else field("temporal.hop_s")
    # The `PreprocessSpec` only carries `num_frames` and `target_fps`: no pixel is touched, so
    # the spatial fields are honest placeholders.
    spec = PreprocessSpec(
        fps=float(field("temporal.target_fps")),
        num_frames=int(field("temporal.num_frames")),
        size=int(field("geometry.size")),
        short_side=int(field("geometry.size")),
        geometry="squash",
        backend="none",
        backend_version="",
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        tokenizer="",
        dtype="fp32",
    )
    return CausalWindowSampler.from_spec(
        spec,
        hop_s=float(hop),
        grid=str(field("temporal.window_grid")),  # type: ignore[arg-type]
        window_s=float(field("temporal.window_s")),
    )


def verifica(righe: list[dict[str, str]], bundle: Any, protocollo: Any) -> list[str]:
    """Run the five checks of the module docstring.

    :return: the report rows, to print and archive in ``PROVENIENZA.md``.
    :raises SystemExit: at the first inconsistency that would make the split unusable (an
        inconsistent split produces numbers, not errors).
    """
    report: list[str] = []
    ids = {r["video_id"] for r in righe}

    # 1. coverage
    outside_inventory = sorted(ids - set(bundle.inventory))
    if outside_inventory:
        raise SystemExit(
            f"{len(outside_inventory)} CSV videos are not in the F2 inventory "
            f"({outside_inventory[:3]}…): inventory and delivery come from different releases"
        )
    canonical = set(bundle.train_videos) | set(bundle.query_videos)
    left_out = sorted(canonical - ids)
    report.append(
        f"coverage: {len(ids)} videos out of {len(canonical)} canonical; "
        f"left out by the laboratory: {len(left_out)}"
        + (f" ({', '.join(left_out)})" if left_out else "")
    )

    # 2. geometry and 3. labels
    geometry: list[str] = []
    labels: list[str] = []
    for row in righe:
        meta = bundle.inventory[row["video_id"]]
        if int(row["total_frames"]) != meta.num_frames or abs(float(row["fps"]) - meta.fps) > 1e-3:
            geometry.append(
                f"{row['video_id']}: CSV {row['total_frames']}f@{row['fps']} "
                f"vs probe {meta.num_frames}f@{meta.fps}"
            )
        ann = bundle.annotations.get(row["video_id"])
        anomalous = ann is not None and not ann.is_normal
        if (row["label"] == "anomalous") != anomalous:
            labels.append(row["video_id"])
    if geometry:
        raise SystemExit(
            f"{len(geometry)} videos have frames or fps different from the measured ones: "
            f"{geometry[:3]}. Our grid would not be theirs"
        )
    if labels:
        raise SystemExit(
            f"{len(labels)} videos have a `label` disagreeing with the "
            f"FS-UCF-Crime annotations ({labels[:3]}…): CSV and annotations are not the same release"
        )
    report.append(f"geometry: {len(righe)} videos, frames and fps agree with the F2 inventory")
    report.append(f"labels: {len(righe)} videos, `label` agrees with the annotations")

    # 4. grid
    samplers = {
        "val": sampler_of(protocollo, validazione=True),
        "test": sampler_of(protocollo, validazione=False),
    }
    discordant: list[str] = []
    totals: dict[str, int] = defaultdict(int)
    counted: dict[str, int] = defaultdict(int)
    for row in righe:
        expected = row["finestre_valutazione"].strip()
        if not expected:
            continue
        part = row["split"]
        ours = len(samplers[part].spans_for_video(bundle.inventory[row["video_id"]]))
        totals[part] += int(expected)
        counted[part] += ours
        if ours != int(expected):
            discordant.append(f"{row['video_id']} ({part}): CSV {expected}, ours {ours}")
    if discordant:
        raise SystemExit(
            f"our grid does not reproduce the laboratory one on {len(discordant)} "
            f"videos: {discordant[:3]}. Section 1 of the document is implemented differently, and every "
            f"resulting number would be on a set different from theirs"
        )
    for part in ("val", "test"):
        report.append(
            f"grid {part}: {counted[part]} windows, identical to the CSV video by video "
            f"(hop {samplers[part].hop_s} s)"
        )

    # 5. duplicates crossing the partitions
    where = {r["video_id"]: r["split"] for r in righe}
    by_sha: dict[str, list[str]] = defaultdict(list)
    for video_id, sha in bundle.content_sha.items():
        if video_id in where:
            by_sha[sha].append(video_id)
    crossings = [
        sorted(group)
        for group in by_sha.values()
        if len(group) > 1 and len({where[v] for v in group}) > 1
    ]
    report.append(f"byte-identical duplicates crossing the partitions: {len(crossings)}")
    for group in sorted(crossings):
        report.append("  " + " = ".join(f"{v} [{where[v]}]" for v in group))
    if crossings:
        report.append(
            "  WARNING: adopted verbatim (ADR-047), they are the partition on which the laboratory "
            "reports its numbers. The leakage is declared in the thesis, not silently corrected."
        )
    return report


def main(argv: list[str] | None = None) -> int:
    from omegaconf import OmegaConf

    args = build_parser().parse_args(argv)
    protocol = OmegaConf.load(args.protocol)
    if not bool(OmegaConf.select(protocol, "approved", default=False)):
        raise SystemExit(f"{args.protocol}: `approved` is not true; no split is imported before")
    # Two fields this script makes TRUE: an imported split excludes nothing and is not carved
    # by us; a protocol declaring otherwise would contradict the written file.
    for path, expected, why in (
        ("dedup", "declared", "an import does not exclude duplicates, it declares them (ADR-047)"),
        (
            "validation.source",
            "laboratory",
            "the partition comes from outside, it is not carved by us",
        ),
    ):
        value = str(OmegaConf.select(protocol, path, default=""))
        if value != expected:
            raise SystemExit(
                f"{args.protocol}: {path} = {value!r}, expected {expected!r} ({why}). "
                f"For a split GENERATED from a rule there is scripts/build_split.py"
            )
    if not args.csv.is_file():
        raise SystemExit(f"Missing CSV: {args.csv}")

    group = OmegaConf.load(args.data)
    # The override policy lives in `evd.exp.sources`, shared by every script.
    try:
        apply_overrides(group, args.overrides)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    bundle = build_dataset(OmegaConf.to_container(group, resolve=True))  # type: ignore[arg-type]

    rows = read_csv(args.csv)
    report = verifica(rows, bundle, protocol)
    print("\n".join(report))

    partitions = {p: tuple(r["video_id"] for r in rows if r["split"] == p) for p in PARTITIONS}
    split = SplitSpec(
        protocol_version=str(OmegaConf.select(protocol, "protocol_version", default="")),
        dataset=bundle.name,
        annotation_source=bundle.annotation_source,
        builder_commit=_git_commit(),
        created=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        train=partitions["train"],
        val=partitions["val"],
        test=partitions["test"],
        # No exclusion: the split is adopted verbatim (ADR-047); cross-partition duplicates are in
        # the report above and in PROVENIENZA.md.
        excluded=(),
    ).sealed()
    print(
        f"\ntrain {len(split.train)} · val {len(split.val)} · test {len(split.test)} "
        f"· sha {split.sha256[:12]}"
    )

    destination = args.out or Path(str(OmegaConf.select(protocol, "split_file", default="")) or "")
    if not str(destination):
        raise SystemExit(f"{args.protocol}: `split_file` is missing")

    if args.verifica:
        if not destination.exists():
            raise SystemExit(f"{destination} does not exist: there is nothing to verify")
        on_disk = load_split(destination)
        for part in PARTITIONS:
            if tuple(getattr(on_disk, part)) != partitions[part]:
                raise SystemExit(
                    f"{destination}: partition '{part}' on disk is not the CSV one. "
                    f"A split is never rewritten: if the laboratory delivered another one, "
                    f"it is a new protocol"
                )
        print(f"verified: {destination} describes the same CSV")
        return 0
    if args.dry_run:
        print("\ndry-run: nothing written")
        return 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    save_split(destination, split)
    print(f"\nwritten {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
