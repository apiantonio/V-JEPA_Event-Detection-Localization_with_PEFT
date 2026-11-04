r"""Count the protocol chunk grid and print its verifiable signature.

The grid is not sealed: it is a pure function of (split, annotations, geometry) and is
recomputed (``docs/04`` §8); what is recorded is the count and the measured positive rate.
The numbers of ``docs/04`` §3.6, ``docs/05`` §9.1 and ``docs/13`` §2.1 come from here, as
does ``max_steps`` of the ``configs/train/`` budgets (``ceil(training_chunks /
effective_batch)``, ADR-047).

It calls the same code as the runs (``sampler_for_protocol`` and ``chunks_for_video``):
a count made with a second windowing implementation would measure that implementation.
Geometry is read from the protocol, not from arguments (ADR-047); explicit overrides
remain, to compare candidate geometries (the ADR-034 table).

Usage, with the ``pytorch`` environment active, from the repository root::

    python scripts/conta_chunk.py \
        --protocol configs/protocol/edp_v2.yaml \
        --inventario data/ucf_crime_inventory/inventory.csv \
        --split-dir  data/ucf_crime_mirror \
        --annotazioni ../data/annotations/fs_ucf_crime
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evd.annotations.schema import TemporalAnnotation  # noqa: E402
from evd.annotations.ucf_classes import ucf_classes  # noqa: E402
from evd.data.windows import CausalWindowSampler  # noqa: E402
from evd.exp.sources import DatasetBundle, build_dataset  # noqa: E402
from evd.protocol.chunks import (  # noqa: E402
    chunks_for_video,
    riepilogo,
    sampler_for_protocol,
)
from evd.protocol.split import load_split  # noqa: E402


def _arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True, help="configs/protocol/edp_v*.yaml")
    p.add_argument("--inventario", type=Path, required=True, help="inventory.csv from F2")
    p.add_argument(
        "--split-dir",
        type=Path,
        required=True,
        help="folder with Anomaly_Train.txt and Anomaly_Test.txt (needed to build the bundle)",
    )
    p.add_argument("--annotazioni", type=Path, required=True, help="FS-UCF-Crime folder")
    p.add_argument("--num-frames", type=int, help="override: samples per window")
    p.add_argument("--target-fps", type=float, help="override: requested frame rate")
    p.add_argument("--hop-s", type=float, help="override: distance between windows, in seconds")
    p.add_argument(
        "--batch-effettivo",
        type=int,
        default=16,
        help="to print how many steps an epoch takes (default 16 = 2 x 8)",
    )
    return p.parse_args()


def _bundle(args: argparse.Namespace) -> DatasetBundle:
    return build_dataset(
        {
            "id": "ucf",
            "name": "ucf_crime",
            "root": str(args.split_dir),
            "inventory": str(args.inventario),
            "splits_dir": str(args.split_dir),
            "annotations": {"source": "ucf_lab", "path": str(args.annotazioni)},
        }
    )


def _sampler(
    protocollo: Any, args: argparse.Namespace, *, validazione: bool
) -> CausalWindowSampler:
    """Return the partition geometry, from the protocol unless explicitly overridden.

    The protocol -> sampler translation is the runner one
    (:func:`evd.protocol.chunks.sampler_for_protocol`): this script is the source of
    ``max_steps``, so it must enumerate the same grid. Command-line overrides are applied
    afterwards with :func:`dataclasses.replace`, so they stay visible as deviations.
    """
    from omegaconf import OmegaConf

    sampler = sampler_for_protocol(
        None, OmegaConf.to_container(protocollo, resolve=True), validazione=validazione
    )
    override: dict[str, Any] = {}
    if args.num_frames:
        override["num_frames"] = int(args.num_frames)
    if args.target_fps:
        override["target_fps"] = float(args.target_fps)
    if args.hop_s is not None:
        override["hop_s"] = float(args.hop_s)
    return replace(sampler, **override) if override else sampler


def _partition(
    bundle: DatasetBundle, video_ids: tuple[str, ...], sampler: CausalWindowSampler
) -> int:
    """Print counts, rates and tails of a partition.

    :param bundle: dataset with loaded inventory and annotations.
    :param video_ids: the partition videos.
    :param sampler: the grid geometry.
    :return: how many chunks were produced.
    """
    classes = ucf_classes()
    items = []
    without_windows: list[tuple[str, float, int]] = []
    per_class: Counter[str] = Counter()
    coverages: list[float] = []
    truncated = 0
    repeated = 0
    duration_s = 0.0
    for video_id in video_ids:
        meta = bundle.inventory[video_id]
        duration_s += meta.duration_s
        ann = bundle.annotations.get(video_id, TemporalAnnotation(video_id=video_id))
        chunk = chunks_for_video(meta, ann, sampler)
        if not chunk:
            # A video shorter than a window produces nothing on the `endpoint` grid: a dataset
            # exclusion to count and name (docs/04 §3.6). It never happens on the `fixed` grid.
            without_windows.append((video_id, meta.duration_s, len(ann.intervals)))
            continue
        full = sampler.window_frames(meta)
        items.extend(chunk)
        for c in chunk:
            length = c.span[1] - c.span[0] + 1
            truncated += int(length < full)
            repeated += int(len(set(c.window.frame_indices)) < c.window.num_frames)
            # `class_idx is not None` rather than `label == 1`: equivalent by construction
            # (`chunks_for_video`), but only the former tells the type.
            if c.class_idx is not None:
                per_class[classes[c.class_idx].name] += 1
                coverages.append(min(1.0, c.overlap_frames / length))
    r = riepilogo(items)
    coverages.sort()
    print(f"  videos in the split        {len(video_ids)}")
    print(f"  videos with >= 1 window    {r.n_video}")
    print(f"  duration                   {duration_s / 3600:.1f} h")
    print(f"  chunk                      {r.n_chunk}")
    print(f"  of which truncated         {truncated}")
    print(f"  with repeated frames       {repeated}")
    print(f"  positives                  {r.n_positivi} ({r.tasso_positivi_uniforme:.4%})")
    print(f"  weighted positive rate     {r.tasso_positivi_pesato:.4%}")
    if coverages:
        print(f"  median positive coverage   {coverages[len(coverages) // 2]:.3f}")
        quota = sum(1 for x in coverages if x < 0.25) / len(coverages)
        print(f"  positives below 25%        {quota:.1%}")
    print(f"  richest classes            {per_class.most_common(2)}")
    print(f"  poorest classes            {per_class.most_common()[-2:]}")
    if without_windows:
        print(f"  videos WITHOUT windows     {len(without_windows)}")
        for video_id, duration, n_int in without_windows:
            print(f"    {video_id:28s} {duration:6.2f} s, {n_int} annotated intervals")
    return r.n_chunk


def main() -> None:
    from omegaconf import OmegaConf

    args = _arguments()
    protocol = OmegaConf.load(args.protocol)
    bundle = _bundle(args)
    split = load_split(Path(str(OmegaConf.select(protocol, "split_file"))))
    train = _sampler(protocol, args, validazione=False)
    val = _sampler(protocol, args, validazione=True)

    tokens = (int(OmegaConf.select(protocol, "geometry.size")) // 16) ** 2 * (train.num_frames // 2)
    print(
        f"protocol {split.protocol_version} · grid {train.grid!r} · "
        f"{train.num_frames} samples, window {train.window_s or train.span_s:.2f} s, "
        f"hop {train.hop_s} s (validation {val.hop_s} s) "
        f"-> {tokens} tokens/window, "
        f"{tokens * 1024 * 2 / 1024**2:.1f} MiB per window per layer in fp16"
    )

    n_train = 0
    for name, ids, sampler in (
        ("training", split.train, train),
        ("validazione", split.val, val),
        ("valutazione", split.test, train),
    ):
        print(f"\n{name}")
        count = _partition(bundle, ids, sampler)
        if name == "training":
            n_train = count

    steps = math.ceil(n_train / args.batch_effettivo)
    print(
        f"\none epoch at effective batch {args.batch_effettivo}: {steps} steps "
        f"(the `max_steps` of configs/train/, ADR-047)"
    )

    reachable = 0
    totals = 0
    for _name, ids, sampler in (
        ("training", split.train, train),
        ("validazione", split.val, val),
        ("valutazione", split.test, train),
    ):
        for video_id in ids:
            ann = bundle.annotations.get(video_id)
            if ann is None or not ann.intervals:
                continue
            meta = bundle.inventory[video_id]
            windows = sampler.spans_for_video(meta)
            for start, end, _class_name in ann.intervals:
                totals += 1
                if any(min(end, e) - max(start, s) + 1 > 0 for s, e in windows):
                    reachable += 1
    print(f"reachable intervals: {reachable}/{totals}")


if __name__ == "__main__":
    main()
