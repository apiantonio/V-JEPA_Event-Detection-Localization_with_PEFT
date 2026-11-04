#!/usr/bin/env python
"""Thin CLI for E5b: inference efficiency (docs/05 §6, docs/10 §E5b).

    python scripts/e5b_efficienza.py --checkpoint $EVD_CKPT_VITL

An experiment, not a diagnostic: the run is official, so the tracker rejects a dirty
working tree and the row goes into the registry. It does not start on the login node:
``profila_e5b`` requires CUDA, since a time taken on CPU would not be comparable with
those of E3.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evd.exp.e5b import VARIANTS, E5bConfig, profile_e5b  # noqa: E402
from evd.models.backbones.vjepa21_source import ARCHS, default_checkpoint  # noqa: E402
from evd.utils import env  # noqa: E402
from evd.utils.logging import setup_logging  # noqa: E402


def _integers(testo: str) -> tuple[int, ...]:
    values = tuple(int(piece) for piece in testo.split(",") if piece.strip())
    if not values:
        raise argparse.ArgumentTypeError("empty list")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--arch", choices=sorted(ARCHS), default="vit_large")
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"))
    parser.add_argument(
        "--varianti",
        nargs="+",
        default=list(VARIANTS),
        choices=list(VARIANTS),
        help="which variants to measure; default all six",
    )
    parser.add_argument(
        "--batches",
        type=_integers,
        default=(1, 2, 4, 8),
        help="1 is the deployment latency, 2 the batch E3 evaluated with, "
        "4 and 8 show where throughput saturates (cameras per GPU)",
    )
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30, help="iterations at batch > 1")
    parser.add_argument(
        "--iters-batch1",
        type=int,
        default=200,
        help="iterations at batch 1, where latency is a response time and its "
        "tail (p95/p99) is a deployment number",
    )
    parser.add_argument(
        "--senza-energia",
        action="store_true",
        help="do not sample the GPU power",
    )
    parser.add_argument("--intervallo-energia-s", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hop-s", type=float, default=4.0)
    parser.add_argument(
        "--run-id",
        default="e5b2_efficienza",
        help="⚠️ default changed with schema 2: a run folder is its identity "
        "on disk, and reusing `e5b_efficienza` would overwrite the artifact "
        "of job 542380, the source of the docs/10 §E5b numbers and of the "
        "already composed deployment study. The two profiles must coexist",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()
    checkpoint = args.checkpoint or default_checkpoint(args.arch)
    if not Path(checkpoint).is_file():
        raise SystemExit(
            f"missing checkpoint: {checkpoint}. Pass --checkpoint, or export EVD_CKPT_DIR."
        )
    config = E5bConfig(
        checkpoint=Path(checkpoint),
        run_id=args.run_id,
        repo=args.repo,
        arch=args.arch,
        configs_dir=args.configs_dir,
        varianti=tuple(args.varianti),
        batches=tuple(args.batches),
        num_frames=args.num_frames,
        size=args.size,
        precision=args.precision,
        layer=args.layer,
        warmup=args.warmup,
        iters=args.iters,
        iters_batch1=args.iters_batch1,
        misura_energia=not args.senza_energia,
        intervallo_energia_s=args.intervallo_energia_s,
        device=args.device or env.device(),
        seed=args.seed,
        hop_s=args.hop_s,
    )
    payload = profile_e5b(config)
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2))
    print(json.dumps(payload["rows"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
