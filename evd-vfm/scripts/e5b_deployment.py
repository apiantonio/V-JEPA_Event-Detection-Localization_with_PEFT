#!/usr/bin/env python
"""Thin CLI for E5b.2: the deployment study (``docs/10`` §E5b.2).

    python scripts/e5b_deployment.py

Not an experiment and no GPU required: it composes existing artifacts (the E5b profile,
the E0d decoder throughput, the E2/E3 cells) and writes ``deployment.json``,
``deployment.md`` and the figures inside the E5b run folder. It runs on the laptop and
does not go through the tracker (it produces no measurement); the provenance of every
number is in the ``sorgenti`` payload field.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evd.eval.deployment import Geometry  # noqa: E402
from evd.exp.e5b_deploy import confronto, study  # noqa: E402
from evd.utils import env  # noqa: E402
from evd.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger(__name__)

#: The two E0d artifacts, and which one is used: the ``throughput-aug`` leg ran on a GPU
#: node with 8 CPUs (the configuration of a real job), while ``throughput`` ran on a
#: 32-core CPU node (higher number, not the deployment machine).
DECODER_DEFAULT = "e0d_ucf_vjepa21l_frozen-none_s1__throughput-aug"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="runs folder; default from EVD_RUNS_DIR",
    )
    parser.add_argument(
        "--efficienza-run",
        action="append",
        default=None,
        help="E5b profile run to compose; **repeatable**. With two or more, "
        "each produces its study and a **machine comparison** is added "
        "in `runs/figures/`. `action='append'` rather than `nargs` with a "
        "string default: a string IS a sequence of strings, and the default "
        "would be iterated character by character (docs/11 v1.59)",
    )
    parser.add_argument(
        "--confronto-dir",
        type=Path,
        default=None,
        help="where to write the machine comparison; default `<runs>/figures`. "
        "Not in a run folder: a comparison belongs to none "
        "of the profiles it combines",
    )
    parser.add_argument("--decoder-run", default=DECODER_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=None, help="default: the E5b run")
    parser.add_argument("--workers-decoder", type=int, default=8)
    parser.add_argument(
        "--augment-decoder",
        default="none",
        help="augmentation does not run at inference (ADR-028); 'official_nocrop' tells "
        "how much the number would change if it did",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="table batch; 1 is the online deployment latency",
    )
    parser.add_argument(
        "--batch-verifica",
        type=int,
        default=2,
        help="batch the test passes ran with (2 on edp_v2)",
    )
    parser.add_argument(
        "--host-verifica",
        default="ainode02",
        help="host on which observed durations are compared; ONE only, or the slope "
        "of the fit would absorb a machine difference",
    )
    parser.add_argument("--senza-verifica", action="store_true")
    parser.add_argument("--senza-figure", action="store_true")
    parser.add_argument("--hop-s", type=float, default=4.0)
    parser.add_argument("--span-s", type=float, default=7.75)
    parser.add_argument("--n-chunk", type=int, default=9116)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()
    runs = args.runs_dir or env.runs_dir()
    profiles = args.efficienza_run or ["e5b_efficienza"]
    decoder = Path(runs) / args.decoder_run / "throughput.json"
    geometry = Geometry(hop_s=args.hop_s, span_s=args.span_s, n_chunk_valutazione=args.n_chunk)

    paths: list[Path] = []
    last: dict[str, object] = {}
    for name in profiles:
        efficiency = Path(runs) / name / "efficiency_inference.json"
        paths.append(efficiency)
        # `--out-dir` applies only with ONE profile: with two, writing both into the same folder
        # would keep the second and lose the first.
        out_dir = (args.out_dir if len(profiles) == 1 else None) or Path(runs) / name
        last = study(
            efficiency=efficiency,
            decoder=decoder,
            runs_dir=runs,
            out_dir=out_dir,
            registry=None if args.senza_verifica else Path(runs) / "registry.jsonl",
            host_verifica=None if args.senza_verifica else args.host_verifica,
            workers_decoder=args.workers_decoder,
            augment_decoder=args.augment_decoder,
            batch=args.batch,
            batch_verifica=args.batch_verifica,
            geometria=geometry,
            figure=not args.senza_figure,
        )
        log.info("study of %s written to %s", name, out_dir)

    if len(paths) >= 2:
        where = args.confronto_dir or Path(runs) / "figures"
        outcome = confronto(efficiencies=paths, out_dir=where, figure=not args.senza_figure)
        print(json.dumps(outcome, indent=2, default=str)[:4000])
        log.info("machine comparison written to %s", where)
    else:
        print(json.dumps({k: v for k, v in last.items() if k != "righe"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
