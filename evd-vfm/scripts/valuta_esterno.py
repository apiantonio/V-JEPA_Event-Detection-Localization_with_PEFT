#!/usr/bin/env python
r"""Evaluate an already closed run on an external test set. Thin CLI (docs/02 par.5).

    # on the cluster, where EVD_MIVIA_DIR points to the laboratory folder
    python scripts/valuta_esterno.py \
        --run runs/e3r_ucf_vjepa21l_dora-r32-qkvomlp_s42 --dati mivia_crime

    # locally, with the paths of this machine
    python scripts/valuta_esterno.py --run runs/... --dati mivia_crime \
        --set root=../data/mivia_crime \
        --set inventory=../data/mivia_crime_inventory/inventory.csv \
        --backbone-checkpoint ../data/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt

    # a trial on a few videos, before paying for the queue
    python scripts/valuta_esterno.py --run ... --dati mivia_crime --limite 4

Writes a new run to ``runs/{run_id}__{dataset}/`` (``config.yaml``, ``meta.json``,
``metrics.json``, ``scores.jsonl``) plus the registry row: a regular run folder, which the
demonstrator replays in ``--modo replay`` and ``eval/report.py`` plots. Threshold,
geometry and metrics come from :mod:`evd.exp.esterno`.

The source run checkpoint (``checkpoint/checkpoint.pt``), the backbone weights and a GPU
are required: this is a real inference pass (thousands of windows on the 270 MIVIA-Crime
videos).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from evd.exp.esterno import evaluate_external_dataset  # noqa: E402
from evd.exp.sources import DATA_PATH_OVERRIDES, load_data_group  # noqa: E402
from evd.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("valuta_esterno")

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--run", type=Path, required=True, help="the runs/{run_id}/ folder of the source run"
    )
    p.add_argument(
        "--dati", default="mivia_crime", help="the configs/data/ group of the external test set"
    )
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=f"paths of the data group; allowed: {sorted(DATA_PATH_OVERRIDES)}",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="decoding workers; changes no number, only the time",
    )
    p.add_argument(
        "--backbone-checkpoint",
        default=None,
        help="backbone weights, if not where the run had them (the config records the "
        "path of the machine it ran on)",
    )
    p.add_argument(
        "--limite",
        type=int,
        default=None,
        help="evaluate only the first N videos: a TRIAL, declared in metrics.json "
        "with `sottoinsieme: true`. It produces no thesis numbers",
    )
    p.add_argument(
        "--non-ufficiale",
        action="store_true",
        help="do not require a clean working tree. For trials, not for numbers",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()
    data_cfg = load_data_group(args.dati, args.overrides, configs_dir=ROOT / "configs" / "data")

    video_ids = None
    if args.limite is not None:
        from evd.exp.sources import build_dataset

        video_ids = build_dataset(data_cfg).query_videos[: args.limite]
        log.warning(
            "TRIAL on %d videos: the resulting numbers are not comparable with any thesis table",
            len(video_ids),
        )

    outcome = evaluate_external_dataset(
        args.run,
        data_cfg,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        backbone_checkpoint=args.backbone_checkpoint,
        video_ids=video_ids,
        official=not args.non_ufficiale,
    )
    m = outcome.metrics
    print(f"\n{outcome.run_id}")
    print(f"  source     {outcome.run_sorgente}  (tau {m['threshold']['value']:.6f}, transferred)")
    print(f"  dataset    {outcome.dataset}: {m['n_eval_videos']} video, {m['n_chunk_eval']} chunk")
    print(f"  chunk AUC  {m['chunk_auc']:.4f}   AP {m['chunk_ap']:.4f}")
    print(f"  AUC_a      {m['chunk_auc_anomalous']:.4f}   AP_a {m['chunk_ap_anomalous']:.4f}")
    print(f"  chunk F1   {m['chunk_prf']['f1']:.4f}   video F1 {m['video_prf']['f1']:.4f}")
    oracle = m.get("soglia_oracolo") or {}
    if oracle:
        print(
            f"  [oracle]   tau {oracle['threshold']:.6f} -> chunk F1 "
            f"{oracle['chunk_prf']['f1']:.4f} (NOT protocol, upper bound)"
        )
    print(f"\nartifacts in {outcome.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
