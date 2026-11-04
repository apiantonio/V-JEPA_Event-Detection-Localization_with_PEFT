#!/usr/bin/env python
"""The smoke test, without Hydra: ``python scripts/smoke.py``.

A second path besides ``run_experiment.py +experiment=smoke``, independent of config
composition: if this passes and that does not, the problem is Hydra; if neither passes,
the problem is the code.

Exit code 0 = green. With ``--strict`` it also fails when the synthetic signal is not
visible (AUC below the guard), the form used by CI and the ``fatq`` job.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evd.exp.smoke import SmokeConfig, run_smoke  # noqa: E402

#: Guard threshold, not a thesis metric: it must signal a broken pipeline, not a worse
#: model. Measured on 28 August (F11) with the ``edp_v1`` regime: 1.000 · 0.984 · 1.000 on
#: world seeds 0/1/2 (the world seed is part of the guard). The value belongs to the regime
#: it was measured in: if the objective changes, it must be measured again.
MIN_CHUNK_AUC = 0.70


def main() -> int:
    ap = argparse.ArgumentParser(description="end-to-end smoke test on CPU")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: $EVD_CACHE_DIR/smoke")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--steps",
        type=int,
        default=SmokeConfig.max_steps,
        help="default: the toy budget of configs/train/probe_toy.yaml",
    )
    ap.add_argument("--no-figura", action="store_true")
    ap.add_argument("--no-run", action="store_true", help="do not write to runs/")
    ap.add_argument("--strict", action="store_true", help="fail if the AUC is below the guard")
    args = ap.parse_args()

    out = run_smoke(
        SmokeConfig(
            out_dir=args.out_dir,
            seed=args.seed,
            max_steps=args.steps,
            make_figure=not args.no_figura,
            write_run=not args.no_run,
        )
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))

    auc = out["chunk_auc"]
    if args.strict and not (auc >= MIN_CHUNK_AUC):
        print(
            f"\nSMOKE RED: chunk AUC {auc:.3f} < {MIN_CHUNK_AUC}. The synthetic signal is "
            f"coarse on purpose: if it is not visible, something among windows, mapping and "
            f"metrics is not talking to the rest.",
            file=sys.stderr,
        )
        return 1
    print(f"\nSMOKE GREEN — chunk AUC {auc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
