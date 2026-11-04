#!/usr/bin/env python
"""From registry rows to figures and tables: numbers are never copied by hand.

    python scripts/make_figures.py --exp e3r
    python scripts/make_figures.py --exp e3r --metric chunk_auc --metric chunk_ap
    python scripts/make_figures.py --exp e2d --exp e3p --exp e3r --per-run   # E3-X.1
    python scripts/make_figures.py --exp smoke --non-ufficiali    # the F6 dummy sweep

ADR-014 makes this script mandatory (figures come from the registry), and ADR-009 says
which registry: the local one, brought back from the cluster by ``sync.ps1 -Azione
pull-run``. The chain uses existing pieces only:

    read_registry()            evd.exp.registry     — tolerant: a broken row does not kill it
    grouped_for_aggregation()  evd.exp.registry     — the bridge
    SeedAggregator             evd.eval.aggregate   — mean ± spread over seeds, tidy CSV
    curve_png()                evd.eval.plots       — the curve with error bars

The script is thin (``docs/02`` §5). It refuses to:

* aggregate unofficial runs with official ones (``--non-ufficiali`` takes them instead,
  not in addition);
* invent a spread: with a single seed ``std`` stays empty in the CSV;
* put figures in the cache: they go to ``$EVD_RUNS_DIR``, the tree that comes back from
  the cluster (invariant 2).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evd.eval.aggregate import SeedAggregator  # noqa: E402
from evd.eval.plots import curves_png
from evd.eval.report import comparison_png, run_figures  # noqa: E402
from evd.exp.registry import grouped_for_aggregation, read_registry  # noqa: E402
from evd.utils import env  # noqa: E402
from evd.utils.io import ensure_dir  # noqa: E402
from evd.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("make_figures")

#: The primary ADR-024 metric: chunk-level AUC. ``--metric frame_auc`` gives an EMPTY figure
#: on any run after ADR-031 (no `frame_*` key in `metrics.json`); it only applies to
#: archived `edp_v1` runs.
DEFAULT_METRIC = "chunk_auc"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--exp",
        action="append",
        default=[],
        help="filter by experiment (e2d, e3r, e3p, smoke…); REPEATABLE, which is how "
        "E3-X.1 comes from a single call: --exp e2d --exp e3p --exp e3r. `exp` stays "
        "part of the group key, so cells sit side by side and are not merged",
    )
    ap.add_argument(
        "--metric",
        action="append",
        default=[],
        help=f"metric to draw; repeatable. Default: {DEFAULT_METRIC}",
    )
    ap.add_argument(
        "--non-ufficiali",
        action="store_true",
        help="aggregate runs with official=false (smoke, dummy sweeps) INSTEAD of the official ones",
    )
    ap.add_argument(
        "--registry", type=Path, default=None, help="default: $EVD_RUNS_DIR/registry.jsonl"
    )
    ap.add_argument("--out-dir", type=Path, default=None, help="default: $EVD_RUNS_DIR/figures")
    ap.add_argument(
        "--seeds-attesi",
        type=int,
        default=None,
        help="how many seeds the schedule expects: flag incomplete groups instead of dropping them",
    )
    ap.add_argument("--niente-figure", action="store_true", help="write only CSV and table")
    ap.add_argument(
        "--run",
        type=Path,
        action="append",
        default=[],
        help="write the figures of ONE run folder only, without the registry; "
        "repeatable. For runs outside the run_id grammar (the "
        "cross-dataset evaluations of exp/esterno.py, named {run_id}__{dataset}, ADR-066), "
        "which `--per-run` skips because `parsed_id()` does not recognise them. No "
        "aggregate or CSV: those need the grammar, the figures of a run do not",
    )
    ap.add_argument(
        "--per-run",
        action="store_true",
        help="also write the figures INSIDE every run folder (ROC, PR, confusion "
        "matrices, event level, per class, threshold sensitivity). They go there and "
        "not in --out-dir because they describe ONE run, not the aggregate",
    )
    return ap


def _md_table(righe: list[dict[str, Any]], metrica: str) -> str:
    """Return the per-method table, in a format pasted into the thesis as is.

    ``n`` is always reported next to mean and spread (``docs/03`` §6.4); a group below the
    seed schedule is flagged ``⚠``, not hidden.

    :param righe: the tidy rows of :meth:`SeedAggregator.tidy`.
    :param metrica: which metric to tabulate.
    :return: the Markdown table; an empty string if the metric does not appear.
    """
    choices = [r for r in righe if r["metric"] == metrica]
    if not choices:
        return ""
    # The `k` column appears only if it exists: `GroupKey.as_dict()` omits it when `None`,
    # which since ADR-031 is the case of every run.
    with_k = any(r.get("k") is not None for r in choices)
    header_spec = "| backbone | adapt |" + (" k |" if with_k else "") + " mean | std | n |"
    ruler = "|---|---|" + ("---:|" if with_k else "") + "---:|---:|---:|"
    outside = [f"### {metrica}", "", header_spec, ruler]
    for row in sorted(choices, key=lambda r: (r["backbone"], r["adapt"], str(r.get("k") or ""))):
        deviation = "—" if row["std"] is None else f"{row['std']:.4f}"
        mark = "" if row["complete"] else " ⚠"
        cells = f"| {row['backbone']} | {row['adapt']} |"
        if with_k:
            cells += f" {row.get('k')} |"
        outside.append(f"{cells} {row['mean']:.4f} | {deviation} | {row['n']}{mark} |")
    return "\n".join(outside)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()

    if args.run:
        # The figures of ONE run do not need the registry: `figure_di_un_run` reads `metrics.json`
        # and, if present, `scores.jsonl` (filtering by `parsed_id()` would fail on a cross-dataset
        # run).
        per_run: dict[str, list[str]] = {}
        for folder in args.run:
            if not (folder / "metrics.json").is_file():
                log.error("%s is not a run folder: metrics.json is missing", folder)
                return 2
            per_run[folder.name] = [str(v) for v in run_figures(folder).values()]
            log.info("%s: %d figure", folder.name, len(per_run[folder.name]))
        print(json.dumps({"figure_per_run": per_run}, indent=2, ensure_ascii=False))
        return 0

    metrics = args.metric or [DEFAULT_METRIC]

    rows = read_registry(args.registry)
    if not rows:
        raise SystemExit(
            "empty or missing registry. On the laptop runs live in evd-vfm/runs/; if the "
            "numbers are on the cluster, bring them back with .\\hpc\\sync.ps1 -Azione pull-run "
            "(invariant 2: runs/ is gitignored, durability is transfer)."
        )

    # `None`, not an empty list: a missing `--exp` means «all», while an empty list would keep
    # no run.
    which_exp = args.exp or None
    run = grouped_for_aggregation(rows, exp=which_exp, official=not args.non_ufficiali)
    if not run:
        which = "unofficial" if args.non_ufficiali else "official"
        raise SystemExit(
            f"no complete {which} run for exp={which_exp!r} among {len(rows)} registry "
            f"rows. A run counts as complete only with status 'ok'; run_ids outside the "
            f"grammar cannot be grouped (see the warnings above)."
        )

    aggregator = SeedAggregator(metrics, seeds_expected=args.seeds_attesi)
    aggregator.add_all(run)
    tidy = aggregator.tidy()

    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else env.runs_dir() / "figures")
    # File names: with several experiments, their sorted conjunction, so the same request
    # always produces the same file.
    label = "-".join(sorted(args.exp)) if args.exp else "tutti"
    csv = aggregator.to_csv(out_dir / f"{label}_aggregato.csv")

    tables = [t for t in (_md_table(tidy, m) for m in metrics) if t]
    table = out_dir / f"{label}_tabella.md"
    table.write_text(
        f"# {label} — {len(run)} runs over {len({k for k, _, _ in run})} groups\n\n"
        + "\n\n".join(tables)
        + "\n",
        encoding="utf-8",
    )

    figure: list[str] = []
    skipped: dict[str, str] = {}
    # Which figure makes sense depends on the regime, as the data say: with a `k` axis
    # (few-shot `fsp_v1` runs) the curve; without it (every run since ADR-031) the comparison
    # of adaptation methods with the seed spread.
    with_k = any(r.get("k") is not None for r in tidy)
    for metric in metrics:
        if args.niente_figure:
            continue
        destination = out_dir / f"{label}_{metric}.png"
        try:
            if with_k:
                figure.append(str(curves_png(tidy, destination, metric=metric)))
            else:
                figure.append(str(comparison_png(tidy, destination, metrica=metric)))
        except ValueError as error:
            # Not a script failure: a figure that does not exist for those data, replaced by the table.
            skipped[metric] = str(error)
            log.warning("figure %s not drawn: %s", metric, error)

    # The figures of every cell, inside its folder; a run predating ADR-051 has no
    # `scores.jsonl` and produces fewer.
    per_run: dict[str, list[str]] = {}
    if args.per_run:
        # The SAME runs used by the aggregate, each ONCE: iterating over `righe` would ignore
        # `--exp` and redraw a run once per registry row (the registry is append-only, ADR-009).
        for row in rows:
            identity = row.parsed_id()
            if identity is None or (which_exp is not None and identity.exp not in which_exp):
                continue
            if row.official is (True if args.non_ufficiali else False):
                continue
            run_id = str(row.run_id)
            if run_id in per_run:
                continue
            folder = env.runs_dir() / run_id
            if not (folder / "metrics.json").is_file():
                continue
            try:
                per_run[run_id] = [str(v) for v in run_figures(folder).values()]
            except (ValueError, FileNotFoundError) as error:
                log.warning("figures of %s not written: %s", run_id, error)

    outcome = {
        "n_run": len(run),
        "n_gruppi": len({key for key, _, _ in run}),
        "metriche": metrics,
        "csv": str(csv),
        "tabella": str(table),
        "figure": figure,
        "curve_non_disegnate": skipped,
        "figure_per_run": per_run,
        "gruppi_incompleti": [
            {"adapt": key.adapt, "k": key.k, "metric": name, "n": n}
            for key, name, n in aggregator.incomplete()
        ],
    }
    print(json.dumps(outcome, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
