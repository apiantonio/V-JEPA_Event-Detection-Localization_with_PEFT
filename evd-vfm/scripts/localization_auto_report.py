"""Evaluate every sealed stage of the localization campaign and write the comparison.

Runs locally, where the frozen GT snapshot lives (``evaluation-only/``, never shipped to the
cluster). For every experiment of the reduced plan found in a sealed stage it calls
``localization_auto_evaluate.evaluate`` — the single metric: plane → event → class → mean of
the 13 classes — on UCF-Crime and MIVIA-Crime separately, then writes:

- ``results/<dataset>/<experiment>.json``: the evaluator output, unchanged;
- ``comparison.md`` and ``comparison_<dataset>.csv``: all-versus-all ranking (UCF selects,
  MIVIA is reported apart), the best of each family, the 13 class columns, fit and
  inference times. Ties share a rank. Experiments still missing are listed, not dropped.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from evd.exp.localization_auto import FAMILY_LABELS  # noqa: E402
from localization_auto_evaluate import evaluate  # noqa: E402

PLAN = ROOT / "configs/localization_auto_20261016_reduced.json"
DATASETS = {"ucf_test": "ucf_crime", "mivia_test": "mivia_crime"}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def ranks(values: list[float]) -> list[int]:
    """:return: competition ranks, ties share the best position."""
    ordered = sorted(values, reverse=True)
    return [ordered.index(v) + 1 for v in values]


def collect(predictions: Path, snapshot: Path, out: Path, plan: dict) -> dict:
    """:return: results per dataset and experiment (evaluating what is not cached).

    :raises ValueError: if a result does not cover every annotated event and plane of the
        snapshot (17 October: every method is scored on every annotated plane).
    """
    wanted = {e["id"]: e for e in plan["experiments"]}
    counts = read(snapshot / "snapshot.json")["counts"]
    results = {d: {} for d in DATASETS}
    for stage in sorted(p for p in predictions.iterdir() if p.is_dir()):
        for folder, dataset in DATASETS.items():
            seal = stage / "operations" / f"seal-{folder}.json"
            if not seal.exists():
                continue
            for experiment in read(seal)["methods"]:
                if experiment not in wanted:
                    continue
                target = out / "results" / folder / f"{experiment}.json"
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    evaluate(snapshot, stage, seal, dataset, experiment, target)
                result = read(target)
                covered = len(result["per_event"]), sum(e["planes"] for e in result["per_event"])
                if covered != (counts[f"{dataset}_events"], counts[f"{dataset}_planes"]):
                    raise ValueError(f"{experiment} on {folder} misses annotated planes: {covered}")
                results[folder][experiment] = {**result, "stage": stage.name}
    return results


def chunk_count(predictions: Path, stage: str, folder: str, experiment: str) -> int:
    """:return: chunks the experiment's inference time covers (``inference_chunks``).

    In the gated layout every chunk is counted (S0 cost on all of them); in the merged
    layout of 17 October only the localized chunks.
    """
    seal = read(predictions / stage / "operations" / f"seal-{folder}.json")
    total = 0
    for v in seal["videos"]:
        complete = read(predictions / stage / v["directory"] / "complete.json")
        total += complete["timing"][experiment].get("inference_chunks", len(complete["chunks"]))
    return total


def table(rows: list[dict], classes: list[str]) -> list[str]:
    head = "| # | Experiment | Family | Macro IoU, 13 classes | Fit (s) | Inference (s/chunk) |"
    lines = [head, "|---:|---|---|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r['rank']} | {r['experiment']} | {FAMILY_LABELS.get(r['family'], r['family'])} | "
            f"{r['macro']:.4f} | {r['fit']:.1f} | {r['per_chunk']:.3f} |"
        )
    lines += ["", "| Experiment | " + " | ".join(classes) + " |", "|---|" + "---:|" * len(classes)]
    for r in rows:
        lines.append(
            f"| {r['experiment']} | " + " | ".join(f"{r['classes'][c]:.3f}" for c in classes) + " |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="pulled predictions/ root")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=PLAN, help="frozen plan of the experiments")
    args = parser.parse_args()
    plan = read(args.plan)
    family = {e["id"]: e["family"] for e in plan["experiments"]}
    results = collect(args.predictions, args.snapshot, args.output, plan)
    counts = read(args.snapshot / "snapshot.json")["counts"]
    lines = [
        "# Automatic localization: comparison",
        "",
        "Every experiment is evaluated on every annotated plane: UCF-Crime"
        f" {counts['ucf_crime_events']} events / {counts['ucf_crime_planes']} planes,"
        f" MIVIA-Crime {counts['mivia_crime_events']} events / {counts['mivia_crime_planes']}"
        " planes (checked for every result).",
        "",
        "Single metric: IoU per annotated plane → mean per event → mean per class (13 values)"
        " → mean of the 13 classes. The winner is chosen on UCF-Crime test: an exploratory"
        " selection, not an independent estimate. MIVIA-Crime is reported apart and chooses"
        " nothing.",
        "",
    ]
    for folder in DATASETS:
        rows = []
        for experiment, result in results[folder].items():
            per_class = {c["class"]: c["iou"] for c in result["per_class"]}
            chunks = chunk_count(args.predictions, result["stage"], folder, experiment)
            rows.append(
                {
                    "experiment": experiment,
                    "family": family[experiment],
                    "macro": result["value"],
                    "classes": per_class,
                    "fit": result["timing"]["fit_seconds"],
                    "per_chunk": result["timing"]["inference_compute_seconds"] / chunks,
                    "events": {c["class"]: c["events"] for c in result["per_class"]},
                }
            )
        if not rows:
            continue
        for r, k in zip(rows, ranks([r["macro"] for r in rows]), strict=True):
            r["rank"] = k
        rows.sort(key=lambda r: (r["rank"], r["experiment"]))
        classes = sorted(rows[0]["classes"])
        title = (
            "UCF-Crime test (selection)"
            if folder == "ucf_test"
            else "MIVIA-Crime test (reported only)"
        )
        lines += [f"## {title}", "", "### All against all", ""] + table(rows, classes)
        lines += [
            "",
            "### By family",
            "",
            "| Family | Best | Macro IoU | Members (macro IoU) |",
            "|---|---|---:|---|",
        ]
        for key, name in FAMILY_LABELS.items():
            members = [r for r in rows if r["family"] == key]
            if members:
                best = members[0]
                listed = ", ".join(f"{m['experiment']} {m['macro']:.3f}" for m in members)
                lines.append(f"| {name} | {best['experiment']} | {best['macro']:.4f} | {listed} |")
        support = rows[0]["events"]
        lines += ["", "Events per class: " + ", ".join(f"{c} {support[c]}" for c in classes), ""]
        missing = sorted(set(family) - {r["experiment"] for r in rows})
        if missing:
            lines += [f"Not yet evaluated ({len(missing)}): " + ", ".join(missing), ""]
        with (args.output / f"comparison_{folder}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "rank",
                    "experiment",
                    "family",
                    "macro_iou",
                    "fit_s",
                    "inference_s_per_chunk",
                    *classes,
                ]
            )
            for r in rows:
                writer.writerow(
                    [r["rank"], r["experiment"], r["family"], r["macro"], r["fit"], r["per_chunk"]]
                    + [r["classes"][c] for c in classes]
                )
    (args.output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written {args.output / 'comparison.md'}")


if __name__ == "__main__":
    main()
