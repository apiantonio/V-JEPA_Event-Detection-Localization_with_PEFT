#!/usr/bin/env python
"""Re-evaluate every saved model with the delivered OVAD code (docs/19).

Writes exports, judged rows, events, per-video metrics, original results and separate
rankings. Reuses the ``classifiche_modelli`` catalogue; MIVIA always uses the UCF
validation of the same model and seed. No GPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "ovad_protocol"), str(ROOT / "scripts")]

from evaluate_ovad import evaluate_ovad, from_triples  # noqa: E402
from protocol_core import metrics as M  # noqa: E402
from protocol_core import report, to_json  # noqa: E402
from protocol_core.evaluate import METRIC_NAMES  # noqa: E402
from protocol_core.inputs import build_entries  # noqa: E402

from classifiche_modelli import (  # noqa: E402
    MODELS,
    VERSION_GROUPS,
    find_cells,
    find_mivia,
    version_epochs,
)
from esporta_score_ovad import esporta  # noqa: E402
from evd.utils.hashing import file_sha256  # noqa: E402
from evd.utils.io import read_json, write_json, write_text_atomic  # noqa: E402


def save_json(path: Path, obj: object) -> None:
    """Write standard JSON: undefined metrics become null."""
    write_text_atomic(
        path, json.dumps(to_json(obj), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def write_rankings_csv(path: Path, rows: list[dict]) -> None:
    """Write a complete table, keeping the union of the columns."""
    if not rows:
        write_text_atomic(path, "")
        return
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(
        [
            {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in r.items()}
            for r in rows
        ]
    )
    write_text_atomic(path, buf.getvalue())


def verified_export(run: Path, out: Path, split: str, overrides: list[str], manifest: Path) -> dict:
    """Reuse an export only when sources, converter and outputs match.

    :return: the verified export receipt.
    :raises ValueError: if the conversion finds a data inconsistency.
    """
    receipt_path = out / "provenienza.json"
    converter_sha = file_sha256(ROOT / "scripts/esporta_score_ovad.py")
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if (
            receipt.get("converter_sha256") == converter_sha
            and receipt.get("overrides") == overrides
        ):
            sources_ok = all(
                Path(p).is_file() and file_sha256(p) == h
                for p, h in receipt["sorgenti_sha256"].items()
            )
            outputs_ok = all(
                (out / p).is_file() and file_sha256(out / p) == h
                for p, h in receipt["output_sha256"].items()
            )
            if sources_ok and outputs_ok:
                return receipt
    receipt = esporta(run, out, split=split, overrides=overrides, manifest=manifest)
    receipt.update({"converter_sha256": converter_sha, "overrides": overrides})
    write_json(receipt_path, receipt)
    return receipt


def details(labels: dict, scores: dict, theta: float, guard: float, out: Path) -> dict:
    """Save the original intermediate tables and check the denominators.

    :return: counts rebuilt from the tables, for the comparison with the result.
    :raises ValueError: if a per-video metric cannot be computed.
    """
    entries, hp = build_entries(from_triples(labels, "dettagli"), scores)
    rows, videos, events = [], [], []
    for entry in entries:
        bounds = M.make_rows(entry["duration"], entry["events"])
        table = M.row_table(entry, c=hp["chunk_size"], g=guard)
        for i, ((a, b, label), (score, judged)) in enumerate(zip(bounds, table, strict=True)):
            rows.append(
                {
                    "video_id": entry["key"],
                    "row_id": i,
                    "start": a,
                    "end": b,
                    "label": label,
                    "judgeable": judged >= 0,
                    "score": score if judged >= 0 else None,
                    "prediction": int(score >= theta) if judged >= 0 else None,
                }
            )
        if any(y >= 0 for _, y in table):
            vm = M.compute_metrics([entry], theta, c=hp["chunk_size"], g=guard)
        else:
            vm = {"n_rows": 0, "n_unjudgeable_rows": len(table)}
        videos.append({"video_id": entry["key"], **vm})
        for i, (a, b) in enumerate(entry["events"]):
            own = [
                x
                for (r0, r1, _), (x, y) in zip(bounds, table, strict=True)
                if y == 1 and r0 >= a - M.EPS and r1 <= b + M.EPS
            ]
            events.append(
                {
                    "video_id": entry["key"],
                    "event_id": i,
                    "start": a,
                    "end": b,
                    "judgeable": bool(own),
                    "score": max(own) if own else None,
                    "detected": bool(own) and max(own) >= theta,
                }
            )
    for name, data in (("righe", rows), ("per_video", videos), ("eventi", events)):
        save_json(out / f"{name}.json", data)
        write_rankings_csv(out / f"{name}.csv", data)
    return {
        "n_rows": sum(r["judgeable"] for r in rows),
        "n_unjudgeable_rows": sum(not r["judgeable"] for r in rows),
        "n_events": sum(e["judgeable"] for e in events),
        "n_events_detected": sum(e["detected"] for e in events),
        "n_events_unseen": sum(not e["judgeable"] for e in events),
    }


def validated_threshold(labels: dict, scores: dict, guard: float) -> float:
    """Compute Youden only if validation contains rows of both classes.

    :return: the original protocol threshold, without touching the tie-break.
    :raises ValueError: if ROC and Youden are undefined on validation.
    """
    entries, hp = build_entries(from_triples(labels, "validation"), scores)
    table = [r for e in entries for r in M.row_table(e, c=hp["chunk_size"], g=guard) if r[1] >= 0]
    if {y for _, y in table} != {0, 1}:
        raise ValueError("Validation must contain judgeable positive and negative rows.")
    return M.youden_threshold([x for x, _ in table], [y for _, y in table])


def select_versions(records: list[dict]) -> tuple[dict, list[dict]]:
    """Choose one version per training from the mean of the means on the two s42 tests.

    The candidates are the versions of the same training extended in continuation
    (:data:`classifiche_modelli.VERSION_GROUPS`: one, three and four epochs; three and four for
    the combinations). On an exact tie fewer epochs are preferred. Other seeds are not
    reselected. If a test pair is missing for a competing version, the choice is refused.
    """
    available = {r["modello"] for r in records}
    alternatives = {k for keys in VERSION_GROUPS.values() for k in keys}
    catalog = {k: dict(MODELS[k]) for k in available if k not in alternatives}
    lookup = {(r["modello"], r["dataset"], r["seed"]): r for r in records}
    choices = []
    for base, keys in VERSION_GROUPS.items():
        candidates = [k for k in keys if k in available]
        if len(candidates) == 1:
            # Nothing to choose: no pair is required, as before the fourth epoch existed.
            catalog[candidates[0]] = {**MODELS[candidates[0]], "nome": MODELS[base]["nome"]}
        if len(candidates) < 2:
            continue
        scores = {}
        for key in candidates:
            pair = [lookup.get((key, dataset, 42)) for dataset in ("ucf", "mivia")]
            if not all(pair):
                raise ValueError(f"Version selection: missing UCF/MIVIA s42 pair for {key}")
            scores[key] = (pair[0]["mean"] + pair[1]["mean"]) / 2
        winner = max(candidates, key=lambda k: (scores[k], -keys.index(k)))
        catalog[winner] = {**MODELS[winner], "nome": MODELS[base]["nome"]}
        choices.append(
            {
                "modello_base": base,
                "nome": MODELS[base]["nome"],
                "candidati": {k: {"epoche": version_epochs(k), "mean": scores[k]} for k in scores},
                "versione_scelta": winner,
                "epoche": version_epochs(winner),
                "criterio": "mean of the UCF and MIVIA means, seed 42, among the versions of the "
                "same training extended in continuation; tie: fewer epochs. Selection on the "
                "test set",
            }
        )
    return catalog, choices


def rank(records: list[dict], out: Path, *, tutte: bool) -> None:
    """Publish every metric: UCF s42, MIVIA s42, averages and other seeds.

    :raises ValueError: if the needed cohorts or pairs are not comparable.
    """
    if tutte:
        catalog = {k: MODELS[k] for k in {r["modello"] for r in records}}
        choices = []
    else:
        catalog, choices = select_versions(records)
    chosen = [
        {**r, "nome": catalog[r["modello"]]["nome"]} for r in records if r["modello"] in catalog
    ]
    metrics = [k for k, _ in METRIC_NAMES]
    for dataset in {r["dataset"] for r in chosen}:
        source = [r for r in chosen if r["dataset"] == dataset]
        for field in ("labels_sha256", "cohort_sha256", "chunk_size", "asking_rate", "guard", "T"):
            if len({r[field] for r in source}) != 1:
                raise ValueError(f"{dataset}: {field} differs between models; split the tables.")
    note = (
        "All available versions: diagnostic collection, including the three- and four-epoch "
        "alternatives. "
        "The main collection is [../classifiche/classifiche.md](../classifiche/classifiche.md)."
        if tutte
        else "Main collection: one version per model. Among the versions of the same training "
        "extended in continuation (one, three and four epochs; three and four for the "
        "combinations) the higher mean of the UCF/MIVIA means at seed 42 is chosen (tie: fewer "
        "epochs). It is a selection on "
        "the test set, not on validation. The version stays fixed for both datasets and all "
        "seeds; the name is plain. The choices are in `selezione_versioni.json` and the "
        "alternatives in "
        "[../classifiche_tutte_versioni/classifiche.md](../classifiche_tutte_versioni/classifiche.md)."
    )
    lines = [
        "# OVAD rankings",
        "",
        note,
        "",
        "All 15 metrics and the protocol mean, in percent; lower FPR is better, "
        "all other metrics higher is better. Ordered by mean. No average across seeds in the following tables.",
        "",
        "Mean = (AUC + macro AUC_a + row F1 + (1 − row FPR) + video F1 + event recall) / 6. "
        "Mean of means = (UCF mean + MIVIA mean) / 2, equal weight per dataset.",
        "",
        f"c={records[0]['chunk_size']:g} s, g={records[0]['guard']:g} s, h_test=4 s, h_val=8 s (warning expected from the code), T=32 s. "
        "Youden on the UCF validation rows, transferred to MIVIA. Rows are not chunks; "
        "a video can contribute both to the event part and to the clean part. "
        "Thresholds in the original results.json, counts and provenance in the per-seed CSV/JSON.",
    ]

    def table(dataset: str, seed: int) -> list[str]:
        rows = sorted(
            [r for r in chosen if r["dataset"] == dataset and r["seed"] == seed],
            key=lambda r: (-r["mean"], r["nome"]),
        )
        if not rows:
            return []
        stem = f"{dataset}_s{seed}"
        save_json(out / f"{stem}.json", rows)
        write_rankings_csv(out / f"{stem}.csv", rows)
        part = [
            f"# {'UCF-Crime' if dataset == 'ucf' else 'MIVIA-Crime'} — seed {seed}",
            "",
            f"{len(rows)} models/versions available. Values in percent; ordered by mean.",
            "",
            "| # | Model | " + " | ".join(label for _, label in METRIC_NAMES) + " |",
            "|" + "---|" * (2 + len(metrics)),
        ]
        for i, row in enumerate(rows, 1):
            vals = ["n/d" if row[k] is None else f"{100 * row[k]:.2f}" for k in metrics]
            part.append("| " + " | ".join([str(i), row["nome"], *vals]) + " |")
        write_text_atomic(out / f"{stem}.md", "\n".join(part) + "\n")
        return ["#" + s if s.startswith("# ") else s for s in part]

    for dataset in ("ucf", "mivia"):
        lines += ["", *table(dataset, 42)]
    lookup = {(r["dataset"], r["modello"]): r for r in chosen if r["seed"] == 42}
    paired = []
    for model in catalog:
        a, b = lookup.get(("ucf", model)), lookup.get(("mivia", model))
        if a and b:
            paired.append(
                {
                    "modello": model,
                    "nome": catalog[model]["nome"],
                    "seed": 42,
                    "mean_ucf": a["mean"],
                    "mean_mivia": b["mean"],
                    "mean_fra_dataset": (a["mean"] + b["mean"]) / 2,
                    "auc_ucf": a["auc"],
                    "auc_mivia": b["auc"],
                    "auc_media_fra_dataset": (a["auc"] + b["auc"]) / 2,
                }
            )
    paired.sort(key=lambda r: (-r["mean_fra_dataset"], r["nome"]))
    cross = [
        "# Mean of the UCF/MIVIA means — seed 42",
        "",
        "Mean of means = (UCF mean + MIVIA mean) / 2. It is not the mean of the AUCs.",
        "",
        "| # | Model | UCF mean | MIVIA mean | Mean of means |",
        "|---|---|---|---|---|",
    ]
    for i, row in enumerate(paired, 1):
        cross.append(
            "| "
            + " | ".join(
                [
                    str(i),
                    row["nome"],
                    *[
                        f"{100 * row[k]:.2f}"
                        for k in ("mean_ucf", "mean_mivia", "mean_fra_dataset")
                    ],
                ]
            )
            + " |"
        )
    save_json(out / "cross_dataset_s42.json", paired)
    write_rankings_csv(out / "cross_dataset_s42.csv", paired)
    write_text_atomic(out / "cross_dataset_s42.md", "\n".join(cross) + "\n")
    lines += ["", *["#" + s if s.startswith("# ") else s for s in cross]]
    for seed in sorted({r["seed"] for r in chosen} - {42}):
        for dataset in ("ucf", "mivia"):
            lines += ["", *table(dataset, seed)]
    # The aggregate stays separate: it does not replace the uniform s42 comparison.
    groups = defaultdict(list)
    for r in chosen:
        groups[(r["dataset"], r["modello"])].append(r)
    aggregated = []
    for (dataset, model), items in sorted(groups.items()):
        row = {
            "dataset": dataset,
            "modello": model,
            "nome": catalog[model]["nome"],
            "semi": ",".join(str(r["seed"]) for r in sorted(items, key=lambda r: r["seed"])),
            "n_semi": len(items),
        }
        for key in metrics:
            vals = [r[key] for r in items]
            valid = all(v is not None and math.isfinite(v) for v in vals)
            row[key] = statistics.mean(vals) if valid else None
            row[key + "_std"] = statistics.stdev(vals) if valid and len(vals) > 1 else None
        aggregated.append(row)
    rankings = {}
    for dataset in ("ucf", "mivia"):
        if not any(r["dataset"] == dataset for r in chosen):
            continue
        rankings[dataset] = {}
        for metric in metrics:
            ordered = sorted(
                [r for r in aggregated if r["dataset"] == dataset],
                key=lambda r: (
                    r[metric] is None,
                    (r[metric] if metric.endswith("fpr") else -r[metric])
                    if r[metric] is not None
                    else 0,
                    r["nome"],
                ),
            )
            rankings[dataset][metric] = [{"rank": i + 1, **r} for i, r in enumerate(ordered)]
            write_rankings_csv(
                out / dataset / f"classifica_{metric}.csv", rankings[dataset][metric]
            )
    save_json(
        out / "classifiche.json",
        {
            "aggregazione": "mean and standard deviation over the available seeds; use the s42 tables for the uniform comparison",
            "rankings": rankings,
            "celle": chosen,
            "selezione_versioni": choices,
        },
    )
    save_json(out / "selezione_versioni.json", choices)
    write_rankings_csv(out / "selezione_versioni.csv", choices)
    write_rankings_csv(out / "classifiche.csv", aggregated)
    write_rankings_csv(out / "per_run.csv", chosen)
    write_text_atomic(out / "classifiche.md", "\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--solo-classifiche",
        action="store_true",
        help="Regenerate the tables from risultati.json without re-evaluating",
    )
    parser.add_argument(
        "--run", type=Path, help="Pilot on a single UCF cell and its MIVIA counterpart"
    )
    parser.add_argument("--set-ucf", action="append", default=[])
    parser.add_argument("--set-mivia", action="append", default=[])
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "manifests/edp_v2/ucf_crime/split.json"
    )
    parser.add_argument("--guard", type=float, default=8.0)
    parser.add_argument(
        "--da-export",
        type=Path,
        help="Reuse plan and verified inputs of an existing OVAD collection",
    )
    args = parser.parse_args()
    if args.solo_classifiche:
        records = read_json(args.out / "risultati.json")
        for all_items, folder in ((False, "classifiche"), (True, "classifiche_tutte_versioni")):
            rank(records, args.out / folder, tutte=all_items)
        return 0
    cells = find_cells(args.runs, solo=args.run)
    external = find_mivia(args.runs)
    records, errors, plan = [], [], []
    for model, seeds in cells.items():
        for seed, run in seeds.items():
            plan.append(
                {
                    "modello": model,
                    "seed": seed,
                    "ucf": str(run),
                    "mivia": str(external[model][seed])
                    if seed in external.get(model, {})
                    else None,
                }
            )
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": sys.version,
        "protocol_sha256": {
            str(p.relative_to(ROOT)): file_sha256(p)
            for p in sorted((ROOT / "ovad_protocol").rglob("*"))
            if p.suffix in (".py", ".md")
        },
        "integration_sha256": {
            str(p.relative_to(ROOT)): file_sha256(p)
            for p in [
                Path(__file__),
                ROOT / "scripts/esporta_score_ovad.py",
                ROOT / "scripts/classifiche_modelli.py",
            ]
        },
        "piano": plan,
        "modelli_senza_run": sorted(set(MODELS) - set(cells)) if args.run is None else [],
    }
    if args.da_export:
        if args.da_export.resolve() == args.out.resolve():
            raise ValueError("The new collection must be distinct from the source.")
        previous = read_json(args.da_export / "manifest.json")
        if previous["status"] != "complete":
            raise ValueError("The source collection must be complete.")
        plan = previous["piano"]
        provenance.update(
            piano=plan,
            source_collection=str(args.da_export.resolve()),
            source_manifest_sha256=file_sha256(args.da_export / "manifest.json"),
        )

    def prepare_export(run: Path, out: Path, split: str, overrides: list[str]) -> dict:
        if not args.da_export:
            return verified_export(run, out, split, overrides, args.manifest)
        source = args.da_export / out.relative_to(args.out)
        receipt = read_json(source / "provenienza.json")
        for name, expected_hash in receipt["output_sha256"].items():
            if file_sha256(source / name) != expected_hash:
                raise ValueError(f"Altered source export: {source / name}")
        out.mkdir(parents=True, exist_ok=True)
        for name in [*receipt["output_sha256"], "provenienza.json"]:
            shutil.copy2(source / name, out / name)
        return receipt

    provenance["guard"] = args.guard
    save_json(args.out / "manifest.json", {**provenance, "status": "running"})
    for index, item in enumerate(plan, 1):
        model, seed, run = item["modello"], item["seed"], Path(item["ucf"])
        base = args.out / "celle" / run.name
        try:
            prepare_export(run, base / "val", "val", args.set_ucf)
            val_labels, val_scores = (
                read_json(base / "val/labels.json"),
                read_json(base / "val/scores.json"),
            )
            theta = validated_threshold(val_labels, val_scores, args.guard)
            val_counts = details(val_labels, val_scores, theta, args.guard, base / "val")
            save_json(
                base / "val/soglia.json",
                {
                    "threshold": theta,
                    "source": "validation: Youden point on the judgeable rows",
                    "conteggi": val_counts,
                },
            )
        except Exception as exc:
            errors.append({"run": run.name, "fase": "val", "errore": str(exc)})
            print(f"ERROR {run.name}: {exc}", flush=True)
            continue
        for dataset in ("ucf", "mivia"):
            source = item[dataset]
            if source is None:
                continue
            dest = base / dataset
            try:
                receipt = prepare_export(
                    Path(source),
                    dest / "input",
                    "test",
                    args.set_ucf if dataset == "ucf" else args.set_mivia,
                )
                labels, scores = (
                    read_json(dest / "input/labels.json"),
                    read_json(dest / "input/scores.json"),
                )
                if set(labels) & set(val_labels):
                    raise ValueError("Videos shared between validation and test: check the splits.")
                result = evaluate_ovad(labels, scores, val_labels, val_scores, guard=args.guard)
                if result["threshold"]["value"] != theta:
                    raise ValueError("The test threshold does not match the validation one.")
                metrics = result["levels"]["anomalies"]
                counts = details(labels, scores, result["threshold"]["value"], args.guard, dest)
                if any(metrics[k] != v for k, v in counts.items()):
                    raise ValueError("The intermediate tables do not reproduce the OVAD counts.")
                if not math.isfinite(metrics["mean"]):
                    raise ValueError("Protocol mean undefined: do not publish a ranking.")
                save_json(dest / "results.json", result)
                write_text_atomic(dest / "report.txt", report(result) + "\n")
                # The signature excludes scores/predictions, which must vary across models.
                cohort = [
                    {k: v for k, v in r.items() if k not in ("score", "prediction")}
                    for r in read_json(dest / "righe.json")
                ]
                cohort_sha = hashlib.sha256(json.dumps(cohort, sort_keys=True).encode()).hexdigest()
                record = {
                    "dataset": dataset,
                    "modello": model,
                    "nome": MODELS[model]["nome"],
                    "seed": seed,
                    "run_id": Path(source).name,
                    **result["parameters"],
                    **metrics,
                    "labels_sha256": receipt["output_sha256"]["labels.json"],
                    "cohort_sha256": cohort_sha,
                    "results": str((dest / "results.json").relative_to(args.out)),
                }
                records.append(record)
                save_json(
                    dest / "verifica.json",
                    {
                        "status": "complete",
                        "conteggi": counts,
                        "val_source_run": run.name,
                        "val_scores_sha256": file_sha256(base / "val/scores.json"),
                        "results_sha256": file_sha256(dest / "results.json"),
                        "cohort_sha256": cohort_sha,
                    },
                )
                print(
                    f"[{index}/{len(plan)}] {dataset} {run.name}: mean={100 * metrics['mean']:.4f}",
                    flush=True,
                )
            except Exception as exc:
                errors.append({"run": source, "fase": dataset, "errore": str(exc)})
                print(f"ERROR {source}: {exc}", flush=True)
        save_json(args.out / "progress.json", {"complete": len(records), "errors": errors})
    save_json(args.out / "risultati.json", records)
    write_rankings_csv(args.out / "risultati.csv", records)
    for all_items, folder in ((False, "classifiche"), (True, "classifiche_tutte_versioni")):
        rank(records, args.out / folder, tutte=all_items)
    expected = sum(1 + (p["mivia"] is not None) for p in plan)
    save_json(
        args.out / "manifest.json",
        {
            **provenance,
            "status": "complete" if not errors and len(records) == expected else "incomplete",
            "n_attese": expected,
            "n_complete": len(records),
            "errors": errors,
        },
    )
    return 1 if errors or len(records) != expected else 0


if __name__ == "__main__":
    raise SystemExit(main())
