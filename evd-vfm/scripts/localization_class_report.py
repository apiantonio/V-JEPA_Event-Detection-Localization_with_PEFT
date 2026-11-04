"""One interactive page with every localization metric of the class-routing campaign (CPU).

Reads what the campaign folder holds and writes ``evaluation/report_localizzazione.html``, a
single offline file with its data inside (decided on 27 October 2026: one file, always up to
date). It is regenerated at the end of every ``localization_class_ap.py evaluate`` and must be
rerun after every pull from the cluster; a source that is not there yet is shown as missing.

- **AP, RBDC, RBDR** (``evaluation/ap-rbdc-<stage>/``, else the AP-only ``ap-iou-<stage>/``):
  per event at the 19 IoU thresholds 0.05:0.95, then every summary of
  ``evd.eval.localization_ap.IOU_SUMMARIES`` per event, then the class means and the mean over
  the thirteen classes; AP, RBDC and RBDR pooled over the dataset's images as well.
- **Real use** (28 October 2026, ``evaluation/ap-real-<stage>/`` and ``evaluation/real-<stage>*``):
  the same metrics with boxes only on the chunks the detector flags, the pooled ones with the
  negative images, and the IoU with the share of negative images that get a box.
- **IoU with a mass threshold** (``evaluation/classifica_iou.csv``), the 88 configurations.
- **Detectors and classifiers**: ``detection/ovad_guard30/risultati.json`` and
  ``evaluation/metriche_classificatori_proposta_b.json``.

The macro value at each threshold is checked against ``summary_by_threshold.csv`` of the
evaluator, so the page and the evaluator cannot disagree.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402

from evd.eval.localization_ap import IOU_SUMMARIES, summary_columns  # noqa: E402
from evd.utils.hashing import file_sha256  # noqa: E402

CAMPAIGN = ROOT / "runs" / "localization-class-routing-20261021"
TEMPLATE = Path(__file__).with_suffix(".html")
#: Detection stages of the AP and their detector, in the order of the page.
STAGES = (("campaign", "LoRA+VPT base"), ("b3s", "B3-sum"), ("b3g", "B3-GradNorm"))
#: Evaluation regimes: boxes on the annotated test chunks (27 October 2026), or only on the
#: chunks the detector flags, every test chunk seen (real use, 28 October 2026).
REGIMES = (("annotated", "annotated chunks"), ("real", "real use"))
#: Threshold rules of the IoU experiments, by suffix (as ``operations/ranking.py`` names them).
IOU_RULES = {
    "phase1": "fixed_55",
    "phase2": "calibrated",
    "mass50": "fixed_50",
    "mass70": "fixed_70",
}
#: Names of the detectors in ``classifica_iou.csv`` (its values, written by the campaign) → page.
IOU_DETECTORS = {"campagna": "LoRA+VPT base", "B3-somma": "B3-sum", "B3-GradNorm": "B3-GradNorm"}
DATASETS = (("ucf_test", "FS-UCF-Crime"), ("mivia_test", "MIVIA-Crime"))
#: GT readings, from the most joined to the most split.
GT_ORDER = ("union", "touching", "instances")
METRICS = ("ap", "rbdc", "rbdr")
POOLED = ("ap", "rbdc", "rbdr")
#: Statistics kept per class (the page keeps all of them for the mean over the classes).
CLASS_STATS = (
    *(k for k in IOU_SUMMARIES if ":0.05:" not in k or k == "[0.5:0.05:0.95]"),
    "@0.10",
    "@0.30",
    "@0.50",
    "@0.70",
)
MISSING = 65535


def _source(evaluation: Path, stage: str, regime: str) -> tuple[Path | None, str | None]:
    """:return: the folder with this stage's metrics and its kind, the newest first."""
    candidates = (
        ((f"ap-rbdc-{stage}", "full"), (f"ap-iou-{stage}", "ap"))
        if regime == "annotated"
        else ((f"ap-real-{stage}", "full"),)
    )
    for folder, kind in candidates:
        if (evaluation / folder / "summary.json").exists():
            return evaluation / folder, kind
    return None, None


def _stats(values: np.ndarray, thresholds: list[float], keys: list[str]) -> np.ndarray:
    """:return: ``values`` (``..., T``) as the statistics ``keys``: single thresholds and means."""
    out = []
    for key in keys:
        if key.startswith("@"):
            out.append(values[..., summary_columns(thresholds, [float(key[1:])])[0]])
        else:
            out.append(values[..., summary_columns(thresholds, IOU_SUMMARIES[key])].mean(-1))
    return np.stack(out, -1)


def _check(folder: Path, dataset: str, metric: str, macro: np.ndarray, axes: dict) -> None:
    """Compare the macro value at each threshold with the evaluator's CSV (6 decimals)."""
    rows = list(csv.DictReader((folder / "summary_by_threshold.csv").open(encoding="utf-8")))
    for row in rows:
        if row["dataset"] != dataset or row.get("metric", "ap") != metric:
            continue
        index = (
            axes["configuration"].index(row["configuration"]),
            axes["variant"].index(row["variant"]),
            axes["score"].index(row["score"]),
            axes["gt"].index(row["gt"]),
        )
        stored = [float(v) for k, v in row.items() if k.startswith(("@", "AP@"))]
        if np.abs(macro[index][: len(stored)] - stored).max() > 6e-7:
            raise ValueError(f"{folder.name} {dataset} {metric}: the page disagrees with the CSV")


def _encode(array: np.ndarray) -> str:
    coded = np.where(np.isnan(array), MISSING, np.round(np.clip(array, 0, 1) * 10000))
    return base64.b64encode(coded.astype("<u2").tobytes()).decode("ascii")


def metric_data(evaluation: Path) -> dict:
    """:return: the methods, their statistics and the provenance of every stage folder."""
    thresholds = [round(0.05 * i, 2) for i in range(1, 20)]
    keys = [f"@{t:.2f}" for t in thresholds] + list(IOU_SUMMARIES)
    class_index = [keys.index(k) for k in CLASS_STATS]
    methods, loaded, sources = [], [], []
    classes = None
    for (regime, _), (stage, detector) in ((r, s) for r in REGIMES for s in STAGES):
        folder, kind = _source(evaluation, stage, regime)
        info = {
            "stage": stage,
            "detector": detector,
            "regime": regime,
            "folder": folder and folder.name,
        }
        if folder is None:
            sources.append(info)
            continue
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        info.update(kind=kind, datasets={})
        start, names = len(methods), None
        for d, (dataset, _) in enumerate(DATASETS):
            axes = json.loads((folder / f"axes_{dataset}.json").read_text(encoding="utf-8"))
            name = "metrics_per_event" if kind == "full" else "ap_per_event"
            path = folder / f"{name}_{dataset}.npz"
            stored = np.load(path)
            if not np.allclose(axes["iou_threshold"], thresholds):
                raise ValueError(f"{folder.name}: unexpected IoU thresholds")
            event_classes = [e["class"] for e in axes["event"]]
            if classes is None:
                classes = sorted(set(event_classes))
                if len(classes) != 13:
                    raise ValueError("The page expects thirteen event classes")
            axis_names = (axes["configuration"], axes["variant"], axes["score"])
            if names is None:
                names = axis_names
                methods.extend(
                    [len(sources), c, v, s] for c in names[0] for v in names[1] for s in names[2]
                )
            elif axis_names != names:
                raise ValueError(f"{folder.name}: the datasets disagree on the methods")
            per_event = {m: stored[m] for m in METRICS if m in stored.files}
            info["datasets"][dataset] = {
                "events": len(event_classes),
                "events_per_class": {c: event_classes.count(c) for c in classes},
                "images": summary["datasets"][dataset]["images"],
                "ground_truth": summary["datasets"][dataset].get("ground_truth"),
                "metrics": [m for m in METRICS if m in per_event],
                "pooled": [m for m in POOLED if f"pooled_{m}" in stored.files],
                "gt": axes["gt"],
                "negative": summary["datasets"][dataset].get("negative"),
                "sha256": file_sha256(path),
            }
            loaded.append((start, d, axes, per_event, stored, folder))
        info["configurations"] = names[0]
        sources.append(info)
    shape = (len(methods), len(DATASETS), len(GT_ORDER))
    macro = np.full((*shape, len(METRICS), len(keys)), np.nan)
    per_class = np.full((*shape, len(METRICS), 13, len(class_index)), np.nan)
    pooled = np.full((*shape, len(POOLED), len(keys)), np.nan)
    for start, d, axes, per_event, stored, folder in loaded:
        n_c, n_v, n_s = len(axes["configuration"]), len(axes["variant"]), len(axes["score"])
        rows = slice(start, start + n_c * n_v * n_s)
        members = [[i for i, e in enumerate(axes["event"]) if e["class"] == c] for c in classes]
        for g_src, gt in enumerate(axes["gt"]):
            g = GT_ORDER.index(gt)
            for k, metric in enumerate(METRICS):
                if metric not in per_event:
                    continue
                values = per_event[metric][:, :, :, g_src]  # C,V,S,E,T
                stats = _stats(values, axes["iou_threshold"], keys)  # C,V,S,E,K
                classes_mean = np.stack([stats[:, :, :, m].mean(3) for m in members], 3)
                mean = classes_mean.mean(3)  # C,V,S,K
                macro[rows, d, g, k] = mean.reshape(-1, len(keys))
                per_class[rows, d, g, k] = classes_mean[..., class_index].reshape(
                    -1, 13, len(class_index)
                )
            for k, metric in enumerate(POOLED):
                if f"pooled_{metric}" in stored.files:
                    values = stored[f"pooled_{metric}"][:, :, :, g_src]
                    pooled[rows, d, g, k] = _stats(values, axes["iou_threshold"], keys).reshape(
                        -1, len(keys)
                    )
        # The evaluator's macro at every threshold, from the same arrays (and checked).
        for k, metric in enumerate(METRICS):
            if metric in per_event:
                full = macro[rows, d][:, [GT_ORDER.index(g) for g in axes["gt"]], k, :19]
                _check(folder, DATASETS[d][0], metric, full.reshape(n_c, n_v, n_s, -1, 19), axes)
    return {
        "thresholds": thresholds,
        "stats": keys,
        "classStats": list(CLASS_STATS),
        "classes": classes or [],
        "methods": methods,
        "sources": sources,
        "macro": _encode(macro),
        "perClass": _encode(per_class),
        "pooled": _encode(pooled),
    }


def iou_data(evaluation: Path) -> dict | None:
    """:return: the 88 IoU configurations with a mass threshold, overall and per class."""
    path = evaluation / "classifica_iou.csv"
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    table: dict = {}
    for row in rows:
        key = (IOU_DETECTORS[row["detector"]], row["configuration"], row["threshold"])
        entry = table.setdefault(
            key, {"detector": key[0], "configuration": key[1], "threshold": key[2], "values": {}}
        )
        entry["values"].setdefault(row["dataset"], {})[row["class"]] = float(row["iou"])
    return {"rows": list(table.values()), "sha256": file_sha256(path)}


def iou_real_data(evaluation: Path) -> dict | None:
    """:return: the IoU configurations in real use, with the share of negative images boxed.

    Read from the evaluations ``real-<stage>``, ``real-<stage>-mass50`` and ``-mass70``;
    ``P07-global-bridge70`` (P07 at 70%) is left out, ``P07-global-mass70`` is the same.
    """
    rows, sources = [], []
    for stage, detector in STAGES:
        for suffix in ("", "-mass50", "-mass70"):
            folder = evaluation / f"real-{stage}{suffix}"
            if not (folder / "summary.json").exists():
                continue
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            alarms = summary.get("false_alarms", {})
            sources.append({"folder": folder.name, "detector": detector})
            for experiment in summary["datasets"]["ucf_test"]:
                name, rule = experiment.rsplit("-", 1)
                if name == "P07-global-bridge70":
                    continue
                entry = {
                    "detector": detector,
                    "configuration": name,
                    "threshold": IOU_RULES[rule],
                    "values": {},
                    "falseAlarms": {},
                }
                for dataset, _ in DATASETS:
                    result = json.loads(
                        (folder / "results" / dataset / f"{experiment}.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    entry["values"][dataset] = {
                        **{r["class"]: r["iou"] for r in result["per_class"]},
                        "MEAN": result["value"],
                    }
                    share = alarms.get(dataset, {}).get("image_share", {}).get(experiment)
                    entry["falseAlarms"][dataset] = share
                rows.append(entry)
    if not rows:
        return None
    return {"rows": rows, "sources": sources}


def context_data(campaign: Path) -> dict:
    """:return: the detectors' binary metrics and the routing classifiers' metrics."""
    out: dict = {}
    ovad = campaign / "detection" / "ovad_guard30" / "risultati.json"
    if ovad.exists():
        out["detectors"] = json.loads(ovad.read_text(encoding="utf-8"))
    classifiers = campaign / "evaluation" / "metriche_classificatori_proposta_b.json"
    if classifiers.exists():
        models = json.loads(classifiers.read_text(encoding="utf-8"))["models"]
        out["classifiers"] = [
            {
                "model": model,
                "dataset": dataset,
                "chunks": values[dataset]["chunks"],
                "accuracy": values[dataset]["accuracy"],
                "balanced_accuracy": values[dataset]["balanced_accuracy"],
                "macro_f1": values[dataset]["macro"]["f1-score"],
                "weighted_f1": values[dataset]["weighted"]["f1-score"],
            }
            for model, values in models.items()
            for dataset in ("ucf_test", "mivia_test", "pooled")
            if isinstance(values.get(dataset), dict) and "accuracy" in values[dataset]
        ]
    return out


def build(campaign: Path, output: Path) -> Path:
    """Write the page and return its path."""
    evaluation = campaign / "evaluation"
    data = {
        "generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "campaign": campaign.name,
        "datasets": [{"key": k, "label": v} for k, v in DATASETS],
        "regimes": [{"key": k, "label": v} for k, v in REGIMES],
        "gt": list(GT_ORDER),
        "metrics": list(METRICS),
        "pooledMetrics": list(POOLED),
        **metric_data(evaluation),
        "iou": iou_data(evaluation),
        "iouReal": iou_real_data(evaluation),
        **context_data(campaign),
    }
    page = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    )
    output.write_text(page, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN)
    parser.add_argument("--output", type=Path, help="default: <campaign>/evaluation/...")
    args = parser.parse_args()
    output = args.output or args.campaign / "evaluation" / "report_localizzazione.html"
    path = build(args.campaign, output)
    print(f"{path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
