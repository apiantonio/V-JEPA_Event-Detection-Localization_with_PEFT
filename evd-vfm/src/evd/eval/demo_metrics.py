"""Single-video diagnostics on real scores and the effective geometry."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from evd.annotations.schema import window_label
from evd.eval.metrics import detection_latency, event_prf


def saved_labels(path: Path, video_id: str) -> dict[str, int]:
    """Return the labels of the saved windows only, never transferred to a new grid.

    :param path: score file.
    :param video_id: video identifier.
    :return: window id to label.
    """
    if not path.is_file():
        return {}
    labels = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["video_id"] == video_id and row.get("label") in (0, 1):
                labels[row["window_id"]] = int(row["label"])
    return labels


def _finite(value):
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def compute_timing(costs):
    """Return the measured per-window timings, excluding frame waits and weight loading."""
    if not costs:
        return {}
    return {
        "compute_mean_ms": float(np.mean(costs)),
        "compute_p95_ms": float(np.percentile(costs, 95)),
        "compute_total_s": sum(costs) / 1000,
        "measured_windows": len(costs),
    }


def metriche_video(
    scores,
    meta,
    threshold,
    *,
    annotation=None,
    labels=None,
    timing=None,
    complete=True,
    threshold_source="val_youden",
):
    """Compute chunk P/R/F1 and AUC/AP; events and latency use the shared metrics.

    Live metrics are provisional, computed on the windows received so far. Grid latency
    excludes compute and is reported on detected events only; without temporal
    annotations no events are inferred from chunk labels.
    """
    values = np.asarray([s.anomaly for s in scores], dtype=float)
    positive = values >= threshold
    output = {
        "complete": complete,
        "n_windows": len(scores),
        "above_threshold": int(positive.sum()),
        "threshold": threshold,
        "ground_truth": "annotations"
        if annotation is not None
        else "saved_labels"
        if labels
        else None,
        "chunk": None,
        "events": None,
        "latency_grid": None,
        "timing": dict(timing or {}),
    }
    truth = None
    if annotation is not None:
        truth = [window_label(annotation, *s.window.frame_span) for s in scores]
    elif labels and all(s.window.window_id in labels for s in scores):
        truth = [labels[s.window.window_id] for s in scores]
    if truth is not None and len(truth):
        y = np.asarray(truth, dtype=bool)
        tp = int((positive & y).sum())
        fp = int((positive & ~y).sum())
        fn = int((~positive & y).sum())
        tn = int((~positive & ~y).sum())
        both = bool(y.any() and (~y).any())
        output["chunk"] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp + fn else None,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None,
            "accuracy": (tp + tn) / len(y),
            "auc": float(roc_auc_score(y, values)) if both else None,
            "ap": float(average_precision_score(y, values)) if both else None,
        }
    if annotation is not None:
        mapped = {meta.video_id: scores}
        gt = {meta.video_id: annotation}
        output["events"] = event_prf(mapped, gt, threshold, source=threshold_source).as_dict()
        event = output["events"]
        if not event["n_eventi"]:
            event["recall"] = event["f1"] = None
        if not event["n_allarmi"]:
            event["precision"] = None
        output["latency_grid"] = detection_latency(
            mapped,
            gt,
            {meta.video_id: meta.fps},
            threshold=threshold,
            rule="intersezione",
        ).as_dict()
    return _finite(output)
