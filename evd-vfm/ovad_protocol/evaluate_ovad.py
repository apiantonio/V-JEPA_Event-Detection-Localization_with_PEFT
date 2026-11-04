"""Evaluation for the OVAD framework: anomaly detection, one entry per video.

The model gives one anomaly score per chunk, without a query: it looks for any anomaly in the video.
The labels of a video are all its annotated anomalies, as triples [start, end, class]; the events of
the protocol are these intervals, whatever the class. Rows, threshold and metrics are those of the
protocol, computed by protocol_core/: this script only decides which data go in.

Labels (one JSON file):

    {
      "video_001": {"duration": 73.07, "anomalies": [[16.57, 24.03, "Fighting"]]},
      "video_002": {"duration": 120.0, "anomalies": []}
    }

Scores: the format of protocol_core/inputs.py, with the same key (the video).

Command line:

    python evaluate_ovad.py \\
        --labels test/labels.json --scores test/scores.json \\
        --val-labels val/labels.json --val-scores val/scores.json \\
        --output results.json

Instead of the validation set, --threshold 0.35 can be passed. From Python: evaluate_ovad(...).
"""
from __future__ import annotations

import argparse
import json
import sys

from protocol_core import InputError, evaluate_levels, read_json, report, to_json

LEVEL = "anomalies"


def from_triples(labels: dict, where: str) -> dict:
    """OVAD labels in the format of the core: the intervals of all the anomalies of the video.

    The class is checked but does not enter the metrics: the model looks for any anomaly."""
    if not isinstance(labels, dict):
        raise InputError(f"{where}: the labels must be an object keyed by video")
    out = {}
    for video, lab in labels.items():
        if not isinstance(lab, dict) or "duration" not in lab or "anomalies" not in lab:
            raise InputError(f"{where}: {video}: 'duration' and 'anomalies' are required")
        if not isinstance(lab["anomalies"], list):
            raise InputError(f"{where}: {video}.anomalies must be a list of [start, end, class]")
        iv = []
        for i, t in enumerate(lab["anomalies"]):
            if not isinstance(t, (list, tuple)) or len(t) != 3:
                raise InputError(f"{where}: {video}.anomalies[{i}] must be [start, end, class]")
            if not isinstance(t[2], str) or not t[2].strip():
                raise InputError(f"{where}: {video}.anomalies[{i}]: the class must be a non-empty string")
            iv.append([t[0], t[1]])
        out[video] = {"duration": lab["duration"], "intervals": iv}
    return out


def evaluate_ovad(labels: dict, scores: dict, val_labels: dict | None = None,
                  val_scores: dict | None = None, threshold: float | None = None,
                  guard: float | None = None) -> dict:
    """labels: {video: {"duration", "anomalies": [[start, end, class], ...]}}; scores: the format of
    protocol_core/inputs.py. The same for the validation set. The threshold comes from the validation
    set or is passed with threshold=."""
    if (val_labels is None) != (val_scores is None):
        raise InputError("the validation set needs both labels and scores")
    val = None if val_labels is None else {LEVEL: (from_triples(val_labels, "validation"), val_scores)}
    return evaluate_levels({LEVEL: (from_triples(labels, "test"), scores)}, val, threshold, guard)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Protocol for OVAD: anomaly detection, one entry per video.")
    ap.add_argument("--labels", required=True, metavar="FILE")
    ap.add_argument("--scores", required=True, metavar="FILE")
    ap.add_argument("--val-labels", metavar="FILE")
    ap.add_argument("--val-scores", metavar="FILE")
    ap.add_argument("--threshold", type=float, help="threshold by hand, instead of the validation set")
    ap.add_argument("--guard", type=float, help="guard g in seconds (default: the chunk size)")
    ap.add_argument("--output", help="JSON file in which to write all the results")
    a = ap.parse_args(argv)
    try:
        r = evaluate_ovad(read_json(a.labels), read_json(a.scores),
                          read_json(a.val_labels) if a.val_labels else None,
                          read_json(a.val_scores) if a.val_scores else None,
                          a.threshold, a.guard)
    except (InputError, OSError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(report(r))
    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            json.dump(to_json(r), f, indent=2, ensure_ascii=False)
        print(f"\nfull results in {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
