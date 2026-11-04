"""From label and score files to the entries of the protocol.

Labels (one JSON file per data set; in OV-CAVED one per query granularity):

    {
      "<key>": {"duration": 73.07, "intervals": [[16.57, 24.03]]},
      "<key>": {"duration": 120.0, "intervals": []}
    }

  <key>      a free identifier of the entry: the video, or the video and the query together.
  duration   the duration of the video in seconds.
  intervals  the intervals in which the queried event is present, [start, end) in seconds.
             Empty list for a video in which the event does not occur (for example a normal video).

Scores (one JSON file for every label file):

    {
      "hyperparameters": {"chunk_size": 8, "asking_rate": 4, "grid": "start"},
      "scores": {
        "<key>": [0.01, 0.02, 0.35, ...],
        "<key>": [{"start": 0.0, "end": 8.0, "score": 0.01}, ...]
      }
    }

  chunk_size   c, the seconds of video the model reads to produce one score.
  asking_rate  h, the seconds between two consecutive queries to the model (the protocol uses 4).
  grid         where chunk j lies when the scores are given as a list of numbers:
                 "start"     [j*h, j*h + c]            (the most common sliding window)
                 "centered"  centered on (j + 1/2)*h   [(j+1/2)*h - c/2, (j+1/2)*h + c/2]
               In both cases the extent is clipped to [0, duration].
  For every key the scores are either a list of numbers, one per chunk in grid order, or a list of
  objects with the explicit extent in seconds (start, end) and the score.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .metrics import H, make_entry

GRIDS = ("start", "centered")


class InputError(ValueError):
    """A label or score file does not follow the format."""


def read_json(path) -> dict:
    with open(Path(path), encoding="utf-8") as f:
        return json.load(f)


def _number(x, where: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise InputError(f"{where}: expected a finite number, found {x!r}")
    return float(x)


def hyperparameters(scores: dict, where: str = "scores") -> dict:
    """Reads and checks the "hyperparameters" block of a score file."""
    if not isinstance(scores, dict) or "hyperparameters" not in scores or "scores" not in scores:
        raise InputError(f"{where}: the two keys 'hyperparameters' and 'scores' are required")
    hp = scores["hyperparameters"]
    if not isinstance(hp, dict):
        raise InputError(f"{where}: 'hyperparameters' must be an object")
    c = _number(hp.get("chunk_size"), f"{where}: hyperparameters.chunk_size")
    h = _number(hp.get("asking_rate"), f"{where}: hyperparameters.asking_rate")
    grid = hp.get("grid", "start")
    if c <= 0 or h <= 0:
        raise InputError(f"{where}: chunk_size and asking_rate must be positive")
    if h > c:
        raise InputError(f"{where}: asking_rate ({h:g} s) larger than chunk_size ({c:g} s): "
                         "parts of the video would never be seen by the model")
    if grid not in GRIDS:
        raise InputError(f"{where}: unknown grid {grid!r}, use 'start' or 'centered'")
    return {"chunk_size": c, "asking_rate": h, "grid": grid}


def extents_from_indices(n: int, c: float, h: float, duration: float, grid: str, where: str) -> list:
    """The extents of chunks 0..n-1 on the grid, clipped to [0, duration]."""
    out = []
    for j in range(n):
        a = j * h if grid == "start" else (j + 0.5) * h - c / 2
        b = a + c
        a, b = max(0.0, a), min(duration, b)
        if b <= a:
            raise InputError(f"{where}: chunk {j} falls outside the video (duration {duration:g} s): "
                             "there are more scores than chunks in the video")
        out.append((a, b))
    return out


def build_entries(labels: dict, scores: dict, where: str = "") -> tuple[list[dict], dict]:
    """The entries of a data set and its hyperparameters.

    Every label key must have its scores and vice versa: a key missing on one side is an error,
    not an entry silently skipped."""
    hp = hyperparameters(scores, where or "scores")
    sc = scores["scores"]
    if not isinstance(labels, dict) or not isinstance(sc, dict):
        raise InputError(f"{where}: labels and 'scores' must be objects keyed by entry")
    no_scores = sorted(set(labels) - set(sc))
    no_labels = sorted(set(sc) - set(labels))
    if no_scores or no_labels:
        raise InputError(f"{where}: keys without scores {no_scores[:5]} ({len(no_scores)}), "
                         f"keys without labels {no_labels[:5]} ({len(no_labels)})")
    out = []
    for k in sorted(labels):
        lab, v = labels[k], sc[k]
        if not isinstance(lab, dict) or "duration" not in lab or "intervals" not in lab:
            raise InputError(f"{where}: {k}: the label needs 'duration' and 'intervals' "
                             "(an empty list for a video without the event)")
        duration = _number(lab["duration"], f"{where}: {k}.duration")
        if duration <= 0:
            raise InputError(f"{where}: {k}.duration must be positive")
        if not isinstance(lab["intervals"], list):
            raise InputError(f"{where}: {k}.intervals must be a list of [start, end]")
        iv = []
        for i, x in enumerate(lab["intervals"]):
            if not isinstance(x, (list, tuple)) or len(x) != 2:
                raise InputError(f"{where}: {k}.intervals[{i}] must be [start, end]")
            a, b = _number(x[0], f"{where}: {k}.intervals[{i}]"), _number(x[1], f"{where}: {k}.intervals[{i}]")
            if b <= a:
                raise InputError(f"{where}: {k}.intervals[{i}] = [{a:g}, {b:g}] has end <= start")
            iv.append((a, b))
        if not isinstance(v, list) or not v:
            raise InputError(f"{where}: {k}: a non-empty list of scores is required")
        if all(isinstance(x, dict) for x in v):
            extents = []
            for i, x in enumerate(v):
                a = _number(x.get("start"), f"{where}: {k}[{i}].start")
                b = _number(x.get("end"), f"{where}: {k}[{i}].end")
                a, b = max(0.0, a), min(duration, b)
                if b <= a:
                    raise InputError(f"{where}: {k}[{i}]: empty extent after clipping to [0, {duration:g}]")
                extents.append((a, b))
            values = [_number(x.get("score"), f"{where}: {k}[{i}].score") for i, x in enumerate(v)]
        else:
            values = [_number(x, f"{where}: {k}[{i}]") for i, x in enumerate(v)]
            extents = extents_from_indices(len(values), hp["chunk_size"], hp["asking_rate"], duration,
                                           hp["grid"], f"{where}: {k}")
        e = make_entry(extents, values, iv)
        e["key"] = k
        out.append(e)
    return out, hp


def protocol_warnings(hp: dict) -> list[str]:
    """Departures from the protocol that do not prevent the computation but must be declared."""
    out = []
    if abs(hp["asking_rate"] - H) > 1e-9:
        out.append(f"asking rate {hp['asking_rate']:g} s: the main protocol uses {H:g} s")
    return out
