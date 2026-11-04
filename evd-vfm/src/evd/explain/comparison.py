"""Paired comparison against random controls, with per-video bootstrap."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_curves(folder: Path, index: int, record: dict) -> list[dict]:
    """Return the deletion/insertion rows of a window, from either source.

    The base campaign writes them to ``perturbations_{i}.json`` (one row per measurement);
    the extension stores parallel columns in ``artifacts/w{i}_deletion_insertion.npz``
    with the strings in the receipt. Callers do not need to know which one produced them.

    :param record: the window row in ``campaign.json``, for the receipt.
    :return: rows in the form consumed by :func:`compare_campaigns`; empty if the window
        has no curves.
    """
    path = folder / f"perturbations_{index:05d}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    payload = folder / "artifacts" / f"w{index:05d}_deletion_insertion.npz"
    receipt = (record.get("checks") or {}).get("deletion_insertion")
    if not payload.exists() or receipt is None:
        return []
    with np.load(payload) as stored:
        columns = {name: stored[name] for name in stored.files}
    return [
        {
            "method": receipt["methods"][int(m)],
            "fill": receipt["fills"][int(f)],
            "mode": receipt["modes"][int(md)],
            "opposition": bool(o),
            "random_seed": None if int(s) < 0 else int(s),
            "actual_fraction": float(fr),
            "logit": float(lg),
            "score": float(sc),
        }
        for m, f, md, o, s, fr, lg, sc in zip(
            columns["method"],
            columns["fill"],
            columns["mode"],
            columns["opposition"],
            columns["random_seed"],
            columns["actual_fraction"],
            columns["logit"],
            columns["score"],
            strict=True,
        )
    ]


def compare_campaigns(
    folders: list[Path], categories: dict[str, str], *, seed: int = 42, replicates: int = 2000
) -> dict:
    """Summarize curve advantages, separating model, category and fill.

    :param folders: per-video folders containing campaign.json and perturbations.
    :param categories: category fixed by the DoRA scores per window_id.
    :return: per-video bootstrap intervals; no promotion with fewer than two videos.
    :raises ValueError: on a duplicated window.
    """
    differences = defaultdict(list)
    checks = defaultdict(list)
    costs = defaultdict(list)
    seen = set()
    for folder in folders:
        meta = json.loads((folder / "campaign.json").read_text(encoding="utf-8"))
        if meta["status"] != "complete":
            continue
        run = meta["run_id"]
        for i, window in enumerate(meta["windows"]):
            identity = (run, window["window_id"])
            if identity in seen:
                raise ValueError(f"Duplicated window in the comparison: {identity}")
            seen.add(identity)
            category = categories.get(window["window_id"], "unclassified")
            video = window["window"]["video_id"]
            checks[run].append(window["checks"])
            costs[run].append(
                {k: window[k] for k in ("forward_s", "explanation_s", "total_s", "peak_gpu_bytes")}
            )
            curves = load_curves(folder, i, window)
            grouped = defaultdict(list)
            for row in curves:
                grouped[
                    (row["method"], row["fill"], row["mode"], row["opposition"], row["random_seed"])
                ].append(row)
            auc = {}
            for key, rows in grouped.items():
                rows.sort(key=lambda r: r["actual_fraction"])
                for metric in ("logit", "score"):
                    y = np.array([r[metric] for r in rows])
                    x = np.array([r["actual_fraction"] for r in rows])
                    auc[key + (metric,)] = float(np.sum((y[1:] + y[:-1]) * np.diff(x) / 2))
            for (method, fill, mode, opposition, _random_seed, metric), value in auc.items():
                if method == "random":
                    continue
                random_values = [
                    v
                    for (m, f, md, op, rs, met), v in auc.items()
                    if m == "random" and f == fill and md == mode and met == metric
                ]
                if len(random_values) != 5:
                    continue
                direction = (1 if mode == "insertion" else -1) * (-1 if opposition else 1)
                advantage = direction * (value - float(np.mean(random_values)))
                key = (run, category, method, fill, mode, opposition, metric)
                differences[key].append((video, advantage))
    summaries = []
    rng = np.random.default_rng(seed)
    for key, values in sorted(differences.items()):
        by_video = defaultdict(list)
        for video, value in values:
            by_video[video].append(value)
        means = np.array([np.mean(v) for _, v in sorted(by_video.items())])
        sampled = means[rng.integers(len(means), size=(replicates, len(means)))].mean(1)
        lower, upper = np.percentile(sampled, [2.5, 97.5])
        summaries.append(
            dict(
                zip(
                    ("run", "category", "method", "fill", "mode", "opposition", "metric"),
                    key,
                    strict=True,
                )
            )
            | {
                "n_videos": len(means),
                "n_windows": len(values),
                "advantage": float(means.mean()),
                "ci95": [float(lower), float(upper)] if len(means) >= 2 else None,
                "better_than_random": bool(len(means) >= 2 and lower > 0),
            }
        )
    return {
        "bootstrap_unit": "video",
        "replicates": replicates,
        "seed": seed,
        "summaries": summaries,
        "checks": dict(checks),
        "costs": dict(costs),
        "recommendation": "No automatic promotion: check development coverage, both fills, numerical controls and cost; disagreement = inconclusive.",
    }
