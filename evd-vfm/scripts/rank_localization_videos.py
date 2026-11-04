"""Rank the videos and the single events by the quality of a run's localization boxes.

Reads the evaluator output of the campaign (``results/<dataset>/<method>.json``, one IoU per
annotated event) for the method the demonstrator shows by default, after checking that it
was computed on the same sealed predictions the boxes were exported from
(``runs/<run_id>/localization/index.json``). An event scores its evaluator IoU; a video the
mean IoU of its annotated events, the same unweighted step the metric takes from events to
classes. The event intervals come from the annotation project of the ground-truth snapshot
stored with the export. Writes two CSVs per run and, optionally, Markdown reports
comparing the runs.

    python scripts/rank_localization_videos.py \
        --run runs/e3s_ucf_vjepa21l_vptshallow-n8-su-lora-r32_s42 \
        --run runs/e3r_ucf_vjepa21l_dora-r32-qkvomlp_s42 \
        --markdown <video report>.md --markdown-events <event report>.md
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from apps.demo.explanation import model_label  # noqa: E402
from apps.demo.localization import FOLDER  # noqa: E402
from scripts.export_demo_localization import method_sources  # noqa: E402

from evd.utils.io import read_json  # noqa: E402

#: Result folders of the campaign and the dataset name each one carries.
DATASETS = {"ucf_test": "ucf_crime", "mivia_test": "mivia_crime"}
DATASET_LABELS = {"ucf_crime": "UCF-Crime", "mivia_crime": "MIVIA-Crime"}
CSV_FIELDS = ("rank", "dataset", "video_id", "classes", "events", "planes", "iou", "best_event_iou")
EVENT_FIELDS = (
    "rank",
    "dataset",
    "video_id",
    "event",
    "event_id",
    "class",
    "start_frame",
    "end_frame",
    "start_s",
    "end_s",
    "planes",
    "iou",
)


def rank_videos(per_event: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average the event IoUs of each video and rank the videos.

    :param per_event: evaluator rows with ``event_id`` (``<dataset>/<video>/eventNNN``),
        ``class``, ``iou`` and ``planes``.
    :return: one row per video, by decreasing IoU (ties: more planes first, then name), with
        competition ranks.
    """
    videos: dict[str, dict[str, Any]] = {}
    for event in per_event:
        dataset, *video, _ = str(event["event_id"]).split("/")
        key = "/".join(video)
        row = videos.setdefault(
            key, {"dataset": dataset, "video_id": key, "classes": [], "ious": [], "planes": 0}
        )
        if event["class"] not in row["classes"]:
            row["classes"].append(event["class"])
        row["ious"].append(float(event["iou"]))
        row["planes"] += int(event["planes"])
    rows = [
        {
            "dataset": r["dataset"],
            "video_id": r["video_id"],
            "classes": ", ".join(r["classes"]),
            "events": len(r["ious"]),
            "planes": r["planes"],
            "iou": sum(r["ious"]) / len(r["ious"]),
            "best_event_iou": max(r["ious"]),
        }
        for r in videos.values()
    ]
    rows.sort(key=lambda r: (-r["iou"], -r["planes"], r["video_id"]))
    _competition_ranks(rows)
    return rows


def rank_events(
    per_event: Sequence[dict[str, Any]], videos: dict[tuple[str, str], dict[str, Any]]
) -> list[dict[str, Any]]:
    """Rank the single annotated events by their IoU, with where they lie in the video.

    :param per_event: evaluator rows (``event_id`` is ``<dataset>/<video>/eventNNN``, the
        index of the event in the annotation project).
    :param videos: project videos by ``(dataset, video_id)``, with ``fps`` and ``events``
        (``start``, ``end`` inclusive frames, ``class_name``).
    :return: one row per event by decreasing IoU (ties: more planes first, then id), with
        competition ranks, the 1-based event number and its interval in frames and seconds.
    :raises ValueError: if an event is missing from the project or has another class.
    """
    rows = []
    for event in per_event:
        dataset, *video, name = str(event["event_id"]).split("/")
        key = "/".join(video)
        project = videos.get((dataset, key))
        position = int(name.removeprefix("event"))
        if project is None or position >= len(project["events"]):
            raise ValueError(f"{event['event_id']} is not in the annotation project")
        annotated = project["events"][position]
        if annotated["class_name"] != event["class"]:
            raise ValueError(f"{event['event_id']}: the project names another class")
        fps = float(project["fps"])
        rows.append(
            {
                "dataset": dataset,
                "video_id": key,
                "event": position + 1,
                "event_id": event["event_id"],
                "class": event["class"],
                "start_frame": int(annotated["start"]),
                "end_frame": int(annotated["end"]),
                "start_s": round(annotated["start"] / fps, 2),
                "end_s": round((annotated["end"] + 1) / fps, 2),
                "planes": int(event["planes"]),
                "iou": float(event["iou"]),
            }
        )
    rows.sort(key=lambda r: (-r["iou"], -r["planes"], r["event_id"]))
    _competition_ranks(rows)
    return rows


def _competition_ranks(rows: list[dict[str, Any]]) -> None:
    """Write ``rank`` in place: ties share the best position."""
    for row in rows:
        row["rank"] = 1 + sum(other["iou"] > row["iou"] for other in rows)


def ranking_for_run(run: Path, method: str | None = None) -> dict[str, Any]:
    """Rank videos and events of both datasets for one run.

    :param run: run folder with a localization export, ground truth included.
    :param method: experiment to rank, the demonstrator default if ``None``.
    :return: run, method, model label and the ranked rows per dataset, for videos and events.
    :raises ValueError: if the method is unknown, its result was computed on other
        predictions or another ground truth than the export, or the export has no ground
        truth.
    """
    index = read_json(run / FOLDER / "index.json")
    method = method or index["default"]
    entry = next((m for m in index["methods"] if m["id"] == method), None)
    if entry is None:
        raise ValueError(f"{method} is not among the methods exported for {run.name}")
    truth = index.get("ground_truth")
    if truth is None:
        raise ValueError(
            f"{run.name} has no ground truth in its export: run export_demo_localization.py "
            "with --ground-truth first"
        )
    project = read_json(Path(truth["snapshot"]) / "project.json")
    videos = {(v["dataset"], v["video_id"]): v for v in project["videos"]}
    sources = method_sources(index, method)
    results = Path(sources["report"]) / "results"
    rankings, events = {}, {}
    for folder, dataset in DATASETS.items():
        result = read_json(results / folder / f"{method}.json")
        sealed = sources["seal_sha256"][f"{entry['stage']}/{folder}"]
        if result["prediction_seal_sha256"] != sealed:
            raise ValueError(
                f"{method} on {folder} was evaluated on other predictions than the exported "
                "boxes: re-export or re-evaluate before ranking"
            )
        if result["gt_snapshot_sha256"] != truth["snapshot_sha256"]:
            raise ValueError(f"{method} on {folder} was evaluated on another ground truth")
        rankings[dataset] = rank_videos(result["per_event"])
        events[dataset] = rank_events(result["per_event"], videos)
    return {
        "run_id": run.name,
        "label": model_label(run.name),
        "method": method,
        "rankings": rankings,
        "events": events,
        "results": str(results),
    }


def write_csv(rows_by_dataset: dict[str, list[dict[str, Any]]], fields, path: Path) -> None:
    """Write one run's ranking, both datasets, with the given columns."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rows in rows_by_dataset.values():
            for row in rows:
                writer.writerow({k: row[k] for k in fields})


#: Report columns per level: header, then the cell of a row; ``key`` identifies a row
#: across runs.
LEVELS: dict[str, dict[str, Any]] = {
    "rankings": {
        "title": "videos",
        "intro": "Video IoU = mean IoU of its annotated events (plane IoU → event)",
        "key": lambda r: r["video_id"],
        "columns": (
            ("Video", lambda r: r["video_id"]),
            ("Class", lambda r: r["classes"]),
            ("Events", lambda r: r["events"]),
            ("Planes", lambda r: r["planes"]),
        ),
    },
    "events": {
        "title": "events",
        "intro": "Event IoU = mean over its annotated planes of the frame IoU; events are "
        "numbered as in the video (1 = first annotated), with their annotated interval",
        "key": lambda r: r["event_id"],
        "columns": (
            ("Video", lambda r: r["video_id"]),
            ("Event", lambda r: r["event"]),
            ("Class", lambda r: r["class"]),
            ("Interval", lambda r: f"{r['start_s']:.1f}–{r['end_s']:.1f} s"),
            ("Planes", lambda r: r["planes"]),
        ),
    },
}


def markdown(rankings: Sequence[dict[str, Any]], level: str = "rankings") -> str:
    """:return: a report with one table per run and dataset, the other runs alongside."""
    spec = LEVELS[level]
    key: Callable[[dict[str, Any]], str] = spec["key"]
    lines = [
        f"# Localization: ranking of the {spec['title']}",
        "",
        f"{spec['intro']}, from the campaign evaluator; method = the demonstrator default. "
        "The method is chosen on the test set, so the ranking is exploratory. Few planes = "
        "an unstable number.",
        "",
    ]
    for r in rankings:
        lines.append(
            f"- **{r['label']}** (`{r['run_id']}`): {r['method']}; results in `{r['results']}`"
        )
    for r in rankings:
        others = [o for o in rankings if o is not r]
        for dataset, rows in r[level].items():
            lookup = [{key(row): row for row in o[level][dataset]} for o in others]
            names = [name for name, _ in spec["columns"]]
            lines += [
                "",
                f"## {r['label']} · {r['method']} · {DATASET_LABELS[dataset]} "
                f"({len(rows)} {spec['title']})",
                "",
                "| # | "
                + " | ".join(names)
                + " | IoU |"
                + "".join(f" {o['label']} |" for o in others),
                "|---:|" + "---|" * len(names) + "---:|" + "---:|" * len(others),
            ]
            for row in rows:
                cells = " | ".join(str(cell(row)) for _, cell in spec["columns"])
                extra = "".join(
                    f" {table[key(row)]['iou']:.3f} (#{table[key(row)]['rank']}) |"
                    if key(row) in table
                    else " — |"
                    for table in lookup
                )
                lines.append(f"| {row['rank']} | {cells} | {row['iou']:.3f} |" + extra)
    return "\n".join(lines) + "\n"


def main() -> None:
    """Rank every requested run and write the outputs."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--method", help="experiment to rank (default: the demonstrator default)")
    parser.add_argument("--markdown", type=Path, help="optional video report comparing the runs")
    parser.add_argument("--markdown-events", type=Path, help="optional event report")
    args = parser.parse_args()
    rankings = []
    for run in args.run:
        ranking = ranking_for_run(run, args.method)
        method = ranking["method"]
        videos = run / FOLDER / f"video_ranking__{method}.csv"
        events = run / FOLDER / f"event_ranking__{method}.csv"
        write_csv(ranking["rankings"], CSV_FIELDS, videos)
        write_csv(ranking["events"], EVENT_FIELDS, events)
        rankings.append(ranking)
        print(f"{ranking['run_id']}: {videos}, {events}")
    for path, level in ((args.markdown, "rankings"), (args.markdown_events, "events")):
        if path:
            path.write_text(markdown(rankings, level), encoding="utf-8")
            print(f"report: {path}")


if __name__ == "__main__":
    main()
