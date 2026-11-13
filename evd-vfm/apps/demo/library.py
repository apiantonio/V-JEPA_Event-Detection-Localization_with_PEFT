"""Per-video and per-event numbers of a run, for sorting the demonstrator library.

Everything comes from what a run already wrote (``scores.jsonl``), the annotation deliveries and
the localization export; no model is involved (ADR-015). The metrics are those of the page of a
video, from the same function (:func:`evd.eval.demo_metrics.metriche_video`), so a value next to a
card is the value the page shows once the video is open.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from apps.demo.localization import default_method_events, match_event
from evd.annotations.schema import TemporalAnnotation
from evd.annotations.ucf_classes import ucf_classes
from evd.annotations.ucf_lab import UcfLabAnnotations, delivery_of
from evd.eval.demo_metrics import event_rows, metriche_video
from evd.utils.io import read_jsonl


def read_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    """:return: the ``scores.jsonl`` rows of a run by video, ordered as the windows close."""
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        rows[str(row["video_id"])].append(row)
    for items in rows.values():
        items.sort(key=lambda r: (r["end_frame"], r["start_frame"]))
    return dict(rows)


def load_annotations(
    folders: Iterable[Path], num_frames: Mapping[str, int]
) -> dict[str, TemporalAnnotation]:
    """Read the annotation deliveries once, for all the videos of a run.

    :param folders: delivery folders; the first naming a video wins, as in
        :func:`evd.exp.demo.annotation`.
    :param num_frames: frames per video, to clamp the intervals.
    :return: ``video_id`` to annotation.
    """
    found: dict[str, TemporalAnnotation] = {}
    for folder in folders:
        try:
            delivery = delivery_of(folder)
        except FileNotFoundError:
            continue
        source = UcfLabAnnotations.from_dir(folder, delivery=delivery, num_frames=num_frames)
        for video_id, annotation in source.load().items():
            found.setdefault(video_id, annotation)
    return found


def video_entry(
    video_id: str,
    rows: Sequence[Mapping[str, Any]],
    threshold: float,
    annotation: TemporalAnnotation | None,
    fps: float | None,
    placed: Sequence[Mapping[str, Any]] = (),
    threshold_source: str = "val_youden",
) -> dict[str, Any]:
    """Compute what the library sorts a video and its events by.

    :param video_id: the video.
    :param rows: its ``scores.jsonl`` rows.
    :param threshold: the operating threshold of the source run.
    :param annotation: its annotation, ``None`` if no delivery names it.
    :param fps: native frames per second, ``None`` if unknown.
    :param placed: the evaluated events of the video, from
        :func:`apps.demo.localization.default_method_events`.
    :param threshold_source: how the threshold was estimated.
    :return: ``metrics``, flat and with ``None`` where a value is not defined, and ``events``.
    """
    scores = [
        SimpleNamespace(
            anomaly=float(r["score"]),
            window=SimpleNamespace(
                frame_span=(int(r["start_frame"]), int(r["end_frame"])), window_id=r["window_id"]
            ),
        )
        for r in rows
    ]
    meta = SimpleNamespace(video_id=video_id, fps=fps)
    # Without fps the latency has no unit: the rest of the metrics does not need it.
    known = annotation if fps else None
    measured = metriche_video(
        scores,
        meta,
        threshold,
        annotation=known,
        labels={r["window_id"]: int(r["label"]) for r in rows if r.get("label") in (0, 1)},
        threshold_source=threshold_source,
    )
    chunk = measured["chunk"] or {}
    events = measured["events"] or {}
    latency = measured["latency_grid"] or {}
    metrics = {
        "auc": chunk.get("auc"),
        "ap": chunk.get("ap"),
        "f1": chunk.get("f1"),
        "precision": chunk.get("precision"),
        "recall": chunk.get("recall"),
        "accuracy": chunk.get("accuracy"),
        "fpr": chunk.get("false_positive_rate"),
        "tp": chunk.get("tp"),
        "fp": chunk.get("fp"),
        "fn": chunk.get("fn"),
        "tn": chunk.get("tn"),
        "above_threshold": measured["above_threshold"],
        "peak_score": max((s.anomaly for s in scores), default=None),
        "event_recall": events.get("recall"),
        "event_precision": events.get("precision"),
        "event_f1": events.get("f1"),
        "events": events.get("n_eventi"),
        "events_missed": (
            events["n_eventi"] - events["n_eventi_rilevati"] if "n_eventi" in events else None
        ),
        "alarms": events.get("n_allarmi"),
        "false_alarms": (
            events["n_allarmi"] - events["n_allarmi_veri"] if "n_allarmi" in events else None
        ),
        "latency_s": latency.get("mediana_s"),
    }
    names = {int(c.idx): str(c.name) for c in ucf_classes()}
    rows_of_events = []
    for event in event_rows(scores, annotation, threshold, fps) if known else ():
        label = names.get(event["class_idx"], f"class {event['class_idx']}")
        match = match_event(event["start"], event["end"], label, placed)
        rows_of_events.append(
            {
                **event,
                "class": label,
                "loc_iou": None if match is None else match["iou"],
                "loc_ap": None if match is None else match["ap"],
            }
        )
    for name, key in (("loc_iou", "iou"), ("loc_ap", "ap")):
        values = [e[key] for e in placed if e.get(key) is not None]
        metrics[name] = sum(values) / len(values) if values else None
    return {"metrics": metrics, "events": rows_of_events}


def run_entries(
    run: Any,
    annotations: Iterable[Path],
    fps_of: Any,
    wanted: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """Compute the library entries of one source run (the run, or an external evaluation).

    :param run: the opened run.
    :param annotations: annotation delivery folders.
    :param fps_of: ``video_id -> fps or None``.
    :param wanted: the videos to compute, the others are skipped.
    :return: ``video_id`` to :func:`video_entry`.
    """
    rows = read_rows(run.score_path)
    keep = [v for v in wanted if v in rows]
    # The grid covers each video to its last frame (ADR-047), so no decoding is needed.
    frames = {v: max(int(r["end_frame"]) for r in rows[v]) + 1 for v in keep}
    truth = load_annotations(annotations, frames)
    placed = default_method_events(run.dir, run.metrics)
    return {
        v: video_entry(
            v,
            rows[v],
            run.threshold,
            truth.get(v),
            fps_of(v),
            placed.get(v, ()),
            run.threshold_source,
        )
        for v in keep
    }
