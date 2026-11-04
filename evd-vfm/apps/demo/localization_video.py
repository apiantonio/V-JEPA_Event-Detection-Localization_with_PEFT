"""Localization movies: the demonstrator layout around the saved boxes of one method.

What the demonstrator video shows (``overlay.py``: the frame, a reading panel, the whole
timeline below), plus what a localization method adds: a header naming the model, the
localization method and the video; on the frame the method boxes (solid, numbered) and the
spatial ground truth the IoU was computed on (dashed); in the panel the verdict of the chunk
the boxes come from, the annotated events with their evaluated IoU, and the frame IoU. The
strip is the one of the explanation movies (score curve, chunks coloured TP/FP/FN/TN,
annotation), so the two collections read the same way.

Everything comes from one :class:`~apps.demo.payload.DemoPayload` with its localization
block: no model is involved (ADR-015). Boxes exist only inside the annotated intervals,
where :func:`apps.demo.localization.for_video` leaves them, and the chunk described for a
frame is the one its boxes were projected from (``fine``, as in the evaluator).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np

from apps.demo.explanation import (
    BOX_PALETTE,
    COLORS,
    STRIP_H,
    VERDICT_COLORS,
    Timeline,
    _ascii,
    _truncate,
    _VideoWriter,
    chunk_verdict,
    strip,
    strip_background,
)
from apps.demo.overlay import VIDEO_H, VIDEO_W
from apps.demo.payload import DemoPayload
from evd.eval.localization import union_box_iou
from evd.explain.project import offline_assignment
from evd.types import WindowSpec

#: Canvas geometry: header band, then the frame (left) and the panel (right), then the strip.
CANVAS_W = 1280
HEADER_H = 76
CANVAS_H = HEADER_H + VIDEO_H + STRIP_H

#: Rows of the panel lists; what does not fit is counted, not dropped silently.
MAX_EVENTS, MAX_DETECTIONS = 5, 3

_FONT = cv2.FONT_HERSHEY_DUPLEX
_FONT_MONO = cv2.FONT_HERSHEY_SIMPLEX

#: x264 threads per movie: a worker encodes one movie at a time, and its memory, not the
#: cores, limits how many workers a machine runs.
ENCODER_THREADS = 2

#: Characters of the readable method names that the Hershey fonts cannot draw.
_ARROWS = {"→": "->", "←": "<-"}


@dataclass(frozen=True)
class EventRow:
    """An annotated event as the panel lists it.

    :param start: first frame, inclusive.
    :param end: last frame, inclusive.
    :param name: annotated class.
    :param iou: the evaluator IoU of the method on the event, ``None`` if not evaluated
        (no reviewed spatial annotation for it).
    """

    start: int
    end: int
    name: str
    iou: float | None


@dataclass(frozen=True)
class Movie:
    """Everything a localization movie shows, decided once per video.

    :param payload: the demonstrator data of the video, localization block included.
    :param method: exported method id whose boxes are drawn.
    :param method_label: the readable name of the method, shown in the header.
    :param model_label: the readable name of the detector.
    :param dataset_label: dataset and partition of the video.
    :param line: the strip timeline, with the ``fine`` assignment of frames to chunks.
    :param method_ref: per frame, the index of its method box list in the payload table,
        ``-1`` where the method draws nothing.
    :param truth_ref: per frame, the index of its ground-truth box list, ``-1`` if none.
    :param events: the annotated events, in annotation order.
    :param video_iou: mean IoU of the evaluated events of the video, ``None`` if none.
    :param dataset_mean: mean IoU of the method on the video's dataset (mean of classes).
    """

    payload: DemoPayload
    method: str
    method_label: str
    model_label: str
    dataset_label: str
    line: Timeline
    method_ref: np.ndarray
    truth_ref: np.ndarray
    events: tuple[EventRow, ...]
    video_iou: float | None
    dataset_mean: float | None

    @property
    def method_table(self) -> list:
        """:return: the box-list table the method segments point into."""
        return self.payload.localization["boxes"]

    @property
    def truth_table(self) -> list:
        """:return: the ground-truth box-list table, empty without a ground truth."""
        truth = self.payload.localization.get("truth")
        return truth["boxes"] if truth else []


def frame_refs(segments: Sequence[Sequence[int]], num_frames: int) -> np.ndarray:
    """Expand inclusive ``[start, end, index]`` segments into one index per frame.

    :param segments: segments in frame order, possibly empty.
    :param num_frames: native frames of the video.
    :return: ``[num_frames]`` int, ``-1`` on frames no segment covers.
    :raises ValueError: if a segment lies outside the video.
    """
    refs = np.full(num_frames, -1, dtype=np.int64)
    for start, end, index in segments:
        if not 0 <= start <= end < num_frames:
            raise ValueError(f"Segment {start}..{end} outside a video of {num_frames} frames")
        refs[start : end + 1] = index
    return refs


def build_movie(
    payload: DemoPayload,
    windows: Sequence[WindowSpec],
    *,
    method: str,
    method_label: str,
    model_label: str,
    dataset_label: str,
    labels: Sequence[int] | None = None,
) -> Movie:
    """Decide what the movie of one method on one video shows.

    :param payload: the demonstrator data, with the run localization block for the video.
    :param windows: the windows of ``payload.finestre``, in the same order.
    :param method: exported method id to draw.
    :param method_label: its readable name.
    :param model_label: the readable name of the detector.
    :param dataset_label: dataset and partition of the video.
    :param labels: the protocol labels of the windows as the run saved them, to check the
        ones derived here from the annotation; ``None`` skips the check.
    :return: the movie description.
    :raises ValueError: without localization boxes for the method, with windows that are not
        those of the payload, or with labels that disagree with the annotation.
    """
    block = payload.localization
    if block is None or method not in block["segments"]:
        raise ValueError(f"{payload.video_id}: no saved boxes of {method} for this run")
    spans = tuple(tuple(int(v) for v in w.frame_span) for w in windows)
    if spans != tuple((int(a), int(b)) for a, b, _ in payload.finestre):
        raise ValueError(f"{payload.video_id}: the windows are not those of the payload")
    n = payload.num_frames
    scores = np.asarray([s for _, _, s in payload.finestre], dtype=float)
    assignment = offline_assignment(list(windows), n, politica="fine")
    intervals = tuple((seg.start, seg.end) for seg in payload.annotati)
    # The protocol rule: a window is positive if it intersects an annotated interval.
    positive = (
        tuple(int(any(a <= e and s <= b for s, e in intervals)) for a, b in spans)
        if payload.ha_verita
        else tuple(None for _ in spans)
    )
    if labels is not None and tuple(labels) != positive:
        raise ValueError(f"{payload.video_id}: window labels disagree with the run's")
    projected = np.where(assignment >= 0, scores[np.clip(assignment, 0, None)], 0.0)
    line = Timeline(
        projected.astype(np.float32),
        assignment,
        spans,
        tuple(float(s) for s in scores),
        payload.threshold,
        positive,
        intervals if payload.ha_verita else None,
        tuple(seg.label or "" for seg in payload.annotati),
    )
    truth = block.get("truth")
    entry = next(m for m in block["methods"] if m["id"] == method)
    own = entry.get(block["dataset"], {})
    evaluated = own.get("events") or {}
    events = _event_rows(payload, truth["events"] if truth else [], evaluated)
    known = [e.iou for e in events if e.iou is not None]
    return Movie(
        payload=payload,
        method=method,
        method_label=method_label,
        model_label=model_label,
        dataset_label=dataset_label,
        line=line,
        method_ref=frame_refs(block["segments"][method], n),
        truth_ref=frame_refs(truth["segments"], n) if truth else np.full(n, -1, dtype=np.int64),
        events=events,
        video_iou=sum(known) / len(known) if known else None,
        dataset_mean=own.get("mean"),
    )


def _event_rows(
    payload: DemoPayload, placed: Sequence[dict[str, Any]], evaluated: dict[str, float]
) -> tuple[EventRow, ...]:
    """Attach to each annotated interval the IoU of the evaluated event lying on it.

    An evaluated event is placed on its annotation-project interval; it matches the
    annotated interval of the same class that overlaps it most.
    """
    rows = []
    for seg in payload.annotati:
        best, overlap = None, 0
        for event in placed:
            if event["class"] != seg.label or event["id"] not in evaluated:
                continue
            common = min(seg.end, event["end"]) - max(seg.start, event["start"]) + 1
            if common > overlap:
                best, overlap = evaluated[event["id"]], common
        rows.append(EventRow(seg.start, seg.end, seg.label or "event", best))
    return tuple(rows)


def _text(canvas: np.ndarray, text: str, org, scale: float, color, font=_FONT_MONO, width=None):
    """Write ASCII-only text (the fonts draw nothing else), shortened to ``width`` pixels."""
    for origin, destination in _ARROWS.items():
        text = text.replace(origin, destination)
    text = _truncate(text, width, font, scale) if width else _ascii(text)
    cv2.putText(canvas, text, org, font, scale, color, 1, cv2.LINE_AA)


def _clock(frame: int, fps: float) -> str:
    """:return: ``mm:ss`` of a frame, ``--:--`` without fps."""
    if not fps:
        return "--:--"
    seconds = int(frame / fps)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def header_band(movie: Movie) -> np.ndarray:
    """Return the top band: the localization method, the model and the video.

    :return: BGR ``[HEADER_H, CANVAS_W, 3]``, the same for every frame.
    """
    band = np.empty((HEADER_H, CANVAS_W, 3), dtype=np.uint8)
    band[:] = COLORS["pannello"]
    cv2.line(band, (0, HEADER_H - 1), (CANVAS_W, HEADER_H - 1), COLORS["linea"], 1)
    width = CANVAS_W - 36
    _text(band, "LOCALIZATION", (18, 22), 0.36, COLORS["calma"])
    _text(band, movie.method_label, (130, 24), 0.62, COLORS["testo"], _FONT, width - 112)
    _text(
        band,
        f"Model: {movie.model_label}  ({movie.payload.run_id})",
        (18, 47),
        0.42,
        COLORS["tenue"],
        width=width,
    )
    classes = ", ".join(dict.fromkeys(e.name for e in movie.events)) or "none"
    iou = []
    if movie.dataset_mean is not None:
        iou.append(f"method IoU {movie.dataset_mean:.3f} on the dataset")
    if movie.video_iou is not None:
        iou.append(f"{movie.video_iou:.3f} on this video")
    _text(
        band,
        f"Video: {movie.payload.video_id}  -  {movie.dataset_label}  -  annotated: {classes}"
        + (f"  -  {', '.join(iou)}" if iou else ""),
        (18, 68),
        0.42,
        COLORS["verita"],
        width=width,
    )
    return band


def _scaled_rect(box, scale: float, ox: int, oy: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """:return: the corners of an inclusive native box on the scaled frame."""
    x0, y0, x1, y1 = box
    return (
        (ox + int(round(x0 * scale)), oy + int(round(y0 * scale))),
        (ox + int(round((x1 + 1) * scale)) - 1, oy + int(round((y1 + 1) * scale)) - 1),
    )


def _dashed_rect(canvas: np.ndarray, p0, p1, color, thickness: int = 2, dash: int = 8) -> None:
    """Draw a dashed rectangle, the demonstrator convention for the ground truth."""
    (x0, y0), (x1, y1) = p0, p1
    for a, b, fixed, horizontal in ((x0, x1, y0, True), (x0, x1, y1, True)) + (
        (y0, y1, x0, False),
        (y0, y1, x1, False),
    ):
        for start in range(a, b + 1, 2 * dash):
            end = min(start + dash, b)
            pts = ((start, fixed), (end, fixed)) if horizontal else ((fixed, start), (fixed, end))
            cv2.line(canvas, pts[0], pts[1], color, thickness, cv2.LINE_AA)


def _draw_frame(canvas: np.ndarray, frame: np.ndarray, movie: Movie, index: int, alarm: bool):
    """Draw the frame, scaled into its box, with ground truth and method boxes."""
    h, w = frame.shape[:2]
    scale = min(VIDEO_W / w, VIDEO_H / h)
    nw, nh = int(w * scale), int(h * scale)
    ox, oy = (VIDEO_W - nw) // 2, HEADER_H + (VIDEO_H - nh) // 2
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    canvas[oy : oy + nh, ox : ox + nw] = cv2.resize(frame, (nw, nh), interpolation=interpolation)
    truth = int(movie.truth_ref[index])
    for box in movie.truth_table[truth] if truth >= 0 else []:
        p0, p1 = _scaled_rect(box, scale, ox, oy)
        _dashed_rect(canvas, p0, p1, COLORS["verita"])
        _text(canvas, "GT", (p0[0] + 2, max(p0[1] - 4, oy + 12)), 0.42, COLORS["verita"])
    ref = int(movie.method_ref[index])
    for number, box in enumerate(movie.method_table[ref] if ref >= 0 else []):
        color = BOX_PALETTE[number % len(BOX_PALETTE)]
        p0, p1 = _scaled_rect(box, scale, ox, oy)
        cv2.rectangle(canvas, p0, p1, color, 2)
        label = str(number + 1)
        y_text = p0[1] - 5 if p0[1] - oy >= 16 else min(p1[1] - 4, oy + nh - 4)
        _text(canvas, label, (max(p0[0] + 2, ox), y_text), 0.5, color)
    color = COLORS["allarme"] if alarm else COLORS["linea"]
    cv2.rectangle(canvas, (ox, oy), (ox + nw - 1, oy + nh - 1), color, 3 if alarm else 1)


def _bar(canvas, x: int, y: int, width: int, value: float, threshold: float, color) -> None:
    """Draw the score gauge with the threshold mark, as the demonstrator panel does."""
    cv2.rectangle(canvas, (x, y), (x + width, y + 8), COLORS["linea"], -1)
    filled = int(width * min(max(value, 0.0), 1.0))
    if filled > 0:
        cv2.rectangle(canvas, (x, y), (x + filled, y + 8), color, -1)
    xs = x + int(width * min(max(threshold, 0.0), 1.0))
    cv2.line(canvas, (xs, y - 4), (xs, y + 12), COLORS["testo"], 1)


def _draw_panel(canvas: np.ndarray, movie: Movie, index: int, alarm: bool) -> None:
    """Draw the reading column.

    The top is the demonstrator panel: the score the online system has on this frame
    (causal projection) and its alarm state. Below, the chunk the boxes come from (the one
    containing the frame and closing first, as the evaluator projects boxes), with its
    verdict; the two can differ, because an online alarm starts when a window closes and
    lasts with the hysteresis.
    """
    data = movie.payload
    x0, top = VIDEO_W, HEADER_H
    width = CANVAS_W - x0 - 44
    cv2.rectangle(canvas, (x0, top), (CANVAS_W, top + VIDEO_H), COLORS["pannello"], -1)
    cv2.line(canvas, (x0, top), (x0, top + VIDEO_H), COLORS["linea"], 1)
    px = x0 + 22
    covered = bool(data.covered[index])
    score = float(data.frame_scores[index]) if covered else 0.0
    accent = COLORS["allarme"] if alarm else COLORS["calma"]
    y = top + 28
    _text(canvas, "ANOMALY SCORE (online)", (px, y), 0.38, COLORS["tenue"])
    shown = f"{score:.3f}" if covered else "--.--"
    _text(canvas, shown, (px, y + 46), 1.3, accent if covered else COLORS["tenue"], _FONT)
    _bar(canvas, px, y + 60, width, score, data.threshold, accent)
    y += 82
    state = "EVENT DETECTED" if alarm else ("MONITORING" if covered else "WARM-UP")
    fill = COLORS["allarme"] if alarm else COLORS["linea"]
    cv2.rectangle(canvas, (px, y), (px + width, y + 30), fill, -1)
    _text(canvas, state, (px + 12, y + 21), 0.50, COLORS["fondo"] if alarm else COLORS["testo"])
    y += 52
    _text(
        canvas,
        f"t {_clock(index, data.fps)} / {_clock(data.num_frames, data.fps)}",
        (px, y),
        0.42,
        COLORS["testo"],
    )
    active = [e for e in movie.events if e.start <= index <= e.end]
    annotated = (
        "annotated: " + " + ".join(dict.fromkeys(e.name for e in active))
        if active
        else ("annotated: none" if data.ha_verita else "annotated: n/a")
    )
    _text(canvas, annotated, (px, y + 24), 0.42, COLORS["verita"], width=width)
    y += 42
    y = _draw_chunk(canvas, movie, index, px, y, width)
    y += 14
    cv2.line(canvas, (px, y), (px + width, y), COLORS["linea"], 1)
    y += 22
    _text(canvas, f"ANNOTATED EVENTS ({len(movie.events)})", (px, y), 0.38, COLORS["tenue"])
    _text(canvas, "IoU", (px + width - 44, y), 0.38, COLORS["tenue"])
    for event in movie.events[:MAX_EVENTS]:
        y += 24
        on = event.start <= index <= event.end
        if on:
            cv2.rectangle(canvas, (px - 8, y - 16), (px + width, y + 7), COLORS["linea"], -1)
        cv2.circle(canvas, (px + 4, y - 4), 3, COLORS["verita"], -1 if on else 1)
        when = f"{_clock(event.start, data.fps)}-{_clock(event.end + 1, data.fps)}"
        iou = "n/e" if event.iou is None else f"{event.iou:.3f}"
        tone = COLORS["testo"] if on else COLORS["tenue"]
        _text(canvas, f"{event.name}  {when}", (px + 16, y), 0.38, tone, width=width - 70)
        _text(canvas, iou, (px + width - 44, y), 0.38, tone)
    if len(movie.events) > MAX_EVENTS:
        y += 22
        more = f"+{len(movie.events) - MAX_EVENTS} more"
        _text(canvas, more, (px + 16, y), 0.36, COLORS["tenue"])
    y += 26
    cv2.line(canvas, (px, y), (px + width, y), COLORS["linea"], 1)
    y += 22
    _text(canvas, f"DETECTIONS ({len(data.predetti)})", (px, y), 0.38, COLORS["tenue"])
    for seg in data.predetti[:MAX_DETECTIONS]:
        y += 24
        on = seg.start <= index <= seg.end
        cv2.circle(canvas, (px + 4, y - 4), 3, COLORS["allarme"], -1 if on else 1)
        peak = "" if seg.peak is None else f"  peak {seg.peak:.2f}"
        when = f"{_clock(seg.start, data.fps)}-{_clock(seg.end + 1, data.fps)}{peak}"
        _text(canvas, when, (px + 16, y), 0.38, COLORS["testo"] if on else COLORS["tenue"])
    if len(data.predetti) > MAX_DETECTIONS:
        y += 22
        more = f"+{len(data.predetti) - MAX_DETECTIONS} more"
        _text(canvas, more, (px + 16, y), 0.36, COLORS["tenue"])
    _draw_boxes_summary(canvas, movie, index, px, width)


def _draw_chunk(canvas: np.ndarray, movie: Movie, index: int, px: int, y: int, width: int) -> int:
    """Draw the chunk the boxes of this frame come from, with its score and verdict.

    :return: the row below the block.
    """
    line, data = movie.line, movie.payload
    cv2.line(canvas, (px, y), (px + width, y), COLORS["linea"], 1)
    y += 22
    _text(canvas, "CHUNK OF THE BOXES", (px, y), 0.38, COLORS["tenue"])
    window = int(line.assegnazioni[index])
    if window < 0:
        _text(canvas, "no chunk contains this frame", (px, y + 26), 0.40, COLORS["tenue"])
        return y + 38
    score = line.score[window]
    code = chunk_verdict(score >= data.threshold, line.label_of(window))
    tone = COLORS[VERDICT_COLORS[code]] if code and code != "TN" else COLORS["tenue"]
    if code:
        cv2.rectangle(canvas, (px + width - 56, y - 16), (px + width, y + 14), tone, -1)
        _text(canvas, code, (px + width - 46, y + 7), 0.62, COLORS["fondo"], _FONT)
    a, b = line.spans[window]
    _text(
        canvas,
        f"{window + 1}/{len(line.spans)}  {_clock(a, data.fps)}-{_clock(b + 1, data.fps)}"
        f"  score {score:.3f}",
        (px, y + 26),
        0.42,
        COLORS["testo"],
        width=width,
    )
    return y + 38


def _draw_boxes_summary(canvas: np.ndarray, movie: Movie, index: int, px: int, width: int):
    """Draw the bottom rows of the panel: boxes on the frame, frame IoU, operating point."""
    data = movie.payload
    y = HEADER_H + VIDEO_H - 128
    cv2.line(canvas, (px, y), (px + width, y), COLORS["linea"], 1)
    y += 22
    _text(canvas, "BOXES ON THIS FRAME", (px, y), 0.38, COLORS["tenue"])
    ref, truth = int(movie.method_ref[index]), int(movie.truth_ref[index])
    method = movie.method_table[ref] if ref >= 0 else None
    gt = movie.truth_table[truth] if truth >= 0 else None
    inside = any(e.start <= index <= e.end for e in movie.events)
    if not inside:
        _text(canvas, "outside annotated intervals: no box", (px, y + 24), 0.38, COLORS["tenue"])
    else:
        count = "not computed" if method is None else f"{len(method)} (solid)"
        _text(canvas, f"method: {count}", (px, y + 24), 0.40, COLORS["testo"])
        reviewed = "no reviewed box" if gt is None else f"{len(gt)} (dashed)"
        _text(canvas, f"ground truth: {reviewed}", (px, y + 46), 0.40, COLORS["verita"])
        if gt:
            iou = union_box_iou(method or [], gt)
            _text(canvas, f"frame IoU {iou:.3f}", (px, y + 68), 0.40, COLORS["testo"])
    metric = data.metriche.get("chunk_auc")
    footer = f"tau {data.threshold:.4f} ({data.threshold_source})"
    if isinstance(metric, float):
        footer += f"  chunk AUC {metric:.4f}"
    _text(canvas, footer, (px, HEADER_H + VIDEO_H - 12), 0.34, COLORS["tenue"], width=width)


def _panel_key(movie: Movie, index: int, alarm: bool) -> tuple:
    """:return: everything the panel shows on a frame; equal keys draw equal panels."""
    data = movie.payload
    covered = bool(data.covered[index])
    return (
        _clock(index, data.fps),
        f"{float(data.frame_scores[index]):.3f}" if covered else None,
        alarm,
        int(movie.line.assegnazioni[index]),
        tuple(e.start <= index <= e.end for e in movie.events),
        tuple(s.start <= index <= s.end for s in data.predetti[:MAX_DETECTIONS]),
        int(movie.method_ref[index]),
        int(movie.truth_ref[index]),
    )


def compose(
    movie: Movie,
    index: int,
    frame: np.ndarray,
    header: np.ndarray,
    background: np.ndarray,
    cache: dict | None = None,
) -> np.ndarray:
    """Return the canvas of one native frame.

    :param movie: the movie description.
    :param index: native frame index.
    :param frame: the native BGR frame.
    :param header: :func:`header_band` of the movie.
    :param background: the strip drawn once by ``strip_background``.
    :param cache: the last panel drawn and its key, reused while the key does not change
        (the panel changes at most once a second); ``None`` draws it every time.
    :return: BGR ``[CANVAS_H, CANVAS_W, 3]``.
    """
    canvas = np.empty((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    canvas[:] = COLORS["fondo"]
    canvas[:HEADER_H] = header
    alarm = any(seg.start <= index <= seg.end for seg in movie.payload.predetti)
    _draw_frame(canvas, frame, movie, index, alarm)
    region = (slice(HEADER_H, HEADER_H + VIDEO_H), slice(VIDEO_W, CANVAS_W))
    key = _panel_key(movie, index, alarm)
    if cache is not None and cache.get("key") == key:
        canvas[region] = cache["panel"]
    else:
        _draw_panel(canvas, movie, index, alarm)
        if cache is not None:
            cache.update(key=key, panel=canvas[region].copy())
    canvas[HEADER_H + VIDEO_H :] = strip(background, index, movie.line)
    return canvas


def write(movie: Movie, video_path: str | Path, out_path: str | Path) -> dict[str, int]:
    """Write the H.264 movie of every native frame, stopping at the probed frame count.

    :param movie: the movie description.
    :param video_path: the original video file.
    :param out_path: destination ``.mp4``; parents are created.
    :return: frames written, frames with method boxes and with ground-truth boxes.
    :raises ValueError: if the file yields fewer frames than probed, or another geometry.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = movie.payload
    header = header_band(movie)
    background = strip_background(CANVAS_W, movie.line)
    writer = _VideoWriter(out, data.fps, (CANVAS_W, CANVAS_H))
    writer.stream.thread_count = ENCODER_THREADS
    cache: dict = {}
    count = 0
    try:
        with av.open(str(video_path)) as container:
            container.streams.video[0].thread_type = "AUTO"
            for frame in container.decode(video=0):
                if count >= data.num_frames:
                    break
                body = frame.to_ndarray(format="bgr24")
                if body.shape[:2] != (data.height, data.width):
                    raise ValueError(f"{video_path}: frames are not {data.width}x{data.height}")
                writer.write(compose(movie, count, body, header, background, cache))
                count += 1
    finally:
        writer.release()
    if count != data.num_frames:
        raise ValueError(f"{video_path}: decoded {count} frames, expected {data.num_frames}")
    return {
        "frames": count,
        "frames_with_method_boxes": int(
            sum(1 for r in movie.method_ref if r >= 0 and movie.method_table[r])
        ),
        "frames_with_truth_boxes": int(np.count_nonzero(movie.truth_ref >= 0)),
    }
