"""``annotated.mp4``: the video with the decision overlaid (ADR-015).

The deliverable for a presentation without network, GPU or browser (R7: the session
video is recorded in advance). Layout: the video on the left, a reading panel on the
right (current score, state, alarms), and a strip below with the whole timeline and an
advancing cursor. All numbers come from the same :class:`~apps.demo.payload.DemoPayload`
as the viewer.

The announced state is binary and the class shown is the annotated one: the head emits
a single logit (ADR-031 point 4), so the class appears only next to the ground truth,
labelled as such (a declared deviation from PdD §6, ``docs/01`` §3bis).

Drawn with OpenCV, using its Hershey fonts so no ``.ttf`` has to ship with the video.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from apps.demo.payload import DemoPayload

#: Names used here and the template custom properties (``assets/viewer.html``, ``:root``
#: block) they take their value from: the session video and the page share one theme.
#: A test checks the correspondence by reading the CSS
#: (``test_demo.py::test_la_palette_del_video_e_quella_della_pagina``).
TOKEN = {
    "fondo": "--ink",
    "pannello": "--ink-2",
    "linea": "--rule",
    "testo": "--paper",
    "tenue": "--haze",
    "calma": "--calm",
    "allarme": "--alarm",
    "verita": "--truth",
}

#: The palette, in BGR (OpenCV channel order). The comment on each row is the value of
#: the corresponding token in :data:`TOKEN`.
COLORS = {
    "fondo": (25, 20, 16),  # --ink      #101419
    "pannello": (38, 30, 24),  # --ink-2    #181E26
    "linea": (65, 53, 44),  # --rule     #2C3541
    "testo": (242, 238, 232),  # --paper    #E8EEF2
    "tenue": (193, 179, 165),  # --haze     #A5B3C1
    "calma": (188, 223, 114),  # --calm     #72DFBC
    "allarme": (73, 91, 255),  # --alarm    #FF5B49  above threshold
    "verita": (39, 162, 201),  # --truth    #C9A227  the annotation
}

#: Canvas geometry: the video in ``VIDEO_W x VIDEO_H``, the reading panel in the right
#: column, the timeline in the bottom strip.
CANVAS_W, VIDEO_W, VIDEO_H, STRIP_H = 1280, 900, 720, 160
CANVAS_H = VIDEO_H + STRIP_H

_FONT = cv2.FONT_HERSHEY_DUPLEX
_FONT_MONO = cv2.FONT_HERSHEY_SIMPLEX


def scrivi(
    payload: DemoPayload,
    video_path: str | Path,
    out_path: str | Path,
    *,
    fourcc: str = "mp4v",
) -> Path:
    """Write the annotated video, reading frames from the original file.

    Frames are read sequentially with OpenCV, not through
    :class:`~evd.data.decoder.VideoDecoder`, which produces the pixels the model sees: here
    the video is wanted as it is, at its resolution and with all its frames.

    :param payload: the demonstrator data.
    :param video_path: the original video file.
    :param out_path: where to write the mp4; parents are created.
    :param fourcc: the four-character codec; ``mp4v`` is always available in OpenCV,
        ``avc1`` gives smaller files but is not available everywhere.
    :return: the written path.
    :raises RuntimeError: if the video does not open or the writer cannot be created
        (typically a missing codec).
    """
    p_out = Path(out_path)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {video_path}")
    writer = cv2.VideoWriter(
        str(p_out), cv2.VideoWriter_fourcc(*fourcc), payload.fps or 25.0, (CANVAS_W, CANVAS_H)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"OpenCV cannot create the writer for {p_out} with fourcc {fourcc!r}: the codec is "
            f"not available in this build. Try `--fourcc avc1` or `mp4v`"
        )
    background = _strip_background(payload)
    try:
        for index, frame in _frames(cap, payload.num_frames):
            writer.write(_compose(payload, frame, index, background))
    finally:
        writer.release()
        cap.release()
    return p_out


def _frames(cap: cv2.VideoCapture, expected: int) -> Iterator[tuple[int, np.ndarray]]:
    """Yield the video frames in order, stopping at the inventory count.

    Scores are ``[num_frames]`` from the probe, and containers with an inexact declared
    count can yield one extra frame.
    """
    index = 0
    while index < expected:
        ok, frame = cap.read()
        if not ok:
            return
        yield index, frame
        index += 1


def _compose(
    payload: DemoPayload, frame: np.ndarray, indice: int, background: np.ndarray
) -> np.ndarray:
    """Return one canvas (video, panel, strip) for frame ``indice``."""
    canvas = np.empty((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    canvas[:] = COLORS["fondo"]
    alarm = _in_alarm(payload, indice)

    _draw_video(canvas, frame, alarm)
    _draw_panel(canvas, payload, indice, alarm)
    canvas[VIDEO_H:] = background
    _draw_cursor(canvas, payload, indice)
    return canvas


def _draw_video(canvas: np.ndarray, frame: np.ndarray, alarm: bool) -> None:
    """Draw the resized frame inside its box, with the state frame."""
    h, w = frame.shape[:2]
    scale = min(VIDEO_W / w, VIDEO_H / h)
    nw, nh = int(w * scale), int(h * scale)
    x0, y0 = (VIDEO_W - nw) // 2, (VIDEO_H - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    color = COLORS["allarme"] if alarm else COLORS["linea"]
    cv2.rectangle(canvas, (x0, y0), (x0 + nw - 1, y0 + nh - 1), color, 3 if alarm else 1)


def _draw_panel(canvas: np.ndarray, payload: DemoPayload, indice: int, alarm: bool) -> None:
    """Draw the reading column: identity, current score, state, alarms, metrics."""
    x0 = VIDEO_W
    width = CANVAS_W - x0
    cv2.rectangle(canvas, (x0, 0), (CANVAS_W, VIDEO_H), COLORS["pannello"], -1)
    cv2.line(canvas, (x0, 0), (x0, VIDEO_H), COLORS["linea"], 1)
    px = x0 + 22
    accent = COLORS["allarme"] if alarm else COLORS["calma"]

    cv2.putText(canvas, "EVD", (px, 44), _FONT, 0.82, COLORS["testo"], 1, cv2.LINE_AA)
    cv2.putText(
        canvas, "event detection", (px + 62, 44), _FONT_MONO, 0.40, COLORS["tenue"], 1, cv2.LINE_AA
    )
    _truncated_text(canvas, payload.run_id, (px, 70), width - 44, 0.36, COLORS["tenue"])
    cv2.putText(canvas, f"[{payload.modo}]", (px, 92), _FONT_MONO, 0.38, accent, 1, cv2.LINE_AA)

    # Current score. An uncovered frame has no score: during the causal warm-up the panel
    # says there is no full window to judge yet.
    covered = bool(payload.covered[indice])
    score = float(payload.frame_scores[indice]) if covered else 0.0
    cv2.putText(
        canvas, "ANOMALY SCORE", (px, 148), _FONT_MONO, 0.38, COLORS["tenue"], 1, cv2.LINE_AA
    )
    cv2.putText(
        canvas,
        f"{score:.3f}" if covered else "--.--",
        (px, 208),
        _FONT,
        1.55,
        accent if covered else COLORS["tenue"],
        2,
        cv2.LINE_AA,
    )
    _bar(canvas, px, 226, width - 44, score if covered else 0.0, payload.threshold, accent)

    state = "EVENT DETECTED" if alarm else ("WARM-UP" if not covered else "MONITORING")
    cv2.rectangle(
        canvas, (px, 250), (px + width - 44, 282), accent if alarm else COLORS["linea"], -1
    )
    cv2.putText(
        canvas,
        state,
        (px + 12, 272),
        _FONT_MONO,
        0.50,
        COLORS["fondo"] if alarm else COLORS["testo"],
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        f"t {_timecode(indice, payload.fps)} / {_timecode(payload.num_frames, payload.fps)}",
        (px, 312),
        _FONT_MONO,
        0.42,
        COLORS["testo"],
        1,
        cv2.LINE_AA,
    )
    class_name = _in_event(payload, indice)
    truth = (
        f"annotated: {class_name}"
        if class_name
        else ("annotated: none" if payload.ha_verita else "annotated: n/d")
    )
    cv2.putText(canvas, truth, (px, 336), _FONT_MONO, 0.40, COLORS["verita"], 1, cv2.LINE_AA)

    cv2.line(canvas, (px, 360), (px + width - 44, 360), COLORS["linea"], 1)
    cv2.putText(
        canvas,
        f"DETECTIONS ({len(payload.predetti)})",
        (px, 384),
        _FONT_MONO,
        0.40,
        COLORS["tenue"],
        1,
        cv2.LINE_AA,
    )
    for row, seg in enumerate(payload.predetti[:8]):
        y = 412 + row * 26
        active = seg.start <= indice <= seg.end
        if active:
            cv2.rectangle(canvas, (px - 8, y - 16), (px + width - 44, y + 8), COLORS["linea"], -1)
        cv2.circle(canvas, (px + 4, y - 4), 3, COLORS["allarme"], -1 if active else 1)
        when = f"{_timecode(seg.start, payload.fps)}-{_timecode(seg.end, payload.fps)}"
        peak = "" if seg.peak is None else f"  peak {seg.peak:.2f}"
        cv2.putText(
            canvas,
            when + peak,
            (px + 18, y),
            _FONT_MONO,
            0.38,
            COLORS["testo"] if active else COLORS["tenue"],
            1,
            cv2.LINE_AA,
        )
    if len(payload.predetti) > 8:
        cv2.putText(
            canvas,
            f"+{len(payload.predetti) - 8} more",
            (px + 18, 412 + 8 * 26),
            _FONT_MONO,
            0.36,
            COLORS["tenue"],
            1,
            cv2.LINE_AA,
        )

    _footer(canvas, payload, px, width)


def _footer(canvas: np.ndarray, payload: DemoPayload, px: int, width: int) -> None:
    """Draw the rows at the bottom of the panel: chunk geometry and run metrics."""
    y = VIDEO_H - 118
    cv2.line(canvas, (px, y), (px + width - 44, y), COLORS["linea"], 1)
    prototype = payload.protocollo
    rows = [
        f"chunk {prototype.get('window_s', '?')}s / hop {prototype.get('hop_s', '?')}s",
        f"{prototype.get('num_frames', '?')} frames @ {prototype.get('target_fps', '?')} fps",
        f"tau {payload.threshold:.4f} ({payload.threshold_source})",
    ]
    for key, label in (("chunk_auc", "chunk AUC"), ("video_f1", "video F1")):
        value = payload.metriche.get(key)
        if isinstance(value, (int, float)):
            rows.append(f"{label} {value:.4f}")
    for row, text in enumerate(rows[:5]):
        cv2.putText(
            canvas,
            text,
            (px, y + 24 + row * 20),
            _FONT_MONO,
            0.36,
            COLORS["tenue"],
            1,
            cv2.LINE_AA,
        )


def _bar(
    canvas: np.ndarray, x: int, y: int, width: int, valore: float, soglia: float, color: tuple
) -> None:
    """Draw the score bar with the threshold mark: the panel gauge."""
    cv2.rectangle(canvas, (x, y), (x + width, y + 8), COLORS["linea"], -1)
    filled = int(width * min(max(valore, 0.0), 1.0))
    if filled > 0:
        cv2.rectangle(canvas, (x, y), (x + filled, y + 8), color, -1)
    xs = x + int(width * min(max(soglia, 0.0), 1.0))
    cv2.line(canvas, (xs, y - 4), (xs, y + 12), COLORS["testo"], 1)


def _timecode(frame: int, fps: float) -> str:
    """:return: ``mm:ss`` from the frame number; ``--:--`` if the fps is unknown."""
    if not fps:
        return "--:--"
    seconds = int(frame / fps)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _truncated_text(
    canvas: np.ndarray, testo: str, org: tuple[int, int], width: int, scala: float, color: tuple
) -> None:
    """Write the text, shortening it with an ellipsis until it fits."""
    t = testo
    while t and cv2.getTextSize(t, _FONT_MONO, scala, 1)[0][0] > width:
        t = t[:-1]
    if t != testo:
        t = t[:-1] + "…"
    cv2.putText(canvas, t, org, _FONT_MONO, scala, color, 1, cv2.LINE_AA)


def _in_alarm(payload: DemoPayload, indice: int) -> bool:
    """:return: whether the frame falls inside a detected interval."""
    return any(seg.start <= indice <= seg.end for seg in payload.predetti)


def _in_event(payload: DemoPayload, indice: int) -> str | None:
    """:return: the name of the class annotated on this frame, or ``None``."""
    for seg in payload.annotati:
        if seg.start <= indice <= seg.end:
            return seg.label or "evento"
    return None


def _strip_background(payload: DemoPayload) -> np.ndarray:
    """Return the timeline strip without the cursor, drawn once.

    Curve, threshold and the two lanes do not change between frames, so they are drawn
    once and the cursor is added per frame.

    :return: ``[STRIP_H, CANVAS_W, 3]`` BGR.
    """
    s = np.empty((STRIP_H, CANVAS_W, 3), dtype=np.uint8)
    s[:] = COLORS["pannello"]
    cv2.line(s, (0, 0), (CANVAS_W, 0), COLORS["linea"], 1)

    margin, top = 24, 78
    width = CANVAS_W - 2 * margin
    n = max(payload.num_frames, 1)

    def _x(frame: float) -> int:
        return int(margin + width * min(max(frame / n, 0.0), 1.0))

    # The curve, one point per pixel column: denser sampling is invisible, sparser sampling
    # would lose narrow peaks.
    y = payload.frame_scores
    covered = payload.covered
    base_y = 12 + top
    points: list[tuple[int, int]] = []
    for col in range(width):
        a = int(col * n / width)
        b = max(int((col + 1) * n / width), a + 1)
        if not covered[a:b].any():
            continue
        v = float(y[a:b].max())
        points.append((margin + col, int(base_y - v * top)))
    if len(points) > 1:
        arr = np.array(points, dtype=np.int32)
        fill = np.vstack([arr, [[arr[-1, 0], base_y], [arr[0, 0], base_y]]])
        stratum = s.copy()
        cv2.fillPoly(stratum, [fill], COLORS["calma"])
        cv2.addWeighted(stratum, 0.16, s, 0.84, 0.0, dst=s)
        cv2.polylines(s, [arr], False, COLORS["calma"], 2, cv2.LINE_AA)

    y_threshold = int(base_y - payload.threshold * top)
    for x in range(margin, CANVAS_W - margin, 10):
        cv2.line(s, (x, y_threshold), (x + 5, y_threshold), COLORS["tenue"], 1)
    cv2.putText(
        s,
        f"tau {payload.threshold:.3f}",
        (margin, y_threshold - 6),
        _FONT_MONO,
        0.36,
        COLORS["tenue"],
        1,
        cv2.LINE_AA,
    )

    # The two lanes: the decision above, the annotation below.
    for seg in payload.predetti:
        cv2.rectangle(
            s,
            (_x(seg.start), base_y + 12),
            (_x(seg.end + 1), base_y + 22),
            COLORS["allarme"],
            -1,
        )
    for seg in payload.annotati:
        cv2.rectangle(
            s,
            (_x(seg.start), base_y + 28),
            (_x(seg.end + 1), base_y + 38),
            COLORS["verita"],
            -1,
        )
    cv2.putText(
        s, "DETECTED", (margin, base_y + 9), _FONT_MONO, 0.34, COLORS["tenue"], 1, cv2.LINE_AA
    )
    label = "GROUND TRUTH" if payload.ha_verita else "GROUND TRUTH — n/d"
    cv2.putText(s, label, (margin, base_y + 51), _FONT_MONO, 0.34, COLORS["tenue"], 1, cv2.LINE_AA)
    return s


def _draw_cursor(canvas: np.ndarray, payload: DemoPayload, indice: int) -> None:
    """Draw the vertical line of the current time on the strip."""
    margin = 24
    width = CANVAS_W - 2 * margin
    x = int(margin + width * min(indice / max(payload.num_frames, 1), 1.0))
    cv2.line(canvas, (x, VIDEO_H + 6), (x, CANVAS_H - 8), COLORS["testo"], 1, cv2.LINE_AA)
