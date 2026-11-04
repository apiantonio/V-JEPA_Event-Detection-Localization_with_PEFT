"""``attention.mp4``: where the model looks, overlaid on the video (ADR-063).

``annotated.mp4`` shows when the system decides; this shows on what. Three cuts:

* ``overlay``: the video with the map on top, for figures and slides;
* ``affiancato``: original on the left, map on the right, to judge the map against a
  clean reference;
* ``mappa``: the heatmap alone, the check that tells whether a barely visible map is
  weak or just covered by the frame.

Nothing is computed here: map, contrast and frame pairing are decided in
:mod:`evd.exp.explain`. ``heat`` arrives in ``[0, 1]`` but, with the default percentile
stretch (:data:`~evd.explain.attention.CONTRAST_MODES`), that range is no longer "zero to
maximum"; a stretch without its scale would make a flat map look selective, so the
caller passes ``scale_lo``/``scale_hi`` in multiples of the uniform weight and the strip
prints them. The legend states that a hot area is a position the pooler query reads
from, not a pixel that alone caused the decision.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from apps.demo.overlay import COLORS

__all__ = ["ALFA", "GAMMA", "LAYOUT", "DIVERGING_LUT", "SEQUENTIAL_LUT", "scrivi"]

#: Available cuts. ``entrambi`` writes overlay and side-by-side (historical CLI default);
#: ``tutti`` adds the bare map. Separate files, one per use.
LAYOUT = ("overlay", "affiancato", "mappa", "entrambi", "tutti")

#: Height of the reading strip below the video.
STRIP_H = 112

#: Map weight over the frame at the top of the scale. Alpha stays proportional to the
#: intensity, so cold areas leave the video visible; verifiability comes from the
#: side-by-side cut.
ALFA = 0.80

#: Exponent applied to the already contrasted map; it only changes how fast colour grows
#: within the declared scale (0.7 lifts mid intensities).
GAMMA = 0.7

#: Width of the scale bar in the reading strip.
BAR_W, BAR_H = 260, 12


def _diverging_lut() -> np.ndarray:
    """Return the two-sided colormap for deviation maps, as a ``[256,1,3]`` LUT.

    Built by hand: ``COLORMAP_COOLWARM`` does not exist in this OpenCV build (5.0.0 exposes
    22 tables, none diverging). Three BGR nodes (cold blue, neutral grey, warm red),
    linearly interpolated; the light centre makes zero ("like the video mean") visible.

    :return: the uint8 LUT accepted by ``cv2.applyColorMap``.
    """
    nodes = np.array([[176, 96, 58], [226, 226, 226], [58, 76, 200]], dtype=np.float32)
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    channels = [np.interp(x, [0.0, 0.5, 1.0], nodes[:, c]) for c in range(3)]
    return np.stack(channels, axis=-1).reshape(256, 1, 3).astype(np.uint8)


#: The diverging LUT, built once.
DIVERGING_LUT = _diverging_lut()

#: Sequential colormap for modes with a zero rather than a centre (``massimo``,
#: ``percentile``, ``finestra``). Viridis is perceptually uniform, readable in greyscale
#: and by common colour-vision deficiencies; it starts from dark violet, so alpha (which
#: follows the intensity) keeps cold areas transparent.
SEQUENTIAL_LUT = cv2.COLORMAP_VIRIDIS

_FONT = cv2.FONT_HERSHEY_DUPLEX
_FONT_MONO = cv2.FONT_HERSHEY_SIMPLEX


def scrivi(
    video_path: str | Path,
    out_path: str | Path,
    *,
    heat: np.ndarray,
    anomaly: np.ndarray,
    threshold: float,
    fps: float,
    video_id: str,
    run_id: str,
    layout: str = "overlay",
    fourcc: str = "mp4v",
    max_width: int = 960,
    alfa: float = ALFA,
    gamma: float = GAMMA,
    divergente: bool = False,
    scale_lo: float | None = None,
    scale_hi: float | None = None,
    contrast_mode: str = "",
) -> Path:
    """Draw the attention map on the video, in the requested cut.

    :param video_path: the original video file, read sequentially with OpenCV (the video as
        it is, not the preprocessed clip).
    :param out_path: where to write the mp4; parents are created.
    :param heat: ``[num_frames, H', W']`` float32 in ``[0, 1]``, ``nan`` on frames no window
        covers.
    :param anomaly: ``[num_frames]`` float32, the score projected on the frame.
    :param threshold: the run operating point, to tell whether the frame is in alarm.
    :param fps: frames per second of the output file.
    :param video_id: video name, for the strip.
    :param run_id: run the weights come from, for the strip.
    :param layout: ``"overlay"``, ``"affiancato"`` or ``"mappa"`` (:data:`LAYOUT`);
        ``"entrambi"`` and ``"tutti"`` are handled by the caller.
    :param fourcc: the four-character codec.
    :param max_width: the video is downscaled to this width if larger (never upscaled).
    :param alfa: map opacity at the top of the scale (:data:`ALFA`).
    :param gamma: exponent on the contrasted map (:data:`GAMMA`).
    :param divergente: if ``True`` the map is a deviation from the video mean (0.5 = mean),
        drawn with a two-sided colormap, transparent at the centre.
    :param scale_lo: value of map 0 in multiples of the uniform weight; ``None`` removes the
        scale bar.
    :param scale_hi: value of map 1.
    :param contrast_mode: mode name, printed in the strip.
    :return: the written path.
    :raises ValueError: if the layout is unknown, ``heat`` and ``anomaly`` differ in frame
        count, or alpha is outside ``[0, 1]``.
    :raises RuntimeError: if the video does not open or the codec is missing.
    """
    if layout not in ("overlay", "affiancato", "mappa"):
        raise ValueError(f"unknown layout {layout!r}: expected 'overlay', 'affiancato' or 'mappa'")
    if heat.shape[0] != anomaly.shape[0]:
        raise ValueError(
            f"heat covers {heat.shape[0]} frames and anomaly {anomaly.shape[0]}: they are two "
            f"projections of the same thing and must have the same length"
        )
    if not 0.0 <= alfa <= 1.0:
        raise ValueError(f"alpha {alfa} outside [0, 1]: it is an opacity, not a gain")
    p_out = Path(out_path)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 320
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 240
    scale = min(1.0, max_width / max(1, width))
    w, h = int(width * scale), int(height * scale)
    canvas_w = w * 2 if layout == "affiancato" else w
    writer = cv2.VideoWriter(
        str(p_out), cv2.VideoWriter_fourcc(*fourcc), fps or 25.0, (canvas_w, h + STRIP_H)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"OpenCV cannot create the writer for {p_out} with fourcc {fourcc!r}: the codec is "
            f"not available in this build. Try `--fourcc avc1` or `mp4v`"
        )
    try:
        for index, frame in _frames(cap, heat.shape[0]):
            small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            background = small if layout != "mappa" else np.zeros_like(small)
            opacity = 1.0 if layout == "mappa" else alfa
            hot = _overlay(background, heat[index], opacity, gamma, divergente)
            canvas = np.full((h + STRIP_H, canvas_w, 3), COLORS["fondo"], dtype=np.uint8)
            if layout == "affiancato":
                canvas[:h, :w] = small
                canvas[:h, w:] = hot
                cv2.line(canvas, (w, 0), (w, h), COLORS["linea"], 1)
            else:
                canvas[:h, :w] = hot
            _strip(
                canvas,
                h,
                canvas_w,
                index,
                anomaly,
                threshold,
                fps,
                video_id,
                run_id,
                layout,
                divergente,
                scale_lo,
                scale_hi,
                contrast_mode,
            )
            writer.write(canvas)
    finally:
        writer.release()
        cap.release()
    return p_out


def _frames(cap: cv2.VideoCapture, expected: int) -> Iterator[tuple[int, np.ndarray]]:
    """Yield the video frames in order, stopping at the inventory count.

    Arrays are ``num_frames`` long, and some containers yield one extra frame.

    :param cap: the open video.
    :param attesi: how many frames the arrays describe.
    :yield: ``(index, BGR frame)``.
    """
    index = 0
    while index < expected:
        ok, frame = cap.read()
        if not ok:
            return
        yield index, frame
        index += 1


def _colorize(map_data: np.ndarray, divergente: bool) -> np.ndarray:
    """Turn the ``[0, 1]`` map into a BGR image.

    :param mappa: ``[H, W]`` in ``[0, 1]``.
    :param divergente: whether the scale has a neutral centre at 0.5.
    :return: ``[H, W, 3]`` uint8.
    """
    levels = (np.clip(map_data, 0.0, 1.0) * 255).astype(np.uint8)
    lut = DIVERGING_LUT if divergente else SEQUENTIAL_LUT
    return cv2.applyColorMap(levels, lut)


def _overlay(
    frame: np.ndarray,
    map_data: np.ndarray,
    alfa: float,
    gamma: float,
    divergente: bool,
) -> np.ndarray:
    """Blend the map onto the frame, or return it greyed if there is no map.

    A frame no window covers (the first frames of every video under the causal policy) gets
    the dimmed frame, not a flat map: "not decided yet" and "looking everywhere" must look
    different.

    :param frame: the resized BGR frame (black in the ``mappa`` cut).
    :param mappa: ``[H', W']`` in ``[0, 1]``, or all ``nan``.
    :param alfa: opacity at the top of the scale.
    :param gamma: exponent on the intensity.
    :param divergente: whether 0.5 is the neutral centre of the scale.
    :return: the frame with the map on top.
    """
    if not np.isfinite(map_data).any():
        return cv2.addWeighted(frame, 0.45, np.zeros_like(frame), 0.55, 0)
    clean = np.nan_to_num(map_data, nan=0.5 if divergente else 0.0).astype(np.float32)
    h, w = frame.shape[:2]
    large = np.clip(cv2.resize(clean, (w, h), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)
    color = _colorize(large, divergente)
    # Alpha follows the intensity, so the video stays visible where the model does not look;
    # on a deviation map the ramp is measured from the centre (0.5).
    intensity = np.abs(large - 0.5) * 2.0 if divergente else large
    weight = (np.power(intensity, gamma) * alfa)[..., None]
    return (frame * (1 - weight) + color * weight).astype(np.uint8)


def _scale_bar(
    canvas: np.ndarray, x: int, y: int, width: int, lo: float, hi: float, divergente: bool
) -> None:
    """Draw the colormap gradient with its extremes, in place.

    :param canvas: the canvas.
    :param x: starting column.
    :param y: starting row.
    :param larghezza: gradient width in pixels (narrow videos cannot fit :data:`BAR_W`).
    :param lo: value at the left end, in multiples of the uniform weight.
    :param hi: value at the right end.
    :param divergente: which colormap to use.
    """
    ramp = np.linspace(0.0, 1.0, max(2, width), dtype=np.float32)[None, :]
    canvas[y : y + BAR_H, x : x + width] = _colorize(np.repeat(ramp, BAR_H, axis=0), divergente)
    cv2.rectangle(canvas, (x, y), (x + width, y + BAR_H), COLORS["linea"], 1)
    cv2.putText(
        canvas, f"{lo:+.2f}", (x - 42, y + 11), _FONT_MONO, 0.34, COLORS["tenue"], 1, cv2.LINE_AA
    )
    cv2.putText(
        canvas,
        f"{hi:+.2f} x unif.",
        (x + width + 6, y + 11),
        _FONT_MONO,
        0.34,
        COLORS["tenue"],
        1,
        cv2.LINE_AA,
    )


def _row(
    canvas: np.ndarray,
    testo: str,
    x: int,
    y: int,
    width: int,
    scala: float,
    color: tuple[int, int, int],
    font: int = _FONT_MONO,
) -> None:
    """Write a row, shortening it if it does not fit, in place.

    The canvas can be 320 px wide (native UCF-Crime) or 1920 (side-by-side): without
    shortening, provenance and caveat would leave the frame.

    :param canvas: the canvas.
    :param testo: the row to write.
    :param x: starting column.
    :param y: baseline.
    :param larghezza: canvas width.
    :param scala: font scale.
    :param colore: BGR colour.
    :param font: OpenCV font.
    """
    available = width - x - 8
    current = testo
    while current:
        (w, _), _ = cv2.getTextSize(current, font, scala, 1)
        if w <= available:
            break
        current = current[:-2] if len(current) > 2 else ""
    if current and current != testo:
        current = current[:-1] + "…"
    if current:
        cv2.putText(canvas, current, (x, y), font, scala, color, 1, cv2.LINE_AA)


def _strip(
    canvas: np.ndarray,
    y: int,
    width: int,
    indice: int,
    anomaly: np.ndarray,
    threshold: float,
    fps: float,
    video_id: str,
    run_id: str,
    layout: str,
    divergente: bool,
    scale_lo: float | None,
    scale_hi: float | None,
    contrast_mode: str,
) -> None:
    """Draw the reading strip below the video, in place.

    :param canvas: the canvas to draw on.
    :param y: row where the strip starts.
    :param larghezza: canvas width.
    :param indice: current frame.
    :param anomaly: per-frame scores.
    :param threshold: the operating point.
    :param fps: to convert the frame into seconds.
    :param video_id: video name.
    :param run_id: run of the weights.
    :param layout: printed in the strip, so a detached file still tells its cut.
    :param divergente: which colormap the bar shows.
    :param scale_lo: left scale extreme, in multiples of the uniform weight; ``None``
        removes the bar.
    :param scale_hi: right extreme.
    :param contrast_mode: mode name, next to the bar.
    """
    cv2.rectangle(canvas, (0, y), (width, canvas.shape[0]), COLORS["pannello"], -1)
    cv2.line(canvas, (0, y), (width, y), COLORS["linea"], 1)
    score = float(anomaly[indice]) if np.isfinite(anomaly[indice]) else float("nan")
    alarm = np.isfinite(score) and score >= threshold
    accent = COLORS["allarme"] if alarm else COLORS["calma"]
    text = "n/d" if not np.isfinite(score) else f"{score:.3f}"

    # The bar sits right of the first row on wide canvases and below the text on narrow ones.
    bar_w = 0
    x_bar, y_bar = 62, y + 84
    if scale_lo is not None and scale_hi is not None:
        wide = width >= 620
        bar_w = max(60, min(BAR_W, width - (300 if wide else 170)))
        if wide:
            x_bar, y_bar = width - bar_w - 110, y + 24
    reserved = width - bar_w - 130 if bar_w and y_bar == y + 24 else width

    cv2.putText(canvas, "ATTENTION MAP", (16, y + 26), _FONT, 0.52, COLORS["testo"], 1, cv2.LINE_AA)
    _row(
        canvas,
        f"{video_id}  ·  {run_id}  ·  {layout}",
        16,
        y + 48,
        reserved,
        0.36,
        COLORS["tenue"],
    )
    _row(
        canvas,
        f"score {text}  (tau {threshold:.3f})   t={indice / (fps or 25.0):6.2f}s",
        16,
        y + 70,
        width,
        0.40,
        accent,
    )
    _row(
        canvas,
        "pooler cross-attention: where the query reads, not what caused the decision",
        16,
        y + 104,
        width,
        0.32,
        COLORS["tenue"],
    )
    if not bar_w:
        return
    _scale_bar(canvas, x_bar, y_bar, bar_w, scale_lo, scale_hi, divergente)
    if contrast_mode:
        cv2.putText(
            canvas,
            f"contrast: {contrast_mode}",
            (x_bar - 42, y_bar - 6),
            _FONT_MONO,
            0.32,
            COLORS["tenue"],
            1,
            cv2.LINE_AA,
        )
