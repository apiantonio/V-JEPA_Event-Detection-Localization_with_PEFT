"""The data feeding every demonstrator artifact.

``timeline.png``, ``annotated.mp4`` and ``index.html`` show the same thing in three
forms: the renderers receive an already decided :class:`DemoPayload` (which intervals,
which threshold, which ground truth), so they cannot diverge. Nothing model-related is
computed here (ADR-015): scores come from :mod:`evd.exp.demo`, intervals from
:func:`evd.infer.engine.intervals_from_scores`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: Digits of the scores in the JSON: the ``edp_v2`` operating point is 0.1469 and the
#: threshold sensitivity reads on the third digit.
DIGITS = 4


@dataclass(frozen=True)
class Segment:
    """A frame interval, with inclusive extremes as everywhere in the repository.

    :param start: first frame.
    :param end: last frame, inclusive.
    :param label: text to draw on it (class name, or ``None``).
    :param peak: maximum score in the interval, ``None`` for the ground truth.
    """

    start: int
    end: int
    label: str | None = None
    peak: float | None = None

    def as_dict(self, fps: float) -> dict[str, Any]:
        """:return: the JSON form, with seconds already computed for the renderers."""
        return {
            "start": int(self.start),
            "end": int(self.end),
            "start_s": round(self.start / fps, 3) if fps else 0.0,
            "end_s": round((self.end + 1) / fps, 3) if fps else 0.0,
            "label": self.label,
            "peak": None if self.peak is None else round(float(self.peak), DIGITS),
        }


@dataclass(frozen=True)
class DemoPayload:
    """Everything the demonstrator shows about a video, already decided.

    :param run_id: run the weights or scores come from.
    :param modo: ``live`` or ``replay``, written because the two look alike.
    :param video_id: the video.
    :param video_src: how the viewer reaches the file, relative to the page.
    :param fps: native frames per second of the file.
    :param num_frames: native frames.
    :param width: width in pixels.
    :param height: height in pixels.
    :param threshold: the applied operating point tau.
    :param threshold_source: how it was estimated.
    :param frame_scores: ``[num_frames]`` scores projected on the frames.
    :param covered: ``[num_frames]`` frames covered by a causal window.
    :param finestre: ``(start, end, score)`` per window, in temporal order.
    :param predetti: intervals above threshold, after hysteresis.
    :param annotati: the ground truth, empty if unavailable.
    :param ha_verita: whether an annotation existed (an empty ``annotati`` is the right
        answer on a normal video).
    :param metriche: run summary, for the strip at the bottom of the page.
    :param protocollo: geometry fields describing a window.
    :param metriche_video: per-video metrics shown on the page.
    :param localization: saved localization boxes of the run for this video
        (:func:`apps.demo.localization.for_video`), ``None`` when there are none.
    """

    run_id: str
    modo: str
    video_id: str
    video_src: str
    fps: float
    num_frames: int
    width: int
    height: int
    threshold: float
    threshold_source: str
    frame_scores: np.ndarray
    covered: np.ndarray
    finestre: tuple[tuple[int, int, float], ...]
    predetti: tuple[Segment, ...]
    annotati: tuple[Segment, ...]
    ha_verita: bool
    metriche: Mapping[str, Any]
    protocollo: Mapping[str, Any]
    metriche_video: Mapping[str, Any] = field(default_factory=dict)
    localization: Mapping[str, Any] | None = None

    @property
    def durata_s(self) -> float:
        return self.num_frames / self.fps if self.fps else 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON form read inline by the viewer.

        Scores are rounded to :data:`DIGITS` and the coverage mask travels as run-length.

        :return: a dictionary serializable with ``json.dumps``.
        """
        return {
            "run_id": self.run_id,
            "modo": self.modo,
            "video": {
                "id": self.video_id,
                "src": self.video_src,
                "fps": round(float(self.fps), 6),
                "num_frames": int(self.num_frames),
                "width": int(self.width),
                "height": int(self.height),
                "duration_s": round(self.durata_s, 3),
            },
            "threshold": {
                "value": round(float(self.threshold), 6),
                "source": self.threshold_source,
            },
            "frame_scores": [round(float(v), DIGITS) for v in self.frame_scores],
            "covered_runs": [[int(a), int(b)] for a, b in _run_length(self.covered)],
            "windows": [
                {
                    "start": int(a),
                    "end": int(b),
                    "score": round(float(s), DIGITS),
                    "start_s": round(a / self.fps, 3) if self.fps else 0.0,
                    "end_s": round((b + 1) / self.fps, 3) if self.fps else 0.0,
                }
                for a, b, s in self.finestre
            ],
            "predicted": [seg.as_dict(self.fps) for seg in self.predetti],
            "ground_truth": [seg.as_dict(self.fps) for seg in self.annotati],
            "has_ground_truth": bool(self.ha_verita),
            "metrics": dict(self.metriche),
            "video_metrics": dict(self.metriche_video),
            "protocol": dict(self.protocollo),
            "localization": None if self.localization is None else dict(self.localization),
        }

    def save_json(self, out_path: str | Path) -> Path:
        """Write ``demo.json``, the archivable form of the payload.

        Lets a figure be redone or a new question answered ("and if tau were 0.3?") without
        running anything again, as ``scores.jsonl`` does (ADR-051).

        :param out_path: where to write it; parents are created.
        :return: the written path.
        """
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.as_dict(), indent=1), encoding="utf-8")
        return p


def _run_length(mask: np.ndarray) -> list[tuple[int, int]]:
    """:return: inclusive ``(start, end)`` intervals where ``mask`` is true."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    edges = np.flatnonzero(np.diff(np.r_[False, m, False].astype(np.int8)))
    return [(int(a), int(b) - 1) for a, b in zip(edges[0::2], edges[1::2], strict=True)]


def build_catalog(
    risultato: Any,
    *,
    run_id: str,
    modo: str,
    threshold: float,
    threshold_source: str,
    video_src: str,
    annotation: Any | None,
    classi: Sequence[Any],
    metriche: Mapping[str, Any],
    protocollo: Mapping[str, Any],
    metriche_video: Mapping[str, Any] | None = None,
    localization: Mapping[str, Any] | None = None,
) -> DemoPayload:
    """Pack a ``VideoResult`` into what the renderers can draw.

    :param risultato: the :class:`~evd.infer.engine.VideoResult` of the video.
    :param run_id: source run.
    :param modo: ``live`` or ``replay``.
    :param threshold: the applied operating point.
    :param threshold_source: how it was estimated.
    :param video_src: video path relative to the page that will show it.
    :param annotation: the ``TemporalAnnotation`` of the video, or ``None`` if unavailable
        (an unseen video, not an error).
    :param classi: the catalogue, to turn an index into a name.
    :param metriche: run summary shown on the page.
    :param protocollo: window geometry shown on the page.
    :param metriche_video: per-video metrics shown on the page, if any.
    :param localization: saved localization boxes for this video, if any.
    :return: the payload.
    """
    if metriche_video is None:
        from evd.eval.demo_metrics import metriche_video as compute_video_metrics

        metriche_video = compute_video_metrics(
            risultato.window_scores,
            risultato.meta,
            threshold,
            annotation=annotation,
            threshold_source=threshold_source,
        )
    meta = risultato.meta
    y = np.asarray(risultato.frame_scores.scores, dtype=float)
    names = {int(c.idx): str(c.name) for c in classi}
    predicted = tuple(
        Segment(start=s, end=e, label=None, peak=float(y[s : e + 1].max()) if e >= s else None)
        for s, e in risultato.intervals
    )
    annotated = (
        ()
        if annotation is None
        else tuple(
            Segment(start=s, end=e, label=names.get(int(c), f"class {c}"))
            for s, e, c in annotation.intervals
        )
    )
    return DemoPayload(
        run_id=run_id,
        modo=modo,
        video_id=meta.video_id,
        video_src=video_src,
        fps=float(meta.fps),
        num_frames=int(meta.num_frames),
        width=int(meta.width),
        height=int(meta.height),
        threshold=float(threshold),
        threshold_source=threshold_source,
        frame_scores=y,
        covered=np.asarray(risultato.frame_scores.covered, dtype=bool),
        finestre=tuple(
            (int(w.window.frame_span[0]), int(w.window.frame_span[1]), float(w.anomaly))
            for w in sorted(
                risultato.window_scores,
                key=lambda w: (w.window.end_frame, w.window.frame_span[0]),
            )
        ),
        predetti=predicted,
        annotati=annotated,
        ha_verita=annotation is not None,
        metriche=dict(metriche),
        protocollo=dict(protocollo),
        metriche_video=dict(metriche_video or {}),
        localization=localization,
    )
