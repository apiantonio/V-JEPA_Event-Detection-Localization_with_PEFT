"""``timeline.png``: the first ADR-015 artifact.

Calls :func:`evd.eval.plots.timeline_png`, the same function the runner uses for a run
timeline, so the demonstrator and the run never draw two different readings of a video.
"""

from __future__ import annotations

from pathlib import Path

from apps.demo.payload import DemoPayload
from evd.annotations.schema import TemporalAnnotation
from evd.eval.plots import timeline_png as _timeline_png
from evd.types import FrameScores


def scrivi(payload: DemoPayload, out_path: str | Path) -> Path:
    """Write the payload timeline: score, threshold, prediction and ground truth.

    :param payload: the demonstrator data.
    :param out_path: where to write the PNG.
    :return: the written path.
    """
    return _timeline_png(
        FrameScores(
            video_id=payload.video_id,
            scores=payload.frame_scores.astype("float32"),
            covered=payload.covered,
        ),
        out_path,
        gt=_truth(payload),
        threshold=payload.threshold,
        fps=payload.fps,
        title=f"{payload.video_id} — {payload.run_id} ({payload.modo})",
        predicted=tuple((s.start, s.end) for s in payload.predetti),
    )


def _truth(payload: DemoPayload) -> TemporalAnnotation | None:
    """Return the annotation in the form the figure expects, or ``None`` if missing.

    ``ha_verita`` and an empty ``annotati`` differ: on a normal video the annotation exists
    and has no intervals.
    """
    if not payload.ha_verita:
        return None
    intervals: tuple[tuple[int, int, int], ...] = tuple(
        (s.start, s.end, 0) for s in payload.annotati
    )
    return TemporalAnnotation(video_id=payload.video_id, intervals=intervals)
