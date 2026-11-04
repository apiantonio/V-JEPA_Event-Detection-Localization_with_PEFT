"""Scores of a run read back from disk, without running the model.

Used by the demonstrator (``apps/demo/``) and to recompute metrics from the
same predictions (``scores.jsonl``). Only the score comes from the file: the
geometry is rebuilt by :class:`~evd.data.windows.CausalWindowSampler`, and a mismatch
raises. The partition travels with the scores because in ``edp_v2`` validation has its
own contiguous grid.

Rows are written by :func:`evd.eval.metrics.score_record`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from evd.data.windows import CausalWindowSampler
from evd.types import VideoMeta, WindowScores, WindowSpec
from evd.utils.io import read_jsonl
from evd.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class ScoreVideo:
    """Scores of one video read from ``scores.jsonl``, with its partition.

    :param video_id: the video.
    :param split: ``test`` or ``val``: which grid these windows belong to.
    :param per_finestra: ``window_id`` to anomaly score.
    """

    video_id: str
    split: str
    per_finestra: Mapping[str, float]

    @property
    def validazione(self) -> bool:
        """:return: whether the video must be replayed with the validation sampler."""
        return self.split == "val"


def scores_from_disk(percorso: str | Path, *, split: str | None = None) -> dict[str, ScoreVideo]:
    """Read ``scores.jsonl`` and index it by video, keeping the partition.

    Unlike :func:`evd.eval.report.read_scores` it keeps the window identity and the
    windows with unknown label (``-1``), which a timeline still draws. Both readers share
    the :func:`evd.eval.metrics.score_record` schema.

    :param percorso: ``scores.jsonl`` file of a run.
    :param split: partition to keep (``test`` | ``val``), or ``None`` for all.
    :return: ``video_id`` to :class:`ScoreVideo`.
    :raises FileNotFoundError: if the file does not exist.
    :raises ValueError: if the requested partition is missing, or a video appears in two
        partitions.
    """
    p = Path(percorso)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} does not exist: the run predates ADR-051 (31 August 2026), when scores "
            f"started being written. A re-evaluation or a more recent run is "
            f"needed"
        )
    rows = read_jsonl(p)
    present = {str(r.get("split", "test")) for r in rows}
    if split is not None and split not in present:
        raise ValueError(
            f"{p} does not contain partition {split!r}: found {sorted(present)}. "
            f"A run evaluated with `eval_split: val` has no 'test' partition"
        )
    accumulator: dict[str, tuple[str, dict[str, float]]] = {}
    for r in rows:
        part = str(r.get("split", "test"))
        if split is not None and part != split:
            continue
        video = str(r["video_id"])
        if_already = accumulator.get(video)
        if if_already is None:
            accumulator[video] = (part, {})
        elif if_already[0] != part:
            raise ValueError(
                f"{video} appears in {p} both as {if_already[0]!r} and as {part!r}: the two "
                f"partitions have different grids (ADR-047) and neither can be chosen. "
                f"Pass `split=` to select one"
            )
        accumulator[video][1][str(r["window_id"])] = float(r["score"])
    return {
        video: ScoreVideo(video_id=video, split=part, per_finestra=windows)
        for video, (part, windows) in accumulator.items()
    }


def replay_video(
    punteggi: Mapping[str, ScoreVideo],
    meta: VideoMeta,
    sampler: CausalWindowSampler,
) -> tuple[WindowScores, ...]:
    """Rebuild the window scores of a video from the file instead of the model.

    Windows come from ``sampler``, the same object used in training, evaluation and the
    inference engine; the file only provides the score.

    :param punteggi: result of :func:`scores_from_disk`.
    :param meta: video to replay.
    :param sampler: sampler of this video's partition (see :attr:`ScoreVideo.validazione`);
        a wrong one finds no window and raises.
    :return: window scores in temporal order, as
        :meth:`~evd.infer.engine.InferenceEngine.stream` would yield them.
    :raises KeyError: if the video is not in the file.
    :raises ValueError: if a grid window has no score in the file.
    """
    if meta.video_id not in punteggi:
        raise KeyError(
            f"{meta.video_id} is not in the run scores: the run did not evaluate it. "
            f"Use `--modo live` with a checkpoint, or `--elenca` to list what it covers"
        )
    known = punteggi[meta.video_id].per_finestra
    windows: list[WindowSpec] = sampler.windows_for_video(meta)
    missing = [w.window_id for w in windows if w.window_id not in known]
    if missing:
        raise ValueError(
            f"{meta.video_id}: {len(missing)} of {len(windows)} windows have no score "
            f"in the file (first: {missing[0]}). The replay grid is not the one the run "
            f"measured: usually the video is from validation, which in edp_v2 has a "
            f"contiguous, non-overlapping grid (ADR-047), or the protocol changed "
            f"after the run"
        )
    if len(known) != len(windows):
        log.warning(
            "%s: the file has %d scores and the grid %d windows; the extra ones are ignored",
            meta.video_id,
            len(known),
            len(windows),
        )
    return tuple(WindowScores(window=w, anomaly=known[w.window_id]) for w in windows)


def available_videos(punteggi: Mapping[str, ScoreVideo]) -> tuple[str, ...]:
    """:return: the ``video_id`` values covered by the file, in alphabetical order."""
    return tuple(sorted(punteggi))
