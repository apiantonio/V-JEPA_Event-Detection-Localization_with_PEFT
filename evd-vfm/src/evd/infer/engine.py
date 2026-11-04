"""``InferenceEngine``: the only place where the system builds windows.

Evaluation, efficiency profiling and the demonstrator all go through it, so offline
numbers and online behaviour cannot differ because of windowing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from evd.data.dataset import WindowDataset, WindowItem, collate_windows, homogeneous_batches
from evd.data.decoder import VideoDecoder
from evd.data.feature_store import FeatureReader
from evd.data.windows import CausalWindowSampler
from evd.eval.frame_map import FrameScoreMapper
from evd.eval.scorers import WindowScorer
from evd.types import FrameScores, PreprocessSpec, VideoMeta, WindowScores, WindowSpec


@dataclass(frozen=True)
class VideoResult:
    video_id: str
    meta: VideoMeta
    window_scores: tuple[WindowScores, ...]
    frame_scores: FrameScores
    intervals: tuple[tuple[int, int], ...] = ()
    """Detected intervals (threshold + hysteresis); empty without a threshold."""


def intervals_from_scores(
    scores: FrameScores, *, on: float, off: float, min_len: int = 1
) -> tuple[tuple[int, int], ...]:
    """Threshold scores with hysteresis: open at ``on``, close at ``off`` (<= ``on``).

    Hysteresis avoids bursts of one-frame intervals when the score oscillates around a
    single threshold; ``min_len`` drops short blips.

    :raises ValueError: if ``on < off``.
    """
    if off > on:
        raise ValueError(f"inverted hysteresis: on={on} < off={off}")
    y = np.asarray(scores.scores, dtype=float)
    cov = np.asarray(scores.covered, dtype=bool)
    out: list[tuple[int, int]] = []
    start: int | None = None
    for i in range(len(y)):
        if not cov[i]:
            continue
        if start is None:
            if y[i] >= on:
                start = i
        elif y[i] < off:
            if i - start >= min_len:
                out.append((start, i - 1))
            start = None
    if start is not None and len(y) - start >= min_len:
        out.append((start, len(y) - 1))
    return tuple(out)


def video_result(
    mapper: FrameScoreMapper,
    meta: VideoMeta,
    scores: Sequence[WindowScores],
    *,
    threshold: float | None = None,
    off_threshold: float | None = None,
    min_len: int = 1,
) -> VideoResult:
    """Turn window scores into a video result: frame projection and intervals.

    Module-level because it has two callers: the engine, which computes the scores, and
    the demonstrator (``apps/demo/``, ADR-015), which replays them from disk
    (:mod:`evd.infer.replay`). Both share the deciding part, so timelines and annotated
    videos follow the same rule.

    :param mapper: window-to-frame projector, with the protocol policy.
    :param meta: the video, providing the native frame count.
    :param scores: window scores, in any order (sorted by the mapper).
    :param threshold: operating point, or ``None`` to skip interval extraction.
    :param off_threshold: hysteresis closing threshold; ``None`` means ``threshold``.
    :param min_len: minimum interval length, in frames.
    :return: the video result, with empty ``intervals`` without a threshold.
    :raises ValueError: if a window belongs to another video or the hysteresis is inverted.
    """
    frames = mapper.to_frames(scores, meta)
    intervals: tuple[tuple[int, int], ...] = ()
    if threshold is not None:
        intervals = intervals_from_scores(
            frames,
            on=threshold,
            off=threshold if off_threshold is None else off_threshold,
            min_len=min_len,
        )
    return VideoResult(meta.video_id, meta, tuple(scores), frames, intervals)


class InferenceEngine:
    def __init__(
        self,
        scorer: WindowScorer,
        sampler: CausalWindowSampler,
        mapper: FrameScoreMapper,
        decoder: VideoDecoder,
        spec: PreprocessSpec,
        *,
        batch_size: int = 4,
        num_workers: int = 0,
        cache: FeatureReader | None = None,
        cache_keys: Callable[[WindowSpec], tuple[str, ...]] | None = None,
    ) -> None:
        """:param num_workers: decoding workers for :meth:`run_videos` only; zero (default) keeps
            the serial behaviour. :meth:`stream` is never parallel.
        :param cache: feature store, for a frozen backbone only (invariant 3).
        :param cache_keys: builds the keys of a window, one per level read. Provided from
            outside because the key ``(backbone_id, ckpt_sha, preprocess_hash, layer,
            window_id)`` depends on information the engine does not own.
        :raises ValueError: if a cache is given without ``cache_keys``.
        """
        self.scorer, self.sampler, self.mapper = scorer, sampler, mapper
        self.decoder, self.spec, self.batch_size = decoder, spec, batch_size
        self.num_workers = int(num_workers)
        if cache is not None and cache_keys is None:
            raise ValueError(
                "cache without cache_keys: the engine cannot build the ADR-007 key "
                "(backbone_id, ckpt_sha, preprocess_hash, layer, window_id)"
            )
        self.cache, self.cache_keys = cache, cache_keys
        self.cache_hits = 0
        self.cache_misses = 0

    def stream(
        self,
        path: str | Path,
        meta: VideoMeta | None = None,
        *,
        before_window: Callable[[WindowSpec], None] | None = None,
    ) -> Iterator[WindowScores]:
        """Yield the windows of a video one at a time, in temporal order.

        Live-like form used by the demonstrator and the latency profiler: the same sequence of
        decisions an online system would take, also with a cache (batches stay homogeneous and
        contiguous in time).

        :raises ValueError: if a timed source is used with ``batch_size != 1``.
        """
        if before_window is not None and self.batch_size != 1:
            raise ValueError("The timed source requires batch_size=1.")
        m = meta or self.decoder.probe(path)
        items = self.finestre(path, m)
        ds = WindowDataset(items, self.decoder, self.spec, cache=self.cache)
        for slice_ in homogeneous_batches(items, self.cache, self.batch_size):
            if before_window is not None:
                before_window(items[slice_[0]].window)
            batch = collate_windows([ds[j] for j in slice_])
            yield from self.scorer.score(batch)
        self.cache_hits += ds.cache_hits
        self.cache_misses += ds.cache_misses

    def cache_hit_rate(self) -> float | None:
        """:return: fraction of windows served by the cache, or ``None`` if no window was looked
        up (no cache configured).
        """
        attempted = self.cache_hits + self.cache_misses
        return self.cache_hits / attempted if attempted else None

    def finestre(self, path: str | Path, m: VideoMeta) -> list[WindowItem]:
        """Return the windows of a video, as consumed by the dataset.

        Both pixel paths (:meth:`stream` and :meth:`run_videos`) start from these windows in
        the same order; it is public because external callers (explanation campaigns,
        :mod:`evd.exp.descrittori`) must reuse it instead of rebuilding windows.

        :param path: video file.
        :param m: its metadata, already probed.
        :return: windows in temporal order.
        """
        return [
            WindowItem(
                window=w,
                video_path=str(path),
                cache_keys=() if self.cache_keys is None else self.cache_keys(w),
            )
            for w in self.sampler.windows_for_video(m)
        ]

    #: Old name, kept for callers that have not migrated yet; new code uses
    #: :meth:`finestre`.
    _windows = finestre

    def _result(
        self,
        m: VideoMeta,
        ws: tuple[WindowScores, ...],
        *,
        threshold: float | None,
        off_threshold: float | None,
        min_len: int,
    ) -> VideoResult:
        """Turn window scores into the video result with the engine's mapper.

        Shared by both ways of computing ``ws``; the rule lives in :func:`video_result`.

        :param m: video metadata.
        :param ws: window scores, in temporal order.
        :param threshold: operating point, or ``None`` to skip interval extraction.
        :param off_threshold: hysteresis closing threshold; ``None`` means ``threshold``.
        :param min_len: minimum interval length, in frames.
        :return: the video result.
        """
        return video_result(
            self.mapper,
            m,
            ws,
            threshold=threshold,
            off_threshold=off_threshold,
            min_len=min_len,
        )

    def run_video(
        self,
        path: str | Path,
        meta: VideoMeta | None = None,
        *,
        threshold: float | None = None,
        off_threshold: float | None = None,
        min_len: int = 1,
    ) -> VideoResult:
        """Process a whole video: window scores, frame scores and intervals.

        ``threshold`` has no default: the operating point comes from the protocol or the
        demonstrator config.
        """
        m = meta or self.decoder.probe(path)
        return self._result(
            m,
            tuple(self.stream(path, m)),
            threshold=threshold,
            off_threshold=off_threshold,
            min_len=min_len,
        )

    def run_videos(self, metas: Sequence[VideoMeta], **kwargs: Any) -> dict[str, VideoResult]:
        """Process many videos; with ``num_workers > 0`` decoding overlaps the forward pass.

        :meth:`stream` stays serial because it is what latency profiling measures. Offline
        evaluation uses the same windows from :meth:`finestre` and only changes who decodes
        them. Without workers the GPU waits for decoding (a cell evaluation once ran at 1.78
        windows/s against 4.04 in training). The parallel branch is not used with a feature
        store: batches must stay homogeneous (:func:`~evd.data.dataset.homogeneous_batches`)
        and there is no decoding to overlap.

        :param metas: videos to evaluate.
        :param kwargs: forwarded to :meth:`run_video` / :meth:`_result` (``threshold``,
            ``off_threshold``, ``min_len``).
        :return: ``video_id`` to result.
        """
        if self.num_workers <= 0 or self.cache is not None:
            return {m.video_id: self.run_video(m.path, m, **kwargs) for m in metas}
        return self._run_videos_with_workers(metas, **kwargs)

    def _run_videos_with_workers(
        self,
        metas: Sequence[VideoMeta],
        *,
        threshold: float | None = None,
        off_threshold: float | None = None,
        min_len: int = 1,
    ) -> dict[str, VideoResult]:
        """Run one ``DataLoader`` over all windows of all videos.

        Concatenating lets decoding of video *n+1* overlap the forward of video *n*. Order is
        preserved (``shuffle=False``), so scores are reassigned by count.

        :param metas: videos to evaluate.
        :return: ``video_id`` to result.
        :raises RuntimeError: if the loader returns a different number of scores.
        """
        from evd.data.dataset import build_adapted_loader

        per_video = [self.finestre(m.path, m) for m in metas]
        items = [it for windows in per_video for it in windows]
        if not items:
            return {}
        loader = build_adapted_loader(
            items,
            decoder=self.decoder,
            spec=self.spec,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            # No augmentation at evaluation: preprocessing is deterministic, so the number of
            # workers does not change the pixels.
            augment=None,
        )
        scores: list[WindowScores] = []
        for batch in loader:
            scores.extend(self.scorer.score(batch))
        if len(scores) != len(items):
            raise RuntimeError(
                f"the loader returned {len(scores)} scores for {len(items)} windows: "
                f"the per-video reassignment would be silently wrong"
            )
        outside: dict[str, VideoResult] = {}
        start = 0
        for m, windows in zip(metas, per_video, strict=True):
            slice_ = tuple(scores[start : start + len(windows)])
            start += len(windows)
            outside[m.video_id] = self._result(
                m, slice_, threshold=threshold, off_threshold=off_threshold, min_len=min_len
            )
        return outside
