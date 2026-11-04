"""Project chunk scores onto frames.

The ``causal`` policy keeps on frame ``t`` the score of the latest chunk whose
endpoint is at most ``t``; overlap aggregations remain available for offline
analyses. On the ``fixed`` grid two windows may end on the same frame (the truncated
tail): the later-starting one wins.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from evd.types import FrameScores, VideoMeta, WindowScores

MappingPolicy = Literal["causal", "max", "mean", "last"]


def _slice(ws: WindowScores, n: int) -> slice | None:
    """:return: the frames covered by the window, or ``None`` if it falls outside the video."""
    lo, hi = ws.window.frame_span
    hi = min(hi, n - 1)
    return slice(lo, hi + 1) if lo <= hi else None


@dataclass(frozen=True)
class FrameScoreMapper:
    policy: MappingPolicy
    pad_first_window: bool = False
    """Offline policies only: extend the first covered score back to the start."""

    def __post_init__(self) -> None:
        """:raises ValueError: if the policy does not exist."""
        if self.policy not in ("causal", "max", "mean", "last"):
            raise ValueError(
                f"unknown policy: {self.policy!r}. Allowed: causal (protocol), "
                f"max, mean, last (offline)"
            )

    def to_frames(self, scores: Sequence[WindowScores], meta: VideoMeta) -> FrameScores:
        """Project chunk scores onto the native frames of the video.

        The frame level is outside the evaluation protocol: this projection draws timelines.

        :param scores: chunk scores of the video, in any order (sorted by ``(end, start)``).
        :param meta: the video, providing the native frame count.
        :return: per-frame scores and coverage mask; ``class_scores`` is ``None`` when chunks
            carry no ``class_probs``.
        :raises ValueError: if a chunk belongs to another video, an endpoint is outside the
            video, ``class_probs`` widths differ, or ``pad_first_window`` is used with the
            causal policy.
        """
        n = meta.num_frames
        out = np.zeros(n, dtype=np.float32)
        covered = np.zeros(n, dtype=bool)
        if not scores:
            return FrameScores(meta.video_id, out, covered)

        # Sort by `(end, start)`: windows sharing an endpoint must end with the later-starting
        # one, which writes last and wins.
        ordered = sorted(
            scores, key=lambda item: (item.window.end_frame, item.window.frame_span[0])
        )
        class_widths = {len(item.class_probs) for item in ordered if item.class_probs is not None}
        if len(class_widths) > 1 or (class_widths and any(x.class_probs is None for x in ordered)):
            raise ValueError("class_probs must be present with the same width on every chunk")
        class_width = next(iter(class_widths), 0)
        class_out = np.zeros((n, class_width), dtype=np.float32) if class_width else None

        for item in ordered:
            if item.window.video_id != meta.video_id:
                raise ValueError(
                    f"window of {item.window.video_id} passed to the mapper of {meta.video_id}"
                )
            if not 0 <= item.window.end_frame < n:
                raise ValueError(
                    f"endpoint {item.window.end_frame} outside the {n} frames of {meta.video_id}"
                )

        if self.policy == "causal":
            if self.pad_first_window:
                raise ValueError("pad_first_window is not allowed with the causal mapping")
            return self._causal(ordered, meta, out, covered, class_out)

        offline = {"max": self._max, "mean": self._mean, "last": self._last}
        offline[self.policy](ordered, n, out, covered, class_out, class_width)

        if self.pad_first_window and covered.any():
            first = int(np.argmax(covered))
            out[:first] = out[first]
            if class_out is not None:
                class_out[:first] = class_out[first]
            covered[:first] = True
        return FrameScores(meta.video_id, out, covered, class_out)

    def _max(
        self,
        ordered: Sequence[WindowScores],
        n: int,
        out: npt.NDArray[np.float32],
        covered: npt.NDArray[np.bool_],
        class_out: npt.NDArray[np.float32] | None,
        class_width: int,
    ) -> None:
        """Write in place the maximum over the windows covering each frame."""
        del class_width
        for ws in ordered:
            sl = _slice(ws, n)
            if sl is None:
                continue
            # Assign where the frame was not covered yet, so negative scores are not clamped by
            # the zero initialization.
            new = np.float32(ws.anomaly)
            out[sl] = np.where(covered[sl], np.maximum(out[sl], new), new)
            if class_out is not None and ws.class_probs is not None:
                new_class = np.asarray(ws.class_probs, dtype=np.float32)
                class_out[sl] = np.where(
                    covered[sl, None], np.maximum(class_out[sl], new_class), new_class
                )
            covered[sl] = True

    def _mean(
        self,
        ordered: Sequence[WindowScores],
        n: int,
        out: npt.NDArray[np.float32],
        covered: npt.NDArray[np.bool_],
        class_out: npt.NDArray[np.float32] | None,
        class_width: int,
    ) -> None:
        """Write in place the mean over the windows covering each frame."""
        acc = np.zeros(n, dtype=np.float64)
        cnt = np.zeros(n, dtype=np.int32)
        class_acc = np.zeros((n, class_width), dtype=np.float64) if class_width else None
        for ws in ordered:
            sl = _slice(ws, n)
            if sl is None:
                continue
            acc[sl] += ws.anomaly
            cnt[sl] += 1
            if class_acc is not None and ws.class_probs is not None:
                class_acc[sl] += np.asarray(ws.class_probs, dtype=np.float64)
            covered[sl] = True
        with np.errstate(invalid="ignore"):
            out[:] = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0).astype(np.float32)
            if class_out is not None and class_acc is not None:
                class_out[:] = np.where(
                    cnt[:, None] > 0,
                    class_acc / np.maximum(cnt[:, None], 1),
                    0.0,
                ).astype(np.float32)

    def _last(
        self,
        ordered: Sequence[WindowScores],
        n: int,
        out: npt.NDArray[np.float32],
        covered: npt.NDArray[np.bool_],
        class_out: npt.NDArray[np.float32] | None,
        class_width: int,
    ) -> None:
        """Write in place the score of the last window covering each frame."""
        del class_width
        for ws in ordered:
            sl = _slice(ws, n)
            if sl is None:
                continue
            # Windows arrive by increasing end frame (guaranteed by CausalWindowSampler), so the
            # last writer is the one ending latest.
            out[sl] = ws.anomaly
            if class_out is not None and ws.class_probs is not None:
                class_out[sl] = np.asarray(ws.class_probs, dtype=np.float32)
            covered[sl] = True

    @staticmethod
    def _causal(
        scores: Sequence[WindowScores],
        meta: VideoMeta,
        out: npt.NDArray[np.float32],
        covered: npt.NDArray[np.bool_],
        class_out: npt.NDArray[np.float32] | None,
    ) -> FrameScores:
        """Keep on each frame the latest score already available.

        Two windows may share an endpoint on the ``fixed`` grid (truncated tail); they are
        different windows, so only identical spans are rejected. The later-starting window
        wins on the shared frame.

        :param scores: chunk scores of the video.
        :param meta: video metadata.
        :param out: per-frame output, written in place.
        :param covered: coverage mask, written in place.
        :param class_out: per-frame class scores, or ``None``.
        :return: the frame scores.
        :raises ValueError: on repeated spans or an incomplete causal mapping.
        """
        endpoints = [item.window.end_frame for item in scores]
        spans = [item.window.frame_span for item in scores]
        if len(set(spans)) != len(spans):
            repeated = sorted({s for s in spans if spans.count(s) > 1})
            raise ValueError(
                f"repeated windows in the causal mapper of {meta.video_id}: {repeated[:5]}. "
                "Equal endpoints are allowed (tail of the `fixed` grid), equal spans are not"
            )
        if endpoints[-1] != meta.num_frames - 1:
            raise ValueError(
                f"incomplete causal mapping for {meta.video_id}: last endpoint "
                f"{endpoints[-1]}, last frame {meta.num_frames - 1}"
            )
        for index, item in enumerate(scores):
            next_end = (
                scores[index + 1].window.end_frame if index + 1 < len(scores) else meta.num_frames
            )
            sl = slice(item.window.end_frame, next_end)
            out[sl] = item.anomaly
            if class_out is not None and item.class_probs is not None:
                class_out[sl] = np.asarray(item.class_probs, dtype=np.float32)
            covered[sl] = True
        return FrameScores(meta.video_id, out, covered, class_out)
