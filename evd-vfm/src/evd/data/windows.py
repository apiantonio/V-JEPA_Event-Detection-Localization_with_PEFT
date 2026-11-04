"""Causal windows on a temporal grid.

Two grids live in this single module, selected by ``grid``:

``"endpoint"``
    Historical grid (``fsp_v1``, ``edp_v1``, ``synthetic_toy``). A window is defined
    by its endpoint: samples lie at ``i / target_fps`` backwards from there, and the
    last window ends on the last frame. A video shorter than one span yields no
    window.

``"fixed"``
    Laboratory protocol grid (``edp_v2``). A window is defined by its start,
    ``start = k * hop`` in native frames, with nominal duration ``window_s`` and
    half-open bounds (240 frames and a 120-frame step at 30 fps). The first window
    shorter than ``window_s`` is still evaluated and closes the video, so every video
    yields at least one window.

``target_fps`` is a physical frequency: sample instants are projected onto native
frames once, here, with a quantization error below half a native frame.

On the ``fixed`` grid the samples cover the whole window (``linspace`` over
``[start, end]``, as upstream ``loadvideo_decord``), so the observed content matches
the labeled interval. The effective frequency is therefore
``(num_frames - 1) / window_duration`` (3.89 fps on a full 30 fps window).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from evd.types import PreprocessSpec, VideoMeta, WindowSpec
from evd.utils.logging import get_logger

log = get_logger(__name__)

SegmentPolicy = Literal["cover", "one_end"]

#: Which grid is in use (see the module docstring).
WindowGrid = Literal["endpoint", "fixed"]


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def frame_offsets_for(native_fps: float, target_fps: float, num_frames: int) -> tuple[int, ...]:
    """Derive the sample offsets relative to the window end.

    :param native_fps: measured frame rate of the file.
    :param target_fps: temporal frequency requested by the protocol.
    :param num_frames: samples per window.
    :return: decreasing offsets, from the oldest sample to zero.
    :raises ValueError: on invalid fps, upsampling or a non-increasing grid.
    """
    if not math.isfinite(native_fps) or not math.isfinite(target_fps):
        raise ValueError(f"non-finite fps: native={native_fps}, target={target_fps}")
    if native_fps <= 0 or target_fps <= 0:
        raise ValueError(f"invalid fps: native={native_fps}, target={target_fps}")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be > 0, not {num_frames}")
    if target_fps > native_fps:
        raise ValueError(
            f"target_fps={target_fps} exceeds native_fps={native_fps}: "
            "upsampling requires an explicit policy"
        )
    ratio = native_fps / target_fps
    offsets = tuple(_round_half_up((num_frames - 1 - i) * ratio) for i in range(num_frames))
    if any(right >= left for left, right in zip(offsets, offsets[1:], strict=False)):
        raise ValueError(
            f"temporal grid not strictly increasing: native_fps={native_fps}, "
            f"target_fps={target_fps}, offsets={offsets}"
        )
    return offsets


def stride_frames_for(native_fps: float, target_fps: float) -> int:
    """Return the nearest uniform stride, for profilers and historical comparisons only.

    The protocol sampler uses :func:`frame_offsets_for`.

    :param native_fps: measured frame rate.
    :param target_fps: requested frequency.
    :return: integer stride in native frames.
    :raises ValueError: on invalid fps.
    """
    if not math.isfinite(native_fps) or not math.isfinite(target_fps):
        raise ValueError(f"non-finite fps: native={native_fps}, target={target_fps}")
    if native_fps <= 0 or target_fps <= 0:
        raise ValueError(f"invalid fps: native={native_fps}, target={target_fps}")
    return max(1, int(round(native_fps / target_fps)))


@dataclass(frozen=True)
class CausalWindowSampler:
    """Generate causal windows with duration and hop expressed in seconds.

    :param num_frames: samples per window.
    :param target_fps: requested frequency (nominal on the ``fixed`` grid).
    :param hop_s: nominal distance between consecutive windows, in seconds.
    :param grid: grid in use (see the module docstring).
    :param window_s: nominal window duration in seconds; required with ``grid="fixed"``
        and forbidden otherwise.
    """

    num_frames: int
    target_fps: float
    hop_s: float
    grid: WindowGrid = "endpoint"
    window_s: float | None = None

    def __post_init__(self) -> None:
        if self.num_frames <= 0:
            raise ValueError(f"num_frames must be > 0, not {self.num_frames}")
        for name in ("target_fps", "hop_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0, not {value}")
        if self.grid not in ("endpoint", "fixed"):
            raise ValueError(
                f"grid={self.grid!r}: the grids are 'endpoint' and 'fixed' (laboratory protocol)"
            )
        if self.grid == "fixed":
            if self.window_s is None:
                raise ValueError(
                    "grid='fixed' without window_s: the nominal duration defines the grid "
                    "(deriving it from (num_frames - 1) / target_fps would give 7.75 s "
                    "instead of 8.0)"
                )
            if not math.isfinite(self.window_s) or self.window_s <= 0:
                raise ValueError(f"window_s must be finite and > 0, not {self.window_s}")
        elif self.window_s is not None:
            raise ValueError(
                f"window_s={self.window_s} with grid={self.grid!r}: on that grid the duration "
                f"is (num_frames - 1) / target_fps, so the field would not affect the "
                f"computation"
            )

    @classmethod
    def from_spec(
        cls,
        spec: PreprocessSpec,
        hop_s: float,
        *,
        grid: WindowGrid = "endpoint",
        window_s: float | None = None,
    ) -> CausalWindowSampler:
        """Build the sampler from the geometry shared by all videos.

        :param spec: preprocessing spec with target fps and sample count.
        :param hop_s: nominal distance between consecutive windows, in seconds.
        :param grid: grid in use.
        :param window_s: nominal window duration, required by ``fixed``.
        :return: a sampler reusable on videos with different native fps.
        """
        return cls(
            num_frames=spec.num_frames,
            target_fps=spec.fps,
            hop_s=hop_s,
            grid=grid,
            window_s=window_s,
        )

    @property
    def span_s(self) -> float:
        """Return the nominal distance between the first and last sampled timestamp."""
        return (self.num_frames - 1) / self.target_fps

    def span_frames(self, meta: VideoMeta) -> int:
        """Return the native frames between the first and last sample, inclusive.

        This is the ``endpoint`` geometry; on the ``fixed`` grid use :meth:`window_frames`
        (234 against 240 frames at 30 fps).

        :param meta: video metadata.
        :return: span in native frames.
        """
        return frame_offsets_for(meta.fps, self.target_fps, self.num_frames)[0] + 1

    def window_frames(self, meta: VideoMeta) -> int:
        """Return the nominal length of a full window, in native frames.

        :param meta: video metadata (native fps).
        :return: ``round_half_up(window_s * fps)`` on the ``fixed`` grid, :meth:`span_frames`
            otherwise.
        """
        if self.grid == "fixed":
            assert self.window_s is not None  # guaranteed by __post_init__
            return _round_half_up(self.window_s * meta.fps)
        return self.span_frames(meta)

    def hop_frames(self, meta: VideoMeta) -> int:
        """:return: the step between consecutive windows, in native frames.
        :raises ValueError: if the hop is shorter than one frame.
        """
        step = _round_half_up(self.hop_s * meta.fps)
        if step < 1:
            raise ValueError(
                f"{meta.video_id}: hop_s={self.hop_s} is shorter than one frame at {meta.fps} fps"
            )
        return step

    def effective_span_s(self, meta: VideoMeta) -> float:
        """Return the window duration after quantization on the native grid."""
        return (self.window_frames(meta) - 1) / meta.fps

    def effective_fps(self, meta: VideoMeta) -> float:
        """Return the effective sampling frequency inside a full window.

        :param meta: video metadata.
        :return: samples per second, or ``inf`` for a single-sample window.
        """
        duration = self.effective_span_s(meta)
        if duration <= 0:
            return float("inf")
        return (self.num_frames - 1) / duration

    def _window(self, meta: VideoMeta, end_frame: int) -> WindowSpec:
        offsets = frame_offsets_for(meta.fps, self.target_fps, self.num_frames)
        indices = tuple(end_frame - offset for offset in offsets)
        return WindowSpec.from_frame_indices(meta.video_id, indices)

    def _video_endpoints(self, meta: VideoMeta) -> list[int]:
        if self.hop_s * meta.fps < 1.0:
            raise ValueError(
                f"{meta.video_id}: hop_s={self.hop_s} is shorter than one frame at {meta.fps} fps"
            )
        first_end = self.span_frames(meta) - 1
        last_end = meta.num_frames - 1
        if last_end < first_end:
            return []
        endpoints: list[int] = []
        index = 0
        while True:
            end = _round_half_up((self.span_s + index * self.hop_s) * meta.fps)
            if end > last_end:
                break
            if not endpoints or end != endpoints[-1]:
                endpoints.append(end)
            index += 1
        if endpoints[-1] != last_end:
            endpoints.append(last_end)
        return endpoints

    def spans_for_video(self, meta: VideoMeta) -> list[tuple[int, int]]:
        """Return the ``(first, last)`` frame of every window, inclusive.

        This is the grid definition from which windows, labels and weights derive. On the
        ``fixed`` grid the last element is the truncated window closing the video.

        :param meta: video metadata.
        :return: intervals in temporal order; empty only on the ``endpoint`` grid for videos
            shorter than one span.
        :raises ValueError: if ``num_frames`` is negative or the hop is below one frame.
        """
        if meta.num_frames < 0:
            raise ValueError(f"{meta.video_id}: negative num_frames ({meta.num_frames})")
        if self.grid == "endpoint":
            span = self.span_frames(meta)
            return [(end - span + 1, end) for end in self._video_endpoints(meta)]
        if meta.num_frames == 0:
            return []
        width = self.window_frames(meta)
        step = self.hop_frames(meta)
        spans: list[tuple[int, int]] = []
        start = 0
        while True:
            spans.append((start, min(start + width, meta.num_frames) - 1))
            if start + width > meta.num_frames:
                # The window just added is the truncated one: evaluate it and close the video.
                break
            start += step
            if start >= meta.num_frames:
                # The video ends exactly on a step boundary and has no tail: without this check a
                # window starting after the end would be added.
                break
        return spans

    def window_for_span(self, meta: VideoMeta, span: tuple[int, int]) -> WindowSpec:
        """Return the window sampled on a grid interval.

        :param meta: video metadata.
        :param span: ``(first, last)`` native frame, both inclusive.
        :return: the window with the indices to decode; on a window shorter than
            ``num_frames`` the indices repeat.
        :raises ValueError: if the interval is inverted or outside the video.
        """
        first, last = int(span[0]), int(span[1])
        if first < 0 or last >= meta.num_frames or last < first:
            raise ValueError(
                f"{meta.video_id}: span ({first}, {last}) outside [0, {meta.num_frames - 1}]"
            )
        if self.grid == "endpoint":
            return self._window(meta, last)
        length = last - first + 1
        if self.num_frames == 1:
            return WindowSpec.from_frame_indices(meta.video_id, (last,))
        indices = tuple(
            first + _round_half_up(i * (length - 1) / (self.num_frames - 1))
            for i in range(self.num_frames)
        )
        return WindowSpec.from_frame_indices(meta.video_id, indices)

    def span_at(self, meta: VideoMeta, end_frame: int) -> tuple[int, int]:
        """Return the span of a full window ending at ``end_frame``.

        Used by dynamic sampling to relabel a window whose endpoint moved inside its cell.

        :param meta: video metadata.
        :param end_frame: last frame of the window.
        :return: ``(first, last)`` frame.
        :raises ValueError: if the window would leave the video.
        """
        width = self.window_frames(meta)
        first = end_frame - width + 1
        if first < 0 or end_frame > meta.num_frames - 1:
            raise ValueError(
                f"{meta.video_id}: endpoint {end_frame} outside the allowed endpoints "
                f"[{width - 1}, {meta.num_frames - 1}]"
            )
        return (first, end_frame)

    def window_at(self, meta: VideoMeta, end_frame: int) -> WindowSpec:
        """Return the grid window ending at ``end_frame``.

        Public because dynamic sampling builds windows at endpoints off the hop grid; the
        sample offsets stay those of :func:`frame_offsets_for`.

        :param meta: video metadata (fps and native frame count).
        :param end_frame: last sampled frame, inclusive.
        :return: the causal window.
        :raises ValueError: if the window would leave the video (see :meth:`endpoint_cell`).
        """
        return self.window_for_span(meta, self.span_at(meta, end_frame))

    def endpoint_cell(self, meta: VideoMeta, end_frame: int) -> tuple[int, int]:
        """Return the allowed endpoints around a grid point, both inclusive.

        The hop cell of dynamic sampling is centered on the grid endpoint and one hop wide
        (``[e - hop/2, e + hop/2]`` in native frames), then clipped to the endpoints the
        video allows. Width and centering are our declared convention.

        :param meta: video metadata.
        :param end_frame: endpoint of the static grid.
        :return: ``(minimum, maximum)``, both allowed.
        :raises ValueError: if the video allows no window.
        """
        first = self.window_frames(meta) - 1
        last = meta.num_frames - 1
        if last < first:
            raise ValueError(
                f"{meta.video_id}: {meta.num_frames} frames are not enough for a window "
                f"of {self.window_frames(meta)}"
            )
        half = (self.hop_s * meta.fps) / 2.0
        bottom = max(first, int(math.ceil(end_frame - half)))
        top = min(last, int(math.floor(end_frame + half)))
        # The starting endpoint is always allowed: if rounding inverts the bounds, the cell
        # collapses on it instead of raising.
        if top < bottom:
            bottom = top = min(max(end_frame, first), last)
        return bottom, top

    def windows_for_video(self, meta: VideoMeta) -> list[WindowSpec]:
        """Return every window of the video, in temporal order.

        On the ``endpoint`` grid a video shorter than one window yields an empty list (no
        padding); on the ``fixed`` grid the list is never empty.

        :param meta: video metadata.
        :return: the windows.
        """
        spans = self.spans_for_video(meta)
        if not spans:
            log.debug(
                "%s: %d frames, a window requires %d -> no window",
                meta.video_id,
                meta.num_frames,
                self.window_frames(meta),
            )
            return []
        return [self.window_for_span(meta, span) for span in spans]

    def windows_for_segment(
        self,
        meta: VideoMeta,
        seg: tuple[int, int],
        policy: SegmentPolicy,
        *,
        max_windows: int | None = None,
    ) -> list[WindowSpec]:
        """Return the windows associated with an annotated segment ``(start, end)``.

        ``policy`` has no default because it is a protocol decision:

        ``"cover"``    grid windows ending inside the segment, at most ``max_windows``
                       equispaced ones when given.
        ``"one_end"``  the last grid window ending inside the segment.

        :param meta: video metadata.
        :param seg: inclusive segment.
        :param policy: ``"cover"`` or ``"one_end"``.
        :param max_windows: maximum number of windows for ``"cover"``.
        :return: the windows.
        :raises NotImplementedError: on the ``fixed`` grid.
        :raises ValueError: on an unknown policy or an invalid segment.
        """
        if self.grid != "endpoint":
            raise NotImplementedError(
                f"windows_for_segment is not defined on the {self.grid!r} grid: it serves the "
                f"few-shot manifest builder (fsp_v1), which runs on the 'endpoint' grid"
            )
        start, end = int(seg[0]), int(seg[1])
        if policy not in ("cover", "one_end"):
            raise ValueError(f"unknown policy: {policy!r} (expected 'cover' | 'one_end')")
        if end < start:
            raise ValueError(f"inverted segment: {seg}")
        if start < 0 or end >= meta.num_frames:
            raise ValueError(
                f"segment {seg} outside video {meta.video_id} ({meta.num_frames} frames)"
            )
        if max_windows is not None and max_windows <= 0:
            raise ValueError(f"max_windows must be > 0, not {max_windows}")
        ends = [endpoint for endpoint in self._video_endpoints(meta) if start <= endpoint <= end]
        if not ends:
            return []
        if policy == "one_end":
            return [self._window(meta, ends[-1])]
        if policy == "cover":
            if max_windows is not None and len(ends) > max_windows:
                if max_windows == 1:
                    ends = [ends[-1]]
                else:
                    positions = tuple(
                        _round_half_up(i * (len(ends) - 1) / (max_windows - 1))
                        for i in range(max_windows)
                    )
                    ends = [ends[position] for position in positions]
            return [self._window(meta, candidate) for candidate in ends]
        raise AssertionError("validated policy not handled")  # pragma: no cover
