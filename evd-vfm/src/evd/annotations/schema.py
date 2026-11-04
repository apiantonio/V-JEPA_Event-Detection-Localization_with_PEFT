"""Temporal annotation schema and its projection onto frames.

Intervals are expressed in native frames: the seconds-to-frames conversion
happens once, in the adapter that reads the source format.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from evd.types import EventClass

NORMAL_CLASS_NAME = "Normal"

TimeUnit = Literal["frames", "seconds"]


@dataclass(frozen=True)
class TemporalAnnotation:
    """Annotated intervals of one video.

    ``intervals`` holds ``(start, end, class_idx)`` triples with both ends
    inclusive, in native frames. A normal video has no intervals.
    """

    video_id: str
    intervals: tuple[tuple[int, int, int], ...] = ()

    def __post_init__(self) -> None:
        for s, e, c in self.intervals:
            if e < s:
                raise ValueError(f"{self.video_id}: inverted interval ({s}, {e})")
            if s < 0:
                raise ValueError(f"{self.video_id}: negative interval ({s}, {e})")
            if c < 0:
                raise ValueError(f"{self.video_id}: negative class_idx ({c})")

    @property
    def is_normal(self) -> bool:
        return len(self.intervals) == 0


@dataclass(frozen=True)
class RawInterval:
    """One row of an annotation file before validation.

    ``TemporalAnnotation`` raises on inverted intervals while the validator must
    report them, hence the two adapter stages ``parse()`` and ``load()``.
    """

    video_id: str
    class_name: str
    start: float
    end: float
    unit: TimeUnit
    source: str = ""
    """``file:line`` source, so that reports point at the offending row."""


def to_frames(value: float, unit: TimeUnit, fps: float) -> int:
    """Convert a value to native frames (the only conversion point of the system).

    :param value: value to convert.
    :param unit: ``"frames"`` or ``"seconds"``.
    :param fps: frames per second, used only when ``unit`` is ``"seconds"``.
    :return: the nearest frame index.
    :raises ValueError: if ``unit`` is ``"seconds"`` and ``fps`` is not finite and positive.
    """
    if unit == "frames":
        return int(round(value))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"invalid fps to convert {value}s into frames: {fps}")
    return int(round(value * fps))


def materialize(
    rows: Iterable[RawInterval],
    *,
    class_index: Mapping[str, int],
    fps: Mapping[str, float] | None = None,
    known_videos: Iterable[str] = (),
) -> dict[str, TemporalAnnotation]:
    """Turn ``RawInterval`` rows into valid annotations, raising on the first problem.

    :param rows: parsed rows.
    :param class_index: mapping from class name to index.
    :param fps: frames per second, for rows expressed in seconds.
    :param known_videos: videos that must appear even without intervals, so that a
        normal video is distinguishable from a missing one.
    :return: annotations keyed by video id.
    :raises ValueError: on unreadable rows or unknown classes.
    """
    per_video: dict[str, list[tuple[int, int, int]]] = {v: [] for v in known_videos}
    for row in rows:
        if not row.class_name:  # empty class name marks an unreadable row
            raise ValueError(f"{row.source}: unreadable row")
        if row.class_name not in class_index:
            raise ValueError(f"{row.source or row.video_id}: unknown class {row.class_name!r}")
        rate = (fps or {}).get(row.video_id, 0.0)
        s = to_frames(row.start, row.unit, rate)
        e = to_frames(row.end, row.unit, rate)
        per_video.setdefault(row.video_id, [])
        if row.class_name == NORMAL_CLASS_NAME:
            continue
        per_video[row.video_id].append((s, e, class_index[row.class_name]))
    return {
        video_id: TemporalAnnotation(video_id, tuple(sorted(intervals)))
        for video_id, intervals in per_video.items()
    }


@runtime_checkable
class AnnotationSource(Protocol):
    """Source of temporal annotations (FS-UCF-Crime, Sultani, MIVIA-Crime)."""

    def parse(self) -> list[RawInterval]: ...

    def load(self) -> dict[str, TemporalAnnotation]: ...

    def classes(self) -> tuple[EventClass, ...]: ...


def frame_labels(ann: TemporalAnnotation, num_frames: int) -> npt.NDArray[np.uint8]:
    """Binary per-frame vector: 1 where an event is annotated, 0 elsewhere.

    Out-of-range intervals raise, since they usually reveal a wrong fps.

    :param ann: annotation of the video.
    :param num_frames: number of native frames.
    :return: ``[num_frames]`` uint8 array.
    :raises ValueError: if an interval ends after ``num_frames - 1``.
    """
    y = np.zeros(num_frames, dtype=np.uint8)
    for s, e, _ in ann.intervals:
        if e >= num_frames:
            raise ValueError(
                f"{ann.video_id}: interval ({s}, {e}) beyond the {num_frames} frames of the video"
            )
        y[s : e + 1] = 1
    return y


def frame_class_labels(
    ann: TemporalAnnotation, num_frames: int, normal_idx: int
) -> npt.NDArray[np.int64]:
    """Per-frame class index, with ``normal_idx`` where no event is annotated.

    With overlapping intervals the last declared one wins.

    :param ann: annotation of the video.
    :param num_frames: number of native frames.
    :param normal_idx: index of the ``Normal`` class.
    :return: ``[num_frames]`` int64 array.
    :raises ValueError: if an interval ends after ``num_frames - 1``.
    """
    y = np.full(num_frames, normal_idx, dtype=np.int64)
    for s, e, c in ann.intervals:
        if e >= num_frames:
            raise ValueError(
                f"{ann.video_id}: interval ({s}, {e}) beyond the {num_frames} frames of the video"
            )
        y[s : e + 1] = c
    return y


def window_class_label(ann: TemporalAnnotation, end_frame: int, normal_idx: int) -> int:
    """Class of a causal window by the last-frame rule.

    The window takes the label of what happens at ``end_frame``. This differs from
    :func:`window_label`, which looks at the overlap with the whole window.

    :param ann: annotation of the video the window belongs to.
    :param end_frame: last sampled frame of the window, inclusive.
    :param normal_idx: index returned when no interval covers ``end_frame``.
    :return: the class index; with overlapping intervals the last one wins, as in
        :func:`frame_class_labels`.
    """
    label = normal_idx
    for s, e, c in ann.intervals:
        if s <= end_frame <= e:
            label = c
    return label


def window_label(ann: TemporalAnnotation, start: int, end: int, min_overlap: float = 0.0) -> int:
    """Return 1 if the window ``[start, end]`` intersects an event, 0 otherwise.

    :param ann: annotation of the video.
    :param start: first frame of the window, inclusive.
    :param end: last frame of the window, inclusive.
    :param min_overlap: required overlap fraction (``0`` means one frame suffices).
    :return: 1 or 0.
    :raises ValueError: if ``end < start``.
    """
    span = end - start + 1
    if span <= 0:
        raise ValueError(f"empty window: ({start}, {end})")
    covered = 0
    for s, e, _ in ann.intervals:
        covered += max(0, min(end, e) - max(start, s) + 1)
    if min_overlap <= 0.0:
        return int(covered > 0)
    return int(covered / span >= min_overlap)


def window_overlap(ann: TemporalAnnotation, start: int, end: int) -> tuple[int, int | None]:
    """Annotated frames inside ``[start, end]`` and the dominant class.

    Single overlap rule shared by training (chunk classes) and evaluation (chunk
    records). On ties the last interval in annotation order wins, as in
    :func:`frame_class_labels`.

    :param ann: annotation of the video (inclusive intervals, native frames).
    :param start: first frame of the window, inclusive.
    :param end: last frame of the window, inclusive.
    :return: ``(overlapping frames, class)``; the class is ``None`` when the window
        touches no interval.
    """
    covered_items = 0
    best = 0
    class_name: int | None = None
    for s, e, c in ann.intervals:
        common = min(end, e) - max(start, s) + 1
        if common > 0:
            covered_items += common
            if common >= best:
                best = common
                class_name = c
    return covered_items, class_name
