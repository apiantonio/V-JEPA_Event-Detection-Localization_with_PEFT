"""From per-window maps to per-frame maps, with the same rule as the scores.

Heatmap and score timeline must describe the same instant. Two rules serve two
artifacts: :func:`windows_per_frame` asks the run mapper
(:class:`~evd.eval.frame_map.FrameScoreMapper`, invariant 4), causal on ``edp_v2``,
used by attention videos (``exp/explain.py``) and the demonstrator timeline;
:func:`offline_assignment` is the offline view of explainability videos, where the map
drawn on a frame must come from a window containing that frame.

The mapper projects scores, not indices, so it receives scores encoding the window
index (``i/(n-1)``, within ``[0, 1]``) and the winner is decoded back: if ADR-050
changes which window wins a shared frame, this module follows without changes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from evd.eval.frame_map import FrameScoreMapper
from evd.explain.attention import AttentionMap, reduce_over_frames
from evd.types import VideoMeta, WindowScores, WindowSpec

__all__ = [
    "ASSIGNMENT_POLICIES",
    "WindowsPerFrame",
    "windows_per_frame",
    "frame_annotati",
    "map_per_frame",
    "temporal_position",
    "offline_assignment",
    "offline_plane",
]

#: Rules choosing the window shown on a frame in the offline view. ``fine`` (default,
#: 20 September 2026) renders every campaign computed before 29 September;
#: ``overlap_positivo`` follows ADR-077
#: (``docs/incontri/incontro_explain-localizzazione_29-09.md``).
ASSIGNMENT_POLICIES = ("fine", "overlap_positivo")


def frame_annotati(intervalli: Sequence[Sequence[int]], num_frames: int) -> np.ndarray:
    """Return the native frames falling inside at least one annotated interval.

    Intervals come from ``rapporti/verita.json``, in native frames with both ends included,
    clamped to the video (same convention as the ``annotati`` box variant).

    :param intervalli: ``(start, end)`` pairs, ends included.
    :param num_frames: native frames of the video.
    :return: ``[num_frames]`` bool.
    """
    inside = np.zeros(num_frames, dtype=bool)
    for start, end in intervalli:
        inside[max(0, int(start)) : min(num_frames, int(end) + 1)] = True
    return inside


def offline_assignment(
    finestre: Sequence[WindowSpec],
    num_frames: int,
    *,
    politica: str = "fine",
    intervalli: Sequence[Sequence[int]] | None = None,
) -> np.ndarray:
    """Offline view: which of the windows containing a frame shows it.

    With both policies a window contributes only to the frames it contains,
    ``[start_frame, end_frame]``; they differ on who wins where windows overlap.

    - ``fine`` (default): :func:`_by_end`, the window closing first.
    - ``overlap_positivo``: :func:`_by_positive_overlap`, the window containing more
      annotated positive frames; ties go to the nearest centre.

    :param finestre: actual sampler windows, without score aggregation.
    :param num_frames: number of native frames.
    :param politica: one of :data:`ASSIGNMENT_POLICIES`.
    :param intervalli: annotated intervals in native frames, ends included; required by
        ``overlap_positivo`` (empty on a normal video) and ignored by ``fine``.
    :return: index per frame, ``-1`` where no window covers the frame.
    :raises ValueError: on an unknown policy, or ``overlap_positivo`` without intervals.
    """
    if politica == "fine":
        return _by_end(finestre, num_frames)
    if politica == "overlap_positivo":
        if intervalli is None:
            raise ValueError(
                "The 'overlap_positivo' policy picks the window with more annotated frames: "
                "without the intervals of rapporti/verita.json it has no answer. A normal "
                "video passes them EMPTY, not missing"
            )
        return _by_positive_overlap(finestre, num_frames, intervalli)
    raise ValueError(f"Unknown assignment policy: {politica!r}; allowed {ASSIGNMENT_POLICIES}")


def _by_positive_overlap(
    finestre: Sequence[WindowSpec], num_frames: int, intervalli: Sequence[Sequence[int]]
) -> np.ndarray:
    """The window with more annotated frames wins; ties go to the nearest centre.

    Rule of 29 September 2026 (ADR-077): on the ``edp_v2`` grid consecutive chunks always
    overlap by 4 s, and in the shared zone the frame shows the tubelet plane of the chunk
    containing more annotated positive frames, so an event is read entirely on the planes
    of the window that sees it best.

    Lexicographic key, for each frame ``f`` and window ``w`` containing it:

    1. more annotated native frames within ``[start_frame, end_frame]`` of ``w``;
    2. centre ``(start_frame + end_frame) / 2`` closest to ``f``;
    3. lower endpoint;
    4. later start (ADR-050), which makes the rule total.

    Where the overlap does not discriminate, the nearest-centre rule applies, whose switch
    falls half a hop before the chunk end (60 frames, 2 s on ``edp_v2``). It uses annotated
    positivity, not scores: an offline visualization choice used to propose spatial
    annotations inside annotated intervals, not to measure the detector.

    :param finestre: actual sampler windows.
    :param num_frames: number of native frames.
    :param intervalli: annotated intervals, ends included.
    :return: index per frame, ``-1`` where no window covers the frame.
    """
    positives = np.concatenate(([0], np.cumsum(frame_annotati(intervalli, num_frames))))
    result = np.full(num_frames, -1, dtype=np.int32)
    # Best key seen so far, per frame: lower is better on every column.
    best = np.full((num_frames, 4), np.inf)
    for i, w in enumerate(finestre):
        start, term = max(0, w.start_frame), min(num_frames, w.end_frame + 1)
        if term <= start:
            continue
        frames = np.arange(start, term)
        center = (w.start_frame + w.end_frame) / 2
        key = np.column_stack(
            (
                np.full(frames.size, -float(positives[term] - positives[start])),
                np.abs(frames - center),
                np.full(frames.size, float(w.end_frame)),
                np.full(frames.size, -float(w.start_frame)),
            )
        )
        current = best[frames]
        # Strict lexicographic comparison, column by column.
        smaller = np.zeros(frames.size, dtype=bool)
        equal = np.ones(frames.size, dtype=bool)
        for column in range(key.shape[1]):
            smaller |= equal & (key[:, column] < current[:, column])
            equal &= key[:, column] == current[:, column]
        result[frames[smaller]] = i
        best[frames[smaller]] = key[smaller]
    return result


def _by_end(finestre: Sequence[WindowSpec], num_frames: int) -> np.ndarray:
    """Offline view: among the windows containing the frame, the one closing first.

    Frame ``f`` goes to the containing window with the nearest end, so the switch falls
    exactly on a chunk end, the instant the chunk produces its score (as in the
    demonstrator, ``apps/demo/assets/viewer.html``, ``revealed()`` on ``(end + 1) / fps``).
    Unlike the causal policy of :class:`~evd.eval.frame_map.FrameScoreMapper`, the frame is
    always inside the window explaining it and coverage starts at frame 0; the shown score
    belongs to a window not yet closed (an offline reading). With equal ends the later
    start wins (ADR-050).

    :param finestre: actual sampler windows, without score aggregation.
    :param num_frames: number of native frames.
    :return: index per frame, ``-1`` where no window covers the frame.
    """
    result = np.full(num_frames, -1, dtype=np.int32)
    distance = np.full(num_frames, np.inf)
    # Order by `(end, start)` with a strict comparison: the start order is reversed so that,
    # on equal ends, the most recent window comes first and wins (ADR-050).
    order = sorted(
        range(len(finestre)), key=lambda j: (finestre[j].end_frame, -finestre[j].start_frame)
    )
    for i in order:
        w = finestre[i]
        frames = np.arange(max(0, w.start_frame), min(num_frames, w.end_frame + 1))
        if frames.size == 0:
            continue
        # Frames left before the window closes: non-negative by construction.
        d = w.end_frame - frames
        better = d < distance[frames]
        result[frames[better]] = i
        distance[frames[better]] = d[better]
    return result


def offline_plane(window: WindowSpec, frame: int, planes: int) -> int:
    """Map the samples to their tubelet; intermediate frames get a visual interpolation.

    :param window: window with sampled indices, repeated tail ones included.
    :param frame: native frame to show.
    :param planes: temporal planes of the map.
    :return: plane of the nearest sample; ties go to the previous sample.
    :raises ValueError: if the number of samples is not divisible by the planes.
    """
    if planes < 1 or window.num_frames % planes:
        raise ValueError("The number of samples must be divisible by the temporal planes")
    indices = np.asarray(window.frame_indices)
    sample = int(np.argmin(abs(indices - frame)))
    return sample // (window.num_frames // planes)


@dataclass(frozen=True)
class WindowsPerFrame:
    """Which window decides on each native frame of the video.

    :param indice: ``[num_frames]`` int32, index of the window assigned by the mapper,
        ``-1`` where none is assigned.
    :param coperto: ``[num_frames]`` bool, frames that received a window. With the causal
        policy the first frames have none (no decision yet).
    """

    indice: np.ndarray
    coperto: np.ndarray


def windows_per_frame(
    mapper: FrameScoreMapper, meta: VideoMeta, finestre: Sequence[WindowSpec]
) -> WindowsPerFrame:
    """Ask the mapper which window wins on each frame.

    :param mapper: the run mapper, the same one projecting the scores.
    :param meta: video metadata, for the number of frames.
    :param finestre: windows in temporal order.
    :return: frame to window index assignment.
    :raises ValueError: if there are no windows, or decoded indices are not integers (the
        mapper interpolated instead of assigning).
    """
    n = len(finestre)
    if n == 0:
        raise ValueError("no windows: nothing to project")
    # `anomaly` is documented in [0, 1]: the index is encoded as a fraction and decoded by
    # multiplication; the round trip is checked below.
    scale = float(n - 1) if n > 1 else 1.0
    dummy = [
        WindowScores(window=w, anomaly=(i / scale if n > 1 else 0.0))
        for i, w in enumerate(finestre)
    ]
    projected = mapper.to_frames(dummy, meta)
    raw = np.asarray(projected.scores, dtype=np.float64) * scale
    index = np.rint(raw).astype(np.int32)
    covered = np.asarray(projected.covered, dtype=bool)
    residual = np.abs(raw[covered] - index[covered])
    if residual.size and float(residual.max()) > 1e-3:
        raise ValueError(
            f"the projection produced non-integer indices (maximum residual "
            f"{float(residual.max()):.3g}): the mapper is AVERAGING windows instead of "
            f"assigning one, so 'which window decides on this frame' has no single "
            f"answer. With the 'mean' policy the map must be aggregated, not chosen"
        )
    index[~covered] = -1
    return WindowsPerFrame(indice=index, coperto=covered)


def temporal_position(finestra: WindowSpec, frame: int, steps: int) -> int:
    """Return the temporal grid plane of a native frame.

    The ``num_frames`` samples of a window are grouped into ``steps`` (``T'``) planes, one
    per tubelet (32 samples in 16 planes on ``edp_v2``). Sampled indices are
    non-decreasing (ADR-047), so ``searchsorted`` is valid.

    :param finestra: window deciding on that frame.
    :param frame: native frame index.
    :param steps: ``T'``, the temporal grid planes.
    :return: the plane, clamped to the ends (a frame after the window end takes the last
        plane, consistent with the causal policy).
    """
    indices = finestra.explicit_frame_indices
    if not indices:
        first = finestra.end_frame - (finestra.num_frames - 1) * (finestra.stride_frames or 1)
        position = (frame - first) // (finestra.stride_frames or 1)
    else:
        position = int(np.searchsorted(np.asarray(indices), frame, side="right")) - 1
    per_plane = max(1, finestra.num_frames // max(1, steps))
    return int(np.clip(position // per_plane, 0, steps - 1))


def map_per_frame(
    mappe: Sequence[AttentionMap],
    finestre: Sequence[WindowSpec],
    assignment: WindowsPerFrame,
    frame: int,
) -> np.ndarray | None:
    """Return the spatial slice to draw on a frame.

    :param mappe: maps, one per window and in the same order.
    :param finestre: windows in temporal order.
    :param assegnazione: output of :func:`windows_per_frame`.
    :param frame: native frame index.
    :return: ``[H', W']`` float32, or ``None`` where the system has not decided yet.
    :raises IndexError: if the frame is outside the video.
    """
    i = int(assignment.indice[frame])
    if i < 0:
        return None
    map_ = mappe[i]
    plane = temporal_position(finestre[i], frame, map_.grid[0])
    return reduce_over_frames(map_, plane)
