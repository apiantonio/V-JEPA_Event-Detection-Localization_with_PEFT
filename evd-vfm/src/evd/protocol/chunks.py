"""Chunk grid of the fully supervised regime.

This module is the rule and the grid is its output: it is recomputed, never sealed.
Runs record the count and the measured positive rate (:func:`riepilogo`), a
verifiable signature of the grid.

Three protocol decisions live here:

* positive by any intersection: a chunk is positive if it overlaps
  an annotated interval by at least one frame;
* video-duration weight ``1/n_chunk(video)``: otherwise a 9-hour normal
  video alone would be 7.8% of the training set;
* overlap weight ``clip(overlap/duration, floor, 1)`` on positives only;
  the floor is not zero, which would reintroduce a minimum-overlap threshold.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from evd.annotations.schema import TemporalAnnotation, window_overlap
from evd.data.windows import CausalWindowSampler
from evd.types import PreprocessSpec, VideoMeta, WindowSpec

#: Floor of the overlap weight. The value in force comes from
#: ``configs/protocol/*.yaml::sampling.overlap_weight_floor``; callers that omit it are
#: toys or tests.
OVERLAP_FLOOR = 0.1


@dataclass(frozen=True)
class ChunkItem:
    """A training-set window: geometry, label and weight.

    :param window: causal window, in native frames.
    :param label: ``1`` if it intersects an annotated interval, ``0`` otherwise.
    :param class_idx: class of the intersected interval, or ``None`` on negatives. Not a
        model target (ADR-031 point 4): used to break binary errors down by class.
    :param weight: sampling weight ``w_video · w_overlap``, unnormalized. It lives in the
        data so that PEFT families share the same sampling (invariant 5).
    :param overlap_frames: frames shared with annotated intervals, which makes ``weight``
        verifiable.
    """

    window: WindowSpec
    label: int
    class_idx: int | None
    weight: float
    overlap_frames: int

    @property
    def video_id(self) -> str:
        return self.window.video_id

    @property
    def end_frame(self) -> int:
        return self.window.end_frame

    @property
    def span(self) -> tuple[int, int]:
        """:return: the nominal ``(first, last)`` frame interval of the chunk.

        On the ``fixed`` grid it equals ``window.frame_span`` (ADR-047). It is the interval the
        label was computed on; use it instead of rebuilding it from ``end_frame``, which is
        wrong on a truncated window.
        """
        return self.window.frame_span


#: The overlap rule lives in :mod:`evd.annotations.schema`, imported by both
#: ``protocol`` and ``eval``, so training and evaluated chunks are labelled by the same
#: function.
_overlap = window_overlap


def _check_floor(overlap_floor: float) -> None:
    """:raises ValueError: if ``overlap_floor`` is not in ``(0, 1]``."""
    if not 0.0 < overlap_floor <= 1.0:
        raise ValueError(
            f"overlap_floor must be in (0, 1], not {overlap_floor}. Zero would cancel "
            f"low-overlap positives, reintroducing the overlap threshold rejected by "
            f"ADR-031 point 3"
        )


def chunk_at(
    meta: VideoMeta,
    ann: TemporalAnnotation,
    sampler: CausalWindowSampler,
    span: tuple[int, int],
    *,
    weight_video: float,
    overlap_floor: float = OVERLAP_FLOOR,
) -> ChunkItem:
    """Return the chunk on interval ``span``, with label and weight.

    The single implementation of the positivity rule, shared by the static grid
    (:func:`chunks_for_video`) and dynamic sampling (docs/04 §3.5). The overlap weight is
    divided by the duration of this window, not the nominal one, so a truncated tail fully
    covered by an event weighs 1.

    :param meta: video metadata.
    :param ann: video annotation; empty ``intervals`` means a normal video.
    :param sampler: grid geometry.
    :param span: ``(first, last)`` native frame of the window, both included, as returned
        by :meth:`~evd.data.windows.CausalWindowSampler.spans_for_video`.
    :param weight_video: ``1 / n_chunk(video)``, counted on the static grid even when the
        endpoint is drawn at random.
    :param overlap_floor: floor of the overlap weight.
    :return: the chunk.
    :raises ValueError: if ``overlap_floor`` is not in ``(0, 1]`` or the span is not valid
        for the video.
    """
    _check_floor(overlap_floor)
    window = sampler.window_for_span(meta, span)
    start, end = int(span[0]), int(span[1])
    chunk_duration = float(end - start + 1)
    covered_items, class_name = _overlap(ann, start, end)
    if covered_items > 0:
        weight = weight_video * max(overlap_floor, min(1.0, covered_items / chunk_duration))
    else:
        weight = weight_video
    return ChunkItem(
        window=window,
        label=int(covered_items > 0),
        class_idx=class_name,
        weight=weight,
        overlap_frames=covered_items,
    )


def chunks_for_video(
    meta: VideoMeta,
    ann: TemporalAnnotation,
    sampler: CausalWindowSampler,
    *,
    overlap_floor: float = OVERLAP_FLOOR,
) -> list[ChunkItem]:
    """Return all chunks of a video, with labels and weights.

    No window is discarded (ADR-031 point 2): negatives inside an anomalous video are
    valuable supervision.

    :param meta: file metadata (fps and native frame count).
    :param ann: video annotation; a normal video has empty ``intervals``.
    :param sampler: grid geometry (span and hop).
    :param overlap_floor: floor of the overlap weight; ``0`` is rejected.
    :return: chunks in temporal order. Empty only on the ``endpoint`` grid for a video
        shorter than a window; on the ``fixed`` grid the tail rule yields at least one
        (ADR-047).
    :raises ValueError: if ``overlap_floor`` is not in ``(0, 1]``.
    """
    _check_floor(overlap_floor)
    spans = sampler.spans_for_video(meta)
    if not spans:
        return []
    video_weight = 1.0 / len(spans)
    return [
        chunk_at(
            meta,
            ann,
            sampler,
            span,
            weight_video=video_weight,
            overlap_floor=overlap_floor,
        )
        for span in spans
    ]


def enumerate_chunks(
    video_ids: Sequence[str],
    metas: Mapping[str, VideoMeta],
    annotations: Mapping[str, TemporalAnnotation],
    sampler: CausalWindowSampler,
    *,
    overlap_floor: float = OVERLAP_FLOOR,
) -> list[ChunkItem]:
    """Return the full grid of a partition.

    :param video_ids: partition videos, in split order.
    :param metas: per-video metadata; must cover ``video_ids``.
    :param annotations: per-video annotations; a missing video is treated as normal
        (FS-UCF-Crime lists 950 videos without events).
    :param sampler: grid geometry.
    :param overlap_floor: see :func:`chunks_for_video`.
    :return: the chunks of all videos, in video order.
    :raises KeyError: if the metadata of a video is missing.
    """
    outside: list[ChunkItem] = []
    for video_id in video_ids:
        if video_id not in metas:
            raise KeyError(
                f"{video_id}: no metadata in the inventory. Without fps and frame count "
                f"the grid is undefined, and guessing them would silently give wrong windows"
            )
        ann = annotations.get(video_id, TemporalAnnotation(video_id=video_id))
        outside.extend(chunks_for_video(metas[video_id], ann, sampler, overlap_floor=overlap_floor))
    return outside


@dataclass(frozen=True)
class ChunkSummary:
    """Verifiable signature of a grid, written to ``meta.json``.

    :param n_chunk: number of chunks.
    :param n_positivi: number of positives.
    :param n_video: number of videos producing at least one window.
    :param tasso_positivi_uniforme: fraction of positives counting chunks.
    :param tasso_positivi_pesato: fraction of the sampling mass on positives, i.e. the rate
        the model actually sees; measured, not assumed (invariant 6).
    """

    n_chunk: int
    n_positivi: int
    n_video: int
    tasso_positivi_uniforme: float
    tasso_positivi_pesato: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n_chunk": self.n_chunk,
            "n_positivi": self.n_positivi,
            "n_video": self.n_video,
            "tasso_positivi_uniforme": self.tasso_positivi_uniforme,
            "tasso_positivi_pesato": self.tasso_positivi_pesato,
        }


def riepilogo(items: Iterable[ChunkItem]) -> ChunkSummary:
    """:return: counts and positive rates of a grid.
    :raises ValueError: if the grid is empty.
    """
    item_list = list(items)
    if not item_list:
        raise ValueError(
            "empty grid: no chunk to summarize. Check the split and the geometry "
            "(a video shorter than a window produces none)"
        )
    positives = [c for c in item_list if c.label == 1]
    mass = sum(c.weight for c in item_list)
    positive_mass = sum(c.weight for c in positives)
    return ChunkSummary(
        n_chunk=len(item_list),
        n_positivi=len(positives),
        n_video=len({c.video_id for c in item_list}),
        tasso_positivi_uniforme=len(positives) / len(item_list),
        tasso_positivi_pesato=(positive_mass / mass) if mass > 0 else 0.0,
    )


def sampler_for_protocol(
    spec: PreprocessSpec | None,
    protocol: Mapping[str, Any],
    *,
    validazione: bool = False,
) -> CausalWindowSampler:
    """Return the grid geometry for the partition being built.

    The only place where the ``protocol`` group becomes a sampler (ADR-052): copies of this
    translation used to diverge and silently build the ``edp_v1`` grid on ``edp_v2``.
    Validation may use a different hop (contiguous windows, ``val_hop_s``; ADR-047,
    protocol §3.1); when absent it equals ``hop_s``. With ``spec=None`` (diagnostic and
    counting scripts) ``num_frames`` and ``target_fps`` are read from the protocol;
    otherwise the backbone spec is used and :meth:`ExperimentRunner._check_geometry`
    checks that the two match.

    :param spec: backbone preprocessing, or ``None`` to read ``num_frames`` and
        ``target_fps`` from the protocol.
    :param protocol: resolved ``protocol`` group.
    :param validazione: whether the sampler is for the validation partition.
    :return: the sampler of the requested partition.
    :raises ValueError: if ``window_grid`` names an unknown grid.
    :raises KeyError: if ``temporal.hop_s`` is missing (no default) or, with ``spec=None``,
        ``temporal.num_frames`` or ``temporal.target_fps``.
    """
    temporal = protocol.get("temporal") or {}
    grid = str(temporal.get("window_grid", "endpoint"))
    if grid not in ("endpoint", "fixed"):
        raise ValueError(
            f"protocol.temporal.window_grid = {grid!r}: the grids are 'endpoint' "
            f"(ADR-024) and 'fixed' (ADR-047, the laboratory protocol)"
        )
    duration = temporal.get("window_s")
    hop = temporal.get("val_hop_s") if validazione else temporal.get("hop_s")
    if hop is None:
        hop = temporal["hop_s"]
    return CausalWindowSampler(
        num_frames=int(spec.num_frames if spec is not None else temporal["num_frames"]),
        target_fps=float(spec.fps if spec is not None else temporal["target_fps"]),
        hop_s=float(hop),
        grid=grid,  # type: ignore[arg-type]
        window_s=None if duration is None else float(duration),
    )


def balance_classes(chunks: Sequence[ChunkItem]) -> list[ChunkItem]:
    """Give every class the same share of the sampling mass (``class_balance: equal``).

    Each weight ``w_video · w_overlap`` is multiplied by ``(1 / K) / M_c``, where ``M_c`` is
    the total weight of the chunk's class (``Normal``, i.e. ``class_idx is None``, counts as
    one class) and ``K`` the number of classes present. Weights stay proportional inside a
    class, so the per-video and overlap balance of invariant 6 is kept within each class.
    A declared deviation from invariant 6 for the class-routing heads
    (``docs/varie/piano_localizzazione_per_classe_20261021.md`` §5.5).

    :param chunks: the training grid, with protocol weights.
    :return: the same chunks, in the same order, with class-balanced weights summing to one.
    :raises ValueError: if the grid is empty or a class has no weight.
    """
    from collections import defaultdict
    from dataclasses import replace

    if not chunks:
        raise ValueError("class balance of an empty grid")
    mass: dict[int | None, float] = defaultdict(float)
    for chunk in chunks:
        mass[chunk.class_idx] += chunk.weight
    if any(m <= 0 for m in mass.values()):
        raise ValueError("class balance: a class has no sampling weight")
    k = len(mass)
    return [replace(c, weight=c.weight / (k * mass[c.class_idx])) for c in chunks]
