"""Stratified analysis: in which situations the detector performs better or worse.

:mod:`evd.eval.metrics` says how good the system is; this module says where. The axes
make measurable the difficulties named in the PdD (intra-class heterogeneity, rarity
and temporal dynamics of events). Everything is computed from the scores already
written to ``scores.jsonl``, so a closed run is re-analysed in seconds with
``scripts/analisi_strati.py``.

======= =============== ===================================================
level   axis            question
======= =============== ===================================================
chunk   coverage        does the model fail on border chunks, barely touched
                        by the event?
event   class           which of the 13 classes are detected, and how late
event   duration        are short events missed? (bands from the ``edp_v2``
                        geometry, see :data:`DURATION_EDGES_S`)
event   anomaly share   rarity: is an event covering 2% of its video harder
                        than one covering 80%?
======= =============== ===================================================

Chunk-level class rows already exist in :func:`~evd.eval.metrics.per_class_chunk`;
this module reuses its primitive :func:`~evd.eval.metrics.breakdown_vs_negatives`, so
negatives are shared across strata. At event level, alarms do not belong to a stratum,
so false alarms are a shared denominator. Diagnostics, not protocol.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from evd.annotations.schema import TemporalAnnotation, window_overlap
from evd.eval.metrics import (
    LatencyRule,
    ThresholdSource,
    breakdown_vs_negatives,
    chunk_records,
    detection_latency,
    event_prf,
)
from evd.types import EventClass, WindowScores
from evd.utils.logging import get_logger

log = get_logger(__name__)

#: An annotated interval: ``(onset, offset, class_idx)``, both ends included.
Interval = tuple[int, int, int]

#: Key of a chunk stratum: from a :func:`~evd.eval.metrics.chunk_records` row to its
#: stratum label, or ``None`` if the row belongs to none.
ChunkKey = Callable[[Mapping[str, Any]], str | None]

#: Key of an event stratum, from ``(video_id, interval)``.
EventKey = Callable[[str, Interval], str | None]

#: Event duration bands in seconds, from the ``edp_v2`` geometry (hop 4 s, span 7.75 s):
#: under 4 s an event is shorter than the grid step; 4-8 s is about one window; up to
#: 32 s a handful of windows. The edges are stored with the results (:func:`analyse`).
DURATION_EDGES_S: tuple[float, ...] = (4.0, 8.0, 32.0)

#: Coverage bands, as the fraction of the window covered by an annotated interval
#: (neutral quartiles).
COVERAGE_EDGES: tuple[float, ...] = (0.25, 0.5, 0.75)

#: Anomaly-share bands of a video: fraction of its frames covered by at least one
#: interval. 5% separates rare events from the rest.
SHARE_EDGES: tuple[float, ...] = (0.05, 0.20, 0.50)


# --------------------------------------------------------------------------- #
# Bands
# --------------------------------------------------------------------------- #


def _band(valore: float, edges: Sequence[float]) -> int:
    """:return: index of the band of ``valore``, from 0 to ``len(edges)``; a value on an edge
    falls in the lower band.
    """
    for i, border in enumerate(edges):
        if valore <= border:
            return i
    return len(edges)


def band_labels(edges: Sequence[float], *, unit: str = "", scale: float = 1.0) -> tuple[str, ...]:
    """Return the labels of the bands defined by ``edges``, in increasing order.

    All ``len(edges) + 1`` labels are generated, including empty bands. Labels are ASCII
    on purpose: they become ``metrics.json`` keys and are printed on cp1252 consoles.

    :param edges: increasing edges.
    :param unit: suffix appended to the numbers (``" s"``, ``"%"``).
    :param scale: factor applied to the edges before formatting (``100`` for percentages).
    :return: the labels, e.g. ``("<=4 s", "4-8 s", "8-32 s", ">32 s")``.
    :raises ValueError: if ``edges`` is empty or not strictly increasing.
    """
    if not edges:
        raise ValueError("at least two bands, i.e. at least one edge, are required")
    if any(b <= a for a, b in zip(edges, edges[1:], strict=False)):
        raise ValueError(f"edges must be increasing: {list(edges)}")
    n = [f"{b * scale:g}" for b in edges]
    return (
        f"<={n[0]}{unit}",
        *(f"{a}-{b}{unit}" for a, b in zip(n, n[1:], strict=False)),
        f">{n[-1]}{unit}",
    )


# --------------------------------------------------------------------------- #
# Stratum rows
# --------------------------------------------------------------------------- #


def _harmonic_mean(precision: float, recall: float) -> float:
    """:return: harmonic mean of the two, ``nan`` if one is missing, 0 if they sum to zero."""
    if precision != precision or recall != recall:
        return float("nan")
    total_sum = precision + recall
    return float(2.0 * precision * recall / total_sum) if total_sum > 0 else 0.0


@dataclass(frozen=True)
class Stratum:
    """Behaviour of the detector on the positive chunks of a stratum.

    :param n_pos: positive chunks in the stratum; read it first (small strata are noisy).
    :param auc: AUC of the stratum chunks against all negatives.
    :param ap: AP on the same chunks.
    :param recall: fraction of the stratum chunks above the threshold.
    :param precision: ``TP_stratum / (TP_stratum + FP_total)``: false positives are shared
        by every stratum, as negatives are for ``auc`` and ``ap``. It makes strata
        comparable with each other and does not sum to the run precision.
    :param f1: harmonic mean of ``precision`` and ``recall``.
    :param score_mediano: median score of the stratum, to compare with the negatives'
        median (``riferimento_negativi`` in :func:`analyse`).
    """

    n_pos: int
    auc: float
    ap: float
    recall: float
    precision: float
    f1: float
    score_mediano: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class EventStratum:
    """How many events of a stratum the system finds, and how late.

    :param n_eventi: annotated events in the stratum (recall denominator).
    :param n_rilevati: events intersected by at least one above-threshold window (ADR-048).
    :param recall: ``n_rilevati / n_eventi``, ``nan`` on an empty stratum.
    :param n_allarmi_veri: alarms intersecting at least one event of this stratum (an
        alarm spanning two strata is true in both).
    :param precision: ``n_allarmi_veri / (n_allarmi_veri + FP)``, where ``FP`` are alarms
        intersecting no annotated event at all (shared denominator). It is not the
        aggregated ``event_prf`` precision.
    :param f1: harmonic mean of ``precision`` and ``recall``.
    :param latenza_mediana_s: median delay over measurable events, with the
        ``intersezione`` rule (ADR-061).
    :param n_senza_latenza_misurabile: events intersected by no above-threshold window;
        with the ADR-061 rule it equals ``n_eventi - n_rilevati`` (checked by
        :func:`event_strata`). ``strata.json`` files older than 4 September 2026 used the
        historical rule and lack the ``regola_latenza`` field.
    """

    n_eventi: int
    n_rilevati: int
    recall: float
    n_allarmi_veri: int
    precision: float
    f1: float
    latenza_mediana_s: float
    n_senza_latenza_misurabile: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Stratifiers
# --------------------------------------------------------------------------- #


def chunk_strata(
    rows: Sequence[Mapping[str, Any]],
    key: ChunkKey,
    *,
    labels: Sequence[str],
    threshold: float,
) -> dict[str, Stratum]:
    """Split the positive chunks along an axis, against shared negatives.

    Consumes :func:`~evd.eval.metrics.chunk_records` rows, which share the
    ``scores.jsonl`` schema, so ongoing evaluations and closed runs use the same path.

    :param rows: rows of the evaluated partition; rows labelled ``-1`` (unknown) are
        dropped.
    :param key: maps a row to its stratum label; called on positives only.
    :param labels: expected labels, in presentation order; empty strata are kept with
        ``n_pos`` 0.
    :param threshold: run threshold, for the ``recall`` of each stratum.
    :return: one row per label, in the order of ``labels``.
    :raises KeyError: if ``key`` returns a label not in ``labels``.
    """
    row_list = [r for r in rows if int(r["label"]) >= 0]
    nan = float("nan")
    empty = Stratum(n_pos=0, auc=nan, ap=nan, recall=nan, precision=nan, f1=nan, score_mediano=nan)
    if not row_list:
        return dict.fromkeys(labels, empty)
    s = np.asarray([float(r["score"]) for r in row_list], dtype=np.float64)
    y = np.asarray([int(r["label"]) for r in row_list], dtype=np.uint8)
    # False positives are the same for every band: a negative chunk has no coverage, so it
    # belongs to no stratum.
    shared_fp = int(((y == 0) & (s >= threshold)).sum())
    expected = set(labels)
    membership: list[str | None] = []
    for row, binary_label in zip(row_list, y, strict=True):
        stratum = key(row) if binary_label == 1 else None
        if stratum is not None and stratum not in expected:
            raise KeyError(
                f"the axis produced stratum {stratum!r}, which is not among the declared "
                f"labels {sorted(expected)}: it would be invisible in the result"
            )
        membership.append(stratum)
    outside: dict[str, Stratum] = {}
    for label in labels:
        mask: npt.NDArray[np.bool_] = np.asarray([a == label for a in membership], dtype=bool)
        auc_row = breakdown_vs_negatives(s, y, mask, name=f"chunk_strata[{label}]")
        inside = s[mask]
        tp = int((inside >= threshold).sum()) if inside.size else 0
        recall = float((inside >= threshold).mean()) if inside.size else float("nan")
        precision_value = float(tp / (tp + shared_fp)) if (tp + shared_fp) else float("nan")
        outside[label] = Stratum(
            n_pos=auc_row.n_pos,
            auc=auc_row.auc,
            ap=auc_row.ap,
            recall=recall,
            precision=precision_value,
            f1=_harmonic_mean(precision_value, recall),
            score_mediano=float(np.median(inside)) if inside.size else float("nan"),
        )
    return outside


def event_strata(
    scores: Mapping[str, Sequence[WindowScores]],
    gt: Mapping[str, TemporalAnnotation],
    fps: Mapping[str, float],
    key: EventKey,
    *,
    labels: Sequence[str],
    threshold: float,
    source: ThresholdSource,
    latency_rule: LatencyRule = "intersezione",
) -> dict[str, EventStratum]:
    """Return recall and latency of the events of each stratum.

    Annotated intervals are partitioned and :func:`~evd.eval.metrics.event_prf` and
    :func:`~evd.eval.metrics.detection_latency` are called on each subset, so strata cannot
    contradict the aggregated number. Precision uses false alarms counted once on the full
    ``gt`` as shared denominator: bands become comparable but do not sum to the aggregated
    precision.

    :param scores: chunk scores of all evaluated videos, full grid, as ``event_prf``
        requires.
    :param gt: annotations of the evaluated videos.
    :param fps: native fps per video, for latency in seconds.
    :param key: maps ``(video_id, interval)`` to a stratum label, or ``None`` to leave the
        event out.
    :param labels: expected labels, in presentation order.
    :param threshold: run threshold.
    :param source: threshold provenance, required by ``event_prf``.
    :param latency_rule: detection predicate of the latency; the default
        ``intersezione`` matches ``event_prf``.
    :return: one row per label, in the order of ``labels``.
    :raises KeyError: if ``key`` produces a label outside ``labels``, or the fps of a video
        with events is missing.
    """
    expected = set(labels)
    # False alarms are counted once on the full `gt`.
    aggregated = event_prf(scores, gt, threshold, source=source)
    shared_fp = aggregated.n_allarmi - aggregated.n_allarmi_veri
    groups: dict[str, dict[str, list[Interval]]] = {et: {} for et in labels}
    for video_id, ann in gt.items():
        for interval in ann.intervals:
            stratum = key(video_id, interval)
            if stratum is None:
                continue
            if stratum not in expected:
                raise KeyError(
                    f"the axis produced stratum {stratum!r}, which is not among the declared "
                    f"labels {sorted(expected)}: it would be invisible in the result"
                )
            groups[stratum].setdefault(video_id, []).append(interval)
    outside: dict[str, EventStratum] = {}
    for label in labels:
        below = {
            video_id: TemporalAnnotation(video_id=video_id, intervals=tuple(intervals))
            for video_id, intervals in groups[label].items()
        }
        if not below:
            outside[label] = EventStratum(
                n_eventi=0,
                n_rilevati=0,
                recall=float("nan"),
                n_allarmi_veri=0,
                precision=float("nan"),
                f1=float("nan"),
                latenza_mediana_s=float("nan"),
                n_senza_latenza_misurabile=0,
            )
            continue
        ev = event_prf(scores, below, threshold, source=source)
        lat = detection_latency(scores, below, fps, threshold=threshold, rule=latency_rule)
        # With `intersezione` the two predicates coincide, so the counts must match. Log instead
        # of raising: diagnostics must not fail a run after hours of training.
        missed = ev.n_eventi - ev.n_eventi_rilevati
        if latency_rule == "intersezione" and lat.n_senza_latenza_misurabile != missed:
            log.error(
                "stratum %r: n_senza_latenza_misurabile=%d but event_prf reports %d missed; "
                "with the `intersezione` rule the two predicates are the same and must match",
                label,
                lat.n_senza_latenza_misurabile,
                missed,
            )
        true_items = ev.n_allarmi_veri
        precision_value = (
            float(true_items / (true_items + shared_fp))
            if (true_items + shared_fp)
            else float("nan")
        )
        outside[label] = EventStratum(
            n_eventi=ev.n_eventi,
            n_rilevati=ev.n_eventi_rilevati,
            recall=ev.recall,
            n_allarmi_veri=true_items,
            precision=precision_value,
            f1=_harmonic_mean(precision_value, ev.recall),
            latenza_mediana_s=lat.mediana_s,
            n_senza_latenza_misurabile=lat.n_senza_latenza_misurabile,
        )
    return outside


# --------------------------------------------------------------------------- #
# Axes
# --------------------------------------------------------------------------- #


def by_coverage(
    gt: Mapping[str, TemporalAnnotation], edges: Sequence[float] = COVERAGE_EDGES
) -> tuple[ChunkKey, tuple[str, ...]]:
    """Chunk axis: how much of the window the event covers.

    The fraction is ``covered frames / window frames`` from
    :func:`~evd.annotations.schema.window_overlap`, the repository's only overlap rule.
    The denominator is the length of that window, not the nominal one, so a fully covered
    truncated tail window is not demoted.

    :param gt: annotations of the evaluated videos.
    :param edges: band edges, as fractions.
    :return: ``(key, labels)`` for :func:`chunk_strata`.
    """
    labels = band_labels(edges, unit="%", scale=100.0)

    def key(row: Mapping[str, Any]) -> str | None:
        ann = gt.get(str(row["video_id"]))
        if ann is None:
            return None
        start, end = int(row["start_frame"]), int(row["end_frame"])
        covered_items, _ = window_overlap(ann, start, end)
        if covered_items <= 0:
            return None
        return labels[_band(covered_items / (end - start + 1), edges)]

    return key, labels


def by_event_class(classes: Sequence[EventClass]) -> tuple[EventKey, tuple[str, ...]]:
    """Event axis: the annotated class of the interval (used only for grouping).

    :param classes: anomaly classes of the dataset, without ``Normal`` (no interval
        carries it).
    :return: ``(key, labels)`` for :func:`event_strata`.
    """
    per_index = {cls.idx: cls.name for cls in classes}
    labels = tuple(cls.name for cls in classes)

    def key(video_id: str, interval: Interval) -> str | None:  # noqa: ARG001
        return per_index.get(interval[2])

    return key, labels


def by_event_duration(
    fps: Mapping[str, float], edges: Sequence[float] = DURATION_EDGES_S
) -> tuple[EventKey, tuple[str, ...]]:
    """Event axis: duration of the annotated interval, in seconds.

    :param fps: native fps per video (bands are defined in seconds).
    :param edges: band edges, in seconds.
    :return: ``(key, labels)`` for :func:`event_strata`.
    :raises KeyError: if the fps of an annotated video is missing.
    """
    labels = band_labels(edges, unit=" s")

    def key(video_id: str, interval: Interval) -> str | None:
        if video_id not in fps:
            raise KeyError(
                f"{video_id}: missing fps, the event duration cannot be converted into seconds"
            )
        onset, offset, _ = interval
        return labels[_band((offset - onset + 1) / fps[video_id], edges)]

    return key, labels


def anomaly_share(ann: TemporalAnnotation, num_frames: int) -> float:
    """Return the fraction of a video's frames covered by at least one interval.

    Intervals are merged, not summed, so overlapping annotations are not counted twice.

    :param ann: video annotation (inclusive intervals, native frames).
    :param num_frames: native frames of the video.
    :return: the share, in ``[0, 1]``.
    :raises ValueError: if ``num_frames`` is not positive.
    """
    if num_frames <= 0:
        raise ValueError(f"{ann.video_id}: num_frames must be > 0, not {num_frames}")
    covered_items = 0
    current_end = -1
    for onset, offset in sorted((s, e) for s, e, _ in ann.intervals):
        start = max(onset, current_end + 1)
        if offset >= start:
            covered_items += offset - start + 1
            current_end = offset
    return min(covered_items / num_frames, 1.0)


def by_video_anomaly_share(
    gt: Mapping[str, TemporalAnnotation],
    num_frames: Mapping[str, int],
    edges: Sequence[float] = SHARE_EDGES,
) -> tuple[EventKey, tuple[str, ...]]:
    """Event axis: how rare the event is within its video.

    The share is a property of the video, so events of the same video fall in the same
    band.

    :param gt: annotations of the evaluated videos.
    :param num_frames: native frames per video.
    :param edges: band edges, as fractions.
    :return: ``(key, labels)`` for :func:`event_strata`.
    :raises KeyError: if ``num_frames`` of an annotated video is missing.
    """
    labels = band_labels(edges, unit="%", scale=100.0)
    quotas: dict[str, float] = {}
    for video_id, ann in gt.items():
        if not ann.intervals:
            continue
        if video_id not in num_frames:
            raise KeyError(f"{video_id}: missing num_frames, the anomaly share cannot be computed")
        quotas[video_id] = anomaly_share(ann, num_frames[video_id])

    def key(video_id: str, interval: Interval) -> str | None:  # noqa: ARG001
        quota = quotas.get(video_id)
        return None if quota is None else labels[_band(quota, edges)]

    return key, labels


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #


def analyse(
    scores: Mapping[str, Sequence[WindowScores]],
    gt: Mapping[str, TemporalAnnotation],
    fps: Mapping[str, float],
    num_frames: Mapping[str, int],
    *,
    classes: Sequence[EventClass],
    normal_idx: int,
    threshold: float,
    source: ThresholdSource,
    include_event_axes: bool = True,
    latency_rule: LatencyRule = "intersezione",
    coverage_edges: Sequence[float] = COVERAGE_EDGES,
    duration_edges_s: Sequence[float] = DURATION_EDGES_S,
    share_edges: Sequence[float] = SHARE_EDGES,
) -> dict[str, Any]:
    """Compute the four axes in one block, for ``metrics.json``.

    :param scores: chunk scores of the evaluated partition, per video, full grid with
        normal videos included.
    :param gt: annotations of the evaluated videos.
    :param fps: native fps per video.
    :param num_frames: native frames per video.
    :param classes: dataset classes, ``Normal`` included.
    :param normal_idx: index of ``Normal``.
    :param threshold: run threshold, shared by every granularity.
    :param source: threshold provenance.
    :param include_event_axes: whether to compute the three event axes (same gate as
        ``event_prf``, since the event definition is ours); stored as ``livello_evento``.
    :param latency_rule: latency detection predicate (ADR-061), stored as
        ``regola_latenza``.
    :param coverage_edges: coverage band edges.
    :param duration_edges_s: duration band edges, in seconds.
    :param share_edges: anomaly-share band edges.
    :return: the ``strata`` block, including the band edges.
    """
    rows = chunk_records(scores, gt)
    anomalous = tuple(cls for cls in classes if cls.idx != normal_idx)
    negatives = np.asarray(
        [float(r["score"]) for r in rows if int(r["label"]) == 0], dtype=np.float64
    )

    coverage_key, coverage_labels = by_coverage(gt, coverage_edges)

    def events(chiave: EventKey, etichette: Sequence[str]) -> dict[str, dict[str, float | int]]:
        return {
            name: row.as_dict()
            for name, row in event_strata(
                scores,
                gt,
                fps,
                chiave,
                labels=etichette,
                threshold=threshold,
                source=source,
                latency_rule=latency_rule,
            ).items()
        }

    outside: dict[str, Any] = {
        "threshold": float(threshold),
        "source": source,
        # Reference for every `score_mediano`: tells a hard stratum from a model that scores low
        # everywhere.
        "riferimento_negativi": {
            "n": int(negatives.size),
            # Shared false positives: the common denominator of every band's `precision`.
            "n_sopra_soglia": int((negatives >= threshold).sum()) if negatives.size else 0,
            "score_mediano": float(np.median(negatives)) if negatives.size else float("nan"),
        },
        "chunk": {
            "copertura": {
                item_name: row_data.as_dict()
                for item_name, row_data in chunk_strata(
                    rows, coverage_key, labels=coverage_labels, threshold=threshold
                ).items()
            }
        },
        "livello_evento": bool(include_event_axes),
        "regola_latenza": latency_rule,
        "bordi": {
            "copertura": [float(b) for b in coverage_edges],
            "durata_s": [float(b) for b in duration_edges_s],
            "quota_anomala_video": [float(b) for b in share_edges],
        },
    }
    if include_event_axes:
        # Event axes are built here because `by_video_anomaly_share` needs `num_frames`, which
        # is only required when the event level is requested.
        outside["evento"] = {
            "classe": events(*by_event_class(anomalous)),
            "durata": events(*by_event_duration(fps, duration_edges_s)),
            "quota_anomala_video": events(*by_video_anomaly_share(gt, num_frames, share_edges)),
        }
    return outside
