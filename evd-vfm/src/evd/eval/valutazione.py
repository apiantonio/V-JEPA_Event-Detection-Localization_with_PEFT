"""Metrics of an evaluated partition, computed in one place.

Shared by :class:`~evd.exp.runner.ExperimentRunner` and the cross-dataset evaluation of
:mod:`evd.exp.esterno`, so the two tables cannot diverge. The function neither estimates
a threshold nor chooses the partition: the caller passes the threshold with its
provenance. Run counts, the ``threshold`` block, cache statistics and figures stay with
the caller.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from evd.annotations.schema import TemporalAnnotation
from evd.eval import metrics as M
from evd.eval import strata as S
from evd.types import EventClass, WindowScores

__all__ = ["metriche_di_valutazione", "threshold_sensitivity"]


def threshold_sensitivity(
    chunk: Mapping[str, Sequence[WindowScores]],
    gt: Mapping[str, TemporalAnnotation],
    negative_scores: Sequence[float],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Compute P/R/F1 at other quantiles as a secondary analysis (not protocol).

    Shows how much a conclusion depends on the operating point; it only recounts scores
    that already exist.

    :param chunk: chunk scores of the evaluated videos.
    :param gt: annotations of the evaluated videos.
    :param negative_scores: scores the quantiles are read from (validation negatives, not
        those of the evaluated partition).
    :param protocol: protocol group, providing ``threshold``.
    :return: one row per quantile; empty if the protocol declares none or there are no
        negative scores (cross-dataset evaluation).
    """
    threshold = protocol.get("threshold") or {}
    quantiles = threshold.get("sensitivity_quantiles") or ()
    if not quantiles or not negative_scores:
        return []
    # Sensitivity is a quantile analysis by construction: labelling it `val_youden` would be
    # wrong.
    source = "val" if str(threshold.get("source", "support")) == "val" else "support"
    rows: list[dict[str, Any]] = []
    for q in quantiles:
        value, provenance = M.threshold_from_scores(
            negative_scores,
            quantile=float(q),
            source=source,  # type: ignore[arg-type]
        )
        rows.append(
            {
                "quantile": float(q),
                "threshold": value,
                "chunk_prf": M.dataset_chunk_prf(chunk, gt, value, source=provenance).as_dict(),
            }
        )
    return rows


def metriche_di_valutazione(
    *,
    chunk: Mapping[str, Sequence[WindowScores]],
    gt: Mapping[str, TemporalAnnotation],
    soglia: float,
    sorgente: Any,
    classi: Sequence[EventClass],
    normal_idx: int,
    fps: Mapping[str, float],
    num_frames: Mapping[str, int],
    protocol: Mapping[str, Any],
    requests: Mapping[str, Any] | None = None,
    negative_scores: Sequence[float] = (),
) -> dict[str, Any]:
    """Compute all metrics of docs/04 §4.3 on an evaluated partition.

    :param chunk: chunk scores per video, full grid with normal videos included (AUC_a
        excludes them by itself).
    :param gt: annotations of the evaluated videos; a video without events has empty
        ``intervals``.
    :param soglia: operating point, shared by chunk, video and event level.
    :param sorgente: threshold provenance carried by every P/R/F1 triple (``val_youden``
        for a matrix run, ``fissa`` for a threshold transferred from another dataset).
    :param classi: dataset classes, ``Normal`` included.
    :param normal_idx: index of ``Normal``, skipped by per-class rows.
    :param fps: native fps per video, to express temporal metrics in seconds.
    :param num_frames: native frames per video, for the coverage axes.
    :param protocol: protocol group, providing ``video_policy`` and sensitivity quantiles.
    :param requests: ``eval.metrics`` config block enabling non-protocol metrics:
        ``video_auc``/``video_ap`` (``edp_v1`` only), ``event_level`` (ADR-048) and
        ``strata`` (ADR-060).
    :param negative_scores: validation negative scores for the threshold sensitivity
        analysis; empty disables it.
    :return: the dictionary written into ``metrics.json``.
    """
    requests = requests or {}
    anomalous = M.restrict_to_anomalous(chunk, gt)
    classi = tuple(classi)
    video_scores = M.aggregate_video_scores(chunk, policy=str(protocol["video_policy"]))
    video_labels = {v: int(not gt[v].is_normal) for v in chunk}

    # Frame-level metrics are outside the protocol (ADR-031) and are not reported here.
    metrics: dict[str, Any] = {
        "chunk_auc": M.dataset_chunk_auc(chunk, gt),
        "chunk_ap": M.dataset_chunk_ap(chunk, gt),
        # AUC_a and AP_a: the same metrics restricted to anomalous videos (laboratory protocol
        # §4.1).
        "chunk_auc_anomalous": M.dataset_chunk_auc(anomalous, gt),
        "chunk_ap_anomalous": M.dataset_chunk_ap(anomalous, gt),
        "n_video_anomalous": len(anomalous),
        "chunk_prf": M.dataset_chunk_prf(chunk, gt, soglia, source=sorgente).as_dict(),
        "per_class_chunk": {
            name: row.as_dict()
            for name, row in M.per_class_chunk(chunk, gt, classi, normal_idx=normal_idx).items()
        },
        # Both latency rules are written (ADR-061): the historical one keeps existing runs
        # comparable, the intersection one matches `event_prf` and is the one to report.
        "detection_latency": M.detection_latency(chunk, gt, fps, threshold=soglia).as_dict(),
        "detection_latency_intersezione": M.detection_latency(
            chunk, gt, fps, threshold=soglia, rule="intersezione"
        ).as_dict(),
        # Video level: thresholded metrics only, with the asymmetric rule and the chunk threshold
        # (laboratory protocol §4.2).
        "video_prf": M.video_decision_prf(chunk, gt, soglia, source=sorgente).as_dict(),
        "threshold_sensitivity": threshold_sensitivity(chunk, gt, negative_scores, protocol),
    }
    # Video-level AUC/AP are no longer protocol (ADR-047 §4.2); computed only on request, to
    # reread `edp_v1` runs.
    if bool(requests.get("video_auc", True)):
        metrics["video_auc"] = M.video_auc(video_scores, video_labels)
    if bool(requests.get("video_ap", True)):
        metrics["video_ap"] = M.video_ap(video_scores, video_labels)
    # Event level (ADR-048): opt-in, since the definition is ours and not the laboratory's.
    event_level = bool(requests.get("event_level", False))
    if event_level:
        metrics["event_prf"] = M.event_prf(chunk, gt, soglia, source=sorgente).as_dict()
    # Stratified analysis (ADR-060): diagnostics, not protocol. Event axes follow the
    # `event_prf` gate because they share its definition.
    if bool(requests.get("strata", True)):
        metrics["strata"] = S.analyse(
            chunk,
            gt,
            fps,
            num_frames,
            classes=classi,
            normal_idx=normal_idx,
            threshold=soglia,
            source=sorgente,
            include_event_axes=event_level,
        )
    # Per-class P/R/F1 at the operating point, with the same negatives and threshold as
    # `per_class_chunk`.
    metrics["per_class_chunk_prf"] = {
        name: row.as_dict()
        for name, row in M.per_class_chunk_prf(
            chunk, gt, classi, soglia, source=sorgente, normal_idx=normal_idx
        ).items()
    }
    return metrics
