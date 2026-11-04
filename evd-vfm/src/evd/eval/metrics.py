"""Evaluation metrics with explicit conventions for degenerate cases.

Three granularities are in use: chunk (the window scored by the model), video (an
aggregation of the chunk scores) and event (annotated intervals versus alarms, with
our own definition). Threshold-independent metrics (AUC, AP) sit next to fixed-threshold
metrics (precision, recall, F1).

Frame-level metrics are outside the evaluation protocol; ``frame_auc``, ``frame_ap``,
``dataset_frame_auc``, ``dataset_frame_ap``, ``frame_prf``, ``dataset_frame_prf``,
``per_class_ap`` and ``per_class_prf`` remain only to reread ``edp_v1`` runs.

Degenerate cases: without positives or negatives AUC/AP return ``nan`` with a warning
(never 0.5); a threshold above every score predicts no positive, giving recall 0 and
precision 0.0 by convention (``zero_division=0``), hence F1 0. Every function is
checked against scikit-learn, degenerate cases included.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from evd.annotations.schema import (
    TemporalAnnotation,
    frame_class_labels,
    frame_labels,
    window_label,
    window_overlap,
)
from evd.types import EventClass, FrameScores, WindowScores, WindowSpec
from evd.utils.logging import get_logger

log = get_logger(__name__)

#: Where the threshold comes from, and with which rule. ``"val_youden"`` is the current
#: one (the validation ROC point maximizing ``TPR - FPR``); ``"val"`` is the negative-score
#: quantile of ``edp_v1``; ``"support"`` and ``"val_episodico"`` belong to the few-shot
#: regime and stay readable for existing runs.
ThresholdSource = Literal["val_youden", "val", "fissa", "support", "val_episodico"]


def _pair(
    scores: FrameScores, gt: TemporalAnnotation, restrict_to_covered: bool
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8]]:
    y = frame_labels(gt, len(scores.scores))
    s = np.asarray(scores.scores, dtype=np.float64)
    if restrict_to_covered:
        # The first frames are not covered by any complete causal window: they are excluded from
        # every frame metric.
        mask = np.asarray(scores.covered, dtype=bool)
        return s[mask], y[mask]
    return s, y


def _guard(y: npt.NDArray[np.uint8], name: str) -> bool:
    if y.size == 0:
        log.warning("%s: no evaluable samples", name)
        return False
    pos = int(y.sum())
    if pos == 0 or pos == y.size:
        log.warning(
            "%s: degenerate ground truth (%d positives out of %d) -> nan", name, pos, y.size
        )
        return False
    return True


# Threshold-independent metrics


def _chunk_arrays(
    scores: Mapping[str, Sequence[WindowScores]],
    gt: Mapping[str, TemporalAnnotation],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8], npt.NDArray[np.int64]]:
    """Return score, binary label and class of every chunk, in video order.

    A chunk is positive if it overlaps an annotated interval by at least one frame.

    :param scores: chunk scores grouped by video.
    :param gt: annotations of the evaluated videos.
    :return: ``(scores, binary labels, classes)``; the class is ``-1`` on negatives.
    :raises ValueError: if a chunk is indexed under another video.
    """
    rows = chunk_records(scores, gt)
    return (
        np.asarray([r["score"] for r in rows], dtype=np.float64),
        np.asarray([r["label"] for r in rows], dtype=np.uint8),
        np.asarray([r["class"] for r in rows], dtype=np.int64),
    )


def score_record(
    window: WindowSpec,
    score: float,
    label: int,
    class_idx: int | None,
) -> dict[str, Any]:
    """Build one ``scores.jsonl`` row: window identity, score and label.

    The row schema has a single author, while labels come from two sources (evaluation
    uses :func:`chunk_records`, validation uses ``WindowItem.label``).

    :param window: the window (id, video and frame span).
    :param score: anomaly score.
    :param label: 1 positive, 0 negative, ``-1`` unknown.
    :param class_idx: dominant class, or ``None`` on negatives (written as ``-1``).
    :return: the row, without the ``split`` field added by the writer.
    """
    start, end = window.frame_span
    return {
        "video_id": window.video_id,
        "window_id": window.window_id,
        "start_frame": int(start),
        "end_frame": int(end),
        "score": float(score),
        "label": int(label),
        "class": -1 if class_idx is None else int(class_idx),
    }


def chunk_records(
    scores: Mapping[str, Sequence[WindowScores]],
    gt: Mapping[str, TemporalAnnotation],
) -> list[dict[str, Any]]:
    """Return one row per chunk with identity, score, label and class.

    This is the only function that labels chunks for evaluation, and :func:`_chunk_arrays`
    derives from it, so metrics, curves and the score file see the same labels. A chunk is
    positive if it overlaps an annotated interval by at least one frame.

    :param scores: chunk scores grouped by video.
    :param gt: annotations of the evaluated videos.
    :return: rows in video order and window order; ``class`` is ``-1`` on negatives.
    :raises ValueError: if a chunk is indexed under another video.
    """
    rows: list[dict[str, Any]] = []
    for video_id, chunks in scores.items():
        ann = gt[video_id]
        for item in chunks:
            if item.window.video_id != video_id:
                raise ValueError(f"chunk of {item.window.video_id} indexed under {video_id}")
            start, end = item.window.frame_span
            # A single pass: `window_overlap` returns the covered frames and the dominant class, the
            # same function used to label training chunks.
            covered_items, class_name = window_overlap(ann, start, end)
            rows.append(score_record(item.window, item.anomaly, int(covered_items > 0), class_name))
    return rows


def restrict_to_anomalous(
    scores: Mapping[str, Sequence[WindowScores]],
    gt: Mapping[str, TemporalAnnotation],
) -> dict[str, Sequence[WindowScores]]:
    """Keep only the videos with at least one annotated event.

    It makes ``AUC_a`` and ``AP_a`` a restriction of :func:`dataset_chunk_auc` and
    :func:`dataset_chunk_ap`: inside a video where something happens, can the model tell
    when?

    :param scores: chunk scores grouped by video.
    :param gt: annotations of the evaluated videos.
    :return: the sub-dictionary of anomalous videos, in the order of ``scores``.
    :raises KeyError: if a video of ``scores`` has no annotation.
    """
    return {v: c for v, c in scores.items() if not gt[v].is_normal}


def dataset_chunk_auc(
    scores: Mapping[str, Sequence[WindowScores]],
    gt: Mapping[str, TemporalAnnotation],
) -> float:
    """Return the primary AUC over the pooled chunks of the evaluation split.

    :param scores: chunk scores grouped by video.
    :param gt: annotations of the evaluated videos.
    :return: pooled AUC, or ``nan`` with degenerate ground truth.
    """
    s, y, _ = _chunk_arrays(scores, gt)
    if not _guard(y, "dataset_chunk_auc"):
        return float("nan")
    return float(roc_auc_score(y, s))


def dataset_chunk_ap(
    scores: Mapping[str, Sequence[WindowScores]],
    gt: Mapping[str, TemporalAnnotation],
) -> float:
    """Return the primary AP over the pooled chunks of the evaluation split.

    :param scores: chunk scores grouped by video.
    :param gt: annotations of the evaluated videos.
    :return: pooled AP, or ``nan`` with degenerate ground truth.
    """
    s, y, _ = _chunk_arrays(scores, gt)
    if not _guard(y, "dataset_chunk_ap"):
        return float("nan")
    return float(average_precision_score(y, s))


@dataclass(frozen=True)
class ClassBreakdown:
    """Behaviour of the binary detector on the events of one class.

    :param auc: AUC of this class's chunks against all negatives.
    :param ap: AP on the same chunks.
    :param n_pos: number of positive chunks of this class; report it next to the value
        (some classes have only a handful of positive chunks).
    """

    auc: float
    ap: float
    n_pos: int

    def as_dict(self) -> dict[str, float | int]:
        return {"auc": self.auc, "ap": self.ap, "n_pos": self.n_pos}


def breakdown_vs_negatives(
    s: npt.NDArray[np.float64],
    y: npt.NDArray[np.uint8],
    positives: npt.NDArray[np.bool_],
    *,
    name: str,
) -> ClassBreakdown:
    """Return AUC and AP of a stratum of positives against all negatives.

    Used for classes and for the coverage/duration strata of :mod:`evd.eval.strata`. The
    negatives are the same in every stratum, which keeps rows comparable.

    :param s: scores of all chunks of the partition.
    :param y: binary labels aligned with ``s``.
    :param positives: boolean mask of the positives forming the stratum.
    :param name: stratum name used in degenerate-case warnings.
    :return: the row, with ``nan`` AUC and AP when the stratum is empty.
    """
    selection = (y == 0) | positives
    ss, yy = s[selection], y[selection]
    n_pos = int(yy.sum())
    if not _guard(yy, name):
        return ClassBreakdown(float("nan"), float("nan"), n_pos)
    return ClassBreakdown(
        auc=float(roc_auc_score(yy, ss)),
        ap=float(average_precision_score(yy, ss)),
        n_pos=n_pos,
    )


def per_class_chunk(
    scores: Mapping[str, Sequence[WindowScores]],
    gt: Mapping[str, TemporalAnnotation],
    classes: Sequence[EventClass],
    *,
    normal_idx: int,
) -> dict[str, ClassBreakdown]:
    """Break the binary errors down by ground-truth class.

    Not one-vs-rest AP: the model emits a single score, so for class ``c`` the evaluation
    is restricted to the positive chunks of ``c`` plus all negatives.

    :param scores: chunk scores grouped by video.
    :param gt: annotations of the evaluated videos.
    :param classes: dataset classes, ``Normal`` included.
    :param normal_idx: index of ``Normal``, which is skipped.
    :return: one row per anomaly class, with its number of chunks.
    """
    s, y, c = _chunk_arrays(scores, gt)
    out: dict[str, ClassBreakdown] = {}
    for cls in classes:
        if cls.idx == normal_idx:
            continue
        # `class` is -1 on negatives, so `c == cls.idx` selects positives only.
        out[cls.name] = breakdown_vs_negatives(
            s, y, c == cls.idx, name=f"per_class_chunk[{cls.name}]"
        )
    return out


#: The two rules deciding which above-threshold chunk detects an event.
#:
#: * ``endpoint_dentro``: historical rule, the chunk must end inside the annotated
#:   interval. It stays the default so that existing ``metrics.json`` keep their meaning.
#: * ``intersezione``: the chunk only has to touch the interval, the same predicate used
#:   for chunk labels, event detection and video decisions.
LatencyRule = Literal["endpoint_dentro", "intersezione"]


@dataclass(frozen=True)
class DetectionLatency:
    """How late the system notices an event.

    :param regola: which :data:`LatencyRule` produced the numbers (the two rules give
        different medians on the same run).
    :param mediana_s: median delay over detected events.
    :param p90_s: 90th percentile over the same events.
    :param n_eventi: evaluated events.
    :param n_senza_latenza_misurabile: events with no measurable latency. Report it next
        to the percentiles: a median over measurable events improves when more events are
        missed. With ``endpoint_dentro`` it is not the number of missed events (an event
        touched by an above-threshold chunk that does not end inside it is detected but not
        measurable); with ``intersezione`` the two coincide.
    """

    mediana_s: float
    p90_s: float
    n_eventi: int
    n_senza_latenza_misurabile: int
    regola: LatencyRule = "endpoint_dentro"

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "regola": self.regola,
            "mediana_s": self.mediana_s,
            "p90_s": self.p90_s,
            "n_eventi": self.n_eventi,
            "n_senza_latenza_misurabile": self.n_senza_latenza_misurabile,
        }


def detection_latency(
    scores: Mapping[str, Sequence[WindowScores]],
    gt: Mapping[str, TemporalAnnotation],
    fps: Mapping[str, float],
    *,
    threshold: float,
    rule: LatencyRule = "endpoint_dentro",
) -> DetectionLatency:
    """Return the delay between the annotated onset and the first above-threshold chunk detecting it.

    The delay is the distance between the onset and the endpoint of the detecting chunk,
    since a chunk score exists only once its last frame has arrived.
    ``endpoint_dentro`` requires the chunk to end inside the interval (otherwise a missed
    event would be attributed to a later, unrelated alarm). ``intersezione`` only requires
    the chunk to touch the interval, which bounds the delay by ``duration + span``; with
    this rule the delay may exceed the event duration.

    :param scores: chunk scores grouped by video.
    :param gt: annotations of the evaluated videos.
    :param fps: native fps per video, to convert frames into seconds.
    :param threshold: threshold estimated on validation.
    :param rule: detection predicate; the default is the historical rule.
    :return: median, p90, counts and the rule that produced them.
    :raises KeyError: if the fps of a video is missing.
    :raises ValueError: if ``rule`` is unknown.
    """
    if rule not in ("endpoint_dentro", "intersezione"):
        raise ValueError(f"unknown latency rule: {rule!r}")
    delays: list[float] = []
    n_events = 0
    never = 0
    for video_id, ann in gt.items():
        if not ann.intervals:
            continue
        if video_id not in fps:
            raise KeyError(f"{video_id}: missing fps, latency cannot be converted into seconds")
        # Sort by (end, start): on a shared frame the later-starting window wins, as in the
        # frame mapper.
        above = sorted(
            (item.window.frame_span[1], item.window.frame_span[0])
            for item in scores.get(video_id, ())
            if item.anomaly >= threshold
        )
        for onset, offset, _ in ann.intervals:
            n_events += 1
            if rule == "endpoint_dentro":
                first = next((f for f, _a in above if onset <= f <= offset), None)
            else:
                first = next((f for f, a in above if a <= offset and f >= onset), None)
            if first is None:
                never += 1
            else:
                delays.append((first - onset) / fps[video_id])
    if not delays:
        return DetectionLatency(float("nan"), float("nan"), n_events, never, rule)
    array = np.asarray(delays, dtype=np.float64)
    return DetectionLatency(
        mediana_s=float(np.median(array)),
        p90_s=float(np.percentile(array, 90)),
        n_eventi=n_events,
        n_senza_latenza_misurabile=never,
        regola=rule,
    )


def frame_auc(
    scores: FrameScores, gt: TemporalAnnotation, *, restrict_to_covered: bool = True
) -> float:
    s, y = _pair(scores, gt, restrict_to_covered)
    if not _guard(y, "frame_auc"):
        return float("nan")
    return float(roc_auc_score(y, s))


def frame_ap(
    scores: FrameScores, gt: TemporalAnnotation, *, restrict_to_covered: bool = True
) -> float:
    """:return: frame-level AP of one video, or ``nan`` with degenerate ground truth."""
    s, y = _pair(scores, gt, restrict_to_covered)
    if not _guard(y, "frame_ap"):
        return float("nan")
    return float(average_precision_score(y, s))


def dataset_frame_auc(
    scores: Mapping[str, FrameScores],
    gt: Mapping[str, TemporalAnnotation],
    *,
    restrict_to_covered: bool = True,
) -> float:
    """Return the frame-level AUC over the pooled frames of all videos.

    This is how the VAD literature reports frame-level AUC on UCF-Crime: a single ROC on
    the union of frames, not the mean of per-video AUCs.

    :param scores: frame-level scores per video.
    :param gt: annotations of the evaluated videos.
    :param restrict_to_covered: whether to use only frames covered by a window.
    :return: pooled AUC, or ``nan`` with degenerate ground truth.
    """
    ss, ys = [], []
    for vid, fs in scores.items():
        s, y = _pair(fs, gt[vid], restrict_to_covered)
        ss.append(s)
        ys.append(y)
    if not ss:
        return float("nan")
    s_all, y_all = np.concatenate(ss), np.concatenate(ys)
    if not _guard(y_all, "dataset_frame_auc"):
        return float("nan")
    return float(roc_auc_score(y_all, s_all))


def dataset_frame_ap(
    scores: Mapping[str, FrameScores],
    gt: Mapping[str, TemporalAnnotation],
    *,
    restrict_to_covered: bool = True,
) -> float:
    """Return the frame-level AP over the pooled frames of all videos.

    :param scores: frame-level scores per video.
    :param gt: annotations of the evaluated videos.
    :param restrict_to_covered: whether to use only frames covered by a window.
    :return: pooled AP, or ``nan`` with degenerate ground truth.
    """
    ss, ys = [], []
    for vid, fs in scores.items():
        s, y = _pair(fs, gt[vid], restrict_to_covered)
        ss.append(s)
        ys.append(y)
    if not ss:
        return float("nan")
    s_all, y_all = np.concatenate(ss), np.concatenate(ys)
    if not _guard(y_all, "dataset_frame_ap"):
        return float("nan")
    return float(average_precision_score(y_all, s_all))


def per_class_ap(
    scores: Mapping[str, FrameScores],
    gt: Mapping[str, TemporalAnnotation],
    classes: Sequence[EventClass],
    *,
    normal_idx: int,
    restrict_to_covered: bool = True,
) -> dict[str, float]:
    """Return one-vs-rest AP per event class.

    Report it with the number of videos: rare classes dominate the macro average.

    :param scores: frame-level scores per video.
    :param gt: annotations of the evaluated videos.
    :param classes: dataset classes.
    :param normal_idx: index of ``Normal``.
    :param restrict_to_covered: whether to use only covered frames.
    :return: AP per class.
    :raises ValueError: if a class score is missing.
    """
    out: dict[str, float] = {}
    for cls in classes:
        if cls.idx == normal_idx:
            continue
        ss, ys = [], []
        for vid, fs in scores.items():
            n = len(fs.scores)
            cl = frame_class_labels(gt[vid], n, normal_idx)
            keep = np.ones(n, dtype=bool)
            if restrict_to_covered:
                keep &= np.asarray(fs.covered, dtype=bool)
            if not keep.any():
                continue
            if fs.class_scores is None or cls.idx >= fs.class_scores.shape[1]:
                raise ValueError(f"{vid}: frame-level score of class {cls.idx} missing")
            ss.append(np.asarray(fs.class_scores[:, cls.idx], dtype=np.float64)[keep])
            ys.append((cl[keep] == cls.idx).astype(np.uint8))
        if not ss:
            out[cls.name] = float("nan")
            continue
        s_all, y_all = np.concatenate(ss), np.concatenate(ys)
        out[cls.name] = (
            float(average_precision_score(y_all, s_all))
            if _guard(y_all, f"per_class_ap[{cls.name}]")
            else float("nan")
        )
    return out


def aggregate_video_scores(
    scores: Mapping[str, Sequence[WindowScores]], policy: Literal["max", "mean"] = "max"
) -> dict[str, float]:
    """Reduce chunk scores to one score per video.

    :param scores: chunk scores grouped by video.
    :param policy: ``max`` or ``mean``.
    :return: ``video_id`` to video score.
    :raises ValueError: on an unknown policy.
    """
    out: dict[str, float] = {}
    for vid, chunks in scores.items():
        s = np.asarray([item.anomaly for item in chunks], dtype=np.float64)
        if s.size == 0:
            out[vid] = float("nan")
        elif policy == "max":
            out[vid] = float(s.max())
        elif policy == "mean":
            out[vid] = float(s.mean())
        else:
            raise ValueError(f"unknown video aggregation policy: {policy!r}")
    return out


def video_auc(scores: Mapping[str, float], labels: Mapping[str, int]) -> float:
    vids = [v for v in scores if not np.isnan(scores[v])]
    y = np.array([labels[v] for v in vids], dtype=np.uint8)
    s = np.array([scores[v] for v in vids], dtype=np.float64)
    if not _guard(y, "video_auc"):
        return float("nan")
    return float(roc_auc_score(y, s))


def video_ap(scores: Mapping[str, float], labels: Mapping[str, int]) -> float:
    vids = [v for v in scores if not np.isnan(scores[v])]
    y = np.array([labels[v] for v in vids], dtype=np.uint8)
    s = np.array([scores[v] for v in vids], dtype=np.float64)
    if not _guard(y, "video_ap"):
        return float("nan")
    return float(average_precision_score(y, s))


def video_decision_scores(
    scores: Mapping[str, Sequence[WindowScores]],
    gt: Mapping[str, TemporalAnnotation],
) -> dict[str, float]:
    """Return the score used to decide on each video, with the asymmetric rule.

    ====================== =========================================================
    **anomalous** video     positive if at least one window that actually contains the
                            event exceeds the threshold
    **normal** video        positive if any window exceeds the threshold
    ====================== =========================================================

    The rule is a maximum over different sets (positive chunks for anomalous videos, all
    chunks for normal ones), so comparing it with the threshold reproduces the rule and
    reuses :func:`_prf`. It differs from :func:`aggregate_video_scores` with ``max``, which
    is the symmetric ``edp_v1`` rule.

    :param scores: chunk scores grouped by video.
    :param gt: annotations of the evaluated videos.
    :return: ``video_id`` to decision score; ``-inf`` for an anomalous video whose event no
        window covers, ``nan`` for a video without windows.
    """
    out: dict[str, float] = {}
    for video_id, chunks in scores.items():
        ann = gt[video_id]
        if not chunks:
            out[video_id] = float("nan")
            continue
        if ann.is_normal:
            out[video_id] = float(max(item.anomaly for item in chunks))
            continue
        covered = [
            float(item.anomaly) for item in chunks if window_label(ann, *item.window.frame_span)
        ]
        out[video_id] = max(covered) if covered else float("-inf")
    return out


# Fixed-threshold metrics


@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    f1: float
    threshold: float
    source: ThresholdSource
    """How the threshold was chosen; a P/R/F1 without it is not reportable."""
    n_pred_pos: int = 0
    n_true_pos: int = 0

    def as_dict(self) -> dict[str, float | str | int]:
        return asdict(self)


def _prf(
    y: npt.NDArray[np.uint8],
    s: npt.NDArray[np.float64],
    threshold: float,
    source: ThresholdSource,
) -> PRF:
    pred = (s >= threshold).astype(np.uint8)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return PRF(
        precision=precision,
        recall=recall,
        f1=f1,
        threshold=float(threshold),
        source=source,
        n_pred_pos=int(pred.sum()),
        n_true_pos=int(y.sum()),
    )


def frame_prf(
    scores: FrameScores,
    gt: TemporalAnnotation,
    threshold: float,
    *,
    source: ThresholdSource,
    restrict_to_covered: bool = True,
) -> PRF:
    """Return frame-level P/R/F1 at the given threshold (``source`` is required)."""
    s, y = _pair(scores, gt, restrict_to_covered)
    return _prf(y, s, threshold, source)


def dataset_frame_prf(
    scores: Mapping[str, FrameScores],
    gt: Mapping[str, TemporalAnnotation],
    threshold: float,
    *,
    source: ThresholdSource,
    restrict_to_covered: bool = True,
) -> PRF:
    ss, ys = [], []
    for vid, fs in scores.items():
        s, y = _pair(fs, gt[vid], restrict_to_covered)
        ss.append(s)
        ys.append(y)
    if not ss:
        return PRF(0.0, 0.0, 0.0, float(threshold), source)
    return _prf(np.concatenate(ys), np.concatenate(ss), threshold, source)


def dataset_chunk_prf(
    scores: Mapping[str, Sequence[WindowScores]],
    gt: Mapping[str, TemporalAnnotation],
    threshold: float,
    *,
    source: ThresholdSource,
) -> PRF:
    """Return the primary chunk P/R/F1 at the declared threshold.

    :param scores: chunk scores grouped by video.
    :param gt: annotations of the evaluated videos.
    :param threshold: the threshold, with its provenance in ``source``.
    :param source: where the threshold comes from; it ends up in the reported value.
    :return: precision, recall, F1 and the threshold with its provenance.
    """
    s, y, _ = _chunk_arrays(scores, gt)
    return _prf(y, s, threshold, source)


def video_decision_prf(
    scores: Mapping[str, Sequence[WindowScores]],
    gt: Mapping[str, TemporalAnnotation],
    threshold: float,
    *,
    source: ThresholdSource,
) -> PRF:
    """Return video-level P/R/F1 with the asymmetric rule.

    At video level only thresholded metrics are reported: the decision score depends on
    which windows exceeded the threshold, so ranking videos is not defined independently
    of the threshold.

    :param scores: chunk scores grouped by video.
    :param gt: annotations of the evaluated videos.
    :param threshold: the same threshold used at chunk level.
    :param source: where the threshold comes from.
    :return: precision, recall and F1 over videos.
    """
    decision = video_decision_scores(scores, gt)
    vids = [v for v in decision if not np.isnan(decision[v])]
    y = np.array([int(not gt[v].is_normal) for v in vids], dtype=np.uint8)
    s = np.array([decision[v] for v in vids], dtype=np.float64)
    if y.size == 0:
        return PRF(0.0, 0.0, 0.0, float(threshold), source)
    return _prf(y, s, threshold, source)


# Event-level metrics


@dataclass(frozen=True)
class EventPRF:
    """P/R/F1 with the event as unit, plus the counts that produce them.

    Recall counts annotated events and precision counts emitted alarms, so the two sides
    have different denominators and do not form a confusion matrix.
    """

    precision: float
    recall: float
    f1: float
    threshold: float
    source: ThresholdSource
    n_eventi: int
    """Annotated events in the evaluated videos (recall denominator)."""
    n_eventi_rilevati: int
    n_allarmi: int
    """Emitted alarms (precision denominator)."""
    n_allarmi_veri: int
    n_eventi_irraggiungibili: int
    """Events that no grid window intersects (undetectable regardless of the model);
    zero on ``edp_v2``, where all 1830 intervals are reachable."""

    def as_dict(self) -> dict[str, float | str | int]:
        return asdict(self)


def _intersects(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Return ``True`` if the two closed frame intervals share a frame."""
    return min(a_end, b_end) >= max(a_start, b_start)


def _alarms(finestre: Sequence[WindowScores], threshold: float) -> list[tuple[int, int]]:
    """Merge the above-threshold windows into alarms and return their spans.

    An alarm is a maximal run of grid-consecutive windows above the threshold, spanning
    from the first frame of the first window to the last frame of the last one. Windows are
    sorted by position first, so the result does not depend on loader order.

    :param finestre: chunk scores of a single video.
    :param threshold: the threshold.
    :return: alarm spans in temporal order (empty when the video has none).
    """
    sorted_items = sorted(finestre, key=lambda item: item.window.frame_span)
    outside: list[tuple[int, int]] = []
    current: tuple[int, int] | None = None
    previous_above = False
    for item in sorted_items:
        above = item.anomaly >= threshold
        start, end = item.window.frame_span
        if above and previous_above and current is not None:
            current = (current[0], max(current[1], end))
        elif above:
            if current is not None:
                outside.append(current)
            current = (start, end)
        previous_above = above
    if current is not None:
        outside.append(current)
    return outside


def event_prf(
    scores: Mapping[str, Sequence[WindowScores]],
    gt: Mapping[str, TemporalAnnotation],
    threshold: float,
    *,
    source: ThresholdSource,
) -> EventPRF:
    """Return P/R/F1 with the event as unit.

    Our definition (the laboratory protocol does not define the event level): an event is
    **detected** if at least one above-threshold window intersects it, even by one frame;
    consecutive above-threshold windows merge into an **alarm**, and an alarm intersecting
    no annotated event is a false positive (every alarm on a normal video is false). The
    matching is many-to-many: a long alarm covering two events is one true positive that
    detects two events. Precision and recall have different denominators (region-based
    evaluation).

    :param scores: chunk scores grouped by video. All evaluated videos must be present
        (normal ones included) with their full grid, since alarm adjacency is the grid
        adjacency.
    :param gt: annotations of the evaluated videos.
    :param threshold: the same threshold used at chunk and video level.
    :param source: where the threshold comes from.
    :return: the triple with the four counts producing it.
    """
    n_events = 0
    detected = 0
    unreachable = 0
    n_alarms = 0
    true_items = 0
    for video_id, ann in gt.items():
        windows = list(scores.get(video_id, ()))
        for span in _alarms(windows, threshold):
            n_alarms += 1
            true_items += window_label(ann, *span)
        if not ann.intervals:
            continue
        all_items = [item.window.frame_span for item in windows]
        above = [item.window.frame_span for item in windows if item.anomaly >= threshold]
        for onset, offset, _ in ann.intervals:
            n_events += 1
            if not any(_intersects(s, e, onset, offset) for s, e in all_items):
                unreachable += 1
            if any(_intersects(s, e, onset, offset) for s, e in above):
                detected += 1
    recall = detected / n_events if n_events > 0 else 0.0
    precision = true_items / n_alarms if n_alarms > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    if unreachable:
        log.warning(
            "%d events out of %d are not intersected by any window: the grid is not "
            "the edp_v2 one, where all 1830 intervals are reachable",
            unreachable,
            n_events,
        )
    return EventPRF(
        precision=precision,
        recall=recall,
        f1=f1,
        threshold=float(threshold),
        source=source,
        n_eventi=n_events,
        n_eventi_rilevati=detected,
        n_allarmi=n_alarms,
        n_allarmi_veri=true_items,
        n_eventi_irraggiungibili=unreachable,
    )


def per_class_chunk_prf(
    scores: Mapping[str, Sequence[WindowScores]],
    gt: Mapping[str, TemporalAnnotation],
    classes: Sequence[EventClass],
    threshold: float,
    *,
    source: ThresholdSource,
    normal_idx: int,
) -> dict[str, PRF]:
    """Return per-class P/R/F1, breaking down the binary errors.

    Same semantics as :func:`per_class_chunk`: for class ``c``, its positives plus all
    negatives, at the same threshold for every class.

    :param scores: chunk scores grouped by video.
    :param gt: annotations of the evaluated videos.
    :param classes: dataset classes, ``Normal`` included.
    :param threshold: shared threshold.
    :param source: threshold provenance.
    :param normal_idx: index of ``Normal``, which is skipped.
    :return: one row per anomaly class.
    """
    s, y, c = _chunk_arrays(scores, gt)
    negatives = y == 0
    out: dict[str, PRF] = {}
    for cls in classes:
        if cls.idx == normal_idx:
            continue
        selection = negatives | (c == cls.idx)
        ss, yy = s[selection], y[selection]
        out[cls.name] = (
            _prf(yy, ss, threshold, source)
            if yy.size
            else PRF(0.0, 0.0, 0.0, float(threshold), source)
        )
    return out


def per_class_prf(
    scores: Mapping[str, FrameScores],
    gt: Mapping[str, TemporalAnnotation],
    classes: Sequence[EventClass],
    threshold: float,
    *,
    source: ThresholdSource,
    normal_idx: int,
    restrict_to_covered: bool = True,
) -> dict[str, PRF]:
    out: dict[str, PRF] = {}
    for cls in classes:
        if cls.idx == normal_idx:
            continue
        ss, ys = [], []
        for vid, fs in scores.items():
            n = len(fs.scores)
            cl = frame_class_labels(gt[vid], n, normal_idx)
            keep = np.ones(n, dtype=bool)
            if restrict_to_covered:
                keep &= np.asarray(fs.covered, dtype=bool)
            if not keep.any():
                continue
            if fs.class_scores is None or cls.idx >= fs.class_scores.shape[1]:
                raise ValueError(f"{vid}: frame-level score of class {cls.idx} missing")
            ss.append(np.asarray(fs.class_scores[:, cls.idx], dtype=np.float64)[keep])
            ys.append((cl[keep] == cls.idx).astype(np.uint8))
        out[cls.name] = (
            _prf(np.concatenate(ys), np.concatenate(ss), threshold, source)
            if ss
            else PRF(0.0, 0.0, 0.0, float(threshold), source)
        )
    return out


def threshold_from_scores(
    scores: Sequence[float], *, quantile: float, source: ThresholdSource
) -> tuple[float, ThresholdSource]:
    """Return a threshold as a quantile of scores that are not the evaluation queries.

    :param scores: scores to take the quantile of (support or validation).
    :param quantile: quantile in [0, 1].
    :param source: provenance label of the threshold.
    :return: the threshold and its provenance.
    :raises ValueError: on an invalid quantile or no scores.
    """
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile outside [0,1]: {quantile}")
    arr = np.asarray(list(scores), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("no scores to estimate the threshold from")
    return float(np.quantile(arr, quantile)), source


def threshold_youden(
    scores: Sequence[float], labels: Sequence[int]
) -> tuple[float, ThresholdSource]:
    """Return the threshold at the best ROC point, ``argmax(TPR - FPR)`` (Youden index).

    This is the laboratory protocol rule: unlike a negative-score quantile it looks at both
    classes. It is estimated on validation only and shared by chunk and video level.

    :param scores: validation chunk scores.
    :param labels: matching binary labels, in the same order.
    :return: the threshold and its provenance (``"val_youden"``).
    :raises ValueError: if the vectors differ in length, are empty, or the ground truth is
        degenerate.
    """
    s = np.asarray(list(scores), dtype=np.float64)
    y = np.asarray(list(labels), dtype=np.int64)
    if s.size != y.size:
        raise ValueError(f"{s.size} scores and {y.size} labels: not the same set")
    if s.size == 0:
        raise ValueError("no scores to estimate the threshold from")
    positives = int((y == 1).sum())
    if positives == 0 or positives == y.size:
        raise ValueError(
            f"degenerate validation ground truth ({positives} positives out of {y.size}): the "
            f"ROC and the Youden threshold are undefined. Check split and grid "
            f"before reading any thresholded number"
        )
    fpr, tpr, thresholds = roc_curve(y, s)
    chosen = int(np.argmax(tpr - fpr))
    value = float(thresholds[chosen])
    if not np.isfinite(value):
        # `roc_curve` starts with a +inf threshold (the (0, 0) point), reached only when the model
        # does not separate at all: declare the degenerate "no predicted positive" point instead
        # of returning inf.
        log.warning(
            "degenerate Youden threshold: no ROC point beats the diagonal, "
            "the operating point predicts no positive"
        )
        value = float(np.nextafter(s.max(), np.inf))
    return value, "val_youden"
