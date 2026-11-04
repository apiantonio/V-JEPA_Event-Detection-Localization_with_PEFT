"""The metrics of the OV-CAVED experimental protocol.

This is the only place where rows, threshold and metrics are computed: the evaluation scripts
(OV-CAVED and OVAD) differ only in how they prepare the data, and both go through here.

Conventions (see README.md for the full explanation):
  * the extent of a chunk is the time interval whose frames the model has seen; a chunk cut by the
    end of the video has a shorter extent. The center is the midpoint of the extent;
  * the events of an entry are its annotated intervals, in seconds, clipped to [0, D] and merged
    when they touch or overlap. D is the end of the last chunk;
  * rows: the timeline is cut at the event boundaries, then every segment of length L is split into
    k = max(1, ceil(L/T)) equal parts. Every row is entirely inside an event or entirely outside;
  * evidence: an event row takes the maximum score of the chunks whose extent overlaps it
    (overlap > 0); a clean row takes the maximum score of the chunks centered in it whose guard
    window does not overlap any event. The guard window is the extent widened by (g - c)/2 on each
    side (for a full chunk: a window of length g centered on the chunk). If no chunk is left, the
    row is unjudgeable (label -1) and is left out of the row metrics. A clean row that contains no
    chunk center uses the chunk whose center is nearest, under the same guard condition; an event
    row that no extent overlaps is unjudgeable;
  * video level, two outcomes per entry: the event part (its judgeable event rows, if any) is a TP
    if one of them reaches the threshold, otherwise a FN; the clean part (its judgeable clean rows,
    if any) is a FP if one of them reaches the threshold, otherwise a TN;
  * event level, also computed from the row table: an event is detected if one of its judgeable
    event rows reaches the threshold; an event with no judgeable row is not counted (n_events_unseen);
  * AUROC with average ranks; step-wise AP, with tied scores entering together; thresholds with >=;
  * threshold: Youden point on the judgeable validation rows;
  * protocol mean: AUC, AUC_a macro, row F1, 1 - row FPR, video F1, event recall.
Depends only on numpy.
"""
from __future__ import annotations

import math

import numpy as np

H = 4.0     # asking rate of the protocol, in seconds
T = 32.0    # maximum length of a row (row size), in seconds
C = 8.0     # default chunk size and guard: evaluate.py always passes the declared ones
G = 8.0
EPS = 1e-9  # only against rounding residues, not a tolerance


class NoJudgeableRows(ValueError):
    """No entry has a judgeable row: there is nothing to evaluate."""


# --------------------------------------------------------------------------- entries and rows
def merge_events(intervals, duration: float) -> list[tuple[float, float]]:
    """Intervals in seconds -> events: clipped to [0, duration], merged when they touch or overlap."""
    out: list[tuple[float, float]] = []
    for a, b in sorted((max(0.0, a), min(duration, b)) for a, b in intervals):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def make_entry(extents, scores, intervals, duration: float | None = None) -> dict:
    """extents: [(start, end)] in seconds, one per chunk; scores: one per chunk; intervals in seconds.

    The timeline ends where the last chunk ends, unless `duration` is given."""
    o = sorted(range(len(extents)), key=lambda i: extents[i][0])
    ex = [(float(extents[i][0]), float(extents[i][1])) for i in o]
    d = max(b for _, b in ex) if duration is None else float(duration)
    return {"extents": ex, "centers": [(a + b) / 2 for a, b in ex], "scores": [float(scores[i]) for i in o],
            "duration": d, "events": merge_events(intervals, d)}


def overlaps(x0: float, x1: float, intervals) -> bool:
    return any(min(x1, b) - max(x0, a) > 0 for a, b in intervals)


def make_rows(duration: float, events, t: float = T) -> list[tuple[float, float, int]]:
    """The rows of an entry: (start, end, label), label 1 for an event row and 0 for a clean row."""
    segments, t0 = [], 0.0
    for a, b in events:
        if a > t0 + EPS:
            segments.append((t0, a, 0))
        segments.append((a, b, 1))
        t0 = b
    if t0 < duration - EPS:
        segments.append((t0, duration, 0))
    out = []
    for s0, s1, label in segments or [(0.0, duration, 0)]:
        k = max(1, math.ceil((s1 - s0) / t - EPS))
        out += [(s0 + i * (s1 - s0) / k, s0 + (i + 1) * (s1 - s0) / k, label) for i in range(k)]
    return out


def row_table(e: dict, t: float = T, c: float = C, g: float = G) -> list[tuple[float, int]]:
    """One (score, label) per row: 1 event row, 0 clean row, -1 unjudgeable row."""
    s, cs, ex, ev, dur = e["scores"], e["centers"], e["extents"], e["events"], e["duration"]
    m = (g - c) / 2                                   # how much the guard widens the extent
    guard = [(max(0.0, a - m), min(dur, b + m)) for a, b in ex]
    nearest = lambda r0, r1: [min(range(len(cs)), key=lambda i: abs(cs[i] - (r0 + r1) / 2))]
    out = []
    for r0, r1, label in make_rows(dur, ev, t):
        if label == 1:
            # no fallback to the nearest chunk: it would not have seen the event
            idx = [i for i, (a, b) in enumerate(ex) if min(b, r1) - max(a, r0) > 0]
            out.append((max(s[i] for i in idx), 1) if idx else (float("nan"), -1))
        else:
            inside = [i for i, x in enumerate(cs) if r0 <= x < r1] or nearest(r0, r1)
            valid = [i for i in inside if not overlaps(guard[i][0], guard[i][1], ev)]
            out.append((max(s[i] for i in valid), 0) if valid else (max(s[i] for i in inside), -1))
    return out


# --------------------------------------------------------------------------- basic metrics
def auroc(s, y) -> float:
    s, y = np.asarray(s, float), np.asarray(y, float)
    if not len(s) or y.min() == y.max():
        return float("nan")
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    ss, i = s[o], 0
    while i < len(ss):                                # ranks from 1, ties get the average rank
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    p, n = y.sum(), len(y) - y.sum()
    return float((r[y > 0.5].sum() - p * (p + 1) / 2) / (p * n))


def ap(s, y) -> float:
    """AP = sum over every distinct score value of (increase in recall) x (precision)."""
    s, y = np.asarray(s, float), np.asarray(y, float)
    if not len(s) or not y.sum():
        return float("nan")
    o = np.argsort(-s, kind="mergesort")
    s, y = s[o], y[o]
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]      # last element of every group of ties
    tp, fp = np.cumsum(y)[last], np.cumsum(1 - y)[last]
    rec = tp / y.sum()
    return float(((rec - np.r_[0.0, rec[:-1]]) * tp / (tp + fp)).sum())


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    return p, r, (2 * p * r / (p + r) if p and r and p + r else 0.0)


# --------------------------------------------------------------------------- the threshold
def youden_threshold(s, y) -> float:
    """The best point of the ROC curve, maximum of TPR - FPR, with these rules: candidates only at
    the boundaries between distinct scores, cut halfway between the two, among ties the highest.
    The result does not depend on the order in which the rows arrive."""
    s, y = np.asarray(s, float), np.asarray(y, float)
    o = np.argsort(-s, kind="mergesort")
    s, y = s[o], y[o]
    tpr = np.cumsum(y) / max(1.0, y.sum())
    fpr = np.cumsum(1 - y) / max(1.0, (1 - y).sum())
    j = np.where(np.r_[s[:-1] != s[1:], True], tpr - fpr, -2.0)
    i = int(np.nonzero(j >= j.max() - 1e-12)[0][0])
    return float(s[i] - 1e-9) if i == len(s) - 1 else float((s[i] + s[i + 1]) / 2)


def validation_threshold(entries_per_level, t: float = T, c: float = C, g: float = G) -> float:
    """entries_per_level: the validation entries of every query level. The judgeable rows of all
    levels enter the Youden point together."""
    s, y = [], []
    for ents in entries_per_level:
        for e in ents:
            for x, label in row_table(e, t, c, g):
                if label >= 0:
                    s.append(x); y.append(label)
    if not s:
        raise NoJudgeableRows("the validation set has no judgeable row: the threshold cannot be fixed")
    return youden_threshold(s, y)


MEAN_METRICS = ("auc", "auc_a_macro", "row_f1", "row_fpr", "video_f1", "event_recall")


def protocol_mean(m: dict) -> float:
    """The protocol mean: six metrics, one per question, with the row FPR entering as 1 - FPR."""
    v = [1 - m[k] if k == "row_fpr" else m[k] for k in MEAN_METRICS]
    return float(np.mean(v))


# --------------------------------------------------------------------------- the 15 metrics
def compute_metrics(entries, threshold: float, t: float = T, c: float = C, g: float = G) -> dict:
    rs, ry, as_, ay, per_auc, per_ap = [], [], [], [], [], []
    n_with_event = unjudgeable = n_ev = detected = unseen = 0
    vtp = vfn = vfp = vtn = vfp_ev = vtn_ev = 0
    hours = 0.0
    for e in entries:
        tb = row_table(e, t, c, g)
        rows = make_rows(e["duration"], e["events"], t)
        unjudgeable += sum(1 for _, y in tb if y == -1)
        hours += e["duration"] / 3600.0
        has_event = bool(e["events"])
        # video level, two outcomes: the event part (TP/FN) if the entry has judgeable event rows,
        # the clean part (FP/TN) if it has judgeable clean rows; unjudgeable rows do not count
        event_rows = [x for x, y in tb if y == 1]
        if event_rows:
            fired = any(x >= threshold for x in event_rows); vtp += fired; vfn += not fired
        clean_rows = [x for x, y in tb if y == 0]
        if clean_rows:
            fired = any(x >= threshold for x in clean_rows); vfp += fired; vtn += not fired
            if has_event:
                vfp_ev += fired; vtn_ev += not fired
        # event level, from the row table: event rows are grouped by event, and an event is detected
        # if one of its judgeable rows reaches the threshold. An event with no judgeable row (no chunk
        # overlaps it) was never seen: it is not counted, and ends up in n_events_unseen
        for a, b in e["events"]:
            own = [x for (r0, r1, _), (x, y) in zip(rows, tb) if y == 1 and r0 >= a - EPS and r1 <= b + EPS]
            if own:
                n_ev += 1; detected += max(own) >= threshold
            else:
                unseen += 1
        good = [(x, y) for x, y in tb if y >= 0]
        if not good:
            continue
        s = np.array([x for x, _ in good]); y = np.array([float(v) for _, v in good])
        rs.append(s); ry.append(y)
        if has_event:
            n_with_event += 1
            as_.append(s); ay.append(y)
            if y.min() != y.max():                    # entry usable for the macro averages
                per_auc.append(auroc(s, y)); per_ap.append(ap(s, y))
    if not rs:
        raise NoJudgeableRows("no entry has a judgeable row: there is nothing to evaluate")
    S, Y = np.concatenate(rs), np.concatenate(ry)
    AS = np.concatenate(as_) if as_ else np.array([]); AY = np.concatenate(ay) if ay else np.array([])
    pr = S >= threshold
    rp, rr, rf = _prf(int((pr & (Y == 1)).sum()), int((pr & (Y == 0)).sum()), int((~pr & (Y == 1)).sum()))
    neg = int((Y == 0).sum())
    vp, vr, vf = _prf(vtp, vfp, vfn)
    out = {"auc": auroc(S, Y), "ap": ap(S, Y),
           "auc_a_macro": float(np.mean(per_auc)) if per_auc else float("nan"),
           "ap_a_macro": float(np.mean(per_ap)) if per_ap else float("nan"),
           "auc_a_micro": auroc(AS, AY), "ap_a_micro": ap(AS, AY),
           "row_precision": rp, "row_recall": rr, "row_f1": rf,
           "row_fpr": int((pr & (Y == 0)).sum()) / neg if neg else float("nan"),
           "video_precision": vp, "video_recall": vr, "video_f1": vf,
           "video_fpr": vfp / (vfp + vtn) if vfp + vtn else float("nan"),
           "event_recall": detected / n_ev if n_ev else float("nan"),
           "threshold": threshold, "n_rows": int(len(S)), "n_unjudgeable_rows": unjudgeable,
           "n_events": n_ev, "n_events_detected": detected, "n_events_unseen": unseen,
           "n_entries_with_event": n_with_event, "n_entries_in_macro": len(per_auc),
           "video_tp": vtp, "video_fn": vfn, "video_fp": vfp, "video_tn": vtn,
           "video_fp_with_event": vfp_ev, "video_tn_with_event": vtn_ev, "hours": hours}
    out["mean"] = protocol_mean(out)
    return out
