"""The shared engine of the evaluation scripts: from test (and validation) entries to the results.

The OV-CAVED script passes three levels (coarse, mid, fine), the OVAD script only one.
Everything else, threshold included, is identical.
"""
from __future__ import annotations

import math

from . import metrics as M
from .inputs import InputError, build_entries, protocol_warnings

# The metrics in printing order, with their readable names.
METRIC_NAMES = [
    ("mean", "protocol mean"),
    ("auc", "row · AUC"),
    ("ap", "row · AP"),
    ("auc_a_macro", "row · AUC_a macro"),
    ("ap_a_macro", "row · AP_a macro"),
    ("auc_a_micro", "row · AUC_a micro"),
    ("ap_a_micro", "row · AP_a micro"),
    ("row_precision", "row · precision"),
    ("row_recall", "row · recall"),
    ("row_f1", "row · F1"),
    ("row_fpr", "row · FPR"),
    ("video_precision", "video · precision"),
    ("video_recall", "video · recall"),
    ("video_f1", "video · F1"),
    ("video_fpr", "video · FPR"),
    ("event_recall", "event · recall"),
]
COUNT_NAMES = [
    ("n_rows", "judgeable rows"), ("n_unjudgeable_rows", "unjudgeable rows"), ("n_events", "events"),
    ("n_events_detected", "events detected"), ("n_events_unseen", "events never seen, excluded"),
    ("n_entries_with_event", "entries with an event"), ("n_entries_in_macro", "entries in the macro averages"),
    ("video_tp", "video TP"), ("video_fn", "video FN"), ("video_fp", "video FP"), ("video_tn", "video TN"),
    ("hours", "hours of video"),
]


def _same_model(hps: list[tuple[str, dict]]) -> dict:
    """All score files must come from the same model: the same chunk size everywhere, and on the test
    set also the same asking rate and grid. The validation set may have a different asking rate (it
    only serves to fix the threshold): this is declared among the warnings."""
    ref_name, ref = hps[0]
    for name, hp in hps[1:]:
        if hp["chunk_size"] != ref["chunk_size"]:
            raise InputError(f"different chunk size between {ref_name} ({ref['chunk_size']:g} s) and {name} "
                             f"({hp['chunk_size']:g} s): the scores must come from the same model")
        if name.startswith("test") and hp != ref:
            raise InputError(f"different hyperparameters between {ref_name} {ref} and {name} {hp}: "
                             "all test score files must have the same ones")
    return ref


def _validation_warnings(hps: list[tuple[str, dict]], ref: dict) -> list[str]:
    other = sorted({f"{hp['asking_rate']:g} s" for name, hp in hps
                    if name.startswith("validation") and hp["asking_rate"] != ref["asking_rate"]})
    return ([f"the validation set has asking rate {', '.join(other)}, the test set {ref['asking_rate']:g} s"]
            if other else [])


def evaluate_levels(test: dict, val: dict | None = None, threshold: float | None = None,
                    guard: float | None = None, T: float = M.T) -> dict:
    """test: {level: (labels, scores)}; val: the same shape, or None.

    The threshold is taken from the validation set (Youden point on the judgeable rows of all levels
    together) or passed by hand: exactly one of the two."""
    if (val is None) == (threshold is None):
        raise InputError("exactly one of the validation set (labels and scores) and a threshold passed by hand is required")
    ent, hps = {}, []
    for lv, (lab, sc) in test.items():
        ent[lv], hp = build_entries(lab, sc, f"test {lv}")
        hps.append((f"test {lv}", hp))
    ent_val = {}
    if val is not None:
        if set(val) != set(test):
            raise InputError(f"the validation set has levels {sorted(val)}, the test set {sorted(test)}")
        for lv, (lab, sc) in val.items():
            ent_val[lv], hp = build_entries(lab, sc, f"validation {lv}")
            hps.append((f"validation {lv}", hp))
    hp = _same_model(hps)
    c = hp["chunk_size"]
    g = c if guard is None else float(guard)
    if not math.isfinite(g) or g < c:
        raise InputError(f"guard {g:g} s smaller than the chunk size {c:g} s: the guard is the longest chunk "
                         "size among the models compared, so at least the model's own")
    notes = protocol_warnings(hp) + _validation_warnings(hps, hp)
    if abs(T - M.T) > 1e-9:
        notes.append(f"T = {T:g} s: the main protocol uses T = {M.T:g} s")
    try:
        if val is not None:
            theta = M.validation_threshold([ent_val[lv] for lv in ent_val], T, c, g)
            source = "validation: Youden point on the judgeable rows" + (
                ", all levels together" if len(ent_val) > 1 else "")
        else:
            theta = float(threshold)
            if not math.isfinite(theta):
                raise InputError(f"threshold {threshold!r} is not finite")
            source = "passed by hand"
            notes.append("threshold passed by hand: the protocol fixes it on the validation set")
        levels = {lv: M.compute_metrics(ent[lv], theta, T, c, g) for lv in ent}
    except M.NoJudgeableRows as e:
        raise InputError(str(e)) from None
    out = {
        "parameters": {"chunk_size": c, "asking_rate": hp["asking_rate"], "grid": hp["grid"],
                       "guard": g, "T": T},
        "threshold": {"value": theta, "source": source},
        "warnings": notes,
        "levels": levels,
    }
    if len(levels) > 1:
        out["mean_over_levels"] = float(sum(m["mean"] for m in levels.values()) / len(levels))
    return out


def to_json(x):
    """The results ready for json.dump: NaN (metric not defined) becomes null."""
    if isinstance(x, dict):
        return {k: to_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_json(v) for v in x]
    if isinstance(x, float) and x != x:
        return None
    return x


def report(r: dict) -> str:
    """The results as a readable table: metrics in percent, counts as integers."""
    lv = list(r["levels"])
    p = r["parameters"]
    lines = [f"chunk size {p['chunk_size']:g} s · asking rate {p['asking_rate']:g} s · grid {p['grid']} · "
             f"guard {p['guard']:g} s · T {p['T']:g} s",
             f"threshold θ = {r['threshold']['value']:.6f} ({r['threshold']['source']})"]
    lines += [f"WARNING: {a}" for a in r["warnings"]]
    w = max(len(n) for _, n in METRIC_NAMES + COUNT_NAMES)
    lines.append("")
    lines.append(" " * w + "".join(f"{x:>12}" for x in lv))
    for k, n in METRIC_NAMES:
        vals = [r["levels"][x][k] for x in lv]
        lines.append(f"{n:<{w}}" + "".join("         n/a" if v != v else f"{100 * v:12.2f}" for v in vals))
    lines.append("")
    for k, n in COUNT_NAMES:
        vals = [r["levels"][x][k] for x in lv]
        lines.append(f"{n:<{w}}" + "".join(f"{v:12.2f}" if k == "hours" else f"{v:12d}" for v in vals))
    if "mean_over_levels" in r:
        lines += ["", f"protocol mean over the {len(lv)} levels: {100 * r['mean_over_levels']:.2f}"]
    lines += ["", "Metrics in percent (x100). n/a = not defined on these data (for example only one "
                  "class present)."]
    return "\n".join(lines)
