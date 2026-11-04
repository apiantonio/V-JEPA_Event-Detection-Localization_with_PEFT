"""The tests of the package. Run them like this, with no dependency other than numpy:

    python tests/test_protocol.py

Three families of checks: small cases worked out by hand, with the result written next to them; the
basic metrics against a second, brute-force implementation; evaluate_ovad.py on the example data against
the expected values in tests/expected_example.json.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from protocol_core import InputError, evaluate_levels, metrics as M  # noqa: E402
from protocol_core.evaluate import to_json  # noqa: E402
from protocol_core.inputs import build_entries, extents_from_indices  # noqa: E402
from evaluate_ovad import evaluate_ovad, main as main_ovad  # noqa: E402

DATA = ROOT / "example" / "data"
EXPECTED = Path(__file__).resolve().parent / "expected_example.json"


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol


def scores(values, c=8, h=4, grid="start"):
    return {"hyperparameters": {"chunk_size": c, "asking_rate": h, "grid": grid}, "scores": values}


# A 40 s video with the event in [10, 14) and a 20 s normal video. 8 s chunks every 4 s.
WITH_EVENT = {"duration": 40.0, "intervals": [[10.0, 14.0]]}
S_WITH_EVENT = [0.1, 0.9, 0.8, 0.7, 0.2, 0.3, 0.6, 0.2, 0.1]     # chunks [0,8], [4,12], ..., [32,40]
NORMAL = {"duration": 20.0, "intervals": []}
S_NORMAL = [0.2, 0.3, 0.1, 0.4]                                 # chunks [0,8], [4,12], [8,16], [12,20]


# --------------------------------------------------------------------------- 1. small cases worked by hand
def test_rows_cut_at_boundaries():
    # 73 s, event [16.5, 24): one row before, the event row, and the 49 s after split into 2 rows of 24.5 s
    r = M.make_rows(73.0, [(16.5, 24.0)], 32.0)
    assert [(round(a, 6), round(b, 6), l) for a, b, l in r] == [
        (0.0, 16.5, 0), (16.5, 24.0, 1), (24.0, 48.5, 0), (48.5, 73.0, 0)]


def test_intervals_merged_and_clipped():
    assert M.merge_events([(5, 10), (10, 12), (30, 50), (-3, 1)], 40.0) == [(0.0, 1.0), (5.0, 12.0), (30.0, 40.0)]


def test_evidence_rule():
    (e,), _ = build_entries({"a": WITH_EVENT}, scores({"a": S_WITH_EVENT}))
    t = M.row_table(e, 32.0, 8.0, 8.0)
    # row [0,10): chunk [4,12] (0.9) overlaps the event and stays out, only [0,8] (0.1) counts
    # row [10,14): the chunks overlapping it are [4,12], [8,16], [12,20]: maximum 0.9
    # row [14,40): center 16 ([12,20], overlaps the event) excluded; 20..36 remain: maximum 0.6
    assert t == [(0.1, 0), (0.9, 1), (0.6, 0)]


def test_unjudgeable_row_with_long_guard():
    (e,), _ = build_entries({"a": WITH_EVENT}, scores({"a": S_WITH_EVENT}))
    # guard 24 s: every extent is widened by 8 s on each side. In row [0,10) the only candidate,
    # [0,8], becomes [0,16] and overlaps the event: the row is unjudgeable (-1). In row [14,40)
    # the chunks centered from 28 on remain: maximum 0.6
    t = M.row_table(e, 32.0, 8.0, 24.0)
    assert [l for _, l in t] == [-1, 1, 0] and t[2][0] == 0.6


def test_event_row_without_chunks():
    # explicit extents with a gap between 16 and 28 s, event [20,24): no extent overlaps the event row,
    # which is unjudgeable (no fallback to the nearest chunk, which did not see the event).
    # The video has no judgeable event row, so it gives neither TP nor FN, and the event, never seen,
    # does not enter the event recall
    extents = [(0.0, 8.0), (4.0, 12.0), (8.0, 16.0), (28.0, 36.0), (32.0, 40.0)]
    e = M.make_entry(extents, [0.1, 0.2, 0.3, 0.9, 0.4], [(20.0, 24.0)])
    t = M.row_table(e, 32.0, 8.0, 8.0)
    assert [l for _, l in t] == [0, -1, 0] and t[0][0] == 0.3 and t[2][0] == 0.9
    m = M.compute_metrics([e], 0.5, 32.0, 8.0, 8.0)
    assert (m["video_tp"], m["video_fn"], m["video_fp"], m["video_tn"]) == (0, 0, 1, 0)
    assert m["n_unjudgeable_rows"] == 1
    assert (m["n_events"], m["n_events_detected"], m["n_events_unseen"]) == (0, 0, 1)
    assert m["event_recall"] != m["event_recall"]                    # nan: no event to count


def test_event_split_into_several_rows():
    # a 40 s event is split into two 20 s event rows: the event is detected if at least one of the two
    # is, and it counts once. Explicit extents with a gap over the second half of the event: the
    # second row is unjudgeable, but the event can still be judged on the first one
    extents = [(0.0, 8.0), (4.0, 12.0), (8.0, 16.0), (12.0, 20.0)]
    for peak, hit in ((0.9, 1), (0.2, 0)):
        e = M.make_entry(extents, [0.1, 0.1, 0.1, peak], [(10.0, 50.0)], duration=60.0)
        m = M.compute_metrics([e], 0.5, 32.0, 8.0, 8.0)
        assert (m["n_events"], m["n_events_detected"], m["n_events_unseen"]) == (1, hit, 0)


def test_video_level_two_outcomes_and_metrics():
    r = evaluate_levels({"all": ({"a": WITH_EVENT, "n": NORMAL}, scores({"a": S_WITH_EVENT, "n": S_NORMAL}))},
                        threshold=0.5)
    m = r["levels"]["all"]
    # video with the event: event detected (0.9) -> TP, and false alarm outside (0.6) -> FP; normal: 0.4 -> TN
    assert (m["video_tp"], m["video_fn"], m["video_fp"], m["video_tn"]) == (1, 0, 1, 1)
    assert close(m["video_precision"], 0.5) and close(m["video_recall"], 1.0) and close(m["video_fpr"], 0.5)
    assert close(m["video_f1"], 2 / 3)
    # judgeable rows: (0.1, 0) (0.9, 1) (0.6, 0) and (0.4, 0). At threshold 0.5: TP 1, FP 1, FN 0
    assert m["n_rows"] == 4 and m["n_unjudgeable_rows"] == 0
    assert close(m["row_precision"], 0.5) and close(m["row_recall"], 1.0) and close(m["row_fpr"], 1 / 3)
    assert close(m["auc"], 1.0) and close(m["ap"], 1.0) and close(m["auc_a_macro"], 1.0)
    # only the video with the event: rows (0.1, 0) (0.9, 1) (0.6, 0), the event row is on top
    assert close(m["auc_a_micro"], 1.0) and close(m["ap_a_micro"], 1.0)
    assert m["n_events"] == 1 and m["n_events_detected"] == 1 and close(m["event_recall"], 1.0)
    # mean = (AUC 1 + AUC_a 1 + row F1 2/3 + (1 - 1/3) + video F1 2/3 + event recall 1) / 6 = 5/6
    assert close(m["mean"], 5 / 6)


def test_grids():
    assert extents_from_indices(3, 8, 4, 10.0, "start", "") == [(0.0, 8.0), (4.0, 10.0), (8.0, 10.0)]
    assert extents_from_indices(3, 8, 4, 10.0, "centered", "") == [(0.0, 6.0), (2.0, 10.0), (6.0, 10.0)]


def test_explicit_extents_equal_to_indices():
    explicit = [{"start": 4.0 * j, "end": min(40.0, 4.0 * j + 8), "score": s} for j, s in enumerate(S_WITH_EVENT)]
    a, _ = build_entries({"a": WITH_EVENT}, scores({"a": S_WITH_EVENT}))
    b, _ = build_entries({"a": WITH_EVENT}, scores({"a": explicit}))
    assert a[0]["extents"] == b[0]["extents"] and a[0]["scores"] == b[0]["scores"]


def test_input_errors():
    cases = [
        lambda: build_entries({"a": WITH_EVENT, "b": NORMAL}, scores({"a": S_WITH_EVENT})),       # key without scores
        lambda: build_entries({"a": WITH_EVENT}, scores({"a": S_WITH_EVENT}, c=4, h=8)),          # asking rate > chunk
        lambda: build_entries({"a": {"duration": 10, "intervals": [[5, 3]]}}, scores({"a": [0.1]})),  # end < start
        lambda: build_entries({"a": NORMAL}, scores({"a": [0.1] * 6})),                            # more chunks than video
        lambda: build_entries({"a": {"duration": 20.0}}, scores({"a": S_NORMAL})),                 # intervals missing
        lambda: build_entries({"a": [20.0, []]}, scores({"a": S_NORMAL})),                         # label not an object
        lambda: build_entries({"a": NORMAL}, scores({"a": [0.1, float("nan"), 0.2, 0.3]})),        # score not finite
        lambda: evaluate_levels({"all": ({"a": WITH_EVENT}, scores({"a": S_WITH_EVENT}))}),        # neither validation nor threshold
        lambda: evaluate_levels({"all": ({"a": WITH_EVENT}, scores({"a": S_WITH_EVENT}))}, threshold=0.5, guard=4.0),  # guard < c
    ]
    for i, f in enumerate(cases):
        try:
            f()
        except InputError:
            continue
        raise AssertionError(f"wrong input number {i} accepted without error")


def test_no_judgeable_rows_is_a_clear_error():
    # two short chunks, [0,2] and [30,32], around the event [5,25), with a 60 s guard: no chunk overlaps
    # the event row, and both guard windows overlap the event, so every row is unjudgeable
    lab = {"a": {"duration": 32.0, "intervals": [[5.0, 25.0]]}}
    sc = scores({"a": [{"start": 0.0, "end": 2.0, "score": 0.5}, {"start": 30.0, "end": 32.0, "score": 0.5}]})
    try:
        evaluate_levels({"all": (lab, sc)}, threshold=0.5, guard=60.0)
    except InputError:
        return
    raise AssertionError("an evaluation without judgeable rows did not raise InputError")


# --------------------------------------------------------------------------- 2. brute force
def _random_data(rng, n=300):
    s = [round(rng.random(), 1) for _ in range(n)]            # many ties, on purpose
    y = [1 if rng.random() < 0.3 else 0 for _ in range(n)]
    return s, y


def _auc_pairs(s, y):
    pos = [a for a, b in zip(s, y) if b]
    neg = [a for a, b in zip(s, y) if not b]
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def _ap_thresholds(s, y):
    tot, prev_r, out = sum(y), 0.0, 0.0
    for th in sorted(set(s), reverse=True):
        tp = sum(1 for a, b in zip(s, y) if a >= th and b)
        fp = sum(1 for a, b in zip(s, y) if a >= th and not b)
        r = tp / tot
        out += (r - prev_r) * tp / (tp + fp)
        prev_r = r
    return out


def test_auc_ap_against_brute_force():
    rng = random.Random(1)
    for _ in range(20):
        s, y = _random_data(rng)
        assert close(M.auroc(s, y), _auc_pairs(s, y), 1e-12)
        assert close(M.ap(s, y), _ap_thresholds(s, y), 1e-12)


def test_youden_against_brute_force():
    rng = random.Random(2)
    for _ in range(20):
        s, y = _random_data(rng)
        P, N = sum(y), len(y) - sum(y)
        j = lambda th: (sum(1 for a, b in zip(s, y) if a >= th and b) / P
                        - sum(1 for a, b in zip(s, y) if a >= th and not b) / N)
        best = max(j(th) for th in set(s))
        assert close(j(M.youden_threshold(s, y)), best, 1e-12)


# --------------------------------------------------------------------------- 3. the script on the example data
def _read(name):
    return json.loads((DATA / name).read_text())


def _ovad():
    return evaluate_ovad(_read("test_labels.json"), _read("test_scores.json"),
                         _read("val_labels.json"), _read("val_scores.json"))


def _compare(obtained: dict, expected: dict, where: str):
    assert close(obtained["threshold"]["value"], expected["threshold"], 1e-12), f"{where}: threshold"
    for lv, metrics in expected["levels"].items():
        for k, x in metrics.items():
            y = obtained["levels"][lv][k]
            ok = (x is None and y != y) or (x is not None and close(float(y), float(x), 1e-9))
            assert ok, f"{where}, {lv}, {k}: obtained {y}, expected {x}"


def test_example_ovad():
    _compare(_ovad(), json.loads(EXPECTED.read_text()), "OVAD")


def test_triples_and_classes():
    # the class is checked but does not enter the metrics: changing it changes nothing
    lab, sc = _read("test_labels.json"), _read("test_scores.json")
    vlab, vsc = _read("val_labels.json"), _read("val_scores.json")
    a = evaluate_ovad(lab, sc, vlab, vsc)
    lab2 = {k: {**v, "anomalies": [[x[0], x[1], "Other"] for x in v["anomalies"]]} for k, v in lab.items()}
    b = evaluate_ovad(lab2, sc, vlab, vsc)
    assert a["threshold"]["value"] == b["threshold"]["value"]
    for k, x in a["levels"]["anomalies"].items():
        y = b["levels"]["anomalies"][k]
        assert (x != x and y != y) or x == y, k
    for wrong in ([10.0, 14.0], [10.0, 14.0, ""], [10.0, 14.0, 3]):
        try:
            evaluate_ovad({"v": {"duration": 40.0, "anomalies": [wrong]}}, scores({"v": S_WITH_EVENT}), threshold=0.5)
        except InputError:
            continue
        raise AssertionError(f"wrong triple accepted: {wrong}")


def test_command_line():
    # the command line gives the same numbers as the Python call, and writes valid JSON
    import contextlib, io, tempfile
    D = DATA
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "results.json"
        argv = ["--labels", str(D / "test_labels.json"), "--scores", str(D / "test_scores.json"),
                "--val-labels", str(D / "val_labels.json"), "--val-scores", str(D / "val_scores.json"),
                "--output", str(out)]
        with contextlib.redirect_stdout(io.StringIO()):
            assert main_ovad(argv) == 0
        written = json.loads(out.read_text())
    assert written == to_json(_ovad())
# --------------------------------------------------------------------------- direct execution
def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, f in tests:
        try:
            f()
            print(f"  ok        {name}")
        except Exception as e:                                # noqa: BLE001 - report and carry on
            failed += 1
            print(f"  FAILED    {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed} of {len(tests)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
