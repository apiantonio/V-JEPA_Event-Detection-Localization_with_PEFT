"""OVAD rows and emitting chunks of the class-routing localization detectors (CPU, local).

Decisions of 30 October 2026 (``docs/varie/metriche_localizzazione_1fps_fpr_ovad_20261030.md``):
the boxes are shown on the chunks with ``h(c) >= θ_O``, the Youden threshold of the OVAD rows
with guard 30 s (D1, also for the online AP), and the FP metric reads the frames of the OVAD FP
rows through the clean chunks only (D2). This script gathers, per detector, what the cluster
steps of ``localization_class_ap.py`` need and do not recompute: θ_O and, on both test sets,
every row with its label, judgeability and prediction, and every chunk with its extent, head
score and guard condition. Rows and guard come from the delivered protocol code
(``ovad_protocol/protocol_core/metrics.py``): the guard window of a chunk is its extent widened
by ``(g - c)/2`` on each side and clipped to the video, as ``row_table`` does, and every row of
``righe.json`` is recomputed here from the same entries and must match.

A detector of the ``ovad_guard30`` collection is read from there; the others (the B3 detectors,
outside the model catalogue) are evaluated with the functions of ``valuta_ovad.py`` (verified
export, Youden on the judgeable validation rows, row tables) into
``<campaign>/plan/ovad_guard30/<run_id>/``, in the same layout. Writes
``<campaign>/plan/ovad_rows/<detector>.json``.

Run from ``evd-vfm/``: ``python scripts/localization_class_ovad_plan.py``.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "ovad_protocol"), str(ROOT / "scripts")]

from evaluate_ovad import from_triples  # noqa: E402
from protocol_core import metrics as M  # noqa: E402
from protocol_core.inputs import build_entries  # noqa: E402

from evd.utils.hashing import file_sha256  # noqa: E402
from evd.utils.io import read_json, write_json  # noqa: E402

CAMPAIGN = ROOT / "runs/localization-class-routing-20261021"
COLLECTION = ROOT / "runs/classifiche/ovad_guard30"
#: Guard of the official OVAD collection (decision of 28 September 2026).
GUARD = 30.0
#: The detectors of the localization stages (``localization_class_table.DETECTORS``).
DETECTORS = {
    "campaign": "e3s_ucf_vjepa21l_vptshallow-n8-su-lora-r32_s42",
    "b3s": "lcb3s_ucf_vjepa21l_vptshallow-n8-su-lora-r32_s42",
    "b3g": "lcb3g_ucf_vjepa21l_vptshallow-n8-su-lora-r32_s42",
    "attentive": "e2d_ucf_vjepa21l_frozen-attentive_s42",
}
#: Data overrides of the OVAD collections (``runs/classifiche/ovad/manifest.json``).
SET_UCF = [
    "root=data/raw/ucf_crime/UCF_Crimes",
    "inventory=data/ucf_crime_inventory/inventory.csv",
    "annotations.path=data/annotations/fs_ucf_crime",
    "splits_dir=data/raw/ucf_crime/UCF_Crimes/Anomaly_Detection_splits",
]
SET_MIVIA = [
    "root=data/mivia_crime",
    "inventory=data/mivia_crime_inventory/inventory.csv",
    "annotations.path=data/mivia_crime",
]
MANIFEST = ROOT / "manifests/edp_v2/ucf_crime/split.json"
#: OVAD folder of a test set → dataset key of the localization stages.
DATASETS = {"ucf": "ucf_test", "mivia": "mivia_test"}


def evaluate_cell(run: Path, out: Path) -> Path:
    """OVAD rows of a run outside the collection, with the collection's own functions.

    :return: the cell folder, ``val/``, ``ucf/`` and ``mivia/`` as in the collection.
    :raises ValueError: as ``valuta_ovad.py``: a threshold or a count that does not reproduce.
    """
    from evaluate_ovad import evaluate_ovad

    from valuta_ovad import details, save_json, validated_threshold, verified_export

    base = out / run.name
    verified_export(run, base / "val", "val", SET_UCF, MANIFEST)
    val_labels = read_json(base / "val/labels.json")
    val_scores = read_json(base / "val/scores.json")
    theta = validated_threshold(val_labels, val_scores, GUARD)
    counts = details(val_labels, val_scores, theta, GUARD, base / "val")
    save_json(
        base / "val/soglia.json",
        {
            "threshold": theta,
            "source": "validation: Youden point on the judgeable rows",
            "conteggi": counts,
        },
    )
    for folder, source, overrides in (
        ("ucf", run, SET_UCF),
        ("mivia", run / f"{run.name}__mivia", SET_MIVIA),
    ):
        dest = base / folder
        verified_export(source, dest / "input", "test", overrides, MANIFEST)
        labels = read_json(dest / "input/labels.json")
        scores = read_json(dest / "input/scores.json")
        if set(labels) & set(val_labels):
            raise ValueError("Videos shared between validation and test: check the splits.")
        result = evaluate_ovad(labels, scores, val_labels, val_scores, guard=GUARD)
        if result["threshold"]["value"] != theta:
            raise ValueError("The test threshold does not match the validation one.")
        found = details(labels, scores, theta, GUARD, dest)
        metrics = result["levels"]["anomalies"]
        if any(metrics[k] != v for k, v in found.items()):
            raise ValueError("The intermediate tables do not reproduce the OVAD counts.")
        save_json(dest / "results.json", result)
    return base


def cell_of(run_id: str, runs: Path, out: Path) -> Path:
    """:return: the OVAD g30 cell of a run: the collection's, else one evaluated here."""
    cell = COLLECTION / "celle" / run_id
    if all((cell / d / "righe.json").exists() for d in ("val", "ucf", "mivia")):
        return cell
    done = out / run_id
    if all((done / d / "righe.json").exists() for d in ("val", "ucf", "mivia")):
        return done
    return evaluate_cell(runs / run_id, out)


def video_plan(entry: dict, rows: list[dict], chunk: float) -> dict:
    """Extents, scores, guard condition and rows of one video, checked against ``righe.json``.

    :param entry: the OVAD entry (``build_entries``): extents sorted by start, clipped to the
        video, merged events, duration.
    :param rows: the video's rows of ``righe.json``, in order.
    :param chunk: the declared chunk size ``c``.
    :return: ``{"duration", "extents", "scores", "clean", "events", "rows"}``; a row is
        ``[start, end, label, judgeable, prediction]``, prediction ``None`` when not judgeable.
    :raises ValueError: if the protocol's rows differ from ``righe.json``.
    """
    widen = (GUARD - chunk) / 2
    duration, events = entry["duration"], entry["events"]
    clean = [
        not M.overlaps(max(0.0, a - widen), min(duration, b + widen), events)
        for a, b in entry["extents"]
    ]
    bounds = M.make_rows(duration, events)
    table = M.row_table(entry, c=chunk, g=GUARD)
    if len(bounds) != len(rows):
        raise ValueError(
            f"{entry['key']}: {len(rows)} rows on disk, the protocol gives {len(bounds)}"
        )
    out = []
    for (a, b, label), (score, judged), row in zip(bounds, table, rows, strict=True):
        same = (
            math.isclose(a, row["start"], abs_tol=1e-9)
            and math.isclose(b, row["end"], abs_tol=1e-9)
            and label == row["label"]
            and (judged >= 0) == row["judgeable"]
            and (judged < 0 or score == row["score"])
        )
        if not same:
            raise ValueError(f"{entry['key']} row {row['row_id']}: differs from the protocol")
        out.append([a, b, label, row["judgeable"], row["prediction"]])
    return {
        "duration": duration,
        "extents": [[a, b] for a, b in entry["extents"]],
        "scores": entry["scores"],
        "clean": clean,
        "events": [[a, b] for a, b in events],
        "rows": out,
    }


def detector_plan(detector: str, run_id: str, cell: Path, runs: Path) -> dict:
    """:return: θ_O, the chunk threshold τ and, per test set, every video's :func:`video_plan`.

    :raises ValueError: if the test results carry another threshold than the validation one.
    """
    theta = read_json(cell / "val/soglia.json")["threshold"]
    tau = read_json(runs / run_id / "metrics.json")["threshold"]["value"]
    datasets, sources = {}, {"soglia.json": file_sha256(cell / "val/soglia.json")}
    for folder, key in DATASETS.items():
        result = read_json(cell / folder / "results.json")
        if result["threshold"]["value"] != theta:
            raise ValueError(f"{run_id} {folder}: threshold differs from the validation one")
        labels = read_json(cell / folder / "input/labels.json")
        scores = read_json(cell / folder / "input/scores.json")
        entries, hp = build_entries(from_triples(labels, folder), scores)
        by_video: dict[str, list[dict]] = {}
        for row in read_json(cell / folder / "righe.json"):
            by_video.setdefault(row["video_id"], []).append(row)
        videos = {
            e["key"]: video_plan(e, by_video[e["key"]], float(hp["chunk_size"])) for e in entries
        }
        rows = [r for v in videos.values() for r in v["rows"]]
        datasets[key] = {
            "hyperparameters": hp,
            "counts": {
                "videos": len(videos),
                "rows_tp": sum(r[2] == 1 and r[3] and r[4] == 1 for r in rows),
                "rows_fn": sum(r[2] == 1 and r[3] and r[4] == 0 for r in rows),
                "rows_fp": sum(r[2] == 0 and r[3] and r[4] == 1 for r in rows),
                "rows_tn": sum(r[2] == 0 and r[3] and r[4] == 0 for r in rows),
                "chunks_emitting": sum(s >= theta for v in videos.values() for s in v["scores"]),
            },
            "videos": videos,
        }
        for name in ("righe.json", "input/labels.json", "input/scores.json"):
            sources[f"{folder}/{name}"] = file_sha256(cell / folder / name)
    return {
        "detector": detector,
        "run_id": run_id,
        "theta_ovad": theta,
        "theta_chunks": tau,
        "emitting_rule": "h(c) >= theta_ovad",
        "theta_ovad_above_chunks": theta >= tau,
        "guard": GUARD,
        "cell": str(cell.relative_to(ROOT)),
        "sources_sha256": sources,
        "datasets": datasets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN)
    parser.add_argument("--detectors", nargs="*", default=list(DETECTORS))
    args = parser.parse_args()
    for detector in args.detectors:
        run_id = DETECTORS[detector]
        cell = cell_of(run_id, args.runs, args.campaign / "plan/ovad_guard30")
        plan = detector_plan(detector, run_id, cell, args.runs)
        write_json(args.campaign / "plan/ovad_rows" / f"{detector}.json", plan)
        counts = {k: v["counts"] for k, v in plan["datasets"].items()}
        print(detector, f"theta_O={plan['theta_ovad']:.4f} tau={plan['theta_chunks']:.4f}", counts)


if __name__ == "__main__":
    main()
