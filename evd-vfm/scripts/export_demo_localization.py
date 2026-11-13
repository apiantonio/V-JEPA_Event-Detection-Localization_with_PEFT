"""Export sealed localization boxes for the demonstrator, projected onto native frames.

The localization campaign seals, per chunk and per method, one list of boxes per temporal
plane in native inclusive pixels. This script turns them into per-frame segments with the
``fine`` projection of the evaluator (``localization_auto_evaluate.evaluate_video``) and of
``render_proposal_boxes.py``, and stores them with the mean and per-class IoU of every
evaluated experiment under ``runs/<run_id>/localization/``, where the demonstrator reads
them (``apps/demo/localization.py``). No model is loaded.

The detector identity is verified, not assumed: the score of every localized chunk must
equal the run's ``scores.jsonl`` for the same window, or the boxes belong to another model.

    python scripts/export_demo_localization.py \
        --run runs/e3s_ucf_vjepa21l_vptshallow-n8-su-lora-r32_s42 \
        --predictions ../tmp/localization-auto-20261016/pulled_annotated/main/predictions_annotated \
        --report ../tmp/localization-auto-20261016/report_annotated \
        --campaign localization-auto-20261016

The class-routing campaign (21-24 October 2026) seals its boxes in the same layout but
evaluates them into per-experiment JSON results instead of comparison tables, and keeps the
chunk scores in its IG passes. ``--evaluation`` reads those results for the experiments named
by ``--method ID=NAME`` (the readable name shown instead of the campaign acronym), ``--passes``
supplies the chunk scores of the check, and ``--append`` adds the methods to the export the
run already has instead of replacing it:

    python scripts/export_demo_localization.py \
        --run runs/e3s_ucf_vjepa21l_vptshallow-n8-su-lora-r32_s42 \
        --predictions <campaign>/stages/campaign-heads \
        --evaluation <campaign>/evaluation/campaign-heads \
        --passes <campaign>/passes/campaign/test \
        --method "B2-A2-phase1=MultiClass-Full→Concepts-AllChunks, 55% threshold" \
        --campaign localization-class-routing-20261021 --append

The official localization table (``localization_class_table.py``) ranks the box rules by
offline AP, and an AP stage seals the material of the boxes instead of the boxes.
``--official-table`` draws from that material the rule a step of the table names as the best of
the detector, and makes it the method the page opens with. AP has no operating point of its own,
so the rule enters the menu twice: with the boxes above the confidence of TPR 90%, the default,
and with all of them. Before anything is written the per-event AP is recomputed from the stage
and compared with the evaluation the table was built from:

    python scripts/export_demo_localization.py \
        --run runs/e3s_ucf_vjepa21l_vptshallow-n8-su-lora-r32_s42 \
        --official-table <campaign>/evaluation/tabella_localizzazione.json \
        --official-step ours-best --official-stage <campaign>/stages/ign-n32-campaign \
        --official-name "MultiClass-Branch→Concepts-AllChunks, largest 4-conn. component" \
        --intervals <campaign>/plan/real_use/intervals.json \
        --ground-truth <campaign>/evaluation-only/gt-snapshot \
        --campaign localization-class-routing-20261021
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from evd.eval.localization import aggregate_localization  # noqa: E402
from evd.exp.localization_auto import FAMILY_LABELS  # noqa: E402
from evd.explain.project import offline_assignment, offline_plane  # noqa: E402
from evd.types import WindowSpec  # noqa: E402
from evd.utils.hashing import file_sha256  # noqa: E402
from evd.utils.io import read_json, read_jsonl, write_json, write_text_atomic  # noqa: E402

#: Prediction folders of the campaign and the dataset each one describes.
DATASETS = {"ucf_test": "ucf_crime", "mivia_test": "mivia_crime"}

#: Leading CSV columns; every column after them is a class IoU.
CSV_FIELDS = ("rank", "experiment", "family", "macro_iou", "fit_s", "inference_s_per_chunk")

#: Largest score difference accepted between a localized chunk and the run: the two come
#: from the same bf16 forward, and on the campaign detectors they agree within 6e-8. A run
#: scored on another GPU model differs more (2.6e-3 for the attentive probe scored on an
#: A100 and localized on an H100): ``--score-tolerance`` raises it, and the export says so.
SCORE_TOLERANCE = 1e-6

#: Folder of the run that receives the export.
OUTPUT_FOLDER = "localization"

#: Family under which the class-routing experiments appear in the demonstrator menu.
CLASS_ROUTING_FAMILY = ("class_routing", "Class routing (21-24 Oct 2026)")

#: Largest gap accepted between an evaluator result's ``value`` and the mean of its classes.
MEAN_TOLERANCE = 1e-12

#: Family of the rule the official table names as the best of the detector.
OFFICIAL_FAMILY = ("official_table", "Official table (offline AP)")

#: Node set of the official table: AP and false alarms at 1 fps, IG with 32 nodes.
OFFICIAL_SET = "1fps"

#: GT reading of the official table.
OFFICIAL_GT = "union"

#: Share of the annotated frames that keep a box at the confidence of the default method.
OFFICIAL_TPR = 90

#: Largest gap accepted between the recomputed per-event AP and the evaluation on disk.
AP_TOLERANCE = 1e-9

#: Confidence grid of the F1 of the official rule, for a confidence in ``[0, 1]``.
F1_CONFIDENCES = np.round(np.arange(0, 1.0005, 0.005), 3)

#: The dataset that chooses the confidence of the highest F1; the other one receives it, as the
#: chunk threshold moves from UCF-Crime to MIVIA-Crime.
F1_SOURCE = "ucf_test"

#: The table rounds its values to seven digits.
TABLE_TOLERANCE = 1e-6

#: Folder of the export holding the IoU results of the official method.
OFFICIAL_FOLDER = "official"


def read_comparison(report: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """Read the UCF-Crime and MIVIA-Crime rankings of one detector.

    :param report: folder with ``comparison_ucf_test.csv`` and ``comparison_mivia_test.csv``.
    :return: the class names, and one entry per experiment ordered by UCF-Crime rank (ties
        keep the CSV order), with mean and per-class IoU on both datasets.
    :raises ValueError: if the two files list different experiments or classes.
    """
    tables = {}
    classes = None
    for folder in DATASETS:
        with (report / f"comparison_{folder}.csv").open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            header = list(reader.fieldnames or [])
            if tuple(header[: len(CSV_FIELDS)]) != CSV_FIELDS:
                raise ValueError(f"Unexpected columns in comparison_{folder}.csv: {header}")
            found = header[len(CSV_FIELDS) :]
            if classes is not None and found != classes:
                raise ValueError("The two comparisons list different classes")
            classes = found
            tables[folder] = list(reader)
    names = {folder: [r["experiment"] for r in rows] for folder, rows in tables.items()}
    if set(names["ucf_test"]) != set(names["mivia_test"]) or len(set(names["ucf_test"])) != len(
        names["ucf_test"]
    ):
        raise ValueError("The two comparisons do not list the same experiments once each")
    mivia = {r["experiment"]: r for r in tables["mivia_test"]}
    methods = []
    for row in sorted(tables["ucf_test"], key=lambda r: int(r["rank"])):
        other = mivia[row["experiment"]]
        methods.append(
            {
                "id": row["experiment"],
                "family": row["family"],
                "family_label": FAMILY_LABELS.get(row["family"], row["family"]),
                "fit_s": float(row["fit_s"]),
                "inference_s_per_chunk": float(row["inference_s_per_chunk"]),
                **{
                    DATASETS[folder]: {
                        "rank": int(source["rank"]),
                        "mean": float(source["macro_iou"]),
                        "classes": {c: float(source[c]) for c in classes},
                    }
                    for folder, source in (("ucf_test", row), ("mivia_test", other))
                },
            }
        )
    return list(classes), methods


def read_evaluation(
    evaluation: Path, names: Mapping[str, str]
) -> tuple[list[str], list[dict[str, Any]]]:
    """Read the class-routing evaluator results of the chosen experiments of one stage.

    :param evaluation: stage folder of the evaluator, with
        ``results/<dataset folder>/<experiment>.json``.
    :param names: experiment id to the readable name shown in its place, in output order.
    :return: the class names and one entry per experiment in the :func:`read_comparison`
        shape, plus ``label`` and, per dataset, ``events`` (event id to IoU) and the
        digests of the predictions and ground truth it was computed on. Ranks are those
        among the given experiments, by decreasing mean.
    :raises ValueError: if a result is missing, its mean is not the mean of its classes, or
        the datasets list different classes.
    """
    classes: list[str] | None = None
    methods = []
    for method, label in names.items():
        entry: dict[str, Any] = {
            "id": method,
            "label": label,
            "family": CLASS_ROUTING_FAMILY[0],
            "family_label": CLASS_ROUTING_FAMILY[1],
        }
        for folder, dataset in DATASETS.items():
            path = evaluation / "results" / folder / f"{method}.json"
            if not path.is_file():
                raise ValueError(f"No evaluator result for {method} on {folder}: {path}")
            result = read_json(path)
            per_class = {r["class"]: float(r["iou"]) for r in result["per_class"]}
            if classes is not None and list(per_class) != classes:
                raise ValueError(f"{path} lists other classes than the first result")
            classes = list(per_class)
            mean = sum(per_class.values()) / len(per_class)
            if abs(mean - float(result["value"])) > MEAN_TOLERANCE:
                raise ValueError(f"{path}: value {result['value']} is not the class mean {mean}")
            entry[dataset] = {
                "mean": float(result["value"]),
                "classes": per_class,
                "events": {e["event_id"]: float(e["iou"]) for e in result["per_event"]},
                "prediction_seal_sha256": result["prediction_seal_sha256"],
                "gt_snapshot_sha256": result["gt_snapshot_sha256"],
            }
            if folder == "ucf_test" and "timing" in result:
                entry["fit_s"] = float(result["timing"]["fit_seconds"])
        methods.append(entry)
    if classes is None:
        raise ValueError("No experiment to read: give at least one --method")
    rank_methods(methods)
    return classes, methods


def rank_methods(methods: Sequence[dict[str, Any]]) -> None:
    """Number the methods by decreasing mean IoU on each dataset, in place.

    Competition ranks, as the comparison tables: equal means share a rank.

    :param methods: entries with ``ucf_crime`` and ``mivia_crime`` blocks carrying ``mean``.
    """
    for dataset in DATASETS.values():
        means = [m[dataset]["mean"] for m in methods]
        for method, mean in zip(methods, means, strict=True):
            method[dataset]["rank"] = 1 + sum(other > mean for other in means)


def pass_scores(passes: Path) -> dict[str, float]:
    """Read the detector score of every chunk of an IG pass of the class-routing campaign.

    :param passes: folder with one JSON record per chunk (``window_id``, binary ``logit``).
    :return: ``window_id`` to sigmoid of the logit, the quantity ``scores.jsonl`` stores.
    :raises ValueError: if the folder holds no record or a window appears twice.
    """
    scores: dict[str, float] = {}
    for path in sorted(passes.glob("*.json")):
        record = read_json(path)
        if record["window_id"] in scores:
            raise ValueError(f"{record['window_id']} appears twice in {passes}")
        scores[record["window_id"]] = 1.0 / (1.0 + math.exp(-float(record["logit"])))
    if not scores:
        raise ValueError(f"No pass record in {passes}")
    return scores


def find_stages(predictions: Path, methods: Sequence[str]) -> dict[str, dict[str, Path]]:
    """Locate, for every dataset, the sealed stage holding each experiment.

    :param predictions: root of the pulled ``predictions_annotated`` layout, or a single
        stage folder (one with ``operations/``).
    :param methods: experiments to export.
    :return: dataset folder to ``{experiment: stage folder}``.
    :raises ValueError: if a seal is incomplete, or an experiment is missing or sealed twice.
    """
    located: dict[str, dict[str, Path]] = {folder: {} for folder in DATASETS}
    stages = (
        [predictions]
        if (predictions / "operations").is_dir()
        else sorted(p for p in predictions.iterdir() if p.is_dir())
    )
    for stage in stages:
        for folder in DATASETS:
            seal_path = stage / "operations" / f"seal-{folder}.json"
            if not seal_path.exists():
                continue
            seal = read_json(seal_path)
            if seal.get("status") != "complete":
                raise ValueError(f"Incomplete seal: {seal_path}")
            for method in seal["methods"]:
                if method not in methods:
                    continue
                if method in located[folder]:
                    raise ValueError(f"{method} is sealed by two stages on {folder}")
                located[folder][method] = stage
    for folder, found in located.items():
        missing = [m for m in methods if m not in found]
        if missing:
            raise ValueError(f"No sealed boxes on {folder} for: {missing}")
    return located


def run_scores(run: Path, folder: str) -> dict[str, float] | None:
    """Read the window scores the run wrote for one dataset.

    :param run: the run folder; MIVIA-Crime is the nested external evaluation
        ``<run_id>__mivia*`` (ADR-066).
    :param folder: dataset folder of the campaign.
    :return: ``window_id`` to score, or ``None`` if the run has no scores for that dataset.
    """
    if folder == "ucf_test":
        path = run / "scores.jsonl"
    else:
        nested = sorted(run.glob(f"{run.name}__mivia*/scores.jsonl"))
        path = nested[0] if nested else run / "missing"
    if not path.exists():
        return None
    return {str(r["window_id"]): float(r["score"]) for r in read_jsonl(path)}


def project_boxes(
    windows: Sequence[WindowSpec],
    num_frames: int,
    planes: Sequence[int],
    boxes: Mapping[str, Sequence[Sequence[Sequence[Sequence[int]]] | None]],
) -> tuple[list[list[list[int]]], dict[str, list[list[int]]]]:
    """Project per-plane chunk boxes onto native frames, as the evaluator does.

    Each frame goes to the window closing first among those containing it (``fine``) and,
    inside it, to the plane of the nearest sample; the boxes of that plane are the boxes of
    the frame. Consecutive frames with the same boxes become one segment.

    :param windows: the sampled windows of the video, in chunk order.
    :param num_frames: native frames of the video.
    :param planes: temporal planes of each chunk's token grid.
    :param boxes: per experiment and per chunk, the box lists of every plane in native
        inclusive ``[x0, y0, x1, y1]`` pixels; ``None`` for a chunk that was not localized.
    :return: a shared table of distinct box lists, and per experiment the inclusive
        ``[start, end, table index]`` segments, in frame order. Frames whose chunk was not
        localized belong to no segment.
    :raises ValueError: if a chunk's plane count differs from its grid.
    """
    assignment = offline_assignment(windows, num_frames, politica="fine")
    # One run per (chunk, plane): the box lists only change there.
    runs: list[list[int]] = []
    for frame in range(num_frames):
        index = int(assignment[frame])
        if index < 0:
            continue
        plane = offline_plane(windows[index], frame, planes[index])
        if runs and runs[-1][1] == frame - 1 and runs[-1][2:] == [index, plane]:
            runs[-1][1] = frame
        else:
            runs.append([frame, frame, index, plane])
    table: list[list[list[int]]] = []
    refs: dict[str, int] = {}
    segments: dict[str, list[list[int]]] = {}
    for method, per_chunk in boxes.items():
        rows: list[list[int]] = []
        for start, end, index, plane in runs:
            chunk = per_chunk[index]
            if chunk is None:
                continue
            if len(chunk) != planes[index]:
                raise ValueError(
                    f"{method}: chunk {index} has {len(chunk)} planes, not {planes[index]}"
                )
            value = [list(map(int, b)) for b in chunk[plane]]
            key = json.dumps(value)
            if key not in refs:
                refs[key] = len(table)
                table.append(value)
            ref = refs[key]
            if rows and rows[-1][1] == start - 1 and rows[-1][2] == ref:
                rows[-1][1] = end
            else:
                rows.append([start, end, ref])
        segments[method] = rows
    return table, segments


def _sealed_records(stage: Path, entry: dict[str, Any], *, localized_only: bool) -> tuple:
    """Read one sealed video prediction, verifying every file against its seal.

    :param stage: stage folder.
    :param entry: the video row of the stage seal.
    :param localized_only: skip reading chunks marked as not localized.
    :return: video metadata, chunk index entries and records (``None`` where skipped).
    :raises ValueError: if a file changed after sealing.
    """
    target = stage / entry["directory"]
    if file_sha256(target / "complete.json") != entry["complete_sha256"]:
        raise ValueError(f"Prediction changed after sealing: {target}")
    complete = read_json(target / "complete.json")
    records = []
    for chunk in complete["chunks"]:
        path = target / chunk["path"]
        if file_sha256(path) != chunk["sha256"]:
            raise ValueError(f"Prediction chunk changed after sealing: {path}")
        record = read_json(path)
        records.append(record if record.get("localized") or not localized_only else None)
    return complete["video"], records


def export(
    run: Path,
    predictions: Path,
    report: Path,
    campaign: str,
    output: Path | None = None,
    *,
    evaluated: tuple[list[str], list[dict[str, Any]]] | None = None,
    chunk_scores: Mapping[str, float] | None = None,
    score_tolerance: float = SCORE_TOLERANCE,
) -> dict[str, Any]:
    """Write the demonstrator localization of one run.

    :param run: the run whose model produced the boxes.
    :param predictions: pulled ``predictions_annotated`` root of that detector, or one
        stage folder.
    :param report: comparison folder of that detector, or the evaluator stage folder when
        ``evaluated`` is given.
    :param campaign: provenance label of the campaign.
    :param output: destination, ``<run>/localization`` by default.
    :param evaluated: classes and methods from :func:`read_evaluation`, in place of the
        comparison tables of ``report``.
    :param chunk_scores: ``window_id`` to detector score, for predictions whose chunk
        records carry no score (the class-routing campaign, :func:`pass_scores`).
    :param score_tolerance: largest accepted score difference between a localized chunk and
        the run (:data:`SCORE_TOLERANCE` unless the two were scored on different GPUs).
    :return: the written index.
    :raises ValueError: on missing or modified predictions, inconsistent stages, chunk
        scores that do not match the run, or evaluator results computed on other
        predictions than the sealed ones.
    """
    output = output or run / OUTPUT_FOLDER
    classes, methods = evaluated if evaluated is not None else read_comparison(report)
    names = [m["id"] for m in methods]
    stages = find_stages(predictions, names)
    videos: dict[str, str] = {}
    checks: dict[str, Any] = {}
    seals: dict[str, str] = {}
    for folder, dataset in DATASETS.items():
        by_stage: dict[Path, list[str]] = {}
        for method, stage in stages[folder].items():
            by_stage.setdefault(stage, []).append(method)
        entries = {}
        for stage in by_stage:
            seal_path = stage / "operations" / f"seal-{folder}.json"
            seals[f"{stage.name}/{folder}"] = file_sha256(seal_path)
            entries[stage] = {v["video_id"]: v for v in read_json(seal_path)["videos"]}
        reference = min(by_stage)
        expected = run_scores(run, folder)
        compared, worst = 0, 0.0
        for video_id, entry in sorted(entries[reference].items()):
            meta, base = _sealed_records(reference, entry, localized_only=False)
            if not any(r.get("localized") for r in base):
                continue
            windows = [
                WindowSpec(
                    **{
                        **r["window"],
                        "explicit_frame_indices": tuple(r["window"]["explicit_frame_indices"]),
                    }
                )
                for r in base
            ]
            boxes: dict[str, list] = {}
            for stage, members in by_stage.items():
                if stage == reference:
                    records = base
                else:
                    _, records = _sealed_records(
                        stage, entries[stage][video_id], localized_only=True
                    )
                    if len(records) != len(base) or any(
                        (r is not None) != bool(b.get("localized"))
                        or (r is not None and r["window_id"] != b["window_id"])
                        for r, b in zip(records, base, strict=True)
                    ):
                        raise ValueError(f"{stage.name} disagrees on the chunks of {video_id}")
                for method in members:
                    boxes[method] = [
                        r["methods"][method]["boxes"]
                        if r is not None and r.get("localized")
                        else None
                        for r in records
                    ]
            if expected is not None:
                for record in base:
                    if record.get("localized") and record["window_id"] in expected:
                        compared += 1
                        score = _chunk_score(record, chunk_scores)
                        worst = max(worst, abs(expected[record["window_id"]] - score))
            table, segments = project_boxes(
                windows, meta["num_frames"], [r["grid"][0] for r in base], boxes
            )
            if video_id in videos:
                raise ValueError(f"{video_id} appears in two datasets")
            relative = f"{dataset}/{video_id}.json"
            body = {
                "video": {k: meta[k] for k in ("video_id", "num_frames", "fps", "width", "height")},
                "dataset": dataset,
                "boxes": table,
                "segments": segments,
            }
            # Compact: the demonstrator embeds these files in the page.
            write_text_atomic(output / relative, json.dumps(body, separators=(",", ":")) + "\n")
            videos[video_id] = relative
        if expected is not None and (compared == 0 or worst > score_tolerance):
            raise ValueError(
                f"The localized chunk scores on {folder} do not match {run.name} "
                f"({compared} windows, largest difference {worst:.3g}): the boxes belong to "
                "another detector"
            )
        checks[dataset] = (
            {
                "compared_windows": compared,
                "max_abs_score_difference": worst,
                "tolerance": score_tolerance,
            }
            if expected is not None
            else {"compared_windows": 0, "note": "the run has no scores for this dataset"}
        )
    if evaluated is not None:
        for method in methods:
            for folder, dataset in DATASETS.items():
                sealed = seals[f"{stages[folder][method['id']].name}/{folder}"]
                if method[dataset]["prediction_seal_sha256"] != sealed:
                    raise ValueError(
                        f"{method['id']} on {folder} was evaluated on other predictions than "
                        "the sealed boxes"
                    )
        digests = {
            f"{folder}/{m['id']}": file_sha256(report / "results" / folder / f"{m['id']}.json")
            for folder in DATASETS
            for m in methods
        }
    else:
        digests = {f: file_sha256(report / f"comparison_{f}.csv") for f in DATASETS}
    index = {
        "schema": 1,
        "run_id": run.name,
        "campaign": campaign,
        "metric": "IoU per annotated plane, then mean per event, per class, and over the classes",
        "selection": "best UCF-Crime test macro IoU (exploratory: chosen on the test set)",
        "default": min(methods, key=lambda m: m["ucf_crime"]["rank"])["id"],
        "classes": classes,
        "coordinate_space": "native_pixels_xyxy_inclusive",
        "projection": "fine",
        "methods": [
            {**m, "stage": stages["ucf_test"][m["id"]].name}
            | ({"campaign": campaign} if evaluated is not None else {})
            for m in methods
        ],
        "videos": videos,
        "detector_check": checks,
        "sources": {
            "predictions": str(predictions.resolve()),
            "report": str(report.resolve()),
            ("results_sha256" if evaluated is not None else "comparison_sha256"): digests,
            "seal_sha256": seals,
        },
    }
    write_json(output / "index.json", index)
    return index


def _chunk_score(record: dict[str, Any], chunk_scores: Mapping[str, float] | None) -> float:
    """:return: the detector score of a localized chunk, from its record or from the passes.

    :raises ValueError: if the passes have no score for the chunk.
    """
    if chunk_scores is None:
        return float(record["score"])
    if record["window_id"] not in chunk_scores:
        raise ValueError(f"The passes have no score for {record['window_id']}")
    return chunk_scores[record["window_id"]]


def append_export(output: Path, addition: Path) -> dict[str, Any]:
    """Add the methods of a second export of the same run to the one in ``output``.

    Box tables are merged video by video, ranks are recomputed over all methods and the
    default follows the selection rule again (best UCF-Crime mean, unless the official table
    chose it: :func:`export_official`). The first campaign stays the index ``campaign``; the
    added one is recorded under ``appended`` and on each added method.

    :param output: the ``localization`` folder that receives the methods.
    :param addition: a complete export of the same run, written elsewhere.
    :return: the updated index.
    :raises ValueError: if the two exports describe another run, other classes or another
        geometry, share a method id, or were scored against different ground truths.
    """
    index = read_json(output / "index.json")
    extra = read_json(addition / "index.json")
    for name in ("run_id", "classes", "coordinate_space", "projection", "metric"):
        if index[name] != extra[name]:
            raise ValueError(f"The two exports disagree on {name}")
    clash = {m["id"] for m in index["methods"]} & {m["id"] for m in extra["methods"]}
    if clash:
        raise ValueError(f"Method ids already exported: {sorted(clash)}")
    truth = (index.get("ground_truth") or {}).get("snapshot_sha256")
    for method in extra["methods"]:
        for dataset in DATASETS.values():
            digest = method[dataset].get("gt_snapshot_sha256")
            if truth and digest and digest != truth:
                raise ValueError(f"{method['id']} was scored on another ground truth")
    for video_id, relative in extra["videos"].items():
        body = read_json(addition / relative)
        if video_id not in index["videos"]:
            write_text_atomic(output / relative, json.dumps(body, separators=(",", ":")) + "\n")
            index["videos"][video_id] = relative
            continue
        if index["videos"][video_id] != relative:
            raise ValueError(f"{video_id} is filed under another dataset in the two exports")
        merged = read_json(output / relative)
        if merged["video"] != body["video"] or merged["dataset"] != body["dataset"]:
            raise ValueError(f"{video_id}: the two exports describe different videos")
        refs = {json.dumps(b): i for i, b in enumerate(merged["boxes"])}
        for method, rows in body["segments"].items():
            moved = []
            for start, end, ref in rows:
                key = json.dumps(body["boxes"][ref])
                if key not in refs:
                    refs[key] = len(merged["boxes"])
                    merged["boxes"].append(body["boxes"][ref])
                moved.append([start, end, refs[key]])
            merged["segments"][method] = moved
        write_text_atomic(output / relative, json.dumps(merged, separators=(",", ":")) + "\n")
    index["methods"] += extra["methods"]
    rank_methods(index["methods"])
    if "official_best" not in index:
        index["default"] = min(index["methods"], key=lambda m: m["ucf_crime"]["rank"])["id"]
    index.setdefault("appended", []).append(
        {
            "campaign": extra["campaign"],
            "methods": [m["id"] for m in extra["methods"]],
            "detector_check": extra["detector_check"],
            "sources": extra["sources"],
        }
    )
    write_json(output / "index.json", index)
    return index


def truth_segments(
    events: Sequence[dict[str, Any]], num_frames: int
) -> tuple[list[list[list[int]]], list[list[int]]]:
    """Flatten the ground-truth planes of a video into per-frame box segments.

    Events of different classes can overlap in time: on a shared frame the boxes of both
    are drawn, sorted, as one list.

    :param events: snapshot events, each with ``planes`` of inclusive ``segments``
        (``start``, ``end``, ``boxes`` in native inclusive pixels).
    :param num_frames: native frames of the video.
    :return: a table of distinct box lists and the inclusive ``[start, end, index]``
        segments, in frame order; frames without a reviewed box belong to no segment.
    :raises ValueError: if a segment lies outside the video.
    """
    per_frame: dict[int, list[list[int]]] = {}
    for event in events:
        for plane in event["planes"]:
            for segment in plane["segments"]:
                if not 0 <= segment["start"] <= segment["end"] < num_frames:
                    raise ValueError(f"Ground-truth segment outside the video: {segment}")
                for frame in range(segment["start"], segment["end"] + 1):
                    per_frame.setdefault(frame, []).extend(
                        list(map(int, b)) for b in segment["boxes"]
                    )
    table: list[list[list[int]]] = []
    refs: dict[str, int] = {}
    rows: list[list[int]] = []
    for frame in sorted(per_frame):
        value = sorted(per_frame[frame])
        key = json.dumps(value)
        if key not in refs:
            refs[key] = len(table)
            table.append(value)
        if rows and rows[-1][1] == frame - 1 and rows[-1][2] == refs[key]:
            rows[-1][1] = frame
        else:
            rows.append([frame, frame, refs[key]])
    return table, rows


def method_sources(index: dict[str, Any], method: str) -> dict[str, Any]:
    """Return the sources of the campaign that exported a method.

    :param index: an export index, possibly with ``appended`` campaigns.
    :param method: method id.
    :return: the ``sources`` block (predictions, report, seals) behind that method.
    :raises ValueError: if the index does not export the method.
    """
    for block in index.get("appended", []):
        if method in block["methods"]:
            return block["sources"]
    if all(m["id"] != method for m in index["methods"]):
        raise ValueError(f"{method} is not among the methods exported for {index['run_id']}")
    return index["sources"]


def event_intervals(
    events: Sequence[dict[str, Any]], annotated: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Place each evaluated event of a video on its annotated interval.

    The evaluator names an event ``<dataset>/<video>/eventNNN``, NNN being its position
    among the annotation project's events of the video (as in
    ``rank_localization_videos.rank_events``).

    :param events: snapshot events of the video, with ``event_id`` and ``class``.
    :param annotated: project events of the video (``start``, ``end`` inclusive frames,
        ``class_name``).
    :return: one ``{id, class, start, end}`` per evaluated event, in snapshot order.
    :raises ValueError: if an event falls outside the project or the project names another
        class.
    """
    placed = []
    for event in events:
        position = int(str(event["event_id"]).rsplit("/", 1)[1].removeprefix("event"))
        if position >= len(annotated):
            raise ValueError(f"{event['event_id']} is not in the annotation project")
        interval = annotated[position]
        if interval["class_name"] != event["class"]:
            raise ValueError(f"{event['event_id']}: the project names another class")
        placed.append(
            {
                "id": event["event_id"],
                "class": event["class"],
                "start": int(interval["start"]),
                "end": int(interval["end"]),
            }
        )
    return placed


def add_ground_truth(output: Path, snapshot: Path) -> dict[str, Any]:
    """Store, next to the exported boxes, the spatial ground truth they were scored against.

    The snapshot is the frozen one of the campaign evaluator: its digest must equal the
    ``gt_snapshot_sha256`` of the results of the default method on both datasets, so the
    boxes drawn as annotation are exactly those behind the IoU the page shows. Each video
    also lists its evaluated events (``id``, class and annotated interval, from the
    snapshot's ``project.json``), the keys of the per-event IoU of the class-routing methods.

    :param output: the ``localization`` folder of a run, with its ``index.json``.
    :param snapshot: the evaluator snapshot (``snapshot.json`` plus one file per video).
    :return: the updated index.
    :raises ValueError: if the snapshot is not the evaluated one, a video geometry differs
        from the exported boxes, or the project names an event with another class.
    """
    index = read_json(output / "index.json")
    digest = file_sha256(snapshot / "snapshot.json")
    results = Path(method_sources(index, index["default"])["report"]) / "results"
    project_path = snapshot / "project.json"
    project = (
        {(v["dataset"], v["video_id"]): v for v in read_json(project_path)["videos"]}
        if project_path.is_file()
        else {}
    )
    for folder in DATASETS:
        evaluated = read_json(results / folder / f"{index['default']}.json")["gt_snapshot_sha256"]
        if evaluated != digest:
            raise ValueError(
                f"{snapshot} is not the ground truth the results on {folder} were computed on"
            )
    videos: dict[str, str] = {}
    for video_id, relative in index["videos"].items():
        exported = read_json(output / relative)
        source = snapshot / exported["dataset"] / f"{video_id}.json"
        if not source.exists():
            continue
        truth = read_json(source)
        for name in ("num_frames", "width", "height"):
            if truth[name] != exported["video"][name]:
                raise ValueError(f"{video_id}: ground truth and boxes disagree on {name}")
        table, segments = truth_segments(truth["events"], truth["num_frames"])
        target = f"ground_truth/{exported['dataset']}/{video_id}.json"
        body = {"video": exported["video"], "boxes": table, "segments": segments}
        annotated = project.get((exported["dataset"], video_id))
        if annotated is not None:
            body["events"] = event_intervals(truth["events"], annotated["events"])
        write_text_atomic(output / target, json.dumps(body, separators=(",", ":")) + "\n")
        videos[video_id] = target
    meta = read_json(snapshot / "snapshot.json")
    index["ground_truth"] = {
        "snapshot": str(snapshot.resolve()),
        "snapshot_sha256": digest,
        "unit": meta.get("unit"),
        "frames": "reviewed frames with a box of the event class",
        "videos": videos,
    }
    write_json(output / "index.json", index)
    return index


def official_choice(table: Mapping[str, Any], step: str) -> dict[str, Any]:
    """Read what a step of the official table names as the best rule of a detector.

    :param table: the content of ``tabella_localizzazione.json``.
    :param step: a key of ``table["ladder"]["steps"]``, such as ``ours-best``.
    :return: ``id`` of the table row, its ``index``, ``stage``, ``configuration``, ``variant``
        and ``score``.
    :raises ValueError: if the step is unknown or has no row at the official node set.
    """
    steps = {s["key"]: s for s in table["ladder"]["steps"]}
    if step not in steps:
        raise ValueError(f"Unknown step {step!r} of the official table: {sorted(steps)}")
    row = steps[step]["ids"][OFFICIAL_SET]["any"]
    if row is None:
        raise ValueError(f"The step {step} has no row at the node set {OFFICIAL_SET}")
    stage, configuration, variant, score = row.split("|")
    index = next(i for i, r in enumerate(table["rows"]) if r["id"] == row)
    return {
        "id": row,
        "index": index,
        "stage": stage,
        "configuration": configuration,
        "variant": variant,
        "score": score,
    }


def official_values(table: Mapping[str, Any], index: int, dataset: int) -> dict[str, Any]:
    """Read the numbers of one row of the official table on one dataset.

    :param table: the content of ``tabella_localizzazione.json``.
    :param index: position of the row in ``table["rows"]``.
    :param dataset: position of the dataset in ``table["datasets"]``.
    :return: ``main``, ``coco``, ``online_main``, ``online_coco``, ``fpr_tpr90`` and the
        per-class ``classes`` of the main offline AP.
    """
    nodes = table["ig_nodes"]
    values = nodes["values"][OFFICIAL_SET][index][dataset]
    metric = nodes["metrics"].index
    per_class = nodes["per_class"][OFFICIAL_SET][index][dataset]
    main = per_class[nodes["per_class_metrics"].index("offline_main")]
    return {
        "main": values[metric("offline_main")],
        "coco": values[metric("offline_coco")],
        "online_main": values[metric("online_main")],
        "online_coco": values[metric("online_coco")],
        "fpr_tpr90": values[metric(f"fpr{OFFICIAL_TPR}")],
        "classes": dict(zip(table["classes"], main, strict=True)),
    }


def plane_detections(
    chunk: Mapping[str, Any], configuration: str, variant: str, score: str, size: tuple[int, int]
) -> list[list[tuple[float, list[int]]]] | None:
    """Build the scored boxes of every plane of a chunk of an AP stage.

    :param chunk: a chunk record of the stage.
    :param configuration: the configuration that chose the dictionary.
    :param variant: box rule (``evd.exp.localization_ap.BOX_VARIANTS``).
    :param score: confidence, ``mass`` or ``share``.
    :param size: native ``(width, height)``.
    :return: per plane the ``(confidence, inclusive native box)`` pairs by decreasing
        confidence, or ``None`` if the chunk was not localized under the configuration.
    """
    from localization_class_ap import _detections, plane_material

    key = chunk["methods"][configuration]["partition"] if chunk.get("localized") else None
    if key is None:
        return None
    planes = []
    for temporal in range(len(chunk["partitions"][key]["planes"])):
        material = plane_material(chunk, key, temporal)
        planes.append(
            [
                (float(d.score), [int(r[0]), int(r[1]), int(r[2]) - 1, int(r[3]) - 1])
                for d in _detections(variant, material, score, size)
                for r in d.rects
            ]
        )
    return planes


def _stored_ap(
    evaluation: Path, folder: str, choice: Mapping[str, Any]
) -> tuple[dict[str, Any], np.ndarray]:
    """Read the AP of one box rule from an evaluation of the official table.

    :return: the axes of the evaluation and the ``scores × events × thresholds`` AP of the rule.
    """
    axes = read_json(evaluation / f"axes_{folder}.json")
    with np.load(evaluation / f"metrics_per_event_{folder}.npz") as stored:
        ap = stored["ap"][
            axes["configuration"].index(choice["configuration"]),
            axes["variant"].index(choice["variant"]),
            :,
            axes["gt"].index(OFFICIAL_GT),
        ]
    return axes, ap


def _official_theta(evaluation: Path, folder: str, choice: Mapping[str, Any]) -> float:
    """:return: the box confidence that keeps :data:`OFFICIAL_TPR` percent of the annotated
    frames of the true positive rows (``localization_class_ap.frame_fpr``)."""
    axes = read_json(evaluation / f"axes_{folder}.json")
    with np.load(evaluation / f"frame_fpr_{folder}.npz") as stored:
        position = [int(q) for q in stored["q"]].index(OFFICIAL_TPR)
        return float(
            stored["theta"][
                axes["configuration"].index(choice["configuration"]),
                axes["variant"].index(choice["variant"]),
                axes["score"].index(choice["score"]),
                position,
            ]
        )


def _official_seal(stage: Path, folder: str, choice: Mapping[str, Any]) -> dict[str, Any]:
    """:return: the seal of a dataset of the stage.

    :raises ValueError: if the dataset is not sealed for the configuration of the rule.
    """
    path = stage / "operations" / f"seal-{folder}.json"
    seal = read_json(path)
    if seal.get("status") != "complete" or choice["configuration"] not in seal["methods"]:
        raise ValueError(f"{path} does not seal {choice['configuration']}")
    return seal


def _official_f1(
    stage: Path,
    snapshot: Path,
    folder: str,
    intervals: Mapping[str, Any],
    choice: Mapping[str, Any],
    iou_thresholds: Sequence[float],
    theta: float,
) -> tuple[np.ndarray, float]:
    """Read the macro F1 of the rule at every confidence, offline at 1 fps on the official GT.

    :param intervals: temporal events of the videos of the dataset, by video.
    :param theta: one more confidence, off the grid.
    :return: the F1 at each of :data:`F1_CONFIDENCES`, and at ``theta``.
    """
    from evd.eval.localization_ap import Image, confidence_f1, ground_truth
    from localization_auto_evaluate import sealed_videos
    from localization_class_ap import _detections, evaluated_frames, event_images

    configuration = choice["configuration"]
    images: dict[str, list[Image]] = {}
    truth_folder = snapshot / DATASETS[folder]
    seal = _official_seal(stage, folder, choice)
    for entry, complete, chunks in sealed_videos(
        stage, seal, lambda e: (truth_folder / f"{e['video_id']}.json").exists()
    ):
        if chunks is None:
            continue
        truth = read_json(truth_folder / f"{entry['video_id']}.json")
        size = (truth["width"], truth["height"])
        frames, _ = evaluated_frames(
            truth, float(complete["video"]["fps"]), intervals[entry["video_id"]]
        )
        for event in event_images(truth, complete, chunks, [configuration], "offline", frames):
            for image in event["images"]:
                material = image["material"][configuration]
                images.setdefault(event["class"], []).append(
                    Image(
                        truths=ground_truth(image["boxes"], OFFICIAL_GT),
                        detections=[]
                        if material is None
                        else _detections(choice["variant"], material, choice["score"], size),
                    )
                )
    values = confidence_f1(images, [*F1_CONFIDENCES, theta], iou_thresholds)
    return values[:-1], float(values[-1])


def export_official(
    run: Path,
    table_path: Path,
    step: str,
    stage: Path,
    intervals: Path,
    snapshot: Path,
    name: str,
    campaign: str,
    output: Path | None = None,
) -> dict[str, Any]:
    """Export the best box rule of the official table and make it the default of the run.

    :param run: the run of the detector, with an export whose ground truth is ``snapshot``.
    :param table_path: ``tabella_localizzazione.json``; its folder holds the evaluations.
    :param step: step of the table naming the rule (:func:`official_choice`).
    :param stage: the sealed AP stage of the rule, at the table's IG nodes.
    :param intervals: temporal events of the test videos, which place the 1 fps frames.
    :param snapshot: the frozen GT snapshot of the evaluations.
    :param name: readable name of the rule.
    :param campaign: provenance label of the campaign.
    :param output: the ``localization`` folder, ``<run>/localization`` by default.
    :return: the updated index.
    :raises ValueError: if the stage is not the one of the evaluation, the chunk flags disagree
        with the run scores (another detector), the recomputed AP differs from the evaluation,
        the AP on disk is not the table's, or the export lacks the snapshot.
    """
    from localization_auto_evaluate import chunk_windows, evaluate_video, sealed_videos
    from localization_class_ap import evaluate_configuration, evaluated_frames, event_images
    from localization_class_table import range_mean

    output = output or run / OUTPUT_FOLDER
    table = read_json(table_path)
    choice = official_choice(table, step)
    # Reading the snapshot here: an export on another ground truth must fail before the long read.
    digest = file_sha256(snapshot / "snapshot.json")
    index = read_json(output / "index.json")
    if (index.get("ground_truth") or {}).get("snapshot_sha256") != digest:
        raise ValueError(f"{snapshot} is not the ground truth of the export in {output}")
    temporal = read_json(intervals)["intervals"]
    provenance = next(s for s in table["ig_nodes"]["sets"] if s["key"] == OFFICIAL_SET)[
        "provenance"
    ][choice["stage"]]
    all_boxes = "-".join((choice["configuration"], choice["variant"], choice["score"]))
    above_theta = f"{all_boxes}-tpr{OFFICIAL_TPR}"
    above_f1 = f"{all_boxes}-f1max"
    most_confident = f"{all_boxes}-top1"
    # What tells the entries apart comes first: the closed menu shows only the start of a name.
    labels = {
        above_theta: f"θ@TPR{OFFICIAL_TPR} · {name}",
        above_f1: f"θ@F1max · {name}",
        most_confident: f"Most confident box · {name}",
        all_boxes: f"All boxes · {name}",
    }
    report = output / OFFICIAL_FOLDER
    target = output.with_name(output.name + ".official")
    if target.exists():
        raise ValueError(f"{target} exists: remove the leftover of an interrupted export")
    threshold = read_json(run / "metrics.json")["threshold"]["value"]
    videos: dict[str, str] = {}
    ap_blocks: dict[str, dict[str, Any]] = {}
    seals: dict[str, str] = {}
    checks: dict[str, Any] = {}
    meta_snapshot = read_json(snapshot / "snapshot.json")
    previous_default = index["default"]
    low, high = table["ranges"]["main"]
    f1_curves, f1_at_tpr = {}, {}
    for folder in DATASETS:
        axes = read_json(table_path.parent / provenance[folder]["offline"] / f"axes_{folder}.json")
        f1_curves[folder], f1_at_tpr[folder] = _official_f1(
            stage,
            snapshot,
            folder,
            temporal[folder],
            choice,
            [t for t in axes["iou_threshold"] if low - 1e-9 <= t <= high + 1e-9],
            _official_theta(table_path.parent / provenance[folder]["online"], folder, choice),
        )
    theta_f1 = float(F1_CONFIDENCES[int(np.argmax(f1_curves[F1_SOURCE]))])
    for folder, dataset in DATASETS.items():
        seal = _official_seal(stage, folder, choice)
        seals[f"{stage.name}/{folder}"] = file_sha256(stage / "operations" / f"seal-{folder}.json")
        offline = table_path.parent / provenance[folder]["offline"]
        online = table_path.parent / provenance[folder]["online"]
        expected = run_scores(run, folder)
        theta = _official_theta(online, folder, choice)
        entries = {
            above_theta: lambda plane, limit=theta: [b for s, b in plane if s >= limit],
            above_f1: lambda plane: [b for s, b in plane if s >= theta_f1],
            most_confident: lambda plane: [b for s, b in plane if s == plane[0][0]],
            all_boxes: lambda plane: [b for _, b in plane],
        }
        rows: dict[str, list] = {method: [] for method in entries}
        events: list[dict[str, Any]] = []
        sizes: dict[str, tuple[int, int]] = {}
        compared = mismatches = 0

        def truth_of(entry: dict, dataset: str = dataset) -> Path:
            return snapshot / dataset / f"{entry['video_id']}.json"

        for entry, complete, chunks in sealed_videos(stage, seal, lambda e: truth_of(e).exists()):
            video_id = entry["video_id"]
            if chunks is None:
                continue
            truth = read_json(truth_of(entry))
            meta = complete["video"]
            size = (int(meta["width"]), int(meta["height"]))
            boxes: dict[str, list] = {method: [] for method in entries}
            for chunk in chunks:
                if expected is not None and chunk["flagged"] is not None:
                    compared += 1
                    above = expected[chunk["window_id"]] >= threshold
                    mismatches += above != bool(chunk["flagged"])
                planes = plane_detections(
                    chunk, choice["configuration"], choice["variant"], choice["score"], size
                )
                empty = [[] for _ in range(int(chunk["grid"][0]))]
                for method, keep in entries.items():
                    kept = None if planes is None else [keep(plane) for plane in planes]
                    boxes[method].append(kept)
                    chunk["methods"][method] = {"boxes": kept if kept is not None else empty}
            table_boxes, segments = project_boxes(
                chunk_windows(chunks), meta["num_frames"], [c["grid"][0] for c in chunks], boxes
            )
            relative = f"{dataset}/{video_id}.json"
            body = {
                "video": {k: meta[k] for k in ("video_id", "num_frames", "fps", "width", "height")},
                "dataset": dataset,
                "boxes": table_boxes,
                "segments": segments,
            }
            write_text_atomic(target / relative, json.dumps(body, separators=(",", ":")) + "\n")
            videos[video_id] = relative
            for method in rows:
                rows[method].extend(evaluate_video(truth, complete, chunks, method))
            frames, _ = evaluated_frames(truth, float(meta["fps"]), temporal[folder][video_id])
            for event in event_images(
                truth, complete, chunks, [choice["configuration"]], "offline", frames
            ):
                sizes[event["event_id"]] = (truth["width"], truth["height"])
                events.append(event)
        if expected is not None and (compared == 0 or mismatches):
            raise ValueError(
                f"The chunk flags of {stage.name} on {folder} disagree with {run.name} on "
                f"{mismatches} of {compared} windows: the stage belongs to another detector"
            )
        checks[dataset] = {"compared_windows": compared, "flag_mismatches": mismatches}
        axes, stored = _stored_ap(offline, folder, choice)
        stored_events = axes["event"]
        order = {e["event_id"]: i for i, e in enumerate(events)}
        if set(order) != {e["event_id"] for e in stored_events}:
            raise ValueError(f"The events of {stage.name} on {folder} are not the evaluation's")
        classes = sorted(meta_snapshot["class_event_support"][dataset])
        recomputed = evaluate_configuration(
            {
                "name": choice["configuration"],
                "flagged": [],
                "grid": "fine",
                "variants": [choice["variant"]],
                "gt_modes": [OFFICIAL_GT],
                "ap_only": True,
                "classes": classes,
                "sizes": sizes,
                "events": [
                    {
                        "event_id": e["event_id"],
                        "class": e["class"],
                        "images": [
                            {
                                "boxes": i["boxes"],
                                "material": i["material"][choice["configuration"]],
                            }
                            for i in events[order[e["event_id"]]]["images"]
                        ],
                    }
                    for e in stored_events
                ],
                "negatives": [],
                "empty_negatives": 0,
            }
        )["arrays"]["ap"][0, :, 0]
        gap = float(np.nanmax(np.abs(recomputed - stored)))
        if gap > AP_TOLERANCE:
            raise ValueError(
                f"The AP recomputed from {stage.name} on {folder} differs from {offline} by "
                f"{gap:.3g}: these boxes are not the evaluated ones"
            )
        checks[dataset]["recomputed_ap_gap"] = gap
        rule = stored[axes["score"].index(choice["score"])]
        main = range_mean(rule, axes["iou_threshold"], low, high)
        official = official_values(
            table, choice["index"], [d["key"] for d in table["datasets"]].index(folder)
        )
        for cls, value in official["classes"].items():
            mine = [v for e, v in zip(stored_events, main, strict=True) if e["class"] == cls]
            if mine and abs(float(np.mean(mine)) - value) > TABLE_TOLERANCE:
                raise ValueError(f"{cls} on {folder}: {offline} is not the table's value")
        ap_blocks[dataset] = {
            **official,
            "theta_tpr90": theta,
            "theta_f1max": theta_f1,
            "f1": {
                "at_theta_f1max": float(f1_curves[folder][F1_CONFIDENCES.searchsorted(theta_f1)]),
                "at_theta_tpr90": f1_at_tpr[folder],
                "best": float(f1_curves[folder].max()),
                "theta_of_best": float(F1_CONFIDENCES[int(np.argmax(f1_curves[folder]))]),
            },
            "events": {e["event_id"]: float(v) for e, v in zip(stored_events, main, strict=True)},
        }
        for method, method_rows in rows.items():
            result = aggregate_localization(method_rows, classes)
            result.update(
                experiment=method,
                dataset=dataset,
                prediction_seal_sha256=seals[f"{stage.name}/{folder}"],
                gt_snapshot_sha256=digest,
            )
            write_json(report / "results" / folder / f"{method}.json", result)
    _, methods = read_evaluation(report, labels)
    for method in methods:
        method["family"], method["family_label"] = OFFICIAL_FAMILY
        method["ap"] = ap_blocks
    addition = {
        "schema": 1,
        "run_id": run.name,
        "campaign": campaign,
        "metric": index["metric"],
        "selection": index["selection"],
        "default": above_theta,
        "classes": index["classes"],
        "coordinate_space": index["coordinate_space"],
        "projection": index["projection"],
        "methods": [{**m, "stage": stage.name, "campaign": campaign} for m in methods],
        "videos": videos,
        "detector_check": checks,
        "sources": {
            "predictions": str(stage.resolve()),
            "report": str(report.resolve()),
            "table_sha256": file_sha256(table_path),
            "seal_sha256": seals,
        },
    }
    write_json(target / "index.json", addition)
    index = append_export(output, target)
    shutil.rmtree(target)
    first = [m for m in index["methods"] if m["id"] in labels]
    first.sort(key=lambda m: list(labels).index(m["id"]))
    index["methods"] = first + [m for m in index["methods"] if m["id"] not in labels]
    index["official_best"] = {
        "method": above_theta,
        "above_f1": above_f1,
        "most_confident": most_confident,
        "all_boxes": all_boxes,
        "theta_f1_source": F1_SOURCE,
        "previous_default": previous_default,
        "table_row": choice["id"],
        "table_sha256": file_sha256(table_path),
        "step": step,
        "node_set": OFFICIAL_SET,
        "gt": OFFICIAL_GT,
        "criterion": table["ladder"]["criterion"],
    }
    index["default"] = above_theta
    index["selection"] = (
        f"the official table's best rule for this detector, {table['ladder']['criterion']}; "
        f"drawn above the confidence of TPR {OFFICIAL_TPR}% (exploratory: both are chosen on "
        "the test sets)"
    )
    write_json(output / "index.json", index)
    return index


def main() -> None:
    """Parse the arguments and export one detector, its ground truth, or both."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="runs/<run_id> of the detector")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--campaign", help="provenance label of the campaign")
    parser.add_argument(
        "--evaluation", type=Path, help="class-routing evaluator stage folder (with results/)"
    )
    parser.add_argument(
        "--passes", type=Path, help="class-routing IG pass folder: chunk scores of the check"
    )
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        metavar="ID=NAME",
        help="class-routing experiment to export and the name shown for it (repeatable)",
    )
    parser.add_argument(
        "--append", action="store_true", help="add the methods to the existing export"
    )
    parser.add_argument(
        "--score-tolerance",
        type=float,
        default=SCORE_TOLERANCE,
        help="largest chunk score difference from the run (raise it only for a run scored on "
        "another GPU model)",
    )
    parser.add_argument("--ground-truth", type=Path, help="evaluator GT snapshot folder")
    parser.add_argument("--output", type=Path, help="default: <run>/localization")
    parser.add_argument(
        "--official-table",
        type=Path,
        help="tabella_localizzazione.json: export the best rule of a step and make it the default",
    )
    parser.add_argument("--official-step", default="ours-best", help="step of the table")
    parser.add_argument("--official-stage", type=Path, help="sealed AP stage of the rule")
    parser.add_argument("--official-name", help="readable name of the rule")
    parser.add_argument("--intervals", type=Path, help="temporal events of the test videos")
    args = parser.parse_args()
    output = args.output or args.run / OUTPUT_FOLDER
    if args.official_table:
        needed = (
            args.official_stage,
            args.official_name,
            args.intervals,
            args.ground_truth,
            args.campaign,
        )
        if not all(needed):
            parser.error(
                "--official-table needs --official-stage, --official-name, --intervals, "
                "--ground-truth and --campaign"
            )
        index = export_official(
            args.run,
            args.official_table,
            args.official_step,
            args.official_stage,
            args.intervals,
            args.ground_truth,
            args.official_name,
            args.campaign,
            output,
        )
        print(
            f"{index['run_id']}: {len(index['methods'])} methods, default {index['default']}; "
            f"check {index['appended'][-1]['detector_check']}"
        )
        return
    routing = (args.evaluation, args.passes, args.method)
    if any(routing) and not all(routing):
        parser.error("--evaluation, --passes and --method go together")
    if any(routing) and args.report:
        parser.error("--report is for the comparison tables; --evaluation replaces it")
    report = args.evaluation or args.report
    boxes = (args.predictions, report, args.campaign)
    if any(boxes) and not all(boxes):
        parser.error("--predictions, --report (or --evaluation) and --campaign go together")
    if args.append and not all(boxes):
        parser.error("--append needs the predictions to add")
    if not any(boxes) and not args.ground_truth:
        parser.error("nothing to export: give the predictions, --ground-truth, or both")
    if all(boxes):
        options = {}
        if args.evaluation:
            names = dict(item.split("=", 1) for item in args.method)
            options = {
                "evaluated": read_evaluation(args.evaluation, names),
                "chunk_scores": pass_scores(args.passes),
            }
        target = output.with_name(output.name + ".append") if args.append else output
        if args.append and target.exists():
            raise ValueError(f"{target} exists: remove the leftover of an interrupted append")
        index = export(
            args.run,
            args.predictions,
            report,
            args.campaign,
            target,
            score_tolerance=args.score_tolerance,
            **options,
        )
        if args.append:
            index = append_export(output, target)
            shutil.rmtree(target)
        print(
            f"{index['run_id']}: {len(index['methods'])} methods, {len(index['videos'])} videos, "
            f"default {index['default']}; detector check {index['detector_check']}"
        )
    if args.ground_truth:
        index = add_ground_truth(output, args.ground_truth)
        print(f"{index['run_id']}: ground truth for {len(index['ground_truth']['videos'])} videos")


if __name__ == "__main__":
    main()
