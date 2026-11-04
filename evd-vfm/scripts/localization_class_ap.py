"""COCO AP of the class-routing localization, per event, over every box variant (CPU).

Plan: ``docs/varie/campagna_ap_localizzazione_20261027.md``. Reads the records of
``localization_class_pass.py`` and the dictionaries of ``localization_class_fit.py``, as
``localization_class_localize.py``; decodes nothing and uses no GPU.

- ``detections``: for every annotated test chunk and every configuration (``campaign``,
  ``b3`` or ``b4``, as ``localize``), the class choice and, for the chosen dictionary, the
  positive IG mass of each concept over the chunk and every 8-connected component of every
  concept on every plane, with its own positive mass. No threshold is applied: the AP
  sweeps them all. P07's three configurations share the global dictionary and become one,
  ``Global``. The stage is sealed in the format of ``localize`` (every protocol window of
  every test video), the components of a dictionary stored once per chunk.
- ``evaluate``: on the frozen GT snapshot, after the seal is checked, the per-event COCO AP
  (mean over IoU 0.50:0.95, AP50, AP75) of every configuration, box variant
  (``evd.exp.localization_ap.VARIANTS``), score (concept or component mass, or its share of
  the chunk total) and ground-truth reading (``union``, ``instances`` or ``touching``), then
  the mean over the events of each class and over the thirteen classes. With ``--iou-grid
  fine`` every event keeps, at each IoU threshold 0.05:0.95 by 0.05, its AP, RBDC and RBDR
  (``evd.eval.localization_ap``), and AP, RBDC and RBDR are also computed over all the images
  of the dataset at once (``pooled``), as Ramachandra & Jones do for RBDC.

Real use (28 October 2026): ``detections --real-use PLAN --extra-sets real`` keeps only the
chunks the detector's binary head flags (``localization_class_real_use.py``), and ``evaluate
--intervals`` adds to the pooled metrics the negative images, the chunk planes of the frames
outside every temporal event, where every detection is a false positive; the per-event metrics
keep the event's images only. ``evaluate --mode online`` shows every GT frame on every chunk
that holds it: the flagged ones with their boxes, the others (false negatives) with none, so
their GT boxes count as missed (30 October 2026).

``box-scores`` (30 October 2026): on an online evaluation with ``--intervals``, histograms of
the score of every box on its flagged true and false positive chunks, the material of the FP
metric without TPR (``box_score_histograms``); ``box-retention``, the false boxes kept when the
threshold keeps a share of the boxes on the positive planes (``box_retention_step``);
``false-boxes``, the false boxes per false positive chunk at the threshold that keeps a box on a
share of the positive planes (``false_boxes_step``).

1 fps and the OVAD rows (30 October 2026, ``docs/varie/metriche_localizzazione_1fps_fpr_ovad_
20261030.md``): ``evaluate --frames 1fps`` takes one image per second of the video, the native
frame nearest to each whole second, plus the annotated frame of an event without one
(:func:`evaluated_frames`), with no deduplication; ``--ovad-plan`` (``localization_class_ovad_
plan.py``) makes the online boxes those of the chunks with ``h(c) >= θ_O``, the threshold of the
OVAD rows, instead of the chunk threshold of the stage. ``frame-fpr`` computes the FP metric,
``FPR_box@TPR(q)`` on the frames of the OVAD FP rows (:func:`frame_fpr_step`).
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from evd.eval.localization_ap import (  # noqa: E402
    GT_MODES,
    IOU_THRESHOLDS_FINE,
    METRICS,
    Image,
    aggregate_ap,
    aggregate_by_threshold,
    ap_by_threshold,
    box_event_precision,
    box_matches,
    box_rbdc,
    ground_truth,
    prepare_boxes,
    prepare_union,
    summarize,
    touching_groups,
    union_event_precision,
    union_precision,
    union_rbdc,
)
from evd.exp.localization_ap import (  # noqa: E402
    LEGACY_VARIANTS,
    SCORES,
    UNION,
    VARIANTS,
    detections,
    union_concepts,
)
from evd.exp.localization_auto import read_json, seal_stage, video_key, write_json  # noqa: E402
from evd.exp.localization_class import (  # noqa: E402
    CLASS_RULES,
    choose_class,
    class_scores,
    concept_components,
    positive_masses,
    region_boxes,
    stack_dictionaries,
)
from evd.exp.localization_vtcd import CONCEPTS  # noqa: E402
from evd.explain.vtcd import assign_concepts  # noqa: E402
from evd.utils.hashing import file_sha256  # noqa: E402
from evd.vendor.cocoeval_ap import accumulate  # noqa: E402
from localization_class_localize import (  # noqa: E402
    CONFIG,
    DATASETS,
    HEADS,
    _selector_class,
    head_records,
    stage_records,
    with_classifiers,
    with_heads,
    write_real_use,
)
from localization_class_plan import CLASSES  # noqa: E402

#: What a stage of this script stores per chunk plane (28 October 2026): the 8-connected
#: components (the stages of 27 October hold only these), the 4-connected ones and the regions.
MATERIAL = ("components8", "components4", "regions")
#: The configuration of the global dictionary: P07's three IoU configurations differ only
#: by the threshold rule and box union, which the AP sweeps or varies on its own.
GLOBAL = "Global"
#: IoU configurations that the AP folds into :data:`GLOBAL`.
FOLDED = ("P07-global", "P07-global-bridge70", "P07-global-per-concept")


def configurations(config: dict, args) -> list[str]:
    """:return: the AP configurations of a set: :data:`GLOBAL`, then the class-routed ones."""
    names = config["configurations"][args.configurations]
    prefixes = tuple(f"{h}-" for h in HEADS)
    if args.without_heads and args.heads_set:
        raise ValueError("--without-heads excludes --heads-set")
    if args.without_heads:
        names = [n for n in names if not n.startswith(prefixes)]
    if args.configurations == "b4" and not args.classifiers_set:
        raise ValueError("the b4 configurations need --classifiers-set")
    routed = [n for n in names if n not in FOLDED]
    return ([GLOBAL] if any(n in FOLDED for n in names) else []) + routed


def partition_key(name: str, record: dict, stacked_scores: dict) -> str | None:
    """:return: the dictionary a configuration localizes the chunk with (``global``,
    ``a1:<class>`` or ``a2:<class>``), ``None`` when the selector chooses no class."""
    if name == GLOBAL:
        return "global"
    selector, key = (
        (name[:-3], name[-2:].lower()) if name[-3:] in ("-A1", "-A2") else (name, name[:2].lower())
    )
    chosen = _selector_class(selector, record, stacked_scores)
    return None if chosen is None else f"{key}:{CLASSES[chosen]}"


def chunk_partitions(record: dict, arrays: dict, model, stacked, names, meta) -> tuple[dict, dict]:
    """Concept masses and components of every dictionary the configurations use on a chunk.

    :return: ``(methods, partitions)``: per configuration its ``partition`` key, and per key
        ``{"masses", "planes", "planes4", "region_concepts"}``: the 8- and 4-connected
        components of :func:`concept_components` and the concept of each SLIC region (the
        region boxes do not depend on the dictionary: :func:`region_boxes`, once per chunk).
    """
    descriptors = arrays["descriptors"].astype(np.float64)
    labels, counts, ig = arrays["labels"], arrays["counts"], arrays["ig_ensemble"]
    stacked_scores = {}
    for key, (dictionary, owners) in stacked.items():
        _, coefficients = assign_concepts(descriptors, dictionary)
        for rule in CLASS_RULES:
            stacked_scores[f"{key.upper()}-{rule}"] = choose_class(
                class_scores(coefficients, counts, owners, len(CLASSES), rule)
            )
    methods, partitions = {}, {}
    for name in names:
        key = partition_key(name, record, stacked_scores)
        methods[name] = {"partition": key}
        if key is None or key in partitions:
            continue
        if key == "global":
            dictionary = model["global_dictionary"]
        else:
            family, cls = key.split(":")
            dictionary = model[family][CLASSES.index(cls)]
        ids, _ = assign_concepts(descriptors, dictionary)
        partitions[key] = {
            "masses": positive_masses(ig, labels, ids, CONCEPTS).tolist(),
            "planes": concept_components(ig, labels, ids, meta["width"], meta["height"]),
            "planes4": concept_components(
                ig, labels, ids, meta["width"], meta["height"], connectivity=4
            ),
            "region_concepts": [int(i) for i in ids],
        }
    return methods, partitions


def detect(args, config: dict) -> None:
    model = np.load(args.dictionaries)
    # A file without A2 (``localization_class_fit.py --without-a2``) serves the ``global``
    # configurations only, which stack no class dictionary.
    stacked = {key: stack_dictionaries(model[key]) for key in ("a1", "a2") if key in model.files}
    names = configurations(config, args)
    heads = head_records(args.pass_root / args.heads_set) if args.heads_set else None
    classifiers = None
    if args.classifiers_set:
        classifiers = {}
        for name in args.classifiers_set:
            classifiers.update(head_records(args.pass_root / name))
    by_window = {}
    for record, arrays in stage_records(args):
        if heads is not None:
            record = with_heads(record, heads)
        if classifiers is not None:
            record = with_classifiers(record, classifiers)
        start = time.process_time()
        meta = args.metas[record["dataset"]][record["video_id"]]
        methods, partitions = chunk_partitions(record, arrays, model, stacked, names, meta)
        regions = region_boxes(
            arrays["ig_ensemble"], arrays["labels"], meta["width"], meta["height"]
        )
        seconds = time.process_time() - start
        by_window[record["window_id"]] = (record, methods, partitions, regions, seconds)
    write_stage(
        args,
        by_window,
        names,
        {
            "configurations": names,
            "set": args.configurations,
            "pass_root": str(args.pass_root),
            "heads_set": args.heads_set,
            "classifiers_set": args.classifiers_set,
            "dictionaries_sha256": file_sha256(args.dictionaries),
            "material": list(MATERIAL),
            "keep_annotated": bool(getattr(args, "keep_annotated", False)),
        },
    )


def write_stage(args, by_window: dict, names: list[str], receipt: dict) -> None:
    """Write and seal an AP stage: one chunk file for every protocol window of every test video.

    :param by_window: per localized window id ``(record, methods, partitions, regions, cpu
        seconds)``; a window without an entry gets no detection.
    :param names: the stage's configurations.
    :param receipt: the content of ``operations/ap-stage.json``, after ``localized_chunks``.
    :raises ValueError: with ``--real-use``, if the stage did not localize every flagged chunk.
    """
    from evd.protocol.chunks import sampler_for_protocol
    from evd.types import VideoMeta

    flagged = None
    if getattr(args, "real_use", None):
        plan = read_json(args.real_use)
        flagged = {(d, w) for d, ids in plan["positives"].items() for w in ids}
    protocol = yaml.safe_load((ROOT / "configs/protocol/edp_v2.yaml").read_text(encoding="utf-8"))
    sampler = sampler_for_protocol(None, protocol, validazione=False)
    grid = next(iter(by_window.values()))[0]["grid"]
    for dataset, manifest in (("ucf_test", args.manifest_ucf), ("mivia_test", args.manifest_mivia)):
        for row in read_json(manifest):
            meta = VideoMeta(**row)
            target = args.output / dataset / video_key(meta.video_id)
            chunks, cpu = [], 0.0
            for index, window in enumerate(sampler.windows_for_video(meta)):
                found = by_window.get(window.window_id)
                if found is not None and found[0]["dataset"] != dataset:
                    found = None
                if found is None:
                    methods, partitions, regions = {n: {"partition": None} for n in names}, {}, None
                else:
                    _, methods, partitions, regions, seconds = found
                    cpu += seconds
                destination = target / "chunks" / f"{index:06d}.json"
                write_json(
                    destination,
                    {
                        "window": asdict(window),
                        "window_id": window.window_id,
                        "grid": grid,
                        "localized": found is not None,
                        # Whether the detector's binary head flags the chunk (real use), None
                        # outside real use: the online AP reads the boxes of flagged chunks only.
                        "flagged": None
                        if flagged is None
                        else (dataset, window.window_id) in flagged,
                        "methods": methods,
                        "partitions": partitions,
                        "regions": regions,
                    },
                )
                chunks.append(
                    {
                        "path": str(destination.relative_to(target)),
                        "sha256": file_sha256(destination),
                    }
                )
            write_json(
                target / "complete.json",
                {
                    "status": "complete",
                    "video": row,
                    "chunks": chunks,
                    # Masses and components are shared by the configurations: one CPU time.
                    "timing": {n: {"shared_cpu_seconds": cpu} for n in names},
                    "prediction_scope": "entire_video_all_protocol_windows",
                },
            )
        seal_stage(args.output, dataset, names, manifest)
    write_json(
        args.output / "operations" / "ap-stage.json",
        {"localized_chunks": len(by_window), **receipt},
    )
    if getattr(args, "real_use", None):
        if getattr(args, "keep_annotated", False):
            localized_flagged = sum(
                (r["dataset"], w) in flagged for w, (r, *_rest) in by_window.items()
            )
            write_real_use(args, args.output, localized_flagged)
        else:
            write_real_use(args, args.output, len(by_window))


def plane_material(chunk: dict, key: str, temporal: int) -> tuple:
    """Material of one chunk plane under one dictionary partition.

    :return: ``(masses, components8, components4, regions)``; the last two are ``None`` on a
        stage of 27 October 2026, which stores only the 8-connected components, and each
        region carries the concept its partition assigns it.
    """
    partition = chunk["partitions"][key]
    four = partition.get("planes4")
    regions = chunk.get("regions")
    if regions is not None:
        owners = partition["region_concepts"]
        regions = [{**r, "concept": owners[r["region"]]} for r in regions[temporal]]
    return (
        partition["masses"],
        partition["planes"][temporal],
        None if four is None else four[temporal],
        regions,
    )


def online_frames(
    truth: dict, complete: dict, chunks: list[dict], flagged: list[bool] | None = None
):
    """Yield every reviewed GT frame once per protocol chunk that contains it (online AP).

    No chunk is chosen for the frame, unlike ``assigned_frames``: every chunk whose window
    ``[start_frame, end_frame]`` holds the frame shows it on its nearest plane
    (``offline_plane``). A chunk holding an annotated frame is a true positive if the detector
    flags it and a false negative otherwise; :func:`event_images` reads the boxes of the first
    and gives the second none, so its GT boxes count as missed (decision of 30 October 2026).
    Until then the false negative chunks were left out, and an event the detector missed had
    no AP and left its class: the online AP rose by dropping the hardest events.

    :param flagged: per chunk whether it shows boxes (:func:`emitting_chunks`, ``h(c) >= θ_O``);
        ``None`` reads each chunk's ``flagged``, the chunk threshold of the stage.
    :return: ``(event, plane, frame, index, temporal, boxes)`` per (frame, chunk), in GT order.
    :raises ValueError: if prediction and GT disagree on the video geometry, a chunk does not
        say whether it is flagged, or a flagged chunk was not localized.
    """
    from evd.explain.project import offline_plane
    from localization_auto_evaluate import chunk_windows

    for name in ("video_id", "width", "height", "num_frames"):
        if complete["video"][name] != truth[name]:
            raise ValueError(f"Video identity/geometry mismatch: {name}")
    windows = chunk_windows(chunks)
    if flagged is not None and len(flagged) != len(chunks):
        raise ValueError("One flag per chunk is needed")
    for index, chunk in enumerate(chunks):
        shown = chunk.get("flagged") if flagged is None else flagged[index]
        if shown is None:
            raise ValueError("The online AP needs a real-use stage, whose chunks say if flagged")
        if shown and not chunk["localized"]:
            raise ValueError(f"Flagged chunk {chunk['window_id']} has no localization")
    for event in truth["events"]:
        for plane in event["planes"]:
            for segment in plane["segments"]:
                near = [
                    (index, window)
                    for index, window in enumerate(windows)
                    if window.start_frame <= segment["end"] and segment["start"] <= window.end_frame
                ]
                for frame in range(segment["start"], segment["end"] + 1):
                    for index, window in near:
                        if window.start_frame <= frame <= window.end_frame:
                            temporal = offline_plane(window, frame, chunks[index]["grid"][0])
                            yield event, plane, frame, index, temporal, segment["boxes"]


def event_images(
    truth: dict,
    complete: dict,
    chunks: list[dict],
    names: list[str],
    mode: str = "offline",
    frames: np.ndarray | None = None,
    flagged: list[bool] | None = None,
    ignore_head_gate: bool = False,
    exclude_head_fn: bool = False,
) -> list[dict]:
    """Images of every event: distinct (annotated plane, chunk, chunk plane, GT boxes), or one
    per evaluated frame (and chunk, online) with ``frames``.

    :param mode: ``offline``, each GT frame on the one chunk ``assigned_frames`` gives it, or
        ``online`` (:func:`online_frames`), on every chunk that holds it, with no material on
        the chunks the detector does not flag (false negatives); an event with no image keeps
        an empty list.
    :param frames: the evaluated frames ``F_v`` (:func:`evaluated_frames`): only these GT frames
        are images, each one (per chunk online) with no deduplication, so every second weighs
        the same (30 October 2026); ``None`` keeps every GT frame, deduplicated by signature.
    :param flagged: online, per chunk whether it shows boxes (:func:`emitting_chunks`);
        ``None`` reads the stage's ``flagged``.
    :param ignore_head_gate: controlled online counterfactual: retain exactly the same
        images and GT, but restore saved predictions on non-emitting chunks. Missing
        localization material raises instead of masquerading as an empty prediction.
    :param exclude_head_fn: conditional reference using only emitting chunks. Predictions
        stay unchanged; FN images are omitted, and wholly missed events have no AP.
    :return: per event ``{"event_id", "class", "images"}``, each image with its GT boxes and,
        per configuration, the :func:`plane_material` (``None`` when no chunk or no class
        predicts it).
    :raises ValueError: on an unknown mode.
    """
    from localization_auto_evaluate import assigned_frames

    if ignore_head_gate and mode != "online":
        raise ValueError("Ignoring the head gate requires online images")
    if exclude_head_fn and (mode != "online" or ignore_head_gate):
        raise ValueError("FN exclusion requires observed online predictions")
    if mode == "offline":
        found = assigned_frames(truth, complete, chunks)
    elif mode == "online":
        found = online_frames(truth, complete, chunks, flagged)
    else:
        raise ValueError(f"Unknown AP mode {mode!r}: offline or online")
    keep = None if frames is None else {int(f) for f in frames}
    events: dict[str, dict] = {
        e["event_id"]: {"event_id": e["event_id"], "class": e["class"], "images": []}
        for e in truth["events"]
    }
    seen = set()
    for event, plane, frame, index, temporal, boxes in found:
        out = events[event["event_id"]]
        if keep is not None:
            if frame not in keep:
                continue
        else:
            signature = (
                event["event_id"],
                plane["plane_id"],
                index,
                temporal,
                tuple(map(tuple, boxes)),
            )
            if signature in seen:
                continue
            seen.add(signature)
        missed = mode == "online" and not (
            chunks[index]["flagged"] if flagged is None else flagged[index]
        )
        if exclude_head_fn and missed:
            continue
        if ignore_head_gate and not chunks[index]["localized"]:
            raise ValueError(
                f"Counterfactual needs localized GT chunk {chunks[index]['window_id']}"
            )
        material = {}
        for name in names:
            key = (
                None
                if index < 0 or (missed and not ignore_head_gate)
                else chunks[index]["methods"][name]["partition"]
            )
            material[name] = None if key is None else plane_material(chunks[index], key, temporal)
        out["images"].append({"boxes": boxes, "material": material})
    return list(events.values())


def evaluate_configuration(payload: dict) -> dict:
    """AP of one configuration under every variant, score and ground-truth reading.

    With ``payload["grid"] == "fine"`` every event keeps its AP at each of the 19 thresholds of
    :data:`IOU_THRESHOLDS_FINE`, with the class and all-class means at each; otherwise AP,
    AP50 and AP75 on COCO's ten.
    """
    if payload.get("grid") == "fine":
        return _evaluate_fine(payload)
    name, classes, width_height = payload["name"], payload["classes"], payload["sizes"]
    results: dict = {}
    for variant in VARIANTS:
        for score in SCORES:
            rows = {mode: [] for mode in GT_MODES}
            for event in payload["events"]:
                width, height = width_height[event["event_id"]]
                built = []
                for image in event["images"]:
                    material = image["material"]
                    if material is None:
                        built.append(None)
                    else:
                        built.append(_detections(variant, material, score, (width, height)))
                for mode in GT_MODES:
                    images = [
                        Image(
                            truths=ground_truth(image["boxes"], mode),
                            **{("concepts" if variant == UNION else "detections"): made or []},
                        )
                        for image, made in zip(event["images"], built, strict=True)
                    ]
                    precision = (
                        union_event_precision(images)
                        if variant == UNION
                        else box_event_precision(images)
                    )
                    rows[mode].append(
                        {
                            "event_id": event["event_id"],
                            "class": event["class"],
                            "images": len(images),
                            **summarize(precision),
                        }
                    )
            results.setdefault(variant, {})[score] = {
                mode: aggregate_ap(rows[mode], classes) for mode in GT_MODES
            }
    return {"configuration": name, "variants": results}


def _detections(variant: str, material: tuple, score: str, size: tuple[int, int]) -> list:
    """:return: the union material or the detections of ``variant`` for a
    :func:`plane_material` (a two-element tuple on payloads of 27 October)."""
    masses, components = material[0], material[1]
    if variant == UNION:
        return union_concepts(components, masses, score)
    four, regions = (material[2], material[3]) if len(material) > 2 else (None, None)
    return detections(variant, components, masses, score, *size, components4=four, regions=regions)


def _event_material(event: dict, variant: str, score: str, size: tuple[int, int]) -> list:
    """:return: per image the union material or the detections of ``variant`` (``None``: none)."""
    return [
        None if image["material"] is None else _detections(variant, image["material"], score, size)
        for image in event["images"]
    ]


#: Per-event arrays of the fine grid, each event × threshold.
FINE_METRICS = ("ap", "rbdc", "rbdr")
#: Metrics also computed over the dataset's images at once (``pooled``; the AP from 28 October
#: 2026, for the false positives of the negative images of real use).
POOLED_METRICS = ("ap", "rbdc", "rbdr")


def _negative_material(negatives: list[dict], variant: str, score: str) -> list:
    """:return: per negative image the union material or the detections of ``variant``."""
    return [_detections(variant, image["material"], score, image["size"]) for image in negatives]


def _pooled(
    variant: str,
    prepared: list,
    matches: list,
    objects: int,
    frames: int,
    thresholds,
    ap_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:return: ``(ap, rbdc, rbdr)`` per threshold over every image of a dataset at once
    (RBDC and RBDR NaN with ``ap_only``)."""
    nothing = np.full(len(thresholds), np.nan)
    if variant == UNION:
        ap = ap_by_threshold(union_precision(prepared, thresholds))
        area, rate = (nothing, nothing) if ap_only else union_rbdc(prepared, thresholds, frames)
    else:
        scores = [s for s, _ in prepared]
        ap = ap_by_threshold(accumulate(scores, matches, objects))
        area, rate = (nothing, nothing) if ap_only else box_rbdc(prepared, thresholds, frames)
    return ap, area, rate


def _json_values(values: np.ndarray) -> list:
    """:return: ``values`` as a list with ``None`` for NaN, which JSON cannot hold."""
    return [None if np.isnan(x) else float(x) for x in np.asarray(values, dtype=np.float64)]


def aggregate_present(events: list[dict], values: np.ndarray, classes: list[str]) -> dict:
    """Event → class → all-class means at every threshold, skipping the events without a value.

    :func:`evd.eval.localization_ap.aggregate_by_threshold` when every event has a value; an
    event with NaN (no chunk holds its frames) is left out of its class, and a
    class with no event left out of the all-class mean, both counted.

    :param events: ``{"event_id", "class"}`` per event, in the rows' order.
    :param values: ``E×T`` per-event values, NaN where the event has none.
    :param classes: the thirteen event classes.
    :return: ``{"macro": [T], "per_class": {class: [T] or None}, "events_without_value",
        "classes_without_value"}``.
    """
    values = np.asarray(values, dtype=np.float64)
    present = ~np.isnan(values).any(axis=1)
    if present.all():
        out = aggregate_by_threshold(events, values, classes)
        return {**out, "events_without_value": 0, "classes_without_value": []}
    per_class = {}
    for c in classes:
        rows = [i for i, e in enumerate(events) if e["class"] == c and present[i]]
        per_class[c] = values[rows].mean(axis=0).tolist() if rows else None
    kept = [v for v in per_class.values() if v is not None]
    return {
        "macro": np.mean(kept, axis=0).tolist() if kept else None,
        "per_class": per_class,
        "events_without_value": int((~present).sum()),
        "classes_without_value": [c for c, v in per_class.items() if v is None],
    }


def _evaluate_fine(payload: dict) -> dict:
    """Per-event AP, RBDC and RBDR at every threshold of :data:`IOU_THRESHOLDS_FINE`.

    The pooled metrics take every event image of the dataset and, in real use, the negative
    images (``payload["negatives"]``, with ``payload["empty_negatives"]`` more that have no
    detection): no object, the same under every GT reading.

    :return: the class and all-class means of each metric per combination, with the pooled
        AP, RBDC and RBDR; the per-event arrays travel in ``"arrays"``, metric → variant × score
        × GT reading × event × threshold, and are written apart from the JSON.
    """
    name, classes, sizes = payload["name"], payload["classes"], payload["sizes"]
    events = payload["events"]
    negatives = payload.get("negatives", [])
    empty = int(payload.get("empty_negatives", 0))
    variants = payload.get("variants", VARIANTS)
    gt_modes = payload.get("gt_modes", GT_MODES)
    # AP only (28 October 2026): the official table needs no RBDC/RBDR, which cost the most.
    ap_only = bool(payload.get("ap_only", False))
    thresholds = IOU_THRESHOLDS_FINE
    shape = (len(variants), len(SCORES), len(gt_modes), len(events), len(thresholds))
    arrays = {m: np.zeros(shape) for m in FINE_METRICS}
    pooled = {
        m: np.zeros((len(variants), len(SCORES), len(gt_modes), len(thresholds)))
        for m in POOLED_METRICS
    }
    # An event with no image (no chunk holds its frames) has no AP: NaN, and the means skip it.
    # Online, an event the detector misses has images with no detection and AP 0 (30 October).
    imageless = [e for e, event in enumerate(events) if not event["images"]]
    for m in FINE_METRICS:
        arrays[m][:, :, :, imageless] = np.nan
    flagged = payload.get("flagged", [])
    confidence = np.full((len(variants), len(SCORES), len(flagged)), -np.inf)
    boxes = np.zeros((len(variants), len(SCORES), len(flagged)), dtype=np.int64)
    results: dict = {}
    for v, variant in enumerate(variants):
        for s, score in enumerate(SCORES):
            for k, chunk in enumerate(flagged):
                confidence[v, s, k], boxes[v, s, k] = chunk_confidence(
                    chunk["planes"], variant, score, chunk["size"]
                )
            everything = {mode: [] for mode in gt_modes}
            matched = {mode: [] for mode in gt_modes}
            objects = dict.fromkeys(gt_modes, 0)
            for e, event in enumerate(events):
                if not event["images"]:
                    continue
                built = _event_material(event, variant, score, sizes[event["event_id"]])
                for g, mode in enumerate(gt_modes):
                    images = [
                        Image(
                            truths=ground_truth(image["boxes"], mode),
                            **{("concepts" if variant == UNION else "detections"): made or []},
                        )
                        for image, made in zip(event["images"], built, strict=True)
                    ]
                    nothing = np.full(len(thresholds), np.nan)
                    if variant == UNION:
                        prepared = prepare_union(images)
                        precision = union_event_precision(images, thresholds, prepared=prepared)
                        area, rate = (
                            (nothing, nothing) if ap_only else union_rbdc(prepared, thresholds)
                        )
                    else:
                        prepared = prepare_boxes(images)
                        matches = box_matches(prepared, thresholds)
                        precision = box_event_precision(
                            images, thresholds, prepared=prepared, matches=matches
                        )
                        area, rate = (
                            (nothing, nothing) if ap_only else box_rbdc(prepared, thresholds)
                        )
                        matched[mode].extend(matches)
                    arrays["ap"][v, s, g, e] = ap_by_threshold(precision)
                    arrays["rbdc"][v, s, g, e] = area
                    arrays["rbdr"][v, s, g, e] = rate
                    everything[mode].extend(prepared)
                    objects[mode] += sum(len(i.truths) for i in images)
            # The negative images have no object: one preparation serves every GT reading.
            kind = "concepts" if variant == UNION else "detections"
            negative = [
                Image(truths=[], **{kind: made})
                for made in _negative_material(negatives, variant, score)
            ]
            negative = (prepare_union if variant == UNION else prepare_boxes)(negative)
            false = [np.zeros((len(thresholds), len(item[0])), dtype=bool) for item in negative]
            for g, mode in enumerate(gt_modes):
                frames = len(everything[mode]) + len(negative) + empty
                if not everything[mode]:
                    # No image at all (no chunk holds a GT frame): no pooled value either.
                    for m in POOLED_METRICS:
                        pooled[m][v, s, g] = np.nan
                    results.setdefault(variant, {}).setdefault(score, {})[mode] = {
                        "ap": aggregate_present(events, arrays["ap"][v, s, g], classes),
                        "pooled": {"ap": None},
                    }
                    continue
                values = _pooled(
                    variant,
                    everything[mode] + negative,
                    matched[mode] + false,
                    objects[mode],
                    frames,
                    thresholds,
                    ap_only,
                )
                for m, value in zip(POOLED_METRICS, values, strict=True):
                    pooled[m][v, s, g] = value
                metrics = ("ap",) if ap_only else FINE_METRICS
                results.setdefault(variant, {}).setdefault(score, {})[mode] = {
                    **{m: aggregate_present(events, arrays[m][v, s, g], classes) for m in metrics},
                    "pooled": {
                        m: _json_values(pooled[m][v, s, g])
                        for m in (("ap",) if ap_only else POOLED_METRICS)
                    },
                }
    return {
        "configuration": name,
        "iou_grid": "fine",
        "iou_thresholds": thresholds.tolist(),
        "variants_evaluated": list(variants),
        "gt_modes_evaluated": list(gt_modes),
        "ap_only": ap_only,
        "events": [
            {"event_id": e["event_id"], "class": e["class"], "images": len(e["images"])}
            for e in events
        ],
        "negative_images": {"with_detections_material": len(negatives), "empty": empty},
        "variants": results,
        "arrays": {
            **arrays,
            **{f"pooled_{m}": a for m, a in pooled.items()},
            **({"chunk_confidence": confidence, "chunk_boxes": boxes} if flagged else {}),
        },
    }


def gt_statistics(events: list[dict]) -> dict:
    """:return: images, GT boxes and objects of each reading, and how often they differ."""
    boxes = [image["boxes"] for event in events for image in event["images"]]
    groups = [len(touching_groups(b)) for b in boxes]
    return {
        "images": len(boxes),
        "objects": {
            "union": len(boxes),
            "instances": sum(len(b) for b in boxes),
            "touching": sum(groups),
        },
        "images_with_several_boxes": sum(len(b) > 1 for b in boxes),
        "images_where_touching_joins_boxes": sum(
            n < len(b) for n, b in zip(groups, boxes, strict=True)
        ),
        "images_where_touching_keeps_several_objects": sum(n > 1 for n in groups),
    }


def negative_images(complete: dict, chunks: list[dict], intervals, names: list[str]) -> tuple:
    """Material of the negative images of one video under every configuration (real use).

    :return: ``(material, empty, images, frames)``: per configuration the material of the
        negative images its chunk localizes, per configuration the negative images with no
        material (chunk not flagged, or no class chosen), and the video's negative images and
        native frames.
    """
    from localization_auto_evaluate import negative_planes

    planes = negative_planes(complete, chunks, intervals)
    size = (complete["video"]["width"], complete["video"]["height"])
    material = {n: [] for n in names}
    empty = dict.fromkeys(names, 0)
    for index, temporal in sorted(planes):
        for name in names:
            key = chunks[index]["methods"][name]["partition"]
            if key is None:
                empty[name] += 1
                continue
            material[name].append(
                {"material": plane_material(chunks[index], key, temporal), "size": size}
            )
    return material, empty, len(planes), sum(planes.values())


#: Categories of a flagged chunk for the FP metric (``docs/varie/metrica_fp_localizzazione_
#: 20261028.md`` §1.2): ``T`` holds GT frames (the chunks of the online AP), ``F`` meets no
#: temporal event (a false positive), ``Z`` meets an event without holding a GT frame.
CHUNK_CATEGORIES = ("T", "F", "Z")


def flagged_chunk_material(
    complete: dict, chunks: list[dict], truth: dict | None, intervals, names: list[str]
) -> tuple[list[dict], dict]:
    """The flagged chunks of one video with their category and every plane's material.

    :param truth: the video's GT, ``None`` on a video without spatial GT (its flagged chunks
        are ``F`` or ``Z``).
    :param intervals: ``(start, end, ...)`` of every temporal event of the video.
    :return: ``(info, material)``: per flagged chunk ``{"window_id", "category"}``, and per
        configuration the chunk's planes (:func:`plane_material`, ``None`` when no class is
        chosen) with the frame size and its positive planes (:func:`positive_planes`, empty
        outside ``T``).
    """
    from localization_auto_evaluate import chunk_windows

    positive = positive_planes(truth, complete, chunks) if truth is not None else {}
    holding = set(positive)
    size = (complete["video"]["width"], complete["video"]["height"])
    info, material = [], {n: [] for n in names}
    for index, (window, chunk) in enumerate(zip(chunk_windows(chunks), chunks, strict=True)):
        if not (chunk["localized"] and chunk.get("flagged")):
            continue
        meets = any(
            int(a) <= window.end_frame and window.start_frame <= int(b) for a, b, *_ in intervals
        )
        category = "T" if index in holding else ("Z" if meets else "F")
        info.append({"window_id": chunk["window_id"], "category": category})
        for name in names:
            key = chunk["methods"][name]["partition"]
            planes = (
                None
                if key is None
                else [plane_material(chunk, key, t) for t in range(int(chunk["grid"][0]))]
            )
            material[name].append(
                {"planes": planes, "size": size, "positive": sorted(positive.get(index, ()))}
            )
    return info, material


def positive_planes(truth: dict, complete: dict, chunks: list[dict]) -> dict[int, set[int]]:
    """:return: per flagged chunk index holding GT frames (``T``), the planes on which the
    online AP shows them: the positive planes ``P⁺`` of ``docs/varie/proposte_metriche_fp_
    localizzazione_20261029.md`` §2."""
    out: dict[int, set[int]] = {}
    for _, _, _, index, temporal, _ in online_frames(truth, complete, chunks):
        if chunks[index]["flagged"]:
            out.setdefault(index, set()).add(temporal)
    return out


#: Categories whose box scores :func:`box_score_histograms` keeps.
BOX_CATEGORIES = ("T", "F")
#: Bin edges of the box score histograms: 0, then ``10^(k/50)`` from 1e-9 to 1e5. Every power
#: of ten, 1 included, is an exact edge, so the mean kept share of false boxes over ``[0, 1]``,
#: or between two powers of ten, is exact from the histograms.
SCORE_EDGES = np.concatenate(([0.0], 10.0 ** (np.arange(-450, 251) / 50)))


def box_scores(planes: list | None, variant: str, score: str, size) -> np.ndarray:
    """Score of every detection on the planes of a chunk: the box inventory of the FP metrics.

    A composite detection counts once whatever its rectangles. On the cumulative union a plane
    shows one region, the union of the concepts at or above the threshold, so it holds one
    detection scored by its highest concept with a rectangle there: visible at ``θ`` exactly
    when the union is not empty (``docs/varie/proposte_metriche_fp_localizzazione_20261029.md``
    §4, the inventory to declare).

    :param planes: :func:`plane_material` of every plane, ``None`` when no class is chosen.
    :return: the scores, plane after plane (empty without any detection).
    """
    found = []
    for material in planes or ():
        made = _detections(variant, material, score, size)
        if variant == UNION:
            values = [c for c, rects in made if rects]
            found.extend([max(values)] if values else [])
        else:
            found.extend(d.score for d in made)
    return np.asarray(found, dtype=np.float64)


def chunk_confidence(planes: list | None, variant: str, score: str, size) -> tuple[float, int]:
    """Confidence and detection count of a chunk under one variant, for the FP metrics.

    The count is the ``n(c)`` of ``B_FP`` (``docs/varie/proposte_metriche_fp_localizzazione_
    20261029.md`` §4), the size of the :func:`box_scores` inventory: on the cumulative union a
    plane counts one if its union is not empty.

    :return: ``(S_C(c), n(c))``: the highest detection score over the planes (``-inf``
        without any detection) and the detections over the planes.
    """
    values = box_scores(planes, variant, score, size)
    return (float(values.max()) if len(values) else -np.inf), len(values)


def score_histogram(values: np.ndarray) -> np.ndarray:
    """:return: ``2×B`` count and score sum of ``values`` in the bins of :data:`SCORE_EDGES`:
    bin 0 below 0, bin ``k`` in ``[edges[k-1], edges[k])``, the last at or above 1e5."""
    values = np.asarray(values, dtype=np.float64)
    index = np.searchsorted(SCORE_EDGES, values, side="right")
    size = len(SCORE_EDGES) + 1
    return np.stack(
        [
            np.bincount(index, minlength=size).astype(np.float64),
            np.bincount(index, weights=values, minlength=size),
        ]
    )


#: Targets of ``F_box@R⁺_box(q)`` in percent of the boxes emitted on positive planes.
RETENTION_GRID = np.arange(1, 101)
#: Intervals, in percent, of the exact continuous mean ``F̄_box,B[a, b]``.
RETENTION_INTERVALS = ((70, 95), (80, 95))


def box_retention(positive: np.ndarray, negative: np.ndarray) -> dict:
    """``F_box@R⁺_box(q)`` and its means: false boxes kept when the threshold keeps ``q`` of the
    boxes emitted on positive planes.

    ``docs/varie/proposte_metriche_fp_localizzazione_20261029.md`` §4 «Definizione 1 sui box»
    and §5: the threshold is the ``ceil(q·N⁺)``-th highest positive score, applied to every box
    alone; ``F_box`` is the share of the false boxes at or above it. Exact, from the sorted
    scores: ``f_B`` is a step function of ``q``, constant on ``((k-1)/N⁺, k/N⁺]``, so the
    continuous mean over ``[a, b]`` weighs each step by its overlap with the interval.

    :param positive: scores of the boxes on the positive planes ``P⁺`` (``D⁺``).
    :param negative: scores of the boxes on the planes of the false positive chunks (``D⁻``).
    :return: ``{"positive", "negative", "f", "r", "theta", "mean", "auroc"}``: the inventory
        sizes, per :data:`RETENTION_GRID` target the kept false share, the positive retention
        reached and the threshold, the continuous means over :data:`RETENTION_INTERVALS`, and
        the AUROC of the individual scores (``D⁺`` against ``D⁻``, ties half); NaN without
        boxes of a kind.
    """
    pos = np.sort(np.asarray(positive, dtype=np.float64))[::-1]
    neg = np.sort(np.asarray(negative, dtype=np.float64))
    n, m = len(pos), len(neg)
    nan = np.full(len(RETENTION_GRID), np.nan)
    out = {
        "positive": n,
        "negative": m,
        "f": nan,
        "r": nan.copy(),
        "theta": nan.copy(),
        "mean": np.full(len(RETENTION_INTERVALS), np.nan),
        "auroc": np.nan,
    }
    if not n or not m:
        return out
    # Kept false share at each positive score, the k-th highest first.
    kept = (m - np.searchsorted(neg, pos, side="left")) / m
    rank = (RETENTION_GRID * n + 99) // 100  # ceil(q·N⁺) in integers, q in percent
    theta = pos[rank - 1]
    ascending = pos[::-1]
    out["f"] = kept[rank - 1]
    out["r"] = (n - np.searchsorted(ascending, theta, side="left")) / n
    out["theta"] = theta
    steps = np.arange(n + 1) / n
    for i, (low, high) in enumerate(RETENTION_INTERVALS):
        a, b = low / 100, high / 100
        overlap = np.clip(np.minimum(b, steps[1:]) - np.maximum(a, steps[:-1]), 0, None)
        out["mean"][i] = float(overlap @ kept / (b - a))
    below = np.searchsorted(neg, pos, side="left")
    ties = np.searchsorted(neg, pos, side="right") - below
    out["auroc"] = float((below.sum() + 0.5 * ties.sum()) / (n * m))
    return out


#: Intervals, in percent, of the exact continuous mean of ``B^σ_FP@TPR_P``: the recommended
#: ``[0.80, 0.95]`` (30 October 2026) and ``[0.70, 0.95]``, the grid of ``B̄_FP``, beside it.
FP_BOX_INTERVALS = ((80, 95), (70, 95))


def false_boxes_at_planes(
    positive: np.ndarray, negative: np.ndarray, chunk_max: np.ndarray, chunks: int
) -> dict:
    """``B^σ_FP@TPR_P(q)``: false boxes still shown per false positive chunk, each box filtered
    by its own score at the threshold that keeps a box on ``q`` of the positive planes.

    ``docs/varie/proposte_metriche_fp_localizzazione_20261029.md`` §4 (``FPPI_F(θ_{P,q})``,
    counted per false positive chunk instead of per plane): ``θ_{P,q}`` is the
    ``ceil(q·|P⁺|)``-th highest plane maximum ``S_P`` of the positive planes, those without a
    box at ``-inf``, where the target is unreachable. The value splits into the share of false
    positive chunks still showing a box (``far``, the FAR of ``S_C`` at that threshold) times
    the boxes each of them shows. Exact: the curve is a step function of ``q``, constant on
    ``((k-1)/|P⁺|, k/|P⁺|]``, and the continuous mean weighs each step by its overlap; a mean
    whose interval meets an unreachable step is unavailable.

    :param positive: ``S_P`` of every positive plane, ``-inf`` without a box.
    :param negative: scores of every box on the planes of the false positive chunks.
    :param chunk_max: ``S_C`` of every false positive chunk, ``-inf`` without a box.
    :param chunks: ``|F|``.
    :return: ``{"planes", "planes_with_boxes", "chunks", "b", "far", "tpr", "theta",
        "mean"}``: per :data:`RETENTION_GRID` target the false boxes per false positive chunk,
        the share of those chunks showing a box, the positive-plane share reached and the
        threshold (NaN where unreachable), and the means over :data:`FP_BOX_INTERVALS`.
    """
    pos = np.sort(np.asarray(positive, dtype=np.float64))[::-1]
    neg = np.sort(np.asarray(negative, dtype=np.float64))
    tops = np.sort(np.asarray(chunk_max, dtype=np.float64))
    n = len(pos)
    nan = np.full(len(RETENTION_GRID), np.nan)
    out = {
        "planes": n,
        "planes_with_boxes": int(np.isfinite(pos).sum()),
        "chunks": int(chunks),
        "b": nan,
        "far": nan.copy(),
        "tpr": nan.copy(),
        "theta": nan.copy(),
        "mean": np.full(len(FP_BOX_INTERVALS), np.nan),
    }
    if not n or not chunks:
        return out
    finite = np.isfinite(pos)
    # Per step k (the k-th highest plane maximum as threshold): boxes and chunks kept.
    boxes = np.where(finite, (len(neg) - np.searchsorted(neg, pos, side="left")) / chunks, np.nan)
    shown = np.where(finite, (len(tops) - np.searchsorted(tops, pos, side="left")) / chunks, np.nan)
    rank = (RETENTION_GRID * n + 99) // 100
    theta = pos[rank - 1]
    reach = np.isfinite(theta)
    out["b"] = np.where(reach, boxes[rank - 1], np.nan)
    out["far"] = np.where(reach, shown[rank - 1], np.nan)
    ascending = pos[::-1]
    out["tpr"] = np.where(reach, (n - np.searchsorted(ascending, theta, side="left")) / n, np.nan)
    out["theta"] = np.where(reach, theta, np.nan)
    steps = np.arange(n + 1) / n
    for i, (low, high) in enumerate(FP_BOX_INTERVALS):
        a, b = low / 100, high / 100
        overlap = np.clip(np.minimum(b, steps[1:]) - np.maximum(a, steps[:-1]), 0, None)
        used = overlap > 0
        if np.all(finite[used]):
            out["mean"][i] = float(overlap[used] @ boxes[used] / (b - a))
    return out


#: The flagged chunks of the dataset being processed, inherited by the forked workers.
_BOX_MATERIAL: dict = {}


def _box_histograms(task: tuple[str, str]) -> tuple[str, str, np.ndarray]:
    """:return: the configuration, the variant and its histograms, score × category × 2 × bin."""
    name, variant = task
    material, categories = _BOX_MATERIAL["material"][name], _BOX_MATERIAL["categories"]
    out = np.zeros((len(SCORES), len(BOX_CATEGORIES), 2, len(SCORE_EDGES) + 1))
    for s, score in enumerate(SCORES):
        for chunk, category in zip(material, categories, strict=True):
            if category in BOX_CATEGORIES:
                values = box_scores(chunk["planes"], variant, score, chunk["size"])
                out[s, BOX_CATEGORIES.index(category)] += score_histogram(values)
    return name, variant, out


def _box_retention(task: tuple[str, str]) -> tuple[str, str, list[dict]]:
    """:return: the configuration, the variant and per score its :func:`box_retention`."""
    name, variant = task
    material, categories = _BOX_MATERIAL["material"][name], _BOX_MATERIAL["categories"]
    out = []
    for score in SCORES:
        found = {"T": [np.zeros(0)], "F": [np.zeros(0)]}
        for chunk, category in zip(material, categories, strict=True):
            if category == "T" and chunk["planes"] is not None:
                planes = [chunk["planes"][t] for t in chunk["positive"]]
                found["T"].append(box_scores(planes, variant, score, chunk["size"]))
            elif category == "F":
                found["F"].append(box_scores(chunk["planes"], variant, score, chunk["size"]))
        out.append(box_retention(np.concatenate(found["T"]), np.concatenate(found["F"])))
    return name, variant, out


def _false_boxes(task: tuple[str, str]) -> tuple[str, str, list[dict]]:
    """:return: the configuration, the variant and per score its :func:`false_boxes_at_planes`."""
    name, variant = task
    material, categories = _BOX_MATERIAL["material"][name], _BOX_MATERIAL["categories"]
    out = []
    for score in SCORES:
        planes, negative, tops = [], [np.zeros(0)], []
        for chunk, category in zip(material, categories, strict=True):
            if category == "T":
                # A positive plane without material (no class chosen) shows no box.
                for t in chunk["positive"]:
                    found = (
                        box_scores([chunk["planes"][t]], variant, score, chunk["size"])
                        if chunk["planes"] is not None
                        else np.zeros(0)
                    )
                    planes.append(found.max() if len(found) else -np.inf)
            elif category == "F":
                found = box_scores(chunk["planes"], variant, score, chunk["size"])
                negative.append(found)
                tops.append(found.max() if len(found) else -np.inf)
        out.append(
            false_boxes_at_planes(
                np.array(planes), np.concatenate(negative), np.array(tops), len(tops)
            )
        )
    return name, variant, out


def false_boxes_step(args) -> None:
    """``B^σ_FP@TPR_P(q)`` and its continuous means for every row of an online evaluation.

    :func:`false_boxes_at_planes` on the positive planes (the planes of the flagged chunks that
    show a GT frame online, each counted once) and the false positive chunks. Writes
    ``false_boxes_<dataset>.npz`` in ``--evaluation``, each array configuration × variant ×
    score × …: ``planes``, ``planes_with_boxes``, ``chunks``; ``b``, ``far``, ``tpr`` and
    ``theta`` per target of ``q`` (``q`` in percent); ``mean`` per interval of ``intervals``.
    """
    for folder, names, variants, _, found in _each_dataset(args, _false_boxes):
        shape = (len(names), len(variants), len(SCORES))
        grid = (*shape, len(RETENTION_GRID))
        arrays = {
            "planes": np.zeros(shape, dtype=np.int64),
            "planes_with_boxes": np.zeros(shape, dtype=np.int64),
            "chunks": np.zeros(shape, dtype=np.int64),
            **{k: np.zeros(grid) for k in ("b", "far", "tpr", "theta")},
            "mean": np.zeros((*shape, len(FP_BOX_INTERVALS))),
        }
        for (name, variant), per_score in found.items():
            c, v = names.index(name), variants.index(variant)
            for s, result in enumerate(per_score):
                for key, value in result.items():
                    arrays[key][c, v, s] = value
        np.savez_compressed(
            args.evaluation / f"false_boxes_{folder}.npz",
            q=RETENTION_GRID,
            intervals=np.array(FP_BOX_INTERVALS),
            **arrays,
        )


def _each_dataset(args, worker):
    """Run ``worker`` on every (configuration, variant) of the flagged chunks of each dataset.

    The flagged chunks are rebuilt from the stage and must equal, one by one, those of the
    online evaluation's ``summary.json``, so what the worker computes aligns with its
    ``chunk_confidence`` and ``chunk_boxes``. Workers are forked and inherit the material.

    :return: per dataset folder ``(names, variants, flagged, {(name, variant): result})``.
    :raises ValueError: if the stage is not sealed, its configurations are not the
        evaluation's, or its flagged chunks differ from the evaluation's.
    """
    from localization_auto_evaluate import sealed_videos

    global _BOX_MATERIAL
    intervals = read_json(args.intervals)["intervals"]
    summary = read_json(args.evaluation / "summary.json")
    variants = list(summary["variants"])
    for folder, dataset in DATASETS.items():
        seal = read_json(args.predictions / "operations" / f"seal-{folder}.json")
        if seal.get("status") != "complete" or seal["dataset"] != dataset:
            raise ValueError("Predictions must be complete and sealed for the whole dataset")
        names = list(seal["methods"])
        if read_json(args.evaluation / f"axes_{folder}.json")["configuration"] != names:
            raise ValueError(f"{folder}: the stage's configurations are not the evaluation's")
        flagged, material = [], {n: [] for n in names}
        for entry, complete, chunks in sealed_videos(args.predictions, seal, lambda e: True):
            if chunks is None:
                continue
            video = entry["video_id"]
            path = args.snapshot / dataset / f"{video}.json"
            info, found = flagged_chunk_material(
                complete,
                chunks,
                read_json(path) if path.exists() else None,
                intervals[folder][video],
                names,
            )
            flagged.extend({"video_id": video, **i} for i in info)
            for n in names:
                material[n].extend(found[n])
        # The evaluation writes no list when the detector flags nothing in a dataset.
        if flagged != summary["datasets"][folder].get("flagged_chunks", []):
            raise ValueError(f"{folder}: the flagged chunks differ from the evaluation's")
        _BOX_MATERIAL = {"material": material, "categories": [f["category"] for f in flagged]}
        tasks = [(n, v) for n in names for v in variants]
        # Forked workers inherit the material; one task each keeps their copied pages few.
        pool = (
            multiprocessing.get_context("fork").Pool(args.workers, maxtasksperchild=1)
            if args.workers > 1
            else None
        )
        results = pool.imap_unordered(worker, tasks) if pool is not None else map(worker, tasks)
        found = {(name, variant): out for name, variant, out in results}
        if pool is not None:
            pool.close()
            pool.join()
        _BOX_MATERIAL = {}
        print(f"{folder}: {len(flagged)} flagged chunks, {len(tasks)} tasks", flush=True)
        yield folder, names, variants, flagged, found


def box_retention_step(args) -> None:
    """``F_box@R⁺_box(q)`` and ``F̄_box,B`` of every row of an online evaluation, exactly.

    The «definition 1 on boxes» of ``docs/varie/proposte_metriche_fp_localizzazione_
    20261029.md`` §4–5 (:func:`box_retention`): the positive boxes are those emitted on the
    positive planes (the planes of the flagged chunks that show a GT frame online), the false
    ones those on every plane of the false positive chunks.

    Writes ``box_retention_<dataset>.npz`` in ``--evaluation``, each array configuration ×
    variant × score × …: ``positive`` and ``negative`` (box counts), ``f``, ``r`` and ``theta``
    per target of ``q`` (``q`` in percent), ``mean`` per interval of ``intervals``, ``auroc``.
    """
    for folder, names, variants, _, found in _each_dataset(args, _box_retention):
        shape = (len(names), len(variants), len(SCORES))
        arrays = {
            "positive": np.zeros(shape, dtype=np.int64),
            "negative": np.zeros(shape, dtype=np.int64),
            "f": np.zeros((*shape, len(RETENTION_GRID))),
            "r": np.zeros((*shape, len(RETENTION_GRID))),
            "theta": np.zeros((*shape, len(RETENTION_GRID))),
            "mean": np.zeros((*shape, len(RETENTION_INTERVALS))),
            "auroc": np.zeros(shape),
        }
        for (name, variant), per_score in found.items():
            c, v = names.index(name), variants.index(variant)
            for s, result in enumerate(per_score):
                for key, value in result.items():
                    arrays[key][c, v, s] = value
        np.savez_compressed(
            args.evaluation / f"box_retention_{folder}.npz",
            q=RETENTION_GRID,
            intervals=np.array(RETENTION_INTERVALS),
            **arrays,
        )


def box_score_histograms(args) -> None:
    """Histograms of the score of every box on the flagged T and F chunks of an online evaluation.

    The material of the mean kept share of false boxes over the confidence thresholds, with no
    TPR (``docs/varie/proposte_metriche_fp_localizzazione_20261029.md`` §5): ``F̄_box^θ[a, b]`` is
    the mean over the false boxes of ``clip((σ - a)/(b - a), 0, 1)``, exact from the count and
    score sum of each bin when ``a`` and ``b`` are edges of :data:`SCORE_EDGES`. The ``T``
    category holds every plane of the true positive chunks, not only the positive ones.

    Writes ``box_scores_<dataset>.npz`` in ``--evaluation``: ``histogram``, configuration ×
    variant × score × category (:data:`BOX_CATEGORIES`) × (count, score sum) × bin, and
    ``edges``.

    :raises ValueError: as :func:`_each_dataset`.
    """
    for folder, names, variants, _, found in _each_dataset(args, _box_histograms):
        histogram = np.zeros(
            (len(names), len(variants), len(SCORES), len(BOX_CATEGORIES), 2, len(SCORE_EDGES) + 1)
        )
        for (name, variant), out in found.items():
            histogram[names.index(name), variants.index(variant)] = out
        np.savez_compressed(
            args.evaluation / f"box_scores_{folder}.npz", histogram=histogram, edges=SCORE_EDGES
        )


#: The evaluated frames of the localization metrics: every GT frame (until 30 October 2026) or
#: one per second of the video (``1fps``, the official protocol from 30 October 2026).
FRAME_MODES = ("all", "1fps")


def frame_grid(num_frames: int, fps: float) -> np.ndarray:
    """``G_v``: the native frame nearest to every whole second of the video.

    ``n_k = floor(k·fps + 1/2)`` (half-up, as ADR-024) for ``k = 0, 1, ...`` while
    ``n_k <= num_frames - 1`` (note §2.1). The grid belongs to the video: it depends neither on
    the events nor on the rows nor on the model.

    :param num_frames: native frames of the video.
    :param fps: native frame rate, not necessarily an integer.
    :return: the frames, ascending, without repetitions.
    :raises ValueError: on a video without frames or a non-positive frame rate.
    """
    if num_frames < 1 or not fps > 0:
        raise ValueError(f"No 1 fps grid for {num_frames} frames at {fps} fps")
    seconds = np.arange(int(math.ceil((num_frames - 1) / fps)) + 2)
    frames = np.floor(seconds * float(fps) + 0.5).astype(np.int64)
    return np.unique(frames[frames <= num_frames - 1])


def annotated_frames(event: dict) -> list[int]:
    """:return: the annotated frames of a GT event, reviewed with at least one box, ascending."""
    return sorted(
        {
            frame
            for plane in event["planes"]
            for segment in plane["segments"]
            if segment["boxes"]
            for frame in range(segment["start"], segment["end"] + 1)
        }
    )


def event_extent(frames: list[int], intervals, num_frames: int) -> tuple[int, int]:
    """``[α_e, β_e]``: the temporal event holding the annotated frames of a GT event.

    The GT snapshot keeps the event's planes, not its interval: the interval is the narrowest
    temporal event of the video (native frames, ends included, clipped to the video as the
    snapshot does) that contains every annotated frame.

    :raises ValueError: if no temporal event contains them all.
    """
    found = []
    for start, end, *_ in intervals:
        a, b = max(0, int(start)), min(num_frames - 1, int(end))
        if a <= frames[0] and frames[-1] <= b:
            found.append((b - a, a, b))
    if not found:
        raise ValueError("The annotated frames of an event lie in no temporal event")
    _, a, b = min(found)
    return a, b


def evaluated_frames(truth: dict, fps: float, intervals) -> tuple[np.ndarray, dict]:
    """``F_v``: the 1 fps grid plus one annotated frame for each event that has none on it.

    D4 of the note (§2.1): an event with no frame of the grid receives the annotated frame
    nearest to the centre of its interval, the first on a tie, so no event leaves the AP. The
    rule applies only when the temporal interval contains no grid frame (D4), not when
    grid frames exist but lack spatial annotations.

    :param truth: the video's GT.
    :param fps: native frame rate.
    :param intervals: ``(start, end, ...)`` of the video's temporal events, native frames.
    :return: ``(frames, extra)``: ``F_v`` ascending, and per event given a frame
        ``{"frame", "extent", "short"}``.
    """
    grid = frame_grid(int(truth["num_frames"]), fps)
    on_grid = set(grid.tolist())
    extra = {}
    for event in truth["events"]:
        frames = annotated_frames(event)
        if not frames or on_grid.intersection(frames):
            continue
        a, b = event_extent(frames, intervals, int(truth["num_frames"]))
        inside = np.searchsorted(grid, b, side="right") - np.searchsorted(grid, a, side="left")
        if inside:
            continue
        centre = (a + b) / 2
        chosen = min(frames, key=lambda n: (abs(n - centre), n))
        extra[event["event_id"]] = {"frame": int(chosen), "extent": [a, b], "short": not inside}
    added = [e["frame"] for e in extra.values()]
    return np.union1d(grid, np.asarray(added, dtype=np.int64)), extra


def emitting_chunks(complete: dict, chunks: list[dict], video: dict, theta: float) -> list[bool]:
    """``E_v`` (D1): per chunk whether its head score reaches θ_O, from the OVAD plan.

    The plan's chunks are the OVAD extents ``[a/fps, (b+1)/fps)`` in start order: each must be
    the protocol window of the stage's chunk at the same position, or the scores would go to
    the wrong windows.

    :param video: the video's entry of ``localization_class_ovad_plan.py``.
    :raises ValueError: on another number of chunks or another extent, or on an emitting chunk
        the stage did not localize (θ_O below the chunk threshold of the stage).
    """
    from localization_auto_evaluate import chunk_windows

    fps = float(complete["video"]["fps"])
    windows = chunk_windows(chunks)
    if len(windows) != len(video["extents"]):
        raise ValueError(f"{complete['video']['video_id']}: OVAD and stage chunks differ in number")
    out = []
    for window, (a, b), score, chunk in zip(
        windows, video["extents"], video["scores"], chunks, strict=True
    ):
        if not (
            math.isclose(window.start_frame / fps, a, abs_tol=1e-6)
            and math.isclose((window.end_frame + 1) / fps, b, abs_tol=1e-6)
        ):
            raise ValueError(f"{chunk['window_id']}: the OVAD extent is not the chunk's window")
        shown = bool(score >= theta)
        if shown and not chunk["localized"]:
            raise ValueError(f"{chunk['window_id']}: emitting at θ_O but not localized")
        out.append(shown)
    return out


def fpr_frames(
    truth: dict | None, complete: dict, chunks: list[dict], video: dict, theta: float, intervals
) -> dict:
    """The frames of the FP metric in one video, each with the chunk planes that show it.

    Note §3–5: ``P⁺`` holds the frames of ``F_v`` in the OVAD TP rows that are annotated and
    covered by an emitting chunk (D3); ``N⁻`` the frames of the 1 fps grid in the FP rows
    covered by any clean chunk, including non-emitting ones (D2). Only emitting clean chunks
    supply boxes; a frame covered solely by non-emitting chunks is a true negative. A frame
    belongs to row ``[a, b)`` when ``a <= n/fps < b``, the last row closed on the right; a
    chunk shows it on its nearest plane (``offline_plane``), with or without boxes there.

    :param truth: the video's GT, ``None`` without spatial GT (no positive frame).
    :param video: the video's entry of the OVAD plan.
    :param intervals: the video's temporal events, for ``F_v``.
    :return: ``{"positive", "negative", "counts", "extra"}``: per frame of ``P⁺`` and ``N⁻`` the
        ``(chunk index, plane)`` pairs, the counts of the note §5.1 and the frames of D4.
    :raises ValueError: if a frame lies outside every row, or as :func:`emitting_chunks`.
    """
    from evd.explain.project import offline_plane
    from localization_auto_evaluate import chunk_windows

    fps, num_frames = float(complete["video"]["fps"]), int(complete["video"]["num_frames"])
    windows = chunk_windows(chunks)
    emitting = np.array(emitting_chunks(complete, chunks, video, theta), dtype=bool)
    clean = np.array(video["clean"], dtype=bool)
    first = np.array([w.start_frame for w in windows])
    last = np.array([w.end_frame for w in windows])
    rows = video["rows"]
    starts = np.array([r[0] for r in rows])

    def kind(frame: int) -> str | None:
        t = frame / fps
        i = int(np.searchsorted(starts, t, side="right")) - 1
        if i < 0 or t > rows[-1][1] + 1e-9:
            raise ValueError(f"{complete['video']['video_id']}: frame {frame} outside every row")
        _, _, label, judged, predicted = rows[i]
        if judged and predicted == 1:
            return "TP" if label == 1 else "FP"
        return None

    def planes(frame: int, only_clean: bool) -> list[tuple[int, int]]:
        mask = emitting & (first <= frame) & (last >= frame)
        if only_clean:
            mask &= clean
        return [
            (int(i), offline_plane(windows[i], frame, int(chunks[i]["grid"][0])))
            for i in np.flatnonzero(mask)
        ]

    counts = dict.fromkeys(
        ("positive", "positive_uncovered", "negative", "negative_unjudgeable"), 0
    )
    counts["rows_tp"] = sum(r[2] == 1 and r[3] and r[4] == 1 for r in rows)
    counts["rows_fp"] = sum(r[2] == 0 and r[3] and r[4] == 1 for r in rows)
    counts["emitting_chunks"] = int(emitting.sum())
    positive, negative, extra = [], [], {}
    if truth is not None:
        frames, extra = evaluated_frames(truth, fps, intervals)
        annotated = {f for event in truth["events"] for f in annotated_frames(event)}
        for frame in frames:
            if int(frame) not in annotated or kind(int(frame)) != "TP":
                continue
            found = planes(int(frame), only_clean=False)
            if found:
                positive.append(found)
            else:
                counts["positive_uncovered"] += 1
    for frame in frame_grid(num_frames, fps):
        if kind(int(frame)) != "FP":
            continue
        found = planes(int(frame), only_clean=True)
        if np.any(clean & (first <= frame) & (last >= frame)):
            negative.append(found)
        else:
            counts["negative_unjudgeable"] += 1
    counts["positive"], counts["negative"] = len(positive), len(negative)
    return {"positive": positive, "negative": negative, "counts": counts, "extra": extra}


#: Targets of ``FPR_box@TPR(q)``, in percent (D6): 0.90 is the base, 0.95 the «FPR95» of the
#: out-of-distribution literature (Liang, Li and Srikant, ICLR 2018).
FPR_POINTS = (70, 75, 80, 85, 90, 95)
#: Intervals, in percent, of the exact continuous mean of ``FPR_box@TPR(q)`` (D6).
FPR_INTERVALS = ((80, 90), (80, 95), (70, 95))


def frame_fpr(positive: np.ndarray, negative: np.ndarray) -> dict:
    """``FPR_box@TPR(q)``: the share of FP-row frames still showing a box at the threshold that
    keeps one on ``q`` of the positive frames; points, continuous means, AUROC and ceiling.

    Note §4–5: ``θ_q`` is the ``ceil(q·|P⁺|)``-th highest score of the positive frames (``S``,
    ``-inf`` on a covered frame without a box, which stays in the denominator); a negative
    frame counts when ``S_Γ >= θ_q``. Exact: the curve is a step function of ``q``, constant on
    ``((k-1)/|P⁺|, k/|P⁺|]``, and the continuous mean weighs each step by its overlap with the
    interval. A target whose threshold is ``-inf`` is unreachable (NaN), and so is a mean whose
    interval meets such a step; with ties the reached TPR can exceed ``q``.

    :param positive: ``S(n)`` of the frames of ``P⁺``.
    :param negative: ``S_Γ(n)`` of the frames of ``N⁻``.
    :return: ``{"positive", "negative", "fpr", "tpr", "theta", "mean", "auroc", "ceiling"}``:
        the sizes, per :data:`FPR_POINTS` target the FPR, the TPR reached and ``θ_q``, the means
        over :data:`FPR_INTERVALS`, the AUROC of ``S`` against ``S_Γ`` (ties half, ``-inf``
        included) and the ceiling ``FPR(-inf)``, the share of ``N⁻`` with any box; NaN without
        frames of a kind.
    """
    pos = np.sort(np.asarray(positive, dtype=np.float64))[::-1]
    neg = np.sort(np.asarray(negative, dtype=np.float64))
    n, m = len(pos), len(neg)
    nan = np.full(len(FPR_POINTS), np.nan)
    out = {
        "positive": n,
        "negative": m,
        "fpr": nan,
        "tpr": nan.copy(),
        "theta": nan.copy(),
        "mean": np.full(len(FPR_INTERVALS), np.nan),
        "auroc": np.nan,
        "ceiling": np.nan,
    }
    if not n or not m:
        return out
    out["ceiling"] = float(np.isfinite(neg).mean())
    finite = np.isfinite(pos)
    # Share of the negative frames at or above each positive score, the k-th highest first.
    kept = (m - np.searchsorted(neg, pos, side="left")) / m
    rank = (np.asarray(FPR_POINTS) * n + 99) // 100  # ceil(q·|P⁺|) in integers, q in percent
    theta = pos[rank - 1]
    reach = np.isfinite(theta)
    out["fpr"] = np.where(reach, kept[rank - 1], np.nan)
    ascending = pos[::-1]
    out["tpr"] = np.where(reach, (n - np.searchsorted(ascending, theta, side="left")) / n, np.nan)
    out["theta"] = np.where(reach, theta, np.nan)
    steps = np.arange(n + 1) / n
    for i, (low, high) in enumerate(FPR_INTERVALS):
        a, b = low / 100, high / 100
        overlap = np.clip(np.minimum(b, steps[1:]) - np.maximum(a, steps[:-1]), 0, None)
        used = overlap > 0
        if np.all(finite[used]):
            out["mean"][i] = float(overlap[used] @ kept[used] / (b - a))
    below = np.searchsorted(neg, pos, side="left")
    ties = np.searchsorted(neg, pos, side="right") - below
    out["auroc"] = float((below.sum() + 0.5 * ties.sum()) / (n * m))
    return out


def _frame_scores(task: tuple[str, str]) -> tuple[str, str, list]:
    """:return: the configuration, the variant and per score ``(S of P⁺, S_Γ of N⁻)``: the
    highest box score over the planes that show each frame, ``-inf`` without any box."""
    name, variant = task
    material, sizes = _BOX_MATERIAL["material"][name], _BOX_MATERIAL["sizes"]
    out = []
    for score in SCORES:
        top = np.full(len(material), -np.inf)
        for k, (plane, size) in enumerate(zip(material, sizes, strict=True)):
            if plane is not None:
                found = box_scores([plane], variant, score, size)
                if len(found):
                    top[k] = found.max()
        out.append(
            tuple(
                np.array(
                    [top[ids].max() if len(ids) else -np.inf for ids in _BOX_MATERIAL[kind]],
                    dtype=np.float64,
                )
                for kind in ("positive", "negative")
            )
        )
    return name, variant, out


def frame_fpr_step(args) -> None:
    """``FPR_box@TPR(q)`` of every row of a 1 fps online evaluation, on the OVAD FP rows.

    Collects :func:`fpr_frames` over the sealed stage, computes ``S`` and ``S_Γ`` of every frame
    for each configuration, variant and score (:func:`_frame_scores`), then :func:`frame_fpr`.
    Writes in ``--evaluation`` ``frame_fpr_<dataset>.npz`` (each array configuration × variant
    × score × …: the raw ``positive`` and ``negative`` scores, ``fpr``, ``tpr`` and ``theta``
    per target ``q``, ``mean`` per interval, ``auroc``, ``ceiling``) and ``frame_fpr_<dataset>
    .json`` (axes, θ_O, counts, the frames given to short events).

    :raises ValueError: if the evaluation is not a 1 fps online one on the same OVAD plan, the
        stage is not sealed, or its configurations are not the evaluation's.
    """
    from localization_auto_evaluate import sealed_videos

    global _BOX_MATERIAL
    plan = read_json(args.ovad_plan)
    # A dry run (the dress rehearsal of a campaign) only collects the frames of every video,
    # with every check of the plan against the stage, before any evaluation exists.
    dry = bool(getattr(args, "dry_run", False))
    if not dry:
        summary = read_json(args.evaluation / "summary.json")
        if summary.get("mode") != "online" or summary.get("frames") != "1fps":
            raise ValueError("frame-fpr goes with an online evaluation at 1 fps")
        if summary.get("ovad_plan", {}).get("sha256") != file_sha256(args.ovad_plan):
            raise ValueError("The evaluation used another OVAD plan")
    variants = [] if dry else list(summary["variants"])
    intervals = read_json(args.intervals)["intervals"]
    theta = float(plan["theta_ovad"])
    for folder, dataset in DATASETS.items():
        seal = read_json(args.predictions / "operations" / f"seal-{folder}.json")
        if seal.get("status") != "complete" or seal["dataset"] != dataset:
            raise ValueError("Predictions must be complete and sealed for the whole dataset")
        names = list(seal["methods"])
        if not dry and read_json(args.evaluation / f"axes_{folder}.json")["configuration"] != names:
            raise ValueError(f"{folder}: the stage's configurations are not the evaluation's")
        videos = plan["datasets"][folder]["videos"]
        index: dict[tuple, int] = {}
        material, sizes = {n: [] for n in names}, []
        found = {"positive": [], "negative": []}
        counts: dict[str, int] = {}
        extra = {}
        for entry, complete, chunks in sealed_videos(args.predictions, seal):
            video = entry["video_id"]
            path = args.snapshot / dataset / f"{video}.json"
            frames = fpr_frames(
                read_json(path) if path.exists() else None,
                complete,
                chunks,
                videos[video],
                theta,
                intervals[folder][video],
            )
            for key, value in frames["counts"].items():
                counts[key] = counts.get(key, 0) + value
            extra.update(frames["extra"])
            if dry:
                continue
            size = (complete["video"]["width"], complete["video"]["height"])
            for kind in ("positive", "negative"):
                for pairs in frames[kind]:
                    ids = []
                    for chunk, temporal in pairs:
                        key = (video, chunk, temporal)
                        if key not in index:
                            index[key] = len(sizes)
                            sizes.append(size)
                            for n in names:
                                part = chunks[chunk]["methods"][n]["partition"]
                                material[n].append(
                                    None
                                    if part is None
                                    else plane_material(chunks[chunk], part, temporal)
                                )
                        ids.append(index[key])
                    found[kind].append(np.array(ids, dtype=np.int64))
        counts["videos"] = len(seal["videos"])
        if dry:
            short = sum(e["short"] for e in extra.values())
            print(f"{folder} (dry run): {counts}, extra frames {len(extra)} ({short} short)")
            continue
        _BOX_MATERIAL = {"material": material, "sizes": sizes, **found}
        tasks = [(n, v) for n in names for v in variants]
        pool = (
            multiprocessing.get_context("fork").Pool(args.workers, maxtasksperchild=1)
            if args.workers > 1
            else None
        )
        results = pool.imap_unordered(_frame_scores, tasks) if pool else map(_frame_scores, tasks)
        scores = {(name, variant): out for name, variant, out in results}
        if pool is not None:
            pool.close()
            pool.join()
        _BOX_MATERIAL = {}
        shape = (len(names), len(variants), len(SCORES))
        arrays = {
            "positive": np.zeros((*shape, counts["positive"])),
            "negative": np.zeros((*shape, counts["negative"])),
            **{k: np.zeros((*shape, len(FPR_POINTS))) for k in ("fpr", "tpr", "theta")},
            "mean": np.zeros((*shape, len(FPR_INTERVALS))),
            "auroc": np.zeros(shape),
            "ceiling": np.zeros(shape),
        }
        for (name, variant), per_score in scores.items():
            c, v = names.index(name), variants.index(variant)
            for s, (pos, neg) in enumerate(per_score):
                arrays["positive"][c, v, s], arrays["negative"][c, v, s] = pos, neg
                result = frame_fpr(pos, neg)
                for key in ("fpr", "tpr", "theta", "mean", "auroc", "ceiling"):
                    arrays[key][c, v, s] = result[key]
        np.savez_compressed(
            args.evaluation / f"frame_fpr_{folder}.npz",
            q=np.array(FPR_POINTS),
            intervals=np.array(FPR_INTERVALS),
            **arrays,
        )
        write_json(
            args.evaluation / f"frame_fpr_{folder}.json",
            {
                "definition": "docs/varie/metriche_localizzazione_1fps_fpr_ovad_20261030.md",
                "negative_population": "all_clean_coverage_v2",
                "configurations": names,
                "variants": variants,
                "scores": list(SCORES),
                "theta_ovad": theta,
                "ovad_plan": {"path": str(args.ovad_plan), "sha256": file_sha256(args.ovad_plan)},
                "counts": counts,
                "extra_frames": extra,
            },
        )
        print(f"{folder}: {counts}", flush=True)


def check_inside_events(truth: dict, intervals) -> None:
    """:raises ValueError: if a GT frame lies outside every temporal event, where it would also
    be a negative image."""
    inside = set()
    for start, end, *_ in intervals:
        inside.update(range(int(start), int(end) + 1))
    for event in truth["events"]:
        for plane in event["planes"]:
            for segment in plane["segments"]:
                if any(f not in inside for f in range(segment["start"], segment["end"] + 1)):
                    raise ValueError(f"{event['event_id']}: a GT frame is outside every event")


def evaluate(args) -> None:
    from localization_auto_evaluate import sealed_videos

    snapshot = read_json(args.snapshot / "snapshot.json")
    intervals = read_json(args.intervals)["intervals"] if getattr(args, "intervals", None) else None
    if intervals is not None and args.iou_grid != "fine":
        raise ValueError("--intervals needs --iou-grid fine: the pooled metrics live there")
    # Options of 28 and 30 October 2026, with the defaults of the evaluations before them.
    for name, default in (
        ("mode", "offline"),
        ("variants", None),
        ("gt_modes", None),
        ("frames", "all"),
        ("ovad_plan", None),
        ("ignore_head_gate", False),
        ("exclude_head_fn", False),
    ):
        setattr(args, name, getattr(args, name, default))
    args.ap_only = getattr(args, "ap_only", False)
    one_fps = args.frames == "1fps"
    if args.ignore_head_gate and not (one_fps and args.mode == "online" and args.ap_only):
        raise ValueError("The head-gate counterfactual requires online, 1fps and --ap-only")
    if args.exclude_head_fn and (
        args.ignore_head_gate or not (one_fps and args.mode == "online" and args.ap_only)
    ):
        raise ValueError("FN exclusion requires observed online, 1fps and --ap-only")
    if one_fps and intervals is None:
        raise ValueError("--frames 1fps needs --intervals: the frame of a short event needs them")
    if args.ovad_plan is not None and args.mode != "online":
        raise ValueError("--ovad-plan decides which chunks show boxes online: --mode online")
    if one_fps and args.mode == "online" and args.ovad_plan is None:
        raise ValueError("The online AP at 1 fps shows boxes on h(c) >= θ_O: --ovad-plan (D1)")
    plan = read_json(args.ovad_plan) if args.ovad_plan is not None else None
    stage = read_json(args.predictions / "operations" / "ap-stage.json")
    available = VARIANTS if set(MATERIAL) <= set(stage.get("material", ())) else LEGACY_VARIANTS
    variants = list(args.variants or available)
    unknown = sorted(set(variants) - set(available))
    if unknown:
        raise ValueError(f"The stage cannot give the variants {unknown}: it lacks their material")
    gt_modes = list(args.gt_modes or GT_MODES)
    if args.iou_grid != "fine" and (args.variants or args.gt_modes or args.ap_only):
        raise ValueError("--variants, --gt-modes and --ap-only need --iou-grid fine")
    # Online, the intervals do not add negative images: they tell the false positive chunks
    # (flagged, outside every event) apart, whose box confidence the FP metric reads.
    summary = {
        "stage": str(args.predictions),
        "mode": args.mode,
        "variants": variants,
        "gt_modes": gt_modes,
        "ap_only": args.ap_only,
        "head_gate": "disabled_counterfactual" if args.ignore_head_gate else "observed",
        "head_fn": "excluded" if args.exclude_head_fn else "included",
        "datasets": {},
    }
    if intervals is not None:
        summary["intervals"] = {"path": str(args.intervals), "sha256": file_sha256(args.intervals)}
    summary["frames"] = args.frames
    if plan is not None:
        summary["ovad_plan"] = {
            "path": str(args.ovad_plan),
            "sha256": file_sha256(args.ovad_plan),
            "detector": plan["detector"],
            "theta_ovad": plan["theta_ovad"],
        }
    if args.ignore_head_gate or args.exclude_head_fn:
        identity_path = args.output / "evaluation_identity.json"
        identity = {
            k: summary[k]
            for k in (
                "stage",
                "mode",
                "frames",
                "variants",
                "gt_modes",
                "head_gate",
                "head_fn",
                "ovad_plan",
                "intervals",
            )
        }
        identity["gt_snapshot_sha256"] = file_sha256(args.snapshot / "snapshot.json")
        identity["source_sha256"] = file_sha256(Path(__file__))
        if identity_path.exists():
            if read_json(identity_path) != identity:
                raise ValueError("Counterfactual output belongs to a different evaluation")
        elif (args.output / "results").exists():
            raise ValueError("Cannot reuse observed results for a counterfactual")
        else:
            write_json(identity_path, identity)
    for folder, dataset in DATASETS.items():
        seal_path = args.predictions / "operations" / f"seal-{folder}.json"
        seal = read_json(seal_path)
        if seal.get("status") != "complete" or seal["dataset"] != dataset:
            raise ValueError("Predictions must be complete and sealed for the whole dataset")
        names = list(seal["methods"])
        classes = sorted(snapshot["class_event_support"][dataset])
        expected = {row["video_id"] for row in seal["videos"]}
        if {p.stem for p in (args.snapshot / dataset).glob("*.json")} - expected:
            raise ValueError("A ground-truth video has no prediction")

        def truth_of(entry: dict, dataset: str = dataset) -> Path:
            return args.snapshot / dataset / f"{entry['video_id']}.json"

        events, sizes = [], {}
        negatives = {n: [] for n in names}
        empty = dict.fromkeys(names, 0)
        counted = {"images": 0, "frames": 0}
        flagged_chunks: list[dict] = []
        flagged_material = {n: [] for n in names}
        extra_frames: dict = {}
        emitting = 0
        # At 1 fps the per-plane material of negative images and flagged chunks (the FP metrics
        # until 30 October 2026) is not collected: the FP metric is ``frame-fpr``'s.
        legacy = intervals is not None and not one_fps
        wanted = (lambda e: True) if legacy else (lambda e: truth_of(e).exists())
        for entry, complete, chunks in sealed_videos(args.predictions, seal, wanted):
            if chunks is None:
                continue
            if intervals is not None:
                video = entry["video_id"]
                if video not in intervals[folder]:
                    raise ValueError(f"{folder} {video}: no temporal intervals")
            if legacy and args.mode == "offline":
                material, nothing, images, frames = negative_images(
                    complete, chunks, intervals[folder][video], names
                )
                for n in names:
                    negatives[n].extend(material[n])
                    empty[n] += nothing[n]
                counted["images"] += images
                counted["frames"] += frames
            truth = read_json(truth_of(entry)) if truth_of(entry).exists() else None
            if legacy and args.mode == "online":
                info, material = flagged_chunk_material(
                    complete, chunks, truth, intervals[folder][video], names
                )
                flagged_chunks.extend({"video_id": video, **i} for i in info)
                for n in names:
                    flagged_material[n].extend(material[n])
            if truth is None:
                continue
            if intervals is not None:
                check_inside_events(truth, intervals[folder][entry["video_id"]])
            frames = flags = None
            if one_fps:
                frames, extra = evaluated_frames(
                    truth, float(complete["video"]["fps"]), intervals[folder][entry["video_id"]]
                )
                extra_frames.update(extra)
            if plan is not None:
                flags = emitting_chunks(
                    complete,
                    chunks,
                    plan["datasets"][folder]["videos"][entry["video_id"]],
                    float(plan["theta_ovad"]),
                )
                emitting += sum(flags)
            for event in event_images(
                truth,
                complete,
                chunks,
                names,
                args.mode,
                frames,
                flags,
                args.ignore_head_gate,
                args.exclude_head_fn,
            ):
                sizes[event["event_id"]] = (truth["width"], truth["height"])
                events.append(event)
        if len(events) != snapshot["counts"][f"{dataset}_events"]:
            raise ValueError(f"{folder}: {len(events)} events, the snapshot has a different count")
        # A result on disk is reused (stage and snapshot are frozen): the step resumes.
        todo = [n for n in names if not (args.output / "results" / folder / f"{n}.json").exists()]

        def payload(
            n: str,
            classes=classes,
            sizes=sizes,
            events=events,
            negatives=negatives,
            empty=empty,
            flagged_material=flagged_material,
        ) -> dict:
            return {
                "name": n,
                "flagged": flagged_material[n],
                "grid": args.iou_grid,
                "variants": variants,
                "gt_modes": gt_modes,
                "ap_only": args.ap_only,
                "classes": classes,
                "sizes": sizes,
                "events": [
                    {
                        **{k: e[k] for k in ("event_id", "class")},
                        "images": [
                            {"boxes": i["boxes"], "material": i["material"][n]} for i in e["images"]
                        ],
                    }
                    for e in events
                ],
                "negatives": negatives[n],
                "empty_negatives": empty[n],
            }

        pool = multiprocessing.Pool(args.workers) if args.workers > 1 else None
        # One batch of payloads at a time: with the negative images a payload is large.
        batch = max(1, args.workers)
        for start in range(0, len(todo), batch):
            payloads = [payload(n) for n in todo[start : start + batch]]
            results = (
                pool.imap_unordered(evaluate_configuration, payloads)
                if pool is not None
                else map(evaluate_configuration, payloads)
            )
            for result in results:
                result.update(
                    dataset=dataset,
                    head_gate=summary["head_gate"],
                    head_fn=summary["head_fn"],
                    prediction_seal_sha256=file_sha256(seal_path),
                    gt_snapshot_sha256=file_sha256(args.snapshot / "snapshot.json"),
                )
                target = args.output / "results" / folder / f"{result['configuration']}.json"
                arrays = result.pop("arrays", None)
                if arrays is not None:
                    # The arrays first: a JSON on disk means the configuration is complete.
                    target.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(target.with_suffix(".npz"), **arrays)
                write_json(target, result)
            del payloads
        if pool is not None:
            pool.close()
            pool.join()
        counts = [len(e["images"]) for e in events]
        summary["datasets"][folder] = {
            "events": len(events),
            "events_without_images": sum(c == 0 for c in counts),
            "images": sum(counts),
            "images_per_event": {
                "min": min(counts),
                "median": statistics.median(counts),
                "max": max(counts),
                "events_with_one_image": sum(c == 1 for c in counts),
                "events_with_at_most_three": sum(c <= 3 for c in counts),
            },
            "ground_truth": gt_statistics(events),
            "configurations": names,
        }
        if one_fps:
            summary["datasets"][folder]["extra_frames"] = {
                "events": len(extra_frames),
                "short_events": sum(e["short"] for e in extra_frames.values()),
                "frames": extra_frames,
            }
        if plan is not None:
            summary["datasets"][folder]["emitting_chunks_of_videos_with_gt"] = emitting
        if flagged_chunks:
            summary["datasets"][folder]["flagged_chunks"] = flagged_chunks
            summary["datasets"][folder]["flagged_counts"] = {
                c: sum(f["category"] == c for f in flagged_chunks) for c in ("T", "F", "Z")
            }
        if legacy and args.mode == "offline":
            summary["datasets"][folder]["negative"] = {
                "images": counted["images"],
                "frames": counted["frames"],
                "images_with_material": {n: len(negatives[n]) for n in names},
            }
    if args.iou_grid == "fine":
        write_threshold_summary(args.output, summary)
    else:
        write_summary(args.output, summary)
    if args.output.parent.name == "evaluation" and not args.ap_only:
        # The campaign's single page of metrics follows every evaluation.
        from localization_class_report import build

        print(build(args.output.parent.parent, args.output.parent / "report_localizzazione.html"))


def write_threshold_summary(output: Path, summary: dict) -> None:
    """Write the per-threshold results in array form, for intervals and means chosen later.

    Per dataset ``metrics_per_event_<dataset>.npz`` holds ``ap``, ``rbdc`` and ``rbdr``, each
    shaped configuration × variant × score × GT reading × event × threshold, and
    ``pooled_rbdc``/``pooled_rbdr`` without the event axis, with the axes in
    ``axes_<dataset>.json``; ``summary_by_threshold.csv`` has the all-class mean of each metric
    at every threshold, and the pooled values.
    """
    header = ",".join(f"@{t:.2f}" for t in IOU_THRESHOLDS_FINE)
    lines = [f"dataset,metric,gt,configuration,variant,score,{header}"]
    for folder, info in summary["datasets"].items():
        names = info["configurations"]
        loaded = [read_json(output / "results" / folder / f"{n}.json") for n in names]
        events = loaded[0]["events"]
        if any(r["events"] != events for r in loaded):
            raise ValueError(f"{folder}: configurations disagree on the events")
        stored = [np.load(output / "results" / folder / f"{n}.npz") for n in names]
        arrays = {key: np.stack([s[key] for s in stored]) for key in stored[0].files}
        np.savez_compressed(output / f"metrics_per_event_{folder}.npz", **arrays)
        variants, gt_modes = summary.get("variants", VARIANTS), summary.get("gt_modes", GT_MODES)
        write_json(
            output / f"axes_{folder}.json",
            {
                "shape": list(arrays["ap"].shape),
                "mode": summary.get("mode", "offline"),
                "configuration": names,
                "variant": list(variants),
                "score": list(SCORES),
                "gt": list(gt_modes),
                "event": events,
                "iou_threshold": IOU_THRESHOLDS_FINE.tolist(),
                "arrays": {key: list(a.shape) for key, a in arrays.items()},
            },
        )
        metrics = ("ap",) if summary.get("ap_only") else FINE_METRICS
        for name, result in zip(names, loaded, strict=True):
            for v in variants:
                for s in SCORES:
                    for m in gt_modes:
                        cell = result["variants"][v][s][m]
                        rows = [(k, cell[k]["macro"]) for k in metrics]
                        rows += [(f"pooled_{k}", cell["pooled"][k]) for k in cell["pooled"]]
                        lines.extend(
                            f"{folder},{k},{m},{name},{v},{s},"
                            + ",".join("" if x is None else f"{x:.6f}" for x in (xs or []))
                            for k, xs in rows
                        )
    write_json(output / "summary.json", summary)
    (output / "summary_by_threshold.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v["images_per_event"] for k, v in summary["datasets"].items()}))


def write_summary(output: Path, summary: dict) -> None:
    """Write ``summary.json``, every row in ``summary.csv`` and the AP tables in ``summary.md``."""
    lines = ["dataset,gt,configuration,variant,score," + ",".join(METRICS)]
    tables = []
    for folder, info in summary["datasets"].items():
        for mode in GT_MODES:
            header = f"\n### {folder}, GT {mode}, score mass: AP (AP50)\n"
            table = [header, "| Configuration | " + " | ".join(VARIANTS) + " |"]
            table.append("|---|" + "---:|" * len(VARIANTS))
            for name in info["configurations"]:
                result = read_json(output / "results" / folder / f"{name}.json")["variants"]
                cells = []
                for variant in VARIANTS:
                    for score in SCORES:
                        values = [result[variant][score][mode][m]["value"] for m in METRICS]
                        lines.append(
                            f"{folder},{mode},{name},{variant},{score},"
                            + ",".join(f"{v:.6f}" for v in values)
                        )
                    ap = result[variant]["mass"][mode]
                    cells.append(f"{ap['AP']['value']:.4f} ({ap['AP50']['value']:.4f})")
                table.append(f"| {name} | " + " | ".join(cells) + " |")
            tables.extend(table)
    write_json(output / "summary.json", summary)
    (output / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    note = (
        "# Per-event AP of the class-routed localization\n\n"
        "Mean of the per-event AP over the events of each class, then over the 13 classes. "
        "⚠️ The two GT readings (`union`, `instances`) are not comparable with each other. "
        "Every row (the `share` score too) in `summary.csv`.\n"
    )
    (output / "summary.md").write_text(note + "\n".join(tables) + "\n", encoding="utf-8")
    print(json.dumps({k: v["images_per_event"] for k, v in summary["datasets"].items()}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exclude-head-fn",
        action="store_true",
        help="evaluate: conditional online AP excluding FN images; no new predictions",
    )
    parser.add_argument(
        "--ignore-head-gate",
        action="store_true",
        help="evaluate: online counterfactual on identical 1fps images, restore saved boxes on FN chunks",
    )
    parser.add_argument(
        "step",
        choices=(
            "detections",
            "evaluate",
            "box-scores",
            "box-retention",
            "false-boxes",
            "frame-fpr",
        ),
    )
    parser.add_argument(
        "--frames",
        choices=FRAME_MODES,
        default="all",
        help="evaluate: every GT frame, or one per second of the video (30 October 2026)",
    )
    parser.add_argument(
        "--ovad-plan",
        type=Path,
        help="evaluate --mode online and frame-fpr: a detector's plan of "
        "localization_class_ovad_plan.py; the boxes are shown on h(c) >= θ_O",
    )
    parser.add_argument("--pass-root", type=Path, help="detections")
    parser.add_argument("--dictionaries", type=Path, help="detections")
    parser.add_argument(
        "--configurations", choices=("campaign", "b3", "b4", "global"), default="campaign"
    )
    parser.add_argument("--heads-set", help="set folder of the class-head pass (B1/B2 logits)")
    parser.add_argument("--without-heads", action="store_true", help="leave B1/B2 out")
    parser.add_argument(
        "--classifiers-set",
        nargs="*",
        help="set folders of the classifier pass (B4); several are merged (test and real)",
    )
    parser.add_argument(
        "--keep-annotated",
        action="store_true",
        help="detections with --real-use: keep every record of test too (offline and online)",
    )
    parser.add_argument(
        "--mode",
        choices=("offline", "online"),
        default="offline",
        help="evaluate: images of the chunk each GT frame is assigned to, or of every flagged "
        "chunk holding it",
    )
    parser.add_argument("--variants", nargs="*", help="evaluate --iou-grid fine: a subset")
    parser.add_argument(
        "--gt-modes", nargs="*", choices=GT_MODES, help="evaluate --iou-grid fine: a subset"
    )
    parser.add_argument(
        "--ap-only", action="store_true", help="evaluate --iou-grid fine: no RBDC/RBDR"
    )
    parser.add_argument(
        "--real-use",
        type=Path,
        help="detections: a plan of localization_class_real_use.py; only the chunks the "
        "detector flags get detections",
    )
    parser.add_argument(
        "--extra-sets", nargs="*", default=[], help="detections: pass sets read besides test"
    )
    parser.add_argument(
        "--intervals",
        type=Path,
        help="evaluate: intervals.json of the real-use plan; the pooled metrics add the "
        "negative images",
    )
    parser.add_argument("--manifest-ucf", type=Path)
    parser.add_argument("--manifest-mivia", type=Path)
    parser.add_argument("--predictions", type=Path, help="evaluate: a sealed detections stage")
    parser.add_argument("--snapshot", type=Path, help="evaluate: the frozen GT snapshot")
    parser.add_argument("--workers", type=int, default=1, help="evaluate: processes")
    parser.add_argument(
        "--iou-grid",
        choices=("coco", "fine"),
        default="coco",
        help="evaluate: COCO's AP/AP50/AP75, or the per-event AP at IoU 0.05:0.95 by 0.05",
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        help="box-scores: an online evaluation with --intervals; the histograms go there",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="frame-fpr: collect the frames and check the OVAD plan, compute nothing",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.step == "frame-fpr":
        if None in (args.predictions, args.snapshot, args.intervals, args.ovad_plan) or (
            args.evaluation is None and not args.dry_run
        ):
            raise ValueError(
                "frame-fpr needs --predictions, --snapshot, --intervals, --evaluation, --ovad-plan"
            )
        frame_fpr_step(args)
        return
    if args.step in ("box-scores", "box-retention", "false-boxes"):
        if None in (args.predictions, args.snapshot, args.intervals, args.evaluation):
            raise ValueError(
                f"{args.step} needs --predictions, --snapshot, --intervals, --evaluation"
            )
        steps = {
            "box-scores": box_score_histograms,
            "box-retention": box_retention_step,
            "false-boxes": false_boxes_step,
        }
        steps[args.step](args)
        return
    if args.output is None:
        raise ValueError(f"{args.step} needs --output")
    if args.step == "evaluate":
        if args.predictions is None or args.snapshot is None:
            raise ValueError("evaluate needs --predictions and --snapshot")
        evaluate(args)
        return
    if None in (args.pass_root, args.dictionaries, args.manifest_ucf, args.manifest_mivia):
        raise ValueError("detections needs --pass-root, --dictionaries and both test manifests")
    args.metas = {
        "ucf_test": {r["video_id"]: r for r in read_json(args.manifest_ucf)},
        "mivia_test": {r["video_id"]: r for r in read_json(args.manifest_mivia)},
    }
    detect(args, read_json(CONFIG))


if __name__ == "__main__":
    main()
