"""Calibrate the mass fractions and build the boxes of the class-routing campaign (CPU).

Plan: ``docs/varie/piano_localizzazione_per_classe_20261021.md`` §5.2, §7.3, §7.4, §7.5.
Reads the records of ``localization_class_pass.py`` and the dictionaries of
``localization_class_fit.py``; decodes nothing and uses no GPU.

- ``calibrate``: on the validation positives of one detector, the first-concept share of
  every chunk under the global dictionary (one fraction) and under the A1 and A2 dictionary
  of its annotated class (one fraction per class, no fallback): the highest fraction keeping
  one concept on 90% of the chunks, rounded down to the percentage point.
- ``localize``: for every annotated test chunk and every configuration of the config file
  (``campaign`` or ``b3``), the class choice (four rules of proposal A, the B1/B2 heads, the
  B3 class head, the annotated temporal class for the oracle), the concept prefix of the
  chosen dictionary and the boxes (union for P07, one set of boxes per concept otherwise),
  at the 55% threshold (suffix ``phase1``) and at the calibrated one (``phase2``). The
  predictions are written in the sealed format that ``localization_auto_evaluate.py`` reads,
  over every protocol window of every test video (empty boxes outside the annotated chunks),
  with the class-selection diagnostics in ``selectors.json``. The B1/B2 class logits come
  either from the IG records themselves or, with ``--heads-set``, from a separate pass without
  IG over the same chunks (passes 1 and 3 run apart so that A, P07 and the oracle do not wait
  for the B1 and B2 trainings); ``--without-heads`` leaves the B1/B2 configurations out and
  ``--heads-only`` keeps only them. ``--fixed-fraction F`` replaces both rules with one fixed
  fraction for every dictionary (experiments ``<configuration>-massNN``, e.g. ``-mass70``):
  the stage then has no calibrated rule, and ``P07-global-bridge70`` (P07-global at 70%
  whatever the rule, already in the campaign stage) is left out. The ``b4`` configurations
  (26 October 2026) choose the class with the B4 classifiers, class heads alone on their own
  trained encoder, whose logits come from a ``--classifiers`` pass over the same chunks
  (``--classifiers-set``); the second step is the campaign detector's (its IG, regions,
  dictionaries and calibrated fractions), as for B1 and B2. With ``--real-use`` (28 October
  2026) the chunks are those the detector's binary head flags in a plan of
  ``localization_class_real_use.py``, read from ``test`` and ``--extra-sets``, on the videos of
  the plan's scope manifests; an annotated chunk the detector does not flag gets no box.
- ``evaluate``: every method of one sealed stage on the frozen spatial GT snapshot, with
  ``localization_auto_evaluate.evaluate`` (which checks the seal before reading the GT), and
  a summary per dataset with the best first-step rule of A on FS-UCF-Crime (plan 5.3 point
  10: the rule is chosen on the test boxes). With ``--intervals`` it also counts the boxes on
  the negative images, the chunk planes of the frames outside every event
  (``false_alarms.json``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from evd.exp.localization_auto import read_json, seal_stage, video_key, write_json  # noqa: E402
from evd.exp.localization_class import (  # noqa: E402
    CLASS_RULES,
    balanced_accuracy,
    calibrate_fraction,
    choose_class,
    class_scores,
    concept_boxes,
    confusion,
    first_concept_share,
    per_class_fractions,
    positive_masses,
    prefix_concepts,
    stack_dictionaries,
)
from evd.exp.localization_vtcd import CONCEPTS  # noqa: E402
from evd.explain.localization import native_boxes  # noqa: E402
from evd.explain.vtcd import assign_concepts  # noqa: E402
from evd.utils.hashing import file_sha256  # noqa: E402
from localization_class_plan import CLASSES  # noqa: E402

CONFIG = ROOT / "configs/localization_class_routing_20261021.json"
#: Index of ``Normal`` in the class heads' outputs (``eval.normal_idx`` of the runs).
NORMAL = len(CLASSES)
#: Selectors that read the class heads trained on the campaign detector.
HEADS = ("B1", "B2")
#: Selectors that read a class head on its own trained encoder (B4, 26 October 2026).
CLASSIFIERS = ("B4L", "B4V")


def iter_records(folder: Path, keep=lambda record: True):
    """Yield ``(record, arrays)`` of the chunks saved by the pass in ``folder``.

    :param keep: ``record -> bool``; the arrays of a chunk not kept are not read.
    """
    for path in sorted(folder.glob("*.json")):
        record = read_json(path)
        if not keep(record):
            continue
        arrays = np.load(path.with_suffix(".npz"))
        yield record, {k: arrays[k] for k in arrays.files}


def records_of(folder: Path) -> list[tuple[dict, dict]]:
    """:return: ``(record, arrays)`` of every chunk saved by the pass in ``folder``."""
    return list(iter_records(folder))


def stage_records(args):
    """Yield ``(record, arrays)`` of the chunks a stage localizes.

    Without ``--real-use``, the annotated test chunks of the set ``test``. With it (real use,
    28 October 2026), the chunks the detector flags in the plan of
    ``localization_class_real_use.py``, from ``test`` and the ``--extra-sets`` (the pass over
    the flagged chunks without a record): the annotated chunks it does not flag get no box,
    unless ``--keep-annotated`` (the AP stages of 28 October, evening, that serve both the
    offline AP on every annotated chunk and the online AP on the flagged ones) keeps every
    record of ``test`` too.

    :raises ValueError: if a chunk is in two sets, a record comes from another detector than
        the plan's, or a flagged chunk has no record.
    """
    plan = read_json(args.real_use) if getattr(args, "real_use", None) else None
    wanted = None
    if plan is not None:
        wanted = {(d, w) for d, ids in plan["positives"].items() for w in ids}
    seen: set[tuple[str, str]] = set()

    annotated = getattr(args, "keep_annotated", False)

    def keeper(name: str):
        def keep(record: dict) -> bool:
            key = record["dataset"], record["window_id"]
            if key in seen:
                raise ValueError(f"{key} is in two pass sets")
            seen.add(key)
            if wanted is None:
                return True
            if record["identity"]["detector"] != plan["run_id"]:
                raise ValueError(
                    f"{key}: pass of {record['identity']['detector']}, not {plan['run_id']}"
                )
            return key in wanted or (annotated and name == "test")

        return keep

    for name in ("test", *getattr(args, "extra_sets", ())):
        yield from iter_records(args.pass_root / name, keeper(name))
    if wanted is not None and wanted - seen:
        missing = sorted(wanted - seen)
        raise ValueError(f"{len(missing)} flagged chunks have no pass record, e.g. {missing[:3]}")


def write_real_use(args, output: Path, localized: int) -> None:
    """Write the stage's real-use receipt: the plan, the detector's threshold and the counts.

    :raises ValueError: if the stage did not localize every flagged chunk.
    """
    plan = read_json(args.real_use)
    flagged = sum(len(ids) for ids in plan["positives"].values())
    if localized != flagged:
        raise ValueError(f"{localized} chunks localized, the plan flags {flagged}")
    write_json(
        output / "operations" / "real-use.json",
        {
            "plan": str(args.real_use),
            "plan_sha256": plan["sha256"],
            "detector": plan["run_id"],
            "threshold": plan["threshold"],
            "scope": {d: len(s["videos"]) for d, s in plan["scope"].items()},
            "counts": plan["counts"],
            "sets": ["test", *getattr(args, "extra_sets", ())],
            "localized_chunks": localized,
        },
    )


def head_records(folder: Path) -> dict[tuple[str, str], dict]:
    """:return: the records of a class-head pass keyed by ``(dataset, window_id)``."""
    out = {}
    for path in sorted(folder.glob("*.json")):
        record = read_json(path)
        out[record["dataset"], record["window_id"]] = record
    return out


def with_heads(record: dict, heads: dict[tuple[str, str], dict]) -> dict:
    """Return the IG record with the B1/B2 class logits and head times of its head record.

    A record whose pass computed B1 and B2 itself (the real-use pass) is returned as it is.

    :raises ValueError: if the chunk has no head record, the head record lacks B1 or B2, or
        the two passes ran on different detectors.
    """
    if set(HEADS) <= set(record["class_logits"]):
        return record
    found = heads.get((record["dataset"], record["window_id"]))
    if found is None:
        raise ValueError(f"No class-head record for {record['dataset']}:{record['window_id']}")
    missing = set(HEADS) - set(found["class_logits"])
    if missing:
        raise ValueError(f"Class-head record of {record['window_id']} lacks {sorted(missing)}")
    for field in ("detector", "checkpoint_sha256"):
        if found.get("identity", {}).get(field) != record.get("identity", {}).get(field):
            raise ValueError(f"Class-head pass and IG pass differ on {field}")
    seconds = record["seconds"] | {
        k: v for k, v in found["seconds"].items() if k.startswith("head_")
    }
    return record | {
        "class_logits": record["class_logits"] | found["class_logits"],
        "seconds": seconds,
    }


def with_classifiers(record: dict, classifiers: dict[tuple[str, str], dict]) -> dict:
    """Return the IG record with the B4 class logits and seconds of its classifier record.

    The B4 seconds (``encode_<name>``, ``head_<name>``) are kept apart from the detector ones:
    a B4 configuration pays its own encoder on top of the detector chain.

    :raises ValueError: if the chunk has no classifier record, the record lacks a classifier,
        or the two passes ran on different chunk plans.
    """
    found = classifiers.get((record["dataset"], record["window_id"]))
    if found is None:
        raise ValueError(f"No classifier record for {record['dataset']}:{record['window_id']}")
    missing = set(CLASSIFIERS) - set(found["class_logits"])
    if missing:
        raise ValueError(f"Classifier record of {record['window_id']} lacks {sorted(missing)}")
    if found["identity"]["plan_sha256"] != record["identity"]["plan_sha256"]:
        raise ValueError("Classifier pass and IG pass ran on different chunk plans")
    own = {k: v for k, v in found["seconds"].items() if k.split("_", 1)[-1] in CLASSIFIERS}
    return record | {
        "class_logits": record["class_logits"] | found["class_logits"],
        "classifier_seconds": own,
    }


def shares(arrays: dict, dictionary: np.ndarray) -> float | None:
    """:return: the first-concept share of a chunk under ``dictionary``."""
    ids, _ = assign_concepts(arrays["descriptors"].astype(np.float64), dictionary)
    return first_concept_share(
        positive_masses(arrays["ig_ensemble"], arrays["labels"], ids, len(dictionary))
    )


def calibrate(args, config: dict) -> None:
    model = np.load(args.dictionaries)
    target = config["fraction"]["phase2_single_concept_target"]
    globals_, per_set = [], {"a1": defaultdict(list), "a2": defaultdict(list)}
    empty = defaultdict(int)
    for record, arrays in records_of(args.pass_root / "val"):
        name = record["class_name"]
        value = shares(arrays, model["global_dictionary"])
        if value is None:
            empty["global"] += 1
        else:
            globals_.append(value)
        for key in ("a1", "a2"):
            value = shares(arrays, model[key][CLASSES.index(name)])
            if value is None:
                empty[key] += 1
            else:
                per_set[key][name].append(value)
    global_fraction = calibrate_fraction(globals_, target)
    result = {
        "target": target,
        "global": {
            "fraction": global_fraction,
            "chunks": len(globals_),
            "single_concept_share": float(np.mean(np.asarray(globals_) >= global_fraction)),
        },
        **{key: per_class_fractions(dict(values), target) for key, values in per_set.items()},
        "chunks_without_positive_mass": dict(empty),
        "dictionaries": str(args.dictionaries),
    }
    missing = {key: sorted(set(CLASSES) - set(result[key])) for key in ("a1", "a2")}
    if any(missing.values()):
        raise ValueError(f"Classes without validation positives: {missing}")
    write_json(args.output / "thresholds.json", result)
    print(json.dumps({k: result[k] for k in ("global", "chunks_without_positive_mass")}))


def _selector_class(name: str, record: dict, stacked_scores: dict) -> int | None:
    """:return: the class index chosen for the chunk by selector ``name``, ``None`` if none."""
    if name.startswith(("A1-", "A2-")):
        return stacked_scores[name]
    if name == "Oracle":
        return None if record["class_name"] is None else CLASSES.index(record["class_name"])
    key = "own" if name == "B3" else name
    if key not in record["class_logits"]:
        raise ValueError(
            f"{record['window_id']} has no {key} class logits: pass --heads-set with the "
            "class-head pass, or --without-heads"
        )
    logits = np.asarray(record["class_logits"][key], dtype=float)
    logits[NORMAL] = -np.inf
    return choose_class(logits)


def selector_seconds(experiment: str, record: dict) -> float:
    """:return: the GPU seconds a configuration adds to the detector chain to choose its class:
    the class head of B1, B2 and B3, the encoder and class head of a B4 classifier, nothing
    for P07, A and the oracle."""
    selector = experiment.split("-", 1)[0]
    if selector in CLASSIFIERS:
        own = record["classifier_seconds"]
        return own[f"encode_{selector}"] + own[f"head_{selector}"]
    head = {"B1": "head_B1", "B2": "head_B2", "B3": "head_own"}.get(selector)
    return record["seconds"].get(head, 0.0) if head else 0.0


def localize(args, config: dict) -> None:
    from evd.protocol.chunks import sampler_for_protocol
    from evd.types import VideoMeta

    model = np.load(args.dictionaries)
    stacked = {key: stack_dictionaries(model[key]) for key in ("a1", "a2")}
    thresholds = read_json(args.thresholds) if args.thresholds else None
    names = config["configurations"][args.configurations]
    if args.without_heads and (args.heads_set or args.heads_only):
        raise ValueError("--without-heads excludes --heads-set and --heads-only")
    prefixes = tuple(f"{h}-" for h in HEADS)
    if args.without_heads:
        names = [n for n in names if not n.startswith(prefixes)]
    if args.heads_only:
        names = [n for n in names if n.startswith(prefixes)]
        if not names:
            raise ValueError(f"No B1/B2 configuration among {args.configurations}")
    classifiers_set = getattr(args, "classifiers_set", None)
    if args.configurations == "b4" and not classifiers_set:
        raise ValueError("the b4 configurations need --classifiers-set")
    classifiers = head_records(args.pass_root / classifiers_set) if classifiers_set else None
    heads = head_records(args.pass_root / args.heads_set) if args.heads_set else None
    fixed, bridge = config["fraction"]["phase1"], config["fraction"]["bridge"]
    phases = ["phase1"] + (["phase2"] if thresholds else [])
    override = getattr(args, "fixed_fraction", None)
    if override is not None:
        if thresholds is not None:
            raise ValueError("--fixed-fraction has no calibrated rule: drop --thresholds")
        if not 0.0 < override <= 1.0:
            raise ValueError(f"--fixed-fraction must be in (0, 1], got {override}")
        fixed, phases = override, [f"mass{round(override * 100)}"]
        # The bridge is P07-global at 70% whatever the rule: campaign-core has it already.
        names = [n for n in names if n != "P07-global-bridge70"]
    experiments = [
        f"{n}-{p}"
        for n in names
        for p in phases
        if not (p == "phase2" and n in config["configurations"]["phase2_excludes"])
    ]
    by_window, selections = {}, defaultdict(list)
    for record, arrays in stage_records(args):
        if heads is not None:
            record = with_heads(record, heads)
        if classifiers is not None:
            record = with_classifiers(record, classifiers)
        start = time.process_time()
        descriptors = arrays["descriptors"].astype(np.float64)
        labels, counts, ig = arrays["labels"], arrays["counts"], arrays["ig_ensemble"]
        global_ids, _ = assign_concepts(descriptors, model["global_dictionary"])
        global_masses = positive_masses(ig, labels, global_ids, CONCEPTS)
        stacked_scores, class_ids = {}, {}
        for key, (dictionary, owners) in stacked.items():
            _, coefficients = assign_concepts(descriptors, dictionary)
            for rule in CLASS_RULES:
                stacked_scores[f"{key.upper()}-{rule}"] = choose_class(
                    class_scores(coefficients, counts, owners, len(CLASSES), rule)
                )
        shared_cpu = time.process_time() - start

        def ids_for(
            key: str, index: int, class_ids=class_ids, descriptors=descriptors
        ) -> np.ndarray:
            if (key, index) not in class_ids:
                class_ids[key, index] = assign_concepts(descriptors, model[key][index])[0]
            return class_ids[key, index]

        meta = args.metas[record["dataset"]][record["video_id"]]
        methods, recorded = {}, set()
        for name in names:
            for phase in phases:
                experiment = f"{name}-{phase}"
                if experiment not in experiments:
                    continue
                begin = time.process_time()
                if name.startswith("P07-global"):
                    if name == "P07-global-bridge70":
                        fraction = bridge
                    else:
                        fraction = thresholds["global"]["fraction"] if phase == "phase2" else fixed
                    kept = prefix_concepts(global_masses, fraction)
                    if name == "P07-global-per-concept":
                        boxes = concept_boxes(
                            labels, global_ids, kept, meta["width"], meta["height"]
                        )
                    else:
                        boxes = native_boxes(
                            np.isin(global_ids[labels], kept), meta["width"], meta["height"]
                        )
                    chosen = None
                else:
                    selector, key = (
                        (name[:-3], name[-2:].lower())
                        if name[-3:] in ("-A1", "-A2")
                        else (name, name[:2].lower())
                    )
                    chosen = _selector_class(selector, record, stacked_scores)
                    if chosen is None:
                        boxes = [[] for _ in range(labels.shape[0])]
                    else:
                        ids = ids_for(key, chosen)
                        fraction = (
                            thresholds[key][CLASSES[chosen]]["fraction"]
                            if phase == "phase2"
                            else fixed
                        )
                        kept = prefix_concepts(positive_masses(ig, labels, ids, CONCEPTS), fraction)
                        boxes = concept_boxes(labels, ids, kept, meta["width"], meta["height"])
                    # One diagnostic row per selector and chunk: a B head chooses the same class
                    # whatever dictionary set follows it.
                    if selector not in recorded:
                        recorded.add(selector)
                        selections[selector].append((record["class_name"], chosen))
                seconds = shared_cpu + time.process_time() - begin
                methods[experiment] = {"boxes": boxes, "method_seconds": seconds}
        shared_gpu = sum(v for k, v in record["seconds"].items() if not k.startswith("head_"))
        for experiment, output in methods.items():
            extra = selector_seconds(experiment, record)
            output["inference_compute_seconds"] = shared_gpu + extra + output["method_seconds"]
        by_window[record["window_id"]] = (record, methods)
    write_stage(args, config, experiments, by_window, sampler_for_protocol, VideoMeta)
    write_selectors(args.output / "selectors.json", selections)
    if getattr(args, "real_use", None):
        write_real_use(args, args.output, len(by_window))


def write_selectors(path: Path, selections: dict) -> None:
    """Write the balanced accuracy and confusion of every class selector on the test chunks."""
    out = {}
    for name, pairs in sorted(selections.items()):
        known = [(CLASSES.index(t), p) for t, p in pairs if t is not None and p is not None]
        true, predicted = zip(*known, strict=True) if known else ((), ())
        out[name] = {
            "chunks": len(pairs),
            "chunks_with_temporal_class": len(known),
            "balanced_accuracy": balanced_accuracy(true, predicted, len(CLASSES)),
            "confusion": confusion(true, predicted, len(CLASSES)),
            "classes": CLASSES,
        }
    write_json(path, out)


def write_stage(args, config, experiments, by_window, sampler_for_protocol, video_meta) -> None:
    """Write one sealed prediction stage per dataset, every protocol window of every video."""
    protocol = yaml.safe_load((ROOT / "configs/protocol/edp_v2.yaml").read_text(encoding="utf-8"))
    sampler = sampler_for_protocol(None, protocol, validazione=False)
    fit = read_json(args.fit_seconds) if args.fit_seconds else {}
    grid = next(iter(by_window.values()))[0]["grid"]
    for dataset, manifest in (("ucf_test", args.manifest_ucf), ("mivia_test", args.manifest_mivia)):
        for row in read_json(manifest):
            meta = video_meta(**row)
            target = args.output / dataset / video_key(meta.video_id)
            chunks = []
            for index, window in enumerate(sampler.windows_for_video(meta)):
                found = by_window.get(window.window_id)
                if found is not None and found[0]["dataset"] != dataset:
                    found = None
                methods = (
                    found[1]
                    if found is not None
                    else {
                        e: {
                            "boxes": [[] for _ in range(grid[0])],
                            "method_seconds": 0.0,
                            "inference_compute_seconds": 0.0,
                        }
                        for e in experiments
                    }
                )
                destination = target / "chunks" / f"{index:06d}.json"
                write_json(
                    destination,
                    {
                        "window": asdict(window),
                        "window_id": window.window_id,
                        "grid": grid,
                        "localized": found is not None,
                        "methods": methods,
                    },
                )
                chunks.append(
                    {
                        "path": str(destination.relative_to(target)),
                        "sha256": file_sha256(destination),
                    }
                )
            records = [read_json(target / c["path"]) for c in chunks]
            write_json(
                target / "complete.json",
                {
                    "status": "complete",
                    "video": row,
                    "chunks": chunks,
                    "timing": {
                        e: {
                            "fit_seconds": float(fit.get(e.rsplit("-", 1)[0], 0.0)),
                            "inference_compute_seconds": sum(
                                r["methods"][e]["inference_compute_seconds"] for r in records
                            ),
                            "inference_chunks": len(records),
                            "fit_shared_cost_do_not_sum_over_videos": True,
                        }
                        for e in experiments
                    },
                    "prediction_scope": "entire_video_all_protocol_windows",
                },
            )
        seal_stage(args.output, dataset, experiments, manifest)


#: Stage folders of ``localize`` and the dataset names of the GT snapshot.
DATASETS = {"ucf_test": "ucf_crime", "mivia_test": "mivia_crime"}


def best_rules(values: dict[str, float]) -> dict[str, dict]:
    """:return: per dictionary set and threshold rule of A, the first-step rule with the
    highest IoU (rules ``phase1``, ``phase2`` and ``massNN``, as the experiment suffixes)."""
    out = {}
    for key in ("A1", "A2"):
        for phase in sorted({name.rsplit("-", 1)[1] for name in values}):
            scored = {
                rule: values[f"{key}-{rule}-{phase}"]
                for rule in CLASS_RULES
                if f"{key}-{rule}-{phase}" in values
            }
            if scored:
                rule = max(scored, key=lambda r: (scored[r], -CLASS_RULES.index(r)))
                out[f"{key}-{phase}"] = {"rule": rule, "value": scored[rule], "all": scored}
    return out


def false_alarms(predictions: Path, intervals: dict) -> dict:
    """Boxes on the negative images of a sealed stage, per dataset and experiment (real use).

    The IoU measures the GT frames only; on the chunk planes of the frames outside every
    temporal event (``localization_auto_evaluate.negative_planes``) any box is a false alarm.
    Counted per image and per native frame.

    :param intervals: the ``intervals`` of the real-use plan, dataset → video → events.
    :return: per dataset the negative images and frames and, per experiment, those with a box.
    :raises ValueError: if a video of the seal has no temporal intervals.
    """
    from localization_auto_evaluate import negative_planes, sealed_videos

    out = {}
    for folder in DATASETS:
        seal = read_json(predictions / "operations" / f"seal-{folder}.json")
        methods = list(seal["methods"])
        boxed = {m: [0, 0] for m in methods}
        images = frames = 0
        for _, complete, chunks in sealed_videos(predictions, seal):
            video = complete["video"]["video_id"]
            if video not in intervals[folder]:
                raise ValueError(f"{folder} {video}: no temporal intervals")
            for (index, temporal), count in negative_planes(
                complete, chunks, intervals[folder][video]
            ).items():
                images += 1
                frames += count
                for m in methods:
                    if chunks[index]["methods"][m]["boxes"][temporal]:
                        boxed[m][0] += 1
                        boxed[m][1] += count
        out[folder] = {
            "negative_images": images,
            "negative_frames": frames,
            "methods": {
                m: {
                    "images_with_boxes": i,
                    "image_share": i / images if images else None,
                    "frames_with_boxes": f,
                    "frame_share": f / frames if frames else None,
                }
                for m, (i, f) in boxed.items()
            },
        }
    return out


def evaluate_stage(args) -> None:
    """Score every method of a sealed stage on the GT snapshot and summarize it.

    A result already on disk is reused (the stage and the snapshot are frozen), so the step
    resumes after a timeout. Every result must cover every annotated event and plane.

    :raises ValueError: if a result misses annotated planes or a seal is incomplete.
    """
    from localization_auto_evaluate import evaluate

    counts = read_json(args.snapshot / "snapshot.json")["counts"]
    summary = {"stage": str(args.predictions), "datasets": {}}
    for folder, dataset in DATASETS.items():
        seal = args.predictions / "operations" / f"seal-{folder}.json"
        rows = {}
        for experiment in read_json(seal)["methods"]:
            target = args.output / "results" / folder / f"{experiment}.json"
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                evaluate(args.snapshot, args.predictions, seal, dataset, experiment, target)
            result = read_json(target)
            covered = len(result["per_event"]), sum(e["planes"] for e in result["per_event"])
            if covered != (counts[f"{dataset}_events"], counts[f"{dataset}_planes"]):
                raise ValueError(f"{experiment} on {folder} misses annotated planes: {covered}")
            rows[experiment] = result["value"]
            summary["gt_snapshot_sha256"] = result["gt_snapshot_sha256"]
        summary["datasets"][folder] = dict(sorted(rows.items(), key=lambda kv: -kv[1]))
    summary["best_rule_on_ucf_test"] = best_rules(summary["datasets"]["ucf_test"])
    if getattr(args, "intervals", None):
        alarms = false_alarms(args.predictions, read_json(args.intervals)["intervals"])
        write_json(args.output / "false_alarms.json", alarms)
        summary["false_alarms"] = {
            folder: {
                "negative_images": body["negative_images"],
                "image_share": {m: v["image_share"] for m, v in body["methods"].items()},
            }
            for folder, body in alarms.items()
        }
    write_json(args.output / "summary.json", summary)
    mivia = summary["datasets"]["mivia_test"]
    lines = ["| Configuration | FS-UCF-Crime | MIVIA-Crime |", "|---|---:|---:|"]
    lines += [
        f"| {name} | {value:.4f} | {mivia.get(name, float('nan')):.4f} |"
        for name, value in summary["datasets"]["ucf_test"].items()
    ]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("calibrate", "localize", "evaluate"))
    parser.add_argument("--pass-root", type=Path, help="calibrate, localize")
    parser.add_argument("--dictionaries", type=Path, help="calibrate, localize")
    parser.add_argument("--predictions", type=Path, help="evaluate: a sealed localize stage")
    parser.add_argument("--snapshot", type=Path, help="evaluate: the frozen GT snapshot")
    parser.add_argument(
        "--thresholds", type=Path, help="thresholds.json; without it the 55% threshold only"
    )
    parser.add_argument("--configurations", choices=("campaign", "b3", "b4"), default="campaign")
    parser.add_argument(
        "--classifiers-set",
        help="set folder of the classifier pass (B4 logits on their own encoder)",
    )
    parser.add_argument(
        "--fixed-fraction",
        type=float,
        help="localize: one fixed fraction instead of the 55%% and calibrated rules "
        "(experiments end in -massNN; no --thresholds)",
    )
    parser.add_argument("--manifest-ucf", type=Path)
    parser.add_argument("--manifest-mivia", type=Path)
    parser.add_argument("--fit-seconds", type=Path, help="configuration -> fit seconds")
    parser.add_argument(
        "--heads-set", help="set folder of the class-head pass (B1/B2 logits without IG)"
    )
    parser.add_argument(
        "--without-heads", action="store_true", help="leave the B1/B2 configurations out"
    )
    parser.add_argument(
        "--heads-only", action="store_true", help="keep only the B1/B2 configurations"
    )
    parser.add_argument(
        "--real-use",
        type=Path,
        help="localize: a plan of localization_class_real_use.py; only the chunks the detector "
        "flags get boxes",
    )
    parser.add_argument(
        "--extra-sets", nargs="*", default=[], help="localize: pass sets read besides test"
    )
    parser.add_argument(
        "--intervals",
        type=Path,
        help="evaluate: intervals.json of the real-use plan; counts the boxes on the frames "
        "outside every event",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(CONFIG)
    if args.step == "evaluate":
        if args.predictions is None or args.snapshot is None:
            raise ValueError("evaluate needs --predictions and --snapshot")
        evaluate_stage(args)
        return
    if args.pass_root is None or args.dictionaries is None:
        raise ValueError(f"{args.step} needs --pass-root and --dictionaries")
    if args.step == "calibrate":
        calibrate(args, config)
        return
    if args.manifest_ucf is None or args.manifest_mivia is None:
        raise ValueError("localize needs both test manifests")
    args.metas = {
        "ucf_test": {r["video_id"]: r for r in read_json(args.manifest_ucf)},
        "mivia_test": {r["video_id"]: r for r in read_json(args.manifest_mivia)},
    }
    localize(args, config)


if __name__ == "__main__":
    main()
