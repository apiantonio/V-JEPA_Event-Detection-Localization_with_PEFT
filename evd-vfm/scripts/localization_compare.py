"""CPU steps of the localization comparisons (methods approved on 29 October 2026).

Methods of other paradigms evaluated with the official AP of ``localization_class_ap.py`` on
the campaign detector (LoRA+VPT base) and its chunks: IG alone, Grad-CAM, CRIS and the VTCD
concepts without IG (rules in :mod:`evd.exp.localization_compare`).

- ``plan-enrichment``: the plan of the training chunks the concept enrichment is measured on,
  the positive and normal chunks of the 16 October train plan (P05), for
  ``localization_class_pass.py`` without IG.
- ``enrichment``: the P05/P10 log-ratio of every concept of the global, A1 and A2
  dictionaries (``localization_vtcd.enrichment``: presence in the positive chunks of the class,
  all classes for the global dictionary, against the normal ones, smoothed per video), checked
  against the ratios the 16 October fit saved for the global and A1 dictionaries.
- ``timing``: the median seconds per chunk of the parts of every method's cost.
- ``detections``: an AP stage of the comparison configurations (:data:`IG_CONFIGURATIONS` on
  the IG maps of a pass root, :data:`OTHER_CONFIGURATIONS` on the Grad-CAM and CRIS pass and
  the dictionaries), in the format of ``localization_class_ap.py detections --real-use
  --keep-annotated``: every annotated test chunk and every flagged chunk; ``evaluate`` of that
  script reads it with ``--variants component``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402

from evd.exp import localization_compare as lc  # noqa: E402
from evd.exp.localization_auto import read_json, write_json  # noqa: E402
from localization_class_plan import CLASSES, sealed  # noqa: E402

#: Configurations of IG alone: name -> keyword arguments of ``ig_tree_boxes`` plus ``map``,
#: the array of the pass record (``abs``: the absolute ensemble). The first is the method,
#: the others its ablations, one axis at a time.
IG_CONFIGURATIONS = {
    "IG-tree": {},
    "IG-tree-density": {"score": "density"},
    "IG-tree-peak": {"score": "peak"},
    "IG-tree-level": {"score": "level"},
    "IG-tree-persistence": {"score": "persistence"},
    "IG-tree-conn4": {"connectivity": 4},
    "IG-tree-nms30": {"rule": "nms30-component"},
    "IG-tree-nms70": {"rule": "nms70-component"},
    "IG-tree-softnms": {"rule": "softnms-component"},
    "IG-tree-wbf": {"rule": "wbf-component"},
    "IG-tree-nonms": {"rule": "component"},
    "IG-tree-largest": {"rule": "largest"},
    "IG-q100": {"thresholds": "q100"},
    "IG-nested": {"thresholds": "nested"},
    "IG-top5": {"thresholds": "top5"},
    "IG-tree-min01": {"min_share": 0.001},
    "IG-tree-black": {"map": "ig_black"},
    "IG-tree-noise": {"map": "ig_noise"},
    "IG-tree-mean": {"map": "ig_mean"},
    "IG-tree-blur": {"map": "ig_blur"},
    "IG-tree-abs": {"map": "abs"},
}
#: Configurations without IG: Grad-CAM, CRIS and the VTCD dictionaries ordered by enrichment.
OTHER_CONFIGURATIONS = (
    "GradCAM",
    "GradCAM-tree",
    "CRIS-top10",
    "CRIS-top1",
    "VTCD-global",
    "VTCD-global-components",
    "VTCD-global-tokens",
    "VTCD-class",
    "VTCD-class-components",
    "VTCD-class-tokens",
    "VTCD-class-A1",
)


def plan_enrichment(args) -> None:
    """Write the plan of the enrichment chunks: the 16 October P05 positives and normals."""
    train = read_json(args.train_plan)
    rows = [
        {**r, "dataset": "ucf_train", "positive": role == "positive"}
        for role in ("positive", "normal")
        for r in train[role]
    ]
    body = {"rows": rows, "train_plan_sha256": train["sha256"], "source": "P05 positive+normal"}
    write_json(args.output, sealed(body))
    print(json.dumps({"rows": len(rows), "positive": sum(r["positive"] for r in rows)}))


def enrichment(args) -> None:
    """Write the enrichment of every concept of the three dictionary families.

    :raises ValueError: if a chunk of the plan has no record, or the global or A1 dictionary
        differs from the one of the 16 October fit while its ratios are compared.
    """
    from evd.exp import localization_vtcd as lv
    from localization_class_pass import _key

    plan = read_json(args.plan)
    rows = plan["rows"]
    model = np.load(args.dictionaries)
    old = np.load(args.fit16 / "vtcd_models.npz")
    fit16 = read_json(args.fit16 / "vtcd_fit.json")
    arrays = []
    for row in rows:
        path = args.pass_set / f"{_key(row)}.npz"
        if not path.exists():
            raise ValueError(f"No record for {row['window_id']}")
        arrays.append(np.load(path))
    positive = np.array([r["positive"] for r in rows], dtype=bool)
    videos = [r["video_id"] for r in rows]

    def presence(dictionary):
        return np.stack(
            [
                lv.chunk_concepts(a["descriptors"].astype(np.float64), a["counts"], dictionary)[
                    "presence"
                ]
                for a in arrays
            ]
        )

    out = {"plan_sha256": plan["sha256"], "dictionaries": str(args.dictionaries)}
    out["global"] = lv.enrichment(presence(model["global_dictionary"]), positive, videos).tolist()
    for family in ("a1", "a2"):
        ratios = {}
        for index, name in enumerate(CLASSES):
            keep = [i for i, r in enumerate(rows) if not r["positive"] or r["class_name"] == name]
            chunk = presence(model[family][index])[keep]
            ratios[name] = lv.enrichment(chunk, positive[keep], [videos[i] for i in keep]).tolist()
        out[family] = ratios
    checks = {}
    if np.array_equal(model["global_dictionary"], old["global_dictionary"]):
        checks["global_max_abs_diff_vs_16oct"] = float(
            np.max(np.abs(np.array(out["global"]) - np.array(fit16["p05_ratios"])))
        )
    if np.array_equal(model["a1"], old["class_dictionaries"]):
        checks["a1_max_abs_diff_vs_16oct"] = float(
            max(
                np.max(np.abs(np.array(out["a1"][n]) - np.array(fit16["p10_ratios"][n])))
                for n in CLASSES
            )
        )
    out["checks"] = checks
    write_json(args.output, out)
    print(json.dumps(checks))


def _ig_partitions(arrays, meta: dict, seconds: dict | None = None) -> dict:
    """:return: the partition of every :data:`IG_CONFIGURATIONS` entry on one chunk; with
    ``seconds``, the CPU seconds of each are added to it."""
    out = {}
    for name, spec in IG_CONFIGURATIONS.items():
        start = time.process_time()
        spec = dict(spec)
        source = spec.pop("map", "ig_ensemble")
        if source == "abs":
            values = np.abs(arrays["ig_ensemble"].astype(np.float64))
        else:
            values = arrays[source].astype(np.float64)
        out[name] = lc.ig_tree_boxes(values, meta["width"], meta["height"], **spec)
        if seconds is not None:
            seconds[name] = time.process_time() - start
    return out


def _cris_partition(compare, concepts: list[dict], meta: dict, top: int) -> dict:
    """:return: the boxes of the ``top`` CRIS concepts, one per component, merged per plane."""
    planes = None
    for head in sorted({c["head"] for c in concepts[:top]}):
        scores = {c["concept"]: c["score"] for c in concepts[:top] if c["head"] == head}
        part = lc.concept_planes(
            compare[f"cris_{head}"], scores, meta["width"], meta["height"], per_component=True
        )
        planes = (
            part["planes"]
            if planes is None
            else [a + b for a, b in zip(planes, part["planes"], strict=True)]
        )
    return {"masses": [1.0], "planes": planes}


def _other_partitions(record, arrays, compare, model, enrich, concepts, meta) -> dict:
    """:return: the partition of every :data:`OTHER_CONFIGURATIONS` entry on one chunk."""
    from evd.explain.localization import MaskRule, map_mask
    from evd.explain.vtcd import assign_concepts
    from localization_class_localize import _selector_class

    width, height = meta["width"], meta["height"]
    out = {}
    cam = compare["gradcam"].astype(np.float64)
    out["GradCAM"] = lc.mask_components(cam, map_mask(cam, MaskRule()), width, height)
    out["GradCAM-tree"] = lc.ig_tree_boxes(cam, width, height)
    out["CRIS-top10"] = _cris_partition(compare, concepts, meta, 10)
    out["CRIS-top1"] = _cris_partition(compare, concepts, meta, 1)
    descriptors, labels = arrays["descriptors"].astype(np.float64), arrays["labels"]

    def tokens_of(dictionary):
        ids, _ = assign_concepts(descriptors, dictionary)
        return np.asarray(ids)[labels]

    def dictionary_set(prefix: str, dictionary, ratios) -> None:
        ids = tokens_of(dictionary)
        scores = dict(enumerate(ratios))
        out[prefix] = lc.concept_planes(ids, scores, width, height)
        out[f"{prefix}-components"] = lc.concept_planes(
            ids, scores, width, height, per_component=True
        )
        # Token share: every concept present scores its share; the enrichment only says
        # which concepts count (a concept enriched in the normal chunks gives no box).
        share = lc.token_share_scores(ids, len(ratios))
        out[f"{prefix}-tokens"] = lc.concept_planes(
            ids, {c: share[c] if ratios[c] > 0 else 0.0 for c in share}, width, height
        )

    dictionary_set("VTCD-global", model["global_dictionary"], enrich["global"])
    chosen = _selector_class("B1", record, {})
    name = CLASSES[chosen]
    dictionary_set("VTCD-class", model["a2"][chosen], enrich["a2"][name])
    ids = tokens_of(model["a1"][chosen])
    out["VTCD-class-A1"] = lc.concept_planes(
        ids, dict(enumerate(enrich["a1"][name])), width, height
    )
    return out


def _chunk_job(payload: tuple) -> tuple:
    """Material of one chunk (a worker task). :return: ``(window_id, partitions, seconds)``."""
    kind, record, ig_path, compare_path, meta, context = payload
    start = time.process_time()
    arrays = np.load(ig_path)
    if kind == "ig":
        partitions = _ig_partitions(arrays, meta)
    else:
        model = np.load(context["dictionaries"])
        partitions = _other_partitions(
            record,
            arrays,
            np.load(compare_path),
            model,
            context["enrichment"],
            context["concepts"],
            meta,
        )
    return record["window_id"], partitions, time.process_time() - start


def detections(args) -> None:
    """Build and seal the AP stage of the comparison configurations of ``--kind``.

    :raises ValueError: if a chunk has no Grad-CAM/CRIS record (``--kind other``).
    """
    from localization_class_ap import write_stage
    from localization_class_localize import head_records, stage_records, with_heads
    from localization_class_pass import _key

    names = list(IG_CONFIGURATIONS) if args.kind == "ig" else list(OTHER_CONFIGURATIONS)
    heads, context = None, {}
    if args.kind == "other":
        heads = head_records(args.pass_root / args.heads_set)
        context = {
            "dictionaries": str(args.dictionaries),
            "enrichment": read_json(args.enrichment),
            "concepts": None,
        }
    payloads = []
    for record, _ in stage_records(args):
        if args.kind == "other":
            record = with_heads(record, heads)
        key, folder = _key(record), record["identity"]["set"]
        ig_path = args.pass_root / folder / f"{key}.npz"
        compare_path = None
        if args.kind == "other":
            compare_path = args.compare_root / folder / f"{key}.npz"
            if not compare_path.exists():
                raise ValueError(f"No Grad-CAM/CRIS record for {record['window_id']}")
            if context["concepts"] is None:
                context["concepts"] = read_json(compare_path.with_suffix(".json"))["identity"][
                    "cris_concepts"
                ]
        meta = args.metas[record["dataset"]][record["video_id"]]
        payloads.append((args.kind, record, ig_path, compare_path, meta, context))
    by_window = {}
    records = {p[1]["window_id"]: p[1] for p in payloads}
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for window_id, partitions, seconds in pool.map(_chunk_job, payloads, chunksize=4):
            methods = {n: {"partition": n} for n in names}
            by_window[window_id] = (records[window_id], methods, partitions, None, seconds)
    write_stage(args, by_window, names, {"kind": args.kind, "pass_root": str(args.pass_root)})


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def timing(args) -> None:
    """Write the median seconds per chunk of the parts of every method's cost.

    From the pass records (H100): decoding, encoder and binary head (``forward``), the encoder
    on the IG baselines, the IG integral at 256 and 32 nodes, SLIC (CPU), Grad-CAM, the CRIS
    key segmentation of the top-10 heads (CPU, their sum) and the B1 head. The box step of
    every method (CPU, one process) is timed here on the first ``--sample`` annotated test
    chunks, ours as the class choice, dictionary assignment and components of B1-A2.
    """
    from evd.exp.localization_class import stack_dictionaries
    from evd.explain.localization import MaskRule, map_mask
    from evd.explain.vtcd import assign_concepts
    from localization_class_ap import chunk_partitions
    from localization_class_localize import _selector_class, head_records, with_heads
    from localization_class_pass import _key

    def records(folder: Path) -> list[dict]:
        return [read_json(f) for f in sorted(folder.glob("*.json"))]

    parts: dict[str, list[float]] = {}

    def add(name: str, value) -> None:
        if value is not None:
            parts.setdefault(name, []).append(float(value))

    for name in ("test", "real"):
        for record in records(args.pass_root / name):
            seconds = record["seconds"]
            for key, part in (
                ("decode", "decode"),
                ("forward", "forward"),
                ("baselines", "baselines"),
                ("integral", "integral_256"),
                ("slic_cpu", "slic"),
                ("head_B1", "head"),
            ):
                add(part, seconds.get(key))
        for record in records(args.nodes32_root / name):
            add("integral_32", record["seconds"].get("integral"))
        for record in records(args.compare_root / name):
            seconds = record["seconds"]
            add("gradcam", seconds["gradcam"])
            add("cris", sum(v for k, v in seconds.items() if k.startswith("cris_cpu_")))
    for record in records(args.pass_root / "test_heads"):
        add("head", record["seconds"].get("head_B1"))
    model = np.load(args.dictionaries)
    enrich = read_json(args.enrichment)
    stacked = {key: stack_dictionaries(model[key]) for key in ("a1", "a2")}
    heads = head_records(args.pass_root / "test_heads")
    sample = records(args.pass_root / "test")[: args.sample]
    concepts = None
    for record in sample:
        key = _key(record)
        arrays = np.load(args.pass_root / "test" / f"{key}.npz")
        compare = np.load(args.compare_root / "test" / f"{key}.npz")
        if concepts is None:
            concepts = read_json(args.compare_root / "test" / f"{key}.json")["identity"][
                "cris_concepts"
            ]
        meta = args.metas[record["dataset"]][record["video_id"]]
        width, height = meta["width"], meta["height"]
        seconds: dict[str, float] = {}
        _ig_partitions(arrays, meta, seconds)
        for name, value in seconds.items():
            add(f"box:{name}", value)
        cam = compare["gradcam"].astype(np.float64)
        start = time.process_time()
        lc.mask_components(cam, map_mask(cam, MaskRule()), width, height)
        add("box:GradCAM", time.process_time() - start)
        start = time.process_time()
        _cris_partition(compare, concepts, meta, 10)
        add("box:CRIS-top10", time.process_time() - start)
        descriptors, labels = arrays["descriptors"].astype(np.float64), arrays["labels"]
        record = with_heads(record, heads)
        for name, family in (("VTCD-global", None), ("VTCD-class", "a2")):
            start = time.process_time()
            if family is None:
                dictionary, ratios = model["global_dictionary"], enrich["global"]
            else:
                chosen = _selector_class("B1", record, {})
                dictionary, ratios = model[family][chosen], enrich[family][CLASSES[chosen]]
            ids, _ = assign_concepts(descriptors, dictionary)
            lc.concept_planes(np.asarray(ids)[labels], dict(enumerate(ratios)), width, height)
            add(f"box:{name}", time.process_time() - start)
        start = time.process_time()
        chunk_partitions(record, arrays, model, stacked, ["B1-A2"], meta)
        add("box:ours", time.process_time() - start)
    write_json(
        args.output,
        {
            "medians": {name: _median(values) for name, values in sorted(parts.items())},
            "counts": {name: len(values) for name, values in sorted(parts.items())},
            "sample": args.sample,
        },
    )
    print(json.dumps({name: round(_median(v), 4) for name, v in sorted(parts.items())}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("plan-enrichment", "enrichment", "detections", "timing"))
    parser.add_argument("--train-plan", type=Path, help="plan-enrichment: 16 Oct train-plan")
    parser.add_argument("--plan", type=Path, help="enrichment: the plan-enrichment file")
    parser.add_argument("--pass-set", type=Path, help="enrichment: the pass set of that plan")
    parser.add_argument("--fit16", type=Path, help="enrichment: the 16 October fits folder")
    parser.add_argument("--dictionaries", type=Path, help="enrichment, detections")
    parser.add_argument("--kind", choices=("ig", "other"), help="detections")
    parser.add_argument("--pass-root", type=Path, help="detections: records with IG maps")
    parser.add_argument("--compare-root", type=Path, help="detections --kind other")
    parser.add_argument("--enrichment", type=Path, help="detections --kind other")
    parser.add_argument("--heads-set", help="detections: set with the B1 logits of test")
    parser.add_argument("--real-use", type=Path, help="detections: the detector's real-use plan")
    parser.add_argument("--extra-sets", nargs="*", default=["real"])
    parser.add_argument("--manifest-ucf", type=Path)
    parser.add_argument("--manifest-mivia", type=Path)
    parser.add_argument("--nodes32-root", type=Path, help="timing: the 32-node pass root")
    parser.add_argument("--sample", type=int, default=60, help="timing: chunks timed on CPU")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.step == "plan-enrichment":
        plan_enrichment(args)
    elif args.step == "enrichment":
        enrichment(args)
    else:
        args.keep_annotated = True
        args.metas = {
            "ucf_test": {r["video_id"]: r for r in read_json(args.manifest_ucf)},
            "mivia_test": {r["video_id"]: r for r in read_json(args.manifest_mivia)},
        }
        if args.step == "timing":
            timing(args)
        else:
            detections(args)


if __name__ == "__main__":
    main()
