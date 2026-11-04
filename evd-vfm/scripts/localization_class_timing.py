"""Time the class-routing localization chain as a deployed system would run it.

Plan: ``docs/varie/piano_localizzazione_per_classe_20261021.md`` §7.6. Two steps:

- ``chain`` (GPU or CPU-only, ``--device``): on 4 annotated test chunks per class (both test
  sets, seed-42 hash order) the whole chain runs with a synchronization at every step
  boundary: decoding and preprocessing, encoder with the binary head, baseline encodings,
  fixed-node IG, SLIC, NNLS on the global (8), stacked (104) and class (8) dictionaries,
  concept prefix and boxes, and the class heads: B1 whole and as a branch on its root (the
  root repeats the first blocks of the binary head, so a deployment can share it), B2, and
  the B3-GradNorm head (its class head alone and both heads). The binary head is also timed
  alone. The first ``--warmup`` chunks are not counted.

  ``--groups 1 2 4 8`` runs the sample once per group size. With a group of N the chunks run
  N at a time: the encoder takes the N clips, then the N references of each baseline, as one
  batch; the class heads take the group as one batch; SLIC starts in a process pool as soon
  as the tokens exist and overlaps with IG; IG stays per chunk (its queue is already batched).
  The group wall time is the latency of every chunk of the group; chunks per second are the
  throughput.

  CPU-only (``--device cpu``) runs everything in fp32 with every CPU of the job, one chunk at
  a time; ``--ig-nodes`` lowers the IG nodes for a first cost probe (the result records the
  nodes used, and the report scales the integral linearly to the campaign nodes, declared).
- ``report`` (CPU): median, 95th percentile and mean per step and per method (P07, A, B1 with
  a shared root, B1 whole, B2, B3), per group size latency and throughput, and the fit
  seconds of every dictionary set and training hours.

The IG settings (nodes, batch, precision, baselines, tolerances) are those of the campaign
config on GPU; the CPU run keeps nodes and batch and runs in fp32. Nothing is written outside
``--output``; no ground truth is read.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402

from evd.exp.localization_auto import read_json, write_json  # noqa: E402

CONFIG = ROOT / "configs/localization_class_routing_20261021.json"
PER_CLASS = 4
#: Steps shared by every method: the detector, its IG map and the SLIC regions.
COMMON = ("decode", "encode", "baselines", "integral", "slic")
#: Steps of each method, summed into its total. ``encode`` includes the binary head; B3 adds
#: its class head only (its binary head is the detector's).
METHOD_STEPS = {
    "detector": ("decode", "encode"),
    "P07": (*COMMON, "nnls_global", "boxes_p07"),
    "A": (*COMMON, "nnls_stacked", "nnls_class", "boxes_class"),
    "B1-shared-root": (*COMMON, "head_B1_branch", "nnls_class", "boxes_class"),
    "B1": (*COMMON, "head_B1", "nnls_class", "boxes_class"),
    "B2": (*COMMON, "head_B2", "nnls_class", "boxes_class"),
    "B3": (*COMMON, "head_B3_classes", "nnls_class", "boxes_class"),
}


def _sample(test: dict) -> list[dict]:
    from evd.exp.vtcd_sampling import stable
    from localization_class_plan import CLASSES

    rows = []
    for name in CLASSES:
        pool = sorted(
            (r for r in test["rows"] if r["class_name"] == name),
            key=lambda r: stable(r["window_id"]),
        )
        rows += pool[:PER_CLASS]
    return rows


def _reference(clip, fill: str, window_id: str, spec):
    """:return: the IG reference clip of one baseline.

    Same arguments as :func:`evd.exp.localization_class.encode_with_baselines`, which encodes
    one clip at a time; the grouped run needs the references to batch them, and the campaign
    module is left as it was when the numbers were produced.
    """
    from evd.exp.explanation_campaign import seed_from_window_id
    from evd.explain.perturbation import reference_clip

    if fill == "black":
        return reference_clip(clip, fill, 31, mean=spec.mean, std=spec.std)
    if fill == "noise":
        return reference_clip(clip, fill, 31, seed=seed_from_window_id(window_id))
    return reference_clip(clip, fill, 31)


def _job_cpus() -> int:
    """:return: the CPUs of this job.

    ``SLURM_CPUS_PER_TASK`` first: the cluster nodes do not confine the affinity of a job, so
    ``sched_getaffinity`` returns every CPU of the node (112), not the ones requested.
    """
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _settings(config: dict, device: str, nodes: int | None, batch: int | None):
    from evd.exp.localization_class import IGSettings

    ig = config["ig"]
    if ig["nodes"] is None or ig["batch_size"] is None:
        raise ValueError("IG nodes and batch are not set: run and accept the pilot first")
    return IGSettings(
        nodes=int(nodes or ig["nodes"]),
        batch_size=int(batch or ig["batch_size"]),
        precision=ig["precision"] if device == "cuda" else "fp32",
        fills=tuple(ig["fills"]),
        atol=ig["tolerance"]["atol"],
        rtol=ig["tolerance"]["rtol"],
    )


def chain(args, config: dict) -> None:
    import torch

    from evd.data.dataset import WindowItem
    from evd.exp.demo import live_engine, open_run
    from evd.exp.explanation_advanced import prepare_clip, window_from_record
    from evd.exp.localization_class import (
        choose_class,
        class_scores,
        concept_boxes,
        encode_with_baselines,
        integrate,
        positive_masses,
        prefix_concepts,
        stack_dictionaries,
    )
    from evd.exp.localization_vtcd import CONCEPTS, segment
    from evd.explain.localization import native_boxes
    from evd.explain.vtcd import assign_concepts
    from evd.models.backbones.base import encode_multilevel
    from evd.models.heads.class_queries import _per_class_logits
    from localization_class_pass import _class_head

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available; no silent CPU fallback")
    if device.type == "cpu" and args.groups != [1]:
        raise ValueError("The CPU-only chain runs one chunk at a time: use --groups 1")
    torch.manual_seed(42)
    # As the GPU pass: one torch thread, SLIC in other processes. CPU-only: every CPU.
    torch.set_num_threads(1 if device.type == "cuda" else _job_cpus())

    def sync() -> float:
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    settings = _settings(config, device.type, args.ig_nodes, args.ig_batch)
    runs = Path(os.environ["EVD_RUNS_DIR"])
    run = open_run(runs / config["detectors"]["campaign"])
    engine, spec = live_engine(run, device=str(device), batch_size=1, num_workers=0)
    scorer = engine.scorer
    if device.type == "cpu":
        scorer.precision = "fp32"
    dim = scorer.backbone.embed_dim
    heads = {
        name: _class_head(runs / config["class_heads"][name], dim, str(device))[0]
        for name in ("B1", "B2")
    }
    b3 = _class_head(runs / config["detectors"]["b3_gradnorm"], dim, str(device))[0]
    if not callable(getattr(b3, "forward_tasks", None)):
        raise ValueError("The B3-GradNorm head has no class head (forward_tasks)")
    model = np.load(args.dictionaries)
    dictionaries = model[args.dictionary_set]
    stacked, owners = stack_dictionaries(dictionaries)
    rows = _sample(read_json(args.test))[: args.max_chunks]

    def timed(fn):
        t0 = sync()
        with torch.no_grad(), scorer.autocast():
            out = fn()
        return out, sync() - t0

    def class_heads(feats, n: int) -> dict:
        """:return: seconds per chunk of each class head on a batch of ``n`` chunks."""
        steps = {}
        _, steps["head_binary"] = timed(lambda: scorer.head(feats).float().cpu())
        _, steps["head_B1"] = timed(lambda: heads["B1"](feats).float().cpu())
        root, steps["head_B1_root"] = timed(lambda: heads["B1"].root(feats))
        _, steps["head_B1_branch"] = timed(
            lambda: _per_class_logits(heads["B1"].branch(root), heads["B1"].linear).float().cpu()
        )
        _, steps["head_B2"] = timed(lambda: heads["B2"](feats).float().cpu())
        _, steps["head_B3_classes"] = timed(lambda: b3.classes(feats).float().cpu())
        _, steps["head_B3_both"] = timed(
            lambda: {k: v.float().cpu() for k, v in b3.forward_tasks(feats).items()}
        )
        return {k: v / n for k, v in steps.items()}

    def localize(row, steps, ig_map, segmented) -> None:
        labels, descriptors, counts, slic_process = segmented
        # CPU seconds of SLIC in its own process, also when it overlapped with IG.
        steps["slic_process"] = float(slic_process)
        descriptors = descriptors.astype(np.float64)
        width, height = int(row["width"]), int(row["height"])
        t0 = time.perf_counter()
        ids, _ = assign_concepts(descriptors, model["global_dictionary"])
        steps["nnls_global"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        kept = prefix_concepts(positive_masses(ig_map, labels, ids, CONCEPTS), args.fraction)
        native_boxes(np.isin(ids[labels], kept), width, height)
        steps["boxes_p07"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        _, coefficients = assign_concepts(descriptors, stacked)
        chosen = choose_class(
            class_scores(coefficients, counts, owners, len(dictionaries), args.rule)
        )
        steps["nnls_stacked"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        class_ids, _ = assign_concepts(descriptors, dictionaries[chosen])
        steps["nnls_class"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        kept = prefix_concepts(positive_masses(ig_map, labels, class_ids, CONCEPTS), args.fraction)
        concept_boxes(labels, class_ids, kept, width, height)
        steps["boxes_class"] = time.perf_counter() - t0

    for group in args.groups:
        pool = None
        if group > 1:
            pool = ProcessPoolExecutor(
                max_workers=group, mp_context=multiprocessing.get_context("spawn")
            )
        out, path = [], args.output / f"chain-{args.device}-group{group}.json"
        for index, begin in enumerate(range(0, len(rows), group)):
            batch = rows[begin : begin + group]
            n = len(batch)
            start = sync()
            clips = [
                prepare_clip(engine, WindowItem(window_from_record(r), r["path"])) for r in batch
            ]
            decoded = sync()
            records = [{"decode": (decoded - start) / n} for _ in batch]
            if group == 1:
                feats, _, bases, seconds = encode_with_baselines(
                    scorer, spec, clips[0], batch[0]["window_id"], settings.fills
                )
                records[0].update(encode=seconds["forward"], baselines=seconds["baselines"])
                tokens = feats.tokens[0].float().cpu().numpy().reshape(*feats.grid, -1)
                if args.device == "cpu":
                    print(json.dumps({"chunk": begin, **records[0]}), flush=True)
                maps, _, records[0]["integral"] = integrate(scorer, feats, bases, settings)
                records[0].update(class_heads(feats, 1))
                t0 = time.perf_counter()
                segmented = [segment(tokens)]
                records[0]["slic"] = time.perf_counter() - t0
                ig_maps = [maps["ig_ensemble"]]
                del feats, bases
            else:
                with torch.no_grad(), scorer.autocast():
                    all_feats = encode_multilevel(scorer.backbone, torch.cat(clips), scorer.levels)
                    scorer.head(all_feats).float().cpu()
                encoded = sync()
                tokens = all_feats.tokens.float().cpu().numpy()
                jobs = [
                    pool.submit(segment, tokens[i].reshape(*all_feats.grid, -1)) for i in range(n)
                ]
                bases = {}
                with torch.no_grad(), scorer.autocast():
                    for fill in settings.fills:
                        refs = torch.cat(
                            [
                                _reference(c, fill, r["window_id"], spec)
                                for c, r in zip(clips, batch, strict=True)
                            ]
                        )
                        bases[fill] = encode_multilevel(scorer.backbone, refs, scorer.levels).tokens
                        del refs
                based = sync()
                ig_maps = []
                for i, record in enumerate(records):
                    feats = replace(
                        all_feats,
                        tokens=all_feats.tokens[i : i + 1],
                        pooled=None if all_feats.pooled is None else all_feats.pooled[i : i + 1],
                    )
                    one = {k: v[i : i + 1] for k, v in bases.items()}
                    maps, _, record["integral"] = integrate(scorer, feats, one, settings)
                    ig_maps.append(maps["ig_ensemble"])
                    record.update(encode=(encoded - decoded) / n, baselines=(based - encoded) / n)
                shared_heads = class_heads(all_feats, n)
                t0 = time.perf_counter()
                segmented = [job.result() for job in jobs]
                waited = time.perf_counter() - t0
                for record in records:
                    record.update(shared_heads)
                    # SLIC ran during IG: only the wait after IG is on the critical path.
                    record["slic"] = waited / n
                del all_feats, bases
            for row, record, ig_map, seg in zip(batch, records, ig_maps, segmented, strict=True):
                localize(row, record, ig_map, seg)
            wall = time.perf_counter() - start
            for row, record in zip(batch, records, strict=True):
                out.append(
                    {
                        "window_id": row["window_id"],
                        "dataset": row["dataset"],
                        "class_name": row["class_name"],
                        "group": n,
                        "group_index": index,
                        "group_wall_seconds": wall,
                        "steps": record,
                    }
                )
            print(json.dumps({"group": group, "done": begin + n, "of": len(rows)}), flush=True)
            write_json(
                path,
                {
                    "device": args.device,
                    "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
                    # Peak since the process started, so across the groups written so far.
                    "peak_memory_mb": (
                        torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
                    ),
                    "cpus": _job_cpus(),
                    "torch_threads": torch.get_num_threads(),
                    "slurm_job": os.environ.get("SLURM_JOB_ID"),
                    "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
                    "node": os.environ.get("SLURMD_NODENAME"),
                    "detector": config["detectors"]["campaign"],
                    "b3_head": config["detectors"]["b3_gradnorm"],
                    "ig": settings.__dict__ | {"fills": list(settings.fills)},
                    "ig_campaign_nodes": int(config["ig"]["nodes"]),
                    "dictionary_set": args.dictionary_set,
                    "rule": args.rule,
                    "fraction": args.fraction,
                    "group_size": group,
                    "warmup": args.warmup,
                    "chunks": out,
                },
            )
        if pool is not None:
            pool.shutdown()


def _stats(values: list[float]) -> dict:
    return {
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "mean": float(np.mean(values)),
        "n": len(values),
    }


def summarize(data: dict) -> dict:
    """Return per-step and per-method statistics of one chain file, warm-up excluded.

    The warm-up counts chunks; with groups it excludes every group that holds one of the first
    ``warmup`` chunks, so a group is either all counted or not at all.

    :param data: a ``chain-*.json`` file.
    :return: ``steps`` and ``methods`` (seconds per chunk), ``latency`` (group wall seconds),
        ``seconds_per_chunk`` and ``chunks_per_second``; empty when nothing is left.
    """
    group = max(1, int(data.get("group_size") or 1))
    skip = math.ceil(int(data["warmup"]) / group)
    kept = [c for c in data["chunks"] if c["group_index"] >= skip]
    if not kept:
        return {}
    steps = sorted({s for c in kept for s in c["steps"]})
    out = {
        "chunks": len(kept),
        "steps": {s: _stats([c["steps"][s] for c in kept if s in c["steps"]]) for s in steps},
        "methods": {
            m: _stats([sum(c["steps"][s] for s in names) for c in kept])
            for m, names in METHOD_STEPS.items()
            if all(s in kept[0]["steps"] for s in names)
        },
    }
    walls = {c["group_index"]: (c["group_wall_seconds"], c["group"]) for c in kept}
    out["latency"] = _stats([w for w, _ in walls.values()])
    total_wall = sum(w for w, _ in walls.values())
    out["seconds_per_chunk"] = total_wall / len(kept)
    out["chunks_per_second"] = len(kept) / total_wall
    nodes, campaign = int(data["ig"]["nodes"]), int(data.get("ig_campaign_nodes") or 0)
    if campaign and nodes != campaign:
        integral = out["steps"]["integral"]
        out["integral_scaled_to_campaign_nodes"] = {
            "nodes_used": nodes,
            "campaign_nodes": campaign,
            "rule": "linear in the nodes (head calls = baselines x nodes / batch)",
            **{k: v * campaign / nodes for k, v in integral.items() if k != "n"},
        }
    return out


def report(args) -> None:
    result = {"chains": {}}
    for path in sorted(args.output.glob("chain-*.json")):
        data = read_json(path)
        result["chains"][path.stem] = {
            k: data[k] for k in ("device", "gpu", "cpus", "torch_threads", "slurm_job", "node")
        } | {"ig": data["ig"], "group_size": data["group_size"], **summarize(data)}
    result["fit"] = {str(p): read_json(p) for p in args.fit_receipts}
    if args.training_hours:
        result["training_hours"] = read_json(args.training_hours)
    write_json(args.output / "timing_report.json", result)
    print(json.dumps({k: v.get("methods") for k, v in result["chains"].items()}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("chain", "report"))
    parser.add_argument("--test", type=Path, help="test.json of localization_class_plan.py")
    parser.add_argument("--dictionaries", type=Path, help="dictionaries.npz of the campaign")
    parser.add_argument("--dictionary-set", choices=("a1", "a2"), default="a2")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--groups", type=int, nargs="+", default=[1], help="chunks per group")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-chunks", type=int, default=10_000)
    parser.add_argument("--rule", default="regione-un-voto", help="class rule of proposal A")
    parser.add_argument("--fraction", type=float, default=0.5, help="fixed mass fraction")
    parser.add_argument("--ig-nodes", type=int, help="IG nodes (default: the campaign's)")
    parser.add_argument("--ig-batch", type=int, help="IG batch (default: the campaign's)")
    parser.add_argument("--fit-receipts", type=Path, nargs="*", default=[])
    parser.add_argument("--training-hours", type=Path, help="training hours file")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(g < 1 for g in args.groups):
        parser.error("--groups must be positive")
    if args.step == "chain":
        args.output.mkdir(parents=True, exist_ok=True)
        chain(args, read_json(CONFIG))
    else:
        report(args)


if __name__ == "__main__":
    main()
