"""GPU pass of the class-routing localization campaign over a frozen list of chunks.

Plan: ``docs/varie/piano_localizzazione_per_classe_20261021.md`` §6.2 and §7.2. For every
chunk of a list written by ``localization_class_plan.py`` (test, validation, references or
pure train chunks) and one detector run, this script saves what the CPU steps need:

- the binary logit, and with ``--ig`` the fixed-node IG maps of the four baselines and their
  mean (settings from ``configs/localization_class_routing_20261021.json``, filled by the
  pilot; the pass refuses to run while they are null);
- the 32 SLIC regions of the token volume, their descriptors and token counts (SLIC runs on
  the encoder tokens, so it is redone for every encoder);
- the class logits of the class heads: B1 and B2 (``--class-heads``, applied to the
  detector features after checking that their frozen encoder is the detector's) or the class
  head of a B3 detector.

With ``--classifiers`` (B4, 26 October 2026) the pass runs no detector: every chunk is
encoded by each classifier's OWN trained encoder (V-JEPA 2.1-L with its LoRA, or with the
merged LoRA and its VPT prompts) and its class head, and only the class logits and the
seconds are saved, one ``.json`` per chunk (no IG, no SLIC: the second step uses the
detector pass of the same chunks).

One ``.json`` record and one ``.npz`` per chunk under ``--output/<set>/``; resumes per chunk;
shards on the window id. No spatial ground truth is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from evd.exp.localization_auto import in_shard, read_json, write_json  # noqa: E402

CONFIG = ROOT / "configs/localization_class_routing_20261021.json"
#: Outputs of a class head: the 13 anomalous classes of UCF-Crime and Normal.
NUM_OUTPUTS = 14


def rows_of(plan: dict) -> list[dict]:
    """:return: the chunk rows of a plan file (``rows``, or ``by_class`` for pure chunks)."""
    if "rows" in plan:
        return list(plan["rows"])
    return [row for rows in plan["by_class"].values() for row in rows]


def _key(row: dict) -> str:
    return hashlib.sha256(f"{row['dataset']}:{row['window_id']}".encode()).hexdigest()[:24]


def _class_head(run_dir: Path, embed_dim: int, device: str):
    """:return: a B1/B2 class head with its selected weights, and its selected adapter."""
    import torch

    from evd.models.heads.base import HEADS

    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    group = {k: v for k, v in config["head"].items() if k != "id"}
    head = HEADS.get(str(config["head"]["id"]))(embed_dim=embed_dim, **group)
    data = torch.load(run_dir / "checkpoint/checkpoint.pt", map_location="cpu", weights_only=True)
    if not data.get("selected_head"):
        raise ValueError(f"{run_dir} has no selected head")
    head.load_state_dict(data["selected_head"])
    return (
        head.to(device).eval(),
        dict(data.get("selected_adapter") or {}),
        data.get("selected_step"),
    )


def classifier_identity(runs: Path, config: dict, names) -> dict:
    """:return: per classifier, its run id, checkpoint hash and selected step.

    :raises ValueError: if a run has no selected head: the class choice uses the weights of
        the model selection, never the last step.
    """
    import torch

    from evd.utils.hashing import file_sha256

    out = {}
    for name in names:
        run_id = config["class_classifiers"][name]
        path = runs / run_id / "checkpoint/checkpoint.pt"
        data = torch.load(path, map_location="cpu", weights_only=True)
        if not data.get("selected_head"):
            raise ValueError(f"{run_id} has no selected head")
        out[name] = {
            "run_id": run_id,
            "checkpoint_sha256": file_sha256(path),
            "selected_step": data.get("selected_step"),
        }
        del data
    return out


def classifiers_pass(args) -> None:
    """Save the class logits of the B4 classifiers, each on its own encoder, per chunk.

    The clip is decoded once with the preprocessing of the first classifier; every classifier
    must declare the same preprocessing, or the logits would not describe the same pixels.
    ``live_engine`` rebuilds each model as the runner did (pre-adaptation included, ``docs/09``
    R24) with its selected weights.

    :raises ValueError: if the preprocessing differs, a head does not have 14 outputs, or a
        record on disk has another identity.
    """
    import torch

    from evd.data.dataset import WindowItem
    from evd.exp.demo import live_engine, open_run
    from evd.exp.explanation_advanced import prepare_clip, window_from_record
    from evd.models.backbones.base import encode_multilevel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no silent CPU fallback")
    torch.manual_seed(42)
    config = read_json(CONFIG)
    runs = Path(os.environ["EVD_RUNS_DIR"])
    identity_of = classifier_identity(runs, config, args.classifiers)
    engines = {
        name: live_engine(open_run(runs / identity_of[name]["run_id"]), device="cuda", batch_size=1)
        for name in args.classifiers
    }
    first = args.classifiers[0]
    if any(spec != engines[first][1] for _, spec in engines.values()):
        raise ValueError("The classifiers declare different preprocessing")
    paths = {}
    for file in args.inputs:
        paths.update({r["video_id"]: r["path"] for r in read_json(file)})
    plan = read_json(args.plan)
    rows = [r for r in rows_of(plan) if in_shard(r["window_id"], args.shard, args.shards)]
    identity = {
        "purpose": "class_routing_classifiers",
        "plan_sha256": plan["sha256"],
        "set": args.set,
        "classifiers": identity_of,
        "source_commit": args.source_commit,
    }
    out = args.output / args.set
    out.mkdir(parents=True, exist_ok=True)
    for row in rows:
        stem = out / _key(row)
        if stem.with_suffix(".json").exists():
            if read_json(stem.with_suffix(".json"))["identity"] != identity:
                raise ValueError(f"Existing record with another identity: {stem}")
            continue
        torch.cuda.synchronize()
        start = time.perf_counter()
        clip = prepare_clip(
            engines[first][0],
            WindowItem(window_from_record(row), row.get("path") or paths[row["video_id"]]),
        )
        torch.cuda.synchronize()
        seconds, class_logits = {"decode": time.perf_counter() - start}, {}
        for name, (engine, _) in engines.items():
            scorer = engine.scorer
            with torch.no_grad(), scorer.autocast():
                begin = time.perf_counter()
                feats = encode_multilevel(scorer.backbone, clip, scorer.levels)
                torch.cuda.synchronize()
                middle = time.perf_counter()
                class_logits[name] = scorer.head(feats).float()[0].cpu().tolist()
                torch.cuda.synchronize()
            seconds[f"encode_{name}"] = middle - begin
            seconds[f"head_{name}"] = time.perf_counter() - middle
            if len(class_logits[name]) != NUM_OUTPUTS:
                raise ValueError(f"{name}: {len(class_logits[name])} logits, not {NUM_OUTPUTS}")
            del feats
        write_json(
            stem.with_suffix(".json"),
            {
                "identity": identity,
                "dataset": row["dataset"],
                "video_id": row["video_id"],
                "window_id": row["window_id"],
                "class_name": row.get("class_name"),
                "index": row.get("index"),
                "class_logits": class_logits,
                "seconds": seconds,
                "gpu": torch.cuda.get_device_name(),
                "slurm_job": os.environ.get("SLURM_JOB_ID"),
            },
        )
        del clip
    write_json(
        args.output / "operations" / f"{args.set}-shard{args.shard:02d}.json",
        {"status": "complete", "identity": identity, "chunks": len(rows)},
    )
    print(json.dumps({"set": args.set, "chunks": len(rows)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, required=True, help="a file of localization_class_plan"
    )
    parser.add_argument("--set", required=True, help="name of the output folder (test, val, ...)")
    parser.add_argument("--detector", choices=("campaign", "b3_sum", "b3_gradnorm", "attentive"))
    parser.add_argument("--inputs", type=Path, nargs="*", default=[], help="rows with video paths")
    parser.add_argument("--ig", action="store_true", help="compute the fixed-node IG maps")
    parser.add_argument("--class-heads", nargs="*", default=[], choices=("B1", "B2"))
    parser.add_argument(
        "--classifiers",
        nargs="+",
        choices=("B4L", "B4V"),
        help="class heads on their own trained encoder (no detector, no IG, no SLIC)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    # CPUs of this job, not of the node: on a cluster node os.cpu_count() is the whole machine.
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    parser.add_argument("--workers", type=int, default=max(1, (cpus or 2) - 2))
    parser.add_argument("--source-commit", default="")
    args = parser.parse_args()
    if (args.detector is None) == (args.classifiers is None):
        parser.error("pass exactly one of --detector and --classifiers")
    if args.classifiers and (args.ig or args.class_heads):
        parser.error("--classifiers runs no detector: --ig and --class-heads do not apply")
    if args.classifiers:
        classifiers_pass(args)
        return

    import torch

    from evd.data.dataset import WindowItem
    from evd.exp.demo import live_engine, open_run
    from evd.exp.explanation_advanced import prepare_clip, window_from_record
    from evd.exp.localization_class import IGSettings, encode_with_baselines, integrate
    from evd.exp.localization_vtcd import segment
    from evd.utils.hashing import file_sha256

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no silent CPU fallback")
    torch.manual_seed(42)
    torch.set_num_threads(1)
    config = read_json(CONFIG)
    settings = None
    if args.ig:
        ig = config["ig"]
        if ig["nodes"] is None or ig["batch_size"] is None:
            raise ValueError("IG nodes and batch are not set: run and accept the pilot first")
        settings = IGSettings(
            nodes=int(ig["nodes"]),
            batch_size=int(ig["batch_size"]),
            precision=ig["precision"],
            fills=tuple(ig["fills"]),
            atol=ig["tolerance"]["atol"],
            rtol=ig["tolerance"]["rtol"],
        )
    runs = Path(os.environ["EVD_RUNS_DIR"])
    run_id = config["detectors"][args.detector]
    run = open_run(runs / run_id)
    if args.detector == "campaign":
        from localization_auto_preflight import CHECKPOINT_SHA, RUN

        if run_id != RUN or file_sha256(run.checkpoint_path) != CHECKPOINT_SHA:
            raise ValueError("Campaign detector identity mismatch")
    engine, spec = live_engine(run, device="cuda", batch_size=1, num_workers=0)
    scorer = engine.scorer
    heads = {}
    if args.class_heads:
        if args.detector != "campaign":
            raise ValueError("B1 and B2 run on the campaign detector only")
        detector = torch.load(run.checkpoint_path, map_location="cpu", weights_only=True)
        for name in args.class_heads:
            head, adapter, step = _class_head(
                runs / config["class_heads"][name], scorer.backbone.embed_dim, "cuda"
            )
            reference = dict(detector.get("selected_adapter") or {})
            if set(adapter) != set(reference) or any(
                not torch.equal(adapter[k], reference[k]) for k in adapter
            ):
                raise ValueError(f"{name} was not trained on the frozen detector encoder")
            heads[name] = (head, step)
    own_classes = callable(getattr(scorer.head, "forward_tasks", None))
    paths = {}
    for file in args.inputs:
        paths.update({r["video_id"]: r["path"] for r in read_json(file)})
    plan = read_json(args.plan)
    rows = [r for r in rows_of(plan) if in_shard(r["window_id"], args.shard, args.shards)]
    identity = {
        "purpose": "class_routing_pass",
        "plan_sha256": plan["sha256"],
        "set": args.set,
        "detector": run_id,
        "checkpoint_sha256": file_sha256(run.checkpoint_path),
        "ig": None if settings is None else settings.__dict__ | {"fills": list(settings.fills)},
        "class_heads": {k: config["class_heads"][k] for k in args.class_heads},
        "source_commit": args.source_commit,
    }
    out = args.output / args.set
    pool = ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    )

    def finish(row: dict, encoded: dict, segmented) -> None:
        labels, descriptors, counts, slic_s = segmented
        stem = out / _key(row)
        stem.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "labels": labels.astype(np.uint8),
            "descriptors": descriptors.astype(np.float32),
            "counts": counts.astype(np.int32),
            **encoded.pop("maps"),
        }
        with stem.with_suffix(".tmp").open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        stem.with_suffix(".tmp").replace(stem.with_suffix(".npz"))
        encoded["seconds"]["slic_cpu"] = slic_s
        write_json(stem.with_suffix(".json"), {"identity": identity, **encoded})

    pending = []
    for row in rows:
        stem = out / _key(row)
        if stem.with_suffix(".json").exists():
            if read_json(stem.with_suffix(".json"))["identity"] != identity:
                raise ValueError(f"Existing record with another identity: {stem}")
            continue
        torch.cuda.synchronize()
        start = time.perf_counter()
        clip = prepare_clip(
            engine, WindowItem(window_from_record(row), row.get("path") or paths[row["video_id"]])
        )
        torch.cuda.synchronize()
        decode = time.perf_counter() - start
        feats, logit, bases, seconds = encode_with_baselines(
            scorer, spec, clip, row["window_id"], () if settings is None else settings.fills
        )
        maps, receipts = {}, {}
        if settings is not None:
            ig_maps, receipts, seconds["integral"] = integrate(scorer, feats, bases, settings)
            maps = {"ig_ensemble": ig_maps["ig_ensemble"].astype(np.float32)}
            maps.update({k: v.astype(np.float16) for k, v in ig_maps.items() if k != "ig_ensemble"})
        class_logits = {}
        with torch.no_grad(), scorer.autocast():
            for name, (head, _) in heads.items():
                begin = time.perf_counter()
                class_logits[name] = head(feats).float()[0].cpu().tolist()
                torch.cuda.synchronize()
                seconds[f"head_{name}"] = time.perf_counter() - begin
            if own_classes:
                begin = time.perf_counter()
                class_logits["own"] = (
                    scorer.head.forward_tasks(feats)["classes"].float()[0].cpu().tolist()
                )
                torch.cuda.synchronize()
                seconds["head_own"] = time.perf_counter() - begin
        tokens = feats.tokens[0].float().cpu().numpy().reshape(*feats.grid, -1)
        encoded = {
            "dataset": row["dataset"],
            "video_id": row["video_id"],
            "window_id": row["window_id"],
            "class_name": row.get("class_name"),
            "index": row.get("index"),
            "grid": list(feats.grid),
            "logit": logit,
            "class_logits": class_logits,
            "ig_receipts": receipts,
            "seconds": {"decode": decode, **seconds},
            "maps": maps,
            "gpu": torch.cuda.get_device_name(),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
        }
        pending.append((row, encoded, pool.submit(segment, tokens)))
        del clip, feats, bases, tokens
        while len(pending) > 2 * args.workers or (pending and pending[0][2].done()):
            first, record, future = pending.pop(0)
            finish(first, record, future.result())
    for row, record, future in pending:
        finish(row, record, future.result())
    pool.shutdown()
    write_json(
        args.output / "operations" / f"{args.set}-shard{args.shard:02d}.json",
        {"status": "complete", "identity": identity, "chunks": len(rows)},
    )
    print(json.dumps({"set": args.set, "chunks": len(rows)}))


if __name__ == "__main__":
    main()
