"""Recompute the class-routing IG at other node counts over the records of a pass (GPU).

Written on 28 October 2026 to measure the localization metrics while the Gauss-Legendre nodes
of IG go from 16 to 128, and decide whether to lower the campaign's 256. For every record of a
detector pass set (``test``, ``real``) the chunk is decoded and encoded again and the IG
ensemble is integrated at each node count of the queue; every other field of the record (the
binary and class logits, the 32 SLIC regions with their descriptors and token counts) is
copied from it, so between the stages built on the outputs only the integral changes. The IG
settings other than the nodes (batch, precision, baselines, tolerances) are read from the
record's identity, not from the config file.

- ``queue``: write ``<root>/queue.json``, the tasks (detector, set, shard) in priority order,
  each with its node counts and its source and output pass roots; one output pass root per
  node count, ``<root>/n<N>/<detector>/``, that ``localization_class_ap.py detections`` reads
  as the campaign's (the sets without IG are linked, not copied).
- ``work``: drain the queue on one GPU. A task is claimed by making its directory under
  ``<root>/claims`` (atomic) and writing an owner token in it; the token is refreshed after
  every chunk, and a claim older than :data:`STALE_SECONDS` is taken over by replacing the
  token and reading it back after :data:`TAKEOVER_WAIT_SECONDS`, so that of two jobs taking
  over the same claim only the last writer keeps it (renaming the stale directory, the first
  version, let two jobs hold one task: 29 October). A holder checks its token before every
  chunk and leaves a task it lost. Any number of jobs on H100 or A100 share the work, and a
  job killed at its time limit loses at most its current chunk (the outputs resume per chunk;
  every file is written under a temporary name of its own writer). A task done writes
  ``<root>/done/<task>.json``.

The recomputed binary logit is compared with the record's (``logit_drift``); no spatial ground
truth is read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402

from evd.exp.localization_auto import in_shard, read_json, write_json  # noqa: E402

#: Seconds after which a claim not refreshed is taken over (a chunk takes about 20 s on H100
#: for 16+32+64+128 nodes, 35 s with 256 too).
STALE_SECONDS = 15 * 60
#: Seconds between writing a takeover token and reading it back.
TAKEOVER_WAIT_SECONDS = 10
#: Pass sets without IG that the stages read beside ``test`` and ``real``.
LINKED_SETS = ("test_heads", "test_classifiers", "real_classifiers")


def task_name(task: dict) -> str:
    return f"{task['detector']}-{task['set']}-n{'_'.join(map(str, task['nodes']))}-s{task['shard']:02d}"


def build_queue(args) -> None:
    """Write the task list and the output pass roots, with the sets without IG linked."""
    tasks = []
    for item in args.tasks:
        detector, plan, nodes, shards = item.split(":")
        node_list = [int(n) for n in nodes.split(",")]
        source = args.passes / detector
        for name in ("test", "real"):
            plan_file = args.plan_root / (
                "test.json" if name == "test" else f"real_use/{plan}.json"
            )
            for shard in range(int(shards)):
                tasks.append(
                    {
                        "detector": detector,
                        "set": name,
                        "plan": str(plan_file),
                        "source": str(source),
                        "nodes": node_list,
                        "shard": shard,
                        "shards": int(shards),
                    }
                )
        for n in node_list:
            target = args.root / f"n{n}" / detector
            target.mkdir(parents=True, exist_ok=True)
            for linked in LINKED_SETS:
                if (source / linked).is_dir() and not (target / linked).exists():
                    (target / linked).symlink_to(source / linked, target_is_directory=True)
    for name in ("claims", "done"):
        (args.root / name).mkdir(parents=True, exist_ok=True)
    write_json(args.root / "queue.json", {"tasks": tasks, "inputs": [str(p) for p in args.inputs]})
    print(json.dumps({"tasks": len(tasks)}))


def _publish(path: Path, write, token: str) -> None:
    """Write ``path`` through a temporary file of this writer, then rename it into place."""
    temporary = path.with_name(f"{path.name}.{token}.tmp")
    write(temporary)
    temporary.replace(path)


def owner(claimed: Path) -> str | None:
    """:return: the token holding a claim directory, ``None`` if it has none."""
    try:
        return (claimed / "owner").read_text(encoding="utf-8")
    except OSError:
        return None


def claim(root: Path, name: str, wait: float = TAKEOVER_WAIT_SECONDS) -> tuple[Path, str] | None:
    """Claim a task of the queue, or take over a stale claim.

    :param wait: seconds between writing a takeover token and reading it back.
    :return: ``(claim directory, owner token)``, or ``None`` if the task is done or held by
        a live job (or by another job that took it over at the same time).
    """
    if (root / "done" / f"{name}.json").exists():
        return None
    path = root / "claims" / name
    token = uuid.uuid4().hex

    def write(target: Path) -> None:
        target.write_text(token, encoding="utf-8")

    try:
        path.mkdir()
    except FileExistsError:
        token_file = path / "owner"
        try:
            beat = token_file.stat().st_mtime if token_file.exists() else path.stat().st_mtime
        except OSError:
            return None
        if time.time() - beat < STALE_SECONDS:
            return None
        _publish(token_file, write, token)
        time.sleep(wait)
        return (path, token) if owner(path) == token else None
    _publish(path / "owner", write, token)
    return path, token


def work(args) -> None:
    import torch

    from evd.data.dataset import WindowItem
    from evd.exp.demo import live_engine, open_run
    from evd.exp.explanation_advanced import prepare_clip, window_from_record
    from evd.exp.localization_class import IGSettings, encode_with_baselines, integrate
    from evd.utils.hashing import file_sha256

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no silent CPU fallback")
    torch.manual_seed(42)
    torch.set_num_threads(1)
    queue = read_json(args.root / "queue.json")
    paths = {}
    for file in queue["inputs"]:
        paths.update({r["video_id"]: r["path"] for r in read_json(Path(file))})
    runs = Path(os.environ["EVD_RUNS_DIR"])
    loaded: dict[str, tuple] = {}
    finished = 0
    for task in queue["tasks"]:
        name = task_name(task)
        claimed = claim(args.root, name)
        if claimed is None:
            continue
        held, token = claimed
        lost = False
        source = Path(task["source"]) / task["set"]
        records = sorted(source.glob("*.json"))
        first = read_json(records[0])["identity"]
        run_id = first["detector"]
        if run_id not in loaded:
            loaded.clear()
            torch.cuda.empty_cache()
            run = open_run(runs / run_id)
            if file_sha256(run.checkpoint_path) != first["checkpoint_sha256"]:
                raise ValueError(f"{run_id}: checkpoint differs from the pass")
            engine, spec = live_engine(run, device="cuda", batch_size=1, num_workers=0)
            loaded[run_id] = (engine, spec)
        engine, spec = loaded[run_id]
        rows = {r["window_id"]: r for r in read_json(Path(task["plan"]))["rows"]}
        drift, chunks = [], 0
        for path in records:
            record = read_json(path)
            if not in_shard(record["window_id"], task["shard"], task["shards"]):
                continue
            if owner(held) != token:
                lost = True
                break
            identity = record["identity"]
            base = IGSettings(
                **{k: v for k, v in identity["ig"].items() if k != "fills"},
                fills=tuple(identity["ig"]["fills"]),
            )
            targets = {}
            for n in task["nodes"]:
                out = args.root / f"n{n}" / task["detector"] / task["set"] / path.name
                if not out.exists():
                    targets[n] = out
            if not targets:
                continue
            row = rows[record["window_id"]]
            torch.cuda.synchronize()
            start = time.perf_counter()
            clip = prepare_clip(
                engine,
                WindowItem(window_from_record(row), row.get("path") or paths[row["video_id"]]),
            )
            torch.cuda.synchronize()
            decode = time.perf_counter() - start
            feats, logit, bases, seconds = encode_with_baselines(
                engine.scorer, spec, clip, record["window_id"], base.fills
            )
            drift.append(abs(float(logit) - float(record["logit"])))
            arrays = np.load(path.with_suffix(".npz"))
            kept = {k: arrays[k] for k in ("labels", "descriptors", "counts")}
            for n, out in targets.items():
                settings = replace(base, nodes=n)
                ig_maps, receipts, integral = integrate(engine.scorer, feats, bases, settings)
                maps = {"ig_ensemble": ig_maps["ig_ensemble"].astype(np.float32)}
                maps.update(
                    {k: v.astype(np.float16) for k, v in ig_maps.items() if k != "ig_ensemble"}
                )
                out.parent.mkdir(parents=True, exist_ok=True)

                def save(target: Path, arrays=kept | maps) -> None:
                    with target.open("wb") as stream:
                        np.savez_compressed(stream, **arrays)

                _publish(out.with_suffix(".npz"), save, token)
                ig = settings.__dict__ | {"fills": list(settings.fills)}
                document = json.dumps(
                    {
                        **record,
                        "identity": {**identity, "ig": ig},
                        "ig_receipts": receipts,
                        "seconds": {
                            **record["seconds"],
                            "decode": decode,
                            **seconds,
                            "integral": integral,
                        },
                        "ig_nodes_source": {
                            "record": str(path),
                            "logit_recomputed": float(logit),
                            "gpu": torch.cuda.get_device_name(),
                            "slurm_job": os.environ.get("SLURM_JOB_ID"),
                        },
                    },
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )

                def text(target: Path, document=document) -> None:
                    target.write_text(document + "\n", encoding="utf-8")

                _publish(out, text, token)
            del clip, feats, bases
            chunks += 1
            os.utime(held / "owner")
        if lost:
            print(json.dumps({"lost": name, "chunks": chunks}), flush=True)
            continue
        write_json(
            args.root / "done" / f"{name}.json",
            {
                "task": task,
                "chunks_computed": chunks,
                "max_logit_drift": max(drift, default=0.0),
                "gpu": torch.cuda.get_device_name(),
                "slurm_job": os.environ.get("SLURM_JOB_ID"),
            },
        )
        finished += 1
        print(json.dumps({"done": name, "chunks": chunks}), flush=True)
    print(json.dumps({"tasks_finished": finished}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("queue", "work"))
    parser.add_argument("--root", type=Path, required=True, help="output root of the node sweep")
    parser.add_argument("--passes", type=Path, help="queue: the campaign's passes/ folder")
    parser.add_argument("--plan-root", type=Path, help="queue: the campaign's plan/ folder")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[],
        help="queue: DETECTOR:REAL_USE_PLAN:N1,N2,...:SHARDS in priority order",
    )
    parser.add_argument("--inputs", type=Path, nargs="*", default=[], help="rows with video paths")
    args = parser.parse_args()
    if args.step == "queue":
        if args.passes is None or args.plan_root is None or not args.tasks:
            parser.error("queue needs --passes, --plan-root and --tasks")
        build_queue(args)
    else:
        work(args)


if __name__ == "__main__":
    main()
