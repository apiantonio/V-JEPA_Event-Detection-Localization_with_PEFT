"""Localization movies of chosen methods in the demonstrator layout, and their catalogue.

Each movie is one method on one annotated video: the demonstrator payload of the run, in
replay (no model), with the saved boxes of the method (``apps/demo/localization_video.py``).
The steps can be re-run, and a finished movie is not redone:

- ``plan`` (where the exports are): one task per method and video of the export, with the
  paths the render machine will read. A dataset the run has no scores for is skipped and
  the plan says so;
- ``render`` (a CPU job, sharded): writes ``<target>/movie.mp4`` and ``manifest.json``, after
  decoding the encoded movie whole;
- ``catalog``: ``catalog.json`` and the page ``index.html`` served through the SSH tunnel;
- ``verify``: every task complete, every file matching its manifest.

    python scripts/render_localization_videos.py plan --spec <spec.json> --out plan.json
    python scripts/render_localization_videos.py render --plan plan.json --output <root> \\
        --shard 0 --shards 4 --workers 4
    python scripts/render_localization_videos.py catalog --plan plan.json --output <root>
    python scripts/render_localization_videos.py verify --plan plan.json --output <root>

The spec names the local runs and exports (``plan`` reads them) and the same folders on
the render machine (``render`` reads those): ``remote_runs``, ``remote_exports``, video paths
by id, annotation folders, and per method ``key``, ``run``, ``method``, ``selection``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

#: Readable dataset names, as the header writes them.
DATASET_LABELS = {"ucf_crime": "FS-UCF-Crime test", "mivia_crime": "MIVIA-Crime test"}

#: Movie file of a task, inside its target folder.
MOVIE = "movie.mp4"

#: Heads trained with the encoder in the class-routing campaign (B3), by run prefix: the
#: model name says how, since the acronym does not.
JOINT_HEADS = {"lcb3s_": "summed losses", "lcb3g_": "GradNorm"}


def model_name(run_id: str) -> str:
    """:return: the readable name of a detector run, the multiclass training included."""
    from apps.demo.explanation import model_label

    name = model_label(run_id)
    for prefix, balance in JOINT_HEADS.items():
        if run_id.startswith(prefix):
            return f"{name}, with a multiclass head trained jointly ({balance})"
    return name


def read(path: Path) -> Any:
    """:return: the decoded UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    """:return: the SHA256 of a file, read in bounded blocks."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def plan(spec: dict[str, Any]) -> dict[str, Any]:
    """Build the render tasks of every method of a spec.

    :param spec: local ``runs`` folder, ``remote_runs`` and ``remote_exports`` folders on the
        render machine, ``videos`` (video id to path there), ``annotations`` (delivery
        folders there), and ``methods``: ``key``, ``run`` (run id with a local export),
        ``method`` (exported id), ``selection`` (why it was chosen).
    :return: the plan: methods with their names, IoU and evaluated IoU per event id, tasks,
        the evaluated events of every video with their ids (``events``, for the
        catalogue, skipped videos included), and skipped datasets. The event ids and IoU
        stay out of the tasks, whose digest is the identity of a finished movie.
    :raises ValueError: on a method the export does not have, or a video without a path.
    """
    from apps.demo.localization import FOLDER

    runs, tasks, methods, skipped = Path(spec["runs"]), [], [], []
    video_events: dict[str, list[dict[str, Any]]] = {}
    for item in spec["methods"]:
        run_id = item["run"]
        index = read(runs / run_id / FOLDER / "index.json")
        entry = next((m for m in index["methods"] if m["id"] == item["method"]), None)
        if entry is None:
            raise ValueError(f"{item['method']} is not exported for {run_id}")
        label = entry.get("label") or entry["id"]
        methods.append(
            {
                "key": item["key"],
                "method": entry["id"],
                "label": label,
                "model": model_name(run_id),
                "run_id": run_id,
                "selection": item["selection"],
                "campaign": entry.get("campaign", index["campaign"]),
                "iou": {d: entry[d]["mean"] for d in DATASET_LABELS},
                "event_iou": {
                    event: iou
                    for d in DATASET_LABELS
                    for event, iou in entry[d].get("events", {}).items()
                },
            }
        )
        nested = f"{run_id}__mivia"
        for video_id, relative in sorted(index["videos"].items()):
            dataset = relative.split("/", 1)[0]
            truth = index.get("ground_truth", {}).get("videos", {}).get(video_id)
            events = read(runs / run_id / FOLDER / truth).get("events", []) if truth else []
            video_events.setdefault(
                f"{dataset}/{video_id}",
                [{k: e[k] for k in ("id", "class", "start", "end")} for e in events],
            )
            if dataset == "mivia_crime" and not (runs / run_id / nested / "scores.jsonl").is_file():
                skipped.append({"key": item["key"], "dataset": dataset, "video_id": video_id})
                continue
            if video_id not in spec["videos"]:
                raise ValueError(f"No path on the render machine for {video_id}")
            run_dir = f"{spec['remote_runs']}/{run_id}"
            if dataset == "mivia_crime":
                run_dir += f"/{nested}"
            fps = read(runs / run_id / FOLDER / relative)["video"]["fps"]
            tasks.append(
                {
                    "key": item["key"],
                    "method": entry["id"],
                    "method_label": label,
                    "model_label": model_name(run_id),
                    "run_dir": run_dir,
                    "export": f"{spec['remote_exports']}/{run_id}",
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS[dataset],
                    "video_id": video_id,
                    "video": spec["videos"][video_id],
                    "annotations": spec["annotations"],
                    "target": f"videos/{dataset}/{video_id}/{item['key']}",
                    "fps": fps,
                    "events": [{k: e[k] for k in ("class", "start", "end")} for e in events],
                }
            )
    reasons = {
        s["key"]: "the run has no MIVIA-Crime scores (no external evaluation)" for s in skipped
    }
    return {
        "name": spec["name"],
        "methods": methods,
        "tasks": tasks,
        "events": video_events,
        "skipped": skipped,
        "skipped_reason": reasons,
    }


def identity(task: dict[str, Any]) -> str:
    """:return: the digest of a task, which a finished movie must carry to be reused."""
    return hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()


def render(task: dict[str, Any], output: Path) -> dict[str, Any]:
    """Render one movie, decode it whole, then publish it with its manifest.

    :param task: one plan task.
    :param output: collection root; the movie goes to ``output / task["target"]``.
    :return: the manifest, also written next to the movie.
    :raises ValueError: if the run cannot replay the video, the boxes are missing, or the
        encoded movie fails validation.
    """
    import av
    from apps.demo import localization_video as movie_mod
    from apps.demo.run_demo import payload_for

    from evd.eval.demo_metrics import saved_labels
    from evd.exp.demo import analyze, open_run

    target = output / task["target"]
    target.mkdir(parents=True, exist_ok=True)
    key = identity(task)
    manifest_path = target / "manifest.json"
    if manifest_path.exists():
        previous = read(manifest_path)
        if previous.get("identity") != key:
            raise ValueError(f"Existing movie has different inputs: {target}")
        if previous.get("status") == "complete" and digest(target / MOVIE) == previous["sha256"]:
            return previous
    run = open_run(task["run_dir"])
    result, mode = analyze(run, task["video"], modo="replay")
    data = payload_for(
        run,
        result,
        mode,
        video_src=MOVIE,
        annotations=[Path(p) for p in task["annotations"]],
        export=Path(task["export"]),
    )
    ordered = sorted(
        result.window_scores, key=lambda w: (w.window.end_frame, w.window.frame_span[0])
    )
    saved = saved_labels(run.score_path, data.video_id)
    labels = [saved[w.window.window_id] for w in ordered] if saved else None
    movie = movie_mod.build_movie(
        data,
        [w.window for w in ordered],
        method=task["method"],
        method_label=task["method_label"],
        model_label=task["model_label"],
        dataset_label=task["dataset_label"],
        labels=labels,
    )
    pending = target / f".{MOVIE}.pending.mp4"
    counts = movie_mod.write(movie, task["video"], pending)
    with av.open(str(pending)) as container:
        stream = container.streams.video[0]
        context = stream.codec_context
        decoded = sum(1 for _ in container.decode(video=0))
        geometry = (context.width, context.height)
        valid = (
            decoded == counts["frames"]
            and context.name == "h264"
            and context.pix_fmt == "yuv420p"
            and geometry == (movie_mod.CANVAS_W, movie_mod.CANVAS_H)
        )
    if not valid:
        raise ValueError(f"Encoded movie failed validation: {pending}")
    os.replace(pending, target / MOVIE)
    manifest = {
        "status": "complete",
        "identity": key,
        "task": task,
        "file": MOVIE,
        "sha256": digest(target / MOVIE),
        "codec": "h264",
        "pixel_format": "yuv420p",
        "size": list(geometry),
        **counts,
        "windows": len(ordered),
        "threshold": data.threshold,
        "video_iou": movie.video_iou,
        "events": [e.__dict__ for e in movie.events],
        "renderer": "apps.demo.localization_video over apps.demo.run_demo.payload_for (replay)",
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def run_shard(args: argparse.Namespace) -> None:
    """Render one shard of the plan, in parallel workers, printing one line per movie."""
    tasks = read(args.plan)["tasks"][args.shard :: args.shards]
    if args.limit:
        tasks = tasks[: args.limit]
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(render, task, args.output): task for task in tasks}
        for future in as_completed(futures):
            done += 1
            task = futures[future]
            try:
                status = future.result()["status"]
            except Exception as exc:  # one broken video must not stop the shard
                status = f"failed: {type(exc).__name__}: {exc}"
            print(json.dumps({"task": task["target"], "status": status, "done": done}), flush=True)
    print(f"shard {args.shard}/{args.shards}: {done} tasks", flush=True)


def catalog(plan_data: dict[str, Any], output: Path) -> dict[str, Any]:
    """Write ``catalog.json`` and the page that lists, filters, sorts and plays the movies.

    Every event carries the evaluated IoU of each method (``iou``, method key to value,
    ``None`` where the method has none), so the page can sort events and videos by it,
    also for a method whose movie of that video is not rendered.

    :param plan_data: the plan.
    :param output: collection root.
    :return: the catalogue.
    """
    by_event = {m["key"]: m.get("event_iou", {}) for m in plan_data["methods"]}
    videos: dict[str, dict[str, Any]] = {}
    for task in plan_data["tasks"]:
        key = f"{task['dataset']}/{task['video_id']}"
        if key not in videos:
            events = [
                {**e, "iou": {m: values.get(e["id"]) for m, values in by_event.items()}}
                for e in plan_data.get("events", {}).get(key, [])
            ] or task["events"]
            videos[key] = {
                "id": key,
                "dataset": task["dataset"],
                "dataset_label": task["dataset_label"],
                "video_id": task["video_id"],
                "fps": task["fps"],
                "classes": sorted({e["class"] for e in task["events"]}),
                "events": events,
                "movies": {},
            }
        videos[key]["movies"][task["key"]] = task["target"]
    body = {
        "name": plan_data["name"],
        "methods": [{k: v for k, v in m.items() if k != "event_iou"} for m in plan_data["methods"]],
        "skipped": plan_data["skipped"],
        "skipped_reason": plan_data["skipped_reason"],
        "videos": sorted(
            videos.values(), key=lambda v: (v["dataset"] != "ucf_crime", v["video_id"])
        ),
    }
    (output / "catalog.json").write_text(json.dumps(body, indent=1), encoding="utf-8")
    shutil.copyfile(
        ROOT / "apps" / "demo" / "assets" / "localization_videos.html", output / "index.html"
    )
    return body


def verify(plan_data: dict[str, Any], output: Path) -> dict[str, Any]:
    """Check that every task has a complete movie matching its manifest and its plan entry.

    :return: the verification, also written to ``final-verification.json``.
    """
    failures, frames = [], 0
    for task in plan_data["tasks"]:
        target = output / task["target"]
        try:
            manifest = read(target / "manifest.json")
            if manifest["status"] != "complete" or manifest["identity"] != identity(task):
                raise ValueError("incomplete or rendered from other inputs")
            if digest(target / MOVIE) != manifest["sha256"]:
                raise ValueError("movie differs from its manifest")
            frames += int(manifest["frames"])
        except (OSError, ValueError, KeyError) as exc:
            failures.append({"target": task["target"], "error": str(exc)})
    result = {
        "status": "complete" if not failures else "incomplete",
        "tasks": len(plan_data["tasks"]),
        "complete": len(plan_data["tasks"]) - len(failures),
        "frames": frames,
        "failures": failures,
        "skipped": plan_data["skipped"],
    }
    (output / "final-verification.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def main() -> None:
    """Parse the step and run it."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    steps = parser.add_subparsers(dest="step", required=True)
    p = steps.add_parser("plan")
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    r = steps.add_parser("render")
    r.add_argument("--plan", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--shards", type=int, default=1)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--limit", type=int)
    for name in ("catalog", "verify"):
        c = steps.add_parser(name)
        c.add_argument("--plan", type=Path, required=True)
        c.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.step == "plan":
        data = plan(read(args.spec))
        args.out.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{len(data['tasks'])} tasks, {len(data['skipped'])} skipped")
    elif args.step == "render":
        if not 0 <= args.shard < args.shards:
            parser.error("invalid shard index")
        run_shard(args)
    elif args.step == "catalog":
        body = catalog(read(args.plan), args.output)
        print(f"catalogue: {len(body['videos'])} videos")
    else:
        result = verify(read(args.plan), args.output)
        print(json.dumps({k: v for k, v in result.items() if k != "skipped"}))


if __name__ == "__main__":
    main()
