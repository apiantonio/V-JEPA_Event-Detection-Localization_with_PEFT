"""Screen recordings of the demonstrator page, one per test video, and their catalogue.

Each recording is the page the local console opens in replay (``apps/demo/assets/viewer.html``
filled by :func:`apps.demo.run_demo.payload_for`), shown in a headless Chromium at the size of
a full-screen window and stepped frame by frame: seek the video, let the page draw, take a
screenshot, encode. Every frame is therefore what the page shows at that instant, with the
chosen toggles (chunk outcome, interpolated planes, YOLO style, GT and method boxes). No model
runs: the scores come from the run (ADR-015).

Steps, re-runnable, a finished video is not redone:

- ``plan`` (on the render machine, project environment): the test videos of a run and of its
  external evaluations, the video files, the library numbers of each video (for sorting the
  catalogue) and the recording settings;
- ``render`` (a CPU job, sharded by duration): per video a VP9 copy of the source (the
  Chromium of Playwright has no H.264 decoder), the page, the recording in segments of
  ``segment_s`` seconds in parallel workers, and the assembled ``<video_id>.mp4`` with its
  manifest, after counting its frames;
- ``catalog``: ``catalog.json`` and the page ``index.html`` (``apps/demo/assets/
  demo_recordings.html``), served through the SSH tunnel;
- ``verify``: every video complete and matching its manifest.

    python scripts/record_demonstrator.py plan --run <runs>/<run_id> --videos <dir> \\
        --annotations <dir> [--annotations <dir>] --out plan.json
    python scripts/record_demonstrator.py render --plan plan.json --output <root> \\
        --shard 0 --shards 3 --workers 48
    python scripts/record_demonstrator.py catalog --plan plan.json --output <root>
    python scripts/record_demonstrator.py verify --plan plan.json --output <root>

``render`` needs Playwright with its Chromium (``PLAYWRIGHT_BROWSERS_PATH``); decoding and
encoding go through PyAV (libvpx-vp9 for the browser copy, libx264 for the recording), so no
``ffmpeg`` executable is needed on the compute nodes.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from multiprocessing import get_context
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

#: Readable dataset names.
DATASET_LABELS = {"ucf_crime": "FS-UCF-Crime test", "mivia_crime": "MIVIA-Crime test"}

#: Recording settings: a full-screen 1080p window, the toggles asked for the presentation clips.
SETTINGS = {
    "width": 1920,
    "height": 1080,
    "max_fps": 30.0,
    "segment_s": 60.0,
    "jpeg_quality": 92,
    "crf": 18,
    "preset": "veryfast",
    "toggles": {
        "outcomes": True,
        "interpolate": True,
        "yolo": True,
        "truth_boxes": True,
        "method_boxes": True,
    },
}

#: Largest source size kept for the browser copy: the stage of a 1080p window is smaller.
SOURCE_MAX = (1920, 1080)

#: The pause icon of the page: the recording shows a video that is playing.
PAUSE_ICON = '<svg viewBox="0 0 16 16"><path d="M3.5 2h3.2v12H3.5zM9.3 2h3.2v12H9.3z"/></svg>'

SETUP_JS = """(opts) => {
  const $ = (id) => document.getElementById(id);
  const v = $("v"); v.muted = true; v.pause();
  const press = (id, want) => {
    const b = $(id);
    if (!b || b.disabled) return null;
    if ((b.getAttribute("aria-pressed") === "true") !== want) b.click();
    return b.getAttribute("aria-pressed") === "true";
  };
  const check = (id, want) => {
    const c = $(id);
    if (!c || $("loc").hidden) return null;
    if (c.checked !== want) c.click();
    return c.checked;
  };
  const state = {
    outcomes: press("toggle-outcomes", opts.toggles.outcomes),
    truth_boxes: press("toggle-truth-boxes", opts.toggles.truth_boxes),
    method_boxes: press("toggle-method-boxes", opts.toggles.method_boxes),
    interpolate: check("loc-interpolate", opts.toggles.interpolate),
    yolo: check("loc-yolo", opts.toggles.yolo),
  };
  $("btn-play").innerHTML = opts.pause_icon;
  const m = $("loc").hidden ? null : $("loc-method");
  // A menu entry other than the default, chosen as the user would: value, then `change`.
  if (m && opts.method && m.value !== opts.method) {
    m.value = opts.method;
    m.dispatchEvent(new Event("change"));
  }
  state.method = m ? m.value : null;
  state.method_missing = !!(m && opts.method && m.value !== opts.method);
  state.method_label = m ? m.title : null;
  state.media_error = !$("media-error").hidden;
  state.video_size = [v.videoWidth, v.videoHeight];
  state.duration = v.duration;
  return state;
}"""

# Seek, wait for the page listeners (paint on `seeked`) and for the new frame to be presented,
# put CSS animations at the video time (transitions finished, the alarm blink in phase), then two
# animation frames so that layout and canvases are on screen before the screenshot.
SEEK_JS = """async (t) => {
  const v = document.getElementById("v");
  if (Math.abs(v.currentTime - t) > 1e-6) {
    const frame = new Promise((r) => v.requestVideoFrameCallback ? v.requestVideoFrameCallback(() => r(true)) : r(true));
    const seeked = new Promise((r) => v.addEventListener("seeked", r, {once: true}));
    v.currentTime = t;
    await seeked;
    await Promise.race([frame, new Promise((r) => setTimeout(() => r(false), 120))]);
  }
  for (const a of document.getAnimations()) {
    if (typeof CSSTransition !== "undefined" && a instanceof CSSTransition) a.finish();
    else { a.pause(); a.currentTime = t * 1000; }
  }
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  return v.currentTime;
}"""


def read(path: Path) -> Any:
    """:return: the decoded UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path: Path, data: Any) -> None:
    """Write JSON through a temporary file, so a reader never sees a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending")
    pending.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(pending, path)


def digest(path: Path) -> str:
    """:return: the SHA256 of a file, read in bounded blocks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def identity(task: dict[str, Any], settings: dict[str, Any]) -> str:
    """:return: the digest of the inputs a finished recording must carry to be reused."""
    keys = ("video_id", "dataset", "source_dir", "video", "annotations")
    page = ROOT / "apps" / "demo" / "assets" / "viewer.html"
    data = {k: task[k] for k in keys} | {"settings": settings, "viewer": digest(page)}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------------------------------- plan


def plan(args: argparse.Namespace) -> dict[str, Any]:
    """Build the recording tasks: the evaluated test videos of the run and the videos with boxes
    of its external evaluations.

    :return: the plan, with ``tasks`` (one per video), ``entries`` (library numbers, outside the
        identity) and ``settings``.
    :raises ValueError: on a planned video without a file.
    """
    from apps.demo import localization
    from apps.demo.serve import Console

    from evd.exp.demo import open_run
    from evd.infer.replay import scores_from_disk

    run_dir = Path(args.run).resolve()
    console = Console(
        run_dir.parent, [Path(v) for v in args.videos], [Path(a) for a in args.annotations]
    )
    run = open_run(run_dir)
    boxes = localization.exported_videos(run.dir, run.metrics)
    split = run.metrics.get("eval_split", "test")
    sources = [(run, "ucf_crime", lambda key, saved: saved.split == split)]
    for other in console.external_runs(run):
        dataset = str(other.cfg.get("data", {}).get("name") or other.run_id)
        # External test sets: only the videos with spatial annotation (the localization export).
        sources.append((other, dataset, lambda key, saved: key in boxes))
    tasks, entries, missing = [], {}, []
    for source, dataset, wanted in sources:
        saved = scores_from_disk(source.score_path)
        keys = sorted(k for k, v in saved.items() if wanted(k, v))
        numbers = console.library_entries(source)
        for key in keys:
            if args.only_boxes and key not in boxes:
                continue
            path = console.library.get(key)
            if path is None:
                missing.append(key)
                continue
            duration, fps = console.video_timing(key)
            tasks.append(
                {
                    "video_id": key,
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS.get(dataset, dataset),
                    "run_id": run.run_id,
                    "source_dir": str(source.dir),
                    "video": str(path),
                    "annotations": [str(a) for a in console.annotations],
                    "duration_s": duration,
                    "fps": fps,
                    "boxes": key in boxes,
                    "normal": key.lower().startswith("normal_video"),
                }
            )
            entries[key] = numbers.get(key, {})
    if missing:
        raise ValueError(f"{len(missing)} planned videos have no file: {missing[:5]}")
    from apps.demo.explanation import model_label

    settings, variant = dict(SETTINGS), None
    if args.method:
        # A menu entry of the export, checked here so that a typo fails before any job.
        index = json.loads(
            (
                localization.export_dir(run.dir, run.metrics) / localization.FOLDER / "index.json"
            ).read_text(encoding="utf-8")
        )
        entry = next((m for m in index["methods"] if m["id"] == args.method), None)
        if entry is None:
            raise ValueError(f"{args.method} is not a localization method of {run.run_id}")
        settings["method"] = args.method
        variant = entry.get("label") or args.method
    return {
        "name": f"Demonstrator recordings of {run.run_id}" + (f", {variant}" if variant else ""),
        "run_id": run.run_id,
        "model": model_label(run.run_id),
        "variant": variant,
        "threshold": run.threshold,
        "settings": settings,
        "tasks": tasks,
        "entries": entries,
    }


# ------------------------------------------------------------------------------------- render


def shard_of(tasks: list[dict[str, Any]], shard: int, shards: int) -> list[dict[str, Any]]:
    """Split the tasks by duration (longest first, to the least loaded shard): deterministic."""
    load, assigned = [0.0] * shards, [[] for _ in range(shards)]
    for task in sorted(tasks, key=lambda t: (-(t["duration_s"] or 0.0), t["video_id"])):
        i = min(range(shards), key=lambda s: (load[s], s))
        assigned[i].append(task)
        load[i] += task["duration_s"] or 0.0
    return assigned[shard]


def target_of(output: Path, task: dict[str, Any]) -> Path:
    """:return: the recording of a task."""
    return output / "videos" / task["dataset"] / f"{task['video_id']}.mp4"


def work_of(output: Path, task: dict[str, Any]) -> Path:
    """:return: the scratch folder of a task (browser copy, page, segments)."""
    return output / "work" / task["dataset"] / task["video_id"]


def finished(output: Path, task: dict[str, Any], key: str) -> bool:
    """:return: whether the recording exists with this identity and its recorded digest."""
    manifest = target_of(output, task).with_suffix(".json")
    if not manifest.exists():
        return False
    data = read(manifest)
    return data.get("identity") == key and data.get("status") == "complete"


def prepare(task: dict[str, Any], output: Path, settings: dict[str, Any]) -> dict[str, Any]:
    """Write the browser copy of the video, the page and the frame plan of one task.

    :return: the frame plan (``frames.json``): output fps, frames, segments.
    :raises ValueError: if the browser copy has a different number of frames.
    """
    from apps.demo import viewer
    from apps.demo.run_demo import payload_for

    from evd.exp.demo import analyze, open_run

    work = work_of(output, task)
    plan_path = work / "frames.json"
    if plan_path.exists():
        return read(plan_path)
    work.mkdir(parents=True, exist_ok=True)
    run = open_run(task["source_dir"])
    result, mode = analyze(run, task["video"], modo="replay")
    copy = work / "source.webm"
    data = payload_for(
        run,
        result,
        mode,
        video_src=copy.name,
        annotations=[Path(p) for p in task["annotations"]],
    )
    payload = data.as_dict()
    video = payload["video"]
    fps, frames = float(video["fps"]), int(video["num_frames"])
    width, height = int(video["width"]), int(video["height"])
    pending = work / ".source.pending.webm"
    copied = transcode(Path(task["video"]), pending, fps)
    if copied != frames:
        raise ValueError(f"{task['video_id']}: browser copy has {copied} frames, the run {frames}")
    os.replace(pending, copy)
    viewer.scrivi(data, work / "page.html")
    out_fps = min(fps, settings["max_fps"])
    total = int(math.floor(frames / fps * out_fps + 1e-9))
    per = max(1, int(round(settings["segment_s"] * out_fps)))
    frame_plan = {
        "fps": fps,
        "num_frames": frames,
        "size": [width, height],
        "out_fps": out_fps,
        "frames": total,
        "segments": [[a, min(total, a + per)] for a in range(0, total, per)],
        "default_method": (payload.get("localization") or {}).get("default"),
    }
    write(plan_path, frame_plan)
    return frame_plan


def rate_of(fps: float) -> Fraction:
    """:return: a frame rate as a fraction (29.97 becomes 30000/1001)."""
    return Fraction(fps).limit_denominator(1001)


def transcode(source: Path, target: Path, fps: float) -> int:
    """Write the VP9 copy the browser plays, one frame per decoded source frame.

    Frames get uniform timestamps ``k / fps``, the clock the page reads (``floor(t * fps)``); a
    source larger than :data:`SOURCE_MAX` is scaled down, keeping the aspect ratio.

    :return: the frames written.
    """
    import av

    rate = rate_of(fps)
    with av.open(str(source)) as inp, av.open(str(target), "w", format="webm") as out:
        src = inp.streams.video[0]
        src.thread_type = "AUTO"
        width, height = src.codec_context.width, src.codec_context.height
        k = min(1.0, SOURCE_MAX[0] / width, SOURCE_MAX[1] / height)
        if k < 1.0:
            width, height = int(width * k) // 2 * 2, int(height * k) // 2 * 2
        stream = out.add_stream(
            "libvpx-vp9",
            rate=rate,
            options={
                "crf": "12",
                "b": "0",
                "deadline": "realtime",
                "cpu-used": "8",
                "row-mt": "1",
                "g": "30",
            },
        )
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.codec_context.thread_count = 2
        n = 0
        for frame in inp.decode(src):
            frame = frame.reformat(width=width, height=height, format="yuv420p")
            frame.pts, frame.time_base = n, 1 / rate
            for packet in stream.encode(frame):
                out.mux(packet)
            n += 1
        for packet in stream.encode():
            out.mux(packet)
    return n


def count_packets(path: Path) -> int:
    """:return: the packets of the first video stream (one per frame without B-frames)."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return sum(1 for p in container.demux(stream) if p.size)


class Encoder:
    """H.264 writer of screenshots, without B-frames so that segments join by remuxing."""

    def __init__(self, path: Path, fps: float, settings: dict[str, Any]) -> None:
        import av

        self.rate = rate_of(fps)
        self.out = av.open(str(path), "w", format="mp4")
        self.stream = self.out.add_stream(
            "libx264",
            rate=self.rate,
            options={"crf": str(settings["crf"]), "preset": settings["preset"], "bf": "0"},
        )
        self.stream.width, self.stream.height = settings["width"], settings["height"]
        self.stream.pix_fmt = "yuv420p"
        self.stream.codec_context.thread_count = 2
        self.frames = 0

    def add(self, jpeg: bytes) -> None:
        """Encode one screenshot."""
        import av
        import cv2
        import numpy as np

        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        frame.pts, frame.time_base = self.frames, 1 / self.rate
        for packet in self.stream.encode(frame):
            self.out.mux(packet)
        self.frames += 1

    def close(self) -> None:
        """Flush the encoder and close the file."""
        for packet in self.stream.encode():
            self.out.mux(packet)
        self.out.close()


def join(segments: list[Path], target: Path, fps: float) -> int:
    """Concatenate H.264 segments by remuxing, shifting each by the frames before it.

    :return: the packets written.
    """
    import av

    rate, written, offset = rate_of(fps), 0, Fraction(0)
    with av.open(str(target), "w", format="mp4", options={"movflags": "faststart"}) as out:
        ostream = None
        for path in segments:
            with av.open(str(path)) as inp:
                stream = inp.streams.video[0]
                if ostream is None:
                    ostream = out.add_stream_from_template(stream)
                count = 0
                for packet in inp.demux(stream):
                    if not packet.size:
                        continue
                    shift = int(round(offset / packet.time_base))
                    packet.pts += shift
                    packet.dts += shift
                    packet.stream = ostream
                    out.mux(packet)
                    count += 1
                written += count
                offset += count / rate
    return written


def capture(job: dict[str, Any]) -> dict[str, Any]:
    """Record one segment of a page: seek, draw, screenshot, encode.

    :param job: ``work`` folder, ``index``, ``first`` and ``end`` output frames, ``out_fps`` and
        the settings.
    :return: the segment receipt, also written next to it.
    :raises RuntimeError: if the page cannot play the video or the encoder fails.
    """
    from playwright.sync_api import sync_playwright

    work, settings = Path(job["work"]), job["settings"]
    name = f"seg_{job['index']:04d}"
    done = work / f"{name}.json"
    if done.exists():
        return read(done)
    pending = work / f".{name}.pending.mp4"
    started = time.time()
    encoder = Encoder(pending, job["out_fps"], settings)
    late = 0
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                args=[
                    "--hide-scrollbars",
                    "--mute-audio",
                    "--force-color-profile=srgb",
                    "--allow-file-access-from-files",
                    "--autoplay-policy=no-user-gesture-required",
                ]
            )
            page = browser.new_page(
                viewport={"width": settings["width"], "height": settings["height"]},
                device_scale_factor=1,
            )
            page.goto((work / "page.html").as_uri(), wait_until="load")
            page.wait_for_function(
                "document.getElementById('v').readyState >= 2 || !!document.getElementById('v').error",
                timeout=120_000,
            )
            page.evaluate("document.fonts.ready.then(() => true)")
            state = page.evaluate(
                SETUP_JS,
                {
                    "toggles": settings["toggles"],
                    "method": settings.get("method"),
                    "pause_icon": PAUSE_ICON,
                },
            )
            if state["media_error"]:
                raise RuntimeError(f"the page cannot play {work / 'source.webm'}")
            if state.get("method_missing"):
                raise RuntimeError(f"the page has no method {settings['method']} for this video")
            cdp = page.context.new_cdp_session(page)
            for k in range(job["first"], job["end"]):
                t = (k + 0.5) / job["out_fps"]
                shown = page.evaluate(SEEK_JS, t)
                late += abs(shown - t) > 1e-3
                shot = cdp.send(
                    "Page.captureScreenshot",
                    {
                        "format": "jpeg",
                        "quality": settings["jpeg_quality"],
                        "optimizeForSpeed": True,
                    },
                )
                encoder.add(base64.b64decode(shot["data"]))
            browser.close()
    finally:
        encoder.close()
    frames = count_packets(pending)
    expected = job["end"] - job["first"]
    if frames != expected:
        raise RuntimeError(f"{pending}: {frames} frames encoded, {expected} expected")
    os.replace(pending, work / f"{name}.mp4")
    receipt = {
        "index": job["index"],
        "first": job["first"],
        "end": job["end"],
        "frames": frames,
        "seconds": round(time.time() - started, 2),
        "late_seeks": late,
        "state": state,
    }
    write(done, receipt)
    return receipt


def assemble(
    task: dict[str, Any], output: Path, frame_plan: dict[str, Any], key: str
) -> dict[str, Any]:
    """Join the segments of a task, count the frames and publish the recording with its manifest.

    :return: the manifest.
    :raises RuntimeError: if a segment is missing or the joined file has the wrong frames.
    """
    work, target = work_of(output, task), target_of(output, task)
    receipts = [read(work / f"seg_{i:04d}.json") for i in range(len(frame_plan["segments"]))]
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(f".{target.name}.pending.mp4")
    join([work / f"seg_{i:04d}.mp4" for i in range(len(receipts))], pending, frame_plan["out_fps"])
    frames = count_packets(pending)
    if frames != frame_plan["frames"]:
        raise RuntimeError(f"{pending}: {frames} frames, {frame_plan['frames']} planned")
    os.replace(pending, target)
    states = {json.dumps(r["state"], sort_keys=True) for r in receipts}
    manifest = {
        "status": "complete",
        "identity": key,
        "task": task,
        "file": target.name,
        "sha256": digest(target),
        "bytes": target.stat().st_size,
        "codec": "h264",
        "size": [SETTINGS["width"], SETTINGS["height"]],
        **{k: frame_plan[k] for k in ("fps", "num_frames", "out_fps", "frames", "default_method")},
        "source_size": frame_plan["size"],
        "segments": len(receipts),
        "capture_seconds": round(sum(r["seconds"] for r in receipts), 1),
        "late_seeks": sum(r["late_seeks"] for r in receipts),
        "page_state": json.loads(next(iter(states))),
        "page_state_consistent": len(states) == 1,
        "recorder": "scripts/record_demonstrator.py: apps/demo/assets/viewer.html over "
        "apps.demo.run_demo.payload_for (replay), headless Chromium, frame-stepped",
    }
    write(target.with_suffix(".json"), manifest)
    shutil.rmtree(work, ignore_errors=True)
    return manifest


def run_shard(args: argparse.Namespace) -> None:
    """Prepare, record and assemble the videos of one shard, printing one line per step."""
    data = read(args.plan)
    settings = data["settings"]
    tasks = shard_of(data["tasks"], args.shard, args.shards)
    if args.only:
        tasks = [t for t in data["tasks"] if t["video_id"] in set(args.only)]
    if args.limit:
        tasks = tasks[: args.limit]
    keys = {t["video_id"]: identity(t, settings) for t in tasks}
    todo = [t for t in tasks if not finished(args.output, t, keys[t["video_id"]])]
    print(
        f"shard {args.shard}/{args.shards}: {len(tasks)} videos, {len(todo)} to record", flush=True
    )
    failures = 0
    plans: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        futures = {pool.submit(prepare, t, args.output, settings): t for t in todo}
        for future in as_completed(futures):
            task = futures[future]
            try:
                plans[task["video_id"]] = future.result()
            except Exception as exc:  # noqa: BLE001 - one video must not stop the shard
                failures += 1
                print(f"PREPARE FAILED {task['video_id']}: {exc}", flush=True)
        jobs = []
        for task in todo:
            frame_plan = plans.get(task["video_id"])
            if frame_plan is None:
                continue
            for index, (first, end) in enumerate(frame_plan["segments"]):
                jobs.append(
                    {
                        "video_id": task["video_id"],
                        "work": str(work_of(args.output, task)),
                        "index": index,
                        "first": first,
                        "end": end,
                        "out_fps": frame_plan["out_fps"],
                        "settings": settings,
                    }
                )
        print(f"prepared {len(plans)} videos, {len(jobs)} segments", flush=True)
        pending = {
            t["video_id"]: len(plans[t["video_id"]]["segments"])
            for t in todo
            if t["video_id"] in plans
        }
        broken: set[str] = set()
        futures = {pool.submit(capture, j): j for j in jobs}
        for future in as_completed(futures):
            job = futures[future]
            video_id = job["video_id"]
            try:
                receipt = future.result()
            except Exception as exc:  # noqa: BLE001
                failures += 1
                broken.add(video_id)
                print(f"SEGMENT FAILED {video_id} #{job['index']}: {exc}", flush=True)
                continue
            pending[video_id] -= 1
            rate = receipt["frames"] / max(receipt["seconds"], 1e-6)
            print(
                f"{video_id} #{job['index']} {receipt['frames']} frames {rate:.1f} fps", flush=True
            )
            if pending[video_id] == 0 and video_id not in broken:
                task = next(t for t in todo if t["video_id"] == video_id)
                try:
                    manifest = assemble(task, args.output, plans[video_id], keys[video_id])
                    print(f"DONE {video_id}: {manifest['frames']} frames", flush=True)
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    print(f"ASSEMBLE FAILED {video_id}: {exc}", flush=True)
    print(f"shard {args.shard}/{args.shards}: {failures} failures", flush=True)
    if failures:
        raise SystemExit(1)


# ------------------------------------------------------------------------- catalog and verify


def catalog(args: argparse.Namespace) -> dict[str, Any]:
    """Write ``catalog.json`` and ``index.html`` for the finished recordings."""
    data = read(args.plan)
    videos = []
    for task in data["tasks"]:
        manifest_path = target_of(args.output, task).with_suffix(".json")
        if not manifest_path.exists():
            continue
        manifest = read(manifest_path)
        entry = data["entries"].get(task["video_id"], {})
        videos.append(
            {
                "id": task["video_id"],
                "dataset": task["dataset"],
                "dataset_label": task["dataset_label"],
                "normal": task["normal"],
                "boxes": task["boxes"],
                "file": target_of(args.output, task).relative_to(args.output).as_posix(),
                "duration_s": manifest["frames"] / manifest["out_fps"],
                "out_fps": manifest["out_fps"],
                "bytes": manifest["bytes"],
                "method": manifest["page_state"].get("method_label"),
                "metrics": entry.get("metrics") or {},
                "events": [
                    {
                        k: e.get(k)
                        for k in (
                            "class",
                            "start_s",
                            "end_s",
                            "detected",
                            "latency_s",
                            "chunk_recall",
                            "peak_score",
                            "loc_iou",
                            "loc_ap",
                        )
                    }
                    for e in entry.get("events", [])
                ],
            }
        )
    result = {
        "name": data["name"],
        "model": data["model"],
        "variant": data.get("variant"),
        "run_id": data["run_id"],
        "threshold": data["threshold"],
        "settings": data["settings"],
        "planned": len(data["tasks"]),
        "links": dict(link.split("=", 1) for link in (args.link or [])),
        "videos": videos,
    }
    write(args.output / "catalog.json", result)
    page = (ROOT / "apps" / "demo" / "assets" / "demo_recordings.html").read_text(encoding="utf-8")
    (args.output / "index.html").write_text(page, encoding="utf-8")
    print(f"catalog: {len(videos)} of {len(data['tasks'])} videos")
    return result


def verify(args: argparse.Namespace) -> None:
    """Check that every planned video is recorded and matches its manifest."""
    data = read(args.plan)
    problems = []
    for task in data["tasks"]:
        target = target_of(args.output, task)
        manifest_path = target.with_suffix(".json")
        if not manifest_path.exists():
            problems.append(f"missing {task['video_id']}")
            continue
        manifest = read(manifest_path)
        if manifest.get("identity") != identity(task, data["settings"]):
            problems.append(f"stale {task['video_id']}")
        elif digest(target) != manifest["sha256"]:
            problems.append(f"digest {task['video_id']}")
        elif not manifest.get("page_state_consistent"):
            problems.append(f"page state differs between segments {task['video_id']}")
    report = {"planned": len(data["tasks"]), "problems": problems}
    write(args.output / "verification.json", report)
    print(json.dumps(report, indent=1)[:4000])
    if problems:
        raise SystemExit(1)


def main() -> None:
    """Parse the step and run it."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    steps = parser.add_subparsers(dest="step", required=True)
    p = steps.add_parser("plan")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--videos", type=Path, action="append", required=True)
    p.add_argument("--annotations", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--method", help="localization menu entry to show instead of the default")
    p.add_argument("--only-boxes", action="store_true", help="only the videos with boxes")
    r = steps.add_parser("render")
    r.add_argument("--plan", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--shards", type=int, default=1)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--limit", type=int)
    r.add_argument("--only", action="append", help="record these video ids (a pilot)")
    for name in ("catalog", "verify"):
        c = steps.add_parser(name)
        c.add_argument("--plan", type=Path, required=True)
        c.add_argument("--output", type=Path, required=True)
        c.add_argument("--link", action="append", help="LABEL=URL of another collection")
    args = parser.parse_args()
    if args.step == "plan":
        result = plan(args)
        write(args.out, result)
        hours = sum(t["duration_s"] or 0 for t in result["tasks"]) / 3600
        print(f"plan: {len(result['tasks'])} videos, {hours:.2f} h")
    elif args.step == "render":
        run_shard(args)
    elif args.step == "catalog":
        catalog(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
