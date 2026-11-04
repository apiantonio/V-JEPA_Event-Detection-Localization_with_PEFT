"""Render saved proposal boxes with the demonstrator's header, palette and timeline.

No detector is loaded. Each task names a sealed video prediction and inclusive frame
intervals in which boxes may appear. Native frames and native inclusive coordinates are
preserved; the temporal projection is the same ``fine`` rule used by the evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import av  # noqa: E402
import numpy as np  # noqa: E402
from apps.demo.explanation import (  # noqa: E402
    STRIP_H,
    Header,
    Timeline,
    _VideoWriter,
    draw_boxes,
    header,
    header_height,
    model_label,
    strip,
    strip_background,
)

from evd.explain.boxes import Box  # noqa: E402
from evd.explain.project import offline_assignment, offline_plane  # noqa: E402
from evd.types import WindowSpec  # noqa: E402
from evd.utils.io import write_json  # noqa: E402


def digest(path: Path) -> str:
    """Hash a file in bounded blocks.

    :param path: input file.
    :return: hexadecimal SHA256.
    """
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict:
    """Read one artifact.

    :param path: UTF-8 JSON file.
    :return: decoded artifact.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def frame_scope(intervals: list[dict], count: int) -> np.ndarray:
    """Return the exact union of inclusive annotated segments.

    :param intervals: inclusive start/end frame dictionaries.
    :param count: native video frame count.
    :return: boolean mask over native frames.
    :raises ValueError: if a segment falls outside the native video.
    """
    mask = np.zeros(count, dtype=bool)
    for item in intervals:
        a, b = item["start"], item["end"]
        if not 0 <= a <= b < count:
            raise ValueError(f"Invalid annotated segment: {a}..{b} for {count} frames")
        mask[a : b + 1] = True
    return mask


def native_box(coords: list[int], width: int, height: int) -> Box:
    """Adapt inclusive native pixels to the shared box drawer without rounding drift.

    :param coords: inclusive native ``[x0, y0, x1, y1]``.
    :param width: native frame width.
    :param height: native frame height.
    :return: rectangle accepted by the shared drawer.
    :raises ValueError: if the box lies outside the native frame.
    """
    x0, y0, x1, y1 = coords
    if not (0 <= x0 <= x1 < width and 0 <= y0 <= y1 < height):
        raise ValueError(f"Box outside native frame: {coords} ({width}x{height})")
    return Box(
        token=(x0, y0, x1 + 1, y1 + 1),
        norm=(x0 / width, y0 / height, (x1 + 1) / width, (y1 + 1) / height),
        area_token=(x1 - x0 + 1) * (y1 - y0 + 1),
    )


def prepare(task: dict) -> tuple:
    """Check sealed inputs and build frame assignments before writing any movie.

    :param task: frozen render input with prediction hash, intervals and model identity.
    :return: metadata, records, windows, assignments, mask, plane boxes and timeline.
    :raises ValueError: on modified predictions, missing localizations or bad geometry.
    """
    source = Path(task["source"])
    complete_path = source / "complete.json"
    if digest(complete_path) != task["complete_sha256"]:
        raise ValueError(f"Sealed prediction changed: {complete_path}")
    complete = read(complete_path)
    meta = complete["video"]
    records = []
    for chunk in complete["chunks"]:
        path = source / chunk["path"]
        if digest(path) != chunk["sha256"]:
            raise ValueError(f"Prediction chunk changed: {path}")
        records.append(read(path))
    windows = [
        WindowSpec(
            **{
                **r["window"],
                "explicit_frame_indices": tuple(r["window"]["explicit_frame_indices"]),
            }
        )
        for r in records
    ]
    count = meta["num_frames"]
    assignments = offline_assignment(windows, count, politica="fine")
    inside = frame_scope(task["intervals"], count)
    if np.any(inside & (assignments < 0)):
        raise ValueError("An annotated frame has no covering prediction window")
    used = sorted({int(i) for i in assignments[inside]})
    planes = {}
    for i in used:
        record = records[i]
        if not record.get("localized", record.get("gate_passed", False)):
            raise ValueError(f"Annotated frames require uncomputed chunk {i}: {meta['video_id']}")
        boxes = record["methods"][task["method"]]["boxes"]
        if len(boxes) != record["grid"][0]:
            raise ValueError("Box plane count differs from the token grid")
        planes[i] = [
            [native_box(b, meta["width"], meta["height"]) for b in plane] for plane in boxes
        ]
    intervals = tuple((r["start"], r["end"]) for r in task["intervals"])
    labels = tuple(str(r["class"]) for r in task["intervals"])
    spans = tuple(w.frame_span for w in windows)
    scores = tuple(float(r["score"]) for r in records)
    projected = np.array([scores[i] if i >= 0 else 0 for i in assignments], dtype=np.float32)
    positive = tuple(int(any(a <= d and c <= b for c, d in intervals)) for a, b in spans)
    line = Timeline(
        projected, assignments, spans, scores, task["threshold"], positive, intervals, labels
    )
    return meta, records, windows, assignments, inside, planes, line


def render(task: dict, output: Path) -> dict:
    """Render both existing box styles, validate full decodes and publish atomically.

    :param task: one frozen model/video render input.
    :param output: collection root containing the task's relative target.
    :return: complete rendering manifest, also published in the video folder.
    :raises ValueError: if source or encoded geometry, codec or frame counts disagree.
    """
    target = output / task["target"]
    target.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()
    manifest_path = target / "manifest.json"
    if manifest_path.exists():
        previous = read(manifest_path)
        if previous.get("identity") != identity:
            raise ValueError(f"Existing render has different inputs: {target}")
        if previous["status"] == "complete" and all(
            digest(target / v["file"]) == v["sha256"] for v in previous["videos"]
        ):
            return previous
    meta, records, windows, assignment, inside, planes, line = prepare(task)
    width, height = meta["width"], meta["height"]
    even_w, even_h = width + width % 2, height + height % 2
    geometries = {
        "interfaccia": (even_w, even_h + header_height(even_w) + STRIP_H),
        "pulito": (even_w, even_h),
    }
    paths = {
        s: target / f"bbox__{task['method']}__annotati{'_pulito' if s == 'pulito' else ''}.mp4"
        for s in geometries
    }
    pending = {s: p.with_name(f".{p.stem}.pending.mp4") for s, p in paths.items()}
    writers = {s: _VideoWriter(pending[s], meta["fps"], size) for s, size in geometries.items()}
    background = strip_background(even_w, line)
    cached_headers = {}
    count = box_frames = box_count = 0
    try:
        with av.open(meta["path"]) as container:
            container.streams.video[0].thread_count = 1
            for frame_index, frame in enumerate(container.decode(video=0)):
                if frame_index >= meta["num_frames"]:
                    raise ValueError("Source video contains more frames than its sealed metadata")
                body = frame.to_ndarray(format="bgr24")
                if body.shape[:2] != (height, width):
                    raise ValueError("Native source dimensions differ from sealed metadata")
                i = int(assignment[frame_index])
                plane = (
                    offline_plane(windows[i], frame_index, records[i]["grid"][0]) if i >= 0 else -1
                )
                boxes = planes[i][plane] if inside[frame_index] else []
                box_count += len(boxes)
                box_frames += bool(boxes)
                clean = np.zeros((even_h, even_w, 3), dtype=np.uint8)
                clean[:height, :width] = draw_boxes(body.copy(), boxes, etichette=False)
                writers["pulito"].write(clean)
                interface = clean.copy()
                interface[:height, :width] = draw_boxes(body, boxes)
                event = line.event_at(frame_index)
                signature = (i, plane, bool(inside[frame_index]), event)
                if signature not in cached_headers:
                    spec = Header(
                        metodo=f"Box: {task['method']}",
                        famiglia="Proposed localization",
                        modello=model_label(task["run_id"]),
                        video=meta["video_id"],
                        dettaglio=f"win. {i + 1}/{len(records)} - plane {plane + 1}"
                        + (f" - event {event}" if event else ""),
                        significato="VTCD concepts covering 70% of positive IG mass",
                        score=records[i]["score"] if i >= 0 else None,
                        soglia=task["threshold"],
                        chunk_positivo=line.label_of(i),
                        extra=f"{len(boxes)} box - annotated interval"
                        if inside[frame_index]
                        else "outside annotated intervals: no box",
                    )
                    cached_headers[signature] = header(even_w, spec)
                writers["interfaccia"].write(
                    np.concatenate(
                        (
                            cached_headers[signature],
                            interface,
                            strip(background, frame_index, line),
                        ),
                        axis=0,
                    )
                )
                count += 1
    finally:
        for writer in writers.values():
            writer.release()
    if count != meta["num_frames"]:
        raise ValueError(f"Source decoded {count} frames, expected {meta['num_frames']}")
    videos = []
    for style, path in paths.items():
        with av.open(str(pending[style])) as container:
            stream = container.streams.video[0]
            context = stream.codec_context
            decoded = sum(1 for _ in container.decode(video=0))
            if (
                decoded != count
                or context.name != "h264"
                or context.pix_fmt != "yuv420p"
                or (context.width, context.height) != geometries[style]
            ):
                raise ValueError(f"Encoded movie failed validation: {pending[style]}")
        os.replace(pending[style], path)
        videos.append(
            {
                "file": path.name,
                "style": style,
                "frames": decoded,
                "codec": "h264",
                "pixel_format": "yuv420p",
                "sha256": digest(path),
            }
        )
    coordinates = {
        str(i): {
            "window_id": records[i]["window_id"],
            "boxes": records[i]["methods"][task["method"]]["boxes"],
        }
        for i in planes
    }
    write_json(
        target / "boxes.json",
        {
            "coordinate_space": "native_pixels_xyxy_inclusive",
            "projection": "fine",
            "intervals": task["intervals"],
            "windows": coordinates,
        },
    )
    result = {
        "status": "complete",
        "identity": identity,
        "task": task,
        "videos": videos,
        "source_frames": count,
        "annotated_frames": int(inside.sum()),
        "frames_with_boxes": box_frames,
        "boxes_drawn": box_count,
        "boxes_outside_annotated_intervals": 0,
        "renderer": "apps.demo.explanation shared header, timeline, draw_boxes and H264 writer",
    }
    write_json(manifest_path, result)
    return result


def main() -> None:
    """Run the requested shard with optional CPU workers and per-task progress."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        raise ValueError("Invalid shard index")
    tasks = read(args.plan)["tasks"][args.shard :: args.shards]
    if args.limit:
        tasks = tasks[: args.limit]
    done = []

    def progress(task: dict) -> None:
        done.append(task["target"])
        write_json(
            args.output / f"progress-{args.plan.stem}-{args.shard}-of{args.shards}.json",
            {"complete": done, "total": len(tasks)},
        )

    if args.workers > 1 and not args.preflight:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(render, task, args.output): task for task in tasks}
            for number, future in enumerate(as_completed(futures)):
                result = future.result()
                progress(futures[future])
                print(
                    json.dumps(
                        {
                            "task": futures[future]["target"],
                            "status": result["status"],
                            "progress": [number + 1, len(tasks)],
                        }
                    ),
                    flush=True,
                )
        return
    for number, task in enumerate(tasks):
        if args.preflight:
            prepare(task)
        else:
            result = render(task, args.output)
            progress(task)
            print(
                json.dumps(
                    {
                        "task": task["target"],
                        "status": result["status"],
                        "progress": [number + 1, len(tasks)],
                    }
                ),
                flush=True,
            )
    print(f"{'preflight' if args.preflight else 'render'} complete: {len(tasks)} tasks", flush=True)


if __name__ == "__main__":
    main()
