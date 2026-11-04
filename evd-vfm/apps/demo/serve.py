"""Local demonstrator console, on top of the existing CLI and engine (ADR-055).

Serves only registered files, on loopback. Analyses run as separate processes, so the
page stays responsive and cancelling also frees the model memory. No extra web
dependency; historical results are never modified.
"""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import secrets
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
for entry in (str(ROOT), str(ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from apps.demo import localization  # noqa: E402
from evd.exp.demo import demo_geometry, open_run, validate_hop  # noqa: E402
from evd.infer.replay import scores_from_disk  # noqa: E402
from evd.utils.env import device as local_device  # noqa: E402

ASSETS = Path(__file__).parent / "assets"
EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
MAX_UPLOAD = 2 * 1024**3
#: Library names of the external test sets (``data.name`` of an external evaluation).
DATASET_LABELS = {"mivia_crime": "MIVIA-Crime"}


class Console:
    """Hold catalogue, authorised files and at most one local analysis at a time.

    :param runs: folder of the existing results.
    :param videos: local folders to index.
    :param annotations: annotation delivery folders, for library videos only (UCF-Crime
        and MIVIA-Crime have different deliveries; the video decides which applies).
    """

    def __init__(
        self, runs: Path, videos: list[Path], annotations: list[Path] | Path | None = None
    ) -> None:
        self.runs = runs.resolve()
        # `None` is legitimate ("no annotation": the demonstrator draws no comparison row), and a
        # single folder is still accepted.
        if annotations is None:
            annotations = []
        elif isinstance(annotations, Path):
            annotations = [annotations]
        self.annotations = [d for d in annotations if d.is_dir()]
        self.workspace = self.runs / "demo-local"
        self.files: dict[str, Path] = {}
        self.library: dict[str, Path] = {}
        self.video_durations: dict[tuple[Path, int, int], float | None] = {}
        for directory in videos:
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix.lower() in EXTENSIONS:
                    # Nominal duplicates are not resolved by overwriting the last one.
                    self.library.setdefault(path.stem, path.resolve())
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()

    def register(self, path: Path) -> str:
        """:return: an opaque URL for a single file, without exposing the filesystem."""
        token = secrets.token_hex(16)
        self.files[token] = path.resolve()
        return "/file/" + token

    def run_path(self, name: str) -> Path:
        """:return: a run directly inside the configured folder.

        :raises ValueError: for arbitrary paths or missing runs.
        """
        path = (self.runs / name).resolve()
        if path.parent != self.runs or not path.is_dir():
            raise ValueError("Choose a model available in the library.")
        return path

    def catalog(self) -> dict[str, Any]:
        """:return: models with results, without loading any model into memory."""
        runs = []
        for path in sorted(self.runs.iterdir()):
            if not (path / "config.yaml").is_file() or not (path / "metrics.json").is_file():
                continue
            try:
                run = open_run(path)
                if not run.score_path.exists() and not run.checkpoint_path.exists():
                    continue
                runs.append(
                    {
                        "id": run.run_id,
                        "protocol": run.protocol.get("id", ""),
                        "live": run.checkpoint_path.is_file(),
                    }
                )
            except (ValueError, OSError, KeyError):
                continue
        return {"runs": runs, "local_videos": len(self.library)}

    def external_runs(self, run: Any) -> list[Any]:
        """Return the external evaluations of a run's weights (ADR-066).

        They live in ``<run_id>__<dataset>`` folders, nested in the run or next to it; one
        counts only if its threshold is declared transferred from this run, so other
        ``__`` folders (reruns, profiles) are not mistaken for it.

        :param run: the opened run.
        :return: the opened external evaluations, in name order.
        """
        found = []
        for path in sorted(
            {*run.dir.glob(f"{run.run_id}__*"), *self.runs.glob(f"{run.run_id}__*")}
        ):
            if not (path / "config.yaml").is_file() or not (path / "scores.jsonl").is_file():
                continue
            try:
                other = open_run(path)
            except (ValueError, OSError, KeyError):
                continue
            if (other.metrics.get("threshold") or {}).get("trasferita_da") == run.run_id:
                found.append(other)
        return found

    def video_duration(self, key: str) -> float | None:
        """Read only the file metadata and reuse the duration until the file changes."""
        path = self.library.get(key)
        if path is None:
            return None
        try:
            stat = path.stat()
            cache_key = (path, stat.st_mtime_ns, stat.st_size)
            if cache_key not in self.video_durations:
                import av

                with av.open(str(path)) as container:
                    stream = container.streams.video[0]
                    duration = (
                        float(stream.duration * stream.time_base)
                        if stream.duration is not None and stream.time_base is not None
                        else float(container.duration / av.time_base)
                        if container.duration
                        else None
                    )
                    self.video_durations[cache_key] = (
                        duration
                        if duration is not None and math.isfinite(duration) and duration > 0
                        else None
                    )
            return self.video_durations[cache_key]
        except (OSError, ValueError, IndexError, ImportError):
            return None

    def videos(self, name: str) -> dict[str, Any]:
        """:return: replayable videos, partition and availability of the local file."""
        run = open_run(self.run_path(name))
        declared = str(run.cfg.get("backbone", {}).get("checkpoint", ""))
        local = ROOT.parent / "data" / "checkpoints" / Path(declared).name
        backbone = str(local) if local.is_file() else declared
        scores = scores_from_disk(run.score_path) if run.score_path.exists() else {}
        partitions = {key: value.split for key, value in scores.items()}
        # The external test sets of the same weights become partitions of the library.
        external = []
        for other in self.external_runs(run):
            name = str(other.cfg.get("data", {}).get("name") or other.run_id)
            external.append({"id": name, "label": DATASET_LABELS.get(name, name) + " test"})
            for key, value in scores_from_disk(other.score_path).items():
                if key not in scores:
                    scores[key] = value
                    partitions[key] = name
        boxes = localization.exported_videos(run.dir, run.metrics)
        split_path = Path(str(run.protocol.get("split_file", "")))
        if not split_path.is_absolute():
            split_path = ROOT / split_path
        if split_path.is_file():
            manifest = json.loads(split_path.read_text(encoding="utf-8"))
            for split in ("train", "val", "test"):
                for key in manifest.get(split, []):
                    if key in self.library:
                        partitions.setdefault(key, split)
        return {
            "videos": [
                {
                    "id": key,
                    "split": split,
                    "available": key in self.library,
                    "replay": key in scores,
                    "normal": key.lower().startswith("normal_video"),
                    "duration_s": self.video_duration(key),
                    "saved_windows": len(scores[key].per_finestra) if key in scores else None,
                    "boxes": key in boxes,
                }
                for key, split in sorted(partitions.items())
            ],
            "partitions": external,
            "hop_s": run.protocol["temporal"]["hop_s"],
            "window_s": run.protocol["temporal"].get("window_s", 8),
            "window_configurable": run.protocol["temporal"].get("window_grid") == "fixed",
            "live": run.checkpoint_path.is_file(),
            "threshold": run.threshold,
            "checkpoint": str(run.checkpoint_path),
            "backbone": backbone,
        }

    def start(self, request: dict[str, Any]) -> dict[str, Any]:
        """Start the existing CLI with separate arguments and isolated output.

        :return: the analysis identifier, queryable without blocking the UI.
        :raises ValueError: if an input is missing or another analysis is running.
        """
        with self.lock:
            if any(job["process"].poll() is None for job in self.jobs.values()):
                raise ValueError("An analysis is already running. Wait or press Cancel.")
            run = open_run(self.run_path(str(request.get("run", ""))))
            mode = str(request.get("mode", "replay"))
            realtime = request.get("realtime", False)
            if not isinstance(realtime, bool) or (realtime and mode != "live"):
                raise ValueError("The live stream requires a new local inference.")
            source = run
            if mode == "replay":
                video_id = str(request.get("video", ""))
                if video_id not in scores_from_disk(run.score_path):
                    # A video of an external test set is replayed from that evaluation.
                    source = next(
                        (
                            other
                            for other in self.external_runs(run)
                            if video_id in scores_from_disk(other.score_path)
                        ),
                        None,
                    )
                    if source is None:
                        raise ValueError("This video has no saved results for the chosen model.")
                video = self.library.get(video_id)
            elif mode == "live":
                video = (
                    self.files.get(str(request.get("upload", "")))
                    if request.get("upload")
                    else self.library.get(str(request.get("video", "")))
                )
                if not run.checkpoint_path.is_file():
                    raise ValueError(
                        f"Trained weights are missing. Copy them from the cluster: {run.checkpoint_path}"
                    )
            else:
                raise ValueError("Invalid mode. Choose saved results or a new analysis.")
            if video is None or not video.is_file():
                raise ValueError("Video not available on this PC. Choose an existing file.")
            hop = request.get("hop_s")
            if hop is not None:
                if mode != "live":
                    raise ValueError("Changing the hop requires a new inference.")
                hop = validate_hop(hop)
            window = request.get("window_s")
            if window is not None and mode != "live":
                raise ValueError("Changing the window requires a new inference.")
            demo_geometry(run.protocol, hop_s=hop, window_s=window)
            identifier = secrets.token_hex(12)
            out = self.workspace / "sessions" / identifier
            out.mkdir(parents=True)
            if realtime:
                (out / "clock.json").write_text(
                    json.dumps({"position_s": 0, "updated_at": time.time()}), encoding="utf-8"
                )
            command = [
                sys.executable,
                str(Path(__file__).with_name("run_demo.py")),
                "--run",
                str(source.dir),
                "--video",
                str(video),
                "--modo",
                mode,
                "--out",
                str(out),
                "--no-video",
                "--no-timeline",
                "--no-copia-video",
            ]
            if mode == "live":
                if realtime:
                    command += ["--realtime", "--playback-clock", str(out / "clock.json")]
                if window is not None:
                    command += ["--window-s", str(window)]
                if hop is not None:
                    command += ["--hop-s", str(hop)]
                selected_device = str(request.get("device", "auto"))
                if selected_device == "auto":
                    selected_device = local_device()
                command += ["--device", selected_device, "--batch-size", "1"]
                if request.get("backbone"):
                    command += ["--backbone-checkpoint", str(request["backbone"])]
            if not request.get("upload"):
                for folder in self.annotations:
                    command += ["--annotazioni", str(folder)]
            log_path = out / "analysis.log"
            with log_path.open("wb") as stream:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=stream,
                    stderr=stream,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            self.jobs[identifier] = {
                "process": process,
                "out": out,
                "video": video,
                "cancelled": False,
                "realtime": realtime,
                "position_s": 0.0,
            }
            return {
                "id": identifier,
                "realtime": realtime,
                "video_url": self.register(video) if realtime else None,
            }

    def status(self, identifier: str, after: int | None = None) -> dict[str, Any]:
        """:return: state and links, registered only after the CLI has finished."""
        with self.lock:
            job = self.jobs[identifier]
            progress = self.progress(job, after) if after is not None else {}
            code = job["process"].poll()
            if job["cancelled"]:
                return {"state": "cancelled", **progress}
            if code is None:
                return {"state": "running", **progress}
            if code != 0:
                log = (job["out"] / "analysis.log").read_text(encoding="utf-8", errors="replace")
                return {"state": "error", "message": log[-5000:], **progress}
            if "result" not in job:
                # Reuse the renderer, replacing only the video file URL.
                page = (job["out"] / "index.html").read_text(encoding="utf-8")
                start = page.index(">", page.index('<script id="payload"')) + 1
                end = page.index("</script>", start)
                data = json.loads(page[start:end])
                data["video"]["src"] = self.register(job["video"])
                page = page[:start] + json.dumps(data).replace("</", "<\\/") + page[end:]
                served = job["out"] / "console.html"
                served.write_text(page, encoding="utf-8")
                job["result"] = {
                    "state": "done",
                    "video_metrics": data.get("video_metrics"),
                    "metrics": data.get("metrics"),
                    "url": self.register(served),
                    "json": self.register(job["out"] / "demo.json"),
                }
            return {**job["result"], **progress}

    @staticmethod
    def progress(job: dict[str, Any], after: int) -> dict[str, Any]:
        """Read only complete journal lines, with a byte cursor."""
        if after < 0:
            raise ValueError("Invalid cursor.")
        path = job["out"] / "progress.jsonl"
        if not path.exists():
            return {"events": [], "next": 0, "more": False}
        with path.open("rb") as stream:
            if after > path.stat().st_size:
                raise ValueError("Cursor beyond the journal.")
            stream.seek(after)
            chunk = stream.read(64 * 1024)
        complete = chunk[: chunk.rfind(b"\n") + 1]
        return {
            "events": [json.loads(line) for line in complete.splitlines()],
            "next": after + len(complete),
            "more": len(chunk) == 64 * 1024,
        }

    def playback(self, identifier: str, position: float) -> None:
        """Update the video clock atomically; late requests never move it back."""
        if isinstance(position, bool) or not isinstance(position, (float, int)):
            raise ValueError("Invalid video time.")
        if not math.isfinite(position) or position < 0:
            raise ValueError("Invalid video time.")
        with self.lock:
            job = self.jobs[identifier]
            if not job.get("realtime"):
                raise ValueError("This analysis is not a live stream.")
            if job["process"].poll() is not None:
                return
            job["position_s"] = max(position, job["position_s"])
            path = job["out"] / "clock.json"
            temporary = path.with_suffix(".tmp")
            # On Windows the player (or OneDrive) may keep the file open without delete sharing, and
            # replace fails until that handle is closed. Never expose a partial JSON.
            for attempt in range(21):
                try:
                    temporary.write_text(
                        json.dumps({"position_s": job["position_s"], "updated_at": time.time()}),
                        encoding="utf-8",
                    )
                    temporary.replace(path)
                    break
                except PermissionError:
                    if attempt == 20:
                        raise
                    time.sleep(0.025)

    def cancel(self, identifier: str) -> None:
        """Stop only the analysis process created by this console."""
        with self.lock:
            job = self.jobs[identifier]
            if job["process"].poll() is None:
                job["cancelled"] = True
                job["process"].terminate()
                job["process"].wait(timeout=10)


def byte_range(header: str | None, size: int) -> tuple[int, int, bool]:
    """Parse a single HTTP Range header, needed for seeking in the browser.

    :return: start, inclusive end and a partial-response flag.
    :raises ValueError: for unsatisfiable or multiple ranges.
    """
    if not header:
        return 0, size - 1, False
    if not header.startswith("bytes=") or "," in header:
        raise ValueError("Range not supported")
    left, right = header[6:].split("-", 1)
    if not left:
        count = int(right)
        if count <= 0:
            raise ValueError("Empty range")
        start, end = max(0, size - count), size - 1
    else:
        start, end = int(left), min(int(right), size - 1) if right else size - 1
    if start < 0 or start >= size or end < start:
        raise ValueError("Range outside the file")
    return start, end, True


def handler_for(console: Console) -> type[BaseHTTPRequestHandler]:
    """:return: an HTTP handler limited to registered files and the console APIs."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def allowed(self) -> bool:
            host = f"127.0.0.1:{self.server.server_port}"
            return (
                self.headers.get("Host") == host
                and self.headers.get("Origin", "http://" + host) == "http://" + host
            )

        def reply(self, data: dict[str, Any], code: int = 200) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_file(self, path: Path, head: bool = False) -> None:
            size = path.stat().st_size
            try:
                start, end, partial = byte_range(self.headers.get("Range"), size)
            except (ValueError, TypeError):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(206 if partial else 200)
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-cache")
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if not head:
                try:
                    with path.open("rb") as stream:
                        stream.seek(start)
                        remaining = end - start + 1
                        while remaining > 0:
                            chunk = stream.read(min(256 * 1024, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass  # A seek normally interrupts the previous request.

        def do_HEAD(self) -> None:
            self.do_GET(head=True)

        def do_GET(self, head: bool = False) -> None:
            if not self.allowed():
                self.reply({"message": "Open the console from its local address."}, 403)
                return
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            try:
                if url.path == "/":
                    self.send_file(ASSETS / "console.html", head)
                elif url.path == "/api/catalog":
                    self.reply(console.catalog())
                elif url.path == "/api/videos":
                    self.reply(console.videos(query["run"][0]))
                elif url.path == "/api/job":
                    after = int(query["after"][0]) if "after" in query else None
                    self.reply(console.status(query["id"][0], after))
                elif url.path in ("/assets/realtime.js", "/assets/metrics.js"):
                    self.send_file(ASSETS / Path(url.path).name, head)
                elif url.path.startswith("/file/"):
                    self.send_file(console.files[url.path.removeprefix("/file/")], head)
                else:
                    self.reply({"message": "Page not found."}, 404)
            except (KeyError, FileNotFoundError):
                self.reply(
                    {"message": "File not available. Reopen the video from the library."}, 404
                )
            except (ValueError, OSError) as exc:
                self.reply({"message": str(exc)}, 400)

        def do_POST(self) -> None:
            if not self.allowed() or self.headers.get("X-EVD-Console") != "1":
                self.reply({"message": "External request rejected."}, 403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if self.path == "/api/upload":
                    if not 0 < size <= MAX_UPLOAD:
                        raise ValueError("Choose a non-empty video, up to 2 GB.")
                    name = Path(unquote(self.headers.get("X-Filename", "video.mp4"))).name
                    if Path(name).suffix.lower() not in EXTENSIONS:
                        raise ValueError("Unsupported format: use MP4, MOV, MKV, AVI or WebM.")
                    directory = console.workspace / "uploads" / secrets.token_hex(12)
                    directory.mkdir(parents=True)
                    path = directory / name
                    self.connection.settimeout(120)
                    try:
                        with path.open("wb") as stream:
                            remaining = size
                            while remaining:
                                chunk = self.rfile.read(min(1024 * 1024, remaining))
                                if not chunk:
                                    raise ValueError("Upload interrupted. Select the file again.")
                                stream.write(chunk)
                                remaining -= len(chunk)
                    except Exception:
                        path.unlink(missing_ok=True)
                        raise
                    url = console.register(path)
                    self.reply({"token": url.removeprefix("/file/"), "url": url, "name": name})
                    return
                if not 0 < size < 16384:
                    raise ValueError("Invalid request.")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError("Invalid request: a JSON object is required.")
                if self.path == "/api/analyze":
                    self.reply(console.start(data))
                elif self.path == "/api/cancel":
                    console.cancel(data["id"])
                    self.reply({"state": "cancelled"})
                elif self.path == "/api/playback":
                    console.playback(data["id"], data["position_s"])
                    self.reply({"ok": True})
                else:
                    self.reply({"message": "Command not found."}, 404)
            except (ValueError, KeyError, OSError) as exc:
                self.reply({"message": str(exc)}, 400)

    return Handler


def main() -> None:
    """Start the console on a free port and open the local browser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--videos", type=Path, action="append")
    parser.add_argument("--annotations", type=Path, action="append")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    console = Console(
        args.runs,
        args.videos or [ROOT.parent / "data/raw", ROOT.parent / "data/mivia_crime"],
        args.annotations
        or [ROOT.parent / "data/annotations/fs_ucf_crime", ROOT.parent / "data/mivia_crime"],
    )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(console))
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"\nEVD Studio: {url}\nKeep this window open. Ctrl+C to close.\n", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for identifier in console.jobs:
            console.cancel(identifier)
        server.server_close()


if __name__ == "__main__":
    main()
