"""Loopback server of the annotator: native frames, revisions and export package.

Interface in ``docs/20_annotatore_spaziale.md``. No model is loaded and there are no web
dependencies; like ``apps/demo/serve.py``, only registered files are exposed.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import secrets
import socket
import sqlite3
import tempfile
import threading
import urllib.request
import webbrowser
import zipfile
from collections import OrderedDict
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from apps.annotator.preload import FrameStore, Preloader, encode_frame
from apps.annotator.project import (
    ConflictError,
    Project,
    add_candidates,
    create_local_project,
    now,
    sha,
)

ASSETS = Path(__file__).parent / "assets"


class Frames:
    """Decode by ordinal index, with a cursor and a JPEG cache bounded to 32 MiB.

    A persistent cursor makes advancing by one frame cheap; counting decoded images avoids
    confusing PTS and ordinals on pathological MIVIA videos.
    """

    def __init__(self, store: FrameStore | None = None) -> None:
        self.store = store
        self.lock = threading.Lock()
        self.container = None
        self.path = None
        self.identity = None
        self.position = -1
        self.cache: OrderedDict[tuple[str, int], bytes] = OrderedDict()
        self.cache_bytes = 0

    def get(self, video: dict, index: int) -> bytes:
        """:return: JPEG of the requested native frame, without approximate seeks.

        :raises ValueError: if the index is out of range or the geometry differs from the
            inventory.
        """
        import av

        if not 0 <= index < video["num_frames"]:
            raise ValueError("Frame outside the video.")
        identity = self.store.identity(video) if self.store else video["path"]
        key = (identity, index)
        if self.store:
            cached = self.store.get(identity, index)
            if cached is not None:
                return cached
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            if self.identity != identity or index <= self.position:
                if self.container is not None:
                    self.container.close()
                self.container = av.open(video["path"])
                self.iterator = iter(self.container.decode(video=0))
                self.path, self.position = video["path"], -1
                self.identity = identity
            try:
                while self.position < index:
                    frame = next(self.iterator)
                    self.position += 1
            except StopIteration as exc:
                raise ValueError("Video ended earlier than the inventory.") from exc
            payload = encode_frame(frame, video)
            if self.store:
                self.store.put(identity, index, payload)
            self.cache[key] = payload
            self.cache_bytes += len(payload)
            while self.cache_bytes > 32 * 1024**2 and len(self.cache) > 1:
                _, removed = self.cache.popitem(last=False)
                self.cache_bytes -= len(removed)
            return payload


def export_project(project: Project) -> bytes:
    """Export reviewed ground truth, drafts, sources and a restorable copy of the project.

    :return: ZIP; CSV/COCO contain only reviewed frames, empty ones included.
    """
    output = io.BytesIO()
    boxes_buffer, frames_buffer = io.StringIO(), io.StringIO()
    boxes_csv, frames_csv = csv.writer(boxes_buffer), csv.writer(frames_buffer)
    boxes_csv.writerow(
        [
            "dataset",
            "video_id",
            "frame",
            "box",
            "x0",
            "y0",
            "x1",
            "y1",
            "class_name",
            "track_id",
            "origin",
            "source_id",
            "reviewer",
            "display_id",
            "color",
            "token_x0",
            "token_y0",
            "token_x1_exclusive",
            "token_y1_exclusive",
            "token_rows",
            "token_columns",
        ]
    )
    frames_csv.writerow(
        [
            "dataset",
            "video_id",
            "frame",
            "box_count",
            "reviewer",
            "note",
            "review_scope",
            "unit_id",
            "unit_start",
            "unit_end",
        ]
    )
    classes = sorted({e["class_name"] for v in project.videos.values() for e in v["events"]})
    cat_ids = {name: i + 1 for i, name in enumerate(classes)}
    coco: dict = {
        "info": {"created_at": now(), "description": "Only explicitly reviewed frames"},
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": n} for n, i in cat_ids.items()],
    }
    gt = {
        "schema": 1,
        "coordinate_convention": project.manifest["coordinate_convention"],
        "created_at": now(),
        "videos": [],
    }
    # Consistent SQLite snapshot even while another tab is saving.
    with tempfile.TemporaryDirectory(prefix="evd-annotation-export-") as tmp:
        backup = Path(tmp) / "annotations.sqlite3"
        with project.connect() as source, closing(sqlite3.connect(backup)) as dest:
            source.backup(dest)
        with closing(sqlite3.connect(backup)) as con:
            states = {r[0]: json.loads(r[1]) for r in con.execute("SELECT video,body FROM states")}
            with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(
                    "project.json", json.dumps(project.manifest, ensure_ascii=False, indent=2)
                )
                archive.write(backup, "annotations.sqlite3")
                for vid, video in project.videos.items():
                    state = states.get(vid, {"revision": 0, "frames": {}})
                    archive.writestr(f"work/{vid}.json", json.dumps(state, ensure_ascii=False))
                    reviewed = {
                        k: f for k, f in state["frames"].items() if f["status"] == "reviewed"
                    }
                    if not reviewed:
                        continue
                    gt["videos"].append(
                        {k: val for k, val in video.items() if k not in {"candidates", "path"}}
                        | {"revision": state["revision"], "frames": reviewed}
                    )
                    for key in sorted(reviewed, key=int):
                        item, frame = reviewed[key], int(key)
                        image_id = len(coco["images"]) + 1
                        coco["images"].append(
                            {
                                "id": image_id,
                                "file_name": f"{video['dataset']}/{video['video_id']}/{frame:08d}.jpg",
                                "width": video["width"],
                                "height": video["height"],
                                "video_id": video["video_id"],
                                "dataset": video["dataset"],
                                "frame": frame,
                                "review_unit": item.get("review_unit"),
                            }
                        )
                        frames_csv.writerow(
                            [
                                video["dataset"],
                                video["video_id"],
                                frame,
                                len(item["boxes"]),
                                item["reviewer"],
                                item.get("note", ""),
                                "plane_hold" if item.get("review_unit") else "frame",
                                item.get("review_unit", {}).get("id", ""),
                                item.get("review_unit", {}).get("start", ""),
                                item.get("review_unit", {}).get("end", ""),
                            ]
                        )
                        for box_index, box in enumerate(item["boxes"], 1):
                            x0, y0, x1, y1 = box["xyxy"]
                            w, h = x1 - x0 + 1, y1 - y0 + 1
                            boxes_csv.writerow(
                                [
                                    video["dataset"],
                                    video["video_id"],
                                    frame,
                                    box["id"],
                                    x0,
                                    y0,
                                    x1,
                                    y1,
                                    box["class_name"],
                                    box.get("track_id", ""),
                                    box["origin"],
                                    box.get("source_id", ""),
                                    item["reviewer"],
                                    box.get("display_id", box_index),
                                    box.get("color", "#8bffbd"),
                                    *box.get("token_space", {}).get("xyxy", ["", "", "", ""]),
                                    *box.get("token_space", {}).get("grid", ["", ""]),
                                ]
                            )
                            coco["annotations"].append(
                                {
                                    "id": len(coco["annotations"]) + 1,
                                    "image_id": image_id,
                                    "category_id": cat_ids[box["class_name"]],
                                    "bbox": [x0, y0, w, h],
                                    "area": w * h,
                                    "iscrowd": 0,
                                    "attributes": box,
                                }
                            )
                for vid, source_id, label, metadata, csv_text in con.execute(
                    "SELECT video,id,label,metadata,csv FROM sources"
                ):
                    prefix = f"sources/{vid}/{source_id}"
                    archive.writestr(f"{prefix}/boxes.json", metadata)
                    archive.writestr(f"{prefix}/boxes_per_frame.csv", csv_text)
                    archive.writestr(f"{prefix}/label.txt", label)
                for name, data in [("ground_truth.json", gt), ("coco.json", coco)]:
                    archive.writestr(name, json.dumps(data, ensure_ascii=False, indent=2))
                archive.writestr("boxes_per_frame.csv", boxes_buffer.getvalue())
                archive.writestr("reviewed_frames.csv", frames_buffer.getvalue())
                for vid, sid, body in con.execute("SELECT video,id,body FROM source_units"):
                    archive.writestr(f"sources/{vid}/{sid}/temporal_units.json", body)
                # Base planes are not a source: they are exported only where a saved frame declares them,
                # since the unit ID refers to them.
                for vid, state in states.items():
                    if any(
                        f.get("review_unit") and not f["review_unit"].get("source_id")
                        for f in state["frames"].values()
                    ):
                        view = {
                            k: v for k, v in project.protocol_view(vid).items() if k != "frames"
                        }
                        archive.writestr(
                            f"protocol_units/{vid}/temporal_units.json",
                            json.dumps(view, ensure_ascii=False),
                        )
                archive.writestr(
                    "LEGGIMI.txt",
                    "Zero-based frames; inclusive xyxy pixels. COCO uses xywh.\n"
                    "GT, CSV and COCO include only explicitly reviewed frames, even without boxes.\n"
                    "Drafts and excluded frames are in work/. Original sources are in sources/.\n"
                    "Base planes without a source (edp_v2, overlap_positivo): protocol_units/.\n"
                    "COCO image names are references: images are not included.\n"
                    "To resume: extract project.json and annotations.sqlite3 into a new folder;\n"
                    "run annota_spaziale.py --project FOLDER. The original videos must be available.\n"
                    "Proposals may bias the review: document this provenance in the metrics.\n",
                )
    return output.getvalue()


def make_server(project: Project, port: int = 8766) -> ThreadingHTTPServer:
    """:return: a server bound to 127.0.0.1, with a token for every write."""
    token_path = project.root / ".server-token"
    try:
        with token_path.open("x", encoding="ascii") as stream:
            stream.write(secrets.token_urlsafe(32))
    except FileExistsError:
        pass
    token = token_path.read_text(encoding="ascii").strip()
    if len(token) < 32:
        raise ValueError("Incomplete local server token.")
    store = FrameStore(project.root / ".frame-cache")
    frames = Frames(store)
    preloader = Preloader(store)

    from apps.annotator.geometry import token_grid
    from apps.annotator.imports import collection_candidates, import_pair

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            pass

        def send(self, status: int, payload: bytes | dict, mime: str = "application/json") -> None:
            if isinstance(payload, dict):
                payload = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' blob:; style-src 'self' 'unsafe-inline'; object-src 'none'; frame-ancestors 'none'",
            )
            if mime == "application/zip":
                self.send_header(
                    "Content-Disposition", 'attachment; filename="spatial-annotations.zip"'
                )
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def safe_host(self) -> bool:
            port_here = self.server.server_address[1]
            return self.headers.get("Host") in {f"127.0.0.1:{port_here}", f"localhost:{port_here}"}

        def do_GET(self) -> None:
            if not self.safe_host():
                self.send(403, {"error": "Host not allowed."})
                return
            url = urlsplit(self.path)
            parts = url.path.strip("/").split("/")
            try:
                if url.path in {
                    "/",
                    "/app.js",
                    "/guide.js",
                    "/box-tools.js",
                    "/collection.js",
                    "/workflow.js",
                    "/playback.js",
                    "/stats.js",
                    "/style.css",
                }:
                    name = "index.html" if url.path == "/" else url.path[1:]
                    mime = {
                        "index.html": "text/html; charset=utf-8",
                        "app.js": "text/javascript; charset=utf-8",
                        "style.css": "text/css; charset=utf-8",
                    }.get(name, "text/javascript; charset=utf-8")
                    self.send(200, (ASSETS / name).read_bytes(), mime)
                elif url.path == "/api/catalog":
                    self.send(
                        200,
                        {
                            "name": project.manifest["name"],
                            "videos": project.catalog(),
                            "token": token,
                            "project_key": sha(str(project.root).encode())[:16],
                        },
                    )
                elif url.path == "/api/stats":
                    self.send(200, project.stats())
                elif url.path == "/api/export":
                    self.send(200, export_project(project), "application/zip")
                elif url.path == "/api/import-candidates":
                    self.send(200, {"items": collection_candidates(project)})
                elif len(parts) >= 3 and parts[:2] == ["api", "video"]:
                    vid = parts[2]
                    video = project.videos[vid]
                    if len(parts) == 3:
                        self.send(
                            200,
                            {
                                "video": {
                                    k: v
                                    for k, v in video.items()
                                    if k not in {"path", "candidates"}
                                },
                                "state": project.state(vid),
                                "preferred_source": project.preferred_source(vid),
                                "grid": token_grid(video),
                                "sources": project.sources(vid),
                                "candidates": [
                                    {"index": i, "label": Path(p).parent.name}
                                    for i, p in enumerate(video.get("candidates", []))
                                ],
                            },
                        )
                    elif len(parts) == 4 and parts[3] == "units":
                        self.send(200, {**project.protocol_view(vid), "grid": token_grid(video)})
                    elif len(parts) == 4 and parts[3] == "preload":
                        self.send(200, preloader.snapshot(vid))
                    elif len(parts) == 4 and parts[3] == "frame":
                        index = int(parse_qs(url.query).get("n", ["0"])[0])
                        self.send(200, frames.get(video, index), "image/jpeg")
                    elif len(parts) == 5 and parts[3] == "source":
                        meta = next(
                            s["metadata"] for s in project.sources(vid) if s["id"] == parts[4]
                        )
                        self.send(
                            200,
                            {**project.source_view(vid, parts[4]), "grid": token_grid(video, meta)},
                        )
                    else:
                        self.send(404, {"error": "Resource not found."})
                else:
                    self.send(404, {"error": "Resource not found."})
            except (ValueError, KeyError, TypeError, StopIteration, OSError, sqlite3.Error) as exc:
                self.send(400, {"error": str(exc)})

        def do_POST(self) -> None:
            origin = self.headers.get("Origin")
            expected = f"http://{self.headers.get('Host')}"
            if (
                not self.safe_host()
                or (origin and origin != expected)
                or self.headers.get("X-Annotation-Token") != token
            ):
                self.send(403, {"error": "Invalid session. Reload the page."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 25 * 1024**2:
                    raise ValueError("Empty request or larger than 25 MiB.")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("The request must be a JSON object.")
                parts = urlsplit(self.path).path.strip("/").split("/")
                if parts == ["api", "import-preview"]:
                    self.send(200, import_pair(project, body))
                    return
                if parts == ["api", "import-pair"]:
                    self.send(200, import_pair(project, body, commit=True))
                    return
                if parts == ["api", "import-defaults"]:
                    self.send(200, project.set_source_defaults(body["items"]))
                    return
                if len(parts) not in (3, 4) or parts[:2] != ["api", "video"]:
                    self.send(404, {"error": "Resource not found."})
                    return
                vid = parts[2]
                if len(parts) == 3:
                    self.send(200, project.save(vid, body))
                elif parts[3] == "preload":
                    current = body.get("frame", 0)
                    if (
                        type(current) is not int
                        or not 0 <= current < project.videos[vid]["num_frames"]
                    ):
                        raise ValueError("Invalid preload frame.")
                    self.send(200, preloader.start(project.videos[vid], current))
                elif parts[3] == "import":
                    if "candidate" in body:
                        candidates = project.videos[vid].get("candidates", [])
                        index = body["candidate"]
                        if type(index) is not int or not 0 <= index < len(candidates):
                            raise ValueError("Source not registered in the project.")
                        csv_path = Path(candidates[index])
                        csv_text = csv_path.read_bytes().decode("utf-8")
                        metadata = csv_path.with_name("boxes.json").read_bytes().decode("utf-8")
                        label = f"{csv_path.parent.name} · {csv_path.parents[2].name}"
                    else:
                        csv_text, metadata, label = body["csv"], body["metadata"], body["label"]
                    source_id = project.add_source(vid, metadata, csv_text, label)
                    self.send(200, {"id": source_id, "sources": project.sources(vid)})
                else:
                    self.send(404, {"error": "Resource not found."})
            except ConflictError as exc:
                self.send(409, {"error": str(exc)})
            except (ValueError, KeyError, TypeError, IndexError, OSError, sqlite3.Error) as exc:
                self.send(400, {"error": str(exc)})

    class LocalServer(ThreadingHTTPServer):
        # On Windows SO_REUSEADDR lets two processes bind the same port; the existing instance
        # must be recognised and reopened instead.
        allow_reuse_address = False

        def server_close(self) -> None:
            preloader.close()
            with frames.lock:
                if frames.container is not None:
                    frames.container.close()
            super().server_close()

        def server_bind(self) -> None:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            super().server_bind()

    server = LocalServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main() -> None:
    """Start or initialize the local editor from the data in the workspace."""
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[2]
    parser.add_argument("--project", type=Path, default=repo / "runs/annotazioni_spaziali")
    parser.add_argument(
        "--init-local", action="store_true", help="Create the UCF test + MIVIA test project"
    )
    parser.add_argument(
        "--proposal-root",
        type=Path,
        default=repo.parent / "presentazione_localizzazione_sorgenti/cluster",
    )
    parser.add_argument(
        "--add-proposals",
        type=Path,
        metavar="FOLDER",
        help="Register the boxes.json/CSV pairs found in the folder in the project and exit",
    )
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--recover", type=Path, help="Apply a browser recovery copy and exit")
    parser.add_argument(
        "--restore-revision",
        nargs=2,
        metavar=("VIDEO_ID", "REVISION"),
        help="Restore a revision keeping the history and exit",
    )
    parser.add_argument(
        "--relink",
        nargs=2,
        metavar=("VIDEO_ID", "FILE"),
        help="Relink a video after checking its SHA256 and exit",
    )
    args = parser.parse_args()
    if args.init_local:
        manifest = create_local_project(args.project, repo.parent, args.proposal_root)
        print(f"Project created: {len(manifest['videos'])} videos.", flush=True)
    if args.add_proposals:
        added, removed = add_candidates(args.project, args.add_proposals)
        print(
            f"Candidates registered: {added}, removed because missing from the folder: {removed}. "
            "Restart the server to see them."
        )
        return
    project = Project(args.project)
    if args.restore_revision:
        vid, number = args.restore_revision
        body = project.revision(vid, int(number))
        body["revision"] = project.state(vid)["revision"]
        result = project.save(vid, body)
        print(f"Revision {number} restored as {result['revision']}. History kept.")
        return
    if args.recover:
        result = project.recover(args.recover)
        print(f"Copy recovered into revision {result['revision']}. History kept.")
        return
    if args.relink:
        project.relink(args.relink[0], Path(args.relink[1]))
        print("Video relinked: content hash verified. Restart the server.")
        return
    try:
        server = make_server(project, args.port)
    except OSError:
        url = f"http://127.0.0.1:{args.port}"
        try:
            with urllib.request.urlopen(url + "/api/catalog", timeout=3) as response:
                existing = json.load(response)
        except (OSError, ValueError):
            existing = {}
        if existing.get("project_key") != sha(str(project.root).encode())[:16]:
            raise
        print(f"The project is already open: {url}")
        if not args.no_browser:
            webbrowser.open(url)
        return
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"Annotator: {url}\nProject: {args.project.resolve()}\nCtrl+C to stop.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
