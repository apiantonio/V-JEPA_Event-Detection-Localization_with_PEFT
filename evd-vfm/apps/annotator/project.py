"""Projects, VTCD imports and spatial reviews, separate from the temporal annotations.

Contract in ``docs/20_annotatore_spaziale.md``. Pixels follow the inclusive extremes of
``evd.explain.boxes.in_pixel``; frames are zero-based. The renderer CSV is the authority
for the frame/window assignment: it is not rebuilt from ``boxes.json`` alone.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evd.annotations.ucf_lab import FS_UCF_CRIME, MIVIA_CRIME, UcfLabAnnotations
from evd.utils.io import write_json


def sha(data: bytes) -> str:
    """:return: SHA256 of the original bytes, used to identify sources."""
    return hashlib.sha256(data).hexdigest()


def now() -> str:
    """:return: UTC timestamp of the revision."""
    return datetime.now(timezone.utc).isoformat()


def event_classes(video: dict, frame: int) -> list[str]:
    """:return: classes of the events including the frame, never the folder class."""
    return sorted({e["class_name"] for e in video["events"] if e["start"] <= frame <= e["end"]})


def validate_box(box: dict, video: dict, frame: int) -> None:
    """Validate an inclusive native box and its temporal class.

    :raises ValueError: on incompatible coordinates, identity or label.
    """
    if not isinstance(box, dict):
        raise ValueError("The box must be a JSON object.")
    coords = box.get("xyxy", [])
    if len(coords) != 4 or any(type(x) is not int for x in coords):
        raise ValueError("The edges must be four integer pixels.")
    x0, y0, x1, y1 = coords
    if not (0 <= x0 <= x1 < video["width"] and 0 <= y0 <= y1 < video["height"]):
        raise ValueError("Box outside the frame or with inverted edges.")
    if not isinstance(box.get("id"), str) or not 1 <= len(box["id"]) <= 100:
        raise ValueError("Missing box identity.")
    if box.get("class_name") not in event_classes(video, frame):
        raise ValueError("The box class must belong to an active event.")
    if "display_id" in box and (
        type(box["display_id"]) is not int or not 1 <= box["display_id"] <= 999999
    ):
        raise ValueError("The visible ID must be an integer between 1 and 999999.")
    if "color" in box and (
        not isinstance(box["color"], str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", box["color"])
    ):
        raise ValueError("Invalid box colour: use #RRGGBB.")
    if not isinstance(box.get("track_id", ""), str) or len(box.get("track_id", "")) > 100:
        raise ValueError("Invalid track identity.")
    if box.get("origin") not in {"manual", "pseudo", "propagated", "interpolated"}:
        raise ValueError("Invalid box origin.")


def import_proposals(video: dict, metadata: dict, csv_text: str, source_id: str) -> dict:
    """Import the per-frame CSV and verify its geometry, video and windows.

    :return: per-frame proposals, without any review state.
    :raises ValueError: if the format differs from the renderer contract or data are
        inconsistent.
    """
    if metadata.get("schema") not in (1, 2) or metadata.get("geometry") != "squash":
        raise ValueError("Expected boxes.json schema 1 or 2 and squash geometry.")
    if metadata.get("variante") != "annotati":
        raise ValueError("Import the annotated variant with its boxes_per_frame.csv.")
    native = metadata["nativo"]
    for name in ("width", "height", "num_frames"):
        if native[name] != video[name]:
            raise ValueError(f"Geometry differs from the video: {name}.")
    if not math.isclose(float(native["fps"]), video["fps"], rel_tol=1e-6):
        raise ValueError("Source FPS differs from the video.")
    # Schema 2 (one video, the boxes of the dictionary of each event class, 4 October 2026):
    # a window appears once per class, and the key is the pair.
    composite = metadata["schema"] == 2
    windows = window_index(metadata)
    reader = csv.DictReader(io.StringIO(csv_text.lstrip("\ufeff")))
    required = {"video_id", "frame", "x0", "y0", "x1", "y1", "window_id", "piano", "box"}
    if not required.issubset(reader.fieldnames or []) or (
        composite and "classe" not in (reader.fieldnames or [])
    ):
        raise ValueError("Incomplete boxes_per_frame.csv header.")
    frames: dict[str, list] = {}
    seen = set()
    for row in reader:
        frame, plane, number = int(row["frame"]), int(row["piano"]), int(row["box"])
        if row["video_id"] != video["video_id"] or not 0 <= frame < video["num_frames"]:
            raise ValueError("The CSV contains another video or a frame out of range.")
        classes = event_classes(video, frame)
        if not classes:
            raise ValueError(f"Frame {frame} outside the annotated events.")
        if metadata.get("classi") and not set(metadata["classi"]) & set(classes):
            raise ValueError(f"Frame {frame} outside the events of the source classes.")
        if composite and row["classe"] not in classes:
            raise ValueError(f"Frame {frame}: class {row['classe']} differs from its event.")
        window = windows.get((row["window_id"], row["classe"] if composite else None))
        if window is None:
            raise ValueError("CSV window missing from boxes.json.")
        planes = {p["t"]: p["box"] for p in window["piani"]}
        if plane not in planes or not 1 <= number <= len(planes[plane]):
            raise ValueError("Plane or box count inconsistent with boxes.json.")
        xyxy = [int(row[k]) for k in ("x0", "y0", "x1", "y1")]
        if xyxy != planes[plane][number - 1]["nativo"]:
            raise ValueError("CSV coordinates differ from boxes.json: unmatched sources.")
        identity = (frame, row["window_id"], plane, number)
        if identity in seen:
            raise ValueError("Duplicated box in the CSV.")
        seen.add(identity)
        box = {
            "id": f"p-{plane}-{number}",
            "xyxy": xyxy,
            "class_name": row["classe"] if composite else classes[0],
            "track_id": "",
            "origin": "pseudo",
            "source_id": source_id,
            "window_id": row["window_id"],
            "plane": plane,
            "source_box": number,
            "class_ambiguous": len(classes) > 1,
        }
        validate_box(box, video, frame)
        frames.setdefault(str(frame), []).append(box)
    return frames


def window_index(metadata: dict) -> dict[tuple[str, str | None], dict]:
    """Return the ``boxes.json`` windows keyed by ``(window_id, classe)``.

    In schema 1 the class is ``None``; in schema 2 a window straddling two events of
    different classes appears once per class, with the boxes of its dictionary.

    :raises ValueError: on a repeated key.
    """
    composite = metadata.get("schema") == 2
    index = {}
    for window in metadata["finestre"]:
        key = (window["window_id"], window["classe"] if composite else None)
        if key in index:
            raise ValueError(f"Repeated window in boxes.json: {key}.")
        index[key] = window
    return index


def find_candidates(proposal_root: Path | None) -> dict[str, list[str]]:
    """Return the ``boxes.json``/``boxes_per_frame.csv`` pairs under a folder, per video.

    :return: ``video_id -> absolute CSV paths``; empty if the folder does not exist.
    """
    candidates: dict[str, list[str]] = {}
    if proposal_root and proposal_root.is_dir():
        for csv_path in sorted(proposal_root.rglob("boxes_per_frame.csv")):
            # Only folders BELOW the root: a hidden folder above it (`.claude`, a user profile) does
            # not hide the proposals.
            relative = csv_path.relative_to(proposal_root).parts
            if any(p.startswith(".") or p == "archivio" for p in relative):
                continue
            if not csv_path.with_name("boxes.json").is_file():
                continue
            with csv_path.open(encoding="utf-8-sig", newline="") as f:
                first = next(csv.DictReader(f), None)
            video_id = first["video_id"] if first else _window_videos(csv_path)
            if video_id:
                candidates.setdefault(video_id, []).append(str(csv_path.resolve()))
    return candidates


def _window_videos(csv_path: Path) -> str | None:
    """Return the video of a pair without boxes, from the ``boxes.json`` windows.

    A CSV with only the header (the chosen concept never appears during the events) is
    still a source: it carries the planes, which are reviewed and confirmed empty or
    annotated by hand. A window ID is ``<video>@<end>x<samples>i<profile>``.

    :return: the video ID, or ``None`` if ``boxes.json`` has no windows.
    """
    metadata = json.loads(csv_path.with_name("boxes.json").read_text(encoding="utf-8-sig"))
    videos = {w["window_id"].rsplit("@", 1)[0] for w in metadata.get("finestre") or ()}
    if len(videos) > 1:
        raise ValueError(f"Windows of several videos in {csv_path.with_name('boxes.json')}.")
    return next(iter(videos), None)


def add_candidates(root: Path, proposal_root: Path) -> tuple[int, int]:
    """Align the candidates of an existing project with the pairs under ``proposal_root``.

    For sources produced after initialization (per-class VTCD, 3 October 2026). New pairs
    are registered and candidates under ``proposal_root`` that no longer exist are removed;
    those outside the folder stay. No source is imported and no annotation changes.
    ``project.json`` is rewritten after a ``project.prima_candidati_<time>.json`` copy.

    :return: ``(candidates added, candidates removed)``.
    :raises ValueError: if the project does not exist.
    """
    path = root / "project.json"
    if not path.is_file():
        raise ValueError("Missing project: initialize it first with --init-local.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    found = find_candidates(proposal_root)
    base = proposal_root.resolve()
    added = removed = 0
    for video in manifest["videos"]:
        present = set(found.get(video["video_id"], []))
        kept = [
            c
            for c in video.get("candidates", [])
            if c in present or not Path(c).resolve().is_relative_to(base)
        ]
        removed += len(video.get("candidates", [])) - len(kept)
        new = [c for c in found.get(video["video_id"], []) if c not in set(kept)]
        video["candidates"] = [*kept, *new]
        added += len(new)
    if added or removed:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        (root / f"project.prima_candidati_{stamp}.json").write_bytes(path.read_bytes())
        write_json(path, manifest)
    return added, removed


def create_local_project(root: Path, workspace: Path, proposal_root: Path | None = None) -> dict:
    """Create a project from the local test sets and the existing temporal adapters.

    :param root: new project directory; existing projects are not overwritten.
    :param workspace: ``TESI/codice``, containing ``data/`` and ``evd-vfm/``.
    :return: manifest with inventories, splits, events and delivery hashes.
    :raises ValueError: if the project exists, or videos or annotations are missing.
    """
    if (root / "project.json").exists():
        raise ValueError("The project already exists: start it without initializing it.")
    repo = workspace / "evd-vfm"
    split_path = repo / "manifests/edp_v2/ucf_crime/split.json"
    test_ids = set(json.loads(split_path.read_text(encoding="utf-8"))["test"])
    candidates = find_candidates(proposal_root)
    specs = [
        (
            "ucf_crime",
            workspace / "data/ucf_crime_inventory/inventory.csv",
            workspace / "data/annotations/fs_ucf_crime",
            workspace / "data/raw",
            FS_UCF_CRIME,
        ),
        (
            "mivia_crime",
            repo / "data/mivia_crime_inventory/inventory.csv",
            workspace / "data/mivia_crime",
            workspace / "data/mivia_crime",
            MIVIA_CRIME,
        ),
    ]
    videos = []
    provenance = {"split_sha256": sha(split_path.read_bytes())}
    for dataset, inventory, annotation_dir, mean, delivery in specs:
        with inventory.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        lengths = {r["video_id"]: int(r["num_frames"]) for r in rows}
        adapter = UcfLabAnnotations.from_dir(annotation_dir, delivery=delivery, num_frames=lengths)
        temporal = adapter.load()
        classes = {c.idx: c.name for c in adapter.classes()}
        provenance[dataset] = {
            "inventory_sha256": sha(inventory.read_bytes()),
            "annotations_sha256": sha(adapter.path.read_bytes()),
            "classes_sha256": sha((annotation_dir / delivery.classes_name).read_bytes()),
        }
        for row in rows:
            vid = row["video_id"]
            if dataset == "ucf_crime" and vid not in test_ids:
                continue
            relative = Path(row["relative_path"])
            if relative.parts[0] == "Videos":
                relative = Path(*relative.parts[1:])
            path = mean / relative
            if vid not in temporal:
                raise ValueError(f"Missing annotations: {dataset}/{vid}")
            events = [
                {"start": a, "end": b, "class_name": classes[c]}
                for a, b, c in temporal[vid].intervals
            ]
            videos.append(
                {
                    "id": f"{dataset}--{vid}",
                    "video_id": vid,
                    "dataset": dataset,
                    "split": "test",
                    "path": str(path.resolve()),
                    "available": path.is_file(),
                    "sha256": row["sha256"],
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                    "num_frames": lengths[vid],
                    "fps": float(row["fps"]),
                    "events": events,
                    "candidates": candidates.get(vid, []),
                }
            )
    manifest = {
        "schema": 1,
        "name": "Localization · Ground truth",
        "created_at": now(),
        "coordinate_convention": "native_pixels_xyxy_inclusive_zero_based",
        "provenance": provenance,
        "videos": videos,
    }
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "project.json", manifest)
    return manifest


class ConflictError(ValueError):
    """The submitted revision is no longer the one on disk."""


class Project:
    """Keep immutable sources and revisions with SQLite transactions.

    The database is local, with optimistic control for concurrent tabs. Every save keeps
    the previous version for recovery.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.manifest = json.loads((root / "project.json").read_text(encoding="utf-8"))
        self.videos = {v["id"]: v for v in self.manifest["videos"]}
        self._protocol_views: dict[str, dict] = {}
        self.db = root / "annotations.sqlite3"
        with self.connect() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS states(video TEXT PRIMARY KEY, revision INTEGER, body TEXT);
                CREATE TABLE IF NOT EXISTS history(video TEXT, revision INTEGER, body TEXT,
                                                  PRIMARY KEY(video, revision));
                CREATE TABLE IF NOT EXISTS sources(video TEXT, id TEXT, label TEXT, metadata TEXT,
                                                  csv TEXT, frames TEXT, PRIMARY KEY(video, id));
                CREATE TABLE IF NOT EXISTS source_units(video TEXT, id TEXT, body TEXT,
                                                       PRIMARY KEY(video, id));
                CREATE TABLE IF NOT EXISTS source_defaults(video TEXT PRIMARY KEY, source_id TEXT);
                CREATE TABLE IF NOT EXISTS protocol_units(video TEXT PRIMARY KEY, fingerprint TEXT,
                                                         units INTEGER, body TEXT);
            """)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """:return: transactional connection with a timeout for concurrent writers."""
        connection = sqlite3.connect(self.db, timeout=15)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def state(self, video_id: str) -> dict:
        """:return: current state; a missing frame means not reviewed."""
        if video_id not in self.videos:
            raise ValueError("Video not in the project.")
        with self.connect() as con:
            row = con.execute("SELECT body FROM states WHERE video=?", (video_id,)).fetchone()
        return (
            json.loads(row[0])
            if row
            else {"revision": 0, "frames": {}, "last_frame": 0, "reviewer": "", "source_id": ""}
        )

    def sources(self, video_id: str) -> list[dict]:
        """:return: imported sources of a video, identified by content."""
        with self.connect() as con:
            rows = con.execute("SELECT id,label,metadata FROM sources WHERE video=?", (video_id,))
            return [
                {"id": r[0], "label": r[1], "metadata": json.loads(r[2].lstrip("\ufeff"))}
                for r in rows
            ]

    def preferred_source(self, video_id: str) -> str:
        """:return: starting source chosen in bulk, without changing corrections."""
        with self.connect() as con:
            row = con.execute(
                "SELECT source_id FROM source_defaults WHERE video=?", (video_id,)
            ).fetchone()
        return row[0] if row else ""

    def set_source_defaults(self, items: list[dict]) -> dict:
        """Set one source per video, only where no human annotations exist.

        :return: updated videos and already annotated videos left unchanged.
        :raises ValueError: on missing sources or several choices for the same video.
        """
        if not isinstance(items, list) or len(items) > len(self.videos):
            raise ValueError("Invalid video list.")
        applied, skipped, seen = [], [], set()
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            for item in items:
                vid, sid = item["video"], item["source_id"]
                if (
                    vid in seen
                    or not con.execute(
                        "SELECT 1 FROM sources WHERE video=? AND id=?", (vid, sid)
                    ).fetchone()
                ):
                    raise ValueError("Missing source or repeated video in the starting choice.")
                seen.add(vid)
                row = con.execute("SELECT body FROM states WHERE video=?", (vid,)).fetchone()
                if row and json.loads(row[0])["frames"]:
                    skipped.append(vid)
                    continue
                con.execute("INSERT OR REPLACE INTO source_defaults VALUES (?,?)", (vid, sid))
                applied.append(vid)
        return {"applied": applied, "skipped": skipped}

    def source(self, video_id: str, source_id: str) -> dict:
        """:return: per-frame proposals of an already imported source.

        :raises ValueError: if the source is not registered for this video.
        """
        with self.connect() as con:
            row = con.execute(
                "SELECT frames FROM sources WHERE video=? AND id=?", (video_id, source_id)
            ).fetchone()
        if not row:
            raise ValueError("Source not present.")
        return json.loads(row[0])

    def add_source(self, video_id: str, metadata_text: str, csv_text: str, label: str) -> str:
        """Freeze a JSON/CSV pair; a new configuration becomes a new source.

        :return: SHA256 identity of the original pair.
        :raises ValueError: on an invalid pair or an empty label.
        """
        if not label.strip() or len(label) > 200:
            raise ValueError("Give the source a name (at most 200 characters).")
        source_id = sha(metadata_text.encode() + b"\0" + csv_text.encode())
        metadata = json.loads(metadata_text.lstrip("\ufeff"))
        frames = import_proposals(self.videos[video_id], metadata, csv_text, source_id)
        with self.connect() as con:
            con.execute(
                "INSERT OR IGNORE INTO sources VALUES (?,?,?,?,?,?)",
                (video_id, source_id, label.strip(), metadata_text, csv_text, json.dumps(frames)),
            )
        return source_id

    def source_view(self, video_id: str, source_id: str) -> dict:
        """Return proposals and plane intervals, including planes without boxes.

        :return: original frames and review units persisted per source.
        :raises ValueError: if the mapping is incompatible with the proposals.
        """
        from apps.annotator.planes import original_manifest, temporal_units

        frames = self.source(video_id, source_id)
        with self.connect() as con:
            cached = con.execute(
                "SELECT body FROM source_units WHERE video=? AND id=?", (video_id, source_id)
            ).fetchone()
            if cached:
                return {"frames": frames, **json.loads(cached[0])}
            row = con.execute(
                "SELECT metadata,csv FROM sources WHERE video=? AND id=?", (video_id, source_id)
            ).fetchone()
        metadata = json.loads(row[0].lstrip("\ufeff"))
        manifest = original_manifest(self.videos[video_id], metadata, row[1])
        timeline = temporal_units(self.videos[video_id], metadata, frames, manifest=manifest)
        with self.connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO source_units VALUES (?,?,?)",
                (video_id, source_id, json.dumps(timeline)),
            )
        return {"frames": frames, **timeline}

    def protocol_view(self, video_id: str) -> dict:
        """Return the base VTCD planes of the video, for work without a source.

        They are not an imported source and do not enter ``source_units``. The computation is
        kept in ``protocol_units`` with its fingerprint and redone when algorithm, protocol or
        video events change.

        :return: review units and provenance, without proposals.
        """
        from apps.annotator.planes import protocol_fingerprint, protocol_units

        video = self.videos[video_id]
        fingerprint = protocol_fingerprint(video)
        cached = self._protocol_views.get(video_id)
        if not cached or cached["provenance"]["fingerprint"] != fingerprint:
            with self.connect() as con:
                row = con.execute(
                    "SELECT body FROM protocol_units WHERE video=? AND fingerprint=?",
                    (video_id, fingerprint),
                ).fetchone()
            if row:
                cached = json.loads(row[0])
            else:
                cached = protocol_units(video)
                with self.connect() as con:
                    con.execute(
                        "INSERT OR REPLACE INTO protocol_units VALUES (?,?,?,?)",
                        (video_id, fingerprint, len(cached["units"]), json.dumps(cached)),
                    )
            self._protocol_views[video_id] = cached
        return {"frames": {}, **cached}

    def plane_counts(self) -> dict[str, int]:
        """Return the number of base planes of every video, from the cache or computed once.

        :return: video -> review units of its events (0 without events).
        """
        from apps.annotator.planes import protocol_fingerprint

        with self.connect() as con:
            stored = {
                vid: (fingerprint, units)
                for vid, fingerprint, units in con.execute(
                    "SELECT video,fingerprint,units FROM protocol_units"
                )
            }
        counts = {}
        for vid, video in self.videos.items():
            if not video["events"]:
                counts[vid] = 0
                continue
            fingerprint, units = stored.get(vid, (None, 0))
            if fingerprint != protocol_fingerprint(video):
                units = len(self.protocol_view(vid)["units"])
            counts[vid] = units
        return counts

    def save(self, video_id: str, body: dict) -> dict:
        """Validate and save a revision, without automatically promoting proposals.

        :return: saved state with updated revision and temporal attribution.
        :raises ConflictError: if another tab saved first.
        :raises ValueError: on invalid frames, boxes, reviewer or sources.
        """
        video = self.videos[video_id]
        source_metadata = {s["id"]: s["metadata"] for s in self.sources(video_id)}
        sources = set(source_metadata)
        reviewer = body.get("reviewer", "").strip()
        if not reviewer or len(reviewer) > 100:
            raise ValueError("Enter the reviewer name before saving.")
        if body.get("source_id", "") not in sources | {""}:
            raise ValueError("Selected source not registered.")
        if (
            type(body.get("last_frame")) is not int
            or not 0 <= body["last_frame"] < video["num_frames"]
        ):
            raise ValueError("Current frame outside the video.")
        if not isinstance(body.get("frames"), dict):
            raise ValueError("Invalid frame list.")
        from apps.annotator.geometry import token_box, token_grid

        source_units, grids, projected = {}, {}, {}
        for key, item in body["frames"].items():
            if not isinstance(item, dict):
                raise ValueError("The frame must be a JSON object.")
            if not key.isdigit() or str(int(key)) != key or not 0 <= int(key) < video["num_frames"]:
                raise ValueError("Invalid frame index.")
            if item.get("status") not in {"draft", "reviewed", "excluded"}:
                raise ValueError("Invalid review state.")
            scope = item.get("review_unit")
            if scope:
                # Empty source: base protocol planes (work from scratch).
                sid = scope.get("source_id") or ""
                if sid not in sources | {""}:
                    raise ValueError("Plane source not registered.")
                if sid not in source_units:
                    view = self.source_view(video_id, sid) if sid else self.protocol_view(video_id)
                    source_units[sid] = {u["id"]: u for u in view["units"]}
                unit = source_units[sid].get(scope.get("id"))
                if (
                    not unit
                    or not unit["known"]
                    or not unit["start"] <= int(key) <= unit["end"]
                    or scope.get("start") != unit["start"]
                    or scope.get("end") != unit["end"]
                ):
                    raise ValueError("The frame does not belong to the declared review plane.")
            if not isinstance(item.get("note", ""), str) or len(item.get("note", "")) > 2000:
                raise ValueError("Note too long.")
            if item["status"] == "excluded" and not item.get("note", "").strip():
                raise ValueError("Specify why the frame is excluded.")
            if not isinstance(item.get("boxes"), list) or len(item["boxes"]) > 500:
                raise ValueError("Invalid box list.")
            ids, display_ids = set(), set()
            for index, box in enumerate(item["boxes"], 1):
                validate_box(box, video, int(key))
                if box["id"] in ids:
                    raise ValueError("Duplicated box identities in the frame.")
                ids.add(box["id"])
                display_id = box.get("display_id", index)
                if display_id in display_ids:
                    raise ValueError("Duplicated visible IDs in the same plane/frame.")
                display_ids.add(display_id)
                if box.get("source_id") and box["source_id"] not in sources:
                    raise ValueError("Box provenance not registered.")
                if box["origin"] == "pseudo" and not box.get("source_id"):
                    raise ValueError("An accepted proposal must keep its source.")
                if item["status"] == "reviewed" and box.get("class_ambiguous"):
                    raise ValueError("Choose the box class for overlapping events.")
                sid = (scope or {}).get("source_id") or body.get("source_id", "")
                if sid not in grids:
                    grids[sid] = token_grid(video, source_metadata.get(sid))
                projection_key = (sid, tuple(box["xyxy"]))
                if projection_key not in projected:
                    projected[projection_key] = token_box(box["xyxy"], grids[sid])
                # Derived data: never trust the version sent by the browser.
                box["token_space"] = projected[projection_key]
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT revision,body FROM states WHERE video=?", (video_id,)
            ).fetchone()
            revision, old = (row[0], json.loads(row[1])) if row else (0, {"frames": {}})
            if type(body.get("revision")) is not int or body["revision"] != revision:
                raise ConflictError(
                    "This video was modified in another tab. Export the recovery copy and reload."
                )
            stamp = now()
            for key, item in body["frames"].items():
                before = old["frames"].get(key)
                # Attribution comes from the server, never from browser timestamps.
                content = {k: v for k, v in item.items() if k not in {"updated_at", "reviewer"}}
                previous = {
                    k: v for k, v in (before or {}).items() if k not in {"updated_at", "reviewer"}
                }
                item["updated_at"] = (
                    before["updated_at"] if content == previous and before else stamp
                )
                item["reviewer"] = (
                    before["reviewer"] if content == previous and before else reviewer
                )
            result = {
                "revision": revision + 1,
                "frames": body["frames"],
                "reviewer": reviewer,
                "last_frame": body["last_frame"],
                "source_id": body.get("source_id", ""),
                "updated_at": stamp,
            }
            serialized = json.dumps(result, ensure_ascii=False, allow_nan=False)
            # Keep only the changed frames, avoiding quadratic growth of the history.
            delta = {k: v for k, v in result.items() if k != "frames"}
            delta["changed_frames"] = {
                k: v for k, v in result["frames"].items() if old["frames"].get(k) != v
            }
            delta["removed_frames"] = sorted(set(old["frames"]) - set(result["frames"]))
            con.execute(
                "INSERT INTO history VALUES (?,?,?)",
                (video_id, revision + 1, json.dumps(delta, ensure_ascii=False, allow_nan=False)),
            )
            con.execute(
                "INSERT OR REPLACE INTO states VALUES (?,?,?)", (video_id, revision + 1, serialized)
            )
            # After the first save the reviewer choice overrides the batch default, even when
            # working from scratch without boxes yet.
            con.execute("DELETE FROM source_defaults WHERE video=?", (video_id,))
        return result

    def revision(self, video_id: str, number: int) -> dict:
        """Rebuild a revision from the incremental history.

        :return: full state at the requested revision.
        :raises ValueError: if the revision does not exist.
        """
        result = {"frames": {}}
        with self.connect() as con:
            rows = con.execute(
                "SELECT revision,body FROM history WHERE video=? AND revision<=? ORDER BY revision",
                (video_id, number),
            ).fetchall()
        if not rows or rows[-1][0] != number:
            raise ValueError("Revision not in the history.")
        for _, serialized in rows:
            item = json.loads(serialized)
            if "frames" in item:
                result = item
            else:
                for key in item.pop("removed_frames"):
                    result["frames"].pop(key, None)
                result["frames"].update(item.pop("changed_frames"))
                result.update(item)
        return result

    def recover(self, path: Path) -> dict:
        """Recover a browser copy as a new revision, preserving the history.

        :return: the new saved revision.
        :raises ValueError: if video, hash or state are incompatible with this project.
        """
        payload = json.loads(path.read_text(encoding="utf-8"))
        video = payload["video"]
        vid = video["id"]
        if vid not in self.videos or video["sha256"] != self.videos[vid]["sha256"]:
            raise ValueError("The recovery copy belongs to another video or content.")
        body = payload["state"]
        body["revision"] = self.state(vid)["revision"]
        return self.save(vid, body)

    def relink(self, video_id: str, path: Path) -> None:
        """Update the video path only if the bytes have the inventory hash.

        :raises ValueError: if the video is not registered or the content differs.
        """
        if video_id not in self.videos:
            raise ValueError("Video not registered in the project.")
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != self.videos[video_id]["sha256"]:
            raise ValueError("The content does not match the original inventory hash.")
        self.videos[video_id]["path"] = str(path.resolve())
        self.videos[video_id]["available"] = True
        write_json(self.root / "project.json", self.manifest)

    def stats(self) -> dict:
        """Return the annotation progress over all videos with events.

        :return: one entry per video (``stats.video_stats``), the videos without events per
            dataset and the computation time.
        """
        from apps.annotator.stats import video_stats

        with self.connect() as con:
            states = {r[0]: json.loads(r[1]) for r in con.execute("SELECT video,body FROM states")}
        rows, normal = [], {}
        for vid, video in self.videos.items():
            if not video["events"]:
                normal[video["dataset"]] = normal.get(video["dataset"], 0) + 1
                continue
            units = self.protocol_view(vid)["units"]
            rows.append(video_stats(video, states.get(vid, {"frames": {}}), units))
        return {"videos": rows, "normal": normal, "generated_at": now()}

    def catalog(self) -> list[dict[str, Any]]:
        """:return: videos, sources, reviewed frames and base VTCD planes."""
        with self.connect() as con:
            states = {r[0]: json.loads(r[1]) for r in con.execute("SELECT video,body FROM states")}
            sources = dict(con.execute("SELECT video,COUNT(*) FROM sources GROUP BY video"))
        planes = self.plane_counts()
        rows = []
        for vid, v in self.videos.items():
            # Files may have arrived after the project was created.
            v["available"] = Path(v["path"]).is_file()
            frames = states.get(vid, {}).get("frames", {})
            rows.append(
                {k: val for k, val in v.items() if k not in {"path", "candidates"}}
                | {
                    "sources": sources.get(vid, 0) + len(v.get("candidates", [])),
                    "reviewed": sum(f["status"] == "reviewed" for f in frames.values()),
                    "drafts": sum(f["status"] == "draft" for f in frames.values()),
                    "planes": planes[vid],
                }
            )
        return rows
