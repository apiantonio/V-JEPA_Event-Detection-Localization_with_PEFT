"""On-disk JPEG cache and progressive preloading of the events of the open video.

The foreground decoder stays independent: preloading neither blocks its cursor nor
replaces native ordinals with approximate seeks. The cache can be regenerated, is
bounded, and is not part of the annotation backup.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from pathlib import Path


def encode_frame(frame, video: dict) -> bytes:
    """:return: native JPEG; rejects resolution changes with respect to the project."""
    import cv2

    if frame.width != video["width"] or frame.height != video["height"]:
        raise ValueError("File resolution differs from the project.")
    ok, encoded = cv2.imencode(
        ".jpg", frame.to_ndarray(format="bgr24"), [cv2.IMWRITE_JPEG_QUALITY, 94]
    )
    if not ok:
        raise ValueError("Cannot encode the frame.")
    return encoded.tobytes()


class FrameStore:
    """Cache shared by the two decoders, bounded by default to 1 GiB on disk."""

    def __init__(self, root: Path, budget: int = 1024**3) -> None:
        self.root, self.budget = root, budget
        root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.entries = OrderedDict(
            (p.name, p.stat().st_size)
            for p in sorted(root.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
        )
        self.size = sum(self.entries.values())
        with self.lock:
            self._trim(0)

    def _trim(self, extra: int) -> None:
        while self.entries and self.size + extra > self.budget:
            name, size = self.entries.popitem(last=False)
            (self.root / name).unlink(missing_ok=True)
            self.size -= size

    @staticmethod
    def identity(video: dict) -> str:
        """:return: key invalidated by a changed content, path, geometry or file."""
        path = Path(video["path"])
        stat = path.stat()
        value = (
            str(path.resolve()),
            video.get("sha256"),
            stat.st_size,
            stat.st_mtime_ns,
            video["width"],
            video["height"],
            "jpeg94-v1",
        )
        return hashlib.sha256(repr(value).encode()).hexdigest()[:32]

    def get(self, identity: str, index: int) -> bytes | None:
        """:return: the cached JPEG, or None; updates the retention priority."""
        name = f"{identity}-{index:09d}.jpg"
        with self.lock:
            if name not in self.entries:
                return None
            self.entries.move_to_end(name)
            try:
                return (self.root / name).read_bytes()
            except FileNotFoundError:
                self.size -= self.entries.pop(name)
                return None

    def put(self, identity: str, index: int, payload: bytes, *, background: bool = False) -> bool:
        """Store atomically; background work never evicts frames of the active video.

        :return: False if the limit prevents preloading more content.
        """
        name = f"{identity}-{index:09d}.jpg"
        with self.lock:
            if name in self.entries:
                return True
            if len(payload) > self.budget:
                return False
            if background:
                for old in list(self.entries):
                    if self.size + len(payload) <= self.budget:
                        break
                    if not old.startswith(identity + "-"):
                        (self.root / old).unlink(missing_ok=True)
                        self.size -= self.entries.pop(old)
                if self.size + len(payload) > self.budget:
                    return False
            else:
                self._trim(len(payload))
            path = self.root / name
            temporary = path.with_suffix(".part")
            temporary.write_bytes(payload)
            os.replace(temporary, path)
            self.entries[name] = len(payload)
            self.size += len(payload)
            return True


class Preloader:
    """One active video; cancellable progress, separate from the annotations."""

    def __init__(self, store: FrameStore) -> None:
        self.store = store
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.thread = None
        self.status = {"video": "", "state": "idle", "ready": 0, "total": 0}

    def start(self, video: dict, current: int = 0) -> dict:
        """Preload the events starting from the current one; the whole video for normal ones.

        :return: initial state; repeated requests on the same video are idempotent.
        """
        identity = self.store.identity(video)
        with self.lock:
            if self.status.get("identity") == identity:
                return dict(self.status)
            self.cancel.set()
            previous = self.thread
            cancel = self.cancel = threading.Event()
            intervals = sorted((e["start"], e["end"]) for e in video["events"])
            merged = []
            for a, b in intervals or [(0, video["num_frames"] - 1)]:
                if merged and a <= merged[-1][1] + 1:
                    merged[-1][1] = max(merged[-1][1], b)
                else:
                    merged.append([a, b])
            # Two ordinal passes only if the reviewer opens a later event.
            pivot = next((i for i, (_, b) in enumerate(merged) if b >= current), 0)
            ranges = merged[pivot:] + merged[:pivot]
            status = {
                "video": video["id"],
                "identity": identity,
                "state": "loading",
                "ready": 0,
                "total": sum(b - a + 1 for a, b in merged),
            }
            self.status = status
            self.thread = threading.Thread(
                target=self._run, args=(video, ranges, status, cancel, previous), daemon=True
            )
            self.thread.start()
            return dict(status)

    def snapshot(self, video_id: str) -> dict:
        """:return: progress of the requested video, even with several tabs open."""
        with self.lock:
            return dict(self.status) if self.status["video"] == video_id else {"state": "inactive"}

    def _run(self, video, ranges, status, cancel, previous) -> None:
        import av

        if previous:
            previous.join()
        container, position, iterator = None, -1, None
        try:
            for a, b in ranges:
                for index in range(a, b + 1):
                    if cancel.is_set():
                        return
                    if self.store.get(status["identity"], index) is None:
                        if container is None or index <= position:
                            if container:
                                container.close()
                            container = av.open(video["path"])
                            iterator = iter(container.decode(video=0))
                            position = -1
                        while position < index:
                            if cancel.is_set():
                                return
                            frame = next(iterator)
                            position += 1
                        if not self.store.put(
                            status["identity"], index, encode_frame(frame, video), background=True
                        ):
                            status["state"] = "limited"
                            return
                    status["ready"] += 1
            status["state"] = "complete"
        except Exception as exc:
            status.update(state="error", error=str(exc) or "Video ended early.")
        finally:
            if container:
                container.close()

    def close(self) -> None:
        """Stop the worker; the regenerable cache remains available on reopening."""
        self.cancel.set()
        if self.thread:
            self.thread.join(timeout=5)
