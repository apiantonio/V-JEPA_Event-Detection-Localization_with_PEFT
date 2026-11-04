"""Deterministic synthetic "toy surveillance" dataset.

It makes the whole pipeline (decoding, windows, backbone, head, mapping, metrics,
figures) testable without real data, GPU or annotations. The event signal is
deliberately coarse, a bright square crossing the scene at an anomalous speed, so
that even a randomly initialized backbone separates it: the smoke test checks the
pipeline, not learning. Videos are 64x64, 120 frames, with three event classes plus
``Normal``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from evd.annotations.schema import TemporalAnnotation
from evd.types import EventClass, PreprocessSpec, VideoMeta
from evd.utils.hashing import stable_hash
from evd.utils.io import read_json, write_json
from evd.utils.logging import get_logger

log = get_logger(__name__)

#: In the synthetic dataset ``Normal`` has index 0 (UCF-Crime uses 13).
SYNTH_CLASSES: tuple[EventClass, ...] = (
    EventClass(0, "Normal"),
    EventClass(1, "Sprint"),  # fast crossing from left to right
    EventClass(2, "Reverse"),  # fast crossing in the opposite direction
    EventClass(3, "Erratic"),  # oscillates back and forth
)
NORMAL_IDX = 0


@dataclass(frozen=True)
class SyntheticSpec:
    """Parameters of the synthetic world; they enter the hash that decides regeneration."""

    n_normal: int = 9
    n_anomalous: int = 9
    """Nine anomalous videos (not six) keep enough positive chunks in the query split."""
    num_frames: int = 120
    """120 frames keep the CPU smoke test under a minute."""
    size: int = 64
    fps: float = 25.0
    seed: int = 0
    event_fraction: tuple[float, float] = (0.45, 0.70)
    """Fraction of the video covered by the event, as a sampling range.

    It is also the fraction of positive chunks, so it is large enough to make the
    signal obvious."""
    object_size: int = 14
    """Object side in pixels: 14 (not 10) makes the brightness difference obvious."""
    codec: str = "auto"

    def hash(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True)
class SyntheticDataset:
    root: Path
    videos: tuple[VideoMeta, ...]
    annotations: dict[str, TemporalAnnotation]
    classes: tuple[EventClass, ...]
    spec: SyntheticSpec

    @property
    def normal_ids(self) -> tuple[str, ...]:
        return tuple(v.video_id for v in self.videos if self.annotations[v.video_id].is_normal)

    @property
    def anomalous_ids(self) -> tuple[str, ...]:
        return tuple(v.video_id for v in self.videos if not self.annotations[v.video_id].is_normal)


def synthetic_preprocess_spec(size: int = 64, num_frames: int = 16) -> PreprocessSpec:
    """Return the spec shared by the synthetic world and ``TinyBackbone``.

    Videos are generated at the right size, so geometry and backend are ``"none"``.

    :param size: frame size.
    :param num_frames: samples per window.
    :return: the preprocessing spec.
    """
    return PreprocessSpec(
        fps=8.0,
        num_frames=num_frames,
        size=size,
        short_side=size,
        geometry="none",
        backend="none",
        backend_version="none",
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        tokenizer="tubelet_2x8x8",
        dtype="fp32",
    )


# Pixel generation


def _render_video(
    spec: SyntheticSpec, rng: np.random.Generator, event: tuple[int, int, int] | None
) -> np.ndarray:
    """Render a ``[T, H, W, 3]`` uint8 video; ``event`` is ``(start, end, class_idx)`` or ``None``."""
    n, s, obj = spec.num_frames, spec.size, spec.object_size
    texture = rng.integers(45, 75, size=(s, s), dtype=np.int16)  # fixed background of this video
    frames = np.empty((n, s, s, 3), dtype=np.uint8)

    x = float(rng.integers(0, s - obj))
    y = int(rng.integers(0, s - obj))
    drift = 0.35  # normal speed: about 1 pixel every 3 frames
    for t in range(n):
        canvas = texture + rng.integers(-4, 5, size=(s, s), dtype=np.int16)
        in_event = event is not None and event[0] <= t <= event[1]
        if in_event:
            cls = event[2]
            if cls == 1:
                x += 4.0
            elif cls == 2:
                x -= 4.0
            else:
                x += 5.0 if (t // 3) % 2 == 0 else -5.0
            brightness = 250
        else:
            x += drift
            brightness = 100
        # Bounce on the borders: teleporting would add an unwanted discontinuity signal.
        limit = s - obj
        if x < 0:
            x = -x
        elif x > limit:
            x = 2 * limit - x
        xi = int(round(min(max(x, 0), limit)))
        canvas[y : y + obj, xi : xi + obj] = brightness
        frames[t] = np.clip(canvas, 0, 255).astype(np.uint8)[:, :, None].repeat(3, axis=2)
    return frames


def _encode(path: Path, frames: np.ndarray, fps: float, codec: str) -> None:
    """Write an mp4; ``codec="auto"`` tries h264 and falls back to mpeg4.

    PyAV wheels do not always ship ``libx264``, while ``mpeg4`` is always available.

    :param path: destination file.
    :param frames: ``[T, H, W, 3]`` uint8 frames.
    :param fps: frame rate.
    :param codec: codec name or ``"auto"``.
    :raises RuntimeError: if no codec works.
    """
    import av

    candidates = ["libx264", "mpeg4"] if codec == "auto" else [codec]
    last: Exception | None = None
    for name in candidates:
        try:
            with av.open(str(path), mode="w") as container:
                stream = container.add_stream(name, rate=int(round(fps)))
                stream.width = frames.shape[2]
                stream.height = frames.shape[1]
                stream.pix_fmt = "yuv420p"
                if name == "libx264":
                    stream.options = {"crf": "16", "preset": "veryfast"}
                for f in frames:
                    packet = stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24"))
                    container.mux(packet)
                container.mux(stream.encode())  # flush the encoder, otherwise the last GOP is lost
            return
        except Exception as e:  # noqa: BLE001 - try the next codec
            last = e
            path.unlink(missing_ok=True)
            log.debug("codec %s not usable: %s", name, e)
    raise RuntimeError(f"no usable codec among {candidates}: {last}")


# API


def generate(out_dir: str | Path, spec: SyntheticSpec | None = None) -> SyntheticDataset:
    """Generate (or reuse) the synthetic world in ``out_dir``.

    Idempotent with respect to the spec: an existing ``index.json`` with a matching hash
    is reused without re-encoding.

    :param out_dir: output folder.
    :param spec: world parameters (defaults when ``None``).
    :return: the synthetic dataset description.
    :raises RuntimeError: if a written video does not have the expected frame count.
    """
    spec = spec or SyntheticSpec()
    root = Path(out_dir)
    index_path = root / "index.json"
    if index_path.is_file():
        index = read_json(index_path)
        if index.get("spec_hash") == spec.hash() and all(
            (root / v["file"]).is_file() for v in index["videos"]
        ):
            log.info(
                "synthetic world already present in %s (%d videos)", root, len(index["videos"])
            )
            return _from_index(root, index, spec)
        log.info("spec changed: regenerating the synthetic world in %s", root)

    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(spec.seed)
    videos: list[VideoMeta] = []
    anns: dict[str, TemporalAnnotation] = {}
    records: list[dict] = []

    plan = [(i, None) for i in range(spec.n_normal)]
    for i in range(spec.n_anomalous):
        plan.append((i, 1 + i % (len(SYNTH_CLASSES) - 1)))

    for i, cls in plan:
        if cls is None:
            vid = f"synth_normal_{i:02d}"
            event = None
        else:
            vid = f"synth_{SYNTH_CLASSES[cls].name.lower()}_{i:02d}"
            min_fraction, max_fraction = spec.event_fraction
            length = int(
                rng.integers(
                    int(min_fraction * spec.num_frames), int(max_fraction * spec.num_frames)
                )
            )
            last_start = max(1, spec.num_frames - length - 1)
            start = int(rng.integers(int(0.10 * spec.num_frames), last_start))
            event = (start, min(start + length, spec.num_frames - 1), cls)
        frames = _render_video(spec, rng, event)
        path = root / f"{vid}.mp4"
        _encode(path, frames, spec.fps, spec.codec)

        # Read the frame count back from the file: a lost GOP would shift every window.
        from evd.data.decoder import PyAVDecoder

        meta = PyAVDecoder().probe(path)
        if meta.num_frames != spec.num_frames:
            raise RuntimeError(
                f"{path.name}: wrote {spec.num_frames} frames, the file declares "
                f"{meta.num_frames}; windows and annotations would not match."
            )
        videos.append(meta)
        anns[vid] = TemporalAnnotation(vid, () if event is None else (event,))
        records.append(
            {
                "video_id": vid,
                "file": path.name,
                "num_frames": meta.num_frames,
                "fps": meta.fps,
                "width": meta.width,
                "height": meta.height,
                "codec": meta.codec,
                "intervals": [] if event is None else [list(event)],
            }
        )

    write_json(
        index_path,
        {
            "spec_hash": spec.hash(),
            "spec": asdict(spec),
            "classes": [{"idx": c.idx, "name": c.name} for c in SYNTH_CLASSES],
            "videos": records,
        },
    )
    log.info("synthetic world: %d videos in %s", len(videos), root)
    return SyntheticDataset(root, tuple(videos), anns, SYNTH_CLASSES, spec)


def _from_index(root: Path, index: dict, spec: SyntheticSpec) -> SyntheticDataset:
    videos: list[VideoMeta] = []
    anns: dict[str, TemporalAnnotation] = {}
    for v in index["videos"]:
        videos.append(
            VideoMeta(
                video_id=v["video_id"],
                path=str(root / v["file"]),
                num_frames=int(v["num_frames"]),
                fps=float(v["fps"]),
                width=int(v["width"]),
                height=int(v["height"]),
                codec=v.get("codec", ""),
            )
        )
        anns[v["video_id"]] = TemporalAnnotation(
            v["video_id"], tuple(tuple(int(x) for x in iv) for iv in v["intervals"])
        )
    return SyntheticDataset(root, tuple(videos), anns, SYNTH_CLASSES, spec)
