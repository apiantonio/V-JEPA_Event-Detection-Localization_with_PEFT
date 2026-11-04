"""Video decoding behind an interface.

``decord`` is unreliable on Windows and ``torchcodec`` is not installed on the
project machines, so no module imports a decoder directly: the choice is a config
field. :class:`PyAVDecoder` seeks to the keyframe before the first requested
index and rebuilds absolute indices from PTS, falling back to sequential decoding
when the mapping is unreliable. Partial reads use slice threading, since PyAV
frame threading can hang when the decoder is closed long before EOF.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from evd.types import VideoMeta
from evd.utils.logging import get_logger

log = get_logger(__name__)


@runtime_checkable
class VideoDecoder(Protocol):
    """Minimal decoder contract: probe a file and read frames by index."""

    name: str

    def probe(self, path: str | Path) -> VideoMeta: ...

    def read_frames(self, path: str | Path, indices: Sequence[int]) -> np.ndarray:
        """Return ``[T, H, W, C]`` uint8 RGB frames, in the order of ``indices``."""
        ...


class PyAVDecoder:
    """PyAV decoder with keyframe seek and exact fallback.

    After ``container.seek(..., backward=True, any_frame=False)`` frame indices are
    rebuilt from PTS, ``time_base`` and frame rate. Without PTS, with VFR content or
    when some index is not recovered, the file is reopened and read sequentially.
    """

    name = "pyav"
    # ``AUTO`` also enables frame threading, which can hang when an H.264 read is closed
    # after a few frames; ``SLICE`` keeps intra-frame parallelism and the same pixels.
    _partial_decode_thread_type = "SLICE"

    def __init__(self, *, seek: bool = True) -> None:
        self.seek = seek

    def probe(self, path: str | Path) -> VideoMeta:
        """Return video metadata with a verified ``num_frames``.

        Container headers can overstate the frame count (53 of 270 MIVIA-Crime videos do),
        and ``num_frames`` drives the window geometry. The header is compared with
        ``duration x fps`` and frames are counted by decoding only when the two disagree.

        :param path: video file.
        :return: metadata with the exact ``num_frames``.
        """
        import av

        p = Path(path)
        with av.open(str(p)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else 0.0
            declared = int(stream.frames or 0)
            dur = float(stream.duration * stream.time_base) if stream.duration else 0.0
            estimated = int(round(dur * fps))
            num_frames = declared
            if declared <= 0:
                # The container does not declare a count: count by decoding.
                num_frames = self._count_frames(container, stream)
                log.warning("%s does not declare its frame count: counted %d", p.name, num_frames)
            elif declared != estimated:
                num_frames = self._count_frames(container, stream)
                log.warning(
                    "%s: header declares %d frames but duration x fps gives %d; counted %d",
                    p.name,
                    declared,
                    estimated,
                    num_frames,
                )
            return VideoMeta(
                video_id=p.stem,
                path=str(p),
                num_frames=num_frames,
                fps=fps,
                width=int(stream.codec_context.width),
                height=int(stream.codec_context.height),
                codec=str(stream.codec_context.name),
            )

    @staticmethod
    def _count_frames(container: Any, stream: Any) -> int:
        """Count the frames actually produced by the stream, rewinding before and after.

        :param container: open container.
        :param stream: video stream.
        :return: number of decoded frames.
        """
        container.seek(0, stream=stream)
        n = sum(1 for _ in container.decode(stream))
        container.seek(0, stream=stream)
        return n

    def read_frames(self, path: str | Path, indices: Sequence[int]) -> np.ndarray:
        if not indices:
            raise ValueError("read_frames called without indices")
        wanted = sorted({int(i) for i in indices})
        if wanted[0] < 0:
            raise ValueError(f"negative index: {wanted[0]}")
        found: dict[int, np.ndarray] | None = None
        if self.seek and wanted[0] > 0:
            found = self._read_frames_seek(path, wanted)
        if found is None:
            found = self._read_frames_sequential(path, wanted)
        return self._stack_requested(path, indices, wanted, found)

    @staticmethod
    def _read_frames_sequential(path: str | Path, wanted: Sequence[int]) -> dict[int, np.ndarray]:
        import av

        found: dict[int, np.ndarray] = {}
        last = wanted[-1]
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = PyAVDecoder._partial_decode_thread_type
            for i, frame in enumerate(container.decode(stream)):
                if i > last:
                    break
                if i in wanted:
                    found[i] = frame.to_ndarray(format="rgb24")
        return found

    @staticmethod
    def _read_frames_seek(path: str | Path, wanted: Sequence[int]) -> dict[int, np.ndarray] | None:
        """Read from the previous keyframe; ``None`` requests the sequential fallback."""
        import av

        first, last = wanted[0], wanted[-1]
        found: dict[int, np.ndarray] = {}
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            rate = stream.average_rate
            time_base = stream.time_base
            if rate is None or time_base is None:
                return None
            fps = float(rate)
            if not math.isfinite(fps) or fps <= 0:
                return None
            start_pts = int(stream.start_time or 0)
            units_per_frame = Fraction(1, 1) / rate / time_base
            target_pts = start_pts + int(first * units_per_frame)
            try:
                container.seek(
                    max(start_pts, target_pts),
                    stream=stream,
                    backward=True,
                    any_frame=False,
                )
            except (av.error.FFmpegError, ValueError):
                return None

            stream.thread_type = PyAVDecoder._partial_decode_thread_type
            for frame in container.decode(stream):
                if frame.pts is None:
                    return None
                seconds = Fraction(int(frame.pts) - start_pts) * time_base
                frame_index = int(round(float(seconds * rate)))
                if frame_index > last:
                    break
                if frame_index in wanted and frame_index not in found:
                    found[frame_index] = frame.to_ndarray(format="rgb24")

        if any(i not in found for i in wanted):
            log.debug(
                "%s: seek/PTS did not recover every index; sequential fallback",
                Path(path).name,
            )
            return None
        return found

    @staticmethod
    def _stack_requested(
        path: str | Path,
        indices: Sequence[int],
        wanted: Sequence[int],
        found: dict[int, np.ndarray],
    ) -> np.ndarray:
        missing = [i for i in wanted if i not in found]
        if missing:
            raise IndexError(
                f"{Path(path).name}: frames not decoded {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''} (the file has fewer frames than "
                f"requested or is truncated; see the inventory)"
            )
        return np.stack([found[int(i)] for i in indices])


class TorchcodecDecoder:  # pragma: no cover - optional dependency
    """Placeholder: torchcodec is neither installed nor verified."""

    name = "torchcodec"

    def probe(self, path: str | Path) -> VideoMeta:
        raise NotImplementedError(
            "TorchcodecDecoder is neither installed nor verified; use pyav "
            "(future [codec] extra of the pyproject)"
        )

    def read_frames(self, path: str | Path, indices: Sequence[int]) -> np.ndarray:
        raise NotImplementedError(
            "TorchcodecDecoder is neither installed nor verified; use pyav "
            "(future [codec] extra of the pyproject)"
        )


_DECODERS = {"pyav": PyAVDecoder, "torchcodec": TorchcodecDecoder}


def make_decoder(name: str = "pyav") -> VideoDecoder:
    """Build a decoder from its config name (``data.decoder``).

    :param name: decoder name.
    :return: the decoder instance.
    :raises KeyError: if the name is unknown.
    """
    try:
        cls = _DECODERS[name]
    except KeyError:
        raise KeyError(f"unknown decoder '{name}': choose among {sorted(_DECODERS)}") from None
    return cls()
