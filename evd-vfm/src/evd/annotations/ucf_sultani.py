"""Public fallback annotations by Sultani et al. (CVPR 2018), test set only.

One line per video, whitespace-separated::

    Abuse028_x264.mp4  Abuse  165  240  -1  -1
    Normal_Videos_003_x264.mp4  Normal  -1  -1  -1  -1

Up to two intervals, ``-1`` meaning absent; indices are frames.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from evd.annotations.schema import (
    NORMAL_CLASS_NAME,
    RawInterval,
    TemporalAnnotation,
    materialize,
)
from evd.annotations.ucf_classes import class_index, ucf_classes
from evd.types import EventClass


@dataclass(frozen=True)
class UcfSultaniAnnotations:
    """``Temporal_Anomaly_Annotation_for_Testing_Videos.txt`` adapter.

    ``index_base`` is a declared assumption: with ``index_base=1`` indices are
    shifted to 0-based. The validator counts out-of-range intervals under both
    conventions.
    """

    path: Path
    index_base: int = 0
    catalog: tuple[EventClass, ...] = field(default_factory=ucf_classes)

    def classes(self) -> tuple[EventClass, ...]:
        return self.catalog

    def parse(self) -> list[RawInterval]:
        rows: list[RawInterval] = []
        for lineno, line in _lines(self.path):
            source = f"{self.path.name}:{lineno}"
            fields = line.split()
            if len(fields) < 2:
                rows.append(RawInterval("", "", 0.0, 0.0, "frames", source))
                continue
            video_id = Path(fields[0]).stem
            class_name = fields[1]
            bounds = _pairs(fields[2:])
            if not bounds or class_name == NORMAL_CLASS_NAME:
                rows.append(RawInterval(video_id, NORMAL_CLASS_NAME, 0.0, 0.0, "frames", source))
                continue
            rows += [
                RawInterval(
                    video_id,
                    class_name,
                    start - self.index_base,
                    end - self.index_base,
                    "frames",
                    source,
                )
                for start, end in bounds
            ]
        return rows

    def load(self) -> dict[str, TemporalAnnotation]:
        return materialize(self.parse(), class_index=class_index(self.catalog))


def _lines(path: Path) -> Iterable[tuple[int, str]]:
    text = Path(path).read_text(encoding="utf-8-sig")
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if line:
            yield lineno, line


def _pairs(fields: list[str]) -> list[tuple[int, int]]:
    """Return the valid ``(start, end)`` pairs; a pair with a single ``-1`` is kept for
    the validator to flag.
    """
    out: list[tuple[int, int]] = []
    for i in range(0, len(fields) - 1, 2):
        try:
            start, end = int(fields[i]), int(fields[i + 1])
        except ValueError:
            continue
        if start == -1 and end == -1:
            continue
        out.append((start, end))
    return out
