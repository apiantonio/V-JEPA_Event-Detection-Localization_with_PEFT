"""Adapter for the laboratory annotations (FS-UCF-Crime and MIVIA-Crime).

One line per video, whitespace-separated::

    Abuse/Abuse001_x264.mp4 A1 257 316 A1 1018 1081 A1 1823 1844
    Training_Normal_Videos_Anomaly/Normal_Videos_003_x264.mp4

After the path come repeated ``<code> <start> <end>`` triples, in frames with
inclusive ends. A line with the path only is a video without events
(MIVIA-Crime writes ``-1 -1`` instead). Codes are resolved with the
``*_target_classes.json`` dictionary.

Two facts shape the consumers: the class of an interval may differ from the
folder of its video (the label is always read from the interval), and a few
intervals end after the last frame; they are clamped to ``num_frames - 1`` and
counted (:meth:`UcfLabAnnotations.clamps`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from evd.annotations.schema import (
    NORMAL_CLASS_NAME,
    RawInterval,
    TemporalAnnotation,
    materialize,
)
from evd.annotations.ucf_classes import class_index, ucf_classes
from evd.types import EventClass
from evd.utils.io import read_json
from evd.utils.logging import get_logger

log = get_logger(__name__)

#: File names of the FS-UCF-Crime delivery.
ANNOTATIONS_NAME = "FS-UCF-Crime_official_annotations.txt"
CLASSES_NAME = "ALL_target_classes.json"

#: File names of the MIVIA-Crime delivery.
MIVIA_ANNOTATIONS_NAME = "MIVIA_CRIME_manual_class_temporal_annots.txt"
MIVIA_CLASSES_NAME = "MIVIA_CRIME_TEST_target_classes.json"

#: MIVIA-Crime marks a video without events with ``-1 -1`` after the path.
MIVIA_NO_EVENTS: tuple[str, ...] = ("-1", "-1")


@dataclass(frozen=True)
class Delivery:
    """Names and conventions of one laboratory annotation delivery.

    :param nome: human-readable name used in messages.
    :param annotations_name: interval file name.
    :param classes_name: code-to-class dictionary file name.
    :param senza_eventi: fields that, alone after the path, mark a video without
        events; empty means a line with the path only (FS-UCF-Crime).
    """

    nome: str
    annotations_name: str
    classes_name: str
    senza_eventi: tuple[str, ...] = ()


FS_UCF_CRIME = Delivery("FS-UCF-Crime", ANNOTATIONS_NAME, CLASSES_NAME)
MIVIA_CRIME = Delivery("MIVIA-Crime", MIVIA_ANNOTATIONS_NAME, MIVIA_CLASSES_NAME, MIVIA_NO_EVENTS)

#: Known deliveries.
DELIVERIES: tuple[Delivery, ...] = (FS_UCF_CRIME, MIVIA_CRIME)


def delivery_of(directory: str | Path) -> Delivery:
    """Recognise which delivery is stored in a folder, from its annotation file name.

    :param directory: delivery folder.
    :return: the recognised delivery.
    :raises FileNotFoundError: if no known delivery is in the folder.
    :raises ValueError: if the folder contains more than one delivery.
    """
    base = Path(directory)
    found = [c for c in DELIVERIES if (base / c.annotations_name).is_file()]
    if not found:
        searched = ", ".join(c.annotations_name for c in DELIVERIES)
        raise FileNotFoundError(
            f"{base} contains no known annotation delivery (looked for: {searched})"
        )
    if len(found) > 1:
        raise ValueError(
            f"{base} contains {len(found)} deliveries ({', '.join(c.nome for c in found)}): "
            f"keep one delivery per folder"
        )
    return found[0]


@dataclass(frozen=True)
class ClampedInterval:
    """Interval that exceeded the video length, and by how much."""

    video_id: str
    start: int
    end_originale: int
    end_clampato: int
    source: str = ""

    @property
    def excess(self) -> int:
        """Return how many frames the interval exceeded."""
        return self.end_originale - self.end_clampato


@dataclass(frozen=True)
class UcfLabAnnotations:
    """FS-UCF-Crime / MIVIA-Crime temporal annotations.

    :param path: annotation ``.txt`` delivered by the laboratory.
    :param class_map: code-to-class mapping.
    :param num_frames: ``video_id`` to frame count; without it no clamp is applied
        and out-of-range intervals are kept as they are.
    :param catalog: allowed classes (the 13 UCF-Crime classes plus ``Normal`` by
        default); a code outside the catalog makes ``load`` raise.
    :param senza_eventi: fields marking a video without events (see
        :class:`Delivery`).
    """

    path: Path
    class_map: Mapping[str, str]
    num_frames: Mapping[str, int] | None = None
    catalog: tuple[EventClass, ...] = field(default_factory=ucf_classes)
    senza_eventi: tuple[str, ...] = ()

    @classmethod
    def from_dir(
        cls,
        directory: str | Path,
        *,
        delivery: Delivery = FS_UCF_CRIME,
        num_frames: Mapping[str, int] | None = None,
    ) -> UcfLabAnnotations:
        """Build the adapter from a delivery folder.

        :param directory: folder holding the annotation and class files.
        :param delivery: which delivery is being read; use :func:`delivery_of` to
            recognise an unknown folder.
        :param num_frames: see the field with the same name.
        :return: the adapter.
        :raises FileNotFoundError: if the class dictionary is missing.
        """
        base = Path(directory)
        return cls(
            path=base / delivery.annotations_name,
            class_map=read_json(base / delivery.classes_name),
            num_frames=num_frames,
            senza_eventi=delivery.senza_eventi,
        )

    def classes(self) -> tuple[EventClass, ...]:
        return self.catalog

    def parse(self) -> list[RawInterval]:
        """Return the rows of the file, without clamping or validation.

        :return: one ``RawInterval`` per interval plus a ``Normal`` row for every video
            without events; malformed lines yield a row with an empty class name.
        """
        rows: list[RawInterval] = []
        for lineno, line in _lines(self.path):
            source = f"{self.path.name}:{lineno}"
            fields = line.replace("\\", "/").split()
            video_id = PurePosixPath(fields[0]).stem if fields else ""
            rest = fields[1:]
            if not video_id:
                rows.append(RawInterval("", "", 0.0, 0.0, "frames", source))
                continue
            if not rest or (self.senza_eventi and tuple(rest) == self.senza_eventi):
                rows.append(RawInterval(video_id, NORMAL_CLASS_NAME, 0.0, 0.0, "frames", source))
                continue
            if len(rest) % 3:
                # Truncated triple: report the whole line as unreadable.
                rows.append(RawInterval(video_id, "", 0.0, 0.0, "frames", source))
                continue
            for i in range(0, len(rest), 3):
                code_name = rest[i]
                try:
                    start, end = int(rest[i + 1]), int(rest[i + 2])
                except ValueError:
                    rows.append(RawInterval(video_id, "", 0.0, 0.0, "frames", source))
                    continue
                rows.append(
                    RawInterval(
                        video_id,
                        self.class_map.get(code_name, code_name),
                        start,
                        end,
                        "frames",
                        source,
                    )
                )
        return rows

    def clamps(self) -> list[ClampedInterval]:
        """List the intervals that exceed ``num_frames - 1``.

        :return: an empty list when ``num_frames`` was not provided.
        """
        if not self.num_frames:
            return []
        outside: list[ClampedInterval] = []
        for row in self.parse():
            if not row.class_name or row.class_name == NORMAL_CLASS_NAME:
                continue
            total = self.num_frames.get(row.video_id)
            if total is None:
                continue
            last = total - 1
            if int(row.end) > last:
                outside.append(
                    ClampedInterval(row.video_id, int(row.start), int(row.end), last, row.source)
                )
        return outside

    def load(self) -> dict[str, TemporalAnnotation]:
        """Return valid annotations with the clamp applied and counted.

        :return: ``video_id`` to annotation for every video of the file, including the
            videos without events.
        :raises ValueError: on codes outside the catalog, unreadable lines, or intervals
            that start after the end of the video.
        """
        last_frame = {v: n - 1 for v, n in (self.num_frames or {}).items()}
        cut_items = 0
        rows: list[RawInterval] = []
        for row in self.parse():
            last = last_frame.get(row.video_id)
            if last is None or not row.class_name or row.class_name == NORMAL_CLASS_NAME:
                rows.append(row)
                continue
            if int(row.start) > last:
                raise ValueError(
                    f"{row.source or row.video_id}: the interval starts at frame "
                    f"{int(row.start)}, after the last frame of the video ({last}). "
                    "Not a clamp case: annotation and inventory refer to different videos"
                )
            if int(row.end) > last:
                cut_items += 1
                row = RawInterval(
                    row.video_id, row.class_name, row.start, last, row.unit, row.source
                )
            rows.append(row)
        if cut_items:
            log.warning(
                "%s: %d intervals ended after the last frame and were clamped to "
                "num_frames-1; see clamps() for the list",
                self.path.name,
                cut_items,
            )
        return materialize(rows, class_index=class_index(self.catalog))


def _lines(path: Path) -> Iterable[tuple[int, str]]:
    text = Path(path).read_text(encoding="utf-8-sig")
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if line:
            yield lineno, line
