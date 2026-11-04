"""Reproducible video inventory.

The CSV describes what PyAV actually sees and ``MANIFEST.sha256`` binds it to the
bytes. Scanning is shardable; full verification decodes every video to compare the
real frame count with the header, and the canonical merge keeps files outside the
official splits apart.
"""

from __future__ import annotations

import csv
import io
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from evd.types import VideoMeta
from evd.utils.hashing import file_sha256
from evd.utils.io import ensure_dir, write_json, write_text_atomic

VIDEO_EXTENSIONS = frozenset({".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg"})


@dataclass(frozen=True)
class InventoryRecord:
    relative_path: str
    video_id: str
    size_bytes: int
    sha256: str
    fps: float
    duration_s: float
    num_frames: int
    num_frames_header: int
    num_frames_decoded: int
    width: int
    height: int
    codec: str
    issues: tuple[str, ...] = ()
    error: str = ""

    @property
    def status(self) -> str:
        return "ok" if not self.issues else "pathological"

    def to_meta(self) -> VideoMeta:
        """Return the ``VideoMeta`` view of the record."""
        return VideoMeta(
            video_id=self.video_id,
            path=self.relative_path,
            num_frames=self.num_frames,
            fps=self.fps,
            width=self.width,
            height=self.height,
            codec=self.codec,
        )

    def csv_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["issues"] = ";".join(self.issues)
        row["status"] = self.status
        return row


CSV_FIELDS = (
    "relative_path",
    "video_id",
    "size_bytes",
    "sha256",
    "fps",
    "duration_s",
    "num_frames",
    "num_frames_header",
    "num_frames_decoded",
    "width",
    "height",
    "codec",
    "status",
    "issues",
    "error",
)


def discover_videos(root: str | Path) -> list[Path]:
    """List video files under ``root``, sorted by relative path.

    :param root: dataset root.
    :return: video paths.
    :raises FileNotFoundError: if the folder does not exist.
    """
    base = Path(root)
    if not base.is_dir():
        raise FileNotFoundError(f"missing video folder: {base}")
    return sorted(
        (p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda p: p.relative_to(base).as_posix(),
    )


def load_split_paths(
    split_files: Sequence[str | Path],
    *,
    prefix: str = "Videos",
) -> list[str]:
    """Load the canonical paths declared by the official split files.

    Paths are normalized to POSIX relative paths, so the manifest does not depend on
    the operating system.

    :param split_files: official split files.
    :param prefix: prefix to strip from the paths.
    :return: canonical relative paths.
    :raises FileNotFoundError: if a split file is missing.
    :raises ValueError: on absolute or duplicate paths.
    """
    normalized_prefix = prefix.strip().replace("\\", "/").strip("/")
    paths: list[str] = []
    for split_file in split_files:
        source = Path(split_file)
        if not source.is_file():
            raise FileNotFoundError(f"missing split file: {source}")
        for line_number, raw in enumerate(
            source.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            value = raw.strip().replace("\\", "/")
            if not value:
                continue
            candidate = PurePosixPath(value)
            if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
                raise ValueError(f"non-relative path in {source}:{line_number}: {raw!r}")
            relative = candidate.as_posix()
            paths.append(f"{normalized_prefix}/{relative}" if normalized_prefix else relative)

    duplicates = sorted(path for path, count in Counter(paths).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate paths in the splits: {duplicates[:5]}")
    return sorted(paths)


def shard_paths(paths: Sequence[Path], shard_index: int, num_shards: int) -> list[Path]:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be > 0, not {num_shards}")
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index {shard_index} outside [0, {num_shards})")
    return list(paths[shard_index::num_shards])


def inspect_video(
    path: str | Path,
    *,
    dataset_root: str | Path,
    verify_decode: bool,
) -> InventoryRecord:
    """Return hash, metadata and, if requested, the full frame count of a video.

    :param path: video file.
    :param dataset_root: dataset root, for the relative path.
    :param verify_decode: whether to count frames by decoding.
    :return: the inventory record.
    """
    import av

    p = Path(path)
    root = Path(dataset_root)
    relative = p.relative_to(root).as_posix()
    digest = file_sha256(p)
    issues: list[str] = []
    fps = duration_s = 0.0
    header_frames = decoded_frames = width = height = 0
    codec = ""
    error = ""

    try:
        with av.open(str(p)) as container:
            if not container.streams.video:
                raise ValueError("nessuno stream video")
            stream = container.streams.video[0]
            rate = stream.average_rate
            fps = float(rate) if rate is not None else 0.0
            header_frames = int(stream.frames or 0)
            width = int(stream.codec_context.width or 0)
            height = int(stream.codec_context.height or 0)
            codec = str(stream.codec_context.name or "")
            if stream.duration is not None and stream.time_base is not None:
                duration_s = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration_s = float(container.duration / av.time_base)

            if verify_decode:
                stream.thread_type = "AUTO"
                decoded_frames = sum(1 for _ in container.decode(stream))
    except Exception as exc:  # keep broken files in the CSV instead of aborting the shard
        error = f"{type(exc).__name__}: {exc}"
        issues.append("decode_error")

    if not math.isfinite(fps) or fps <= 0:
        issues.append("fps_missing_or_invalid")
        fps = 0.0
    if header_frames <= 0:
        issues.append("header_frame_count_missing")
    if verify_decode and decoded_frames <= 0:
        issues.append("decoded_frame_count_empty")
    if (
        verify_decode
        and header_frames > 0
        and decoded_frames > 0
        and header_frames != decoded_frames
    ):
        issues.append("header_decoded_frame_mismatch")
    if width <= 0 or height <= 0:
        issues.append("invalid_resolution")
    if not codec:
        issues.append("codec_missing")

    num_frames = decoded_frames if verify_decode and decoded_frames > 0 else header_frames
    if num_frames <= 0 and duration_s > 0 and fps > 0:
        num_frames = int(round(duration_s * fps))
        issues.append("frame_count_estimated_from_duration")
    if duration_s <= 0 and num_frames > 0 and fps > 0:
        duration_s = num_frames / fps
        issues.append("duration_estimated_from_frames")

    return InventoryRecord(
        relative_path=relative,
        video_id=p.stem,
        size_bytes=p.stat().st_size,
        sha256=digest,
        fps=fps,
        duration_s=duration_s,
        num_frames=num_frames,
        num_frames_header=header_frames,
        num_frames_decoded=decoded_frames,
        width=width,
        height=height,
        codec=codec,
        issues=tuple(dict.fromkeys(issues)),
        error=error,
    )


def scan_shard(
    paths: Sequence[Path],
    *,
    dataset_root: str | Path,
    verify_decode: bool,
) -> list[InventoryRecord]:
    return [
        inspect_video(path, dataset_root=dataset_root, verify_decode=verify_decode)
        for path in paths
    ]


def _csv_text(records: Sequence[InventoryRecord]) -> str:
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow(record.csv_row())
    return buf.getvalue()


def _part_name(shard_index: int, num_shards: int) -> str:
    return f"inventory.part-{shard_index:04d}-of-{num_shards:04d}.csv"


def write_shard(
    output_dir: str | Path,
    records: Sequence[InventoryRecord],
    *,
    shard_index: int,
    num_shards: int,
) -> Path:
    out = ensure_dir(output_dir)
    path = out / _part_name(shard_index, num_shards)
    return write_text_atomic(path, _csv_text(records))


def read_records(path: str | Path) -> list[InventoryRecord]:
    out: list[InventoryRecord] = []
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            out.append(
                InventoryRecord(
                    relative_path=row["relative_path"],
                    video_id=row["video_id"],
                    size_bytes=int(row["size_bytes"]),
                    sha256=row["sha256"],
                    fps=float(row["fps"]),
                    duration_s=float(row["duration_s"]),
                    num_frames=int(row["num_frames"]),
                    num_frames_header=int(row["num_frames_header"]),
                    num_frames_decoded=int(row["num_frames_decoded"]),
                    width=int(row["width"]),
                    height=int(row["height"]),
                    codec=row["codec"],
                    issues=tuple(x for x in row["issues"].split(";") if x),
                    error=row["error"],
                )
            )
    return out


def index_by_video_id(records: Sequence[InventoryRecord]) -> dict[str, InventoryRecord]:
    """Map ``video_id`` to its record, raising on duplicates.

    :param records: inventory records.
    :return: records keyed by video id.
    :raises ValueError: if two records share a video id.
    """
    out: dict[str, InventoryRecord] = {}
    for record in records:
        if record.video_id in out:
            raise ValueError(f"duplicated video_id in the inventory: {record.video_id}")
        out[record.video_id] = record
    return out


def merge_shards(
    output_dir: str | Path,
    *,
    num_shards: int,
    expected_count: int | None = None,
    canonical_paths: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Merge all inventory parts or raise, so that no partial inventory looks complete.

    :param output_dir: folder with the shard parts.
    :param num_shards: expected number of parts.
    :param expected_count: expected number of canonical videos.
    :param canonical_paths: canonical relative paths.
    :return: merge summary.
    :raises FileNotFoundError: if a part is missing.
    :raises ValueError: on duplicates, missing or unexpected videos.
    """
    out = Path(output_dir)
    part_paths = [out / _part_name(i, num_shards) for i in range(num_shards)]
    missing = [p.name for p in part_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing inventory parts: {missing[:5]}")
    records = [record for path in part_paths for record in read_records(path)]
    records.sort(key=lambda record: record.relative_path)

    paths = [record.relative_path for record in records]
    if len(paths) != len(set(paths)):
        duplicates = sorted(path for path, count in Counter(paths).items() if count > 1)
        raise ValueError(f"duplicate paths across shards: {duplicates[:5]}")
    scanned_count = len(records)
    excluded: list[InventoryRecord] = []
    if canonical_paths is not None:
        canonical = list(canonical_paths)
        duplicate_canonical = sorted(
            path for path, count in Counter(canonical).items() if count > 1
        )
        if duplicate_canonical:
            raise ValueError(f"duplicate canonical paths: {duplicate_canonical[:5]}")
        canonical_set = set(canonical)
        missing_canonical = sorted(canonical_set.difference(paths))
        if missing_canonical:
            raise ValueError(
                f"canonical videos missing from the inventory: {missing_canonical[:5]}"
            )
        excluded = [
            InventoryRecord(
                **{
                    **asdict(record),
                    "issues": tuple(dict.fromkeys((*record.issues, "outside_official_splits"))),
                }
            )
            for record in records
            if record.relative_path not in canonical_set
        ]
        records = [record for record in records if record.relative_path in canonical_set]

    if expected_count is not None and len(records) != expected_count:
        raise ValueError(f"expected {expected_count} videos, found {len(records)}")

    ids = Counter(record.video_id for record in records)
    duplicate_ids = {video_id for video_id, count in ids.items() if count > 1}
    if duplicate_ids:
        records = [
            InventoryRecord(
                **{
                    **asdict(record),
                    "issues": tuple(dict.fromkeys((*record.issues, "duplicate_video_id"))),
                }
            )
            if record.video_id in duplicate_ids
            else record
            for record in records
        ]

    hashes = Counter(record.sha256 for record in records if record.sha256)
    duplicate_hashes = {digest for digest, count in hashes.items() if count > 1}
    if duplicate_hashes:
        records = [
            InventoryRecord(
                **{
                    **asdict(record),
                    "issues": tuple(dict.fromkeys((*record.issues, "duplicate_content_sha256"))),
                }
            )
            if record.sha256 in duplicate_hashes
            else record
            for record in records
        ]

    pathological = [record for record in records if record.issues]
    manifest = "".join(f"{record.sha256}  {record.relative_path}\n" for record in records)
    write_text_atomic(out / "inventory.csv", _csv_text(records))
    write_text_atomic(out / "pathologies.csv", _csv_text(pathological))
    write_text_atomic(out / "excluded.csv", _csv_text(excluded))
    write_text_atomic(out / "MANIFEST.sha256", manifest)

    summary: dict[str, Any] = {
        "n_scanned": scanned_count,
        "n_videos": len(records),
        "n_excluded": len(excluded),
        "n_pathological": len(pathological),
        "total_bytes": sum(record.size_bytes for record in records),
        "total_duration_h": sum(record.duration_s for record in records) / 3600,
        "codecs": dict(sorted(Counter(record.codec or "<missing>" for record in records).items())),
        "resolutions": dict(
            sorted(Counter(f"{record.width}x{record.height}" for record in records).items())
        ),
        "fps": dict(sorted(Counter(f"{record.fps:.6g}" for record in records).items())),
        "issues": dict(
            sorted(Counter(issue for record in records for issue in record.issues).items())
        ),
    }
    write_json(out / "summary.json", summary)
    return summary
