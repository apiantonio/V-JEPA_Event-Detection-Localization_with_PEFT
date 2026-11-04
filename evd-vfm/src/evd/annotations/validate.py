"""Annotation validation report for the laboratory.

The module never raises: a broken file produces a report that says what is
broken and where. Cross-checking with the video inventory (true fps and frame
counts) reveals intervals expressed in the wrong unit.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from evd.annotations.schema import NORMAL_CLASS_NAME, RawInterval, to_frames
from evd.types import EventClass, VideoMeta

Severity = Literal["error", "warning", "info"]

MAX_FINDINGS_PER_CODE = 50
"""Above this many findings per code the report counts instead of listing."""


@dataclass(frozen=True)
class Finding:
    code: str
    severity: Severity
    video_id: str
    detail: str
    source: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity,
            "video_id": self.video_id,
            "detail": self.detail,
            "source": self.source,
        }


@dataclass(frozen=True)
class ValidationReport:
    source_name: str
    source_sha256: str
    n_rows: int
    n_videos: int
    findings: tuple[Finding, ...]
    counts: dict[str, int]
    """Count per code before truncation (``findings`` may hold fewer)."""
    stats: dict[str, Any]

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == "error")

    @property
    def n_errors(self) -> int:
        severity = {f.code: f.severity for f in self.findings}
        return sum(n for code, n in self.counts.items() if severity.get(code) == "error")

    @property
    def ok(self) -> bool:
        return self.n_errors == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "n_rows": self.n_rows,
            "n_videos": self.n_videos,
            "ok": self.ok,
            "n_errors": self.n_errors,
            "counts": self.counts,
            "findings": [f.as_dict() for f in self.findings],
            "stats": self.stats,
        }

    def to_markdown(self) -> str:
        return _markdown(self)


def validate(
    rows: Iterable[RawInterval],
    *,
    inventory: Mapping[str, VideoMeta],
    classes: Sequence[EventClass],
    expect_videos: Iterable[str] | None = None,
    declared_class: Mapping[str, str] | None = None,
    source_name: str = "",
    source_sha256: str = "",
) -> ValidationReport:
    """Cross-check annotation rows against the inventory and the class table.

    :param rows: tolerant rows from ``AnnotationSource.parse()``.
    :param inventory: ``video_id`` to measured video metadata.
    :param classes: allowed classes.
    :param expect_videos: videos that should be annotated; missing ones are findings.
    :param declared_class: ``video_id`` to class inferred from the official split
        folder, enabling the "anomalous video without intervals" and "class differs
        from folder" checks.
    :param source_name: declared source name, reported as is.
    :param source_sha256: fingerprint of the annotation file, reported as is.
    :return: the full report with typed findings and per-class statistics.
    """
    rows = list(rows)
    known = {c.name for c in classes}
    declared = dict(declared_class or {})
    findings: list[Finding] = []

    per_class_videos: dict[str, set[str]] = {name: set() for name in known}
    per_class_segments: dict[str, int] = dict.fromkeys(known, 0)
    spans_frames: dict[str, list[float]] = {name: [] for name in known}
    spans_seconds: dict[str, list[float]] = {name: [] for name in known}
    intervals_by_video: dict[str, list[tuple[int, int, str, str]]] = {}
    listed: set[str] = set()
    with_intervals: set[str] = set()
    out_of_range_by_base = {0: 0, 1: 0}

    for row in rows:
        if not row.class_name:
            # An empty class name marks a row the parser could not read.
            findings.append(
                Finding("unparsable_line", "error", row.video_id, "unreadable row", row.source)
            )
            continue
        listed.add(row.video_id)
        meta = inventory.get(row.video_id)
        if meta is None:
            findings.append(
                Finding(
                    "video_not_in_inventory",
                    "error",
                    row.video_id,
                    "annotated but missing from the canonical inventory videos",
                    row.source,
                )
            )
        if row.class_name not in known:
            findings.append(
                Finding(
                    "unknown_class",
                    "error",
                    row.video_id,
                    f"class {row.class_name!r} outside the {len(known)} expected ones",
                    row.source,
                )
            )
            continue
        if row.class_name == NORMAL_CLASS_NAME:
            per_class_videos[NORMAL_CLASS_NAME].add(row.video_id)
            continue

        expected = declared.get(row.video_id)
        if expected is not None and expected != row.class_name:
            findings.append(
                Finding(
                    "class_mismatch",
                    "error",
                    row.video_id,
                    f"annotated {row.class_name}, but the official folder says {expected}",
                    row.source,
                )
            )

        fps = meta.fps if meta is not None else 0.0
        try:
            start = to_frames(row.start, row.unit, fps)
            end = to_frames(row.end, row.unit, fps)
        except ValueError as exc:
            findings.append(
                Finding("unconvertible_interval", "error", row.video_id, str(exc), row.source)
            )
            continue

        if start < 0 or end < 0:
            findings.append(
                Finding(
                    "interval_negative",
                    "error",
                    row.video_id,
                    f"({start}, {end}) has a negative end",
                    row.source,
                )
            )
            continue
        if end < start:
            findings.append(
                Finding(
                    "interval_inverted",
                    "error",
                    row.video_id,
                    f"end {end} before start {start}",
                    row.source,
                )
            )
            continue

        if meta is not None:
            for base in (0, 1):
                if end - base >= meta.num_frames:
                    out_of_range_by_base[base] += 1
            if end >= meta.num_frames:
                findings.append(
                    Finding(
                        "interval_beyond_num_frames",
                        "error",
                        row.video_id,
                        f"({start}, {end}) beyond the {meta.num_frames} frames of the video",
                        row.source,
                    )
                )
            if meta.fps > 0:
                spans_seconds[row.class_name].append((end - start + 1) / meta.fps)

        with_intervals.add(row.video_id)
        per_class_videos[row.class_name].add(row.video_id)
        per_class_segments[row.class_name] += 1
        spans_frames[row.class_name].append(end - start + 1)
        intervals_by_video.setdefault(row.video_id, []).append(
            (start, end, row.class_name, row.source)
        )

    findings += _overlap_findings(intervals_by_video)
    findings += _coverage_findings(
        listed=listed,
        with_intervals=with_intervals,
        expect_videos=set(expect_videos or ()),
        declared=declared,
        per_class_videos=per_class_videos,
        known=known,
    )

    counts: dict[str, int] = {}
    for f in findings:
        counts[f.code] = counts.get(f.code, 0) + 1

    stats: dict[str, Any] = {
        "videos_per_class": {k: len(v) for k, v in sorted(per_class_videos.items())},
        "segments_per_class": dict(sorted(per_class_segments.items())),
        "segment_frames": {k: _summary(v) for k, v in sorted(spans_frames.items()) if v},
        "segment_seconds": {k: _summary(v) for k, v in sorted(spans_seconds.items()) if v},
        "out_of_range_by_index_base": out_of_range_by_base,
    }

    return ValidationReport(
        source_name=source_name,
        source_sha256=source_sha256,
        n_rows=len(rows),
        n_videos=len(listed),
        findings=_cap(findings),
        counts=dict(sorted(counts.items())),
        stats=stats,
    )


def _cap(findings: Sequence[Finding]) -> tuple[Finding, ...]:
    seen: dict[str, int] = {}
    out: list[Finding] = []
    for f in findings:
        seen[f.code] = seen.get(f.code, 0) + 1
        if seen[f.code] <= MAX_FINDINGS_PER_CODE:
            out.append(f)
    return tuple(out)


def _overlap_findings(
    intervals_by_video: Mapping[str, list[tuple[int, int, str, str]]],
) -> list[Finding]:
    """Find duplicates and overlaps between different classes.

    Overlaps matter because ``frame_class_labels`` resolves them with "last
    declared wins".
    """
    out: list[Finding] = []
    for video_id, items in sorted(intervals_by_video.items()):
        ordered = sorted(items)
        for i, (start, end, name, source) in enumerate(ordered):
            for other_start, other_end, other_name, _ in ordered[i + 1 :]:
                if other_start > end:
                    break
                if (start, end, name) == (other_start, other_end, other_name):
                    out.append(
                        Finding(
                            "duplicate_interval",
                            "warning",
                            video_id,
                            f"({start}, {end}) {name} declared twice",
                            source,
                        )
                    )
                elif name != other_name:
                    out.append(
                        Finding(
                            "overlap_different_class",
                            "error",
                            video_id,
                            f"({start}, {end}) {name} overlaps "
                            f"({other_start}, {other_end}) {other_name}",
                            source,
                        )
                    )
    return out


def _coverage_findings(
    *,
    listed: set[str],
    with_intervals: set[str],
    expect_videos: set[str],
    declared: Mapping[str, str],
    per_class_videos: Mapping[str, set[str]],
    known: set[str],
) -> list[Finding]:
    out = [
        Finding(
            "video_never_annotated",
            "error",
            video_id,
            "expected in the annotation file but missing",
        )
        for video_id in sorted(expect_videos - listed)
    ]
    for video_id in sorted(listed - with_intervals):
        if declared.get(video_id, NORMAL_CLASS_NAME) != NORMAL_CLASS_NAME:
            out.append(
                Finding(
                    "anomalous_video_without_intervals",
                    "error",
                    video_id,
                    f"declared {declared[video_id]} by the official folder, "
                    f"but without annotated intervals",
                )
            )
    out += [
        Finding("class_never_annotated", "warning", "", f"no segment for {name}")
        for name in sorted(known - {NORMAL_CLASS_NAME})
        if not per_class_videos.get(name)
    ]
    return out


def _summary(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    n = len(ordered)
    return {
        "n": n,
        "min": round(ordered[0], 3),
        "median": round(ordered[n // 2], 3),
        "max": round(ordered[-1], 3),
        "mean": round(sum(ordered) / n, 3),
    }


# Rendering


def _markdown(report: ValidationReport) -> str:
    outcome = "no errors" if report.ok else f"{report.n_errors} errors"
    lines = [
        f"# Validation report — {report.source_name or 'annotations'}",
        "",
        f"- file sha256: `{report.source_sha256 or 'n/a'}`",
        f"- rows read: **{report.n_rows}**, over **{report.n_videos}** videos",
        f"- outcome: **{outcome}**",
        "",
        "Generated by `evd.annotations.validate`. The reference inventory is",
        "`data/ucf_crime_inventory/inventory.csv`: fps and `num_frames` are",
        "measured by decoding the videos, not read from headers.",
        "",
    ]

    if report.counts:
        lines += ["## Findings summary", "", "| code | occurrences |", "|---|---|"]
        lines += [f"| `{code}` | {n} |" for code, n in report.counts.items()]
        lines.append("")

    segments = report.stats.get("segments_per_class", {})
    videos = report.stats.get("videos_per_class", {})
    if segments or videos:
        frames = report.stats.get("segment_frames", {})
        seconds = report.stats.get("segment_seconds", {})
        lines += [
            "## Per-class coverage",
            "",
            "| class | videos | segments | median duration (frames) | median duration (s) |",
            "|---|---|---|---|---|",
        ]
        for name in sorted(set(segments) | set(videos)):
            lines.append(
                f"| {name} | {videos.get(name, 0)} | {segments.get(name, 0)} "
                f"| {frames.get(name, {}).get('median', '—')} "
                f"| {seconds.get(name, {}).get('median', '—')} |"
            )
        lines.append("")

    base = report.stats.get("out_of_range_by_index_base", {})
    if base:
        lines += [
            "## Index base",
            "",
            "Intervals that would exceed `num_frames` reading the file as",
            f"0-based: **{base.get(0, 0)}** · as 1-based: **{base.get(1, 0)}**.",
            "If the two numbers differ, the convention can be inferred from the data;",
            "if both are zero, the data cannot tell and it must be asked.",
            "",
        ]

    if report.findings:
        lines += [
            "## Findings",
            "",
            "| severity | code | video | detail | source |",
            "|---|---|---|---|---|",
        ]
        lines += [
            f"| {f.severity} | `{f.code}` | {f.video_id or '—'} | {f.detail} | {f.source or '—'} |"
            for f in report.findings
        ]
        lines.append("")

    truncated = {code: n for code, n in report.counts.items() if n > MAX_FINDINGS_PER_CODE}
    if truncated:
        lines += [
            f"> List truncated to {MAX_FINDINGS_PER_CODE} cases per code: "
            + ", ".join(f"`{k}` ({v} in total)" for k, v in sorted(truncated.items())),
            "",
        ]

    return "\n".join(lines)
