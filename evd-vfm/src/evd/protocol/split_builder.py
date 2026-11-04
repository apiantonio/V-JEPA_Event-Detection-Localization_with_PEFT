"""How the sealed ``edp_v1`` split is generated.

The official UCF-Crime splits declare only train and test; the validation partition
(used for learning rate, early stopping and threshold) was not delivered.
This module carves it out of training, per video, stratified by class, with the quota
declared in ``configs/protocol/edp_v1.yaml`` (``validation.fraction``):

* carved from training, never from test;
* the class of a video comes from its annotation, not its folder;
* stratified per video, not per chunk, so no video straddles the boundary.

``edp_v2`` uses the laboratory partition instead.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np

from evd.annotations.schema import TemporalAnnotation
from evd.protocol.split import SplitSpec
from evd.utils.logging import get_logger

log = get_logger(__name__)

ValSource = Literal["holdout_from_train", "laboratory"]
StratifyRule = Literal["video_class", "none"]
DedupRule = Literal["none", "sha256_representative"]

#: Stratum key of a video: the sorted classes of its intervals. The empty tuple is
#: ``Normal`` (a normal video has no intervals).
Stratum = tuple[int, ...]


@dataclass(frozen=True)
class ValidationRules:
    """Carving rule; no field has a default on purpose (it is a protocol decision).

    :param source: ``holdout_from_train`` is the Q-C fallback; ``laboratory`` is for a
        delivered partition, which this module does not generate.
    :param fraction: fraction of training videos moved to validation.
    :param stratify: ``video_class`` stratifies on the set of annotated classes of the
        video; ``none`` carves uniformly (toy datasets only).
    :param seed: carving seed, which makes the split repeatable (not a run seed).
    :param dedup: ``sha256_representative`` applies Q19 to the training pool: files whose
        content appears in test are dropped and one representative is kept per hash.
    """

    source: ValSource
    fraction: float
    stratify: StratifyRule
    seed: int
    dedup: DedupRule

    def __post_init__(self) -> None:
        if self.source not in ("holdout_from_train", "laboratory"):
            raise ValueError(f"unknown validation.source: {self.source!r}")
        if self.source == "laboratory":
            raise NotImplementedError(
                "validation.source='laboratory' means the partition was delivered: "
                "read it from their file, do not generate it here (docs/04 §7.3-Q-C)"
            )
        if not 0.0 < self.fraction < 1.0:
            raise ValueError(
                f"validation.fraction must be in (0, 1), not {self.fraction}: "
                f"0 would mean no validation (rejected by the runner) and 1 no training"
            )
        if self.stratify not in ("video_class", "none"):
            raise ValueError(f"unknown validation.stratify: {self.stratify!r}")
        if self.dedup not in ("none", "sha256_representative"):
            raise ValueError(f"unknown dedup: {self.dedup!r}")


@dataclass(frozen=True)
class SplitReport:
    """Outcome of the carving, printable before writing the file.

    :param per_stratum: stratum to ``(n_train, n_val)`` after the carving.
    :param nomi: stratum to readable name, for the table.
    :param strati_senza_validazione: strata too small for ``round(fraction · n)`` to give
        at least 1, kept whole in training (on UCF-Crime only rare class combinations).
    :param esclusi: dedup exclusions, ``(video_id, reason)``.
    """

    per_stratum: dict[Stratum, tuple[int, int]]
    nomi: dict[Stratum, str]
    strati_senza_validazione: tuple[Stratum, ...]
    esclusi: tuple[tuple[str, str], ...]

    def tabella(self) -> str:
        """:return: the strata table, one row per stratum plus the total."""
        rows = ["  stratum                         train    val   share"]
        for stratum in sorted(self.per_stratum, key=lambda s: self.nomi[s]):
            n_train, n_val = self.per_stratum[stratum]
            total = n_train + n_val
            quota = n_val / total if total else 0.0
            rows.append(f"  {self.nomi[stratum]:<28} {n_train:>6} {n_val:>6}  {quota:>6.1%}")
        t = sum(n for n, _ in self.per_stratum.values())
        v = sum(n for _, n in self.per_stratum.values())
        rows.append(f"  {'TOTALE':<28} {t:>6} {v:>6}  {v / (t + v) if t + v else 0:>6.1%}")
        return "\n".join(rows)


def stratum_of(
    video_id: str, annotations: Mapping[str, TemporalAnnotation], *, normal_idx: int
) -> Stratum:
    """Return the stratum of a video: the classes of its intervals, or ``Normal``.

    :param video_id: the video.
    :param annotations: loaded annotations.
    :param normal_idx: canonical index of ``Normal`` (13 on UCF-Crime).
    :return: sorted tuple of annotated classes; ``(normal_idx,)`` for a video without
        intervals.
    :raises KeyError: if the video is missing from the annotations (never treated as
        normal).
    """
    ann = annotations[video_id]
    classes = sorted({class_idx for _, _, class_idx in ann.intervals})
    return tuple(classes) if classes else (normal_idx,)


def build_split(
    *,
    dataset: str,
    protocol_version: str,
    train_videos: Sequence[str],
    test_videos: Sequence[str],
    annotations: Mapping[str, TemporalAnnotation],
    normal_idx: int,
    rules: ValidationRules,
    class_names: Mapping[int, str] | None = None,
    content_sha: Mapping[str, str] | None = None,
    annotation_source: str = "",
    builder_commit: str = "",
    created: str = "",
) -> tuple[SplitSpec, SplitReport]:
    """Build the sealed ``edp_v1`` split.

    :param dataset: dataset name (``"ucf_crime"``).
    :param protocol_version: ``"edp_v1"``.
    :param train_videos: official training pool (``Anomaly_Train.txt``).
    :param test_videos: official evaluation split, left untouched.
    :param annotations: loaded annotations, indexed by ``video_id``.
    :param normal_idx: canonical index of ``Normal``.
    :param rules: carving rule, from the protocol file.
    :param class_names: index to name, only for the report.
    :param content_sha: ``video_id`` to sha256 from the inventory, used by the dedup.
    :param annotation_source: ``"ucf_lab@<sha>"``, bound to the split by the seal.
    :param builder_commit: sha of the generating commit.
    :param created: ISO date; defaults to today.
    :return: the sealed split and the strata report.
    :raises ValueError: if the official pools overlap, dedup is requested without
        ``content_sha``, a video is missing from the annotations, or two byte-identical
        duplicates carry discordant classes (Q19).
    """
    overlap = set(train_videos) & set(test_videos)
    if overlap:
        raise ValueError(
            f"the two official splits overlap on {len(overlap)} videos "
            f"(e.g. {sorted(overlap)[:3]}): not a partition, and no carving can fix it"
        )
    if rules.dedup != "none" and not content_sha:
        raise ValueError(f"dedup={rules.dedup!r} requires content_sha from the inventory")

    pool, excluded = _dedup(
        train_videos,
        test_videos,
        annotations=annotations,
        content_sha=content_sha or {},
        rule=rules.dedup,
    )

    strata: dict[Stratum, list[str]] = defaultdict(list)
    for video_id in pool:
        key = (
            stratum_of(video_id, annotations, normal_idx=normal_idx)
            if rules.stratify == "video_class"
            else ()
        )
        strata[key].append(video_id)

    val: list[str] = []
    per_stratum: dict[Stratum, tuple[int, int]] = {}
    without_val: list[Stratum] = []
    for stratum in sorted(strata):
        candidati = sorted(strata[stratum])
        how_many = _quota(len(candidati), rules.fraction)
        if how_many == 0:
            without_val.append(stratum)
        # One generator per stratum, seeded by its key: adding a class does not reshuffle the
        # others.
        rng = np.random.default_rng([rules.seed, *stratum])
        chosen = [candidati[i] for i in rng.permutation(len(candidati))[:how_many]]
        val.extend(chosen)
        per_stratum[stratum] = (len(candidati) - how_many, how_many)

    val_chosen = set(val)
    split = SplitSpec(
        protocol_version=protocol_version,
        dataset=dataset,
        annotation_source=annotation_source,
        builder_commit=builder_commit,
        created=created or date.today().isoformat(),
        train=tuple(v for v in pool if v not in val_chosen),
        val=tuple(sorted(val_chosen)),
        test=tuple(sorted(test_videos)),
        excluded=excluded,
    ).sealed()

    names = {
        stratum: _stratum_name(stratum, class_names or {}, normal_idx=normal_idx)
        for stratum in per_stratum
    }
    report = SplitReport(
        per_stratum=per_stratum,
        nomi=names,
        strati_senza_validazione=tuple(without_val),
        esclusi=excluded,
    )
    if without_val:
        log.warning(
            "%d strata are too small for the quota and stay whole in training: %s",
            len(without_val),
            [names[s] for s in without_val],
        )
    return split, report


def _quota(n: int, fraction: float) -> int:
    """Return how many videos of a stratum go to validation: ``round(fraction · n)``.

    No floor at 1: small strata are rare class combinations, and forcing one video out of
    two would move 50% of a stratum to validation. Strata left at zero are declared in the
    report. The ``n - 1`` cap keeps at least one training video per stratum.

    :param n: videos in the stratum.
    :param fraction: requested quota.
    :return: number of videos to move; ``0`` if the stratum is too small.
    """
    return min(n - 1, round(fraction * n)) if n >= 2 else 0


def _dedup(
    train_videos: Sequence[str],
    test_videos: Sequence[str],
    *,
    annotations: Mapping[str, TemporalAnnotation],
    content_sha: Mapping[str, str],
    rule: DedupRule,
) -> tuple[list[str], tuple[tuple[str, str], ...]]:
    """Apply Q19 to the training pool: the 21 byte-identical pairs.

    The test split is never pruned (it keeps metrics comparable with the literature): the
    duplicate is removed from training.

    :return: the remaining pool, sorted, and the exclusions with their reason.
    :raises ValueError: if two duplicates carry discordant classes.
    """
    if rule == "none":
        return sorted(train_videos), ()

    hash_test = {content_sha[v] for v in test_videos if v in content_sha}
    per_hash: dict[str, list[str]] = defaultdict(list)
    kept: list[str] = []
    excluded: list[tuple[str, str]] = []
    for video_id in sorted(train_videos):
        digest = content_sha.get(video_id)
        if digest is None:
            kept.append(video_id)
            continue
        if digest in hash_test:
            excluded.append((video_id, f"sha256 {digest[:12]} presente nello split di test"))
            continue
        per_hash[digest].append(video_id)

    for digest, group in sorted(per_hash.items()):
        if len(group) > 1:
            labels = {_classes_of(v, annotations) for v in group}
            if len(labels) > 1:
                raise ValueError(
                    f"duplicates with discordant classes (sha {digest[:12]}): {sorted(group)} "
                    f"-> {sorted(labels)}. This is Q19: resolve it with the laboratory "
                    f"annotations, not by picking the file that looks right"
                )
            excluded.extend(
                (v, f"duplicato byte-identico di {group[0]} (sha {digest[:12]})") for v in group[1:]
            )
        kept.append(group[0])
    return sorted(kept), tuple(sorted(excluded))


def _classes_of(video_id: str, annotations: Mapping[str, TemporalAnnotation]) -> tuple[int, ...]:
    ann = annotations.get(video_id)
    return tuple(sorted({c for _, _, c in ann.intervals})) if ann else ()


def _stratum_name(stratum: Stratum, class_names: Mapping[int, str], *, normal_idx: int) -> str:
    """:return: readable stratum name, or ``'(tutti)'`` without stratification."""
    if not stratum:
        return "(tutti)"
    return "+".join(class_names.get(idx, str(idx)) for idx in stratum) or str(normal_idx)
