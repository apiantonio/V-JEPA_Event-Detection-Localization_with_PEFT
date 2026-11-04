"""Sealed split of the EDP protocol.

In the fully supervised regime the protocol declares which videos belong to which
partition; the chunk grid, labels and sampling weights are a pure function of that and
of the annotations (:mod:`evd.protocol.chunks`). The grid is not sealed here: runs
record its count and measured positive rate, a verifiable signature of the grid.

Invariant 1: a protocol file consumed by a run is never rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from evd.utils.hashing import stable_hash
from evd.utils.io import read_json, write_json

SCHEMA_FIELDS = (
    "protocol_version",
    "dataset",
    "annotation_source",
    "builder_commit",
    "created",
    "train",
    "val",
    "test",
    "excluded",
)


@dataclass(frozen=True)
class SplitSpec:
    """The three sealed video partitions.

    :param protocol_version: protocol form (``"edp_v1"``, ``"edp_v2"``); a few-shot
        ``fsp_*`` file cannot be loaded by mistake.
    :param dataset: dataset name (``"ucf_crime"``).
    :param annotation_source: ``"ucf_lab@<annotation file sha>"``, binding the split to
        the exact annotation version.
    :param train: training videos, in order.
    :param val: validation videos (LR selection, early stopping, threshold; docs/04 §5.3).
        May be empty, in which case trainers must say so.
    :param test: evaluation videos, never used in training.
    :param excluded: ``(video_id, reason)`` pairs (byte-identical duplicates and files
        outside the split), counted and declared.
    :param sha256: seal of the content without this field.
    """

    protocol_version: str
    dataset: str
    annotation_source: str
    builder_commit: str
    created: str
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    excluded: tuple[tuple[str, str], ...] = ()
    sha256: str = ""

    def __post_init__(self) -> None:
        for name, group in (("train", self.train), ("val", self.val), ("test", self.test)):
            if len(set(group)) != len(group):
                raise ValueError(f"split '{name}' contains repeated videos")
        # Partitions must be disjoint: a video in two partitions makes every evaluation a
        # measurement on seen data.
        for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
            common = set(getattr(self, a)) & set(getattr(self, b))
            if common:
                raise ValueError(
                    f"splits not disjoint: {len(common)} videos in '{a}' and '{b}' "
                    f"(e.g. {sorted(common)[:3]}). A video belongs to one partition only."
                )
        if not self.train:
            raise ValueError("the training split is empty: nothing to train on")
        if not self.test:
            raise ValueError("the evaluation split is empty: nothing to measure")

    @property
    def has_validation(self) -> bool:
        """:return: whether a validation partition exists (a legitimate but declared state when
        false; trainers must never silently fall back on the test set).
        """
        return bool(self.val)

    def payload(self) -> dict[str, Any]:
        """:return: the content without ``sha256``, i.e. what the seal covers."""
        return {
            "protocol_version": self.protocol_version,
            "dataset": self.dataset,
            "annotation_source": self.annotation_source,
            "builder_commit": self.builder_commit,
            "created": self.created,
            "train": list(self.train),
            "val": list(self.val),
            "test": list(self.test),
            "excluded": [list(pair) for pair in self.excluded],
        }

    def sealed(self) -> SplitSpec:
        """:return: the same spec with the computed seal."""
        return replace(self, sha256=split_sha256(self.payload()))


def split_sha256(payload: dict[str, Any]) -> str:
    """:return: stable sha256 of the content, excluding the ``sha256`` field."""
    return stable_hash({k: v for k, v in payload.items() if k != "sha256"})


def save_split(path: str | Path, split: SplitSpec) -> Path:
    """Write the sealed split, refusing to overwrite an existing one.

    :param path: destination.
    :param split: the split; sealed here if it is not already.
    :return: the written path.
    :raises FileExistsError: if the file exists (invariant 1).
    """
    output = Path(path)
    if output.exists():
        raise FileExistsError(
            f"{output} already exists: a split is never rewritten (invariant 1). "
            f"A different value is a new protocol, with a new file."
        )
    sealed = split.sealed()
    return write_json(output, {**sealed.payload(), "sha256": sealed.sha256})


def load_split(path: str | Path) -> SplitSpec:
    """Read a split, verifying schema and seal.

    :param path: file to read.
    :return: the split.
    :raises ValueError: on a missing schema field, a missing or mismatching seal, or a
        ``protocol_version`` of another regime.
    """
    source = Path(path)
    raw = read_json(source)
    missing = [field for field in SCHEMA_FIELDS if field not in raw]
    if missing:
        raise ValueError(f"{source}: missing fields in the split: {missing}")
    if "sha256" not in raw:
        raise ValueError(f"{source}: split without sha256 seal")
    version = str(raw["protocol_version"])
    if version.startswith("fsp"):
        raise ValueError(
            f"{source}: protocol_version '{version}' belongs to the few-shot regime, "
            f"replaced by ADR-031. Those files remain readable with "
            f"evd.protocol.manifest, not with this module."
        )
    split = SplitSpec(
        protocol_version=version,
        dataset=str(raw["dataset"]),
        annotation_source=str(raw["annotation_source"]),
        builder_commit=str(raw["builder_commit"]),
        created=str(raw["created"]),
        train=tuple(raw["train"]),
        val=tuple(raw["val"]),
        test=tuple(raw["test"]),
        excluded=tuple((str(a), str(b)) for a, b in raw["excluded"]),
        sha256=str(raw["sha256"]),
    )
    expected = split_sha256(split.payload())
    if expected != split.sha256:
        raise ValueError(
            f"{source}: invalid seal (expected {expected}, found {split.sha256}). "
            f"The file was modified after generation: not usable."
        )
    return split
