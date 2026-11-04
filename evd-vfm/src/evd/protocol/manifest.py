"""Immutable schema of the few-shot manifest.

``k`` counts annotated event segments per class and ``m`` caps the training windows
produced by each segment. Manifest intervals are half-open seconds
``[start_s, end_s)``; inclusive frames belong only to the native annotations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from evd.types import EventClass
from evd.utils.hashing import stable_hash
from evd.utils.io import read_json, write_json

BudgetUnit = Literal["annotated_event_segments_per_class"]

#: The "all available supervision" budget. Not an integer because it differs per class
#: (docs/12 Q2): from 44 ``Shooting`` segments to 250 ``Robbery`` ones on FS-UCF-Crime.
K_FULL = "full"

#: Protocol ``k``: a positive integer or :data:`K_FULL`.
KBudget = int | str

SCHEMA_FIELDS = (
    "protocol_version",
    "dataset",
    "budget",
    "seed",
    "created",
    "builder_commit",
    "annotation_source",
    "classes",
    "support",
    "support_normal",
    "query",
)


def normalize_k(value: Any) -> KBudget:
    """Return a protocol ``k`` in canonical form.

    :param value: value read from YAML, manifest or config.
    :return: ``int > 0`` or :data:`K_FULL`.
    :raises ValueError: if it is neither a positive integer nor ``"full"``.
    """
    if isinstance(value, str) and value.strip() == K_FULL:
        return K_FULL
    try:
        whole = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"unrecognized k: {value!r} (expected integers > 0 or {K_FULL!r})"
        ) from None
    if whole <= 0:
        raise ValueError(f"k must be > 0, not {whole}")
    return whole


def k_sort_key(value: KBudget) -> tuple[int, int]:
    """Return a sort key placing :data:`K_FULL` last.

    :param value: an already normalized ``k``.
    :return: the sort key.
    """
    return (1, 0) if value == K_FULL else (0, int(value))


@dataclass(frozen=True)
class ProtocolBudget:
    """Supervision budget associated with a manifest.

    :param k: annotated event segments per anomaly class, or :data:`K_FULL` for all
        available (a different number per class, always reported with the actual counts).
    :param m: maximum number of training windows derived from a segment.
    :param unit: canonical unit of ``k``; no implicit aliases.
    """

    k: KBudget
    m: int
    unit: BudgetUnit

    def __post_init__(self) -> None:
        object.__setattr__(self, "k", normalize_k(self.k))
        if self.m <= 0:
            raise ValueError(f"m must be > 0, not {self.m}")
        if self.unit != "annotated_event_segments_per_class":
            raise ValueError(f"unrecognized budget unit: {self.unit!r}")

    @property
    def is_full(self) -> bool:
        """Return ``True`` if the budget is :data:`K_FULL`."""
        return self.k == K_FULL

    def as_dict(self) -> dict[str, int | str]:
        return {"k": self.k, "m": self.m, "unit": self.unit}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ProtocolBudget:
        return cls(k=normalize_k(raw["k"]), m=int(raw["m"]), unit=raw["unit"])


@dataclass(frozen=True)
class SupportItem:
    """Labelled segment with a half-open interval in seconds.

    :param video_id: canonical video identifier.
    :param interval_s: interval ``[start_s, end_s)`` in seconds.
    :param class_idx: canonical class index.
    """

    video_id: str
    interval_s: tuple[float, float]
    class_idx: int

    def __post_init__(self) -> None:
        start, end = self.interval_s
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError(f"{self.video_id}: non-finite interval {self.interval_s}")
        if start < 0.0 or end <= start:
            raise ValueError(f"{self.video_id}: invalid half-open interval {self.interval_s}")
        if self.class_idx < 0:
            raise ValueError(f"{self.video_id}: negative class_idx {self.class_idx}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "interval_s": [self.interval_s[0], self.interval_s[1]],
            "class_idx": self.class_idx,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SupportItem:
        start, end = raw["interval_s"]
        return cls(raw["video_id"], (float(start), float(end)), int(raw["class_idx"]))


@dataclass(frozen=True)
class FewShotManifest:
    protocol_version: str
    dataset: str
    budget: ProtocolBudget
    seed: int
    created: str
    builder_commit: str
    annotation_source: str
    """``ucf_lab@<sha>``: binds the manifest to the source annotations."""
    classes: tuple[EventClass, ...]
    support: tuple[SupportItem, ...]
    support_normal: tuple[SupportItem, ...]
    query: tuple[str, ...]
    sha256: str = ""

    @property
    def k(self) -> KBudget:
        """Return the number of annotated event segments per class, or :data:`K_FULL`."""
        return self.budget.k

    @property
    def m(self) -> int:
        """Return the maximum number of training windows per annotated segment."""
        return self.budget.m

    def payload(self) -> dict[str, Any]:
        """Return the content without ``sha256``, used for the seal."""
        return {
            "protocol_version": self.protocol_version,
            "dataset": self.dataset,
            "budget": self.budget.as_dict(),
            "seed": self.seed,
            "created": self.created,
            "builder_commit": self.builder_commit,
            "annotation_source": self.annotation_source,
            "classes": [
                {"idx": c.idx, "name": c.name, "text_prompts": list(c.text_prompts)}
                for c in self.classes
            ],
            "support": [item.as_dict() for item in self.support],
            "support_normal": [item.as_dict() for item in self.support_normal],
            "query": list(self.query),
        }

    def sealed(self) -> FewShotManifest:
        from dataclasses import replace

        return replace(self, sha256=manifest_sha256(self.payload()))

    def support_videos(self) -> frozenset[str]:
        return frozenset(item.video_id for item in (*self.support, *self.support_normal))


def manifest_sha256(payload: dict[str, Any]) -> str:
    return stable_hash({key: value for key, value in payload.items() if key != "sha256"})


def save_manifest(path: str | Path, manifest: FewShotManifest) -> Path:
    """Write the sealed manifest without overwriting existing files.

    :raises FileExistsError: if the file already exists.
    """
    output = Path(path)
    if output.exists():
        raise FileExistsError(
            f"{output} already exists. Manifests are immutable: to change the "
            "protocol generate fsp_v(N+1) with a changelog and approval."
        )
    sealed = manifest.sealed()
    return write_json(output, {**sealed.payload(), "sha256": sealed.sha256})


def load_manifest(path: str | Path) -> FewShotManifest:
    """Load a manifest, verifying schema and seal.

    :raises ValueError: on missing fields or an invalid seal.
    """
    source = Path(path)
    raw: dict[str, Any] = read_json(source)
    missing = [field for field in SCHEMA_FIELDS if field not in raw]
    if missing:
        raise ValueError(f"{source}: missing fields in the manifest: {missing}")
    if "sha256" not in raw:
        raise ValueError(f"{source}: manifest without sha256 seal")

    manifest = FewShotManifest(
        protocol_version=raw["protocol_version"],
        dataset=raw["dataset"],
        budget=ProtocolBudget.from_dict(raw["budget"]),
        seed=int(raw["seed"]),
        created=raw["created"],
        builder_commit=raw["builder_commit"],
        annotation_source=raw["annotation_source"],
        classes=tuple(
            EventClass(
                idx=int(item["idx"]),
                name=item["name"],
                text_prompts=tuple(item.get("text_prompts", ())),
            )
            for item in raw["classes"]
        ),
        support=tuple(SupportItem.from_dict(item) for item in raw["support"]),
        support_normal=tuple(SupportItem.from_dict(item) for item in raw["support_normal"]),
        query=tuple(raw["query"]),
        sha256=raw["sha256"],
    )
    expected = manifest_sha256(manifest.payload())
    if expected != manifest.sha256:
        raise ValueError(
            f"{source}: invalid seal (expected {expected}, found {manifest.sha256}). "
            "The file was modified after generation."
        )
    return manifest
