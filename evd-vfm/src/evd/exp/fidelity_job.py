"""fp16 vs fp32 comparison between two feature stores.

Does rereading from an fp16 cache return the same features as an fp32 cache? The
criterion lives in ``evd/eval/fidelity.py`` and is declared before the measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evd.data.feature_store import SafetensorsFeatureStore
from evd.eval.fidelity import compare_stores
from evd.exp.tracker import LocalTracker
from evd.utils.io import write_json


class StoresNotComparable(RuntimeError):
    """The two stores do not describe the same computation, so they cannot be compared."""


#: Axes along which two stores can be compared.
#:
#: ``storage_dtype``: pure rounding; the threshold derives from the dtype epsilon and the
#: comparison has a verdict.
#:
#: ``precision``: AMP bf16 versus fp32 computation; the error accumulates over 24 blocks
#: with no a priori bound, so the comparison measures without a verdict.
AXES = ("storage_dtype", "precision")


@dataclass(frozen=True)
class FidelityConfig:
    reference_dir: Path
    candidate_dir: Path
    run_id: str
    axis: str = "storage_dtype"
    tolerance: float | None = None


def _assert_comparable(
    reference: SafetensorsFeatureStore,
    candidate: SafetensorsFeatureStore,
    *,
    axis: str = "storage_dtype",
) -> None:
    """Check that only the declared axis varies and everything else is identical.

    :raises StoresNotComparable: on an unknown axis or any other differing field.
    """
    if axis not in AXES:
        raise StoresNotComparable(f"unknown axis {axis!r}; allowed {list(AXES)}")
    invariants = [
        field_name for field_name in ("backbone_id", "ckpt_sha", *AXES) if field_name != axis
    ]
    for field in invariants:
        expected = getattr(reference, field)
        actual = getattr(candidate, field)
        if expected != actual:
            raise StoresNotComparable(
                f"{field}: reference {expected!r}, candidate {actual!r}. "
                f"Comparing along {axis!r}, {field} must match."
            )
    if reference.preprocess.hash() != candidate.preprocess.hash():
        raise StoresNotComparable(
            f"different preprocess_hash: {reference.preprocess.hash()[:12]} vs "
            f"{candidate.preprocess.hash()[:12]}"
        )
    reference_value = getattr(reference, axis)
    candidato = getattr(candidate, axis)
    if reference_value == candidato:
        raise StoresNotComparable(
            f"both stores have {axis}={reference_value!r}: nothing to compare."
        )
    if reference_value != "fp32":
        raise StoresNotComparable(
            f"the reference has {axis}={reference_value!r}, not fp32: fidelity is "
            "measured against the most precise representation."
        )


def run_fidelity_job(config: FidelityConfig) -> dict[str, Any]:
    """Open the two stores, compare and register; never raises if the criterion fails.

    A failing criterion is a result: the report is written and registered with ``passed``
    false, and the CLI exits non-zero.

    :raises StoresNotComparable: if the stores are not comparable or the reference is empty.
    """
    reference = SafetensorsFeatureStore.open_existing(config.reference_dir)
    candidate = SafetensorsFeatureStore.open_existing(config.candidate_dir)
    _assert_comparable(reference, candidate, axis=config.axis)

    keys = reference.keys()
    if not keys:
        raise StoresNotComparable(f"the reference store is empty: {config.reference_dir}")

    tracker = LocalTracker()
    handle = tracker.start(
        config.run_id,
        {
            "reference_dir": str(reference.root),
            "candidate_dir": str(candidate.root),
            "reference_meta": reference.metadata,
            "candidate_meta": candidate.metadata,
            "axis": config.axis,
            "tolerance": config.tolerance,
        },
        official=True,
        regime={
            "axis": config.axis,
            "attn": "sdpa",
            "reference": f"{reference.precision}/{reference.storage_dtype}",
            "candidate": f"{candidate.precision}/{candidate.storage_dtype}",
        },
    )

    report = compare_stores(
        reference,
        candidate,
        keys,
        storage_dtype=candidate.storage_dtype,
        tolerance=config.tolerance,
        # On the `precision` axis there is no derivable threshold: measure and report.
        derive_tolerance=config.axis == "storage_dtype",
    )
    payload = {
        "run_id": config.run_id,
        "axis": config.axis,
        "reference_dir": str(reference.root),
        "candidate_dir": str(candidate.root),
        "reference": f"{reference.precision}/{reference.storage_dtype}",
        "candidate": f"{candidate.precision}/{candidate.storage_dtype}",
        **report.as_dict(),
    }
    # Axis and geometry in the name, so different comparisons under the same run_id do not
    # overwrite each other.
    name = f"fidelity-{config.axis}-{reference.preprocess.hash()[:12]}.json"
    artifact = write_json(handle.dir / name, payload)
    tracker.finish(
        handle,
        {**report.as_dict(), "fidelity_json": str(artifact)},
        extra={"exp": "e0d", "part": "fidelity"},
    )
    return payload
