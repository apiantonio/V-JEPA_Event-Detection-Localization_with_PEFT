"""Numerical fidelity of a feature store against the fp32 reference.

Used to choose the storage dtype of the feature store. The tolerance is derived from
the dtype epsilon before running the comparison, never chosen after seeing the error.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch

from evd.data.feature_store import FeatureReader

_DTYPES = {"fp16": torch.float16, "fp32": torch.float32}

#: Multiple of the theoretical rounding bound granted as tolerance.
TOLERANCE_MARGIN = 2.0


def storage_dtype_tolerance(storage_dtype: str, *, margin: float = TOLERANCE_MARGIN) -> float:
    """Return the maximum relative L2 error allowed for a store in the given dtype.

    Rounding to nearest bounds the relative error of each element by ``finfo.eps / 2``
    (4.88e-4 for fp16), hence the threshold ``margin * eps / 2``. It bounds rounding only:
    overflow and underflow produce ``inf`` or zeros, counted separately by
    :func:`compare_stores`.

    :param storage_dtype: storage dtype.
    :param margin: multiple of the rounding bound.
    :return: the tolerance.
    :raises ValueError: on an unknown dtype or a non-positive margin.
    """
    if storage_dtype not in _DTYPES:
        raise ValueError(f"storage_dtype={storage_dtype!r}: allowed {sorted(_DTYPES)}")
    if margin <= 0:
        raise ValueError(f"margin must be > 0, not {margin}")
    return margin * torch.finfo(_DTYPES[storage_dtype]).eps / 2


def relative_l2(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """Return ``||candidate - reference|| / ||reference||``, computed in fp64.

    :param reference: reference tensor.
    :param candidate: candidate tensor.
    :return: the relative L2 error.
    :raises ValueError: if the shapes differ.
    """
    if reference.shape != candidate.shape:
        raise ValueError(
            f"different shapes: reference {tuple(reference.shape)}, "
            f"candidate {tuple(candidate.shape)}"
        )
    ref = reference.detach().to(dtype=torch.float64)
    cand = candidate.detach().to(dtype=torch.float64)
    denominator = torch.linalg.vector_norm(ref).item()
    if denominator == 0.0:
        # An all-zero reference has no scale: the error is 0 if the candidate is also zero,
        # otherwise infinite.
        return 0.0 if torch.equal(ref, cand) else float("inf")
    return float(torch.linalg.vector_norm(cand - ref).item() / denominator)


def _percentile(valori: Sequence[float], q: float) -> float:
    """Return a percentile by linear interpolation, without numpy."""
    if not valori:
        return float("nan")
    sorted_items = sorted(valori)
    if len(sorted_items) == 1:
        return sorted_items[0]
    position = q * (len(sorted_items) - 1)
    bottom = int(position)
    top = min(bottom + 1, len(sorted_items) - 1)
    fraction = position - bottom
    return sorted_items[bottom] * (1.0 - fraction) + sorted_items[top] * fraction


@dataclass(frozen=True)
class ScaleReference:
    """Typical distance between different windows inside one store.

    The denominator that makes a fidelity error readable; the deciding statistic is
    ``same_video_p05``, since perturbations larger than the smallest separations can flip
    comparisons.
    """

    n_same_video: int
    n_cross_video: int
    same_video_mean: float
    same_video_p05: float
    cross_video_mean: float
    cross_video_p05: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def scale_reference(
    reader: FeatureReader,
    groups: Sequence[Sequence[str]],
    *,
    n_pairs: int = 200,
    seed: int = 0,
) -> ScaleReference:
    """Sample window pairs and measure their distance.

    ``groups`` holds keys grouped by video and layer. Two populations are measured:
    same video with different windows, and different videos.

    :param reader: feature reader.
    :param groups: keys grouped by video and layer.
    :param n_pairs: pairs to sample per population.
    :param seed: sampling seed.
    :return: the scale reference.
    :raises ValueError: if no group has at least two keys.
    """
    rng = random.Random(seed)
    usable = [list(g) for g in groups if len(g) >= 2]
    if not usable:
        raise ValueError("at least two keys in at least one group are required to sample pairs")

    def distance(a: str, b: str) -> float | None:
        first, second = reader.get(a), reader.get(b)
        if first is None or second is None or first.tokens.shape != second.tokens.shape:
            return None
        return relative_l2(first.tokens, second.tokens)

    same: list[float] = []
    for _ in range(n_pairs):
        group = rng.choice(usable)
        a, b = rng.sample(group, 2)
        value = distance(a, b)
        if value is not None:
            same.append(value)

    cross: list[float] = []
    if len(usable) >= 2:
        for _ in range(n_pairs):
            ga, gb = rng.sample(usable, 2)
            value = distance(rng.choice(ga), rng.choice(gb))
            if value is not None:
                cross.append(value)

    return ScaleReference(
        n_same_video=len(same),
        n_cross_video=len(cross),
        same_video_mean=sum(same) / len(same) if same else float("nan"),
        same_video_p05=_percentile(same, 0.05),
        cross_video_mean=sum(cross) / len(cross) if cross else float("nan"),
        cross_video_p05=_percentile(cross, 0.05),
    )


@dataclass(frozen=True)
class FidelityReport:
    """Outcome of the comparison, with the criterion included."""

    n_keys: int
    n_compared: int
    n_missing: int
    n_nonfinite: int
    n_shape_mismatch: int
    rel_l2_max: float
    rel_l2_mean: float
    max_abs_reference: float
    tolerance: float | None
    storage_dtype: str

    @property
    def passed(self) -> bool:
        """Return whether the structural checks pass and, if a tolerance exists, the threshold.

        ``tolerance is None`` means a measurement without verdict (e.g. accumulated bf16
        error, which has no a priori bound).
        """
        structural = (
            self.n_missing == 0
            and self.n_shape_mismatch == 0
            and self.n_nonfinite == 0
            and self.n_compared > 0
        )
        if self.tolerance is None:
            return structural
        return structural and self.rel_l2_max <= self.tolerance

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "passed": self.passed}


def compare_stores(
    reference: FeatureReader,
    candidate: FeatureReader,
    keys: Iterable[str],
    *,
    storage_dtype: str,
    tolerance: float | None = None,
    derive_tolerance: bool = True,
) -> FidelityReport:
    """Compare two stores on the same keys, the reference being the fp32 one.

    :param reference: fp32 reference reader.
    :param candidate: candidate reader.
    :param keys: keys to compare.
    :param storage_dtype: candidate storage dtype.
    :param tolerance: explicit tolerance, or ``None``.
    :param derive_tolerance: derive the tolerance from ``storage_dtype``.
    :return: the fidelity report.
    """
    if tolerance is not None:
        limit: float | None = tolerance
    elif derive_tolerance:
        limit = storage_dtype_tolerance(storage_dtype)
    else:
        limit = None
    n_keys = n_compared = n_missing = n_nonfinite = n_shape_mismatch = 0
    rel_l2_max = 0.0
    rel_l2_total = 0.0
    max_abs_reference = 0.0

    for key in keys:
        n_keys += 1
        expected = reference.get(key)
        actual = candidate.get(key)
        if expected is None or actual is None:
            n_missing += 1
            continue
        if expected.tokens.shape != actual.tokens.shape:
            n_shape_mismatch += 1
            continue
        if expected.grid != actual.grid or expected.layer != actual.layer:
            # Same key with different metadata means the key does not cover the spec: a key bug,
            # counted as such.
            n_shape_mismatch += 1
            continue
        if not bool(torch.isfinite(actual.tokens).all().item()):
            n_nonfinite += 1
            continue
        n_compared += 1
        value = relative_l2(expected.tokens, actual.tokens)
        rel_l2_max = max(rel_l2_max, value)
        rel_l2_total += value
        max_abs_reference = max(
            max_abs_reference,
            float(expected.tokens.detach().abs().max().item()),
        )

    return FidelityReport(
        n_keys=n_keys,
        n_compared=n_compared,
        n_missing=n_missing,
        n_nonfinite=n_nonfinite,
        n_shape_mismatch=n_shape_mismatch,
        rel_l2_max=rel_l2_max,
        rel_l2_mean=rel_l2_total / n_compared if n_compared else 0.0,
        max_abs_reference=max_abs_reference,
        tolerance=limit,
        storage_dtype=storage_dtype,
    )
