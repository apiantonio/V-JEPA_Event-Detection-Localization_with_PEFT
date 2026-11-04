"""The ``Head`` contract."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from torch import Tensor

from evd.models.registry import Registry
from evd.types import BackboneOutput


@runtime_checkable
class Head(Protocol):
    """Consume a ``BackboneOutput`` and produce logits.

    ``[B, C]`` with C classes (``Normal`` included) or ``[B, 1]`` for a binary head. The
    logit-to-anomaly-score conversion lives in the scorer, since it is a protocol
    convention (docs/03 §4.4).
    """

    n_classes: int

    def forward(self, feats: BackboneOutput) -> Tensor: ...


HEADS: Registry[Callable[..., Head]] = Registry("head")
