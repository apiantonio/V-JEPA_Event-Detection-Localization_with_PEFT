"""``LinearProbe``: mean pooling plus one ``Linear``.

A collapse detector, not the main head: it tells whether the domain semantics
are linearly accessible from frozen tokens.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from evd.models.heads.base import HEADS
from evd.types import BackboneOutput


@HEADS.register("linear")
class LinearProbe(nn.Module):
    def __init__(self, embed_dim: int, n_classes: int, norm: bool = True) -> None:
        super().__init__()
        self.n_classes = n_classes
        # LayerNorm before the Linear: intermediate layers have different token scales, which
        # would make the learning rate layer-dependent.
        self.norm = nn.LayerNorm(embed_dim) if norm else nn.Identity()
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, feats: BackboneOutput) -> Tensor:
        """:raises ValueError: if the tokens are multilevel (ADR-029): averaging concatenated
        levels mixes different spaces.
        """
        if feats.levels:
            raise ValueError(
                f"LinearProbe reads a single level, got {list(feats.levels)}: mean "
                f"pooling over concatenated levels mixes different spaces. Use the "
                f"`attentive` head for the multilevel arm (ADR-029)"
            )
        x = feats.tokens.mean(dim=1)  # [B, D], mean pooling over all tokens
        return self.fc(self.norm(x))
