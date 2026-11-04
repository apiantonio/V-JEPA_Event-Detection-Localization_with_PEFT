"""Attentive probe for the frozen baselines."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from evd.models.heads.base import HEADS
from evd.models.heads.pooling import AttentivePooling
from evd.types import BackboneOutput


@HEADS.register("attentive")
class AttentiveProbe(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        n_classes: int,
        n_queries: int = 1,
        n_heads: int = 16,
        depth: int = 4,
        mlp_ratio: float = 4.0,
        norm: bool = True,
        use_activation_checkpointing: bool = False,
    ) -> None:
        """:param use_activation_checkpointing: recompute block activations instead of storing
        them. Logits are unchanged (no dropout), only memory: used by the multilevel arm
        of ADR-029. The default ``False`` lives in the signature, not in the YAML, so the
        ``_config_sha`` of single-layer cells does not change.
        """
        super().__init__()
        self.n_classes = n_classes
        self.norm = nn.LayerNorm(embed_dim) if norm else nn.Identity()
        self.pooler = AttentivePooling(
            num_queries=n_queries,
            embed_dim=embed_dim,
            num_heads=n_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            use_activation_checkpointing=use_activation_checkpointing,
        )
        self.linear = nn.Linear(embed_dim, n_classes)

    def forward(self, feats: BackboneOutput) -> Tensor:
        pooled = self.pooler(self.norm(feats.tokens))
        if pooled.shape[1] != 1:
            raise ValueError(
                f"AttentiveProbe requires n_queries=1 to produce [B,C], got {pooled.shape[1]}"
            )
        return self.linear(pooled.squeeze(1))
