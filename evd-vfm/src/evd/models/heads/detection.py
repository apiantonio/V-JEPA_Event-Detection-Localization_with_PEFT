"""Detection head used by E2/E3 and the demonstrator.

Binary output: a single logit (the model detects events, it does
not predict their class). Per-class metrics break binary errors down by the chunk
class, which lives in the data (:class:`~evd.protocol.chunks.ChunkItem`). ``n_classes``
remains because it sets the output size; the default is 1, and the multi-class head
stays available for numerical parity with the official probes.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from evd.models.heads.base import HEADS
from evd.models.heads.pooling import AttentivePooling
from evd.types import BackboneOutput


@HEADS.register("detection")
class DetectionHead(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        n_classes: int = 1,
        n_queries: int = 1,
        n_heads: int = 16,
        depth: int = 4,
        mlp_ratio: float = 4.0,
        norm: bool = True,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.norm = nn.LayerNorm(embed_dim) if norm else nn.Identity()
        self.pooler = AttentivePooling(
            num_queries=n_queries,
            embed_dim=embed_dim,
            num_heads=n_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
        )
        self.linear = nn.Linear(embed_dim, n_classes)

    def forward(self, feats: BackboneOutput) -> Tensor:
        pooled = self.pooler(self.norm(feats.tokens))
        if pooled.shape[1] != 1:
            raise ValueError(
                f"DetectionHead requires n_queries=1 to produce [B,C], got {pooled.shape[1]}"
            )
        return self.linear(pooled.squeeze(1))
