"""Attentive pooling faithful to the Meta reference.

Replicates ``vjepa2/src/models/attentive_pooler.py::AttentivePooler`` without importing
upstream at runtime; the original is the numerical oracle in tests. Submodule names and
operation order are unchanged, so the same state dict loads without conversion.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(0.0)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"embed_dim={dim} not divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = 0.0
        self.proj_drop = nn.Dropout(0.0)
        self.use_sdpa = True
        self.is_causal = False

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        del mask
        batch, tokens, dim = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch, tokens, 3, self.num_heads, dim // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.proj_drop_prob,
            is_causal=self.is_causal,
            attn_mask=attn_mask,
        )
        out = out.transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj_drop(self.proj(out))


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, qkv_bias: bool) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads, qkv_bias)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, int(dim * mlp_ratio))

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), mask=mask, attn_mask=attn_mask))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class _CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"embed_dim={dim} not divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.use_sdpa = True

    def forward(self, q: Tensor, x: Tensor) -> Tensor:
        batch, n_queries, dim = q.shape
        q = (
            self.q(q)
            .reshape(batch, n_queries, self.num_heads, dim // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        batch, n_tokens, dim = x.shape
        kv = (
            self.kv(x)
            .reshape(batch, n_tokens, 2, self.num_heads, dim // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)
        q = F.scaled_dot_product_attention(q, k, v)
        return q.transpose(1, 2).reshape(batch, n_queries, dim)


class _CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, qkv_bias: bool) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.xattn = _CrossAttention(dim, num_heads, qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, int(dim * mlp_ratio))

    def forward(self, q: Tensor, x: Tensor) -> Tensor:
        q = q + self.xattn(q, self.norm1(x))
        return q + self.mlp(self.norm2(q))


class AttentivePooling(nn.Module):
    """Learned queries in cross-attention over the backbone tokens."""

    def __init__(
        self,
        *,
        num_queries: int = 1,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        depth: int = 1,
        init_std: float = 0.02,
        qkv_bias: bool = True,
        complete_block: bool = True,
        use_activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if num_queries <= 0 or depth <= 0:
            raise ValueError(f"num_queries and depth must be > 0: {num_queries}, {depth}")
        self.use_activation_checkpointing = use_activation_checkpointing
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        self.complete_block = complete_block
        self.cross_attention_block: nn.Module
        if complete_block:
            self.cross_attention_block = _CrossAttentionBlock(
                embed_dim, num_heads, mlp_ratio, qkv_bias
            )
        else:
            self.cross_attention_block = _CrossAttention(embed_dim, num_heads, qkv_bias)
        self.blocks = (
            nn.ModuleList(
                [_Block(embed_dim, num_heads, mlp_ratio, qkv_bias) for _ in range(depth - 1)]
            )
            if depth > 1
            else None
        )
        self.init_std = init_std
        nn.init.trunc_normal_(self.query_tokens, std=init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _rescale_blocks(self) -> None:
        def rescale(parameter: nn.Parameter, layer_id: int) -> None:
            parameter.data.div_(math.sqrt(2.0 * layer_id))

        layer_id = 0
        if self.blocks is not None:
            for layer_id, layer in enumerate(self.blocks):
                rescale(layer.attn.proj.weight, layer_id + 1)
                rescale(layer.mlp.fc2.weight, layer_id + 1)
        if self.complete_block:
            rescale(self.cross_attention_block.mlp.fc2.weight, layer_id + 1)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected tokens [B,N,D], got {tuple(x.shape)}")
        if self.blocks is not None:
            for block in self.blocks:
                if self.use_activation_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)
        queries = self.query_tokens.repeat(len(x), 1, 1)
        return self.cross_attention_block(queries, x)
