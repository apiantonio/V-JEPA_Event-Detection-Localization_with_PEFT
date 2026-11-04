"""``TinyBackbone``: the toy ViT where every bug shows up in seconds.

Two blocks, dim 32, tubelet 2×8×8: 512 tokens on 16 frames at 64 px and a millisecond
forward on CPU. Used by the smoke test and the PEFT unit tests, it reproduces the
pitfalls of the real model:

* fused QKV on purpose: ``attn.qkv`` is a single ``Linear(32, 96)``, as on V-JEPA 2.1
  (``Linear(768, 2304)``, layout ``[q|k|v]``);
* optional fake RoPE (1D rotation over the flattened token index), so that
  ``pos_policy="rope_identity"`` can be tested;
* no prefix tokens (no CLS, no registers), declared by ``token_layout()``;
* the forward registered in ``KV_PROMPT_ATTENTION`` at the end of the file is the CPU
  counterpart of the vendored V-JEPA forward used by APT.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from evd.models.adapt.apt import KV_PROMPT_ATTENTION
from evd.models.adapt.target_maps import TargetModuleMap
from evd.models.backbones.base import BACKBONES
from evd.types import BackboneOutput, PreprocessSpec, TokenLayout


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class FakeRope(nn.Module):
    """1D positional rotation over the token index (fake on purpose).

    It only makes the model position-sensitive. Like upstream ``rotate_queries_or_keys``
    (``vjepa2/app/vjepa_2_1/models/utils/modules.py:14``), the last ``n_registers``
    tokens are not rotated, which ``PromptPEFT`` relies on.
    """

    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, not {head_dim}")
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, x: Tensor, positions: Tensor, n_registers: int = 0) -> Tensor:
        """Rotate the context and leave the tail unchanged.

        :param x: ``[B, H, N, Dh]``.
        :param positions: ``[N - n_registers]``, positions of the context only.
        :param n_registers: number of trailing tokens left unrotated.
        :return: tensor with the shape of ``x``.
        """
        end = x.shape[-2] - n_registers
        ctx, reg = x[..., :end, :], x[..., end:, :]
        freqs = positions.float()[:, None] * self.inv_freq[None, :]  # [N_ctx, Dh/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [N_ctx, Dh]
        cos, sin = emb.cos()[None, None], emb.sin()[None, None]
        rotated = ctx * cos + _rotate_half(ctx) * sin
        return torch.cat((rotated, reg), dim=-2) if n_registers else rotated


class TinyAttention(nn.Module):
    """Attention with fused QKV: a single ``Linear(dim, 3*dim)``, as upstream.

    ``n_registers`` is a mutable attribute, as in upstream ``RoPEAttention``, which reads
    it at every forward; ``PromptPEFT`` increments it when installing prompts.
    """

    def __init__(self, dim: int, num_heads: int, rope: bool = False) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.n_registers = 0
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.rope = FakeRope(self.head_dim) if rope else None

    def forward(self, x: Tensor) -> Tensor:
        return tiny_attention_forward(self, x)


@KV_PROMPT_ATTENTION.register("TinyAttention")
def tiny_attention_forward(
    attn: TinyAttention,
    x: Tensor,
    *args: object,
    p_k: Tensor | None = None,
    p_v: Tensor | None = None,
    **kwargs: object,
) -> Tensor:
    """Run :class:`TinyAttention` with optional key and value prompts.

    CPU counterpart of ``evd/vendor/vjepa21_rope.py``: prompts are injected after the
    rotation and before SDPA, at the same point and with the same shape.

    :param attn: attention module to run.
    :param x: input sequence ``[B, N, D]``; the output has the same length.
    :param p_k: key prompts ``[n, head_dim]``; ``None`` disables them.
    :param p_v: value prompts, same shape.
    :return: ``[B, N, D]``.
    :raises ValueError: if only one of ``p_k`` and ``p_v`` is given.
    """
    if (p_k is None) != (p_v is None):
        raise ValueError("p_k and p_v go together: keys and values must stay aligned")
    b, n, d = x.shape
    qkv = attn.qkv(x).reshape(b, n, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    if attn.rope is not None:
        pos = torch.arange(n - attn.n_registers, device=x.device)
        q = attn.rope(q, pos, attn.n_registers)
        k = attn.rope(k, pos, attn.n_registers)
    if p_k is not None and p_v is not None:
        # After the rotation: a prompt has no position. `expand`, not `repeat`, as in the
        # official APT.
        k = torch.cat((k, p_k.to(k.dtype).expand(b, attn.num_heads, -1, -1)), dim=2)
        v = torch.cat((v, p_v.to(v.dtype).expand(b, attn.num_heads, -1, -1)), dim=2)
    # SDPA rather than eager attention: the project path, and the only one where bf16 does
    # not produce NaN on V-JEPA 2.1.
    out = F.scaled_dot_product_attention(q, k, v)
    return cast(Tensor, attn.proj(out.transpose(1, 2).reshape(b, n, d)))


class TinyMlp(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.fc2(F.gelu(self.fc1(x))))


class TinyBlock(nn.Module):
    """Pre-norm block, as in the reference implementations."""

    def __init__(self, dim: int, num_heads: int, rope: bool = False) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TinyAttention(dim, num_heads, rope=rope)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = TinyMlp(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        return cast(Tensor, x + self.mlp(self.norm2(x)))


class TinyViT(nn.Module):
    def __init__(
        self,
        dim: int = 32,
        depth: int = 2,
        num_heads: int = 4,
        tubelet: tuple[int, int, int] = (2, 8, 8),
        in_chans: int = 3,
        rope: bool = False,
    ) -> None:
        super().__init__()
        self.dim, self.depth, self.tubelet = dim, depth, tubelet
        self.patch_embed = nn.Conv3d(in_chans, dim, kernel_size=tubelet, stride=tubelet)
        self.blocks = nn.ModuleList([TinyBlock(dim, num_heads, rope=rope) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def tokenize(self, clips: Tensor) -> tuple[Tensor, tuple[int, int, int]]:
        if clips.ndim != 5:
            raise ValueError(f"expected clips [B,C,T,H,W], got {tuple(clips.shape)}")
        for size, ker, name in zip(clips.shape[2:], self.tubelet, "THW", strict=True):
            if size % ker != 0:
                raise ValueError(
                    f"dimension {name}={size} not divisible by the tubelet {ker}: "
                    f"the tokenizer would silently drop pixels"
                )
        x = self.patch_embed(clips)  # [B, D, T', H', W']
        grid = (x.shape[2], x.shape[3], x.shape[4])
        return x.flatten(2).transpose(1, 2), grid  # [B, N, D]

    def forward_layers(self, clips: Tensor) -> tuple[list[Tensor], tuple[int, int, int]]:
        """Return the output of every block (the last one normalized) and the grid."""
        x, grid = self.tokenize(clips)
        outs: list[Tensor] = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            outs.append(self.norm(x) if i == self.depth - 1 else x)
        return outs, grid


@BACKBONES.register("tiny")
class TinyBackbone:
    """Toy implementation of the whole ``VideoBackbone`` contract (docs/08 §3)."""

    def __init__(
        self,
        dim: int = 32,
        depth: int = 2,
        num_heads: int = 4,
        size: int = 64,
        num_frames: int = 16,
        rope: bool = False,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.backbone_id = "tiny"
        self.embed_dim = dim
        self.size, self.num_frames = size, num_frames
        self.rope_enabled = rope
        # Local seed: Tiny weights are identical in every test, whatever the global RNG state.
        gen_state = torch.get_rng_state()
        torch.manual_seed(seed)
        self.model = TinyViT(dim=dim, depth=depth, num_heads=num_heads, rope=rope).to(device)
        torch.set_rng_state(gen_state)
        self.model.eval()
        self.device = device

    # --- contract ----------------------------------------------------------
    def preprocess_spec(self) -> PreprocessSpec:
        from evd.data.synthetic import synthetic_preprocess_spec

        # The only exception to "models does not import data": the synthetic spec lives where
        # synthetic pixels are generated. Local import, so the dependency rule holds at import
        # time.
        return synthetic_preprocess_spec(size=self.size, num_frames=self.num_frames)

    def encode(self, clips: Tensor, layer: int = -1) -> BackboneOutput:
        return self.encode_layers(clips, (layer,))[layer]

    def encode_layers(self, clips: Tensor, layers: Sequence[int]) -> dict[int, BackboneOutput]:
        requested = tuple(int(layer) for layer in layers)
        if not requested:
            raise ValueError("encode_layers requires at least one layer")
        outs, grid = self.model.forward_layers(clips)
        resolved: list[int] = []
        for layer in requested:
            idx = layer if layer >= 0 else len(outs) + layer
            if not 0 <= idx < len(outs):
                raise IndexError(f"layer {layer} outside the {len(outs)} blocks of Tiny")
            resolved.append(idx)
        if len(set(resolved)) != len(resolved):
            raise ValueError(
                f"layer aliases in the same forward: requested {requested}, resolved {resolved}"
            )
        return {
            layer: BackboneOutput(
                tokens=outs[idx],
                grid=grid,
                pooled=outs[idx].mean(dim=1),
                layer=layer,
            )
            for layer, idx in zip(requested, resolved, strict=True)
        }

    def module(self) -> nn.Module:
        return self.model

    def blocks(self) -> nn.ModuleList:
        return self.model.blocks

    def target_map(self) -> TargetModuleMap:
        n = self.model.depth
        return TargetModuleMap(
            o=tuple(f"blocks.{i}.attn.proj" for i in range(n)),
            mlp=tuple(f"blocks.{i}.mlp.{p}" for i in range(n) for p in ("fc1", "fc2")),
            fused_qkv=tuple(f"blocks.{i}.attn.qkv" for i in range(n)),
            fused_out_dim=self.embed_dim,
            fused_order=("q", "k", "v"),
        )

    def token_layout(self) -> TokenLayout:
        return TokenLayout(
            n_cls=0,
            n_register=0,
            positional="rope_3d" if self.rope_enabled else "none",
        )

    def predictor(self) -> nn.Module | None:
        return None  # Tiny has no world model: a tokenizer with two blocks

    def hierarchical_layers(self) -> tuple[int, ...]:
        """Return the depth quarters, as the real model declares them (both blocks at depth 2)."""
        d = self.model.depth
        return tuple(sorted({max(0, int(math.ceil(d * q / 4)) - 1) for q in (1, 2, 3, 4)}))

    # --- convenience -------------------------------------------------------
    def parameters(self) -> Iterator[nn.Parameter]:
        return self.model.parameters()

    def to(self, device: str) -> TinyBackbone:
        self.model.to(device)
        self.device = device
        return self

    def eval(self) -> TinyBackbone:
        self.model.eval()
        return self

    def freeze(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(False)
