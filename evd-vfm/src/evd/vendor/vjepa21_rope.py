"""V-JEPA 2.1 ``RoPEAttention.forward``, copied to inject K/V prompts.

Provenance: ``vjepa2/app/vjepa_2_1/models/utils/modules.py``, ``RoPEAttention.forward``
(lines 199-305), upstream commit ``b921227e3e602220e8220001a17b3edff9edb474``. Copied
rather than wrapped because APT injects in the middle of the forward pass, after the
3D-RoPE rotation and before SDPA, where no hook or wrapper sees ``k`` and ``v``.

Changes with respect to the original:

1. ``self`` becomes the ``attn`` parameter: a free function applied to an existing
   instance, so the module that runs is the one that loaded the checkpoint.
2. After the rotation and before SDPA, ``p_k`` is concatenated to ``k`` and ``p_v`` to
   ``v``, as in ``wgcban/apt`` (``modeling_VPT.py``, ``Attention.forward``)::

       k = torch.cat((k, p_k.expand(B, self.num_heads, -1, -1)), dim=2)
       v = torch.cat((v, p_v.expand(B, self.num_heads, -1, -1)), dim=2)

3. Two checks the original does not need: ``is_causal`` with ``k`` longer than ``q``
   (SDPA's triangular mask would hide the prompts), and ``p_k``/``p_v`` of different
   lengths.

Only the forward is copied: ``rotate_queries_or_keys`` is resolved at runtime from the
instance's module (``_rotation_of``), so the 3D-RoPE is literally the upstream one.
Prompts enter after the RoPE, so they get identity rotation; the input sequence does not
grow, so ``N_ctx``, ``grid_depth``, ``pos`` and ``n_registers`` are untouched.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _rotation_of(attn: nn.Module) -> Callable[..., Tensor]:
    """Return the ``rotate_queries_or_keys`` of the module this attention comes from.

    Resolving it from the instance's module makes APT use the same function as the normal
    path, instead of a second 3D-RoPE implementation.

    :param attn: the attention instance.
    :return: the positional rotation function.
    :raises TypeError: if the module defining the attention does not expose it.
    """
    module = sys.modules.get(type(attn).__module__)
    rotation = getattr(module, "rotate_queries_or_keys", None)
    if rotation is None:
        raise TypeError(
            f"{type(attn).__name__} does not come from a module exposing "
            "`rotate_queries_or_keys`: this vendored forward replicates V-JEPA 2.1 "
            "RoPEAttention and does not apply to other attentions."
        )
    return rotation


def rope_attention_forward(
    attn: nn.Module,
    x: Tensor,
    mask: Tensor | None = None,
    T: int | None = None,
    H_patches: int | None = None,
    W_patches: int | None = None,
    return_attn: bool = False,
    *,
    p_k: Tensor | None = None,
    p_v: Tensor | None = None,
) -> tuple[Tensor, Any]:
    """Run ``RoPEAttention`` forward with optional prompts on keys and values.

    :param attn: the ``RoPEAttention`` instance to run; not modified.
    :param x: input sequence, ``[B, N, C]``.
    :param mask: token position indices, as upstream; ``None`` derives them from the
        geometry.
    :param T: tubelet frames, to derive positions.
    :param H_patches: patches along the height.
    :param W_patches: patches along the width.
    :param return_attn: whether to also return the attention matrix (eager path only).
    :param p_k: prompts concatenated to the keys, ``[n, head_dim]`` or
        ``[B, num_heads, n, head_dim]``; ``None`` disables the injection.
    :param p_v: prompts concatenated to the values, same shape as ``p_k``.
    :return: the pair ``(output [B, N, C], attention | None)``, as upstream.
    :raises ValueError: if only one of ``p_k`` and ``p_v`` is given, if their lengths
        differ, or if the attention is causal.
    """
    if (p_k is None) != (p_v is None):
        raise ValueError(
            "p_k and p_v go together: SDPA needs keys and values of the same length"
        )
    if p_k is not None and p_v is not None:
        if p_k.shape[-2] != p_v.shape[-2]:
            raise ValueError(
                f"K and V prompts of different length: {p_k.shape[-2]} vs {p_v.shape[-2]}"
            )
        if getattr(attn, "is_causal", False):
            raise ValueError(
                "APT on a causal attention is undefined: with k longer than q, SDPA's "
                "triangular mask aligns bottom-right and masks exactly "
                "the prompts. The V-JEPA 2.1 encoder is not causal."
            )

    rotate_queries_or_keys = _rotation_of(attn)
    B, N, C = x.size()
    N_ctx = N - attn.n_registers
    grid_depth = int(N_ctx // (attn.grid_size * attn.grid_size))

    qkv = attn.qkv(x).unflatten(-1, (3, attn.num_heads, -1)).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    if mask is not None:
        mask = mask.unsqueeze(1).repeat(1, attn.num_heads, 1)
        d_mask, h_mask, w_mask = attn.separate_positions(mask, H_patches, W_patches)
    else:
        if T is None or H_patches is None or W_patches is None:
            mask = torch.arange(int(grid_depth * attn.grid_size * attn.grid_size), device=x.device)
        else:
            mask = torch.arange(int(T * H_patches * W_patches), device=x.device)
        d_mask, h_mask, w_mask = attn.separate_positions(mask, H_patches, W_patches)

    if attn.interpolate_rope:
        if H_patches is None:
            H_patches = int(attn.grid_size)
        if W_patches is None:
            W_patches = int(attn.grid_size)
        h_mask = h_mask * (attn.pretrained_grid_size - 1) / (H_patches - 1)
        w_mask = w_mask * (attn.pretrained_grid_size - 1) / (W_patches - 1)

    s = 0
    qd = rotate_queries_or_keys(
        q[..., s : s + attn.d_dim],
        pos=d_mask,
        n_registers=attn.n_registers,
        has_cls_first=attn.has_cls_first,
    )
    kd = rotate_queries_or_keys(
        k[..., s : s + attn.d_dim],
        pos=d_mask,
        n_registers=attn.n_registers,
        has_cls_first=attn.has_cls_first,
    )
    s += attn.d_dim
    qh = rotate_queries_or_keys(
        q[..., s : s + attn.h_dim],
        pos=h_mask,
        n_registers=attn.n_registers,
        has_cls_first=attn.has_cls_first,
    )
    kh = rotate_queries_or_keys(
        k[..., s : s + attn.h_dim],
        pos=h_mask,
        n_registers=attn.n_registers,
        has_cls_first=attn.has_cls_first,
    )
    s += attn.h_dim
    qw = rotate_queries_or_keys(
        q[..., s : s + attn.w_dim],
        pos=w_mask,
        n_registers=attn.n_registers,
        has_cls_first=attn.has_cls_first,
    )
    kw = rotate_queries_or_keys(
        k[..., s : s + attn.w_dim],
        pos=w_mask,
        n_registers=attn.n_registers,
        has_cls_first=attn.has_cls_first,
    )
    s += attn.w_dim

    if s < attn.head_dim:
        qr = q[..., s:]
        kr = k[..., s:]
        q = torch.cat([qd, qh, qw, qr], dim=-1)
        k = torch.cat([kd, kh, kw, kr], dim=-1)
    else:
        q = torch.cat([qd, qh, qw], dim=-1)
        k = torch.cat([kd, kh, kw], dim=-1)

    # --- APT: the only addition to the upstream forward ---------------------- #
    # Prompts enter after the rotation (they have no position). `expand`, not `repeat`: no
    # copy, and the gradient accumulates on the single bank shared across batch and heads.
    if p_k is not None and p_v is not None:
        k = torch.cat((k, _expand(p_k, B, attn.num_heads, k)), dim=2)
        v = torch.cat((v, _expand(p_v, B, attn.num_heads, v)), dim=2)
    # ------------------------------------------------------------------------- #

    if attn.use_sdpa:
        with torch.backends.cuda.sdp_kernel():
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=attn.proj_drop_prob, is_causal=attn.is_causal
            )
            attn_weights = None
    else:
        attn_weights = (q @ k.transpose(-2, -1)) * attn.scale
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = attn.attn_drop(attn_weights)
        x = attn_weights @ v
    x = x.transpose(1, 2).reshape(B, N, C)
    x = attn.proj(x)
    x = attn.proj_drop(x)
    if return_attn:
        return x, attn_weights
    return x, None


def _expand(prompts: Tensor, batch: int, num_heads: int, reference: Tensor) -> Tensor:
    """Bring the prompts to the shape of ``k``/``v``, without copying them.

    :param prompts: ``[n, head_dim]`` (shared across heads, as in official APT) or already
        ``[B, num_heads, n, head_dim]``.
    :param batch: batch size.
    :param num_heads: attention heads.
    :param reference: the tensor they will be concatenated to; used only for dtype and
        device, which under autocast differ from the parameter's.
    :return: a ``[B, num_heads, n, head_dim]`` view.
    :raises ValueError: if the number of dimensions is neither 2 nor 4.
    """
    if prompts.dim() == 4:
        return prompts.to(dtype=reference.dtype)
    if prompts.dim() != 2:
        raise ValueError(
            f"K/V prompts of shape {tuple(prompts.shape)}: expected [n, head_dim] or "
            "[B, num_heads, n, head_dim]"
        )
    return prompts.to(dtype=reference.dtype).expand(batch, num_heads, -1, -1)
