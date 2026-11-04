"""Attention map of the pooler: where the model looks to decide.

The head (:class:`~evd.models.heads.detection.DetectionHead`) is an
:class:`~evd.models.heads.pooling.AttentivePooling` with ``depth=4``: three
self-attention blocks over the backbone tokens and a cross-attention block with a
single learned query. That cross-attention gives one weight per token
(``T/2 × H/16 × W/16 = 16 × 24 × 24 = 9216`` on ``edp_v2``), so the spatio-temporal map
is read from the model, not invented.

``_CrossAttention.forward`` uses ``scaled_dot_product_attention``, which does not
return the weights, and ``pooling.py`` replicates the Meta source: a
``forward_hook`` receives inputs and output, ``q`` and ``k`` are recomputed with the
module's weights, and ``attn @ v`` is compared with the actual output as a fidelity
check.

What the map does not say: three self-attention blocks precede the cross-attention, so
a high weight is a position the query reads from, not a pixel that caused the decision
alone (no attention rollout: 9216 × 9216 matrices per head).

The map is almost flat, a result rather than a fault: on ``Arson016_x264`` with
``dora-r32`` weights lie between 0.60x and 1.70x the uniform value ``1/9216`` and the
normalized softmax entropy is 0.997-0.9999. Head averaging, ignoring the norm of ``v``
and extraction noise were ruled out by measurement. The informative part is the
deviation from the video mean map, which motivates :data:`CONTRAST_MODES`.

Prompts do not reach the pooler (``PromptedBlock`` strips them after the last block);
the guard records the attention mass outside the grid if they ever did.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

__all__ = [
    "CONTRAST_MODES",
    "FIDELITY_TOLERANCE",
    "Contrast",
    "AttentionMap",
    "MapStatistics",
    "capture_attention",
    "contrast_of",
    "cross_attention_module",
    "reduce_over_frames",
    "map_statistics",
]

#: Maximum gap between the recomputation and the actual module output, per arithmetic
#: dtype: measured 1.0e-03 in bf16 and 6e-07 in fp32 on ViT-L, while a wrong
#: recomputation gives errors of order one. The key is the output dtype (see
#: :class:`AttentionMap`).
FIDELITY_TOLERANCE: dict[torch.dtype, float] = {
    torch.float32: 1e-4,
    torch.float64: 1e-6,
    torch.bfloat16: 5e-2,
    torch.float16: 5e-2,
}


@dataclass(frozen=True)
class AttentionMap:
    """Attention weights of the pooler query on one window.

    :param pesi: ``[T', H', W']`` float32, unnormalized softmax weights over all tokens
        (they sum to 1 over the window).
    :param grid: backbone grid ``(T', H', W')`` of this window.
    :param massa_prompt: fraction of weight on tokens outside the grid (0.0 when prompts
        are stripped).
    :param scarto_fedelta: relative distance between the recomputed ``attn @ v`` and the
        actual module output; compare with :data:`FIDELITY_TOLERANCE`.
    :param dtype: arithmetic dtype, i.e. the module output dtype. Under ``autocast`` the
        LayerNorm before the cross-attention returns fp32 even in bf16 regime, so the
        input dtype would trigger the fidelity check on a normal bf16 error.
    :param n_teste: number of averaged heads.
    """

    pesi: np.ndarray
    grid: tuple[int, int, int]
    massa_prompt: float
    scarto_fedelta: float
    dtype: str
    n_teste: int

    @property
    def fedele(self) -> bool:
        """:return: whether the gap is within the tolerance of its dtype."""
        limit = FIDELITY_TOLERANCE.get(getattr(torch, self.dtype, torch.float32), 1e-4)
        return self.scarto_fedelta <= limit


def cross_attention_module(testa: nn.Module) -> nn.Module:
    """Find the cross-attention module of a head's pooler.

    With ``complete_block=True`` (default, used by E2/E3) the cross-attention sits inside a
    block with norm and MLP; with ``complete_block=False`` the block is the
    cross-attention.

    :param testa: run head (``DetectionHead``, ``AttentiveProbe``...).
    :return: the module exposing ``q``, ``kv`` and ``num_heads``.
    :raises TypeError: if the head has no attentive pooler (e.g. a linear probe).
    """
    pooler = getattr(testa, "pooler", None)
    block = getattr(pooler, "cross_attention_block", None)
    if block is None:
        raise TypeError(
            f"{type(testa).__name__} has no attentive pooler: without cross-attention there "
            f"is no map to read. This is the case of the LINEAR probe, by design "
            f"(it averages the tokens and looks at none in particular)"
        )
    xattn = getattr(block, "xattn", block)
    for expected in ("q", "kv", "num_heads"):
        if not hasattr(xattn, expected):
            raise TypeError(
                f"the cross-attention block of {type(testa).__name__} does not expose {expected!r}: "
                f"the pooler architecture changed and this module is reading it from memory"
            )
    return xattn


def _weights_of(xattn: nn.Module, query: Tensor, tokens: Tensor) -> tuple[Tensor, Tensor]:
    """Recompute attention weights and output with the module's actual weights.

    Replicates ``_CrossAttention.forward`` up to ``attn @ v``; the module ``scale`` equals
    the ``scaled_dot_product_attention`` default (``1/sqrt(head_dim)``).

    :param xattn: cross-attention module.
    :param query: queries ``[B, Nq, D]``, as received by the module.
    :param tokens: backbone tokens, already normalized, ``[B, N, D]``.
    :return: ``(attn [B, heads, Nq, N], output [B, Nq, D])``.
    """
    heads = int(xattn.num_heads)
    b, nq, dim = query.shape
    n = tokens.shape[1]
    head_dim = dim // heads
    q = xattn.q(query).reshape(b, nq, heads, head_dim).permute(0, 2, 1, 3)
    kv = xattn.kv(tokens).reshape(b, n, 2, heads, head_dim).permute(2, 0, 3, 1, 4)
    k, v = kv.unbind(0)
    attn = torch.softmax(q @ k.transpose(-2, -1) * (head_dim**-0.5), dim=-1)
    output = (attn @ v).transpose(1, 2).reshape(b, nq, dim)
    return attn, output


@contextmanager
def capture_attention(testa: nn.Module) -> Iterator[list[AttentionMap]]:
    """Collect one map for every window passing through the head.

    Usage::

        with capture_attention(head) as maps:
            scores = list(engine.stream(video))
        assert len(maps) == len(scores)

    The pairing with scores is checked by counting in the caller. The grid is read from
    the ``BackboneOutput`` received by the head (a second hook), not derived from
    ``num_frames`` and patch sizes.

    :param testa: run head, built and with weights loaded.
    :yield: the list filled during the ``with`` block, one entry per window.
    :raises TypeError: if the head has no attentive pooler.
    :raises ValueError: if the input tokens are fewer than the grid, there is more than
        one query, or the head does not receive a ``BackboneOutput`` with its grid.
    """
    xattn = cross_attention_module(testa)
    collections: list[AttentionMap] = []
    geometry: dict[str, tuple[int, int, int] | None] = {"grid": None}

    def read_grid(_module: nn.Module, args: tuple[Any, ...]) -> None:
        grid = getattr(args[0], "grid", None) if args else None
        if grid is None or len(tuple(grid)) != 3:
            raise ValueError(
                f"the head did not receive a BackboneOutput with its grid (grid={grid!r}): "
                f"without it the attention weights are a vector, not a spatio-temporal map"
            )
        geometry["grid"] = (int(grid[0]), int(grid[1]), int(grid[2]))

    def hook(_module: nn.Module, args: tuple[Any, ...], uscita: Tensor) -> None:
        """Single forward hook: it receives inputs and output together, so the fidelity check
        compares the recomputation with the output of this very call.
        """
        grid = geometry["grid"]
        if grid is None:
            raise ValueError(
                "the cross-attention was called before the head declared its "
                "grid: the pooler is running outside the head forward"
            )
        expected = int(np.prod(grid))
        query, tokens = args[0], args[1]
        if query.shape[1] != 1:
            raise ValueError(
                f"the pooler has {query.shape[1]} queries: there is no single attention map, "
                f"but one per query. The E2/E3 head has a single query"
            )
        if tokens.shape[1] < expected:
            raise ValueError(
                f"the pooler receives {tokens.shape[1]} tokens but the grid {grid} expects "
                f"{expected}: the map cannot be laid out in the video space-time"
            )
        with torch.no_grad():
            attn, redone = _weights_of(xattn, query, tokens)
            denominator = float(uscita.float().norm()) or 1.0
            gap = float((redone - uscita).float().norm() / denominator)
            # Mean over heads: with a single query each head gives one `[N]` row and the map shows
            # their consensus.
            mean_values = attn.mean(dim=1)[:, 0, :].float().cpu()
        for row in mean_values:
            collections.append(
                AttentionMap(
                    pesi=row[:expected].reshape(grid).numpy().astype(np.float32),
                    grid=grid,
                    massa_prompt=float(row[expected:].sum()),
                    scarto_fedelta=gap,
                    dtype=str(uscita.dtype).replace("torch.", ""),
                    n_teste=int(xattn.num_heads),
                )
            )

    handles = [
        testa.register_forward_pre_hook(read_grid),
        xattn.register_forward_hook(hook),
    ]
    try:
        yield collections
    finally:
        for handle in handles:
            handle.remove()


def reduce_over_frames(map_data: AttentionMap, position: int) -> np.ndarray:
    """Return the spatial slice of the map at temporal position ``posizione``.

    :param mappa: map of the window.
    :param posizione: index on the temporal grid ``T'``, clamped to the ends.
    :return: ``[H', W']`` float32.
    """
    t = int(np.clip(position, 0, map_data.grid[0] - 1))
    return map_data.pesi[t]


#: Ways a raw map becomes an image in ``[0, 1]``; needed because the pooler map is
#: almost uniform (entropy 0.997-0.9999).
#:
#: * ``massimo``: divide by the video maximum (zero weight is black).
#: * ``percentile``: linear stretch between two video percentiles (default 1 and 99);
#:   the colour bar reports the extremes in multiples of the uniform weight.
#: * ``scarto``: subtract the video mean map and normalize on a percentile of the
#:   absolute difference, with a diverging colormap: the content-dependent part.
#: * ``finestra``: each window on its own maximum; for inspecting one window, never
#:   for comparing two.
CONTRAST_MODES = ("massimo", "percentile", "scarto", "finestra")


@dataclass(frozen=True)
class Contrast:
    """Transformation bringing a raw map into ``[0, 1]`` for drawing.

    The transformation and its provenance travel together, so colour bars can report the
    extremes in ``x uniform`` and the image can be redrawn without rerunning the forward.

    :param modo: one of :data:`CONTRAST_MODES`.
    :param lo: raw value mapped to 0; ``-hi`` for ``scarto``.
    :param hi: raw value mapped to 1.
    :param uniforme: ``1/N``, the weight of every token under a flat softmax.
    :param base: video mean map ``[H', W']`` for ``scarto``, ``None`` otherwise.
    :param divergente: whether to use a two-sided colormap.
    """

    modo: str
    lo: float
    hi: float
    uniforme: float
    base: np.ndarray | None = None
    divergente: bool = False

    def apply_to(self, slice_: np.ndarray) -> np.ndarray:
        """Bring a spatial slice into ``[0, 1]``.

        :param slice_: ``[H', W']`` raw weights.
        :return: ``[H', W']`` float32 in ``[0, 1]``, clamped.
        :raises ValueError: if the mode is ``scarto`` without a base.
        """
        x = slice_.astype(np.float32)
        if self.modo == "finestra":
            maximum = float(x.max())
            return (x / maximum if maximum > 0 else x).astype(np.float32)
        if self.modo == "scarto":
            if self.base is None:
                raise ValueError("the 'scarto' mode requires the video mean map")
            width = self.hi if self.hi > 0 else 1.0
            return np.clip((x - self.base) / (2.0 * width) + 0.5, 0.0, 1.0).astype(np.float32)
        width_px = self.hi - self.lo
        if width_px <= 0:
            return np.zeros_like(x, dtype=np.float32)
        return np.clip((x - self.lo) / width_px, 0.0, 1.0).astype(np.float32)

    def estremi_in_uniforme(self) -> tuple[float, float]:
        """:return: the scale extremes in multiples of the uniform weight: ``(-hi, +hi)`` for
        ``scarto``, absolute ``lo`` and ``hi`` otherwise.
        """
        u = self.uniforme or 1.0
        if self.modo == "scarto":
            return (-self.hi / u, self.hi / u)
        return (self.lo / u, self.hi / u)


@dataclass(frozen=True)
class MapStatistics:
    """How (little) selective the attention of a video is, in numbers.

    Recorded next to the image: a contrast stretch makes even a flat map readable, and
    these fields keep the two cases distinguishable.

    :param uniforme: ``1/N``.
    :param minimo_su_uniforme: lowest weight of the video, in uniform units.
    :param massimo_su_uniforme: highest weight.
    :param entropia_minima: normalized entropy of the most concentrated window (1.0 = flat
        softmax).
    :param entropia_mediana: median over windows.
    :param massa_top1: mass share of the top 1% of tokens, median over windows (0.01 for a
        uniform map).
    :param scarto_su_deviazione: ratio between the absolute maximum of the ``scarto`` map
        and its standard deviation (structure versus noise).
    """

    uniforme: float
    minimo_su_uniforme: float
    massimo_su_uniforme: float
    entropia_minima: float
    entropia_mediana: float
    massa_top1: float
    scarto_su_deviazione: float


def map_statistics(mappe: Sequence[AttentionMap]) -> MapStatistics:
    """Measure how concentrated the attention of a video is.

    :param mappe: maps of the video, at least one.
    :return: statistics to record next to the image.
    :raises ValueError: if there is no map.
    """
    if not mappe:
        raise ValueError("no map: nothing to measure")
    stacks = np.stack([m.pesi.reshape(-1) for m in mappe]).astype(np.float64)
    n = stacks.shape[1]
    u = 1.0 / n
    entropies = -(stacks * np.log(stacks + 1e-30)).sum(axis=1) / np.log(n)
    how_many = max(1, int(0.01 * n))
    top1 = np.sort(stacks, axis=1)[:, ::-1][:, :how_many].sum(axis=1) / stacks.sum(axis=1)
    discards = stacks - stacks.mean(axis=0, keepdims=True)
    deviation = float(discards.std()) or 1.0
    return MapStatistics(
        uniforme=u,
        minimo_su_uniforme=float(stacks.min()) / u,
        massimo_su_uniforme=float(stacks.max()) / u,
        entropia_minima=float(entropies.min()),
        entropia_mediana=float(np.median(entropies)),
        massa_top1=float(np.median(top1)),
        scarto_su_deviazione=float(np.abs(discards).max()) / deviation,
    )


def contrast_of(
    mappe: Sequence[AttentionMap],
    *,
    modo: str = "percentile",
    percentili: tuple[float, float] = (1.0, 99.0),
) -> Contrast:
    """Build the contrast transformation for a video.

    :param mappe: maps of the video, at least one, all on the same grid.
    :param modo: one of :data:`CONTRAST_MODES`.
    :param percentili: extremes of the ``percentile`` mode and tail dropped by ``scarto``;
        not 0 and 100, since a single extreme token would decide the scale.
    :return: the contrast with its provenance.
    :raises ValueError: on an unknown mode, no maps, or non-increasing percentiles within
        ``[0, 100]``.
    """
    if modo not in CONTRAST_MODES:
        raise ValueError(f"unknown contrast mode {modo!r}: expected {CONTRAST_MODES}")
    if not mappe:
        raise ValueError("no map: nothing to normalize")
    p_lo, p_hi = percentili
    if not 0.0 <= p_lo < p_hi <= 100.0:
        raise ValueError(f"percentiles {percentili} not increasing within [0, 100]")

    stacks = np.stack([m.pesi for m in mappe]).astype(np.float32)
    u = 1.0 / float(np.prod(mappe[0].grid))

    if modo in ("massimo", "finestra"):
        return Contrast(modo=modo, lo=0.0, hi=float(stacks.max()), uniforme=u)
    if modo == "percentile":
        lo, hi = (float(v) for v in np.percentile(stacks, [p_lo, p_hi]))
        return Contrast(modo=modo, lo=lo, hi=hi, uniforme=u)
    # `scarto`: the base is the video mean over windows and planes; the temporal profile
    # within a window is flat, so a spatial `[H', W']` base suffices.
    base = stacks.mean(axis=(0, 1))
    width = float(np.percentile(np.abs(stacks - base), p_hi))
    return Contrast(modo=modo, lo=-width, hi=width, uniforme=u, base=base, divergente=True)
