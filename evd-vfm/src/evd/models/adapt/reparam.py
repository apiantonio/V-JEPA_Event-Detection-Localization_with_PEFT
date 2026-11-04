r"""Reparametrization PEFT: LoRA, rsLoRA and DoRA.

Pure PyTorch; ``peft`` is not a dependency: on a fused QKV it applies a single rank-*r*
adapter to the whole matrix, which makes ``qk_only`` inexpressible. It remains the
semantic reference for tests.

Official references replicated here:

* ``microsoft/LoRA``, ``loralib/layers.py``: ``A`` from ``kaiming_uniform_(a=√5)``,
  ``B`` zero, ``scaling = alpha/r``, frozen base weight; ``MergedLinear`` gives an
  independent adapter to each enabled projection of a fused QKV.
  ``FusedQKVLoRALinear`` rewrites that semantics with explicit slices, also allowing
  slices of different widths.
* ``NVlabs/DoRA`` (``commonsense_reasoning/peft/src/peft/tuners/dora.py``): one
  magnitude per output feature (``torch.linalg.norm(weight, dim=1)``); the forward
  computes ``norm_scale = m / ‖V‖`` first and the merge writes ``(m/‖V‖)·V``.
* rsLoRA (Kalajdzievski 2023): ``scaling = alpha/√r``, combinable with DoRA.

Factorization lattice: on a fused QKV the same ``ΔW ∈ R^{3d×d}`` can be factorized in
ways with different cost and constraints (``ReparamSpec.factorization``)::

                          fused_nr                      12·r·d
                             |
                          separate                       6·r·d
                        /    |    \
                   fused  shared_b  (qk_only)           4·r·d each
                 (one A)  (one B)   (B_v = 0)

The bottom three cost the same and are incomparable: ``fused`` forces Q, K and V to
read the same input subspace, ``shared_b`` to write the same output directions, and
``qk_only`` (a placement, not a factorization) leaves V untouched. Equal budget
therefore does not identify a configuration (Q14).

Two explicit decisions: the scaling always follows the declared rank
(``s = alpha/r_eff`` with ``r_eff = n·r`` for ``fused_nr``, i.e. three times smaller
than ``separate`` at equal ``alpha``); on a single-slice target all factorizations
degenerate to the same block.

DoRA details measured on the four V-JEPA 2.1 variants
(``../lab/notes/03_target_modules.md``): ``V·(m/‖V‖)`` is bitwise the frozen model at
t=0 in float32 while ``(m·V)/‖V‖`` drifts up to 6.7e-04; the output-axis norm also
fixes the magnitude budget (36,864 magnitudes on ViT-B ``qkvo``).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from evd.models.adapt.base import (
    ADAPTERS,
    AdaptationReport,
    AdaptedBackbone,
    Family,
)
from evd.models.adapt.target_maps import OutSlice, Placement, TargetRef

if TYPE_CHECKING:
    from evd.models.backbones.base import VideoBackbone

ReparamMethod = Literal["lora", "dora"]

#: How ``ΔW`` is factorized when a target is requested on several slices (fused QKV).
Factorization = Literal["separate", "fused", "fused_nr", "shared_b"]

#: Allowed factorizations.
FACTORIZATIONS: tuple[Factorization, ...] = ("separate", "fused", "fused_nr", "shared_b")

#: Factorizations requiring slices of equal width: ``shared_b`` shares a ``B`` of shape
#: ``(width, r)``.
EQUAL_SLICE_FACTORIZATIONS: frozenset[str] = frozenset({"shared_b"})


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReparamSpec:
    """Configuration of a reparametrization adaptation.

    :param method: ``"lora"`` (low-rank additive update) or ``"dora"``
        (magnitude-direction decomposition on top of the same update).
    :param r: update rank. ``0`` disables the low-rank part: a full no-op for LoRA, only
        the magnitude rescaling for DoRA (the official ``Wdecompose`` mode).
    :param alpha: scaling numerator.
    :param placement: projections to adapt, in
        :class:`~evd.models.adapt.target_maps.Placement` syntax (``"qkvo"``,
        ``"attn_mlp:4"``, ``"qk_only"``, ``"last_n_blocks:6"``); on a fused QKV the same
        ``Linear`` is adapted once per requested slice.
    :param factorization: how ``ΔW`` is factorized over the requested slices of the same
        module: ``"separate"`` (one ``(A, B)`` per slice), ``"fused"`` (one rank-``r``
        adapter), ``"fused_nr"`` (one rank-``n·r`` adapter), ``"shared_b"`` (one ``B``,
        one ``A`` per slice).
    :param use_rslora: use ``alpha/√r`` instead of ``alpha/r``.
    :param dropout: dropout on the input of the adapted branch only.
    :param init: initialization scheme; only ``A`` kaiming uniform with ``B`` zero is
        allowed, so the output at t=0 equals the base model.
    :param seed: seed for drawing ``A`` (``kaiming_uniform_`` would draw from the global
        RNG).
    """

    method: ReparamMethod = "lora"
    r: int = 16
    alpha: float = 32.0
    placement: str = "qkvo"
    factorization: Factorization = "separate"
    use_rslora: bool = False
    dropout: float = 0.0
    init: Literal["kaiming_b_zero"] = "kaiming_b_zero"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.method not in ("lora", "dora"):
            raise ValueError(f"method={self.method!r}: ammessi 'lora' | 'dora'")
        if self.factorization not in FACTORIZATIONS:
            raise ValueError(
                f"factorization={self.factorization!r}: allowed {sorted(FACTORIZATIONS)}"
            )
        if self.r < 0:
            raise ValueError(f"r must be >= 0, not {self.r}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, not {self.alpha}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), not {self.dropout}")
        if self.init != "kaiming_b_zero":
            raise ValueError(
                f"init={self.init!r}: the only allowed scheme is 'kaiming_b_zero'. "
                "With B != 0 the adapted model does not start from the frozen model, and "
                "every comparison at t=0 loses its meaning."
            )
        # Placement.parse raises on wrong syntax: fail when the config is built, not mid-job.
        Placement.parse(self.placement)

    def scaling(self, n_slices: int = 1) -> float:
        """Return the factor ``s`` multiplying ``B·A`` for the rank being built.

        :param n_slices: slices of the same module merged by the factorization; only affects
            ``fused_nr``, whose rank is ``n·r``.
        :return: the multiplicative factor; ``0.0`` if the rank is ``0``.
        """
        rank = effective_rank(factorization=self.factorization, r=self.r, n_slices=n_slices)
        return lora_scaling(self.alpha, rank, use_rslora=self.use_rslora)


def lora_scaling(alpha: float, r: int, *, use_rslora: bool) -> float:
    """Return the scaling factor of the low-rank update.

    :param alpha: numerator.
    :param r: rank; ``0`` cancels the adapted branch.
    :param use_rslora: ``alpha/√r`` (rsLoRA) instead of ``alpha/r`` (LoRA, Hu et al. 2021).
    :return: the multiplicative factor.
    """
    if r == 0:
        return 0.0
    return alpha / math.sqrt(r) if use_rslora else alpha / r


def effective_rank(*, factorization: str, r: int, n_slices: int) -> int:
    """Return the rank of the adapter built by a factorization.

    Equal to ``r`` except for ``fused_nr``, defined as the single adapter costing as much
    as the ``n`` separate ones together.

    :param factorization: one of :data:`FACTORIZATIONS`.
    :param r: per-slice rank, as in the spec.
    :param n_slices: number of slices of the same module merged by the factorization.
    :return: the effective rank, also the scaling denominator.
    :raises ValueError: if ``n_slices`` is not positive.
    """
    if n_slices <= 0:
        raise ValueError(f"n_slices must be > 0, not {n_slices}")
    return n_slices * r if factorization == "fused_nr" else r


def trainable_parameters_formula(
    *,
    method: ReparamMethod,
    r: int,
    in_features: int,
    block_widths: Sequence[int],
    factorization: Factorization = "separate",
) -> int:
    """Return how many parameters a target must introduce, in closed form.

    Pure function, without torch, compared with the count on the real model: a mismatch
    means a missed target or an unfrozen backbone part. It deliberately does not call the
    measuring code (docs/08 §2).

    With ``span = Σ block_widths`` and ``n = len(block_widths)``:

    ===============  =========================================
    factorization    low-rank parameters
    ===============  =========================================
    ``separate``     ``Σ r·(in + wᵢ)``      = ``6·r·d`` on qkv
    ``fused``        ``r·(in + span)``      = ``4·r·d``
    ``fused_nr``     ``n·r·(in + span)``    = ``12·r·d``
    ``shared_b``     ``n·r·in + r·w``       = ``4·r·d``
    ===============  =========================================

    ``dora`` adds ``span`` magnitudes in all four cases (one per covered output row).

    :param method: ``"lora"`` or ``"dora"``.
    :param r: per-slice rank.
    :param in_features: columns of the target ``Linear``.
    :param block_widths: rows of each adapted slice (one entry for a whole target).
    :param factorization: one of :data:`FACTORIZATIONS`.
    :return: number of trainable parameters introduced by the target.
    :raises ValueError: on an unknown factorization, or ``shared_b`` with slices of
        different widths.
    """
    widths = list(block_widths)
    if not widths:
        raise ValueError("empty block_widths: a target without slices adapts nothing")
    span = sum(widths)
    n = len(widths)
    if factorization == "separate":
        total = sum(r * (in_features + width) for width in widths)
    elif factorization in ("fused", "fused_nr"):
        rank = effective_rank(factorization=factorization, r=r, n_slices=n)
        total = rank * (in_features + span)
    elif factorization == "shared_b":
        if len(set(widths)) != 1:
            raise ValueError(
                f"shared_b requires slices of equal width, got {widths}: "
                "the shared B has shape (width, r) and no common one exists"
            )
        total = n * r * in_features + r * widths[0]
    else:
        raise ValueError(f"unknown factorization {factorization!r}")
    if method == "dora":
        # One magnitude per covered output feature (NVlabs, peft).
        total += span
    return total


# --------------------------------------------------------------------------- #
# Parametrizations of a block of rows
# --------------------------------------------------------------------------- #


def _kaiming_uniform_bound(fan_in: int) -> float:
    """Return the bound of ``kaiming_uniform_(a=√5)``, in closed form.

    Used to draw ``A`` from an explicit ``Generator`` without relying on the
    ``generator=`` signature, missing in some torch versions used here:

        gain = √(2 / (1 + a²)) = √(1/3)   for a = √5
        std  = gain / √fan_in
        b    = √3 · std = 1 / √fan_in

    i.e. the same bound as the ``nn.Linear`` default.

    :param fan_in: columns of the tensor to initialize.
    :return: the bound ``b`` of ``U(-b, +b)``.
    :raises ValueError: if ``fan_in`` is not positive.
    """
    if fan_in <= 0:
        raise ValueError(f"fan_in must be > 0, not {fan_in}")
    return 1.0 / math.sqrt(fan_in)


class LowRankBlock(nn.Module, ABC):
    """How a block of rows of an ``nn.Linear`` is reparametrized.

    Abstracting at block level makes LoRA and DoRA orthogonal to the fused QKV: a whole
    target is a block covering all rows, a QKV slice a block covering a third. The base
    weight is received at every call, so it stays owned by the wrapped ``nn.Linear``.
    """

    def __init__(self, *, r: int, scaling: float, dropout: float) -> None:
        super().__init__()
        self.r = int(r)
        self.scaling = float(scaling)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @abstractmethod
    def delta(self, x: Tensor, w0: Tensor, base_out: Tensor) -> Tensor:
        """Return the contribution to add to the block output.

        :param x: module input, ``[..., in_features]``.
        :param w0: base-weight rows covered by the block, ``[width, in_features]``.
        :param base_out: ``F.linear(x, w0)`` without bias, ``[..., width]``, already computed
            by the wrapper.
        :return: tensor ``[..., width]`` to add to ``base_out``.
        """

    @abstractmethod
    def effective_weight(self, w0: Tensor) -> Tensor:
        """Return the materialized weight represented by the block.

        Used by ``merge_()`` and equivalence tests, never by the forward.

        :param w0: base-weight rows, ``[width, in_features]``.
        :return: tensor of the same shape.
        """

    def adapter_parameters(self) -> Iterator[nn.Parameter]:
        """:return: the parameters introduced by the block, in declaration order."""
        return self.parameters()


class LoRABlock(LowRankBlock):
    """``ΔW = s · B·A``, with ``A`` kaiming uniform and ``B`` zero.

    At t=0 the contribution is exactly zero, in exact and floating-point arithmetic.

    :param in_features: columns of the base weight.
    :param width: rows of the block.
    :param r: rank; with ``0`` the block allocates nothing and is a no-op.
    :param scaling: factor ``s``, from :func:`lora_scaling`.
    :param dropout: dropout on the input of the adapted branch only.
    :param generator: explicit random source for ``A``.
    """

    def __init__(
        self,
        *,
        in_features: int,
        width: int,
        r: int,
        scaling: float,
        dropout: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__(r=r, scaling=scaling, dropout=dropout)
        self.in_features = int(in_features)
        self.width = int(width)
        if self.r > 0:
            bound = _kaiming_uniform_bound(self.in_features)
            a = torch.empty(self.r, self.in_features).uniform_(-bound, bound, generator=generator)
            self.lora_A = nn.Parameter(a)
            self.lora_B = nn.Parameter(torch.zeros(self.width, self.r))

    def delta(self, x: Tensor, w0: Tensor, base_out: Tensor) -> Tensor:
        if self.r == 0:
            return torch.zeros_like(base_out)
        # Two projections in sequence, never the B@A product: materializing ΔW would cost
        # `width × in_features` per module at every forward.
        return self.scaling * F.linear(F.linear(self.drop(x), self.lora_A), self.lora_B)

    def effective_weight(self, w0: Tensor) -> Tensor:
        if self.r == 0:
            return w0
        return w0 + self.scaling * (self.lora_B @ self.lora_A)


class SharedBBlock(LowRankBlock):
    """One ``B`` and one ``A`` per slice: the dual of ``fused``.

    Covers a span of ``n`` contiguous bands of equal width and writes ``ΔW_i = s·B·A_i``:
    ``fused`` constrains what is read, this constrains where it is written. It costs
    exactly as much as ``fused`` (``n·r·in + r·w``). ``A`` is stacked in a single
    ``(n·r, in)`` tensor and bands follow row order; ``ΔW`` is never materialized.

    :param in_features: columns of the base weight.
    :param width: rows of the covered span; must be ``n_bands`` times the band width.
    :param n_bands: number of slices sharing ``B``.
    :param r: rank of each band; ``0`` makes the block a parameter-free no-op.
    :param scaling: factor ``s``, from :func:`lora_scaling`.
    :param dropout: dropout on the input of the adapted branch only.
    :param generator: random source for the ``A`` matrices.
    :raises ValueError: if ``width`` is not divisible by ``n_bands``.
    """

    def __init__(
        self,
        *,
        in_features: int,
        width: int,
        n_bands: int,
        r: int,
        scaling: float,
        dropout: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__(r=r, scaling=scaling, dropout=dropout)
        if n_bands <= 0:
            raise ValueError(f"n_bands must be > 0, not {n_bands}")
        if width % n_bands:
            raise ValueError(
                f"shared_b: span of {width} rows not divisible into {n_bands} equal bands"
            )
        self.in_features = int(in_features)
        self.width = int(width)
        self.n_bands = int(n_bands)
        self.band = self.width // self.n_bands
        if self.r > 0:
            bound = _kaiming_uniform_bound(self.in_features)
            a = torch.empty(self.n_bands * self.r, self.in_features).uniform_(
                -bound, bound, generator=generator
            )
            self.lora_A = nn.Parameter(a)
            self.lora_B = nn.Parameter(torch.zeros(self.band, self.r))

    def delta(self, x: Tensor, w0: Tensor, base_out: Tensor) -> Tensor:
        if self.r == 0:
            return torch.zeros_like(base_out)
        h = F.linear(self.drop(x), self.lora_A)  # [..., n_bands·r]
        h = h.unflatten(-1, (self.n_bands, self.r))  # [..., n_bands, r]
        out = F.linear(h, self.lora_B)  # [..., n_bands, band]
        return self.scaling * out.flatten(-2)

    def effective_weight(self, w0: Tensor) -> Tensor:
        if self.r == 0:
            return w0
        a = self.lora_A.unflatten(0, (self.n_bands, self.r))  # [n_bands, r, in]
        delta = (self.lora_B @ a).flatten(0, 1)  # [n_bands·band, in]
        return w0 + self.scaling * delta


class DoRABlock(LowRankBlock):
    """``W = m · V/‖V‖`` with ``V = W₀ + s·B·A`` and ``m`` per output feature.

    Composition rather than inheritance from :class:`LoRABlock`: the DoRA contribution is
    not additive, since it rescales the base branch too. The direction is any
    :class:`LowRankBlock`, which makes DoRA orthogonal to the factorization lattice
    (``dora`` + ``shared_b`` is a composition, not a new class); :func:`_make_block`
    builds direction and magnitude on the same rows.

    The forward follows the official NVlabs form::

        ns  = m / ‖V‖                        one factor per row
        y   = W₀x + (ns−1)·W₀x + ns·s·BAx    = ns·(W₀ + s·BA)x

    which at t=0 (``B = 0``, hence ``ns = 1.0`` exactly) is bitwise the frozen model.

    :param direction: low-rank parametrization producing ``V − W₀``; it must have
        ``dropout=0``, since DoRA owns the dropout.
    :param w0: covered base-weight rows ``[width, in_features]``, used to initialize
        ``m = ‖W₀‖``.
    :param dropout: input dropout; a single mask is shared by both branches, as in the
        official source (``dropout_x`` in ``dora.py``).
    """

    def __init__(
        self,
        direction: LowRankBlock,
        *,
        w0: Tensor,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(r=direction.r, scaling=direction.scaling, dropout=dropout)
        if not isinstance(direction.drop, nn.Identity):
            raise ValueError(
                "the direction of a DoRABlock must not have its own dropout: the official "
                "source draws a single mask and applies it to both branches"
            )
        self.in_features = int(w0.shape[1])
        self.width = int(w0.shape[0])
        self.direction = direction
        # `m` is computed with the same operation `_norm_scale` applies to V: at t=0 the two
        # norms match bit for bit and the ratio is exactly 1.0.
        self.magnitude = nn.Parameter(torch.linalg.norm(w0.detach(), dim=1).clone())

    def _norm_scale(self, w0: Tensor) -> Tensor:
        """:return: ``m/‖V‖``, one factor per row, shape ``[width]``."""
        v = self.direction.effective_weight(w0)
        return self.magnitude / torch.linalg.norm(v, dim=1)

    def delta(self, x: Tensor, w0: Tensor, base_out: Tensor) -> Tensor:
        ns = self._norm_scale(w0)
        # A single dropout draw shared by both branches, as in NVlabs; with dropout off (E3)
        # `base_dropped` is `base_out` and a second matmul is avoided.
        if isinstance(self.drop, nn.Identity):
            x_dir, base_dropped = x, base_out
        else:
            x_dir = self.drop(x)
            base_dropped = F.linear(x_dir, w0)
        out = (ns - 1.0) * base_dropped
        if self.r > 0:
            out = out + ns * self.direction.delta(x_dir, w0, base_out)
        return out

    def effective_weight(self, w0: Tensor) -> Tensor:
        v = self.direction.effective_weight(w0)
        # Scale first: one factor per row, exactly 1.0 at t=0, and `V * 1.0 == V` in IEEE 754.
        return v * self._norm_scale(w0).unsqueeze(1)


# --------------------------------------------------------------------------- #
# Module wrappers
# --------------------------------------------------------------------------- #


class AdaptedLinear(nn.Module):
    """A frozen ``nn.Linear`` plus one parametrization per block of rows.

    Shared machinery of the three wrappers named in docs/03 §4.5: ``forward``, ``merge_()``
    and parameter counting have a single implementation.

    :param base: ``Linear`` to wrap; its parameters are frozen.
    :param blocks: mapping ``label -> (slice | None, parametrization)``; ``None`` covers
        all rows. Labels become the adapter ``state_dict`` names (``q``, ``k``, ``v`` or
        ``all``).
    :raises ValueError: if two slices overlap or fall outside the module.
    """

    def __init__(
        self,
        base: nn.Linear,
        blocks: Mapping[str, tuple[OutSlice | None, LowRankBlock]],
    ) -> None:
        super().__init__()
        if not blocks:
            raise ValueError("an AdaptedLinear without blocks adapts nothing")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self._slices = _validated_slices(
            {label: out_slice for label, (out_slice, _) in blocks.items()},
            out_features=base.out_features,
        )
        self.blocks = nn.ModuleDict({label: blk for label, (_, blk) in blocks.items()})
        # An adapter lives where its base weight lives: parametrizations are created on CPU and
        # grafted into a model already on GPU.
        self.blocks.to(device=base.weight.device, dtype=base.weight.dtype)
        self.merged = False
        # A single block as wide as the module (unfused LoRA/DoRA): the forward adds directly
        # instead of going through a full-size delta tensor.
        only = next(iter(self._slices)) if len(self._slices) == 1 else None
        self._full_label = (
            only if only is not None and self._slices[only].width == base.out_features else None
        )

    # --- reading -----------------------------------------------------------
    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def slice_of(self, label: str) -> OutSlice:
        """:return: the rows covered by block ``label``."""
        return self._slices[label]

    def n_trainable(self) -> int:
        """:return: trainable parameters introduced by this wrapper."""
        return sum(p.numel() for p in self.adapter_parameters())

    def adapter_parameters(self) -> Iterator[nn.Parameter]:
        """:return: only the new parameters, never those of the base ``Linear``."""
        for block in self.blocks.values():
            yield from block.parameters()

    def adapter_state_dict(self) -> dict[str, Tensor]:
        """:return: the new weights, keyed ``{label}.{name}``."""
        return {
            f"{label}.{name}": param.detach()
            for label, block in self.blocks.items()
            for name, param in block.named_parameters()
        }

    # --- computation -------------------------------------------------------
    def effective_weight(self) -> Tensor:
        """Return the full weight represented by the wrapper, adapted and unadapted rows.

        :return: tensor ``[out_features, in_features]``; after ``merge_()`` the base weight,
            which is then already the effective one.
        """
        if self.merged:
            return self.base.weight
        w = self.base.weight.clone()
        for label, block in self.blocks.items():
            lo, hi = self._slices[label].start, self._slices[label].stop
            w[lo:hi] = block.effective_weight(self.base.weight[lo:hi])
        return w

    def forward(self, x: Tensor) -> Tensor:
        """:param x: ``[..., in_features]``.
        :return: ``[..., out_features]``.
        """
        if self.merged:
            return self.base(x)
        # Base output without bias: DoRA rescales the weight branch, not the bias.
        base_out = F.linear(x, self.base.weight)
        if self._full_label is not None:
            out = base_out + self.blocks[self._full_label].delta(x, self.base.weight, base_out)
        else:
            # `delta` is a new tensor: writing into `base_out` would modify it after DoRA fed it to
            # autograd.
            delta = torch.zeros_like(base_out)
            for label, block in self.blocks.items():
                lo, hi = self._slices[label].start, self._slices[label].stop
                delta[..., lo:hi] = block.delta(x, self.base.weight[lo:hi], base_out[..., lo:hi])
            out = base_out + delta
        if self.base.bias is not None:
            out = out + self.base.bias
        return out

    @torch.no_grad()
    def merge_(self) -> None:
        """Merge the adaptation into the base weight, in place. Idempotent.

        The wrapper stays in place as a pass-through; ``ReparamPEFT.unload_()`` removes it.
        """
        if self.merged:
            return
        self.base.weight.copy_(self.effective_weight())
        self.merged = True

    def extra_repr(self) -> str:
        slices = ", ".join(f"{k}[{v.start}:{v.stop}]" for k, v in self._slices.items())
        return f"in={self.in_features}, out={self.out_features}, blocks={slices}"


class LoRALinear(AdaptedLinear):
    """LoRA on the whole module: ``y = W₀x + b + s·B·A·x`` (Hu et al. 2021).

    :param base: ``Linear`` to wrap.
    :param r: rank; ``0`` makes the wrapper an exact, parameter-free no-op.
    :param alpha: scaling numerator.
    :param dropout: dropout on the input of the adapted branch.
    :param use_rslora: use ``alpha/√r``.
    :param generator: random source for ``A``.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        r: int,
        alpha: float,
        dropout: float = 0.0,
        use_rslora: bool = False,
        generator: torch.Generator | None = None,
    ) -> None:
        block = LoRABlock(
            in_features=base.in_features,
            width=base.out_features,
            r=r,
            scaling=lora_scaling(alpha, r, use_rslora=use_rslora),
            dropout=dropout,
            generator=generator,
        )
        super().__init__(base, {"all": (None, block)})


class DoRALinear(AdaptedLinear):
    """DoRA on the whole module: ``W = m·V/‖V‖``, ``m`` per output feature.

    :param base: ``Linear`` to wrap.
    :param r: direction rank; ``0`` keeps only the rescaling.
    :param alpha: scaling numerator.
    :param dropout: dropout on the input of the adapted branch.
    :param use_rslora: use ``alpha/√r`` (combinable with DoRA).
    :param generator: random source for ``A``.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        r: int,
        alpha: float,
        dropout: float = 0.0,
        use_rslora: bool = False,
        generator: torch.Generator | None = None,
    ) -> None:
        block = _make_block(
            method="dora",
            factorization="separate",
            in_features=base.in_features,
            width=base.out_features,
            n_bands=1,
            w0=base.weight,
            r=r,
            scaling=lora_scaling(alpha, r, use_rslora=use_rslora),
            dropout=dropout,
            generator=generator,
        )
        super().__init__(base, {"all": (None, block)})


class FusedQKVLoRALinear(AdaptedLinear):
    """Independent adapters on the output slices of a fused QKV.

    What ``microsoft/LoRA`` (``MergedLinear``) does and ``peft`` does not: given
    ``Linear(768 → 2304)`` with layout ``[q|k|v]``, each requested projection gets its own
    ``A`` and ``B`` and the others stay exactly unchanged, so ``qk_only`` is a structural
    constraint. One rank-*r* adapter on the whole matrix is not equivalent
    (``fused-r ⊊ separate-r ⊊ fused-3r``), hence E3-R budget parity in trainable
    parameters. ``factorization`` selects the lattice element (``"separate"`` by default).

    :param base: fused ``Linear`` to wrap.
    :param slices: mapping ``role -> output-row slice``.
    :param r: rank of each slice (``n·r`` for ``fused_nr``).
    :param alpha: scaling numerator.
    :param method: ``"lora"`` or ``"dora"``; with ``"dora"`` and ``separate`` each slice
        has its own magnitudes, leaving unadapted slices untouched.
    :param factorization: one of :data:`FACTORIZATIONS`.
    :param dropout: dropout on the input of the adapted branch.
    :param use_rslora: use ``alpha/√r``.
    :param generator: random source for the ``A`` matrices.
    """

    def __init__(
        self,
        base: nn.Linear,
        slices: Mapping[str, OutSlice],
        *,
        r: int,
        alpha: float,
        method: ReparamMethod = "lora",
        factorization: Factorization = "separate",
        dropout: float = 0.0,
        use_rslora: bool = False,
        generator: torch.Generator | None = None,
    ) -> None:
        if not slices:
            raise ValueError("FusedQKVLoRALinear requires at least one slice")
        super().__init__(
            base,
            build_blocks(
                base,
                slices,
                method=method,
                factorization=factorization,
                r=r,
                alpha=alpha,
                dropout=dropout,
                use_rslora=use_rslora,
                generator=generator,
            ),
        )


def build_blocks(
    base: nn.Linear,
    slices: Mapping[str, OutSlice],
    *,
    method: ReparamMethod = "lora",
    factorization: Factorization = "separate",
    r: int,
    alpha: float,
    dropout: float = 0.0,
    use_rslora: bool = False,
    generator: torch.Generator | None = None,
) -> dict[str, tuple[OutSlice | None, LowRankBlock]]:
    """Turn a factorization into blocks of rows, ready for ``AdaptedLinear``.

    The only place where a factorization name becomes a geometry: ``separate`` yields one
    block per slice, the others a single block over the span. Merged slices must tile a
    contiguous span, otherwise the single adapter would write into rows of a projection
    that was not requested.

    :param base: target ``Linear``, providing ``in_features`` and the ``W₀`` rows for DoRA.
    :param slices: mapping ``label -> output-row slice``.
    :param method: ``"lora"`` or ``"dora"``.
    :param factorization: one of :data:`FACTORIZATIONS`.
    :param r: per-slice rank.
    :param alpha: scaling numerator.
    :param dropout: dropout on the input of the adapted branch.
    :param use_rslora: use ``alpha/√r``.
    :param generator: random source for the ``A`` matrices.
    :return: mapping ``label -> (slice, parametrization)``; with a merged factorization
        the label concatenates the merged ones in row order (``"qkv"``, ``"qk"``).
    :raises ValueError: on an unknown factorization, slices not tiling a contiguous span,
        or ``shared_b`` with slices of different widths.
    """
    if factorization not in FACTORIZATIONS:
        raise ValueError(f"unknown factorization {factorization!r}")
    # Validate before building: `base.weight[start:stop]` silently truncates an out-of-range
    # `stop`.
    resolved = _validated_slices(dict(slices), out_features=base.out_features)

    def block(slice_: OutSlice, *, n_bands: int) -> LowRankBlock:
        rank = effective_rank(factorization=factorization, r=r, n_slices=n_bands)
        return _make_block(
            method=method,
            factorization=factorization,
            in_features=base.in_features,
            width=slice_.width,
            n_bands=n_bands,
            w0=base.weight[slice_.start : slice_.stop],
            r=rank,
            scaling=lora_scaling(alpha, rank, use_rslora=use_rslora),
            dropout=dropout,
            generator=generator,
        )

    if factorization == "separate" or len(resolved) == 1:
        # With a single slice all factorizations coincide.
        return {
            label: (qkv_slice, block(qkv_slice, n_bands=1)) for label, qkv_slice in resolved.items()
        }

    sorted_items = sorted(resolved.items(), key=lambda kv: kv[1].start)
    span = _contiguous_span(sorted_items)
    widths = [qkv_slice.width for _, qkv_slice in sorted_items]
    if factorization in EQUAL_SLICE_FACTORIZATIONS and len(set(widths)) != 1:
        raise ValueError(f"{factorization} requires slices of equal width, got {widths}")
    label = "".join(name for name, _ in sorted_items)
    return {label: (span, block(span, n_bands=len(sorted_items)))}


def _contiguous_span(sorted_items: Sequence[tuple[str, OutSlice]]) -> OutSlice:
    """Return the span covered by the slices, if they cover it without gaps.

    :param sorted_items: ``(label, slice)`` pairs sorted by first row.
    :return: the slice ``[first.start, last.stop)``.
    :raises ValueError: naming the two slices around the gap.
    """
    for (name_a, a), (name_b, b) in zip(sorted_items, sorted_items[1:], strict=False):
        if a.stop != b.start:
            raise ValueError(
                f"slices {name_a}=[{a.start},{a.stop}) and {name_b}=[{b.start},{b.stop}) "
                "are not contiguous: a merged factorization would also cover the rows "
                "in between, which belong to a projection that was not requested"
            )
    return OutSlice(sorted_items[0][1].start, sorted_items[-1][1].stop)


def _make_block(
    *,
    method: ReparamMethod,
    factorization: Factorization,
    in_features: int,
    width: int,
    n_bands: int,
    w0: Tensor,
    r: int,
    scaling: float,
    dropout: float,
    generator: torch.Generator | None,
) -> LowRankBlock:
    """Compose the parametrization of a block of rows: direction plus method.

    The factorization chooses how the low-rank direction is written; the method decides
    whether it is the update (LoRA) or only its directional part (DoRA). A new
    factorization is a :class:`LowRankBlock` subclass plus a branch here; a new method is a
    decorator like :class:`DoRABlock`.

    :param method: ``"lora"`` or ``"dora"``.
    :param factorization: one of :data:`FACTORIZATIONS`.
    :param in_features: columns of the base weight.
    :param width: rows covered by the block.
    :param n_bands: bands of the span (used only by ``shared_b``).
    :param w0: covered base-weight rows, used by DoRA for ``m = ‖W₀‖``.
    :param r: rank, already effective (see :func:`effective_rank`).
    :param scaling: factor ``s``, consistent with ``r``.
    :param dropout: dropout on the input of the adapted branch.
    :param generator: random source for the ``A`` matrices.
    :return: the initialized parametrization.
    :raises ValueError: on an unknown method or factorization.
    """
    if method not in ("lora", "dora"):
        raise ValueError(f"unknown adaptation method {method!r}")
    # With DoRA the dropout belongs to the decorator (one mask for both branches); the
    # direction gets zero.
    direction_dropout = 0.0 if method == "dora" else dropout
    direction: LowRankBlock
    if factorization == "shared_b":
        direction = SharedBBlock(
            in_features=in_features,
            width=width,
            n_bands=n_bands,
            r=r,
            scaling=scaling,
            dropout=direction_dropout,
            generator=generator,
        )
    elif factorization in FACTORIZATIONS:
        direction = LoRABlock(
            in_features=in_features,
            width=width,
            r=r,
            scaling=scaling,
            dropout=direction_dropout,
            generator=generator,
        )
    else:
        raise ValueError(f"unknown factorization {factorization!r}")
    if method == "lora":
        return direction
    return DoRABlock(direction, w0=w0, dropout=dropout)


def _validated_slices(
    slices: Mapping[str, OutSlice | None], *, out_features: int
) -> dict[str, OutSlice]:
    """Normalize the slices and check that they are inside the module and disjoint.

    ``None`` means all rows. An out-of-range slice would be truncated by
    ``Tensor.__getitem__`` and overlapping slices would add up on the same rows.

    :param slices: mapping ``label -> slice`` (or ``None``).
    :param out_features: output rows of the target module.
    :return: the same mapping with ``None`` resolved.
    :raises ValueError: naming the out-of-range slice or the first overlapping pair.
    """
    resolved = {
        label: (slice_ if slice_ is not None else OutSlice(0, out_features))
        for label, slice_ in slices.items()
    }
    for label, slice_ in resolved.items():
        if slice_.stop > out_features:
            raise ValueError(
                f"slice {label}=[{slice_.start},{slice_.stop}) outside the "
                f"{out_features} output rows of the module"
            )
    sorted_items = sorted(resolved.items(), key=lambda kv: kv[1].start)
    for (name_a, a), (name_b, b) in zip(sorted_items, sorted_items[1:], strict=False):
        if b.start < a.stop:
            raise ValueError(
                f"overlapping slices {name_a}=[{a.start},{a.stop}) and "
                f"{name_b}=[{b.start},{b.stop}): two adapters on the same rows "
                "add their contributions, and the effective rank is no longer the "
                "declared one"
            )
    return resolved


# --------------------------------------------------------------------------- #
# Strategy
# --------------------------------------------------------------------------- #


@ADAPTERS.register("reparam")
class ReparamPEFT:
    """Apply LoRA or DoRA to the targets declared by the backbone.

    Module names come from ``backbone.target_map()`` resolved with the spec ``placement``,
    so the same config works on fused and separate QKV and no hard-coded strings appear in
    configs (a bare ``proj`` would also match the two ``patch_embed`` ``Conv3d``).

    :param spec: adaptation configuration.
    """

    family: Family = "reparam"

    def __init__(self, spec: ReparamSpec) -> None:
        self.spec = spec
        self._adapted: dict[str, AdaptedLinear] = {}
        # Where each wrapper was grafted (parent module and attribute), captured at injection: in
        # a composition prompts replace `blocks.{i}`, so qualified names no longer resolve while
        # the `attn` object is the same. Anchors make `unload_()` independent of the unload order
        # (ADR-064).
        self._anchors: dict[str, tuple[nn.Module, str]] = {}
        self._module: nn.Module | None = None

    # --- AdaptationStrategy contract ---------------------------------------
    def apply(self, backbone: VideoBackbone, /) -> AdaptedBackbone:
        """Freeze the backbone and replace every target with its wrapper.

        :param backbone: backbone to adapt, modified in place.
        :return: the receipt with counts and targets.
        :raises RuntimeError: if the strategy was already applied or a target is already
            adapted.
        :raises TypeError: if a resolved target is not an ``nn.Linear``.
        """
        if self._adapted:
            raise RuntimeError(
                "this ReparamPEFT was already applied: a second apply() "
                "would nest the adapters and the parameter count "
                "would still look plausible"
            )
        backbone.freeze()
        module = backbone.module()
        target_map = backbone.target_map()
        refs = target_map.resolve(self.spec.placement)

        generator = torch.Generator().manual_seed(self.spec.seed)
        expected = 0
        for name, group in _group_by_module(refs).items():
            base = _linear_at(module, name)
            wrapper = self._build(base, group, generator)
            self._anchors[name] = _replace_submodule(module, name, wrapper)
            self._adapted[name] = wrapper
            # The formula uses the requested widths, not those of the built blocks (docs/08 §2).
            expected += trainable_parameters_formula(
                method=self.spec.method,
                r=self.spec.r,
                in_features=base.in_features,
                block_widths=[_width(ref, base) for ref in group],
                factorization=self.spec.factorization,
            )

        self._module = module
        report = AdaptationReport(
            family=self.family,
            method=self.spec.method,
            n_targets=len(self._adapted),
            n_trainable=sum(p.numel() for p in module.parameters() if p.requires_grad),
            n_trainable_formula=expected,
            extra_tokens=0,
            targets=tuple(self._adapted),
            notes={
                "placement": str(Placement.parse(self.spec.placement)),
                "factorization": self.spec.factorization,
                "r": self.spec.r,
                "alpha": self.spec.alpha,
                "use_rslora": self.spec.use_rslora,
            },
        )
        return AdaptedBackbone(backbone=backbone, strategy=self, report=report)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """:return: the parameters of all adapters, in target order."""
        for wrapper in self._adapted.values():
            yield from wrapper.adapter_parameters()

    def can_merge(self) -> bool:
        """:return: always ``True``: the defining property of this family."""
        return True

    def merge_parts(self) -> tuple[str, ...]:
        """:return: always ``("reparam",)``: everything merges."""
        return (self.family,)

    def merge_(self) -> None:
        """Merge all adapters into the backbone weights. Idempotent."""
        for wrapper in self._adapted.values():
            wrapper.merge_()

    def unload_(self) -> int:
        """Put the bare ``nn.Linear`` modules back.

        Call it after ``merge_()`` to keep the effect; alone it discards the adaptation and
        restores the backbone bit for bit. ``requires_grad`` is not restored.

        :return: number of removed wrappers.
        """
        return self._unwrap(list(self._adapted))

    def unload_merged_(self) -> int:
        """Remove only the already merged wrappers, leaving the others in place.

        After ``merge_()`` an ``AdaptedLinear`` is a pass-through still traversed at every
        forward. Equivalent to ``unload_()`` here; it matters in compositions.

        :return: number of removed wrappers.
        """
        return self._unwrap([name for name, w in self._adapted.items() if w.merged])

    def _unwrap(self, nomi: list[str]) -> int:
        """Put the bare ``nn.Linear`` modules back in place of the named wrappers.

        Goes through the anchors rather than ``get_submodule``, since in a composition the
        qualified names may no longer resolve.

        :param nomi: targets to unwrap.
        :return: number of removed wrappers.
        """
        if self._module is None:
            return 0
        for name in nomi:
            parent, leaf = self._anchors.pop(name)
            setattr(parent, leaf, self._adapted.pop(name).base)
        if not self._adapted:
            self._module = None
        return len(nomi)

    def extra_tokens(self) -> int:
        """:return: always 0: reparam does not lengthen the sequence."""
        return 0

    def state_dict_adapter(self) -> dict[str, Tensor]:
        """:return: only the new weights, keyed ``{target}.{block}.{tensor}``."""
        return {
            f"{name}.{key}": value
            for name, wrapper in self._adapted.items()
            for key, value in wrapper.adapter_state_dict().items()
        }

    # --- internals ---------------------------------------------------------
    def _build(
        self,
        base: nn.Linear,
        refs: Sequence[TargetRef],
        generator: torch.Generator,
    ) -> AdaptedLinear:
        """Choose the wrapper for the references pointing to the same module.

        Several references on the same name mean a fused QKV, each carrying its projection
        slice.

        :return: the wrapper, not yet installed.
        :raises ValueError: if a target is requested both whole and sliced, or several times
            without slices.
        """
        with_slice = [ref for ref in refs if ref.out_slice is not None]
        if with_slice:
            if len(with_slice) != len(refs):
                raise ValueError(
                    f"target {refs[0].name!r} is requested both whole and sliced: "
                    "the backbone map is inconsistent"
                )
            return FusedQKVLoRALinear(
                base,
                {ref.role: ref.out_slice for ref in with_slice if ref.out_slice},
                r=self.spec.r,
                alpha=self.spec.alpha,
                method=self.spec.method,
                factorization=self.spec.factorization,
                dropout=self.spec.dropout,
                use_rslora=self.spec.use_rslora,
                generator=generator,
            )
        if len(refs) != 1:
            raise ValueError(
                f"target {refs[0].name!r} appears {len(refs)} times without slices: "
                "they would be overlapping adapters on the same rows"
            )
        class_name = DoRALinear if self.spec.method == "dora" else LoRALinear
        return class_name(
            base,
            r=self.spec.r,
            alpha=self.spec.alpha,
            dropout=self.spec.dropout,
            use_rslora=self.spec.use_rslora,
            generator=generator,
        )


def _width(ref: TargetRef, base: nn.Linear) -> int:
    """:return: output rows requested by the reference (the slice, or the whole module)."""
    return ref.out_slice.width if ref.out_slice is not None else base.out_features


def _group_by_module(refs: Sequence[TargetRef]) -> dict[str, list[TargetRef]]:
    """Group references by qualified name, keeping their order.

    On a fused QKV the same ``Linear`` appears once per requested projection.

    :return: ``name -> references``, in order of first appearance.
    """
    groups: dict[str, list[TargetRef]] = {}
    for ref in refs:
        groups.setdefault(ref.name, []).append(ref)
    return groups


def _linear_at(module: nn.Module, name: str) -> nn.Linear:
    """Return an ``nn.Linear`` by qualified name, with errors naming the cause.

    :raises RuntimeError: if the module is already adapted (checked before the type).
    :raises TypeError: if the target exists but is not an ``nn.Linear`` (e.g. a bare
        ``proj`` pattern matching the ``patch_embed.proj`` ``Conv3d``).
    """
    target = module.get_submodule(name)
    if isinstance(target, AdaptedLinear):
        raise RuntimeError(f"{name} is already adapted: the injection is not idempotent")
    if not isinstance(target, nn.Linear):
        raise TypeError(f"{name} is {type(target).__name__}, not nn.Linear")
    return target


def _replace_submodule(module: nn.Module, name: str, new: nn.Module) -> tuple[nn.Module, str]:
    """Replace a submodule by qualified name, in place.

    ``get_submodule`` also handles ``nn.ModuleList`` indices (``blocks.0``).

    :return: the anchor (parent module and attribute), so the module can be unwrapped
        without resolving the name from the root (ADR-064).
    """
    *parent_module, leaf = name.split(".")
    parent = module.get_submodule(".".join(parent_module)) if parent_module else module
    setattr(parent, leaf, new)
    return parent, leaf
