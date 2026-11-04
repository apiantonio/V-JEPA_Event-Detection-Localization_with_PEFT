"""Prompt PEFT: shallow and deep VPT.

Replicates the official ``KMnP/vpt`` (``src/models/vit_prompt/vit.py``, Jia et al.,
*Visual Prompt Tuning*, ECCV 2022):

* uniform initialization with the Xavier bound ``val = √(6 / (3·∏patch_size + D))``;
* shallow: prompts enter once before the first block and traverse the whole network;
* deep: at every following block the previous prompts are discarded and replaced by
  new parameters while image tokens continue (``forward_deep_prompt``);
* dropout on prompts only.

Deviation: VPT places prompts after the CLS token. V-JEPA 2.1 has no CLS and uses
3D-RoPE, so prompts are appended at the end and declared as registers. Upstream
already leaves trailing registers unrotated::

    rotate_queries_or_keys(x, pos, n_registers, has_cls_first)
        x_cls = x[..., :n_cls, :]              # head, not rotated
        x_ctx = x[..., n_cls : N-n_registers]  # rotated with `pos`
        x_reg = x[..., N-n_registers:, :]      # tail, not rotated
    (``vjepa2/app/vjepa_2_1/models/utils/modules.py:14``)

so the ``rope_identity`` policy comes from the reference code and ``N_ctx`` stays
consistent with ``pos = arange(T·H·W)``. Without it V-JEPA 2.1 fails with a shape
error rather than a wrong result.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from evd.models.adapt.base import (
    AdaptationReport,
    AdaptationStrategy,
    AdaptedBackbone,
    Family,
)
from evd.models.registry import Registry
from evd.types import TokenLayout

if TYPE_CHECKING:
    from evd.models.backbones.base import VideoBackbone

PromptVariant = Literal["vpt_shallow", "vpt_deep", "apt", "da_vpt"]
PosPolicy = Literal["rope_identity", "learned_abs", "none"]
PromptNorm = Literal["none", "l2", "layernorm"]

#: Variants inserting prompts into the sequence, handled by `PromptedBlock`; ``apt``
#: injects inside attention instead (``adapt/apt.py``).
SEQUENTIAL_VARIANTS: tuple[PromptVariant, ...] = ("vpt_shallow", "vpt_deep", "da_vpt")

#: Strategy implementing each variant. One family (``prompt``), three mechanisms
#: (sequence insertion, K/V injection, sequence insertion with a loss term), one class
#: each; the family factory is in ``adapt/prompt_family.py``.
PROMPT_VARIANTS: Registry[Callable[[PromptSpec], AdaptationStrategy]] = Registry("prompt variant")

#: Positional policies allowed for each kind of positional encoding: a compatibility
#: table, so a new encoding is one more row, not a branch in ``apply()``.
ALLOWED_POLICIES: dict[str, tuple[PosPolicy, ...]] = {
    "rope_3d": ("rope_identity",),
    "sincos_abs": ("none", "learned_abs"),
    "learned_abs": ("none", "learned_abs"),
    "none": ("none",),
}


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PromptSpec:
    """Configuration of a prompt adaptation.

    :param variant: ``"vpt_shallow"`` (prompts inserted once and carried to the end),
        ``"vpt_deep"`` (new prompts at every block, previous ones discarded), ``"apt"``
        (prompts as keys and values inside attention, ``adapt/apt.py``) or ``"da_vpt"``
        (deep VPT with class-proxy banks in a metric-learning term, ``adapt/davpt.py``).
    :param n_tokens: prompts per block; directly drives the inference cost.
    :param block_indices: explicit sorted block indices of a sparse VPT-deep bank;
        excludes a nonzero ``start_block`` and an explicit ``depth``.
    :param depth: number of prompted blocks from ``start_block``; ``None`` means to the
        end. Not allowed for ``vpt_shallow``.
    :param start_block: first prompted block (``prompt_start`` in the official APT); 0
        for VPT.
    :param pos_policy: how prompts relate to the backbone positional encoding; checked
        against ``token_layout()`` in ``apply()``.
    :param init: ``"xavier_uniform"`` (official VPT), ``"normal"``, or
        ``"from_patch_stats"`` (not implemented, raises).
    :param init_divider: divides the initialization bound (``init_divide_by8`` of
        DA-VPT); 1.0 elsewhere.
    :param norm: normalization applied to the bank before insertion (``vpt_norm`` of
        DA-VPT); ``"none"`` is official VPT.
    :param apt_reparam: ``"apt"`` only: add the per-token ``prompt_reparam`` scale,
        initialized to 1.
    :param dropout: dropout on prompts only, as in VPT.
    :param seed: initialization seed.
    """

    variant: PromptVariant = "vpt_deep"
    n_tokens: int = 8
    depth: int | None = None
    start_block: int = 0
    pos_policy: PosPolicy = "rope_identity"
    init: Literal["xavier_uniform", "normal", "from_patch_stats"] = "xavier_uniform"
    init_divider: float = 1.0
    norm: PromptNorm = "none"
    apt_reparam: bool = False
    dropout: float = 0.0
    seed: int = 0
    block_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.block_indices is not None:
            indices = tuple(self.block_indices)
            if self.variant != "vpt_deep" or self.start_block != 0 or self.depth is not None:
                raise ValueError("block_indices requires vpt_deep, start_block=0 and depth=None")
            if (
                not indices
                or any(type(i) is not int or i < 0 for i in indices)
                or tuple(sorted(set(indices))) != indices
            ):
                raise ValueError(
                    "block_indices must be nonempty, sorted, unique nonnegative integers"
                )
            object.__setattr__(self, "block_indices", indices)
        if self.variant not in get_args(PromptVariant):
            raise ValueError(f"variant={self.variant!r}: allowed {list(get_args(PromptVariant))}")
        if self.n_tokens <= 0:
            raise ValueError(f"n_tokens must be > 0, not {self.n_tokens}")
        if self.start_block < 0:
            raise ValueError(f"start_block must be >= 0, not {self.start_block}")
        if self.depth is not None:
            if self.depth <= 0:
                raise ValueError(f"depth must be > 0, not {self.depth}")
            if self.variant == "vpt_shallow":
                raise ValueError(
                    "vpt_shallow has no depth parameter: prompts enter once "
                    "and traverse the whole network. To limit the blocks receiving "
                    "new prompts, use the 'vpt_deep' variant."
                )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), not {self.dropout}")
        if self.init_divider <= 0:
            raise ValueError(f"init_divider must be > 0, not {self.init_divider}")
        if self.norm not in get_args(PromptNorm):
            raise ValueError(f"norm={self.norm!r}: allowed {list(get_args(PromptNorm))}")
        if self.apt_reparam and self.variant != "apt":
            raise ValueError(
                f"apt_reparam is the per-token scale of the official APT and is meaningless "
                f"for variant={self.variant!r}"
            )
        if self.init == "from_patch_stats":
            raise NotImplementedError(
                "init='from_patch_stats' is in the docs/03 contract but not implemented: "
                "it needs patch-token statistics on a data sample, i.e. an extra pass "
                "before training. Use 'xavier_uniform'."
            )

    def prompted_blocks(self, n_blocks: int) -> set[int]:
        """Return the indices of the blocks receiving a new bank.

        The only place where ``variant``, ``start_block`` and ``depth`` become indices, shared
        by VPT, APT and DA-VPT.

        :param n_blocks: network depth.
        :return: the indices; empty if ``start_block`` is beyond the end.
        :raises ValueError: if ``n_blocks`` is not positive.
        """
        if n_blocks <= 0:
            raise ValueError(f"n_blocks must be > 0, not {n_blocks}")
        if self.block_indices is not None:
            if self.block_indices[-1] >= n_blocks:
                raise ValueError(f"block_indices exceeds backbone depth {n_blocks}")
            return set(self.block_indices)
        if self.start_block >= n_blocks:
            raise ValueError(
                f"start_block={self.start_block} beyond the {n_blocks} blocks of the network: "
                "the adaptation would resolve to zero banks"
            )
        if self.variant == "vpt_shallow":
            return {self.start_block}
        end = n_blocks if self.depth is None else min(self.start_block + self.depth, n_blocks)
        return set(range(self.start_block, end))


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #


def xavier_uniform_bound(embed_dim: int, patch_numel: int) -> float:
    """Return the initialization bound of the official VPT.

    ``val = √(6 / (3·∏patch_size + prompt_dim))``: Xavier with fan-in one flattened patch
    (3 channels × tubelet volume) and fan-out one prompt.

    :param embed_dim: model dimension.
    :param patch_numel: elements of a flattened patch, channels included.
    :return: the bound ``val`` of ``U(-val, +val)``.
    :raises ValueError: on non-positive dimensions.
    """
    if embed_dim <= 0 or patch_numel <= 0:
        raise ValueError(f"invalid dimensions: embed_dim={embed_dim}, patch={patch_numel}")
    return math.sqrt(6.0 / float(patch_numel + embed_dim))


class PromptTokens(nn.Module):
    """A bank of ``n_tokens`` learned vectors, with VPT dropout.

    :param n_tokens: number of prompts.
    :param embed_dim: size of each prompt.
    :param bound: bound of the uniform initialization; the standard deviation with
        ``init="normal"``.
    :param init: initialization scheme.
    :param norm: normalization applied before every use (``vpt_norm`` of DA-VPT):
        ``"none"`` (official VPT), ``"l2"`` (unit-norm prompts) or ``"layernorm"`` (two
        trainable parameters per dimension, counted).
    :param dropout: dropout on prompts only.
    :param generator: explicit random source, independent of the global RNG.
    """

    def __init__(
        self,
        *,
        n_tokens: int,
        embed_dim: int,
        bound: float,
        init: str = "xavier_uniform",
        norm: PromptNorm = "none",
        dropout: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        values = torch.empty(n_tokens, embed_dim)
        if init == "normal":
            values.normal_(0.0, bound, generator=generator)
        else:
            values.uniform_(-bound, bound, generator=generator)
        self.prompt = nn.Parameter(values)
        self.norm_kind = norm
        self.norm = nn.LayerNorm(embed_dim) if norm == "layernorm" else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def n_tokens(self) -> int:
        return int(self.prompt.shape[0])

    def tokens(self) -> Tensor:
        """Return the normalized bank, without dropout or batch expansion.

        Used by the DA-VPT loss as the proxy set, where dropout would make both sides of the
        similarity stochastic.

        :return: ``[n_tokens, D]``.
        """
        if self.norm_kind == "l2":
            return F.normalize(self.prompt, p=2.0, dim=-1, eps=1e-6)
        return cast(Tensor, self.norm(self.prompt))

    def expand(self, batch: int) -> Tensor:
        """:param batch: batch size.
        :return: ``[batch, n_tokens, D]``.
        """
        return self.drop(self.tokens()).unsqueeze(0).expand(batch, -1, -1)


class PromptedBlock(nn.Module):
    """A transformer block that sees ``n_tokens`` trailing prompts.

    Three behaviours reproduce the two VPT variants:

    ``insert``
        append the prompts to the sequence (first prompted block, both variants).
    ``replace``
        discard the incoming prompts and add new ones (``forward_deep_prompt``).
    ``carry``
        leave the sequence untouched (shallow after the first block, deep beyond the last
        bank).

    ``strip`` is true only on the last block of the network: with prompts installed,
    intermediate block outputs contain the prompts, so building a ``BackboneOutput`` on
    them fails the grid check. ``n_in`` and ``n_out`` differ because DA-VPT proxy banks
    are class-wide next to ``n_tokens``-wide banks.

    :param block: block to wrap; it stays frozen and unmodified.
    :param prompts: prompts of this block, or ``None`` for ``carry``.
    :param n_in: prompts arriving from the previous block (0 before insertion).
    :param action: ``"insert"``, ``"replace"`` or ``"carry"``.
    :param strip: whether to remove the prompts from the output.
    :raises ValueError: if action and prompts are inconsistent.
    """

    def __init__(
        self,
        block: nn.Module,
        prompts: PromptTokens | None,
        *,
        n_in: int,
        action: Literal["insert", "replace", "carry"],
        strip: bool,
    ) -> None:
        super().__init__()
        if action in ("insert", "replace") and prompts is None:
            raise ValueError(f"action={action!r} requires prompts")
        if action == "carry" and prompts is not None:
            raise ValueError("action='carry' must not receive its own prompts")
        self.block = block
        self.prompts = prompts
        self.n_in = int(n_in)
        self.n_out = int(n_in) if prompts is None else prompts.n_tokens
        self.action = action
        self.strip = bool(strip)
        # Prompts are created on CPU and grafted into a block already on GPU; without this line
        # `torch.cat` would mix devices.
        reference = next(block.parameters(), None)
        if prompts is not None and reference is not None:
            prompts.to(device=reference.device, dtype=reference.dtype)

    @property
    def n_tokens(self) -> int:
        """:return: prompts emitted by this block, i.e. its ``n_registers``."""
        return self.n_out

    def patch_tokens(self, x: Tensor) -> Tensor:
        """Separate the patch tokens from the incoming prompts.

        :param x: ``[B, N, D]`` as received by ``forward``.
        :return: the ``[B, N - n_in, D]`` view of the patch tokens only.
        """
        return x if self.n_in == 0 else x[:, : -self.n_in]

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Any:
        """Insert or replace the prompts, run the block and strip them if requested.

        :param x: ``[B, N, D]`` block input.
        :return: what the block returns (a tensor, or the V-JEPA 2.1 ``(x, attn)`` tuple) with
            the corrected sequence.
        """
        if self.prompts is not None:
            base = x if self.action == "insert" else self.patch_tokens(x)
            x = torch.cat((base, self.prompts.expand(x.shape[0])), dim=1)
        out = self.block(x, *args, **kwargs)
        if not self.strip or self.n_out == 0:
            return out
        if isinstance(out, tuple):
            return (out[0][:, : -self.n_out], *out[1:])
        return out[:, : -self.n_out]


# --------------------------------------------------------------------------- #
# Strategy
# --------------------------------------------------------------------------- #


class PromptPEFT:
    """Install VPT prompts on the blocks declared by the backbone.

    :param spec: adaptation configuration.
    :raises ValueError: if the variant does not insert prompts into the sequence (``apt``
        has its own strategy).
    """

    family: Family = "prompt"

    def __init__(self, spec: PromptSpec) -> None:
        if spec.variant not in SEQUENTIAL_VARIANTS:
            raise ValueError(
                f"variant={spec.variant!r} does not insert prompts into the sequence: it uses "
                f"another strategy (see PROMPT_VARIANTS). Handled here: {list(SEQUENTIAL_VARIANTS)}."
            )
        self.spec = spec
        self._blocks: list[PromptedBlock] = []
        self._container: nn.ModuleList | None = None
        self._original_registers: list[tuple[nn.Module, int]] = []

    # --- AdaptationStrategy contract ---------------------------------------
    def apply(self, backbone: VideoBackbone, /) -> AdaptedBackbone:
        """Freeze the backbone and wrap the blocks that must see the prompts.

        :param backbone: backbone to adapt, modified in place.
        :return: the receipt with counts and touched blocks.
        :raises RuntimeError: if the strategy was already applied.
        :raises ValueError: if the ``pos_policy`` is incompatible with the backbone or cannot
            be realized.
        """
        if self._blocks:
            raise RuntimeError("this PromptPEFT was already applied")
        layout = backbone.token_layout()
        validate_pos_policy(layout, self.spec.pos_policy)

        backbone.freeze()
        container = backbone.blocks()
        n_blocks = len(container)
        prompted = self.spec.prompted_blocks(n_blocks)

        generator = torch.Generator().manual_seed(self.spec.seed)
        bound = self._bound(backbone)
        n_in = 0
        for i in range(n_blocks):
            action = self._action(i, prompted)
            # Prompts are removed once, at the network output: with fewer banks than blocks the
            # final blocks keep seeing the last bank, as in the official VPT-deep.
            strip = i == n_blocks - 1
            tokens = (
                PromptTokens(
                    n_tokens=self._n_tokens_of(i),
                    embed_dim=backbone.embed_dim,
                    bound=bound,
                    init=self.spec.init,
                    norm=self.spec.norm,
                    dropout=self.spec.dropout,
                    generator=generator,
                )
                if action != "carry"
                else None
            )
            wrapper = PromptedBlock(container[i], tokens, n_in=n_in, action=action, strip=strip)
            container[i] = wrapper
            self._blocks.append(wrapper)
            n_in = wrapper.n_out

        self._container = container
        if self.spec.pos_policy == "rope_identity":
            self._declare_prompts_as_registers()

        # Closed formula, not a count of the modules just built (docs/08 §2); with
        # `norm="layernorm"` every bank also carries the two LayerNorm vectors.
        per_bank_extra = 2 * backbone.embed_dim if self.spec.norm == "layernorm" else 0
        expected = sum(
            self._n_tokens_of(i) * backbone.embed_dim + per_bank_extra for i in sorted(prompted)
        )
        report = AdaptationReport(
            family=self.family,
            method=self.spec.variant,
            n_targets=len(self._blocks),
            n_trainable=sum(p.numel() for p in backbone.module().parameters() if p.requires_grad),
            n_trainable_formula=expected,
            # Output sequence growth is decided by the last bank, since every `replace` substitutes.
            extra_tokens=self._blocks[-1].n_out if self._blocks else 0,
            targets=tuple(f"blocks.{i}" for i in sorted(prompted)),
            notes={
                "n_tokens": self.spec.n_tokens,
                "start_block": self.spec.start_block,
                "block_indices": sorted(prompted),
                "n_banchi": len(prompted),
                "pos_policy": self.spec.pos_policy,
                "norm": self.spec.norm,
                "init": self.spec.init,
            },
        )
        return AdaptedBackbone(backbone=backbone, strategy=self, report=report)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """:return: the prompt banks, in block order."""
        for wrapper in self._blocks:
            if wrapper.prompts is not None:
                yield from wrapper.prompts.parameters()

    def can_merge(self) -> bool:
        """:return: always ``False``: prompts lengthen the sequence and no weight can absorb them
        (the trade-off E3-X measures against reparam).
        """
        return False

    def merge_parts(self) -> tuple[str, ...]:
        """:return: always empty: no part of the prompts merges into the weights."""
        return ()

    def merge_(self) -> None:
        raise NotImplementedError(
            "prompts do not merge into the weights: they lengthen the sequence, and the "
            "resulting inference cost is what E3-X compares with reparam. For a "
            "model without overhead, use the other family."
        )

    def unload_merged_(self) -> int:
        """:return: always 0: nothing was merged, so nothing is removed (a no-op, as the contract
        requires).
        """
        return 0

    def unload_(self) -> int:
        """Put the original blocks back and restore the registers.

        :return: number of released blocks.
        """
        if self._container is None:
            return 0
        for i, module in enumerate(self._container):
            if isinstance(module, PromptedBlock):
                self._container[i] = module.block
        for attn, value in self._original_registers:
            attn.n_registers = value
        removed = len(self._blocks)
        self._blocks.clear()
        self._original_registers.clear()
        self._container = None
        return removed

    def extra_tokens(self) -> int:
        """:return: how many tokens longer the sequence leaves a block.

        The bank of the last block, not ``spec.n_tokens`` (banks may differ in width, as in
        DA-VPT); ``spec.n_tokens`` before ``apply()``.
        """
        return self._blocks[-1].n_out if self._blocks else self.spec.n_tokens

    def state_dict_adapter(self) -> dict[str, Tensor]:
        """:return: the prompt banks, keyed ``blocks.{i}.{name}``."""
        state: dict[str, Tensor] = {}
        for wrapper in self._blocks:
            if wrapper.prompts is None:
                continue
            index = self._index_of(wrapper)
            for name, parameter in wrapper.prompts.named_parameters():
                state[f"blocks.{index}.{name}"] = parameter.detach()
        return state

    # --- internals ---------------------------------------------------------
    def _n_tokens_of(self, i: int) -> int:
        """Return the number of prompts in the bank of block ``i``.

        Extension point: ``spec.n_tokens`` here, class-wide banks on DA-VPT proxy blocks.

        :param i: block index.
        :return: the bank width.
        """
        return self.spec.n_tokens

    def _action(self, i: int, prompted: set[int]) -> Literal["insert", "replace", "carry"]:
        """:return: what block ``i`` does with the prompts.

        ``insert`` at the first prompted block only (the only point where the sequence grows),
        ``replace`` at the following banks (deep), ``carry`` elsewhere.
        """
        if i not in prompted:
            return "carry"
        return "insert" if i == min(prompted) else "replace"

    def _bound(self, backbone: VideoBackbone) -> float:
        """Return the initialization bound from the backbone geometry.

        The VPT ``patch_numel`` is ``3 · ∏patch_size``, taken from the tubelet declared by the
        ``PreprocessSpec``; ``init_divider`` narrows it.

        :param backbone: backbone to adapt.
        :return: the bound ``val`` of ``U(-val, +val)``, or the standard deviation with
            ``init="normal"``.
        """
        if self.spec.init == "normal":
            return 0.02 / self.spec.init_divider
        raw = xavier_uniform_bound(backbone.embed_dim, 3 * tubelet_numel(backbone))
        return raw / self.spec.init_divider

    def _declare_prompts_as_registers(self) -> None:
        """Tell every attention how many trailing tokens of its sequence have no position.

        This is where ``pos_policy`` becomes behaviour. The number is per block
        (``wrapper.n_out``) since banks may differ in width.

        :raises ValueError: naming the block that does not expose ``n_registers``.
        """
        for i, wrapper in enumerate(self._blocks):
            found = [m for m in wrapper.block.modules() if hasattr(m, "n_registers")]
            if not found:
                raise ValueError(
                    f"pos_policy='rope_identity' requires the attention of block {i} to "
                    "declare how many trailing tokens have no position (the "
                    "`n_registers` attribute read by RoPEAttention at every forward), and "
                    "this block does not expose it. Without it, prompts would receive a "
                    "spatial position they do not have."
                )
            for attn in found:
                self._original_registers.append((attn, int(attn.n_registers)))
                attn.n_registers = int(attn.n_registers) + wrapper.n_out

    def _index_of(self, wrapper: PromptedBlock) -> int:
        assert self._container is not None
        for i, module in enumerate(self._container):
            if module is wrapper:
                return i
        raise RuntimeError("adapted block no longer present in the backbone container")


PROMPT_VARIANTS.register("vpt_shallow")(PromptPEFT)
PROMPT_VARIANTS.register("vpt_deep")(PromptPEFT)


def validate_pos_policy(layout: TokenLayout, policy: PosPolicy) -> None:
    """Check that the policy makes sense for the backbone positional encoding.

    The correct default lives in the backbone config, not in the adapter one (ADR-004);
    only the consistency of the pair is checked here (e.g. ``learned_abs`` on 3D-RoPE is
    rejected).

    :param layout: token layout declared by the backbone.
    :param policy: policy requested by the spec.
    :raises ValueError: naming the policies allowed for that kind of position.
    """
    allowed = ALLOWED_POLICIES.get(layout.positional)
    if allowed is None:
        raise ValueError(f"unknown positional encoding: {layout.positional!r}")
    if policy not in allowed:
        raise ValueError(
            f"pos_policy={policy!r} is meaningless on a backbone with positions "
            f"{layout.positional!r}: allowed {list(allowed)}. "
            "On 3D-RoPE the only one is 'rope_identity', since a prompt has no "
            "spatio-temporal position to rotate; without a policy V-JEPA 2.1 would "
            "fail with a shape error, because the position vector covers the patch "
            "tokens only."
        )


def tubelet_numel(backbone: VideoBackbone) -> int:
    """Return the spatio-temporal elements of a patch, from the spec ``tokenizer``.

    :return: product of the tubelet dimensions (e.g. 512 for ``tubelet_2x16x16``).
    :raises ValueError: if the tokenizer name does not contain a readable geometry.
    """
    name = backbone.preprocess_spec().tokenizer
    numbers = [int(piece) for piece in name.replace("_", "x").split("x") if piece.isdigit()]
    if not numbers:
        raise ValueError(
            f"cannot read the patch geometry from tokenizer {name!r}; it is needed "
            "by the VPT initialization (val = √(6/(3·∏patch + D))). Use init='normal' "
            "or declare a tokenizer with the dimensions in its name."
        )
    return math.prod(numbers)


def prompt_overhead(n_patch_tokens: int, spec: PromptSpec) -> dict[str, float]:
    """Return the inference cost of lengthening the sequence, in declared form.

    Attention is quadratic in the sequence length, so *n* prompts on *N* patches cost about
    ``(1 + n/N)² − 1``. An analytical estimate to put next to the profiler measurement.

    :param n_patch_tokens: patch tokens of the geometry in use.
    :param spec: prompt configuration.
    :return: ``{"frazione_token", "fattore_attenzione", "fattore_mlp"}``.
    :raises ValueError: if ``n_patch_tokens`` is not positive.
    """
    if n_patch_tokens <= 0:
        raise ValueError(f"n_patch_tokens must be > 0, not {n_patch_tokens}")
    report = spec.n_tokens / n_patch_tokens
    return {
        "frazione_token": report,
        "fattore_attenzione": (1.0 + report) ** 2,
        "fattore_mlp": 1.0 + report,
    }
