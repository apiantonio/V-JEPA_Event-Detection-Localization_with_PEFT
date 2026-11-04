"""APT: prompts entering attention as keys and values, never as queries.

Replicates the official ``wgcban/apt`` (Bandara & Patel, *Attention Prompt Tuning*,
FG 2024, ``modeling_VPT.py``):

* banks are ``[n_layer, n_tokens, head_dim]``, shared across heads (on ViT-L an APT
  bank costs 1/16 of a VPT bank with the same ``n_tokens``);
* inside attention ``k = cat(k, p_k.expand(B, num_heads, -1, -1))``, same for ``v``;
  ``q`` is untouched;
* uniform initialization with the VPT Xavier bound using ``head_dim``:
  ``val = √(6/(3·∏patch + head_dim))``;
* optional ``prompt_reparam``: a per-token scale ``[n_layer, 2, n_tokens]`` for keys
  and values, applied before concatenation;
* ``prompt_start``/``prompt_end`` select the block range; dropout on prompts only.

On V-JEPA 2.1 the attention input sequence does not grow (only ``k`` and ``v`` do), so
positions, ``n_registers`` and the grid are untouched; the injection happens inside
``RoPEAttention.forward``, copied to ``evd/vendor/vjepa21_rope.py`` because
``../vjepa2`` is not modified (invariant 11). Attention costs ``N·(N+n)`` instead of
``(N+n)²`` and the MLP does not see the prompts (:func:`apt_overhead`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch import Tensor

from evd.models.adapt.base import (
    AdaptationReport,
    AdaptedBackbone,
    Family,
)
from evd.models.adapt.prompt import (
    PROMPT_VARIANTS,
    PromptSpec,
    tubelet_numel,
    validate_pos_policy,
    xavier_uniform_bound,
)
from evd.models.registry import Registry

if TYPE_CHECKING:
    from evd.models.backbones.base import VideoBackbone

#: How to run an attention with key/value prompts, per attention class: adding a
#: backbone means registering a forward, not adding an ``if`` to ``apply()``. The
#: signature is the original forward plus the keyword-only ``p_k`` and ``p_v``.
KV_PROMPT_ATTENTION: Registry[Callable[..., Any]] = Registry("K/V prompt attention")


def _register_vjepa21() -> None:
    """Register the vendored ``RoPEAttention`` forward.

    Kept here, not in the vendored file, so ``evd/vendor/`` stays a plain copy of upstream
    code that does not know ``evd.models``.
    """
    from evd.vendor.vjepa21_rope import rope_attention_forward

    KV_PROMPT_ATTENTION.register("RoPEAttention")(rope_attention_forward)


_register_vjepa21()


class KVPromptTokens(nn.Module):
    """The two banks of an APT block: ``p_k`` and ``p_v``, ``head_dim`` wide.

    Two separate parameters, as in the official source (``prompt_embeddings_k`` /
    ``prompt_embeddings_v``), read separately in the adapter ``state_dict``.

    :param n_tokens: number of prompts.
    :param head_dim: width of one head; banks are shared across heads.
    :param bound: bound of the uniform initialization.
    :param reparam: add the per-token ``prompt_reparam`` scale, initialized to 1.
    :param dropout: dropout on prompts only.
    :param generator: explicit random source, independent of the global RNG.
    """

    def __init__(
        self,
        *,
        n_tokens: int,
        head_dim: int,
        bound: float,
        reparam: bool = False,
        dropout: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        self.prompt_k = nn.Parameter(
            torch.empty(n_tokens, head_dim).uniform_(-bound, bound, generator=generator)
        )
        self.prompt_v = nn.Parameter(
            torch.empty(n_tokens, head_dim).uniform_(-bound, bound, generator=generator)
        )
        self.scale = nn.Parameter(torch.ones(2, n_tokens)) if reparam else None
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def n_tokens(self) -> int:
        return int(self.prompt_k.shape[0])

    def kv(self) -> tuple[Tensor, Tensor]:
        """Return the two banks ready for concatenation.

        :return: ``(p_k, p_v)``, each ``[n_tokens, head_dim]``, with dropout and the optional
            per-token scale applied.
        """
        p_k, p_v = self.drop(self.prompt_k), self.drop(self.prompt_v)
        if self.scale is not None:
            p_k = p_k * self.scale[0].unsqueeze(-1)
            p_v = p_v * self.scale[1].unsqueeze(-1)
        return p_k, p_v


class KVPromptedAttention(nn.Module):
    """A frozen attention that sees ``n_tokens`` extra keys and values.

    Replaces the block's ``attn`` submodule keeping its signature, so ``mask``, ``T``,
    ``H_patches``, ``W_patches`` and ``return_attn`` pass through unchanged.

    :param attn: attention to wrap; it stays frozen and unmodified.
    :param prompts: the two banks of this block.
    :raises KeyError: if no forward is registered in :data:`KV_PROMPT_ATTENTION` for the
        class of ``attn``.
    """

    def __init__(self, attn: nn.Module, prompts: KVPromptTokens) -> None:
        super().__init__()
        self.forward_impl = KV_PROMPT_ATTENTION.get(type(attn).__name__)
        self.attn = attn
        self.prompts = prompts
        # An adapter lives where the adapted module lives: banks are created on CPU while the
        # attention may already be on GPU.
        reference = next(attn.parameters(), None)
        if reference is not None:
            prompts.to(device=reference.device, dtype=reference.dtype)

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Any:
        """Run the attention with the prompts grafted onto keys and values.

        :param x: input sequence ``[B, N, D]``; the output has the same length.
        :return: what the registered forward of this attention returns.
        """
        p_k, p_v = self.prompts.kv()
        return self.forward_impl(self.attn, x, *args, p_k=p_k, p_v=p_v, **kwargs)


class AptPEFT:
    """Install APT banks on the attention of the selected blocks.

    :param spec: prompt configuration; uses ``n_tokens``, ``start_block``, ``depth``,
        ``dropout``, ``apt_reparam``, ``seed``.
    """

    family: Family = "prompt"

    def __init__(self, spec: PromptSpec) -> None:
        if spec.variant != "apt":
            raise ValueError(f"AptPEFT requires variant='apt', not {spec.variant!r}")
        self.spec = spec
        self._wrapped: dict[int, KVPromptedAttention] = {}
        self._container: nn.ModuleList | None = None

    # --- AdaptationStrategy contract ---------------------------------------
    def apply(self, backbone: VideoBackbone, /) -> AdaptedBackbone:
        """Freeze the backbone and wrap the attention of the prompted blocks.

        :param backbone: backbone to adapt, modified in place.
        :return: the receipt with counts and touched blocks.
        :raises RuntimeError: if the strategy was already applied.
        :raises ValueError: if the backbone blocks do not expose ``attn``, or the
            ``pos_policy`` is incompatible with its positional encoding.
        """
        if self._wrapped:
            raise RuntimeError("this AptPEFT was already applied")
        # The policy is validated here too, although APT does not touch `n_registers`.
        validate_pos_policy(backbone.token_layout(), self.spec.pos_policy)
        backbone.freeze()
        container = backbone.blocks()
        indices = self.spec.prompted_blocks(len(container))

        head_dim = self._head_dim(backbone, container)
        bound = xavier_uniform_bound(head_dim, 3 * tubelet_numel(backbone))
        generator = torch.Generator().manual_seed(self.spec.seed)
        for i in sorted(indices):
            block = container[i]
            attn = getattr(block, "attn", None)
            if attn is None:
                raise ValueError(
                    f"block {i} does not expose an `attn` submodule: APT injects inside "
                    "attention, so it needs to know which module it is"
                )
            wrapper = KVPromptedAttention(
                attn,
                KVPromptTokens(
                    n_tokens=self.spec.n_tokens,
                    head_dim=head_dim,
                    bound=bound,
                    reparam=self.spec.apt_reparam,
                    dropout=self.spec.dropout,
                    generator=generator,
                ),
            )
            block.attn = wrapper
            self._wrapped[i] = wrapper

        self._container = container
        per_bank = 2 * self.spec.n_tokens * head_dim
        if self.spec.apt_reparam:
            per_bank += 2 * self.spec.n_tokens
        report = AdaptationReport(
            family=self.family,
            method="apt",
            n_targets=len(self._wrapped),
            n_trainable=sum(p.numel() for p in backbone.module().parameters() if p.requires_grad),
            n_trainable_formula=len(self._wrapped) * per_bank,
            # Zero, not `n_tokens`: the output sequence does not grow (the cost difference between
            # APT and VPT).
            extra_tokens=0,
            targets=tuple(f"blocks.{i}.attn" for i in sorted(self._wrapped)),
            notes={
                "n_tokens": self.spec.n_tokens,
                "head_dim": head_dim,
                "reparam": self.spec.apt_reparam,
                "kv_extra_tokens": self.spec.n_tokens,
                "pos_policy": self.spec.pos_policy,
            },
        )
        return AdaptedBackbone(backbone=backbone, strategy=self, report=report)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """:return: the K/V banks, in block order."""
        for i in sorted(self._wrapped):
            yield from self._wrapped[i].prompts.parameters()

    def can_merge(self) -> bool:
        """:return: always ``False``: K/V prompts lengthen keys and values at every forward."""
        return False

    def merge_parts(self) -> tuple[str, ...]:
        """:return: always empty: no part of the K/V prompts merges into the weights."""
        return ()

    def merge_(self) -> None:
        raise NotImplementedError(
            "APT K/V prompts do not merge into the weights: they lengthen keys and values at "
            "every forward. The resulting cost (linear in the prompts, not quadratic "
            "as in VPT) is what E3-P compares."
        )

    def unload_merged_(self) -> int:
        """:return: always 0: nothing was merged, so nothing is removed."""
        return 0

    def unload_(self) -> int:
        """Put the original attention modules back.

        :return: number of released attention modules.
        """
        if self._container is None:
            return 0
        for i, wrapper in self._wrapped.items():
            self._container[i].attn = wrapper.attn
        removed = len(self._wrapped)
        self._wrapped.clear()
        self._container = None
        return removed

    def extra_tokens(self) -> int:
        """:return: 0: the output sequence is as long as the input one.

        The APT cost is not zero: it lives in keys and values and is reported by
        ``notes["kv_extra_tokens"]`` and :func:`apt_overhead`.
        """
        return 0

    def state_dict_adapter(self) -> dict[str, Tensor]:
        """:return: the banks, keyed ``blocks.{i}.attn.{prompt_k|prompt_v|scale}``."""
        state: dict[str, Tensor] = {}
        for i in sorted(self._wrapped):
            for name, parameter in self._wrapped[i].prompts.named_parameters():
                state[f"blocks.{i}.attn.{name}"] = parameter.detach()
        return state

    # --- internals ---------------------------------------------------------
    def _head_dim(self, backbone: VideoBackbone, container: nn.ModuleList) -> int:
        """Return the width of one head, read from the attention module.

        :raises ValueError: if the attention does not declare ``num_heads`` or ``embed_dim``
            is not a multiple of it.
        """
        attn = getattr(container[0], "attn", None)
        num_heads = getattr(attn, "num_heads", None)
        if num_heads is None:
            raise ValueError(
                "APT needs the number of attention heads: its banks are "
                "`head_dim` wide and shared across heads, not `embed_dim` wide"
            )
        if backbone.embed_dim % int(num_heads):
            raise ValueError(
                f"embed_dim={backbone.embed_dim} not divisible by num_heads={num_heads}"
            )
        return backbone.embed_dim // int(num_heads)


PROMPT_VARIANTS.register("apt")(AptPEFT)


def apt_overhead(n_patch_tokens: int, spec: PromptSpec) -> dict[str, float]:
    """Return the cost of lengthening keys and values, in declared form.

    With *n* prompts on *N* patches, VPT attention costs ``(1 + n/N)²`` (queries and keys
    grow) while APT costs ``1 + n/N`` (only keys and values), and the APT MLP pays nothing.

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
        "fattore_attenzione": 1.0 + report,
        "fattore_mlp": 1.0,
    }
