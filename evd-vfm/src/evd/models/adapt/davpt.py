"""DA-VPT: prompts acting as class proxies in a metric-learning loss.

Replicates the official ``Noahsark/DA-VPT`` (Ren, Chen, Wang, Hua, *DA-VPT:
Semantic-Guided Visual Prompt Tuning for Vision Transformers*, CVPR 2025,
``models/vpt.py`` and ``models/criterion.py``):

* deep VPT prompts in two groups: proxy banks (blocks ``[proxy_start, proxy_end]``)
  enter the loss, the others are plain VPT;
* ``vpt_norm`` normalizes the bank before insertion (``none`` | L2 | LayerNorm) and
  ``init_divide_by8`` narrows the initialization;
* the loss is Proxy-Anchor (Kim et al. 2020), or Proxy-NCA by changing the
  ``logsumexp`` axis, between batch features and prompts::

      sims  = normalize(feature) @ (normalize(proxy) · scale)ᵀ
      pos   = LSE_{anchors}  ( −α_pos·(sims − δ_pos) )   on the class proxies
      neg   = LSE_{anchors}  ( +α_neg·(sims − δ_neg) )   on the others
      loss  = mean(pos) + mean(neg)

  where ``masked_logsumexp`` prepends a zero (the "+1" inside the Proxy-Anchor log).
  Repository defaults: ``α = 32``, ``δ_pos = 0.1``, ``δ_neg = −0.1``, ``scale = 1``;
* the patch anchor is the mean of the block patch tokens, the cls anchor the CLS
  token; the total is ``w_vpt·patch + w_cls·cls``.

Three explicit choices for our backbone:

1. No CLS term: V-JEPA 2.1 has no CLS token, so ``cls_weight`` stays 0 and
   ``apply()`` raises if it is set on a backbone without CLS.
2. Similarities in token space, not in query/key space (on a fused QKV with 3D-RoPE
   proxies would be unrotated and patches rotated); this is also what the released
   ``ml_patch_forward`` actually computes.
3. One proxy per class (``initial_mapping='all_classes'``), so proxy banks are
   ``n_classes`` wide.

Using this module in an experiment adds a loss term and is a protocol decision; the
defaults are the paper's.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from evd.models.adapt.base import AdaptedBackbone
from evd.models.adapt.prompt import (
    PROMPT_VARIANTS,
    PromptedBlock,
    PromptPEFT,
    PromptSpec,
    PromptTokens,
)

if TYPE_CHECKING:
    from evd.models.backbones.base import VideoBackbone

ProxyCriterion = Literal["proxyanchor", "proxynca"]


@dataclass(frozen=True)
class ProxyLossSpec:
    """Hyper-parameters of the DA-VPT metric-learning loss.

    Defaults come from ``Noahsark/DA-VPT`` (``params.py``), i.e. from Proxy-Anchor.

    :param criterion: ``"proxyanchor"`` reduces the ``logsumexp`` over anchors,
        ``"proxynca"`` over proxies.
    :param n_classes: number of classes, i.e. proxies per bank; ``0`` means undecided and
        makes ``apply()`` raise.
    :param pos_alpha: slope of the positive term.
    :param neg_alpha: slope of the negative term.
    :param pos_delta: margin of the positive term.
    :param neg_delta: margin of the negative term (negative in the paper).
    :param scale: factor multiplying the normalized proxies (``loss_vpt_scale_ratio``).
    :param patch_weight: weight of the patch term in the total loss.
    :param cls_weight: weight of the CLS term; 0 on V-JEPA 2.1.
    :param proxy_blocks: number of blocks, counted from the end, with proxy banks (one by
        default, the last).
    """

    criterion: ProxyCriterion = "proxyanchor"
    n_classes: int = 0
    pos_alpha: float = 32.0
    neg_alpha: float = 32.0
    pos_delta: float = 0.1
    neg_delta: float = -0.1
    scale: float = 1.0
    patch_weight: float = 1.0
    cls_weight: float = 0.0
    proxy_blocks: int = 1

    def __post_init__(self) -> None:
        if self.criterion not in ("proxyanchor", "proxynca"):
            raise ValueError(f"criterion={self.criterion!r}: ammessi 'proxyanchor' | 'proxynca'")
        if self.n_classes < 0:
            raise ValueError(f"n_classes must be >= 0, not {self.n_classes}")
        if self.proxy_blocks <= 0:
            raise ValueError(f"proxy_blocks must be > 0, not {self.proxy_blocks}")
        if self.scale <= 0:
            raise ValueError(f"scale must be > 0, not {self.scale}")

    @property
    def proxy_dim(self) -> int:
        """:return: the ``logsumexp`` axis on the ``[anchors, proxies]`` matrix: 0 for
        Proxy-Anchor, 1 for Proxy-NCA.
        """
        return 1 if self.criterion == "proxynca" else 0


def masked_logsumexp(sims: Tensor, *, mask: Tensor, dim: int) -> Tensor:
    """Proxy-Anchor ``logsumexp`` with mask and prepended zero.

    As in the official source, excluded positions become the minimum representable value
    instead of ``-inf`` (no ``nan`` in bf16) and a zero vector is concatenated before the
    ``logsumexp``; rows with an empty mask are zeroed.

    :param sims: weighted similarities, shape ``[anchors, proxies]``.
    :param mask: boolean mask of the pairs that count, same shape.
    :param dim: reduction axis, ``0`` for Proxy-Anchor and ``1`` for NCA.
    :return: the reduced tensor, with ``keepdim=True``.
    """
    sims = sims.masked_fill(~mask, torch.finfo(sims.dtype).min)
    dims = list(sims.shape)
    dims[dim] = 1
    zeros = torch.zeros(dims, dtype=sims.dtype, device=sims.device)
    reduced = torch.logsumexp(torch.cat([sims, zeros], dim=dim), dim=dim, keepdim=True)
    return reduced.masked_fill(~torch.any(mask, dim=dim, keepdim=True), 0.0)


class ProxyPromptLoss(nn.Module):
    """The DA-VPT loss: prompts as proxies, features as anchors.

    It has no parameters (the proxies are the adapter prompts); it is an ``nn.Module`` to
    follow the official source.

    :param spec: hyper-parameters.
    """

    def __init__(self, spec: ProxyLossSpec) -> None:
        super().__init__()
        self.spec = spec

    def forward(self, features: Tensor, proxies: Tensor, labels: Tensor) -> Tensor:
        """Compute the metric-learning term for a set of anchors and proxies.

        :param features: anchors ``[B, D]`` (the mean patch tokens of a block).
        :param proxies: proxy bank prompts ``[P, D]``; proxy ``c`` represents class ``c``.
        :param labels: class of each anchor, ``[B]``, values in ``[0, P)``.
        :return: a scalar.
        :raises ValueError: on inconsistent shapes or a label without a proxy.
        """
        if features.dim() != 2 or proxies.dim() != 2:
            raise ValueError(
                f"expected anchors [B, D] and proxies [P, D], got {tuple(features.shape)} "
                f"and {tuple(proxies.shape)}"
            )
        if features.shape[1] != proxies.shape[1]:
            raise ValueError(
                f"anchors and proxies have different sizes: {features.shape[1]} vs "
                f"{proxies.shape[1]}"
            )
        n_proxy = proxies.shape[0]
        label_values = labels.view(-1).long()
        if label_values.numel() != features.shape[0]:
            raise ValueError(f"{label_values.numel()} labels for {features.shape[0]} anchors")
        if int(label_values.max()) >= n_proxy or int(label_values.min()) < 0:
            raise ValueError(
                f"labels outside the {n_proxy} available proxies: "
                f"[{int(label_values.min())}, {int(label_values.max())}]. "
                "Every class needs its proxy (initial_mapping='all_classes')."
            )
        # Similarities are cosines: both sides normalized, proxies rescaled after normalization
        # as in the official source.
        anchors = F.normalize(features.float(), dim=-1)
        p = F.normalize(proxies.float(), dim=-1) * self.spec.scale
        sims = anchors @ p.T  # [B, P]

        proxy_class = torch.arange(n_proxy, device=sims.device)
        same_value = proxy_class.unsqueeze(0) == label_values.unsqueeze(1)  # [B, P]
        pos = masked_logsumexp(
            -self.spec.pos_alpha * (sims - self.spec.pos_delta),
            mask=same_value,
            dim=self.spec.proxy_dim,
        )
        neg = masked_logsumexp(
            self.spec.neg_alpha * (sims - self.spec.neg_delta),
            mask=~same_value,
            dim=self.spec.proxy_dim,
        )
        return pos.mean() + neg.mean()


class DaVptPEFT(PromptPEFT):
    """Deep VPT whose last banks are class proxies in a loss.

    Extends :class:`PromptPEFT` because the insertion mechanism is the same, which keeps
    DA-VPT comparable with VPT. It adds class-wide proxy banks and a loss term collected by
    forward hooks, and implements :class:`~evd.models.adapt.base.AuxiliaryLoss`.

    :param spec: prompt configuration; ``variant`` must be ``"da_vpt"``.
    :param loss: loss hyper-parameters; defaults to the paper.
    :raises ValueError: if the variant is not ``"da_vpt"``.
    """

    def __init__(self, spec: PromptSpec, loss: ProxyLossSpec | None = None) -> None:
        if spec.variant != "da_vpt":
            raise ValueError(f"DaVptPEFT requires variant='da_vpt', not {spec.variant!r}")
        super().__init__(spec)
        self.loss_spec = loss or ProxyLossSpec()
        self.criterion = ProxyPromptLoss(self.loss_spec)
        self._proxy_blocks: set[int] = set()
        self._handles: list[RemovableHandle] = []
        self._sites: list[tuple[Tensor, PromptTokens]] = []

    # --- AdaptationStrategy contract ---------------------------------------
    def apply(self, backbone: VideoBackbone, /) -> AdaptedBackbone:
        """Install the prompts and hook the loss collection.

        :param backbone: backbone to adapt, modified in place.
        :return: the receipt, noting which blocks are proxies.
        :raises ValueError: if ``n_classes`` is 0, or the CLS term is requested on a backbone
            without CLS.
        """
        if self.loss_spec.n_classes <= 0:
            raise ValueError(
                "DA-VPT needs the number of classes: proxy banks are "
                "`n_classes` wide, one proxy per class. There is no sensible default: "
                "the head knows it, and it must be passed in ProxyLossSpec."
            )
        if self.loss_spec.cls_weight and backbone.token_layout().n_cls == 0:
            raise ValueError(
                f"cls_weight={self.loss_spec.cls_weight} on a backbone without CLS token: "
                "the DA-VPT CLS term has no anchor. On V-JEPA 2.1 it stays 0 "
                "(see the adapt/davpt.py docstring)."
            )
        # Proxy blocks are the last prompted ones, as in the official source: class semantics
        # lives at the end of the network.
        with_prompts = sorted(self.spec.prompted_blocks(len(backbone.blocks())))
        self._proxy_blocks = set(with_prompts[-self.loss_spec.proxy_blocks :])

        adapted = super().apply(backbone)
        for i in sorted(self._proxy_blocks):
            wrapper = self._blocks[i]
            self._handles.append(wrapper.register_forward_pre_hook(self._collect))
        note = dict(adapted.report.notes)
        note.update(
            {
                "proxy_blocks": sorted(self._proxy_blocks),
                "n_proxy": self.loss_spec.n_classes,
                "criterion": self.loss_spec.criterion,
                "patch_weight": self.loss_spec.patch_weight,
                "cls_weight": self.loss_spec.cls_weight,
            }
        )
        return AdaptedBackbone(
            backbone=adapted.backbone,
            strategy=self,
            report=replace(adapted.report, notes=note),
        )

    def unload_(self) -> int:
        """Remove the hooks, then unload the prompts as ``PromptPEFT`` does.

        :return: number of released blocks.
        """
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._sites.clear()
        self._proxy_blocks.clear()
        return super().unload_()

    # --- AuxiliaryLoss contract --------------------------------------------
    def reset_auxiliary(self) -> None:
        """Discard the sites collected by the previous forward."""
        self._sites.clear()

    def auxiliary_loss(self, labels: Tensor) -> Tensor:
        """Return the sum of the metric-learning terms over the proxy blocks.

        :param labels: class of each batch element, ``[B]``.
        :return: a scalar, already weighted by ``patch_weight``.
        :raises RuntimeError: if no forward happened after the last reset.
        """
        if not self._sites:
            raise RuntimeError(
                "auxiliary_loss() without a preceding forward: the proxy blocks collected "
                "nothing. The order is reset_auxiliary() -> forward -> auxiliary_loss()."
            )
        terms = [
            self.criterion(patch.mean(dim=1), bank.tokens(), labels) for patch, bank in self._sites
        ]
        return self.loss_spec.patch_weight * torch.stack(terms).sum()

    # --- internals ---------------------------------------------------------
    def _n_tokens_of(self, i: int) -> int:
        """:return: ``n_classes`` on proxy blocks, ``spec.n_tokens`` on the others."""
        return self.loss_spec.n_classes if i in self._proxy_blocks else self.spec.n_tokens

    def _collect(self, module: nn.Module, args: tuple[Any, ...]) -> None:
        """Forward pre-hook: store the patch tokens and the bank of the block.

        A pre-hook because the anchors are the block inputs; the prompts at the end of ``x``
        belong to the previous block and are removed.
        """
        assert isinstance(module, PromptedBlock)
        assert module.prompts is not None
        self._sites.append((module.patch_tokens(args[0]), module.prompts))


PROMPT_VARIANTS.register("da_vpt")(DaVptPEFT)
