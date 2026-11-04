"""Multiclass heads with one query per class, for the class-routing localization campaign.

These heads choose the
class of a chunk (13 anomaly classes plus ``Normal``) so that the localization can apply the
dictionary of that class; detection stays binary.

- :class:`ClassQueryHead` (B2): normalization, ``depth - 1`` self-attention blocks and one
  cross-attention block with one learned query per class. The class logit reads only its own
  query: the weights are those of one ``Linear(embed_dim, n_classes)``, row ``c`` applied to
  query ``c`` (the per-class projection of query decoders such as Query2Label).
- :class:`ClassQueryBranchHead` (B1): a frozen root (normalization and the first blocks of a
  trained binary head, loaded by the runner's ``inherit`` block) and a new branch on top.
- :class:`BinaryAndClassHead` (B3): a binary :class:`DetectionHead` and a
  :class:`ClassQueryHead` on the same tokens, trained together with a loss balance (sum or
  GradNorm). ``forward`` returns the binary logit only, so evaluation, thresholds and
  Integrated Gradients treat the model as the binary detector it is.

The attentive blocks are those of :mod:`evd.models.heads.pooling` (Meta ``AttentivePooler``);
queries do not interact in the cross-attention block (``pooling.py``, ``_CrossAttentionBlock``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from evd.models.heads.base import HEADS
from evd.models.heads.detection import DetectionHead
from evd.models.heads.pooling import AttentivePooling, _Block
from evd.types import BackboneOutput

#: Loss balances of :class:`BinaryAndClassHead`.
BALANCES = ("sum", "gradnorm")


def _per_class_logits(pooled: Tensor, linear: nn.Linear) -> Tensor:
    """:return: ``[B, C]`` logits, row ``c`` of ``linear`` applied to query ``c`` of ``pooled``."""
    return (pooled * linear.weight.unsqueeze(0)).sum(-1) + linear.bias


@HEADS.register("class_queries")
class ClassQueryHead(nn.Module):
    """Attentive head with one learned query per class and a per-class projection.

    :param embed_dim: token dimension.
    :param n_classes: output classes, ``Normal`` included (one query each).
    :param n_heads: attention heads.
    :param depth: attentive blocks, the last being the cross-attention one.
    :param mlp_ratio: MLP width ratio of the blocks.
    :param norm: apply a ``LayerNorm`` to the tokens first, as :class:`DetectionHead`.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        n_classes: int,
        n_heads: int = 16,
        depth: int = 4,
        mlp_ratio: float = 4.0,
        norm: bool = True,
    ) -> None:
        super().__init__()
        if n_classes < 2:
            raise ValueError(f"A class-query head needs at least two classes, got {n_classes}")
        self.n_classes = n_classes
        self.norm = nn.LayerNorm(embed_dim) if norm else nn.Identity()
        self.pooler = AttentivePooling(
            num_queries=n_classes,
            embed_dim=embed_dim,
            num_heads=n_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
        )
        self.linear = nn.Linear(embed_dim, n_classes)

    def forward(self, feats: BackboneOutput) -> Tensor:
        """:return: ``[B, n_classes]`` logits."""
        return _per_class_logits(self.pooler(self.norm(feats.tokens)), self.linear)


@HEADS.register("class_queries_branch")
class ClassQueryBranchHead(nn.Module):
    """Class-query branch on top of a frozen root taken from a trained binary head.

    The root mirrors the first ``root_blocks`` self-attention blocks of a
    :class:`DetectionHead` (``root_norm`` is its ``norm``, ``root_blocks.i`` its
    ``pooler.blocks.i``); the runner copies and freezes them from the detector checkpoint
    (``inherit.head`` with a prefix map). The branch is a new :class:`AttentivePooling` of
    ``branch_depth`` blocks with one query per class.

    :param embed_dim: token dimension.
    :param n_classes: output classes, ``Normal`` included.
    :param n_heads: attention heads (must match the binary head for the root).
    :param root_blocks: self-attention blocks of the root.
    :param branch_depth: blocks of the branch, the last being the cross-attention one.
    :param mlp_ratio: MLP width ratio of every block.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        n_classes: int,
        n_heads: int = 16,
        root_blocks: int = 2,
        branch_depth: int = 2,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if n_classes < 2 or root_blocks < 1:
            raise ValueError("The branch needs at least two classes and one root block")
        self.n_classes = n_classes
        self.root_norm = nn.LayerNorm(embed_dim)
        self.root_blocks = nn.ModuleList(
            [_Block(embed_dim, n_heads, mlp_ratio, True) for _ in range(root_blocks)]
        )
        self.branch = AttentivePooling(
            num_queries=n_classes,
            embed_dim=embed_dim,
            num_heads=n_heads,
            mlp_ratio=mlp_ratio,
            depth=branch_depth,
        )
        self.linear = nn.Linear(embed_dim, n_classes)

    def root(self, feats: BackboneOutput) -> Tensor:
        """:return: the root tokens, ``[B, N, D]``."""
        x = self.root_norm(feats.tokens)
        for block in self.root_blocks:
            x = block(x)
        return x

    def forward(self, feats: BackboneOutput) -> Tensor:
        """:return: ``[B, n_classes]`` logits."""
        return _per_class_logits(self.branch(self.root(feats)), self.linear)


class GradNormBalance(nn.Module):
    """GradNorm task weights (Chen et al., ICML 2018, arXiv:1711.02257, Algorithm 1).

    At every micro-batch the gradient norm of each weighted task loss with respect to the
    shared representation is pulled towards the mean norm times the task's relative inverse
    training rate raised to ``alpha``. The weights are parameters of the head, updated by the
    run optimizer from the gradient written here, and renormalized to sum to the number of
    tasks after every optimizer step (:meth:`renormalize`); the initial losses are a buffer,
    so a resumed run continues the same balance. The norms are taken on the encoder output
    tokens rather than on encoder weights (plan §5.4): with VPT-shallow the only shared
    trainable weights are the input prompts, and their per-task norm would cost two extra
    full backward passes.

    :param tasks: task names, in a fixed order.
    :param alpha: asymmetry strength (1.5, the NYUv2 optimum of the paper).
    """

    def __init__(self, tasks: tuple[str, ...], alpha: float) -> None:
        super().__init__()
        if alpha < 0:
            raise ValueError(f"GradNorm alpha must be non-negative, got {alpha}")
        self.tasks, self.alpha = tasks, float(alpha)
        self.weights = nn.Parameter(torch.ones(len(tasks)))
        self.register_buffer("initial", torch.zeros(len(tasks)))

    def combine(self, losses: dict[str, Tensor], shared: Tensor, scale: float = 1.0) -> Tensor:
        """Return the weighted loss and add the GradNorm gradient to ``weights.grad``.

        :param losses: scalar loss per task, still attached to the graph.
        :param shared: the representation both heads read, attached to the graph.
        :param scale: factor of the weight gradient (``1 / grad_accum``).
        :return: ``sum_i w_i L_i`` with the weights detached, to back-propagate as usual.
        """
        if not shared.requires_grad:
            raise ValueError(
                "GradNorm needs a shared representation attached to the graph: with a frozen "
                "encoder there is nothing shared to balance (use balance: sum)"
            )
        values = torch.stack([losses[t] for t in self.tasks]).float()
        if not bool(self.initial.any()):
            self.initial.copy_(values.detach())
        norms = torch.stack(
            [
                torch.autograd.grad(losses[t], shared, retain_graph=True)[0].float().norm()
                for t in self.tasks
            ]
        )
        w = self.weights.detach()
        scaled = w * norms
        ratios = values.detach() / self.initial.clamp_min(1e-12)
        target = scaled.mean() * (ratios / ratios.mean()) ** self.alpha
        gradient = torch.sign(scaled - target) * norms * scale
        if self.weights.grad is None:
            self.weights.grad = gradient.to(self.weights.dtype)
        else:
            self.weights.grad += gradient.to(self.weights.dtype)
        return (w * values).sum()

    @torch.no_grad()
    def renormalize(self) -> None:
        """Keep the weights positive and summing to the number of tasks (in place)."""
        self.weights.clamp_(min=1e-4)
        self.weights.mul_(len(self.tasks) / self.weights.sum())


@HEADS.register("binary_and_classes")
class BinaryAndClassHead(nn.Module):
    """A binary detection head and a class-query head trained together (B3).

    ``forward`` returns the binary logit, so everything downstream of training (evaluation,
    threshold, Integrated Gradients) sees the binary detector; :meth:`forward_tasks` returns
    both outputs for the multitask objective of the ``Trainer``.

    :param embed_dim: token dimension.
    :param n_classes: classes of the class head, ``Normal`` included.
    :param n_heads: attention heads of both heads.
    :param depth: attentive blocks of each head.
    :param balance: ``sum`` (weights 1) or ``gradnorm``.
    :param gradnorm_alpha: GradNorm asymmetry, read only with ``balance: gradnorm``.
    """

    #: Task names, in the order of the GradNorm weights.
    TASKS = ("binary", "classes")

    def __init__(
        self,
        *,
        embed_dim: int,
        n_classes: int,
        n_heads: int = 16,
        depth: int = 4,
        balance: str = "sum",
        gradnorm_alpha: float | None = None,
    ) -> None:
        super().__init__()
        if balance not in BALANCES:
            raise ValueError(f"Unknown loss balance {balance!r}; known: {BALANCES}")
        if (balance == "gradnorm") != (gradnorm_alpha is not None):
            raise ValueError("gradnorm_alpha is required with balance: gradnorm, and only there")
        # One logit: the Head contract describes what `forward` returns.
        self.n_classes = 1
        self.n_task_classes = n_classes
        self.balance = balance
        self.binary = DetectionHead(embed_dim=embed_dim, n_classes=1, n_heads=n_heads, depth=depth)
        self.classes = ClassQueryHead(
            embed_dim=embed_dim, n_classes=n_classes, n_heads=n_heads, depth=depth
        )
        self.balancer = (
            GradNormBalance(self.TASKS, gradnorm_alpha) if balance == "gradnorm" else None
        )

    def forward(self, feats: BackboneOutput) -> Tensor:
        """:return: ``[B, 1]`` binary logits."""
        return self.binary(feats)

    def forward_tasks(self, feats: BackboneOutput) -> dict[str, Tensor]:
        """:return: ``binary`` ``[B, 1]`` and ``classes`` ``[B, n_task_classes]`` logits."""
        return {"binary": self.binary(feats), "classes": self.classes(feats)}

    def combine(self, losses: dict[str, Tensor], shared: Tensor, scale: float = 1.0) -> Tensor:
        """:return: the training loss of the two tasks (sum, or GradNorm-weighted sum)."""
        if self.balancer is None:
            return losses["binary"] + losses["classes"]
        return self.balancer.combine(losses, shared, scale)

    def after_step(self) -> None:
        """Renormalize the GradNorm weights after the optimizer step (no-op for the sum)."""
        if self.balancer is not None:
            self.balancer.renormalize()

    def task_weights(self) -> dict[str, float]:
        """:return: the current weight of each task (1 for the sum)."""
        if self.balancer is None:
            return dict.fromkeys(self.TASKS, 1.0)
        return dict(zip(self.TASKS, self.balancer.weights.detach().tolist(), strict=True))
