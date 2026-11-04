"""The ``AdaptationStrategy`` contract and its value objects..

Only what is shared by the two PEFT families compared in the thesis: reparam
(LoRA/DoRA) merges into the weights, prompt (VPT/APT/DA-VPT) lengthens the sequence
and cannot be merged. Since ADR-064 the contract asks two questions: ``can_merge()``
(does the whole adaptation disappear?) and ``merge_parts()`` (which parts disappear?);
on a LoRA+VPT composition the reparam half merges and the prompts remain.

A runner that only sees this protocol does not need to know the family, so E3-X
compares families through a single code path.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from torch import Tensor

from evd.models.registry import Registry

if TYPE_CHECKING:  # avoids the backbones -> adapt -> backbones import cycle
    from evd.models.backbones.base import VideoBackbone

#: Families of docs/03 §4.5. ``none`` lets E1/E2 and E3 share the same runner (no
#: ``if adapt is None`` branch). ``composite`` (E3-C, ADR-062) is a strategy in its own
#: right; it is not ``none``, so ``preserves_backbone_output`` keeps the feature store
#: closed (invariant 3).
Family = Literal["none", "reparam", "prompt", "composite"]


@dataclass(frozen=True)
class AdaptationReport:
    """What was adapted, in a form written to ``meta.json`` (invariant 2).

    :param family: family of the strategy that produced the report.
    :param method: concrete variant (``"lora"``, ``"dora"``, ``"vpt_deep"``...).
    :param n_targets: number of replaced or hooked modules.
    :param n_trainable: parameters with ``requires_grad``, counted on the model.
    :param n_trainable_formula: the same parameters from the closed formula; must equal
        ``n_trainable`` (an assertion, not redundant data).
    :param extra_tokens: tokens added to the sequence at every forward (0 for reparam).
    :param targets: qualified target names, in stable order.
    :param notes: spec fields that distinguish configurations with the same ``method``
        and parameter count (``ΔW`` factorization, prompt positional policy, number of
        DA-VPT proxies).
    """

    family: Family
    method: str
    n_targets: int
    n_trainable: int
    n_trainable_formula: int
    extra_tokens: int = 0
    targets: tuple[str, ...] = field(default_factory=tuple)
    notes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_trainable != self.n_trainable_formula:
            raise ValueError(
                f"inconsistent trainable parameters for {self.method!r}: the model "
                f"exposes {self.n_trainable}, the formula expects "
                f"{self.n_trainable_formula}. Either the injection missed a target, or "
                f"part of the backbone was left unfrozen."
            )

    def fraction_of(self, total_parameters: int) -> float:
        """Return the trained fraction of the backbone.

        Report it with the backbone it refers to: the same LoRA configuration ranges from
        1.019% to 0.416% across the four V-JEPA 2.1 variants.

        :param total_parameters: total parameters of the unadapted backbone.
        :return: fraction in [0, 1]; 0.0 if ``total_parameters`` is 0.
        """
        return self.n_trainable / total_parameters if total_parameters else 0.0

    def as_dict(self) -> dict[str, object]:
        """:return: a serializable copy, ready for ``meta.json``."""
        return {
            "family": self.family,
            "method": self.method,
            "n_targets": self.n_targets,
            "n_trainable": self.n_trainable,
            "extra_tokens": self.extra_tokens,
            "targets": list(self.targets),
            "notes": dict(self.notes),
        }


@dataclass(frozen=True)
class AdaptedBackbone:
    """The backbone after ``apply()``, with the strategy that adapted it.

    Adaptation is in place: ``backbone`` is the same object passed to ``apply()``. This
    dataclass is a receipt of who changed what.

    :param backbone: the adapted backbone (same object as the input).
    :param strategy: the strategy that adapted it; keeps the state needed by ``merge_()``,
        ``unload_()`` and ``state_dict_adapter()``.
    :param report: counts and targets, for ``meta.json``.
    """

    backbone: VideoBackbone
    strategy: AdaptationStrategy
    report: AdaptationReport


@runtime_checkable
class AdaptationStrategy(Protocol):
    """How a frozen backbone is adapted, without the runner knowing which method.

    Life cycle: ``apply()`` once on a backbone, ``trainable_parameters()`` to the
    optimizer, then ``state_dict_adapter()`` for the checkpoint and, where possible,
    ``merge_()`` followed by ``unload_()`` for inference.
    """

    family: Family

    def apply(self, backbone: VideoBackbone, /) -> AdaptedBackbone:
        """Freeze the backbone and install the adaptation, in place.

        :param backbone: backbone to adapt; it is modified.
        :return: the receipt with backbone, strategy and counts.
        :raises RuntimeError: if the strategy was already applied (nested adapters).
        """
        ...

    def trainable_parameters(self) -> Iterator[Tensor]:
        """:return: only the parameters introduced by the adaptation, in stable order."""
        ...

    def can_merge(self) -> bool:
        """:return: ``True`` if the whole adaptation can be merged into the weights.

        True for reparam, false for prompt. On a composition it differs from
        :meth:`merge_parts` (ADR-064).
        """
        ...

    def merge_parts(self) -> tuple[str, ...]:
        """:return: names of the parts that ``merge_()`` merges into the weights.

        Empty when there is nothing to merge, which makes ``merge_()`` illegal. On a
        composition only the reparam part merges (ADR-064).
        """
        ...

    def merge_(self) -> None:
        """Merge into the backbone weights everything that can be merged, in place.

        :meth:`merge_parts` says what stays out; callers read it beforehand.

        :raises NotImplementedError: if ``merge_parts()`` is empty.
        """
        ...

    def unload_(self) -> int:
        """Remove the adaptation structures and return the bare backbone.

        Call it after ``merge_()`` to keep the effect; alone it discards the adaptation.
        Merging alone leaves a Python wrapper per module, paid at every forward.

        :return: number of removed structures.
        """
        ...

    def unload_merged_(self) -> int:
        """Remove only the already merged structures, leaving the rest in place.

        After a partial merge on LoRA+VPT it removes the LoRA wrappers and keeps the prompts.
        Equivalent to ``unload_()`` on a fully mergeable strategy; a no-op returning 0 (not an
        exception) on a non-mergeable one.

        :return: number of removed structures.
        """
        ...

    def extra_tokens(self) -> int:
        """:return: tokens added to the sequence at every forward (0 for reparam)."""
        ...

    def state_dict_adapter(self) -> dict[str, Tensor]:
        """:return: only the new weights, with the qualified target names (a PEFT checkpoint
        must not contain the backbone).
        """
        ...


@runtime_checkable
class AuxiliaryLoss(Protocol):
    """A strategy contributing its own loss term.

    A separate contract (ISP), so strategies without a loss do not implement it; the
    ``Trainer`` asks "are you an ``AuxiliaryLoss``?" instead of "are you DA-VPT?".
    Life cycle per forward: ``reset_auxiliary()``, the backbone forward (which collects
    what is needed), then ``auxiliary_loss(labels)``.
    """

    def reset_auxiliary(self) -> None:
        """Discard what the previous forward collected."""
        ...

    def auxiliary_loss(self, labels: Tensor) -> Tensor:
        """Return the term to add to the classification loss.

        :param labels: class labels of the batch, shape ``(B,)``.
        :return: a scalar, already weighted by the method configuration.
        :raises RuntimeError: if no forward happened after the last reset.
        """
        ...


#: Strategy registry by family: ``ADAPTERS.get("reparam")(spec)``. The concrete variant
#: lives in the spec, since E3-X compares families across experiments.
ADAPTERS: Registry[Callable[..., AdaptationStrategy]] = Registry("adapter")


@ADAPTERS.register("none")
class NoAdapt:
    """Identity adaptation: freeze the backbone and install nothing.

    Lets E2 (frozen probes) and E3 (PEFT) go through the same runner (invariant 5).
    ``unload_()`` is a no-op returning 0, ``merge_()`` raises as the contract requires.
    The report compares the measured trainable parameters with the formula (zero), so an
    incomplete ``freeze()`` is caught before training.
    """

    family: Family = "none"

    def __init__(self) -> None:
        self._applied = False

    # --- AdaptationStrategy contract ---------------------------------------
    def apply(self, backbone: VideoBackbone, /) -> AdaptedBackbone:
        """Freeze the backbone and produce a zero-parameter receipt.

        :param backbone: backbone to freeze; modified in place.
        :return: the receipt, with ``n_trainable`` measured on the model.
        :raises RuntimeError: if the strategy was already applied.
        :raises ValueError: if trainable parameters remain after ``freeze()`` (raised by
            :class:`AdaptationReport`).
        """
        if self._applied:
            raise RuntimeError(
                "this NoAdapt was already applied: a second apply() would add nothing, "
                "but would hide that the runner is reusing a strategy instead of "
                "building one per run"
            )
        backbone.freeze()
        module = backbone.module()
        measured = sum(p.numel() for p in module.parameters() if p.requires_grad)
        self._applied = True
        report = AdaptationReport(
            family=self.family,
            method="frozen",
            n_targets=0,
            n_trainable=measured,
            n_trainable_formula=0,
            extra_tokens=0,
            targets=(),
            notes={},
        )
        return AdaptedBackbone(backbone=backbone, strategy=self, report=report)

    def trainable_parameters(self) -> Iterator[Tensor]:
        """:return: nothing: what distinguishes a frozen probe from a PEFT run."""
        return iter(())

    def can_merge(self) -> bool:
        """:return: always ``False``: nothing to merge."""
        return False

    def merge_parts(self) -> tuple[str, ...]:
        """:return: always empty: nothing to merge."""
        return ()

    def merge_(self) -> None:
        """:raises NotImplementedError: always, since there is nothing to merge."""
        raise NotImplementedError(
            "NoAdapt has nothing to merge: call merge_() only if merge_parts() "
            "is not empty (AdaptationStrategy contract)"
        )

    def unload_merged_(self) -> int:
        """:return: always 0: nothing was merged, so nothing is removed."""
        return 0

    def unload_(self) -> int:
        """:return: always 0: no structure was installed.

        The backbone stays frozen: as for the other families, ``unload_()`` does not restore
        ``requires_grad``.
        """
        return 0

    def extra_tokens(self) -> int:
        """:return: always 0: the sequence does not grow."""
        return 0

    def state_dict_adapter(self) -> dict[str, Tensor]:
        """:return: an empty dictionary: a probe checkpoint is the head alone."""
        return {}


def freezes_backbone(adapt: AdaptationStrategy | None) -> bool:
    """Return whether the backbone stays frozen under this adaptation.

    It decides whether the forward runs under ``no_grad`` (``Trainer``) and whether the
    loader may read from the feature store (invariant 3, ADR-007). A free function rather
    than a protocol method, since only two callers need it.

    :param adapt: the strategy, or ``None`` for no adaptation.
    :return: ``True`` if no adaptation parameter requires gradients.
    """
    if adapt is None:
        return True
    return not any(p.requires_grad for p in adapt.trainable_parameters())


def preserves_backbone_output(adapt: AdaptationStrategy | None) -> bool:
    """Return whether features extracted from the bare backbone remain valid.

    Not the same question as :func:`freezes_backbone`: prompts require no backbone
    gradients but change what the backbone computes. The answer looks at ``family``, which
    is declared at construction (ADR-056): checking materialized parameters failed on a
    strategy not yet applied, and LoRA/DoRA would have read bare-backbone features from
    the cache.

    :param adapt: the strategy, or ``None``.
    :return: ``True`` only for the ``none`` family, i.e. when reading from the cache gives
        the same result as the forward.
    """
    if adapt is None:
        return True
    # `getattr` rather than `adapt.family`: an unknown strategy gets the conservative answer
    # (no cache) instead of an `AttributeError`.
    return str(getattr(adapt, "family", "")) == "none"
