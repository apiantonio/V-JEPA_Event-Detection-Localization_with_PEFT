"""Composite adaptation: two PEFT families on the same backbone.

Experiment block E3-C combines the best reparam (LoRA/DoRA) with the best VPT to test
whether rewriting weights and lengthening the sequence are complementary or redundant
priors. A composition is a strategy in its own right, so the runner keeps a single
code path (invariant 5).

Two facts drive the design:

1. Order is not symmetric. :meth:`PromptPEFT.apply` replaces ``blocks[i]`` with a
   ``PromptedBlock`` while :class:`ReparamPEFT` resolves targets by qualified name
   (``blocks.{i}.attn.qkv``), so prompts must be applied last. The order is imposed
   here, by family, and is not a config field.
2. Both families call ``backbone.freeze()`` inside ``apply()``, which would freeze
   adapters installed earlier. Gradients are re-armed after the last ``apply()`` and
   ``AdaptationReport`` checks that the measured trainable parameters equal the sum of
   the component formulas.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING

from torch import Tensor

from evd.models.adapt.base import (
    ADAPTERS,
    AdaptationReport,
    AdaptationStrategy,
    AdaptedBackbone,
    AuxiliaryLoss,
    Family,
)

if TYPE_CHECKING:  # avoids the backbones -> adapt -> backbones import cycle
    from evd.models.backbones.base import VideoBackbone

__all__ = ["CANONICAL_ORDER", "CompositeAdapt"]

#: Application order of the components, which is also the set of allowed slot names;
#: the only order in which injection works (point 1 of the module docstring).
CANONICAL_ORDER: tuple[str, ...] = ("reparam", "prompt")


@ADAPTERS.register("composite")
class CompositeAdapt:
    """Apply two or more adaptations to the same backbone, in canonical order.

    :param componenti: strategies to compose, by slot; the slot name is the family of the
        strategy it holds (:data:`CANONICAL_ORDER`).
    :raises ValueError: with fewer than two slots, an unknown slot, a strategy whose
        family differs from its slot, or a component carrying its own loss.
    """

    family: Family = "composite"

    def __init__(self, componenti: Mapping[str, AdaptationStrategy]) -> None:
        if len(componenti) < 2:
            raise ValueError(
                f"a composition has at least two components, got {sorted(componenti)}: "
                f"with a single one use that family directly, as the run_id would say"
            )
        unknown = sorted(set(componenti) - set(CANONICAL_ORDER))
        if unknown:
            raise ValueError(
                f"slots {unknown} not allowed: components are declared by family, "
                f"and the composable families are {list(CANONICAL_ORDER)}"
            )
        for slot, strategy in componenti.items():
            declared = str(getattr(strategy, "family", ""))
            if declared != slot:
                raise ValueError(
                    f"slot {slot!r} holds a strategy of family {declared!r}: "
                    f"the slot name decides the application order, so it "
                    f"must match the family it holds"
                )
            # DA-VPT contributes a loss term that the `Trainer` finds via `isinstance(adapt,
            # AuxiliaryLoss)` on this object; a composition would run DA-VPT without its loss, so it
            # is rejected.
            if isinstance(strategy, AuxiliaryLoss):
                raise ValueError(
                    f"component {slot!r} ({type(strategy).__name__}) carries its own loss "
                    f"(AuxiliaryLoss), which this composition would not forward to the "
                    f"Trainer: it would silently run without its loss term"
                )
        self._components: dict[str, AdaptationStrategy] = {
            slot: componenti[slot] for slot in CANONICAL_ORDER if slot in componenti
        }
        self._is_applied = False
        self._rearmed = 0

    @property
    def slots(self) -> tuple[str, ...]:
        """:return: the slots present, in application order."""
        return tuple(self._components)

    def componente(self, slot: str) -> AdaptationStrategy:
        """:param slot: slot name.
        :return: the strategy in that slot.
        :raises KeyError: if the slot is not present.
        """
        return self._components[slot]

    # --- AdaptationStrategy contract ---------------------------------------
    def apply(self, backbone: VideoBackbone, /) -> AdaptedBackbone:
        """Apply every component in canonical order and re-arm the gradients.

        Each component calls ``freeze()`` inside its own ``apply()``, disabling the adapters
        installed before; re-arming after the last application keeps the whole composition
        trainable, and :class:`~evd.models.adapt.base.AdaptationReport` checks the measured
        parameters against the sum of the component formulas.

        :param backbone: backbone to adapt; modified in place.
        :return: the single receipt of the composition.
        :raises RuntimeError: if the strategy was already applied.
        :raises ValueError: if the measured trainable parameters are not the sum of those
            expected by the components.
        """
        if self._is_applied:
            raise RuntimeError(
                "this CompositeAdapt was already applied: a second apply() "
                "would nest the components and the counts would still look plausible"
            )
        backbone.freeze()
        below: dict[str, AdaptationReport] = {}
        for slot, strategy in self._components.items():
            below[slot] = strategy.apply(backbone).report

        self._rearmed = 0
        for strategy in self._components.values():
            for parameter in strategy.trainable_parameters():
                if not parameter.requires_grad:
                    parameter.requires_grad_(True)
                    self._rearmed += 1
        self._is_applied = True

        module = backbone.module()
        measured = sum(p.numel() for p in module.parameters() if p.requires_grad)
        expected = sum(r.n_trainable_formula for r in below.values())
        report = AdaptationReport(
            family=self.family,
            method="+".join(r.method for r in below.values()),
            n_targets=sum(r.n_targets for r in below.values()),
            n_trainable=measured,
            n_trainable_formula=expected,
            extra_tokens=sum(r.extra_tokens for r in below.values()),
            targets=tuple(f"{slot}:{t}" for slot, r in below.items() for t in r.targets),
            notes={
                "ordine": list(self._components),
                "tensori_riarmati": self._rearmed,
                "componenti": {slot: r.as_dict() for slot, r in below.items()},
            },
        )
        return AdaptedBackbone(backbone=backbone, strategy=self, report=report)

    def trainable_parameters(self) -> Iterator[Tensor]:
        """:return: the parameters of every component, in application order."""
        for strategy in self._components.values():
            yield from strategy.trainable_parameters()

    def can_merge(self) -> bool:
        """:return: true only if every component merges into the weights.

        Always false with a prompt component: composing the two families costs full
        mergeability (a result for C7). The reparam half still merges; see
        :meth:`merge_parts` (ADR-064).
        """
        return all(c.can_merge() for c in self._components.values())

    def merge_parts(self) -> tuple[str, ...]:
        """:return: the slots that ``merge_()`` merges into the weights, in order.

        ``("reparam",)`` on ``reparam+prompt``: the union of what components can merge, not
        their unanimity (ADR-064). An unmerged reparam costs +42.8% (LoRA) or +83.7% (DoRA)
        inference latency (E5b).
        """
        return tuple(slot for slot, c in self._components.items() if c.merge_parts())

    def merge_(self) -> None:
        """Merge into the weights every mergeable component, and only those.

        Merging LoRA into ``base.weight`` does not touch the prompts and does not change the
        output: the families act on disjoint things (``Linear`` weights and sequence length).
        After the merge the reparam delta no longer exists separately, so the shipped model is
        the whole backbone plus the prompts.

        :raises NotImplementedError: only if nothing can be merged.
        """
        mergeable = self.merge_parts()
        if not mergeable:
            raise NotImplementedError(
                f"nothing in this composition can be merged: the components "
                f"{list(self._components)} all lengthen the sequence, and no weight can "
                f"absorb a longer sequence. This is the trade-off measured by C7, not a "
                f"limitation of this class"
            )
        for slot in mergeable:
            self._components[slot].merge_()

    def unload_(self) -> int:
        """Unload the components in reverse application order.

        Prompts wrap the blocks containing the reparam-adapted modules, so reparam is removed
        last.

        :return: total number of removed structures.
        """
        return sum(s.unload_() for s in reversed(list(self._components.values())))

    def unload_merged_(self) -> int:
        """Unload only the already merged structures, leaving the others in place.

        Second half of the partial merge: after ``merge_()`` the reparam wrappers are
        pass-through modules still paid at every forward. Prompts remain, so ``extra_tokens()``
        is unchanged and the composition is still the adaptation. The reverse order is no
        longer required since ``ReparamPEFT`` unloads from its anchors (ADR-064).

        :return: total number of removed structures.
        """
        return sum(s.unload_merged_() for s in reversed(list(self._components.values())))

    def extra_tokens(self) -> int:
        """:return: tokens added to the sequence, summed over the components."""
        return sum(c.extra_tokens() for c in self._components.values())

    def state_dict_adapter(self) -> dict[str, Tensor]:
        """:return: the new weights, with keys prefixed by the slot.

        The prefix prevents collisions between families (``blocks.3...`` exists in both).
        After ``merge_()`` and ``unload_merged_()`` the keys of the merged component disappear,
        since those weights now live in the backbone.
        """
        return {
            f"{slot}/{key}": value
            for slot, strategy in self._components.items()
            for key, value in strategy.state_dict_adapter().items()
        }
