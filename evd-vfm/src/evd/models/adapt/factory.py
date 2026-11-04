"""From config to strategy, without the orchestrator knowing the families.

``ADAPTERS`` maps family to class, but which fields form each spec is family knowledge.
This second registry, indexed by family, goes from the config mapping to the built
strategy, so adding a PEFT method never adds an ``if`` to the runner.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from evd.models.adapt.base import AdaptationStrategy, NoAdapt
from evd.models.adapt.composite import CANONICAL_ORDER, CompositeAdapt
from evd.models.adapt.davpt import DaVptPEFT, ProxyLossSpec
from evd.models.adapt.prompt import PromptSpec
from evd.models.adapt.prompt_family import make_prompt_strategy
from evd.models.adapt.reparam import ReparamPEFT, ReparamSpec
from evd.models.registry import Registry
from evd.types import check_known_fields

__all__ = [
    "ADAPT_BUILDERS",
    "AdaptContext",
    "build_adaptation",
]

#: Fields carried by every ``configs/adapt/*.yaml`` group that are not part of the spec:
#: the family selects the constructor, the name ends up in the ``run_id``.
HEADER_FIELDS = frozenset({"family", "name"})

#: Hydra sentinel for an undecided value; here it appears on ``adapt.loss.n_classes``,
#: which the head knows.
MISSING = "???"


@dataclass(frozen=True)
class AdaptContext:
    """What an adaptation cannot read from its own config.

    Run-level values ignored by most families; passing them all keeps constructor
    signatures uniform for the registry.

    :param n_classes: head outputs; used only by DA-VPT, whose last prompt banks are class
        proxies.
    :param n_blocks: backbone depth, for placements expressed as fractions.
    :param seed: run seed, used to initialize the adapters.
    """

    n_classes: int
    n_blocks: int
    seed: int


ADAPT_BUILDERS: Registry[Callable[[Mapping[str, Any], AdaptContext], AdaptationStrategy]] = (
    Registry("adapt-builder")
)


def _spec_fields(spec_cls: type, valori: Mapping[str, Any], famiglia: str) -> dict[str, Any]:
    """Filter and validate the fields of an adaptation spec.

    The rule (an unknown config field stops the run) is :func:`evd.types.check_known_fields`;
    this function only adapts the message to the PEFT family.

    :param spec_cls: target dataclass.
    :param valori: config fields, without the header.
    :param famiglia: used only in the error message.
    :return: the fields, ready for the constructor.
    :raises ValueError: if a field does not belong to the spec.
    """
    check_known_fields(spec_cls, valori, f"adapt of family {famiglia!r}")
    return dict(valori)


@ADAPT_BUILDERS.register("none")
def _build_none(valori: Mapping[str, Any], ctx: AdaptContext) -> AdaptationStrategy:
    """No adaptation: the frozen probe of E1/E2.

    :param valori: must be empty (``configs/adapt/none.yaml`` has only the header).
    :param ctx: ignored.
    :return: a :class:`~evd.models.adapt.base.NoAdapt`, not yet applied.
    :raises ValueError: if the config carries fields beyond the header.
    """
    del ctx
    if valori:
        raise ValueError(
            f"family 'none' has no parameters, but the config carries {sorted(valori)}: "
            f"none of them would affect the computation"
        )
    return NoAdapt()


@ADAPT_BUILDERS.register("reparam")
def _build_reparam(valori: Mapping[str, Any], ctx: AdaptContext) -> AdaptationStrategy:
    """LoRA, rsLoRA or DoRA (ADR-019).

    :param valori: fields of :class:`~evd.models.adapt.reparam.ReparamSpec`.
    :param ctx: only the seed is used.
    :return: the strategy, not yet applied.
    :raises ValueError: if a config field does not belong to the spec.
    """
    fields = _spec_fields(ReparamSpec, valori, "reparam")
    fields.setdefault("seed", ctx.seed)
    return ReparamPEFT(ReparamSpec(**fields))


@ADAPT_BUILDERS.register("prompt")
def _build_prompt(valori: Mapping[str, Any], ctx: AdaptContext) -> AdaptationStrategy:
    """Shallow/deep VPT, APT or DA-VPT (ADR-004).

    Two config fields depend on the run and are resolved here: ``start_frac`` (placement as
    a fraction of the depth) and ``loss.n_classes`` (``???`` in the YAML, known by the
    head).

    :param valori: spec fields, plus the optional ``loss`` block.
    :param ctx: depth, classes and seed of the run.
    :return: the strategy, not yet applied.
    :raises ValueError: if a field is unknown, a variant without its own loss declares
        one, or ``loss.n_classes`` contradicts the head.
    """
    fields = dict(valori)
    loss = dict(fields.pop("loss", None) or {})
    fraction = fields.pop("start_frac", None)
    if fraction is not None:
        fields["start_block"] = int(ctx.n_blocks * float(fraction))
    spec = PromptSpec(**_spec_fields(PromptSpec, fields, "prompt"), seed=ctx.seed)
    if spec.variant != "da_vpt":
        if loss:
            raise ValueError(
                f"variant {spec.variant!r} has no loss of its own, but the config "
                f"declares a 'loss' block: {sorted(loss)}"
            )
        return make_prompt_strategy(spec)

    declared = loss.pop("n_classes", MISSING)
    if declared != MISSING and int(declared) != ctx.n_classes:
        raise ValueError(
            f"adapt.loss.n_classes declares {declared} but the head has {ctx.n_classes} outputs: "
            f"DA-VPT has one proxy per class, so the two numbers must match"
        )
    return DaVptPEFT(
        spec,
        ProxyLossSpec(
            **_spec_fields(ProxyLossSpec, loss, "prompt/da_vpt"),
            n_classes=ctx.n_classes,
        ),
    )


@ADAPT_BUILDERS.register("composite")
def _build_composite(valori: Mapping[str, Any], ctx: AdaptContext) -> AdaptationStrategy:
    """Two families on the same backbone: E3-C (ADR-062).

    Components are declared in ``componenti`` as a per-slot mapping, each block being a
    full ``adapt`` group built by :func:`build_adaptation`. It is not a list: the
    application order is fixed (only one order works), so the config cannot express it.

    :param valori: the ``componenti`` field only.
    :param ctx: depth, classes and seed of the run, forwarded to every component.
    :return: the composition, not yet applied.
    :raises ValueError: if ``componenti`` is missing or not a mapping, or the config
        carries other fields.
    """
    components = valori.get("componenti")
    if not isinstance(components, Mapping) or not components:
        raise ValueError(
            f"family 'composite' needs a non-empty 'componenti' block, with one slot "
            f"per family among {list(CANONICAL_ORDER)}; got {components!r}"
        )
    extraneous = sorted(set(valori) - {"componenti"})
    if extraneous:
        raise ValueError(
            f"family 'composite' has no parameters of its own, but the config carries "
            f"{extraneous}: none of them would affect the computation. Component "
            f"settings go inside their slot in 'componenti'"
        )
    return CompositeAdapt(
        {str(slot): build_adaptation(cfg, ctx) for slot, cfg in components.items()}
    )


def build_adaptation(cfg: Mapping[str, Any], ctx: AdaptContext) -> AdaptationStrategy:
    """Build the strategy described by the ``adapt`` config group.

    :param cfg: resolved ``adapt`` group, with ``family`` and ``name``.
    :param ctx: run-level values.
    :return: the strategy, not yet applied (the runner applies it and records the
        ``AdaptationReport``).
    :raises KeyError: if the family is not registered.
    :raises ValueError: if ``family`` is missing or a field does not belong to the spec.
    """
    if "family" not in cfg:
        raise ValueError(
            "the adapt group does not declare 'family': the constructor cannot be chosen"
        )
    family = str(cfg["family"])
    values = {k: v for k, v in cfg.items() if k not in HEADER_FIELDS}
    return ADAPT_BUILDERS.get(family)(values, ctx)
