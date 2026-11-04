"""Factory of the ``prompt`` family: one registry key, four variants.

A separate module to avoid the ``prompt -> apt -> prompt`` import cycle; importing it
populates ``PROMPT_VARIANTS``. ``ADAPTERS`` is indexed by family because E3-X compares
families (invariant 5); the concrete variant lives in the spec.
"""

from __future__ import annotations

from evd.models.adapt.apt import AptPEFT
from evd.models.adapt.base import ADAPTERS, AdaptationStrategy
from evd.models.adapt.davpt import DaVptPEFT, ProxyLossSpec
from evd.models.adapt.prompt import PROMPT_VARIANTS, PromptPEFT, PromptSpec

__all__ = [
    "AptPEFT",
    "DaVptPEFT",
    "PromptPEFT",
    "ProxyLossSpec",
    "make_prompt_strategy",
]


@ADAPTERS.register("prompt")
def make_prompt_strategy(spec: PromptSpec, **kwargs: object) -> AdaptationStrategy:
    """Build the prompt strategy described by the spec.

    :param spec: the configuration; ``spec.variant`` selects the class.
    :param kwargs: extra variant arguments (the ``ProxyLossSpec`` for ``da_vpt``).
    :return: the strategy, not yet applied.
    :raises KeyError: if the variant is not registered in
        :data:`~evd.models.adapt.prompt.PROMPT_VARIANTS`.
    """
    return PROMPT_VARIANTS.get(spec.variant)(spec, **kwargs)  # type: ignore[call-arg]
