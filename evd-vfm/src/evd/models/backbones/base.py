"""The ``VideoBackbone`` contract and the backbone registry."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import torch.nn as nn
from torch import Tensor

from evd.models.adapt.target_maps import TargetModuleMap
from evd.models.registry import Registry
from evd.types import BackboneOutput, PreprocessSpec, TokenLayout, concat_levels


@runtime_checkable
class VideoBackbone(Protocol):
    """Uniform output, declared preprocessing spec, known target modules.

    Implementations wrap upstream code without modifying it (invariant 11): the Meta
    source is the numerical reference of the project.
    """

    backbone_id: str
    embed_dim: int

    def preprocess_spec(self) -> PreprocessSpec: ...

    def encode(self, clips: Tensor, layer: int = -1) -> BackboneOutput:
        """Encode ``clips`` ``[B, C, T, H, W]`` into a ``BackboneOutput``; a negative ``layer``
        counts from the end, as in configs.
        """
        ...

    def encode_layers(self, clips: Tensor, layers: Sequence[int]) -> dict[int, BackboneOutput]:
        """Extract several layers with a single encoder forward.

        Keys are the indices requested by the caller, negative ones included. Implementations
        reject aliases of the same block in one call (e.g. ``-1`` and ``depth-1``).
        """
        ...

    def freeze(self) -> None: ...

    def eval(self) -> VideoBackbone: ...

    def module(self) -> nn.Module:
        """Return the ``nn.Module`` on which ``target_map()`` names its targets.

        Qualified names of ``TargetModuleMap`` are relative to this module, so
        ``AdaptationStrategy.apply()`` does not guess the attribute holding the network.

        :return: the actual network, not a container.
        """
        ...

    def blocks(self) -> nn.ModuleList:
        """Return the transformer blocks, in order.

        ``PromptPEFT`` works on blocks, which ``TargetModuleMap`` does not name.

        :return: the ``nn.ModuleList`` of blocks; ``len()`` is the depth.
        """
        ...

    def target_map(self) -> TargetModuleMap: ...

    def token_layout(self) -> TokenLayout: ...

    def predictor(self) -> nn.Module | None:
        """Return the predictor if the backbone exposes it, ``None`` otherwise."""
        ...

    def hierarchical_layers(self) -> tuple[int, ...]:
        """Return the intermediate layer indices declared by the model.

        V-JEPA 2.1 exposes normalized intermediate outputs (``norms_block``) at quarters of
        the depth (``[2,5,8,11]`` on ViT-B, ``[5,11,17,23]`` on ViT-L); see
        ``../lab/notes/03_target_modules.md``.
        """
        ...


def encode_multilevel(
    backbone: VideoBackbone, clips: Tensor, levels: Sequence[int]
) -> BackboneOutput:
    """Run a single forward and stack several levels on the token axis.

    Pixel path of the V-JEPA 2.1 multilevel strategy
    (``vjepa2/evals/video_classification_frozen/modelcustom/vit_encoder_multiclip_multilevel.py:122-123``).
    The concatenation itself is :func:`~evd.types.concat_levels`, shared with the
    feature-store path. A free function, not a protocol method, because only the trainer
    and the scorer need it.

    :param backbone: any ``VideoBackbone`` implementation.
    :param clips: ``[B, C, T, H, W]``.
    :param levels: block indices in concatenation order; negative indices allowed. With a
        single element the result equals ``backbone.encode(clips, layer=levels[0])``.
    :return: tokens of the concatenated levels, with ``levels`` set (empty for a single
        level).
    :raises ValueError: if ``levels`` is empty; aliases of the same block are rejected by
        the implementation's ``encode_layers``.
    """
    requested = tuple(int(level) for level in levels)
    if not requested:
        raise ValueError("encode_multilevel requires at least one level")
    outputs = backbone.encode_layers(clips, requested)
    return concat_levels([outputs[level] for level in requested])


#: Backbone registry: ``@BACKBONES.register("tiny")`` on a factory.
BACKBONES: Registry[Callable[..., VideoBackbone]] = Registry("backbone")
