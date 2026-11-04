"""``VideoBackbone`` on V-JEPA 2.1 built from the Meta source.

The wrapped encoder is
``vjepa2/app/vjepa_2_1/models/vision_transformer.py::VisionTransformer``; building and
checkpoint loading live in ``vjepa21_source``, this class only maps the model onto the
docs/03 §4.1 contract.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from torch import Tensor

from evd.models.adapt.target_maps import TargetModuleMap
from evd.models.backbones.base import BACKBONES
from evd.models.backbones.vjepa21_source import (
    ARCHS,
    ArchitectureMismatch,
    ArchSpec,
    build_loaded_encoder,
    checkpoint_sha,
)
from evd.types import BackboneOutput, PreprocessSpec, TokenLayout
from evd.utils import env


class VJEPA21Backbone:
    """Adapt an already built Meta encoder to the ``VideoBackbone`` contract.

    Receiving the module instead of a checkpoint path allows meta-device tests and keeps
    I/O separate from the contract.
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        arch: ArchSpec,
        ckpt_sha: str,
        spec: PreprocessSpec,
        device: str = "cpu",
        load_report: Mapping[str, Any] | None = None,
    ) -> None:
        self.encoder = encoder
        self.model = encoder
        self.arch = arch
        self.backbone_id = arch.backbone_id
        self.embed_dim = arch.embed_dim
        self.ckpt_sha = ckpt_sha
        self._spec = spec
        self.device = device
        self.load_report = dict(load_report) if load_report is not None else None
        self.encoder.eval()

    def preprocess_spec(self) -> PreprocessSpec:
        return self._spec

    def _layer_index(self, layer: int) -> int:
        idx = layer if layer >= 0 else self.arch.depth + layer
        if not 0 <= idx < self.arch.depth:
            raise IndexError(
                f"layer {layer} outside the {self.arch.depth} blocks of {self.arch.name}"
            )
        return idx

    def encode(self, clips: Tensor, layer: int = -1) -> BackboneOutput:
        return self.encode_layers(clips, (layer,))[layer]

    def encode_layers(self, clips: Tensor, layers: Sequence[int]) -> dict[int, BackboneOutput]:
        """Extract several blocks in a single forward.

        A hook is installed on each requested block and always removed in ``finally``.
        Hierarchical layers go through their ``norms_block``; the others stay raw, as in the
        single-layer path.

        :raises ValueError: on an empty list, a non-divisible clip geometry or layer aliases.
        :raises TypeError: if a block returns something other than a tensor.
        :raises RuntimeError: if the hooks do not capture one output per layer.
        """
        requested = tuple(int(layer) for layer in layers)
        if not requested:
            raise ValueError("encode_layers requires at least one layer")
        if clips.ndim != 5:
            raise ValueError(f"expected clips [B,C,T,H,W], got {tuple(clips.shape)}")
        _, _, frames, height, width = clips.shape
        tubelet = int(getattr(self.encoder, "tubelet_size", 2))
        patch = int(getattr(self.encoder, "patch_size", 16))
        if frames % tubelet or height % patch or width % patch:
            raise ValueError(
                f"clip {frames}x{height}x{width} not divisible by the tokenizer "
                f"{tubelet}x{patch}x{patch}"
            )
        resolved = tuple(self._layer_index(layer) for layer in requested)
        if len(set(resolved)) != len(resolved):
            raise ValueError(
                f"layer aliases in the same forward: requested {requested}, resolved {resolved}"
            )
        captured: dict[int, list[Tensor]] = {idx: [] for idx in resolved}

        def capture_for(
            idx: int,
        ) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
            def capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
                value = output[0] if isinstance(output, tuple) else output
                if not isinstance(value, Tensor):
                    raise TypeError(f"block {idx} returned {type(value).__name__}, not a Tensor")
                captured[idx].append(value)

            return capture

        handles = [
            self.encoder.blocks[idx].register_forward_hook(capture_for(idx)) for idx in resolved
        ]
        try:
            try:
                param = next(self.encoder.parameters())
                clips = clips.to(device=param.device, dtype=param.dtype)
            except StopIteration:
                clips = clips.to(self.device)
            self.encoder(clips)
        finally:
            for handle in handles:
                handle.remove()
        invalid = {idx: len(values) for idx, values in captured.items() if len(values) != 1}
        if invalid:
            raise RuntimeError(f"block hooks: expected one output per layer, got {invalid}")
        hierarchical = tuple(self.encoder.hierarchical_layers)
        grid = (frames // tubelet, height // patch, width // patch)
        outputs: dict[int, BackboneOutput] = {}
        for layer, idx in zip(requested, resolved, strict=True):
            tokens = captured[idx][0]
            if idx in hierarchical:
                norm_idx = hierarchical.index(idx)
                tokens = self.encoder.norms_block[norm_idx](tokens)
            outputs[layer] = BackboneOutput(
                tokens=tokens,
                grid=grid,
                pooled=None,
                layer=layer,
            )
        return outputs

    def target_map(self) -> TargetModuleMap:
        depth = self.arch.depth
        return TargetModuleMap(
            o=tuple(f"blocks.{i}.attn.proj" for i in range(depth)),
            mlp=tuple(
                f"blocks.{i}.mlp.{projection}"
                for i in range(depth)
                for projection in ("fc1", "fc2")
            ),
            fused_qkv=tuple(f"blocks.{i}.attn.qkv" for i in range(depth)),
            fused_out_dim=self.embed_dim,
            fused_order=("q", "k", "v"),
        )

    def module(self) -> nn.Module:
        """Return the encoder: ``target_map()`` names (``blocks.0.attn.qkv`` and similar) are
        relative to it, not to this wrapper.
        """
        return self.encoder

    def blocks(self) -> nn.ModuleList:
        return cast(nn.ModuleList, self.encoder.blocks)

    def token_layout(self) -> TokenLayout:
        return TokenLayout(n_cls=0, n_register=0, positional="rope_3d")

    def predictor(self) -> nn.Module | None:
        """Return ``None``: the predictor is not built."""
        return None

    def hierarchical_layers(self) -> tuple[int, ...]:
        return tuple(int(layer) for layer in self.encoder.hierarchical_layers)

    def parameters(self) -> Iterator[nn.Parameter]:
        return self.encoder.parameters()

    def to(self, device: str | torch.device) -> VJEPA21Backbone:
        self.encoder.to(device)
        self.device = str(device)
        return self

    def eval(self) -> VJEPA21Backbone:
        self.encoder.eval()
        return self

    def freeze(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)


@BACKBONES.register("vjepa21gigantic")
@BACKBONES.register("vjepa21giant")
@BACKBONES.register("vjepa21l")
@BACKBONES.register("vjepa21b")
def vjepa21_backbone(
    *,
    arch: str,
    checkpoint: str | Path,
    state_dict_key: str,
    size: int,
    num_frames: int,
    preprocess_variant: str,
    preprocess_fps: float,
    preprocess_dtype: str,
    expected: Mapping[str, int] | None = None,
    hierarchical_layers: Sequence[int] | None = None,
    repo: str | Path | None = None,
    device: str | None = None,
    strict: bool = True,
    id: str | None = None,
) -> VJEPA21Backbone:
    """Registered factory: resolve paths, weights and spec without implicit values.

    :raises ValueError: if ``strict`` is false or ``arch`` is unknown.
    :raises ArchitectureMismatch: if the built model differs from the declared one.
    """
    if not strict:
        raise ValueError("V-JEPA 2.1 is loaded only with strict=True (ADR-018)")
    try:
        arch_spec = ARCHS[arch]
    except KeyError:
        raise ValueError(f"unknown arch {arch!r}; allowed: {sorted(ARCHS)}") from None
    if id is not None and id != arch_spec.backbone_id:
        raise ArchitectureMismatch(
            f"arch {arch!r} requires id={arch_spec.backbone_id!r}, got {id!r}"
        )
    if state_dict_key != arch_spec.ckpt_key:
        raise ArchitectureMismatch(
            f"arch {arch!r} requires state_dict_key={arch_spec.ckpt_key!r}, got {state_dict_key!r}"
        )
    resolved_device = device or env.device()
    ckpt = Path(checkpoint)
    digest = checkpoint_sha(ckpt)
    encoder, load_report = build_loaded_encoder(
        arch,
        checkpoint=ckpt,
        key=state_dict_key,
        img_size=int(size),
        num_frames=int(num_frames),
        device=resolved_device,
        repo=repo,
    )
    actual = {
        "embed_dim": int(encoder.embed_dim),
        "depth": len(encoder.blocks),
        "num_heads": int(encoder.num_heads),
        "head_dim": int(encoder.embed_dim) // int(encoder.num_heads),
    }
    if expected is not None:
        declared = {key: int(value) for key, value in expected.items()}
        mismatches = {
            key: (declared[key], actual.get(key))
            for key in declared
            if actual.get(key) != declared[key]
        }
        if mismatches:
            raise ArchitectureMismatch(f"expected config does not match the model: {mismatches}")
    model_layers = tuple(int(value) for value in encoder.hierarchical_layers)
    if hierarchical_layers is not None and tuple(map(int, hierarchical_layers)) != model_layers:
        raise ArchitectureMismatch(
            f"hierarchical_layers in the config={tuple(hierarchical_layers)}, model={model_layers}"
        )
    # Same declared exception as in `tiny.py`: the preprocessing spec lives in `data`, where
    # pixels are produced. Local import, so "models does not import data" holds at import
    # time.
    from evd.data.preprocess import vjepa21_preprocess_spec

    spec = vjepa21_preprocess_spec(
        preprocess_variant,  # type: ignore[arg-type]
        fps=float(preprocess_fps),
        num_frames=int(num_frames),
        size=int(size),
        dtype=preprocess_dtype,  # type: ignore[arg-type]
    )
    return VJEPA21Backbone(
        encoder,
        arch=arch_spec,
        ckpt_sha=digest,
        spec=spec,
        device=resolved_device,
        load_report=load_report,
    )
