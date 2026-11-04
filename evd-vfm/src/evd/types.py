"""Shared value objects, the common language of the system.

They live in this neutral module because ``evd.data`` and ``evd.models`` must not import
each other while both need these types. Package-specific types stay where
they belong (``WindowItem`` in ``evd.data.dataset``, ``TemporalAnnotation`` in
``evd.annotations.schema``, ``TargetModuleMap`` in ``evd.models.adapt``).

All types are immutable; those entering a cache key are hashable and canonicalized
through ``evd.utils.hashing``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclass_fields
from typing import TYPE_CHECKING, Literal

from evd.utils.hashing import stable_hash

if TYPE_CHECKING:  # no torch import at import time for modules that do not use it
    import numpy as np
    import numpy.typing as npt
    from torch import Tensor


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #

Geometry = Literal["center_crop", "eval_center", "three_views", "squash", "resize_only", "none"]
ResizeBackend = Literal["cv2", "torchvision", "none"]


@dataclass(frozen=True)
class PreprocessSpec:
    """Preprocessing matching the backbone pre-training; it determines the cache key.

    No field has a default on purpose: the Meta repository alone has two preprocessing
    chains (eval and demo/HF), and geometry and resize backend measurably change the
    encoder output (``../lab/notes/02_preprocessing.md``).
    """

    fps: float
    """Sampling frame rate (not the native rate of the file)."""
    num_frames: int
    """Frames per window."""
    size: int
    """Side of the final crop, in pixels."""
    short_side: int
    """Short side resized to before the crop. With ``geometry="squash"`` the whole
    frame goes to ``size``×``size``, so the two fields must match (checked by
    ``_squash``)."""
    geometry: Geometry
    backend: ResizeBackend
    backend_version: str
    """Version of the resize library: part of the hash because the backend changes
    the numbers. ``"none"`` when there is no resize."""
    mean: tuple[float, ...]
    std: tuple[float, ...]
    tokenizer: str
    """``"tubelet_2x16x16"`` (VJ2.1) | ``"patch_1x14x14_f16"`` (IVN) |
    ``"frame_patch_16"`` (DINOv3) | ``"tubelet_2x8x8"`` (Tiny)."""
    dtype: Literal["fp32", "fp16"]
    """Pixel dtype after preprocessing. Never ``bf16``: it produces NaN with eager
    attention on V-JEPA 2.1 ViT-B/L/g."""

    def hash(self) -> str:
        """Return a stable sha256 of all fields; a preprocessing change must change it (ADR-007)."""
        return stable_hash(asdict(self))


@dataclass(frozen=True)
class AugmentSpec:
    """Training augmentation, replicating the official V-JEPA 2.1 chain.

    Reference: ``vjepa2/evals/video_classification_frozen/eval.py:414-447``
    (``make_dataloader``, hard-wired for every dataset) applied by ``utils.py:104-131``;
    fields are its arguments, with the same names. Evaluation loaders never receive an
    ``AugmentSpec``, which keeps the feature store valid (ADR-007).

    ``id`` enters the ``run_id`` through ``adapt.name`` (docs/05 §8) and ``meta.json``. The
    ``none`` chain applies no transformation and is declared explicitly rather than as
    ``None`` (docs/04 §7.2 D5).
    """

    id: str
    """Chain identity: ``"none"`` | ``"mild"`` | ``"official"``."""
    random_resize_scale: tuple[float, float]
    """Crop **area** range, as a fraction of the whole frame. Official:
    ``(0.08, 1.0)``; pre-training uses ``(0.3, 1.0)``. Ignored when :attr:`crop` is
    false."""
    random_resize_ratio: tuple[float, float]
    """Crop aspect-ratio range. Official: ``(0.75, 4/3)``."""
    rand_augment: str | None
    """RandAugment policy in ``timm`` syntax (official:
    ``"rand-m7-n4-mstd0.5-inc1"``), applied to whole frames before the crop.
    ``None`` disables it."""
    horizontal_flip: bool
    """Horizontal flip; off in the official probe (``make_dataloader`` passes
    ``random_horizontal_flip=False``)."""
    reprob: float
    """Random erasing probability (official ``0.25``); 0 disables it. The rectangle is
    the same in all frames (``cube=True``)."""
    rand_augment_exclude: tuple[str, ...] = ()
    """RandAugment operations to **remove**, with ``timm`` names (``"Rotate"``,
    ``"ShearX"``, ``"ShearY"``, ``"TranslateXRel"``, ``"TranslateYRel"``). Empty means
    the whole repertoire, i.e. the official chain.

    Removing the five geometric operations keeps the content aligned with the
    temporal annotation (laboratory indication of 31 August 2026); the remaining ten
    are photometric. Subtracting instead of listing keeps the difference from the
    upstream repertoire exactly the declared one.
    """
    crop: bool = True
    """Whether the random crop is applied. ``False`` sends the whole frame to
    ``size``×``size`` (ADR-031 point 8).

    Not an alias of ``scale=(1,1), ratio=(1,1)``: on a 4:3 frame that setting falls
    back to a 240×240 centre crop.
    """
    motion_shift: bool = False
    """``random_resized_crop_with_shift`` variant; ``False`` in the official chain
    and everywhere here.
    """

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("AugmentSpec without id: the run_id would have no name")
        for name, pair in (
            ("random_resize_scale", self.random_resize_scale),
            ("random_resize_ratio", self.random_resize_ratio),
        ):
            bottom, top = pair
            if not 0 < bottom <= top:
                raise ValueError(f"{name} = {pair}: expected an increasing pair of positive values")
        if not 0.0 <= self.reprob <= 1.0:
            raise ValueError(f"reprob is a probability: {self.reprob} is outside [0, 1]")
        if self.rand_augment_exclude and not self.rand_augment:
            raise ValueError(
                f"rand_augment_exclude={self.rand_augment_exclude} without rand_augment: "
                f"there is no chain to remove those operations from, so the field would "
                f"not affect the computation"
            )
        if not self.crop and (
            self.random_resize_scale != (1.0, 1.0) or self.random_resize_ratio != (1.0, 1.0)
        ):
            raise ValueError(
                f"crop=False but scale={self.random_resize_scale} and "
                f"ratio={self.random_resize_ratio}: the chain declares two contradictory "
                f"things. Without a crop those ranges are meaningless and must stay "
                f"(1,1)"
            )
        if self.motion_shift:
            raise ValueError(
                "motion_shift=True is not implemented: the official probe chain uses "
                "random_resized_crop, not the shift variant (eval.py:444). If needed, "
                "copy it into evd/vendor/ first"
            )

    @property
    def activate(self) -> bool:
        """:return: whether the chain actually changes the pixels."""
        return self.id != "none"

    def hash(self) -> str:
        """:return: a stable sha256 of all fields, for ``meta.json``."""
        return stable_hash(asdict(self))


# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WindowSpec:
    """Causal window: it ends at ``end_frame``, never centred.

    ``stride_frames`` describes uniform windows; when the native/target fps ratio is not an
    integer, ``explicit_frame_indices`` keeps the timestamp-derived indices instead.
    Indices are native frames of the file; the conversion happens only in
    ``evd.data.windows``.

    Explicit indices are non-decreasing, not strictly increasing: on the ``fixed`` grid
    (ADR-047) a video tail can be shorter than ``num_frames`` and frames are repeated, so
    the window shows exactly the labelled content (``np.linspace`` upstream does the same).
    Decreasing indices are rejected.
    """

    video_id: str
    end_frame: int
    num_frames: int
    stride_frames: int | None
    explicit_frame_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.num_frames <= 0:
            raise ValueError(f"num_frames must be > 0, not {self.num_frames}")
        if self.explicit_frame_indices:
            if len(self.explicit_frame_indices) != self.num_frames:
                raise ValueError(
                    "explicit_frame_indices must have num_frames elements: "
                    f"{len(self.explicit_frame_indices)} != {self.num_frames}"
                )
            if self.explicit_frame_indices[-1] != self.end_frame:
                raise ValueError(
                    "the last explicit index must equal end_frame: "
                    f"{self.explicit_frame_indices[-1]} != {self.end_frame}"
                )
            if any(
                right < left
                for left, right in zip(
                    self.explicit_frame_indices,
                    self.explicit_frame_indices[1:],
                    strict=False,
                )
            ):
                raise ValueError(
                    "explicit_frame_indices must be non-decreasing: duplicates are "
                    "allowed (a window shorter than num_frames repeats frames), indices "
                    "going backwards are not"
                )
            if self.stride_frames is not None:
                raise ValueError("stride_frames must be None when indices are explicit")
        elif self.stride_frames is None or self.stride_frames <= 0:
            raise ValueError(f"stride_frames must be > 0, not {self.stride_frames}")
        if self.start_frame < 0:
            raise ValueError(
                f"window before the start of the video: end_frame={self.end_frame}, "
                f"num_frames={self.num_frames}, stride={self.stride_frames} -> "
                f"first index {self.start_frame}"
            )

    @classmethod
    def from_frame_indices(cls, video_id: str, indices: tuple[int, ...]) -> WindowSpec:
        """Build a window from an already discretized temporal grid.

        Keeps the historical identity format when the indices have a constant step and uses
        explicit indices only when needed.

        :param video_id: video identifier.
        :param indices: native, non-decreasing indices (see the class docstring).
        :return: causal window ending at the last index.
        :raises ValueError: if ``indices`` is empty or goes backwards.
        """
        if not indices:
            raise ValueError("at least one index is required to build a window")
        values = tuple(int(index) for index in indices)
        if len(values) == 1:
            return cls(video_id, values[0], 1, 1)
        steps = tuple(right - left for left, right in zip(values, values[1:], strict=False))
        if min(steps) < 0:
            raise ValueError("window indices cannot go backwards")
        # The compact `stride_frames` form only describes a constant positive step; a zero step
        # would be rebuilt from `end_frame` and the stride, not from the real duplicates.
        if len(set(steps)) == 1 and steps[0] > 0:
            return cls(video_id, values[-1], len(values), steps[0])
        return cls(video_id, values[-1], len(values), None, values)

    @property
    def start_frame(self) -> int:
        """Return the first sampled index (included)."""
        if self.explicit_frame_indices:
            return self.explicit_frame_indices[0]
        assert self.stride_frames is not None
        return self.end_frame - (self.num_frames - 1) * self.stride_frames

    @property
    def frame_span(self) -> tuple[int, int]:
        """Return ``(first, last)`` sampled index, both included."""
        return (self.start_frame, self.end_frame)

    @property
    def frame_indices(self) -> tuple[int, ...]:
        """Return the indices to pass to the decoder, in temporal order."""
        if self.explicit_frame_indices:
            return self.explicit_frame_indices
        assert self.stride_frames is not None
        return tuple(
            self.end_frame - (self.num_frames - 1 - i) * self.stride_frames
            for i in range(self.num_frames)
        )

    @property
    def window_id(self) -> str:
        """Return the window identity used in the cache key (ADR-007)."""
        if not self.explicit_frame_indices:
            return f"{self.video_id}@{self.end_frame}x{self.num_frames}s{self.stride_frames}"
        offsets = tuple(self.end_frame - index for index in self.explicit_frame_indices)
        profile = stable_hash({"offsets": offsets})[:12]
        return f"{self.video_id}@{self.end_frame}x{self.num_frames}i{profile}"


@dataclass(frozen=True)
class VideoMeta:
    """What is known about a video file before decoding it (``probe``)."""

    video_id: str
    path: str
    num_frames: int
    fps: float
    width: int
    height: int
    codec: str = ""

    @property
    def duration_s(self) -> float:
        return self.num_frames / self.fps if self.fps > 0 else 0.0


# --------------------------------------------------------------------------- #
# Backbone output
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BackboneOutput:
    """Uniform output of every backbone."""

    tokens: Tensor
    """``[B, N, D]`` flattened sequence. With ``levels`` set, N is
    ``len(levels) * T'*H'*W'``: levels are stacked on the token axis."""
    grid: tuple[int, int, int]
    """``(T', H', W')`` to unflatten **one** level. DINOv3-frame: ``(T, H', W')``."""
    pooled: Tensor | None
    """``[B, D]`` if the backbone has a native pooling, otherwise ``None``."""
    layer: int
    """Index of the extracted layer (negative = from the end, as written in configs).
    With ``levels`` set it is its last element, i.e. the deepest block."""
    levels: tuple[int, ...] = ()
    """Empty means a single layer.

    Non-empty means the tokens concatenate the listed levels on the token axis, in
    this order (the V-JEPA 2.1 multilevel strategy,
    ``vjepa2/evals/video_classification_frozen/modelcustom/vit_encoder_multiclip_multilevel.py:123``).
    ``__post_init__`` uses it to validate the geometry.
    """

    def __post_init__(self) -> None:
        t, h, w = self.grid
        if self.tokens.ndim != 3:
            raise ValueError(f"tokens must be [B, N, D], got {tuple(self.tokens.shape)}")
        if self.levels and self.layer != self.levels[-1]:
            raise ValueError(
                f"layer={self.layer} but levels={self.levels}: with stacked levels "
                f"`layer` must be the last one"
            )
        if len(set(self.levels)) != len(self.levels):
            raise ValueError(f"repeated levels in levels={self.levels}")
        expected = t * h * w * max(1, len(self.levels))
        if expected != self.tokens.shape[1]:
            how_many = "" if not self.levels else f" x {len(self.levels)} levels"
            raise ValueError(
                f"grid {self.grid}{how_many} implies {expected} tokens, but tokens has "
                f"{self.tokens.shape[1]}. Prefix tokens added by the backbone "
                f"(CLS/register/prompt) must be removed before building BackboneOutput."
            )

    @property
    def n_levels(self) -> int:
        """:return: how many levels are stacked on the token axis; 1 for a single level."""
        return max(1, len(self.levels))


def concat_levels(outs: Sequence[BackboneOutput]) -> BackboneOutput:
    """Stack several levels of the same forward on the token axis.

    The only implementation of "multilevel" in the repository, shared by the pixel path
    (``encode_multilevel``) and the feature-store path (``WindowDataset.__getitem__``). The
    token axis grows and ``embed_dim`` is unchanged, as in the official reference, so the
    head does not change.

    :param outs: outputs to stack, in concatenation order; they must share ``grid``, batch,
        dtype and device, and be single-level.
    :return: a ``BackboneOutput`` with ``tokens`` ``[B, n*N, D]``, the grid of one level,
        ``pooled=None`` and ``levels`` equal to the input levels. A single element is
        returned unchanged.
    :raises ValueError: if ``outs`` is empty, an element is already multilevel, or grid,
        batch or token size differ.
    """
    import torch

    if not outs:
        raise ValueError("concat_levels requires at least one BackboneOutput")
    if len(outs) == 1:
        return outs[0]
    already_multi = [out.layer for out in outs if out.levels]
    if already_multi:
        raise ValueError(
            f"concat_levels does not nest: already multilevel at layers {already_multi}"
        )
    grids = {out.grid for out in outs}
    if len(grids) != 1:
        raise ValueError(f"different grids across levels: {sorted(grids)}")
    shapes = {(out.tokens.shape[0], out.tokens.shape[2]) for out in outs}
    if len(shapes) != 1:
        raise ValueError(f"different batch or embed_dim across levels: {sorted(shapes)}")
    return BackboneOutput(
        tokens=torch.cat([out.tokens for out in outs], dim=1),
        grid=outs[0].grid,
        # The native pooling of one level is not the pooling of the stacked sequence: drop it.
        pooled=None,
        layer=outs[-1].layer,
        levels=tuple(out.layer for out in outs),
    )


@dataclass(frozen=True)
class TokenLayout:
    """Position of the non-patch tokens in the sequence, used by ``PromptPEFT``.

    ``positional`` decides which ``pos_policy`` values are allowed: with 3D-RoPE (V-JEPA
    2.1) prompts have no natural spatio-temporal position (ADR-004).
    """

    n_cls: int
    n_register: int
    positional: Literal["rope_3d", "sincos_abs", "learned_abs", "none"]

    @property
    def n_prefix(self) -> int:
        """Return how many tokens precede the patches in the sequence."""
        return self.n_cls + self.n_register


# --------------------------------------------------------------------------- #
# Event classes and scores
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EventClass:
    idx: int
    name: str
    text_prompts: tuple[str, ...] = field(default_factory=tuple)
    """Documentation glossary (docs/04 §6); not a model input (ADR-016)."""


@dataclass(frozen=True)
class WindowScores:
    """Score of one window (the "chunk" of the PdD); no other code path computes it."""

    window: WindowSpec
    anomaly: float
    """In [0, 1]."""
    class_probs: tuple[float, ...] | None = None
    """Per-class probabilities, if the head is multi-class."""


@dataclass(frozen=True, eq=False)
class FrameScores:
    """Chunk scores projected onto frames."""

    video_id: str
    scores: npt.NDArray[np.float32]
    """``[num_frames]`` float32."""
    covered: npt.NDArray[np.bool_]
    """``[num_frames]`` bool: frames that received a score from the mapper."""
    class_scores: npt.NDArray[np.float32] | None = None
    """``[num_frames, n_classes]`` projected probabilities, if available."""


def check_known_fields(
    spec_cls: type,
    valori: Mapping[str, object],
    gruppo: str,
    *,
    read_elsewhere: Sequence[str] = (),
) -> None:
    """Check that every config field has a reader.

    A field nobody reads is a parameter someone believed would change the computation
    (``num_workers`` and ``crop`` in ADR-038, ``model_selection`` in ADR-047); unknown
    fields stop the run before the first forward.

    :param spec_cls: target dataclass; its fields are the allowed ones.
    :param valori: resolved config group.
    :param gruppo: group name, for the error message.
    :param read_elsewhere: fields legitimately declared in the config but read by another
        component (e.g. ``train.checkpoint_every``), listed explicitly.
    :raises ValueError: if a field is neither in the spec nor in ``read_elsewhere``.
    """
    allowed = {f.name for f in dataclass_fields(spec_cls)} | set(read_elsewhere)
    unknown = sorted(set(valori) - allowed)
    if unknown:
        raise ValueError(
            f"unrecognized fields in config {gruppo!r}: {unknown}. "
            f"Allowed: {sorted(allowed)}. A field nobody reads is a misleading "
            f"metadata: remove it from the config or make some component read it "
            f"(CLAUDE.md §3.1)."
        )
