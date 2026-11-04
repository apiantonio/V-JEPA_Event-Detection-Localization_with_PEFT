"""Explicit preprocessing specs for V-JEPA 2.1.

The upstream repository contains two different chains: the *eval* chain (short side
to crop size, three spatial views, ``cv2.resize`` without antialias) and the
*demo/hub* chain (short side ``int(256/224 * crop)``, center crop, ``torchvision``
with antialias). Both geometry and backend change the encoder output, so both are
fields of ``PreprocessSpec`` and enter the cache hash.

Named variants: ``eval_three_views``, ``eval_center``, ``demo_center`` and
``squash``. ``squash`` is not an upstream chain: it resizes the whole frame to
``size x size`` without preserving the aspect ratio, keeping the full field of view
of surveillance footage. The 4:3 to 1:1 distortion (ratio 0.75) lies within the
``random_resize_aspect_ratio`` range used to pretrain V-JEPA 2.1.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from evd.types import PreprocessSpec, ResizeBackend

PreprocessVariant = Literal["eval_three_views", "eval_center", "demo_center", "squash"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def vjepa21_preprocess_spec(
    variant: PreprocessVariant,
    *,
    fps: float,
    num_frames: int,
    size: int,
    dtype: Literal["fp32", "fp16"],
) -> PreprocessSpec:
    """Build one of the verified preprocessing chains.

    All fields stay explicit in the returned spec (and in its hash); the factory only
    prevents inconsistent combinations such as eval geometry with the demo backend.

    :param variant: ``eval_three_views``, ``eval_center``, ``demo_center`` or ``squash``.
    :param fps: target sampling frequency.
    :param num_frames: samples per window.
    :param size: output resolution.
    :param dtype: output dtype.
    :return: the preprocessing spec.
    :raises ValueError: if the variant is unknown.
    """
    if variant == "eval_three_views":
        short_side, geometry, backend = size, "three_views", "cv2"
    elif variant == "eval_center":
        short_side, geometry, backend = size, "eval_center", "cv2"
    elif variant == "demo_center":
        short_side = int(256 / 224 * size)
        geometry, backend = "center_crop", "torchvision"
    elif variant == "squash":
        # The whole frame becomes square; cv2 backend as in the eval chain, so that
        # `squash` and `eval_center` differ only in the crop.
        short_side, geometry, backend = size, "squash", "cv2"
    else:
        raise ValueError(
            f"unknown V-JEPA 2.1 preprocessing variant: {variant!r}; "
            "expected eval_three_views | eval_center | demo_center | squash"
        )
    return PreprocessSpec(
        fps=fps,
        num_frames=num_frames,
        size=size,
        short_side=short_side,
        geometry=geometry,
        backend=backend,
        backend_version=resolve_backend_version(backend),
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        tokenizer="tubelet_2x16x16",
        dtype=dtype,
    )


def resolve_backend_version(backend: ResizeBackend) -> str:
    """Return the version of the resize library, which enters the preprocessing hash.

    :param backend: resize backend name.
    :return: library version string.
    :raises ValueError: if the backend is unknown.
    """
    if backend == "cv2":
        import cv2

        return f"cv2-{cv2.__version__}"
    if backend == "torchvision":
        import torchvision

        return f"torchvision-{torchvision.__version__}"
    if backend == "none":
        return "none"
    raise ValueError(f"unknown backend: {backend!r}")


def _resize_short_side(frames: np.ndarray, short_side: int, backend: ResizeBackend) -> np.ndarray:
    """Resize ``[T,H,W,C]`` uint8 frames so that the short side equals ``short_side``."""
    _, h, w, _ = frames.shape
    if h == 0 or w == 0:
        raise ValueError(f"clip with a zero dimension: {frames.shape}")
    if short_side <= 0:
        raise ValueError(f"short_side must be > 0, not {short_side}")
    # Same as Meta `get_resize_sizes`: truncation with int, not round.
    if w < h:
        new_h, new_w = int(short_side * h / w), short_side
    else:
        new_h, new_w = short_side, int(short_side * w / h)
    if (new_h, new_w) == (h, w):
        return frames
    if backend == "cv2":
        import cv2

        # INTER_LINEAR without antialias, as in the Meta eval chain.
        return np.stack(
            [cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_LINEAR) for f in frames]
        )
    if backend == "torchvision":
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import resize

        x = torch.from_numpy(frames).permute(0, 3, 1, 2)
        y = resize(
            x,
            [new_h, new_w],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        return y.permute(0, 2, 3, 1).numpy()
    raise ValueError(
        f"backend='{backend}' does not resize, but the clip is {h}x{w} and the requested "
        f"short side is {short_side}. With backend='none' the sizes must already match."
    )


def _squash(frames: np.ndarray, spec: PreprocessSpec) -> np.ndarray:
    """Resize ``[T,H,W,C]`` uint8 frames to ``[T,size,size,C]`` without keeping the aspect ratio.

    :param frames: ``[T,H,W,3]`` uint8 clip.
    :param spec: spec providing ``size`` and ``backend``; ``short_side`` must equal ``size``.
    :return: ``[T,size,size,3]`` uint8 clip.
    :raises ValueError: on an inconsistent spec, a non-positive size or a backend that
        does not resize.
    """
    if spec.short_side != spec.size:
        raise ValueError(
            f"geometry='squash' resizes the whole frame to {spec.size}x{spec.size}, "
            f"so short_side must equal size ({spec.size}), not {spec.short_side}"
        )
    if spec.size <= 0:
        raise ValueError(f"size must be > 0, not {spec.size}")
    _, h, w, _ = frames.shape
    if h == 0 or w == 0:
        raise ValueError(f"clip with a zero dimension: {frames.shape}")
    if (h, w) == (spec.size, spec.size):
        return frames
    if spec.backend == "cv2":
        import cv2

        return np.stack(
            [cv2.resize(f, (spec.size, spec.size), interpolation=cv2.INTER_LINEAR) for f in frames]
        )
    if spec.backend == "torchvision":
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import resize

        x = torch.from_numpy(frames).permute(0, 3, 1, 2)
        y = resize(
            x,
            [spec.size, spec.size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        return y.permute(0, 2, 3, 1).numpy()
    raise ValueError(
        f"backend='{spec.backend}' does not resize, but squash must map "
        f"{h}x{w} to {spec.size}x{spec.size}"
    )


def _center_crop(frames: np.ndarray, size: int) -> np.ndarray:
    _, h, w, _ = frames.shape
    if h < size or w < size:
        raise ValueError(f"crop {size} larger than the clip {h}x{w}")
    # Same as Meta `CenterCrop`: with an odd difference `round` may pick the next pixel.
    top, left = int(round((h - size) / 2.0)), int(round((w - size) / 2.0))
    return frames[:, top : top + size, left : left + size, :]


def _three_views(frames: np.ndarray, size: int) -> tuple[np.ndarray, ...]:
    """Return the three ``EvalVideoTransform`` crops along the longer side."""
    _, h, w, _ = frames.shape
    if h < size or w < size:
        raise ValueError(f"crop {size} larger than the clip {h}x{w}")
    step = (max(h, w) - size) // 2
    if h > w:
        return tuple(frames[:, i * step : i * step + size, :, :] for i in range(3))
    return tuple(frames[:, :, i * step : i * step + size, :] for i in range(3))


def _normalize(frames: np.ndarray, spec: PreprocessSpec) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(frames)).float().div_(255.0)
    mean = torch.tensor(spec.mean, dtype=torch.float32).view(1, 1, 1, -1)
    std = torch.tensor(spec.std, dtype=torch.float32).view(1, 1, 1, -1)
    x = ((x - mean) / std).permute(3, 0, 1, 2)
    return x.half() if spec.dtype == "fp16" else x.float()


def preprocess_clip(frames: np.ndarray, spec: PreprocessSpec) -> torch.Tensor:
    """Turn uint8 pixels into a normalized clip according to ``spec``.

    A single geometry returns ``[C,T,H,W]``; ``three_views`` returns ``[V,C,T,H,W]`` with
    ``V=3``, the views being aggregated after the head as in the Meta eval path.

    :param frames: ``[T,H,W,3]`` uint8 clip.
    :param spec: preprocessing spec.
    :return: the normalized clip.
    :raises TypeError: if the pixels are not uint8.
    :raises ValueError: on a wrong shape, frame count or geometry.
    """
    if frames.dtype != np.uint8:
        raise TypeError(f"expected uint8 pixels, got {frames.dtype}")
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected a [T,H,W,3] clip, got {frames.shape}")
    if frames.shape[0] != spec.num_frames:
        raise ValueError(f"clip with {frames.shape[0]} frames, spec requires {spec.num_frames}")

    if spec.geometry == "none":
        _, h, w, _ = frames.shape
        if (h, w) != (spec.size, spec.size):
            raise ValueError(
                f"geometry='none' leaves pixels untouched, but the clip is {h}x{w} and spec.size="
                f"{spec.size}. Use an explicit geometry ('resize_only', 'center_crop')."
            )
        pieces = (frames,)
    elif spec.geometry == "resize_only":
        pieces = (_resize_short_side(frames, spec.short_side, spec.backend),)
    elif spec.geometry == "center_crop":
        pieces = (
            _center_crop(_resize_short_side(frames, spec.short_side, spec.backend), spec.size),
        )
    elif spec.geometry == "eval_center":
        # The central view uses the same floor as the three views (CenterCrop rounds).
        resized = _resize_short_side(frames, spec.short_side, spec.backend)
        pieces = (_three_views(resized, spec.size)[1],)
    elif spec.geometry == "squash":
        pieces = (_squash(frames, spec),)
    elif spec.geometry == "three_views":
        pieces = _three_views(_resize_short_side(frames, spec.short_side, spec.backend), spec.size)
    else:  # pragma: no cover - closed Literal
        raise ValueError(f"unknown geometry: {spec.geometry!r}")

    normalized = tuple(_normalize(piece, spec) for piece in pieces)
    return normalized[0] if len(normalized) == 1 else torch.stack(normalized)
