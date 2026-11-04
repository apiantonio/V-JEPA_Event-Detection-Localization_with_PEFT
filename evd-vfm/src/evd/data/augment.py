"""Training augmentation: the official V-JEPA 2.1 frozen-probe chain.

Reproduces ``vjepa2/evals/video_classification_frozen/eval.py`` (``make_dataloader``)
and the operation order of ``utils.py`` (RandAugment on full frames, normalization,
then the crop). The transforms themselves are vendored verbatim in ``evd/vendor/``;
this module only orchestrates order, seeding and type conversion.

Augmentation applies to training clips only; evaluation uses the deterministic
:func:`evd.data.preprocess.preprocess_clip`. Unlike upstream, every window gets a
seed derived from (run seed, window id, draw count) and the global generators
are restored afterwards, so runs are reproducible and the global state is not
consumed.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
import torch
from torchvision import transforms

from evd.types import AugmentSpec, PreprocessSpec
from evd.utils.hashing import stable_hash
from evd.vendor.vjepa21_randaugment import NAME_TO_OP, RandAugment
from evd.vendor.vjepa21_randerase import RandomErasing
from evd.vendor.vjepa21_video_transforms import (
    create_random_augment,
    horizontal_flip,
    random_resized_crop,
)

if TYPE_CHECKING:
    from evd.types import WindowSpec

#: Empty chain: deterministic preprocessing only. It is a declared value (not
#: ``None``) because "no augmentation" ends up in the run id and metadata.
NO_AUGMENTATION = AugmentSpec(
    id="none",
    random_resize_scale=(1.0, 1.0),
    random_resize_ratio=(1.0, 1.0),
    rand_augment=None,
    horizontal_flip=False,
    reprob=0.0,
)


def randaugment_chain(spec: PreprocessSpec, augment: AugmentSpec):  # type: ignore[no-untyped-def]
    """Return the RandAugment chain of the spec, optionally reduced.

    Without ``rand_augment_exclude`` this is exactly upstream ``create_random_augment``;
    otherwise the same object is built and the named operations are removed.

    :param spec: preprocessing spec, whose target size scales ``translate_const``.
    :param augment: declared augmentation chain.
    :return: the ``Compose`` transform applied to the list of PIL frames.
    :raises ValueError: if an excluded name is unknown, the policy uses choice weights,
        or no operation is left.
    """
    chain = create_random_augment(
        input_size=(spec.size, spec.size),
        auto_augment=augment.rand_augment,
        interpolation="bicubic",
    )
    if not augment.rand_augment_exclude:
        return chain
    randaugment = chain.transforms[0]
    if randaugment.choice_weights is not None:
        raise ValueError(
            "rand_augment_exclude on a policy with choice weights ('w' in the string): "
            "removing operations would leave probabilities that do not sum to 1. "
            "The official policy does not use 'w'"
        )
    unknown = [name for name in augment.rand_augment_exclude if name not in NAME_TO_OP]
    if unknown:
        raise ValueError(
            f"rand_augment_exclude names unknown operations: {unknown}. "
            f"Use timm names (e.g. 'Rotate', 'ShearX', 'TranslateXRel')"
        )
    forbidden = {NAME_TO_OP[name] for name in augment.rand_augment_exclude}
    kept = [op for op in randaugment.ops if op.aug_fn not in forbidden]
    removed = len(randaugment.ops) - len(kept)
    if removed != len(forbidden):
        # The repertoire depends on the policy, so a valid name may still be absent here.
        raise ValueError(
            f"rand_augment_exclude={augment.rand_augment_exclude} should remove "
            f"{len(forbidden)} operations from policy {augment.rand_augment!r} but removed "
            f"{removed}: some names are not in this policy's repertoire"
        )
    if not kept:
        raise ValueError(
            "rand_augment_exclude emptied the repertoire: use `rand_augment: null` to "
            "disable augmentation explicitly"
        )
    return transforms.Compose([RandAugment(kept, randaugment.num_layers, None)])


@contextmanager
def _isolated_rng(seed: int) -> Iterator[None]:
    """Seed the three global generators and restore them afterwards.

    The vendored functions draw from ``random``, ``numpy.random`` and torch; seeding
    without restoring would also shift head initialization and loader shuffling.

    :param seed: seed of this window, not of the run.
    """
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def window_seed(run_seed: int, window: WindowSpec, draw: int, worker: int = 0) -> int:
    """Return the augmentation seed of a window at a given draw.

    :param run_seed: seed of the cell.
    :param window: the window, identified by its ``window_id``.
    :param draw: how many times this window has already been served.
    :param worker: ``DataLoader`` worker index (each worker keeps its own counter).
    :return: a 63-bit integer, stable across platforms and runs.
    """
    digest = stable_hash([int(run_seed), window.window_id, int(draw), int(worker)])
    return int(digest[:16], 16) % (2**63)


def augment_clip(
    frames: np.ndarray,
    spec: PreprocessSpec,
    augment: AugmentSpec,
    *,
    seed: int,
) -> torch.Tensor:
    """Turn native uint8 frames into an augmented, normalized ``[C,T,H,W]`` clip.

    Mirrors ``VideoTransform.__call__`` with ``training=True``
    (``vjepa2/evals/video_classification_frozen/utils.py``): RandAugment on full
    frames, ``ToTensor``, normalization, ``random_resized_crop`` to the target size,
    flip, random erasing. With ``augment.crop`` false the crop is replaced by a
    resize of the whole frame. Only ``size``, ``num_frames``, ``mean``/``std`` and
    ``dtype`` of ``spec`` are used.

    :param frames: native clip ``[T,H,W,3]`` uint8.
    :param spec: backbone preprocessing (size, normalization, output dtype).
    :param augment: augmentation chain; an inactive chain is an error (use
        ``preprocess_clip`` instead).
    :param seed: seed of this window (:func:`window_seed`).
    :return: ``[C,T,H,W]`` tensor, ``float32`` or ``float16`` according to ``spec.dtype``.
    :raises TypeError: if the pixels are not uint8.
    :raises ValueError: on a wrong shape or frame count, or an empty chain.
    """
    if frames.dtype != np.uint8:
        raise TypeError(f"expected uint8 pixels, got {frames.dtype}")
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected a [T,H,W,3] clip, got {frames.shape}")
    if frames.shape[0] != spec.num_frames:
        raise ValueError(f"clip with {frames.shape[0]} frames, spec requires {spec.num_frames}")
    if not augment.activate:
        raise ValueError(
            "augment_clip called with the empty chain: use preprocess_clip for the "
            "deterministic path"
        )

    with _isolated_rng(seed):
        buffer: list = [transforms.ToPILImage()(frame) for frame in frames]

        if augment.rand_augment:
            # `input_size` only scales the RandAugment translation: upstream passes the target
            # size even though the images are still native.
            buffer = randaugment_chain(spec, augment)(buffer)

        clip = torch.stack([transforms.ToTensor()(img) for img in buffer])
        clip = clip.permute(0, 2, 3, 1)
        mean = torch.tensor(spec.mean, dtype=torch.float32)
        deviation = torch.tensor(spec.std, dtype=torch.float32)
        clip = (clip - mean) / deviation
        clip = clip.permute(3, 0, 1, 2)

        if augment.crop:
            clip = random_resized_crop(
                images=clip,
                target_height=spec.size,
                target_width=spec.size,
                scale=augment.random_resize_scale,
                ratio=augment.random_resize_ratio,
            )
        else:
            # No crop: the whole frame is resized to size x size. Fixing scale and ratio to
            # (1, 1) would not work: on a 4:3 frame the crop falls back to a center crop.
            clip = torch.nn.functional.interpolate(
                clip,
                size=(spec.size, spec.size),
                mode="bilinear",
                align_corners=False,
            )

        if augment.horizontal_flip:
            clip, _ = horizontal_flip(0.5, clip)

        if augment.reprob > 0:
            # `RandomErasing` works per image, so time must be dimension 0; with `cube=True`
            # the erased rectangle is the same in every frame.
            clip = clip.permute(1, 0, 2, 3)
            clip = RandomErasing(
                augment.reprob,
                mode="pixel",
                max_count=1,
                num_splits=1,
                device="cpu",
            )(clip)
            clip = clip.permute(1, 0, 2, 3)

    return clip.half() if spec.dtype == "fp16" else clip.float()
