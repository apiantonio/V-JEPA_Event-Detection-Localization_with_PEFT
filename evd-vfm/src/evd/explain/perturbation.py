"""Deletion/insertion on preprocessed pixels, with a new backbone forward."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def reference_clip(
    clip: Tensor,
    fill: str,
    blur_kernel: int = 31,
    *,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
    seed: int | None = None,
) -> Tensor:
    """Return a per-frame spatial reference, with no temporal mixing.

    Also used as Integrated Gradients baseline since 10 September 2026.

    :param clip: B,C,T,H,W, after the affine preprocessing of the detector.
    :param fill: ``blur`` (reflected box filter), ``mean`` (mean frame colour),
        ``black`` (true black pixel, which depends on the normalization) or ``noise``
        (independent seeded Gaussian noise per pixel and frame).
    :param blur_kernel: odd side of the blur filter.
    :param mean: per-channel normalization mean, required by ``fill="black"``.
    :param std: per-channel normalization standard deviation, required by
        ``fill="black"``.
    :param seed: seed for ``fill="noise"``, derived from the window identity (never from
        the global RNG).
    :return: reference clip with the same geometry.
    :raises ValueError: on missing normalization or seed, or an invalid fill or kernel.
    """
    if fill == "mean":
        return clip.mean((-2, -1), keepdim=True).expand_as(clip)
    if fill == "black":
        if mean is None or std is None:
            raise ValueError("fill=black requires the normalization mean and std")
        m = torch.tensor(mean, device=clip.device, dtype=clip.dtype).view(1, -1, 1, 1, 1)
        s = torch.tensor(std, device=clip.device, dtype=clip.dtype).view(1, -1, 1, 1, 1)
        return (-m / s).expand_as(clip).contiguous()
    if fill == "noise":
        if seed is None:
            raise ValueError("fill=noise requires a seed derived from the window")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(tuple(clip.shape), generator=generator, dtype=torch.float32)
        return noise.to(clip.device, clip.dtype)
    if fill != "blur" or blur_kernel < 1 or blur_kernel % 2 != 1:
        raise ValueError("Use fill blur/mean/black/noise and a positive odd kernel")
    b, c, t, h, w = clip.shape
    x = clip.transpose(1, 2).reshape(b * t, c, h, w)
    p = blur_kernel // 2
    if p >= min(h, w):
        raise ValueError("Kernel too large for the clip")
    x = F.avg_pool2d(F.pad(x, (p, p, p, p), mode="reflect"), blur_kernel, stride=1)
    return x.reshape(b, t, c, h, w).transpose(1, 2)


def perturbation_curve(
    forward: Callable[[Tensor], Tensor],
    clip: Tensor,
    relevance: np.ndarray,
    *,
    fill: str,
    grid: tuple[int, int, int] = (4, 6, 6),
    fractions: tuple[float, ...] = (0, 0.1, 0.2, 0.3, 0.5, 0.75, 1),
    opposition: bool = False,
    random_seed: int | None = None,
    blur_kernel: int = 31,
) -> list[dict]:
    """Rank equal-volume regions and rerun the detector on the modified pixels.

    :param forward: full detector returning a logit.
    :param clip: original clip, batch 1.
    :param relevance: raw map T',H',W'.
    :return: logit and score per fraction, separated for deletion and insertion.
    :raises ValueError: on a grid not dividing the clip, non-finite maps or invalid
        fractions.
    """
    shape = clip.shape[-3:]
    if any(s % g for s, g in zip(shape, grid, strict=True)):
        raise ValueError("The perturbation grid must divide the clip exactly")
    if not np.isfinite(relevance).all() or any(not 0 <= f <= 1 for f in fractions):
        raise ValueError("Non-finite maps or fractions outside [0,1]")
    if (
        not fractions
        or fractions[0] != 0
        or fractions[-1] != 1
        or tuple(sorted(set(fractions))) != fractions
    ):
        raise ValueError("Fractions must increase strictly from 0 to 1")
    ranks = (
        F.adaptive_avg_pool3d(torch.as_tensor(relevance).float()[None, None], grid)
        .flatten()
        .numpy()
    )
    order = np.argsort(ranks if opposition else -ranks, kind="stable")
    if random_seed is not None:
        order = np.random.default_rng(random_seed).permutation(len(order))
    base = reference_clip(clip, fill, blur_kernel)
    rows = []
    with torch.no_grad():
        for fraction in fractions:
            mask = torch.zeros(len(order), device=clip.device)
            count = int(round(fraction * len(order)))
            mask[torch.as_tensor(order[:count].copy(), device=clip.device)] = 1
            mask = F.interpolate(mask.reshape(1, 1, *grid), size=shape, mode="nearest")
            for mode, perturbed in (
                ("deletion", clip * (1 - mask) + base * mask),
                ("insertion", base * (1 - mask) + clip * mask),
            ):
                logit = float(forward(perturbed).float().item())
                rows.append(
                    {
                        "fraction": fraction,
                        "actual_fraction": count / len(order),
                        "mode": mode,
                        "logit": logit,
                        "score": float(torch.sigmoid(torch.tensor(logit))),
                        "fill": fill,
                        "opposition": opposition,
                        "random_seed": random_seed,
                    }
                )
    return rows
