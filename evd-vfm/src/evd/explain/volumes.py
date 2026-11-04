"""Spatio-temporal perturbations on the full detector, batch 1.

Volumes are defined in tokens and converted exactly into pixels; overlapping masks and
their unions are choices of our campaign, documented in
docs/varie/interpretabilita_cinque_metodi.md.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import product

import numpy as np
import torch
from torch import Tensor

from evd.explain.perturbation import reference_clip


@dataclass(frozen=True)
class TokenRegion:
    """A perturbation test, possibly a union of boxes.

    :param name: stable test identifier.
    :param family: geometry it belongs to, kept separate in the artifacts.
    :param boxes: exclusive bounds (t0,t1,y0,y1,x0,x1) in tokens.
    """

    name: str
    family: str
    boxes: tuple[tuple[int, int, int, int, int, int], ...]


def token_regions(grid: tuple[int, int, int], settings: dict) -> list[TokenRegion]:
    """Build all valid positions and the special unions.

    :param grid: actual backbone grid, not a pixel partition.
    :param settings: sliding windows with size/stride; special_side for the unions.
    :return: tests in stable order, without padding or truncated volumes.
    :raises ValueError: on unrealizable geometries or an unsupported grid.
    """
    if len(grid) != 3 or any(v < 1 for v in grid):
        raise ValueError("Invalid token grid")
    result = []
    for spec in settings["sliding"]:
        size, stride = tuple(spec["size"]), tuple(spec["stride"])
        if (
            len(size) != 3
            or len(stride) != 3
            or any(s < 1 for s in stride)
            or any(v < 1 or v > g for v, g in zip(size, grid, strict=True))
        ):
            raise ValueError("Volume/stride not compatible with the token grid")
        family = "x".join(map(str, size)) + "_s" + "x".join(map(str, stride))
        starts = [range(0, g - v + 1, s) for g, v, s in zip(grid, size, stride, strict=True)]
        for t, y, x in product(*starts):
            box = (t, t + size[0], y, y + size[1], x, x + size[2])
            result.append(TokenRegion(f"{family}_t{t}_y{y}_x{x}", family, (box,)))
    side = settings.get("special_side")
    if side is not None:
        t, h, w = grid
        if h != w or h % side or h < 2 * side or (h - side) % 2:
            raise ValueError("Special unions require a square grid divisible by the side")
        starts = range(0, h, side)
        center = (h - side) // 2

        def box(y, x):
            return (0, t, y, y + side, x, x + side)

        horizontal = tuple(box(center, x) for x in starts)
        vertical = tuple(box(y, center) for y in starts)
        diagonal = tuple(box(i, i) for i in starts)
        antidiagonal = tuple(box(i, h - side - i) for i in starts)
        top, bottom = tuple(box(0, x) for x in starts), tuple(box(h - side, x) for x in starts)
        left, right = tuple(box(y, 0) for y in starts), tuple(box(y, w - side) for y in starts)
        specials = {
            "central_horizontal": horizontal,
            "central_vertical": vertical,
            "central_cross": horizontal + vertical,
            "diagonal": diagonal,
            "antidiagonal": antidiagonal,
            "diagonals": diagonal + antidiagonal,
            "top": top,
            "bottom": bottom,
            "left": left,
            "right": right,
            "sides": top + bottom + left + right,
        }
        result.extend(TokenRegion(name, "special", boxes) for name, boxes in specials.items())
    if not result or len({r.name for r in result}) != len(result):
        raise ValueError("Missing or duplicated masks")
    return result


def region_masks(regions: list[TokenRegion], grid: tuple[int, int, int]) -> np.ndarray:
    """:return: boolean masks N,T,H,W; union intersections counted once.
    :raises ValueError: on bounds outside the grid or an empty mask.
    """
    masks = np.zeros((len(regions), *grid), dtype=bool)
    for index, region in enumerate(regions):
        for t0, t1, y0, y1, x0, x1 in region.boxes:
            if not (
                0 <= t0 < t1 <= grid[0] and 0 <= y0 < y1 <= grid[1] and 0 <= x0 < x1 <= grid[2]
            ):
                raise ValueError("Volume bounds outside the grid")
            masks[index, t0:t1, y0:y1, x0:x1] = True
    if not masks.reshape(len(regions), -1).any(1).all():
        raise ValueError("Empty mask")
    return masks


def masked_clip(
    clip: Tensor,
    mask: np.ndarray,
    *,
    mode: str,
    reference: Tensor | None = None,
    blur_kernel: int = 31,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
) -> Tensor:
    """Rebuild the perturbed input of one test from its mask.

    The only place where a token mask becomes replaced pixels, shared by
    :func:`masked_effects` and the perturbed views (:mod:`evd.explain.perturbed`).

    :param mask: boolean ``T,H,W`` token mask, expanded by integer repetition (never
        interpolated) to the pixel geometry.
    :param mode: ``blur``, ``mean``, ``black`` or ``local_freeze``.
    :param reference: fill already computed for the whole clip; computed here if missing
        and required.
    :return: a new clip with the same shape; ``clip`` is not modified.
    :raises ValueError: on an unknown mode.
    """
    if mode not in ("blur", "mean", "black", "local_freeze"):
        raise ValueError(f"Unknown perturbation: {mode}")
    if reference is None and mode != "local_freeze":
        reference = reference_clip(clip, mode, blur_kernel, mean=mean, std=std)
    expanded = torch.as_tensor(np.ascontiguousarray(mask), device=clip.device)
    for axis, (s, g) in enumerate(zip(clip.shape[-3:], mask.shape, strict=True)):
        expanded = expanded.repeat_interleave(s // g, dim=axis)
    if mode == "local_freeze":
        # The centre is the volume's, not the clip's: time is untouched outside the mask.
        active = torch.where(expanded.any((1, 2)))[0]
        center = (int(active[0]) + int(active[-1]) + 1) // 2
        source = clip[:, :, center : center + 1].expand_as(clip)
    else:
        source = reference
    return torch.where(expanded[None, None], source, clip)


def masked_effects(
    forward: Callable[[Tensor], Tensor],
    clip: Tensor,
    masks: np.ndarray,
    *,
    mode: str,
    original_logit: float,
    blur_kernel: int = 31,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Perturb each mask once on the full detector.

    :param masks: N,T,H,W token masks; integer expansion, never interpolation.
    :param mode: blur, mean, black or local_freeze, applied to the whole union.
    :param mean: preprocessing mean, needed for true black.
    :param std: preprocessing standard deviation.
    :return: delta-logit vectors and raw logits per test, not per token.
    :raises ValueError: on a non-divisible geometry or a non-finite response.
    """
    if clip.ndim != 5 or clip.shape[0] != 1 or masks.ndim != 4 or not len(masks):
        raise ValueError("Expected batch-1 clips and N,T,H,W masks")
    if mode not in ("blur", "mean", "black", "local_freeze"):
        raise ValueError(f"Unknown perturbation: {mode}")
    if any(s % g for s, g in zip(clip.shape[-3:], masks.shape[-3:], strict=True)):
        raise ValueError("Token grid not divisible into the clip pixels")
    reference = (
        reference_clip(clip, mode, blur_kernel, mean=mean, std=std)
        if mode != "local_freeze"
        else None
    )
    values = []
    with torch.no_grad():
        for mask in masks:
            altered = masked_clip(clip, mask, mode=mode, reference=reference)
            values.append(float(forward(altered).float().item()))
    logits = np.asarray(values, dtype=np.float32)
    if not np.isfinite(logits).all() or not np.isfinite(original_logit):
        raise ValueError("Non-finite logit: artifact rejected")
    return original_logit - logits, logits


def overlap_projection(effects: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the descriptive mean of the effects of the tests covering each token.

    :return: map and coverage count; zero without coverage is not evidence of no effect.
    """
    coverage = masks.sum(0)
    total = np.einsum("n,nthw->thw", effects, masks, optimize=True)
    return np.divide(total, coverage, out=np.zeros_like(total), where=coverage > 0), coverage


def volume_slices(shape: tuple[int, ...], grid: tuple[int, int, int]) -> list[tuple[slice, ...]]:
    """Partition T,H,W without overlaps or gaps.

    :return: B,C,T,H,W slices in temporal order, then row and column.
    :raises ValueError: if the grid does not divide the clip exactly.
    """
    if (
        len(shape) != 3
        or len(grid) != 3
        or any(g < 1 or s % g for s, g in zip(shape, grid, strict=True))
    ):
        raise ValueError("The grid must divide T,H,W exactly")
    step = [s // g for s, g in zip(shape, grid, strict=True)]
    return [
        (
            slice(None),
            slice(None),
            *(slice(i * s, (i + 1) * s) for i, s in zip(index, step, strict=True)),
        )
        for index in product(*(range(g) for g in grid))
    ]


def global_motion(clip: Tensor, *, seeds: tuple[int, ...]) -> Iterator[tuple[str, Tensor, dict]]:
    """Yield freeze, reversal and permutations of the temporal samples.

    :param clip: preprocessed 1,C,T,H,W clip; never modified in place.
    :param seeds: one local seed per permutation, independent of the global RNG.
    :return: name, modified clip and reordering metadata.
    """
    t = clip.shape[2]
    middle = t // 2
    yield "freeze", clip[:, :, middle : middle + 1].expand_as(clip), {"center_sample": middle}
    yield "reverse", clip.flip(2), {"order": list(reversed(range(t)))}
    for seed in seeds:
        order = np.random.default_rng(seed).permutation(t)
        yield (
            f"permutation_{seed}",
            clip[:, :, order.tolist()],
            {"seed": seed, "order": order.tolist()},
        )


def volume_effects(
    forward: Callable[[Tensor], Tensor],
    clip: Tensor,
    *,
    mode: str,
    original_logit: float,
    grid: tuple[int, int, int] = (4, 6, 6),
    blur_kernel: int = 31,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure original minus perturbed logit for each volume.

    :param mode: blur, mean or local_freeze (temporal centre of the volume).
    :param forward: full detector, never the head alone.
    :return: signed map on the grid and perturbed logits, both raw.
    :raises ValueError: on an unknown mode, an invalid grid or a non-finite logit.
    """
    volume_slices(tuple(clip.shape[-3:]), grid)  # validates the historical contract
    masks = np.eye(int(np.prod(grid)), dtype=bool).reshape(-1, *grid)
    effects, logits = masked_effects(
        forward,
        clip,
        masks,
        mode=mode,
        original_logit=original_logit,
        blur_kernel=blur_kernel,
    )
    return effects.reshape(grid), logits.reshape(grid)
