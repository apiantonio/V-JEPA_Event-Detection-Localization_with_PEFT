"""Make a detector input visible: IG references and clips in pixels.

An Integrated Gradients attribution is the integral from a reference to the real clip,
so the reference is shown too. It is rebuilt in pixels by calling
:func:`evd.explain.perturbation.reference_clip`, the function the detector actually
used. Unlike ``apps.demo.explanation_advanced.visual_reference`` (a second,
torch-free implementation kept honest by a test), this matters for ``noise``: the
noise comes from ``torch.randn`` with a seeded Generator, which NumPy cannot reproduce.

Governed by docs/varie/interpretabilita_cinque_metodi.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from evd.explain.perturbation import reference_clip
from evd.types import WindowSpec


def renormalize(rgb: np.ndarray, mean: tuple[float, ...], std: tuple[float, ...]) -> torch.Tensor:
    """Turn the samples saved for display into the tensor consumed by the detector.

    The round trip is not exact: uint8 samples are rounded to 1/255. Pixel-dependent
    references (``blur``, ``mean``) differ by at most 1/255 per channel; ``black`` and
    ``noise`` do not depend on the clip values and stay exact.

    :param rgb: ``[T, H, W, 3]`` uint8 samples in RGB order.
    :param mean: per-channel normalization mean, from ``preprocess``.
    :param std: per-channel standard deviation, from ``preprocess``.
    :return: normalized clip ``[1, 3, T, H, W]`` float32.
    """
    values = rgb.astype(np.float32) / 255.0
    normalized = (values - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(normalized).permute(3, 0, 1, 2)[None].contiguous()


def denormalize(
    clip: torch.Tensor, mean: tuple[float, ...], std: tuple[float, ...]
) -> tuple[np.ndarray, float]:
    """Turn the detector tensor into pixels, reporting how much had to be clipped.

    Gaussian noise in normalized space is unbounded, so part of it is clipped to the
    extremes; the clipped fraction is returned and stored in the manifest. Only clipping
    that changes an 8-bit pixel value is counted (float round trips leave blacks at
    ``-4.9e-06``).

    :param clip: normalized clip ``[1, 3, T, H, W]``.
    :return: ``[T, H, W, 3]`` uint8 RGB samples and the fraction of pixels actually moved
        by clipping.
    """
    values = clip[0].permute(1, 2, 3, 0).float().numpy()
    pixel = values * np.asarray(std, dtype=np.float32) + np.asarray(mean, dtype=np.float32)
    samples = np.rint(np.clip(pixel, 0, 1) * 255)
    cut_items = float(np.mean(samples != np.rint(pixel * 255)))
    return samples.astype(np.uint8), cut_items


def ig_reference(
    rgb: np.ndarray,
    fill: str,
    *,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    blur_kernel: int = 31,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Return the reference of an IG baseline in pixels, from the detector function.

    :param rgb: real clip samples, ``[T, H, W, 3]`` uint8 RGB.
    :param fill: ``blur``, ``mean``, ``black`` or ``noise``.
    :param seed: required by ``noise``; it is ``seed_from_window_id(window_id)``.
    :return: reference samples and diagnostics (clipped fraction, exactness).
    :raises ValueError: on an unknown fill, or ``noise`` without a seed.
    """
    clip = renormalize(rgb, mean, std)
    reference = reference_clip(clip, fill, blur_kernel, mean=mean, std=std, seed=seed)
    samples, cut_items = denormalize(reference, mean, std)
    return samples, {
        "fill": fill,
        "clipped_fraction": cut_items,
        # On black and noise the reference does not read the clip values, so the uint8 rounding
        # of the samples does not affect it.
        "exact": fill in ("black", "noise"),
        "seed": seed,
    }


def window_from_id(window_id: str, modello: WindowSpec) -> WindowSpec:
    """Rebuild a window from its identifier, and verify it again.

    The ``window_id`` carries video, end frame and number of samples but only a hash of
    the temporal profile, so offsets are taken from a window of the same campaign. The id
    of the rebuilt window is recomputed and compared with the requested one.

    :param window_id: id of the form ``video@end x samples i profile``.
    :param modello: a campaign window providing the offsets.
    :return: the window, with the same ``window_id`` as requested.
    :raises ValueError: on an unparsable id or a mismatching reconstruction.
    """
    head, _, tail = window_id.partition("@")
    if not tail or "x" not in tail:
        raise ValueError(f"window_id cannot be parsed: {window_id}")
    text_end, _, rest = tail.partition("x")
    text_samples = rest.split("i")[0].split("s")[0]
    try:
        end, samples = int(text_end), int(text_samples)
    except ValueError as error:
        raise ValueError(f"window_id cannot be parsed: {window_id}") from error
    offsets = tuple(modello.end_frame - index for index in modello.explicit_frame_indices)
    if len(offsets) != samples:
        raise ValueError(f"Template window with {len(offsets)} samples, the id requires {samples}")
    rebuilt = WindowSpec(
        video_id=head,
        end_frame=end,
        num_frames=samples,
        stride_frames=modello.stride_frames,
        explicit_frame_indices=tuple(end - offset for offset in offsets),
    )
    if rebuilt.window_id != window_id:
        raise ValueError(
            f"Rebuilt window differs from the requested one: "
            f"{rebuilt.window_id} instead of {window_id}"
        )
    return rebuilt


def window_from_grid(
    window_id: str,
    video: Path,
    protocollo: dict,
    preprocess: dict,
    *,
    validazione: bool = False,
) -> WindowSpec:
    """Find a window by regenerating the grid of its own video.

    The authoritative reconstruction, needed when the window belongs to a video other than
    the campaign one: :func:`window_from_id` only works for videos sharing the temporal
    profile (same native fps). The grid is rebuilt with
    :func:`~evd.protocol.chunks.sampler_for_protocol` on the ``VideoMeta`` probed by the
    decoder; no model is built (ADR-015).

    :param window_id: requested id, ``video@end x samples i profile``.
    :param video: video file of the reference, not of the explained video.
    :param protocollo: resolved ``protocol`` group, from ``source_config.yaml``.
    :param preprocess: ``preprocess`` block of the campaign, verbatim.
    :param validazione: whether the campaign evaluated validation (contiguous
        ``val_hop_s`` grid in ``edp_v2``).
    :return: the window, with exactly the requested ``window_id``.
    :raises ValueError: if the grid of that video does not contain the id.
    """
    from evd.data.decoder import make_decoder
    from evd.data.preprocess import PreprocessSpec
    from evd.protocol.chunks import sampler_for_protocol

    spec = PreprocessSpec(**dict(preprocess))
    sampler = sampler_for_protocol(spec, protocollo, validazione=validazione)
    windows = list(sampler.windows_for_video(make_decoder("pyav").probe(str(video))))
    for window in windows:
        if window.window_id == window_id:
            return window
    raise ValueError(
        f"Window {window_id} missing from the grid of {Path(video).name}: "
        f"{len(windows)} windows generated, the first "
        f"{windows[0].window_id if windows else '(none)'}"
    )


def window_samples(video: Path, finestra: WindowSpec, preprocess: dict) -> np.ndarray:
    """Decode and preprocess a window, and bring it back to pixels.

    Used by the ``tn_same_video`` baseline, whose reference is another real clip, decoded
    with the detector path; no model is built (ADR-015).

    :param video: video file containing the window.
    :param preprocess: ``preprocess`` block of the campaign, verbatim.
    :return: ``[T, H, W, 3]`` uint8 RGB samples.
    """
    from evd.data.dataset import WindowDataset, WindowItem
    from evd.data.decoder import make_decoder
    from evd.data.preprocess import PreprocessSpec

    spec = PreprocessSpec(**preprocess)
    clip = WindowDataset([WindowItem(finestra, str(video))], make_decoder("pyav"), spec)[0]["clip"]
    samples, _ = denormalize(clip[None], spec.mean, spec.std)
    return samples
