"""The eleven perturbed views of the input, and the maps computed on them.

The ``explain/`` methods answer "where does the model look on this clip". Here the clip
is the one the detector saw when the score changed (occluded, frozen, reversed,
permuted): the same maps computed there show where the explanation moves when the input
changes.

There are eleven views with the three campaign fills (four localized, seven global):
localized views use, per window, the maximum-effect mask among the localized tests (the
one shown by the comparison video, :mod:`evd.explain.selection`), not all 45.

Requested on 14 September 2026; governed by
docs/varie/interpretabilita_cinque_metodi.md.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from torch import Tensor

from evd.explain.boxes import concept_mask, top_fraction_mask
from evd.explain.selection import PERTURBATIVE, local_family, mask_choice
from evd.explain.volumes import global_motion, masked_clip

#: Prefix of the map names produced here; the renderer uses it to recover the base
#: method, whose colormap, sign and colour scale are kept.
PREFIX = "pert__"


def view_name(vista: str, metodo: str) -> str:
    """:return: the name under which the map of ``metodo`` on view ``vista`` is saved."""
    return f"{PREFIX}{vista}__{metodo}"


def vtcd_concept_view(
    clip: Tensor,
    concept_ids: np.ndarray,
    *,
    concept_id: int,
    mean: tuple[float, ...] | None,
    std: tuple[float, ...] | None,
) -> tuple[Tensor, np.ndarray, dict]:
    """Black out the tokens assigned to a global VTCD concept.

    The mask stays on the token grid and is expanded by :func:`masked_clip`, as for the
    existing occlusions. ``black`` is true black in RGB space, then normalized for the
    backbone (not a zeroed normalized tensor).
    """
    mask = concept_mask(concept_ids, concept_id=concept_id)
    altered = masked_clip(clip, mask, mode="black", mean=mean, std=std)
    receipt = {
        "kind": "locale",
        "mode": "black",
        "mask": f"global VTCD: concept {concept_id}",
        "concept_id": concept_id,
        "concept_scope": "global",
        "source_map": "vtcd_ids",
        "volume_tokens": int(mask.sum()),
        "total_tokens": int(mask.size),
        "coverage": float(mask.mean()),
    }
    return altered, mask, receipt


def top_attribution_view(
    clip: Tensor,
    map_data: np.ndarray,
    *,
    frazione: float,
    mean: tuple[float, ...] | None,
    std: tuple[float, ...] | None,
    sorgente: str = "ig_ensemble",
    absolute: bool = False,
) -> tuple[Tensor, np.ndarray, dict]:
    """Black out the tokens with the highest attribution of the window.

    The counterpart of :func:`vtcd_concept_view` for continuous maps: the mask is obtained
    by ranking the map and cutting by count (:func:`evd.explain.boxes.top_fraction_mask`),
    the same function used to draw bounding boxes. :func:`~evd.explain.volumes.masked_clip`
    remains the only place where a token mask becomes replaced pixels.

    :param clip: preprocessed clip, batch 1.
    :param mappa: attribution map on the ``[T', H', W']`` grid.
    :param frazione: fraction of tokens to black out, in ``(0, 1]``.
    :param sorgente: map name, for the receipt.
    :param assoluto: rank by magnitude instead of signed value.
    :return: ``(perturbed clip, mask, receipt)``.
    :raises ValueError: on a non-3D map, an out-of-range fraction or non-finite values.
    """
    mask, detail = top_fraction_mask(map_data, frazione=frazione, absolute=absolute)
    altered = masked_clip(clip, mask, mode="black", mean=mean, std=std)
    receipt = {
        "kind": "locale",
        "mode": "black",
        "mask": f"attribution {sorgente}: top {frazione:.0%} of tokens",
        "source_map": sorgente,
        "volume_tokens": int(mask.sum()),
        **detail,
    }
    return altered, mask, receipt


def decompose(nome: str) -> tuple[str, str] | None:
    """:return: ``(view, method)`` if ``nome`` is a perturbed map, otherwise ``None``."""
    if not nome.startswith(PREFIX):
        return None
    view, _, method = nome[len(PREFIX) :].partition("__")
    return (view, method) if method else None


def chosen_masks(siblings: dict, regions: list[dict]) -> dict[str, dict]:
    """Return, for each localized method, the mask the view will use.

    :param siblings: arrays already computed for this window, per method (``occlusion``
        and ``motion_ablation``): ``effects_*`` and ``masks`` are needed.
    :param regions: test descriptions, from the ``occlusion`` receipt.
    :return: method to choice, with ``mask_array`` added; only for the fills actually
        measured.
    :raises KeyError: if the perturbations were not computed first.
    """
    family = local_family(regions)
    choices = {}
    for method, (suffix, artifact) in PERTURBATIVE.items():
        if artifact not in siblings:
            raise KeyError(
                f"Perturbed views require '{artifact}' already computed on this "
                f"window: run occlusion and motion_ablation before perturbed_views"
            )
        arrays = siblings[artifact]
        if f"effects_{suffix}" not in arrays:
            # Fill not measured in this campaign (``fills`` is a config choice): the view is skipped.
            continue
        choice = mask_choice(arrays[f"effects_{suffix}"], regions, family)
        choice["logit"] = float(arrays[f"logits_{suffix}"][choice["mask_index"]])
        choice["score"] = float(arrays[f"scores_{suffix}"][choice["mask_index"]])
        choice["mask_array"] = np.asarray(siblings["occlusion"]["masks"][choice["mask_index"]])
        choices[method] = choice
    return choices


def viste(
    clip: Tensor,
    choices: dict[str, dict],
    *,
    blur_kernel: int,
    mean: tuple[float, ...] | None,
    std: tuple[float, ...] | None,
    motion_seeds: tuple[int, ...],
) -> Iterator[tuple[str, Tensor, dict]]:
    """Yield the eleven perturbed clips, in saving order.

    :param choices: result of :func:`chosen_masks`.
    :param motion_seeds: seeds of the global permutations, from the settings.
    :return: ``(view, clip, receipt)``; the original clip is not among them.
    """
    for method, choice in choices.items():
        mode = (
            "local_freeze" if method == "motion_local_freeze" else method.removeprefix("occlusion_")
        )
        altered = masked_clip(
            clip,
            choice["mask_array"],
            mode=mode,
            blur_kernel=blur_kernel,
            mean=mean,
            std=std,
        )
        yield (
            method,
            altered,
            {
                "kind": "locale",
                "mode": mode,
                "mask": choice["mask"],
                "mask_index": choice["mask_index"],
                "volume_tokens": choice["volume_tokens"],
                "perturbed_logit": choice["logit"],
                "perturbed_score": choice["score"],
                "logit_difference": choice["effect"],
                "global_maximum": choice["global_maximum"],
            },
        )
    for name, altered, receipt in global_motion(clip, seeds=motion_seeds):
        yield name, altered, {"kind": "globale", **receipt}
