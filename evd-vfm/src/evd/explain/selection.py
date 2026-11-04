"""Which mask is shown and how it is chosen: a single rule, without torch.

Shared by the comparison renderer (``apps/demo/explanation_advanced.py``) and the
perturbed views (:mod:`evd.explain.perturbed`), so the video and the map describe the
same mask. Torch-free because ``apps/`` imports it (ADR-015).

Governed by docs/varie/interpretabilita_cinque_metodi.md.
"""

from __future__ import annotations

import numpy as np

#: Perturbative method -> suffix of the saved vectors and artifact carrying them.
PERTURBATIVE = {
    "occlusion_blur": ("blur", "occlusion"),
    "occlusion_mean": ("mean", "occlusion"),
    "occlusion_black": ("black", "occlusion"),
    "motion_local_freeze": ("local_freeze", "motion_ablation"),
}


def local_family(regions: list[dict]) -> str:
    """:return: the family with the fewest tokens per test, derived from the artifacts.
    :raises ValueError: if there is no perturbative region.
    """
    if not regions:
        raise ValueError("No perturbative region: inconsistent artifact")
    return min(regions, key=lambda r: r["volume_tokens"])["family"]


def mask_choice(effects: np.ndarray, regions: list[dict], famiglia: str) -> dict:
    """Return the maximum-effect mask among the localized ones only.

    The global maximum would almost always fall on a special union (``diagonals`` covers
    4,608 tokens, half a frame) and would localize nothing; it is kept in the result as
    ``global_maximum``.

    :param effetti: logit deltas per test, in the order of ``regions``.
    :param regions: test descriptions, with ``family`` and ``volume_tokens``.
    :param famiglia: result of :func:`local_family`.
    :return: index, name, volume and effect of the choice, plus ``global_maximum``.
    :raises ValueError: if the family does not appear among the regions.
    """
    if len(effects) != len(regions):
        raise ValueError("Effects and regions have different lengths")
    localized = np.array([r["family"] == famiglia for r in regions])
    if not localized.any():
        raise ValueError(f"Family '{famiglia}' missing from the regions")
    index = int(np.flatnonzero(localized)[np.argmax(effects[localized])])
    global_ = int(np.argmax(effects))
    return {
        "mask_index": index,
        "mask": regions[index]["name"],
        "volume_tokens": int(regions[index]["volume_tokens"]),
        "effect": float(effects[index]),
        "global_maximum": {
            "mask": regions[global_]["name"],
            "volume_tokens": int(regions[global_]["volume_tokens"]),
            "effect": float(effects[global_]),
        },
    }
