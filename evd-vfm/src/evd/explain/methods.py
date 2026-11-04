"""Registry of the five additional explanation methods; the runner holds none of their
algorithms.

Methods receive two separate functions: the full detector and the head on the final features.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from evd.explain.concepts import class_direction_maps, direction_maps, prototype_maps
from evd.explain.perturbation import perturbation_curve
from evd.explain.perturbed import chosen_masks, view_name, viste
from evd.explain.volumes import (
    TokenRegion,
    global_motion,
    masked_effects,
    overlap_projection,
    region_masks,
    token_regions,
    volume_slices,
)
from evd.explain.vtcd import assign_concepts, cris, slic_regions
from evd.models.registry import Registry


@dataclass(frozen=True)
class MethodContext:
    """Common immutable inputs; no method updates weights.

    :param class_idx: chunk class, ``-1`` on normal chunks; the class dominating the
        overlap with the annotation, as assigned by ``metrics.chunk_records`` and stored
        in ``scores.jsonl`` (not the folder class).
    :param detector: preprocessed clip -> logit of the full detector.
    :param head: tokens -> logit, original precision.
    :param head_fp32: tokens -> logit with the head alone in float32.
    """

    clip: Tensor
    tokens: Tensor
    grid: tuple[int, int, int]
    logit: float
    label: int
    detector: Callable[[Tensor], Tensor]
    head: Callable[[Tensor], Tensor]
    head_fp32: Callable[[Tensor], Tensor]
    references: dict[str, np.ndarray]
    settings: dict
    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None
    # Among the defaulted fields rather than next to `label`: callers build the context
    # positionally.
    class_idx: int = -1
    explain: Callable[[Tensor], tuple[dict[str, np.ndarray], dict, float]] | None = None
    siblings: dict[str, tuple[dict, dict]] | None = None
    source_maps: dict[str, np.ndarray] | None = None


ADVANCED_METHODS = (
    "anomaly_direction",
    "class_direction",
    "class_direction_all",
    "anomaly_prototypes",
    "vtcd_final",
    "vtcd_concepts",
    "vtcd_class",
    "occlusion",
    "motion_ablation",
    "perturbed_views",
    "deletion_insertion",
)
HANDLERS: Registry[Callable[[MethodContext], tuple[dict[str, np.ndarray], dict]]] = Registry(
    "explanation"
)


def _maps(maps: dict[str, np.ndarray], grid: tuple[int, ...]) -> dict[str, np.ndarray]:
    return {f"map__{key}": value.reshape(grid).astype(np.float32) for key, value in maps.items()}


@HANDLERS.register("anomaly_direction")
def anomaly_direction(context: MethodContext) -> tuple[dict[str, np.ndarray], dict]:
    """:return: token projection and logit sensitivity to the unit normal."""
    maps, receipt = direction_maps(context.head_fp32, context.tokens, context.references)
    return _maps(maps, context.grid), receipt


@HANDLERS.register("anomaly_prototypes")
def anomaly_prototypes(context: MethodContext) -> tuple[dict[str, np.ndarray], dict]:
    """:return: two cosines and their signed difference, with fixed prototypes."""
    maps = prototype_maps(context.tokens[0].float().cpu().numpy(), context.references)
    return _maps(maps, context.grid), {"diagnostic_only": True, "labels": "temporal only"}


def _concepts(context: MethodContext) -> tuple[np.ndarray, dict, dict]:
    """Segment the tokens and assign them to the concepts of the frozen dictionary.

    The half of VTCD that does not touch the head (SLIC on tokens and a per-region NNLS
    against the dictionary learned once per checkpoint), shared by the variants with and
    without CRIS. It is also the expensive part: its cost does not depend on the model
    (about 29 s per window), so it is spent in ``slic_regions``, not in forwards.

    :param context: the window, with extracted tokens and references.
    :return: ``(concept per token, raw arrays, receipt)``.
    """
    settings = context.settings["vtcd"]
    regions, descriptors, segmentation = slic_regions(
        context.tokens[0].float().cpu().numpy().reshape(*context.grid, -1), **settings["slic"]
    )
    assignments, coefficients = assign_concepts(descriptors, context.references["dictionary"])
    arrays = {"regions": regions, "coefficients": coefficients, "descriptors": descriptors}
    receipt = {
        "segmentation": segmentation,
        "adaptation": "final level, frozen dictionary; no multi-head/multi-layer replication",
        "all_zero_assignment_regions": np.flatnonzero(~coefficients.any(1)).tolist(),
    }
    return assignments[regions], arrays, receipt


def _concept_maps(concepts: np.ndarray, n_concepts: int) -> dict[str, np.ndarray]:
    """:return: the map of all concepts together and one per concept.

    ``vtcd_ids`` holds concept identifiers, so it is discrete and upscaled by blocks
    (:func:`apps.demo.explanation.interpolation_of`): interpolating would invent concepts.
    """
    maps = {"vtcd_ids": concepts}
    for k in range(n_concepts):
        maps[f"vtcd_concept_{k:02d}"] = (concepts == k).astype(np.float32)
    return maps


@HANDLERS.register("class_direction")
def class_direction(context: MethodContext) -> tuple[dict[str, np.ndarray], dict]:
    """:return: projection and sensitivity along the CAV of this chunk's class.

    ``anomaly_direction`` per class: the direction of the concept the window actually
    carries, chosen window by window (multi-class videos change concept between chunks).
    The header declares which class was used; directions are unit vectors, so projections
    are comparable across classes. On normal chunks (``class_idx`` ``-1``) the maps are
    zero and the status says so.

    :raises ValueError: if the references lack ``class_directions``.
    """
    if "class_directions" not in context.references:
        raise ValueError(
            "class_direction requires per-class CAVs: 'class_directions' is missing from the "
            "references. They are prepared with prepare_class_references()"
        )
    maps, receipt = class_direction_maps(
        context.head_fp32, context.tokens, context.references, context.class_idx
    )
    return _maps(maps, context.grid), receipt


@HANDLERS.register("class_direction_all")
def class_direction_all(context: MethodContext) -> tuple[dict[str, np.ndarray], dict]:
    """:return: every CAV on annotated positive chunks, besides the true class."""
    maps, receipt = class_direction_maps(
        context.head_fp32, context.tokens, context.references, context.class_idx
    )
    checks = {}
    for k in context.references["class_indices"]:
        k = int(k)
        values, check = class_direction_maps(
            context.head_fp32,
            context.tokens,
            context.references,
            k if context.label == 1 else -1,
        )
        maps.update({f"{name}__class_{k:02d}": value for name, value in values.items()})
        checks[str(k)] = check
    return _maps(maps, context.grid), {**receipt, "per_class": checks}


@HANDLERS.register("vtcd_class")
def vtcd_class(context: MethodContext) -> tuple[dict[str, np.ndarray], dict]:
    """Assign regions to the class dictionaries frozen on training data.

    SLIC is computed once per window; IDs are local to the class dictionary and are not
    predictions of the binary head.

    :return: concept IDs and masks per class, plus the true-class view.
    """
    indices = context.references["vtcd_class_indices"]
    dictionaries = context.references["vtcd_class_dictionaries"]
    n_concepts = int(dictionaries.shape[1])
    maps = {}
    arrays = {}
    checks = {}
    if context.label == 1:
        regions, descriptors, segmentation = slic_regions(
            context.tokens[0].float().cpu().numpy().reshape(*context.grid, -1),
            **context.settings["vtcd"]["slic"],
        )
        arrays.update(regions=regions, descriptors=descriptors)
        checks["segmentation"] = segmentation
    for k, dictionary in zip(indices, dictionaries, strict=True):
        k = int(k)
        if context.label == 1:
            ids, coefficients = assign_concepts(descriptors, dictionary)
            concepts = ids[regions]
            arrays[f"coefficients_class_{k:02d}"] = coefficients
            values = _concept_maps(concepts, n_concepts)
        else:
            values = {"vtcd_ids": np.full(context.grid, -1, dtype=np.float32)}
            values.update(
                {f"vtcd_concept_{j:02d}": np.zeros(context.grid) for j in range(n_concepts)}
            )
        maps.update({f"{name}__class_{k:02d}": value for name, value in values.items()})
        if k == context.class_idx:
            maps.update({f"{name}__class_true": value for name, value in values.items()})
    if "vtcd_ids__class_true" not in maps:
        maps["vtcd_ids__class_true"] = np.full(context.grid, -1, dtype=np.float32)
        maps.update(
            {f"vtcd_concept_{j:02d}__class_true": np.zeros(context.grid) for j in range(n_concepts)}
        )
    arrays.update(_maps(maps, context.grid))
    return arrays, {
        **checks,
        "class_idx": context.class_idx,
        "label": context.label,
        "class_indices": indices.tolist(),
        "id_scope": "checkpoint and class; not predictions",
    }


@HANDLERS.register("vtcd_concepts")
def vtcd_concepts(context: MethodContext) -> tuple[dict[str, np.ndarray], dict]:
    """:return: the eight concepts and their stable IDs, without CRIS importance.

    ``vtcd_final`` without the ablation (17 September 2026): concepts are inspected without
    any perturbation. Not a cheap variant, since VTCD time is spent in SLIC. VTCD is
    excluded from perturbed views and from deletion/insertion ordering: an identifier map
    has no importance order.
    """
    concepts, arrays, receipt = _concepts(context)
    arrays.update(
        _maps(_concept_maps(concepts, context.settings["vtcd"]["n_concepts"]), context.grid)
    )
    receipt["cris"] = "not computed: concepts are inspected without ablations"
    return arrays, receipt


@HANDLERS.register("vtcd_final")
def vtcd_final(context: MethodContext) -> tuple[dict[str, np.ndarray], dict]:
    """:return: eight concepts, stable IDs, CRIS importance and all raw masks."""
    settings = context.settings["vtcd"]
    concepts, base, receipt = _concepts(context)
    arrays, cris_receipt = cris(
        context.head,
        context.tokens,
        concepts,
        label=context.label,
        n_concepts=settings["n_concepts"],
        **settings["cris"],
    )
    maps = _concept_maps(concepts, settings["n_concepts"])
    maps["vtcd_importance"] = arrays["importance"][concepts]
    arrays.update(_maps(maps, context.grid))
    arrays.update(base)
    receipt.update(cris_receipt)
    return arrays, receipt


def _volume_arrays(
    context: MethodContext, modes: tuple[str, ...], prefix: str
) -> tuple[dict, dict]:
    """:return: per-mask measurements and descriptive projection with explicit coverage."""
    if "volumes" in context.settings:
        regions = token_regions(context.grid, context.settings["volumes"])
    else:
        # Historical configs: the partition stays readable without migrating artifacts.
        slices = volume_slices(context.grid, tuple(context.settings["volume_grid"]))
        regions = [
            TokenRegion(
                f"legacy_{i}", "legacy", (tuple(v for s in region[2:] for v in (s.start, s.stop)),)
            )
            for i, region in enumerate(slices)
        ]
    masks = region_masks(regions, context.grid)
    arrays = {"masks": masks, "volume_tokens": masks.sum((1, 2, 3)), "coverage": masks.sum(0)}
    families = sorted({region.family for region in regions})
    for mode in modes:
        effect, logits = masked_effects(
            context.detector,
            context.clip,
            masks,
            mode=mode,
            original_logit=context.logit,
            blur_kernel=context.settings["blur_kernel"],
            mean=context.mean,
            std=context.std,
        )
        arrays[f"map__{prefix}_{mode}"] = overlap_projection(effect, masks)[0]
        arrays[f"effects_{mode}"] = effect
        arrays[f"logits_{mode}"] = logits
        arrays[f"scores_{mode}"] = torch.sigmoid(torch.from_numpy(logits)).numpy()
        for family in families:
            indices = [i for i, region in enumerate(regions) if region.family == family]
            projection, coverage = overlap_projection(effect[indices], masks[indices])
            arrays[f"projection_{mode}_{family}"] = projection
            arrays[f"coverage_{family}"] = coverage
    return arrays, {
        "grid": context.grid,
        "schema": 2,
        "regions": [
            {
                "name": r.name,
                "family": r.family,
                "boxes": r.boxes,
                "volume_tokens": int(masks[i].sum()),
            }
            for i, r in enumerate(regions)
        ],
        "difference": "original_logit - perturbed_logit",
        "projection": "mean of the deltas of the tests covering the token, not a pointwise attribution; volumes of different area",
        "uncovered": "zero in the projection with coverage=0 means not measured",
        "full_detector_forwards": len(modes) * len(regions),
    }


@HANDLERS.register("occlusion")
def occlusion(context: MethodContext) -> tuple[dict[str, np.ndarray], dict]:
    """:return: signed effects with blur, frame mean and normalized black."""
    return _volume_arrays(
        context, tuple(context.settings.get("fills", ("blur", "mean"))), "occlusion"
    )


@HANDLERS.register("motion_ablation")
def motion_ablation(context: MethodContext) -> tuple[dict[str, np.ndarray], dict]:
    """:return: local freeze and global ablations with comparable orders and scores."""
    arrays, receipt = _volume_arrays(context, ("local_freeze",), "motion")
    rows = []
    with torch.no_grad():
        for name, altered, motion_receipt in global_motion(
            context.clip, seeds=tuple(context.settings["motion_seeds"])
        ):
            logit = float(context.detector(altered).float().item())
            rows.append(
                {
                    "name": name,
                    **motion_receipt,
                    "original_logit": context.logit,
                    "logit": logit,
                    "score": float(torch.sigmoid(torch.tensor(logit))),
                    "logit_difference": context.logit - logit,
                }
            )
    return arrays, {
        **receipt,
        "globals": rows,
        "full_detector_forwards": receipt["full_detector_forwards"] + len(rows),
        "local_freeze": "central frame of each volume repeated only within the volume",
        "global_freeze": "central frame of the whole clip repeated in all samples",
        "temporal_order": "reverse and five permutations, no global spatial heatmap",
    }


@HANDLERS.register("perturbed_views")
def perturbed_views(context: MethodContext) -> tuple[dict[str, np.ndarray], dict]:
    """Recompute the same maps on each of the eleven perturbed views.

    Depends on ``occlusion`` and ``motion_ablation``: the mask of a localized view is the
    maximum-effect localized one, chosen from the already measured deltas passed in
    ``context.siblings``.

    :return: maps ``map__pert__{view}__{method}`` and a receipt per view.
    :raises ValueError: without an ``explain`` runner, or if the perturbations were not
        computed.
    """
    if context.explain is None or context.siblings is None:
        raise ValueError(
            "perturbed_views requires a runner able to recompute the maps on any "
            "clip: use extend_campaign, not a direct caller"
        )
    arrays_per_method = {name: arrays for name, (arrays, _) in context.siblings.items()}
    if "occlusion" not in context.siblings:
        raise ValueError(
            "perturbed_views requires 'occlusion' and 'motion_ablation' on the same window"
        )
    regions = context.siblings["occlusion"][1]["regions"]
    choices = chosen_masks(arrays_per_method, regions)
    maps: dict[str, np.ndarray] = {}
    receipts: dict[str, dict] = {}
    for view, clip, receipt in viste(
        context.clip,
        choices,
        blur_kernel=context.settings["blur_kernel"],
        mean=context.mean,
        std=context.std,
        motion_seeds=tuple(context.settings["motion_seeds"]),
    ):
        values, checks, logit = context.explain(clip)
        for method, map_ in values.items():
            maps[f"map__{view_name(view, method)}"] = map_.reshape(context.grid).astype(np.float32)
        row = {
            **receipt,
            "logit": logit,
            "score": float(torch.sigmoid(torch.tensor(logit))),
            "logit_difference": context.logit - logit,
            "checks": checks,
        }
        if "perturbed_logit" in receipt:
            # The same input already went through the detector in `occlusion`: two independent
            # measurements of the same quantity, so a gap would mean the rebuilt mask differs.
            row["logit_vs_occlusione"] = abs(logit - receipt["perturbed_logit"])
        receipts[view] = row
        if view in choices:
            # Used by the renderer to rebuild the perturbed input; global views have no mask.
            maps[f"mask__{view}"] = choices[view]["mask_array"].astype(bool)
    return maps, {
        "views": receipts,
        "n_views": len(receipts),
        "maps_per_view": sum(1 for k in maps if k.startswith("map__")) // max(len(receipts), 1),
        "mask_choice": "maximum effect among localized tests only; the global one is in global_maximum",
        "adaptation": "the maps describe the PERTURBED clip, not the original",
    }


#: Maps excluded from deletion/insertion: they describe the representation, not a
#: contribution to the logit.
NOT_SORTABLE = ("eigencam", "pca_rgb", "cosine")


@HANDLERS.register("deletion_insertion")
def deletion_insertion(context: MethodContext) -> tuple[dict[str, np.ndarray], dict]:
    """Run the protocol putting the methods on the same scale, with its controls.

    Tokens are ranked by each method's map, deleted (or restored) by increasing fractions
    and the detector is rerun: a good map drops the logit faster than random removal. The
    five random orderings are the denominator and are measured once. Curves live in the
    arrays (about 700 rows per window); string tables are in the receipt and the arrays
    hold their indices.

    :return: parallel columns, one row per measurement, and the receipt with the name
        tables and protocol parameters.
    :raises ValueError: if there is no map to rank.
    """
    maps = dict(context.source_maps or {})
    for name, (arrays, _) in (context.siblings or {}).items():
        if name in ("occlusion", "motion_ablation", "perturbed_views", "deletion_insertion"):
            continue
        maps.update(
            {k.removeprefix("map__"): v for k, v in arrays.items() if k.startswith("map__")}
        )
    maps = {
        k: v for k, v in sorted(maps.items()) if k not in NOT_SORTABLE and not k.startswith("vtcd_")
    }
    if not maps:
        raise ValueError(
            "deletion_insertion has no maps to rank: the source campaign "
            "(source_maps) or the concept methods must be computed first"
        )
    settings = context.settings.get("deletion_insertion", {})
    fills = tuple(settings.get("fills", ("blur", "mean")))
    seeds = tuple(settings.get("random_seeds", (42, 43, 44, 45, 46)))
    extra = {k: tuple(settings[k]) for k in ("grid", "fractions") if k in settings}
    modes = ("deletion", "insertion")
    names = [*maps, "random"]
    columns: dict[str, list] = {
        k: []
        for k in (
            "method",
            "fill",
            "mode",
            "opposition",
            "random_seed",
            "actual_fraction",
            "logit",
            "score",
        )
    }

    def add(righe, method_index, seme):
        for row in righe:
            columns["method"].append(method_index)
            columns["fill"].append(fills.index(row["fill"]))
            columns["mode"].append(modes.index(row["mode"]))
            columns["opposition"].append(bool(row["opposition"]))
            columns["random_seed"].append(-1 if seme is None else seme)
            columns["actual_fraction"].append(row["actual_fraction"])
            columns["logit"].append(row["logit"])
            columns["score"].append(row["score"])

    for index, map_ in enumerate(maps.values()):
        signs = (False, True) if np.any(map_ < 0) else (False,)
        for fill in fills:
            for opposite in signs:
                add(
                    perturbation_curve(
                        context.detector,
                        context.clip,
                        map_,
                        fill=fill,
                        opposition=opposite,
                        blur_kernel=context.settings["blur_kernel"],
                        **extra,
                    ),
                    index,
                    None,
                )
    for fill in fills:
        for seed in seeds:
            add(
                perturbation_curve(
                    context.detector,
                    context.clip,
                    next(iter(maps.values())),
                    fill=fill,
                    random_seed=seed,
                    blur_kernel=context.settings["blur_kernel"],
                    **extra,
                ),
                len(maps),
                seed,
            )
    arrays = {
        "method": np.asarray(columns["method"], dtype=np.int16),
        "fill": np.asarray(columns["fill"], dtype=np.int8),
        "mode": np.asarray(columns["mode"], dtype=np.int8),
        "opposition": np.asarray(columns["opposition"], dtype=bool),
        "random_seed": np.asarray(columns["random_seed"], dtype=np.int32),
        "actual_fraction": np.asarray(columns["actual_fraction"], dtype=np.float32),
        "logit": np.asarray(columns["logit"], dtype=np.float32),
        "score": np.asarray(columns["score"], dtype=np.float32),
    }
    return arrays, {
        "methods": names,
        "fills": list(fills),
        "modes": list(modes),
        "random_seeds": list(seeds),
        "grid": list(extra.get("grid", (4, 6, 6))),
        "fractions": list(extra.get("fractions", (0, 0.1, 0.2, 0.3, 0.5, 0.75, 1))),
        "rows": len(columns["logit"]),
        "full_detector_forwards": len(columns["logit"]),
        "random_reference": "the random ordering does not depend on the map: the first one only provides the shape",
    }
