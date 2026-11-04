"""Class-routed VTCD localization.

A chunk is localized in two steps: a class is chosen (proposal A: NNLS on the 13 per-class 
dictionaries stacked into 104 concepts; proposal B: a multiclass head; the oracle: the annotated 
class), then P07 runs on the chosen class's dictionary and every kept concept gives its own boxes.

This module holds the pure logic shared by the GPU passes, the CPU configurations and the
threshold calibration: fixed-node Integrated Gradients of a clip, the four class rules of
proposal A, the concept prefix, per-concept boxes, the per-dictionary fraction calibration
and the Convex NMF fit with the halving rule. The VTCD geometry (32 regions, 8 concepts, the
NMF parameters) is the one of ``exp/localization_vtcd.py``, imported, not repeated.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import torch

from evd.exp.localization_vtcd import CONCEPTS, NMF, SEED
from evd.explain.attribution import full_precision, integrated_gradients_fixed
from evd.explain.localization import native_boxes
from evd.explain.vtcd import convex_nmf

#: Mass fraction of the 55% threshold (decision of 21 October, ``p07_frazione_massa_20261021.md``).
FRACTION = 0.55
#: Target of the calibrated threshold: share of validation positives keeping a single concept.
SINGLE_CONCEPT_TARGET = 0.9
#: The four class rules of proposal A (plan §5.3), in tie-breaking order of the report.
CLASS_RULES = ("somma-grezza", "regione-un-voto", "token-un-voto", "voto-netto")
#: The four synthetic baselines of the campaign IG ensemble (``exp/localization_ig.IG_FILLS``).
IG_FILLS = ("black", "noise", "mean", "blur")
#: Below this many chunks a class dictionary of A2 falls back to the A1 one (plan §5.3).
MIN_CHUNKS = 16


@dataclass(frozen=True)
class IGSettings:
    """Fixed-node Integrated Gradients settings, frozen by the pilot (plan §7.1).

    :param nodes: Gauss-Legendre nodes per baseline.
    :param batch_size: rows of the single (baseline, node) queue per head call.
    :param precision: ``fp32`` runs the head in full precision (autocast and TF32 off),
        ``bf16`` keeps the scorer autocast of the 16 October campaign.
    :param fills: baselines, averaged into ``ig_ensemble``.
    :param atol: absolute tolerance of the completeness residual.
    :param rtol: tolerance relative to the logit difference.
    """

    nodes: int
    batch_size: int
    precision: str = "fp32"
    fills: tuple[str, ...] = IG_FILLS
    atol: float = 0.01
    rtol: float = 0.01

    def __post_init__(self) -> None:
        if self.nodes < 1 or self.batch_size < 1:
            raise ValueError(
                f"IG nodes and batch must be positive: {self.nodes}, {self.batch_size}"
            )
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError(f"IG precision must be fp32 or bf16, not {self.precision!r}")
        if not self.fills or len(set(self.fills)) != len(self.fills):
            raise ValueError(f"IG fills must be distinct and non-empty: {self.fills}")


def _sync(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def encode_with_baselines(scorer, spec, clip: torch.Tensor, window_id: str, fills: Sequence[str]):
    """Encode a clip and its IG reference clips, as the 16 October campaign does.

    The reference clips are those of :func:`evd.exp.explanation_campaign.explain_clip` (same
    ``reference_clip``, blur kernel 31, noise seeded by the window id). ``explain_clip`` is
    not called because its file enters the identity of running explainability campaigns, and
    the integral here differs.

    :param scorer: the live scorer (``backbone``, ``head``, ``levels``, ``autocast``).
    :param spec: run preprocessing; ``black`` needs its mean and std.
    :param clip: ``B,C,T,H,W`` preprocessed clip on the scorer device, ``B = 1``.
    :param window_id: protocol window id, seeding the ``noise`` baseline.
    :param fills: the baselines to encode.
    :return: ``(feats, logit, bases, seconds)``: clip features, binary logit, the baseline
        tokens per fill and the seconds of ``forward`` and ``baselines`` (synchronized).
    """
    from evd.exp.explanation_campaign import seed_from_window_id
    from evd.explain.perturbation import reference_clip
    from evd.models.backbones.base import encode_multilevel

    start = _sync(clip.device)
    with torch.no_grad(), scorer.autocast():
        feats = encode_multilevel(scorer.backbone, clip, scorer.levels)
        logit = float(scorer.head(feats).float().item())
    forward_done = _sync(clip.device)
    bases = {}
    with torch.no_grad(), scorer.autocast():
        for fill in fills:
            if fill == "black":
                ref = reference_clip(clip, fill, 31, mean=spec.mean, std=spec.std)
            elif fill == "noise":
                ref = reference_clip(clip, fill, 31, seed=seed_from_window_id(window_id))
            else:
                ref = reference_clip(clip, fill, 31)
            bases[fill] = encode_multilevel(scorer.backbone, ref, scorer.levels).tokens
            del ref
    done = _sync(clip.device)
    return feats, logit, bases, {"forward": forward_done - start, "baselines": done - forward_done}


def integrate(scorer, feats, bases: Mapping[str, torch.Tensor], settings: IGSettings):
    """Fixed-node IG of the scorer head between each baseline and the clip features.

    :param scorer: the live scorer.
    :param feats: clip features (``BackboneOutput``, batch one).
    :param bases: baseline tokens per fill, shaped like ``feats.tokens``.
    :param settings: nodes, batch and precision (``fp32``: head in full precision).
    :return: ``(maps, receipts, seconds)``: ``T×H×W`` float32 maps per ``ig_<fill>`` plus
        ``ig_ensemble`` (their mean), the convergence receipt per fill, the integral seconds.
    """
    start = _sync(feats.tokens.device)
    fp32 = settings.precision == "fp32"
    original = feats.tokens.float() if fp32 else feats.tokens
    inputs = {k: (v.float() if fp32 else v) for k, v in bases.items()}
    context = full_precision if fp32 else scorer.autocast

    def forward(tokens: torch.Tensor) -> torch.Tensor:
        with context():
            return scorer.head(replace(feats, tokens=tokens, pooled=None))

    with context():
        out = integrated_gradients_fixed(
            forward,
            original,
            inputs,
            nodes=settings.nodes,
            batch_size=settings.batch_size,
            atol=settings.atol,
            rtol=settings.rtol,
        )
    maps = {
        f"ig_{fill}": values.sum(-1).reshape(feats.grid).cpu().numpy().astype(np.float32)
        for fill, (values, _) in out.items()
    }
    maps["ig_ensemble"] = np.mean([maps[f"ig_{fill}"] for fill in bases], axis=0)
    receipts = {fill: receipt for fill, (_, receipt) in out.items()}
    return maps, receipts, _sync(feats.tokens.device) - start


def explain_ig_fixed(scorer, spec, clip: torch.Tensor, window_id: str, settings: IGSettings):
    """Explain a preprocessed clip with fixed-node IG on the synthetic baselines.

    :func:`encode_with_baselines` followed by :func:`integrate`.

    :return: ``(maps, receipts, feats, logit, seconds)`` with the seconds of ``forward``,
        ``baselines`` and ``integral``.
    """
    feats, logit, bases, seconds = encode_with_baselines(
        scorer, spec, clip, window_id, settings.fills
    )
    maps, receipts, integral = integrate(scorer, feats, bases, settings)
    return maps, receipts, feats, logit, {**seconds, "integral": integral}


def stack_dictionaries(class_dictionaries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stack per-class dictionaries into one, remembering the class of each concept.

    :param class_dictionaries: ``C×K×D`` dictionaries, in class order.
    :return: the ``(C·K)×D`` stacked dictionary and the ``C·K`` class index of each concept.
    :raises ValueError: if the input is not three-dimensional.
    """
    stacked = np.asarray(class_dictionaries, dtype=np.float64)
    if stacked.ndim != 3:
        raise ValueError(f"Expected C×K×D class dictionaries, got shape {stacked.shape}")
    classes, concepts, dim = stacked.shape
    return stacked.reshape(classes * concepts, dim), np.repeat(np.arange(classes), concepts)


def class_scores(
    coefficients: np.ndarray,
    counts: np.ndarray,
    owners: np.ndarray,
    n_classes: int,
    rule: str,
) -> np.ndarray:
    """Score each class from the NNLS coefficients of the regions on the stacked dictionary.

    The four rules of plan §5.3: ``somma-grezza`` sums the raw coefficients of each class's
    concepts over the regions; ``regione-un-voto`` first normalizes each region's coefficients
    to sum one; ``token-un-voto`` also weighs each region by its token count;
    ``voto-netto`` gives each region's token count to the class of its highest coefficient
    (lowest concept id on ties, as ``assign_concepts``). A region whose coefficients are all
    zero votes for no class.

    :param coefficients: ``R×(C·K)`` non-negative NNLS coefficients.
    :param counts: ``R`` token counts of the regions.
    :param owners: ``C·K`` class index of each stacked concept.
    :param n_classes: number of classes ``C``.
    :param rule: one of :data:`CLASS_RULES`.
    :return: ``C`` class scores.
    :raises ValueError: on an unknown rule or inconsistent shapes.
    """
    beta = np.asarray(coefficients, dtype=np.float64)
    weights = np.asarray(counts, dtype=np.float64)
    if beta.ndim != 2 or beta.shape[1] != len(owners) or len(weights) != len(beta):
        raise ValueError("Coefficients, counts and concept owners disagree on their shapes")
    if rule == "voto-netto":
        votes = np.zeros(n_classes)
        voting = beta.max(1) > 0
        np.add.at(votes, owners[beta.argmax(1)[voting]], weights[voting])
        return votes
    if rule == "somma-grezza":
        per_region = beta
    elif rule in {"regione-un-voto", "token-un-voto"}:
        totals = beta.sum(1, keepdims=True)
        per_region = np.divide(beta, totals, out=np.zeros_like(beta), where=totals > 0)
        if rule == "token-un-voto":
            per_region = per_region * weights[:, None]
    else:
        raise ValueError(f"Unknown class rule {rule!r}; known: {CLASS_RULES}")
    return np.bincount(owners, weights=per_region.sum(0), minlength=n_classes)


def choose_class(scores: np.ndarray) -> int:
    """:return: the index of the highest score, the lowest index on ties."""
    return int(np.argmax(np.asarray(scores, dtype=np.float64)))


def positive_masses(
    ig: np.ndarray, labels: np.ndarray, ids: np.ndarray, concepts: int
) -> np.ndarray:
    """Positive IG mass of each concept of a chunk.

    :param ig: ``T×H×W`` signed IG ensemble map.
    :param labels: ``T×H×W`` SLIC region of each token.
    :param ids: concept id of each region.
    :param concepts: number of concepts of the dictionary.
    :return: ``concepts`` masses.
    :raises ValueError: if map and regions disagree on the grid or the map is not finite.
    """
    if ig.shape != labels.shape:
        raise ValueError(f"IG map {ig.shape} and SLIC labels {labels.shape} differ")
    if not np.isfinite(ig).all():
        raise ValueError("Non-finite IG map")
    return np.bincount(
        np.asarray(ids)[labels].ravel(), weights=np.maximum(ig, 0).ravel(), minlength=concepts
    )


def prefix_concepts(masses: np.ndarray, fraction: float) -> list[int]:
    """Concepts of the minimal prefix reaching ``fraction`` of the positive mass.

    The P07 rule of ``localization_vtcd.mass_prefix_concepts`` (concepts by decreasing mass,
    ties by id, zero-mass concepts never kept), returning the concepts instead of their union
    mask, because each kept concept gives its own boxes here.

    :param masses: positive IG mass of each concept.
    :param fraction: share of the total the prefix must reach, in ``(0, 1]``.
    :return: kept concept ids in prefix order; empty when the total mass is zero.
    :raises ValueError: if ``fraction`` is outside ``(0, 1]``.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"The mass fraction must be in (0, 1], not {fraction}")
    values = np.asarray(masses, dtype=np.float64)
    if values.sum() <= 0:
        return []
    order = np.lexsort((np.arange(len(values)), -values))
    cumulative = np.cumsum(values[order])
    count = int(np.searchsorted(cumulative, fraction * cumulative[-1], side="left")) + 1
    return [int(k) for k in order[:count] if values[k] > 0]


def concept_boxes(
    labels: np.ndarray, ids: np.ndarray, concepts: Sequence[int], width: int, height: int
) -> list[list[list[int]]]:
    """Native boxes of each kept concept, extracted separately and listed together per plane.

    Each concept's token mask gives its own connected components (8-connectivity, as
    ``native_boxes``); masks of different concepts are never merged, so two touching blobs
    of two concepts give two boxes (plan §3).

    :param labels: ``T×H×W`` SLIC region of each token.
    :param ids: concept id of each region.
    :param concepts: kept concept ids.
    :param width: native frame width, in pixels.
    :param height: native frame height, in pixels.
    :return: boxes per temporal plane, inclusive native coordinates.
    """
    of_token = np.asarray(ids)[labels]
    planes: list[list[list[int]]] = [[] for _ in range(labels.shape[0])]
    for concept in concepts:
        for plane, boxes in enumerate(native_boxes(of_token == concept, width, height)):
            planes[plane].extend(boxes)
    return planes


def concept_components(
    ig: np.ndarray,
    labels: np.ndarray,
    ids: np.ndarray,
    width: int,
    height: int,
    connectivity: int = 8,
) -> list[list[dict]]:
    """Every connected component of every concept, with its positive IG mass, per plane.

    The detections of the localization AP (``docs/varie/campagna_ap_localizzazione_20261027.md``):
    the components of :func:`concept_boxes`, but all of them (no box contained in another
    is removed, unlike ``native_boxes``) and each with the positive IG mass of its tokens.
    The native box is the one ``native_boxes`` gives the same component.

    :param ig: ``T×H×W`` signed IG ensemble map.
    :param labels: ``T×H×W`` SLIC region of each token.
    :param ids: concept id of each region.
    :param width: native frame width, in pixels.
    :param height: native frame height, in pixels.
    :param connectivity: 8 (the campaign's, as ``native_boxes``) or 4 (variants of 28 October
        2026: two tokens touching only at a corner are two components).
    :return: per temporal plane, ``{"concept", "box", "tokens", "mass"}`` per component, with
        ``box`` inclusive native ``[x0, y0, x1, y1]``; by concept, then ``(-tokens, y0, x0)``
        on the token grid as ``boxes_from_mask``.
    :raises ValueError: if map and regions disagree on the grid, the map is not finite or the
        connectivity is neither 4 nor 8.
    """
    import cv2

    from evd.explain.boxes import Box, in_pixel

    if ig.shape != labels.shape:
        raise ValueError(f"IG map {ig.shape} and SLIC labels {labels.shape} differ")
    if not np.isfinite(ig).all():
        raise ValueError("Non-finite IG map")
    if connectivity not in (4, 8):
        raise ValueError(f"Connectivity must be 4 or 8, not {connectivity}")
    of_token = np.asarray(ids)[labels]
    positive = np.maximum(np.asarray(ig, dtype=np.float64), 0)
    _, rows, cols = labels.shape
    planes: list[list[dict]] = []
    for plane in range(labels.shape[0]):
        found = []
        for concept in np.unique(of_token[plane]):
            binary = np.ascontiguousarray((of_token[plane] == concept).astype(np.uint8))
            number, component, stats, _ = cv2.connectedComponentsWithStats(
                binary, connectivity=connectivity
            )
            masses = np.bincount(
                component.ravel(), weights=positive[plane].ravel(), minlength=number
            )
            boxes = []
            for index in range(1, number):
                x0, y0 = int(stats[index, cv2.CC_STAT_LEFT]), int(stats[index, cv2.CC_STAT_TOP])
                x1 = x0 + int(stats[index, cv2.CC_STAT_WIDTH])
                y1 = y0 + int(stats[index, cv2.CC_STAT_HEIGHT])
                box = Box((x0, y0, x1, y1), (x0 / cols, y0 / rows, x1 / cols, y1 / rows), 0)
                tokens = int(stats[index, cv2.CC_STAT_AREA])
                boxes.append(((-tokens, y0, x0), tokens, float(masses[index]), box))
            for _, tokens, mass, box in sorted(boxes, key=lambda b: b[0]):
                found.append(
                    {
                        "concept": int(concept),
                        "box": list(in_pixel(box, width, height)),
                        "tokens": tokens,
                        "mass": mass,
                    }
                )
        planes.append(found)
    return planes


def region_boxes(ig: np.ndarray, labels: np.ndarray, width: int, height: int) -> list[list[dict]]:
    """The box of every SLIC region on every plane, with its positive IG mass there.

    The material of the region variants of the AP (28 October 2026): a concept's regions are
    never merged, so two touching regions of one concept give two boxes. The box encloses the
    region's tokens on the plane, even when the 3D region is cut into several pieces there. It
    does not depend on the dictionary: a configuration reads the region's concept from its ids.

    :param ig: ``T×H×W`` signed IG ensemble map.
    :param labels: ``T×H×W`` SLIC region of each token.
    :param width: native frame width, in pixels.
    :param height: native frame height, in pixels.
    :return: per temporal plane, ``{"region", "box", "tokens", "mass"}`` per region present on
        it, by region id, with ``box`` inclusive native ``[x0, y0, x1, y1]`` (the rule of
        :func:`concept_components`).
    :raises ValueError: if map and regions disagree on the grid or the map is not finite.
    """
    from evd.explain.boxes import Box, in_pixel

    if ig.shape != labels.shape:
        raise ValueError(f"IG map {ig.shape} and SLIC labels {labels.shape} differ")
    if not np.isfinite(ig).all():
        raise ValueError("Non-finite IG map")
    positive = np.maximum(np.asarray(ig, dtype=np.float64), 0)
    _, rows, cols = labels.shape
    planes: list[list[dict]] = []
    for plane in range(labels.shape[0]):
        found = []
        for region in np.unique(labels[plane]):
            ys, xs = np.nonzero(labels[plane] == region)
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            box = Box((x0, y0, x1, y1), (x0 / cols, y0 / rows, x1 / cols, y1 / rows), 0)
            found.append(
                {
                    "region": int(region),
                    "box": list(in_pixel(box, width, height)),
                    "tokens": int(len(xs)),
                    "mass": float(positive[plane][ys, xs].sum()),
                }
            )
        planes.append(found)
    return planes


def first_concept_share(masses: np.ndarray) -> float | None:
    """:return: the share of the positive mass carried by the heaviest concept, or ``None``
    when the chunk has no positive mass (such chunks are left out of the calibration)."""
    values = np.asarray(masses, dtype=np.float64)
    total = values.sum()
    return None if total <= 0 else float(values.max() / total)


def calibrate_fraction(shares: Sequence[float], target: float = SINGLE_CONCEPT_TARGET) -> float:
    """Highest fraction, at whole percentage points, that keeps one concept on ``target`` chunks.

    A chunk keeps a single concept exactly when its heaviest concept carries at least the
    fraction. As for the 55% of ``p07_frazione_massa_20261021.md``, the fraction is the
    ``1 - target`` quantile of the first-concept shares, rounded down to the percentage
    point; it is then lowered by one point at a time until at least ``target`` of the chunks
    really keep one concept, since the interpolated quantile does not guarantee it on small
    samples.

    :param shares: first-concept shares of the chunks with positive mass.
    :param target: required share of single-concept chunks.
    :return: the fraction, a multiple of 0.01 in ``[0.01, 1]``.
    :raises ValueError: without shares, or with a share or target outside ``(0, 1]``.
    """
    values = np.asarray(shares, dtype=np.float64)
    if not len(values) or ((values <= 0) | (values > 1)).any() or not 0 < target <= 1:
        raise ValueError("Calibration needs shares and a target in (0, 1]")
    fraction = max(1, math.floor(round(float(np.quantile(values, 1 - target)) * 100, 9)))
    while fraction > 1 and np.mean(values >= fraction / 100) < target:
        fraction -= 1
    return fraction / 100


def peak_memory_mb() -> float:
    """:return: peak resident memory of this process so far, in MB (Linux reports KB); ``nan``
    where the ``resource`` module does not exist (Windows)."""
    try:
        import resource
    except ImportError:
        return float("nan")
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def fit_with_halving(
    descriptors: Sequence[np.ndarray],
    fallback: np.ndarray | None,
    *,
    min_chunks: int = MIN_CHUNKS,
    seed: int = SEED,
    fit: Callable[[np.ndarray], tuple[dict, dict]] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Fit a class dictionary on all its chunks, halving them while Convex NMF fails.

    Rule of plan §5.3 (decision of 21 October): every failure is recorded; the next attempt
    keeps half of the chunks, drawn with ``seed``; below ``min_chunks`` the class uses its
    A1 dictionary (``fallback``). A class that starts below ``min_chunks`` uses the fallback
    without fitting.

    :param descriptors: per chunk, the ``R×D`` region descriptors.
    :param fallback: the A1 dictionary of the class, or ``None`` if there is none.
    :param min_chunks: smallest number of chunks a fit may use.
    :param seed: seed of the halving draws.
    :param fit: the fit, ``descriptors -> (factors, receipt)``; default Convex NMF with the
        campaign parameters (8 concepts, ``NMF``).
    :return: the ``K×D`` dictionary and one receipt per attempt; the last receipt says which
        dictionary was kept (``fitted`` or ``fallback``).
    :raises RuntimeError: if the fallback is needed and missing.
    """
    fit = fit or (lambda x: convex_nmf(x, n_concepts=CONCEPTS, seed=SEED, **NMF))
    rng = np.random.default_rng(seed)
    chosen = np.arange(len(descriptors))
    receipts: list[dict] = []
    while len(chosen) >= min_chunks:
        x = np.concatenate([descriptors[i] for i in chosen])
        start = time.perf_counter()
        attempt = {"chunks": int(len(chosen)), "regions": int(len(x))}
        try:
            factors, receipt = fit(x)
        except MemoryError as error:
            receipts.append({**attempt, "outcome": "out_of_memory", "error": str(error)})
        else:
            attempt.update(
                converged=bool(receipt["converged"]),
                iterations=int(receipt["iterations"]),
                relative_squared_error=receipt["history"][-1]["relative_squared_error"]
                if receipt.get("history")
                else None,
            )
            if receipt["converged"]:
                receipts.append(
                    {
                        **attempt,
                        "outcome": "fitted",
                        "seconds": time.perf_counter() - start,
                        "peak_memory_mb": peak_memory_mb(),
                    }
                )
                return np.asarray(factors["dictionary"]), receipts
            receipts.append({**attempt, "outcome": "not_converged"})
        receipts[-1].update(seconds=time.perf_counter() - start, peak_memory_mb=peak_memory_mb())
        chosen = np.sort(rng.choice(chosen, size=len(chosen) // 2, replace=False))
    if fallback is None:
        raise RuntimeError("Class dictionary needs the A1 fallback, which is missing")
    receipts.append({"chunks": int(len(chosen)), "outcome": "fallback"})
    return np.asarray(fallback), receipts


def balanced_accuracy(true: Sequence[int], predicted: Sequence[int], classes: int) -> float:
    """Mean over the classes present in ``true`` of the share of their chunks predicted right.

    :param true: class index of each chunk.
    :param predicted: predicted class index of each chunk.
    :param classes: number of classes considered.
    :return: the balanced accuracy, ``nan`` without chunks.
    """
    t, p = np.asarray(true), np.asarray(predicted)
    recalls = [float(np.mean(p[t == c] == c)) for c in range(classes) if (t == c).any()]
    return float(np.mean(recalls)) if recalls else float("nan")


def confusion(true: Sequence[int], predicted: Sequence[int], classes: int) -> list[list[int]]:
    """:return: ``classes×classes`` counts, rows true and columns predicted."""
    matrix = np.zeros((classes, classes), dtype=int)
    np.add.at(matrix, (np.asarray(true), np.asarray(predicted)), 1)
    return matrix.tolist()


def per_class_fractions(
    shares_by_class: Mapping[str, Sequence[float]], target: float = SINGLE_CONCEPT_TARGET
) -> dict[str, dict]:
    """Calibrate one fraction per class dictionary, with its sample size (plan §7.3).

    :param shares_by_class: first-concept shares of each class's validation positives.
    :param target: required share of single-concept chunks.
    :return: class name to ``{"fraction", "chunks", "single_concept_share"}``.
    :raises ValueError: if a class has no share (no fallback by decision of 21 October).
    """
    out = {}
    for name, shares in shares_by_class.items():
        if not len(shares):
            raise ValueError(f"No validation share for class {name}: no fallback is allowed")
        fraction = calibrate_fraction(shares, target)
        out[name] = {
            "fraction": fraction,
            "chunks": len(shares),
            "single_concept_share": float(np.mean(np.asarray(shares) >= fraction)),
        }
    return out
