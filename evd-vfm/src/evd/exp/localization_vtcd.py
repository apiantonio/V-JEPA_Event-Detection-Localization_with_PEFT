"""VTCD concept localizers of the automatic-localization campaign.

Main configurations of ``docs/varie/proposte_localizzazione_automatica_20261016.md`` seed 42, 
Common dictionary R=32 regions, K=8 concepts, fitted on 16 train references per class (V3, one chunk per event); 
P05 negatives are the normal chunks (V2); P06 is an L1 logistic on mean concept coefficients (V4); P10
routes 13 per-class dictionaries with a multinomial logistic on mean tokens (V6). Building
blocks are the existing ``evd.explain.vtcd`` (3D SLIC, Convex NMF, NNLS); dictionaries must
converge.

SLIC is CPU work (~9 s per chunk): a stage encodes every gated chunk of a video on the GPU
first and segments them in a process pool, timing each segmentation where it runs.
"""

from __future__ import annotations

import math
import multiprocessing
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from evd.exp.localization_auto import ChunkResult, MethodOutput, S0Video, read_json, video_key
from evd.explain.localization import MaskRule, map_mask, native_boxes
from evd.explain.vtcd import assign_concepts, convex_nmf, slic_regions
from evd.types import VideoMeta

SEED = 42
REGIONS, CONCEPTS = 32, 8
NMF = {"max_iter": 10_000, "tolerance": 1e-5}
MASS70 = MaskRule("mass", 0.7)
METHODS = ("P04-top1", "P05-presence", "P06-soft", "P07-mass70", "P10-soft")


def segment(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """SLIC a ``T×H×W×D`` token volume.

    :return: labels, region descriptors, region token counts and CPU seconds.
    """
    start = time.process_time()
    labels, descriptors, _ = slic_regions(tokens, n_segments=REGIONS)
    counts = np.bincount(labels.ravel(), minlength=len(descriptors))
    return labels, descriptors, counts, time.process_time() - start


def fit_dictionary(descriptors: np.ndarray) -> tuple[np.ndarray, dict]:
    """:return: the ``K×D`` dictionary and the fit receipt.

    :raises RuntimeError: if Convex NMF does not converge (the branch must stop).
    """
    factors, receipt = convex_nmf(descriptors, n_concepts=CONCEPTS, seed=SEED, **NMF)
    if not receipt["converged"]:
        raise RuntimeError(f"Convex NMF did not converge in {NMF['max_iter']} iterations")
    return factors["dictionary"], receipt


def chunk_concepts(descriptors: np.ndarray, counts: np.ndarray, dictionary: np.ndarray) -> dict:
    """:return: region concept ids, coefficients, presence and mean coefficient per concept."""
    ids, coefficients = assign_concepts(descriptors, dictionary)
    presence = np.bincount(ids, minlength=CONCEPTS) > 0
    mean = (coefficients * counts[:, None]).sum(0) / counts.sum()
    return {"ids": ids, "coefficients": coefficients, "presence": presence, "mean": mean}


def enrichment(presence: np.ndarray, positive: np.ndarray, videos: list[str]) -> np.ndarray:
    """Log-ratio of smoothed presence frequency, positive vs negative chunks, per video.

    Per video and side: ``(count + 1) / (total + 2)``; then the mean over the videos that
    have chunks on that side.

    :param presence: ``C × K`` boolean, concept present in chunk.
    :param positive: ``C`` boolean, chunk temporally positive.
    :return: ``K`` log-ratios.
    """
    rates = {}
    for side in (True, False):
        per_video = defaultdict(list)
        for row, flag, video in zip(presence, positive, videos, strict=True):
            if flag == side:
                per_video[video].append(row)
        rates[side] = np.mean(
            [(np.sum(rows, 0) + 1) / (len(rows) + 2) for rows in per_video.values()], axis=0
        )
    return np.log(rates[True] / rates[False])


def token_values(labels: np.ndarray, region_values: np.ndarray) -> np.ndarray:
    """:return: the ``T×H×W`` map giving each token the value of its region."""
    return np.asarray(region_values, dtype=np.float64)[labels]


def mass_prefix_concepts(
    ig_positive: np.ndarray, labels: np.ndarray, ids: np.ndarray, fraction: float = 0.7
) -> np.ndarray:
    """P07: concepts whose positive IG mass makes the minimal prefix; stable ties by id.

    :param fraction: share of the positive mass the prefix must reach (0.7 for P07).
    :return: boolean token mask (empty when the mass is zero).
    """
    concept_of_token = ids[labels]
    masses = np.bincount(concept_of_token.ravel(), weights=ig_positive.ravel(), minlength=CONCEPTS)
    if masses.sum() <= 0:
        return np.zeros(labels.shape, dtype=bool)
    order = np.lexsort((np.arange(CONCEPTS), -masses))
    cumulative = np.cumsum(masses[order])
    count = int(np.searchsorted(cumulative, fraction * cumulative[-1], side="left")) + 1
    chosen = [k for k in order[:count] if masses[k] > 0]
    return np.isin(concept_of_token, chosen)


def top_drop_concepts(
    drops: np.ndarray, labels: np.ndarray, ids: np.ndarray, k: int, tolerance: float
) -> np.ndarray:
    """P04 and its top-k variants: the ``k`` concepts whose removal lowers the logit most.

    Only concepts with a drop above ``tolerance`` enter; ties break by concept id.

    :return: boolean token mask (empty when no concept lowers the logit).
    """
    finite = np.where(np.isfinite(drops), drops, -np.inf)
    order = np.lexsort((np.arange(len(finite)), -finite))
    chosen = [int(c) for c in order[:k] if finite[c] > tolerance]
    return np.isin(ids[labels], chosen)


class VTCDStage:
    """P04, P05, P06, P07 and P10 on the gated chunks."""

    name = "vtcd"
    methods = METHODS

    def __init__(
        self, engine, spec, gate_fit_seconds: float, fits: Path, ig_dataset_root: Path, workers: int
    ) -> None:
        from evd.exp.explanation_advanced import extract_features, prepare_clip

        self.engine, self.spec, self.scorer = engine, spec, engine.scorer
        self.cuda = next(self.scorer.head.parameters()).is_cuda
        self.extract, self.prepare = extract_features, prepare_clip
        model = np.load(fits / "vtcd_models.npz")
        self.dictionary = model["global_dictionary"]
        self.class_dictionaries = model["class_dictionaries"]
        self.class_names = [str(c) for c in model["class_names"]]
        self.p05_concept = int(model["p05_concept"])
        self.p06_weights = model["p06_token_weights"]
        self.p10_concepts = model["p10_concepts"]
        self.router = {k: model[f"router_{k}"] for k in ("mean", "scale", "coef", "intercept")}
        self.router_classes = [str(c) for c in model["router_classes"]]
        self.fit_seconds_ = read_json(fits / "vtcd_fit_seconds.json")
        self.gate_fit_seconds = gate_fit_seconds
        self.ig_dataset_root = ig_dataset_root
        self.ig_root = ig_dataset_root
        # spawn: workers run numpy SLIC only and must not inherit the parent's CUDA state.
        self.pool = ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        )
        self.cache: dict[int, tuple] = {}

    def parameters(self) -> dict:
        """:return: frozen parameters, part of the stage identity."""
        return {
            "regions": REGIONS,
            "concepts": CONCEPTS,
            "nmf": NMF,
            "p05_concept": self.p05_concept,
            "p10_concepts": self.p10_concepts.tolist(),
            "router_classes": self.router_classes,
            "mask_rule": "mass70 for P06, P07 (prefix), P10; concept support for P04, P05",
            "precision": self.scorer.precision,
        }

    def fit_seconds(self) -> dict[str, float]:
        """:return: gate plus each method's dictionary, statistics and classifier fits."""
        return {name: self.gate_fit_seconds + self.fit_seconds_[name] for name in self.methods}

    def _sync(self) -> float:
        if self.cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    def begin_video(self, meta: VideoMeta, items: list, s0: S0Video) -> None:
        """Encode every gated chunk of the video, then segment them in parallel."""
        self.cache.clear()
        self.ig_root = self.ig_dataset_root / video_key(meta.video_id)
        encoded = {}
        for index in s0.gated:
            start = self._sync()
            clip = self.prepare(self.engine, items[index])
            decoded = self._sync()
            feats, logit = self.extract(self.engine, clip)
            done = self._sync()
            tokens = feats.tokens[0].float().cpu().numpy().reshape(*feats.grid, -1)
            encoded[index] = (feats, logit, tokens, decoded - start, done - decoded)
            del clip
        futures = {i: self.pool.submit(segment, v[2]) for i, v in encoded.items()}
        for index, (feats, logit, tokens, dec, enc) in encoded.items():
            self.cache[index] = (feats, logit, tokens, dec, enc, futures[index].result())

    def _context(self, index: int) -> tuple:
        """:return: cached encoding and SLIC of a gated chunk, its concept assignment, and the
        shared and SLIC seconds."""
        feats, logit, tokens, decode_s, encode_s, (labels, descriptors, counts, slic_s) = (
            self.cache.pop(index)
        )
        start = time.perf_counter()
        assigned = chunk_concepts(descriptors, counts, self.dictionary)
        shared = slic_s + time.perf_counter() - start
        return (
            feats,
            logit,
            tokens,
            decode_s,
            encode_s,
            labels,
            descriptors,
            assigned,
            shared,
            slic_s,
        )

    def _concept_drops(self, feats, logit: float, ids, labels, assigned) -> np.ndarray:
        """:return: logit drop when each present concept's tokens are zeroed (-inf if absent)."""
        drops = np.full(CONCEPTS, -np.inf)
        index_map = torch.as_tensor(ids[labels].reshape(-1), device=feats.tokens.device)
        with torch.no_grad(), self.scorer.autocast():
            for concept in np.flatnonzero(assigned["presence"]):
                zeroed = feats.tokens * (index_map != int(concept))[None, :, None]
                ablated = float(
                    self.scorer.head(replace(feats, tokens=zeroed, pooled=None)).float()
                )
                drops[concept] = logit - ablated
        return drops

    def chunk(self, meta: VideoMeta, item, index: int) -> ChunkResult:
        """Assign the chunk's regions and build the five concept localizations."""
        feats, logit, tokens, decode_s, encode_s, labels, descriptors, assigned, shared, slic_s = (
            self._context(index)
        )
        grid = tuple(feats.grid)
        ids = assigned["ids"]
        outputs, saved = {}, {}

        def emit(name, mask_or_map, seconds, rule=None):
            post = time.perf_counter()
            mask = mask_or_map if rule is None else map_mask(mask_or_map, rule)
            outputs[name] = MethodOutput(
                native_boxes(mask, meta.width, meta.height), seconds + time.perf_counter() - post
            )
            saved[name] = np.asarray(mask_or_map, dtype=np.float32)

        begin = self._sync()
        drops = self._concept_drops(feats, logit, ids, labels, assigned)
        best = int(np.argmax(drops)) if np.isfinite(drops).any() else -1
        tolerance = 1e-6 * max(1.0, abs(logit))
        p04 = (
            (ids[labels] == best) if best >= 0 and drops[best] > tolerance else np.zeros(grid, bool)
        )
        emit("P04-top1", p04, shared + self._sync() - begin)

        p05 = ids[labels] == self.p05_concept if self.p05_concept >= 0 else np.zeros(grid, bool)
        emit("P05-presence", p05, shared)

        begin = time.perf_counter()
        p06 = token_values(labels, assigned["coefficients"] @ self.p06_weights)
        emit("P06-soft", p06, shared + time.perf_counter() - begin, MASS70)

        begin = time.perf_counter()
        ig = np.load(self.ig_root / "maps" / f"{index:06d}.npz")["ig_ensemble"]
        ig_record = read_json(self.ig_root / "chunks" / f"{index:06d}.json")
        ig_seconds = ig_record["methods"]["P14-mass70"]["method_seconds"]
        p07 = mass_prefix_concepts(np.maximum(ig, 0), labels, ids)
        emit("P07-mass70", p07, shared + ig_seconds + time.perf_counter() - begin)

        begin = time.perf_counter()
        z = (tokens.reshape(-1, tokens.shape[-1]).mean(0) - self.router["mean"]) / self.router[
            "scale"
        ]
        scores = self.router["coef"] @ z + self.router["intercept"]
        probabilities = np.exp(scores - scores.max())
        probabilities /= probabilities.sum()
        p10 = np.zeros(grid)
        probability = dict(zip(self.router_classes, probabilities, strict=True))
        for name, dictionary, concept in zip(
            self.class_names, self.class_dictionaries, self.p10_concepts, strict=True
        ):
            if concept < 0:
                continue
            class_ids, _ = assign_concepts(descriptors, dictionary)
            p10 += probability[name] * (class_ids[labels] == concept)
        emit("P10-soft", p10, shared + time.perf_counter() - begin, MASS70)
        return ChunkResult(
            outputs,
            maps=saved,
            receipts={
                "regions": int(len(descriptors)),
                "slic_cpu_seconds": slic_s,
                "present": np.flatnonzero(assigned["presence"]).tolist(),
                "p04_drops": [d if math.isfinite(d) else None for d in drops.tolist()],
                "router": dict(zip(self.router_classes, probabilities.tolist(), strict=True)),
                "ig_stage_record": ig_record["window_id"],
            },
            logit=logit,
            stage_decode_seconds=decode_s,
            stage_encode_seconds=encode_s,
        )


#: Follow-up of 17 October (proposal 3): the cut variants of the two best concept methods.
ABLATION = {"P04-top2": 2, "P04-top3": 3, "P07-mass50": 0.5, "P07-mass90": 0.9}
#: Second follow-up of 17 October: P07 with 30% of the IG mass.
ABLATION_MASS30 = {"P07-mass30": 0.3}
#: Third follow-up of 17 October: the IG mass curve of P07 between 10% and 50%.
ABLATION_SWEEP = {"P07-mass10": 0.1, "P07-mass20": 0.2, "P07-mass40": 0.4}
#: Paired comparison requested on 21 October: absolute value of the signed ensemble.
ABSOLUTE_MASS70 = {"P07-mass70": 0.7, "P07-abs-mass70": 0.7}


class VTCDAblationStage(VTCDStage):
    """Variants of P04 (top-k by logit drop) and P07 (IG mass prefix) on one assignment.

    ``variants`` maps an experiment to its k (``P04-*``) or mass fraction (``P07-*``); the
    concept assignment is the one of :class:`VTCDStage`. ``mass_modes`` changes selected
    P07 variants from positive evidence to absolute attribution, after the signed baseline
    average; every unspecified variant retains the original positive-only rule.
    """

    def __init__(
        self,
        *args,
        variants: dict = ABLATION,
        name: str = "vtcd_ablation",
        mass_modes: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        if any(not k.startswith(("P04-", "P07-")) for k in variants):
            raise ValueError("Ablation variants must derive from P04 or P07")
        modes = dict(mass_modes or {})
        if any(k not in variants or not k.startswith("P07-") for k in modes) or any(
            mode not in {"positive", "absolute"} for mode in modes.values()
        ):
            raise ValueError("IG mass modes must be positive/absolute for declared P07 variants")
        super().__init__(*args, **kwargs)
        self.variants, self.name, self.methods = dict(variants), name, tuple(variants)
        self.mass_modes = modes

    def parameters(self) -> dict:
        """:return: frozen parameters, part of the stage identity."""
        return {
            **super().parameters(),
            "mask_rule": "top-k drops for P04, IG mass prefix for P07",
            "ablation": self.variants,
            **({"ig_mass_modes": self.mass_modes} if self.mass_modes else {}),
        }

    def fit_seconds(self) -> dict[str, float]:
        """:return: the fit of the method each variant derives from."""
        return {
            name: self.gate_fit_seconds
            + self.fit_seconds_["P04-top1" if name.startswith("P04-") else "P07-mass70"]
            for name in self.methods
        }

    def chunk(self, meta: VideoMeta, item, index: int) -> ChunkResult:
        """Build every variant from one concept assignment of the chunk."""
        feats, logit, _, decode_s, encode_s, labels, _, assigned, shared, _ = self._context(index)
        ids = assigned["ids"]
        outputs, receipts = {}, {"present": np.flatnonzero(assigned["presence"]).tolist()}

        def emit(name, mask, seconds):
            post = time.perf_counter()
            outputs[name] = MethodOutput(
                native_boxes(mask, meta.width, meta.height), seconds + time.perf_counter() - post
            )

        if any(n.startswith("P04-") for n in self.methods):
            begin = self._sync()
            drops = self._concept_drops(feats, logit, ids, labels, assigned)
            drop_seconds = self._sync() - begin
            tolerance = 1e-6 * max(1.0, abs(logit))
            receipts["p04_drops"] = [d if math.isfinite(d) else None for d in drops.tolist()]
            for name in (n for n in self.methods if n.startswith("P04-")):
                begin = time.perf_counter()
                mask = top_drop_concepts(drops, labels, ids, self.variants[name], tolerance)
                emit(name, mask, shared + drop_seconds + time.perf_counter() - begin)

        if any(n.startswith("P07-") for n in self.methods):
            ig = np.load(self.ig_root / "maps" / f"{index:06d}.npz")["ig_ensemble"]
            ig_record = read_json(self.ig_root / "chunks" / f"{index:06d}.json")
            ig_seconds = ig_record["methods"]["P14-mass70"]["method_seconds"]
            receipts["ig_stage_record"] = ig_record["window_id"]
            for name in (n for n in self.methods if n.startswith("P07-")):
                begin = time.perf_counter()
                transform = {
                    "positive": lambda values: np.maximum(values, 0),
                    "absolute": np.abs,
                }[self.mass_modes.get(name, "positive")]
                mask = mass_prefix_concepts(transform(ig), labels, ids, self.variants[name])
                emit(name, mask, shared + ig_seconds + time.perf_counter() - begin)
        return ChunkResult(
            outputs,
            # Preserve complete assignments for CPU replay of subsequent sign rules.
            maps={"concept_ids": ids[labels].astype(np.uint8)} if self.mass_modes else {},
            receipts=receipts,
            logit=logit,
            stage_decode_seconds=decode_s,
            stage_encode_seconds=encode_s,
        )
