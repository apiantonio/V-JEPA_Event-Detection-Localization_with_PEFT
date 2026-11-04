"""Perturbation stages of the automatic-localization campaign (P22, P23, P41–P46).

Main configurations of ``docs/varie/proposte_localizzazione_automatica_20261016.md``, seed 42, 
perturbed forwards in batches of 8, the same 128 Bernoulli masks for RISE, LIME and HSIC, 
black = RGB 0 before normalization.
Estimators are in :mod:`evd.explain.region_attribution`; this module only builds perturbed
inputs and runs the frozen detector on them. Every forward group is timed and charged in
full to each experiment that needs it.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, replace

import numpy as np
import torch

from evd.exp.explanation_advanced import extract_features, prepare_clip
from evd.exp.localization_auto import ChunkResult, MethodOutput, S0Video
from evd.explain import region_attribution as ra
from evd.explain.localization import MaskRule, map_mask, native_boxes
from evd.explain.perturbation import reference_clip
from evd.models.backbones.base import encode_multilevel
from evd.types import VideoMeta

SEED = 42
SAMPLES = 128
BATCH = 8
MASS70 = MaskRule("mass", 0.7)
PERTURB_METHODS = (
    "P22-masks128",
    "P41-spatial25",
    "P42-samples128",
    "P43-samples128",
    "P44-masks128",
    "P45-channels32",
    "P46-channels32",
)
EXTREMAL = "P23-area20"
EXTREMAL_SETTINGS = {"area": 0.2, "steps": 100, "step": 0.1, "tv_weight": 0.01}


def token_to_pixel(masks: torch.Tensor, clip_shape: tuple[int, ...]) -> torch.Tensor:
    """Upsample ``N × T' × H' × W'`` token masks to ``N × 1 × T × H × W`` pixels (nearest)."""
    _, _, t, h, w = clip_shape
    n, tt, hh, ww = masks.shape
    return (
        masks[:, None]
        .repeat_interleave(t // tt, 2)
        .repeat_interleave(h // hh, 3)
        .repeat_interleave(w // ww, 4)
    )


class _Detector:
    """Batched logits of the frozen detector on perturbed clips."""

    def __init__(self, engine, spec) -> None:
        self.scorer = engine.scorer
        self.spec = spec
        self.cuda = next(self.scorer.head.parameters()).is_cuda

    def sync(self) -> float:
        if self.cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    def logits(self, clips_fn, count: int) -> tuple[np.ndarray, float]:
        """:return: ``count`` logits of ``clips_fn(begin, end)`` batches, and the seconds."""
        start = self.sync()
        values = []
        with torch.no_grad(), self.scorer.autocast():
            for begin in range(0, count, BATCH):
                clips = clips_fn(begin, min(begin + BATCH, count))
                feats = encode_multilevel(self.scorer.backbone, clips, self.scorer.levels)
                values.append(self.scorer.head(feats).float().reshape(-1).cpu())
                del clips, feats
        return torch.cat(values).numpy().astype(np.float64), self.sync() - start

    def masked(self, clip, black, keep: torch.Tensor):
        """:return: a ``clips_fn`` blending ``clip`` and ``black`` with token keep masks."""

        def build(begin: int, end: int) -> torch.Tensor:
            m = token_to_pixel(keep[begin:end].to(clip.device, clip.dtype), tuple(clip.shape))
            return clip * m + black * (1 - m)

        return build


class PerturbStage:
    """P22, P41, P42, P43, P44, P45, P46 in one pass over the gated chunks."""

    name = "perturb"
    methods = PERTURB_METHODS

    def __init__(self, engine, spec, gate_fit_seconds: float, channels: dict) -> None:
        self.engine, self.spec = engine, spec
        self.detector = _Detector(engine, spec)
        self.gate_fit_seconds = gate_fit_seconds
        self.channels = np.asarray(channels["channels"], dtype=int)
        self.channel_fit_seconds = channels["fit_seconds"]
        self.bernoulli = ra.bernoulli_presence(SAMPLES, 32, SEED)
        self.coalitions = ra.kernel_shap_coalitions(SAMPLES, 32, SEED)

    def parameters(self) -> dict:
        """:return: frozen parameters, part of the stage identity."""
        return {
            "seed": SEED,
            "samples": SAMPLES,
            "batch": BATCH,
            "regions": list(ra.REGION_GRID),
            "fill": "black_rgb0",
            "shared_bernoulli": ["P22", "P42", "P44"],
            "lime": {"width": 0.25, "ridge": 1.0},
            "occlusion": {"cells": 6, "window": 2, "stride": 1},
            "channels": self.channels.tolist(),
            "mask_rule": asdict(MASS70),
            "precision": self.detector.scorer.precision,
        }

    def fit_seconds(self) -> dict[str, float]:
        """:return: the gate, plus the train channel choice for P45/P46."""
        fits = dict.fromkeys(self.methods, self.gate_fit_seconds)
        for name in ("P45-channels32", "P46-channels32"):
            fits[name] += self.channel_fit_seconds
        return fits

    def begin_video(self, meta: VideoMeta, items: list, s0: S0Video) -> None:
        """No per-video state."""

    def _region_keep(self, presence: np.ndarray, grid) -> torch.Tensor:
        index = torch.as_tensor(ra.region_index(grid))
        return torch.as_tensor(presence)[:, index]

    def chunk(self, meta: VideoMeta, item, index: int) -> ChunkResult:
        """Run every perturbation of one gated chunk."""
        d = self.detector
        start = d.sync()
        clip = prepare_clip(self.engine, item)
        decoded = d.sync()
        feats, logit = extract_features(self.engine, clip)
        encoded = d.sync()
        grid = tuple(feats.grid)
        black = reference_clip(clip, "black", mean=self.spec.mean, std=self.spec.std)
        seconds = {}
        bern, seconds["bernoulli"] = d.logits(
            d.masked(clip, black, self._region_keep(self.bernoulli, grid)), SAMPLES
        )
        singles_presence = ~np.eye(32, dtype=bool)
        singles, seconds["singles"] = d.logits(
            d.masked(clip, black, self._region_keep(singles_presence, grid)), 32
        )
        coal, seconds["kernel_shap"] = d.logits(
            d.masked(clip, black, self._region_keep(self.coalitions, grid)), SAMPLES
        )
        (black_logit,), seconds["black"] = d.logits(lambda b, e: black, 1)
        occlusion = ra.occlusion_token_masks(grid)
        occ, seconds["occlusion"] = d.logits(
            d.masked(clip, black, torch.as_tensor(~occlusion)), len(occlusion)
        )
        tokens = feats.tokens[0].float()
        activations, kept = [], []
        for channel in self.channels:
            amap = tokens[:, channel].reshape(grid)
            if float(amap.max() - amap.min()) > 1e-8:
                activations.append(amap)
                kept.append(int(channel))
        acts = torch.stack(activations) if activations else torch.zeros((0, *grid))
        if activations:
            low = acts.amin(dim=(1, 2, 3), keepdim=True)
            soft = (acts - low) / (acts.amax(dim=(1, 2, 3), keepdim=True) - low)
            score_logits, seconds["score_cam"] = d.logits(d.masked(clip, black, soft), len(soft))
        else:
            score_logits, seconds["score_cam"] = np.zeros(0), 0.0
        begin = d.sync()
        ablated = []
        with torch.no_grad(), d.scorer.autocast():
            for channel in self.channels:
                zeroed = feats.tokens.clone()
                zeroed[..., channel] = 0
                ablated.append(
                    float(d.scorer.head(replace(feats, tokens=zeroed, pooled=None)).float())
                )
        seconds["ablation_cam"] = d.sync() - begin
        acts_np = acts.cpu().numpy().astype(np.float64)
        all_acts = np.stack(
            [tokens[:, c].reshape(grid).cpu().numpy() for c in self.channels]
        ).astype(np.float64)

        tolerance = 1e-6 * max(1.0, abs(logit))
        hsic = ra.hsic_scores(self.bernoulli, bern)
        hsic = np.where(logit - singles > tolerance, hsic, 0.0)
        score = ra.score_cam(acts_np, score_logits - black_logit, logit)
        ablation = ra.ablation_cam(all_acts, np.asarray(ablated), logit)
        zero = np.zeros(grid)
        maps = {
            "P22-masks128": (ra.rise_map(self.bernoulli, bern, grid), seconds["bernoulli"]),
            "P41-spatial25": (ra.occlusion_map(occlusion, logit - occ), seconds["occlusion"]),
            "P42-samples128": (
                ra.regions_to_tokens(ra.lime_coefficients(self.bernoulli, bern), grid),
                seconds["bernoulli"],
            ),
            "P43-samples128": (
                ra.regions_to_tokens(
                    ra.kernel_shap(self.coalitions, coal, black_logit, logit), grid
                ),
                seconds["kernel_shap"] + seconds["black"],
            ),
            "P44-masks128": (
                ra.regions_to_tokens(hsic, grid),
                seconds["bernoulli"] + seconds["singles"],
            ),
            "P45-channels32": (
                zero if score is None else score,
                seconds["score_cam"] + seconds["black"],
            ),
            "P46-channels32": (zero if ablation is None else ablation, seconds["ablation_cam"]),
        }
        outputs, saved = {}, {}
        for name, (values, cost) in maps.items():
            post = time.perf_counter()
            boxes = native_boxes(map_mask(values, MASS70), meta.width, meta.height)
            outputs[name] = MethodOutput(boxes, cost + time.perf_counter() - post)
            saved[name] = np.asarray(values, dtype=np.float32)
        return ChunkResult(
            outputs,
            maps=saved,
            receipts={
                "forward_seconds": seconds,
                "black_logit": black_logit,
                "score_cam_channels_kept": kept,
                "score_cam_empty": score is None,
                "ablation_cam_empty": ablation is None,
                "hsic_regions_kept": int((hsic > 0).sum()),
            },
            logit=logit,
            stage_decode_seconds=decoded - start,
            stage_encode_seconds=encoded - decoded,
        )


def total_variation(mask: torch.Tensor) -> torch.Tensor:
    """:return: spatial anisotropic L1 total variation summed over planes."""
    return (mask[:, 1:, :] - mask[:, :-1, :]).abs().sum() + (
        mask[:, :, 1:] - mask[:, :, :-1]
    ).abs().sum()


class ExtremalStage:
    """P23: a token-grid mask preserving the anomaly logit at 20% area, by projected ascent."""

    name = "extremal"
    methods = (EXTREMAL,)

    def __init__(self, engine, spec, gate_fit_seconds: float) -> None:
        self.engine, self.spec = engine, spec
        self.detector = _Detector(engine, spec)
        self.gate_fit_seconds = gate_fit_seconds

    def parameters(self) -> dict:
        """:return: frozen parameters, part of the stage identity."""
        return {
            **EXTREMAL_SETTINGS,
            "init": "uniform at the area budget",
            "step_rule": "gradient divided by its max abs value (step = max change per token)",
            "projection": "exact on [0,1] with fixed mean (bisection)",
            "reference": "black_rgb0",
            "binarization": "first ceil(0.2 P) positive values, stable ties",
            "precision": self.detector.scorer.precision,
        }

    def fit_seconds(self) -> dict[str, float]:
        """:return: the shared gate only; the per-query optimization is inference."""
        return {EXTREMAL: self.gate_fit_seconds}

    def begin_video(self, meta: VideoMeta, items: list, s0: S0Video) -> None:
        """No per-video state."""

    def chunk(self, meta: VideoMeta, item, index: int) -> ChunkResult:
        """Optimize the preservation mask of one gated chunk."""
        d, scorer, s = self.detector, self.detector.scorer, EXTREMAL_SETTINGS
        start = d.sync()
        clip = prepare_clip(self.engine, item)
        decoded = d.sync()
        feats, logit = extract_features(self.engine, clip)
        encoded = d.sync()
        grid = tuple(feats.grid)
        black = reference_clip(clip, "black", mean=self.spec.mean, std=self.spec.std)
        mask = torch.full(grid, s["area"], device=clip.device, dtype=torch.float32)
        objective = []
        for _ in range(s["steps"]):
            current = mask.clone().requires_grad_(True)
            with torch.enable_grad(), scorer.autocast():
                pixels = token_to_pixel(current[None], tuple(clip.shape)).to(clip.dtype)
                preserved = clip * pixels + black * (1 - pixels)
                feats_p = encode_multilevel(scorer.backbone, preserved, scorer.levels)
                value = scorer.head(feats_p).float().sum() - s["tv_weight"] * total_variation(
                    current
                )
                gradient = torch.autograd.grad(value, current)[0]
            objective.append(float(value.detach()))
            scale = float(gradient.abs().max())
            if not math.isfinite(scale):
                raise RuntimeError("Non-finite gradient in the extremal mask")
            step = mask + s["step"] * gradient / scale if scale > 0 else mask
            mask = torch.as_tensor(
                ra.project_mean_box(step.cpu().numpy(), s["area"]),
                device=clip.device,
                dtype=torch.float32,
            )
            del current, pixels, preserved, feats_p, gradient
        values = mask.cpu().numpy()
        optimized = d.sync()
        boxes = native_boxes(ra.top_fraction_mask(values, s["area"]), meta.width, meta.height)
        method_seconds = d.sync() - encoded
        return ChunkResult(
            {EXTREMAL: MethodOutput(boxes, method_seconds)},
            maps={EXTREMAL: values.astype(np.float32)},
            receipts={
                "objective_first_last": [objective[0], objective[-1]],
                "mask_mean": float(values.mean()),
                "optimization_seconds": optimized - encoded,
            },
            logit=logit,
            stage_decode_seconds=decoded - start,
            stage_encode_seconds=encoded - decoded,
        )
