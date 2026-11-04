"""Integrated Gradients stage of the automatic-localization campaign.

One explanation of the original clip (:func:`evd.exp.explanation_campaign.explain_clip`,
four synthetic baselines black/noise/mean/blur, signed mean = Expected Gradients) feeds the
five threshold rules and the four-baseline vote; P36 replaces the baseline with the
encoding of the video's first protocol chunk. Every shared computation is charged in full
to each experiment that needs it.

P16 (absolute threshold calibrated on normal validation chunks) and P07 (VTCD concepts
weighted by IG mass) reuse the ensemble maps saved here and run as post-processing.
"""

from __future__ import annotations

import time
from dataclasses import asdict, replace

import numpy as np
import torch

from evd.exp.explanation_advanced import extract_features, prepare_clip
from evd.exp.explanation_campaign import explain_clip, seed_from_window_id
from evd.exp.localization_auto import ChunkResult, MethodOutput, S0Video
from evd.explain.attribution import integrated_gradients
from evd.explain.localization import MaskRule, map_mask, native_boxes
from evd.types import VideoMeta

IG_STEPS = (16, 32, 64)
IG_FILLS = ("black", "noise", "mean", "blur")
IG_BATCH = 16
IG_TOLERANCE = {"atol": 0.01, "rtol": 0.01}
THRESHOLD_RULES = {
    "P12-range70": MaskRule("range", 0.7),
    "P13-count10": MaskRule("count", 0.1),
    "P14-mass70": MaskRule("mass", 0.7),
    "P15-otsu256": MaskRule("otsu", bins=256),
    "P17-range70_40": MaskRule("hysteresis", 0.7, low=0.4),
}
MASS70 = MaskRule("mass", 0.7)
VOTE = ("P18-vote3of4", 3)
FIRST_CHUNK = "P36-first_chunk"
METHODS = (*THRESHOLD_RULES, VOTE[0], FIRST_CHUNK)


def explain_ig(scorer, spec, clip: torch.Tensor, window_id: str):
    """Explain a preprocessed clip with the four synthetic baselines and schedule I1.

    Shared by the test stage and the P16 validation calibration, so the calibrated maps are
    the maps the threshold is applied to.

    :return: the :func:`explain_clip` tuple ``(maps, checks, feats, logit, timings)``.
    """
    return explain_clip(
        scorer,
        clip,
        ("ig",),
        spec=spec,
        ig_kwargs={"steps": IG_STEPS, "batch_size": IG_BATCH, **IG_TOLERANCE},
        noise_seed=seed_from_window_id(window_id),
        ig_fills=IG_FILLS,
    )


def vote_mask(maps: list[np.ndarray], votes: int) -> np.ndarray:
    """:return: tokens selected by the 70%-mass rule in at least ``votes`` maps."""
    return np.sum([map_mask(m, MASS70) for m in maps], axis=0) >= votes


class IGStage:
    """IG experiments computed in one pass over the chunks above the gate."""

    name = "ig"
    methods = METHODS

    def __init__(self, engine, spec, gate_fit_seconds: float) -> None:
        self.engine, self.spec = engine, spec
        self.scorer = engine.scorer
        self.cuda = next(self.scorer.head.parameters()).is_cuda
        self.gate_fit_seconds = gate_fit_seconds
        self._first_item = None
        self._first = None

    def parameters(self) -> dict:
        """:return: frozen parameters, part of the stage identity."""
        return {
            "ig_steps": list(IG_STEPS),
            "ig_fills": list(IG_FILLS),
            "ig_batch": IG_BATCH,
            "ig_tolerance": IG_TOLERANCE,
            "ensemble": "signed_mean_of_fills",
            "rules": {name: asdict(rule) for name, rule in THRESHOLD_RULES.items()},
            "vote": {"experiment": VOTE[0], "votes": VOTE[1], "rule": asdict(MASS70)},
            "p36_reference": "encoding_of_first_protocol_chunk_of_the_video",
            "precision": self.scorer.precision,
        }

    def fit_seconds(self) -> dict[str, float]:
        """:return: the shared gate fit, the only fitted quantity of these experiments."""
        return dict.fromkeys(self.methods, self.gate_fit_seconds)

    def begin_video(self, meta: VideoMeta, items: list, s0: S0Video) -> None:
        """Reset the per-video P36 reference; it is encoded lazily on first use."""
        self._first_item, self._first = items[0], None

    def _sync(self) -> float:
        if self.cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    def _explain(self, clip: torch.Tensor, window_id: str):
        return explain_ig(self.scorer, self.spec, clip, window_id)

    def _first_features(self) -> tuple[object, float]:
        """:return: the P36 reference features and the seconds spent recomputing them."""
        if self._first is not None:
            return self._first, 0.0
        start = self._sync()
        clip = prepare_clip(self.engine, self._first_item)
        self._first, _ = extract_features(self.engine, clip)
        return self._first, self._sync() - start

    def chunk(self, meta: VideoMeta, item, index: int) -> ChunkResult:
        """Compute the seven IG experiments on one gated chunk."""
        start = self._sync()
        clip = prepare_clip(self.engine, item)
        decoded = self._sync()
        maps, checks, feats, logit, timings = self._explain(clip, item.window.window_id)
        ig_seconds = timings["explanation_s"] - timings["forward_s"]
        ensemble = maps["ig_ensemble"]
        outputs = {}

        def boxes_of(mask: np.ndarray) -> list:
            return native_boxes(mask, meta.width, meta.height)

        for name, rule in THRESHOLD_RULES.items():
            begin = time.perf_counter()
            boxes = boxes_of(map_mask(ensemble, rule))
            outputs[name] = MethodOutput(boxes, ig_seconds + time.perf_counter() - begin)
        begin = time.perf_counter()
        boxes = boxes_of(vote_mask([maps[f"ig_{fill}"] for fill in IG_FILLS], VOTE[1]))
        outputs[VOTE[0]] = MethodOutput(boxes, ig_seconds + time.perf_counter() - begin)

        reference, recompute_seconds = self._first_features()
        begin = self._sync()

        def head_forward(tokens, feats=feats):
            with self.scorer.autocast():
                return self.scorer.head(replace(feats, tokens=tokens, pooled=None))

        values, first_receipt = integrated_gradients(
            head_forward,
            feats.tokens,
            reference.tokens,
            steps=IG_STEPS,
            batch_size=IG_BATCH,
            **IG_TOLERANCE,
        )
        first_map = values.sum(-1).reshape(feats.grid).cpu().numpy()
        boxes = boxes_of(map_mask(first_map, MASS70))
        outputs[FIRST_CHUNK] = MethodOutput(boxes, self._sync() - begin)

        saved = {f"ig_{fill}": maps[f"ig_{fill}"] for fill in IG_FILLS}
        saved.update(ig_ensemble=ensemble, p36_first_chunk=first_map)
        receipts = {
            "ig": {fill: checks[f"ig_{fill}"] for fill in IG_FILLS},
            "ig_all_converged": all(checks[f"ig_{fill}"]["converged"] for fill in IG_FILLS),
            "p36": first_receipt,
            "p36_reference_window": self._first_item.window.window_id,
            "p36_reference_recompute_seconds_not_charged": recompute_seconds,
            "forward_logit_delta": checks["forward_logit_delta"],
        }
        return ChunkResult(
            outputs,
            maps={k: np.asarray(v, dtype=np.float32) for k, v in saved.items()},
            receipts=receipts,
            logit=logit,
            stage_decode_seconds=decoded - start,
            stage_encode_seconds=timings["forward_s"],
        )
