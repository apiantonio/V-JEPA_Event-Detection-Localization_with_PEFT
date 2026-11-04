"""Stage of the automatic-localization campaign.

pass (``scripts/localization_auto_predict.py``) on the chunks above the gate. 
Boxes on every chunk holding an annotated plane, whatever the score) the
missing chunks go through :func:`evd.exp.localization_auto.run_stage` like every other
stage; the computation is the one of ``localization_auto_predict``: ``explain_head`` with
the extra CAMs, the map named in ``READOUTS``, the default mass-70 rule, one call and one
timing per experiment.
"""

from __future__ import annotations

import time

import torch

from evd.exp.localization_auto import ChunkResult, MethodOutput, S0Video
from evd.explain.attribution import explain_head
from evd.explain.localization import MaskRule, map_mask, native_boxes
from evd.types import VideoMeta

#: Experiment -> map of ``explain_head`` (same table as ``localization_auto_preflight``).
READOUTS = {"P37-last": "legrad_last", "P38-postnorm": "gradcam", "P39-postnorm": "layercam"}


class ReadoutStage:
    """The three gradient/attention readouts of the reduced plan on the selected chunks."""

    name = "readouts"
    methods = tuple(READOUTS)

    def __init__(self, engine, spec, gate_fit_seconds: float) -> None:
        from evd.exp.explanation_advanced import extract_features, prepare_clip

        self.engine, self.spec, self.scorer = engine, spec, engine.scorer
        self.cuda = next(self.scorer.head.parameters()).is_cuda
        self.extract, self.prepare = extract_features, prepare_clip
        self.gate_fit_seconds = gate_fit_seconds

    def parameters(self) -> dict:
        """:return: frozen parameters, part of the stage identity."""
        return {
            "readouts": READOUTS,
            "mask_rule": vars(MaskRule()),
            "precision": self.scorer.precision,
        }

    def fit_seconds(self) -> dict[str, float]:
        """:return: no learned parameter: the shared gate fit only, as in the S0 pass."""
        return dict.fromkeys(self.methods, self.gate_fit_seconds)

    def begin_video(self, meta: VideoMeta, items: list, s0: S0Video) -> None:
        """No per-video state."""

    def _sync(self) -> float:
        if self.cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    def chunk(self, meta: VideoMeta, item, index: int) -> ChunkResult:
        """Compute the three readouts of one chunk, each with its own timed call."""
        start = self._sync()
        clip = self.prepare(self.engine, item)
        decoded = self._sync()
        features, logit = self.extract(self.engine, clip)
        encoded = self._sync()
        outputs, maps = {}, {}
        for experiment, readout in READOUTS.items():
            begin = self._sync()
            with self.scorer.autocast():
                attribution = explain_head(self.scorer.head, features, extra_cams=True)
            if abs(attribution.logit - logit) > 1e-5:
                raise RuntimeError("Attribution altered the detector logit")
            values = attribution.maps[readout]
            boxes = native_boxes(map_mask(values, MaskRule()), meta.width, meta.height)
            outputs[experiment] = MethodOutput(boxes, self._sync() - begin)
            maps[experiment] = values
        return ChunkResult(
            outputs,
            maps=maps,
            logit=logit,
            stage_decode_seconds=decoded - start,
            stage_encode_seconds=encoded - decoded,
        )
