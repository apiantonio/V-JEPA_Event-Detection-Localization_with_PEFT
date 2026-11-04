"""CRIS concept importance on every layer and head of the detector (P01).

port of the public VTCD code (``YorkUCVIL/VTCD``: ``run_vcd.py``, ``utilities/clustering.py``, ``CRIS.py``)

- concepts are discovered on the **keys** of every head of the 24 blocks, read at the fused
  QKV output (before RoPE, as the public hook); the volume is the **token grid**
  ``16 × 24 × 24`` instead of features upsampled to half the pixel resolution, which would
  make SLIC impractical on 384 heads;
- intra-video: SLIC with 12 segments and compactness 0.1 (public defaults), here the
  project's ``evd.explain.vtcd.slic_regions``; segment descriptor = mean key of the segment;
- inter-video: k-means elbow on 1..14 clusters with the public ratio test (stop when the
  mean squared error exceeds 0.95 times the previous one, keeping the failing k as the public
  code does), then Convex NMF with that k for 10,000 iterations (the project's
  ``convex_nmf`` instead of PyMF);
- discovery on 30 train videos, ranking on 30 **validation** videos with events, both
  class-balanced (the public code ranks the discovery videos themselves); validation
  segments are assigned to the nearest concept of their head (NNLS, ``assign_concepts``);
- ranking: 4,000 masks per video, each removing ``int(0.5 · C)`` concepts drawn uniformly;
  removal multiplies the removed tokens' keys by zero in their head (public intervention);
  importance is the paper's Eq. 6 with ``D = -BCE(logit, 1)``,
  ``s_i = mean_m (D_orig - D_m) · 1[i removed]``, averaged over videos (the paper's formula
  prevails over the public aggregation).

At test time the globally most important positive concept selects one layer and head: the
chunk's keys there are segmented, assigned, and the concept's tokens form the mask.
"""

from __future__ import annotations

import math
import multiprocessing
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn

from evd.exp.localization_auto import ChunkResult, MethodOutput, S0Video
from evd.explain.localization import native_boxes
from evd.explain.vtcd import assign_concepts, convex_nmf, slic_regions
from evd.models.backbones.base import encode_multilevel
from evd.types import VideoMeta

SEGMENTS, COMPACTNESS = 12, 0.1
MAX_CLUSTERS, ELBOW = 15, 0.95
NMF_ITERATIONS = 10_000
MASKS, RATIO, SEED = 4000, 0.5, 42
BATCH = 8
EXPERIMENT = "P01-top1"


def qkv_modules(model: nn.Module) -> list[nn.Module]:
    """:return: the fused QKV projection of every block, in block order."""
    found = []
    for name, module in model.named_modules():
        match = re.search(r"blocks\.(\d+)\..*attn\.qkv$", name)
        if match:
            found.append((int(match.group(1)), module))
    found.sort(key=lambda item: item[0])
    return [module for _, module in found]


class KeyHooks:
    """Capture or scale the keys of the video tokens at every fused QKV output.

    ``capture``: after a forward, ``keys[l]`` is ``B × N × heads × head_dim`` (video tokens).
    ``keep``: a list (one per block) of ``B × N × heads`` multipliers applied to the keys.
    """

    def __init__(self, modules: list[nn.Module], tokens: int, heads: int) -> None:
        self.modules, self.tokens, self.heads = modules, tokens, heads
        self.capture_layers: set[int] = set()
        self.keys: dict[int, torch.Tensor] = {}
        self.keep: list[torch.Tensor] | None = None
        self.handles = [m.register_forward_hook(self._hook(i)) for i, m in enumerate(modules)]

    def _hook(self, layer: int):
        def hook(module, inputs, output):
            width = output.shape[-1] // 3
            keys = output[:, : self.tokens, width : 2 * width]
            if layer in self.capture_layers:
                self.keys[layer] = keys.detach().reshape(*keys.shape[:2], self.heads, -1)
            if self.keep is None:
                return None
            scaled = keys.reshape(*keys.shape[:2], self.heads, -1) * self.keep[layer][..., None].to(
                keys.dtype
            )
            output = output.clone()
            output[:, : self.tokens, width : 2 * width] = scaled.reshape(keys.shape)
            return output

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def segment_head(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """SLIC one head's ``T×H×W×d`` keys. :return: labels, segment means, CPU seconds."""
    start = time.process_time()
    labels, descriptors, _ = slic_regions(keys, n_segments=SEGMENTS, compactness=COMPACTNESS)
    return labels, descriptors, time.process_time() - start


def elbow_from_errors(errors: list[float]) -> int:
    """Public ratio test: the first k whose error exceeds 0.95 times the previous is kept.

    :param errors: mean squared error for k = 1, 2, ...
    :return: the chosen number of clusters (the last k tried if the test never fires).
    """
    for k in range(2, len(errors) + 1):
        if errors[k - 1] > ELBOW * errors[k - 2]:
            return k
    return len(errors)


def elbow_clusters(descriptors: np.ndarray, seed: int = SEED) -> int:
    """k-means mean squared error for k = 1..14 (fewer with fewer segments), then the test."""
    from sklearn.cluster import KMeans

    errors = []
    for k in range(1, min(MAX_CLUSTERS, len(descriptors))):
        model = KMeans(n_clusters=k, n_init=1, random_state=seed).fit(descriptors)
        errors.append(model.inertia_ / len(descriptors))
        if elbow_from_errors(errors) < len(errors) or (
            len(errors) > 1 and errors[-1] > ELBOW * errors[-2]
        ):
            break
    return elbow_from_errors(errors)


def fit_head(descriptors: np.ndarray) -> tuple[np.ndarray, dict]:
    """Elbow then Convex NMF on one head's segment means. :return: ``k × d`` concepts, receipt."""
    start = time.process_time()
    k = elbow_clusters(descriptors)
    factors, receipt = convex_nmf(
        descriptors, n_concepts=k, seed=SEED, max_iter=NMF_ITERATIONS, tolerance=0.0
    )
    return factors["dictionary"], {
        "k": k,
        "segments": len(descriptors),
        "converged": receipt["converged"],
        "iterations": receipt["iterations"],
        "cpu_seconds": time.process_time() - start,
    }


def assign_head(keys: np.ndarray, dictionary: np.ndarray) -> tuple[np.ndarray, float]:
    """Segment one head's keys and give each token the concept of its segment."""
    labels, descriptors, seconds = segment_head(keys)
    start = time.process_time()
    ids, _ = assign_concepts(descriptors, dictionary)
    return ids[labels].astype(np.int16), seconds + time.process_time() - start


def eq6_importance(removed: np.ndarray, performance: np.ndarray, original: float) -> np.ndarray:
    """Paper Eq. 6: ``s_i = mean_m (D_orig - D_m) · 1[i removed]``.

    :param removed: ``M × C`` boolean.
    :param performance: ``M`` values of ``D`` on the masked forwards.
    """
    return ((original - np.asarray(performance))[:, None] * removed).mean(0)


def performance(logits: np.ndarray) -> np.ndarray:
    """:return: ``D = -BCE(logit, 1) = -softplus(-logit)``."""
    return -np.logaddexp(0.0, -np.asarray(logits, dtype=np.float64))


def make_pool(workers: int) -> ProcessPoolExecutor:
    """:return: a spawn pool (workers run numpy only and must not inherit CUDA)."""
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))


class CRISStage:
    """P01: the top CRIS concept's support on the gated chunks."""

    name = "cris"
    methods = (EXPERIMENT,)

    def __init__(self, engine, spec, gate_fit_seconds: float, fits: Path) -> None:
        from evd.exp.explanation_advanced import prepare_clip

        self.engine, self.scorer, self.prepare = engine, engine.scorer, prepare_clip
        self.cuda = next(self.scorer.head.parameters()).is_cuda
        model = np.load(fits / "cris_top.npz")
        self.layer, self.head, self.concept = (int(model[k]) for k in ("layer", "head", "concept"))
        self.dictionary = model["dictionary"]
        self.fit = gate_fit_seconds + float(model["fit_seconds"])
        backbone = self.scorer.backbone.module()
        self.heads = int(model["heads"])
        self.hooks = KeyHooks(qkv_modules(backbone), int(model["tokens"]), self.heads)
        self.hooks.capture_layers = {self.layer}

    def parameters(self) -> dict:
        """:return: frozen parameters, part of the stage identity."""
        return {
            "layer": self.layer,
            "head": self.head,
            "concept": self.concept,
            "segments": SEGMENTS,
            "compactness": COMPACTNESS,
            "precision": self.scorer.precision,
        }

    def fit_seconds(self) -> dict[str, float]:
        """:return: gate plus discovery and ranking."""
        return {EXPERIMENT: self.fit}

    def begin_video(self, meta: VideoMeta, items: list, s0: S0Video) -> None:
        """No per-video state."""

    def _sync(self) -> float:
        if self.cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    def chunk(self, meta: VideoMeta, item, index: int) -> ChunkResult:
        """Segment the chunk at the selected head and keep the selected concept."""
        start = self._sync()
        clip = self.prepare(self.engine, item)
        decoded = self._sync()
        with torch.no_grad(), self.scorer.autocast():
            feats = encode_multilevel(self.scorer.backbone, clip, self.scorer.levels)
            logit = float(self.scorer.head(feats).float().item())
        encoded = self._sync()
        grid = tuple(feats.grid)
        keys = self.hooks.keys.pop(self.layer)[0, :, self.head].float().cpu().numpy()
        ids, seconds = assign_head(keys.reshape(*grid, -1), self.dictionary)
        post = time.perf_counter()
        mask = ids == self.concept
        boxes = native_boxes(mask, meta.width, meta.height)
        return ChunkResult(
            {EXPERIMENT: MethodOutput(boxes, seconds + time.perf_counter() - post)},
            maps={EXPERIMENT: ids.astype(np.float32)},
            receipts={"concept_tokens": int(mask.sum())},
            logit=logit,
            stage_decode_seconds=decoded - start,
            stage_encode_seconds=encoded - decoded,
        )


def removal_masks(concepts: int, seed: int) -> np.ndarray:
    """:return: ``MASKS × concepts`` boolean, each row removing ``int(RATIO·C)`` concepts."""
    rng = np.random.default_rng(seed)
    count = int(concepts * RATIO)
    masks = np.zeros((MASKS, concepts), dtype=bool)
    for row in masks:
        row[rng.choice(concepts, size=count, replace=False)] = True
    return masks


def keep_tensors(ids: list[np.ndarray], removed: np.ndarray, device) -> list[torch.Tensor]:
    """Per-block keep multipliers for a batch of masks.

    :param ids: one ``N × heads`` array of global concept ids per block.
    :param removed: ``B × C`` boolean.
    :return: one ``B × N × heads`` float tensor per block.
    """
    table = torch.as_tensor(~removed, device=device, dtype=torch.float32)
    return [
        table[:, torch.as_tensor(layer_ids.astype(np.int64), device=device)] for layer_ids in ids
    ]


def rank_video(
    scorer, hooks: KeyHooks, clip: torch.Tensor, ids: list[np.ndarray], concepts: int, seed: int
) -> tuple[np.ndarray, dict]:
    """Run the 4,000 masked forwards of one ranking video. :return: Eq. 6 scores, receipt."""
    device = clip.device
    hooks.keep = None
    with torch.no_grad(), scorer.autocast():
        original = float(
            scorer.head(encode_multilevel(scorer.backbone, clip, scorer.levels)).float().item()
        )
    masks = removal_masks(concepts, seed)
    logits = []
    start = time.perf_counter()
    with torch.no_grad(), scorer.autocast():
        for begin in range(0, MASKS, BATCH):
            batch = masks[begin : begin + BATCH]
            hooks.keep = keep_tensors(ids, batch, device)
            feats = encode_multilevel(
                scorer.backbone, clip.expand(len(batch), *clip.shape[1:]), scorer.levels
            )
            logits.append(scorer.head(feats).float().reshape(-1).cpu().numpy())
            hooks.keep = None
    if clip.is_cuda:
        torch.cuda.synchronize()
    values = np.concatenate(logits).astype(np.float64)
    d = performance(values)
    d0 = float(performance(np.array([original]))[0])
    return eq6_importance(masks, d, d0), {
        "original_logit": original,
        "mean_masked_logit": float(values.mean()),
        "forward_seconds": time.perf_counter() - start,
        "removed_per_mask": int(masks[0].sum()),
        "finite": bool(np.isfinite(values).all() and math.isfinite(original)),
    }
