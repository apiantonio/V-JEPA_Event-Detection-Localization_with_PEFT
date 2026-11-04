"""One descriptor per window: the mean of the backbone tokens.

Used by the 2D projections (PCA, UMAP, t-SNE) showing whether representations separate
anomaly from normality and classes from each other. The descriptor is the token mean
of the read layer (a 1024-vector per window on ViT-L), the same quantity
:func:`evd.explain.concepts.fit_binary_references` calls a descriptor.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from evd.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

    from evd.data.feature_store import SafetensorsFeatureStore
    from evd.data.windows import CausalWindowSampler
    from evd.infer.engine import InferenceEngine
    from evd.types import PreprocessSpec, VideoMeta

log = get_logger(__name__)

__all__ = [
    "DescriptorCell",
    "Descriptors",
    "load_descriptors",
    "descriptors_from_forward",
    "descriptors_from_store",
    "save_descriptors",
]


@dataclass(frozen=True)
class Descriptors:
    """Descriptors of a set of windows, in reading order.

    :param window_id: window identities (ADR-007), one per row of ``valori``; the key used
        to join labels from ``scores.jsonl``.
    :param video_id: video of each window, repeated.
    :param valori: ``[N, D]`` in fp32.
    """

    window_id: tuple[str, ...]
    video_id: tuple[str, ...]
    valori: np.ndarray

    def __post_init__(self) -> None:
        if self.valori.ndim != 2:
            raise ValueError(f"expected descriptors [N, D], not {self.valori.shape}")
        if len(self.window_id) != len(self.valori) or len(self.video_id) != len(self.valori):
            raise ValueError(
                f"identities and values do not match: {len(self.window_id)} window_id, "
                f"{len(self.video_id)} video_id, {len(self.valori)} rows"
            )
        if len(set(self.window_id)) != len(self.window_id):
            raise ValueError("repeated window_id: two rows would describe the same window")
        # A NaN is realistic here (bf16 overflow with eager attention), and a PCA on a matrix
        # with a NaN does not raise.
        if not np.isfinite(self.valori).all():
            n = int((~np.isfinite(self.valori)).any(axis=1).sum())
            raise ValueError(
                f"{n} of {len(self.valori)} descriptors are not finite: the forward overflowed "
                f"or the read is corrupted. No dimensionality reduction would report it "
                f"downstream"
            )

    def __len__(self) -> int:
        return len(self.valori)

    @property
    def dim(self) -> int:
        return int(self.valori.shape[1])


def _means(tokens: torch.Tensor) -> np.ndarray:
    """Return the token mean, in fp32.

    :param tokens: ``[B, N, D]``.
    :return: ``[B, D]`` NumPy fp32 array, detached and on CPU.
    :raises ValueError: if the tokens are not ``[B, N, D]``.
    """
    if tokens.ndim != 3:
        raise ValueError(f"expected tokens [B, N, D], not {tuple(tokens.shape)}")
    return tokens.float().mean(dim=1).cpu().numpy().astype(np.float32, copy=False)


def _single_level(levels: Sequence[int]) -> int:
    """:raises ValueError: if there is more than one level (the mean would mix concatenated
    levels).
    """
    if len(levels) != 1:
        raise ValueError(
            f"descriptors are computed on a single level, not on {list(levels)}: the mean "
            f"over concatenated tokens represents none of the levels"
        )
    return int(levels[0])


def descriptors_from_forward(
    engine: InferenceEngine,
    metas: Sequence[VideoMeta],
    *,
    log_every: int = 2000,
) -> Descriptors:
    """Recompute the descriptors with the engine model, one loader for the whole corpus.

    :param engine: engine already built (``live_engine`` for a closed run); its
        ``num_workers`` decides whether decoding overlaps the forward.
    :param metas: videos, in reading order.
    :param log_every: progress logging period, in windows.
    :return: descriptors, in temporal order within each video and across videos.
    :raises ValueError: if the scorer reads more than one level, the preprocessing yields
        several views, or a descriptor is not finite.
    :raises RuntimeError: if the number of produced rows differs from the requested
        windows.
    """
    import torch

    from evd.data.dataset import build_adapted_loader
    from evd.models.backbones.base import encode_multilevel

    _single_level(engine.scorer.levels)
    items = [it for m in metas for it in engine.finestre(m.path, m)]
    if not items:
        return Descriptors((), (), np.zeros((0, 0), dtype=np.float32))

    # A single loader over all windows, `shuffle=False`: decoding of video *n+1* overlaps the
    # forward of video *n*. No augmentation, so workers do not change the pixels. Going
    # through `build_adapted_loader` also for a frozen backbone keeps both halves of a joint
    # projection on the same path (no cache).
    loader = build_adapted_loader(
        items,
        decoder=engine.decoder,
        spec=engine.spec,
        batch_size=engine.batch_size,
        shuffle=False,
        num_workers=engine.num_workers,
        augment=None,
    )

    window_id: list[str] = []
    video_id: list[str] = []
    blocks: list[np.ndarray] = []
    for batch in loader:
        clips, _, n_views = engine.scorer.prepare_clips(batch)
        if n_views is not None:
            raise ValueError(
                "the preprocessing yields several views per window: a per-window descriptor "
                "is undefined until it is decided how to average them"
            )
        with torch.no_grad(), engine.scorer.autocast():
            feats = encode_multilevel(engine.scorer.backbone, clips, engine.scorer.levels)
        blocks.append(_means(feats.tokens))
        for w in batch["window"]:
            window_id.append(w.window_id)
            video_id.append(w.video_id)
        if log_every and len(window_id) % log_every < engine.batch_size:
            log.info("descriptors: %d/%d windows", len(window_id), len(items))

    if len(window_id) != len(items):
        raise RuntimeError(
            f"the loader returned {len(window_id)} descriptors for {len(items)} windows: "
            f"joining the labels would be silently wrong"
        )
    return Descriptors(tuple(window_id), tuple(video_id), np.concatenate(blocks, axis=0))


def descriptors_from_store(
    store: SafetensorsFeatureStore,
    spec: PreprocessSpec,
    sampler: CausalWindowSampler,
    metas: Sequence[VideoMeta],
    *,
    layer: int,
    log_every: int = 2000,
) -> Descriptors:
    """Read the already extracted tokens and average them, without building a model.

    Mirrors ``scripts/replay_store.py``: the key comes from the store metadata and the
    protocol grid, so it runs without GPU or weights.

    :param store: store opened with ``open_existing``.
    :param spec: store preprocessing (``store.preprocess``), used to build the key.
    :param sampler: protocol sampler from :func:`evd.protocol.chunks.sampler_for_protocol`;
        a sampler of another protocol gives zero hits (ADR-049).
    :param metas: videos, in reading order.
    :param layer: level to read (a single one, see :func:`_single_level`).
    :return: descriptors, in the same order as :func:`descriptors_from_forward`.
    :raises KeyError: at the first window missing from the store, naming it.
    """
    window_id: list[str] = []
    video_id: list[str] = []
    rows: list[np.ndarray] = []
    for meta in metas:
        for window in sampler.windows_for_video(meta):
            key = store.key(store.backbone_id, store.ckpt_sha, spec, int(layer), window)
            output = store.get(key)
            if output is None:
                raise KeyError(
                    f"{window.window_id} (layer {layer}) is not in store {store.root}: "
                    f"the extraction is incomplete, or the sampler grid differs from the "
                    f"one the store was written with"
                )
            rows.append(_means(output.tokens)[0])
            window_id.append(window.window_id)
            video_id.append(window.video_id)
            if log_every and len(window_id) % log_every == 0:
                log.info("descriptors from the store: %d windows", len(window_id))
    if not rows:
        return Descriptors((), (), np.zeros((0, 0), dtype=np.float32))
    return Descriptors(tuple(window_id), tuple(video_id), np.stack(rows, axis=0))


@dataclass(frozen=True)
class DescriptorCell:
    """A cell read back from disk: descriptors, labels and provenance.

    Defined next to its writer because the two callers live in different environments
    (extraction with torch, reduction without); importable without torch.
    """

    nome: str
    valori: np.ndarray
    window_id: np.ndarray
    video_id: np.ndarray
    label: np.ndarray
    classe: np.ndarray
    score: np.ndarray
    meta: dict[str, Any]


def save_descriptors(
    percorso: Path,
    descrittori: Descriptors,
    *,
    label: np.ndarray,
    classe: np.ndarray,
    score: np.ndarray,
    meta: Mapping[str, Any],
) -> Path:
    """Write a cell to a compressed ``.npz``, with provenance inside the file.

    :param percorso: file to write; parent folders are created.
    :return: the written path.
    """
    percorso = Path(percorso)
    percorso.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        percorso,
        valori=descrittori.valori,
        window_id=np.asarray(descrittori.window_id),
        video_id=np.asarray(descrittori.video_id),
        label=np.asarray(label),
        classe=np.asarray(classe),
        score=np.asarray(score),
        meta=np.asarray(json.dumps(dict(meta), ensure_ascii=False)),
    )
    return percorso


def load_descriptors(percorso: Path) -> DescriptorCell:
    """Read a cell written by :func:`save_descriptors`.

    :raises FileNotFoundError: if the file does not exist, listing the folder contents.
    """
    percorso = Path(percorso)
    if not percorso.is_file():
        available = sorted(p.stem for p in percorso.parent.glob("*.npz"))
        raise FileNotFoundError(f"{percorso} does not exist: the folder contains {available}")
    # `allow_pickle=False`: an .npz may come from another machine, and pickle would execute
    # code.
    data = np.load(percorso, allow_pickle=False)
    return DescriptorCell(
        nome=percorso.stem,
        valori=data["valori"],
        window_id=data["window_id"],
        video_id=data["video_id"],
        label=data["label"],
        classe=data["classe"],
        score=data["score"],
        meta=json.loads(str(data["meta"])),
    )
