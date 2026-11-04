"""``WindowDataset``: from ``WindowSpec`` to pixels or frozen features."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from evd.data.augment import augment_clip, window_seed
from evd.data.decoder import VideoDecoder
from evd.data.feature_store import FeatureReader
from evd.data.preprocess import preprocess_clip
from evd.types import AugmentSpec, BackboneOutput, PreprocessSpec, WindowSpec, concat_levels
from evd.utils.logging import get_logger

log = get_logger(__name__)

#: Megabytes of decoded pixels kept in RAM before the dataset stops caching.
#:
#: Set to 0 (cache disabled): with-replacement sampling repeats few windows
#: (2772 unique indices out of 3000 draws) and the cache only added bookkeeping
#: cost. It is a module constant, not a config field, so it changes speed but never
#: the pixels a run sees.
CACHE_RAM_MB = 0


#: How dynamic sampling regenerates a window at each draw:
#: ``(item, draw, worker) -> item``. The positivity rule lives in
#: ``evd.protocol.chunks``; ``data`` only receives the function.
Resampler = Callable[["WindowItem", int, int], "WindowItem"]


def _bytes_of(tensor: Any) -> int:
    """:return: bytes used by a tensor, 0 if the object does not report them."""
    number = getattr(tensor, "numel", None)
    element = getattr(tensor, "element_size", None)
    return int(number() * element()) if callable(number) and callable(element) else 0


@dataclass(frozen=True)
class WindowItem:
    """Window with optional labels and optional cache identity.

    ``cache_keys`` are computed by the caller, which knows backbone, checkpoint and
    levels; the dataset only sees a read-only ``FeatureReader``.
    """

    window: WindowSpec
    video_path: str
    label: int | None = None
    class_label: int | None = None
    cache_keys: tuple[str, ...] = ()
    """One key per level, in stacking order; empty means no cache identity."""
    weight: float = 1.0
    """Unnormalized sampling weight (``w_video * w_overlap``).

    It lives in the data, not in the loader, so that every adapter family shares
    the same sampling. It is read only with ``weighted=True``; ``1.0`` is uniform.
    """


class WindowDataset(Dataset):
    """``WindowItem`` to preprocessed clip or already extracted tokens.

    Pixels are ``[C,T,H,W]`` for one view and ``[V,C,T,H,W]`` for ``three_views``; on a
    cache hit tokens are ``[N,D]`` (the batch dimension of the file is dropped and
    restored by the collate). The feature store is valid only for the frozen backbone,
    and ``build_adapted_loader`` does not expose ``cache``.
    """

    def __init__(
        self,
        items: Sequence[WindowItem],
        decoder: VideoDecoder,
        spec: PreprocessSpec,
        cache: FeatureReader | None = None,
        *,
        cache_in_memory: bool = False,
        cache_in_memory_mb: int = CACHE_RAM_MB,
        augment: AugmentSpec | None = None,
        run_seed: int = 0,
        resample: Resampler | None = None,
    ) -> None:
        self.items = list(items)
        self.decoder = decoder
        self.spec = spec
        self.cache = cache
        self.cache_hits = 0
        self.cache_misses = 0
        if resample is not None and cache is not None:
            raise ValueError(
                "dynamic sampling and feature store together: store keys are windows, and a "
                "randomly drawn window has no stable key (every draw would miss, or hit the "
                "static grid window and silently disable dynamic sampling)"
            )
        if cache is not None:
            without_key = [item.window.window_id for item in self.items if not item.cache_keys]
            if without_key:
                suffix = "..." if len(without_key) > 3 else ""
                raise ValueError(
                    "WindowDataset with a FeatureReader requires cache_keys on every WindowItem; "
                    f"missing for {without_key[:3]}{suffix}"
                )
            how_many = {len(item.cache_keys) for item in self.items}
            if len(how_many) > 1:
                raise ValueError(
                    f"inconsistent number of levels across windows: {sorted(how_many)}. "
                    "All windows of a dataset must read the same levels"
                )

        self.augment = augment if (augment is not None and augment.activate) else None
        self.run_seed = int(run_seed)
        self.resample = resample
        if self.augment is not None and cache is not None:
            raise ValueError(
                "augmentation and feature store together: store keys declare deterministic "
                "preprocessing, so a hit would return features of NON-augmented pixels. "
                "Build the augmented training loader with cache=None"
            )

        # Pixel cache in RAM (not on-disk features), valid also for PEFT. With augmentation
        # it stores the decoded uint8 frames, since the clip changes at every draw.
        self.cache_in_memory = cache_in_memory
        self._mem: dict[int, dict[str, Any]] = {}
        self._mem_frames: dict[int, Any] = {}
        self._draws: dict[int, int] = {}
        self._budget_ram = max(0, int(cache_in_memory_mb)) * 1_000_000
        self._ram_used = 0
        self._budget_exhausted = False
        if cache_in_memory:
            log.info(
                "dataset RAM cache: %d windows, budget %d MB, content %s",
                len(self.items),
                self._budget_ram // 1_000_000,
                "frame decodificati" if self.augment is not None else "clip preprocessate",
            )

    def __len__(self) -> int:
        return len(self.items)

    def _remember(self, where: dict[int, Any], i: int, valore: Any, byte: int) -> None:
        """Store ``valore`` in the RAM cache if the budget allows it.

        Once the budget is exhausted the dataset stops caching new entries, without
        raising or evicting: it stays correct, only slower.
        """
        if not self.cache_in_memory or self.resample is not None:
            return
        if self._ram_used + byte > self._budget_ram:
            if not self._budget_exhausted:
                self._budget_exhausted = True
                log.info(
                    "dataset RAM cache budget exhausted at %d/%d windows (%d MB): "
                    "recomputing from now on",
                    len(where),
                    len(self.items),
                    self._budget_ram // 1_000_000,
                )
            return
        where[i] = valore
        self._ram_used += byte

    def _drawn_item(self, i: int) -> WindowItem:
        """Return the window to serve now and advance the draw counter.

        With dynamic sampling the endpoint, and therefore the label, changes at every
        draw; ``resample`` recomputes it with the static-grid positivity rule.
        """
        item = self.items[i]
        if self.resample is None:
            return item
        from torch.utils.data import get_worker_info

        info = get_worker_info()
        draw = self._draws.get(i, 0)
        self._draws[i] = draw + 1
        return self.resample(item, draw, 0 if info is None else int(info.id))

    def __getitem__(self, i: int | tuple[int, bool]) -> dict[str, Any]:
        if isinstance(i, tuple):
            i, replay = i
            if replay:
                # Fast resume: advance the draw counters without decoding the clip.
                if self.resample is not None:
                    self._drawn_item(i)
                elif self.augment is not None:
                    self._draws[i] = self._draws.get(i, 0) + 1
                return {"_replay": True}
        # With dynamic sampling the RAM cache would hold the clip of another endpoint.
        if self.cache_in_memory and self.resample is None and i in self._mem:
            return self._mem[i]
        item = self._drawn_item(i)
        common = {
            "label": -1 if item.label is None else int(item.label),
            "class_label": -1 if item.class_label is None else int(item.class_label),
            "window": item.window,
        }
        if self.augment is not None:
            return {**common, "clip": self._augmented_clip(i, item)}
        if self.cache is not None:
            assert item.cache_keys  # checked in __init__
            levels = [self.cache.get(key) for key in item.cache_keys]
            # All or nothing: a missing level makes the window a miss.
            if all(level is not None for level in levels):
                for level_value in levels:
                    assert level_value is not None
                    if level_value.tokens.shape[0] != 1:
                        raise ValueError(
                            "a per-window feature must have batch=1, got "
                            f"{tuple(level_value.tokens.shape)} for {item.window.window_id}"
                        )
                cached = concat_levels(
                    [level_value for level_value in levels if level_value is not None]
                )
                self.cache_hits += 1
                out = {
                    **common,
                    "features": cached.tokens.squeeze(0),
                    "grid": cached.grid,
                    "layer": cached.layer,
                    "levels": cached.levels,
                    "pooled": None if cached.pooled is None else cached.pooled.squeeze(0),
                }
                self._remember(self._mem, i, out, _bytes_of(cached.tokens))
                return out
            self.cache_misses += 1

        frames = self.decoder.read_frames(item.video_path, item.window.frame_indices)
        clip = preprocess_clip(frames, self.spec)
        out = {**common, "clip": clip}
        self._remember(self._mem, i, out, _bytes_of(clip))
        return out

    def _augmented_clip(self, i: int, item: WindowItem) -> torch.Tensor:
        """Return the augmented training clip, seeded by the draw count.

        Decoded frames are cached, the clip is not. The counter is per worker and the
        worker index enters the seed (:func:`evd.data.augment.window_seed`).
        """
        from torch.utils.data import get_worker_info

        frames = self._mem_frames.get(i) if self.resample is None else None
        if frames is None:
            frames = self.decoder.read_frames(item.video_path, item.window.frame_indices)
            self._remember(self._mem_frames, i, frames, int(getattr(frames, "nbytes", 0)))
        if self.resample is None:
            draw = self._draws.get(i, 0)
            self._draws[i] = draw + 1
        else:
            # With dynamic sampling `_drawn_item` already advanced the counter.
            draw = self._draws.get(i, 1) - 1
        info = get_worker_info()
        seed = window_seed(
            self.run_seed, item.window, draw, worker=0 if info is None else int(info.id)
        )
        return augment_clip(frames, self.spec, self.augment, seed=seed)

    @property
    def cache_hit_rate(self) -> float | None:
        attempted = self.cache_hits + self.cache_misses
        return self.cache_hits / attempted if attempted else None


def _fully_cached(item: WindowItem, cache: FeatureReader) -> bool:
    """:return: ``True`` only if every level of the window is in the store."""
    return bool(item.cache_keys) and all(key in cache for key in item.cache_keys)


def collate_windows(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate a batch explicitly, rejecting batches that mix features and pixels.

    :param batch: dataset items.
    :return: the collated batch.
    :raises ValueError: on empty, mixed or inconsistent batches.
    """
    if not batch:
        raise ValueError("collate_windows cannot aggregate an empty batch")
    if all(row.get("_replay") is True for row in batch):
        return {"_replay": True}
    if any(row.get("_replay") is True for row in batch):
        raise ValueError("mixed replay/training batch")
    feature_rows = ["features" in row for row in batch]
    if any(feature_rows) and not all(feature_rows):
        raise ValueError("mixed feature/pixel batch: split cache hits and misses before collating")
    out: dict[str, Any] = {
        "label": torch.tensor([row["label"] for row in batch], dtype=torch.long),
        "class_label": torch.tensor([row["class_label"] for row in batch], dtype=torch.long),
        "window": [row["window"] for row in batch],
    }
    if not all(feature_rows):
        out["clip"] = torch.stack([row["clip"] for row in batch])
        return out

    grids = {tuple(row["grid"]) for row in batch}
    layers = {int(row["layer"]) for row in batch}
    # A missing `levels` key means a single level.
    levels = {tuple(row.get("levels", ())) for row in batch}
    if len(grids) != 1 or len(layers) != 1 or len(levels) != 1:
        raise ValueError(
            f"incompatible features in the batch: grid={grids}, layer={layers}, levels={levels}"
        )
    pooled_rows = [row["pooled"] is not None for row in batch]
    if any(pooled_rows) and not all(pooled_rows):
        raise ValueError("batch with pooled features only for some items")
    out.update(
        {
            "features": torch.stack([row["features"] for row in batch]),
            "grid": next(iter(grids)),
            "layer": next(iter(layers)),
            "levels": next(iter(levels)),
            "pooled": (torch.stack([row["pooled"] for row in batch]) if all(pooled_rows) else None),
        }
    )
    return out


class ReplayBatchSampler:
    """Mark the first ``batches`` batches as replay, keeping the original sampler and RNG."""

    def __init__(self, batch_sampler: Any, batches: int) -> None:
        self.batch_sampler = batch_sampler
        self.remaining = batches

    def __len__(self) -> int:
        return len(self.batch_sampler)

    def __iter__(self) -> Iterator[list[int | tuple[int, bool]]]:
        for batch in self.batch_sampler:
            if self.remaining:
                self.remaining -= 1
                yield [(int(i), True) for i in batch]
            else:
                yield list(batch)


def enable_fast_replay(loader: Any, batches: int) -> bool:
    """Skip decoding of the batches replayed on resume, without changing order or workers.

    Must be called before the first iteration: the original batch sampler still yields the
    same indices and consumes the same generator in every epoch.

    :param loader: the training loader.
    :param batches: batches already consumed by the checkpointed run.
    :return: ``True`` if the loader was wrapped; ``False`` if it is not a fresh
        ``WindowDataset`` loader, which then replays by decoding.
    """
    if batches <= 0 or not isinstance(loader, DataLoader):
        return False
    if not isinstance(loader.dataset, WindowDataset):
        return False
    if loader.collate_fn is not collate_windows or loader._iterator is not None:
        return False
    # DataLoader forbids attribute assignment after __init__: only the batch sampler it
    # built is wrapped, every other field is left untouched.
    object.__setattr__(loader, "batch_sampler", ReplayBatchSampler(loader.batch_sampler, batches))
    return True


def homogeneous_batches(
    items: Sequence[WindowItem], cache: FeatureReader | None, batch_size: int
) -> Iterator[list[int]]:
    """Yield index batches that are homogeneous by source, without reordering.

    ``collate_windows`` rejects batches mixing features and pixels, so with a partial
    cache the batches are split at the boundary while keeping temporal order.

    :param items: windows in consumption order.
    :param cache: feature reader, or ``None`` for a single group.
    :param batch_size: maximum batch size.
    :return: lists of indices into ``items``.
    """
    current: list[int] = []
    provenance: bool | None = None
    for i, item in enumerate(items):
        from_cache = cache is not None and _fully_cached(item, cache)
        if current and (from_cache != provenance or len(current) >= batch_size):
            yield current
            current = []
        provenance = from_cache
        current.append(i)
    if current:
        yield current


def _weighted_sampler(
    items: Sequence[WindowItem], generator: torch.Generator | None
) -> WeightedRandomSampler:
    """Build the ``WeightedRandomSampler`` from the data weights.

    An epoch draws ``len(items)`` samples with replacement: no window is discarded,
    only its probability changes.

    :param items: windows, each with its ``weight``.
    :param generator: run generator.
    :return: the sampler.
    :raises ValueError: if a weight is not finite and positive (a zero weight would
        silently discard a window).
    """
    weights = [float(item.weight) for item in items]
    bad = [(i, w) for i, w in enumerate(weights) if not (w > 0.0) or w != w or w == float("inf")]
    if bad:
        raise ValueError(
            f"{len(bad)} windows with non-positive or non-finite weight "
            f"(e.g. index {bad[0][0]}, weight {bad[0][1]}): a zero weight silently "
            f"discards a window"
        )
    return WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True, generator=generator
    )


#: ``DataLoader`` arguments that change speed, not content (machine parameters,
#: outside the budget parity). ``pin_memory`` enables asynchronous host-to-device
#: copies; ``persistent_workers`` keeps workers alive between short passes.
def _extra_loader(num_workers: int, pin_memory: bool, persistent_workers: bool) -> dict[str, Any]:
    """:return: optional ``DataLoader`` arguments, consistent with each other
    (``persistent_workers`` requires ``num_workers > 0``).
    """
    extra: dict[str, Any] = {"pin_memory": pin_memory}
    if num_workers > 0:
        extra["persistent_workers"] = persistent_workers
    return extra


def _build_loader(
    dataset: WindowDataset,
    items: Sequence[WindowItem],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    generator: torch.Generator | None,
    weighted: bool,
) -> DataLoader:
    """Build the ``DataLoader`` shared by the frozen and the PEFT branch.

    A single builder keeps the two loaders at budget parity: an option added to one
    branch only would change steps or pixels of that branch without any error.

    :param dataset: dataset built by the caller, with its cache boundary.
    :param items: windows, for the weighted sampler.
    :param batch_size: batch size.
    :param shuffle: whether to shuffle (ignored with a sampler).
    :param num_workers: number of worker processes.
    :param pin_memory: whether to pin batch memory.
    :param persistent_workers: whether to keep workers alive.
    :param generator: run generator.
    :param weighted: sample with the ``WindowItem`` weights instead of uniformly.
    :return: the loader.
    """
    sampler = _weighted_sampler(items, generator) if weighted else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        # A sampler and `shuffle=True` together are an error in PyTorch.
        shuffle=False if sampler is not None else shuffle,
        sampler=sampler,
        collate_fn=collate_windows,
        num_workers=num_workers,
        generator=generator,
        **_extra_loader(num_workers, pin_memory, persistent_workers),
    )


def build_frozen_loader(
    items: Sequence[WindowItem],
    *,
    decoder: VideoDecoder,
    spec: PreprocessSpec,
    cache: FeatureReader | None,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    generator: torch.Generator | None = None,
    cache_in_memory: bool = False,
    cache_in_memory_mb: int = CACHE_RAM_MB,
    augment: AugmentSpec | None = None,
    run_seed: int = 0,
    weighted: bool = False,
    resample: Resampler | None = None,
) -> DataLoader:
    """Build the frozen-backbone loader, the only public entry that accepts cached features.

    ``weighted`` replaces uniform shuffling with weighted sampling. ``cache`` and
    ``augment`` exclude each other: store keys declare deterministic preprocessing.
    With shuffling, cache coverage must be total or none, otherwise the whole loader
    falls back to pixels with a warning.

    :param items: windows to serve.
    :param decoder: video decoder.
    :param spec: preprocessing spec.
    :param cache: feature reader, or ``None``.
    :param batch_size: batch size.
    :param shuffle: whether to shuffle.
    :param num_workers: number of worker processes.
    :param pin_memory: whether to pin batch memory.
    :param persistent_workers: whether to keep workers alive.
    :param generator: run generator.
    :param cache_in_memory: whether to keep decoded pixels in RAM.
    :param cache_in_memory_mb: RAM cache budget.
    :param augment: training augmentation chain, or ``None``.
    :param run_seed: seed of the run, for augmentation.
    :param weighted: whether to use weighted sampling.
    :param resample: dynamic-sampling function, or ``None``.
    :return: the loader.
    """
    if augment is not None and augment.activate:
        cache = None
    if resample is not None:
        # Dynamic sampling excludes the cache.
        cache = None
    if cache is not None and (shuffle or weighted):
        covered = sum(1 for item in items if _fully_cached(item, cache))
        if covered and covered < len(items):
            log.warning(
                "partial cache: %d of %d windows. The loader recomputes everything, since a "
                "mixed feature/pixel batch cannot be collated and the boundary moves with "
                "shuffling",
                covered,
                len(items),
            )
            cache = None
        elif not covered:
            cache = None
    dataset = WindowDataset(
        items,
        decoder,
        spec,
        cache=cache,
        cache_in_memory=cache_in_memory,
        cache_in_memory_mb=cache_in_memory_mb,
        augment=augment,
        run_seed=run_seed,
        resample=resample,
    )
    return _build_loader(
        dataset,
        items,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        generator=generator,
        weighted=weighted,
    )


def build_adapted_loader(
    items: Sequence[WindowItem],
    *,
    decoder: VideoDecoder,
    spec: PreprocessSpec,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    generator: torch.Generator | None = None,
    cache_in_memory: bool = False,
    cache_in_memory_mb: int = CACHE_RAM_MB,
    augment: AugmentSpec | None = None,
    run_seed: int = 0,
    weighted: bool = False,
    resample: Resampler | None = None,
) -> DataLoader:
    """Build the PEFT loader; no ``cache`` can cross this boundary.

    Augmentation must match the frozen runs it is compared with (budget parity).

    :param items: windows to serve.
    :param decoder: video decoder.
    :param spec: preprocessing spec.
    :param batch_size: batch size.
    :param shuffle: whether to shuffle.
    :param num_workers: number of worker processes.
    :param pin_memory: whether to pin batch memory.
    :param persistent_workers: whether to keep workers alive.
    :param generator: run generator.
    :param cache_in_memory: whether to keep decoded pixels in RAM.
    :param cache_in_memory_mb: RAM cache budget.
    :param augment: training augmentation chain, or ``None``.
    :param run_seed: seed of the run, for augmentation.
    :param weighted: whether to use weighted sampling.
    :param resample: dynamic-sampling function, or ``None``.
    :return: the loader.
    """
    dataset = WindowDataset(
        items,
        decoder,
        spec,
        cache_in_memory=cache_in_memory,
        cache_in_memory_mb=cache_in_memory_mb,
        augment=augment,
        run_seed=run_seed,
        resample=resample,
    )
    return _build_loader(
        dataset,
        items,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        generator=generator,
        weighted=weighted,
    )


def cached_output_from_batch(
    batch: dict[str, Any],
    *,
    device: str,
    dtype: torch.dtype = torch.float32,
) -> BackboneOutput:
    """Rebuild a ``BackboneOutput`` from a feature-store batch.

    Inverse of :func:`collate_windows` on the feature branch; it lives in ``data`` so
    that both ``train`` and ``eval`` can use it. Features stored in fp16 are cast once
    to the head dtype.

    :param batch: batch from :func:`collate_windows` on the feature branch.
    :param device: target device.
    :param dtype: dtype of the head parameters.
    :return: tokens with their grid and source level.
    :raises ValueError: if the batch has no features or the grid is not 3D.
    """
    if "features" not in batch:
        raise ValueError("batch without 'features'")
    tokens = batch["features"].to(device=device, dtype=dtype).detach()
    pooled = batch.get("pooled")
    if pooled is not None:
        pooled = pooled.to(device=device, dtype=dtype).detach()
    grid = tuple(int(value) for value in batch["grid"])
    if len(grid) != 3:
        raise ValueError(f"feature grid is not 3D: {grid}")
    return BackboneOutput(
        tokens=tokens,
        grid=(grid[0], grid[1], grid[2]),
        pooled=pooled,
        layer=int(batch["layer"]),
        levels=tuple(int(v) for v in batch.get("levels", ())),
    )
