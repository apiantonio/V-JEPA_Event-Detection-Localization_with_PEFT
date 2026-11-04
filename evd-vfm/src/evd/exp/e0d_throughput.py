"""Throughput leg: how fast a batch arrives, and with how many CPUs.

Training needs 1.6-2.8 windows/s; the hard requirement is
evaluation, where the frozen forward runs at 9.33 windows/s on A100 and 16.55 on H100.

The real path (``WindowDataset`` with PyAV plus ``build_adapted_loader``) is measured
while varying ``num_workers``, ``pin_memory`` and ``persistent_workers`` on real windows
of the sealed split. Start-up (worker fork, file opening) and steady state are timed
separately. ``pin_memory`` cannot be measured without a GPU (PyTorch ignores it).

The ``cache_mb`` axis measures the RAM cost of the ``WindowDataset`` cache by sampling
the actual RSS (main process plus workers, via ``/proc``, Linux only).
``cache_rss_delta_mb`` (peak minus base) also includes the worker fork when
``workers > 0``: the cost of the cache alone is the difference between two rows at the
same ``workers`` (cleanest at ``workers=0``).
"""

from __future__ import annotations

import os
import random
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.utils.data import WeightedRandomSampler

from evd.data.dataset import WindowDataset, WindowItem, build_adapted_loader
from evd.types import AugmentSpec, PreprocessSpec
from evd.utils.logging import get_logger

log = get_logger(__name__)

PROFILE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ThroughputConfig:
    """Measurement axes; no field has a protocol default."""

    n_windows: int
    """Windows traversing the loader per cell; large enough to cover start-up (with 8
    workers and batch 2, fewer than about 100 windows would mostly measure the fork)."""
    batch_size: int
    workers: tuple[int, ...]
    pin_memory: tuple[bool, ...]
    persistent: tuple[bool, ...]
    device: str
    seed: int
    run_id: str
    augments: tuple[str, ...] = ("none",)
    """Augmentation chains to measure, by id. Without this axis the first measurement
    described a loader no run uses (``edp_v1`` runs use ``official_nocrop``). The
    default ``("none",)`` keeps the meaning of the registered measurement (job 539567)."""
    cache_mb: tuple[int, ...] = (0,)
    """``WindowDataset`` budgets (``cache_in_memory_mb``) to measure, per worker. ``0``
    disables the cache, the behaviour of earlier cells (jobs 539567/540229)."""

    def __post_init__(self) -> None:
        if self.n_windows <= 0 or self.batch_size <= 0:
            raise ValueError(
                f"n_windows and batch_size must be > 0, not {self.n_windows}/{self.batch_size}"
            )
        if not self.workers or any(w < 0 for w in self.workers):
            raise ValueError(f"invalid workers: {self.workers}")
        if tuple(sorted(set(self.workers))) != self.workers:
            raise ValueError(f"workers must be unique and increasing: {self.workers}")
        if not self.pin_memory or not self.persistent:
            raise ValueError("pin_memory and persistent must have at least one value")
        if not self.augments:
            raise ValueError("augments must have at least one value (`none` for no chain)")
        if not self.cache_mb or any(c < 0 for c in self.cache_mb):
            raise ValueError(f"invalid cache_mb: {self.cache_mb} (0 = cache disabled)")


def celle(config: ThroughputConfig) -> list[dict[str, Any]]:
    """Return the Cartesian product of the axes, in stable order.

    ``persistent_workers`` does not exist with zero workers (PyTorch raises), so such cells
    are skipped.

    :param config: the configuration.
    :return: one row per cell, with its identifying fields only.
    """
    return [
        {"workers": w, "pin_memory": p, "persistent": s, "augment": a, "cache_mb": c}
        for a in config.augments
        for w in config.workers
        for p in config.pin_memory
        for s in (config.persistent if w > 0 else (False,))
        for c in config.cache_mb
    ]


def cell_id(cella: dict[str, Any]) -> str:
    """:return: the identity of a cell, also its file name."""
    base = (
        f"aug{cella.get('augment', 'none')}-w{cella['workers']}"
        f"-pin{int(cella['pin_memory'])}-per{int(cella['persistent'])}"
    )
    cache = int(cella.get("cache_mb", 0))
    # `cache0` is not appended, so the cell_id of earlier cells does not change.
    return base if cache == 0 else f"{base}-cache{cache}"


def _rss_mb(pids: Sequence[int]) -> float | None:
    """Return the total RSS (MB) of the given processes, summing ``VmRSS`` from ``/proc``.

    Linux only; a pid that already exited is skipped.

    :param pids: process identifiers to sum.
    :return: resident RSS in MB, or ``None`` if not measurable on this platform or no pid is
        readable.
    """
    if not sys.platform.startswith("linux"):
        return None
    total_kb = 0
    read_items = 0
    for pid in pids:
        try:
            with open(f"/proc/{pid}/status", encoding="utf-8") as fh:
                for row in fh:
                    if row.startswith("VmRSS:"):
                        total_kb += int(row.split()[1])
                        read_items += 1
                        break
        except (OSError, ValueError):
            continue
    return total_kb / 1024 if read_items > 0 else None


def _measure_cell(
    items: Sequence[WindowItem],
    *,
    decoder: Any,
    spec: PreprocessSpec,
    config: ThroughputConfig,
    cella: dict[str, Any],
    augment: AugmentSpec | None = None,
) -> dict[str, Any]:
    """Time one pass over the loader, separating start-up and steady state.

    The first batch includes the worker fork and the first file opening, reported
    separately. RSS is also sampled at every batch when ``cella["cache_mb"]`` is positive.

    :return: measurement fields of the row (without identifying fields); memory keys
        (``rss_base_mb``, ``peak_rss_mb``, ``cache_rss_delta_mb``) are ``None`` outside
        Linux.
    :raises RuntimeError: if the loader yields no batch.
    """
    cache_mb = int(cella.get("cache_mb", 0))
    loader = build_adapted_loader(
        list(items),
        decoder=decoder,
        spec=spec,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=int(cella["workers"]),
        pin_memory=bool(cella["pin_memory"]),
        persistent_workers=bool(cella["persistent"]),
        # The augmentation is passed exactly as the runner passes it to `build_adapted_loader`.
        augment=augment,
        run_seed=config.seed,
        cache_in_memory=cache_mb > 0,
        cache_in_memory_mb=max(1, cache_mb),
    )
    device = torch.device(config.device)
    transfer = device.type == "cuda"

    # Base RSS, before any worker exists; the delta below also includes the fork when
    # workers > 0.
    rss_base_mb = _rss_mb([os.getpid()])

    start = time.perf_counter_ns()
    first: int | None = None
    views = 0
    transfer_ns = 0
    peak_rss_mb: float | None = None

    iterator = iter(loader)
    # Worker processes are forked inside `iter()`; `_workers` is a private torch attribute, so
    # `getattr(..., [])` degrades to "main process only" instead of raising.
    pids_worker = [w.pid for w in getattr(iterator, "_workers", []) if getattr(w, "pid", None)]
    try:
        while views < config.n_windows:
            try:
                batch = next(iterator)
            except StopIteration:
                break
            if first is None:
                first = time.perf_counter_ns() - start
            sample = _rss_mb([os.getpid(), *pids_worker])
            if sample is not None:
                peak_rss_mb = sample if peak_rss_mb is None else max(peak_rss_mb, sample)
            clip = batch["clip"]
            if transfer:
                torch.cuda.synchronize()
                t0 = time.perf_counter_ns()
                clip.to(device, non_blocking=bool(cella["pin_memory"]))
                torch.cuda.synchronize()
                transfer_ns += time.perf_counter_ns() - t0
            views += int(clip.shape[0])
    finally:
        del iterator  # close the workers before the next cell opens new ones
    total_ns = time.perf_counter_ns() - start
    if first is None or views == 0:
        raise RuntimeError("the loader produced no batch")

    regime_ns = max(total_ns - first, 1)
    regime_views = max(views - config.batch_size, 1)
    return {
        "status": "ok",
        "n_windows": views,
        "avvio_ms": first / 1e6,
        "totale_s": total_ns / 1e9,
        "throughput_windows_s": views / (total_ns / 1e9),
        "throughput_regime_windows_s": regime_views / (regime_ns / 1e9),
        "ms_per_window": (total_ns / 1e6) / views,
        # GPU only: without one, `pin_memory` is a no-op.
        "h2d_ms_per_window": (transfer_ns / 1e6) / views if transfer else None,
        "device": config.device,
        # Linux only (`_rss_mb`): `None` elsewhere, never a misleading zero.
        "rss_base_mb": rss_base_mb,
        "peak_rss_mb": peak_rss_mb,
        "cache_rss_delta_mb": (
            peak_rss_mb - rss_base_mb
            if peak_rss_mb is not None and rss_base_mb is not None
            else None
        ),
        "n_worker_pids_misurati": len(pids_worker),
    }


def profile_throughput(
    items: Sequence[WindowItem],
    *,
    decoder: Any,
    spec: PreprocessSpec,
    config: ThroughputConfig,
    augments: Mapping[str, AugmentSpec] | None = None,
) -> dict[str, Any]:
    """Run the grid and return the payload, without writing it.

    The CLI writes it with the tracker, so this function can be tested on a synthetic world.

    :param items: windows to traverse, already sampled.
    :param decoder: the run video decoder.
    :param spec: protocol preprocessing geometry.
    :param config: measurement axes.
    :param augments: chain for each id of ``config.augments``; ``"none"`` needs no entry.
    :return: ``{"run_id", "rows", ...}``.
    :raises RuntimeError: if a cell yields no batch.
    :raises ValueError: if an id other than ``"none"`` has no chain.
    """
    if len(items) < config.batch_size:
        raise ValueError(
            f"{len(items)} windows for a batch of {config.batch_size}: "
            f"the measurement would have no steady state to time"
        )
    if any(p for p in config.pin_memory) and not torch.cuda.is_available():
        # Not an error (a CPU-only grid is useful), but the pin_memory column would be identical.
        log.warning(
            "pin_memory=True without CUDA: PyTorch ignores it, and the two columns will be "
            "identical. The relevant cell must run on a GPU node"
        )

    chains = dict(augments or {})
    missing = [a for a in config.augments if a != "none" and a not in chains]
    if missing:
        raise ValueError(
            f"augmentations declared without a chain: {missing}. Measuring them as "
            f'"no chain" would give identical rows under different names, i.e. '
            f"misleading metadata"
        )

    rows: list[dict[str, Any]] = []
    for cell in celle(config):
        measure = _measure_cell(
            items,
            decoder=decoder,
            spec=spec,
            config=config,
            cella=cell,
            augment=chains.get(str(cell.get("augment", "none"))),
        )
        rows.append({**cell, "cell_id": cell_id(cell), **measure})
        log.info(
            "augment=%s workers=%d pin=%s persistent=%s -> %.2f fin/s (steady %.2f, start-up %.0f ms)",
            cell.get("augment", "none"),
            cell["workers"],
            cell["pin_memory"],
            cell["persistent"],
            measure["throughput_windows_s"],
            measure["throughput_regime_windows_s"],
            measure["avvio_ms"],
        )
    return {
        "run_id": config.run_id,
        "profile_schema": PROFILE_SCHEMA_VERSION,
        "config": asdict(config),
        "rows": rows,
    }


@dataclass(frozen=True)
class Knee:
    """Outcome of the criterion, with its honesty next to the number.

    :param workers: smallest number of workers reaching the threshold.
    :param throughput: windows/s it produces.
    :param satura: false if the grid saw no plateau (throughput still growing at the last
        point), in which case ``workers`` is only the largest measured value.
    """

    workers: int
    throughput: float
    satura: bool

    def __str__(self) -> str:
        if self.satura:
            return f"{self.workers} workers ({self.throughput:.2f} windows/s, saturated)"
        return (
            f"{self.workers} workers ({self.throughput:.2f} windows/s) — "
            f"NOT saturated: the grid stops before the plateau"
        )


def find_knee(righe: Sequence[dict[str, Any]], *, tolerance: float = 0.05) -> Knee:
    """Return the number of workers beyond which adding more does not help, if it exists.

    :param righe: measurement rows, with ``workers`` and ``throughput_regime_windows_s``.
    :param tolerance: throughput fraction one accepts to give up (``0.05`` = smallest number
        of workers reaching 95% of the observed maximum).
    :return: the number, its throughput and whether the grid actually saturated.
    :raises ValueError: if there is no valid row.

    On a linearly scaling grid the 95% criterion always returns the last point, so
    saturation is verified instead of assumed.
    """
    valid = [r for r in righe if r.get("status") == "ok"]
    if not valid:
        raise ValueError("no valid cell: there is no knee to find")
    per_worker = {}
    for r in valid:
        w = int(r["workers"])
        per_worker[w] = max(per_worker.get(w, 0.0), float(r["throughput_regime_windows_s"]))

    maximum = max(per_worker.values())
    threshold = (1.0 - tolerance) * maximum
    chosen = min(w for w, v in per_worker.items() if v >= threshold)

    # Saturated if the chosen point is not the last of the grid, or it is but the last jump
    # gained less than the tolerance.
    sorted_items = sorted(per_worker)
    if chosen < sorted_items[-1]:
        saturated = True
    elif len(sorted_items) < 2:
        saturated = False
    else:
        second_to_last, last = sorted_items[-2], sorted_items[-1]
        gain = (per_worker[last] - per_worker[second_to_last]) / max(
            per_worker[second_to_last], 1e-9
        )
        saturated = gain <= tolerance
    return Knee(workers=chosen, throughput=per_worker[chosen], satura=saturated)


def workers_per(righe: Sequence[dict[str, Any]], request: float) -> int | None:
    """Return how many workers keep up with a consumer of ``richiesta`` windows/s.

    The relevant question: exceed what the GPU consumes, not reach the maximum throughput.

    :param righe: measurement rows.
    :param richiesta: windows/s absorbed by the consumer (from E0b.3: 5.01 for training on
        H100, 16.55 for an evaluation pass).
    :return: the smallest number of workers reaching it, or ``None`` if no measured point
        suffices.
    :raises ValueError: if the request is not positive.
    """
    if request <= 0:
        raise ValueError(f"the request must be > 0 windows/s, not {request}")
    sufficient = [
        int(r["workers"])
        for r in righe
        if r.get("status") == "ok" and float(r["throughput_regime_windows_s"]) >= request
    ]
    return min(sufficient) if sufficient else None


# --------------------------------------------------------------------------- #
# The REAL benefit of the cache under true sampling
# --------------------------------------------------------------------------- #
#
# The `cache_mb` axis above measures the cost (RSS), not the benefit: it traverses a fixed
# list once, so every index is a miss by construction. Training uses
# `WeightedRandomSampler(replacement=True)` (one epoch draws as many samples as chunks,
# 88,371 on edp_v2), so high-weight indices can repeat within a worker share; only there
# does the RAM cache have an effect.


def _draw_indices(pesi: Sequence[float], *, n_draws: int, seed: int) -> list[int]:
    """Return the sequence of indices drawn with replacement from the real weights.

    Uses ``torch.utils.data.WeightedRandomSampler`` directly, the same primitive as a real
    run.

    :param pesi: one unnormalized weight per index (``0..len(pesi)-1``).
    :param n_draws: number of draws with replacement.
    :param seed: generator seed, shared by all compared cache budgets.
    :return: the drawn indices, in drawing order.
    """
    sampler = WeightedRandomSampler(
        list(pesi),
        num_samples=n_draws,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return [int(i) for i in sampler]


@dataclass(frozen=True)
class CacheBenefitConfig:
    """Axes of the real-benefit measurement (not its RAM cost)."""

    n_draws: int
    """Draws with replacement to simulate; one worker share of a real epoch is
    ``max_steps · batch_size · grad_accum / num_workers`` (about 7365 on edp_v2). The
    reported repetition rate says whether a smaller value saw enough repetitions."""
    cache_mb: tuple[int, ...]
    """``WindowDataset.cache_in_memory_mb`` budgets to compare, per worker (include 0 as
    the baseline row). Multiply by the real ``num_workers`` to obtain ``CACHE_RAM_MB``."""
    seed: int
    run_id: str

    def __post_init__(self) -> None:
        if self.n_draws <= 0:
            raise ValueError(f"n_draws must be > 0, not {self.n_draws}")
        if not self.cache_mb or any(c < 0 for c in self.cache_mb):
            raise ValueError(f"invalid cache_mb: {self.cache_mb} (0 = cache disabled)")
        if len(set(self.cache_mb)) != len(self.cache_mb):
            raise ValueError(f"duplicated cache_mb: {self.cache_mb}")


def profile_cache_benefit(
    items: Sequence[WindowItem],
    weights: Sequence[float],
    *,
    decoder: Any,
    spec: PreprocessSpec,
    config: CacheBenefitConfig,
    augment: AugmentSpec | None = None,
) -> dict[str, Any]:
    """Return the real benefit of each candidate budget, on the SAME draw sequence.

    Each ``config.cache_mb`` builds a new ``WindowDataset`` and traverses the same index
    sequence from :func:`_draw_indices`, indexing directly (``dataset[i]``) as a
    ``DataLoader`` worker does. RSS is sampled too (Linux only). Candidates are measured in
    a seeded shuffled order so OS/NFS file caching does not systematically favour one; rows
    are returned in the declared order.

    :param items: the full partition population (draws come from here).
    :param weights: one weight per element, in the order of ``items``.
    :param decoder: the run video decoder.
    :param spec: backbone preprocessing.
    :param config: measurement axes.
    :param augment: training chain to reproduce (``edp_v2``: `photometric`).
    :return: ``{"n_population", "n_draws", "n_indici_unici", "tasso_ripetizione", "rows"}``;
        each row also has ``speedup_vs_baseline`` (``None`` if ``0`` is not a candidate).
    :raises ValueError: if ``items`` and ``weights`` have different lengths.
    """
    if len(items) != len(weights):
        raise ValueError(
            f"items and weights must have the same length: {len(items)} vs {len(weights)}"
        )
    indices = _draw_indices(weights, n_draws=config.n_draws, seed=config.seed)
    unique_items = len(set(indices))

    measure_order = list(config.cache_mb)
    random.Random(config.seed).shuffle(measure_order)

    rows: dict[int, dict[str, Any]] = {}
    for cache_mb in measure_order:
        dataset = WindowDataset(
            list(items),
            decoder,
            spec,
            cache=None,
            cache_in_memory=cache_mb > 0,
            cache_in_memory_mb=max(1, cache_mb),
            augment=augment,
            run_seed=config.seed,
        )
        rss_base = _rss_mb([os.getpid()])
        peak_rss: float | None = None
        start = time.perf_counter_ns()
        for i in indices:
            _ = dataset[i]
            sample = _rss_mb([os.getpid()])
            if sample is not None:
                peak_rss = sample if peak_rss is None else max(peak_rss, sample)
        total_ns = time.perf_counter_ns() - start
        rows[cache_mb] = {
            "cache_mb": cache_mb,
            "n_draws": len(indices),
            "totale_s": total_ns / 1e9,
            "throughput_windows_s": len(indices) / (total_ns / 1e9),
            "rss_base_mb": rss_base,
            "peak_rss_mb": peak_rss,
            "cache_rss_delta_mb": (
                peak_rss - rss_base if peak_rss is not None and rss_base is not None else None
            ),
        }
        log.info(
            "cache_mb=%d -> %.2f windows/s over %d draws (RSS delta %s)",
            cache_mb,
            rows[cache_mb]["throughput_windows_s"],
            len(indices),
            "n/d"
            if rows[cache_mb]["cache_rss_delta_mb"] is None
            else f"{rows[cache_mb]['cache_rss_delta_mb']:.0f} MB",
        )

    base = rows.get(0)
    sorted_items = [rows[c] for c in config.cache_mb]
    for r in sorted_items:
        r["speedup_vs_baseline"] = (base["totale_s"] / r["totale_s"]) if base else None

    return {
        "n_population": len(items),
        "n_draws": len(indices),
        "n_indici_unici": unique_items,
        "tasso_ripetizione": 1.0 - unique_items / len(indices),
        "rows": sorted_items,
    }
