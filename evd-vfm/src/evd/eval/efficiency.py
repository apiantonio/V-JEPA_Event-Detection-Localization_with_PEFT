"""Measurement primitives shared by the efficiency profilers."""

from __future__ import annotations

import shutil
import statistics
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn as nn

from evd.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Measurement:
    """Result of a measurement, with its tail and not only its mean.

    Latency distributions are skewed, so percentiles describe them better than a
    standard deviation; the percentile fields have defaults for backward compatibility.

    :param latency_ms_p50: median latency.
    :param latency_ms_p99: recurring worst case; with fewer than about 100 iterations it
        is dominated by one or two samples (see ``n_iters``).
    :param power_w_mean: mean GPU power during the measurement, ``None`` if unavailable.
    :param energy_j_per_call: energy per call, ``None`` if unavailable.
    :param power_source: how power was read, or why it was not.
    """

    latency_ms_mean: float
    latency_ms_std: float
    throughput_per_s: float
    peak_vram_bytes: int | None
    n_warmup: int
    n_iters: int
    synced: bool
    latency_ms_min: float | None = None
    latency_ms_p50: float | None = None
    latency_ms_p90: float | None = None
    latency_ms_p95: float | None = None
    latency_ms_p99: float | None = None
    latency_ms_max: float | None = None
    power_w_mean: float | None = None
    power_w_max: float | None = None
    energy_j_per_call: float | None = None
    power_source: str = "not measured"
    samples_ms: tuple[float, ...] = field(default=(), repr=False)
    """Raw samples, not serialized, so callers can recompute a percentile."""

    def as_dict(self, *, with_samples: bool = False) -> dict[str, Any]:
        """:param with_samples: whether to include ``samples_ms``.
        :return: the measurement as a dictionary.
        """
        outside = asdict(self)
        if not with_samples:
            outside.pop("samples_ms", None)
        else:
            outside["samples_ms"] = list(self.samples_ms)
        return outside


def percentili(campioni: Sequence[float]) -> dict[str, float]:
    """Return latency percentiles without depending on numpy.

    Linear interpolation between adjacent samples, as ``numpy.percentile``.

    :param campioni: latencies in milliseconds, in any order.
    :return: ``min``, ``p50``, ``p90``, ``p95``, ``p99``, ``max``.
    :raises ValueError: if the sequence is empty.
    """
    if not campioni:
        raise ValueError("no samples: percentiles are undefined")
    sorted_items = sorted(float(c) for c in campioni)
    n = len(sorted_items)

    def q(p: float) -> float:
        if n == 1:
            return sorted_items[0]
        position = p * (n - 1)
        bottom = int(position)
        top = min(bottom + 1, n - 1)
        return sorted_items[bottom] + (position - bottom) * (
            sorted_items[top] - sorted_items[bottom]
        )

    return {
        "min": sorted_items[0],
        "p50": q(0.50),
        "p90": q(0.90),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": sorted_items[-1],
    }


class PowerSampler:
    """Sample GPU power in a thread while a measurement runs.

    Sources in order: ``torch.cuda.power_draw`` (requires ``pynvml``), ``nvidia-smi``
    (one process per sample, so slower sampling), otherwise the reason is recorded and
    the fields stay ``None``. Power is per board, not per process.
    """

    def __init__(self, device: str | torch.device, *, interval_s: float = 0.1) -> None:
        self._device = torch.device(device)
        self._interval = interval_s
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._reading, self.sorgente = self._choose_source()

    def _choose_source(self) -> tuple[Callable[[], float] | None, str]:
        if self._device.type != "cuda":
            return None, "non-CUDA device"
        index = self._device.index if self._device.index is not None else 0
        try:
            torch.cuda.power_draw(index)
        except Exception as error:  # noqa: BLE001 - pynvml missing or NVML silent
            torch_reason = type(error).__name__
        else:
            return (lambda: torch.cuda.power_draw(index) / 1000.0), "torch.cuda.power_draw"

        smi = shutil.which("nvidia-smi")
        if smi is None:
            return None, f"pynvml missing ({torch_reason}) and nvidia-smi not on PATH"

        def read() -> float:
            output = subprocess.run(  # noqa: S603 - path resolved by shutil.which
                [
                    smi,
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits",
                    f"--id={index}",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            return float(output.stdout.strip().splitlines()[0])

        try:
            read()
        except Exception as error:  # noqa: BLE001
            return None, f"nvidia-smi not usable: {type(error).__name__}"
        # One process per sample: sample slower instead of stealing CPU from the timed forward.
        self._interval = max(self._interval, 0.2)
        return read, "nvidia-smi"

    @property
    def disponibile(self) -> bool:
        return self._reading is not None

    def _cycle(self) -> None:
        assert self._reading is not None
        while not self._stop.is_set():
            try:
                self._samples.append(self._reading())
            except Exception:  # noqa: BLE001 - a lost reading does not stop the measurement
                pass
            self._stop.wait(self._interval)

    def __enter__(self) -> PowerSampler:
        self._samples.clear()
        if self._reading is not None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._cycle, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    @property
    def mean_w(self) -> float | None:
        return statistics.fmean(self._samples) if self._samples else None

    @property
    def max_w(self) -> float | None:
        return max(self._samples) if self._samples else None

    @property
    def n_samples(self) -> int:
        return len(self._samples)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(
    fn: Callable[[], Any],
    *,
    warmup: int,
    iters: int,
    device: str,
    energy: bool = False,
    intervallo_energia_s: float = 0.1,
) -> Measurement:
    """Time a call; on CUDA synchronize before and after each iteration.

    :param fn: call to time.
    :param warmup: discarded iterations (kernel selection and clock ramp-up).
    :param iters: timed iterations; a readable ``p99`` needs about a hundred.
    :param device: ``cpu`` or ``cuda``.
    :param energy: whether to sample power during the measurement (in a separate thread,
        outside the timed region).
    :param intervallo_energia_s: power sampling period.
    :return: the measurement with percentiles and, when available, power and energy.
    :raises ValueError: on invalid counts or an unsupported device.
    :raises RuntimeError: if CUDA is requested but unavailable.
    """
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, not {warmup}")
    if iters <= 0:
        raise ValueError(f"iters must be > 0, not {iters}")
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported device: {device!r} (allowed cpu|cuda)")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA measurement requested but torch.cuda.is_available() is False")

    for _ in range(warmup):
        fn()
    _synchronize(target)
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)

    sampler = PowerSampler(target, interval_s=intervallo_energia_s) if energy else None
    latencies_ms: list[float] = []
    context = sampler if sampler is not None else _nothing()
    with context:
        for _ in range(iters):
            _synchronize(target)
            start = time.perf_counter_ns()
            fn()
            _synchronize(target)
            latencies_ms.append((time.perf_counter_ns() - start) / 1_000_000)

    mean = statistics.fmean(latencies_ms)
    tail = percentili(latencies_ms)
    mean_w = sampler.mean_w if sampler is not None else None
    return Measurement(
        latency_ms_mean=mean,
        latency_ms_std=statistics.pstdev(latencies_ms),
        throughput_per_s=1000.0 / mean if mean > 0 else float("inf"),
        peak_vram_bytes=(
            int(torch.cuda.max_memory_allocated(target)) if target.type == "cuda" else None
        ),
        n_warmup=warmup,
        n_iters=iters,
        synced=target.type == "cuda",
        latency_ms_min=tail["min"],
        latency_ms_p50=tail["p50"],
        latency_ms_p90=tail["p90"],
        latency_ms_p95=tail["p95"],
        latency_ms_p99=tail["p99"],
        latency_ms_max=tail["max"],
        power_w_mean=mean_w,
        power_w_max=sampler.max_w if sampler is not None else None,
        # Energy per call = mean power x mean duration; an approximation valid in aggregate.
        energy_j_per_call=(mean_w * mean / 1000.0) if mean_w is not None else None,
        power_source=sampler.sorgente if sampler is not None else "not requested",
        samples_ms=tuple(latencies_ms),
    )


class _nothing:
    """No-op context manager, to avoid duplicating the measurement loop."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: object) -> None:
        return None


def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    """Count scalars, optionally only trainable parameters.

    :param module: module to inspect.
    :param trainable_only: whether to count only parameters requiring gradients.
    :return: number of scalars.
    """
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


#: V-JEPA 2.1 tubelet geometry: 2 frames x 16 x 16 pixels.
PATCH_T, PATCH_HW = 2, 16
PATCH_VOLUME = PATCH_T * PATCH_HW * PATCH_HW


def tokens_per_window(*, num_frames: int, size: int) -> int:
    """Return the number of tokens of a window before any adaptation.

    :param num_frames: temporal samples of the window (32 in ``edp_v2``).
    :param size: frame side in pixels (384 in ``edp_v2``).
    :return: number of tokens (9216 in ``edp_v2``).
    :raises ValueError: if the geometry is not divisible by the tubelet.
    """
    if num_frames <= 0 or size <= 0:
        raise ValueError(f"non-positive geometry: num_frames={num_frames}, size={size}")
    if num_frames % PATCH_T or size % PATCH_HW:
        raise ValueError(
            f"geometry not divisible by the tubelet {PATCH_T}x{PATCH_HW}x{PATCH_HW}: "
            f"num_frames={num_frames}, size={size}"
        )
    return (num_frames // PATCH_T) * (size // PATCH_HW) ** 2


def transformer_forward_flops(
    *,
    batch_size: int,
    tokens: int,
    dim: int,
    depth: int,
    patch_volume: int,
    in_channels: int = 3,
) -> int:
    """Estimate ViT forward FLOPs (matmul/conv, multiply-add = 2 FLOPs).

    Each block counts QKV, output projection, 4x MLP and the two attention matmuls;
    norms, activations and RoPE are excluded, so the value is a consistent theoretical
    throughput, not a hardware count.

    :param batch_size: batch size.
    :param tokens: tokens per sample.
    :param dim: embedding dimension.
    :param depth: number of blocks.
    :param patch_volume: pixels per tubelet.
    :param in_channels: input channels.
    :return: FLOPs per forward.
    :raises ValueError: on non-positive sizes.
    """
    values = (batch_size, tokens, dim, depth, patch_volume, in_channels)
    if any(value <= 0 for value in values):
        raise ValueError(f"non-positive FLOP dimensions: {values}")
    patch_embed = 2 * batch_size * tokens * dim * in_channels * patch_volume
    per_block = batch_size * (24 * tokens * dim * dim + 4 * tokens * tokens * dim)
    return patch_embed + depth * per_block
