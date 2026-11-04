"""Reusable feature-extraction orchestration (the CLI stays thin)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import torch

from evd.data.decoder import VideoDecoder
from evd.data.feature_store import SafetensorsFeatureStore, sizeof_estimate
from evd.data.inventory import InventoryRecord, index_by_video_id, read_records
from evd.data.preprocess import preprocess_clip
from evd.data.windows import CausalWindowSampler, WindowGrid
from evd.types import BackboneOutput, PreprocessSpec, WindowSpec

if TYPE_CHECKING:
    # Annotation-only import: ``data`` does not import ``models`` at runtime.
    from evd.models.backbones.base import VideoBackbone


@dataclass(frozen=True)
class ExtractionEstimate:
    n_videos: int
    n_windows: int
    n_layers: int
    tokens_per_window: int
    embed_dim: int
    bytes_per_value: int
    estimated_bytes: int

    @property
    def estimated_gb(self) -> float:
        return self.estimated_bytes / 1_000_000_000

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "estimated_gb": self.estimated_gb}


@dataclass(frozen=True)
class ExtractionResult:
    n_videos: int
    n_windows: int
    n_written: int
    n_hits: int
    layers: tuple[int, ...]

    @property
    def hit_rate_before_write(self) -> float:
        total = self.n_written + self.n_hits
        return self.n_hits / total if total else 1.0

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "hit_rate_before_write": self.hit_rate_before_write}


def read_video_ids(path: str | Path) -> list[str]:
    """Read an explicit video selection, accepting video ids or split paths.

    :param path: text file, one entry per line.
    :return: the selected video ids.
    :raises FileNotFoundError: if the file does not exist.
    :raises ValueError: on absolute entries or an empty list.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"missing video list: {source}")
    ids: list[str] = []
    for line_number, raw in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), start=1):
        value = raw.strip().replace("\\", "/")
        if not value or value.startswith("#"):
            continue
        candidate = PurePosixPath(value)
        if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
            raise ValueError(f"non-relative selection in {source}:{line_number}: {raw!r}")
        ids.append(candidate.stem)
    duplicates = sorted({video_id for video_id in ids if ids.count(video_id) > 1})
    if duplicates:
        raise ValueError(f"duplicated video_id in {source}: {duplicates[:5]}")
    if not ids:
        raise ValueError(f"empty video list: {source}")
    return ids


def select_inventory_records(
    inventory_csv: str | Path,
    videos_from: str | Path,
    *,
    shard_index: int,
    num_shards: int,
) -> list[InventoryRecord]:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be > 0, not {num_shards}")
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index {shard_index} outside [0, {num_shards})")
    inventory = index_by_video_id(read_records(inventory_csv))
    selected_ids = read_video_ids(videos_from)
    missing = [video_id for video_id in selected_ids if video_id not in inventory]
    if missing:
        raise KeyError(f"selected videos missing from the inventory: {missing[:5]}")
    selected = [inventory[video_id] for video_id in selected_ids]
    invalid = [record.video_id for record in selected if record.fps <= 0 or record.num_frames <= 0]
    if invalid:
        raise ValueError(f"videos without valid fps/frames: {invalid[:5]}")
    return selected[shard_index::num_shards]


def windows_for_records(
    records: Sequence[InventoryRecord],
    *,
    preprocess: PreprocessSpec,
    hop_s: float,
    grid: WindowGrid = "endpoint",
    window_s: float | None = None,
) -> dict[str, list[WindowSpec]]:
    """Return the windows to extract, per video.

    ``grid`` and ``window_s`` must match the run sampler: store keys are windows, and
    extracting on one grid while reading on another silently misses every window.
    ``endpoint`` is the default because existing ``edp_v1`` stores use it; ``edp_v2``
    needs ``grid="fixed", window_s=8.0``.

    :param records: videos to cover.
    :param preprocess: backbone geometry (frame count and target fps).
    :param hop_s: distance between consecutive windows (validation and test differ in
        ``edp_v2``: 8 s and 4 s).
    :param grid: ``endpoint`` or ``fixed``.
    :param window_s: nominal window duration, required by ``fixed``.
    :return: ``video_id`` to its windows.
    """
    sampler = CausalWindowSampler.from_spec(preprocess, hop_s=hop_s, grid=grid, window_s=window_s)
    return {record.video_id: sampler.windows_for_video(record.to_meta()) for record in records}


def tokens_per_window(preprocess: PreprocessSpec) -> int:
    tokenizers = {
        "tubelet_2x16x16": (2, 16, 16),
        "tubelet_2x8x8": (2, 8, 8),
        "patch_1x14x14_f16": (1, 14, 14),
        "frame_patch_16": (1, 16, 16),
    }
    try:
        temporal, height, width = tokenizers[preprocess.tokenizer]
    except KeyError:
        raise ValueError(
            f"tokenizer {preprocess.tokenizer!r} unknown to the estimate; declare its grid"
        ) from None
    if preprocess.num_frames % temporal or preprocess.size % height or preprocess.size % width:
        raise ValueError(
            f"geometry {preprocess.num_frames}x{preprocess.size} not divisible by "
            f"tokenizer {temporal}x{height}x{width}"
        )
    return (
        preprocess.num_frames // temporal * (preprocess.size // height) * (preprocess.size // width)
    )


def estimate_extraction(
    records: Sequence[InventoryRecord],
    *,
    preprocess: PreprocessSpec,
    hop_s: float,
    layers: Sequence[int],
    embed_dim: int,
    storage_dtype: str,
    grid: WindowGrid = "endpoint",
    window_s: float | None = None,
) -> ExtractionEstimate:
    if not layers:
        raise ValueError("at least one layer is required")
    if len(set(layers)) != len(layers):
        raise ValueError(f"duplicate layers: {list(layers)}")
    if storage_dtype not in {"fp16", "fp32"}:
        raise ValueError(f"invalid storage_dtype: {storage_dtype!r}")
    windows = windows_for_records(
        records, preprocess=preprocess, hop_s=hop_s, grid=grid, window_s=window_s
    )
    n_windows = sum(map(len, windows.values()))
    tokens = tokens_per_window(preprocess)
    bytes_per_value = 2 if storage_dtype == "fp16" else 4
    return ExtractionEstimate(
        n_videos=len(records),
        n_windows=n_windows,
        n_layers=len(layers),
        tokens_per_window=tokens,
        embed_dim=embed_dim,
        bytes_per_value=bytes_per_value,
        estimated_bytes=sizeof_estimate(n_windows, len(layers), tokens, embed_dim, bytes_per_value),
    )


def enforce_size_limit(estimate: ExtractionEstimate, *, max_gb: float) -> None:
    if max_gb <= 0:
        raise ValueError(f"max_gb must be > 0, not {max_gb}")
    if estimate.estimated_gb > max_gb:
        raise RuntimeError(
            f"estimated extraction of {estimate.estimated_gb:.2f} GB exceeds the "
            f"explicit cap of {max_gb:.2f} GB"
        )


def extract_records(
    records: Sequence[InventoryRecord],
    *,
    dataset_root: str | Path,
    preprocess: PreprocessSpec,
    hop_s: float,
    layers: Sequence[int],
    backbone: VideoBackbone,
    decoder: VideoDecoder,
    store: SafetensorsFeatureStore,
    batch_size: int,
    device: str,
    precision: str,
    grid: WindowGrid = "endpoint",
    window_s: float | None = None,
) -> ExtractionResult:
    """Extract features video by video; an interruption redoes at most one video.

    ``precision`` has no default; ``bf16`` means autocast with FP32 weights and inputs,
    as in the Meta path (casting the encoder to bf16 breaks the interpolated 3D RoPE).

    :param records: videos to extract.
    :param dataset_root: dataset root folder.
    :param preprocess: preprocessing spec.
    :param hop_s: distance between windows.
    :param layers: layers to store.
    :param backbone: frozen backbone.
    :param decoder: video decoder.
    :param store: destination feature store.
    :param batch_size: forward batch size.
    :param device: compute device.
    :param precision: compute regime (``fp32`` or ``bf16``).
    :param grid: window grid.
    :param window_s: nominal window duration.
    :return: extraction summary.
    :raises FileNotFoundError: if a selected video is missing.
    :raises ValueError: on invalid arguments or a precision different from the store.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, not {batch_size}")
    if precision not in {"fp32", "bf16"}:
        raise ValueError(f"precision={precision!r}: allowed fp32|bf16")
    if precision != store.precision:
        raise ValueError(
            f"precision={precision!r} but the store was opened with "
            f"{store.precision!r}: two regimes in one store are not comparable"
        )
    if preprocess.geometry == "three_views":
        raise ValueError(
            "the per-window feature store cannot hide three views in a batch; "
            "choose a single-view geometry in the protocol"
        )
    layer_tuple = tuple(int(layer) for layer in layers)
    # Read the index once (one SQLite round-trip per key does not finish on a parallel
    # filesystem); ``known_keys`` covers every shard index, so appends skip existing keys.
    already_present = store.known_keys()
    windows_by_video = windows_for_records(
        records, preprocess=preprocess, hop_s=hop_s, grid=grid, window_s=window_s
    )
    root = Path(dataset_root)
    backbone.freeze()
    backbone.eval()
    backbone.to(device)  # type: ignore[attr-defined]
    hits = written = total_windows = 0

    with torch.inference_mode():
        for record in records:
            windows = windows_by_video[record.video_id]
            total_windows += len(windows)
            keys = {
                layer: {
                    window.window_id: store.key(
                        backbone.backbone_id,
                        store.ckpt_sha,
                        preprocess,
                        layer,
                        window,
                    )
                    for window in windows
                }
                for layer in layer_tuple
            }
            pending = {
                layer: {
                    window.window_id
                    for window in windows
                    if keys[layer][window.window_id] not in already_present
                }
                for layer in layer_tuple
            }
            hits += sum(len(windows) - len(pending[layer]) for layer in layer_tuple)
            collected: dict[int, dict[str, BackboneOutput]] = {layer: {} for layer in layer_tuple}
            needed = [
                window
                for window in windows
                if any(window.window_id in pending[layer] for layer in layer_tuple)
            ]
            video_path = root / record.relative_path
            if needed and not video_path.is_file():
                raise FileNotFoundError(f"selected video missing: {video_path}")
            for start in range(0, len(needed), batch_size):
                batch_windows = needed[start : start + batch_size]
                clips = []
                for window in batch_windows:
                    frames = decoder.read_frames(video_path, window.frame_indices)
                    clip = preprocess_clip(frames, preprocess)
                    if clip.ndim != 4:
                        raise ValueError(
                            f"{window.window_id}: preprocessing produced {clip.ndim}D; "
                            "a single view is required"
                        )
                    clips.append(clip)
                pixel_batch = torch.stack(clips).to(device)
                needed_layers = tuple(
                    layer
                    for layer in layer_tuple
                    if any(window.window_id in pending[layer] for window in batch_windows)
                )
                with torch.autocast(
                    device_type=pixel_batch.device.type,
                    dtype=torch.bfloat16,
                    enabled=precision == "bf16",
                ):
                    encoded_layers = backbone.encode_layers(pixel_batch, needed_layers)
                for layer, encoded in encoded_layers.items():
                    for output_index, window in enumerate(batch_windows):
                        if window.window_id not in pending[layer]:
                            continue
                        # Move outputs to CPU right away: a whole video is collected before writing, and an
                        # hour-long video would fill the GPU. ``to("cpu", copy=True)`` (not ``.cpu()``) so
                        # that slices do not keep the batch alive.
                        pooled = (
                            None
                            if encoded.pooled is None
                            else encoded.pooled[output_index : output_index + 1]
                            .detach()
                            .to("cpu", copy=True)
                        )
                        collected[layer][keys[layer][window.window_id]] = BackboneOutput(
                            tokens=encoded.tokens[output_index : output_index + 1]
                            .detach()
                            .to("cpu", copy=True),
                            grid=encoded.grid,
                            pooled=pooled,
                            layer=layer,
                        )
            for layer in layer_tuple:
                written += store.put_many(
                    collected[layer],
                    video_id=record.video_id,
                    layer=layer,
                )
                already_present.update(collected[layer])

    return ExtractionResult(
        n_videos=len(records),
        n_windows=total_windows,
        n_written=written,
        n_hits=hits,
        layers=layer_tuple,
    )
