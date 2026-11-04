"""Feature extraction use cases: dry run, sharded extraction and registered merge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evd.data.decoder import make_decoder
from evd.data.feature_extraction import (
    enforce_size_limit,
    estimate_extraction,
    extract_records,
    select_inventory_records,
)
from evd.data.feature_store import SafetensorsFeatureStore
from evd.data.preprocess import vjepa21_preprocess_spec
from evd.exp.tracker import LocalTracker, git_info, no_git_allowed
from evd.models.backbones.vjepa21 import vjepa21_backbone
from evd.models.backbones.vjepa21_source import ARCHS, default_checkpoint


@dataclass(frozen=True)
class ExtractConfig:
    videos_from: Path
    inventory_csv: Path
    dataset_root: Path
    features_root: Path
    arch: str
    checkpoint: Path | None
    repo: Path | None
    size: int
    num_frames: int
    preprocess_variant: str
    preprocess_fps: float
    preprocess_dtype: str
    storage_dtype: str
    precision: str
    layers: tuple[int, ...] | None
    hop_s: float
    batch_size: int | None
    shard_index: int
    num_shards: int
    max_gb: float
    device: str
    dry_run: bool
    #: Which ``index-N.sqlite`` this task writes; ``None`` means ``shard_index``.
    #:
    #: ``shard_index``/``num_shards`` choose the slice of the video list
    #: (``selected[i::n]``), while the store shard chooses the index file name; on an
    #: already populated store the two must differ to avoid rewriting sealed index files.
    store_shard: int | None = None
    #: Extraction grid and nominal duration required by ``fixed``. They must match the
    #: ``protocol.temporal`` of the run that will read the store (``window_grid``,
    #: ``window_s``): a different grid gives zero cache hits. ``edp_v2`` needs
    #: ``grid="fixed", window_s=8.0`` (ADR-047).
    grid: str = "endpoint"
    window_s: float | None = None

    def __post_init__(self) -> None:
        """Validate where the spec is created, not where it is consumed.

        ``layers`` has no default: falling back to all hierarchical layers would write four
        times the disk space for layers never read (ADR-049).

        :raises ValueError: if ``layers`` is empty or missing, or ``grid`` is ``"fixed"``
            without ``window_s``.
        """
        if not self.layers:
            raise ValueError(
                "ExtractConfig.layers is required and has no default: the plausible default "
                "(the four hierarchical_layers) costs four times the disk, and three "
                "quarters would never be read (ADR-049). E2 uses (23,)"
            )
        if self.grid == "fixed" and self.window_s is None:
            raise ValueError(
                "ExtractConfig: grid='fixed' requires window_s (edp_v2: 8.0). Without it "
                "the window differs from the run one and no key would match"
            )


def run_extract_job(config: ExtractConfig) -> dict[str, Any]:
    try:
        architecture = ARCHS[config.arch]
    except KeyError:
        raise ValueError(f"unknown arch {config.arch!r}; allowed: {sorted(ARCHS)}") from None
    layers = config.layers  # required: guaranteed by ExtractConfig.__post_init__
    preprocess = vjepa21_preprocess_spec(
        config.preprocess_variant,  # type: ignore[arg-type]
        fps=config.preprocess_fps,
        num_frames=config.num_frames,
        size=config.size,
        dtype=config.preprocess_dtype,  # type: ignore[arg-type]
    )
    records = select_inventory_records(
        config.inventory_csv,
        config.videos_from,
        shard_index=config.shard_index,
        num_shards=config.num_shards,
    )
    estimate = estimate_extraction(
        records,
        preprocess=preprocess,
        hop_s=config.hop_s,
        layers=layers,
        embed_dim=architecture.embed_dim,
        storage_dtype=config.storage_dtype,
        grid=config.grid,  # type: ignore[arg-type]
        window_s=config.window_s,
    )
    enforce_size_limit(estimate, max_gb=config.max_gb)
    report: dict[str, Any] = {
        "mode": "dry-run" if config.dry_run else "extract",
        "shard": {
            "index": config.shard_index,
            "count": config.num_shards,
            "store_shard": (
                config.shard_index if config.store_shard is None else config.store_shard
            ),
        },
        "layers": list(layers),
        "estimate": estimate.as_dict(),
        # The grid enters the report because it is not part of `store_meta.json`:
        # `preprocess_hash` covers fps, frames, size and geometry, not hop or anchoring.
        "grid": {"grid": config.grid, "window_s": config.window_s, "hop_s": config.hop_s},
        "regime": {
            "precision": config.precision,
            "storage_dtype": config.storage_dtype,
            "attn": "sdpa",
            "size": config.size,
            "batch_size": config.batch_size,
        },
    }
    if config.dry_run:
        return report

    repository = git_info()
    if (repository["dirty"] or not repository["sha"]) and not no_git_allowed(repository):
        raise RuntimeError("official extraction rejected: dirty working tree or SHA unavailable")
    if config.batch_size is None or config.batch_size <= 0:
        raise ValueError("--batch-size is required and must be > 0 for a non-dry-run extraction")
    checkpoint = config.checkpoint or default_checkpoint(config.arch)
    backbone = vjepa21_backbone(
        arch=config.arch,
        checkpoint=checkpoint,
        state_dict_key=architecture.ckpt_key,
        size=config.size,
        num_frames=config.num_frames,
        preprocess_variant=config.preprocess_variant,
        preprocess_fps=config.preprocess_fps,
        preprocess_dtype=config.preprocess_dtype,
        expected={
            "embed_dim": architecture.embed_dim,
            "depth": architecture.depth,
            "num_heads": architecture.num_heads,
            "head_dim": architecture.head_dim,
        },
        hierarchical_layers=architecture.hierarchical_layers,
        repo=config.repo,
        device=config.device,
        strict=True,
    )
    store = SafetensorsFeatureStore(
        config.features_root,
        backbone_id=backbone.backbone_id,
        ckpt_sha=backbone.ckpt_sha,
        preprocess=preprocess,
        storage_dtype=config.storage_dtype,
        precision=config.precision,
        code_version=str(repository["sha"]),
        shard=config.shard_index if config.store_shard is None else config.store_shard,
    )
    result = extract_records(
        records,
        dataset_root=config.dataset_root,
        preprocess=preprocess,
        hop_s=config.hop_s,
        layers=layers,
        backbone=backbone,
        decoder=make_decoder("pyav"),
        store=store,
        batch_size=config.batch_size,
        device=config.device,
        precision=config.precision,
        grid=config.grid,  # type: ignore[arg-type]
        window_s=config.window_s,
    )
    report.update(
        {
            "store_dir": str(store.root),
            "index": str(store.index_path),
            "result": result.as_dict(),
            "git": repository,
            # More than one entry means the store was filled across different commits; not an error,
            # but the first thing to check if a number is surprising.
            "code_versions": list(store.code_versions()),
        }
    )
    return report


def merge_feature_shards(
    *,
    store_dir: Path,
    num_shards: int,
    run_id: str,
    expected_count: int | None,
) -> dict[str, Any]:
    """Merge the shards atomically and register the official E0d row.

    :raises ValueError: if the merged count differs from ``expected_count``.
    :raises RuntimeError: if the index replay is incomplete after the merge.
    """
    store = SafetensorsFeatureStore.open_existing(store_dir, create_index=False)
    tracker = LocalTracker()
    handle = tracker.start(
        run_id,
        {
            "store_dir": str(store.root),
            "num_shards": num_shards,
            "expected_count": expected_count,
            "store_meta": store.metadata,
        },
        official=True,
        regime={
            "backbone": store.backbone_id,
            "preprocess_hash": store.preprocess.hash(),
            "storage_dtype": store.storage_dtype,
            "precision": store.precision,
            "attn": "sdpa",
        },
    )
    count = store.merge_shards(num_shards)
    if expected_count is not None and count != expected_count:
        raise ValueError(f"merge: expected {expected_count} features, found {count}")
    verification = store.verify_index()
    if verification["hit_rate"] != 1.0:
        raise RuntimeError(f"incomplete index replay after the merge: {verification}")
    metrics = {
        "n_features": count,
        "n_shards": num_shards,
        "store_dir": str(store.root),
        "replay_index_hit_rate": verification["hit_rate"],
    }
    tracker.finish(handle, metrics, extra={"exp": "e0d", "artifact": "feature_store"})
    return metrics
