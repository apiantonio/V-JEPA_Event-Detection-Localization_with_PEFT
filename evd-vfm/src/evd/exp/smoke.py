"""End-to-end smoke test: the vertical slice (docs/06 M0, docs/08 §1).

When green it shows that data, causal windows, backbone, adaptation, head, chunk
scores, primary metrics, causal frame projection, figure and registry form a working
path on CPU in under a minute. It does not show that the model learns (``TinyBackbone``
with random weights on a coarse synthetic signal, ``evd.data.synthetic``).

It has no path of its own: it builds the synthetic world, generates the toy split and
calls :class:`~evd.exp.runner.ExperimentRunner`, the same engine as E2 and E3. It is not
an experiment: the split declares ``protocol_version: toy``, the ``run_id`` starts with
``smoke_synth_tiny_`` and the registry row is ``official: false``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch

from evd.data.dataset import WindowDataset, WindowItem, collate_windows
from evd.data.decoder import make_decoder
from evd.data.feature_store import SafetensorsFeatureStore
from evd.data.windows import CausalWindowSampler
from evd.exp.runner import ExperimentRunner, split_path_for
from evd.exp.sources import DatasetBundle, build_dataset
from evd.protocol.split import SplitSpec, save_split
from evd.utils import env
from evd.utils.hashing import stable_hash
from evd.utils.io import ensure_dir
from evd.utils.logging import get_logger, setup_logging

log = get_logger(__name__)

#: Videos per class moved to the toy validation split: the minimum that makes
#: validation non-degenerate (threshold and LR selection are measured there).
TOY_VAL_PER_CLASS = 1

#: Adaptations the smoke test can traverse, deliberately tiny (r=2, n=2): it shows the
#: path exists, not that the method works.
ADAPTATIONS: dict[str, dict[str, Any]] = {
    "none": {"family": "none", "name": "frozen"},
    "lora": {
        "family": "reparam",
        "name": "lora-r2-qkvo",
        "method": "lora",
        "r": 2,
        "alpha": 4.0,
        "placement": "qkvo",
    },
    "vpt": {
        "family": "prompt",
        "name": "vptdeep-n2",
        "variant": "vpt_deep",
        "n_tokens": 2,
        # `none` rather than `rope_identity`: Tiny has no RoPE by default and the policy is
        # validated against the backbone.
        "pos_policy": "none",
    },
    # E3-C (ADR-062): reparam and prompt together, so the gradient re-arming of
    # `CompositeAdapt` is exercised end to end.
    "composite": {
        "family": "composite",
        "name": "lora-r2-vptdeep-n2",
        "componenti": {
            "reparam": {
                "family": "reparam",
                "name": "lora-r2-qkvo",
                "method": "lora",
                "r": 2,
                "alpha": 4.0,
                "placement": "qkvo",
            },
            "prompt": {
                "family": "prompt",
                "name": "vptdeep-n2",
                "variant": "vpt_deep",
                "n_tokens": 2,
                "pos_policy": "none",
            },
        },
    },
}


@dataclass(frozen=True)
class SmokeConfig:
    out_dir: Path | None = None
    """Where the synthetic world goes (regenerable from the seed: a true cache).
    Default: ``$EVD_CACHE_DIR/smoke``, never inside the repository. Run artifacts go to
    ``$EVD_RUNS_DIR/{run_id}/``."""
    seed: int = 0
    """Seed of the synthetic world (pixels), starting from 0."""
    run_seed: int = 1
    """Seed of the run identity: it enters ``run_id`` and split, so it must be a
    positive integer (docs/05 §8); kept separate so regenerating the world does not
    rename the run."""
    layer: int = -1
    max_steps: int = 20
    batch_size: int = 8
    lr: float = 3e-2
    """Toy budget, recalibrated for the fully supervised binary regime (F11): the old
    20 steps did not move the logit and left the smoke AUC below chance. The values are
    also in ``configs/train/probe_toy.yaml``, the Hydra path of the same smoke test."""
    eval_every: int = 10
    """Validation period in steps; ten rather than one to keep the smoke test fast."""
    hop_s: float = 0.25
    """Toy hop, not the protocol one: low on purpose, so each synthetic video yields
    enough chunks for a meaningful AUC."""
    train_sampling: str = "static"
    """``static`` or ``dynamic`` (ADR-031 point 6); static by default so the smoke AUC
    does not vary with random endpoints."""
    frame_mapping: str = "causal"
    threshold_quantile: float = 0.95
    device: str | None = None
    make_figure: bool = True
    write_run: bool = True
    adapt: str = "none"
    """Row of :data:`ADAPTATIONS` to traverse; ``"none"`` is the default, ``lora``,
    ``vpt`` and ``composite`` make the slice also go through an adapter."""
    tiny: dict[str, Any] = field(default_factory=dict)
    world: dict[str, Any] = field(default_factory=dict)
    """Synthetic world parameters (``configs/data/synthetic.yaml``); empty means the
    ``SyntheticSpec`` defaults."""

    def __post_init__(self) -> None:
        if self.adapt not in ADAPTATIONS:
            raise ValueError(
                f"adapt {self.adapt!r} unknown to the smoke test: {sorted(ADAPTATIONS)}"
            )


def _class_of(bundle: DatasetBundle, video_id: str) -> int:
    """:return: the class of the video: that of its first interval, or ``Normal`` (exact in
    the synthetic world, where an anomalous video has a single interval).
    """
    annotation = bundle.annotations.get(video_id)
    if annotation is None or annotation.is_normal:
        return bundle.normal_idx
    return int(annotation.intervals[0][2])


def _split_dir(bundle: DatasetBundle, root: Path) -> Path:
    """Return the toy split folder, indexed by the world fingerprint.

    Binding the split to the world hash means a regenerated world never reuses an old
    partition.

    :return: the folder, created if missing.
    """
    fingerprint = stable_hash({"fonte": bundle.annotation_source, "train": bundle.train_videos})[
        :12
    ]
    return ensure_dir(root / fingerprint)


def _toy_split(bundle: DatasetBundle, cartella: Path) -> Path:
    """Generate the toy split, if not already present.

    The test set is the world query pool; validation is carved from the training pool per
    video and stratified by class (the docs/04 §5.3 fallback, applied to a synthetic
    world). Each video stays in a single partition.

    :param bundle: the already built world.
    :param cartella: folder from :func:`_split_dir`.
    :return: path of the existing, sealed split.
    """
    path = cartella / f"{bundle.name}_split.json"
    if path.exists():
        return path

    per_class: dict[int, list[str]] = {}
    for video_id in bundle.train_videos:
        per_class.setdefault(_class_of(bundle, video_id), []).append(video_id)
    val: set[str] = set()
    for class_name in sorted(per_class):
        candidati = sorted(per_class[class_name])
        if len(candidati) >= 2:
            val.update(candidati[:TOY_VAL_PER_CLASS])

    split = SplitSpec(
        protocol_version="toy",
        dataset=bundle.name,
        annotation_source=bundle.annotation_source,
        builder_commit="",
        created=datetime.now(timezone.utc).date().isoformat(),
        train=tuple(v for v in bundle.train_videos if v not in val),
        val=tuple(v for v in bundle.train_videos if v in val),
        test=tuple(bundle.query_videos),
    )
    save_split(path, split)
    log.info(
        "toy split generated: %s (%d train, %d val, %d test)",
        path,
        len(split.train),
        len(split.val),
        len(split.test),
    )
    return path


def _feature_store_replay(bundle: DatasetBundle, cfg: SmokeConfig, out_dir: Path) -> dict[str, Any]:
    """Extract the features of two videos and read them back without touching pixels.

    Checks the store round trip: the key is computed the same way when writing and
    reading, and ``WindowDataset`` does not silently fall back to the decoder.

    :return: hit rate, number of features and declared dtype.
    :raises RuntimeError: if the replay falls back to pixels or the index is incomplete.
    """
    from evd.models.backbones.base import BACKBONES

    device = cfg.device or env.device()
    backbone = BACKBONES.get("tiny")(device=device, seed=cfg.seed, **cfg.tiny)
    backbone.freeze()
    spec = backbone.preprocess_spec()
    sampler = CausalWindowSampler.from_spec(spec, hop_s=cfg.hop_s)
    decoder = make_decoder("pyav")

    chosen = sorted(bundle.inventory)[:2]
    items = [
        WindowItem(w, bundle.meta(v).path, label=0, class_label=bundle.normal_idx)
        for v in chosen
        for w in sampler.windows_for_video(bundle.meta(v))
    ]
    if not items:
        raise RuntimeError("no window from the synthetic world videos")

    dataset = WindowDataset(items, decoder, spec, cache_in_memory=True)
    checkpoint_id = stable_hash({"tiny_seed": cfg.seed, "tiny_config": cfg.tiny})
    ensure_dir(out_dir)
    # On NFS a file still mapped by the store becomes `.nfs*` when deleted, and its folder cannot
    # be removed until the file closes: the cleanup is best effort, the check does not depend on it.
    with TemporaryDirectory(
        prefix="feature-store-smoke-", dir=out_dir, ignore_cleanup_errors=True
    ) as temporary:
        store = SafetensorsFeatureStore(
            temporary,
            backbone_id=backbone.backbone_id,
            ckpt_sha=checkpoint_id,
            preprocess=spec,
            storage_dtype="fp16",  # format check, not a protocol decision
            precision="fp32",  # Tiny on CPU: AMP bf16 is a cluster choice
            code_version="smoke-v1",
        )
        keys: dict[str, tuple[str, ...]] = {}
        per_video: dict[str, dict[str, Any]] = {}
        with torch.no_grad():
            for index, item in enumerate(items):
                row = dataset[index]
                output = backbone.encode(row["clip"].unsqueeze(0).to(device), layer=cfg.layer)
                key = store.key(backbone.backbone_id, checkpoint_id, spec, cfg.layer, item.window)
                keys[item.window.window_id] = (key,)
                per_video.setdefault(item.window.video_id, {})[key] = output
        for video_id, entry_list in per_video.items():
            store.put_many(entry_list, video_id=video_id, layer=cfg.layer)

        from dataclasses import replace

        replay_items = [replace(i, cache_keys=keys[i.window.window_id]) for i in items]
        replay = WindowDataset(replay_items, decoder, spec, cache=store)
        for start in range(0, len(replay), 8):
            batch = collate_windows([replay[i] for i in range(start, min(start + 8, len(replay)))])
            if "features" not in batch or "clip" in batch:
                raise RuntimeError("the feature store replay fell back to pixels")
        hit_rate = replay.cache_hit_rate
        if hit_rate != 1.0 or store.count() != len(items):
            raise RuntimeError(
                f"incomplete feature replay: hit_rate={hit_rate}, "
                f"index={store.count()}, windows={len(items)}"
            )
        return {
            "hit_rate": hit_rate,
            "n_features": store.count(),
            "storage_dtype": "fp16-smoke-only",
        }


def _config(cfg: SmokeConfig, out_dir: Path, split_file: Path | None) -> dict[str, Any]:
    """Return the config composed by hand, with the same shape as ``configs/``.

    Hydra is bypassed on purpose so the backbone check does not depend on it;
    ``run_experiment.py +experiment=smoke`` covers the Hydra path.
    """
    return {
        "exp_id": "smoke",
        "seed": cfg.run_seed,
        "layer": cfg.layer,
        "device": cfg.device,
        "write_run": cfg.write_run,
        "make_figure": cfg.make_figure,
        "backbone": {"id": "tiny", "seed": cfg.seed, **cfg.tiny},
        "head": {"id": "linear", "norm": True},
        "adapt": ADAPTATIONS[cfg.adapt],
        "data": {
            "id": "synth",
            "name": "synthetic",
            "decoder": "pyav",
            "out_dir": str(out_dir / "synthetic"),
            **cfg.world,
        },
        "protocol": {
            "id": "synth_toy",
            "protocol_version": "toy",
            "split_file": None if split_file is None else str(split_file),
            "output": "binary",
            "temporal": {
                # The geometry comes from the backbone (`synthetic_preprocess_spec`): declared here so it
                # is checked, not chosen.
                "num_frames": 16,
                "target_fps": 8.0,
                "hop_s": cfg.hop_s,
                "chunk_label": "any_overlap",
                "train_sampling": cfg.train_sampling,
                "frame_mapping": cfg.frame_mapping,
            },
            "sampling": {
                "video_balance": "inverse_chunk_count",
                "overlap_weight_floor": 0.1,
                "class_balance": "none",
            },
            "classes": {"normal_idx": 0},
            "video_policy": "max",
            "threshold": {"source": "val", "quantile": cfg.threshold_quantile},
        },
        "train": {
            "max_steps": cfg.max_steps,
            "batch_size": cfg.batch_size,
            "lr_grid": [cfg.lr],
            "eval_every": cfg.eval_every,
            "precision": "fp32",  # the smoke test runs on CPU: fp32
            "tf32": False,
            "attn": "sdpa",
        },
        "eval": {"batch_size": cfg.batch_size},
    }


def run_smoke(cfg: SmokeConfig | None = None) -> dict[str, Any]:
    """Run the vertical slice and return the run metrics.

    :param cfg: smoke configuration; ``None`` uses the defaults.
    :return: runner metrics, plus the feature-store check, ``run_id`` and ``run_dir`` when
        the run was written.
    :raises RuntimeError: if the feature-store replay is not intact.
    """
    cfg = cfg or SmokeConfig()
    setup_logging()
    out_dir = Path(cfg.out_dir) if cfg.out_dir else env.cache_dir() / "smoke"
    log.info("smoke: out_dir=%s adapt=%s", out_dir, cfg.adapt)

    root = ensure_dir(out_dir / "splits")
    bundle = build_dataset(_config(cfg, out_dir, None)["data"])
    path = _toy_split(bundle, _split_dir(bundle, root))

    composite = _config(cfg, out_dir, path)
    # The runner resolves the split itself: check that the declared split is the one just
    # generated.
    expected = split_path_for(composite)
    if not expected.is_file():  # pragma: no cover - only if the conventions diverge
        raise RuntimeError(
            f"split generated but not found where the runner looks for it: {expected}"
        )

    result = ExperimentRunner().run(composite)
    out = dict(result.metrics)
    out["feature_store"] = _feature_store_replay(bundle, cfg, out_dir)
    if result.run_dir is not None:
        out["run_id"] = result.run_id
        out["run_dir"] = str(result.run_dir)
    # Frame AUC is no longer printed (frame-level metrics are outside the protocol, ADR-031);
    # the smoke guard is the chunk AUC.
    log.info(
        "metrics: chunk AUC %.3f · video AUC %.3f",
        out["chunk_auc"],
        out["video_auc"],
    )
    return out
