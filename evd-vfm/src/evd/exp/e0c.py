"""Sanity check on reduced UCF101, entirely on frozen features."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from evd.data.dataset import cached_output_from_batch, collate_windows
from evd.data.feature_store import SafetensorsFeatureStore
from evd.data.inventory import InventoryRecord, index_by_video_id, read_records
from evd.data.windows import CausalWindowSampler
from evd.exp.tracker import LocalTracker, plain_config
from evd.models.heads.attentive import AttentiveProbe
from evd.train.budget import BudgetSpec, apply_numeric_regime
from evd.train.checkpoint import (
    Checkpointer,
    CheckpointIdentity,
    CheckpointSpec,
    InterruptFlag,
    TrainingInterrupted,
    install_interrupt_handler,
)
from evd.train.trainer import Trainer
from evd.types import BackboneOutput, WindowSpec
from evd.utils.hashing import stable_hash
from evd.utils.io import read_json, write_json
from evd.utils.logging import get_logger
from evd.utils.seeding import seed_all

log = get_logger(__name__)


@dataclass(frozen=True)
class LabeledVideo:
    video_id: str
    class_name: str
    label: int


@dataclass(frozen=True)
class CachedExample:
    key: str
    window: WindowSpec
    label: int


@dataclass(frozen=True)
class E0cConfig:
    store_dir: Path
    inventory_csv: Path
    train_split: Path
    test_split: Path
    class_index: Path
    classes_file: Path
    layer: int
    hop_s: float
    max_steps: int
    batch_size: int
    lr: float
    weight_decay: float
    device: str
    seed: int
    reference_top1: float
    reference_source: str
    ballpark_delta_pp: float
    run_id: str
    #: Distinguishes comparisons under the same run_id; derived from layer and on-disk dtype
    #: when missing.
    predictions_label: str | None = None
    #: Training videos per class; ``None`` means all (E0c behaviour). Used to remove the
    #: ceiling (with all videos the probe reaches 99.7%). Unrelated to the few-shot ``k``.
    train_video_per_classe: int | None = None
    #: Windows taken from each training video, evenly spaced, so the budget is identical
    #: across cells with different spans.
    finestre_per_video: int | None = None
    #: Subsampling seed, separate from ``seed`` (initialization), so two cells can be
    #: compared on the same support.
    subset_seed: int | None = None
    #: Budget in windows per class instead of the ``train_video_per_classe ×
    #: finestre_per_video`` rectangle, for cells sharing the span. Mutually exclusive with
    #: ``train_video_per_classe``.
    train_finestre_per_classe: int | None = None
    #: Keep training and test features in RAM instead of rereading them every epoch. No
    #: number changes, but on a network store time drops by an order of magnitude (job
    #: 534406). Off by default for low-RAM machines.
    precarica_feature: bool = False
    #: ``window_id`` -> class name map written by :mod:`evd.exp.e0f`. ``None`` uses the video
    #: label (E0c on UCF101). When given, each window carries the label of its content
    #: (last-frame rule, docs/04 §4.1), windows missing from the map are excluded and top-1
    #: is computed per window.
    window_labels: Path | None = None
    #: Intra-training checkpoint folder; operational, outside the checkpoint identity.
    checkpoint_dir: Path | None = None
    #: Period in steps; zero disables resumption, a positive value requires
    #: ``checkpoint_dir``.
    checkpoint_every: int = 0


def _checkpointer(
    config: E0cConfig,
    store: SafetensorsFeatureStore,
    etichetta: str,
) -> Checkpointer | None:
    """Prepare resumption of the frozen probe, if requested.

    :param config: full probe configuration.
    :param store: already open feature store; binds backbone and checkpoint.
    :param etichetta: identity of the replica within the run.
    :return: the checkpointer, or ``None`` when the period is zero.
    :raises ValueError: if the period is negative or the folder is missing.
    """
    if config.checkpoint_every < 0:
        raise ValueError("checkpoint_every must be >= 0")
    if config.checkpoint_every == 0:
        return None
    if config.checkpoint_dir is None:
        raise ValueError("checkpoint_dir is required when checkpoint_every > 0")

    flat = plain_config(asdict(config))
    assert isinstance(flat, dict)
    for field in ("checkpoint_dir", "checkpoint_every"):
        flat.pop(field, None)
    flag = InterruptFlag()
    installed = install_interrupt_handler(flag)
    log.info(
        "E0c resumption active: checkpoint every %d steps in %s, signals %s",
        config.checkpoint_every,
        config.checkpoint_dir,
        ", ".join(installed) or "nessuno",
    )
    return Checkpointer(
        CheckpointSpec(dir=config.checkpoint_dir, every=config.checkpoint_every),
        CheckpointIdentity(
            run_id=f"{config.run_id}:{etichetta}",
            config_sha=stable_hash(flat),
            backbone_id=store.backbone_id,
            ckpt_sha=store.ckpt_sha,
        ),
        flag=flag,
    )


def read_window_labels(path: Path, classes: dict[str, int]) -> dict[str, int]:
    """Read the ``window_id`` -> class index map produced by E0f.

    :param path: JSON with keys ``classi`` and ``etichette``.
    :param classes: name -> index table of the split, from :func:`read_selected_classes`.
    :return: ``window_id`` -> class index.
    :raises ValueError: if the map names a class missing from the split (labels would
        silently enter permuted).
    """
    raw = read_json(path)
    labels = raw["etichette"]
    unknown = {name for name in raw.get("classi", ()) if name not in classes}
    if unknown:
        raise ValueError(
            f"{path}: classes missing from the split: {sorted(unknown)}. "
            f"The split declares {sorted(classes)}"
        )
    outside = {name for name in labels.values() if name not in classes}
    if outside:
        raise ValueError(f"{path}: labels with unknown classes: {sorted(outside)[:5]}")
    return {window_id: classes[name] for window_id, name in labels.items()}


def read_selected_classes(class_index: Path, classes_file: Path) -> dict[str, int]:
    wanted = {
        line.strip()
        for line in classes_file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    indexed: list[tuple[int, str]] = []
    for line_number, raw in enumerate(
        class_index.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        fields = raw.split()
        if len(fields) != 2 or not fields[0].isdigit():
            raise ValueError(f"malformed classInd at line {line_number}: {raw!r}")
        if fields[1] in wanted:
            indexed.append((int(fields[0]), fields[1]))
    found = {name for _, name in indexed}
    if found != wanted:
        raise ValueError(f"classes not found in classInd: {sorted(wanted - found)}")
    return {name: label for label, (_, name) in enumerate(sorted(indexed))}


def read_labeled_split(path: Path, classes: dict[str, int]) -> list[LabeledVideo]:
    videos: list[LabeledVideo] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        fields = raw.strip().replace("\\", "/").split()
        if not fields:
            continue
        relative = PurePosixPath(fields[0])
        if len(relative.parts) < 2:
            raise ValueError(f"malformed split at line {line_number}: {raw!r}")
        class_name = relative.parts[0]
        if class_name not in classes:
            continue
        video_id = relative.stem
        if video_id in seen:
            raise ValueError(f"duplicated video in split {path}: {video_id}")
        seen.add(video_id)
        videos.append(LabeledVideo(video_id, class_name, classes[class_name]))
    if not videos:
        raise ValueError(f"empty split after the class filter: {path}")
    return videos


#: Maximum fraction of videos hosting no complete window before the result stops
#: describing the declared dataset: below it they are dropped and counted, above it the
#: run stops.
MAX_SHORT_FRACTION = 0.05


def build_cached_examples(
    videos: Sequence[LabeledVideo],
    *,
    inventory: dict[str, InventoryRecord],
    store: SafetensorsFeatureStore,
    layer: int,
    hop_s: float,
    window_labels: Mapping[str, int] | None = None,
) -> tuple[list[CachedExample], list[str]]:
    """Return the cached examples and the list of videos too short for a window.

    A video shorter than a causal window is a dataset property (3 of 1363 UCF101 clips),
    so such videos are dropped, counted and reported in the metrics.

    :raises KeyError: if a video is missing from the inventory.
    :raises RuntimeError: if features are missing from the store.
    :raises ValueError: if too many videos are too short or no example remains.
    """
    examples: list[CachedExample] = []
    missing: list[str] = []
    too_short: list[str] = []
    without_labels: list[str] = []
    for video in videos:
        try:
            record = inventory[video.video_id]
        except KeyError:
            raise KeyError(f"{video.video_id} missing from the UCF101 inventory") from None
        sampler = CausalWindowSampler.from_spec(store.preprocess, hop_s)
        windows = sampler.windows_for_video(record.to_meta())
        if not windows:
            too_short.append(video.video_id)
            continue
        first = len(examples)
        for window in windows:
            if window_labels is None:
                label = video.label
            else:
                found = window_labels.get(window.window_id)
                if found is None:
                    continue  # window outside the E0f plan: excluded
                label = found
            key = store.key(
                store.backbone_id,
                store.ckpt_sha,
                store.preprocess,
                layer,
                window,
            )
            if key not in store:
                missing.append(window.window_id)
            else:
                examples.append(CachedExample(key, window, label))
        if len(examples) == first:
            without_labels.append(video.video_id)
    if missing:
        raise RuntimeError(
            f"features missing for {len(missing)} windows: {missing[:3]}; E0c does not fall back to pixels"
        )
    if videos and len(too_short) / len(videos) > MAX_SHORT_FRACTION:
        raise ValueError(
            f"{len(too_short)} of {len(videos)} videos "
            f"({len(too_short) / len(videos):.1%}) host no complete "
            f"window at hop={hop_s}s: above the allowed {MAX_SHORT_FRACTION:.0%}. "
            f"First: {too_short[:3]}. With this hop the result would no longer "
            "describe the declared split."
        )
    if without_labels and window_labels is not None:
        log.warning(
            "%d videos contribute no labelled window: %s",
            len(without_labels),
            without_labels[:3],
        )
    if not examples:
        raise ValueError("no cached example: empty split or hop too large")
    return examples, too_short


def subsample_support(
    examples: Sequence[CachedExample],
    *,
    videos_per_class: int,
    finestre_per_video: int,
    seed: int,
    windows_per_class: int | None = None,
) -> list[CachedExample]:
    """Reduce training to a fixed budget, identical across cells.

    Video choice depends only on ``seed`` and the sorted video list of each class, never on
    the number of windows (which changes with the span); windows are evenly spaced within
    each video, so cells with different spans train on the same number of examples from
    the same videos.

    :param examples: cached training examples, already verified in the store.
    :param video_per_classe: videos to keep per class.
    :param finestre_per_video: windows to keep per video, evenly spaced.
    :param seed: subsampling seed (not the initialization one).
    :param finestre_per_classe: budget in windows per class; when set, the other two are
        ignored and the windows are spread over all videos of the class (for cells sharing
        the span).
    :return: the chosen subset, in stable order.
    :raises ValueError: if a class lacks videos or a chosen video lacks windows.
    """
    if windows_per_class is not None:
        return _subsample_to_windows(examples, windows_per_class, seed)
    if videos_per_class <= 0 or finestre_per_video <= 0:
        raise ValueError("video_per_classe and finestre_per_video must be positive")
    per_class: dict[int, dict[str, list[CachedExample]]] = defaultdict(lambda: defaultdict(list))
    for example in examples:
        per_class[example.label][example.window.video_id].append(example)
    rng = random.Random(seed)
    chosen: list[CachedExample] = []
    for label in sorted(per_class):
        video = sorted(per_class[label])
        if len(video) < videos_per_class:
            raise ValueError(
                f"class {label}: {len(video)} videos available, {videos_per_class} "
                f"needed. The budget cannot be met on this split"
            )
        for video_id in sorted(rng.sample(video, videos_per_class)):
            windows = sorted(per_class[label][video_id], key=lambda e: e.window.end_frame)
            if len(windows) < finestre_per_video:
                raise ValueError(
                    f"{video_id}: {len(windows)} windows, {finestre_per_video} needed. "
                    "With this span the video cannot meet the budget, and keeping it with "
                    "fewer windows would break parity across cells"
                )
            if finestre_per_video == 1:
                indices = [0]
            else:
                step = (len(windows) - 1) / (finestre_per_video - 1)
                indices = [round(i * step) for i in range(finestre_per_video)]
            chosen.extend(windows[i] for i in indices)
    return chosen


def _subsample_to_windows(
    examples: Sequence[CachedExample], how_many: int, seed: int
) -> list[CachedExample]:
    """Take ``how_many`` windows per class, spread over the videos of the class.

    The quota is split as evenly as possible, the remainder going to the richest videos in
    deterministic order; within each video windows are evenly spaced in time.

    :raises ValueError: if a class does not have enough windows in total.
    """
    per_class: dict[int, dict[str, list[CachedExample]]] = defaultdict(lambda: defaultdict(list))
    for example in examples:
        per_class[example.label][example.window.video_id].append(example)
    chosen: list[CachedExample] = []
    for label in sorted(per_class):
        available = sum(len(v) for v in per_class[label].values())
        if available < how_many:
            raise ValueError(
                f"class {label}: {available} windows available, {how_many} "
                f"needed. The budget cannot be met on this split"
            )
        video = sorted(per_class[label], key=lambda v: (-len(per_class[label][v]), v))
        base, rest = divmod(how_many, len(video))
        taken: list[CachedExample] = []
        remainder: list[CachedExample] = []
        for i, video_id in enumerate(video):
            windows = sorted(per_class[label][video_id], key=lambda e: e.window.end_frame)
            quota = base + (1 if i < rest else 0)
            if quota >= len(windows):
                taken.extend(windows)
                continue
            if quota <= 1:
                choices = windows[:1] if quota == 1 else []
            else:
                step = (len(windows) - 1) / (quota - 1)
                choices = [windows[round(j * step)] for j in range(quota)]
            taken.extend(choices)
            remainder.extend(e for e in windows if e not in choices)
        # A video poorer than its quota leaves a gap, filled from the richer ones instead of
        # returning a smaller class.
        if len(taken) < how_many:
            remainder.sort(key=lambda e: (e.window.video_id, e.window.end_frame))
            taken.extend(remainder[: how_many - len(taken)])
        chosen.extend(taken[:how_many])
    # The seed does not choose which windows (deterministic rule, shared by all cells) but
    # their order in the loader.
    random.Random(seed).shuffle(chosen)
    return chosen


class _FeatureDataset(Dataset[dict[str, Any]]):
    """Cached features as a dataset, with or without a RAM copy.

    ``precarica`` changes no number, only time (rereading half a gigabyte from NFS 138k
    times took about four hours, job 534406). Off by default for low-RAM machines.
    """

    def __init__(
        self,
        examples: Sequence[CachedExample],
        store: SafetensorsFeatureStore,
        preload: bool = False,
    ) -> None:
        self.examples = list(examples)
        self.store = store
        self._cache: dict[str, BackboneOutput] | None = None
        if preload:
            self._cache = {}
            for example in self.examples:
                if example.key in self._cache:
                    continue
                output = self.store.get(example.key)
                if output is None:
                    raise RuntimeError(f"feature disappeared during preloading: {example.key}")
                self._cache[example.key] = output
            byte = sum(
                output.tokens.numel() * output.tokens.element_size()
                for output in self._cache.values()
            )
            log.info(
                "preloaded %d windows in RAM (%.1f GB): the probe no longer reads from disk",
                len(self._cache),
                byte / 1e9,
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        output = (
            self._cache[example.key] if self._cache is not None else self.store.get(example.key)
        )
        if output is None:
            raise RuntimeError(f"feature disappeared during E0c: {example.key}")
        return {
            "features": output.tokens.squeeze(0),
            "grid": output.grid,
            "layer": output.layer,
            "levels": output.levels,
            "pooled": None if output.pooled is None else output.pooled.squeeze(0),
            "label": -1,
            "class_label": example.label,
            "window": example.window,
        }


def budget_unit_of(config: E0cConfig) -> str:
    """Return the unit the training budget is expressed in, if any.

    :param config: probe configuration.
    :return: ``"finestre"``, ``"video"`` or ``"nessuno"`` (saturated regime, all training
        examples of the split).
    :raises ValueError: if both units are set, or one of them is incomplete.
    """
    if config.train_finestre_per_classe is not None:
        if config.train_video_per_classe is not None:
            raise ValueError(
                "train_finestre_per_classe and train_video_per_classe are mutually exclusive: "
                "the budget is expressed in windows **or** in the "
                "video x windows rectangle, never both"
            )
        if config.subset_seed is None:
            raise ValueError(
                "train_finestre_per_classe without subset_seed: without the subsampling "
                "seed two cells would see different supports, and the paired "
                "comparison would lose its only advantage"
            )
        return "finestre"
    rectangle = (
        config.train_video_per_classe,
        config.finestre_per_video,
        config.subset_seed,
    )
    if all(value is None for value in rectangle):
        return "nessuno"
    if any(value is None for value in rectangle):
        raise ValueError(
            "train_video_per_classe, finestre_per_video and subset_seed must be "
            f"set together or not at all: got {rectangle}"
        )
    return "video"


def run_e0c(config: E0cConfig) -> dict[str, Any]:
    if not 0 <= config.reference_top1 <= 1:
        raise ValueError("reference_top1 must be a fraction in [0,1]")
    if config.max_steps <= 0 or config.batch_size <= 0 or config.lr <= 0:
        raise ValueError("max_steps, batch_size and lr must be positive")
    seed_all(config.seed)
    classes = read_selected_classes(config.class_index, config.classes_file)
    train_videos = read_labeled_split(config.train_split, classes)
    test_videos = read_labeled_split(config.test_split, classes)
    overlap = {video.video_id for video in train_videos}.intersection(
        video.video_id for video in test_videos
    )
    if overlap:
        raise ValueError(f"videos shared between train and test: {sorted(overlap)[:3]}")
    inventory = index_by_video_id(read_records(config.inventory_csv))
    store = SafetensorsFeatureStore.open_existing(config.store_dir, create_index=False)
    label_text = config.predictions_label or f"L{config.layer}-{store.storage_dtype}"
    checkpoint = _checkpointer(config, store, label_text)
    verification = store.verify_index()
    if verification["hit_rate"] != 1.0:
        if not verification["n_indexed"]:
            raise RuntimeError(
                f"feature store {config.store_dir} has no merged index: "
                "extraction writes index-<shard>.sqlite, the reader looks for "
                "index.sqlite. merge_feature_shards is missing, not the features"
            )
        raise RuntimeError(f"incomplete feature store: {verification}")
    window_labels = (
        None if config.window_labels is None else read_window_labels(config.window_labels, classes)
    )
    train_examples, short_train = build_cached_examples(
        train_videos,
        inventory=inventory,
        store=store,
        layer=config.layer,
        hop_s=config.hop_s,
        window_labels=window_labels,
    )
    test_examples, test_short = build_cached_examples(
        test_videos,
        inventory=inventory,
        store=store,
        layer=config.layer,
        hop_s=config.hop_s,
        window_labels=window_labels,
    )
    budget_unit = budget_unit_of(config)
    if budget_unit == "finestre":
        train_examples = subsample_support(
            train_examples,
            videos_per_class=1,
            finestre_per_video=1,
            seed=config.subset_seed,  # type: ignore[arg-type]
            windows_per_class=config.train_finestre_per_classe,
        )
    elif budget_unit == "video":
        train_examples = subsample_support(
            train_examples,
            videos_per_class=config.train_video_per_classe,  # type: ignore[arg-type]
            finestre_per_video=config.finestre_per_video,  # type: ignore[arg-type]
            seed=config.subset_seed,  # type: ignore[arg-type]
        )
    first = store.get(train_examples[0].key)
    assert first is not None
    embed_dim = int(first.tokens.shape[-1])
    budget = BudgetSpec(
        max_steps=config.max_steps,
        batch_size=config.batch_size,
        lr_grid=(config.lr,),
        weight_decay=config.weight_decay,
        precision="fp32",
        tf32=False,
        attn="sdpa",
    )
    numeric_regime = apply_numeric_regime(budget, config.device)
    head = AttentiveProbe(
        embed_dim=embed_dim,
        n_classes=len(classes),
        n_queries=1,
        n_heads=16,
        depth=4,
        norm=True,
    ).to(config.device)
    train_loader = DataLoader(
        _FeatureDataset(train_examples, store, preload=config.precarica_feature),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_windows,
        generator=torch.Generator().manual_seed(config.seed),
    )
    test_loader = DataLoader(
        _FeatureDataset(test_examples, store, preload=config.precarica_feature),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_windows,
    )
    tracker = LocalTracker()
    handle = tracker.start(
        config.run_id,
        asdict(config),
        official=True,
        regime={
            **numeric_regime,
            "size": store.preprocess.size,
            "protocol": "causal-window-mean-logit-ballpark",
        },
    )

    if checkpoint is not None and checkpoint.should_stop():
        raise TrainingInterrupted(
            f"{config.run_id}:{label_text}: interruption requested before the fit; "
            "relaunch the same replica"
        )
    fit = Trainer(device=config.device).fit(
        None,
        head,
        train_loader,
        budget,
        levels=(config.layer,),
        lr=config.lr,
        checkpoint=checkpoint,
    )
    if fit.interrupted:
        raise TrainingInterrupted(
            f"{config.run_id}:{label_text}: training interrupted at step "
            f"{fit.steps}/{config.max_steps}; relaunch the same replica"
        )

    # The evaluation unit follows the label unit: with per-window labels, averaging the logits
    # of a video would mix windows of different classes.
    per_window = window_labels is not None
    logits_per_unit: dict[str, list[torch.Tensor]] = defaultdict(list)
    expected: dict[str, int] = {}
    head.eval()
    with torch.inference_mode():
        for batch in test_loader:
            output = cached_output_from_batch(
                batch,
                device=config.device,
                dtype=torch.float32,
            )
            logits = head(output).cpu()
            for window, label, row in zip(
                batch["window"], batch["class_label"].tolist(), logits, strict=True
            ):
                key = window.window_id if per_window else window.video_id
                logits_per_unit[key].append(row)
                expected[key] = int(label)
    # Per-video predictions and logits are saved, not only counted: at 99.5% top-1 is
    # saturated, so runs are compared on paired decisions.
    correct = 0
    predictions: dict[str, dict[str, Any]] = {}
    for unit, rows in sorted(logits_per_unit.items()):
        mean_values = torch.stack(rows).mean(dim=0)
        prediction = int(mean_values.argmax())
        expected_value = expected[unit]
        correct += int(prediction == expected_value)
        predictions[unit] = {
            "predetto": prediction,
            "atteso": expected_value,
            "n_finestre": len(rows),
            "logit": [round(float(v), 6) for v in mean_values.tolist()],
        }
    top1 = correct / len(expected)
    delta_pp = 100 * (top1 - config.reference_top1)
    within = abs(delta_pp) <= config.ballpark_delta_pp
    metrics = {
        "top1": top1,
        # Number of units top-1 is computed on, so runs with different granularity are not
        # confused.
        "granularita": "finestra" if per_window else "video",
        "n_unita_test": len(expected),
        "window_labels": None if config.window_labels is None else str(config.window_labels),
        "n_classes": len(classes),
        "n_train_videos": len(train_videos),
        "n_test_videos": len(test_videos),
        "n_train_windows": len(train_examples),
        "n_test_windows": len(test_examples),
        # Videos dropped for being shorter than a causal window are part of the result.
        "n_train_video_troppo_corti": len(short_train),
        "n_test_video_troppo_corti": len(test_short),
        "video_troppo_corti": sorted(short_train + test_short),
        "feature_hit_rate": verification["hit_rate"],
        "reference_top1": config.reference_top1,
        "reference_source": config.reference_source,
        "delta_pp": delta_pp,
        "within_declared_ballpark": within,
        "comparison_caveat": (
            "not commensurable: causal windows and reduced subset against the "
            "published multi-segment/multi-view protocol"
        ),
        "train_video_per_classe": config.train_video_per_classe,
        "train_finestre_per_classe": config.train_finestre_per_classe,
        "unita_budget": budget_unit,
        "finestre_per_video": config.finestre_per_video,
        "subset_seed": config.subset_seed,
        "train_steps": fit.steps,
        "resumed_from": fit.resumed_from,
        "final_loss": fit.final_loss,
        "trainable_params": fit.trainable_params,
        "lr": fit.lr,
    }
    # The label distinguishes comparisons under the same run_id; a fixed name would silently
    # overwrite them.
    artifact = write_json(
        handle.dir / f"predizioni-{label_text}.json",
        {
            "run_id": config.run_id,
            "store_dir": str(store.root),
            "precision": store.precision,
            "storage_dtype": store.storage_dtype,
            "layer": config.layer,
            "seed": config.seed,
            "top1": top1,
            "predizioni": predictions,
        },
    )
    metrics["predictions_json"] = str(artifact)
    tracker.log_history(handle, fit.history)
    tracker.finish(
        handle,
        metrics,
        status="ok" if within else "needs-analysis",
        extra={"exp": "e0c", "sanity_only": True},
    )
    return metrics
