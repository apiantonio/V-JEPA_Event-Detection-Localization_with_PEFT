"""The input geometry is decided by measurement.

V-JEPA 2.1 does not state which input it expects: the released training configs are at
256 px, the README declares 384 for the distilled checkpoints and every eval config, and
the 384 stage recipe is not in the repository. UCF-Crime is 320×240, so 384 upsamples a
240p source by 1.6×, and the centre view of chain A discards 25% of the width.

The pilot measures the Cartesian product of three axes, with a frozen probe and an
identical budget, on the E0c sanity task (reduced UCF101, 10 classes):

==========  ==========================================================
resolution  256 (native RoPE grid) · 384 (checkpoint and eval)
crop        ``eval_center`` (chain A) · ``squash`` (whole field of view)
span        fps 7.5 (eval geometry) · fps 4 (pre-training sampling)
==========  ==========================================================

A full factorial separates the effect of resolution from that of the crop.

``three_views`` is excluded: the per-window feature store rejects a three-view geometry
by contract; the lateral views can be measured later as a delta on the winning geometry.

UCF101 is used instead of UCF-Crime to avoid choosing the geometry on the protocol query
set. UCF101 shares the 320×240 resolution, so the cost of upsampling a 240p source
transfers; the task does not. The pilot ranks geometries by the information surviving
preprocessing, not by anomaly-detection performance.

All cells see the same videos (:func:`prepare_lists`): eligibility is decided once, on
the longest window of the plan, so cells are not evaluated on different video sets.

Decision criterion (declared before the results; second version, see ``docs/10`` §E0e):

1. The ceiling is removed by reducing the probe data budget
   (``train_video_per_classe`` × ``finestre_per_video``); :func:`calibrate` fixes it on
   the reference cell only, aiming at a pre-declared top-1 band.
2. Cells are compared paired on the same support: for replica *r* all cells use
   ``subset_seed = r``, and subsampling depends only on the seed and the sorted video
   list.
3. The primary statistic is the sign: per axis, 4 contrasts × 5 replicas = 20 paired
   comparisons; an axis concludes if a level wins at least 16 of the non-tied ones
   (P(X ≥ 16 | n = 20, p = 0.5) ≈ 0.006).

Final configuration: the winning level of each concluding axis; otherwise the declared
fallback: 256 (fewer tokens), ``eval_center`` (official chain), fps 7.5 (eval geometry).
"""

from __future__ import annotations

import shutil
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from math import ceil, comb
from pathlib import Path, PurePosixPath
from typing import Any

from evd.data.feature_extraction import tokens_per_window
from evd.data.inventory import read_records
from evd.data.preprocess import vjepa21_preprocess_spec
from evd.data.windows import CausalWindowSampler
from evd.exp.e0c import E0cConfig, read_labeled_split, read_selected_classes, run_e0c
from evd.exp.feature_jobs import ExtractConfig, merge_feature_shards, run_extract_job
from evd.exp.tracker import LocalTracker
from evd.models.backbones.vjepa21_source import ARCHS
from evd.types import PreprocessSpec
from evd.utils.io import read_json, write_json, write_text_atomic
from evd.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class E0eConfig:
    """Pilot axes and everything held constant across cells.

    No protocol field has a default: they are exactly the choices the pilot must inform.
    """

    # --- axes (their Cartesian product is the pilot matrix) ---
    sizes: tuple[int, ...]
    variants: tuple[str, ...]
    target_fps: tuple[float, ...]
    seeds: tuple[int, ...]
    """Replicas. Each seed drives both support subsampling and initialization, and
    is the same in every cell, which makes comparisons paired."""

    # --- extraction (identical across cells except for the axes) ---
    videos_from: Path
    inventory_csv: Path
    dataset_root: Path
    features_root: Path
    layer: int
    hop_s: float
    num_frames: int
    pixel_dtype: str
    storage_dtype: str
    precision: str
    batch_size: int

    # --- probe (identical across cells: the pilot budget parity) ---
    train_split: Path
    test_split: Path
    class_index: Path
    classes_file: Path
    #: Keep the probe features in RAM instead of reading them at every epoch (time only).
    precarica_feature: bool = False
    #: Per-window labels (``window_id`` -> class), written by E0f. ``None`` on UCF101, where
    #: each window inherits its video class. The map does not depend on ``size`` or
    #: ``variant``, so the same map serves every cell and keeps the comparison paired.
    window_labels: Path | None = None

    checkpoint: Path | None = None
    repo: Path | None = None
    arch: str = "vit_large"
    max_gb: float = 500.0
    max_steps: int = 2000
    probe_batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    #: Operational checkpoint frequency of the probe; does not change the budget.
    checkpoint_every: int = 0
    reference_top1: float = 0.90
    reference_source: str = "sanity threshold declared in E0c, not a published ballpark"
    ballpark_delta_pp: float = 10.0
    #: Unit of the data budget.
    #:
    #: ``"video"``    the ``video_per_classe × finestre_per_video`` rectangle; required when
    #:                cells have different spans (the same video offers a different number
    #:                of windows).
    #: ``"finestre"`` a number of windows per class, spread over all videos of the class;
    #:                usable only when the span is fixed across cells.
    unita_budget: str = "video"
    #: Probe data budget in videos per class (``unita_budget="video"``); ``None`` = all
    #: videos. Fixed by :func:`calibrate`.
    train_video_per_classe: int | None = None
    #: Probe data budget in windows per class (``unita_budget="finestre"``); a separate field
    #: so that a field never means two things depending on another one.
    train_finestre_per_classe: int | None = None
    finestre_per_video: int = 2
    #: Top-1 band the calibration aims at for the reference cell.
    banda_calibrazione: tuple[float, float] = (0.70, 0.90)
    #: What to do if no budget of the grid reaches the band: ``False`` raises; ``True``
    #: chooses anyway with the declared rule and records ``banda_raggiunta: false`` in
    #: ``calibrazione.json``.
    ripiego_fuori_banda: bool = False
    #: Cell used for calibration: the closest to the official recipe.
    cella_di_riferimento: str = "s256-eval_center-fps7p5"
    #: Epochs used by the calibration for every budget, so that a budget sweep does not vary
    #: data and epochs together.
    epoche_calibrazione: int = 60
    #: Paired comparisons per axis (4 contrasts × replicas) and win threshold (16 of 20,
    #: p ≈ 0.006 under the null hypothesis).
    vittorie_per_concludere: int = 16
    #: Fallback level for each axis that does not conclude.
    ripiego: tuple[int, str, float] = (256, "eval_center", 7.5)

    #: Axes in the plan closed without measurement, with the reason: ``{field: reason}``.
    #: Different from "does not conclude" and from "fixed a priori".
    assi_chiusi_senza_misura: dict[str, str] = field(default_factory=dict)

    device: str = "cuda"
    run_id: str = "e0e_ucf101mini_vjepa21l_frozen-attentive_kna_s1"
    #: Delete the cell store after the probes.
    rimuovi_store: bool = False

    def __post_init__(self) -> None:
        if self.arch not in ARCHS:
            raise ValueError(f"unknown arch {self.arch!r}; allowed: {sorted(ARCHS)}")
        if not self.sizes or not self.variants or not self.target_fps:
            raise ValueError("every pilot axis needs at least one value")
        if len(set(self.seeds)) != len(self.seeds) or len(self.seeds) < 2:
            raise ValueError(
                f"at least two distinct seeds are needed to measure the noise floor, "
                f"not {self.seeds}: with one seed the §4.4 criterion does not apply "
                "and the winning cell would be chosen on noise"
            )
        if self.storage_dtype not in {"fp16", "fp32"}:
            raise ValueError(f"storage_dtype {self.storage_dtype!r}: ammessi fp16|fp32")
        if self.hop_s <= 0 or self.num_frames <= 0:
            raise ValueError("hop_s and num_frames must be positive")
        if self.checkpoint_every < 0:
            raise ValueError("checkpoint_every must be >= 0")
        if self.unita_budget not in {"video", "finestre"}:
            raise ValueError(f"unita_budget={self.unita_budget!r}: allowed 'video' | 'finestre'")
        if self.unita_budget == "finestre" and len(self.target_fps) > 1:
            raise ValueError(
                "unita_budget='finestre' with several spans in play: a window budget is "
                "correct only if cells see the same windows, which different spans do "
                "not. Use 'video', the rectangle that exists for that case"
            )
        levels = {
            "size": self.sizes,
            "variant": self.variants,
            "target_fps": self.target_fps,
        }
        fallbacks = dict(zip(("size", "variant", "target_fps"), self.ripiego, strict=True))
        for axis_field, reason in self.assi_chiusi_senza_misura.items():
            if axis_field not in levels:
                raise ValueError(
                    f"unknown axis closed without measurement {axis_field!r}; allowed: {sorted(levels)}"
                )
            if not reason.strip():
                raise ValueError(
                    f"axis {axis_field!r} closed without measurement but without a reason: "
                    "the reason is what distinguishes this label from a shortcut"
                )
            if len(levels[axis_field]) != 1:
                raise ValueError(
                    f"axis {axis_field!r} declared closed without measurement but the plan "
                    f"puts {len(levels[axis_field])} levels in play ({levels[axis_field]}): "
                    "the declaration contradicts the plan"
                )
            if levels[axis_field][0] != fallbacks[axis_field]:
                # Closing an axis for cost while keeping a non-fallback level would adopt an untested
                # level.
                raise ValueError(
                    f"axis {axis_field!r} closed without measurement on level "
                    f"{levels[axis_field][0]!r}, which is not the declared fallback "
                    f"({fallbacks[axis_field]!r}): an axis closed without measurement can "
                    "only fall back, otherwise it is an untested choice"
                )


def arm_id(size: int, variant: str, fps: float) -> str:
    """:return: the identity of a cell, also its file name."""
    return f"s{size}-{variant}-fps{fps:g}".replace(".", "p")


def preprocess_of(config: E0eConfig, arm: dict[str, Any]) -> PreprocessSpec:
    """:return: the complete ``PreprocessSpec`` of the cell, from which the cache key derives."""
    return vjepa21_preprocess_spec(
        arm["variant"],
        fps=arm["target_fps"],
        num_frames=config.num_frames,
        size=arm["size"],
        dtype=config.pixel_dtype,
    )


def arms(config: E0eConfig) -> list[dict[str, Any]]:
    """Return the Cartesian product of the axes, in stable order.

    The order binds the job-array index to the cell, so it must not change after a partial
    submission. ``tokens_per_window`` comes from
    :func:`evd.data.feature_extraction.tokens_per_window`.

    :param config: the pilot configuration.
    :return: one row per cell, with its identifying fields plus tokens per window.
    """
    rows: list[dict[str, Any]] = []
    for size in config.sizes:
        for variant in config.variants:
            for fps in config.target_fps:
                arm = {
                    "arm_id": arm_id(size, variant, fps),
                    "size": int(size),
                    "variant": variant,
                    "target_fps": float(fps),
                }
                arm["tokens_per_window"] = tokens_per_window(preprocess_of(config, arm))
                rows.append(arm)
    return rows


def _video_id_of(row: str) -> str | None:
    """:return: the ``video_id`` of a UCF101 split row, or ``None`` if empty."""
    fields = row.strip().replace("\\", "/").split()
    if not fields:
        return None
    return PurePosixPath(fields[0]).stem


def allowed_videos(config: E0eConfig) -> dict[str, Any]:
    """Return the videos that host a complete window in every cell of the plan.

    The constraint comes from the cell with the longest span (lowest ``target_fps``); a
    video shorter than the window produces no window (no padding).

    :param config: the pilot configuration.
    :return: the admitted set and the counts to report next to the result.
    """
    binding_fps = min(config.target_fps)
    allowed: set[str] = set()
    excluded: list[str] = []
    for record in read_records(config.inventory_csv):
        meta = record.to_meta()
        sampler = CausalWindowSampler(
            num_frames=config.num_frames,
            target_fps=binding_fps,
            hop_s=config.hop_s,
        )
        # One window is not enough: the budget takes `finestre_per_video` windows from every
        # chosen video.
        if len(sampler.windows_for_video(meta)) >= config.finestre_per_video:
            allowed.add(record.video_id)
        else:
            excluded.append(record.video_id)
    return {
        "fps_vincolante": binding_fps,
        "finestre_richieste": config.finestre_per_video,
        "n_ammessi": len(allowed),
        "n_esclusi": len(excluded),
        "esclusi": sorted(excluded),
        "ammessi": allowed,
    }


def prepare_lists(config: E0eConfig) -> dict[str, Path]:
    """Filter lists and splits on the admitted set, once for all cells.

    Writes are atomic and deterministic, so the array tasks can call it in parallel.

    :param config: the pilot configuration.
    :return: the filtered paths, in the order ``videos_from``, ``train``, ``test``.
    :raises ValueError: if the filter empties one of the splits.
    """
    outcome = allowed_videos(config)
    allowed: set[str] = outcome["ammessi"]
    folder = LocalTracker().runs_dir / config.run_id / "liste"
    result: dict[str, Path] = {}
    for name, source in (
        ("videos_from", config.videos_from),
        ("train", config.train_split),
        ("test", config.test_split),
    ):
        rows = [
            row
            for row in Path(source).read_text(encoding="utf-8-sig").splitlines()
            if (vid := _video_id_of(row)) is not None and vid in allowed
        ]
        if not rows:
            raise ValueError(
                f"the filter on the longest span empties {source}: with "
                f"target_fps={outcome['fps_vincolante']} no video hosts a window"
            )
        result[name] = write_text_atomic(folder / f"{name}.txt", "\n".join(rows) + "\n")
    write_json(
        folder / "ammissibilita.json",
        {key: value for key, value in outcome.items() if key != "ammessi"},
    )
    log.info(
        "E0e: %d videos admitted, %d excluded as shorter than the window at fps %g",
        outcome["n_ammessi"],
        outcome["n_esclusi"],
        outcome["fps_vincolante"],
    )
    return result


def _extract_config(
    config: E0eConfig, arm: dict[str, Any], lists: dict[str, Path]
) -> ExtractConfig:
    return ExtractConfig(
        videos_from=lists["videos_from"],
        inventory_csv=config.inventory_csv,
        dataset_root=config.dataset_root,
        features_root=config.features_root,
        arch=config.arch,
        checkpoint=config.checkpoint,
        repo=config.repo,
        size=arm["size"],
        num_frames=config.num_frames,
        preprocess_variant=arm["variant"],
        preprocess_fps=arm["target_fps"],
        preprocess_dtype=config.pixel_dtype,
        storage_dtype=config.storage_dtype,
        precision=config.precision,
        layers=(config.layer,),
        hop_s=config.hop_s,
        batch_size=config.batch_size,
        shard_index=0,
        num_shards=1,
        max_gb=config.max_gb,
        device=config.device,
        dry_run=False,
    )


def estimate_cell(config: E0eConfig, arm: dict[str, Any]) -> dict[str, Any]:
    """Estimate windows and GB of a cell without touching the GPU or the weights.

    Sizes the job before requesting it, with the same ``estimate_extraction`` that guards
    the real extraction with ``--max-gb``.

    :param config: the pilot configuration.
    :param arm: the row produced by :func:`arms`.
    :return: the cell row with ``n_windows``, ``gb_stimati`` and the raw estimate.
    """
    lists = prepare_lists(config)
    outcome = run_extract_job(replace(_extract_config(config, arm, lists), dry_run=True))
    estimate = outcome["estimate"]
    return {
        **arm,
        "n_videos": estimate["n_videos"],
        "n_windows": estimate["n_windows"],
        "gb_stimati": round(estimate["estimated_gb"], 2),
        "estimate": estimate,
    }


def _complete_cell_cleanup(
    config: E0eConfig,
    risultato: Mapping[str, Any],
    arm_id: str,
) -> None:
    """Complete the repeatable deletions of an already recorded cell.

    The final JSON is written before the store is removed: if the wall time interrupts
    ``rmtree``, a relaunch retries it.

    :param config: cell configuration.
    :param risultato: final JSON, already persisted.
    :param arm_id: cell identity, used for the replica checkpoints.
    """
    if config.rimuovi_store and risultato.get("store_dir"):
        store_dir = Path(str(risultato["store_dir"]))
        if store_dir.exists():
            root = config.features_root.resolve()
            target = store_dir.resolve()
            if target == root or not target.is_relative_to(root):
                raise RuntimeError(
                    f"refusing to delete a store outside features_root: "
                    f"{target} is not under {root}"
                )
            shutil.rmtree(store_dir)
            if store_dir.exists():
                raise RuntimeError(f"incomplete removal of store {store_dir}")
            log.info("store of cell %s removed on request", arm_id)

    checkpoint_dir = LocalTracker().runs_dir / config.run_id / "checkpoint" / arm_id
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        if checkpoint_dir.exists():
            raise RuntimeError(f"incomplete removal of checkpoints {checkpoint_dir}")


def run_arm(config: E0eConfig, index: int) -> dict[str, Any]:
    """Run a cell: extraction, merge, one probe per seed, and record it.

    The probes share the same features, so their spread measures training noise, not
    geometry. An already completed cell returns its result immediately (the store may have
    been deleted by ``rimuovi_store``), which makes job chains idempotent.

    :param config: the pilot configuration.
    :param index: cell index in :func:`arms`, i.e. ``SLURM_ARRAY_TASK_ID``.
    :return: the cell result, already written to disk.
    :raises IndexError: if the index matches no cell.
    """
    plan = arms(config)
    if not 0 <= index < len(plan):
        raise IndexError(
            f"cell {index} outside the plan: the pilot has {len(plan)} (array 0-{len(plan) - 1})"
        )
    arm = plan[index]
    concluded = LocalTracker().runs_dir / config.run_id / "celle" / f"{arm['arm_id']}.json"
    if concluded.exists():
        log.info(
            "cell %s already completed in %s: nothing is re-extracted",
            arm["arm_id"],
            concluded,
        )
        result = read_json(concluded)
        _complete_cell_cleanup(config, result, arm["arm_id"])
        return result
    preprocess = preprocess_of(config, arm)
    log.info(
        "E0e cell %d/%d — %s (preprocess_hash=%s)",
        index,
        len(plan),
        arm["arm_id"],
        preprocess.hash()[:12],
    )

    lists = prepare_lists(config)
    extraction = run_extract_job(_extract_config(config, arm, lists))
    store_dir = Path(extraction["store_dir"])
    merge = merge_feature_shards(
        store_dir=store_dir,
        num_shards=1,
        run_id=config.run_id,
        expected_count=None,
    )

    cell_budget = (
        config.train_finestre_per_classe
        if config.unita_budget == "finestre"
        else config.train_video_per_classe
    )
    if cell_budget is None:
        steps = config.max_steps
    else:
        # Same regime as the calibration: steps derived from `epoche_calibrazione` epochs.
        steps = steps_per_epochs(
            config,
            cell_budget,
            len(read_selected_classes(config.class_index, config.classes_file)),
        )
        if steps != config.max_steps:
            log.info(
                "cell: %d steps derived from %d epochs on budget %d (max_steps=%d ignored)",
                steps,
                config.epoche_calibrazione,
                cell_budget,
                config.max_steps,
            )

    per_seed: list[dict[str, Any]] = []
    seeds_dir = LocalTracker().runs_dir / config.run_id / "semi"
    for seed in config.seeds:
        label = f"{arm['arm_id']}-s{seed}"
        already_done = seeds_dir / f"{label}.json"
        # A completed probe is not redone: the artifact is the source of truth, not the queue
        # state.
        if already_done.exists():
            log.info("cell %s, seed %d: already measured, reusing", arm["arm_id"], seed)
            per_seed.append(read_json(already_done))
            continue
        metrics = run_e0c(
            E0cConfig(
                store_dir=store_dir,
                inventory_csv=config.inventory_csv,
                train_split=lists["train"],
                test_split=lists["test"],
                class_index=config.class_index,
                classes_file=config.classes_file,
                layer=config.layer,
                hop_s=config.hop_s,
                max_steps=steps,
                batch_size=config.probe_batch_size,
                lr=config.lr,
                weight_decay=config.weight_decay,
                device=config.device,
                seed=seed,
                reference_top1=config.reference_top1,
                reference_source=config.reference_source,
                ballpark_delta_pp=config.ballpark_delta_pp,
                run_id=config.run_id,
                window_labels=config.window_labels,
                precarica_feature=config.precarica_feature,
                predictions_label=label,
                train_video_per_classe=config.train_video_per_classe,
                train_finestre_per_classe=config.train_finestre_per_classe,
                finestre_per_video=(
                    config.finestre_per_video if config.train_video_per_classe is not None else None
                ),
                # Same subsampling seed in every cell: this is the pairing.
                subset_seed=(
                    seed
                    if config.train_video_per_classe is not None
                    or config.train_finestre_per_classe is not None
                    else None
                ),
                checkpoint_dir=(
                    LocalTracker().runs_dir
                    / config.run_id
                    / "checkpoint"
                    / arm["arm_id"]
                    / f"s{seed}"
                ),
                checkpoint_every=config.checkpoint_every,
            )
        )
        row = {"seed": seed, **metrics}
        write_json(already_done, row)
        per_seed.append(row)

    top1 = [row["top1"] for row in per_seed]
    result = {
        **arm,
        "preprocess_hash": preprocess.hash(),
        "store_dir": str(store_dir),
        "n_windows": extraction["result"]["n_windows"],
        "n_features": merge["n_features"],
        "max_steps": steps,
        "epoche": config.epoche_calibrazione if cell_budget is not None else None,
        "top1_medio": sum(top1) / len(top1),
        "top1_min": min(top1),
        "top1_max": max(top1),
        "escursione_pp": 100 * (max(top1) - min(top1)),
        "per_seme": per_seed,
    }
    destination = LocalTracker().runs_dir / config.run_id / "celle" / f"{arm['arm_id']}.json"
    write_json(destination, result)
    _complete_cell_cleanup(config, result, arm["arm_id"])
    return result


def check_budget(
    config: E0eConfig, lists: dict[str, Path], budget: tuple[int, ...]
) -> dict[str, Any]:
    """Check that the requested budgets exist in the split, before the GPU starts.

    :param config: the pilot configuration.
    :param liste: the filtered lists produced by :func:`prepare_lists`.
    :param budget: the ``train_video_per_classe`` values to try.
    :return: how many videos per class the split offers and the minimum.
    :raises ValueError: if a budget exceeds the poorest training class.
    """
    classes = read_selected_classes(config.class_index, config.classes_file)
    if config.unita_budget == "finestre":
        return _check_window_budget(config, lists, budget, classes)
    per_class = Counter(video.class_name for video in read_labeled_split(lists["train"], classes))
    minimum = min(per_class.values())
    too_large = [value for value in budget if value > minimum]
    if too_large:
        raise ValueError(
            f"budgets {too_large} not feasible: the poorest training class "
            f"has {minimum} ({min(per_class, key=lambda c: per_class[c])}). "
            "Reduce the budget grid, or rebuild the split with more training videos "
            "per class, without going below the cap that keeps classes comparable"
        )
    return {"video_per_classe": dict(sorted(per_class.items())), "minimo": minimum}


def _check_window_budget(
    config: E0eConfig,
    lists: dict[str, Path],
    budget: tuple[int, ...],
    classes: dict[str, int],
) -> dict[str, Any]:
    """Same as :func:`check_budget`, but counting windows instead of videos.

    Reads the label map restricted to training videos, the only source that knows how many
    windows of each class exist.
    """
    if config.window_labels is None:
        raise ValueError(
            "unita_budget='finestre' without window_labels: without the map the number of "
            "windows per class is unknown and the budget cannot be checked"
        )
    id_train = {video.video_id for video in read_labeled_split(lists["train"], classes)}
    labels = read_json(config.window_labels)["etichette"]
    per_class: Counter[str] = Counter()
    for window_id, name in labels.items():
        if window_id.split("@")[0] in id_train:
            per_class[name] += 1
    minimum = min(per_class.values()) if per_class else 0
    too_large = [value for value in budget if value > minimum]
    if too_large:
        raise ValueError(
            f"budgets {too_large} exceed the available windows: the poorest class "
            f"has {minimum} ({min(per_class, key=lambda c: per_class[c])}). "
            "Lower the grid instead of loosening the balance: classes of different "
            "sizes would be a different task from the declared one"
        )
    return {
        "unita": "finestre",
        "finestre_per_classe": dict(sorted(per_class.items())),
        "minimo": minimum,
        "budget": list(budget),
    }


def steps_per_epochs(config: E0eConfig, budget: int, n_classes: int) -> int:
    """Return how many steps train ``budget`` for ``epoche_calibrazione`` epochs.

    Shared by the calibration and the cells, so both train in the same regime.

    :param config: the pilot configuration.
    :param budget: the value in the unit of ``config.unita_budget``.
    :param n_classi: number of classes in the split.
    :return: ``max_steps`` for the probe, at least 1.
    """
    per_class = budget * (1 if config.unita_budget == "finestre" else config.finestre_per_video)
    n_train = per_class * n_classes
    return max(1, ceil(config.epoche_calibrazione * n_train / config.probe_batch_size))


def choose_budget(
    measures: list[dict[str, Any]],
    *,
    band: tuple[float, float],
    ripiego_fuori_banda: bool,
) -> dict[str, Any]:
    """Return the budget the cells will use, with its motivation.

    Receives only the reference-cell measurements, so it sees no comparison between
    geometries.

    :param misure: one row per budget, with ``budget`` and ``top1``.
    :param banda: top-1 band declared before the calibration.
    :param ripiego_fuori_banda: if no budget reaches the band, choose anyway (rule below)
        instead of raising.
    :return: ``scelto``, ``banda_raggiunta`` and ``motivazione_scelta``.
    :raises ValueError: if ``misure`` is empty, or no budget reaches the band and the
        fallback was not requested.
    """
    if not measures:
        raise ValueError("no calibration measurement to choose from")
    bottom, top = band
    for measure in measures:
        measure["in_banda"] = bottom <= measure["top1"] <= top
    inside = [m for m in measures if m["in_banda"]]
    if inside:
        # Among those in the band, the closest to the centre: room for a difference in both
        # directions.
        center = (bottom + top) / 2
        return {
            "scelto": min(inside, key=lambda m: abs(m["top1"] - center)),
            "banda_raggiunta": True,
            "motivazione_scelta": f"in band, closest to the centre {center:.2f}",
        }
    if not ripiego_fuori_banda:
        raise ValueError(
            f"no budget brings the reference cell into the band {band}: "
            f"{[(m['budget'], round(m['top1'], 4)) for m in measures]}. "
            "Widen the budget grid, not the band: the band is the power requirement"
        )
    # Declared fallback: the highest top-1 (the band is missed from below, so the risk is the
    # floor and more data means less training noise).
    chosen = max(measures, key=lambda m: m["top1"])
    increasing = all(
        a["top1"] <= b["top1"]
        for a, b in zip(
            sorted(measures, key=lambda m: m["budget"]),
            sorted(measures, key=lambda m: m["budget"])[1:],
            strict=False,
        )
    )
    motivation = (
        f"band {band} not reached by any budget: applying the declared fallback, "
        f"i.e. the highest top-1 ({chosen['top1']:.4f} at budget "
        f"{chosen['budget']}). The curve "
        + ("is monotonically increasing" if increasing else "is not monotonic")
        + ", so the task "
        + (
            "is not saturated and the risk is the floor, not the ceiling"
            if increasing
            else "must be inspected before trusting it"
        )
    )
    log.warning("calibration: %s", motivation)
    return {
        # `banda_raggiunta` is recorded explicitly: `scelto` alone would hide that the declared
        # criterion was not met.
        "scelto": chosen,
        "banda_raggiunta": False,
        "motivazione_scelta": motivation,
    }


def calibrate(config: E0eConfig, budget: tuple[int, ...]) -> dict[str, Any]:
    """Search the data budget that brings the reference cell into the band.

    Removes the ceiling (with all videos the probe reaches 99.7%). Calibration uses one cell
    only, the closest to the official recipe, before looking at the others.

    :param config: the pilot configuration.
    :param budget: the ``train_video_per_classe`` values to try, increasing.
    :return: the measurement of every budget and the chosen one.
    :raises ValueError: if no budget falls in the declared band.
    """
    plan = {arm["arm_id"]: arm for arm in arms(config)}
    try:
        arm = plan[config.cella_di_riferimento]
    except KeyError:
        raise ValueError(
            f"reference cell {config.cella_di_riferimento!r} is not in the plan: {sorted(plan)}"
        ) from None
    lists = prepare_lists(config)
    # Called only to stop on an infeasible budget before the GPU starts.
    check_budget(config, lists, budget)
    # The number of classes comes from its source, not from the check report.
    n_classes = len(read_selected_classes(config.class_index, config.classes_file))
    store_dir = Path(run_extract_job(_extract_config(config, arm, lists))["store_dir"])
    # The extraction writes `index-<shard>.sqlite` while the reader looks for `index.sqlite`:
    # without the merge the store is unreadable (job 534052).
    merge_feature_shards(
        store_dir=store_dir,
        num_shards=1,
        run_id=config.run_id,
        expected_count=None,
    )
    partial = LocalTracker().runs_dir / config.run_id / "calibrazione-parziale.json"
    measures: list[dict[str, Any]] = []
    # Resume reuses only measurements paid in the same unit.
    already_done = {
        int(m["budget"]): m
        for m in (read_json(partial)["misure"] if partial.exists() else [])
        if m.get("unita") == config.unita_budget
    }
    for value in budget:
        # A paid measurement is neither redone nor lost: `calibrazione.json` is written only at
        # the end. Epochs are constant across budgets.
        steps = steps_per_epochs(config, value, n_classes)
        previous = already_done.get(value)
        # A measurement is reused only if paid in the same regime.
        if previous is not None and previous.get("max_steps") == steps:
            measures.append(previous)
            log.info(
                "calibration: %d %s/class already measured at %d steps, skipping",
                value,
                config.unita_budget,
                steps,
            )
            continue
        metrics = run_e0c(
            E0cConfig(
                store_dir=store_dir,
                inventory_csv=config.inventory_csv,
                train_split=lists["train"],
                test_split=lists["test"],
                class_index=config.class_index,
                classes_file=config.classes_file,
                layer=config.layer,
                hop_s=config.hop_s,
                max_steps=steps,
                batch_size=config.probe_batch_size,
                lr=config.lr,
                weight_decay=config.weight_decay,
                device=config.device,
                seed=config.seeds[0],
                reference_top1=config.reference_top1,
                reference_source=config.reference_source,
                ballpark_delta_pp=config.ballpark_delta_pp,
                run_id=config.run_id,
                window_labels=config.window_labels,
                precarica_feature=config.precarica_feature,
                predictions_label=f"calibra-v{value}",
                train_video_per_classe=(None if config.unita_budget == "finestre" else value),
                train_finestre_per_classe=(value if config.unita_budget == "finestre" else None),
                finestre_per_video=(
                    None if config.unita_budget == "finestre" else config.finestre_per_video
                ),
                subset_seed=config.seeds[0],
                checkpoint_dir=(
                    LocalTracker().runs_dir / config.run_id / "checkpoint" / "calibra" / f"v{value}"
                ),
                checkpoint_every=config.checkpoint_every,
            )
        )
        measures.append(
            {
                # `budget` and `unita` together: the number alone cannot be read.
                "budget": value,
                "unita": config.unita_budget,
                "top1": metrics["top1"],
                "n_train_windows": metrics["n_train_windows"],
                "max_steps": steps,
                "epoche": config.epoche_calibrazione,
            }
        )
        write_json(partial, {"misure": measures})
        log.info(
            "calibration: %d %s/class -> top1 %.4f",
            value,
            config.unita_budget,
            metrics["top1"],
        )
    choice = choose_budget(
        measures,
        band=config.banda_calibrazione,
        ripiego_fuori_banda=config.ripiego_fuori_banda,
    )
    outcome = {
        "cella_di_riferimento": config.cella_di_riferimento,
        "banda": list(config.banda_calibrazione),
        "escursione_utile_pp": round(100 * (max(m["top1"] for m in measures) - 1.0 / n_classes), 2),
        "caso": round(1.0 / n_classes, 4),
        "misure": measures,
        **choice,
    }
    write_json(LocalTracker().runs_dir / config.run_id / "calibrazione.json", outcome)
    return outcome


#: The three factorial axes, with the cell-row key and the two levels.
AXES = (
    ("risoluzione", "size", (256, 384)),
    ("ritaglio", "variant", ("eval_center", "squash")),
    ("span", "target_fps", (7.5, 4.0)),
)


def p_sign_test(wins: int, comparisons: int) -> float:
    """Return the upper tail of the sign test under independence.

    Used to declare what a threshold is worth, not to decide: comparisons here are
    correlated within seed, so the value is an optimistic bound (``docs/12`` Q26).

    :param vittorie: observed successes, or the threshold whose p is wanted.
    :param confronti: total number of comparisons.
    :return: ``P(X >= vittorie)`` with ``X ~ Binomial(confronti, 0.5)``.
    :raises ValueError: if ``confronti`` is negative or ``vittorie`` is outside
        ``[0, confronti]``.
    """
    if comparisons < 0 or not 0 <= wins <= comparisons:
        raise ValueError(f"vittorie={wins} outside [0, {comparisons}]")
    return sum(comb(comparisons, k) for k in range(wins, comparisons + 1)) / 2**comparisons


def _pairings(
    per_cell: Mapping[str, Mapping[int, float]], axis: str
) -> tuple[str, tuple[Any, Any], list[tuple[str, str, str]]]:
    """Return the cell pairs that isolate an axis, with the pairing name.

    A pairing is a combination of the other two axes; combinations missing a cell in
    ``per_cell`` are dropped. The pairing name lists only the fields that distinguish the
    remaining combinations. The three readings of the criterion all go through here.

    :param per_cell: ``{arm_id: {replica: top1}}``.
    :param asse: the varying field (``size``, ``variant``, ``target_fps``).
    :return: the field, its two levels and the list
        ``(pairing_name, arm_at_level_0, arm_at_level_1)``.
    :raises ValueError: if the axis is not one of the three factorial axes.
    """
    try:
        _, field, levels = next(a for a in AXES if a[1] == axis)
    except StopIteration:
        raise ValueError(f"unknown axis: {axis!r}; expected {[a[1] for a in AXES]}") from None

    others = [a for a in AXES if a[1] != field]
    raw: list[tuple[dict[str, Any], str, str]] = []
    for value_a in others[0][2]:
        for value_b in others[1][2]:
            fixed = {others[0][1]: value_a, others[1][1]: value_b}
            cells = [
                arm_id(
                    int({**fixed, field: level}["size"]),
                    str({**fixed, field: level}["variant"]),
                    float({**fixed, field: level}["target_fps"]),
                )
                for level in levels
            ]
            if cells[0] in per_cell and cells[1] in per_cell:
                raw.append((fixed, cells[0], cells[1]))

    distinctive = [c for c in (others[0][1], others[1][1]) if len({f[c] for f, _, _ in raw}) > 1]
    if not distinctive:
        distinctive = [others[0][1]]
    pairs = [
        ("-".join(_level_label(fixed[fixed_field]) for fixed_field in distinctive), a, b)
        for fixed, a, b in raw
    ]
    return field, levels, pairs


def _level_label(valore: Any) -> str:
    """:return: the level as it appears in cell names (``7.5`` -> ``"7.5"``, not
    ``"7.5000"``).
    """
    return f"{valore:g}" if isinstance(valore, float) else str(valore)


def paired_comparisons(per_cell: dict[str, dict[int, float]], axis: str) -> dict[str, Any]:
    """Return the sign test on an axis: who wins, and how many times out of how many.

    Each comparison is paired on replica, test videos and the other two axes. This is the
    pre-registered reading (``docs/10``, E0f v2) and counts comparisons as independent,
    which they are not within a seed: read it with :func:`within_seed_unanimity` and
    :func:`within_seed_mean`.

    :param per_cell: ``{arm_id: {replica: top1}}``.
    :param asse: the varying field (``size``, ``variant``, ``target_fps``).
    :return: wins per level, ties, total and mean paired difference
        (``level_0 - level_1``, negative when the second level is better).
    :raises ValueError: if the axis is not one of the three factorial axes.
    """
    field, levels, pairs = _pairings(per_cell, axis)
    wins = {str(levels[0]): 0, str(levels[1]): 0}
    even = 0
    differences: list[float] = []
    for _, a, b in pairs:
        for replicate in sorted(set(per_cell[a]) & set(per_cell[b])):
            delta = per_cell[a][replicate] - per_cell[b][replicate]
            differences.append(delta)
            if delta > 0:
                wins[str(levels[0])] += 1
            elif delta < 0:
                wins[str(levels[1])] += 1
            else:
                even += 1
    return {
        "asse": field,
        "vittorie": wins,
        "pari": even,
        "confronti": sum(wins.values()) + even,
        "delta_medio_appaiato": (sum(differences) / len(differences)) if differences else 0.0,
    }


def within_seed_unanimity(per_cell: dict[str, dict[int, float]], axis: str) -> dict[str, Any]:
    """Return the within-seed paired reading: a seed votes only if it is consistent.

    Within a seed the pairings share support, query, initialization and training data, so
    the unit is the seed: it votes only if all its pairings agree, and the axis concludes
    only unanimously. With at least one mixed seed unanimity is unreachable by
    construction; pair it with :func:`within_seed_mean`.

    :param per_cell: ``{arm_id: {replica: top1}}``.
    :param asse: the varying field (``size``, ``variant``, ``target_fps``).
    :return: signs per pairing (``"+"`` = level 0 wins), unanimous seeds per level
        (``semi_pro_livelli0`` and ``semi_pro_livelli1``, names frozen by the 25 August
        artifact), mixed seeds, whether it concludes, and the sign-test tail at the
        unanimity threshold.
    :raises ValueError: if the axis is not one of the three factorial axes.
    """
    _, _, pairs = _pairings(per_cell, axis)
    axis_name = next(name for name, field, _ in AXES if field == axis)
    signs: dict[str, dict[str, str]] = {}
    seeds: set[int] = set()
    for item_name, a, b in pairs:
        common = sorted(set(per_cell[a]) & set(per_cell[b]))
        seeds.update(common)
        signs[item_name] = {
            str(replicate): ("+" if per_cell[a][replicate] - per_cell[b][replicate] > 0 else "-")
            for replicate in common
        }
    counts = {"+": 0, "-": 0}
    mixed: list[int] = []
    for replicate in sorted(seeds):
        votes = {
            signs[item_name][str(replicate)]
            for item_name in signs
            if str(replicate) in signs[item_name]
        }
        if len(votes) == 1:
            counts[votes.pop()] += 1
        else:
            mixed.append(replicate)
    return {
        "nome_asse": axis_name,
        "regola": (
            "a seed votes only if both pairings agree; "
            "conclusion only with 7/7 votes in the same direction"
        ),
        "segni_per_abbinamento": signs,
        "semi_pro_livelli0": counts["+"],
        "semi_pro_livelli1": counts["-"],
        "semi_misti": mixed,
        "concluso": bool(seeds) and max(counts.values()) == len(seeds),
        "p_una_coda_se_indipendenti": p_sign_test(len(seeds), len(seeds)),
    }


def within_seed_mean(per_cell: dict[str, dict[int, float]], axis: str) -> dict[str, Any]:
    """Return the robustness check: average within the seed, then count the seeds.

    Unlike :func:`within_seed_unanimity`, a discordant seed weighs by how discordant it is.
    Not the declared criterion of the pilot: it only shows whether the outcome depends on
    the reading.

    :param per_cell: ``{arm_id: {replica: top1}}``.
    :param asse: the varying field (``size``, ``variant``, ``target_fps``).
    :return: wins per level counted on seeds, mean difference across seeds and sign-test
        tail on the number of seeds.
    :raises ValueError: if the axis is not one of the three factorial axes.
    """
    _, levels, pairs = _pairings(per_cell, axis)
    axis_name = next(name for name, field, _ in AXES if field == axis)
    seeds: set[int] = set()
    for _, a, b in pairs:
        seeds.update(set(per_cell[a]) & set(per_cell[b]))
    wins = {str(levels[0]): 0, str(levels[1]): 0}
    even = 0
    means: list[float] = []
    for replicate in sorted(seeds):
        delta = [
            per_cell[a][replicate] - per_cell[b][replicate]
            for _, a, b in pairs
            if replicate in per_cell[a] and replicate in per_cell[b]
        ]
        mean = sum(delta) / len(delta)
        means.append(mean)
        if mean > 0:
            wins[str(levels[0])] += 1
        elif mean < 0:
            wins[str(levels[1])] += 1
        else:
            even += 1
    return {
        "nome_asse": axis_name,
        "regola": "paired difference averaged within each seed, then sign test on seeds",
        "vittorie": wins,
        "pari": even,
        "confronti": len(seeds),
        "delta_medio_fra_semi": (sum(means) / len(means)) if means else 0.0,
        "p_una_coda_se_indipendenti": (
            p_sign_test(max(wins.values()), len(seeds)) if seeds else 1.0
        ),
    }


def choose(per_cell: dict[str, dict[int, float]], config: E0eConfig) -> dict[str, Any]:
    """Apply the criterion declared in the module docstring, axis by axis.

    :param per_cell: ``{arm_id: {replica: top1}}``.
    :param config: the configuration, providing threshold and fallback levels.
    :return: the chosen configuration and, per axis, whether it concluded and why.
    """
    fallback = dict(zip(("size", "variant", "target_fps"), config.ripiego, strict=True))
    config_levels = {
        "size": config.sizes,
        "variant": config.variants,
        "target_fps": config.target_fps,
    }
    outcomes: list[dict[str, Any]] = []
    choice: dict[str, Any] = {}
    for name, axis_field, _levels in AXES:
        present = config_levels[axis_field]
        if len(present) < 2:
            # A single-level axis is either fixed a priori (never in play) or closed without
            # measurement; neither is "does not conclude".
            reason = config.assi_chiusi_senza_misura.get(axis_field)
            choice[axis_field] = present[0]
            outcomes.append(
                {
                    "asse": axis_field,
                    "nome": name,
                    "vittorie": {},
                    "pari": 0,
                    "confronti": 0,
                    "delta_medio_appaiato": 0.0,
                    "concluso": False,
                    "fissato_a_priori": reason is None,
                    "chiuso_senza_misura": reason is not None,
                    "livello_scelto": present[0],
                    "motivazione": (
                        f"axis removed from the plan without measurement, and the adopted level "
                        f"({present[0]}) is the declared fallback, not an a posteriori "
                        f"choice: {reason}"
                        if reason is not None
                        else (
                            f"axis not in play: the plan declares a single level "
                            f"({present[0]}), so the pilot neither measures nor chooses it"
                        )
                    ),
                }
            )
            continue
        test = paired_comparisons(per_cell, axis_field)
        winner, wins = max(test["vittorie"].items(), key=lambda kv: kv[1])
        concluded = wins >= config.vittorie_per_concludere
        choice[axis_field] = winner if concluded else fallback[axis_field]
        outcomes.append(
            {
                **test,
                "nome": name,
                "concluso": concluded,
                "fissato_a_priori": False,
                "chiuso_senza_misura": False,
                "livello_scelto": choice[axis_field],
                "motivazione": (
                    f"{winner} wins {wins}/{test['confronti']} paired comparisons"
                    if concluded
                    else (
                        f"no level reaches {config.vittorie_per_concludere}"
                        f"/{test['confronti']}: applying the declared fallback"
                    )
                ),
            }
        )
    return {
        "assi": outcomes,
        "configurazione_scelta": arm_id(
            int(choice["size"]), str(choice["variant"]), float(choice["target_fps"])
        ),
        "assi_conclusi": [e["nome"] for e in outcomes if e["concluso"]],
        "assi_non_conclusi": [
            e["nome"]
            for e in outcomes
            if not e["concluso"] and not e["fissato_a_priori"] and not e["chiuso_senza_misura"]
        ],
        "assi_fissati_a_priori": [e["nome"] for e in outcomes if e["fissato_a_priori"]],
        "assi_chiusi_senza_misura": [e["nome"] for e in outcomes if e["chiuso_senza_misura"]],
    }


def aggregate(config: E0eConfig) -> dict[str, Any]:
    """Gather the cells and apply the sign-test criterion.

    :param config: the pilot configuration.
    :return: the full table, the three axis tests and the chosen configuration.
    :raises RuntimeError: if cells are missing with respect to the plan.
    """
    folder = LocalTracker().runs_dir / config.run_id / "celle"
    plan = arms(config)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for arm in plan:
        path = folder / f"{arm['arm_id']}.json"
        if path.exists():
            rows.append(read_json(path))
        else:
            missing.append(arm["arm_id"])
    if missing:
        raise RuntimeError(
            f"E0e pilot incomplete: {len(missing)} of {len(plan)} cells missing "
            f"({missing[:3]}). Aggregating a partial factorial would compare "
            "cells that did not see the same plan."
        )
    per_cell = {
        row["arm_id"]: {int(s["seed"]): float(s["top1"]) for s in row["per_seme"]} for row in rows
    }
    summary = {
        "run_id": config.run_id,
        "n_celle": len(rows),
        "repliche": list(config.seeds),
        "budget_dati": {
            "unita": config.unita_budget,
            "train_video_per_classe": config.train_video_per_classe,
            "train_finestre_per_classe": config.train_finestre_per_classe,
            "finestre_per_video": config.finestre_per_video,
        },
        "pavimento_di_rumore_pp": max(row_data["escursione_pp"] for row_data in rows),
        "celle": sorted(rows, key=lambda r: -r["top1_medio"]),
        "criterio": choose(per_cell, config),
        "caveat": (
            "reduced UCF101: ranks geometries by information surviving the preprocessing "
            "of a 320x240 source, not by anomaly-detection performance. Not a thesis "
            "number."
        ),
    }
    write_json(LocalTracker().runs_dir / config.run_id / "confronto_geometrie.json", summary)
    return summary
