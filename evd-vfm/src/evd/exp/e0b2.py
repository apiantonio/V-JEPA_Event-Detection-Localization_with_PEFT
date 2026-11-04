"""The cost of training an adapter.

Here the full step (forward, backward and update) is
measured for each PEFT family over a batch grid, to size ``peft_budget_a``.

1. The ``Trainer`` itself is measured, not a copy: ``Trainer.fit`` with a loader holding
   a single in-memory batch.
2. Pixels are synthetic and the data pipeline is outside the measurement.

3. The backbone is rebuilt once per geometry: ``unload_()`` restores it bit for bit
   between variants, and nothing is ever merged.

Resumable per cell, with clip seeds derived from the cell
(:func:`~evd.exp.e0.clip_seed`). It measures the cost; the step budget remains a
protocol decision (docs/05 §5).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from evd.eval.efficiency import transformer_forward_flops
from evd.exp.e0 import _diagnostic_spec, clip_seed, load_progress, save_progress
from evd.exp.minifit import BUILDERS, create_strategy
from evd.exp.tracker import LocalTracker
from evd.models.backbones.vjepa21 import VJEPA21Backbone
from evd.models.backbones.vjepa21_source import ARCHS, build_loaded_encoder, checkpoint_sha
from evd.models.heads.attentive import AttentiveProbe
from evd.train.budget import BudgetSpec
from evd.train.trainer import Objective, Trainer
from evd.utils.io import ensure_dir, read_json, write_json
from evd.utils.logging import get_logger
from evd.utils.seeding import seed_all

log = get_logger(__name__)

PROFILE_SCHEMA_VERSION = 1

#: Variants profiled by default: those with a structurally different cost profile (DoRA
#: row norms, prompts lengthening the sequence, APT lengthening keys and values only,
#: DA-VPT adding a loss term).
COST_VARIANTS: tuple[str, ...] = (
    "lora",
    "dora",
    "lora_fused",
    "vpt_deep",
    "vpt_shallow",
    "apt",
    "da_vpt",
)


@dataclass(frozen=True)
class E0b2Config:
    """Profile axes; no field has a protocol default."""

    checkpoint: Path
    repo: Path | None = None
    arch: str = "vit_large"
    sizes: tuple[int, ...] = (224, 384)
    precisions: tuple[str, ...] = ("bf16",)
    tf32_values: tuple[bool, ...] = (True,)
    batches: tuple[int, ...] = (1, 2, 4, 8)
    varianti: tuple[str, ...] = COST_VARIANTS
    num_frames: int = 16
    warmup: int = 2
    """Steps discarded before timing: the first ones include kernel autotuning and Adam
    state allocation."""
    iters: int = 8
    device: str = "cuda"
    layer: int = -1
    lr: float = 1e-3
    r: int = 16
    alpha: float = 32.0
    placement: str = "qkvo"
    n_tokens: int = 8
    n_classes: int = 10
    """Head classes; only the last ``Linear`` changes (less than a thousandth of the
    cost on ViT-L)."""
    objective: Objective = "multiclass"
    """What the timed step optimizes.

    The loss weighs almost nothing on the cost, but the measured code is
    ``Trainer.fit``: ``binary`` is the ``edp`` regime (ADR-031 point 4) sized by E0b.3,
    ``multiclass`` the historical default used for the original E0b.2 run.
    """
    run_id: str = "e0b2_synth_vjepa21l_peft_kna_s0"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.arch not in ARCHS:
            raise ValueError(f"unknown arch {self.arch!r}; allowed: {sorted(ARCHS)}")
        unknown = set(self.varianti) - set(BUILDERS)
        if unknown:
            raise ValueError(f"unknown variants: {sorted(unknown)}")
        if not self.varianti:
            raise ValueError("no variant to profile")
        if tuple(sorted(set(self.batches))) != self.batches or any(b <= 0 for b in self.batches):
            raise ValueError(f"batches must be positive, unique and increasing: {self.batches}")
        if any(p not in {"fp32", "bf16"} for p in self.precisions):
            raise ValueError(f"precisions not allowed: {self.precisions}")
        if self.iters <= 0 or self.warmup < 0:
            raise ValueError(f"iters > 0 and warmup >= 0 required, not {self.iters}/{self.warmup}")
        if self.objective not in ("binary", "multiclass"):
            raise ValueError(f"unknown objective: {self.objective!r}")
        if self.objective == "binary" and self.n_classes != 1:
            # The Trainer would raise anyway on [B, n>1], but only after building the backbone.
            raise ValueError(
                f"objective='binary' requires a one-logit head, not {self.n_classes} "
                f"(ADR-031 point 4: binary output is protocol)"
            )
        if self.objective == "multiclass" and self.n_classes < 2:
            raise ValueError(f"objective='multiclass' with {self.n_classes} classes is meaningless")


def cell_id(cell: dict[str, Any]) -> str:
    """:return: the identity of a cell, also its file name."""
    return (
        f"s{cell['size']}-p{cell['precision']}-t{int(cell['tf32'])}-"
        f"b{cell['batch']}-v{cell['variante']}"
    )


def celle(config: E0b2Config) -> list[dict[str, Any]]:
    """Return the Cartesian product of the axes, in stable order.

    :param config: the configuration.
    :return: one row per cell, with its identifying fields only.
    """
    return [
        {
            "size": int(size),
            "precision": precision,
            "precision_mode": "amp_autocast" if precision == "bf16" else "native",
            "tf32": bool(tf32),
            "batch": int(batch),
            "variante": variant,
            "famiglia": BUILDERS[variant]["famiglia"],
        }
        for size in config.sizes
        for variant in config.varianti
        for precision in config.precisions
        for tf32 in config.tf32_values
        for batch in config.batches
    ]


def _single_batch_loader(
    *,
    batch: int,
    size: int,
    num_frames: int,
    n_classes: int,
    objective: Objective,
    seed: int,
    device: str,
) -> DataLoader[Any]:
    """Return a loader that always yields the same in-memory batch.

    Removes the data pipeline from the measurement without removing the ``Trainer``. Labels
    cycle over the allowed values, and the target key follows the objective as in the real
    loop (``Trainer._target``): ``class_label`` for multiclass, ``label`` for binary.

    :param batch: batch size.
    :param size: frame side.
    :param num_frames: frames per clip.
    :param n_classes: head classes; 1 in binary mode.
    :param objective: ``binary`` or ``multiclass``; decides key and domain of the target.
    :param seed: clip seed, derived from the cell.
    :param device: where to allocate the clips.
    :return: a ``DataLoader`` with a single batch.
    """
    generator = torch.Generator().manual_seed(seed)
    key = "label" if objective == "binary" else "class_label"
    values = 2 if objective == "binary" else n_classes
    rows = [
        {
            "clip": torch.randn(3, num_frames, size, size, generator=generator).to(device),
            key: index % values,
        }
        for index in range(batch)
    ]
    return DataLoader(rows, batch_size=batch)  # type: ignore[arg-type]


def _measure_cell(
    backbone: VJEPA21Backbone,
    strategy: Any,
    config: E0b2Config,
    *,
    batch: int,
    size: int,
    precision: str,
    tf32: bool,
) -> dict[str, Any]:
    """Time ``warmup + iters`` training steps on a cell.

    :param backbone: the already adapted backbone.
    :param strategy: the applied strategy; its parameters go to the optimizer.
    :param config: profile configuration.
    :param batch: batch size.
    :param size: frame side.
    :param precision: ``"fp32"`` or ``"bf16"``.
    :param tf32: whether to enable TF32 on matmuls.
    :return: measurement fields of the row (without identifying fields).
    """
    loader = _single_batch_loader(
        batch=batch,
        size=size,
        num_frames=config.num_frames,
        n_classes=config.n_classes,
        objective=config.objective,
        seed=clip_seed(config.seed, size=size, batch=batch, num_frames=config.num_frames),
        device=config.device,
    )
    head = AttentiveProbe(
        embed_dim=backbone.embed_dim,
        n_classes=config.n_classes,
        n_queries=1,
        n_heads=16,
        depth=1,
        norm=True,
    ).to(config.device)

    def budget(steps: int) -> BudgetSpec:
        return BudgetSpec(
            max_steps=steps,
            batch_size=batch,
            lr_grid=(config.lr,),
            precision=precision,  # type: ignore[arg-type]
            tf32=tf32,
            attn="sdpa",
        )

    trainer = Trainer(device=config.device)

    if config.warmup:
        trainer.fit(
            backbone,
            head,
            loader,
            budget(config.warmup),
            levels=(config.layer,),
            adapt=strategy,
            objective=config.objective,
        )
    if config.device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter_ns()
    fit = trainer.fit(
        backbone,
        head,
        loader,
        budget(config.iters),
        levels=(config.layer,),
        adapt=strategy,
        objective=config.objective,
    )
    if config.device.startswith("cuda"):
        torch.cuda.synchronize()
    seconds = (time.perf_counter_ns() - start) / 1e9
    peak = int(torch.cuda.max_memory_allocated()) if config.device.startswith("cuda") else None

    architecture = ARCHS[config.arch]
    tokens = (config.num_frames // 2) * (size // 16) ** 2
    forward = transformer_forward_flops(
        batch_size=batch,
        tokens=tokens,
        dim=architecture.embed_dim,
        depth=architecture.depth,
        patch_volume=2 * 16 * 16,
    )
    per_step = seconds / config.iters
    return {
        "status": "ok",
        "attn": "sdpa",
        "tokens": tokens,
        "n_trainable_adapter": sum(p.numel() for p in strategy.trainable_parameters()),
        "n_trainable_head": sum(p.numel() for p in head.parameters()),
        "latency_step_ms": per_step * 1000,
        "latency_window_ms": per_step * 1000 / batch,
        "throughput_windows_s": batch / per_step,
        "peak_vram_bytes": peak,
        # Estimate, not measurement: a transformer backward costs about twice the forward, so a
        # step about three times; declared in `flops_source`.
        "effective_tflops_s": 3 * forward / per_step / 1e12,
        "flops_per_step": 3 * forward,
        "flops_source": "vit_formula_matmul_conv x3 (fwd+bwd estimated)",
        "warmup": config.warmup,
        "iters": config.iters,
        "loss_finale": fit.final_loss,
    }


def profile_e0b2(config: E0b2Config) -> dict[str, Any]:
    """Run the profile and write ``runs/{run_id}/efficiency_peft.json``.

    :param config: the configuration.
    :return: the final payload, already written to disk.
    :raises RuntimeError: if the device is not CUDA, or cells remain unmeasured.
    """
    target = torch.device(config.device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "official E0b.2 requires a CUDA GPU; it does not start on the login node"
        )
    if "bf16" in config.precisions and not torch.cuda.is_bf16_supported():
        raise RuntimeError("the GPU does not declare bf16 support")
    # As in E0b.1: kernels are measured, and `use_deterministic_algorithms` changes their
    # selection.
    seed_all(config.seed, deterministic=False)

    architecture = ARCHS[config.arch]
    digest = checkpoint_sha(config.checkpoint)
    tracker = LocalTracker()
    plan = celle(config)
    expected = {cell_id(cell) for cell in plan}
    run_dir = tracker.runs_dir / config.run_id
    progress_dir = run_dir / "progress"
    completed = load_progress(progress_dir, expected, etichetta="E0b.2")

    handle = tracker.start(
        config.run_id,
        asdict(config),
        official=True,
        regime={
            "attn": "sdpa",
            "device": config.device,
            "profile_schema": PROFILE_SCHEMA_VERSION,
            "resumable_cells": True,
        },
    )
    ensure_dir(progress_dir)

    tf32_matmul, tf32_cudnn = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    try:
        for size in config.sizes:
            to_do = [c for c in plan if c["size"] == size and cell_id(c) not in completed]
            if not to_do:
                continue
            encoder, _ = build_loaded_encoder(
                config.arch,
                checkpoint=config.checkpoint,
                key=architecture.ckpt_key,
                img_size=size,
                num_frames=config.num_frames,
                device=config.device,
                repo=config.repo,
            )
            encoder.to(dtype=torch.float32)
            backbone = VJEPA21Backbone(
                encoder,
                arch=architecture,
                ckpt_sha=digest,
                spec=_diagnostic_spec(size=size, num_frames=config.num_frames),
                device=config.device,
            )
            try:
                for variant in config.varianti:
                    _profile_variant(
                        variant,
                        backbone=backbone,
                        config=config,
                        size=size,
                        piano=plan,
                        completed=completed,
                        progress_dir=progress_dir,
                    )
            finally:
                del backbone, encoder
                torch.cuda.empty_cache()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32_matmul
        torch.backends.cudnn.allow_tf32 = tf32_cudnn

    missing = expected.difference(completed)
    if missing:
        raise RuntimeError(f"incomplete E0b.2 profile: {len(missing)} missing cells")
    rows = [completed[cell_id(cell)] for cell in plan]
    payload = {
        "run_id": config.run_id,
        "checkpoint_sha": digest,
        "profile_schema": PROFILE_SCHEMA_VERSION,
        "gpu": torch.cuda.get_device_name(target),
        "arch": config.arch,
        "rows": rows,
    }
    write_json(run_dir / "efficiency_peft.json", payload)
    tracker.finish(
        handle,
        {"n_celle": len(rows), "n_ok": sum(1 for r in rows if r["status"] == "ok")},
        status="ok",
        extra={"exp": "e0b", "part": "e0b.2"},
    )
    return payload


def _profile_variant(
    variante: str,
    *,
    backbone: VJEPA21Backbone,
    config: E0b2Config,
    size: int,
    piano: Sequence[dict[str, Any]],
    completed: dict[str, dict[str, Any]],
    progress_dir: Path,
) -> None:
    """Apply a variant, measure all its cells at this geometry, then unload it.

    ``unload_()`` in ``finally`` prevents the next variant from measuring two overlapping
    adaptations.
    """
    to_do = [
        c
        for c in piano
        if c["size"] == size and c["variante"] == variante and cell_id(c) not in completed
    ]
    if not to_do:
        return
    strategy = create_strategy(
        variante,
        _minifit_shadow_config(config, size=size),
        n_classes=config.n_classes,
        n_blocks=len(backbone.blocks()),
    )
    strategy.apply(backbone)
    try:
        oom_from = None
        for cell in to_do:
            if oom_from is not None and cell["batch"] >= oom_from:
                save_progress(
                    progress_dir,
                    completed,
                    {**cell, "status": "skipped_after_oom", "oom_batch": oom_from},
                    cell_id=cell_id(cell),
                )
                continue
            torch.backends.cuda.matmul.allow_tf32 = cell["tf32"]
            torch.backends.cudnn.allow_tf32 = cell["tf32"]
            try:
                measure = _measure_cell(
                    backbone,
                    strategy,
                    config,
                    batch=cell["batch"],
                    size=size,
                    precision=cell["precision"],
                    tf32=cell["tf32"],
                )
            except torch.OutOfMemoryError:
                oom_from = cell["batch"]
                measure = {"status": "oom", "attn": "sdpa"}
                torch.cuda.empty_cache()
            save_progress(progress_dir, completed, {**cell, **measure}, cell_id=cell_id(cell))
            log.info(
                "%s s=%d b=%d %s tf32=%s -> %s",
                variante,
                size,
                cell["batch"],
                cell["precision"],
                cell["tf32"],
                measure.get("status"),
            )
    finally:
        strategy.unload_()


def _minifit_shadow_config(config: E0b2Config, *, size: int) -> Any:
    """Adapt ``E0b2Config`` to the signature expected by :func:`create_strategy`.

    Avoids duplicating :data:`~evd.exp.minifit.BUILDERS`, the only place where a variant
    name becomes a spec.

    :param config: E0b.2 configuration.
    :param size: current geometry; metadata only.
    :return: an object with the fields read by ``create_strategy``.
    """
    from evd.exp.minifit import MiniFitConfig

    empty = Path(".")
    return MiniFitConfig(
        checkpoint=config.checkpoint,
        inventory_csv=empty,
        dataset_root=empty,
        split=empty,
        class_index=empty,
        classes_file=empty,
        arch=config.arch,
        size=size,
        num_frames=config.num_frames,
        r=config.r,
        alpha=config.alpha,
        placement=config.placement,
        n_tokens=config.n_tokens,
        seed=config.seed,
        device=config.device,
    )


def reload_profile(run_dir: Path) -> dict[str, Any]:
    """:return: the payload of an already executed E0b.2 profile.
    :raises FileNotFoundError: if the run does not contain ``efficiency_peft.json``.
    """
    return dict(read_json(run_dir / "efficiency_peft.json"))
