"""E0a (real contract) and E0b.1 (frozen forward budget)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch

from evd.data.decoder import make_decoder
from evd.data.feature_extraction import select_inventory_records
from evd.data.feature_store import sizeof_estimate
from evd.data.preprocess import preprocess_clip, vjepa21_preprocess_spec
from evd.data.windows import CausalWindowSampler, stride_frames_for
from evd.eval.efficiency import (
    count_parameters,
    measure,
    transformer_forward_flops,
)
from evd.exp.tracker import LocalTracker, git_info
from evd.models.backbones.vjepa21 import VJEPA21Backbone
from evd.models.backbones.vjepa21_source import (
    ARCHS,
    architecture_report,
    build_encoder,
    build_loaded_encoder,
    checkpoint_sha,
)
from evd.types import PreprocessSpec, WindowSpec
from evd.utils.hashing import stable_hash
from evd.utils.io import ensure_dir, read_json, read_jsonl, write_json, write_text_atomic
from evd.utils.seeding import seed_all


def _diagnostic_spec(*, size: int, num_frames: int) -> PreprocessSpec:
    """Return a spec for synthetic tensors; it does not choose protocol geometry or fps."""
    return PreprocessSpec(
        fps=1.0,
        num_frames=num_frames,
        size=size,
        short_side=size,
        geometry="none",
        backend="none",
        backend_version="none",
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        tokenizer="tubelet_2x16x16",
        dtype="fp32",
    )


def inspect_e0a(
    *,
    arch: str,
    size: int,
    num_frames: int,
    device: str,
    repo: Path | None,
    checkpoint: Path | None,
    seed: int,
    video: Path | None = None,
    preprocess_variant: str | None = None,
    preprocess_fps: float | None = None,
) -> dict[str, Any]:
    """Build the architecture, load it strictly if requested and run a forward.

    ``seed`` comes from ``EVD_SEED``: without a checkpoint the encoder is randomly
    initialized, so the diagnostic forward needs it to be repeatable.

    :raises ValueError: if a video forward lacks preprocessing parameters or the video is
        too short.
    """
    seed_all(seed)
    architecture = ARCHS[arch]
    load_report = None
    digest = "not-loaded"
    if checkpoint is not None:
        digest = checkpoint_sha(checkpoint)
        encoder, load_report = build_loaded_encoder(
            arch,
            checkpoint=checkpoint,
            key=architecture.ckpt_key,
            img_size=size,
            num_frames=num_frames,
            device=device,
            repo=repo,
        )
    else:
        encoder = build_encoder(
            arch,
            img_size=size,
            num_frames=num_frames,
            device=device,
            repo=repo,
        )
    backbone = VJEPA21Backbone(
        encoder,
        arch=architecture,
        ckpt_sha=digest,
        spec=_diagnostic_spec(size=size, num_frames=num_frames),
        device=device,
        load_report=load_report,
    )
    placements: dict[str, dict[str, int]] = {}
    for placement in ("qkvo", "attn_mlp", "qk_only", "last_n_blocks:6"):
        refs = backbone.target_map().resolve(placement)
        placements[placement] = {
            "target_refs": len(refs),
            "distinct_modules": len({ref.name for ref in refs}),
        }
    report: dict[str, Any] = {
        "architecture": architecture_report(arch, encoder),
        "geometry": {"size": size, "num_frames": num_frames},
        "checkpoint": load_report or {"loaded": False},
        "checkpoint_sha": digest,
        "state_tensors": len(encoder.state_dict()),
        "parameters": count_parameters(encoder),
        "hierarchical_layers": list(backbone.hierarchical_layers()),
        "token_layout": asdict(backbone.token_layout()),
        "placements": placements,
    }

    if video is not None:
        if preprocess_variant is None or preprocess_fps is None:
            raise ValueError(
                "a video forward requires --preprocess-variant and --preprocess-fps; "
                "they are protocol values without defaults"
            )
        decoder = make_decoder("pyav")
        meta = decoder.probe(video)
        stride = stride_frames_for(meta.fps, preprocess_fps)
        end = (num_frames - 1) * stride
        if end >= meta.num_frames:
            raise ValueError(
                f"{video}: {meta.num_frames} frames, the forward needs at least {end + 1}"
            )
        window = WindowSpec(meta.video_id, end, num_frames, stride)
        spec = vjepa21_preprocess_spec(
            preprocess_variant,  # type: ignore[arg-type]
            fps=preprocess_fps,
            num_frames=num_frames,
            size=size,
            dtype="fp32",
        )
        clip = preprocess_clip(decoder.read_frames(video, window.frame_indices), spec)
        pixel_batch = clip.unsqueeze(0) if clip.ndim == 4 else clip
        with torch.inference_mode():
            output = backbone.encode(pixel_batch.to(device), layer=-1)
        report["forward"] = {
            "source": str(video),
            "input_shape": list(pixel_batch.shape),
            "tokens_shape": list(output.tokens.shape),
            "grid": list(output.grid),
            "finite": bool(torch.isfinite(output.tokens).all().item()),
        }
    elif torch.device(device).type == "meta":
        output = backbone.encode(
            torch.empty(1, 3, num_frames, size, size, device="meta"),
            layer=-1,
        )
        report["forward"] = {
            "source": "synthetic-meta",
            "input_shape": [1, 3, num_frames, size, size],
            "tokens_shape": list(output.tokens.shape),
            "grid": list(output.grid),
            "finite": None,
        }
    else:
        report["forward"] = {"source": None, "reason": "no --video given"}
    return report


@dataclass(frozen=True)
class E0bConfig:
    checkpoint: Path
    repo: Path | None
    sizes: tuple[int, ...]
    precisions: tuple[str, ...]
    tf32_values: tuple[bool, ...]
    batches: tuple[int, ...]
    layers: tuple[int, ...]
    num_frames: int
    warmup: int
    iters: int
    device: str
    inventory_csv: Path
    videos_from: Path
    target_fps: float
    hop_candidates_s: tuple[float, ...]
    run_id: str
    seed: int


PROFILE_SCHEMA_VERSION = 4


def clip_seed(seed: int, *, size: int, batch: int, num_frames: int) -> int:
    """Return the clip seed of a cell, derived from its geometry.

    The profile is resumable, so a single seed at start-up would give different clips
    after a restart; deriving it from ``(seed, size, batch, num_frames)`` keeps the clip of
    a cell identical.
    """
    digest = stable_hash(
        {"seed": int(seed), "size": int(size), "batch": int(batch), "num_frames": int(num_frames)}
    )
    return int(digest[:16], 16) % (2**63 - 1)


def _encode_profile_cell(
    backbone: VJEPA21Backbone,
    clips: torch.Tensor,
    layers: Sequence[int],
    *,
    precision: str,
) -> Any:
    """Run a cell with the same mixed precision used by the Meta source.

    V-JEPA 2.1 keeps fp32 weights and uses autocast for bf16; converting the whole encoder
    to bf16 breaks the interpolated RoPE (q/k back to fp32, v in bf16, rejected by SDPA).
    """
    with torch.autocast(
        device_type=clips.device.type,
        dtype=torch.bfloat16,
        enabled=precision == "bf16",
    ):
        return backbone.encode_layers(clips, layers)


def profile_layer_sets(
    layers: Sequence[int], *, depth: int
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Return the two useful costs: last layer and hierarchical features in one forward.

    ``-1`` and ``depth-1`` are aliases: they may appear in different sets, never twice in
    the same set.

    :raises ValueError: on an empty multilayer set or aliases within it.
    :raises IndexError: on a layer outside the blocks.
    """
    hierarchical = tuple(int(layer) for layer in layers)
    if not hierarchical:
        raise ValueError("E0b.1 requires at least one layer in the multilayer set")
    resolved: list[int] = []
    for layer in hierarchical:
        idx = layer if layer >= 0 else depth + layer
        if not 0 <= idx < depth:
            raise IndexError(f"E0b.1 layer {layer} outside the {depth} blocks")
        resolved.append(idx)
    if len(set(resolved)) != len(resolved):
        raise ValueError(
            f"layer aliases in the multilayer set: requested {hierarchical}, resolved {resolved}"
        )
    return (("final", (-1,)), ("hierarchical", hierarchical))


def _profile_cells(
    config: E0bConfig,
    layer_sets: Sequence[tuple[str, tuple[int, ...]]],
) -> list[dict[str, Any]]:
    return [
        {
            "size": int(size),
            "precision": precision,
            "precision_mode": "amp_autocast" if precision == "bf16" else "native",
            "tf32": bool(tf32),
            "batch": int(batch),
            "layer_set": name,
            "layers": list(layers),
            "n_layers": len(layers),
        }
        for size in config.sizes
        for precision in config.precisions
        for tf32 in config.tf32_values
        for batch in config.batches
        for name, layers in layer_sets
    ]


def _cell_id(cell: dict[str, Any]) -> str:
    return (
        f"s{cell['size']}-p{cell['precision']}-t{int(cell['tf32'])}-"
        f"b{cell['batch']}-l{cell['layer_set']}"
    )


def _profile_plan(
    config: E0bConfig,
    *,
    checkpoint_digest: str,
    code_sha: str | None,
    layer_sets: Sequence[tuple[str, tuple[int, ...]]],
) -> dict[str, Any]:
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "run_id": config.run_id,
        "code_sha": code_sha,
        "checkpoint_sha": checkpoint_digest,
        "sizes": list(config.sizes),
        "precisions": list(config.precisions),
        "precision_modes": {"fp32": "native", "bf16": "amp_autocast"},
        "tf32": list(config.tf32_values),
        "batches": list(config.batches),
        "layer_sets": [{"name": name, "layers": list(layers)} for name, layers in layer_sets],
        "num_frames": config.num_frames,
        "warmup": config.warmup,
        "iters": config.iters,
        "device": config.device,
        "inventory_csv": str(config.inventory_csv),
        "videos_from": str(config.videos_from),
        "target_fps": config.target_fps,
        "hop_candidates_s": list(config.hop_candidates_s),
        "seed": config.seed,
    }


def load_progress(
    progress_dir: Path, expected_ids: set[str], *, etichetta: str = "E0b.1"
) -> dict[str, dict[str, Any]]:
    """Return the cells already measured, read back from disk after an interruption.

    Each cell is a file named after its id; a cell foreign to the plan stops the run
    instead of being ignored.

    :param progress_dir: folder of the written cells.
    :param expected_ids: ids foreseen by the plan.
    :param etichetta: block name, for error messages.
    :return: mapping ``cell_id -> row``.
    :raises RuntimeError: if a file does not belong to the plan or its name does not match
        the id it contains.
    """
    rows: dict[str, dict[str, Any]] = {}
    if not progress_dir.exists():
        return rows
    for path in sorted(progress_dir.glob("*.json")):
        row = read_json(path)
        cell_id = str(row.get("cell_id", ""))
        if cell_id not in expected_ids:
            raise RuntimeError(f"{etichetta} cell foreign to the plan: {path} ({cell_id!r})")
        if path.stem != cell_id:
            raise RuntimeError(f"inconsistent {etichetta} cell name: {path.name} != {cell_id}.json")
        rows[cell_id] = row
    return rows


def save_progress(
    progress_dir: Path,
    completed: dict[str, dict[str, Any]],
    row: dict[str, Any],
    *,
    cell_id: str,
) -> None:
    """Write a cell to disk and record it among the completed ones.

    :param progress_dir: folder of the cells.
    :param completed: in-memory mapping, updated in place.
    :param row: the measurement, without ``cell_id``.
    :param cell_id: cell identity, computed by the caller who knows the axes.
    """
    payload = {**row, "cell_id": cell_id}
    write_json(progress_dir / f"{cell_id}.json", payload)
    completed[cell_id] = payload


def _profile_registered(registry_path: Path, *, run_id: str, code_sha: str | None) -> bool:
    """Tell a finished profile from a crash between artifact and registry."""
    return any(
        row.get("run_id") == run_id
        and row.get("status") == "ok"
        and row.get("git_sha") == code_sha
        and row.get("exp") == "e0b"
        and row.get("part") == "e0b.1"
        for row in read_jsonl(registry_path)
    )


def _count_query_windows(
    records: Sequence[Any],
    *,
    target_fps: float,
    num_frames: int,
    hop_s: float,
) -> int:
    spec = _diagnostic_spec(size=16, num_frames=num_frames)
    spec = PreprocessSpec(**{**asdict(spec), "fps": target_fps})
    return sum(
        len(CausalWindowSampler.from_spec(spec, hop_s).windows_for_video(record.to_meta()))
        for record in records
    )


def build_hop_table(
    *,
    rows: Sequence[dict[str, Any]],
    records: Sequence[Any],
    target_fps: float,
    num_frames: int,
    hop_candidates_s: Sequence[float],
    sizes: Sequence[int],
    dim: int,
    n_layers: int,
) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for size in sizes:
        final_rows = [
            row
            for row in rows
            if row.get("status") == "ok" and row["size"] == size and row["layer_set"] == "final"
        ]
        candidates = [row for row in final_rows if row["precision"] == "fp32"]
        throughput: dict[bool, float | None] = {}
        for tf32 in (False, True):
            values = [
                float(row["throughput_windows_s"]) for row in candidates if row["tf32"] is tf32
            ]
            throughput[tf32] = max(values) if values else None
        throughput_off = throughput[False]
        throughput_on = throughput[True]
        # AMP BF16 is not split by tf32 (median throughput difference 0.025%): the maximum over
        # both flag values is taken.
        bf16_values = [
            float(row["throughput_windows_s"]) for row in final_rows if row["precision"] == "bf16"
        ]
        throughput_bf16 = max(bf16_values) if bf16_values else None
        tokens = (num_frames // 2) * (size // 16) ** 2
        for hop_s in hop_candidates_s:
            windows = _count_query_windows(
                records,
                target_fps=target_fps,
                num_frames=num_frames,
                hop_s=hop_s,
            )
            table.append(
                {
                    "size": size,
                    "hop_s": hop_s,
                    "n_windows": windows,
                    "pass_seconds_fp32_tf32_off": (
                        windows / throughput_off if throughput_off else None
                    ),
                    "pass_seconds_fp32_tf32_on": (
                        windows / throughput_on if throughput_on else None
                    ),
                    "pass_seconds_amp_bf16": (
                        windows / throughput_bf16 if throughput_bf16 else None
                    ),
                    "throughput_windows_s": {
                        "fp32_tf32_off": throughput_off,
                        "fp32_tf32_on": throughput_on,
                        "amp_bf16": throughput_bf16,
                    },
                    "cache_gb_fp16": sizeof_estimate(windows, n_layers, tokens, dim, 2)
                    / 1_000_000_000,
                    "cache_gb_fp32": sizeof_estimate(windows, n_layers, tokens, dim, 4)
                    / 1_000_000_000,
                }
            )
    return table


def _markdown(rows: Sequence[dict[str, Any]], hop_table: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# E0b.1 - frozen forward",
        "",
        "| size | precision | tf32 | batch | set | layers | ms/batch | win/s | peak GiB | TFLOP/s |",
        "|---:|---|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        layers = ",".join(map(str, row["layers"]))
        if row.get("status") != "ok":
            lines.append(
                f"| {row['size']} | {row['precision']} | {int(row['tf32'])} | "
                f"{row['batch']} | {row['layer_set']} | {layers} | "
                f"{row['status']} | - | - | - |"
            )
            continue
        lines.append(
            f"| {row['size']} | {row['precision']} | {int(row['tf32'])} | "
            f"{row['batch']} | {row['layer_set']} | {layers} | "
            f"{row['latency_batch_ms']:.3f} | "
            f"{row['throughput_windows_s']:.3f} | {row['peak_vram_bytes'] / 2**30:.3f} | "
            f"{row['effective_tflops_s']:.3f} |"
        )
    lines += [
        "",
        "## Query-set cost per candidate hop",
        "",
        "| size | hop_s | windows | s/pass fp32 | s/pass fp32+tf32 | s/pass AMP bf16 | "
        "cache4L fp16 GB | cache4L fp32 GB |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def _seconds(value: float | None) -> str:
        return "n/d" if value is None else f"{value:.1f}"

    for row in hop_table:
        lines.append(
            f"| {row['size']} | {row['hop_s']:.3g} | {row['n_windows']} | "
            f"{_seconds(row['pass_seconds_fp32_tf32_off'])} | "
            f"{_seconds(row['pass_seconds_fp32_tf32_on'])} | "
            f"{_seconds(row['pass_seconds_amp_bf16'])} | "
            f"{row['cache_gb_fp16']:.2f} | {row['cache_gb_fp32']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def profile_e0b1(config: E0bConfig) -> dict[str, Any]:
    target = torch.device(config.device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "official E0b.1 requires a CUDA GPU; it does not start on the login node"
        )
    if tuple(sorted(set(config.batches))) != config.batches or any(
        batch <= 0 for batch in config.batches
    ):
        raise ValueError(f"E0b.1 batches must be positive, unique and increasing: {config.batches}")
    if any(precision not in {"fp32", "bf16"} for precision in config.precisions):
        raise ValueError(f"precisions not allowed: {config.precisions}")
    if "bf16" in config.precisions and not torch.cuda.is_bf16_supported():
        raise RuntimeError("the GPU does not declare bf16 support")
    # deterministic=False on purpose: this measures kernels, and
    # ``use_deterministic_algorithms`` changes their selection.
    seed_all(config.seed, deterministic=False)
    architecture = ARCHS["vit_large"]
    layer_sets = profile_layer_sets(config.layers, depth=architecture.depth)
    digest = checkpoint_sha(config.checkpoint)
    tracker = LocalTracker()
    code_sha = git_info().get("sha")
    plan = _profile_plan(
        config,
        checkpoint_digest=digest,
        code_sha=code_sha,
        layer_sets=layer_sets,
    )
    cells = _profile_cells(config, layer_sets)
    expected_ids = {_cell_id(cell) for cell in cells}
    run_dir = tracker.runs_dir / config.run_id
    plan_path = run_dir / "profile_plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise RuntimeError(
            f"{plan_path} belongs to another plan or commit: E0b.1 measurements are not mixed"
        )
    progress_dir = run_dir / "progress"
    completed = load_progress(progress_dir, expected_ids)
    final_path = run_dir / "efficiency.json"
    if (
        len(completed) == len(cells)
        and final_path.exists()
        and _profile_registered(
            tracker.registry_path,
            run_id=config.run_id,
            code_sha=code_sha,
        )
    ):
        return cast(dict[str, Any], read_json(final_path))

    handle = tracker.start(
        config.run_id,
        asdict(config),
        official=True,
        regime={
            "attn": "sdpa",
            "device": config.device,
            "grid_declared": True,
            "profile_schema": PROFILE_SCHEMA_VERSION,
            "resumable_cells": True,
        },
    )
    if not plan_path.exists():
        write_json(plan_path, plan)
    ensure_dir(progress_dir)

    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        for size in config.sizes:
            size_cells = [cell for cell in cells if cell["size"] == size]
            if all(_cell_id(cell) in completed for cell in size_cells):
                continue
            encoder, _load_report = build_loaded_encoder(
                "vit_large",
                checkpoint=config.checkpoint,
                key=architecture.ckpt_key,
                img_size=size,
                num_frames=config.num_frames,
                device=config.device,
                repo=config.repo,
            )
            backbone = VJEPA21Backbone(
                encoder,
                arch=architecture,
                ckpt_sha=digest,
                spec=_diagnostic_spec(size=size, num_frames=config.num_frames),
                device=config.device,
            )
            backbone.freeze()
            # The Meta path uses fp32 weights/inputs and AMP bf16, not a model converted to bf16
            # (app/vjepa_2_1/train.py).
            encoder.to(dtype=torch.float32)
            tokens = (config.num_frames // 2) * (size // 16) ** 2
            for precision in config.precisions:
                for tf32 in config.tf32_values:
                    torch.backends.cuda.matmul.allow_tf32 = tf32
                    torch.backends.cudnn.allow_tf32 = tf32
                    for batch in config.batches:
                        active: list[tuple[dict[str, Any], tuple[int, ...]]] = []
                        for name, layers in layer_sets:
                            cell = {
                                "size": size,
                                "precision": precision,
                                "precision_mode": (
                                    "amp_autocast" if precision == "bf16" else "native"
                                ),
                                "tf32": tf32,
                                "batch": batch,
                                "layer_set": name,
                                "layers": list(layers),
                                "n_layers": len(layers),
                            }
                            cell_id = _cell_id(cell)
                            if cell_id in completed:
                                continue
                            prior_oom = [
                                int(row.get("oom_batch", row["batch"]))
                                for row in completed.values()
                                if row["size"] == size
                                and row["precision"] == precision
                                and row["tf32"] is tf32
                                and row["layer_set"] == name
                                and row["status"] in {"oom", "skipped_after_oom"}
                            ]
                            if prior_oom and batch > min(prior_oom):
                                save_progress(
                                    progress_dir,
                                    completed,
                                    {
                                        **cell,
                                        "status": "skipped_after_oom",
                                        "oom_batch": min(prior_oom),
                                        "attn": "sdpa",
                                    },
                                    cell_id=_cell_id(cell),
                                )
                                continue
                            active.append((cell, layers))
                        if not active:
                            continue
                        try:
                            # Seed the global RNG (CPU and CUDA) right before drawing: the seed depends on the cell,
                            # not on execution order.
                            torch.manual_seed(
                                clip_seed(
                                    config.seed,
                                    size=size,
                                    batch=batch,
                                    num_frames=config.num_frames,
                                )
                            )
                            clips = torch.randn(
                                batch,
                                3,
                                config.num_frames,
                                size,
                                size,
                                device=target,
                                dtype=torch.float32,
                            )
                        except torch.OutOfMemoryError:
                            for cell, _layers in active:
                                save_progress(
                                    progress_dir,
                                    completed,
                                    {
                                        **cell,
                                        "status": "oom",
                                        "oom_stage": "clip_allocation",
                                        "attn": "sdpa",
                                    },
                                    cell_id=_cell_id(cell),
                                )
                            torch.cuda.empty_cache()
                            continue

                        flops = transformer_forward_flops(
                            batch_size=batch,
                            tokens=tokens,
                            dim=architecture.embed_dim,
                            depth=architecture.depth,
                            patch_volume=2 * 16 * 16,
                        )
                        for cell, layers in active:
                            try:
                                with torch.inference_mode():
                                    measured = measure(
                                        partial(
                                            _encode_profile_cell,
                                            backbone,
                                            clips,
                                            layers,
                                            precision=precision,
                                        ),
                                        warmup=config.warmup,
                                        iters=config.iters,
                                        device=config.device,
                                    )
                            except torch.OutOfMemoryError:
                                save_progress(
                                    progress_dir,
                                    completed,
                                    {
                                        **cell,
                                        "status": "oom",
                                        "oom_stage": "encode_layers",
                                        "attn": "sdpa",
                                    },
                                    cell_id=_cell_id(cell),
                                )
                                torch.cuda.empty_cache()
                                continue
                            seconds = measured.latency_ms_mean / 1000
                            save_progress(
                                progress_dir,
                                completed,
                                {
                                    **cell,
                                    "status": "ok",
                                    "attn": "sdpa",
                                    "tokens": tokens,
                                    "latency_batch_ms": measured.latency_ms_mean,
                                    "latency_batch_ms_std": measured.latency_ms_std,
                                    "latency_window_ms": measured.latency_ms_mean / batch,
                                    "throughput_windows_s": batch / seconds,
                                    "peak_vram_bytes": measured.peak_vram_bytes,
                                    "effective_tflops_s": flops / seconds / 1e12,
                                    "flops_per_batch": flops,
                                    "flops_source": "vit_formula_matmul_conv",
                                    "warmup": measured.n_warmup,
                                    "iters": measured.n_iters,
                                    "synced": measured.synced,
                                },
                                cell_id=_cell_id(cell),
                            )
                        del clips
                        torch.cuda.empty_cache()
            del backbone, encoder
            torch.cuda.empty_cache()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32

    missing = expected_ids.difference(completed)
    if missing:
        raise RuntimeError(f"incomplete E0b.1 profile: {len(missing)} missing cells")
    rows = [completed[_cell_id(cell)] for cell in cells]
    records = select_inventory_records(
        config.inventory_csv,
        config.videos_from,
        shard_index=0,
        num_shards=1,
    )
    hop_table = build_hop_table(
        rows=rows,
        records=records,
        target_fps=config.target_fps,
        num_frames=config.num_frames,
        hop_candidates_s=config.hop_candidates_s,
        sizes=config.sizes,
        dim=architecture.embed_dim,
        n_layers=len(architecture.hierarchical_layers),
    )
    payload = {
        "run_id": config.run_id,
        "checkpoint_sha": digest,
        "profile_schema": PROFILE_SCHEMA_VERSION,
        "gpu": torch.cuda.get_device_name(target),
        "seed": config.seed,
        "rows": rows,
        "hop_table": hop_table,
    }
    json_path = write_json(handle.dir / "efficiency.json", payload)
    markdown_path = write_text_atomic(handle.dir / "efficiency.md", _markdown(rows, hop_table))
    metrics = {
        "n_rows_ok": sum(row.get("status") == "ok" for row in rows),
        "n_oom": sum(row.get("status") == "oom" for row in rows),
        "n_skipped_after_oom": sum(row.get("status") == "skipped_after_oom" for row in rows),
        "n_cells": len(rows),
        "n_hop_rows": len(hop_table),
        "efficiency_json": str(json_path),
        "efficiency_markdown": str(markdown_path),
    }
    tracker.finish(handle, metrics, extra={"exp": "e0b", "part": "e0b.1"})
    return payload
