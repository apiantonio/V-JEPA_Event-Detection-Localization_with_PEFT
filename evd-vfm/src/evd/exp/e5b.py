"""Inference efficiency

The PdD claim is that prompt and reparam have opposite trade-offs: a reparam adapter
merges into the weights and disappears at inference, while prompts lengthen the
sequence forever. This module turns that into numbers.

For every variant of :data:`VARIANTS` (frozen, VPT, LoRA, merged LoRA, DoRA, merged DoRA
and the combinations with their partially merged rows) and every batch: latency
per window (mean, deviation, percentiles), throughput, peak VRAM, tokens and declared
FLOPs per window, and parameters counted three ways (adapter, total, and shipped bytes).

1. The ``HeadScorer`` itself is measured, the object ``InferenceEngine`` calls; the
   default precision is bf16, the ``eval_precision`` of the cells.
2. Variants are built from the real ``configs/adapt/`` files.
3. Pixels are synthetic and the decoder is outside the measurement:
   this module measures the model, not the system; end-to-end figures are composed.
4. Merging is verified before timing: merged cells record ``merge_max_abs`` and
   ``merge_rel_l2`` in fp32 and are marked ``sospetta`` outside tolerance.

A combination also has a merged row, and its merge is partial: the reparam half merges,
prompts remain; ``fusione`` says ``nessuna``/``parziale``/``totale`` and
``params_da_spedire`` includes the prompts.

``merge_()`` writes in place into ``base.weight``, so the backbone is restored from a
CPU copy after every merged cell and the restoration is verified with a weight digest.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from evd.eval.deployment import fattore_realtime
from evd.eval.efficiency import (
    PATCH_VOLUME,
    measure,
    tokens_per_window,
    transformer_forward_flops,
)
from evd.eval.scorers import HeadScorer
from evd.exp.e0 import _diagnostic_spec, clip_seed
from evd.exp.tracker import LocalTracker
from evd.models.adapt.factory import AdaptContext, build_adaptation
from evd.models.backbones.base import encode_multilevel
from evd.models.backbones.vjepa21 import VJEPA21Backbone
from evd.models.backbones.vjepa21_source import ARCHS, build_loaded_encoder, checkpoint_sha
from evd.models.heads.attentive import AttentiveProbe
from evd.models.heads.linear import LinearProbe
from evd.types import WindowSpec
from evd.utils.io import ensure_dir, write_json
from evd.utils.logging import get_logger
from evd.utils.seeding import seed_all

log = get_logger(__name__)

#: Schema 2 adds percentiles, batch sweep, energy and cold-start time; stored in
#: the payload so readers can tell a missing column from an empty one.
PROFILE_SCHEMA_VERSION = 2

#: Profiled variants, each with the real config file describing it; the frozen probe is
#: the floor and the merged variants are what would be deployed. VPT has no merged row
#: (``can_merge()`` is false by construction). ``testa`` matters: linear and attentive
#: probes differ by 49.3 M head parameters.
VARIANTS: dict[str, dict[str, Any]] = {
    "frozen": {"config": "none", "fondi": False, "testa": "attentive"},
    "frozen-linear": {"config": "none", "fondi": False, "testa": "linear"},
    "lora-r8": {"config": "lora_r8_qkvomlp", "fondi": False, "testa": "attentive"},
    "lora-r8-fusa": {"config": "lora_r8_qkvomlp", "fondi": True, "testa": "attentive"},
    "lora-r16": {"config": "lora_r16_qkvomlp", "fondi": False, "testa": "attentive"},
    "lora-r16-fusa": {"config": "lora_r16_qkvomlp", "fondi": True, "testa": "attentive"},
    "lora-r32": {"config": "lora_r32_qkvomlp", "fondi": False, "testa": "attentive"},
    "lora-r32-fusa": {"config": "lora_r32_qkvomlp", "fondi": True, "testa": "attentive"},
    "lora-r64": {"config": "lora_r64_qkvomlp", "fondi": False, "testa": "attentive"},
    "lora-r64-fusa": {"config": "lora_r64_qkvomlp", "fondi": True, "testa": "attentive"},
    "dora-r32": {"config": "dora_r32_qkvomlp", "fondi": False, "testa": "attentive"},
    "dora-r32-fusa": {"config": "dora_r32_qkvomlp", "fondi": True, "testa": "attentive"},
    "vptdeep-n8": {"config": "vpt_deep_n8", "fondi": False, "testa": "attentive"},
    "vptdeep-n64": {"config": "vpt_deep_n64", "fondi": False, "testa": "attentive"},
    "vptdeep-n192": {"config": "vpt_deep_n192", "fondi": False, "testa": "attentive"},
    "vptdeep-n576": {"config": "vpt_deep_n576", "fondi": False, "testa": "attentive"},
    "vptshallow-n8": {"config": "vpt_shallow_n8", "fondi": False, "testa": "attentive"},
    "vptshallow-n64": {"config": "vpt_shallow_n64", "fondi": False, "testa": "attentive"},
    "vptshallow-n192": {"config": "vpt_shallow_n192", "fondi": False, "testa": "attentive"},
    "vptshallow-n576": {"config": "vpt_shallow_n576", "fondi": False, "testa": "attentive"},
    # The four E3-C combinations, each in two rows like LoRA and DoRA; the merged row of a
    # combination is partial (`fusione`): the reparam merges, the prompt remains (ADR-064).
    "combo-lora-r32-vptdeep-n8": {
        "config": "combo_lora_r32_vptdeep_n8",
        "fondi": False,
        "testa": "attentive",
    },
    "combo-lora-r32-vptdeep-n8-fusa": {
        "config": "combo_lora_r32_vptdeep_n8",
        "fondi": True,
        "testa": "attentive",
    },
    "combo-lora-r32-vptshallow-n8": {
        "config": "combo_lora_r32_vptshallow_n8",
        "fondi": False,
        "testa": "attentive",
    },
    "combo-lora-r32-vptshallow-n8-fusa": {
        "config": "combo_lora_r32_vptshallow_n8",
        "fondi": True,
        "testa": "attentive",
    },
    "combo-dora-r32-vptdeep-n8": {
        "config": "combo_dora_r32_vptdeep_n8",
        "fondi": False,
        "testa": "attentive",
    },
    "combo-dora-r32-vptdeep-n8-fusa": {
        "config": "combo_dora_r32_vptdeep_n8",
        "fondi": True,
        "testa": "attentive",
    },
    "combo-dora-r32-vptshallow-n8": {
        "config": "combo_dora_r32_vptshallow_n8",
        "fondi": False,
        "testa": "attentive",
    },
    "combo-dora-r32-vptshallow-n8-fusa": {
        "config": "combo_dora_r32_vptshallow_n8",
        "fondi": True,
        "testa": "attentive",
    },
}


@dataclass(frozen=True)
class E5bConfig:
    """Configuration of the efficiency profile.

    :param checkpoint: backbone weights.
    :param configs_dir: Hydra groups folder; the real ``adapt/`` files used by E3 are read
        from here.
    :param batches: batch sizes to measure. 1 is the deployment latency of a single window;
        2 is the E3 evaluation batch; 4 and 8 show where throughput saturates (cameras per
        GPU).
    :param num_frames: samples per window; 32 on ``edp_v2`` (ADR-034).
    :param size: frame side; 384 on ``edp_v2``.
    :param precision: forward regime; bf16 by default, as declared by the E3 cells.
    :param layer: level read by the head; 23 on ViT-L.
    :param hop_s: grid step in seconds, used only for the real-time factor (a streaming
        system keeps up if it produces a window in less than ``hop_s`` seconds).
    :param tolleranza_merge: maximum absolute difference allowed between outputs before and
        after ``merge_()``, in fp32.
    :param scala_perturbazione: standard deviation of the noise filling the adapter
        parameters before checking the merge (``init: kaiming_b_zero`` makes ΔW exactly
        zero, so the check could not fail otherwise).
    :param effetto_minimo: minimum relative output shift of the perturbed adapter for the
        merge check to count; below it the row is marked ``sospetta``.
    """

    checkpoint: Path
    run_id: str = "e5b_efficienza"
    repo: Path | None = None
    arch: str = "vit_large"
    configs_dir: Path = Path("configs")
    varianti: tuple[str, ...] = tuple(VARIANTS)
    batches: tuple[int, ...] = (1, 2, 4, 8)
    num_frames: int = 32
    size: int = 384
    precision: str = "bf16"
    layer: int = 23
    n_classes: int = 1
    warmup: int = 5
    iters: int = 30
    iters_batch1: int = 200
    """Batch 1 is measured much longer: there latency is a response time and its tail
    matters (a p99 over 30 samples would be noise). At larger batches the number is a
    throughput and thirty iterations suffice."""
    misura_energia: bool = True
    intervallo_energia_s: float = 0.1
    device: str = "cuda"
    seed: int = 42
    hop_s: float = 4.0
    tolleranza_merge: float = 1e-4
    scala_perturbazione: float = 0.02
    effetto_minimo: float = 1e-3
    head: dict[str, Any] = field(
        default_factory=lambda: {"n_queries": 1, "n_heads": 16, "depth": 4, "norm": True}
    )


def _weights_digest(module: torch.nn.Module) -> str:
    """Return a weight fingerprint, to verify a restoration instead of assuming it.

    :param module: module to fingerprint.
    :return: first 16 hex digits of a SHA-256 over tensor bytes in name order,
        deterministic and device-independent.
    """
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().to("cpu", torch.float32).numpy().tobytes())
    return h.hexdigest()[:16]


def _synthetic_batch(
    *, batch: int, size: int, num_frames: int, seed: int, device: str
) -> dict[str, Any]:
    """Return a batch of fake pixels, prepared once and reused at every iteration.

    :return: the dictionary expected by ``HeadScorer.score``, with ``clip``
        ``[B, C, T, H, W]`` and the ``window_id`` values carried by the scores.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    clip = torch.rand((batch, 3, num_frames, size, size), generator=generator, dtype=torch.float32)
    # `window`, not `window_id`: the key `HeadScorer._from_logit` reads, as produced by
    # `collate_windows`.
    windows = [
        WindowSpec(
            video_id=f"finto{i}",
            end_frame=num_frames,
            num_frames=num_frames,
            stride_frames=1,
        )
        for i in range(batch)
    ]
    return {"clip": clip.to(device), "window": windows}


def _fp32_output(backbone: Any, head: Any, batch: dict[str, Any], *, layer: int) -> torch.Tensor:
    """Run a control forward in fp32, without autocast and without gradients.

    Used only by the merge check: in bf16 the difference would be dominated by rounding.
    """
    with torch.no_grad():
        feats = encode_multilevel(backbone, batch["clip"], (layer,))
        return head(feats).float().detach()


@torch.no_grad()
def _perturb(strategy: Any, *, seed: int, scala: float) -> None:
    """Fill the adapter parameters with noise, so that ΔW is not zero.

    Makes the merge check able to fail (``init: kaiming_b_zero`` starts ``B`` at zero). The
    perturbed weights are not the trained ones: this checks the merge algebra on arbitrary
    ΔW. The noise is seeded.

    :param strategy: strategy already applied to the backbone.
    :param seed: noise seed.
    :param scala: noise standard deviation; at ``0`` the function does nothing.
    """
    if scala <= 0:
        return
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for parameter in strategy.trainable_parameters():
        noise = torch.randn(parameter.shape, generator=generator, dtype=torch.float32).to(
            parameter.device, parameter.dtype
        )
        parameter.add_(noise * scala)


def _count_parameters(module: Any) -> int:
    return sum(int(p.numel()) for p in module.parameters())


def celle(config: E5bConfig) -> list[dict[str, Any]]:
    """:return: the plan, one row per (variant, batch), in measurement order."""
    return [
        {"variante": variant, "batch": batch}
        for variant in config.varianti
        for batch in config.batches
    ]


def _load_adapt(config: E5bConfig, nome: str) -> dict[str, Any]:
    """Compose an ``adapt`` group as a run does, not as stored on disk.

    Ablation configs inherit from a base through ``defaults`` (``vpt_shallow_n64.yaml`` has
    three lines and no ``family``), so raw YAML reading would profile a config no run used.

    :param config: profile configuration, for the configs folder.
    :param nome: group name, without extension.
    :return: the resolved group.
    :raises FileNotFoundError: naming the missing file.
    """
    from hydra import compose, initialize_config_dir

    folder = (Path(config.configs_dir) / "adapt").resolve()
    path = folder / f"{nome}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing adaptation config: {path}. E5b profiles the REAL E3 configs, "
            f"not a table written in the profiler: without the file, the cell "
            f"would measure a method no run executed."
        )
    # Only the `adapt` folder is composed: composing `config.yaml` would require every
    # mandatory group (`train`, `data`, ...).
    with initialize_config_dir(config_dir=str(folder), version_base="1.3"):
        composite = compose(config_name=nome)
    return dict(OmegaConf.to_container(composite, resolve=True))  # type: ignore[arg-type]


def _measure_variant(
    variante: str,
    *,
    backbone: VJEPA21Backbone,
    heads: Mapping[str, Any],
    config: E5bConfig,
    clean_state: Mapping[str, torch.Tensor],
    clean_digest: str,
    n_blocks: int,
) -> list[dict[str, Any]]:
    """Measure all rows of a variant, and restore the backbone afterwards.

    :param clean_state: original backbone weights, on CPU (in the signature because the
        function's correctness depends on them).
    :param clean_digest: fingerprint of the original weights; restoration after a merge is
        verified against it.
    :return: one row per batch.
    :raises RuntimeError: if the weights are not identical after restoration.
    """
    variant_row = VARIANTS[variante]
    merge = bool(variant_row["fondi"])
    head_name = str(variant_row.get("testa", "attentive"))
    head = heads[head_name]
    cfg_adapt = _load_adapt(config, str(variant_row["config"]))
    strategy = build_adaptation(
        cfg_adapt, AdaptContext(n_classes=config.n_classes, n_blocks=n_blocks, seed=config.seed)
    )
    adapted = strategy.apply(backbone)
    report = adapted.report

    # Two questions (ADR-064): `can_merge()` asks whether the whole adaptation disappears,
    # `merge_parts()` how much of it does; on a combination only the reparam merges.
    mergeable_parts = tuple(strategy.merge_parts())
    merge_outcome: dict[str, Any] = {
        "fondibile": bool(strategy.can_merge()),
        "parti_fondibili": list(mergeable_parts),
        "fusione": "nessuna",
        "fusa": False,
        "merge_max_abs": None,
        "merge_rel_l2": None,
        "effetto_adattatore": None,
        "merge_secondi": None,
    }
    if merge:
        if not mergeable_parts:
            # Not a profile error but the result: the row exists and says it does not merge.
            merge_outcome["motivo"] = "merge_parts() is empty: prompts lengthen the sequence"
        else:
            sample = _synthetic_batch(
                batch=1,
                size=config.size,
                num_frames=config.num_frames,
                seed=clip_seed(
                    config.seed, size=config.size, batch=1, num_frames=config.num_frames
                ),
                device=config.device,
            )
            # Output before any adapter: with `B = 0` at initialization ΔW is exactly zero, so the
            # check needs a perturbed adapter to be able to fail (in the first E5b run LoRA and DoRA
            # gave the same gap to the seventh decimal).
            bare = _fp32_output(backbone, head, sample, layer=config.layer)
            _perturb(strategy, seed=config.seed, scala=config.scala_perturbazione)
            first = _fp32_output(backbone, head, sample, layer=config.layer)
            # The merge is paid once, before shipping the model; measured because it is the
            # procedure that makes the C7 claim true.
            merge_start = time.perf_counter()
            strategy.merge_()
            # `unload_merged_()`, not `unload_()`: on a combination the latter would also remove the
            # prompts and measure the frozen model.
            removed = strategy.unload_merged_()
            if config.device.startswith("cuda"):
                torch.cuda.synchronize()
            merge_seconds = time.perf_counter() - merge_start
            after = _fp32_output(backbone, head, sample, layer=config.layer)

            norm = first.norm().item()
            bare_norm = bare.norm().item()
            merge_outcome.update(
                fusa=True,
                fusione="totale" if strategy.can_merge() else "parziale",
                # What must be shipped next to the merged backbone: zero for a pure reparam, the prompts
                # for a combination.
                params_residui=sum(int(p.numel()) for p in strategy.trainable_parameters()),
                wrapper_rimossi=int(removed),
                merge_secondi=merge_seconds,
                merge_max_abs=float((after - first).abs().max().item()),
                merge_rel_l2=float((after - first).norm().item() / norm) if norm else None,
                # How much the perturbed adapter shifts the output: if ~0, a ~0 `merge_max_abs` would only
                # mean there was nothing to merge.
                effetto_adattatore=(
                    float((first - bare).norm().item() / bare_norm) if bare_norm else None
                ),
            )
            del sample, bare, first, after

    # Before the loop: on a merged cell the strategy no longer describes the measured model;
    # extra tokens of a fully merged reparam are zero by definition.
    token_base = tokens_per_window(num_frames=config.num_frames, size=config.size)
    # Zero only for a full merge; on a partial merge the prompts are still there.
    extra = 0 if merge_outcome["fusione"] == "totale" else int(strategy.extra_tokens())

    # A merged cell that changed the output is another model: it is marked instead of
    # raising. Two different suspicious cases: the merge changed the output, or the merge was
    # not put to the test.
    gap = merge_outcome.get("merge_max_abs")
    effect = merge_outcome.get("effetto_adattatore")
    state = "ok"
    if effect is not None and effect < config.effetto_minimo:
        state = "sospetta"
        log.warning(
            "variant %s: the adapter shifts the output by only %.3e (< %.1e), so the "
            "merge check tested nothing",
            variante,
            effect,
            config.effetto_minimo,
        )
    if gap is not None and gap > config.tolleranza_merge:
        state = "sospetta"
        log.warning(
            "variant %s: the merge shifts the output by %.3e in fp32, above the "
            "tolerance %.1e - the row is marked 'sospetta'",
            variante,
            gap,
            config.tolleranza_merge,
        )

    scorer = HeadScorer(
        backbone,
        head,
        levels=(config.layer,),
        device=config.device,
        precision=config.precision,
    )
    rows: list[dict[str, Any]] = []
    for batch in config.batches:
        data = _synthetic_batch(
            batch=batch,
            size=config.size,
            num_frames=config.num_frames,
            seed=clip_seed(
                config.seed, size=config.size, batch=batch, num_frames=config.num_frames
            ),
            device=config.device,
        )
        # The batch is bound as a default argument, not through the closure name. At batch 1
        # latency is a response time and its tail matters; above it is a throughput.
        n_iters = config.iters_batch1 if batch == 1 else config.iters
        measurement = measure(
            lambda b=data: scorer.score(b),
            warmup=config.warmup,
            iters=n_iters,
            device=config.device,
            energy=config.misura_energia,
            intervallo_energia_s=config.intervallo_energia_s,
        )
        rows.append(
            {
                "variante": variante,
                "adapt_config": str(variant_row["config"]),
                "testa": head_name,
                "family": str(report.family),
                "method": str(report.method),
                "batch": batch,
                "tokens_per_window": token_base + extra,
                "tokens_base": token_base,
                "extra_tokens": extra,
                "params_adapter": int(report.n_trainable),
                "params_backbone": _count_parameters(backbone),
                "params_head": _count_parameters(head),
                # Scalars shipped to deploy this model elsewhere, given the base backbone is already
                # there: after a merge the whole backbone is shipped, plus the residual prompts on a
                # partial merge.
                "params_da_spedire": (
                    _count_parameters(backbone) + int(merge_outcome.get("params_residui") or 0)
                    if merge_outcome["fusa"]
                    else int(report.n_trainable)
                ),
                "latency_ms_mean": measurement.latency_ms_mean,
                "latency_ms_std": measurement.latency_ms_std,
                "latency_ms_min": measurement.latency_ms_min,
                "latency_ms_p50": measurement.latency_ms_p50,
                "latency_ms_p90": measurement.latency_ms_p90,
                "latency_ms_p95": measurement.latency_ms_p95,
                "latency_ms_p99": measurement.latency_ms_p99,
                "latency_ms_max": measurement.latency_ms_max,
                "throughput_windows_s": measurement.throughput_per_s * batch,
                "peak_vram_bytes": measurement.peak_vram_bytes,
                "potenza_w_media": measurement.power_w_mean,
                "potenza_w_massima": measurement.power_w_max,
                # Per window, not per call: at batch B a call produces B windows.
                "energia_j_per_finestra": (
                    measurement.energy_j_per_call / batch
                    if measurement.energy_j_per_call is not None
                    else None
                ),
                "sorgente_potenza": measurement.power_source,
                "n_warmup": measurement.n_warmup,
                "n_iters": measurement.n_iters,
                **merge_outcome,
                "status": state,
            }
        )
        del data
        if config.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # Unconditional: after a full merge nothing is left (returns 0), after a partial one the
    # prompts must be removed before restoring the weights.
    strategy.unload_()
    # Restoration and its verification: `merge_()` wrote into `base.weight`, and the next
    # variant would otherwise measure corrupted weights.
    if merge_outcome["fusa"]:
        backbone.module().load_state_dict(dict(clean_state), strict=True)
    digest = _weights_digest(backbone.module())
    if digest != clean_digest:
        raise RuntimeError(
            f"after variant {variante!r} the backbone weights are not the original "
            f"ones ({digest} instead of {clean_digest}): every later cell "
            f"would measure a model that does not exist. The profile stops here."
        )
    return rows


def profile_e5b(config: E5bConfig) -> dict[str, Any]:
    """Run the profile and write ``runs/{run_id}/efficiency_inference.json``.

    :param config: the configuration.
    :return: the payload, already written to disk.
    :raises RuntimeError: if the device is not CUDA, or the GPU does not support bf16.
    """
    target = torch.device(config.device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("official E5b requires a CUDA GPU; it does not start on the login node")
    if config.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("the GPU does not declare bf16 support, which is the E3 regime")
    # As in E0b: kernels are measured, and `use_deterministic_algorithms` would change their
    # selection.
    seed_all(config.seed, deterministic=False)

    architecture = ARCHS[config.arch]
    digest_ckpt = checkpoint_sha(config.checkpoint)
    # Cold start is a deployment metric (how long a camera stays blind after a restart or a
    # model update), measured in two parts: weight loading (I/O) and first pass (kernel
    # selection).
    load_start = time.perf_counter()
    encoder, _ = build_loaded_encoder(
        config.arch,
        checkpoint=config.checkpoint,
        key=architecture.ckpt_key,
        img_size=config.size,
        num_frames=config.num_frames,
        device=config.device,
        repo=config.repo,
    )
    encoder.to(dtype=torch.float32)
    backbone = VJEPA21Backbone(
        encoder,
        arch=architecture,
        ckpt_sha=digest_ckpt,
        spec=_diagnostic_spec(size=config.size, num_frames=config.num_frames),
        device=config.device,
    )
    backbone.freeze()
    backbone.eval()
    n_blocks = len(backbone.blocks())
    # Both heads are built and chosen per variant: a single head would charge the linear
    # probe with the attentive head cost.
    heads: dict[str, Any] = {
        "attentive": AttentiveProbe(
            embed_dim=backbone.embed_dim, n_classes=config.n_classes, **config.head
        )
        .to(config.device)
        .eval(),
        "linear": LinearProbe(
            embed_dim=backbone.embed_dim,
            n_classes=config.n_classes,
            norm=bool(config.head.get("norm", True)),
        )
        .to(config.device)
        .eval(),
    }

    load_seconds = time.perf_counter() - load_start

    # The first pass is timed separately (kernel selection, cuDNN workspaces), with the
    # attentive head used by every PEFT cell and the frozen reference.
    start_before = time.perf_counter()
    _first = HeadScorer(
        backbone,
        heads["attentive"],
        levels=(config.layer,),
        device=config.device,
        precision=config.precision,
    )
    _first.score(
        _synthetic_batch(
            batch=1,
            size=config.size,
            num_frames=config.num_frames,
            seed=config.seed,
            device=config.device,
        )
    )
    if config.device.startswith("cuda"):
        torch.cuda.synchronize()
    first_pass_seconds = time.perf_counter() - start_before
    del _first

    clean_state = {
        k: v.detach().to("cpu").clone() for k, v in backbone.module().state_dict().items()
    }
    clean_digest = _weights_digest(backbone.module())
    log.info(
        "backbone ready: %d blocks, digest %s, %d parameters; cold start "
        "%.1f s (loading) + %.1f s (first pass)",
        n_blocks,
        clean_digest,
        _count_parameters(backbone),
        load_seconds,
        first_pass_seconds,
    )

    tracker = LocalTracker()
    handle = tracker.start(
        config.run_id,
        asdict(config) | {"checkpoint": str(config.checkpoint)},
        official=True,
        regime={
            "attn": "sdpa",
            "device": config.device,
            "precision": config.precision,
            "profile_schema": PROFILE_SCHEMA_VERSION,
        },
    )

    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for variant in config.varianti:
        log.info("variant %s", variant)
        rows.extend(
            _measure_variant(
                variant,
                backbone=backbone,
                heads=heads,
                config=config,
                clean_state=clean_state,
                clean_digest=clean_digest,
                n_blocks=n_blocks,
            )
        )

    payload = {
        "run_id": config.run_id,
        "profile_schema": PROFILE_SCHEMA_VERSION,
        "checkpoint_sha": digest_ckpt,
        "gpu": torch.cuda.get_device_name(target),
        "arch": config.arch,
        "precision": config.precision,
        "geometria": {
            "size": config.size,
            "num_frames": config.num_frames,
            "layer": config.layer,
            "hop_s": config.hop_s,
        },
        "backbone_digest": clean_digest,
        "avvio_a_freddo": {
            "caricamento_s": load_seconds,
            "prima_passata_s": first_pass_seconds,
            "totale_s": load_seconds + first_pass_seconds,
        },
        "secondi": time.perf_counter() - start,
        "rows": tabella(rows, config),
    }
    run_dir = ensure_dir(tracker.runs_dir / config.run_id)
    write_json(run_dir / "efficiency_inference.json", payload)
    tracker.finish(
        handle,
        {"n_celle": len(rows)},
        status="ok",
        extra={"exp": "e5b", "part": "inference"},
    )
    return payload


def tabella(righe: Sequence[Mapping[str, Any]], config: E5bConfig) -> list[dict[str, Any]]:
    """Add the derived columns to the measured rows.

    * ``overhead_latenza_pct``: adaptation cost relative to the frozen model at the same
      batch (~0 for a merged reparam).
    * ``flops_per_window`` and ``overhead_flops_pct``: the same question in arithmetic; at
      9,216 tokens attention is quadratic, so 576 extra tokens are +6.25% of the sequence
      but not of the work.
    * ``fattore_realtime``: at batch 1, seconds of video processed per second of compute
      (above 1.0 the model keeps up); model only, decoding excluded.

    :param righe: measured rows.
    :param config: configuration, for geometry and ``hop_s``.
    :return: the rows with derived columns, in the same order.
    """
    architecture = ARCHS[config.arch]

    # Two passes: a row overhead is computed from the FLOPs of another row (the frozen one),
    # which a single pass would not have yet.
    with_flops: list[dict[str, Any]] = []
    for row in righe:
        new = dict(row)
        token = row.get("tokens_per_window")
        if token:
            new["flops_per_window"] = transformer_forward_flops(
                batch_size=1,
                tokens=int(token),
                dim=architecture.embed_dim,
                depth=architecture.depth,
                patch_volume=PATCH_VOLUME,
            )
        with_flops.append(new)

    # The reference is the frozen model at the same batch, and only if its measurement is
    # not marked `sospetta`.
    base = {
        int(r["batch"]): r
        for r in with_flops
        if r["variante"] == "frozen" and r.get("status") == "ok"
    }
    outside: list[dict[str, Any]] = []
    for new in with_flops:
        reference = base.get(int(new["batch"]))
        if reference is not None and reference is not new:
            new["overhead_latenza_pct"] = 100.0 * (
                float(new["latency_ms_mean"]) / float(reference["latency_ms_mean"]) - 1.0
            )
            if new.get("flops_per_window") and reference.get("flops_per_window"):
                new["overhead_flops_pct"] = 100.0 * (
                    new["flops_per_window"] / reference["flops_per_window"] - 1.0
                )
        batch = int(new["batch"])
        latency_ms = float(new["latency_ms_mean"])
        new["ms_per_finestra"] = latency_ms / batch
        # `fattore_realtime` stays batch-1 only: it is a one-window-at-a-time question.
        if batch == 1:
            new["fattore_realtime"] = fattore_realtime(latency_ms / 1000.0, hop_s=config.hop_s)
        # `stream_per_gpu` is a stationary question and exists at every batch: how many sources
        # the GPU sustains, each requiring a window every `hop_s` seconds. An arithmetic bound;
        # system capacity is computed by `eval.deployment`.
        new["stream_per_gpu"] = config.hop_s * batch * 1000.0 / latency_ms
        p50, p99 = new.get("latency_ms_p50"), new.get("latency_ms_p99")
        if p50 and p99:
            new["jitter_pct"] = 100.0 * (float(p99) / float(p50) - 1.0)
        if new.get("energia_j_per_finestra"):
            # Watt-hours per hour of watched video: the running-cost unit, additive across cameras.
            new["wh_per_ora_di_video"] = (
                float(new["energia_j_per_finestra"]) * (3600.0 / config.hop_s) / 3600.0
            )
        outside.append(new)
    return outside
