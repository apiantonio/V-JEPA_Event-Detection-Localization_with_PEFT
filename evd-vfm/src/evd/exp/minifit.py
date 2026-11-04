"""GPU integration mini-fit: PEFT on V-JEPA 2.1-L, on the real model.

The *Integration GPU* level of docs/08 §1. Not an experiment: it produces a binary
verdict on five things that unit tests on Tiny cannot prove:

1. The target map resolves on the real model (24 blocks, fused
   ``Linear(1024→3072)`` QKV); a placement resolving to zero modules trains nothing.
2. Forward and backward run under AMP BF16 through the adapter wrappers.
3. Gradients reach the adapters and nothing else.
4. The adapter actually learns. Not "the loss goes down" (noisy over a dozen steps) but
   three deterministic checks: every adapter tensor receives a non-zero gradient, every
   tensor AdamW could move has moved, and the encoder output changed. A step below the
   float32 ULP (APT on the first block) is reported, not failed.
5. ``merge_()`` holds on the real model: the ratio between merge error and adapter
   effect, measured in the same run, with comparison forwards in fp32 (a bf16 forward
   would measure bf16 rounding instead).

Peak memory and seconds per step of every variant are also recorded.
The variants (:data:`BUILDERS`) cover the reparam factorizations plus ``qk_only`` and the
prompt variants (shallow/deep VPT, APT on all or deep blocks, DA-VPT). Invariant 3 holds
by construction: ``build_adapted_loader`` exposes no ``cache`` parameter.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from evd.data.dataset import WindowItem, build_adapted_loader
from evd.data.decoder import make_decoder
from evd.data.inventory import InventoryRecord, index_by_video_id, read_records
from evd.data.preprocess import vjepa21_preprocess_spec
from evd.data.windows import CausalWindowSampler
from evd.exp.e0c import LabeledVideo, read_labeled_split, read_selected_classes
from evd.exp.tracker import LocalTracker
from evd.models.adapt import (
    AdaptedBackbone,
    AuxiliaryLoss,
    DaVptPEFT,
    PromptSpec,
    ProxyLossSpec,
    ReparamPEFT,
    ReparamSpec,
    make_prompt_strategy,
)
from evd.models.backbones.vjepa21 import vjepa21_backbone
from evd.models.backbones.vjepa21_source import ARCHS
from evd.models.heads.attentive import AttentiveProbe
from evd.train.budget import BudgetSpec, apply_numeric_regime
from evd.train.trainer import Trainer
from evd.types import PreprocessSpec
from evd.utils.io import write_json
from evd.utils.logging import get_logger
from evd.utils.seeding import seed_all

log = get_logger(__name__)

#: What each mini-fit variant builds: a table, not a chain of ``if``; names are those
#: written in ``configs/adapt/``. Coverage is chosen, not exhaustive.
BUILDERS: dict[str, dict[str, Any]] = {
    "lora": {"famiglia": "reparam", "method": "lora", "factorization": "separate"},
    "dora": {"famiglia": "reparam", "method": "dora", "factorization": "separate"},
    "lora_fused": {"famiglia": "reparam", "method": "lora", "factorization": "fused"},
    "lora_fused_nr": {"famiglia": "reparam", "method": "lora", "factorization": "fused_nr"},
    "lora_shared_b": {"famiglia": "reparam", "method": "lora", "factorization": "shared_b"},
    "dora_fused": {"famiglia": "reparam", "method": "dora", "factorization": "fused"},
    "lora_qk_only": {
        "famiglia": "reparam",
        "method": "lora",
        "factorization": "fused",
        "placement": "qk_only",
    },
    "vpt_shallow": {"famiglia": "prompt", "variant": "vpt_shallow"},
    "vpt_deep": {"famiglia": "prompt", "variant": "vpt_deep"},
    "apt": {"famiglia": "prompt", "variant": "apt"},
    # The same APT on deep blocks only: on the real model, 24-block ``apt`` has an exactly
    # zero gradient on the first bank (near-uniform attention over 4608 keys, a numerical
    # cancellation, jobs 533071/533073); this cell makes the previous one diagnosable.
    "apt_late": {"famiglia": "prompt", "variant": "apt", "start_frac": 0.5},
    "da_vpt": {"famiglia": "prompt", "variant": "da_vpt"},
}

VARIANTS = tuple(BUILDERS)

#: Minimum steps for the "every tensor moved" check: at t=0 ``B = 0``, so ``A`` only
#: moves from the second step.
MIN_STEPS = 2


@dataclass(frozen=True)
class MiniFitConfig:
    """What a mini-fit needs; no field has a protocol default."""

    checkpoint: Path
    inventory_csv: Path
    dataset_root: Path
    split: Path
    class_index: Path
    classes_file: Path
    repo: Path | None = None
    arch: str = "vit_large"
    size: int = 384
    num_frames: int = 16
    preprocess_variant: str = "eval_center"
    preprocess_fps: float = 7.5
    hop_s: float = 2.0
    layer: int = -1
    n_videos: int = 8
    """Videos to take windows from; small on purpose (a code path is checked, not a
    performance)."""
    max_windows: int = 32
    max_steps: int = 12
    batch_size: int = 2
    lr: float = 1e-3
    precision: str = "bf16"
    tf32: bool = True
    r: int = 16
    alpha: float = 32.0
    placement: str = "qkvo"
    n_tokens: int = 8
    device: str = "cuda"
    seed: int = 1
    varianti: tuple[str, ...] = VARIANTS
    run_id: str = "f5_minifit_vjepa21l_peft_kna_s1"
    num_workers: int = 0

    def __post_init__(self) -> None:
        if self.arch not in ARCHS:
            raise ValueError(f"unknown arch {self.arch!r}; allowed: {sorted(ARCHS)}")
        unknown = set(self.varianti) - set(VARIANTS)
        if unknown:
            raise ValueError(f"unknown variants: {sorted(unknown)}; allowed {VARIANTS}")
        if not self.varianti:
            raise ValueError("no variant to try")
        for name in ("n_videos", "max_windows", "max_steps", "batch_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, not {getattr(self, name)}")


@dataclass
class VariantOutcome:
    """Verdict on a variant, with the numbers supporting it."""

    variante: str
    famiglia: str
    n_bersagli: int
    n_trainable: int
    frazione_backbone: float
    extra_tokens: int
    loss_iniziale: float
    loss_finale: float
    calo_relativo: float
    """Reported, not judged: over a dozen steps the loss oscillates."""
    parametri_mossi: str
    """``"{moved}/{total}"`` adapter tensors changed from their initialization."""
    parametri_sotto_risoluzione: int
    """How many still tensors are still because the AdamW step is below the float32
    ULP: a fact about the method at this ``lr``, not an injection defect."""
    gradiente_minimo: float
    """Smallest gradient among the adapter tensors; it separates "APT learns little"
    from "APT is not attached"."""
    secondi_per_passo: float
    picco_memoria_gib: float | None
    gradiente_solo_sugli_adattatori: bool
    pesi_dopo_unload: str
    """``"intatti"`` for non-mergeable variants (the backbone is restored after
    unloading); ``"fusi"`` when ``merge_()`` wrote into the weights, where changed
    weights are the correct outcome."""
    effetto_rel_l2: float | None = None
    rumore_regime_rel_l2: float | None = None
    """Difference between two identical forwards under the run precision: the
    denominator that makes the other numbers readable (0 in fp32)."""
    merge_rel_l2: float | None = None
    merge_su_effetto: float | None = None
    griglia_invariata: bool | None = None
    note: dict[str, Any] = field(default_factory=dict)
    """Report notes (factorization, placement, positional policy), which distinguish
    rows with the same ``method`` such as ``fused`` and ``shared_b``."""
    loss_ausiliaria: bool = False
    """Whether the variant contributed its own loss term (DA-VPT)."""
    problemi: list[str] = field(default_factory=list)

    @property
    def passato(self) -> bool:
        return not self.problemi

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "passato": self.passato}


def _fingerprint(module: torch.nn.Module) -> tuple[int, float]:
    """Return a cheap fingerprint of the weights: number of tensors and sum of norms.

    Used only to check that ``unload_()`` restored the previous model without keeping a
    second 1.14 GiB copy.
    """
    with torch.no_grad():
        tensors = list(module.parameters())
        return len(tensors), float(sum(p.detach().double().norm().item() for p in tensors))


def choose_windows(
    videos: Sequence[LabeledVideo],
    *,
    inventory: dict[str, InventoryRecord],
    dataset_root: Path,
    preprocess: PreprocessSpec,
    hop_s: float,
    max_windows: int,
) -> list[WindowItem]:
    """Build a tiny support set from real videos, decoded on the fly.

    One window per video first, then more windows from the same videos, so classes are
    represented.

    :param videos: labelled videos, already filtered on the chosen classes.
    :param inventory: inventory records, indexed by ``video_id``.
    :param dataset_root: root of the inventory ``relative_path`` values.
    :param preprocess: spec declared by the backbone.
    :param hop_s: step between consecutive causal windows.
    :param max_windows: cap on the total.
    :return: items with ``class_label`` set and without ``cache_keys``.
    :raises RuntimeError: if no video hosts a complete window.
    """
    per_video: list[tuple[LabeledVideo, list[Any]]] = []
    for video in videos:
        record = inventory.get(video.video_id)
        if record is None:
            continue
        sampler = CausalWindowSampler.from_spec(preprocess, hop_s)
        windows = sampler.windows_for_video(record.to_meta())
        if windows:
            per_video.append((video, list(windows)))
    if not per_video:
        raise RuntimeError("no video hosts a complete window: check hop_s, num_frames and fps")

    items: list[WindowItem] = []
    round_ = 0
    while len(items) < max_windows:
        added = 0
        for video, windows in per_video:
            if round_ >= len(windows) or len(items) >= max_windows:
                continue
            record = inventory[video.video_id]
            items.append(
                WindowItem(
                    window=windows[round_],
                    video_path=str(dataset_root / record.relative_path),
                    class_label=video.label,
                )
            )
            added += 1
        if added == 0:
            break
        round_ += 1
    return items


def choose_videos(videos: Sequence[LabeledVideo], n_videos: int) -> list[LabeledVideo]:
    """Take ``n_videos`` videos cycling over classes, not the first ones of the list.

    UCF101 splits are sorted by class: the first eight rows would be a single class.

    :param videos: labelled videos, in split order.
    :param n_videos: how many to take.
    :return: the selection, ordered by class and then by appearance.
    """
    per_class: dict[int, list[LabeledVideo]] = {}
    for video in videos:
        per_class.setdefault(video.label, []).append(video)
    chosen: list[LabeledVideo] = []
    round_ = 0
    while len(chosen) < n_videos and any(round_ < len(v) for v in per_class.values()):
        for label in sorted(per_class):
            group = per_class[label]
            if round_ < len(group) and len(chosen) < n_videos:
                chosen.append(group[round_])
        round_ += 1
    return chosen


def create_strategy(variante: str, config: MiniFitConfig, *, n_classes: int, n_blocks: int) -> Any:
    """Build the strategy described by the :data:`BUILDERS` row.

    :param variante: a key of :data:`BUILDERS`.
    :param config: mini-fit configuration, providing rank, alpha, placement, number of
        prompts and seed.
    :param n_classes: head classes; used only by ``da_vpt`` (class-wide proxy banks).
    :param n_blocks: backbone depth, to resolve ``start_frac`` (a fraction of the depth).
    :return: the strategy, not yet applied.
    :raises KeyError: if the variant is not in the table.
    """
    row = dict(BUILDERS[variante])
    family = row.pop("famiglia")
    if family == "reparam":
        return ReparamPEFT(
            ReparamSpec(
                r=config.r,
                alpha=config.alpha,
                placement=row.pop("placement", config.placement),
                seed=config.seed,
                **row,
            )
        )
    fraction = row.pop("start_frac", None)
    if fraction is not None:
        row["start_block"] = int(n_blocks * float(fraction))
    spec = PromptSpec(
        n_tokens=config.n_tokens,
        pos_policy="rope_identity",
        seed=config.seed,
        **row,
    )
    if spec.variant == "da_vpt":
        return DaVptPEFT(spec, ProxyLossSpec(n_classes=n_classes))
    return make_prompt_strategy(spec)


def _output_on(
    backbone: Any, batch: dict[str, Any], config: MiniFitConfig, *, precision: str = "fp32"
) -> torch.Tensor:
    """Run a control forward, outside training and without gradients.

    ``precision`` defaults to fp32 and does not inherit the run one: fidelity comparisons
    are statements about weights, and a bf16 forward would measure bf16 rounding (2.85e-02
    against 3.05e-06 in fp32, job 532971). TF32, enabled globally by
    ``apply_numeric_regime``, is also turned off for the comparison and restored afterwards
    (job 533070).

    :param backbone: backbone to run.
    :param batch: a loader batch; only ``clip`` is used.
    :param config: mini-fit configuration (device and layer).
    :param precision: ``"fp32"`` for fidelity comparisons, ``"bf16"`` to measure the noise
        of the real regime.
    :return: last-layer tokens, in fp32.
    """
    clips = batch["clip"].to(config.device)
    fidelity = precision == "fp32"
    tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    tf32_cudnn = torch.backends.cudnn.allow_tf32
    if fidelity:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=torch.device(config.device).type,
                dtype=torch.bfloat16,
                enabled=precision == "bf16",
            ),
        ):
            out = backbone.encode(clips, layer=config.layer)
        return out.tokens.float()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32_matmul
        torch.backends.cudnn.allow_tf32 = tf32_cudnn


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    scale = a.norm().item()
    return float((a - b).norm().item() / scale) if scale else float((a - b).norm().item())


def adamw_step(gradient: float, *, lr: float, eps: float = 1e-8) -> float:
    """Return how far AdamW moves a parameter, given its gradient magnitude.

    Separates a parameter missing from the optimizer (a wiring bug) from one whose step is
    below the float32 resolution. Adam normalizes by the second moment, so for large
    gradients the step is ``lr``; below ``eps`` it degrades linearly (the APT regime).

    :param gradiente: gradient magnitude of an element.
    :param lr: run learning rate.
    :param eps: AdamW ``eps``; the PyTorch default.
    :return: step magnitude at the first update.
    """
    return lr * abs(gradient) / (abs(gradient) + eps)


def _can_move(param: torch.nn.Parameter, *, lr: float) -> bool:
    """Return whether the AdamW step on this tensor is representable in float32.

    Compares the expected step with the ULP of the largest parameter of the tensor; a
    criterion derived in the run, not a hand-written threshold.

    :param param: the tensor, after the backward pass.
    :param lr: run learning rate.
    :return: ``False`` if the step vanishes in rounding.
    """
    if param.grad is None:
        return False
    maximum = float(param.detach().abs().max())
    ulp = float(torch.finfo(param.dtype).eps) * max(maximum, float(torch.finfo(param.dtype).tiny))
    return adamw_step(float(param.grad.abs().max()), lr=lr) > ulp


def try_variant(
    variante: str,
    *,
    backbone: Any,
    items: Sequence[WindowItem],
    preprocess: PreprocessSpec,
    n_classes: int,
    config: MiniFitConfig,
) -> VariantOutcome:
    """Apply a variant, train it for a few steps and unload it.

    The backbone must be fresh: ``merge_()`` writes into the weights, so reusing it would
    start the next variant from a modified encoder.

    :param variante: one of :data:`VARIANTS`.
    :param backbone: a freshly built V-JEPA 2.1 backbone.
    :param items: the support set, windows of real videos.
    :param preprocess: backbone spec, for the loader.
    :param n_classes: declared head classes.
    :param config: mini-fit configuration.
    :return: the outcome, with numbers and the list of problems found (empty = pass).
    """
    problems: list[str] = []
    fingerprint_before = _fingerprint(backbone.module())
    seed_all(config.seed, deterministic=False)

    loader = build_adapted_loader(
        items,
        decoder=make_decoder("pyav"),
        spec=preprocess,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=torch.Generator().manual_seed(config.seed),
    )
    check_batch = next(iter(loader))

    first = _output_on(backbone, check_batch, config)
    adapted: AdaptedBackbone = create_strategy(
        variante, config, n_classes=n_classes, n_blocks=len(backbone.blocks())
    ).apply(backbone)
    strategy = adapted.strategy
    report = adapted.report
    backbone_total = fingerprint_before[0]
    log.info(
        "%s: %d targets, %d trainable parameters over %d backbone tensors",
        variante,
        report.n_targets,
        report.n_trainable,
        backbone_total,
    )

    head = AttentiveProbe(
        embed_dim=backbone.embed_dim,
        n_classes=n_classes,
        n_queries=1,
        n_heads=16,
        depth=1,
        norm=True,
    ).to(config.device)
    budget = BudgetSpec(
        max_steps=config.max_steps,
        batch_size=config.batch_size,
        lr_grid=(config.lr,),
        precision=config.precision,  # type: ignore[arg-type]
        tf32=config.tf32,
        attn="sdpa",
    )

    # Copy of the adapter parameters before the fit: the "learned" check uses these, not the
    # loss.
    before_fit = [p.detach().clone() for p in strategy.trainable_parameters()]
    if config.max_steps < MIN_STEPS:
        problems.append(
            f"max_steps={config.max_steps}: at least {MIN_STEPS} steps are needed for "
            "`A` to move too (at t=0 B=0, so its gradient is zero at the first step)"
        )

    if config.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    fit = Trainer(device=config.device).fit(
        backbone, head, loader, budget, levels=(config.layer,), adapt=strategy, lr=config.lr
    )
    seconds_per_step = (time.perf_counter() - start) / max(1, fit.steps)
    peak = torch.cuda.max_memory_allocated() / 2**30 if config.device.startswith("cuda") else None

    # (3) gradients reach the adapters and nothing else.
    adapters = {id(p) for p in strategy.trainable_parameters()}
    intruders = [
        name
        for name, p in backbone.module().named_parameters()
        if p.grad is not None and id(p) not in adapters
    ]
    after_fit = list(strategy.trainable_parameters())
    # "Without gradient" includes an exactly zero gradient: such a tensor is as detached as
    # one with no gradient.
    without_gradient = [p for p in after_fit if p.grad is None or not bool(p.grad.abs().max() > 0)]
    only_adapters = not intruders and not without_gradient
    if intruders:
        problems.append(f"gradient on {len(intruders)} backbone parameters: {intruders[:3]}")
    if without_gradient:
        problems.append(
            f"{len(without_gradient)} adapter parameters with zero or missing gradient: "
            "the graph does not reach them"
        )
    min_gradient = min(
        (float(p.grad.abs().max()) for p in after_fit if p.grad is not None),
        default=0.0,
    )

    # (4) the adapter learned: the graph reaches every tensor (above) and the optimizer owns
    # them all (here). A still tensor is a defect only if AdamW could have moved it.
    still = [
        index
        for index, (before_t, after_t) in enumerate(zip(before_fit, after_fit, strict=True))
        if torch.equal(before_t, after_t.detach())
    ]
    below_resolution = [i for i in still if not _can_move(after_fit[i], lr=config.lr)]
    not_updated = [i for i in still if i not in below_resolution]
    moved = f"{len(after_fit) - len(still)}/{len(after_fit)}"
    if not_updated:
        problems.append(
            f"{len(not_updated)} adapter tensors did not move in {fit.steps} "
            f"steps despite a sufficient gradient (indices {not_updated[:5]}): "
            "the optimizer is not updating them"
        )
    if below_resolution:
        log.warning(
            "%s: %d tensors still because the AdamW step is below the float32 "
            "resolution (indices %s, max gradient %.2e, lr %g). Not an injection "
            "defect: the method does not move them at this lr.",
            variante,
            len(below_resolution),
            below_resolution[:5],
            max(float(after_fit[i].grad.abs().max()) for i in below_resolution),
            config.lr,
        )
    initial_loss = float(fit.history[0]["loss"]) if fit.history else float("nan")
    drop = (initial_loss - fit.final_loss) / initial_loss if initial_loss else 0.0

    after = _output_on(backbone, check_batch, config)
    effect = _rel_l2(first, after)
    # Noise of the real regime, reported next to the fp32 comparisons: two identical forwards
    # under the run precision.
    noise = (
        _rel_l2(
            _output_on(backbone, check_batch, config, precision=config.precision),
            _output_on(backbone, check_batch, config, precision=config.precision),
        )
        if config.precision != "fp32"
        else 0.0
    )
    merge_rel: float | None = None
    merge_over_effect: float | None = None
    grid_ok: bool | None = None

    if strategy.can_merge():
        # (5) merge on the real model, with the S3 rule: a ratio between two quantities measured
        # here.
        strategy.merge_()
        merge_rel = _rel_l2(after, _output_on(backbone, check_batch, config))
        merge_over_effect = merge_rel / effect if effect else float("inf")
        if not (merge_over_effect < 1e-2):
            problems.append(
                f"non-negligible merge: rel_L2 {merge_rel:.2e} against an effect of "
                f"{effect:.2e} (ratio {merge_over_effect:.1e})"
            )
    else:
        # For prompts the property is the other one: the sequence grows inside the network and
        # shrinks before leaving it.
        grid_ok = after.shape == first.shape
        if not grid_ok:
            problems.append(
                f"the prompts contaminate the output: {tuple(after.shape)} vs {tuple(first.shape)}"
            )
    if effect <= 0.0:
        problems.append("the adapter has no effect on the encoder output")

    strategy.unload_()
    fingerprint_after = _fingerprint(backbone.module())
    equal = fingerprint_before == fingerprint_after
    if strategy.can_merge():
        # Here the weights must have changed: the adapter was just merged.
        weights = "fusi"
        if equal:
            problems.append("merge_() did not modify the backbone weights")
    else:
        weights = "intatti"
        if not equal:
            problems.append(
                f"backbone not intact after unload_: {fingerprint_before} -> {fingerprint_after}"
            )

    return VariantOutcome(
        variante=variante,
        famiglia=report.family,
        n_bersagli=report.n_targets,
        n_trainable=report.n_trainable,
        frazione_backbone=report.fraction_of(
            sum(p.numel() for p in backbone.module().parameters())
        ),
        extra_tokens=report.extra_tokens,
        loss_iniziale=initial_loss,
        loss_finale=fit.final_loss,
        calo_relativo=drop,
        parametri_mossi=moved,
        parametri_sotto_risoluzione=len(below_resolution),
        gradiente_minimo=min_gradient,
        secondi_per_passo=seconds_per_step,
        picco_memoria_gib=peak,
        gradiente_solo_sugli_adattatori=only_adapters,
        pesi_dopo_unload=weights,
        effetto_rel_l2=effect,
        rumore_regime_rel_l2=noise,
        merge_rel_l2=merge_rel,
        merge_su_effetto=merge_over_effect,
        griglia_invariata=grid_ok,
        note=dict(report.notes),
        loss_ausiliaria=isinstance(strategy, AuxiliaryLoss),
        problemi=problems,
    )


def run_minifit(config: MiniFitConfig) -> dict[str, Any]:
    """Run the mini-fit on all requested variants and register the run.

    :param config: the configuration.
    :return: the metrics, already written to ``runs/{run_id}/metrics.json``.
    :raises RuntimeError: never for a negative outcome (recorded with
        ``status="failed"``); only if the run cannot execute (missing checkpoint, no
        window, inconsistent split).
    """
    seed_all(config.seed, deterministic=False)
    architecture = ARCHS[config.arch]
    preprocess = vjepa21_preprocess_spec(
        config.preprocess_variant,  # type: ignore[arg-type]
        fps=config.preprocess_fps,
        num_frames=config.num_frames,
        size=config.size,
        dtype="fp32",
    )
    classes = read_selected_classes(config.class_index, config.classes_file)
    videos = choose_videos(read_labeled_split(config.split, classes), config.n_videos)
    inventory = index_by_video_id(read_records(config.inventory_csv))
    items = choose_windows(
        videos,
        inventory=inventory,
        dataset_root=config.dataset_root,
        preprocess=preprocess,
        hop_s=config.hop_s,
        max_windows=config.max_windows,
    )
    log.info("support set: %d windows from %d videos", len(items), len(videos))

    def build_backbone() -> Any:
        """Return an encoder freshly loaded from the checkpoint, for every variant."""
        return vjepa21_backbone(
            arch=config.arch,
            checkpoint=config.checkpoint,
            state_dict_key=architecture.ckpt_key,
            size=config.size,
            num_frames=config.num_frames,
            preprocess_variant=config.preprocess_variant,
            preprocess_fps=config.preprocess_fps,
            preprocess_dtype="fp32",
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

    declared_budget = BudgetSpec(
        max_steps=config.max_steps,
        batch_size=config.batch_size,
        lr_grid=(config.lr,),
        precision=config.precision,  # type: ignore[arg-type]
        tf32=config.tf32,
        attn="sdpa",
    )
    declared_budget.validate_against_env()
    regime = apply_numeric_regime(declared_budget, config.device)

    tracker = LocalTracker()
    handle = tracker.start(
        config.run_id,
        asdict(config),
        # Not an official run: it is an integration test, most useful on a dirty working tree.
        official=False,
        regime={**regime, "size": config.size, "protocol": "integration-gpu-f5"},
    )

    outcomes: list[VariantOutcome] = []
    for variant in config.varianti:
        backbone = build_backbone()
        try:
            outcomes.append(
                try_variant(
                    variant,
                    backbone=backbone,
                    items=items,
                    preprocess=preprocess,
                    n_classes=len(classes),
                    config=config,
                )
            )
        finally:
            # 1.14 GiB per copy on ViT-L: otherwise two encoders coexist and the reported peak memory
            # is not that of a real run.
            del backbone
            if config.device.startswith("cuda"):
                torch.cuda.empty_cache()
    for outcome in outcomes:
        log.info(
            "%-14s %-7s targets=%d  trainable=%d (%.3f%%)  moved=%s  grad_min=%.1e  "
            "effect=%.2e  merge=%s  noise=%.2e  %.2f s/step  peak=%s",
            outcome.variante,
            "OK" if outcome.passato else "FAILED",
            outcome.n_bersagli,
            outcome.n_trainable,
            100 * outcome.frazione_backbone,
            outcome.parametri_mossi,
            outcome.gradiente_minimo,
            outcome.effetto_rel_l2 or 0.0,
            f"{outcome.merge_rel_l2:.2e}" if outcome.merge_rel_l2 is not None else "n/a",
            outcome.rumore_regime_rel_l2 or 0.0,
            outcome.secondi_per_passo,
            f"{outcome.picco_memoria_gib:.2f} GiB" if outcome.picco_memoria_gib else "n/d",
        )
        for problem in outcome.problemi:
            log.error("  %s: %s", outcome.variante, problem)

    passed = all(outcome_label.passato for outcome_label in outcomes)
    metrics: dict[str, Any] = {
        "arch": config.arch,
        "n_videos": len(videos),
        "n_finestre": len(items),
        "n_classi": len(classes),
        "passato": passed,
        "varianti": [outcome.as_dict() for outcome in outcomes],
        "regime": regime,
    }
    write_json(handle.dir / "minifit.json", metrics)
    tracker.finish(
        handle,
        metrics,
        status="ok" if passed else "failed",
        extra={"exp": "f5-minifit", "integration_only": True},
    )
    return metrics
