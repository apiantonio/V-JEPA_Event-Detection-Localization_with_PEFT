"""``BudgetSpec``: the training budget is data, not a convention.

* ``tf32``: on the A100 TF32 is 7.28× faster than fp32 (7.63× on the H100); timings
  without this declaration are not comparable. Default ``False``: reproducibility
  first.
* ``attn``: ``bf16`` produces NaN with eager attention on V-JEPA 2.1 ViT-B/L/g, so the
  ``(bf16, eager)`` pair is rejected here.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from math import ceil
from typing import Any, Literal

from evd.utils import env
from evd.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class BudgetSpec:
    max_steps: int
    batch_size: int
    grad_accum: int = 1
    lr_grid: tuple[float, ...] = (1e-3,)
    """Learning-rate grid of the selection protocol (docs/04 §5). Its cardinality
    must be identical across PEFT families (invariant 5)."""
    weight_decay: float = 0.0
    lr_schedule: Literal["constant", "cosine", "cosine_cyclic"] = "constant"
    """How the LR varies over the steps. ``constant`` keeps it fixed. ``cosine``
    replicates `WarmupCosineLRSchedule`
    (`vjepa2/evals/video_classification_frozen/eval.py:490-515`) with zero warmup
    steps (the k400 reference value) and decays from `lr` to 0 with a half cosine
    over `max_steps` steps; see :func:`lr_at_step`.

    ``cosine_cyclic`` (ADR-062) repeats that half cosine every ``lr_cycle_steps``
    steps (SGDR warm restarts with a constant period), so a model trained for one
    epoch continues instead of restarting: its first cycle is identical to
    ``cosine`` with ``max_steps = N``, as a test checks step by step."""
    lr_cycle_steps: int | None = None
    """Period of the cyclic cosine, in steps. Required with
    ``lr_schedule='cosine_cyclic'`` and forbidden otherwise. On ``edp_v2`` it is
    5524, i.e. one epoch."""
    precision: Literal["bf16", "fp32"] = "bf16"
    tf32: bool = False
    attn: Literal["sdpa", "eager"] = "sdpa"
    early_stop_patience: int | None = None
    """``None`` disables early stopping. Otherwise the number of consecutive
    evaluations without improvement after which the loop stops."""
    epochs: int | None = None
    """If set, ``max_steps`` is derived from it and the support size instead of
    being constant (few-shot regime).

    With a constant ``max_steps``, k=1 would see 152 epochs and k=full 1.5, mixing
    the amount of supervision with the number of passes (E0e, job 534110). The
    denominator is the whole support of the cell, so LR trials and the final refit
    run the same number of steps. ``None`` (default) means fixed steps."""
    max_steps_cap: int | None = None
    """Cap on the steps derived from ``epochs``; the k at which it starts to bind
    must be declared."""
    num_workers: int = 0
    """Number of worker processes decoding video for the ``DataLoader``.

    Part of the budget because it changes the pixels: the worker index enters the
    augmentation seed and the per-worker draw counters, so it enters the
    ``config_sha`` and must be identical across compared cells (invariant 5). It is
    never read from the environment (it would make the pixels depend on the SLURM
    allocation).

    The default ``0`` keeps the identity of registered runs but is not right for a
    real run: without workers decoding precedes the forward (ADR-038). The
    ``edp_v1`` configs declare ``8``."""
    pin_memory: bool = False
    """Whether the ``DataLoader`` pins batches in page-locked memory.

    It does not change the pixels, but it stays declared in the budget instead of
    being hard-wired by the orchestrator. Default ``False`` for registered runs;
    measure with ``scripts/e0d_throughput.py`` and enable it in the config."""
    persistent_workers: bool = False
    """Whether workers survive across ``DataLoader`` epochs.

    It does not change the pixels; it matters for validation, which runs six times
    per budget epoch on ``edp_v2``. ``__post_init__`` rejects it with
    ``num_workers=0``."""
    eval_every: int = 1
    """Validation period in steps. Only used when ``fit`` receives a
    ``val_loader``; it gives the unit of ``early_stop_patience`` and of model
    selection (:attr:`model_selection`)."""
    model_selection: Literal["last", "val_loss", "val_auroc", "val_balanced_accuracy"] = "last"
    """Which weights are kept at the end of training.

    ``"last"``: those of the last step (historical behaviour and default).

    ``"val_auroc"``: those of the step maximizing validation AUROC, the laboratory
    protocol rule (§6, ADR-047).

    ``"val_loss"``: those of the minimum-loss step (not protocol).

    ``"val_balanced_accuracy"``: those of the step maximizing the balanced accuracy of the
    class head over the anomaly classes, on the positive validation windows, with the
    class of highest logit among the anomaly classes (``Normal`` excluded): the selection
    of the class-routing heads (``docs/varie/piano_localizzazione_per_classe_20261021.md``
    §5.5). Requires a multiclass or multitask objective.

    Not early stopping: this field decides which weights to keep, not when to stop.
    A value other than ``"last"`` without a ``val_loader`` is rejected by the
    ``Trainer``."""

    class_balance: Literal["none", "equal"] = "none"
    """Class factor of the sampling weights. ``"none"`` keeps ``w_video · w_overlap``
    (invariant 6, the protocol regime); ``"equal"`` multiplies each chunk weight by
    ``(1 / K) / M_c``, with ``M_c`` the weight mass of its class (``Normal`` included) and
    ``K`` the number of classes present, so each class receives ``1 / K`` of the draws. A
    declared deviation from invariant 6, decided on 21 October 2026 for the class-routing
    heads only (plan §5.5)."""

    def __post_init__(self) -> None:
        if self.class_balance not in ("none", "equal"):
            raise ValueError(f"class_balance must be 'none' or 'equal', not {self.class_balance!r}")
        for name in ("max_steps", "batch_size", "grad_accum", "eval_every"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, not {getattr(self, name)}")
        if self.early_stop_patience is not None and self.early_stop_patience <= 0:
            raise ValueError(
                f"early_stop_patience must be > 0 or None (no early stopping), "
                f"not {self.early_stop_patience}"
            )
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, not {self.num_workers}")
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError(
                "persistent_workers=True with num_workers=0 does not affect the computation: "
                "there are no workers to keep alive, and the DataLoader would raise. "
                "It would be misleading metadata (CLAUDE.md §3.1)"
            )
        for name in ("epochs", "max_steps_cap"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be > 0 or None, not {value}")
        if self.max_steps_cap is not None and self.epochs is None:
            raise ValueError(
                "max_steps_cap without epochs does not affect the computation: with fixed "
                "steps the cap would be misleading metadata (CLAUDE.md §4). Remove the "
                "cap or declare epochs"
            )
        if not self.lr_grid:
            raise ValueError("empty lr_grid: the LR selection protocol is part of the budget")
        if self.precision == "bf16" and self.attn == "eager":
            raise ValueError(
                "config rejected: (precision=bf16, attn=eager) produces NaN on the "
                "V-JEPA 2.1 ViT-B/L/g variants (verified during the HF port). Use "
                "attn='sdpa' or precision='fp32'."
            )
        if self.lr_schedule not in ("constant", "cosine", "cosine_cyclic"):
            raise ValueError(
                f"unknown lr_schedule: {self.lr_schedule!r}, expected 'constant', "
                f"'cosine' or 'cosine_cyclic'"
            )
        if self.lr_schedule == "cosine_cyclic":
            if not self.lr_cycle_steps or self.lr_cycle_steps <= 0:
                raise ValueError(
                    "lr_schedule='cosine_cyclic' requires lr_cycle_steps > 0: without a period "
                    "the cosine has no cycle"
                )
        elif self.lr_cycle_steps is not None:
            raise ValueError(
                f"lr_cycle_steps={self.lr_cycle_steps} with lr_schedule={self.lr_schedule!r}: "
                f"nobody would read it, and a field that does not affect the computation "
                f"is misleading metadata"
            )

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum

    def validate_against_env(self) -> None:
        """Check the machine-level ``EVD_ALLOW_BF16`` kill switch.

        :raises ValueError: if the config requests bf16 and ``EVD_ALLOW_BF16=0``.
        """
        if self.precision == "bf16" and not env.allow_bf16():
            raise ValueError(
                "this config requests bf16 but EVD_ALLOW_BF16=0 (hpc/env.sh default). "
                "If bf16 is really needed set EVD_ALLOW_BF16=1 in the job environment, "
                "knowing that it produces NaN with eager attention."
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def steps_per_budget(budget: BudgetSpec, n_train: int) -> int:
    """Return how many steps this cell runs, given its support.

    With ``epochs=None`` it is ``budget.max_steps``; otherwise steps are derived from the
    epochs and capped. Shared by the cell and by the cost planner.

    :param budget: budget of the cell.
    :param n_train: support windows used for training.
    :return: effective ``max_steps``, at least 1.
    :raises ValueError: if ``n_train`` is not positive.
    """
    if n_train <= 0:
        raise ValueError(f"n_train must be > 0, not {n_train}")
    if budget.epochs is None:
        return budget.max_steps
    steps = max(1, ceil(budget.epochs * n_train / budget.effective_batch))
    if budget.max_steps_cap is not None:
        steps = min(steps, budget.max_steps_cap)
    return steps


def lr_at_step(
    ref_lr: float,
    schedule: str,
    step: int,
    max_steps: int,
    cycle_steps: int | None = None,
) -> float:
    """Return the learning rate to apply at ``step`` according to the schedule.

    ``constant`` always returns ``ref_lr``. ``cosine`` replicates
    `WarmupCosineLRSchedule` (`vjepa2/evals/video_classification_frozen/eval.py:490-515`)
    with ``warmup_steps = 0`` and ``final_lr = 0``: a half cosine from ``ref_lr`` to 0
    between step 1 and ``max_steps``. ``cosine_cyclic`` repeats it every ``cycle_steps``.

    :param ref_lr: peak learning rate of the cell.
    :param schedule: ``"constant"``, ``"cosine"`` or ``"cosine_cyclic"``.
    :param cycle_steps: period, required for ``cosine_cyclic``.
    :param step: current step, 1-based.
    :param max_steps: total steps of the cell.
    :return: the learning rate to write into ``opt.param_groups``.
    :raises ValueError: on an unknown schedule or a missing period.
    """
    if schedule == "constant":
        return ref_lr
    if schedule == "cosine":
        progress = min(max(step, 0), max_steps) / max_steps
        return 0.5 * ref_lr * (1.0 + math.cos(math.pi * progress))
    if schedule == "cosine_cyclic":
        if not cycle_steps or cycle_steps <= 0:
            raise ValueError(
                f"cosine_cyclic without period (cycle_steps={cycle_steps!r}): the cyclic "
                f"cosine has no cycle, and the value is undefined"
            )
        # `(step - 1) % cycle + 1` rather than `step % cycle`: otherwise step `cycle` would give
        # the peak LR where the cycle must close at zero, and the first cycle would not match
        # `cosine` (ADR-062).
        inside = (max(step, 1) - 1) % cycle_steps + 1
        return 0.5 * ref_lr * (1.0 + math.cos(math.pi * inside / cycle_steps))
    raise ValueError(
        f"unknown lr_schedule: {schedule!r}, expected 'constant', 'cosine' or 'cosine_cyclic'"
    )


def apply_numeric_regime(budget: BudgetSpec, device: str) -> dict[str, Any]:
    """Apply the numeric regime and return what goes into ``meta.json``.

    Every run declares ``(precision, tf32, attn, size)`` (docs/08 §4.3), otherwise
    efficiency measurements are not commensurable.
    """
    import torch

    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = budget.tf32
        torch.backends.cudnn.allow_tf32 = budget.tf32
    elif budget.tf32:
        log.warning("tf32=True ignored: device=%s (TF32 exists only on Ampere+)", device)
    return {
        "precision": budget.precision,
        "tf32": bool(budget.tf32),
        "attn": budget.attn,
        "device": device,
        "torch": torch.__version__,
    }
