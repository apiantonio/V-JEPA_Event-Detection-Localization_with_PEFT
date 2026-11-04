"""The training loop, compact and self-written.

A single loop for probes and PEFT: the adaptation strategy decides what is trainable.
No Lightning or HF Trainer: budget parity across PEFT families must be
readable in one file. The loop covers steps, gradient accumulation, the applied numeric
regime (``torch.autocast`` when ``budget.precision == "bf16"``, with FP32 weights and
inputs and SDPA), history, validation with early stopping, model selection and
intra-training resumption (``train/checkpoint.py``); LR selection lives in
``train/selection.py``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from evd.data.dataset import cached_output_from_batch, enable_fast_replay
from evd.models.adapt.base import (
    AuxiliaryLoss,
    freezes_backbone,
    preserves_backbone_output,
)
from evd.models.backbones.base import encode_multilevel
from evd.train.budget import BudgetSpec, apply_numeric_regime, lr_at_step
from evd.train.checkpoint import Checkpointer, TrainingState, restore_adapter_state
from evd.types import BackboneOutput
from evd.utils.logging import get_logger
from evd.utils.seeding import rng_state, set_rng_state

log = get_logger(__name__)


@dataclass
class FitResult:
    steps: int
    """Executed steps, fewer than ``budget.max_steps`` with early stopping."""
    final_loss: float
    lr: float
    """Peak learning rate of the cell (`lr_grid` or explicit override), not
    necessarily the one applied at every step: with a non-constant schedule the
    per-step value is in ``history[i]["lr"]`` (:func:`~evd.train.budget.lr_at_step`)."""
    history: list[dict[str, Any]] = field(default_factory=list)
    regime: dict[str, Any] = field(default_factory=dict)
    trainable_params: int = 0
    best_val: float | None = None
    """Best validation loss, or ``None`` without ``val_loader``. It stays the loss
    even when model selection looks at AUROC (``select_lr`` minimizes it)."""
    best_step: int | None = None
    """Step that produced it."""
    val_auroc: float | None = None
    """Validation AUROC at the last evaluation, or ``None`` if not computed (no
    ``val_loader``, non-binary objective, degenerate ground truth). The full series
    is in ``history[i]["val_auroc"]``."""
    selection_metric: str = "last"
    """Criterion that chose the returned weights (:attr:`BudgetSpec.model_selection`)."""
    selected_step: int | None = None
    """Step whose weights are in the model on return. Equal to :attr:`steps` with
    ``selection_metric == "last"``; otherwise the step that optimized the criterion,
    to be reported next to the numbers."""
    selected_value: float | None = None
    """Criterion value at the selected step."""
    selected_scores: dict[str, float] = field(default_factory=dict)
    """Validation ``window_id -> logit`` at the selected step, i.e. of the model this
    ``FitResult`` returns (ADR-053); empty with ``model_selection == "last"``.

    These are logits, not anomaly scores: ``train`` does not depend on ``eval``
    (docs/02 §5), so readers convert them with
    :func:`evd.eval.scorers.anomaly_from_logits`. They avoid a seventh validation pass
    to estimate the threshold. Populated only for ``objective == "binary"``."""
    stopped_early: bool = False
    """True if patience ran out before ``max_steps`` (to be declared in E3-X
    comparisons, invariant 5)."""
    resumed_from: int | None = None
    """Step this segment resumed from, or ``None`` for a full fit; a resumed run
    executed ``steps - resumed_from`` steps in this process (ADR-025 point 4)."""
    interrupted: bool = False
    """True if the loop stopped on request (signal or flag) with budget left: the
    checkpoint was written and training is not finished. Distinct from
    ``stopped_early``, which is a protocol decision."""


#: What the loop asks the model to predict.
#:
#: ``binary`` is the active regime (ADR-031 point 4): one logit, target
#: ``batch["label"]`` (positive if the chunk intersects an annotated interval), BCE
#: without ``pos_weight`` (ADR-033 §3). ``multiclass`` remains for the E0c sanity check
#: on UCF101. It is explicit rather than deduced from ``head.n_classes``.
Objective = Literal["binary", "multiclass", "multitask"]


def _loss(objective: Objective) -> nn.Module:
    """:return: the loss of the declared objective.
    :raises ValueError: on an unknown objective.
    """
    if objective == "binary":
        return nn.BCEWithLogitsLoss()
    if objective in ("multiclass", "multitask"):
        # For `multitask` this is the loss of the class task; the binary task uses BCE
        # (`Trainer._multitask_loss`).
        return nn.CrossEntropyLoss()
    raise ValueError(
        f"unknown objective: {objective!r} (expected 'binary' | 'multiclass' | 'multitask')"
    )


def _target(batch: dict[str, Any], objective: Objective, device: str) -> torch.Tensor:
    """Return the batch target, in the dtype required by the objective loss.

    :raises ValueError: if labels are missing (the dataset sentinel ``-1`` would silently
        become a BCE target).
    """
    key = "label" if objective == "binary" else "class_label"
    target = batch[key].to(device)
    if int(target.min()) < 0:
        raise ValueError(
            f"training batch without labels ({key}): windows must carry them (WindowItem.{key})"
        )
    return target.to(torch.float32) if objective == "binary" else target


def _logit(logits: torch.Tensor, objective: Objective) -> torch.Tensor:
    """Return the head outputs in the shape required by the objective loss.

    :raises ValueError: if a binary objective receives a head with more than one logit.
    """
    if objective != "binary":
        return logits
    if logits.ndim != 2 or logits.shape[1] != 1:
        raise ValueError(
            f"objective='binary' but the head produces {tuple(logits.shape)}: [B, 1] is required. "
            f"Binary output is protocol (ADR-031 point 4), not a preference"
        )
    return logits[:, 0]


def _cycle(loader: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Cycle over the loader until the budget steps are exhausted (a step is not an epoch).

    :raises ValueError: if the loader is empty.
    """
    while True:
        empty = True
        for batch in loader:
            empty = False
            yield batch
        if empty:
            raise ValueError("empty loader: no training windows")


def _check_budget(budget: BudgetSpec, val_loader: Any | None, objective: Objective) -> None:
    """Check that budget fields assuming a validation actually have one.

    A field that does not affect the computation is misleading metadata (CLAUDE.md §3.1):
    ``early_stop_patience`` or ``model_selection: val_auroc`` without validation would be
    documented but not applied.

    :param budget: run budget.
    :param val_loader: validation loader, or ``None``.
    :param objective: ``"binary"`` or ``"multiclass"``.
    :raises ValueError: if a budget field assumes a missing validation, or binary AUROC is
        requested on a multi-output head.
    """
    if budget.early_stop_patience is not None and val_loader is None:
        raise ValueError(
            "budget.early_stop_patience is declared but there is no val_loader to stop on: "
            "pass a validation loader (docs/04 §5) or remove the field from the config "
            "instead of keeping metadata that does not affect the computation"
        )
    if budget.model_selection != "last" and val_loader is None:
        raise ValueError(
            f"budget.model_selection = {budget.model_selection!r} but there is no "
            f"val_loader to measure the criterion on: pass a validation loader, or set "
            f"the field back to 'last' instead of keeping metadata that does not affect "
            f"the computation"
        )
    if budget.model_selection == "val_auroc" and objective == "multiclass":
        raise ValueError(
            "model_selection='val_auroc' with objective='multiclass': binary AUROC "
            "is undefined on a multi-output head. The protocol criterion "
            "(ADR-047 §6) applies to the binary detector"
        )
    if budget.model_selection == "val_balanced_accuracy" and objective == "binary":
        raise ValueError(
            "model_selection='val_balanced_accuracy' with objective='binary': a binary head "
            "predicts no class. Use a multiclass or multitask objective"
        )


class Trainer:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def fit(
        self,
        backbone: Any | None,
        head: nn.Module,
        train_loader: DataLoader[Any],
        budget: BudgetSpec,
        *,
        val_loader: DataLoader[Any] | None = None,
        levels: tuple[int, ...] = (-1,),
        adapt: Any | None = None,
        lr: float | None = None,
        checkpoint: Checkpointer | None = None,
        objective: Objective = "multiclass",
        normal_idx: int | None = None,
    ) -> FitResult:
        """Train ``head`` (plus the adaptation parameters, if any).

        An adaptation without trainable parameters (``None`` or
        :class:`~evd.models.adapt.base.NoAdapt`) means a frozen backbone: the forward runs
        under ``no_grad`` and the loader may provide cached ``features``; adapted runs reject
        them (invariant 3, ADR-007).

        :param backbone: the backbone, or ``None`` if the loader provides features.
        :param head: head to train; moved to the device.
        :param train_loader: training batches with labels.
        :param budget: declared budget; ``precision`` and ``tf32`` are applied, not only
            recorded.
        :param val_loader: validation loader (never the evaluation set, invariant 6); used for
            early stopping, model selection and LR selection.
        :param levels: encoder blocks to read tokens from, in concatenation order (a length-1
            tuple is the single-layer case, ADR-029).
        :param adapt: adaptation strategy, if any.
        :param lr: chosen learning rate; ``None`` uses the first grid value.
        :param checkpoint: if given, the loop resumes from a compatible checkpoint on disk and
            writes one at step boundaries (ADR-025 point 4), also restoring the consumed
            batches.
        :param objective: what is predicted: ``binary`` (protocol runs, ADR-031 point 4),
            ``multiclass`` (the historical default, cross-entropy on ``batch["class_label"]``)
            or ``multitask`` (a head with ``forward_tasks``, ``combine`` and ``after_step``,
            such as :class:`~evd.models.heads.class_queries.BinaryAndClassHead`: BCE on the
            ``binary`` output plus cross-entropy on the ``classes`` output, balanced by the
            head).
        :param normal_idx: index of the ``Normal`` class, required by
            ``model_selection: val_balanced_accuracy`` (the anomaly classes are the others).
        :return: executed steps, final loss, history, numeric regime and best validation loss.
        :raises ValueError: on an empty loader, a batch ambiguous between ``clip`` and
            ``features``, missing labels, a multi-logit head in binary mode, or budget fields
            requiring a missing ``val_loader``.
        """
        regime = apply_numeric_regime(budget, self.device)
        _check_budget(budget, val_loader, objective)
        if budget.model_selection == "val_balanced_accuracy" and normal_idx is None:
            raise ValueError("model_selection='val_balanced_accuracy' requires normal_idx")
        if objective == "multitask" and not callable(getattr(head, "forward_tasks", None)):
            raise ValueError(
                f"objective='multitask' but {type(head).__name__} has no forward_tasks: use a "
                f"head with a binary and a class output"
            )
        head = head.to(self.device)
        # The question is whether the adaptation requires gradients, not `adapt is None`
        # (`NoAdapt` is a real strategy).
        frozen = freezes_backbone(adapt)
        # A different question: an adaptation may require no gradients and still change what
        # the backbone computes (prompts). Second line of defence after the runner.
        cache_allowed = preserves_backbone_output(adapt)
        params: list[nn.Parameter] = list(head.parameters())
        if adapt is not None:
            params += list(adapt.trainable_parameters())
        n_trainable = sum(p.numel() for p in params if p.requires_grad)
        chosen_lr = lr if lr is not None else budget.lr_grid[0]
        if lr is None and len(budget.lr_grid) > 1:
            log.warning(
                "lr_grid has %d values but no selection was run: using the first one (%g)",
                len(budget.lr_grid),
                chosen_lr,
            )
        opt = torch.optim.AdamW(params, lr=chosen_lr, weight_decay=budget.weight_decay)
        loss_fn = _loss(objective)
        # Methods with their own loss (DA-VPT) are found via `AuxiliaryLoss`, not by name.
        aux = adapt if isinstance(adapt, AuxiliaryLoss) else None

        head.train()
        it = _cycle(train_loader)
        history: list[dict[str, Any]] = []
        last_loss = float("nan")
        best_val: float | None = None
        val_auroc: float | None = None
        selected_step: int | None = None
        selected_value: float | None = None
        chosen_state: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]] | None = None
        chosen_scores: dict[str, float] = {}
        best_step: int | None = None
        without_improvement = 0
        executed = 0
        stopped_early = False
        interrupted = False
        consumed = 0
        first_step = 1
        resumed_from: int | None = None
        try:
            head_dtype = next(head.parameters()).dtype
        except StopIteration:
            head_dtype = torch.float32

        state = checkpoint.load() if checkpoint is not None else None
        if state is not None:
            head.load_state_dict(state.head)
            restore_adapter_state(adapt, state.adapter)
            opt.load_state_dict(state.optimizer)
            # `load_state_dict` also restores the learning rate of the previous segment: if it
            # differs from the schedule value at the last saved step, the selection was not
            # reproduced and this is not the same training.
            expected = lr_at_step(
                chosen_lr,
                budget.lr_schedule,
                state.step,
                budget.max_steps,
                budget.lr_cycle_steps,
            )
            restored = {float(g["lr"]) for g in opt.param_groups}
            if restored != {float(expected)}:
                raise ValueError(
                    f"the checkpoint was written with learning rate {sorted(restored)} "
                    f"but this segment would use {expected} at step {state.step}: the "
                    f"selection did not reproduce the same choice, so it is not the same "
                    f"training"
                )
            history = list(state.history)
            best_val, best_step = state.best_val, state.best_step
            without_improvement = state.senza_miglioramento
            last_loss = state.last_loss
            # Model selection resumes where it was (see `TrainingState`).
            selected_step, selected_value = state.selected_step, state.selected_value
            if state.selected_head:
                chosen_state = (dict(state.selected_head), dict(state.selected_adapter))
            # Selected-step scores travel with the selection, so a resumption going straight to
            # evaluation (ADR-050) does not rerun validation (ADR-053).
            chosen_scores = dict(state.selected_scores)
            # Consumed batches are re-consumed by running `_cycle` idle, since it has no state to
            # save. When the segment is already complete the rewind is skipped: it would only decode
            # data that is thrown away, and the global RNG state is restored right below anyway.
            if state.step < budget.max_steps:
                # Fast replay marks the skipped batches so the dataset does not decode them;
                # the sampler and its generator advance exactly as before.
                fast = enable_fast_replay(train_loader, state.batches_consumed)
                if fast:
                    log.info(
                        "fast resume: %d previous batches without decoding", state.batches_consumed
                    )
                for _ in range(state.batches_consumed):
                    skipped = next(it)
                    if fast and skipped.get("_replay") is not True:
                        raise RuntimeError("fast resume: batch not marked as replay")
            else:
                log.info(
                    "resuming at step %d of %d: no step to run, the "
                    "rewind of %d batches is skipped",
                    state.step,
                    budget.max_steps,
                    state.batches_consumed,
                )
            consumed = state.batches_consumed
            # RNGs are restored after the rewind, which consumes the loader generator.
            set_rng_state(state.rng)
            executed = resumed_from = state.step
            first_step = state.step + 1

        for step in range(first_step, budget.max_steps + 1):
            # The schedule is applied here, before the forward: `opt.step()` reads
            # `param_groups[...]["lr"]`, and the applied value is what gets logged.
            step_lr = lr_at_step(
                chosen_lr, budget.lr_schedule, step, budget.max_steps, budget.lr_cycle_steps
            )
            for opt_group in opt.param_groups:
                opt_group["lr"] = step_lr
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for _ in range(budget.grad_accum):
                batch = next(it)
                consumed += 1
                target = (
                    None if objective == "multitask" else _target(batch, objective, self.device)
                )
                # `budget.precision` is applied, not only declared: autocast with FP32 weights and
                # inputs, never `encoder.to(torch.bfloat16)`, which breaks the 3D-RoPE. No GradScaler
                # (needed for fp16, not bf16).
                if aux is not None:
                    aux.reset_auxiliary()
                with torch.autocast(
                    device_type=torch.device(self.device).type,
                    dtype=torch.bfloat16,
                    enabled=budget.precision == "bf16",
                ):
                    feats = self._features_from_batch(
                        backbone,
                        batch,
                        levels=levels,
                        frozen=frozen,
                        cache_allowed=cache_allowed,
                        dtype=head_dtype,
                    )
                    if objective == "multitask":
                        losses = self._multitask_loss(head, feats, batch, loss_fn)
                        loss = head.combine(losses, feats.tokens, 1.0 / budget.grad_accum)
                    else:
                        logits = head(feats)
                        loss = loss_fn(_logit(logits, objective), target)
                    if aux is not None:
                        # After the forward that collected it, inside autocast like the rest of the graph.
                        loss = loss + aux.auxiliary_loss(target)
                    loss = loss / budget.grad_accum
                loss.backward()
                total += float(loss.detach()) * budget.grad_accum
            opt.step()
            if objective == "multitask":
                head.after_step()
            last_loss = total / budget.grad_accum
            row: dict[str, Any] = {"step": step, "loss": last_loss, "lr": step_lr}
            if objective == "multitask":
                row["task_weights"] = head.task_weights()

            # Evaluation happens at step boundaries, after `opt.step()`.
            if val_loader is not None and step % budget.eval_every == 0:
                val, auroc, window_scores, balanced = self._val_metrics(
                    backbone,
                    head,
                    val_loader,
                    loss_fn,
                    budget=budget,
                    levels=levels,
                    cache_allowed=cache_allowed,
                    dtype=head_dtype,
                    objective=objective,
                    normal_idx=normal_idx,
                )
                row["val_loss"] = val
                if auroc is not None:
                    row["val_auroc"] = auroc
                    val_auroc = auroc
                if balanced is not None:
                    row["val_balanced_accuracy"] = balanced
                if best_val is None or val < best_val:
                    best_val, best_step, without_improvement = val, step, 0
                else:
                    without_improvement += 1
                # Model selection (ADR-047 §6) is distinct from early stopping: it decides which weights
                # to keep. The state is snapshotted here, where criterion and weights refer to the same
                # model.
                if budget.model_selection != "last":
                    criterion = {
                        "val_loss": val,
                        "val_auroc": auroc,
                        "val_balanced_accuracy": balanced,
                    }[budget.model_selection]
                    better = criterion is not None and not math.isnan(criterion)
                    if better and selected_value is not None:
                        better = (
                            criterion < selected_value
                            if budget.model_selection == "val_loss"
                            else criterion > selected_value
                        )
                    if better:
                        assert criterion is not None
                        selected_value, selected_step = float(criterion), step
                        chosen_state = (
                            {k: v.detach().clone().cpu() for k, v in head.state_dict().items()},
                            self._adapter_state(adapt),
                        )
                        # Validation scores belong to these weights: snapshot them at the same point.
                        chosen_scores = dict(window_scores)
                        log.info(
                            "selection: step %d, %s = %.6f",
                            step,
                            budget.model_selection,
                            criterion,
                        )
                if (
                    budget.early_stop_patience is not None
                    and without_improvement >= budget.early_stop_patience
                ):
                    history.append(row)
                    executed, stopped_early = step, True
                    log.info(
                        "early stop at step %d/%d: %d evaluations without improvement "
                        "(best %.6f at step %d)",
                        step,
                        budget.max_steps,
                        without_improvement,
                        best_val,
                        best_step,
                    )
                    break

            history.append(row)
            executed = step
            if step == 1 or step % max(1, budget.max_steps // 5) == 0:
                log.info("step %d/%d  loss=%.4f", step, budget.max_steps, last_loss)

            # Same step boundary as validation: never resume from a partial gradient.
            if checkpoint is not None:
                if checkpoint.due(step, last=step == budget.max_steps):
                    checkpoint.save(
                        TrainingState(
                            step=step,
                            batches_consumed=consumed,
                            head=head.state_dict(),
                            optimizer=opt.state_dict(),
                            rng=rng_state(),
                            adapter=self._adapter_state(adapt),
                            history=history,
                            best_val=best_val,
                            best_step=best_step,
                            senza_miglioramento=without_improvement,
                            last_loss=last_loss,
                            selected_step=selected_step,
                            selected_value=selected_value,
                            selected_head={} if chosen_state is None else chosen_state[0],
                            selected_adapter={} if chosen_state is None else chosen_state[1],
                            selected_scores=dict(chosen_scores),
                        )
                    )
                # `interrupted` means budget left: a request on the last step does not mark a finished
                # training as interrupted.
                if checkpoint.should_stop() and step < budget.max_steps:
                    interrupted = True
                    log.info(
                        "interrupted at step %d/%d: training is not finished, "
                        "resumption restarts from here",
                        step,
                        budget.max_steps,
                    )
                    break

        if budget.model_selection != "last" and not interrupted:
            if chosen_state is None:
                raise ValueError(
                    f"model_selection = {budget.model_selection!r} but no validation evaluation "
                    f"fell within {executed} steps with eval_every="
                    f"{budget.eval_every}: there is nothing to select, and returning the "
                    f"last step while declaring a selection would be misleading metadata"
                )
            head_weights, adapter_weights = chosen_state
            head.load_state_dict({k: v.to(self.device) for k, v in head_weights.items()})
            restore_adapter_state(adapt, adapter_weights)
            log.info(
                "returned weights: those of step %d (%s = %.6f), not of step %d",
                selected_step,
                budget.model_selection,
                selected_value if selected_value is not None else float("nan"),
                executed,
            )
        head.eval()
        return FitResult(
            steps=executed,
            final_loss=last_loss,
            lr=chosen_lr,
            history=history,
            regime=regime,
            trainable_params=n_trainable,
            best_val=best_val,
            best_step=best_step,
            val_auroc=val_auroc,
            selection_metric=budget.model_selection,
            selected_step=executed if budget.model_selection == "last" else selected_step,
            selected_value=selected_value,
            selected_scores={} if budget.model_selection == "last" else dict(chosen_scores),
            stopped_early=stopped_early,
            resumed_from=resumed_from,
            interrupted=interrupted,
        )

    @staticmethod
    def _adapter_state(adapt: Any | None) -> dict[str, torch.Tensor]:
        """Return the new adapter weights, detached and on CPU.

        ``state_dict_adapter()`` returns views; they are cloned so the checkpoint does not hold
        references that change while ``torch.save`` writes (the opposite of what resumption
        needs, see :func:`restore_adapter_state`).

        :return: qualified name to tensor; empty without adaptation.
        """
        read = getattr(adapt, "state_dict_adapter", None)
        if not callable(read):
            return {}
        return {name: t.detach().clone().cpu() for name, t in read().items()}

    def _val_metrics(
        self,
        backbone: Any | None,
        head: nn.Module,
        val_loader: DataLoader[Any],
        loss_fn: nn.Module,
        *,
        budget: BudgetSpec,
        levels: tuple[int, ...],
        cache_allowed: bool,
        dtype: torch.dtype,
        objective: Objective = "multiclass",
        normal_idx: int | None = None,
    ) -> tuple[float, float | None, dict[str, float], float | None]:
        """Compute loss, AUROC, per-window logits and class accuracy on validation, in one pass.

        The pass is expensive (8383 windows, about 14 min of A100 on ``edp_v2``), so the three
        outputs come together; the per-window logits allow estimating the threshold without
        another pass (:attr:`FitResult.selected_scores`, ADR-053). Evaluation never builds a
        graph. AUROC is computed here because ``train`` does not import ``eval`` (docs/02 §5),
        with the same degenerate-case conventions (``nan`` with a warning, never 0.5).

        :return: ``(mean loss, AUROC, window_id -> logit, balanced accuracy)``. AUROC and the
            mapping come from the binary logit (binary and multitask objectives; ``None`` and
            empty otherwise). The balanced accuracy is the mean, over the anomaly classes, of
            the share of their positive windows whose highest anomaly-class logit is theirs
            (``Normal`` excluded from the argmax); ``None`` without a class output or without
            ``normal_idx``.
        :raises ValueError: if the validation loader is empty or a batch returns a number of
            logits different from its windows.
        """
        was_training = head.training
        head.eval()
        total, n = 0.0, 0
        scores: list[float] = []
        labels: list[int] = []
        per_window: dict[str, float] = {}
        true_classes: list[int] = []
        chosen_classes: list[int] = []
        try:
            with torch.no_grad():
                for batch in val_loader:
                    target = (
                        None if objective == "multitask" else _target(batch, objective, self.device)
                    )
                    with torch.autocast(
                        device_type=torch.device(self.device).type,
                        dtype=torch.bfloat16,
                        enabled=budget.precision == "bf16",
                    ):
                        feats = self._features_from_batch(
                            backbone,
                            batch,
                            levels=levels,
                            frozen=True,  # no graph at evaluation, whatever the training regime
                            cache_allowed=cache_allowed,
                            dtype=dtype,
                        )
                        if objective == "multitask":
                            outputs = head.forward_tasks(feats)
                            losses = self._multitask_loss(head, feats, batch, loss_fn, outputs)
                            total += float(sum(losses.values()))
                            logit = outputs["binary"][:, 0]
                            class_logits = outputs["classes"]
                            target = _target(batch, "binary", self.device)
                        else:
                            raw = head(feats)
                            logit = _logit(raw, objective)
                            total += float(loss_fn(logit, target))
                            class_logits = raw if objective == "multiclass" else None
                    if class_logits is not None and normal_idx is not None:
                        classes = batch["class_label"].detach().cpu()
                        positive = classes != normal_idx
                        masked = class_logits.detach().float().cpu().clone()
                        masked[:, normal_idx] = -float("inf")
                        true_classes.extend(classes[positive].tolist())
                        chosen_classes.extend(masked.argmax(1)[positive].tolist())
                    if objective in ("binary", "multitask"):
                        values = logit.detach().float().cpu().tolist()
                        scores.extend(values)
                        labels.extend(int(v) for v in target.detach().cpu().tolist())
                        # `strict=True`: a multi-view batch would return more logits than windows.
                        for window, value in zip(batch["window"], values, strict=True):
                            per_window[window.window_id] = float(value)
                    n += 1
        finally:
            if was_training:
                head.train()
        if n == 0:
            raise ValueError("empty val_loader: the selection would have nothing to choose on")
        auroc = self._auroc(scores, labels) if objective in ("binary", "multitask") else None
        balanced = None
        if true_classes:
            pairs = list(zip(true_classes, chosen_classes, strict=True))
            recalls = [
                sum(p == c for t, p in pairs if t == c) / sum(t == c for t, _ in pairs)
                for c in sorted(set(true_classes))
            ]
            balanced = float(sum(recalls) / len(recalls))
        return total / n, auroc, per_window, balanced

    def _multitask_loss(
        self,
        head: nn.Module,
        feats: BackboneOutput,
        batch: dict[str, Any],
        class_loss: nn.Module,
        outputs: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """:return: the BCE of the ``binary`` output and the cross-entropy of the ``classes``
        output, each still attached to the graph.

        :raises ValueError: if the batch lacks either label (see :func:`_target`).
        """
        outputs = head.forward_tasks(feats) if outputs is None else outputs
        binary = _target(batch, "binary", self.device)
        classes = _target(batch, "multiclass", self.device)
        return {
            "binary": nn.functional.binary_cross_entropy_with_logits(
                _logit(outputs["binary"], "binary"), binary
            ),
            "classes": class_loss(outputs["classes"], classes),
        }

    @staticmethod
    def _auroc(punteggi: list[float], etichette: list[int]) -> float:
        """:return: binary ROC-AUC, or ``nan`` with a warning if the ground truth is degenerate."""
        positives = sum(etichette)
        if not etichette or positives in (0, len(etichette)):
            log.warning(
                "validation AUROC: degenerate ground truth (%d positives out of %d) -> nan",
                positives,
                len(etichette),
            )
            return float("nan")
        return float(roc_auc_score(etichette, punteggi))

    def _features_from_batch(
        self,
        backbone: Any | None,
        batch: dict[str, Any],
        *,
        levels: tuple[int, ...],
        frozen: bool,
        cache_allowed: bool,
        dtype: torch.dtype,
    ) -> BackboneOutput:
        has_clip = "clip" in batch
        has_features = "features" in batch
        if has_clip == has_features:
            raise ValueError(
                "ambiguous training batch: exactly one of 'clip' and 'features' is required"
            )
        if has_features:
            if not cache_allowed:
                raise ValueError(
                    "cached features are forbidden during PEFT: an adapted backbone must run "
                    "the forward on the fly (invariant 3 / ADR-007)"
                )
            return cached_output_from_batch(batch, device=self.device, dtype=dtype)
        if backbone is None:
            raise ValueError("pixel batch received without a backbone")
        clips = batch["clip"].to(self.device)
        return self._encode(backbone, clips, levels, frozen)

    def _encode(
        self, backbone: Any, clips: torch.Tensor, levels: tuple[int, ...], frozen: bool
    ) -> BackboneOutput:
        if frozen:
            with torch.no_grad():
                out = encode_multilevel(backbone, clips, levels)
            # Tokens stay detached from the backbone graph: the head builds its own graph from here.
            return BackboneOutput(
                tokens=out.tokens.detach(),
                grid=out.grid,
                pooled=None if out.pooled is None else out.pooled.detach(),
                layer=out.layer,
                levels=out.levels,
            )
        return encode_multilevel(backbone, clips, levels)
