"""Experiment engine: a config goes in, a registered run comes out.

The Facade of ``docs/03`` §8. Frozen and PEFT go through the same code, and what
differs between them is a config, not a branch (invariant 5: budget-parity comparisons
do not survive two code paths).

The six steps of the contract: validate config and split down to the ``run_id``, build
backbone/adaptation/head from the registries, train, evaluate, measure, register.

The fully-supervised regime (ADR-031/047): the protocol declares which videos belong to
which partition (:mod:`evd.protocol.split`), and the chunk grid is a pure function of
that plus the annotations (:mod:`evd.protocol.chunks`), recomputed at every run and
checked through the measured positive rate written to ``meta.json``.

Three boundaries enforced here:

* the split is the only source of partitions: the runner reads a
  sealed split and recomputes the grid instead of reading a derived artifact;
* only the frozen regime sees the feature store;
* learning rate, early stopping and threshold are decided on validation, never on test.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch

from evd.annotations.schema import TemporalAnnotation
from evd.data.dataset import (
    CACHE_RAM_MB,
    WindowItem,
    build_adapted_loader,
    build_frozen_loader,
    collate_windows,
)
from evd.data.decoder import make_decoder
from evd.data.feature_store import FeatureReader, SafetensorsFeatureStore
from evd.data.windows import CausalWindowSampler
from evd.eval import metrics as M
from evd.eval import valutazione as V
from evd.eval.frame_map import FrameScoreMapper
from evd.eval.plots import lr_png, timeline_png, training_png
from evd.eval.scorers import HeadScorer, anomaly_from_logits
from evd.exp.lr_trials import RECEIPT_NAME, TrialsReceipt
from evd.exp.preadapt import (
    freeze_slot,
    prepare_backbone,
    resolve_checkpoint,
)
from evd.exp.run_id import RunId
from evd.exp.sources import DatasetBundle, build_dataset
from evd.exp.tracker import LocalTracker, RunHandle, git_info, plain_config
from evd.infer.engine import InferenceEngine
from evd.models.adapt.base import AdaptationReport, AdaptationStrategy, preserves_backbone_output
from evd.models.adapt.factory import AdaptContext, build_adaptation
from evd.models.backbones.base import BACKBONES
from evd.models.heads.base import HEADS
from evd.protocol.chunks import (
    OVERLAP_FLOOR,
    ChunkItem,
    balance_classes,
    chunk_at,
    enumerate_chunks,
    riepilogo,
    sampler_for_protocol,
)
from evd.protocol.split import SplitSpec, load_split
from evd.train.budget import BudgetSpec, apply_numeric_regime, steps_per_budget
from evd.train.checkpoint import (
    Checkpointer,
    CheckpointIdentity,
    CheckpointSpec,
    InterruptFlag,
    TrainingInterrupted,
    install_interrupt_handler,
)
from evd.train.selection import (
    SelectionOutcome,
    fully_supervised_regime,
    select_lr,
)
from evd.train.trainer import Objective, Trainer
from evd.types import (
    AugmentSpec,
    PreprocessSpec,
    VideoMeta,
    WindowScores,
    WindowSpec,
    check_known_fields,
)
from evd.utils import env
from evd.utils.hashing import stable_hash
from evd.utils.io import write_jsonl
from evd.utils.logging import get_logger, setup_logging
from evd.utils.seeding import seed_all

log = get_logger(__name__)

__all__ = [
    "ExperimentRunner",
    "RunResult",
    "TrainingInterrupted",
    "protocol_normal_idx",
    "run_id_for",
    "split_path_for",
    "window_items",
]

#: Fields of the ``backbone`` and ``head`` groups that are not constructor arguments.
_NON_ARGUMENTS = frozenset({"id", "name", "family"})

#: Fields outside the checkpoint identity because they do not change training (where it
#: runs, whether it writes the run, whether it draws the figure): otherwise resuming on
#: another node would be refused. ``train.checkpoint_every`` and ``eval.num_workers`` are
#: excluded in :meth:`ExperimentRunner._config_sha`.
_OUTSIDE_IDENTITY = frozenset({"device", "write_run", "make_figure"})


@dataclass(frozen=True)
class RunResult:
    """What a run produced, in a form the caller can query.

    :param run_id: the identity, validated by the ``docs/05`` §8 grammar.
    :param metrics: the dictionary written to ``metrics.json``.
    :param run_dir: the run folder, or ``None`` if the run was not written.
    :param split_sha: the seal of the consumed split, binding the number to the exact
        partition.
    :param adaptation: the adaptation receipt, for ``meta.json``.
    """

    run_id: str
    metrics: dict[str, Any]
    run_dir: Path | None = None
    split_sha: str | None = None
    adaptation: Mapping[str, Any] = field(default_factory=dict)


def window_items(chunks: Sequence[ChunkItem], bundle: DatasetBundle) -> list[WindowItem]:
    """Turn protocol chunks into the windows consumed by the loader.

    ``label`` is the binary target; ``class_label`` travels alongside without being a target
    (per-class error breakdown, ``docs/04`` §4.2), and is the ``Normal`` index on negatives.
    ``weight`` reaches the ``WeightedRandomSampler`` (ADR-032 §4-5). Public so that E0d
    measures exactly the windows of a run.
    """
    return [
        WindowItem(
            window=c.window,
            video_path=bundle.meta(c.video_id).path,
            label=c.label,
            class_label=bundle.normal_idx if c.class_idx is None else int(c.class_idx),
            weight=c.weight,
        )
        for c in chunks
    ]


def split_path_for(cfg: Mapping[str, Any]) -> Path:
    """Return the split this config would consume.

    The path is declared by the protocol file (``protocol.split_file``). Public so that
    ``campagna.py valida`` can detect a missing split before running a cell.

    :param cfg: the resolved config.
    :return: the path, which may not exist.
    :raises ValueError: if ``protocol.split_file`` is not declared.
    """
    path = (cfg["protocol"] or {}).get("split_file")
    if not path:
        raise ValueError(
            "protocol.split_file is not declared: the runner does not build splits, "
            "it reads sealed files (invariant 1). Generate the split before the run"
        )
    return Path(str(path))


def run_id_for(cfg: Mapping[str, Any], *, dataset_id: str) -> RunId:
    """Return the identity of the run this config would produce.

    Outside the class because it is needed before running: ``campagna.py`` asks the registry
    for the ``run_id`` values of a grid. Takes ``dataset_id`` instead of the bundle so that
    planning does not touch the data. The ``k`` field left the ``run_id`` on 27 August
    (ADR-031); ``RunId.parse`` still reads the five-field form.

    :param cfg: the composed config; ``exp_id``, ``backbone.id``, ``adapt.name`` and
        ``seed`` suffice.
    :param dataset_id: the ``data.id`` field, which enters the ``run_id``.
    :return: the identity, validated by the ``docs/05`` §8 grammar.
    :raises ValueError: if a field violates the grammar.
    """
    return RunId(
        exp=str(cfg["exp_id"]),
        dataset=dataset_id,
        backbone=str(cfg["backbone"]["id"]),
        adapt=str(cfg["adapt"]["name"]),
        seed=int(cfg["seed"]),
    )


def _arguments(gruppo: Mapping[str, Any]) -> dict[str, Any]:
    """:return: the group fields that are constructor arguments."""
    return {k: v for k, v in gruppo.items() if k not in _NON_ARGUMENTS}


class ExperimentRunner:
    """Run an experiment described by a composed config.

    :param tracker: where runs and registry go; defaults to ``$EVD_RUNS_DIR``
        (ADR-009/ADR-020, one runs folder per machine).
    """

    def __init__(self, tracker: LocalTracker | None = None) -> None:
        self.tracker = tracker or LocalTracker()
        # Preadaptation of the current run (ADR-065), decided once in `run` and read by the three
        # places that apply a strategy; empty so a reused runner does not inherit it.
        self._preadapt: Any | None = None
        self._preadapt_meta: dict[str, Any] | None = None
        # Weights copied and frozen from a finished run (`inherit`), with their receipt.
        self._inherit_meta: dict[str, Any] | None = None

    def _apply(self, strategy: AdaptationStrategy, backbone: Any, device: str) -> Any:
        """Apply the strategy and, if the run declares it, freeze the pre-trained slot.

        Strategies are applied in three places (final fit, ``select_only`` and every LR trial);
        the freeze must hold in all of them, or an LR trial would also train the prompts.

        :param strategy: the strategy to apply.
        :param backbone: the backbone, modified in place.
        :param device: where the parameters live.
        :return: the ``apply()`` receipt.
        """
        adapted = strategy.apply(backbone)
        if self._preadapt is not None and self._preadapt.modo == "congela":
            self._preadapt_meta = freeze_slot(self._preadapt, strategy, device=device)
        return adapted

    def _frozen_params(self) -> int:
        """:return: the parameters frozen by the preadaptation, 0 if none.

        Subtracted from the declared trainable parameters: the report is produced by
        ``apply()``, before the freeze (ADR-057).
        """
        meta = self._preadapt_meta or {}
        inherited = self._inherit_meta or {}
        return int(meta.get("params_congelati") or 0) + int(
            inherited.get("adapter_params_frozen") or 0
        )

    # --- docs/03 §8 contract ----------------------------------------------------
    def run(self, cfg: Mapping[str, Any]) -> RunResult:
        """Run and return what the run produced.

        :param cfg: the resolved config; no ``???`` may remain.
        :return: metrics, identity and adaptation receipt.
        :raises ValueError: if a protocol piece is missing or contradictory.
        :raises RuntimeError: if the run is official and the working tree is dirty.
        :raises TrainingInterrupted: if an interruption signal stops training early.
        """
        setup_logging()
        # The signal handler is armed here, always. SLURM sends SIGUSR1 five minutes before the
        # wall time (`--signal=B:USR1@300`); when the handler lived in `_checkpointer` a run
        # without checkpoints was killed by the default action (job 539643).
        interruption = InterruptFlag()
        signals = install_interrupt_handler(interruption)
        log.info(
            "interruption signals handled: %s",
            ", ".join(signals) or "none (platform without SIGUSR1)",
        )
        device = str(cfg.get("device") or env.device())
        # `deterministic` is turned off only where timings are measured (it changes kernel
        # selection); here metrics are measured, so it stays on.
        deterministic = bool(cfg.get("deterministic", True))
        seed = int(cfg["seed"])
        seed_all(seed, deterministic=deterministic)

        bundle = build_dataset(cfg["data"])
        protocol = cfg["protocol"]
        self._check_regime(protocol)
        split = self._split(cfg, bundle)
        run_id = run_id_for(cfg, dataset_id=bundle.dataset_id)
        # The real seed is logged here: the `hpc/env.sh` preamble prints `EVD_SEED`, which this
        # path does not read (ADR-057).
        log.info(
            "run %s · seed %d · device %s · split %s",
            run_id,
            seed,
            device,
            split.sha256[:12],
        )

        backbone = BACKBONES.get(str(cfg["backbone"]["id"]))(
            **_arguments(cfg["backbone"]), device=device
        )
        spec = backbone.preprocess_spec()
        self._check_geometry(spec, protocol)

        objective = self._objective(protocol, cfg)
        n_classes = self._n_classes(cfg, bundle, objective)
        self._inherit_meta = None
        n_blocks = len(backbone.blocks())
        # Before any `apply()` and before the store decision: `fondi` rewrites `base.weight`, so
        # from here on the backbone is no longer the base checkpoint (ADR-065). The translation
        # is shared with `exp.demo.motore_live` (`docs/09` R24).
        self._preadapt, self._preadapt_meta = prepare_backbone(
            cfg, backbone, context=AdaptContext(n_classes, n_blocks, seed), device=device
        )
        trial_strategy = build_adaptation(cfg["adapt"], AdaptContext(n_classes, n_blocks, seed))
        # With a preadapted backbone the feature store is invalid because of the weights: cached
        # features come from the bare backbone (ADR-056).
        frozen = preserves_backbone_output(trial_strategy) and self._preadapt is None
        store, key_of = self._open_store(cfg, backbone, spec, frozen=frozen)
        levels = self._levels(
            cfg,
            n_blocks=n_blocks,
            hierarchical=tuple(backbone.hierarchical_layers()),
            store_layers=self._layer_store(cfg, store),
        )
        context = AdaptContext(n_classes=n_classes, n_blocks=n_blocks, seed=seed)

        def build_catalog() -> tuple[Any, AdaptationStrategy]:
            """Return a new head and adaptation, not yet applied.

            LR selection tries every grid point from scratch, so trials do not depend on their
            order.
            """
            head = HEADS.get(str(cfg["head"]["id"]))(
                embed_dim=backbone.embed_dim,
                **{**_arguments(cfg["head"]), "n_classes": n_classes},
            )
            return head, build_adaptation(cfg["adapt"], context)

        augment = self._augment(cfg)
        if augment.activate:
            log.info(
                "training augmentation %r: crop %s, scale %s, ratio %s, "
                "randaugment %s, flip %s, reprob %.2f",
                augment.id,
                augment.crop,
                augment.random_resize_scale,
                augment.random_resize_ratio,
                augment.rand_augment or "no",
                augment.horizontal_flip,
                augment.reprob,
            )
        sampler = self._sampler(spec, protocol)
        sampler_val = self._sampler(spec, protocol, validazione=True)
        decoder = make_decoder(str(cfg["data"].get("decoder", "pyav")))

        # The grid is recomputed, not read: a pure function of (split, annotations, span, hop);
        # its summary (counts and measured positive rate) is recorded (docs/04 §8).
        floor = self._floor(protocol)
        chunk_train = self._chunks(split.train, bundle, sampler, floor, part="training")
        chunk_val = self._chunks(split.val, bundle, sampler_val, floor, part="validazione")
        chunk_test = self._chunks(split.test, bundle, sampler, floor, part="test")
        summary = {
            "train": riepilogo(chunk_train).as_dict(),
            "val": riepilogo(chunk_val).as_dict(),
            "test": riepilogo(chunk_test).as_dict(),
        }
        log.info(
            "grid: %d training chunks (%.2f%% positive, %.2f%% of the sampling "
            "mass), %d validation, %d test",
            summary["train"]["n_chunk"],
            100 * summary["train"]["tasso_positivi_uniforme"],
            100 * summary["train"]["tasso_positivi_pesato"],
            summary["val"]["n_chunk"],
            summary["test"]["n_chunk"],
        )

        dynamic = self._dynamic(protocol)
        budget = self._budget(cfg)
        if budget.class_balance == "equal":
            # Plan of 21 October §5.5: a declared deviation from invariant 6, for the class heads.
            if dynamic:
                raise ValueError(
                    "train.class_balance: equal is defined on the static grid only; with "
                    "dynamic sampling the weights are drawn per window and would ignore it"
                )
            chunk_train = balance_classes(chunk_train)
            summary["train"] = riepilogo(chunk_train).as_dict()
            log.info(
                "class balance: every class gets an equal share of the draws, positives now "
                "%.2f%% of the sampling mass",
                100 * summary["train"]["tasso_positivi_pesato"],
            )
        train = self._with_keys(window_items(chunk_train, bundle), None if dynamic else key_of)
        val = self._with_keys(window_items(chunk_val, bundle), key_of)
        resample = self._resampler(bundle, sampler, floor, seed=seed) if dynamic else None
        # Steps are fixed once, on the whole grid, for both the LR trials and the final fit.
        steps = steps_per_budget(budget, len(train))
        if steps != budget.max_steps:
            log.info(
                "budget: %d steps derived from %d epochs over %d training chunks "
                "(max_steps=%d ignored%s)",
                steps,
                budget.epochs,
                len(train),
                budget.max_steps,
                ", cap active" if steps == budget.max_steps_cap else "",
            )
            budget = replace(budget, max_steps=steps)

        # The effective budget after every adjustment, logged next to the seed: `batch_size` and
        # `num_workers` appear in three config groups with different values.
        log.info(
            "budget: %d steps · batch %d x %d = %d · %d workers · %s · selection %s every %d",
            budget.max_steps,
            budget.batch_size,
            budget.grad_accum,
            budget.batch_size * budget.grad_accum,
            budget.num_workers,
            budget.precision,
            budget.model_selection,
            budget.eval_every,
        )

        # Receipt of completed trials, only when there is a grid to traverse (with one point
        # `select_lr` trains nothing).
        receipt = (
            TrialsReceipt.open_(
                self.tracker.runs_dir / run_id.format() / RECEIPT_NAME,
                run_id=run_id.format(),
                config_sha=self._config_sha(cfg),
                split_sha=split.sha256,
                lr_grid=budget.lr_grid,
            )
            if len(budget.lr_grid) > 1 and bool(cfg.get("write_run", True))
            else None
        )

        selection = self._select_lr(
            train,
            val,
            ricevuta=receipt,
            flag=interruption,
            backbone=backbone,
            decoder=decoder,
            spec=spec,
            budget=budget,
            build_catalog=build_catalog,
            seed=seed,
            device=device,
            levels=levels,
            store=store,
            augment=augment,
            objective=objective,
            resample=resample,
            normal_idx=bundle.normal_idx,
        )

        if bool(cfg.get("select_only", False)):
            # E0i: the cell ends here. A final fit and an evaluation would touch the TEST split for an
            # experiment that produces no thesis numbers (`docs/10`, E0i).
            return self._close_selection(
                cfg,
                run_id=run_id,
                split=split,
                spec=spec,
                budget=budget,
                device=device,
                seed=seed,
                protocol=protocol,
                build_catalog=build_catalog,
                backbone=backbone,
                selezione=selection,
                summary=summary,
                n_train=len(train),
                n_val=len(val),
                dynamic=dynamic,
            )

        # The final fit restarts from scratch with the chosen LR; validation is still available,
        # so early stopping applies here too (docs/04 §5.3).
        head, strategy = build_catalog()
        adapted = self._apply(strategy, backbone, device)
        report = adapted.report
        warm_start = self._warm_start(cfg, head, strategy, device)
        self._inherit(cfg, head, strategy, device)
        # Second line on invariant 3 (ADR-056): `store` was decided on the trial strategy, not yet
        # attached; this is the attached one. `_evaluate` passes the cache to `InferenceEngine`
        # without further filters, and the ADR-007 key does not name the adapter, so an undue hit
        # would report frozen-probe numbers for a PEFT run.
        if store is not None and not preserves_backbone_output(strategy):
            log.info(
                "adaptation %s/%s: feature store discarded, bare-backbone features "
                "are not valid for this run",
                report.family,
                report.method,
            )
            store, key_of = None, None
        log.info(
            "adaptation %s/%s: %d targets, %d parameters, %d extra tokens",
            report.family,
            report.method,
            report.n_targets,
            report.n_trainable,
            report.extra_tokens,
        )

        fit = Trainer(device=device).fit(
            backbone,
            head,
            self._loader(
                train,
                decoder,
                spec,
                budget,
                strategy,
                seed,
                store=store,
                augment=augment,
                weighted=True,
                resample=resample,
            ),
            budget,
            # Validation enters the final fit only when it serves a decision: early stopping or
            # `model_selection` (ADR-047 §6). Otherwise the `Trainer` would evaluate it every
            # `eval_every` steps for nothing (ADR-036, R15). The threshold τ is estimated later, in
            # `_soglia`.
            val_loader=(
                self._loader(val, decoder, spec, budget, strategy, seed, store=store, shuffle=False)
                if budget.early_stop_patience is not None or budget.model_selection != "last"
                else None
            ),
            levels=levels,
            adapt=strategy,
            lr=selection.lr,
            objective=objective,
            checkpoint=self._checkpointer(cfg, run_id, split, backbone, interruption),
            normal_idx=bundle.normal_idx,
        )
        log.info("training: %d steps, final loss %.4f", fit.steps, fit.final_loss)
        if fit.interrupted:
            # Training is not finished: evaluating now would register a half-budget number. The next
            # slice resumes from the checkpoint.
            raise TrainingInterrupted(
                f"{run_id.format()}: training interrupted at step {fit.steps}/"
                f"{budget.max_steps}. The checkpoint is written: relaunch the same cell "
                f"to resume from here"
            )

        handle = self._open_run(
            cfg, run_id, split, fit.regime, spec, report, summary, budget, seed, warm_start
        )
        metrics = self._evaluate(
            cfg=cfg,
            bundle=bundle,
            split=split,
            backbone=backbone,
            head=head,
            spec=spec,
            sampler=sampler,
            decoder=decoder,
            val=val,
            device=device,
            handle=handle,
            levels=levels,
            store=store,
            key_of=key_of,
            sampler_val=sampler_val,
            # A multitask head returns its binary logit from `forward`: what is evaluated is the
            # binary detector, with its threshold from the selected-step scores (ADR-053).
            objective="binary" if objective == "multitask" else objective,
            selection_scores=fit.selected_scores or None,
        )
        metrics.update(
            {
                "seed": seed,
                "protocol_version": str(protocol.get("protocol_version", "")),
                "split_sha": split.sha256,
                "chunks": summary,
                "n_chunk_train": len(train),
                "n_chunk_val": len(val),
                "train_sampling": "dynamic" if dynamic else "static",
                "final_loss": fit.final_loss,
                # Two counts, two names (ADR-057): `trainable_params` is the TOTAL (head plus adapter),
                # while budget parity (invariant 5) is expressed on the ADAPTER parameters.
                "trainable_params": fit.trainable_params,
                "trainable_params_adapter": report.n_trainable - self._frozen_params(),
                "lr": fit.lr,
                "lr_selection": selection.as_dict(),
                "steps": fit.steps,
                "max_steps": budget.max_steps,
                "budget_epochs": budget.epochs,
                "budget_cap_attivo": (
                    budget.max_steps_cap is not None and budget.max_steps == budget.max_steps_cap
                ),
                "stopped_early": fit.stopped_early,
                "resumed_from": fit.resumed_from,
                # Model selection (ADR-047 §6), always written, even when `last`.
                "model_selection": {
                    "metric": fit.selection_metric,
                    "step": fit.selected_step,
                    "value": fit.selected_value,
                    "eval_every": budget.eval_every,
                },
                "val_auroc_finale": fit.val_auroc,
                "adapt": dict(report.as_dict()),
            }
        )
        if handle is not None:
            self.tracker.log_history(handle, fit.history)
            # Both the data and its plot: `history.jsonl` alone is unreadable, a PNG alone cannot be
            # re-checked. Not behind `make_figure` (a qualitative choice about one video). A failure
            # here must not discard an hours-long run.
            try:
                metrics["figura_training"] = str(
                    training_png(
                        fit.history,
                        handle.dir / "training.png",
                        selected_step=fit.selected_step,
                        selection_metric=fit.selection_metric,
                        title=handle.run_id,
                    )
                )
                # The learning-rate schedule has its own figure: it shows whether the configuration
                # actually governed.
                metrics["figura_lr"] = str(
                    lr_png(
                        fit.history,
                        handle.dir / "lr.png",
                        selected_step=fit.selected_step,
                        title=f"{handle.run_id} — learning-rate schedule",
                    )
                )
            except Exception:  # noqa: BLE001 - see the comment above
                log.warning("training figure not written", exc_info=True)
            self.tracker.finish(
                handle,
                metrics,
                extra={"exp": run_id.exp, "adapt_family": report.family},
            )
        return RunResult(
            run_id=run_id.format(),
            metrics=metrics,
            run_dir=None if handle is None else handle.dir,
            split_sha=split.sha256,
            adaptation=report.as_dict(),
        )

    # --- the pilot that only selects ---------------------------------------------
    def _close_selection(
        self,
        cfg: Mapping[str, Any],
        *,
        run_id: RunId,
        split: SplitSpec,
        spec: PreprocessSpec,
        budget: BudgetSpec,
        device: str,
        seed: int,
        protocol: Mapping[str, Any],
        build_catalog: Callable[[], tuple[Any, AdaptationStrategy]],
        backbone: Any,
        selezione: SelectionOutcome,
        summary: Mapping[str, Any],
        n_train: int,
        n_val: int,
        dynamic: bool,
    ) -> RunResult:
        """Register a cell that stops at LR selection (E0i).

        E0i answers one question, the learning rate (ADR-036); a final fit and an evaluation
        would be the cost the pilot exists to avoid, and would touch the test split. A branch in
        ``run`` rather than an extension point: there is only one way to stop (``CLAUDE.md``
        §3.1). ``metrics.json`` holds only what happened: the selection receipt with its
        ``trials``, the chunk-grid signature and the adaptation; no ``final_loss``, ``steps``
        or test metrics.

        :param selezione: the receipt produced by :meth:`_select_lr`.
        :param summary: counts and positive rates of the three partitions.
        :param dinamico: whether training sampling was dynamic.
        :return: the run result, without evaluation metrics.
        """
        # The adaptation was applied only inside the trials (each removes it with `unload_`): the
        # receipt is rebuilt by applying it once and removing it immediately.
        _, strategy = build_catalog()
        try:
            report = self._apply(strategy, backbone, device).report
        finally:
            strategy.unload_()

        metrics: dict[str, Any] = {
            "select_only": True,
            "seed": seed,
            "protocol_version": str(protocol.get("protocol_version", "")),
            "split_sha": split.sha256,
            "chunks": dict(summary),
            "n_chunk_train": n_train,
            "n_chunk_val": n_val,
            "train_sampling": "dynamic" if dynamic else "static",
            "lr": selezione.lr,
            "lr_selection": selezione.as_dict(),
            "max_steps": budget.max_steps,
            # No final fit here, so the total is not measured: only the adapter count is written,
            # under its own name.
            "trainable_params_adapter": report.n_trainable - self._frozen_params(),
            "adapt": dict(report.as_dict()),
        }
        handle = self._open_run(
            cfg,
            run_id,
            split,
            apply_numeric_regime(budget, device),
            spec,
            report,
            summary,
            budget,
            seed,
        )
        if handle is not None:
            self.tracker.finish(
                handle,
                metrics,
                extra={"exp": run_id.exp, "adapt_family": report.family},
            )
        return RunResult(
            run_id=run_id.format(),
            metrics=metrics,
            run_dir=None if handle is None else handle.dir,
            split_sha=split.sha256,
            adaptation=report.as_dict(),
        )

    # --- hyperparameter selection --------------------------------------------------
    def _select_lr(
        self,
        train: Sequence[WindowItem],
        val: Sequence[WindowItem],
        *,
        ricevuta: TrialsReceipt | None = None,
        flag: InterruptFlag | None = None,
        backbone: Any,
        decoder: Any,
        spec: PreprocessSpec,
        budget: BudgetSpec,
        build_catalog: Callable[[], tuple[Any, AdaptationStrategy]],
        seed: int,
        device: str,
        levels: tuple[int, ...],
        store: FeatureReader | None = None,
        augment: AugmentSpec | None = None,
        objective: Objective = "binary",
        resample: Any | None = None,
        normal_idx: int | None = None,
    ) -> SelectionOutcome:
        """Choose the learning rate on validation (``docs/04`` §5.3).

        Validation is a video partition declared in the split, disjoint from training and test
        (:meth:`evd.protocol.split.SplitSpec.__post_init__`). Augmentation applies to training
        only, and validation uses neither weighted nor dynamic sampling: it is measured on the
        static grid used for evaluation.

        Every grid point starts from a new head and adaptation, removed at the end of the trial
        (``unload_``, bit-exact and idempotent), so the backbone is not reloaded per learning
        rate. Completed trials are not redone (:class:`~evd.exp.lr_trials.TrialsReceipt`).

        :param ricevuta: where completed trials are recorded; ``None`` records nothing.
        :param flag: the run interruption flag, checked between trials.
        :return: the receipt, written to ``metrics.json``.
        :raises TrainingInterrupted: if the signal arrives before a trial starts.
        """
        regime = fully_supervised_regime(n_lr=len(budget.lr_grid), n_val=len(val))

        def evaluate(lr: float) -> float:
            already_done = None if ricevuta is None else ricevuta.score_of(lr)
            if already_done is not None:
                return already_done
            if flag is not None and bool(flag):
                raise TrainingInterrupted(
                    f"LR selection interrupted before the trial lr={lr:g}: the signal "
                    f"arrived and the trial would not fit in the slice. Completed trials "
                    f"are in the receipt; relaunch the same cell to continue"
                )
            head, strategy = build_catalog()
            self._apply(strategy, backbone, device)
            try:
                trial = Trainer(device=device).fit(
                    backbone,
                    head,
                    self._loader(
                        train,
                        decoder,
                        spec,
                        budget,
                        strategy,
                        seed,
                        store=store,
                        augment=augment,
                        weighted=True,
                        resample=resample,
                    ),
                    budget,
                    val_loader=self._loader(
                        val, decoder, spec, budget, strategy, seed, store=store, shuffle=False
                    ),
                    levels=levels,
                    adapt=strategy,
                    lr=lr,
                    objective=objective,
                    normal_idx=normal_idx,
                )
            finally:
                # Even if the fit raises: an adaptation left attached would make the next run different
                # from the declared one.
                strategy.unload_()
            score = trial.best_val if trial.best_val is not None else trial.final_loss
            if ricevuta is not None:
                ricevuta.record(lr, score)
            return score

        outcome = select_lr(budget.lr_grid, evaluate, regime=regime, n_val=len(val))
        log.info("learning rate %g — %s (%s)", outcome.lr, outcome.regime, outcome.provenance)
        return outcome

    # --- resume ---------------------------------------------------------------------
    def _checkpointer(
        self,
        cfg: Mapping[str, Any],
        run_id: RunId,
        split: SplitSpec,
        backbone: Any,
        flag: InterruptFlag,
    ) -> Checkpointer | None:
        """Prepare intra-training resume, if the config asks for it.

        Covers the final fit only: LR trials have no checkpoint (they would share the final-fit
        file, and a resumed trial would not be comparable with the others). ``flag`` comes from
        :meth:`run`, which arms signals once per run.

        :param flag: the interruption flag already armed by the run.
        :return: the checkpointer, or ``None`` if resume is inactive (``checkpoint_every`` at
            zero, or ``write_run: false``).
        """
        every = int(cfg["train"].get("checkpoint_every", 0) or 0)
        if every <= 0 or not bool(cfg.get("write_run", True)):
            return None
        log.info(
            "resume active: checkpoint every %d steps in runs/%s/checkpoint",
            every,
            run_id.format(),
        )
        return Checkpointer(
            CheckpointSpec(dir=self.tracker.runs_dir / run_id.format() / "checkpoint", every=every),
            CheckpointIdentity(
                run_id=run_id.format(),
                config_sha=self._config_sha(cfg),
                backbone_id=str(getattr(backbone, "backbone_id", cfg["backbone"]["id"])),
                git_sha=git_info().get("sha"),
                protocol_sha=split.sha256,
                # `ckpt_sha` is not in the `VideoBackbone` Protocol (ISP, CLAUDE.md §3.1); where missing,
                # the identity has one field fewer.
                ckpt_sha=getattr(backbone, "ckpt_sha", None),
            ),
            flag=flag,
        )

    @staticmethod
    def _config_sha(cfg: Mapping[str, Any]) -> str:
        """Return the fingerprint of the resolved config, without operational fields.

        :return: sha256 of the canonical form; distinguishes two runs that share the ``run_id``
            but not the hyperparameters.
        """
        flat = plain_config(cfg)
        if not isinstance(flat, dict):
            return stable_hash(flat)
        reduced = {k: v for k, v in flat.items() if k not in _OUTSIDE_IDENTITY}
        group = reduced.get("train")
        if isinstance(group, dict):
            reduced["train"] = {k: v for k, v in group.items() if k != "checkpoint_every"}
        # `eval` fields describe how a model is read, not how it was trained, so they stay out of
        # the checkpoint identity. A list instead of dropping the whole `eval` group: removing the
        # group would change the hash and invalidate existing checkpoints (ADR-044).
        _EVAL_EXCLUDED = ("num_workers", "precision")
        evaluation = reduced.get("eval")
        if isinstance(evaluation, dict):
            reduced["eval"] = {k: v for k, v in evaluation.items() if k not in _EVAL_EXCLUDED}
        return stable_hash(reduced)

    # --- identity and protocol --------------------------------------------------------
    @staticmethod
    def _split(cfg: Mapping[str, Any], bundle: DatasetBundle) -> SplitSpec:
        """Load and verify the protocol split.

        :raises ValueError: if the seal does not match, if the split belongs to another dataset,
            or if it declares no validation partition.
        """
        path = split_path_for(cfg)
        split = load_split(path)
        if split.dataset != bundle.name:
            raise ValueError(
                f"{path} belongs to dataset {split.dataset!r} but the run uses "
                f"{bundle.name!r}: two protocols are not mixed"
            )
        if not split.has_validation:
            # A silent fallback here would use the test set for selection: LR, early stopping and
            # threshold are all decided on validation (docs/04 §5.3).
            raise ValueError(
                f"{path}: the split declares no validation video, but "
                f"learning rate, early stopping and threshold are estimated there (docs/04 "
                f"§5.3). Carve a validation partition from training, by video and "
                f"stratified by class, and seal it in a new split; never fall back on "
                f"the test set"
            )
        return split

    @staticmethod
    def _layer_store(cfg: Mapping[str, Any], store: FeatureReader | None) -> tuple[int, ...] | None:
        """:return: the layers contained in the store, or ``None`` without a store."""
        if store is None:
            return None
        declared = (cfg["data"].get("feature_store") or {}).get("layers") or ()
        return tuple(int(layer) for layer in declared)

    @staticmethod
    def _levels(
        cfg: Mapping[str, Any],
        *,
        n_blocks: int,
        hierarchical: tuple[int, ...],
        store_layers: tuple[int, ...] | None,
    ) -> tuple[int, ...]:
        """Return the blocks read by the head, as absolute indices in the given order.

        Negative indices are resolved here with the backbone rule
        (``VJEPA21Backbone._layer_index``) because of the cache key: the forward treats ``-1``
        and ``23`` as the same block, :func:`~evd.data.feature_store.feature_key` does not. A
        single level is a tuple of length 1 (ADR-029).

        :param cfg: the config; ``layer`` is an int, a list of ints, or ``"hierarchical"``.
        :param n_blocks: backbone depth, to resolve negative indices.
        :param hierarchical: the layers declared by the model
            (``backbone.hierarchical_layers()``), the target of ``"hierarchical"``.
        :param store_layers: the store layers, or ``None`` without a cache.
        :return: the absolute indices, in concatenation order.
        :raises ValueError: if a level is out of range, two levels alias the same block, or,
            with the cache active, a level was not extracted.
        """
        field = cfg.get("layer", -1)
        if isinstance(field, str):
            if field != "hierarchical":
                raise ValueError(
                    f"unknown layer {field!r}: the only allowed sentinel is "
                    f"'hierarchical', which resolves to the layers declared by the model"
                )
            if not hierarchical:
                raise ValueError(
                    "`layer: hierarchical` but the backbone declares no intermediate "
                    "layer: use explicit indices"
                )
            requested: tuple[int, ...] = tuple(hierarchical)
        elif isinstance(field, (list, tuple)):
            requested = tuple(int(v) for v in field)
            if not requested:
                raise ValueError("`layer` is an empty list: at least one level is required")
        else:
            requested = (int(field),)

        absolute: list[int] = []
        for requested_value in requested:
            absolute_value = requested_value if requested_value >= 0 else n_blocks + requested_value
            if not 0 <= absolute_value < n_blocks:
                raise ValueError(f"layer {requested_value} outside the {n_blocks} backbone blocks")
            absolute.append(absolute_value)
        if len(set(absolute)) != len(absolute):
            raise ValueError(
                f"aliased layers in the same set: requested {requested}, resolved {tuple(absolute)}"
            )
        if store_layers is not None:
            missing = [a for a in absolute if a not in store_layers]
            if missing:
                raise ValueError(
                    f"layer {requested} (absolute {tuple(absolute)}): {missing} is not among "
                    f"those extracted in the store {list(store_layers)}, the run would read "
                    f"an empty cache. Declare `layer` among them, or turn off "
                    f"`data.feature_store.enabled`"
                )
        return tuple(absolute)

    @staticmethod
    def _open_store(
        cfg: Mapping[str, Any],
        backbone: Any,
        spec: PreprocessSpec,
        *,
        frozen: bool,
    ) -> tuple[FeatureReader | None, Callable[[WindowSpec], tuple[str, ...]] | None]:
        """Open the protocol feature store, if this run may read it.

        Three conditions: the dataset declares a cache (``data.feature_store.enabled``), the
        adaptation preserves the backbone output (invariant 3), and the store exists on disk. A
        missing store is not an error: the run recomputes features. ``store.key`` checks that
        ``backbone_id``, ``ckpt_sha`` and ``preprocess_hash`` match the store, so a diverging
        geometry raises at the first batch.

        :param cfg: the resolved config.
        :param backbone: the built backbone (``backbone_id`` and ``ckpt_sha``).
        :param spec: the run ``PreprocessSpec``, part of the key.
        :param congelato: whether the adaptation preserves the backbone output.
        :return: ``(store, key function)``, or ``(None, None)``.
        :raises ValueError: if the cache is enabled without ``root``, or the backbone does not
            declare ``ckpt_sha``.
        """
        group = (cfg["data"].get("feature_store") or {}) if "data" in cfg else {}
        if not frozen or not bool(group.get("enabled", False)):
            return None, None
        root = group.get("root")
        if not root:
            raise ValueError(
                "data.feature_store.enabled is true but `root` is missing: without a root "
                "the cache cannot be located"
            )
        directory = Path(str(root)) / str(backbone.backbone_id) / spec.hash()[:12]
        if not (directory / "store_meta.json").is_file():
            log.info(
                "no feature store in %s: the run recomputes the features (correct, slower)",
                directory,
            )
            return None, None
        # `ckpt_sha` is not in the `VideoBackbone` Protocol (`TinyBackbone` has no checkpoint);
        # an empty value would be a key that does not identify the weights.
        ckpt_sha = str(getattr(backbone, "ckpt_sha", "") or "")
        if not ckpt_sha:
            raise ValueError(
                f"{backbone.backbone_id} does not declare `ckpt_sha` but the dataset requests the "
                f"cache: the ADR-007 key would not identify the weights"
            )
        store = SafetensorsFeatureStore.open_existing(directory, create_index=False)
        levels = ExperimentRunner._levels(
            cfg,
            n_blocks=len(backbone.blocks()),
            hierarchical=tuple(backbone.hierarchical_layers()),
            store_layers=tuple(int(x) for x in (group.get("layers") or ())),
        )
        log.info("feature store %s opened for reading, levels %s", directory, list(levels))

        def key_of(window: WindowSpec) -> tuple[str, ...]:
            return tuple(
                store.key(backbone.backbone_id, ckpt_sha, spec, level, window) for level in levels
            )

        return store, key_of

    @staticmethod
    def _check_regime(protocol: Mapping[str, Any]) -> None:
        """Reject a protocol of the few-shot regime, which this runner no longer executes.

        ``configs/protocol/fsp_v1.yaml`` and its manifests remain on disk (invariant 1) and are
        read with :mod:`evd.protocol.manifest`.

        :param protocol: the ``protocol`` group of the resolved config.
        :raises NotImplementedError: if the protocol belongs to the few-shot regime.
        """
        version = str(protocol.get("protocol_version", ""))
        if version.startswith("fsp"):
            raise NotImplementedError(
                f"protocol '{version}': few-shot regime, replaced by ADR-031 on "
                f"27 August. This runner executes the fully-supervised regime (sealed "
                f"split, every chunk of every video, binary output) and cannot "
                f"build a support set from a manifest. fsp_v1 manifests remain "
                f"readable with evd.protocol.manifest, but no run consumes them"
            )

    @staticmethod
    def _objective(protocol: Mapping[str, Any], cfg: Mapping[str, Any] | None = None) -> Objective:
        """Return what the head must predict, according to the protocol and the cell.

        ``output: binary`` (ADR-031 point 4) is the value in force; ``multiclass`` remains for
        the E0c sanity check on UCF101. A cell of the class-routing localization campaign may
        declare a top-level ``objective`` (``multiclass`` for a class head, ``multitask`` for a
        binary and a class head trained together, plan of 21 October §5.4): the protocol
        file is not touched, and detection stays the binary one.

        :param protocol: the ``protocol`` group.
        :param cfg: the run config, read for the cell ``objective``.
        :return: the ``Trainer`` objective.
        :raises ValueError: if a field declares an unknown value, or a cell overrides a
            protocol that is not binary.
        """
        value = str(protocol.get("output", "binary"))
        if value not in ("binary", "multiclass"):
            raise ValueError(
                f"protocol.output = {value!r}: allowed values are 'binary' "
                f"(ADR-031 point 4, the regime in force) and 'multiclass'"
            )
        declared = None if cfg is None else cfg.get("objective")
        if declared in (None, value):
            return "binary" if value == "binary" else "multiclass"
        if value != "binary" or declared not in ("multiclass", "multitask"):
            raise ValueError(
                f"objective = {declared!r} on protocol.output = {value!r}: a cell may only "
                f"turn a binary protocol into 'multiclass' or 'multitask'"
            )
        log.info("objective %s declared by the cell (protocol output: %s)", declared, value)
        return declared

    @staticmethod
    def _floor(protocol: Mapping[str, Any]) -> float:
        """:return: the overlap-weight floor (ADR-032 §5), read from the protocol."""
        return float((protocol.get("sampling") or {}).get("overlap_weight_floor", OVERLAP_FLOOR))

    @staticmethod
    def _dynamic(protocol: Mapping[str, Any]) -> bool:
        """Return whether training endpoints are drawn inside the hop cell.

        ``static`` is the grid with step ``hop``; ``dynamic`` draws the endpoint uniformly inside
        the cell and recomputes the label with the same positivity rule (ADR-031 point 6,
        ``docs/04`` §3.5). Dynamic sampling excludes the cache by construction.

        :raises ValueError: if the field declares an unknown value (a typo must not silently
            degrade to static).
        """
        value = str((protocol.get("temporal") or {}).get("train_sampling", "static"))
        if value not in ("static", "dynamic"):
            raise ValueError(
                f"protocol.temporal.train_sampling = {value!r}: allowed values are "
                f"'static' and 'dynamic' (ADR-031 point 6)"
            )
        return value == "dynamic"

    #: The protocol -> sampler translation lives in :mod:`evd.protocol.chunks`, shared by the
    #: diagnostic scripts and the feature extraction (ADR-049).
    _sampler = staticmethod(sampler_for_protocol)

    @staticmethod
    def _check_geometry(spec: PreprocessSpec, protocol: Mapping[str, Any]) -> None:
        """Check that protocol and backbone agree on time, resolution and crop.

        If they diverge, the enumerated chunk grid is not the declared one and the feature store
        would be read empty, without any error.

        :raises ValueError: if ``num_frames``, ``target_fps``, ``size`` or the preprocessing
            variant do not match.
        """
        temporal = protocol.get("temporal", {})
        expected_frames = temporal.get("num_frames")
        expected_fps = temporal.get("target_fps")
        if expected_frames is not None and int(expected_frames) != spec.num_frames:
            raise ValueError(
                f"protocol.temporal.num_frames = {expected_frames} but the backbone samples "
                f"{spec.num_frames} frames: the chunk grid would be different"
            )
        if expected_fps is not None and abs(float(expected_fps) - spec.fps) > 1e-9:
            raise ValueError(
                f"protocol.temporal.target_fps = {expected_fps} but the backbone requests "
                f"{spec.fps}: the span would not be the declared one (ADR-024)"
            )
        geometry = protocol.get("geometry", {})
        expected_size = geometry.get("size")
        expected_variant = geometry.get("preprocess_variant")
        if expected_size is not None and int(expected_size) != spec.size:
            raise ValueError(
                f"protocol.geometry.size = {expected_size} but the backbone uses {spec.size}: "
                f"this is the ADR-031/034 geometry, where the feature store is extracted"
            )
        if expected_variant is not None and str(expected_variant) != spec.geometry:
            raise ValueError(
                f"protocol.geometry.preprocess_variant = {expected_variant!r} but the backbone "
                f"uses {spec.geometry!r}: the preprocess_hash, and thus the cache key, changes"
            )

    @staticmethod
    def _n_classes(cfg: Mapping[str, Any], bundle: DatasetBundle, objective: Objective) -> int:
        """Return the number of head logits, dictated by the protocol, not by the ``head`` group.

        With binary output (ADR-031 point 4) it is one; the 13 classes stay in the data for the
        per-class breakdown (``docs/04`` §4.2). With a multiclass objective the dataset dictates
        the classes. ``head.n_classes`` is ``???`` in the config for this reason.

        :raises ValueError: if the config declares a different number.
        """
        expected = 1 if objective == "binary" else bundle.n_classes()
        declared = cfg["head"].get("n_classes")
        if declared not in (None, "???") and int(declared) != expected:
            reason = (
                "the output is binary (ADR-031 point 4): one logit, not one per class"
                if objective == "binary"
                else f"{bundle.name} has {expected} classes (Normal included)"
            )
            raise ValueError(f"head.n_classes declares {declared} but {reason}")
        return expected

    # --- data -------------------------------------------------------------------------
    @staticmethod
    def _chunks(
        video_ids: Sequence[str],
        bundle: DatasetBundle,
        sampler: CausalWindowSampler,
        overlap_floor: float,
        *,
        part: str,
    ) -> list[ChunkItem]:
        """Return the recomputed chunk grid of a partition (``docs/04`` §3).

        No window is discarded (ADR-031 point 2). Videos shorter than a window produce none:
        they are counted and declared.

        :param video_ids: the partition videos, in split order.
        :param bundle: dataset inventory and annotations.
        :param sampler: the grid geometry.
        :param overlap_floor: the overlap-weight floor.
        :param part: name of this partition in the logs.
        :return: the chunks, ordered by video and then by time.
        :raises ValueError: if the partition produces no chunk.
        """
        chunks = enumerate_chunks(
            video_ids,
            bundle.inventory,
            bundle.annotations,
            sampler,
            overlap_floor=overlap_floor,
        )
        with_windows = {c.video_id for c in chunks}
        without = [v for v in video_ids if v not in with_windows]
        if without:
            log.warning(
                "%s: %d of %d videos are shorter than a window and produce no chunk (%s%s)",
                part,
                len(without),
                len(video_ids),
                ", ".join(without[:5]),
                "…" if len(without) > 5 else "",
            )
        if not chunks:
            raise ValueError(
                f"partition '{part}' without chunks: none of the {len(video_ids)} videos "
                f"produces a window on the {sampler.grid!r} grid. Are the protocol span "
                f"and hop compatible with the video durations?"
            )
        return chunks

    @staticmethod
    def _resampler(
        bundle: DatasetBundle,
        sampler: CausalWindowSampler,
        overlap_floor: float,
        *,
        seed: int,
    ) -> Callable[[WindowItem, int, int], WindowItem]:
        """Return the dynamic endpoint sampler (ADR-031 point 6).

        At each draw the endpoint is drawn uniformly inside the chunk hop cell
        (:meth:`~evd.data.windows.CausalWindowSampler.endpoint_cell`) and the label is recomputed
        with the static-grid positivity rule (:func:`~evd.protocol.chunks.chunk_at`). The weight
        stays the static one, already consumed by the sampler. The seed depends on run, window,
        draw and worker, distinct from the augmentation seed.

        :return: ``(item, draw, worker) -> item``, the :data:`~evd.data.dataset.Resampler`
            contract.
        """

        def resample(item: WindowItem, draw: int, worker: int) -> WindowItem:
            video_id = item.window.video_id
            meta = bundle.meta(video_id)
            ann = bundle.annotations.get(video_id) or TemporalAnnotation(video_id=video_id)
            bottom, top = sampler.endpoint_cell(meta, item.window.end_frame)
            digest = stable_hash(
                ["endpoint", int(seed), item.window.window_id, int(draw), int(worker)]
            )
            end_frame = bottom + int(digest[:16], 16) % (top - bottom + 1)
            chunk = chunk_at(
                meta,
                ann,
                sampler,
                sampler.span_at(meta, end_frame),
                weight_video=1.0,
                overlap_floor=overlap_floor,
            )
            return replace(
                item,
                window=chunk.window,
                label=chunk.label,
                class_label=(
                    bundle.normal_idx if chunk.class_idx is None else int(chunk.class_idx)
                ),
            )

        return resample

    @staticmethod
    def _augment(cfg: Mapping[str, Any]) -> AugmentSpec:
        """Return the training augmentation declared by the config.

        A Hydra group of its own (``configs/augment/``) because its identity enters the
        ``run_id``.

        :param cfg: the resolved run config.
        :return: the chain; :data:`evd.data.augment.NESSUNA` if the group is missing.
        :raises ValueError: if the group values do not form a valid ``AugmentSpec``.
        """
        from evd.data.augment import NO_AUGMENTATION

        group = cfg.get("augment")
        if not group:
            return NO_AUGMENTATION
        # Fields are read by name, not with `_arguments` (which drops `id`, here an argument).
        # Same guard as `_budget`.
        check_known_fields(AugmentSpec, group, "augment")
        policy = group.get("rand_augment")
        return AugmentSpec(
            id=str(group["id"]),
            random_resize_scale=tuple(float(x) for x in group["random_resize_scale"]),
            random_resize_ratio=tuple(float(x) for x in group["random_resize_ratio"]),
            rand_augment=None if policy in (None, "", "null") else str(policy),
            rand_augment_exclude=tuple(str(x) for x in (group.get("rand_augment_exclude") or ())),
            horizontal_flip=bool(group["horizontal_flip"]),
            reprob=float(group["reprob"]),
            # `crop` must be read: without it `official_nocrop` became `crop=True` with
            # `scale=(1,1) ratio=(1,1)`, which on a 4:3 frame falls back to the 240x240 centre crop
            # (ADR-032 §3, ADR-038).
            crop=bool(group.get("crop", True)),
            motion_shift=bool(group.get("motion_shift", False)),
        )

    @staticmethod
    def _loader(
        items: Sequence[WindowItem],
        decoder: Any,
        spec: PreprocessSpec,
        budget: BudgetSpec,
        strategy: AdaptationStrategy,
        seed: int,
        *,
        store: FeatureReader | None = None,
        augment: AugmentSpec | None = None,
        weighted: bool = False,
        shuffle: bool = True,
        resample: Any | None = None,
    ) -> Any:
        """Build the loader of the right regime: invariant 3 is enforced here.

        A run that changes what the backbone computes does not receive the cache reference
        (ADR-007); the ``Trainer`` check is a second line. ``weighted`` enables the weighted
        sampling of ADR-032 §4-5, for training only.
        """
        # `num_workers` comes from the budget, not the environment: it changes the pixels (the
        # worker index is in the augmentation seed). The RAM cache is split across workers, and
        # its flag is on only with a positive budget (`CACHE_RAM_MB` is 0 since ADR-054).
        workers = int(budget.num_workers)
        cache_ram_mb = CACHE_RAM_MB // max(1, workers)
        common = {
            "decoder": decoder,
            "spec": spec,
            "batch_size": budget.batch_size,
            "shuffle": shuffle,
            "num_workers": workers,
            "generator": torch.Generator().manual_seed(seed),
            "cache_in_memory": cache_ram_mb > 0,
            "cache_in_memory_mb": cache_ram_mb,
            # Declared in the budget, not hard-wired here (ADR-038).
            "pin_memory": bool(budget.pin_memory),
            "persistent_workers": bool(budget.persistent_workers),
            "augment": augment,
            "run_seed": seed,
            "weighted": weighted,
            "resample": resample,
        }
        if preserves_backbone_output(strategy):
            return build_frozen_loader(list(items), cache=store, **common)
        return build_adapted_loader(list(items), **common)

    @staticmethod
    def _with_keys(
        items: Sequence[WindowItem], key_of: Callable[[WindowSpec], tuple[str, ...]] | None
    ) -> list[WindowItem]:
        """Return the same windows with the cache key computed.

        ``WindowDataset`` requires ``cache_keys`` on every item when it receives a reader.

        :param items: the windows.
        :param key_of: the key function, or ``None`` without a cache.
        :return: the list, unchanged if ``key_of`` is ``None``.
        """
        if key_of is None:
            return list(items)
        return [replace(item, cache_keys=key_of(item.window)) for item in items]

    @staticmethod
    def _warm_start(
        cfg: Mapping[str, Any],
        head: Any,
        strategy: AdaptationStrategy,
        device: str,
    ) -> dict[str, Any] | None:
        """Load the weights of another run as a starting point (ADR-062).

        The mechanism of multi-epoch training with a cyclic cosine: this run executes the next
        cycles starting where the source run arrived. Not a resume: the run is new (different
        budget, schedule and ``run_id``) and only the weights are inherited; ``meta.json``
        records ``avvio_a_caldo`` and leaves ``resumed_from`` empty. The weights of the last
        step are taken, not the selected ones; optimizer moments are not inherited (a warm
        restart, SGDR).

        :param cfg: the run config; reads the top-level ``init_from`` field.
        :param head: the freshly built head, receiving its weights.
        :param strategy: the already applied strategy, receiving its weights.
        :param device: where the parameters live.
        :return: the provenance for ``meta.json``, or ``None`` if the run starts from scratch.
        :raises FileNotFoundError: if the referenced checkpoint does not exist.
        :raises ValueError: if the checkpoint belongs to this very run (that is a resume).
        """
        path = cfg.get("init_from")
        if not path:
            return None
        import torch

        from evd.train.checkpoint import restore_adapter_state

        # Path resolution lives in `preadapt.resolve_checkpoint`, shared with the preadaptation
        # (ADR-052).
        file = resolve_checkpoint(str(path))
        data = torch.load(file, map_location="cpu", weights_only=True)
        identity = dict(data.get("identity") or {})
        source = str(identity.get("run_id", "?"))
        head_weights = dict(data["head"])
        adapter_weights = dict(data.get("adapter") or {})
        head.load_state_dict({k: v.to(device) for k, v in head_weights.items()})
        restore_adapter_state(strategy, {k: v.to(device) for k, v in adapter_weights.items()})
        log.info(
            "warm start from %s (run %s, step %s): %d head tensors, %d adapter tensors",
            file,
            source,
            data.get("step"),
            len(head_weights),
            len(adapter_weights),
        )
        return {
            "checkpoint": str(file),
            "run_id": source,
            "step": data.get("step"),
            "selected_step_sorgente": data.get("selected_step"),
            "pesi": "last step",
            "momenti_ottimizzatore": "reinizializzati (warm restart)",
            "n_tensori_testa": len(head_weights),
            "n_tensori_adattatore": len(adapter_weights),
        }

    def _inherit(
        self,
        cfg: Mapping[str, Any],
        head: Any,
        strategy: AdaptationStrategy,
        device: str,
    ) -> None:
        """Copy weights of a finished run into this cell and freeze them (``inherit`` block).

        The class heads B1 and B2 of the plan of 21 October (§5.4) train on the encoder of the
        detector, frozen, and B1 also on the first blocks of its binary head. ``inherit.from``
        names the checkpoint; ``inherit.state`` is ``selected`` (the weights chosen by the
        source model selection, the default) or ``last``; ``inherit.adapter: freeze`` restores
        the adapter weights and freezes them; ``inherit.head`` maps source head prefixes to
        prefixes of this head, whose tensors are copied and frozen. Unlike ``init_from``
        (a warm start, ADR-062) nothing inherited here is trained. The receipt goes to
        ``meta.json`` under ``inherit``.

        :param cfg: the run config.
        :param head: the freshly built head.
        :param strategy: the applied strategy.
        :param device: where the parameters live.
        :raises FileNotFoundError: if the checkpoint does not exist.
        :raises ValueError: on an unknown field or value, a missing selected state, or a head
            map that does not cover exactly the mapped parameters of this head.
        """
        block = cfg.get("inherit")
        if not block:
            return
        import torch

        from evd.train.checkpoint import restore_adapter_state

        unknown = sorted(set(block) - {"from", "state", "adapter", "head"})
        if unknown:
            raise ValueError(f"inherit: unknown fields {unknown}")
        state = str(block.get("state", "selected"))
        if state not in ("selected", "last"):
            raise ValueError(f"inherit.state must be selected or last, not {state!r}")
        adapter_mode = str(block.get("adapter") or "none")
        if adapter_mode not in ("none", "freeze"):
            raise ValueError(f"inherit.adapter must be freeze or none, not {adapter_mode!r}")
        file = resolve_checkpoint(str(block["from"]))
        data = torch.load(file, map_location="cpu", weights_only=True)
        if state == "selected":
            head_weights = dict(data.get("selected_head") or {})
            adapter_weights = dict(data.get("selected_adapter") or {})
            step = data.get("selected_step")
            if not head_weights or step is None:
                raise ValueError(f"{file} has no selected state to inherit")
        else:
            head_weights = dict(data["head"])
            adapter_weights = dict(data.get("adapter") or {})
            step = data.get("step")
        receipt: dict[str, Any] = {
            "checkpoint": str(file),
            "run_id": str(dict(data.get("identity") or {}).get("run_id", "?")),
            "state": state,
            "step": step,
            "adapter": adapter_mode,
        }
        if adapter_mode == "freeze":
            restore_adapter_state(strategy, {k: v.to(device) for k, v in adapter_weights.items()})
            frozen = 0
            for parameter in strategy.trainable_parameters():
                parameter.requires_grad_(False)
                frozen += int(parameter.numel())
            receipt.update(adapter_tensors=len(adapter_weights), adapter_params_frozen=frozen)
        mapping = {str(k): str(v) for k, v in dict(block.get("head") or {}).items()}
        if mapping:
            copied = {}
            for source, target in mapping.items():
                hits = {k: v for k, v in head_weights.items() if k.startswith(source)}
                if not hits:
                    raise ValueError(f"inherit.head: no source tensor starts with {source!r}")
                copied.update({target + k[len(source) :]: v for k, v in hits.items()})
            own = head.state_dict()
            expected = {k for k in own if any(k.startswith(t) for t in mapping.values())}
            if set(copied) != expected:
                raise ValueError(
                    f"inherit.head does not cover the mapped parameters exactly: missing "
                    f"{sorted(expected - set(copied))[:5]}, unexpected "
                    f"{sorted(set(copied) - expected)[:5]}"
                )
            if any(own[k].shape != v.shape for k, v in copied.items()):
                raise ValueError("inherit.head: a copied tensor has a different shape")
            head.load_state_dict({**own, **{k: v.to(device) for k, v in copied.items()}})
            parameters = dict(head.named_parameters())
            frozen_head = 0
            for name in copied:
                if name in parameters:
                    parameters[name].requires_grad_(False)
                    frozen_head += int(parameters[name].numel())
            receipt.update(
                head_map=mapping, head_tensors=len(copied), head_params_frozen=frozen_head
            )
        log.info("inherit from %s (%s state, step %s): %s", file, state, step, receipt)
        self._inherit_meta = receipt

    @staticmethod
    def _budget(cfg: Mapping[str, Any]) -> BudgetSpec:
        group = dict(cfg["train"])
        # No `train` field may remain without a reader (ADR-038, ADR-047).
        check_known_fields(BudgetSpec, group, "train", read_elsewhere=("checkpoint_every",))
        lr_grid = group.get("lr_grid", (1e-3,))
        return BudgetSpec(
            max_steps=int(group["max_steps"]),
            batch_size=int(group["batch_size"]),
            grad_accum=int(group.get("grad_accum", 1)),
            lr_grid=tuple(float(v) for v in lr_grid),
            weight_decay=float(group.get("weight_decay", 0.0)),
            lr_schedule=str(group.get("lr_schedule", "constant")),
            # The cyclic-cosine period must be read here (ADR-038 pattern: a declared field the
            # orchestrator does not pass on).
            lr_cycle_steps=(
                None if group.get("lr_cycle_steps") is None else int(group["lr_cycle_steps"])
            ),
            precision=str(group.get("precision", "fp32")),
            tf32=bool(group.get("tf32", False)),
            attn=str(group.get("attn", "sdpa")),
            early_stop_patience=group.get("early_stop_patience"),
            # `model_selection` must be read here (ADR-047).
            model_selection=str(group.get("model_selection", "last")),
            eval_every=int(group.get("eval_every", 1)),
            num_workers=int(group.get("num_workers", 0)),
            pin_memory=bool(group.get("pin_memory", False)),
            persistent_workers=bool(group.get("persistent_workers", False)),
            epochs=None if group.get("epochs") is None else int(group["epochs"]),
            max_steps_cap=(
                None if group.get("max_steps_cap") is None else int(group["max_steps_cap"])
            ),
            class_balance=str(group.get("class_balance", "none")),
        )

    # --- registration and evaluation ----------------------------------------------------
    def _open_run(
        self,
        cfg: Mapping[str, Any],
        run_id: RunId,
        split: SplitSpec,
        regime: Mapping[str, Any],
        spec: PreprocessSpec,
        report: AdaptationReport,
        chunks: Mapping[str, Any],
        budget: BudgetSpec,
        seed: int,
        warm_start: dict[str, Any] | None = None,
    ) -> RunHandle | None:
        """Open the run before the figure, so the artifacts land inside it.

        ``chunks`` is the verifiable grid signature (counts and measured positive rate of the
        three partitions, ``docs/04`` §3.6 and §8), written to ``meta.json`` before training.
        ``budget`` and ``seed`` are the values that governed this run (ADR-057).

        :param regime: the applied numeric regime ``(precision, tf32, attn, device, torch)``, to
            which ``size`` is added.
        :param budget: the cell budget, source of ``esecuzione``.
        :param seed: the run seed applied by ``seed_all``.
        :param avvio_a_caldo: the checkpoint this run inherits weights from, or ``None``
            (ADR-062).
        :return: the handle, or ``None`` if the config asks not to write.
        """
        if not bool(cfg.get("write_run", True)):
            return None
        return self.tracker.start(
            run_id.format(),
            cfg,
            official=run_id.is_official,
            regime={**regime, "size": spec.size},
            protocol_sha=split.sha256,
            extra_meta={
                "adapt": report.as_dict(),
                "chunks": dict(chunks),
                # The seed, at top level.
                "seed": int(seed),
                # Budget fields that change the numbers, next to the seed: `config.yaml` mixes them with
                # the `eval` and `data` groups, which declare the same names with different values.
                "esecuzione": {
                    "seed": int(seed),
                    "max_steps": int(budget.max_steps),
                    "batch_size": int(budget.batch_size),
                    "grad_accum": int(budget.grad_accum),
                    "batch_effettivo": int(budget.batch_size) * int(budget.grad_accum),
                    "num_workers": int(budget.num_workers),
                    "precision": str(budget.precision),
                    "model_selection": str(budget.model_selection),
                    "eval_every": int(budget.eval_every),
                    "lr_schedule": str(budget.lr_schedule),
                    "lr_cycle_steps": budget.lr_cycle_steps,
                },
                **({"avvio_a_caldo": warm_start} if warm_start else {}),
                # Which run the pre-trained family comes from and what was done with it (ADR-065).
                **({"preadattamento": self._preadapt_meta} if self._preadapt_meta else {}),
                **({"inherit": self._inherit_meta} if self._inherit_meta else {}),
                "class_balance": str(budget.class_balance),
            },
            # Variables resolved by `hpc/env.sh` and not consumed here: the matrix budget comes from
            # Hydra. Recorded as provenance under a key saying they did not govern (ADR-057).
            ignored_env=("EVD_SEED", "EVD_PRECISION", "EVD_BATCH_SIZE", "EVD_TF32"),
        )

    def _evaluate(
        self,
        *,
        cfg: Mapping[str, Any],
        bundle: DatasetBundle,
        split: SplitSpec,
        backbone: Any,
        head: Any,
        spec: PreprocessSpec,
        sampler: CausalWindowSampler,
        decoder: Any,
        val: Sequence[WindowItem],
        device: str,
        handle: RunHandle | None,
        levels: tuple[int, ...],
        store: FeatureReader | None = None,
        key_of: Callable[[WindowSpec], tuple[str, ...]] | None = None,
        sampler_val: CausalWindowSampler | None = None,
        objective: Objective = "binary",
        selection_scores: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """Evaluate on a split and produce the ``docs/04`` §4.3 metrics.

        ``cfg["eval_split"]`` selects the split (default ``test``); ``val`` exists for full-budget
        pilots that must not touch the test set (ADR-040). With ``val`` the threshold τ is
        estimated on the evaluated videos, so metrics are optimistic and not comparable with a
        test-evaluated run; ``metrics.json`` declares it with ``threshold_on_eval_split``.

        :raises ValueError: if ``eval_split`` is unknown or the partition is empty.
        """
        protocol = cfg["protocol"]
        evaluation = cfg.get("eval", {})
        split_name = str(cfg.get("eval_split", "test"))
        if split_name not in ("test", "val"):
            raise ValueError(
                f"unknown eval_split: {split_name!r}, expected 'test' or 'val'. "
                f"There is no third place to read a number"
            )
        video_ids: list[str] = list(split.val if split_name == "val" else split.test)
        if not video_ids:
            raise ValueError(
                f"eval_split={split_name!r} but that partition is empty in split "
                f"{split.protocol_version}: there is nothing to evaluate"
            )
        if split_name == "val":
            log.warning(
                "evaluating on VALIDATION (%d videos): the threshold is estimated on the same "
                "videos, so metrics are optimistic and not comparable with "
                "a run evaluated on test",
                len(video_ids),
            )
        normal_idx = protocol_normal_idx(protocol, bundle)
        # Evaluation numeric regime (ADR-044); default `fp32`, the historical behaviour, while
        # `configs/eval/` declares the value in force.
        eval_precision = str(evaluation.get("precision", "fp32"))
        scorer = HeadScorer(
            backbone,
            head,
            levels=levels,
            normal_idx=normal_idx,
            device=device,
            precision=eval_precision,
        )
        mapper = FrameScoreMapper(policy=str(protocol["temporal"]["frame_mapping"]))
        # Evaluation workers: unlike training (ADR-038) they do not change the numbers (no
        # augmentation, deterministic preprocessing). See ADR-041.
        worker_eval = int(evaluation.get("num_workers", 0))
        # Validation has its own contiguous grid (ADR-047): with the test sampler it would have
        # ~16,700 windows instead of 8383 and τ would move.
        engine = InferenceEngine(
            scorer,
            sampler_val if split_name == "val" and sampler_val is not None else sampler,
            mapper,
            decoder,
            spec,
            batch_size=int(evaluation.get("batch_size", 4)),
            num_workers=worker_eval,
            cache=store,
            cache_keys=key_of,
        )
        metas: list[VideoMeta] = [bundle.meta(v) for v in video_ids]
        results = engine.run_videos(metas)
        chunk = {v: r.window_scores for v, r in results.items()}
        frame = {v: r.frame_scores for v, r in results.items()}
        gt: dict[str, TemporalAnnotation] = {
            v: (bundle.annotations.get(v) or TemporalAnnotation(video_id=v)) for v in video_ids
        }

        # ADR-043: when evaluating ON validation, τ is free (the same windows were just scored).
        known: dict[str, float] | None = None
        if split_name == "val":
            known = {
                ws.window.window_id: float(ws.anomaly)
                for video_scores in chunk.values()
                for ws in video_scores
            }
            score_source = "eval_pass"
        else:
            # ADR-053: on `test` the validation scores already exist, from the selected training step
            # (`FitResult.selected_scores`); reusing them removes a seventh pass.
            known = self._known_from_selection(
                selection_scores,
                objective=objective,
                normal_idx=normal_idx,
                precisione_eval=eval_precision,
                precisione_train=str(cfg.get("train", {}).get("precision", "fp32")),
            )
            score_source = "selected_step" if known is not None else "eval_pass"
        val_scores = self._validation_scores(
            scorer,
            val,
            spec,
            decoder,
            store=store,
            num_workers=worker_eval,
            known_scores=known,
        )
        threshold, source = self._soglia(protocol, val_scores, val)
        normal_scores = [
            score for score, item in zip(val_scores, val, strict=True) if item.label == 0
        ]
        # --- scores on disk (ADR-051) ---------------------------------------------------- #
        # Rows come from `M.chunk_records`, the only function that labels a chunk, so file and
        # metrics cannot diverge.
        if handle is not None:
            self._write_scores(handle, chunk, gt, val_scores, val, split_name)
        # Protocol metrics have a single author, `evd.eval.valutazione`, shared with the external
        # evaluation (`exp/esterno.py`). What describes the RUN stays here: partition, numeric
        # regime and where τ comes from.
        metrics: dict[str, Any] = V.metriche_di_valutazione(
            chunk=chunk,
            gt=gt,
            soglia=threshold,
            sorgente=source,
            classi=bundle.classes,
            normal_idx=normal_idx,
            fps={v: bundle.meta(v).fps for v in gt},
            num_frames={v: bundle.meta(v).num_frames for v in gt},
            protocol=protocol,
            requests=evaluation.get("metrics") or {},
            negative_scores=normal_scores,
        )
        metrics.update(
            {
                "eval_split": split_name,
                # The numeric regime of these numbers (ADR-044), always written: `eval.*` is outside the
                # `config_sha`.
                "eval_precision": eval_precision,
                "n_eval_videos": len(video_ids),
                "n_chunk_eval": sum(len(v) for v in chunk.values()),
                "threshold": {
                    "value": threshold,
                    "source": source,
                    # With Youden the sample is the whole validation set; with the quantile only negatives.
                    # Both counts are reported.
                    "n_scores": len(val_scores),
                    "n_negativi": len(normal_scores),
                    # Where the τ scores come from (ADR-053): `eval_pass` or `selected_step`. Always written,
                    # since the shortcut can disable itself.
                    "scores_from": score_source,
                },
            }
        )
        if split_name == "test":
            # The two historical names are emitted only when true (they are in the `docs/03`
            # contract).
            metrics["n_test_videos"] = metrics["n_eval_videos"]
            metrics["n_chunk_test"] = metrics["n_chunk_eval"]
        else:
            # τ was estimated on the same videos it then judged.
            metrics["threshold_on_eval_split"] = True
        if store is not None:
            # A hit rate below 1 is a cost, not an error: it says how many windows were recomputed.
            metrics["cache_hit_rate"] = engine.cache_hit_rate()
            metrics["cache_hits"] = engine.cache_hits
            metrics["cache_misses"] = engine.cache_misses
        if handle is not None and bool(cfg.get("make_figure", False)):
            chosen = next((v for v in video_ids if not gt[v].is_normal), video_ids[0])
            metrics["figure"] = str(
                timeline_png(
                    frame[chosen],
                    handle.dir / "timeline.png",
                    gt=gt[chosen],
                    threshold=threshold,
                    fps=bundle.meta(chosen).fps,
                    title=f"{handle.run_id} — {chosen}",
                )
            )
        return metrics

    @staticmethod
    def _write_scores(
        handle: RunHandle,
        chunk: Mapping[str, Sequence[WindowScores]],
        gt: Mapping[str, TemporalAnnotation],
        val_scores: Sequence[float],
        val: Sequence[WindowItem],
        split_name: str,
    ) -> None:
        """Write ``scores.jsonl``: one row per window, evaluation and validation.

        Makes recomputation from disk possible (ADR-048/049), feeds ROC/PR curves and confusion
        matrices (ADR-051), and answers new questions without paying inference again. Both
        partitions share the file, distinguished by ``split``. Labels come from
        :func:`~evd.eval.metrics.chunk_records` (evaluation) and ``WindowItem.label``
        (validation), the same sources as the metrics.

        :param handle: the current run; its directory is used.
        :param chunk: chunk scores of the evaluated partition, per video.
        :param gt: annotations of the evaluated videos.
        :param val_scores: validation window scores, in ``val`` order.
        :param val: validation windows, with their label.
        :param split_name: ``test`` or ``val``, the evaluated partition.
        """
        rows = [{**r, "split": split_name} for r in M.chunk_records(chunk, gt)]
        # Validation is added only if it is not the evaluated partition (otherwise the same
        # windows would be written twice).
        if split_name != "val":
            for score, item in zip(val_scores, val, strict=True):
                # Same row builder as the evaluation: the label sources differ (ADR-051), the shape does
                # not.
                rows.append(
                    {
                        **M.score_record(
                            item.window,
                            score,
                            -1 if item.label is None else int(item.label),
                            item.class_label,
                        ),
                        "split": "val",
                    }
                )
        path = write_jsonl(handle.dir / "scores.jsonl", rows)
        log.info("scores of %d windows written to %s", len(rows), path)

    @staticmethod
    def _soglia(
        protocol: Mapping[str, Any],
        punteggi: Sequence[float],
        val: Sequence[WindowItem],
    ) -> tuple[float, str]:
        """Apply the protocol threshold rule to the validation sample.

        The rule is declared in ``protocol.threshold.rule``:

        ``"youden"``
            the ROC point maximising ``TPR - FPR`` (ADR-047), the rule in force.
        ``"quantile"``
            the ``threshold.quantile`` quantile of the negative-chunk scores (accepted false
            alarm rate), the ``edp_v1`` rule and the default.

        The sample is always validation, never the evaluation split, and the rule must be the
        same for every method (invariant 5).

        :param protocol: the ``protocol`` group, from which ``threshold`` is read.
        :param punteggi: validation window scores, in ``val`` order.
        :param val: validation windows, carrying the label.
        :return: the threshold and its provenance, stored in ``PRF``.
        :raises ValueError: if the protocol declares a source other than ``val``, an unknown
            rule, or if the sample is degenerate.
        """
        block = protocol["threshold"]
        source = str(block["source"])
        if source != "val":
            raise ValueError(
                f"threshold.source = {source!r}: in the fully-supervised regime the threshold is "
                f"estimated on the validation partition and nothing else (docs/04 §5.3). "
                f"'support' was the few-shot rule, where validation would have spent "
                f"labels outside the budget"
            )
        rule = str(block.get("rule", "quantile"))
        if rule not in ("youden", "quantile"):
            raise ValueError(
                f"threshold.rule = {rule!r}: the rules are 'youden' (ADR-047, in force) "
                f"and 'quantile' (edp_v1)"
            )
        if rule == "youden":
            return M.threshold_youden(punteggi, [item.label for item in val])
        normals = [score for score, item in zip(punteggi, val, strict=True) if item.label == 0]
        if not normals:
            raise ValueError(
                "validation has no negative chunk: the false-alarm quantile is undefined"
            )
        if "quantile" not in block:
            raise ValueError(
                "threshold.rule = 'quantile' but the protocol does not declare "
                "threshold.quantile: the quantile is the rule itself, and choosing one "
                "here would be a protocol decision taken by the runner"
            )
        return M.threshold_from_scores(
            normals,
            quantile=float(block["quantile"]),
            source=source,  # type: ignore[arg-type]
        )

    @staticmethod
    def _known_from_selection(
        selection_scores: Mapping[str, float] | None,
        *,
        objective: Objective,
        normal_idx: int | None,
        precisione_eval: str,
        precisione_train: str,
    ) -> dict[str, float] | None:
        """Return the validation scores of the selected step, if usable for τ.

        ADR-053: with ``model_selection: val_auroc`` training evaluates the whole validation set
        every ``eval_every`` steps, and :class:`~evd.train.trainer.FitResult` carries the logits
        of the selected step, i.e. of the weights being evaluated. Valid only if both forwards
        ran in the same numeric regime (``budget.precision`` vs ``eval.precision``, ADR-044);
        otherwise the pass is redone.

        :param selection_scores: ``window_id -> logit``, or ``None``/empty if training did not
            select.
        :param objective: the training objective; only ``binary`` produces one logit per window.
        :param normal_idx: index of the ``Normal`` class, for the multiclass convention of
            :func:`~evd.eval.scorers.anomaly_from_logits`.
        :param precisione_eval: regime of this evaluation.
        :param precisione_train: regime in which training evaluated validation.
        :return: ``window_id -> anomaly score``, or ``None`` if the shortcut does not apply.
        """
        if not selection_scores:
            return None
        if objective != "binary":
            # Outside the binary objective a window has no single logit, so `FitResult` carries none.
            log.warning(
                "selection scores ignored: objective=%r is not binary, tau is "
                "estimated with a pass over validation",
                objective,
            )
            return None
        if precisione_eval != precisione_train:
            log.warning(
                "selection scores ignored: training validation ran in "
                "%s and this evaluation runs in %s, so tau would be estimated in one "
                "numeric regime and applied in another (ADR-044). Redoing the pass",
                precisione_train,
                precisione_eval,
            )
            return None
        ids = list(selection_scores)
        logit = torch.tensor([float(selection_scores[i]) for i in ids]).unsqueeze(1)
        # The conversion goes through the scorer function: it is THE score definition.
        anomaly, _ = anomaly_from_logits(
            logit, batch_size=len(ids), n_views=None, normal_idx=normal_idx
        )
        log.info(
            "tau from scores already computed at the selected step (%d windows, %s): "
            "no seventh validation pass (ADR-053)",
            len(ids),
            precisione_train,
        )
        return {i: float(v) for i, v in zip(ids, anomaly.tolist(), strict=True)}

    @staticmethod
    def _validation_scores(
        scorer: HeadScorer,
        val: Sequence[WindowItem],
        spec: PreprocessSpec,
        decoder: Any,
        *,
        store: FeatureReader | None = None,
        num_workers: int = 0,
        known_scores: Mapping[str, float] | None = None,
    ) -> list[float]:
        """Return the score of every validation window, in ``val`` order.

        Collecting scores and applying a rule change for different reasons, hence a separate
        function from :meth:`_soglia`.

        :param known_scores: ``window_id -> score`` already computed elsewhere: by
            :meth:`_evaluate` when evaluating on validation (ADR-043), or by training at the
            selected step when evaluating on test (ADR-053). Partial coverage raises instead of
            silently falling back.
        :return: the scores, one per window, in the same order.
        :raises ValueError: if ``known_scores`` does not cover every window.
        :raises RuntimeError: if the loader yields a different number of scores.
        """
        from evd.data.dataset import WindowDataset, homogeneous_batches

        # Without `known_scores` this is an extra pass over validation, as costly as the others;
        # under `edp_v2` it runs only when the shortcut does not apply.
        scores: list[float] = [0.0] * len(val)
        if known_scores is not None:
            # ADR-043: matching by `window_id`, never by position (the two orders differ).
            missing = [
                item.window.window_id for item in val if item.window.window_id not in known_scores
            ]
            if missing:
                raise ValueError(
                    f"{len(missing)} of {len(val)} validation windows have no "
                    f"precomputed score (e.g. {missing[0]!r}): the two passes are "
                    f"looking at different grids, and τ would be estimated on a sample "
                    f"different from the declared one"
                )
            scores = [float(known_scores[item.window.window_id]) for item in val]
        elif store is None and num_workers > 0:
            from evd.data.dataset import build_adapted_loader

            loader = build_adapted_loader(
                list(val),
                decoder=decoder,
                spec=spec,
                batch_size=8,
                shuffle=False,
                num_workers=num_workers,
                # Single pass (`shuffle=False`): the RAM cache can never hit.
                cache_in_memory=False,
                augment=None,
            )
            i = 0
            for batch in loader:
                for ws in scorer.score(batch):
                    scores[i] = ws.anomaly
                    i += 1
            if i != len(val):
                raise RuntimeError(
                    f"the loader yielded {i} scores for {len(val)} validation windows: "
                    f"the threshold would be estimated on a sample different from the declared one"
                )
        else:
            # With the feature store, batches must be homogeneous in provenance (`collate_windows`
            # rejects mixed batches); `homogeneous_batches` is a partition, so no cache hit either.
            dataset = WindowDataset(list(val), decoder, spec, cache=store, cache_in_memory=False)
            for slice_ in homogeneous_batches(val, store, 8):
                batch = collate_windows([dataset[i] for i in slice_])
                for i, ws in zip(slice_, scorer.score(batch), strict=True):
                    scores[i] = ws.anomaly
        return scores


def protocol_normal_idx(protocol: Mapping[str, Any], bundle: DatasetBundle) -> int:
    """Return the ``Normal`` index, checking the two declarations against each other.

    The protocol carries it because per-class metrics skip ``Normal`` (``docs/04`` §4.2);
    the bundle carries the loaded catalogue's. If they diverge, per-class rows would be
    shifted. On UCF-Crime it is 13, in the synthetic world 0 (ADR-024 point 7).

    :return: the agreed index.
    :raises ValueError: if the two disagree.
    """
    declared = (protocol.get("classes") or {}).get("normal_idx")
    if declared is not None and int(declared) != bundle.normal_idx:
        raise ValueError(
            f"the protocol places Normal at index {declared}, the dataset at "
            f"{bundle.normal_idx}: protocol and catalogue come from different versions"
        )
    return bundle.normal_idx
