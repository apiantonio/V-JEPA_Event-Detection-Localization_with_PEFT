"""Hyper-parameter selection within the supervision budget.

In the active fully supervised regime the learning rate is selected on the
validation partition of the split (docs/04 §5.3): see :func:`fully_supervised_regime`.

The few-shot regime, kept to reread registered runs, had three cases decided by ``k``:

* ``k >= 4``: episodic 75/25 split of the support, stratified by class; the grid is
  tried on 75%, measured on 25%, and the chosen config is refit on the whole support;
* ``k in {1, 2}``: no selection, the config chosen at k=4 is inherited;
* ``kna``: runs outside the few-shot curve (smoke, E0 blocks), nothing to select.

"Same grid" means same cardinality, not same values: APT bank gradients are three to
six orders of magnitude below VPT ones. The module receives from the caller a function
mapping a learning rate to a validation score and knows nothing about training.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from evd.utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "VAL_FRACTION",
    "MIN_K_FOR_SELECTION",
    "Regime",
    "SelectionOutcome",
    "episodic_split",
    "fully_supervised_regime",
    "regime_per_k",
    "select_lr",
]

#: Below this k no selection is made and the config is inherited (docs/04 §5).
MIN_K_FOR_SELECTION = 4

#: Fraction of the support used for validation (protocol rule; changing it changes the
#: meaning of the few-shot curve).
VAL_FRACTION = 0.25

Regime = Literal[
    "val_split", "val_episodico", "ereditata", "griglia_a_un_valore", "non_sulla_curva"
]

T = TypeVar("T")


@dataclass(frozen=True)
class SelectionOutcome:
    """What was chosen, with which rule and on which evidence (written to ``metrics.json``).

    :param lr: chosen learning rate.
    :param regime: which docs/04 §5 rule decided.
    :param provenance: where the LR comes from, as a readable sentence.
    :param trials: ``(lr, val_score)`` for each tried point, in grid order; empty without
        selection.
    :param n_val: number of validation windows that decided; ``0`` without selection.
    """

    lr: float
    regime: Regime
    provenance: str
    trials: tuple[tuple[float, float], ...] = ()
    n_val: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "lr": self.lr,
            "regime": self.regime,
            "provenance": self.provenance,
            "trials": [list(t) for t in self.trials],
            "n_val": self.n_val,
        }


def regime_per_k(k: str | int | None, *, n_lr: int) -> Regime:
    """Return which docs/04 §5 rule applies to this run.

    :param k: the run ``k``: a grid integer, or ``"na"``/``None`` for runs outside the
        few-shot curve; ``"full"`` counts as a large k.
    :param n_lr: cardinality of the learning-rate grid.
    :return: the regime.
    :raises ValueError: if ``k`` is neither a grid integer nor ``'full'``/``'na'``.
    """
    text = "na" if k is None else str(k)
    if text in ("na", ""):
        return "non_sulla_curva"
    if n_lr <= 1:
        # A one-point grid is not a selection.
        return "griglia_a_un_valore"
    if text == "full":
        return "val_episodico"
    if not text.isdigit():
        raise ValueError(
            f"k = {text!r} is neither a grid integer nor 'full'/'na': the selection "
            f"rule cannot be chosen (docs/04 §5)"
        )
    return "val_episodico" if int(text) >= MIN_K_FOR_SELECTION else "ereditata"


def fully_supervised_regime(*, n_lr: int, n_val: int) -> Regime:
    """Return which rule applies to a fully supervised run (docs/04 §5.3).

    Either the grid has more than one point and the choice is made on validation, or it
    has one and there is nothing to choose.

    :param n_lr: cardinality of the learning-rate grid.
    :param n_val: windows in the validation partition.
    :return: the regime.
    :raises ValueError: if the grid has several points but validation is empty (no
        fallback on training or test).
    """
    if n_lr <= 1:
        return "griglia_a_un_valore"
    if n_val <= 0:
        raise ValueError(
            f"the grid has {n_lr} learning rates but the validation partition is empty: "
            f"the selection has nothing to measure on. Declare `val` in the split "
            f"(docs/04 §5.3), or reduce `train.lr_grid` to a single point and declare "
            f"where that value comes from"
        )
    return "val_split"


def episodic_split(
    items: Sequence[T],
    *,
    seed: int,
    val_fraction: float = VAL_FRACTION,
    classe: Callable[[T], int] | None = None,
) -> tuple[list[T], list[T]]:
    """Split the support into training and validation, stratified by class.

    Stratified because with a small k uniform sampling leaves whole classes out of
    validation; deterministic for a given seed. A class that would get zero validation
    elements by rounding gives one if it has at least two; a class with a single element
    stays in training.

    :param items: support windows (never query windows, invariant 6).
    :param seed: split seed.
    :param val_fraction: validation fraction, the protocol one by default.
    :param classe: how the class of an element is read; defaults to ``item.class_label``.
    :return: ``(train, val)``, each in the original order.
    :raises ValueError: if ``val_fraction`` is not in ``(0, 1)``, ``items`` is empty, or an
        element has no class.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0,1), not {val_fraction}")
    if not items:
        raise ValueError("empty support: nothing to split into training and validation")

    def _class_name(item: T) -> int:
        if classe is not None:
            return classe(item)
        label = getattr(item, "class_label", None)
        if label is None:
            raise ValueError(
                "a support window without class_label cannot be stratified: "
                "support windows must carry their class (WindowItem.class_label)"
            )
        return int(label)

    per_class: dict[int, list[int]] = {}
    for index, item in enumerate(items):
        per_class.setdefault(_class_name(item), []).append(index)

    val_indices: set[int] = set()
    for label_text in sorted(per_class):
        indices = list(per_class[label_text])
        # One RNG per class, seeded with the class, so adding a class does not reshuffle the
        # others; a string seed goes through sha512 and is deterministic across processes.
        random.Random(f"{seed}:{label_text}").shuffle(indices)
        how_many = int(len(indices) * val_fraction)
        if how_many == 0 and len(indices) >= 2:
            how_many = 1
        val_indices.update(indices[:how_many])

    training = [item for i, item in enumerate(items) if i not in val_indices]
    val = [item for i, item in enumerate(items) if i in val_indices]
    if not training:
        raise ValueError(
            f"the episodic split would leave training empty ({len(items)} windows, "
            f"val_fraction={val_fraction}): the support is too small for a selection"
        )
    log.info(
        "episodic split: %d training windows, %d validation windows over %d classes",
        len(training),
        len(val),
        len(per_class),
    )
    return training, val


def select_lr(
    lr_grid: Sequence[float],
    evaluate: Callable[[float], float],
    *,
    k: str | int | None = None,
    n_val: int = 0,
    regime: Regime | None = None,
    lower_is_better: bool = True,
) -> SelectionOutcome:
    """Select the learning rate according to the protocol rule.

    ``evaluate`` trains on the training split and returns the validation score; the refit
    on the whole data with the chosen LR is the caller's job.

    :param lr_grid: the family grid; its cardinality must match across families
        (invariant 5).
    :param evaluate: maps a learning rate to a validation score; called once per grid
        point, only in the ``val_episodico`` regime.
    :param k: run ``k`` in the few-shot regime, used when ``regime`` is not given.
    :param n_val: number of validation windows, for the receipt.
    :param regime: regime already decided by the caller (fully supervised path, see
        :func:`fully_supervised_regime`); ``None`` derives it from ``k``.
    :param lower_is_better: true if the score is a loss.
    :return: the chosen LR with its provenance, ready for ``metrics.json``.
    :raises ValueError: if the grid is empty.
    """
    if not lr_grid:
        raise ValueError("empty lr_grid: the selection protocol is part of the budget")

    chosen = regime if regime is not None else regime_per_k(k, n_lr=len(lr_grid))
    if chosen not in ("val_episodico", "val_split"):
        reason = {
            "griglia_a_un_valore": "the grid has a single point: nothing to choose",
            "ereditata": (
                f"k < {MIN_K_FOR_SELECTION}: the config is inherited from k="
                f"{MIN_K_FOR_SELECTION} (docs/04 §5), the first grid point is used here"
            ),
            "non_sulla_curva": "run without k on the few-shot curve: no selection",
        }[chosen]
        return SelectionOutcome(lr=float(lr_grid[0]), regime=chosen, provenance=reason)

    trials: list[tuple[float, float]] = []
    for lr in lr_grid:
        score = float(evaluate(float(lr)))
        trials.append((float(lr), score))
        log.info("LR selection: lr=%g → val=%.6f", lr, score)

    best = (min if lower_is_better else max)(trials, key=lambda p: p[1])
    provenance = (
        f"chosen on the validation partition of the split ({n_val} chunks) "
        f"among {len(lr_grid)} points; refit on the whole training set"
        if chosen == "val_split"
        else (
            f"chosen on episodic validation ({VAL_FRACTION:.0%} of the support, {n_val} windows) "
            f"among {len(lr_grid)} points; refit on the whole support"
        )
    )
    return SelectionOutcome(
        lr=best[0],
        regime=chosen,
        provenance=provenance,
        trials=tuple(trials),
        n_val=n_val,
    )
