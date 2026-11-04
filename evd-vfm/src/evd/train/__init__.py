"""Training: a single loop for probes and PEFT."""

from evd.train.budget import BudgetSpec, apply_numeric_regime
from evd.train.trainer import FitResult, Trainer

__all__ = [
    "BudgetSpec",
    "apply_numeric_regime",
    "Trainer",
    "FitResult",
]
