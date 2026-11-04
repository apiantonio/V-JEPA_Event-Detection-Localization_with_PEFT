"""Partial receipt of the LR selection: what was already tried, on disk.

Grid trials have no checkpoints by construction
(:meth:`evd.exp.runner.ExperimentRunner._checkpointer`), so a timeout used to discard
the whole cell. This module remembers finished trials only: a finished trial is a full
fit over ``max_steps`` steps, so reusing it introduces no asymmetry between grid points.

The identity makes resumption well defined: on mismatch (another cell, config, split or
grid) the file is ignored and the selection restarts, with a log line.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evd.utils.io import ensure_dir, read_json, write_json
from evd.utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["RECEIPT_NAME", "TrialsReceipt"]

#: File inside the run folder, rewritten atomically after every trial.
RECEIPT_NAME = "lr_trials.json"

#: Relative tolerance used to recognize a grid learning rate in the file (JSON round
#: trip only).
LR_TOLERANCE = 1e-12


@dataclass
class TrialsReceipt:
    """Grid trials already completed by this cell, on disk.

    :param path: the file; its folder is created on first write.
    :param identita: what must match for the file to be reusable (``run_id``,
        ``config_sha``, ``split_sha`` and the grid).
    :param prove: ``lr -> score``, finished trials only.
    """

    path: Path
    identita: dict[str, Any]
    prove: dict[float, float] = field(default_factory=dict)

    @classmethod
    def open_(
        cls,
        path: str | Path,
        *,
        run_id: str,
        config_sha: str,
        split_sha: str,
        lr_grid: Sequence[float],
    ) -> TrialsReceipt:
        """Load the receipt, if it belongs to this cell.

        :param path: file to read.
        :param run_id: cell identity.
        :param config_sha: fingerprint of the resolved config.
        :param split_sha: seal of the consumed split.
        :param lr_grid: declared grid; a different grid is another experiment.
        :return: the receipt, empty if the file is missing or belongs to another cell.
        """
        identity = {
            "run_id": str(run_id),
            "config_sha": str(config_sha),
            "split_sha": str(split_sha),
            "lr_grid": [float(v) for v in lr_grid],
        }
        receipt = cls(path=Path(path), identita=identity)
        if not receipt.path.exists():
            return receipt
        try:
            content = read_json(receipt.path)
        except Exception as error:  # file truncated by a SIGKILL, or not JSON
            log.warning("receipt %s unreadable (%s): restarting from the first trial", path, error)
            return receipt
        if not isinstance(content, Mapping) or content.get("identita") != identity:
            log.warning(
                "receipt %s belongs to another cell or config: ignoring it and "
                "restarting from the first trial",
                path,
            )
            return receipt
        for row in content.get("prove", []):
            receipt.prove[float(row["lr"])] = float(row["val"])
        if receipt.prove:
            log.info(
                "resuming the selection: %d of %d trials already done (%s)",
                len(receipt.prove),
                len(identity["lr_grid"]),
                ", ".join(f"lr={lr:g}→{v:.6f}" for lr, v in sorted(receipt.prove.items())),
            )
        return receipt

    def score_of(self, lr: float) -> float | None:
        """:return: the score already measured for ``lr``, or ``None``."""
        for tried, score in self.prove.items():
            if math.isclose(tried, float(lr), rel_tol=LR_TOLERANCE, abs_tol=0.0):
                return score
        return None

    def record(self, lr: float, score: float) -> None:
        """Add a finished trial and rewrite the file.

        :param lr: tried learning rate.
        :param punteggio: validation score at the final state.
        """
        self.prove[float(lr)] = float(score)
        ensure_dir(self.path.parent)
        write_json(
            self.path,
            {
                "identita": self.identita,
                "prove": [{"lr": lr, "val": v} for lr, v in sorted(self.prove.items())],
            },
        )
