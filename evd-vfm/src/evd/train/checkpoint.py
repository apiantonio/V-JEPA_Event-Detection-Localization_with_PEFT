"""Intra-training checkpoints and resumption.

On the cluster the wall time is 7 hours per slice and ``--requeue`` does not extend it,
so long trainings must be split into verifiable segments. Guarantees:

1. Backbone weights are never saved: head, ``state_dict_adapter()``, optimizer and RNG
   are; the backbone is identified by ``backbone_id``/``ckpt_sha``.
2. The checkpoint carries its identity and resumption rejects a wrong one (resuming
   another config would silently produce a number).
3. Writes are atomic: a ``torch.save`` killed at wall time must not leave a truncated
   file.

``signal.SIGUSR1`` does not exist on Windows: handlers are installed behind a guard and
the ``Trainer`` relies on the flag, which signals only raise.
"""

from __future__ import annotations

import signal
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from evd.utils.io import ensure_dir, replace_atomic, temp_sibling
from evd.utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "CHECKPOINT_NAME",
    "CheckpointIdentity",
    "CheckpointSpec",
    "Checkpointer",
    "InterruptFlag",
    "TrainingInterrupted",
    "TrainingState",
    "install_interrupt_handler",
    "restore_adapter_state",
]

#: The file inside ``CheckpointSpec.dir``; a single one, overwritten (history lives in
#: ``history.jsonl``).
CHECKPOINT_NAME = "checkpoint.pt"

#: Signals intercepted: ``SIGUSR1`` is sent by SLURM with ``--signal=B:USR1@300``;
#: ``SIGTERM`` is the wall-time kill, which leaves a few tens of seconds.
SIGNALS = ("SIGUSR1", "SIGTERM")


class TrainingInterrupted(RuntimeError):
    """Training stopped on request with budget left.

    Not a partial result to evaluate: the caller must relaunch the same cell, which resumes
    from the checkpoint.
    """


@dataclass(frozen=True)
class CheckpointIdentity:
    """Which run a checkpoint belongs to.

    The condition for resumption to be well defined: a checkpoint of another cell or config
    would resume and produce a number instead of an error.

    :param run_id: cell identity, in the docs/05 §8 grammar.
    :param config_sha: hash of the resolved config; distinguishes runs sharing a
        ``run_id`` but not their hyper-parameters.
    :param backbone_id: which backbone (not contained in the checkpoint).
    :param git_sha: the commit; ``None`` if git does not answer.
    :param protocol_sha: seal of the consumed protocol file (the ``edp`` split, or the
        manifest of registered few-shot runs).
    :param ckpt_sha: digest of the backbone weights, where the wrapper exposes it.
    """

    run_id: str
    config_sha: str
    backbone_id: str
    git_sha: str | None = None
    protocol_sha: str | None = None
    ckpt_sha: str | None = None

    #: Fields that must match for two segments to be the same training. ``git_sha`` is not
    #: one of them: a segment may resume after a documentation-only commit.
    BINDING = ("run_id", "config_sha", "backbone_id", "protocol_sha", "ckpt_sha")

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config_sha": self.config_sha,
            "backbone_id": self.backbone_id,
            "git_sha": self.git_sha,
            "protocol_sha": self.protocol_sha,
            "ckpt_sha": self.ckpt_sha,
        }

    @classmethod
    def from_dict(cls, dati: Mapping[str, Any]) -> CheckpointIdentity:
        """:return: the identity read back from a checkpoint.
        :raises ValueError: if required fields are missing.
        """
        missing = [k for k in ("run_id", "config_sha", "backbone_id") if not dati.get(k)]
        if missing:
            raise ValueError(
                f"checkpoint without identity: missing {missing}. It is not a checkpoint of "
                f"this project, or it was written by a version older than F6"
            )
        return cls(
            run_id=str(dati["run_id"]),
            config_sha=str(dati["config_sha"]),
            backbone_id=str(dati["backbone_id"]),
            git_sha=dati.get("git_sha"),
            protocol_sha=dati.get("protocol_sha"),
            ckpt_sha=dati.get("ckpt_sha"),
        )

    def verify(self, found: CheckpointIdentity) -> None:
        """Compare the expected identity with the one written in the checkpoint.

        :param trovata: identity read from the file.
        :raises ValueError: at the first mismatching binding field, naming it.
        """
        for field_name in self.BINDING:
            expected, read_value = getattr(self, field_name), getattr(found, field_name)
            if expected != read_value:
                raise ValueError(
                    f"checkpoint of another run: {field_name} is {read_value!r} in the file and "
                    f"{expected!r} in this run. Resuming here would produce a number, "
                    f"not an error: delete the checkpoint if restarting is intended"
                )
        if self.git_sha and found.git_sha and self.git_sha != found.git_sha:
            # Not binding (see BINDING), but worth logging: the previous segment ran with other code.
            log.warning(
                "resuming across a different commit: the checkpoint comes from %s, this run is %s",
                found.git_sha[:12],
                self.git_sha[:12],
            )


@dataclass(frozen=True)
class CheckpointSpec:
    """Where and how often to save: the policy, not the state.

    ``every`` is operational and outside budget parity (invariant 5), so it does not live
    in :class:`~evd.train.budget.BudgetSpec`.

    :param dir: folder; ``runs/{run_id}/checkpoint/`` for a run.
    :param every: save period in steps; ``0`` disables periodic saving, leaving on-demand
        and last-step saves.
    :raises ValueError: if ``every`` is negative.
    """

    dir: Path
    every: int = 0

    def __post_init__(self) -> None:
        if self.every < 0:
            raise ValueError(
                f"checkpoint.every must be >= 0 (0 = no periodic saving), not {self.every}"
            )
        object.__setattr__(self, "dir", Path(self.dir))

    @property
    def path(self) -> Path:
        return self.dir / CHECKPOINT_NAME


@dataclass
class TrainingState:
    """Minimal state making two segments a single training.

    Every field, if not restored, would make the resumed segment diverge from a full fit
    (checked by the homonymous test).

    :param step: last completed step; resumption restarts from ``step+1``.
    :param batches_consumed: batches already consumed from the train loader; otherwise the
        two segments would see the same examples twice.
    :param head: ``head.state_dict()``.
    :param adapter: ``state_dict_adapter()``, i.e. only the new weights.
    :param optimizer: AdamW state; the moments are part of the training.
    :param rng: snapshot from :func:`evd.utils.seeding.rng_state`.
    :param history: rows already written, so a resumed run has the history of a full one.
    :param best_val: best validation loss seen so far.
    :param best_step: step that produced it.
    :param senza_miglioramento: consecutive evaluations without improvement; resetting it
        would move the early stop.
    :param last_loss: last training loss, for a resumption running no step.
    :param selected_step: step that so far optimizes the model-selection criterion
        (``BudgetSpec.model_selection``, ADR-047 §6).
    :param selected_value: criterion value at that step.
    :param selected_head: head weights at that step.
    :param selected_adapter: adapter weights at that step.
    :param selected_scores: validation ``window_id -> logit`` at that step (ADR-053), so a
        resumption going straight to evaluation does not rerun validation.

    Without the selection fields a resumed run would keep the best of the last steps while
    claiming the overall best.
    """

    step: int
    batches_consumed: int
    head: dict[str, Tensor]
    optimizer: dict[str, Any]
    rng: dict[str, Any]
    adapter: dict[str, Tensor] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    best_val: float | None = None
    best_step: int | None = None
    senza_miglioramento: int = 0
    last_loss: float = float("nan")
    selected_step: int | None = None
    selected_value: float | None = None
    selected_head: dict[str, Tensor] = field(default_factory=dict)
    selected_adapter: dict[str, Tensor] = field(default_factory=dict)
    selected_scores: dict[str, float] = field(default_factory=dict)

    def payload(self, identity: CheckpointIdentity) -> dict[str, Any]:
        """:return: what goes into the file, identity included."""
        return {
            "identity": identity.as_dict(),
            "step": int(self.step),
            "batches_consumed": int(self.batches_consumed),
            "head": self.head,
            "adapter": self.adapter,
            "optimizer": self.optimizer,
            "rng": self.rng,
            "history": self.history,
            "best_val": self.best_val,
            "best_step": self.best_step,
            "senza_miglioramento": int(self.senza_miglioramento),
            "last_loss": float(self.last_loss),
            "selected_step": self.selected_step,
            "selected_value": self.selected_value,
            "selected_head": self.selected_head,
            "selected_adapter": self.selected_adapter,
            "selected_scores": self.selected_scores,
        }

    @classmethod
    def from_payload(cls, dati: Mapping[str, Any]) -> TrainingState:
        return cls(
            step=int(dati["step"]),
            batches_consumed=int(dati.get("batches_consumed", 0)),
            head=dict(dati["head"]),
            optimizer=dict(dati["optimizer"]),
            rng=dict(dati["rng"]),
            adapter=dict(dati.get("adapter") or {}),
            history=list(dati.get("history") or []),
            best_val=dati.get("best_val"),
            best_step=dati.get("best_step"),
            selected_step=dati.get("selected_step"),
            selected_value=dati.get("selected_value"),
            selected_head=dict(dati.get("selected_head") or {}),
            selected_adapter=dict(dati.get("selected_adapter") or {}),
            # `or {}`: checkpoints older than ADR-053 lack the field.
            selected_scores=dict(dati.get("selected_scores") or {}),
            senza_miglioramento=int(dati.get("senza_miglioramento", 0)),
            last_loss=float(dati.get("last_loss", float("nan"))),
        )


class InterruptFlag:
    """ "Save and stop at the next step boundary."

    An object rather than a global ``bool``: it can be raised by a signal handler, a test
    or a caller; :meth:`set` has the signature of a signal handler.
    """

    def __init__(self) -> None:
        self._raised = False
        self.reason: str | None = None

    def set(self, signum: int | None = None, frame: Any = None) -> None:
        """Raise the flag; accepts ``(signum, frame)`` to be a valid signal handler."""
        del frame
        self._raised = True
        self.reason = f"signal {signum}" if signum is not None else "explicit request"
        log.info("interruption requested (%s): saving at the next step boundary", self.reason)

    def clear(self) -> None:
        self._raised = False
        self.reason = None

    def is_set(self) -> bool:
        return self._raised

    def __bool__(self) -> bool:
        return self._raised


def install_interrupt_handler(
    flag: InterruptFlag, *, signals: tuple[str, ...] = SIGNALS
) -> tuple[str, ...]:
    """Connect ``flag`` to the signals that exist on this platform.

    Never raises: ``signal.SIGUSR1`` does not exist on Windows and ``signal.signal`` only
    works in the main thread.

    :param flag: flag to raise.
    :param signals: names to try, in ``signal``.
    :return: names actually installed; empty is a legitimate outcome.
    """
    installed: list[str] = []
    for name in signals:
        number = getattr(signal, name, None)
        if number is None:
            continue
        try:
            signal.signal(number, flag.set)
        except (ValueError, OSError, RuntimeError) as error:  # not the main thread
            log.debug("handler for %s not installed: %s", name, error)
            continue
        installed.append(name)
    if not installed:
        log.info(
            "no signal handler installed on %s: resumption is still driven by the flag, "
            "which is the actual mechanism",
            sys.platform,
        )
    return tuple(installed)


def restore_adapter_state(adapt: Any | None, stato: Mapping[str, Tensor]) -> None:
    """Restore the adapter weights from the saved state, in place.

    ``state_dict_adapter()`` returns detached views sharing memory, so writing into them
    writes into the parameters; this is verified, since a family returning copies would
    make resumption a silent no-op.

    :param adapt: the applied strategy, or ``None``.
    :param stato: saved tensors.
    :raises ValueError: if saved and current keys differ (another family or placement).
    :raises RuntimeError: if the parameters do not hold the saved values after copying.
    """
    read = getattr(adapt, "state_dict_adapter", None)
    current: dict[str, Tensor] = dict(read()) if callable(read) else {}
    if not current and not stato:
        return
    if set(current) != set(stato):
        only_file = sorted(set(stato) - set(current))
        only_here = sorted(set(current) - set(stato))
        raise ValueError(
            f"adapter state mismatch: {len(only_file)} keys only in the "
            f"checkpoint ({only_file[:3]}) and {len(only_here)} only here "
            f"({only_here[:3]}). The checkpoint comes from another family or placement"
        )
    with torch.no_grad():
        for name, tensor in current.items():
            tensor.copy_(stato[name].to(device=tensor.device, dtype=tensor.dtype))
    for name, tensor in dict(read()).items():
        expected = stato[name].to(device=tensor.device, dtype=tensor.dtype)
        if not torch.equal(tensor, expected):
            raise RuntimeError(
                f"resumption did not write into {name}: state_dict_adapter() of "
                f"{type(adapt).__name__} returns copies instead of parameter views, "
                f"so restoring would be a silent no-op. Return "
                f"`param.detach()` (which shares memory), not `param.clone()`"
            )


class Checkpointer:
    """Save and reload a training state, verifying who it belongs to.

    :param spec: where and how often.
    :param identity: run of what is written, and what is required from what is read.
    :param flag: interruption flag, if the caller installed one.
    """

    def __init__(
        self,
        spec: CheckpointSpec,
        identity: CheckpointIdentity,
        *,
        flag: InterruptFlag | None = None,
    ) -> None:
        self.spec = spec
        self.identity = identity
        self.flag = flag

    @property
    def path(self) -> Path:
        return self.spec.path

    def exists(self) -> bool:
        return self.path.exists()

    def due(self, step: int, *, last: bool = False) -> bool:
        """Return whether to save now.

        The last step is always saved, so a run dying during evaluation does not repeat
        training.

        :param step: step just completed.
        :param ultimo: true on the last budget step.
        """
        if self.flag is not None and self.flag.is_set():
            return True
        if last:
            return True
        return self.spec.every > 0 and step % self.spec.every == 0

    def should_stop(self) -> bool:
        """:return: whether someone requested to stop (signal or flag)."""
        return self.flag is not None and self.flag.is_set()

    def save(self, state: TrainingState) -> Path:
        """Write the state atomically (temporary sibling, then rename).

        :return: the checkpoint path.
        """
        ensure_dir(self.spec.dir)
        tmp = temp_sibling(self.path)
        try:
            with open(tmp, "wb") as fh:
                torch.save(state.payload(self.identity), fh)
                fh.flush()
            replace_atomic(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        log.info("checkpoint at step %d in %s", state.step, self.path)
        return self.path

    def load(self) -> TrainingState | None:
        """Reload the checkpoint, if any, verifying that it belongs to this run.

        Loaded with ``weights_only=True``; :func:`~evd.utils.seeding.rng_state` was designed to
        avoid arbitrary pickle.

        :return: the state, or ``None`` if there is nothing to resume.
        :raises ValueError: if the checkpoint belongs to another run.
        """
        if not self.exists():
            return None
        data = torch.load(self.path, map_location="cpu", weights_only=True)
        self.identity.verify(CheckpointIdentity.from_dict(data.get("identity") or {}))
        state = TrainingState.from_payload(data)
        log.info("resuming from step %d (checkpoint %s)", state.step, self.path)
        return state
