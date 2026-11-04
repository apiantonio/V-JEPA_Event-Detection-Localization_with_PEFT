r"""Local tracker: append-only JSONL, the source of truth (ADR-009).

Everything lives in ``runs/{run_id}/`` (gitignored, ``registry.jsonl`` included,
ADR-020), so durability is a transfer problem: an experiment block is closed only once
``.\hpc\sync.ps1 -Azione pull-run`` has brought registry and artifacts home.

Guarantees:

1. Append-only: a written line is never rewritten, so merging two registry copies is a
   set union (dedup by ``run_id``).
2. An official run rejects a dirty working tree: a number without a commit is not
   reproducible.
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evd.utils import env
from evd.utils.io import append_jsonl, ensure_dir, write_json, write_text_atomic
from evd.utils.logging import get_logger

log = get_logger(__name__)

REGISTRY_NAME = "registry.jsonl"


def plain_config(value: Any) -> Any:
    """Strip Python objects from configs, so run YAML files are portable.

    Public because the checkpoint identity hashes the resolved config (ADR-025 point 4):
    two representations of the same config would give two hashes.
    """
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): plain_config(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [plain_config(item) for item in value]
    return value


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_info(repo: Path | None = None) -> dict[str, Any]:
    """Return SHA, branch and dirty flag of the repository; ``None``/``True`` if git does not
    answer (missing information never looks like a clean tree).
    """
    root = repo or env.repo_root()
    sha = _git(["rev-parse", "HEAD"], root)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    status = _git(["status", "--porcelain"], root)
    return {
        "sha": sha,
        "branch": branch,
        "dirty": True if status is None else bool(status),
        "available": sha is not None,
    }


def no_git_allowed(info: Mapping[str, Any]) -> bool:
    """Return whether an official run may start although git does not answer.

    A copy of the code delivered without its repository would otherwise reject every official
    run. ``EVD_ALLOW_NO_GIT=1`` lifts the check only when git does not answer at all: a real
    repository with local changes stays rejected, and ``meta.json`` records the missing commit
    (``git.available: false``).

    :param info: the output of :func:`git_info`.
    :return: ``True`` if git does not answer and the environment allows it.
    """
    return not info.get("available") and os.environ.get("EVD_ALLOW_NO_GIT") == "1"


def env_fingerprint() -> dict[str, Any]:
    """Return the environment fingerprint for ``meta.json`` (docs/08 §4.3).

    The full ``pip freeze`` is not recomputed at every run: the cluster reference snapshot
    is versioned (``hpc/setup/requisiti-hpc.txt``) and only machine-specific values are
    recorded.
    """
    import torch

    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "requirements_ref": "hpc/setup/requisiti-hpc.txt",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


@dataclass
class RunHandle:
    run_id: str
    dir: Path
    official: bool
    started_at: str
    meta: dict[str, Any] = field(default_factory=dict)


def _env_vars(ignored: Sequence[str] = ()) -> dict[str, Any]:
    """Return the environment fingerprint, separating what governed the run from what did not.

    :param ignored: variables not consumed by the calling path.
    :return: ``resolved`` and ``deduced``, plus ``non_consumate`` when the caller declared
        some.
    """
    report = env.report()
    outside = set(ignored)
    resolved = {k: v for k, v in report.values.items() if k not in outside}
    out: dict[str, Any] = {"resolved": resolved, "deduced": report.deduced}
    if outside:
        out["non_consumate"] = {
            "nota": (
                "variables resolved from hpc/env.sh that this run does NOT read: its "
                "budget comes from the Hydra config. The actual seed and the fields that "
                "governed the run are in `seed` and `esecuzione` (ADR-057)."
            ),
            "valori": {k: v for k, v in report.values.items() if k in outside},
        }
    return out


class LocalTracker:
    """Write ``runs/{run_id}/`` and append to ``runs/registry.jsonl``."""

    def __init__(self, runs_dir: str | Path | None = None) -> None:
        self.runs_dir = Path(runs_dir) if runs_dir is not None else env.runs_dir()

    @property
    def registry_path(self) -> Path:
        return self.runs_dir / REGISTRY_NAME

    def start(
        self,
        run_id: str,
        config: Any,
        *,
        official: bool,
        regime: dict[str, Any] | None = None,
        protocol_sha: str | None = None,
        extra_meta: dict[str, Any] | None = None,
        ignored_env: Sequence[str] = (),
        dentro: str | None = None,
    ) -> RunHandle:
        """Open a run and write ``meta.json`` and ``config.yaml``.

        :param dentro: folder name of another run to nest this one into (cross-dataset
            evaluations, ADR-066); the ``run_id`` does not change, since
            :func:`evd.exp.demo.open_run` takes it from the folder name.
        :param ignored_env: environment variables not consumed by this path; they are recorded
            under ``env_vars.non_consumate`` instead of among the resolved values (ADR-057).
        :raises ValueError: if ``dentro`` is not a plain folder name.
        :raises RuntimeError: if the run is official and the working tree is dirty, or git does
            not answer and :func:`no_git_allowed` is false.
        """
        gi = git_info()
        if official and gi["dirty"]:
            if not no_git_allowed(gi):
                raise RuntimeError(
                    "official run rejected: the working tree is dirty (or git does not answer). "
                    "A number that ends up in the thesis must be traceable to a commit "
                    "(invariant 2, docs/08 §4.3). Commit, then relaunch; in a copy of the code "
                    "without git, set EVD_ALLOW_NO_GIT=1."
                )
            log.warning("run %s starts without git: no commit traces it (EVD_ALLOW_NO_GIT)", run_id)
        if dentro is not None and (Path(dentro).name != dentro or dentro in {"", ".", ".."}):
            raise ValueError(f"`dentro` must be a run folder name, not a path: {dentro!r}")
        root = self.runs_dir if dentro is None else self.runs_dir / dentro
        run_dir = ensure_dir(root / run_id)
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        meta: dict[str, Any] = {
            "run_id": run_id,
            "official": official,
            "started_at": started,
            "git": gi,
            "env": env_fingerprint(),
            "env_vars": _env_vars(ignored_env),
            "regime": regime or {},
            "protocol_sha": protocol_sha,
        }
        # `extra_meta` is written to `meta.json` (it used to be accepted and ignored).
        meta.update(extra_meta or {})
        write_json(run_dir / "meta.json", meta)
        self._write_config(run_dir, config)
        return RunHandle(
            run_id=run_id, dir=run_dir, official=official, started_at=started, meta=meta
        )

    @staticmethod
    def _write_config(run_dir: Path, config: Any) -> None:
        """Write the resolved config, the source of truth of the run (ADR-005)."""
        try:
            from omegaconf import DictConfig, OmegaConf

            if isinstance(config, DictConfig):
                write_text_atomic(run_dir / "config.yaml", OmegaConf.to_yaml(config, resolve=True))
                return
            write_text_atomic(
                run_dir / "config.yaml",
                OmegaConf.to_yaml(OmegaConf.create(plain_config(config))),
            )
        except ImportError:  # pragma: no cover - omegaconf ships with hydra
            write_json(run_dir / "config.json", config)

    def log_history(self, handle: RunHandle, records: list[dict[str, Any]]) -> None:
        for r in records:
            append_jsonl(handle.dir / "history.jsonl", r)

    def finish(
        self,
        handle: RunHandle,
        metrics: dict[str, Any],
        *,
        status: str = "ok",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        write_json(handle.dir / "metrics.json", {"run_id": handle.run_id, **metrics})
        row: dict[str, Any] = {
            "run_id": handle.run_id,
            "status": status,
            "official": handle.official,
            "started_at": handle.started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_sha": handle.meta.get("git", {}).get("sha"),
            "git_dirty": handle.meta.get("git", {}).get("dirty"),
            "protocol_sha": handle.meta.get("protocol_sha"),
            "regime": handle.meta.get("regime", {}),
            "host": handle.meta.get("env", {}).get("host"),
            "metrics": metrics,
        }
        if extra:
            row.update(extra)
        append_jsonl(self.registry_path, row)
        log.info("run %s registrato in %s", handle.run_id, self.registry_path)
        return row
