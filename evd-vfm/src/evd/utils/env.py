"""Python side of the project environment variables.

Defaults live in ``hpc/env.sh`` and nowhere else: this module reads them instead of
redefining them (checked by ``tests/unit/test_env.py``).

===========================  ======================================================
``hpc/env.sh``               source of the defaults, sourced by every SLURM job
``hpc/tools/job_env.py``     stdlib-only diagnostics for the login node
``evd/utils/env.py``         what the thesis code imports (this module)
===========================  ======================================================

Static defaults (``: "${NAME:=value}"``) are read from ``env.sh``; values that depend
on the allocation (device, cores, shard) are deduced where the code runs, not where the
job is submitted, and the deducing functions say so.

Typical use::

    from evd.utils import env
    runs = env.runs_dir()          # $EVD_RUNS_DIR, or the env.sh default
    dev  = env.device()            # deduced here, not inherited

Diagnostics: ``python -m evd.utils.env``.
"""

from __future__ import annotations

import getpass
import os
import re
from dataclasses import dataclass
from pathlib import Path

from evd.utils.logging import get_logger

log = get_logger(__name__)

# Line of the form:  : "${NAME:=value}"
_DEFAULT_RE = re.compile(r'^\s*:\s*"\$\{([A-Za-z_][A-Za-z0-9_]*):=(.*)\}"\s*$')
# Variable references inside a value: ${NAME} or $NAME
_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")

#: Names consumed by the Python code; the test checks that each still exists in
#: ``hpc/env.sh``.
CONSUMED_KEYS = (
    "EVD_VJEPA_DIR",
    "EVD_SHARED_ROOT",
    "EVD_STORE_ROOT",
    "EVD_DATA_DIR",
    "EVD_RAW_DIR",
    "EVD_CKPT_DIR",
    "EVD_UCF_DIR",
    "EVD_MIVIA_DIR",
    "EVD_ANNOT_DIR",
    "EVD_FEATURES_DIR",
    "EVD_RUNS_DIR",
    "EVD_CACHE_DIR",
    "EVD_PRECISION",
    "EVD_ALLOW_BF16",
    "EVD_TF32",
    "EVD_BATCH_SIZE",
    "EVD_SEED",
)


class MissingEnvVar(KeyError):
    """Requested variable missing from the environment, from ``env.sh`` and from the call."""


class UnresolvableDefault(ValueError):
    """The ``env.sh`` default is computed by the shell (``$(...)``) and cannot be evaluated here."""


# --------------------------------------------------------------------------- #
# Locating and reading hpc/env.sh
# --------------------------------------------------------------------------- #


def find_env_sh() -> Path | None:
    """Return ``hpc/env.sh`` if reachable, ``None`` outside the workspace."""
    hpc_dir = os.environ.get("EVD_HPC_DIR")
    if hpc_dir:
        cand = Path(hpc_dir) / "env.sh"
        if cand.is_file():
            return cand
    root = os.environ.get("EVD_ROOT")
    if root:
        cand = Path(root) / "hpc" / "env.sh"
        if cand.is_file():
            return cand
    # ...\codice\evd-vfm\src\evd\utils\env.py: walk up until hpc/env.sh is found
    for parent in Path(__file__).resolve().parents:
        cand = parent / "hpc" / "env.sh"
        if cand.is_file():
            return cand
    return None


def parse_shell_defaults(path: str | Path | None = None) -> dict[str, str]:
    """Return the static defaults of ``env.sh``, unexpanded.

    Only the ``: "${NAME:=value}"`` form is read; deduced values and conditional branches
    are deduced in Python.
    """
    p = Path(path) if path is not None else find_env_sh()
    if p is None or not Path(p).is_file():
        return {}
    out: dict[str, str] = {}
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        m = _DEFAULT_RE.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


_cache: dict[str, str] | None = None


def shell_defaults() -> dict[str, str]:
    """Return :func:`parse_shell_defaults`, read once per process."""
    global _cache
    if _cache is None:
        _cache = parse_shell_defaults()
    return _cache


def reload_shell_defaults() -> None:
    """Clear the cache (used by tests that move ``EVD_HPC_DIR``)."""
    global _cache
    _cache = None


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


def _root_from_file(name: str) -> str | None:
    """Deduce ``EVD_ROOT`` and ``EVD_HPC_DIR`` from the location of ``hpc/env.sh``.

    Their ``env.sh`` default is ``$(cd ... && pwd)``; without this deduction every default
    referencing them (``EVD_RUNS_DIR``, ``EVD_LOG_DIR``) would silently be truncated.
    Returns ``None`` for any other name.
    """
    if name not in {"EVD_ROOT", "EVD_HPC_DIR"}:
        return None
    p = find_env_sh()
    if p is None:
        return None
    hpc = Path(p).resolve().parent
    return str(hpc if name == "EVD_HPC_DIR" else hpc.parent)


def _expand(value: str, _seen: frozenset[str] = frozenset()) -> str:
    if "$(" in value:
        raise UnresolvableDefault(f"default computed by the shell: {value!r}")

    def repl(m: re.Match[str]) -> str:
        name = m.group(1) or m.group(2)
        if name in _seen:  # circular reference: report it instead of looping
            raise UnresolvableDefault(f"circular reference on ${name}")
        if name == "HOME" and "HOME" not in os.environ:
            return str(Path.home())  # HOME does not exist on Windows, USERPROFILE does
        if name == "USER" and "USER" not in os.environ:
            return os.environ.get("USERNAME", "")
        return get(name, "", _seen=_seen | {name})

    return _REF_RE.sub(repl, value)


def get(name: str, default: str | None = None, *, _seen: frozenset[str] = frozenset()) -> str:
    """Resolve a variable: environment, then the expanded ``env.sh`` default, then ``default``.

    The environment always wins: it is the documented override mechanism
    (``EVD_BATCH_SIZE=8 hpc/invia.sh ...``).

    :raises MissingEnvVar: if no source provides a value.
    """
    if name in os.environ:
        return os.environ[name]
    inferred = _root_from_file(name)
    if inferred is not None:
        return inferred
    raw = shell_defaults().get(name)
    if raw is not None:
        return _expand(raw, _seen)
    if default is not None:
        return default
    raise MissingEnvVar(
        f"{name} is not in the environment and has no default in hpc/env.sh. "
        f"On the cluster: `source hpc/env.sh`. If it is new, add the default "
        f"to hpc/env.sh, not here."
    )


def get_path(name: str, default: str | None = None) -> Path:
    return Path(get(name, default))


def get_int(name: str, default: int | None = None) -> int:
    return int(get(name, None if default is None else str(default)))


def get_bool(name: str, default: bool | None = None) -> bool:
    """Parse ``1/true/yes/on`` as True (project shell variables use ``0``/``1``)."""
    raw = get(name, None if default is None else ("1" if default else "0"))
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- #
# Typed accessors (defaults read from env.sh)
# --------------------------------------------------------------------------- #


def store_root() -> Path:
    """Return the project storage root: ``$EVD_SHARED_ROOT/$USER`` on the cluster,
    ``$HOME/evd-store`` elsewhere.

    Mirrors the ``if`` branch of ``env.sh``, so a Python process started without sourcing
    ``env.sh`` does not fall back to the small home filesystem.
    """
    if "EVD_STORE_ROOT" in os.environ:
        return Path(os.environ["EVD_STORE_ROOT"])
    shared = Path(get("EVD_SHARED_ROOT", "/shared/tesi_mllm"))
    if shared.is_dir():
        return shared / getpass.getuser()
    return get_path("EVD_STORE_ROOT")


def data_dir() -> Path:
    return get_path("EVD_DATA_DIR")


def raw_dir() -> Path:
    return get_path("EVD_RAW_DIR")


def ckpt_dir() -> Path:
    return get_path("EVD_CKPT_DIR")


def ucf_dir() -> Path:
    return get_path("EVD_UCF_DIR")


def annot_dir() -> Path:
    return get_path("EVD_ANNOT_DIR")


def features_dir() -> Path:
    return get_path("EVD_FEATURES_DIR")


def runs_dir() -> Path:
    """Return the folder holding ``runs/{run_id}/`` and ``registry.jsonl`` (ADR-009/ADR-020).

    One folder per machine: under the store on the cluster (``/shared``), ``evd-vfm/runs/``
    elsewhere, mirroring ``hpc/env.sh`` §1.
    """
    if "EVD_RUNS_DIR" in os.environ:
        return Path(os.environ["EVD_RUNS_DIR"])
    if Path(get("EVD_SHARED_ROOT", "/shared/tesi_mllm")).is_dir():
        return store_root() / "runs"
    return get_path("EVD_RUNS_DIR")


def cache_dir() -> Path:
    return get_path("EVD_CACHE_DIR")


def precision() -> str:
    """Return the precision, ``fp32`` by default.

    ``bf16`` overflows to NaN with eager attention on V-JEPA 2.1 ViT-B/L/g; the combination
    is rejected by config validation (:mod:`evd.train.budget`).
    """
    return get("EVD_PRECISION")


def allow_bf16() -> bool:
    return get_bool("EVD_ALLOW_BF16")


def tf32() -> bool:
    """Return whether TF32 is enabled (off by default: reproducibility first).

    It must be recorded in ``meta.json``, otherwise timings are not comparable
    (docs/08 §4.3).
    """
    return get_bool("EVD_TF32")


def batch_size() -> int:
    return get_int("EVD_BATCH_SIZE")


def seed() -> int:
    return get_int("EVD_SEED")


# --------------------------------------------------------------------------- #
# Deduced values (machine-dependent, deduced here)
# --------------------------------------------------------------------------- #


def repo_root() -> Path:
    """Return the ``evd-vfm`` root, three levels above ``src/evd/utils``.

    ``EVD_ROOT`` is the workspace root instead.
    """
    return Path(__file__).resolve().parents[3]


def workspace_root() -> Path:
    """Return the workspace root (``TESI/codice``), holding ``evd-vfm``, ``hpc``, ``lab`` and
    ``vjepa2``.
    """
    return Path(os.environ["EVD_ROOT"]) if "EVD_ROOT" in os.environ else repo_root().parent


def vjepa_dir() -> Path:
    """Return the Meta source clone used by the V-JEPA 2.1 wrappers.

    The default lives in ``hpc/env.sh``; existence is checked instead of returning a
    plausible path (ADR-018).

    :raises FileNotFoundError: if the folder does not exist.
    """
    path = get_path("EVD_VJEPA_DIR")
    if not path.is_dir():
        raise FileNotFoundError(
            f"EVD_VJEPA_DIR={path} does not exist or is not a folder. "
            "Set the variable to a clean clone of facebookresearch/vjepa2."
        )
    return path


def cpus() -> int:
    """Return the number of CPUs (deduced): ``EVD_CPUS``, else the SLURM allocation, else 1."""
    for key in ("EVD_CPUS", "SLURM_CPUS_PER_TASK"):
        if os.environ.get(key):
            return int(os.environ[key])
    return 1


def num_workers() -> int:
    """Return the number of DataLoader workers (deduced), at most ``--cpus-per-task``."""
    if os.environ.get("EVD_NUM_WORKERS"):
        return int(os.environ["EVD_NUM_WORKERS"])
    return cpus()


def device(prefer: str | None = None, *, require_cuda: bool = False) -> str:
    """Return the compute device (deduced here and verified).

    ``TORCH_DEVICE`` from the environment if set, otherwise ``cuda`` if torch sees it,
    otherwise ``cpu``; ``cuda`` is returned only if torch actually sees a GPU (SLURM may
    export GPU variables on nodes without one).

    ``require_cuda=True`` raises instead of falling back to CPU, for runs that must have a
    GPU. The default is ``False``: no code path may require CUDA.

    :raises RuntimeError: if ``require_cuda`` is set and no GPU is available.
    """
    requested = prefer or os.environ.get("TORCH_DEVICE") or ""
    try:
        import torch

        available = torch.cuda.is_available()
    except ImportError:  # pragma: no cover - torch is a dependency
        available = False

    if not requested:
        return "cuda" if available else "cpu"
    if requested.startswith("cuda") and not available:
        reason = (
            f"TORCH_DEVICE={requested} requested but torch.cuda.is_available() is False "
            f"(node without GPU, or CUDA_VISIBLE_DEVICES=-1)"
        )
        if require_cuda:
            raise RuntimeError(
                f"{reason}. This run needs a GPU: check the partition "
                f"and `--gpus-per-task` (hpc/README.md §4)."
            )
        log.warning(
            "%s -> falling back to cpu. The actual device is recorded in meta.json.", reason
        )
        return "cpu"
    return requested


def shard() -> tuple[int, int]:
    """Return ``(index, total)`` for array jobs (deduced)."""
    idx = int(os.environ.get("EVD_SHARD", os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    tot = int(os.environ.get("EVD_NUM_SHARDS", os.environ.get("SLURM_ARRAY_TASK_COUNT", "1")))
    return idx, tot


# --------------------------------------------------------------------------- #
# Diagnostics / environment fingerprint for meta.json
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EnvReport:
    env_sh: str | None
    values: dict[str, str]
    deduced: dict[str, str]
    sources: dict[str, str]  # name -> "environment" | "hpc/env.sh" | "missing"


def report() -> EnvReport:
    """Return the resolved table with the provenance of every value.

    Recorded in ``meta.json`` (environment fingerprint, docs/08 §4.3).
    """
    values: dict[str, str] = {}
    sources: dict[str, str] = {}
    defaults = shell_defaults()
    for key in CONSUMED_KEYS:
        if key in os.environ:
            values[key] = os.environ[key]
            sources[key] = "environment"
        elif key in defaults:
            try:
                values[key] = _expand(defaults[key])
                sources[key] = "hpc/env.sh"
            except UnresolvableDefault as e:
                values[key] = f"<{e}>"
                sources[key] = "hpc/env.sh (unresolvable)"
        else:
            values[key] = ""
            sources[key] = "missing"
    idx, tot = shard()
    deduced = {
        "TORCH_DEVICE": device(),
        "EVD_CPUS": str(cpus()),
        "EVD_NUM_WORKERS": str(num_workers()),
        "EVD_SHARD": f"{idx}/{tot}",
        "repo_root": str(repo_root()),
    }
    env_sh = find_env_sh()
    return EnvReport(str(env_sh) if env_sh else None, values, deduced, sources)


def _main() -> int:
    r = report()
    print(f"hpc/env.sh: {r.env_sh or 'NOT FOUND (only environment and explicit defaults apply)'}")
    print("--- read -------------------------------------------------------")
    for k, v in r.values.items():
        print(f"  {k:<20} {v}    [{r.sources[k]}]")
    print("--- dedotti ----------------------------------------------------")
    for k, v in r.deduced.items():
        print(f"  {k:<20} {v}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
