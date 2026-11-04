"""Determinism, item 1 of the reproducibility checklist (docs/08 §4).

On CPU the forward of the same model on the same inputs is not bitwise reproducible
over the process lifetime, so equivalence tests use a measured tolerance and
:func:`seed_all` only guarantees that the same sequence of random choices repeats.
"""

from __future__ import annotations

import os
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from evd.utils.logging import get_logger

log = get_logger(__name__)


def seed_all(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and torch (CPU and CUDA) at once.

    Covers the generators listed by the laboratory protocol §5 plus the two cuDNN switches.
    ``CUBLAS_WORKSPACE_CONFIG`` must be set before the CUDA context exists, so it lives in
    ``hpc/env.sh`` and ``scripts/smoke.ps1`` instead.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # warn_only=True: some ops have no deterministic kernel and must not stop legitimate runs.
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Not implied by `use_deterministic_algorithms` and required by the laboratory protocol
        # (§5, ADR-047): `benchmark=True` would pick kernels depending on the GPU load.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker, so workers do not repeat the same NumPy stream."""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def torch_generator(seed: int) -> torch.Generator:
    """Return an explicit generator for DataLoader shuffling (docs/08 §4.2)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def rng_state() -> dict[str, Any]:
    """Snapshot the pseudo-random generators in a serializable form.

    Used to resume training (:mod:`evd.train.checkpoint`). The state is made of primitives
    and tensors only, never NumPy arrays, so checkpoints load with
    ``torch.load(weights_only=True)``.

    :return: an opaque state for :func:`set_rng_state`.
    """
    version, keys, position, has_gauss, gauss = np.random.get_state()
    py_version, py_internals, py_gauss = random.getstate()
    state: dict[str, Any] = {
        "python": [int(py_version), [int(v) for v in py_internals], py_gauss],
        "numpy": [
            str(version),
            [int(v) for v in keys],
            int(position),
            int(has_gauss),
            float(gauss),
        ],
        "torch": torch.get_rng_state(),
        "torch_cuda": [],
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = list(torch.cuda.get_rng_state_all())
    return state


def set_rng_state(stato: Mapping[str, Any]) -> None:
    """Restore the generators from a :func:`rng_state` snapshot.

    The CUDA state is restored only if this machine has the same number of devices as the
    one that wrote it; resuming on a different node is legitimate and only logged.

    :param stato: dictionary produced by :func:`rng_state`.
    :raises KeyError: if the expected generators are missing.
    """
    py_version, py_internals, py_gauss = stato["python"]
    random.setstate((int(py_version), tuple(int(v) for v in py_internals), py_gauss))
    version, keys, position, has_gauss, gauss = stato["numpy"]
    np.random.set_state(
        (
            str(version),
            np.array(keys, dtype=np.uint32),
            int(position),
            int(has_gauss),
            float(gauss),
        )
    )
    torch.set_rng_state(stato["torch"].to(torch.uint8).cpu())
    cuda = list(stato.get("torch_cuda") or [])
    if not cuda:
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != len(cuda):
        log.warning(
            "RNG state of %d CUDA devices not restored: %d available here. "
            "Resuming anyway, but GPU random draws do not continue the "
            "previous sequence",
            len(cuda),
            torch.cuda.device_count() if torch.cuda.is_available() else 0,
        )
        return
    torch.cuda.set_rng_state_all([t.to(torch.uint8).cpu() for t in cuda])
