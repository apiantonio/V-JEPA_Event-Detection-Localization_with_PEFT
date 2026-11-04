"""Resumable artifacts per window and method, with identity and checksum.

The JSON receipt is the commit of the artifact and is published after the NPZ. Results
never enter the detector feature store.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from evd.utils.hashing import file_sha256
from evd.utils.io import replace_atomic, temp_sibling, write_text_atomic


def identity_sha(value: dict) -> str:
    """:return: SHA256 of the canonical JSON, without NaN or key-order dependence."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def atomic_json(path: Path, value: dict | list) -> None:
    """:param value: finite structure; written with the shared atomic helper."""
    write_text_atomic(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """:param arrays: arrays without pickle; a temporary file is never a complete result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temp_sibling(path)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        replace_atomic(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ArtifactStore:
    """Two files per result, accepted together after verification.

    :param root: dedicated folder, never the checkpoint or a training run.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def load(self, key: str, identity: dict) -> tuple[dict[str, np.ndarray], dict] | None:
        """Read only a committed result with a matching identity.

        :return: arrays and receipt, or None after a pre-commit interruption.
        :raises ValueError: on a different identity, corruption or non-finite data.
        """
        metadata = self.root / f"{key}.json"
        payload = self.root / f"{key}.npz"
        if not metadata.exists():
            return None
        record = json.loads(metadata.read_text(encoding="utf-8"))
        if record.get("identity_sha256") != identity_sha(identity):
            raise ValueError(f"Different identity: {metadata}; choose a new result group")
        if not payload.exists() or file_sha256(payload) != record["payload_sha256"]:
            raise ValueError(
                f"Corrupted or missing artifact: {payload}; restore the pair from a valid copy"
            )
        with np.load(payload, allow_pickle=False) as stored:
            arrays = dict(stored)
        if any(not np.isfinite(array).all() for array in arrays.values()):
            raise ValueError(f"Non-finite array: {payload}")
        return arrays, record["receipt"]

    def save(self, key: str, identity: dict, arrays: dict[str, np.ndarray], receipt: dict) -> None:
        """Publish the arrays before the receipt that makes them reusable.

        :raises ValueError: on non-finite data; no receipt is published.
        """
        if any(not np.isfinite(array).all() for array in arrays.values()):
            raise ValueError(f"Non-finite array: {key}")
        payload = self.root / f"{key}.npz"
        atomic_npz(payload, arrays)
        atomic_json(
            self.root / f"{key}.json",
            {
                "identity": identity,
                "identity_sha256": identity_sha(identity),
                "payload_sha256": file_sha256(payload),
                "receipt": receipt,
            },
        )
