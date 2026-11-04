"""Stable hashes: cache keys are computed, never written by hand.

Stable means independent of dict key order, independent of ``PYTHONHASHSEED`` (never
the builtin ``hash()``), and identical across machines and over time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(obj: Any) -> str:
    """Return the canonical representation: sorted keys, no spaces, escaped ASCII.

    Floats are serialized with ``repr``, so ``224`` and ``224.0`` give different digests
    on purpose.
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=_fallback
    )


def _fallback(o: Any) -> Any:
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Path):
        return o.as_posix()
    if hasattr(o, "__dict__"):
        return vars(o)
    return str(o)


def stable_hash(obj: Any) -> str:
    """Return the hexadecimal sha256 (64 characters) of the canonical form of ``obj``."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def short(digest: str, n: int = 12) -> str:
    """Return a readable digest prefix for file names and logs, never for cache keys."""
    return digest[:n]


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """Return the sha256 of a file read in blocks (checkpoints weigh tens of GB)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()
