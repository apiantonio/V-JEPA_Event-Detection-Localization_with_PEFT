"""Atomic writes and JSON helpers.

A job killed at wall time in the middle of ``json.dump`` would leave a truncated file
that still exists. Files are written to a temporary sibling and renamed with
``os.replace``, which is atomic on the same filesystem, Windows included.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from evd.utils.logging import get_logger

log = get_logger(__name__)


def temp_sibling(p: Path) -> Path:
    """Return a temporary file name unique per writer.

    A fixed ``.tmp`` name is not atomic across processes: two jobs writing the same path
    would share it and the second ``os.replace`` would fail.
    """
    return p.with_name(f"{p.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")


def replace_atomic(tmp: Path, destinazione: Path, *, attempts: int = 20) -> None:
    """Run ``os.replace`` with a short retry, needed only on Windows.

    On Windows ``MoveFileEx`` may transiently return *access denied* while another writer
    replaces the same target. The number of attempts is bounded, so a persistent
    permission error reaches the caller.
    """
    for remaining in range(attempts - 1, -1, -1):
        try:
            os.replace(tmp, destinazione)
            return
        except PermissionError:
            if remaining == 0:
                raise
            time.sleep(0.01)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_text_atomic(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    r"""Write ``text`` to ``path`` atomically, always with ``\n`` newlines."""
    p = Path(path)
    ensure_dir(p.parent)
    tmp = temp_sibling(p)
    try:
        with open(tmp, "w", encoding=encoding, newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        replace_atomic(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return p


def write_json(path: str | Path, obj: Any, *, indent: int = 2) -> Path:
    return write_text_atomic(path, json.dumps(obj, indent=indent, ensure_ascii=False) + "\n")


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def append_jsonl(path: str | Path, record: dict[str, Any]) -> Path:
    """Append one line to a JSONL file.

    Append-only by contract (ADR-009): a written line is never rewritten, so merging two
    copies is a union of lines.
    """
    p = Path(path)
    ensure_dir(p.parent)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    with open(p, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(line + "\n")
    return p


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    """Write a whole JSONL file at once, atomically.

    Unlike :func:`append_jsonl` it replaces the file: right for artifacts regenerated on
    every run, wrong for the registry.

    :param path: destination; missing folders are created.
    :param records: rows, consumed once.
    :return: the written path.
    """
    body = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)
    return write_text_atomic(path, body)


def read_jsonl(path: str | Path, *, skip_invalid: bool = False) -> list[dict[str, Any]]:
    """Read a JSONL file.

    A broken line is an error by default (e.g. a corrupted progress file would resume a
    run from the wrong point); readers of long-lived logs such as the registry pass
    ``skip_invalid=True``.

    :param path: the file; a missing file gives an empty list.
    :param skip_invalid: count and skip undecodable lines instead of raising.
    :return: the decoded objects, in file order.
    :raises ValueError: with ``skip_invalid=False``, at the first invalid line, naming
        file and line number.
    """
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    skipped = 0
    with open(p, encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                if not skip_invalid:
                    raise ValueError(f"{p}:{i} is not valid JSON: {e}") from e
                skipped += 1
    if skipped:
        log.warning("%s: %d undecodable lines skipped", p, skipped)
    return out
