"""Uniform logging, with the format decided in one place.

On the cluster job output goes to a file; ``hpc/env.sh`` exports
``PYTHONUNBUFFERED=1`` so that ``tail -f`` shows it live.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"
_configured = False


def setup_logging(level: int = logging.INFO, *, logfile: str | Path | None = None) -> None:
    """Configure the root logger once.

    Idempotent: calling it twice does not duplicate handlers, and if the root logger is
    already configured (e.g. by Hydra) no extra handler is added.
    """
    global _configured
    root = logging.getLogger()
    if _configured or (root.handlers and logfile is None):
        root.setLevel(level)
        _configured = True
        return
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if logfile is not None:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile, encoding="utf-8"))
    fmt = logging.Formatter(_FORMAT, datefmt=_DATEFMT)
    for h in handlers:
        h.setFormatter(fmt)
        root.addHandler(h)
    root.setLevel(level)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
