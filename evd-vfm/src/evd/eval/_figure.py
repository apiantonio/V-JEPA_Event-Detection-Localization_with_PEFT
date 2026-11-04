"""Shared figure helpers: headless backend and saving.

Kept private and separate from :mod:`evd.eval.report` so that :mod:`evd.eval.plots`
can use them without importing scikit-learn. Figure text is in English.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def agg() -> Any:
    """Import matplotlib with the non-interactive ``Agg`` backend.

    The backend is set before importing ``pyplot``: compute nodes have no display.

    :return: the configured ``matplotlib.pyplot`` module.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save(
    fig: Any, out_path: str | Path, *, rect: tuple[float, float, float, float] | None = None
) -> Path:
    """Write the figure, creating the folder if needed, and close it.

    Closing releases the figure from the pyplot registry, so loops drawing many
    figures do not accumulate memory.

    :param fig: matplotlib figure.
    :param out_path: destination path; parents are created.
    :param rect: ``(x0, y0, x1, y1)`` box in figure coordinates available to
        ``tight_layout``; needed when the figure has ``fig.text`` labels, which
        ``tight_layout`` ignores.
    :return: the written path.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=rect) if rect is not None else fig.tight_layout()
    fig.savefig(p)
    agg().close(fig)
    return p
