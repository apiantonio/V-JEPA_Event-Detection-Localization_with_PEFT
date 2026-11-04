"""Figures generated from run artifacts, never drawn by hand.

:func:`timeline_png` is shared by the smoke test, the qualitative analysis and the
demonstrator; :func:`training_png` and :func:`lr_png` draw the history of one run from
the rows of ``history.jsonl``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from evd.annotations.schema import TemporalAnnotation
from evd.eval._figure import agg, save
from evd.types import FrameScores


def timeline_png(
    scores: FrameScores,
    out_path: str | Path,
    *,
    gt: TemporalAnnotation | None = None,
    threshold: float | None = None,
    fps: float | None = None,
    title: str | None = None,
    predicted: Sequence[tuple[int, int]] | None = None,
) -> Path:
    """Plot the score over time, with ground-truth intervals shaded and the threshold if given.

    ``predicted`` adds a comparison strip (decision above, annotation below), the same two
    lanes drawn by ``apps/demo``.

    :param scores: frame projection with the coverage mask.
    :param out_path: PNG destination; parent folders are created.
    :param gt: annotation used to shade the true events.
    :param threshold: operating point, drawn as a horizontal line.
    :param fps: if given, the x axis is in seconds instead of frames.
    :param title: figure title; ``None`` uses the ``video_id``.
    :param predicted: detected ``(start, end)`` frame intervals, both ends included, as
        returned by :func:`~evd.infer.engine.intervals_from_scores`.
    :return: the written path.
    """
    plt = agg()

    n = len(scores.scores)
    x = np.arange(n) / fps if fps else np.arange(n)
    with_strips = predicted is not None or gt is not None
    if with_strips:
        # No `hspace` in the gridspec: `save` calls `tight_layout`, which warns otherwise.
        fig, (ax, ax_strips) = plt.subplots(
            2, 1, figsize=(9, 3.2), dpi=140, sharex=True, gridspec_kw={"height_ratios": [4, 1]}
        )
    else:
        fig, ax = plt.subplots(figsize=(9, 2.6), dpi=140)
        ax_strips = None

    def _axis(frame: float) -> float:
        return (frame / fps) if fps else frame

    if gt is not None:
        for s, e, _ in gt.intervals:
            ax.axvspan(_axis(s), _axis(e), color="tab:red", alpha=0.15, lw=0, label="_event")
    y = np.asarray(scores.scores, dtype=float).copy()
    covered = np.asarray(scores.covered, dtype=bool)
    y[~covered] = np.nan  # frames without a causal window have no score
    ax.plot(x, y, color="tab:blue", lw=1.2, label="anomaly score")
    if threshold is not None:
        ax.axhline(
            threshold, color="tab:orange", ls="--", lw=1.0, label=f"threshold {threshold:.3f}"
        )

    ax.set_ylabel("score")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title or scores.video_id, fontsize=9)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

    if ax_strips is None:
        ax.set_xlabel("seconds" if fps else "frames")
        return save(fig, out_path)

    # Two lanes: the decision on top, the annotation below.
    lanes = (
        ("prediction", tuple(predicted or ()), "tab:orange", 0.62),
        (
            "ground truth",
            tuple((s, e) for s, e, _ in (gt.intervals if gt else ())),
            "tab:red",
            0.18,
        ),
    )
    # `axvspan`, not `axhspan`: its x limits are in data coordinates, so the lanes stay
    # aligned with the event shading above.
    for _, intervals, color, y0 in lanes:
        for s, e in intervals:
            ax_strips.axvspan(
                _axis(s), _axis(e + 1), ymin=y0, ymax=y0 + 0.20, color=color, alpha=0.85, lw=0
            )
    ax_strips.set_ylim(0.0, 1.0)
    # Lane names as ticks: text inside the axes would cover an event at the video start.
    ax_strips.set_yticks([y0 + 0.10 for *_, y0 in lanes])
    ax_strips.set_yticklabels([name for name, *_ in lanes], fontsize=6.5)
    ax_strips.tick_params(axis="y", length=0)
    for side in ("left", "right", "top"):
        ax_strips.spines[side].set_visible(False)
    ax_strips.set_xlabel("seconds" if fps else "frames")
    return save(fig, out_path)


def history_per_step(history: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Reduce ``history`` rows to one per step, in step order.

    ``history.jsonl`` is append-only across executions of the same ``run_id``: a run resumed
    from a checkpoint rewrites its in-memory history after the existing rows. The last row
    of each step (the most recent execution) is kept.

    :param history: rows as stored in the file, in write order.
    :return: one row per step, sorted by ``step``; rows without ``step`` are dropped.
    """
    per_step: dict[int, Mapping[str, Any]] = {}
    for row in history:
        step = row.get("step")
        if step is None:
            continue
        per_step[int(step)] = row
    return [per_step[k] for k in sorted(per_step)]


def _moving_average(
    valori: Sequence[float], finestra: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return a centred moving average and its ±1σ band over the same window.

    :param valori: the per-step series.
    :param finestra: window width in steps; ``<= 1`` returns the series unchanged.
    :return: ``(mean, mean - sigma, mean + sigma)``.
    """
    x = np.asarray(valori, dtype=np.float64)
    if finestra <= 1 or x.size <= 2:
        return x, x, x
    finestra = min(finestra, x.size)
    kernel = np.ones(finestra) / finestra
    # True denominator at each position: at the edges the window sticks out of the data.
    weight = np.convolve(np.ones_like(x), kernel, mode="same")
    mean = np.convolve(x, kernel, mode="same") / weight
    second = np.convolve(x * x, kernel, mode="same") / weight
    sigma = np.sqrt(np.maximum(second - mean * mean, 0.0))
    return mean, mean - sigma, mean + sigma


def training_png(
    history: Sequence[Mapping[str, Any]],
    out_path: str | Path,
    *,
    selected_step: int | None = None,
    selection_metric: str | None = None,
    title: str | None = None,
    mean_window: int | None = None,
) -> Path:
    """Plot the training and validation curves of a run from its ``history`` rows.

    Two stacked axes: losses on top, validation AUROC (the ``edp_v2`` selection criterion)
    below, with the selected step marked on both. The per-step training loss is drawn
    transparent with a moving average and a ±1σ band on top. Validation loss and AUROC
    exist only at evaluation steps and are drawn there only, without interpolation.

    :param history: rows written by :class:`~evd.train.trainer.Trainer`, with ``step`` and
        ``loss`` always and ``val_loss``/``val_auroc`` at evaluation steps.
    :param out_path: PNG destination; missing folders are created.
    :param selected_step: step whose weights were kept, if the selection is not ``last``.
    :param selection_metric: name of the criterion, for the legend.
    :param title: figure title, usually the ``run_id``.
    :param mean_window: moving-average width in steps; ``None`` uses about 2% of the budget
        (at least 5).
    :return: the written path.
    :raises ValueError: if ``history`` is empty.
    """
    plt = agg()

    rows = history_per_step(history)
    if not rows:
        raise ValueError("empty history: nothing to draw")

    steps = [int(r["step"]) for r in rows]
    with_loss = [r for r in rows if r.get("loss") is not None]
    loss_steps = [int(r["step"]) for r in with_loss]
    loss = [float(r["loss"]) for r in with_loss]
    val = [(int(r["step"]), float(r["val_loss"])) for r in rows if r.get("val_loss") is not None]
    auroc = [
        (int(r["step"]), float(r["val_auroc"])) for r in rows if r.get("val_auroc") is not None
    ]

    # A single axis without AUROC: an empty panel would suggest a flat measurement.
    n_axes = 2 if auroc else 1
    fig, axes = plt.subplots(
        n_axes, 1, figsize=(9, 2.9 * n_axes), dpi=140, sharex=True, squeeze=False
    )
    ax_loss = axes[0][0]

    if loss:
        width = mean_window if mean_window else max(5, len(loss) // 50)
        mean, down, over = _moving_average(loss, width)
        # per step, faded: the spread, not the trend
        ax_loss.plot(
            loss_steps,
            loss,
            color="tab:blue",
            lw=0.6,
            alpha=0.22,
            zorder=1,
            label="training loss (per step)",
        )
        ax_loss.fill_between(
            loss_steps, down, over, color="tab:blue", alpha=0.20, lw=0, zorder=2, label="±1σ"
        )
        ax_loss.plot(
            loss_steps,
            mean,
            color="tab:blue",
            lw=2.0,
            zorder=4,
            label=f"training loss (mean over {width} steps)",
        )
    if val:
        ax_loss.plot(
            [s for s, _ in val],
            [v for _, v in val],
            color="tab:red",
            marker="o",
            ms=4.5,
            lw=1.8,
            zorder=5,
            label="validation loss",
        )
    ax_loss.set_ylabel("loss")
    ax_loss.legend(loc="upper right", fontsize=7, framealpha=0.9, ncol=2)
    ax_loss.grid(alpha=0.2, lw=0.5)

    if auroc:
        ax_auc = axes[1][0]
        ax_auc.plot(
            [s for s, _ in auroc],
            [a for _, a in auroc],
            color="tab:green",
            marker="o",
            ms=4.5,
            lw=1.8,
            label="validation AUROC",
        )
        # 0.5 is chance level.
        ax_auc.axhline(0.5, color="grey", ls=":", lw=0.9, label="chance")
        # Keep the honest scale (0.5 visible) and print the values: rescaling on the values alone
        # would exaggerate the slope.
        for s, a in auroc:
            ax_auc.annotate(
                f"{a:.4f}",
                (s, a),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=7,
            )
        ax_auc.set_ylabel("AUROC")
        ax_auc.legend(loc="lower right", fontsize=7, framealpha=0.9)
        ax_auc.grid(alpha=0.2, lw=0.5)

    if selected_step is not None:
        label = f"selected ({selection_metric or 'step'} @ {selected_step})"
        for row in (a[0] for a in axes):
            row.axvline(selected_step, color="tab:orange", ls="--", lw=1.3, zorder=6, label=label)
            label = "_selected"  # a single legend entry, not one per axis
        axes[0][0].legend(loc="upper right", fontsize=7, framealpha=0.9, ncol=2)

    axes[-1][0].set_xlabel("step")
    # `min < max` rather than `if steps`: with a single step matplotlib warns and widens the
    # axis by itself.
    if steps and min(steps) < max(steps):
        axes[-1][0].set_xlim(min(steps), max(steps))
    fig.suptitle(title or "training", fontsize=10)
    return save(fig, out_path)


def lr_png(
    history: Sequence[Mapping[str, Any]],
    out_path: str | Path,
    *,
    selected_step: int | None = None,
    title: str | None = None,
) -> Path:
    """Plot the learning-rate schedule from the same ``history`` rows.

    A separate figure because it is a configuration check (cosine to zero, no warmup), not
    something to read on every run. The top axis is linear (schedule shape), the bottom one
    logarithmic (how many orders of magnitude the tail decays).

    :param history: rows written by :class:`~evd.train.trainer.Trainer`, with ``step`` and
        ``lr``.
    :param out_path: PNG destination; missing folders are created.
    :param selected_step: step whose weights were kept, marked on the curve.
    :param title: figure title, usually the ``run_id``.
    :return: the written path.
    :raises ValueError: if no row carries ``lr``.
    """
    plt = agg()

    points = [
        (int(r["step"]), float(r["lr"]))
        for r in history_per_step(history)
        if r.get("lr") is not None
    ]
    if not points:
        raise ValueError("no history row carries `lr`: nothing to draw")
    steps = [s for s, _ in points]
    lr = [v for _, v in points]
    peak = max(lr)

    fig, axes = plt.subplots(2, 1, figsize=(9, 4.2), dpi=140, sharex=True)
    for ax, log_y in zip(axes, (False, True), strict=True):
        ax.plot(steps, lr, color="tab:purple", lw=1.8, zorder=3, label="learning rate")
        if log_y:
            # Zero does not fit on a log axis: the fill bottom is the smallest non-zero value.
            minimum = min((v for v in lr if v > 0.0), default=peak)
            ax.set_yscale("log")
            ax.set_ylim(minimum * 0.5, peak * 2.0)
            ax.set_ylabel("lr (log)")
        else:
            ax.fill_between(steps, 0.0, lr, color="tab:purple", alpha=0.12, lw=0, zorder=1)
            ax.set_ylim(0.0, peak * 1.12)
            ax.set_ylabel("lr")
        if selected_step is not None:
            ax.axvline(
                selected_step,
                color="tab:orange",
                ls="--",
                lw=1.3,
                zorder=4,
                label=f"selected step {selected_step}",
            )
        ax.grid(alpha=0.2, lw=0.5)
    axes[0].annotate(
        f"peak {peak:.1e}",
        (steps[int(np.argmax(lr))], peak),
        textcoords="offset points",
        xytext=(8, -4),
        fontsize=8,
    )
    axes[0].legend(loc="upper right", fontsize=7, framealpha=0.9)
    axes[-1].set_xlabel("step")
    if min(steps) < max(steps):
        axes[-1].set_xlim(min(steps), max(steps))
    fig.suptitle(title or "learning-rate schedule", fontsize=10)
    return save(fig, out_path)


def curves_png(
    righe: Sequence[Mapping[str, Any]],
    out_path: str | Path,
    *,
    metric: str,
    title: str | None = None,
) -> Path:
    """Plot performance against k, one line per method, with error bars (few-shot regime).

    Drawn from the tidy CSV of :class:`~evd.eval.aggregate.SeedAggregator`. Single-seed
    points have no error bar (missing ``std`` differs from zero ``std``) and non-numeric k
    values (``full``, ``na``) are skipped.

    :param righe: tidy rows, one per ``(group, metric)``.
    :param out_path: PNG destination.
    :param metric: metric to draw.
    :param title: figure title; defaults to the metric name.
    :return: the written path.
    :raises ValueError: if no row refers to that metric with a numeric k.
    """
    plt = agg()

    per_method: dict[str, list[tuple[int, float, float | None]]] = {}
    for row in righe:
        if str(row.get("metric")) != metric:
            continue
        k = str(row.get("k", ""))
        if not k.isdigit():
            continue
        deviation = row.get("std")
        per_method.setdefault(str(row.get("adapt", "?")), []).append(
            (int(k), float(row["mean"]), None if deviation in (None, "") else float(deviation))
        )
    if not per_method:
        raise ValueError(
            f"no row with metric={metric!r} and a numeric k: the few-shot curve "
            f"is drawn on the grid k values, not on 'full' or 'na'"
        )

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=140)
    for method in sorted(per_method):
        points = sorted(per_method[method])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        # `None` where the spread was not measured: matplotlib draws a zero bar.
        errors = [0.0 if p[2] is None else p[2] for p in points]
        ax.errorbar(xs, ys, yerr=errors, marker="o", capsize=3, lw=1.4, label=method)

    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({p[0] for points in per_method.values() for p in points}))
    from matplotlib.ticker import ScalarFormatter

    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel("k — annotated event segments per class")
    ax.set_ylabel(metric)
    ax.set_title(title or f"few-shot curve — {metric}", fontsize=10)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=8, framealpha=0.9)
    return save(fig, out_path)
