"""Figures of a single cell: curves, confusion matrices, breakdowns.

Companion of :mod:`evd.eval.plots` (timelines and training curves): ROC,
precision-recall, confusion matrices at the two comparable levels, per-class bars,
threshold sensitivity and score distribution. Curves come from the scores on disk
(``scores.jsonl``, ADR-051) and summaries from ``metrics.json``; nothing the metrics
already decided is recomputed. The event level has no confusion matrix (precision and
recall have different denominators), so it is shown as explicit counts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import precision_recall_curve, roc_curve

from evd.eval._figure import agg, save
from evd.utils.io import read_jsonl
from evd.utils.logging import get_logger

log = get_logger(__name__)

#: Green/red are never the only difference: figures must read in grayscale.
COLOR = {
    "test": "tab:blue",
    "val": "tab:orange",
    "anomalous": "tab:purple",
    "positive": "tab:red",
    "negative": "tab:blue",
    "chance": "0.55",
}


_agg = agg
_save = save


# --------------------------------------------------------------------------- #
# Scores on disk
# --------------------------------------------------------------------------- #


def read_scores(percorso: str | Path) -> dict[str, dict[str, npt.NDArray[Any]]]:
    """Read ``scores.jsonl`` and group it by partition.

    :param percorso: file written by ``ExperimentRunner._write_scores``.
    :return: partition name to ``{"score", "label", "class"}`` aligned arrays; windows
        labelled ``-1`` (unknown) are dropped.
    """
    rows = read_jsonl(percorso)
    outside: dict[str, dict[str, npt.NDArray[Any]]] = {}
    for split in sorted({str(r.get("split", "test")) for r in rows}):
        sel = [r for r in rows if str(r.get("split", "test")) == split and int(r["label"]) >= 0]
        if not sel:
            continue
        outside[split] = {
            "score": np.asarray([float(r["score"]) for r in sel], dtype=np.float64),
            "label": np.asarray([int(r["label"]) for r in sel], dtype=np.int64),
            "class": np.asarray([int(r.get("class", -1)) for r in sel], dtype=np.int64),
        }
    return outside


# --------------------------------------------------------------------------- #
# ROC and precision-recall
# --------------------------------------------------------------------------- #


def _closed_roc(
    y: npt.NDArray[Any], s: npt.NDArray[Any]
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Return the ROC with both end points, (0,0) and (1,1), added only if missing."""
    fpr, tpr, _ = roc_curve(y, s)
    if fpr[0] > 0.0 or tpr[0] > 0.0:
        fpr, tpr = np.r_[0.0, fpr], np.r_[0.0, tpr]
    if fpr[-1] < 1.0 or tpr[-1] < 1.0:
        fpr, tpr = np.r_[fpr, 1.0], np.r_[tpr, 1.0]
    return fpr, tpr


#: Series colour order, shared by ROC and PR so the same set has the same colour.
_SERIES_COLORS = (COLOR["test"], COLOR["anomalous"], COLOR["val"])


def _operating_point(
    y: npt.NDArray[Any], s: npt.NDArray[Any], soglia: float
) -> tuple[float, float]:
    """Return where the operating threshold falls on the (FPR, TPR) plane.

    The rates are recomputed with the protocol rule ``score >= soglia`` instead of being
    looked up on the ``roc_curve`` output, which may skip that threshold.

    :param y: binary labels.
    :param s: scores.
    :param soglia: operating threshold.
    :return: ``(fpr, tpr)`` at that threshold.
    """
    positive = s >= soglia
    pos = y > 0.5
    neg = ~pos
    tpr = float(positive[pos].mean()) if pos.any() else float("nan")
    fpr = float(positive[neg].mean()) if neg.any() else float("nan")
    return fpr, tpr


def _curve_png(
    curve: Mapping[str, tuple[npt.NDArray[Any], npt.NDArray[Any]]],
    out_path: str | Path,
    *,
    punti: Any,
    xlabel: str,
    ylabel: str,
    code_name: str,
    valori: Mapping[str, float] | None,
    riferimento: Any,
    per_series: bool = False,
    marker: Any = None,
    legend_loc: str,
    title: str,
) -> Path:
    """Draw the scaffolding shared by ROC and PR: square axes, series, reference.

    :param curve: label to ``(y_true, y_score)``.
    :param out_path: PNG destination.
    :param punti: ``(y, s) -> (xs, ys)``, the curve points of one series.
    :param xlabel: x-axis label.
    :param ylabel: y-axis label.
    :param code_name: ``"AUC"`` or ``"AP"``, for the legend.
    :param valori: precomputed values for the legend (from ``metrics.json``), or ``None``.
    :param riferimento: ``(ax, name, y, color) -> None``, called inside the loop if
        ``per_series``, once after it otherwise.
    :param per_series: whether the reference is drawn per series.
    :param marker: ``(ax, name, y, s, color) -> None`` called after each curve to mark a
        point on it (the operating threshold), or ``None``.
    :param legend_loc: legend corner.
    :param title: figure title.
    :return: the written path.
    :raises ValueError: if ``curve`` is empty.
    """
    if not curve:
        raise ValueError("no curve to draw")
    plt = _agg()
    fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=140)
    for i, (name, (y, s)) in enumerate(curve.items()):
        xs, ys = punti(np.asarray(y), np.asarray(s))
        c = _SERIES_COLORS[i % len(_SERIES_COLORS)]
        label = name
        if valori is not None and name in valori:
            label = f"{name} ({code_name} = {valori[name]:.4f})"
        ax.plot(xs, ys, color=c, lw=1.8, label=label, zorder=3)
        ax.fill_between(xs, ys, alpha=0.10, color=c, lw=0, zorder=1)
        if per_series:
            riferimento(ax, name, np.asarray(y), c)
        if marker is not None:
            marker(ax, name, np.asarray(y), np.asarray(s), c)
    if not per_series:
        riferimento(ax, None, None, None)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(loc=legend_loc, fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.2, lw=0.5)
    ax.set_aspect("equal")
    return _save(fig, out_path)


def roc_png(
    curve: Mapping[str, tuple[npt.NDArray[Any], npt.NDArray[Any]]],
    out_path: str | Path,
    *,
    auc: Mapping[str, float] | None = None,
    soglia: float | None = None,
    title: str | None = None,
) -> Path:
    """Plot overlaid ROC curves, closed and filled, with the operating point if given.

    :param curve: label to ``(y_true, y_score)``; several entries share the axes (e.g. AUC
        against AUC_a).
    :param out_path: PNG destination.
    :param auc: precomputed AUCs for the legend (from ``metrics.json``, never recomputed).
    :param soglia: operating threshold to mark on every series, or ``None``.
    :param title: figure title, usually the ``run_id``.
    :return: the written path.
    :raises ValueError: if ``curve`` is empty.
    """

    def diagonal(ax: Any, _name: Any, _y: Any, _c: Any) -> None:
        ax.plot([0, 1], [0, 1], ls=":", lw=1.0, color=COLOR["chance"], label="chance", zorder=2)

    # A single threshold even with several series: only the first one adds a legend entry.
    already_in_legend: list[bool] = []

    def point(ax: Any, _name: str, y: Any, s: Any, c: Any) -> None:
        if soglia is None:
            return
        fpr, tpr = _operating_point(y, s, float(soglia))
        if fpr != fpr or tpr != tpr:  # NaN: the series lacks one of the two classes
            return
        label = "_soglia" if already_in_legend else f"operating threshold = {float(soglia):.4f}"
        already_in_legend.append(True)
        ax.plot(
            [fpr],
            [tpr],
            marker="o",
            ms=8,
            mfc="none",
            mew=1.8,
            color=COLOR["positive"],
            zorder=6,
            label=label,
        )
        # Label below-right of the point: the ROC rises on the left.
        ax.annotate(
            f"TPR {tpr:.3f}\nFPR {fpr:.3f}",
            (fpr, tpr),
            textcoords="offset points",
            xytext=(9, -14),
            fontsize=7.5,
            color=COLOR["positive"],
            zorder=6,
        )

    return _curve_png(
        curve,
        out_path,
        punti=lambda y, s: _closed_roc(y, s),
        xlabel="False positive rate",
        ylabel="True positive rate",
        code_name="AUC",
        valori=auc,
        riferimento=diagonal,
        marker=None if soglia is None else point,
        legend_loc="lower right",
        title=title or "ROC",
    )


def pr_png(
    curve: Mapping[str, tuple[npt.NDArray[Any], npt.NDArray[Any]]],
    out_path: str | Path,
    *,
    ap: Mapping[str, float] | None = None,
    title: str | None = None,
) -> Path:
    """Plot precision-recall curves with the chance line at the positive rate.

    The chance line is the positive frequency (about 16% of chunks), not 0.5.

    :param curve: label to ``(y_true, y_score)``.
    :param out_path: PNG destination.
    :param ap: precomputed APs for the legend (see :func:`roc_png`).
    :param title: figure title.
    :return: the written path.
    :raises ValueError: if ``curve`` is empty.
    """

    def positive_rate(ax: Any, nome: str, y: Any, c: Any) -> None:
        # One chance line per series: each set has its own positive rate.
        base = float(y.mean())
        ax.axhline(base, color=c, ls=":", lw=1.0, zorder=2, label=f"chance ({nome}) = {base:.3f}")

    return _curve_png(
        curve,
        out_path,
        punti=lambda y, s: precision_recall_curve(y, s)[1::-1],
        xlabel="Recall",
        ylabel="Precision",
        code_name="AP",
        valori=ap,
        riferimento=positive_rate,
        per_series=True,
        legend_loc="upper right",
        title=title or "Precision-recall",
    )


# --------------------------------------------------------------------------- #
# Confusion matrices
# --------------------------------------------------------------------------- #


def confusion_from_prf(prf: Mapping[str, Any], n_total: int) -> tuple[int, int, int, int]:
    """Rebuild ``(tn, fp, fn, tp)`` from a P/R/F1 triple and the total.

    Gives a matrix to runs without ``scores.jsonl`` (before ADR-051): the four counts are
    determined by what ``metrics.json`` stores.

        n_pred_pos = tp + fp      n_true_pos = tp + fn      precision = tp / n_pred_pos

    :param prf: :class:`~evd.eval.metrics.PRF` dictionary, with ``precision``,
        ``n_pred_pos`` and ``n_true_pos``.
    :param n_total: number of judged units (chunks or videos).
    :return: ``(tn, fp, fn, tp)``, the ``sklearn.confusion_matrix`` order.
    :raises ValueError: if the counts do not fit in the total.
    """
    n_pred_pos = int(prf["n_pred_pos"])
    n_true_pos = int(prf["n_true_pos"])
    tp = int(round(float(prf["precision"]) * n_pred_pos))
    fp, fn = n_pred_pos - tp, n_true_pos - tp
    tn = n_total - tp - fp - fn
    if min(tp, fp, fn, tn) < 0:
        raise ValueError(
            f"counts inconsistent with the total {n_total}: tp={tp} fp={fp} fn={fn} tn={tn}. "
            "The given total is not the one this triple was computed on"
        )
    return tn, fp, fn, tp


def confusion_png(
    matrices: Mapping[str, tuple[int, int, int, int]],
    out_path: str | Path,
    *,
    title: str | None = None,
    normalize: bool = True,
) -> Path:
    """Plot side by side the confusion matrices of the levels that have one.

    The event level is excluded (no true negatives, see :func:`event_png`). Each cell shows
    the count and, with ``normalize``, the row share (recall and specificity).

    :param matrices: level name to ``(tn, fp, fn, tp)``.
    :param out_path: PNG destination.
    :param title: figure title.
    :param normalize: add the row share below the count.
    :return: the written path.
    :raises ValueError: if ``matrices`` is empty.
    """
    if not matrices:
        raise ValueError("no matrix to draw")
    plt = _agg()
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.5), dpi=140, squeeze=False)
    for ax, (name, (tn, fp, fn, tp)) in zip(axes[0], matrices.items(), strict=True):
        cm = np.array([[tn, fp], [fn, tp]], dtype=float)
        rows = cm.sum(axis=1, keepdims=True)
        quotas = np.divide(cm, np.maximum(rows, 1))
        ax.imshow(quotas, cmap="Blues", vmin=0.0, vmax=1.0)
        for i in range(2):
            for j in range(2):
                text = f"{int(cm[i, j])}"
                if normalize:
                    text += f"\n{quotas[i, j]:.1%}"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=9,
                    # Text contrast follows the fill.
                    color="white" if quotas[i, j] > 0.55 else "black",
                )
        ax.set_xticks([0, 1], ["pred. normal", "pred. anomalous"], fontsize=8)
        ax.set_yticks([0, 1], ["true normal", "true anomalous"], fontsize=8)
        ax.set_title(f"{name}  (n = {int(cm.sum())})", fontsize=9)
    fig.suptitle(title or "Confusion matrices", fontsize=10)
    return _save(fig, out_path)


def event_png(
    event_prf: Mapping[str, Any],
    out_path: str | Path,
    *,
    latency: Mapping[str, Any] | None = None,
    title: str | None = None,
) -> Path:
    """Plot the event level, which is not a confusion matrix.

    Two bars because the denominators differ: annotated events (detected vs missed) and
    emitted alarms (true vs false).

    :param event_prf: :class:`~evd.eval.metrics.EventPRF` dictionary.
    :param out_path: PNG destination.
    :param latency: ``detection_latency``, if available; median and p90 go in the subtitle.
    :param title: figure title.
    :return: the written path.
    """
    plt = _agg()
    n_ev = int(event_prf["n_eventi"])
    rel = int(event_prf["n_eventi_rilevati"])
    n_at = int(event_prf["n_allarmi"])
    true_items = int(event_prf["n_allarmi_veri"])

    fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=140)
    x = [0, 1]
    below = [rel, true_items]
    above = [n_ev - rel, n_at - true_items]
    ax.bar(x, below, color=COLOR["test"], label="detected / true", width=0.55)
    ax.bar(x, above, bottom=below, color=COLOR["positive"], label="missed / false", width=0.55)
    for xi, (a, b) in enumerate(zip(below, above, strict=True)):
        ax.text(xi, a / 2, str(a), ha="center", va="center", color="white", fontsize=9)
        if b:
            ax.text(xi, a + b / 2, str(b), ha="center", va="center", color="white", fontsize=9)
    ax.set_xticks(
        x,
        [f"annotated events\n(n = {n_ev})", f"emitted alarms\n(n = {n_at})"],
        fontsize=9,
    )
    ax.set_ylabel("count")
    sub = (
        f"recall = {float(event_prf['recall']):.3f}   "
        f"precision = {float(event_prf['precision']):.3f}   "
        f"F1 = {float(event_prf['f1']):.3f}"
    )
    if latency:
        # Not "never detected": the historical latency rule requires a chunk to end inside the
        # interval, while ADR-048 counts any intersection, so the two numbers differ. Runs before
        # 2 September 2026 store the old field `n_mai_rilevati`.
        without = int(latency.get("n_senza_latenza_misurabile", latency.get("n_mai_rilevati", 0)))
        measurable = int(latency["n_eventi"]) - without
        # The rule travels with the number (ADR-061); runs before 4 September 2026 lack the field
        # and use the historical rule.
        rule = str(latency.get("regola", "endpoint_dentro"))
        qualifier = (
            "chunk overlapping the interval"
            if rule == "intersezione"
            else "chunk ending inside the interval"
        )
        sub += (
            f"\nlatency measured on {measurable}/{int(latency['n_eventi'])} events "
            f"({qualifier}): "
            f"median {float(latency['mediana_s']):.2f} s, p90 {float(latency['p90_s']):.2f} s"
        )
    ax.set_title(f"{title or 'Event level'}\n{sub}", fontsize=9)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.2, lw=0.5, axis="y")
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Breakdowns
# --------------------------------------------------------------------------- #


def per_class_png(
    per_class: Mapping[str, Mapping[str, Any]],
    out_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Plot AUC and AP per anomaly class, sorted by AP.

    AP is the metric that exposes failing classes (per-class AUC stays high even with six
    positives); the number of positives is in the label.

    :param per_classe: ``metrics.json::per_class_chunk``.
    :param out_path: PNG destination.
    :param title: figure title.
    :return: the written path.
    :raises ValueError: if there are no classes.
    """
    if not per_class:
        raise ValueError("no class to draw")
    plt = _agg()
    entry_list = sorted(per_class.items(), key=lambda kv: float(kv[1].get("ap", 0.0)))
    names = [f"{k}  (n={int(v.get('n_pos', 0))})" for k, v in entry_list]
    auc = [float(v.get("auc", float("nan"))) for _, v in entry_list]
    ap = [float(v.get("ap", float("nan"))) for _, v in entry_list]
    y = np.arange(len(entry_list))

    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(entry_list) + 1.6), dpi=140)
    ax.barh(y - 0.2, auc, height=0.38, color=COLOR["test"], label="AUC")
    ax.barh(y + 0.2, ap, height=0.38, color=COLOR["anomalous"], label="AP")
    ax.axvline(0.5, color=COLOR["chance"], ls=":", lw=1.0, label="chance (AUC)")
    ax.set_yticks(y, names, fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("score")
    ax.set_title(title or "Per-class chunk AUC and AP", fontsize=10)
    # Legend outside the axes: bars reach the right edge.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10 - 0.02 * (len(entry_list) > 8)),
        ncol=3,
        fontsize=8,
        framealpha=0.9,
    )
    ax.grid(alpha=0.2, lw=0.5, axis="x")
    return _save(fig, out_path)


#: The four stratified axes: block name, figure title and file name. Metrics are in
#: reading order: threshold-free first, then the thresholded triple.
STRATA_AXES: dict[str, dict[str, Any]] = {
    "copertura": {
        "dove": ("chunk", "copertura"),
        "file": "strata_copertura",
        "titolo": "Chunk metrics by how much of the window the event covers",
        "asse": "coverage of the window",
        "conteggio": ("n_pos", "positive chunks"),
        "metriche": (
            ("auc", "AUC", "tab:blue"),
            ("ap", "AP", "tab:purple"),
            ("precision", "precision", "tab:green"),
            ("recall", "recall", "tab:red"),
            ("f1", "F1", "tab:orange"),
        ),
    },
    "durata": {
        "dove": ("evento", "durata"),
        "file": "strata_durata",
        "titolo": "Event metrics by how long the event lasts",
        "asse": "event duration",
        "conteggio": ("n_eventi", "annotated events"),
        "metriche": (
            ("precision", "precision", "tab:green"),
            ("recall", "recall", "tab:red"),
            ("f1", "F1", "tab:orange"),
        ),
    },
    "classe": {
        "dove": ("evento", "classe"),
        "file": "strata_classe",
        "titolo": "Event metrics by annotated class",
        "asse": "annotated class",
        "conteggio": ("n_eventi", "annotated events"),
        "metriche": (
            ("precision", "precision", "tab:green"),
            ("recall", "recall", "tab:red"),
            ("f1", "F1", "tab:orange"),
        ),
    },
    "quota_anomala_video": {
        "dove": ("evento", "quota_anomala_video"),
        "file": "strata_quota",
        "titolo": "Event metrics by the anomalous share of the video",
        "asse": "anomalous share of the video",
        "conteggio": ("n_eventi", "annotated events"),
        "metriche": (
            ("precision", "precision", "tab:green"),
            ("recall", "recall", "tab:red"),
            ("f1", "F1", "tab:orange"),
        ),
    },
}


def strata_axis_png(
    strati: Mapping[str, Any],
    out_path: str | Path,
    axis: str,
    *,
    title: str | None = None,
) -> Path:
    """Plot one stratified axis with all its metrics and the values next to the bars.

    Per-axis version of :func:`strata_png`, readable on a slide. ``precision`` is not the
    aggregated run precision: false positives are shared across bands (see
    :class:`~evd.eval.strata.Stratum` and :class:`~evd.eval.strata.EventStratum`), as the
    figure states below the axis.

    :param strati: the ``strata`` block (from ``metrics.json`` or ``strata.json``).
    :param out_path: PNG destination; for a folder, the file name follows the axis
        (``strata_copertura.png`` and similar).
    :param asse: axis to draw, a key of :data:`STRATA_AXES`.
    :param title: figure title, defaults to the axis title.
    :return: the written path.
    :raises KeyError: if ``asse`` is not a known axis.
    :raises ValueError: if the block does not contain that axis.
    """
    if axis not in STRATA_AXES:
        raise KeyError(f"unknown axis {axis!r}: known axes are {sorted(STRATA_AXES)}")
    spec = STRATA_AXES[axis]
    level, key = spec["dove"]
    rows = (strati.get(level) or {}).get(key)
    if not rows:
        raise ValueError(
            f"the `strata` block does not contain axis {level}.{key}: "
            "a run without event level only has the coverage axis"
        )
    out_path = Path(out_path)
    if out_path.is_dir():
        out_path = out_path / f"{spec['file']}.png"

    plt = _agg()
    entry_list = list(rows.items())
    metrics = spec["metriche"]
    n_field, what_n = spec["conteggio"]
    names = [f"{k}\n(n={int(v.get(n_field, 0))} {what_n.split()[0]})" for k, v in entry_list]

    y = np.arange(len(entry_list))
    step = 0.80 / len(metrics)
    height = step * 0.86
    # The value column lives outside the 0-1 scale (ticks stop at 1.0).
    x_numbers = 1.16
    fig, ax = plt.subplots(
        figsize=(8.2, 0.42 * len(entry_list) * len(metrics) / 3.0 + 1.9), dpi=140
    )
    for j, (field, name, color) in enumerate(metrics):
        gap = (j - (len(metrics) - 1) / 2.0) * step
        values = [float(v.get(field, float("nan"))) for _, v in entry_list]
        ax.barh(y + gap, np.nan_to_num(values), height=height, color=color, label=name)
        for i, value in enumerate(values):
            quota = y[i] + gap
            if value != value:
                ax.text(x_numbers, quota, "--", ha="right", va="center", fontsize=7, color="0.5")
                continue
            # The dashed line links each value to the end of its bar.
            ax.plot(
                [value, x_numbers - 0.035],
                [quota, quota],
                ls=(0, (1.6, 1.8)),
                lw=0.7,
                color=color,
                alpha=0.75,
                zorder=1,
            )
            ax.text(
                x_numbers,
                quota,
                f"{value:.3f}",
                ha="right",
                va="center",
                fontsize=7.2,
                color="0.15",
            )
    ax.axvline(0.5, color=COLOR["chance"], ls=":", lw=1.0)
    ax.set_yticks(y, names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, x_numbers + 0.01)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_ylabel(spec["asse"], fontsize=9)
    ax.grid(alpha=0.2, lw=0.5, axis="x")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=len(metrics),
        fontsize=8,
        framealpha=0.9,
    )
    # No x label: the legend lives below the axis.
    if level == "evento":
        # Latency is not a 0-1 metric: it is printed below the figure, band by band.
        code = []
        for k, v in entry_list:
            lat = float(v.get("latenza_mediana_s", float("nan")))
            code.append(f"{k} {lat:.2f}s" if lat == lat else f"{k} --")
        below = "median detection latency:  " + "   ".join(code)
    else:
        n_fp = int((strati.get("riferimento_negativi") or {}).get("n_sopra_soglia", 0))
        below = (
            f"precision and F1 share the same {n_fp} false positives in every band: "
            "the bands compare with each other, they do not add up to the run's own precision"
        )
    fig.suptitle(title or spec["titolo"], fontsize=10.5)
    # The note goes below the legend: `set_xlabel` would overlap it.
    fig.text(0.5, 0.012, below, ha="center", va="bottom", fontsize=7.4, color="0.32", wrap=True)
    return _save(fig, out_path, rect=(0, 0.045, 1, 1))


def strata_png(
    strati: Mapping[str, Any],
    out_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Plot where the detector performs better or worse: one panel per axis.

    Overview of the ``strata`` block with one metric per axis (AUC/AP for the chunk axis,
    recall with median latency for event axes); :func:`strata_axis_png` draws all metrics
    of one axis. Horizontal bars carry the count in the label, and empty bands are kept
    with ``n=0``.

    :param strati: the ``strata`` block; event axes may be missing (``event_level`` not
        requested).
    :param out_path: PNG destination.
    :param title: figure title.
    :return: the written path.
    :raises ValueError: if the block contains no drawable axis.
    """
    plt = _agg()
    panels: list[tuple[str, Mapping[str, Any], bool]] = []
    coverage = (strati.get("chunk") or {}).get("copertura")
    if coverage:
        panels.append(("chunk: window coverage", coverage, False))
    for key, label in (
        ("classe", "event: annotated class"),
        ("durata", "event: duration"),
        ("quota_anomala_video", "event: anomalous share of the video"),
    ):
        row = (strati.get("evento") or {}).get(key)
        if row:
            panels.append((label, row, True))
    if not panels:
        raise ValueError("the `strata` block contains no drawable axis")

    heights = [0.30 * len(rows) + 1.0 for _, rows, _ in panels]
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(7.6, sum(heights)),
        dpi=140,
        gridspec_kw={"height_ratios": heights},
    )
    for ax, (label, rows, is_event) in zip(np.atleast_1d(axes), panels, strict=True):
        entry_list = list(rows.items())
        y = np.arange(len(entry_list))
        if is_event:
            names = [f"{k}  (n={int(v.get('n_eventi', 0))})" for k, v in entry_list]
            values = [float(v.get("recall", float("nan"))) for _, v in entry_list]
            ax.barh(y, np.nan_to_num(values), height=0.6, color=COLOR["anomalous"])
            for i, (_, v) in enumerate(entry_list):
                latency = float(v.get("latenza_mediana_s", float("nan")))
                text = f"{latency:.1f}s" if latency == latency else "--"
                ax.text(1.03, i, text, ha="left", va="center", fontsize=7, color="0.25")
            # The latency rule goes in the label (ADR-061); `strata.json` files before 4 September
            # 2026 lack the field and use the historical rule.
            rule = str(strati.get("regola_latenza", "endpoint_dentro"))
            how = "overlap" if rule == "intersezione" else "ends-inside"
            ax.set_xlabel(f"event recall  (median latency at the right, {how} rule)", fontsize=8)
        else:
            names = [f"{k}  (n={int(v.get('n_pos', 0))})" for k, v in entry_list]
            auc = [float(v.get("auc", float("nan"))) for _, v in entry_list]
            ap = [float(v.get("ap", float("nan"))) for _, v in entry_list]
            ax.barh(y - 0.18, np.nan_to_num(auc), height=0.34, color=COLOR["test"], label="AUC")
            ax.barh(y + 0.18, np.nan_to_num(ap), height=0.34, color=COLOR["anomalous"], label="AP")
            ax.axvline(0.5, color=COLOR["chance"], ls=":", lw=1.0)
            # Legend in the right strip: a bar at 1.0 is the normal case here.
            ax.legend(fontsize=7, loc="center right", framealpha=0.9)
            ax.set_xlabel("score", fontsize=8)
        ax.set_yticks(y, names, fontsize=8)
        # The right margin hosts legend and latency; ticks stop at 1.0.
        ax.set_xlim(0, 1.22)
        ax.set_xticks(np.linspace(0.0, 1.0, 6))
        ax.set_title(label, fontsize=9, loc="left")
        ax.grid(alpha=0.2, lw=0.5, axis="x")
    fig.suptitle(title or "Where the detector does better and worse", fontsize=10)
    return _save(fig, out_path)


#: How the threshold was chosen, as a phrase for figures (the raw ``val_youden`` code
#: stays in ``metrics.json``).
AS_CHOSEN = {
    "val_youden": "max TPR − FPR on the validation ROC",
    "val_f1": "best F1 on validation",
    "fixed": "fixed value",
}


def _as_choice(sorgente: Any) -> str:
    """:return: how the threshold was chosen, in English, or the raw code if unknown."""
    return AS_CHOSEN.get(str(sorgente), str(sorgente or "threshold"))


def threshold_png(
    sensitivity: Sequence[Mapping[str, Any]],
    out_path: str | Path,
    *,
    operating: Mapping[str, Any] | None = None,
    title: str | None = None,
) -> Path:
    """Plot precision, recall and F1 against the threshold, marking the operating one.

    Shows how much the reported number depends on the chosen point.

    :param sensitivity: ``metrics.json::threshold_sensitivity``.
    :param out_path: PNG destination.
    :param operating: the ``threshold`` block, to mark the operating point.
    :param title: figure title.
    :return: the written path.
    :raises ValueError: if the sensitivity is empty.
    """
    if not sensitivity:
        raise ValueError("no sensitivity point to draw")
    plt = _agg()
    entry_list = sorted(sensitivity, key=lambda r: float(r["threshold"]))
    tau = [float(r["threshold"]) for r in entry_list]
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=140)
    for key, color, name in (
        ("precision", COLOR["test"], "precision"),
        ("recall", COLOR["positive"], "recall"),
        ("f1", COLOR["anomalous"], "F1"),
    ):
        ax.plot(
            tau,
            [float(r["chunk_prf"][key]) for r in entry_list],
            marker="o",
            ms=3.5,
            lw=1.4,
            color=color,
            label=name,
        )
    if operating is not None:
        ax.axvline(
            float(operating["value"]),
            color="tab:green",
            ls="--",
            lw=1.2,
            label=f"operating point — {_as_choice(operating.get('source'))}",
        )
    ax.set_xlabel("threshold")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.02)
    ax.set_title(title or "Chunk-level metrics vs threshold", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.2, lw=0.5)
    return _save(fig, out_path)


def score_hist_png(
    y: npt.NDArray[Any],
    s: npt.NDArray[Any],
    out_path: str | Path,
    *,
    threshold: float | None = None,
    title: str | None = None,
) -> Path:
    """Plot the score distribution, split by true label.

    Shows why precision and recall sit where they do relative to the threshold. The y axis
    is logarithmic because negatives outnumber positives about 5 to 1.

    :param y: binary labels.
    :param s: scores.
    :param out_path: PNG destination.
    :param threshold: operating threshold, to mark.
    :param title: figure title.
    :return: the written path.
    """
    plt = _agg()
    y = np.asarray(y)
    s = np.asarray(s)
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=140)
    bins = np.linspace(0.0, 1.0, 51)
    # Translucent fill plus solid outline, so overlapping profiles stay distinguishable.
    for mask, color, name in (
        (y == 0, COLOR["negative"], "normal"),
        (y == 1, COLOR["positive"], "anomalous"),
    ):
        ax.hist(s[mask], bins=bins, color=color, alpha=0.35)
        ax.hist(
            s[mask],
            bins=bins,
            histtype="step",
            color=color,
            lw=1.6,
            label=f"{name} chunks (n = {int(mask.sum())})",
        )
    if threshold is not None:
        ax.axvline(
            threshold, color="tab:green", ls="--", lw=1.4, label=f"threshold = {threshold:.4f}"
        )
    ax.set_yscale("log")
    ax.set_xlabel("anomaly score")
    ax.set_ylabel("count (log)")
    ax.set_title(title or "Score distribution by true label", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.2, lw=0.5)
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Full figure set of a cell
# --------------------------------------------------------------------------- #

#: Figures drawn from `metrics.json` (any run); the score-based ones need
#: `scores.jsonl` (runs after ADR-051).
METRICS_FIGURES = ("confusion", "event", "per_class", "strata", "threshold")
SCORE_FIGURES = ("roc", "pr", "score_hist")
#: Figures drawn from ``history.jsonl``, redrawn from the same rows so drawing changes
#: apply to completed cells too.
HISTORY_FIGURES = ("training", "lr")


def run_figures(
    run_dir: str | Path,
    *,
    solo: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Write every figure that can be derived from a run folder.

    Reads ``metrics.json`` and, if present, ``scores.jsonl``. A figure that cannot be made
    is skipped and logged (e.g. no ROC for runs before ADR-051).

    :param run_dir: run folder (containing ``metrics.json``).
    :param solo: if given, the names of the only figures to produce.
    :return: figure name to written path.
    :raises FileNotFoundError: if ``metrics.json`` is missing.
    """
    import json

    d = Path(run_dir)
    metrics_path = d / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"no metrics.json in {d}: not a run folder")
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_id = str(m.get("run_id", d.name))
    wanted = lambda n: solo is None or n in solo  # noqa: E731
    outside: dict[str, Path] = {}

    # --- from metrics ------------------------------------------------------ #
    if wanted("confusion"):
        matrices: dict[str, tuple[int, int, int, int]] = {}
        if "chunk_prf" in m and m.get("n_chunk_eval"):
            matrices["chunk level"] = confusion_from_prf(m["chunk_prf"], int(m["n_chunk_eval"]))
        if "video_prf" in m and m.get("n_eval_videos"):
            matrices["video level"] = confusion_from_prf(m["video_prf"], int(m["n_eval_videos"]))
        if matrices:
            outside["confusion"] = confusion_png(
                matrices, d / "confusion.png", title=f"{run_id} — confusion matrices"
            )

    if wanted("event") and "event_prf" in m:
        outside["event"] = event_png(
            m["event_prf"],
            d / "event.png",
            # The intersection rule if present, the historical one otherwise (ADR-061).
            latency=m.get("detection_latency_intersezione") or m.get("detection_latency"),
            title=f"{run_id} — event level",
        )

    if wanted("per_class") and m.get("per_class_chunk"):
        outside["per_class"] = per_class_png(
            m["per_class_chunk"], d / "per_class.png", title=f"{run_id} — per-class chunk metrics"
        )

    if wanted("strata") and m.get("strata"):
        outside["strata"] = strata_png(
            m["strata"], d / "strata.png", title=f"{run_id} — where it does better and worse"
        )

    if wanted("threshold") and m.get("threshold_sensitivity"):
        outside["threshold"] = threshold_png(
            m["threshold_sensitivity"],
            d / "threshold.png",
            operating=m.get("threshold"),
            title=f"{run_id} — sensitivity to the threshold",
        )

    # --- from training history --------------------------------------------- #
    history_path = d / "history.jsonl"
    if history_path.is_file() and any(wanted(n) for n in HISTORY_FIGURES):
        from evd.eval.plots import lr_png, training_png

        history = list(read_jsonl(history_path))
        selection = m.get("model_selection") or {}
        step = selection.get("step")
        step = None if step is None else int(step)
        if wanted("training"):
            outside["training"] = training_png(
                history,
                d / "training.png",
                selected_step=step,
                selection_metric=selection.get("metric"),
                title=run_id,
            )
        # No figure for runs without `lr` in the history (profiles, smoke runs).
        if wanted("lr") and any(r.get("lr") is not None for r in history):
            outside["lr"] = lr_png(
                history,
                d / "lr.png",
                selected_step=step,
                title=f"{run_id} — learning-rate schedule",
            )

    # --- from scores ------------------------------------------------------- #
    scores_path = d / "scores.jsonl"
    if not scores_path.is_file():
        log.info(
            "%s: no scores.jsonl, skipping %s. The run predates ADR-051 "
            "(scores were not written); a re-evaluation produces it",
            d.name,
            ", ".join(SCORE_FIGURES),
        )
        return outside

    per_split = read_scores(scores_path)
    eval_split = str(m.get("eval_split", "test"))
    data = per_split.get(eval_split)
    if data is None:
        log.warning("%s: scores.jsonl does not contain partition %r", d.name, eval_split)
        return outside

    curves = {eval_split: (data["label"], data["score"])}
    auc = {eval_split: float(m["chunk_auc"])} if "chunk_auc" in m else None
    ap = {eval_split: float(m["chunk_ap"])} if "chunk_ap" in m else None
    # Two-line title: the `run_id` is long. The operating threshold comes from
    # `metrics.json`.
    threshold = m.get("threshold", {}).get("value")
    if wanted("roc"):
        outside["roc"] = roc_png(
            curves,
            d / "roc.png",
            auc=auc,
            soglia=None if threshold is None else float(threshold),
            title=f"{run_id}\nchunk-level ROC",
        )
    if wanted("pr"):
        outside["pr"] = pr_png(
            curves, d / "pr.png", ap=ap, title=f"{run_id}\nchunk-level precision-recall"
        )
    if wanted("score_hist"):
        outside["score_hist"] = score_hist_png(
            data["label"],
            data["score"],
            d / "score_hist.png",
            threshold=None if threshold is None else float(threshold),
            title=f"{run_id} — score distribution ({eval_split})",
        )
    return outside


def comparison_png(
    tidy: Sequence[Mapping[str, Any]],
    out_path: str | Path,
    *,
    metrica: str,
    title: str | None = None,
) -> Path:
    """Compare adaptation methods: mean ± spread over seeds.

    Replaces the few-shot curve after ADR-031: the x axis is the method. Single-seed
    points have no error bar (missing ``std`` differs from zero ``std``) and the number of
    seeds is written above each bar. Left panel: bars from zero; right panel: zoom on the
    data range drawn with points, since a truncated axis under bars would exaggerate the
    gaps.

    :param tidy: rows of :meth:`~evd.eval.aggregate.SeedAggregator.tidy`.
    :param out_path: PNG destination.
    :param metrica: metric to compare (one figure per metric).
    :param title: figure title.
    :return: the written path.
    :raises ValueError: if no row carries that metric.
    """
    choices = [r for r in tidy if r.get("metric") == metrica]
    if not choices:
        raise ValueError(f"no row for metric {metrica!r}")
    plt = _agg()
    choices = sorted(choices, key=lambda r: float(r["mean"]))
    names = [str(r.get("adapt", "?")) for r in choices]
    means = [float(r["mean"]) for r in choices]
    # `None`, not 0.0: a zero error bar would claim perfect reproducibility.
    errors = [None if r.get("std") is None else float(r["std"]) for r in choices]
    x = np.arange(len(choices))

    fig, (ax, azoom) = plt.subplots(
        1,
        2,
        figsize=(2.1 * len(choices) + 3.6, 4.2),
        dpi=140,
        gridspec_kw={"width_ratios": (1.35, 1.0)},
    )
    ax.bar(
        x,
        means,
        yerr=[0.0 if e is None else e for e in errors],
        capsize=4,
        color=COLOR["test"],
        width=0.6,
        error_kw={"lw": 1.2, "ecolor": "0.25"},
    )
    for xi, (m, e, r) in enumerate(zip(means, errors, choices, strict=True)):
        n = int(r.get("n", 0))
        note = f"{m:.4f}\nn={n}" + ("" if e is not None else "\n(no σ)")
        ax.text(xi, m, note, ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x, names, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel(metrica)
    ax.set_ylim(0, max(1.0, max(means) * 1.18))
    ax.set_title("full scale", fontsize=9)
    ax.grid(alpha=0.2, lw=0.5, axis="y")

    # Zoom panel: points with error bars have no area to read, so an axis not starting at
    # zero is legitimate.
    azoom.errorbar(
        x,
        means,
        yerr=[0.0 if e is None else e for e in errors],
        fmt="o",
        ms=6,
        capsize=4,
        lw=0,
        elinewidth=1.2,
        color=COLOR["test"],
        ecolor="0.25",
    )
    for xi, (m, e) in enumerate(zip(means, errors, strict=True)):
        azoom.annotate(
            f"{m:.4f}",
            (xi, m + (e or 0.0)),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=7.5,
        )
    extremes = [m + (e or 0.0) for m, e in zip(means, errors, strict=True)]
    extremes += [m - (e or 0.0) for m, e in zip(means, errors, strict=True)]
    bottom, top = min(extremes), max(extremes)
    margin = max((top - bottom) * 0.30, 0.004)
    azoom.set_ylim(bottom - margin, top + margin)
    azoom.set_xlim(-0.6, len(choices) - 0.4)
    azoom.set_xticks(x, names, fontsize=8, rotation=15, ha="right")
    azoom.set_title("zoom — points, not bars (axis does not start at 0)", fontsize=8.5)
    azoom.grid(alpha=0.2, lw=0.5, axis="y")

    fig.suptitle(title or f"{metrica} by adaptation method (mean ± sd over seeds)", fontsize=10)
    return _save(fig, out_path)
