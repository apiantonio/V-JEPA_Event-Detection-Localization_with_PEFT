"""Figures of the deployment study, companion of :mod:`evd.eval.deployment`.

They compare methods and pipeline stages, so they do not belong to
:mod:`evd.eval.report` (single-cell figures); composition stays in
:mod:`evd.eval.deployment`, which does not import matplotlib.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from evd.eval._figure import agg, save

__all__ = [
    "capacita_png",
    "latenza_png",
    "macchine_png",
    "pareto_png",
    "scaling_png",
    "tail_png",
]

#: Colors per pipeline stage; every bar also carries its value as text, so the figure
#: stays readable in grayscale.
COLOR = {
    "decoder": "tab:orange",
    "modello": "tab:blue",
    "algoritmica": "0.62",
    "fusa": "tab:purple",
    "peggiore": "tab:red",
}


def _names(righe: Sequence[Any]) -> list[str]:
    return [str(r.variante) for r in righe]


def latenza_png(
    righe: Sequence[Any],
    out_path: str | Path,
    *,
    hop_s: float = 4.0,
    title: str | None = None,
) -> Path:
    """Show where time goes: compute first, then the actual alarm delay.

    The left panel splits per-window compute latency into decoder and model, with the hop
    as real-time budget; the right panel splits the alarm delay into algorithmic (grid)
    and compute parts. The panels use different units (ms and s) on purpose.

    :param righe: rows from :func:`evd.eval.deployment.compose`.
    :param out_path: PNG destination.
    :param hop_s: grid step, drawn as the real-time budget.
    :param title: figure title.
    :return: the written path.
    :raises ValueError: if there are no rows.
    """
    if not righe:
        raise ValueError("no deployment rows to draw")
    plt = agg()
    y = np.arange(len(righe))
    fig, (ax, aev) = plt.subplots(1, 2, figsize=(12.2, 0.62 * len(righe) + 2.8), dpi=140)

    dec = np.array([float(r.latenza_decoder_ms) for r in righe])
    mod = np.array([float(r.latenza_modello_ms) for r in righe])
    ax.barh(y, dec, color=COLOR["decoder"], label="decoder (CPU)", height=0.6)
    ax.barh(y, mod, left=dec, color=COLOR["modello"], label="model (GPU)", height=0.6)
    for yi, (d, m) in enumerate(zip(dec, mod, strict=True)):
        ax.text((d + m) * 1.06, yi, f"{d + m:.0f} ms", va="center", fontsize=7.5)
    ax.axvline(hop_s * 1000.0, color=COLOR["peggiore"], lw=1.2, ls="--")
    ax.text(
        hop_s * 1000.0 * 0.94,
        -0.5,
        f"real-time budget = hop {hop_s:g} s  ",
        color=COLOR["peggiore"],
        fontsize=7.5,
        va="center",
        ha="right",
    )
    ax.set_xscale("log")
    ax.set_xlim(min(dec.min(), mod.min()) * 0.55, hop_s * 1000.0 * 2.4)
    ax.set_yticks(y, _names(righe), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("compute per window (ms, log scale)")
    ax.set_title("compute cost per window", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.2, lw=0.5, axis="x")

    algo = np.array(
        [
            (
                float(r.latenza_allarme_mediana_s) - float(r.latenza_risposta_ms) / 1000.0
                if r.latenza_allarme_mediana_s is not None
                else np.nan
            )
            for r in righe
        ]
    )
    calc = np.array([float(r.latenza_risposta_ms) / 1000.0 for r in righe])
    if not np.all(np.isnan(algo)):
        aev.barh(y, algo, color=COLOR["algoritmica"], label="grid (video seconds)", height=0.6)
        aev.barh(y, calc, left=algo, color=COLOR["modello"], label="compute", height=0.6)
        total = algo + calc
        for yi, (t, r) in enumerate(zip(total, righe, strict=True)):
            if np.isnan(t):
                continue
            quota = r.quota_calcolo_pct or 0.0
            aev.text(t * 1.02, yi, f"{t:.2f} s   compute {quota:.1f}%", va="center", fontsize=7.5)
        aev.set_xlim(0, float(np.nanmax(total)) * 1.5)
        aev.legend(fontsize=7.5, loc="lower right")
    aev.set_yticks(y, [""] * len(righe))
    aev.invert_yaxis()
    aev.set_xlabel("median delay from event onset to alarm (s)")
    aev.set_title("end-to-end alarm delay", fontsize=9.5)
    aev.grid(alpha=0.2, lw=0.5, axis="x")

    fig.suptitle(title or "Deployment latency budget", fontsize=10.5)
    return save(fig, out_path)


def capacita_png(
    righe: Sequence[Any],
    out_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Show how many cameras a GPU sustains, and which stage is the constraint.

    Model and decoder bars side by side, with the system value (their minimum) in bold.
    It is an arithmetic upper bound without contention, scheduling or network I/O.

    :param righe: rows from :func:`evd.eval.deployment.compose`.
    :param out_path: PNG destination.
    :param title: figure title.
    :return: the written path.
    :raises ValueError: if there are no rows.
    """
    if not righe:
        raise ValueError("no deployment rows to draw")
    plt = agg()
    x = np.arange(len(righe))
    fig, ax = plt.subplots(figsize=(1.7 * len(righe) + 3.4, 4.4), dpi=140)
    gpu = [float(r.capacita.stream_gpu) for r in righe]
    dec = [float(r.capacita.stream_decoder) for r in righe]
    ax.bar(x - 0.19, gpu, width=0.36, color=COLOR["modello"], label="model (GPU) ceiling")
    ax.bar(x + 0.19, dec, width=0.36, color=COLOR["decoder"], label="decoder (CPU) ceiling")
    for xi, r in enumerate(righe):
        ax.annotate(
            f"{r.capacita.stream_sistema:.0f}",
            (xi, min(gpu[xi], dec[xi])),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xticks(x, _names(righe), fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("concurrent camera streams")
    ax.set_title(title or "Streams per GPU — bold = system limit (min of the two)", fontsize=10)
    ax.text(
        0.5,
        -0.32,
        "arithmetic upper bound: no scheduling, no contention, no network I/O",
        transform=ax.transAxes,
        ha="center",
        fontsize=7.5,
        color="0.35",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2, lw=0.5, axis="y")
    return save(fig, out_path)


def pareto_png(
    righe: Sequence[Any],
    out_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Plot accuracy against latency, with the shipped megabytes as bubble area.

    The area is proportional to the square root of the megabytes, since viewers read
    the diameter.

    :param righe: rows from :func:`evd.eval.deployment.compose`; rows without accuracy
        are skipped.
    :param out_path: PNG destination.
    :param title: figure title.
    :return: the written path.
    :raises ValueError: if no row carries an accuracy value.
    """
    choices = [r for r in righe if r.chunk_auc is not None]
    if not choices:
        raise ValueError("no row carries chunk_auc: the Pareto plot would have no vertical axis")
    plt = agg()
    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=140)
    mb = np.array([max(float(r.mb_aggiornamento_fp32), 0.0) for r in choices])
    areas = 90.0 + 700.0 * np.sqrt(mb / max(mb.max(), 1e-9))

    # Three variants lie within half a millisecond of each other, so their labels are
    # stacked below the group with leader lines instead of moving the points.
    latencies = [float(r.latenza_modello_ms) for r in choices]
    aucs = [float(r.chunk_auc) for r in choices]
    width_x = max(max(latencies) - min(latencies), 1e-9)
    width_y = max(max(aucs) - min(aucs), 1e-9)
    # Neighbours on both axes: points with equal latency but distant accuracy do not compete.
    _remains = sorted(range(len(choices)), key=lambda i: -aucs[i])
    groups: list[list[int]] = []
    for index in _remains:
        for group in groups:
            if (
                abs(latencies[index] - latencies[group[0]]) < 0.03 * width_x + 1.0
                and abs(aucs[index] - aucs[group[0]]) < 0.25 * width_y
            ):
                group.append(index)
                break
        else:
            groups.append([index])

    label_position: dict[int, tuple[float, float] | None] = {}
    lowest = min(aucs)
    for group in groups:
        if len(group) == 1:
            label_position[group[0]] = None
            # A single label also takes space below its point; keep it off the axis ticks.
            lowest = min(lowest, aucs[group[0]] - 0.16 * width_y)
            continue
        x_group = sum(latencies[i] for i in group) / len(group)
        base = min(aucs[i] for i in group) - 0.11 * width_y
        for k, member in enumerate(group):
            y = base - k * 0.085 * width_y
            label_position[member] = (x_group, y)
            lowest = min(lowest, y - 0.06 * width_y)

    for i, (r, area) in enumerate(zip(choices, areas, strict=True)):
        ax.scatter(
            r.latenza_modello_ms,
            r.chunk_auc,
            s=area,
            alpha=0.40,
            color=COLOR["fusa"] if r.fusa else COLOR["modello"],
            edgecolor="0.25",
            lw=0.8,
            zorder=2,
        )
        if r.chunk_auc_std:
            ax.errorbar(
                r.latenza_modello_ms,
                r.chunk_auc,
                yerr=r.chunk_auc_std,
                fmt="none",
                ecolor="0.35",
                elinewidth=1.0,
                capsize=3,
                zorder=3,
            )
        label = f"{r.variante} · {r.mb_aggiornamento_fp32:,.0f} MB".replace(",", " ")
        where = label_position.get(i)
        if where is None:
            ax.annotate(
                label,
                (r.latenza_modello_ms, r.chunk_auc),
                textcoords="offset points",
                xytext=(0, -26),
                ha="center",
                va="top",
                fontsize=7.5,
                zorder=4,
            )
        else:
            ax.annotate(
                label,
                (r.latenza_modello_ms, r.chunk_auc),
                xytext=where,
                textcoords="data",
                ha="center",
                va="center",
                fontsize=7.5,
                zorder=4,
                arrowprops={
                    "arrowstyle": "-",
                    "lw": 0.6,
                    "color": "0.6",
                    "shrinkA": 3,
                    "shrinkB": 3,
                },
            )
    ax.set_xlabel("inference latency per window, batch 1 (ms)")
    ax.set_ylabel("chunk-AUC (mean ± sd over seeds)")
    ax.set_title(title or "Accuracy vs latency — bubble area ∝ √(MB per update)", fontsize=10)
    ax.grid(alpha=0.2, lw=0.5)
    ax.set_xlim(min(latencies) - 0.14 * width_x, max(latencies) + 0.10 * width_x)
    ax.set_ylim(lowest - 0.10 * width_y, max(aucs) + 0.16 * width_y)
    ax.annotate(
        "← better (more accurate, faster)",
        (0.015, 0.985),
        xycoords="axes fraction",
        fontsize=8,
        color="0.4",
        ha="left",
        va="top",
    )
    return save(fig, out_path)


def scaling_png(
    efficiency_rows: Sequence[Mapping[str, Any]],
    out_path: str | Path,
    *,
    hop_s: float = 4.0,
    title: str | None = None,
) -> Path:
    """Plot throughput and latency against batch size.

    Left: windows per second, with a secondary axis in cameras. Right: forward latency,
    which grows with the batch.

    :param efficiency_rows: raw profile rows over several batch sizes.
    :param out_path: PNG destination.
    :param hop_s: grid step, for the camera axis.
    :param title: figure title.
    :return: the written path.
    :raises ValueError: if the profile has a single batch size.
    """
    batches = sorted({int(r.get("batch", 1)) for r in efficiency_rows})
    if len(batches) < 2:
        raise ValueError(
            f"scaling with a single batch ({batches}): the schema-2 profile sweep is required"
        )
    plt = agg()
    fig, (ax, alat) = plt.subplots(1, 2, figsize=(11.6, 4.4), dpi=140)
    for variant in sorted({str(r["variante"]) for r in efficiency_rows}):
        points = sorted(
            (r for r in efficiency_rows if str(r["variante"]) == variant),
            key=lambda r: int(r["batch"]),
        )
        bs = [int(r["batch"]) for r in points]
        ax.plot(
            bs,
            [float(r["throughput_windows_s"]) for r in points],
            marker="o",
            ms=4,
            lw=1.4,
            label=variant,
        )
        alat.plot(
            bs,
            [float(r["latency_ms_mean"]) for r in points],
            marker="o",
            ms=4,
            lw=1.4,
            label=variant,
        )
    for a, label, title_text in (
        (ax, "windows / s", "throughput saturation"),
        (alat, "forward latency (ms)", "queueing cost: latency grows with batch"),
    ):
        a.set_xscale("log", base=2)
        a.set_xticks(batches, [str(b) for b in batches])
        a.set_xlabel("batch size")
        a.set_ylabel(label)
        a.set_title(title_text, fontsize=9.5)
        a.grid(alpha=0.2, lw=0.5)
        a.legend(fontsize=7.5)
    second = ax.secondary_yaxis("right", functions=(lambda v: v * hop_s, lambda v: v / hop_s))
    second.set_ylabel(f"camera streams (hop {hop_s:g} s)")
    fig.suptitle(title or "Batch scaling — throughput bought with latency", fontsize=10.5)
    return save(fig, out_path)


def tail_png(
    efficiency_rows: Sequence[Mapping[str, Any]],
    out_path: str | Path,
    *,
    batch: int = 1,
    title: str | None = None,
) -> Path:
    """Plot the latency tail: p50, p90, p99 and the observed worst case.

    Requires the schema-2 profile (schema 1 only stored mean and deviation).

    :param efficiency_rows: profile rows.
    :param out_path: PNG destination.
    :param batch: batch to draw; 1 is where latency is a response time.
    :param title: figure title.
    :return: the written path.
    :raises ValueError: if no row carries percentiles.
    """
    choices = [
        r
        for r in efficiency_rows
        if int(r.get("batch", 1)) == batch and r.get("latency_ms_p99") is not None
    ]
    if not choices:
        raise ValueError(
            f"no row at batch {batch} carries percentiles: the schema-2 profile is required"
        )
    plt = agg()
    y = np.arange(len(choices))
    fig, ax = plt.subplots(figsize=(9.0, 0.56 * len(choices) + 2.6), dpi=140)
    for yi, r in enumerate(choices):
        p50 = float(r["latency_ms_p50"])
        ax.plot(
            [p50, float(r["latency_ms_p99"])],
            [yi, yi],
            lw=7,
            color=COLOR["modello"],
            alpha=0.28,
            solid_capstyle="butt",
        )
        ax.plot(
            [p50, float(r["latency_ms_p90"])],
            [yi, yi],
            lw=7,
            color=COLOR["modello"],
            alpha=0.6,
            solid_capstyle="butt",
        )
        ax.scatter([p50], [yi], s=40, color=COLOR["modello"], zorder=3)
        if r.get("latency_ms_max"):
            ax.scatter(
                [float(r["latency_ms_max"])],
                [yi],
                marker="|",
                s=110,
                color=COLOR["peggiore"],
                zorder=3,
            )
        ax.text(
            float(r["latency_ms_p99"]) * 1.005,
            yi,
            f"  p50 {p50:.1f} → p99 +{100.0 * (float(r['latency_ms_p99']) / p50 - 1.0):.2f}%",
            va="center",
            fontsize=7.5,
        )
    ax.set_yticks(y, [str(r["variante"]) for r in choices], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"latency per forward at batch {batch} (ms) — bar: p50 → p90 → p99, | = max")
    ax.set_title(title or "Latency tail: how far the worst case sits from the typical", fontsize=10)
    ax.grid(alpha=0.2, lw=0.5, axis="x")
    return save(fig, out_path)


def macchine_png(
    confronto: Any,
    out_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Show the two machines side by side: overheads and latency ratio.

    Left: per-method overhead on both GPUs (comparable across machines). Right: latency
    ratio per variant with its median. Absolute latencies are deliberately not plotted
    side by side.

    :param confronto: result of :func:`evd.eval.deployment.compare_machines`.
    :param out_path: PNG destination.
    :param title: figure title.
    :return: the written path.
    :raises ValueError: if the comparison has no overhead at batch 1.
    """
    a, b = confronto.macchine[0], confronto.macchine[1]
    entry_list = [r for r in confronto.overhead if int(r["batch"]) == 1]
    if not entry_list:
        raise ValueError("no overhead at batch 1: the comparison has no main panel")
    plt = agg()
    fig, (ax, arap) = plt.subplots(
        1, 2, figsize=(12.0, 4.6), dpi=140, gridspec_kw={"width_ratios": (1.25, 1.0)}
    )

    x = np.arange(len(entry_list))
    va = [float(r[f"overhead_pct_{a}"]) for r in entry_list]
    vb = [float(r[f"overhead_pct_{b}"]) for r in entry_list]
    ax.bar(x - 0.19, va, width=0.36, color=COLOR["modello"], label=a)
    ax.bar(x + 0.19, vb, width=0.36, color=COLOR["decoder"], label=b)
    for xi, (pa, pb) in enumerate(zip(va, vb, strict=True)):
        for dx, v in ((-0.19, pa), (0.19, pb)):
            ax.annotate(
                f"{v:+.1f}%",
                (xi + dx, v),
                textcoords="offset points",
                xytext=(0, 4 if v >= 0 else -12),
                ha="center",
                fontsize=7.5,
            )
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xticks(x, [str(r["variante"]) for r in entry_list], fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("inference overhead vs frozen (%)")
    state = (
        "same ranking on both GPUs — the C7 verdict is hardware-independent"
        if confronto.ordinamento_conservato
        else "RANKING DIFFERS — the C7 verdict depends on the GPU"
    )
    ax.set_title(f"overhead, batch 1 — {state}", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2, lw=0.5, axis="y")

    ratios = [r for r in confronto.rapporti if r["rapporto"]]
    y = np.arange(len(ratios))
    arap.barh(
        y,
        [float(r["rapporto"]) for r in ratios],
        color=COLOR["algoritmica"],
        height=0.6,
    )
    if confronto.rapporto_mediano:
        arap.axvline(
            float(confronto.rapporto_mediano),
            color=COLOR["peggiore"],
            ls="--",
            lw=1.2,
            label=f"median {confronto.rapporto_mediano:.3f}×",
        )
        arap.legend(fontsize=8, loc="lower right")
    for yi, r in enumerate(ratios):
        arap.text(
            float(r["rapporto"]) * 1.01,
            yi,
            f" {r['rapporto']:.3f}×",
            va="center",
            fontsize=7.5,
        )
    arap.set_yticks(y, [f"{r['variante']} b{r['batch']}" for r in ratios], fontsize=7.5)
    arap.invert_yaxis()
    arap.set_xlabel(f"latency ratio {a} / {b}")
    arap.set_title("how much the machine changes the number", fontsize=9)
    arap.grid(alpha=0.2, lw=0.5, axis="x")

    fig.suptitle(title or f"{a} vs {b} — only ratios and overheads are comparable", fontsize=10.5)
    return save(fig, out_path)
