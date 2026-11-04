"""System-level deployment cost composed from already measured numbers.

The module measures nothing: it combines the model cost
(``efficiency_inference.json``), the decoder cost (``throughput.json``) and the
algorithmic event latency of the E2/E3 cells (``detection_latency`` in
``metrics.json``), and every row keeps the provenance of its inputs.

Three regimes are reported: online with one camera (does the system keep up with
the grid ``hop``?), online with many cameras (how many sources per GPU?) and offline
archive processing (compute time per hour of video). Decoder and model run on
different devices, so in steady state the time per window is the maximum of the two,
while the latency of a single response is their sum. The two terms come from
different machines and exclude network acquisition and alarm dispatch; the model
remains the bottleneck by a factor of at least 3.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from evd.utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "COMPARABLE_FIELDS",
    "Capacity",
    "MachineComparison",
    "Geometry",
    "DeploymentRow",
    "Response",
    "Stage",
    "compose",
    "compare_machines",
    "comparison_markdown",
    "fattore_realtime",
    "markdown_table",
    "check_wallclock",
]

#: Bytes per scalar, to report the shipped model size per distribution dtype.
BYTES_PER_SCALAR = {"fp32": 4, "bf16": 2, "fp16": 2, "int8": 1}


def fattore_realtime(latency_s: float, *, hop_s: float) -> float | None:
    """Return how many times faster than real time a per-window latency is.

    In streaming each window covers ``hop_s`` seconds of new video (not ``span_s``,
    windows overlap), so values above 1.0 keep up.

    :param latency_s: compute seconds per window.
    :param hop_s: grid step in seconds of video.
    :return: the ratio, or ``None`` if the latency is not positive.
    :raises ValueError: if ``hop_s`` is not positive.
    """
    if hop_s <= 0:
        raise ValueError(f"hop_s must be positive, not {hop_s}")
    return hop_s / latency_s if latency_s > 0 else None


@dataclass(frozen=True)
class Geometry:
    """Protocol geometry, to translate windows into seconds of video.

    :param hop_s: grid step (4 s in ``edp_v2``).
    :param span_s: duration covered by a window (7.75 s).
    :param n_chunk_valutazione: windows of the test partition (9116).
    :param ore_video_valutazione: hours of video of that partition.
    """

    hop_s: float = 4.0
    span_s: float = 7.75
    n_chunk_valutazione: int = 9116
    ore_video_valutazione: float | None = None

    def __post_init__(self) -> None:
        if self.hop_s <= 0 or self.span_s <= 0:
            raise ValueError(f"non-positive geometry: hop_s={self.hop_s}, span_s={self.span_s}")
        if self.span_s < self.hop_s:
            # Allowed, but the grid would skip video and change the meaning of every row.
            log.warning(
                "span_s (%.2f) < hop_s (%.2f): the grid leaves video unobserved",
                self.span_s,
                self.hop_s,
            )


@dataclass(frozen=True)
class Stage:
    """Pipeline stage with the provenance of its number.

    :param nome: ``"decoder"`` or ``"modello"``.
    :param ms_per_finestra: cost per window, in milliseconds.
    :param provenienza: run id or job the number comes from (stages come from different
        runs on different machines).
    :param dettaglio: how the number was obtained (workers, batch, augmentation).
    """

    nome: str
    ms_per_finestra: float
    provenienza: str
    dettaglio: str = ""

    @property
    def finestre_s(self) -> float:
        """:return: windows per second of this stage alone."""
        return 1000.0 / self.ms_per_finestra if self.ms_per_finestra > 0 else float("inf")


@dataclass(frozen=True)
class Response:
    """Algorithmic latency, independent of compute speed.

    Read from ``metrics.json`` and measured in seconds of video: the distance between the
    annotated onset and the end of the first chunk above threshold that contains it
    (:func:`evd.eval.metrics.detection_latency`). The grid fixes it.

    :param mediana_s: median over events with measurable latency.
    :param p90_s: 90th percentile over the same events.
    :param n_eventi: evaluated events.
    :param n_senza_latenza: events that no chunk above threshold closes inside; it must
        be reported, since a median over measurable events improves when more are missed.
    """

    mediana_s: float
    p90_s: float
    n_eventi: int
    n_senza_latenza: int

    @property
    def copertura(self) -> float:
        """:return: fraction of events with a defined latency, in [0, 1]."""
        return (self.n_eventi - self.n_senza_latenza) / self.n_eventi if self.n_eventi else 0.0


@dataclass(frozen=True)
class Capacity:
    """How many sources a machine sustains, and which stage limits it.

    :param stream_gpu: cameras sustained by the model.
    :param stream_decoder: cameras sustained by the decoder with the measured workers.
    :param stream_sistema: minimum of the two.
    :param vincolo: limiting stage.
    """

    stream_gpu: float
    stream_decoder: float
    stream_sistema: float
    vincolo: str


@dataclass(frozen=True)
class DeploymentRow:
    """One variant read as a system rather than as a model.

    Fields are grouped by question: does it keep up, how many cameras, how late is the
    alarm, what does distribution cost.
    """

    variante: str
    family: str
    method: str
    fusa: bool

    # Compute, per window
    latenza_modello_ms: float
    latenza_decoder_ms: float
    latenza_risposta_ms: float
    """Sum of the two stages: time between the last frame of a window and its score
    (conservative, decoding on demand)."""
    latenza_risposta_amortizzata_ms: float
    """Model only: a system with a sliding buffer that decodes each frame once."""
    ms_per_finestra_stazionario: float
    """Maximum of the two stages (they overlap on different devices); it drives
    throughput and capacity."""
    stadio_dominante: str

    # Does it keep up?
    fattore_realtime: float | None
    fattore_realtime_risposta: float | None
    margine_per_finestra_ms: float
    """Slack in milliseconds per window before the queue grows; negative means the
    system does not keep up."""

    # How many cameras?
    capacita: Capacity

    # How late is the alarm?
    latenza_allarme_mediana_s: float | None
    latenza_allarme_p90_s: float | None
    quota_calcolo_pct: float | None
    """Fraction of the total delay due to compute; when small, a faster model does not
    shorten the system response."""
    copertura_eventi: float | None

    # Archive
    finestre_s: float
    s_per_ora_di_video: float
    rapporto_elaborazione: float
    """Compute seconds per second of video; below 1 the system is faster than real
    time."""
    minuti_per_partizione_test: float

    # Resources
    vram_gb: float | None
    energia_j_per_finestra: float | None
    wh_per_ora_di_video: float | None
    potenza_w_media: float | None

    # Distribution
    params_delta: int
    """Backbone-side delta as counted by E5b: 0 for frozen, ~14 M for an unmerged
    adapter, ~305 M for a merged one."""
    params_testa: int
    """Head parameters: the attentive head is trained in every cell and is not in the
    base checkpoint, so it always ships with the delta."""
    params_aggiornamento: int
    """Parameters shipped to a device that already has the base backbone: delta + head."""
    params_primo_deploy: int
    """Parameters shipped to an empty device: backbone (merged or base) + separate
    adapter + head. Here the merged variant is the smallest."""
    mb_aggiornamento_fp32: float
    mb_aggiornamento_bf16: float
    mb_primo_deploy_fp32: float

    # Quality
    chunk_auc: float | None
    chunk_auc_std: float | None
    chunk_ap: float | None
    event_f1: float | None
    n_semi: int

    # Latency tail, when the profile reports it (schema 2)
    latenza_p95_ms: float | None = None
    latenza_p99_ms: float | None = None
    jitter_pct: float | None = None
    """``(p99 - p50) / p50``: how far the worst case is from the typical one."""

    batch: int = 1
    note: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        """:return: the row as a dictionary, with ``capacita`` flattened."""
        outside = asdict(self)
        cap = outside.pop("capacita")
        outside.update({f"capacita_{k}": v for k, v in cap.items()})
        outside["note"] = list(self.note)
        return outside


def _mean_spread(valori: Sequence[float]) -> tuple[float | None, float | None]:
    """Return mean and sample standard deviation, ``None`` where undefined.

    With a single value the deviation is ``None``, not ``0``.
    """
    clean = [float(v) for v in valori if v is not None and not math.isnan(float(v))]
    if not clean:
        return None, None
    if len(clean) == 1:
        return clean[0], None
    return statistics.fmean(clean), statistics.stdev(clean)


def _bytes_mb(n_scalars: int, dtype: str) -> float:
    return n_scalars * BYTES_PER_SCALAR[dtype] / 1024**2


def compose(
    efficiency_rows: Iterable[Mapping[str, Any]],
    *,
    decoder: Stage,
    geometria: Geometry,
    responses: Mapping[str, Response] | None = None,
    quality: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
    batch: int = 1,
    distribution_dtype: str = "fp32",
) -> list[DeploymentRow]:
    """Compose the E5b rows with the decoder, the event response and the quality.

    :param efficiency_rows: rows of ``efficiency_inference.json``; only those at the
        requested batch are used.
    :param decoder: CPU stage with its provenance.
    :param geometria: hop, span and size of the test partition.
    :param responses: per variant, the algorithmic latency read from ``metrics.json``.
        Merged variants inherit the response of their unmerged variant, since the merge
        is verified to produce the same scores.
    :param quality: per variant, per-seed accuracy values (``chunk_auc``, ``chunk_ap``,
        ``event_f1``).
    :param batch: batch to read; 1 is the online deployment case.
    :param distribution_dtype: dtype used to ship the model, for the MB columns.
    :return: one row per variant, in input order.
    :raises ValueError: on an unknown dtype or if no row has the requested batch.
    """
    if distribution_dtype not in BYTES_PER_SCALAR:
        raise ValueError(
            f"unknown distribution dtype: {distribution_dtype!r} "
            f"(allowed {sorted(BYTES_PER_SCALAR)})"
        )
    choices = [r for r in efficiency_rows if int(r.get("batch", 1)) == batch]
    if not choices:
        raise ValueError(
            f"no efficiency row at batch {batch}: the deployment table would be empty. "
            f"Check that the E5b profile measured that batch."
        )
    responses = dict(responses or {})
    quality = dict(quality or {})

    outside: list[DeploymentRow] = []
    for row in choices:
        variant = str(row["variante"])
        note: list[str] = []

        # At batch B a forward produces B windows: the cost per window is latency / B, while
        # the response time of a window is the whole latency.
        forward_latency_ms = float(row["latency_ms_mean"])
        model_ms_per_window = forward_latency_ms / batch

        steady = max(model_ms_per_window, decoder.ms_per_finestra)
        dominant = "modello" if model_ms_per_window >= decoder.ms_per_finestra else "decoder"
        response_ms = forward_latency_ms + decoder.ms_per_finestra

        windows_s = 1000.0 / steady
        s_hour = (3600.0 / geometria.hop_s) / windows_s

        # Capacity: every source requests one window every `hop_s` seconds.
        stream_gpu = (1000.0 / model_ms_per_window) * geometria.hop_s
        stream_dec = decoder.finestre_s * geometria.hop_s
        capacity = Capacity(
            stream_gpu=stream_gpu,
            stream_decoder=stream_dec,
            stream_sistema=min(stream_gpu, stream_dec),
            vincolo="modello" if stream_gpu <= stream_dec else "decoder",
        )

        response = responses.get(variant)
        if response is None and variant.endswith("-fusa"):
            response = responses.get(variant[: -len("-fusa")])
            if response is not None:
                note.append(
                    "algorithmic latency inherited from the unmerged variant: the merge is "
                    "verified faithful, so the alarm instants are the same"
                )
        median_alarm = alarm_p90 = quota = coverage = None
        if response is not None:
            median_alarm = response.mediana_s + response_ms / 1000.0
            alarm_p90 = response.p90_s + response_ms / 1000.0
            quota = 100.0 * (response_ms / 1000.0) / median_alarm
            coverage = response.copertura

        q = quality.get(variant) or quality.get(variant.replace("-fusa", "")) or {}
        auc, auc_std = _mean_spread(q.get("chunk_auc", ()))
        ap, _ = _mean_spread(q.get("chunk_ap", ()))
        ev, _ = _mean_spread(q.get("event_f1", ()))
        n_seeds = len(q.get("chunk_auc", ()) or ())

        vram = row.get("peak_vram_bytes")
        delta = int(row.get("params_da_spedire", 0))
        head = int(row.get("params_head", 0))
        # `params_backbone` already includes the adapter when attached and the merged backbone
        # otherwise; adding the delta would count it twice.
        first = int(row.get("params_backbone", 0)) + head
        margin = geometria.hop_s * 1000.0 - response_ms
        if margin <= 0:
            note.append("does NOT keep up: the response exceeds the grid hop")

        p50 = row.get("latency_ms_p50")
        p99 = row.get("latency_ms_p99")
        jitter = 100.0 * (float(p99) / float(p50) - 1.0) if p50 and p99 else None

        outside.append(
            DeploymentRow(
                variante=variant,
                family=str(row.get("family", "")),
                method=str(row.get("method", "")),
                fusa=bool(row.get("fusa", False)),
                latenza_modello_ms=forward_latency_ms,
                latenza_decoder_ms=decoder.ms_per_finestra,
                latenza_risposta_ms=response_ms,
                latenza_risposta_amortizzata_ms=forward_latency_ms,
                ms_per_finestra_stazionario=steady,
                stadio_dominante=dominant,
                fattore_realtime=fattore_realtime(steady / 1000.0, hop_s=geometria.hop_s),
                fattore_realtime_risposta=fattore_realtime(
                    response_ms / 1000.0, hop_s=geometria.hop_s
                ),
                margine_per_finestra_ms=margin,
                capacita=capacity,
                latenza_allarme_mediana_s=median_alarm,
                latenza_allarme_p90_s=alarm_p90,
                quota_calcolo_pct=quota,
                copertura_eventi=coverage,
                finestre_s=windows_s,
                s_per_ora_di_video=s_hour,
                rapporto_elaborazione=s_hour / 3600.0,
                minuti_per_partizione_test=geometria.n_chunk_valutazione / windows_s / 60.0,
                vram_gb=float(vram) / 1024**3 if vram else None,
                energia_j_per_finestra=row.get("energia_j_per_finestra"),
                wh_per_ora_di_video=(
                    row["energia_j_per_finestra"] * (3600.0 / geometria.hop_s) / 3600.0
                    if row.get("energia_j_per_finestra")
                    else None
                ),
                potenza_w_media=row.get("potenza_w_media"),
                params_delta=delta,
                params_testa=head,
                params_aggiornamento=delta + head,
                params_primo_deploy=first,
                mb_aggiornamento_fp32=_bytes_mb(delta + head, "fp32"),
                mb_aggiornamento_bf16=_bytes_mb(delta + head, "bf16"),
                mb_primo_deploy_fp32=_bytes_mb(first, "fp32"),
                chunk_auc=auc,
                chunk_auc_std=auc_std,
                chunk_ap=ap,
                event_f1=ev,
                n_semi=n_seeds,
                latenza_p95_ms=row.get("latency_ms_p95"),
                latenza_p99_ms=row.get("latency_ms_p99"),
                jitter_pct=jitter,
                batch=batch,
                note=tuple(note),
            )
        )
    return outside


def check_wallclock(
    efficiency_rows: Iterable[Mapping[str, Any]],
    observed: Mapping[str, float],
    *,
    n_chunk: int,
    batch: int = 2,
) -> dict[str, Any]:
    """Compare the times predicted from the profile with observed test passes.

    Fits ``observed = a + b * predicted`` by least squares: ``a`` is the fixed overhead per
    window (non-overlapped decoding, I/O, collate) and ``1/b`` the hardware ratio between
    the two machines. With three points and two parameters this is not a statistical
    test: ordering and residuals matter. Passes must come from the same host.

    :param efficiency_rows: profile rows.
    :param observed: per variant, wall-clock minutes of the test pass.
    :param n_chunk: windows evaluated in that pass (9116 in ``edp_v2``).
    :param batch: batch used by the passes.
    :return: coefficients, residuals and the implied hardware ratio.
    :raises ValueError: with fewer than two comparable variants (a two-parameter line
        through one point is vacuous).
    """
    per_variant = {
        str(r["variante"]): r for r in efficiency_rows if int(r.get("batch", 1)) == batch
    }
    pairs = [
        (
            name,
            float(per_variant[name]["latency_ms_mean"]) / batch,
            observed[name] * 60_000.0 / n_chunk,
        )
        for name in observed
        if name in per_variant
    ]
    if len(pairs) < 2:
        raise ValueError(
            f"at least two common variants are needed for a non-vacuous comparison, "
            f"found {len(pairs)}: {[c[0] for c in pairs]}"
        )
    x = [c[1] for c in pairs]
    y = [c[2] for c in pairs]
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sxx = sum((xi - mx) ** 2 for xi in x)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y, strict=True))
    b = sxy / sxx if sxx else float("nan")
    a = my - b * mx
    residuals = [
        {
            "variante": name,
            "previsto_ms": xi,
            "osservato_ms": yi,
            "atteso_dal_fit_ms": a + b * xi,
            "residuo_pct": 100.0 * ((a + b * xi) / yi - 1.0) if yi else None,
        }
        for (name, xi, yi) in pairs
    ]
    return {
        "n_punti": len(pairs),
        "overhead_fisso_ms_per_finestra": a,
        "pendenza": b,
        "rapporto_hardware_implicito": 1.0 / b if b else None,
        "residuo_max_pct": max(abs(r["residuo_pct"] or 0.0) for r in residuals),
        "ordinamento_rispettato": [c[0] for c in sorted(pairs, key=lambda c: c[1])]
        == [c[0] for c in sorted(pairs, key=lambda c: c[2])],
        "residui": residuals,
    }


def _fmt(valore: Any, digits: int = 2, *, suffisso: str = "") -> str:
    if valore is None:
        return "—"
    if isinstance(valore, float) and math.isnan(valore):
        return "—"
    if isinstance(valore, (int,)) and not isinstance(valore, bool):
        return f"{valore:,}".replace(",", ".") + suffisso
    return f"{float(valore):,.{digits}f}".replace(",", "~").replace(".", ",").replace("~", ".") + (
        suffisso
    )


def markdown_table(righe: Sequence[DeploymentRow]) -> str:
    """Return the deployment study tables in Markdown.

    :param righe: composed rows.
    :return: the Markdown text.
    """
    if not righe:
        return "_no rows_\n"
    pieces: list[str] = []

    pieces.append("#### 1. Does it keep up? *(online, one camera)*\n")
    pieces.append(
        "| variant | model (ms) | decoder (ms) | response (ms) | hop margin (ms) "
        "| real-time | win/s |\n|---|---:|---:|---:|---:|---:|---:|"
    )
    for r in righe:
        pieces.append(
            f"| {r.variante} | {_fmt(r.latenza_modello_ms, 1)} | {_fmt(r.latenza_decoder_ms, 1)} "
            f"| {_fmt(r.latenza_risposta_ms, 1)} | {_fmt(r.margine_per_finestra_ms, 0)} "
            f"| {_fmt(r.fattore_realtime, 1)}× | {_fmt(r.finestre_s, 2)} |"
        )

    pieces.append("\n#### 2. How many cameras per GPU?\n")
    pieces.append("| variant | model | decoder | system | constraint |\n|---|---:|---:|---:|---|")
    for r in righe:
        pieces.append(
            f"| {r.variante} | {_fmt(r.capacita.stream_gpu, 1)} "
            f"| {_fmt(r.capacita.stream_decoder, 1)} | **{_fmt(r.capacita.stream_sistema, 1)}** "
            f"| {r.capacita.vincolo} |"
        )

    pieces.append("\n#### 3. How late is the alarm?\n")
    pieces.append(
        "| variant | algorithmic median (s) | + compute | total median (s) | total p90 (s) "
        "| compute share | events with latency |\n|---|---:|---:|---:|---:|---:|---:|"
    )
    for r in righe:
        pieces.append(
            f"| {r.variante} | "
            f"{_fmt((r.latenza_allarme_mediana_s or 0) - r.latenza_risposta_ms / 1000.0, 2)} "
            f"| {_fmt(r.latenza_risposta_ms / 1000.0, 3)} "
            f"| {_fmt(r.latenza_allarme_mediana_s, 2)} | {_fmt(r.latenza_allarme_p90_s, 2)} "
            f"| {_fmt(r.quota_calcolo_pct, 1, suffisso='%')} "
            f"| {_fmt(100.0 * r.copertura_eventi if r.copertura_eventi else None, 0, suffisso='%')} |"
        )

    pieces.append("\n#### 4. Archive: processing recorded video\n")
    pieces.append(
        "| variant | s per hour of video | x real time | test set (min) | VRAM (GB) "
        "| chunk-AUC | AP | event-F1 | seeds |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for r in righe:
        auc = _fmt(r.chunk_auc, 4)
        if r.chunk_auc_std is not None:
            auc += f" ± {_fmt(r.chunk_auc_std, 4)}"
        pieces.append(
            f"| {r.variante} | {_fmt(r.s_per_ora_di_video, 0)} "
            f"| {_fmt(1.0 / r.rapporto_elaborazione if r.rapporto_elaborazione else None, 1)}× "
            f"| {_fmt(r.minuti_per_partizione_test, 1)} | {_fmt(r.vram_gb, 2)} "
            f"| {auc} | {_fmt(r.chunk_ap, 4)} | {_fmt(r.event_f1, 3)} | {r.n_semi} |"
        )

    pieces.append("\n#### 5. Distribution cost\n")
    pieces.append(
        "The **head** (49.3 M) is trained in every cell and is not in the base checkpoint: "
        "it always ships with the delta. The two columns answer different questions, and "
        "the sign of the comparison **flips** between them.\n"
    )
    pieces.append(
        "| variant | delta | + head = **update** | MB (fp32) | MB (bf16) "
        "| **first deploy** | MB (fp32) |\n|---|---:|---:|---:|---:|---:|---:|"
    )
    for r in righe:
        pieces.append(
            f"| {r.variante} | {_fmt(r.params_delta)} | {_fmt(r.params_aggiornamento)} "
            f"| {_fmt(r.mb_aggiornamento_fp32, 1)} | {_fmt(r.mb_aggiornamento_bf16, 1)} "
            f"| {_fmt(r.params_primo_deploy)} | {_fmt(r.mb_primo_deploy_fp32, 1)} |"
        )
    return "\n".join(pieces) + "\n"


# Machine comparison (E5b.2, second leg)


@dataclass(frozen=True)
class MachineComparison:
    """The same profile on two GPUs, read only for what can be compared.

    Latencies of two machines are not compared directly: only per-variant ratios and the
    overhead columns (ratios within one run) are exposed.

    :param macchine: GPU names, in input order.
    :param rapporti: per (variant, batch), first-machine latency over second-machine latency.
    :param overhead: per (variant, batch), the overhead on each machine.
    :param ordinamento_conservato: whether the overhead ranking of the methods is the
        same on both machines.
    :param note: findings to read next to the numbers.
    """

    macchine: tuple[str, ...]
    run_id: tuple[str, ...]
    rapporti: list[dict[str, Any]]
    overhead: list[dict[str, Any]]
    ordinamento_conservato: bool
    rapporto_mediano: float | None
    rapporto_min: float | None
    rapporto_max: float | None
    note: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        outside = asdict(self)
        outside["note"] = list(self.note)
        return outside


#: Fields two profiles must share for the comparison to measure the machine only.
COMPARABLE_FIELDS = ("arch", "precision", "geometria")


def compare_machines(
    profiles: Sequence[Mapping[str, Any]], *, points_tolerance: float = 1.0
) -> MachineComparison:
    """Compare the same profile run on different GPUs.

    Two things are produced: the latency ratio per variant with its range (a constant
    ratio would allow "machine B is k times faster") and the overhead on each machine,
    plus whether the overhead ranking of the methods is preserved.

    :param profili: two or more ``efficiency_inference.json`` payloads.
    :param points_tolerance: percentage points below which two methods tie in the
        overhead ranking (merged variants all lie within +-0.3 points of zero).
    :return: the comparison.
    :raises ValueError: with fewer than two profiles, two profiles of the same GPU, or
        profiles differing on a field of :data:`COMPARABLE_FIELDS`.
    """
    if len(profiles) < 2:
        raise ValueError(f"a machine comparison needs at least two profiles, got {len(profiles)}")
    gpu = [str(p.get("gpu", "?")) for p in profiles]
    if len(set(gpu)) != len(gpu):
        raise ValueError(
            f"two profiles declare the same GPU ({gpu}): this is a repeatability check, "
            f"not a machine comparison"
        )
    reference = profiles[0]
    for other in profiles[1:]:
        different = [c for c in COMPARABLE_FIELDS if reference.get(c) != other.get(c)]
        if different:
            raise ValueError(
                f"the profiles differ on {different} besides the GPU: a comparison with two free "
                f"axes measures their sum ({reference.get('run_id')} vs {other.get('run_id')})"
            )

    per_profile = [{(str(r["variante"]), int(r["batch"])): r for r in p["rows"]} for p in profiles]
    sets = [set(d) for d in per_profile]
    keys = sorted(set.intersection(*sets))
    note: list[str] = []
    missing = set.union(*sets) - set(keys)
    if missing:
        note.append(
            f"{len(missing)} cells are not in every profile and are left out of the "
            f"comparison: {sorted(missing)[:4]}"
        )

    ratios: list[dict[str, Any]] = []
    overhead: list[dict[str, Any]] = []
    for variant, batch in keys:
        latencies = [float(d[(variant, batch)]["latency_ms_mean"]) for d in per_profile]
        ratios.append(
            {
                "variante": variant,
                "batch": batch,
                **{f"latency_ms_{g}": lat for g, lat in zip(gpu, latencies, strict=True)},
                # The ratio is first over second: > 1 means the first machine is slower.
                "rapporto": latencies[0] / latencies[1] if latencies[1] else None,
            }
        )
        values = [d[(variant, batch)].get("overhead_latenza_pct") for d in per_profile]
        if any(v is not None for v in values):
            overhead.append(
                {
                    "variante": variant,
                    "batch": batch,
                    **{f"overhead_pct_{g}": v for g, v in zip(gpu, values, strict=True)},
                    "scarto_punti": (
                        values[0] - values[1]
                        if values[0] is not None and values[1] is not None
                        else None
                    ),
                }
            )

    # The ranking is checked at batch 1 (the deployment case), on methods with a defined
    # overhead, and up to a tolerance: two methods closer than `points_tolerance` are tied,
    # so swaps among merged variants (all within 0.3 points of zero) are not verdict changes.
    def _ranking(indice: int) -> list[tuple[str, float]]:
        return sorted(
            (
                (str(r["variante"]), float(r[f"overhead_pct_{gpu[indice]}"]))
                for r in overhead
                if int(r["batch"]) == 1 and r.get(f"overhead_pct_{gpu[indice]}") is not None
            ),
            key=lambda v: v[1],
        )

    def _classes(entry_list: Sequence[tuple[str, float]]) -> list[frozenset[str]]:
        """Return the tie classes in order; a jump below tolerance does not separate them."""
        outside: list[list[str]] = []
        last: float | None = None
        for name, value in entry_list:
            if last is None or (value - last) > points_tolerance:
                outside.append([name])
            else:
                outside[-1].append(name)
            last = value
        return [frozenset(c) for c in outside]

    rankings = [_ranking(i) for i in range(len(profiles))]
    classes = [_classes(g) for g in rankings]
    preserved = bool(classes[0]) and all(c == classes[0] for c in classes[1:])
    if not preserved:
        note.append(
            "the overhead ranking of the methods is NOT the same on the two machines, "
            f"even up to {points_tolerance:g} points: "
            f"{[[sorted(c) for c in cc] for cc in classes]}; the C7 verdict depends "
            f"on the hardware and must be declared as such"
        )
    elif any(g != rankings[0] for g in rankings[1:]):
        note.append(
            f"the raw ranking differs only **within** a tie class "
            f"(tolerance {points_tolerance:g} points): "
            f"{[[n for n, _ in g] for g in rankings]}; the C7 verdict is preserved"
        )

    ratio_values = [r["rapporto"] for r in ratios if r["rapporto"]]
    if ratio_values:
        mean = statistics.fmean(ratio_values)
        if (max(ratio_values) - min(ratio_values)) > 0.15 * mean:
            note.append(
                f"the latency ratio ranges from {min(ratio_values):.2f}x to "
                f'{max(ratio_values):.2f}x across variants: "the machine is k times '
                f'faster" is not a valid statement, and the median does not replace it'
            )
    return MachineComparison(
        macchine=tuple(gpu),
        run_id=tuple(str(p.get("run_id", "?")) for p in profiles),
        rapporti=ratios,
        overhead=overhead,
        ordinamento_conservato=preserved,
        rapporto_mediano=statistics.median(ratio_values) if ratio_values else None,
        rapporto_min=min(ratio_values) if ratio_values else None,
        rapporto_max=max(ratio_values) if ratio_values else None,
        note=tuple(note),
    )


def comparison_markdown(confronto: MachineComparison) -> str:
    """Return the machine comparison in Markdown, with both caveats in view.

    :param confronto: result of :func:`compare_machines`.
    :return: the Markdown text.
    """
    a, b = confronto.macchine[0], confronto.macchine[1]
    pieces = [
        f"### The two machines: `{confronto.run_id[0]}` vs `{confronto.run_id[1]}`\n",
        f"**Latencies of {a} and {b} are not compared directly**: they appear only as a "
        '**ratio**, and the range of that ratio across variants tells whether "k times '
        'faster" is a valid statement.\n',
        f"| variant | batch | {a} | {b} | ratio |\n|---|:-:|---:|---:|---:|",
    ]
    for r in confronto.rapporti:
        pieces.append(
            f"| {r['variante']} | {r['batch']} | {_fmt(r[f'latency_ms_{a}'], 1)} ms "
            f"| {_fmt(r[f'latency_ms_{b}'], 1)} ms | {_fmt(r['rapporto'], 3)}× |"
        )
    pieces.append(
        f"\nMedian ratio **{_fmt(confronto.rapporto_mediano, 3)}x**, range "
        f"{_fmt(confronto.rapporto_min, 3)}-{_fmt(confronto.rapporto_max, 3)}x.\n"
    )
    pieces.append(
        "**The overhead, which is comparable** (a ratio within each run, the C7 number):\n"
    )
    pieces.append(f"| variant | batch | {a} | {b} | gap |\n|---|:-:|---:|---:|---:|")
    for r in confronto.overhead:
        pieces.append(
            f"| {r['variante']} | {r['batch']} | {_fmt(r[f'overhead_pct_{a}'], 2, suffisso='%')} "
            f"| {_fmt(r[f'overhead_pct_{b}'], 2, suffisso='%')} "
            f"| {_fmt(r['scarto_punti'], 2)} points |"
        )
    outcome = (
        "**yes**: the C7 verdict does not depend on the hardware"
        if confronto.ordinamento_conservato
        else "**NO**: the C7 verdict depends on the machine and must be declared"
    )
    pieces.append(f"\n**Is the overhead ranking of the methods preserved?** {outcome}\n")
    for note in confronto.note:
        pieces.append(f"* {note}")
    return "\n".join(pieces) + "\n"
