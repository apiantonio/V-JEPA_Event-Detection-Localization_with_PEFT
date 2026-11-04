"""Deployment study

This module measures nothing: it opens existing artifacts and puts them in one table.

=========================  =================================================
what                       from
=========================  =================================================
**model** cost             ``runs/e5b_efficienza/efficiency_inference.json``
**decoder** cost           ``runs/e0d_..._throughput*/throughput.json``
**event response**         ``detection_latency`` ``metrics.json``
**accuracy**               ``chunk_auc`` / ``chunk_ap`` / ``event_prf``, same files
**independent check**      ``runs/registry.jsonl``, durations of the test passes
=========================  =================================================

It lives in ``exp/`` because it knows what a ``run_id`` is; the arithmetic lives in
:mod:`evd.eval.deployment`, which knows neither the registry nor the naming grid.

The composition is checked against real test passes over 9116 windows: if the times
predicted by the synthetic profile relate affinely to the registry durations (same
ranking, small residuals), the profile describes the real load. Only passes from the
same host are compared (otherwise the slope would absorb a machine difference), and
only PEFT cells.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evd.eval.deployment import (
    DeploymentRow,
    Geometry,
    Response,
    Stage,
    check_wallclock,
    compare_machines,
    comparison_markdown,
    compose,
    markdown_table,
)
from evd.utils.io import ensure_dir, read_json, read_jsonl, write_json, write_text_atomic
from evd.utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "REFERENCE_CELLS",
    "confronto",
    "collect_cells",
    "decoder_stage",
    "study",
    "observed_wallclock",
]

#: E5b variant -> cell that executed it, by the ``adapt`` field of the ``run_id``.
#: ``frozen`` is E2d attentive, not E2c linear: E5b timed an
#: :class:`~evd.models.heads.attentive.AttentiveProbe`. Merged variants have no cell of
#: their own (they are the deployment form of a run, with a verified-faithful merge) and
#: inherit from their unmerged variant.
REFERENCE_CELLS: dict[str, str] = {
    "frozen": "frozen-attentive",
    "vptdeep-n576": "vptdeep-n576",
    "lora-r32": "lora-r32-qkvomlp",
    "dora-r32": "dora-r32-qkvomlp",
}

# ``run_id`` is ``{exp}_{dataset}_{backbone}_{adapt}_{seed}`` (``docs/05`` §8).
_RUN_ID = re.compile(
    r"^(?P<exp>[^_]+)_(?P<dataset>[^_]+)_(?P<backbone>[^_]+)_(?P<adapt>.+)_s(?P<seed>\d+)$"
)


def decoder_stage(
    percorso: str | Path,
    *,
    workers: int = 8,
    augment: str = "none",
    persistent: bool | None = None,
) -> Stage:
    """Extract the per-window decoding cost from an E0d profile.

    :param percorso: ``throughput.json`` of an E0d leg.
    :param workers: number of workers; 8 is the ADR-049 choice for GPU jobs.
    :param augment: augmentation chain; ``none`` is right for inference (augmentation is
        training-only, ADR-028).
    :param persistent: filter on ``persistent_workers``; ``None`` takes the faster of the
        two (they differ by less than 1%).
    :return: the stage, with provenance filled in.
    :raises FileNotFoundError: if the artifact does not exist.
    :raises ValueError: if no row matches the filters.
    """
    p = Path(percorso)
    if not p.is_file():
        raise FileNotFoundError(
            f"decoder profile not found: {p}. Without it the composition would have a single "
            f"stage and would claim the model is the bottleneck without checking."
        )
    payload = read_json(p)
    choices = [
        r
        for r in payload.get("rows", ())
        if int(r.get("workers", -1)) == workers
        and str(r.get("augment", "none")) == augment
        and r.get("status") == "ok"
        and (persistent is None or bool(r.get("persistent")) == persistent)
    ]
    if not choices:
        available = sorted(
            {(int(r.get("workers", -1)), str(r.get("augment", "none"))) for r in payload["rows"]}
        )
        raise ValueError(
            f"no decoder row with workers={workers}, augment={augment!r} in {p}. "
            f"Available (workers, augment) pairs: {available}"
        )
    best = min(choices, key=lambda r: float(r["ms_per_window"]))
    return Stage(
        nome="decoder",
        ms_per_finestra=float(best["ms_per_window"]),
        provenienza=str(payload.get("run_id", p.parent.name)),
        dettaglio=(
            f"{workers} worker, augment={augment}, "
            f"persistent={bool(best.get('persistent'))}, "
            f"{best.get('n_windows')} finestre"
        ),
    )


def _variant_of(run_id: str) -> str | None:
    m = _RUN_ID.match(run_id)
    if not m:
        return None
    adapt = m.group("adapt")
    for variant, expected in REFERENCE_CELLS.items():
        if adapt == expected:
            return variant
    return None


def collect_cells(
    runs_dir: str | Path,
    *,
    experiments: tuple[str, ...] = ("e2", "e3"),
    expected_n_chunks: int | None = 9116,
) -> tuple[dict[str, dict[str, list[float]]], dict[str, Response], dict[str, list[str]]]:
    """Read the E2/E3 cells and extract quality and event response.

    Accuracy metrics are collected per seed and averaged by
    :func:`evd.eval.deployment.compose`. The :class:`~evd.eval.deployment.Response` comes
    from the median seed by chunk-AUC: a real cell with its provenance, not an average of
    percentiles.

    The ``run_id`` alone does not identify the right cell: E0i cells carry the same
    ``adapt`` but run on ``edp_v1`` and stop at selection. Two filters apply: the experiment
    belongs to the matrix, and the cell evaluated the expected grid.

    :param runs_dir: runs folder.
    :param experiments: accepted ``exp`` prefixes.
    :param expected_n_chunks: windows the evaluated partition must have; ``None`` disables
        the check (the grid declared in :class:`~evd.eval.deployment.Geometry`).
    :return: the triple ``(quality, responses, run_ids used)``.
    :raises FileNotFoundError: if the folder does not exist.
    """
    root = Path(runs_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"runs folder not found: {root}")

    per_variant: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for folder in sorted(root.iterdir()):
        metrics = folder / "metrics.json"
        if not folder.is_dir() or not metrics.is_file():
            continue
        variant = _variant_of(folder.name)
        if variant is None or not folder.name.startswith(experiments):
            continue
        try:
            payload = read_json(metrics)
        except (json.JSONDecodeError, OSError) as error:  # pragma: no cover - I/O
            log.warning("%s: unreadable metrics.json (%s), cell skipped", folder.name, error)
            continue
        if payload.get("chunk_auc") is None:
            continue
        if expected_n_chunks is not None and payload.get("n_chunk_eval") not in (
            None,
            expected_n_chunks,
        ):
            log.warning(
                "%s: evaluated %s windows instead of %d, cell skipped (different grid)",
                folder.name,
                payload.get("n_chunk_eval"),
                expected_n_chunks,
            )
            continue
        per_variant.setdefault(variant, []).append((folder.name, payload))

    quality: dict[str, dict[str, list[float]]] = {}
    responses: dict[str, Response] = {}
    used: dict[str, list[str]] = {}
    for variant, cells in per_variant.items():
        cells = sorted(cells, key=lambda c: float(c[1]["chunk_auc"]))
        used[variant] = [name for name, _ in cells]
        quality[variant] = {
            "chunk_auc": [
                float(m["chunk_auc"]) for _, m in cells if m.get("chunk_auc") is not None
            ],
            "chunk_ap": [float(m["chunk_ap"]) for _, m in cells if m.get("chunk_ap") is not None],
            "event_f1": [
                float(m["event_prf"]["f1"])
                for _, m in cells
                if (m.get("event_prf") or {}).get("f1")
            ],
        }
        # Median seed by chunk-AUC: a real cell, not an average of cells.
        _, median = cells[len(cells) // 2]
        dl = median.get("detection_latency") or {}
        if dl.get("mediana_s") is not None:
            responses[variant] = Response(
                mediana_s=float(dl["mediana_s"]),
                p90_s=float(dl["p90_s"]),
                n_eventi=int(dl["n_eventi"]),
                n_senza_latenza=int(
                    dl.get("n_senza_latenza_misurabile", dl.get("n_mai_rilevati", 0))
                ),
            )
    return quality, responses, used


def observed_wallclock(
    registry: str | Path,
    *,
    host: str,
    only_peft: bool = True,
    experiments: tuple[str, ...] = ("e2", "e3"),
) -> dict[str, list[float]]:
    """Return the observed durations of the evaluation passes, per variant, on one host.

    Three filters: (a) the experiment belongs to the E2/E3 matrix (E0i cells share the
    ``adapt`` field but run on ``edp_v1`` and never evaluate); (b) the row carries
    ``chunk_auc``, i.e. it completed an evaluation; (c) a single host.

    :param registry: ``runs/registry.jsonl``.
    :param host: node to compare on; a single one, because
        :func:`~evd.eval.deployment.check_wallclock` reads the slope as a hardware ratio.
    :param only_peft: drop the frozen cells (default ``True``): E2c/E2d evaluate from the
        feature store, so their duration measures the filesystem, not the model.
    :param experiments: accepted ``exp`` prefixes.
    :return: per variant, the minutes of every run found.
    :raises FileNotFoundError: if the registry does not exist.
    """
    import datetime as dt

    p = Path(registry)
    if not p.is_file():
        raise FileNotFoundError(f"registry not found: {p}")
    outside: dict[str, list[float]] = {}
    for row in read_jsonl(p, skip_invalid=True):
        run_id = str(row.get("run_id", ""))
        if row.get("host") != host or not row.get("finished_at") or row.get("status") != "ok":
            continue
        if not run_id.startswith(experiments):
            continue
        if (row.get("metrics") or {}).get("chunk_auc") is None:
            continue
        variant = _variant_of(run_id)
        if variant is None or (only_peft and variant == "frozen"):
            continue
        minutes = (
            dt.datetime.fromisoformat(row["finished_at"])
            - dt.datetime.fromisoformat(row["started_at"])
        ).total_seconds() / 60.0
        outside.setdefault(variant, []).append(minutes)
    return outside


def study(
    *,
    efficiency: str | Path,
    decoder: str | Path,
    runs_dir: str | Path,
    out_dir: str | Path,
    registry: str | Path | None = None,
    host_verifica: str | None = None,
    workers_decoder: int = 8,
    augment_decoder: str = "none",
    batch: int = 1,
    batch_verifica: int = 2,
    geometria: Geometry | None = None,
    figure: bool = True,
) -> dict[str, Any]:
    """Assemble the study: JSON, Markdown tables and figures.

    :param efficienza: E5b ``efficiency_inference.json``.
    :param decoder: E0d ``throughput.json``.
    :param runs_dir: runs folder, for the E2/E3 cells.
    :param out_dir: where ``deployment.json``, ``deployment.md`` and the PNGs are written.
    :param registry: ``registry.jsonl`` for the independent check; ``None`` skips it and the
        payload declares so.
    :param host_verifica: host on which observed durations are compared.
    :param workers_decoder: decoder workers used in the composition.
    :param augment_decoder: decoder augmentation chain (``none`` at inference).
    :param batch: batch of the main table; 1 is online deployment.
    :param batch_verifica: batch of the test passes (2 on ``edp_v2``).
    :param geometria: protocol geometry; defaults to ``edp_v2``.
    :param figure: whether to draw the PNGs (``False`` in tests).
    :return: the payload, already written to disk.
    :raises FileNotFoundError: if the efficiency profile is missing.
    """
    efficiency_path = Path(efficiency)
    if not efficiency_path.is_file():
        raise FileNotFoundError(
            f"E5b profile not found: {efficiency_path}. The deployment study composes "
            f"existing measurements: without that file there is nothing to compose."
        )
    payload_eff = read_json(efficiency_path)
    efficiency_rows: list[dict[str, Any]] = list(payload_eff["rows"])
    geo = geometria or Geometry()
    stage = decoder_stage(decoder, workers=workers_decoder, augment=augment_decoder)
    # The expected grid comes from the geometry, not from a literal.
    quality, responses, used_cells = collect_cells(
        runs_dir, expected_n_chunks=geo.n_chunk_valutazione
    )

    rows = compose(
        efficiency_rows,
        decoder=stage,
        geometria=geo,
        responses=responses,
        quality=quality,
        batch=batch,
    )

    verify: dict[str, Any] = {
        "eseguita": False,
        "motivo": "registry or host not provided",
    }
    if registry and host_verifica:
        observed = observed_wallclock(registry, host=host_verifica)
        means = {k: sum(v) / len(v) for k, v in observed.items() if v}
        if len(means) >= 2:
            verify = check_wallclock(
                efficiency_rows,
                means,
                n_chunk=geo.n_chunk_valutazione,
                batch=batch_verifica,
            )
            verify |= {
                "eseguita": True,
                "host": host_verifica,
                "batch": batch_verifica,
                "minuti_osservati": observed,
                "minuti_medi": means,
            }
        else:
            verify = {
                "eseguita": False,
                "motivo": (
                    f"on {host_verifica} there are {len(means)} comparable PEFT variants; "
                    f"with fewer than two the line fits exactly and the check is vacuous"
                ),
            }

    folder = ensure_dir(out_dir)
    payload = {
        "schema": 1,
        "sorgenti": {
            "efficienza": {
                "run_id": payload_eff.get("run_id"),
                "gpu": payload_eff.get("gpu"),
                "precision": payload_eff.get("precision"),
                "profile_schema": payload_eff.get("profile_schema"),
                "percorso": str(efficiency_path),
            },
            "decoder": asdict(stage),
            "celle": used_cells,
        },
        "geometria": asdict(geo),
        "batch": batch,
        "righe": [r.as_dict() for r in rows],
        "verifica_wallclock": verify,
    }
    write_json(folder / "deployment.json", payload)
    write_text_atomic(
        folder / "deployment.md",
        _markdown(payload, rows),
    )
    if figure:
        payload["figure"] = _figure(folder, rows, efficiency_rows, geo)
    return payload


def _figure(
    cartella: Path,
    righe: Sequence[DeploymentRow],
    efficiency_rows: Sequence[Mapping[str, Any]],
    geo: Geometry,
) -> dict[str, str]:
    """Draw the figures the data allow, and record which ones were skipped.

    Scaling and tail figures need the schema-2 profile (batch sweep, percentiles); with
    schema 1 they are skipped and the payload records the reason.
    """
    from evd.eval import deployment_plots as dp

    written: dict[str, str] = {}
    written["latenza"] = str(
        dp.latenza_png(righe, cartella / "deployment_latency.png", hop_s=geo.hop_s)
    )
    written["capacita"] = str(dp.capacita_png(righe, cartella / "deployment_capacity.png"))
    for name, function, arguments in (
        ("pareto", dp.pareto_png, (righe, cartella / "deployment_pareto.png")),
        ("scaling", dp.scaling_png, (efficiency_rows, cartella / "deployment_scaling.png")),
        ("coda", dp.tail_png, (efficiency_rows, cartella / "deployment_tail.png")),
    ):
        try:
            written[name] = str(function(*arguments))
        except ValueError as error:
            written[name] = f"skipped: {error}"
            log.info("figure %s skipped: %s", name, error)
    return written


def _markdown(payload: Mapping[str, Any], righe: Sequence[DeploymentRow]) -> str:
    """Return the study document: provenance first, then tables, then the check."""
    eff = payload["sorgenti"]["efficienza"]
    dec = payload["sorgenti"]["decoder"]
    head = [
        "# E5b.2 — deployment study",
        "",
        "> Composed from existing measurements: **no new job**. Every number carries its",
        "> provenance, and the two stages come from **different runs**: this is the",
        "> declared limit of the composition.",
        "",
        f"* **model** — `{eff['run_id']}` on **{eff['gpu']}**, {eff['precision']}, "
        f"profile schema {eff['profile_schema']}",
        f"* **decoder** — `{dec['provenienza']}` ({dec['dettaglio']}), "
        f"{dec['ms_per_finestra']:.1f} ms/window",
        "* **cells** — "
        + "; ".join(f"{k}: {len(v)} seeds" for k, v in payload["sorgenti"]["celle"].items()),
        f"* **geometry** — hop {payload['geometria']['hop_s']:g} s, "
        f"span {payload['geometria']['span_s']:g} s, "
        f"{payload['geometria']['n_chunk_valutazione']} test windows; "
        f"table at **batch {payload['batch']}**",
        "",
        "",
    ]
    tail: list[str] = []
    v = payload.get("verifica_wallclock") or {}
    if v.get("eseguita"):
        tail += [
            "",
            "## Independent check",
            "",
            f"Real test passes on `{v['host']}` (batch {v['batch']}, "
            f"{payload['geometria']['n_chunk_valutazione']} windows) against the times predicted "
            "by the synthetic profile. The fit is `observed = a + b · predicted`, with `a` "
            "the fixed overhead per window and `1/b` the implied hardware ratio; "
            "neither is imposed, both come from the fit.",
            "",
            f"* **ranking preserved**: {'yes' if v['ordinamento_rispettato'] else 'NO'}",
            f"* **fixed overhead**: {v['overhead_fisso_ms_per_finestra']:.1f} ms/window",
            f"* **implied hardware ratio**: {v['rapporto_hardware_implicito']:.2f}×",
            f"* **max residual**: {v['residuo_max_pct']:.2f}%",
            "",
            "| variant | predicted (ms/win) | observed (ms/win) | fit expectation | residual |",
            "|---|---:|---:|---:|---:|",
        ]
        for r in v["residui"]:
            tail.append(
                f"| {r['variante']} | {r['previsto_ms']:.1f} | {r['osservato_ms']:.1f} "
                f"| {r['atteso_dal_fit_ms']:.1f} | {r['residuo_pct']:+.2f}% |"
            )
        tail.append("")
        tail.append(
            "⚠️ **Not a statistical test**: three points and two parameters leave one degree "
            "of freedom. What matters is the ranking and the size of the residuals."
        )
    else:
        tail += ["", "## Independent check", "", f"⚠️ **not run** — {v.get('motivo')}"]
    return "\n".join(head) + markdown_table(righe) + "\n".join(tail) + "\n"


def confronto(
    *,
    efficiencies: Sequence[str | Path],
    out_dir: str | Path,
    figure: bool = True,
) -> dict[str, Any]:
    """Compare two or more profiles from different machines and write the artifacts.

    The output goes to a shared folder, not into a run: a comparison belongs to none of the
    profiles it combines (as ``make_figures.py`` writes to ``runs/figures/``).

    :param efficiencies: paths of the ``efficiency_inference.json`` files to compare.
    :param out_dir: where ``confronto_macchine.{json,md}`` and the PNG are written.
    :param figure: whether to draw (``False`` in tests).
    :return: the payload, already written to disk.
    :raises FileNotFoundError: naming the first missing profile.
    """
    profiles: list[dict[str, Any]] = []
    for path in efficiencies:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(
                f"profile not found: {p}. The machine comparison needs every declared "
                f"profile: silently skipping one would compare one machine fewer "
                f"without the table saying so."
            )
        profiles.append(read_json(p))

    outcome = compare_machines(profiles)
    folder = ensure_dir(out_dir)
    payload = {
        "schema": 1,
        "profili": [
            {
                "run_id": p.get("run_id"),
                "gpu": p.get("gpu"),
                "precision": p.get("precision"),
                "profile_schema": p.get("profile_schema"),
                "avvio_a_freddo": p.get("avvio_a_freddo"),
                "percorso": str(q),
            }
            for p, q in zip(profiles, efficiencies, strict=True)
        ],
        "confronto": outcome.as_dict(),
    }
    write_json(folder / "confronto_macchine.json", payload)
    write_text_atomic(folder / "confronto_macchine.md", comparison_markdown(outcome))
    if figure:
        from evd.eval import deployment_plots as dp

        try:
            payload["figura"] = str(dp.macchine_png(outcome, folder / "confronto_macchine.png"))
        except ValueError as error:  # pragma: no cover - data dependent
            payload["figura"] = f"skipped: {error}"
            log.info("comparison figure skipped: %s", error)
    return payload
