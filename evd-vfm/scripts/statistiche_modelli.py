#!/usr/bin/env python
"""Cost statistics of the thesis models: four-epoch training, selected step, response, memory.

    python scripts/statistiche_modelli.py --elenca-job > job.txt
    # on the cluster: sacct -P -j <ids> --format=JobIDRaw,JobID,Partition,NodeList,State,ElapsedRaw,MaxRSS,TRESUsageInMax%400
    python scripts/statistiche_modelli.py --sacct runs/statistiche_modelli/sacct.txt

Writes ``statistiche_modelli.{json,md,csv}`` in ``--out``. It reads artifacts already on disk and
uses no GPU: run folders, SLURM logs, the SLURM accounting dump and the E5b inference profiles.

The eighteen models follow the chapter-4 BRIEF: the ten trainings extended in continuation
(:data:`~classifiche_modelli.VERSION_GROUPS`), in the version published by OVAD guard 30 s,
and the eight E3-S sequential combinations. E3-SV/E3-SO and unmerged inference are excluded.
The OVAD version is frozen across seeds; checkpoints within it use validation AUROC. For each:

* **the chain of four epochs**, rebuilt from the run receipts (``avvio_a_caldo``,
  ``preadattamento`` in ``meta.json``), not assumed: one stage per run, each starting from
  the step of its parent it inherited. The global step of a checkpoint is the stage offset
  plus its local step;
* **training cost**: seconds per step and per validation pass measured in the SLURM logs
  (:mod:`evd.exp.training_log`), pooled over every run with the same architecture (a
  continuation does not change the model, nor does the state of the inherited prior), per
  GPU and never converted between GPUs; hours = steps × s/step + validations × s/validation;
* **the selected checkpoint**: local and global step, validation AUROC, and the training
  time needed to reach it;
* **memory**: the peak GPU memory of the real training jobs (``gres/gpumem`` of SLURM
  accounting, NVML, i.e. including the allocator cache and the CUDA context) and the host
  RSS; the peak at inference is the allocated one measured by E5b (PyTorch);
* **response and detection delay**: compute per window of the deployed form (reparam merged,
  fully or partially, ADR-064) from the E5b profiles, plus decoding, plus the grid delay of
  the intersection rule (ADR-061) of the reported cells.

The combinations' merged rows come from a second profile (E5b4); the variants measured by
both profiles are compared and the deviation is written next to the numbers.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from classifiche_modelli import (  # noqa: E402
    DECODE_MS,
    HOP_S,
    VERSION_GROUPS,
    aggregate,
    find_cells,
    load_cell,
    model_key,
    reported_models,
)
from evd.exp import training_log as TL  # noqa: E402
from evd.utils.hashing import file_sha256  # noqa: E402
from evd.utils.io import ensure_dir, read_json, write_text_atomic  # noqa: E402
from evd.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger(__name__)

GPUS: tuple[str, ...] = ("A100", "H100")

#: Experiment prefixes whose architecture is that of another key: a continuation (E3-M,
#: E3-M4, E3-C4) does not change the model, and E3-SV differs from E3-S only in the state of
#: the inherited prior. The cost of a step is pooled over the runs sharing the target key.
SAME_ARCHITECTURE: dict[str, str] = {"e3m": "", "e3m4": "", "e3c4": "e3c", "e3sv": "e3s"}

#: Deployed form of each model: ``(merged variant, unmerged variant)`` of the inference
#: profile. A ``fondi`` cell has no unmerged form (its reparam was merged before training);
#: E3-SO uses the DoRA r=32 combination rows, the same graph once merged (the rank does not
#: exist after ``merge_()``), and its unmerged r=16 form was not profiled.
DEPLOYED: dict[str, tuple[str, str | None]] = {
    "frozen-linear": ("frozen-linear", None),
    "frozen-attentive": ("frozen", None),
    "vptshallow-n8": ("vptshallow-n8", None),
    "vptdeep-n8": ("vptdeep-n8", None),
    "lora-r32-qkvomlp": ("lora-r32-fusa", "lora-r32"),
    "dora-r32-qkvomlp": ("dora-r32-fusa", "dora-r32"),
    **{
        f"{exp}:{r}-r32-vpt{v}-n8": (f"combo-{r}-r32-vpt{v}-n8-fusa", f"combo-{r}-r32-vpt{v}-n8")
        for exp in ("e3c", "e3s", "e3sv")
        for r in ("lora", "dora")
        for v in ("shallow", "deep")
    },
    **{
        f"{exp}:vpt{v}-n8-su-{r}-r32": (f"combo-{r}-r32-vpt{v}-n8-fusa", None)
        for exp in ("e3s", "e3sv")
        for r in ("lora", "dora")
        for v in ("shallow", "deep")
    },
    **{
        f"e3so:dora-r16-vpt{v}-n8": (f"combo-dora-r32-vpt{v}-n8-fusa", None)
        for v in ("shallow", "deep")
    },
    **{
        f"e3so:vpt{v}-n8-su-dora-r16": (f"combo-dora-r32-vpt{v}-n8-fusa", None)
        for v in ("shallow", "deep")
    },
}


def architecture(key: str) -> str:
    """Return the key whose architecture a :data:`MODELS` key shares.

    :param key: a :data:`MODELS` key, e.g. ``"e3m4:lora-r32-qkvomlp"``.
    :return: the key of :data:`SAME_ARCHITECTURE`, e.g. ``"lora-r32-qkvomlp"``; the key itself
        when its prefix is not listed.
    """
    prefix, sep, rest = key.partition(":")
    if not sep or prefix not in SAME_ARCHITECTURE:
        return key
    target = SAME_ARCHITECTURE[prefix]
    return f"{target}:{rest}" if target else rest


def thesis_models(
    reported: Mapping[str, Mapping[str, str]], chosen: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Select the thesis models among the reported ones.

    :param reported: the catalogue with the published OVAD version choices applied.
    :param chosen: the version choices of the same call.
    :return: the ten :data:`VERSION_GROUPS` trainings and eight E3-S combinations,
        with ``nome``, ``famiglia`` and ``gruppo`` (``None`` for E3-S).
    """
    group_of = {choice["versione"]: group for group, choice in chosen.items()}
    out: dict[str, dict[str, Any]] = {}
    for key, entry in reported.items():
        if key in group_of or key.startswith("e3s:"):
            out[key] = {**entry, "gruppo": group_of.get(key)}
    return out


def ovad_choices(path: Path, cells: Mapping[str, Mapping[int, Path]]) -> dict[str, dict[str, Any]]:
    """Read the published OVAD choices, refusing missing or unavailable versions.

    :param path: ``ovad_guard30/classifiche/selezione_versioni.json``.
    :param cells: available run folders by model and seed.
    :return: version choices in the format consumed by :func:`thesis_models`.
    :raises ValueError: if a group is missing, duplicated, or has no selected seed-42 run.
    """
    choices = {}
    for row in read_json(path):
        group, key = row["modello_base"], row["versione_scelta"]
        if (
            group in choices
            or key not in VERSION_GROUPS.get(group, ())
            or 42 not in cells.get(key, {})
        ):
            raise ValueError(f"Invalid or unavailable OVAD choice: {group} -> {key}")
        choices[group] = {"versione": key, "epoche": row["epoche"], "criterio": row["criterio"]}
    if set(choices) != set(VERSION_GROUPS):
        raise ValueError(
            f"Missing OVAD version groups: {sorted(set(VERSION_GROUPS) - set(choices))}"
        )
    return choices


def selected_models(
    args: argparse.Namespace, cells: Mapping[str, Mapping[int, Path]]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], Path]:
    """:return: the thesis models and frozen OVAD version choices."""
    path = (
        args.selezione_versioni
        or args.runs_dir / "classifiche/ovad_guard30/classifiche/selezione_versioni.json"
    )
    chosen = ovad_choices(path, cells)
    reported = reported_models(chosen={g: c["versione"] for g, c in chosen.items()})
    return thesis_models(reported, chosen), chosen, path


def parent_of(run_dir: Path) -> tuple[str, int] | None:
    """Return the run a run starts from and the step it inherited.

    :param run_dir: the run folder.
    :return: ``(run_id, step)`` from ``avvio_a_caldo`` (E3-M, E3-M4, E3-C4) or from
        ``preadattamento`` (E3-S family) in ``meta.json``; ``None`` for a run trained from the
        pre-trained backbone.
    :raises ValueError: if a receipt names no run or no step.
    """
    meta = read_json(run_dir / "meta.json")
    for field in ("avvio_a_caldo", "preadattamento"):
        receipt = meta.get(field)
        if receipt:
            if not receipt.get("run_id") or receipt.get("step") is None:
                raise ValueError(f"{run_dir.name}: receipt {field} without run_id or step")
            return str(receipt["run_id"]), int(receipt["step"])
    return None


def ancestry(runs_dir: Path, run_id: str) -> list[dict[str, Any]]:
    """Rebuild the chain of trainings behind a run, root first.

    :param runs_dir: the ``runs/`` folder.
    :param run_id: the last run of the chain.
    :return: one stage per run with ``run_id``, ``chiave`` (:data:`MODELS` key),
        ``architettura``, ``max_steps``, ``eval_every``, ``passo_ereditato`` (step of the
        parent it starts from, 0 for the root) and ``offset`` (global step at the start of the
        stage, i.e. the sum of the inherited steps above it).
    :raises FileNotFoundError: if a run of the chain is not on disk.
    :raises ValueError: if the chain loops.
    """
    stages: list[dict[str, Any]] = []
    current: str | None = run_id
    seen: set[str] = set()
    while current is not None:
        if current in seen:
            raise ValueError(f"loop in the chain of {run_id}: {current}")
        seen.add(current)
        folder = runs_dir / current
        metrics = read_json(folder / "metrics.json")
        key = model_key(current) or current
        parent = parent_of(folder)
        stages.append(
            {
                "run_id": current,
                "chiave": key,
                "architettura": architecture(key),
                "max_steps": int(metrics["max_steps"]),
                "eval_every": int((metrics.get("model_selection") or {}).get("eval_every") or 0)
                or None,
                "passo_ereditato": 0 if parent is None else parent[1],
            }
        )
        current = None if parent is None else parent[0]
    stages.reverse()
    offset = 0
    for stage in stages:
        offset += stage["passo_ereditato"]
        stage["offset"] = offset
    return stages


def architecture_pools(cells: Mapping[str, Mapping[int, Path]]) -> dict[str, dict[str, int | None]]:
    """Group the runs on disk by architecture.

    :param cells: ``{key: {seed: folder}}`` (:func:`~classifiche_modelli.find_cells`).
    :return: ``{architecture: {run_id: eval_every}}``.
    """
    pools: dict[str, dict[str, int | None]] = defaultdict(dict)
    for key, seeds in cells.items():
        for folder in seeds.values():
            selection = read_json(folder / "metrics.json").get("model_selection") or {}
            pools[architecture(key)][folder.name] = selection.get("eval_every")
    return dict(pools)


def costs_by_architecture(
    logs: Sequence[TL.TrainingLog], pools: Mapping[str, Mapping[str, int | None]]
) -> dict[str, dict[str, TL.StepCost | None]]:
    """Measure step and validation cost for every architecture, on both GPUs.

    :param logs: the parsed SLURM logs.
    :param pools: :func:`architecture_pools`.
    :return: ``{architecture: {"A100": cost, "H100": cost}}``; ``None`` where the GPU has too
        few intervals.
    """
    return {
        arch: {gpu: TL.step_cost(logs, runs, gpu) for gpu in GPUS} for arch, runs in pools.items()
    }


def per_run_step_costs(
    logs: Sequence[TL.TrainingLog], runs: Mapping[str, int | None], gpu: str
) -> dict[str, float]:
    """Return the step cost of every run of a pool taken alone, to show its spread.

    :param logs: the parsed SLURM logs.
    :param runs: one pool of :func:`architecture_pools`.
    :param gpu: the GPU.
    :return: ``{run_id: s/step}`` for the runs with enough intervals on that GPU.
    """
    out: dict[str, float] = {}
    for run_id, eval_every in runs.items():
        cost = TL.step_cost(logs, {run_id: eval_every}, gpu)
        if cost is not None:
            out[run_id] = cost.s_per_step
    return out


def stage_hours(
    stage: Mapping[str, Any], cost: TL.StepCost | None, steps: int | None = None
) -> float | None:
    """Return the training hours of a stage, validations included.

    :param stage: a stage of :func:`ancestry`.
    :param cost: the cost of its architecture on one GPU.
    :param steps: steps actually needed (to a selected checkpoint); ``None`` = the whole stage.
    :return: hours, or ``None`` if the cost (or the validation cost, when the stage validates)
        is unknown.
    """
    if cost is None:
        return None
    n = stage["max_steps"] if steps is None else steps
    validations = n // stage["eval_every"] if stage["eval_every"] else 0
    if validations and cost.validation_s is None:
        return None
    return (n * cost.s_per_step + validations * (cost.validation_s or 0.0)) / 3600.0


def chain_hours(
    stages: Sequence[Mapping[str, Any]],
    costs: Mapping[str, Mapping[str, TL.StepCost | None]],
    gpu: str,
    *,
    last_steps: int | None = None,
) -> float | None:
    """Return the training hours of a chain on one GPU.

    :param stages: the chain (:func:`ancestry`).
    :param costs: :func:`costs_by_architecture`.
    :param gpu: the GPU.
    :param last_steps: steps of the last stage to count (to a selected checkpoint); ``None``
        counts it whole.
    :return: the sum over the stages, ``None`` if any stage cost is unknown on this GPU.
    """
    total = 0.0
    for i, stage in enumerate(stages):
        steps = last_steps if i == len(stages) - 1 else None
        hours = stage_hours(stage, (costs.get(stage["architettura"]) or {}).get(gpu), steps)
        if hours is None:
            return None
        total += hours
    return total


_UNITS = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}


def _bytes(text: str | None) -> int | None:
    """:return: a SLURM size (``61904264K``, ``44984M``) in bytes, or ``None``."""
    if not text:
        return None
    found = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGT]?)", text.strip())
    if not found:
        return None
    return int(float(found.group(1)) * _UNITS.get(found.group(2), 1))


def parse_sacct(text: str) -> dict[str, dict[str, Any]]:
    """Read a ``sacct -P`` dump into one record per job.

    :param text: pipe-separated output with a header naming at least ``JobIDRaw`` and some of
        ``Partition``, ``NodeList``, ``State``, ``ElapsedRaw``, ``MaxRSS``, ``TRESUsageInMax``.
    :return: ``{job id: {...}}`` keyed on the raw id of the job (``SLURM_JOB_ID``, what
        ``hpc/env.sh`` prints); allocation fields come from the job row, usage fields from its
        ``.batch`` step: ``elapsed_s``, ``max_rss_bytes``, ``gpu_mem_bytes`` (``gres/gpumem``,
        absent where the node does not account it), ``gpu_util_pct``.
    :raises ValueError: if the header lacks ``JobIDRaw``.
    """
    rows = list(csv.DictReader(io.StringIO(text.strip()), delimiter="|"))
    if rows and "JobIDRaw" not in rows[0]:
        raise ValueError("the sacct dump needs the JobIDRaw column")
    jobs: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in rows:
        raw = row["JobIDRaw"]
        job, _, step = raw.partition(".")
        record = jobs[job]
        if not step:
            record.update(
                partition=row.get("Partition") or None,
                node=row.get("NodeList") or None,
                state=row.get("State") or None,
                elapsed_s=int(row["ElapsedRaw"])
                if (row.get("ElapsedRaw") or "").isdigit()
                else None,
            )
        elif step == "batch":
            usage = dict(
                item.split("=", 1)
                for item in (row.get("TRESUsageInMax") or "").split(",")
                if "=" in item
            )
            record.update(
                max_rss_bytes=_bytes(row.get("MaxRSS")),
                gpu_mem_bytes=_bytes(usage.get("gres/gpumem")),
                gpu_util_pct=float(usage["gres/gpuutil"]) if usage.get("gres/gpuutil") else None,
            )
    return dict(jobs)


def training_jobs(
    logs: Sequence[TL.TrainingLog], run_ids: Iterable[str]
) -> dict[str, TL.TrainingLog]:
    """Return the jobs that trained any of the given runs.

    :param logs: the parsed SLURM logs.
    :param run_ids: the runs.
    :return: ``{job id: log}`` for the logs with at least one checkpoint of those runs (a job
        that died before its first checkpoint, or only evaluated, is not a training job).
    """
    wanted = set(run_ids)
    out: dict[str, TL.TrainingLog] = {}
    for log_ in logs:
        if log_.job_id and any(r in wanted and s.checkpoints for r, s in log_.segments.items()):
            out[log_.job_id] = log_
    return out


def memory(
    jobs: Mapping[str, TL.TrainingLog], sacct: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Return the peak memory of a set of training jobs.

    :param jobs: :func:`training_jobs`.
    :param sacct: :func:`parse_sacct`.
    :return: maximum and minimum ``gres/gpumem`` over the jobs that account it, maximum host
        RSS, and how many jobs each number is over.
    """
    gpu = [sacct[j]["gpu_mem_bytes"] for j in jobs if (sacct.get(j) or {}).get("gpu_mem_bytes")]
    rss = [sacct[j]["max_rss_bytes"] for j in jobs if (sacct.get(j) or {}).get("max_rss_bytes")]
    return {
        "gpu_picco_gib": max(gpu) / 1024**3 if gpu else None,
        "gpu_minimo_dei_picchi_gib": min(gpu) / 1024**3 if gpu else None,
        "n_job_gpu": len(gpu),
        "rss_picco_gib": max(rss) / 1024**3 if rss else None,
        "n_job_rss": len(rss),
        "n_job": len(jobs),
    }


def consumed_hours(
    jobs: Mapping[str, TL.TrainingLog], sacct: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Return the GPU-hours the cluster actually spent on a set of training jobs, per GPU.

    :param jobs: :func:`training_jobs`.
    :param sacct: :func:`parse_sacct`; a job missing there counts with its log duration.
    :return: ``{"A100": h, "H100": h, "n_job": n, "da_log": n}``.
    """
    hours: dict[str, float] = defaultdict(float)
    from_log = 0
    for job, log_ in jobs.items():
        elapsed = (sacct.get(job) or {}).get("elapsed_s")
        if elapsed is None:
            elapsed, from_log = log_.duration_s, from_log + 1
        hours[log_.gpu or "?"] += elapsed / 3600.0
    return {
        **{g: hours.get(g, 0.0) for g in (*GPUS, "?") if g in hours or g in GPUS},
        "n_job": len(jobs),
        "da_log": from_log,
    }


def load_profile_rows(folder: Path) -> dict[str, Any]:
    """Read the batch-1 rows of an E5b profile.

    :param folder: a folder with ``efficiency_inference.json``.
    :return: ``{"gpu", "run_id", "righe": {variant: row}}``; empty if the file is missing.
    """
    path = folder / "efficiency_inference.json"
    if not path.is_file():
        log.warning("inference profile missing: %s", path)
        return {}
    payload = read_json(path)
    rows = {
        str(r["variante"]): r
        for r in payload.get("rows", [])
        if int(r["batch"]) == 1 and r.get("status") == "ok"
    }
    return {"gpu": payload.get("gpu"), "run_id": payload.get("run_id"), "righe": rows}


def profile_agreement(reference: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    """Compare two profiles on the variants both measured.

    :param reference: :func:`load_profile_rows` of the reference profile.
    :param second: the same of the second profile.
    :return: the relative latency deviation per shared variant and its maximum in modulus.
    """
    a, b = reference.get("righe") or {}, second.get("righe") or {}
    shared = sorted(set(a) & set(b))
    deviation = {v: b[v]["latency_ms_mean"] / a[v]["latency_ms_mean"] - 1.0 for v in shared}
    return {
        "varianti": shared,
        "scarto_relativo": deviation,
        "scarto_massimo": max((abs(d) for d in deviation.values()), default=None),
    }


def inference(variant: str | None, profiles: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Return the inference measures of a variant from the first profile that has it.

    :param variant: the E5b variant, or ``None``.
    :param profiles: profiles of ONE GPU in order of preference.
    :return: latency (mean, sd, p95, p99), peak allocated VRAM, throughput, merge receipt and
        the profile it came from; ``None`` if no profile has the variant.
    """
    if variant is None:
        return None
    for profile in profiles:
        row = (profile.get("righe") or {}).get(variant)
        if row is not None:
            return {
                "variante": variant,
                "profilo": profile.get("run_id"),
                "latenza_ms": float(row["latency_ms_mean"]),
                "latenza_ms_sd": row.get("latency_ms_std"),
                "latenza_ms_p95": row.get("latency_ms_p95"),
                "latenza_ms_p99": row.get("latency_ms_p99"),
                "vram_picco_gib": None
                if row.get("peak_vram_bytes") is None
                else row["peak_vram_bytes"] / 1024**3,
                "finestre_s": row.get("throughput_windows_s"),
                "fusione": row.get("fusione") or ("totale" if row.get("fusa") else "nessuna"),
                "extra_tokens": row.get("extra_tokens"),
                "merge_max_abs": row.get("merge_max_abs"),
                "effetto_adattatore": row.get("effetto_adattatore"),
            }
    return None


def describe_model(
    key: str,
    entry: Mapping[str, Any],
    *,
    runs_dir: Path,
    cells: Mapping[str, Mapping[int, Path]],
    logs: Sequence[TL.TrainingLog],
    costs: Mapping[str, Mapping[str, TL.StepCost | None]],
    spread: Mapping[str, Mapping[str, Mapping[str, float]]],
    sacct: Mapping[str, Mapping[str, Any]],
    profiles: Mapping[str, Sequence[Mapping[str, Any]]],
    cache_dir: Path | None,
) -> dict[str, Any]:
    """Gather every statistic of one thesis model.

    :param key: the reported :data:`MODELS` key (the version entering the leaderboards).
    :param entry: its :func:`thesis_models` entry.
    :param runs_dir: the ``runs/`` folder.
    :param cells: :func:`~classifiche_modelli.find_cells`.
    :param logs: the parsed SLURM logs.
    :param costs: :func:`costs_by_architecture`.
    :param spread: ``{architecture: {gpu: per_run_step_costs}}``.
    :param sacct: :func:`parse_sacct` (may be empty).
    :param profiles: ``{gpu: [profiles in order of preference]}``.
    :param cache_dir: the ablation analysis cache of :func:`~classifiche_modelli.load_cell`.
    :return: the model record (see the module docstring).
    :raises KeyError: if the reported version has no seed-42 run (the chain is read at 42).
    """
    group = entry.get("gruppo")
    # The four-epoch chain is that of the LAST version of the group at seed 42 (E3-M4/E3-C4,
    # the only seed of the fourth epoch); for a sequential model, its own.
    tail_key = VERSION_GROUPS[group][-1] if group else key
    chain = ancestry(runs_dir, cells[tail_key][42].name)
    total_steps = sum(s["max_steps"] for s in chain)
    chain_runs = [s["run_id"] for s in chain]

    seeds: dict[int, dict[str, Any]] = {}
    for seed, folder in sorted(cells[key].items()):
        cell = load_cell(folder, cache_dir)
        metrics = cell["m"]
        selection = metrics.get("model_selection") or {}
        own_chain = ancestry(runs_dir, folder.name)
        last = own_chain[-1]
        local = int(selection["step"]) if selection.get("step") is not None else None
        evaluation = [
            (log_.gpu, log_.segments[folder.name].evaluation_seconds)
            for log_ in logs
            if folder.name in log_.segments
            and log_.segments[folder.name].evaluation_seconds is not None
        ]
        seeds[seed] = {
            "run_id": folder.name,
            "catena": [s["run_id"] for s in own_chain],
            "passo_locale": local,
            "max_steps_versione": last["max_steps"],
            "passi_catena_versione": sum(s["max_steps"] for s in own_chain),
            "offset": last["offset"],
            "passo_globale": None if local is None else last["offset"] + local,
            "val_auroc": selection.get("value"),
            "ore_al_checkpoint": {
                gpu: chain_hours(own_chain, costs, gpu, last_steps=local) for gpu in GPUS
            },
            "valutazione_test_s": [{"gpu": g, "secondi": s} for g, s in evaluation],
            "ritardo_griglia_mediana_s": (cell["lat"] or {}).get("mediana_s"),
            "ritardo_griglia_p90_s": (cell["lat"] or {}).get("p90_s"),
            "params_adattatore": (metrics.get("adapt") or {}).get("n_trainable"),
            "params_addestrabili": metrics.get("trainable_params"),
        }

    first = seeds[min(seeds)]
    params_head = (
        None
        if first["params_addestrabili"] is None or first["params_adattatore"] is None
        else int(first["params_addestrabili"]) - int(first["params_adattatore"])
    )
    final_arch = chain[-1]["architettura"]
    same_arch_runs = [
        f.name
        for k, by_seed in cells.items()
        if architecture(k) == final_arch
        for f in by_seed.values()
    ]
    chain_arch_runs = [
        f.name
        for k, by_seed in cells.items()
        if architecture(k) in {s["architettura"] for s in chain}
        for f in by_seed.values()
    ]

    merged_variant, _ = DEPLOYED.get(group or key, (None, None))
    deployed: dict[str, Any] = {}
    for gpu in GPUS:
        merged = inference(merged_variant, profiles.get(gpu, ()))
        response = None if merged is None else merged["latenza_ms"] + DECODE_MS
        delays = {
            s: None
            if response is None or v["ritardo_griglia_mediana_s"] is None
            else v["ritardo_griglia_mediana_s"] + response / 1000.0
            for s, v in seeds.items()
        }
        deployed[gpu] = {
            "fusa": merged,
            "risposta_ms": response,
            "fattore_realtime": None if not response else HOP_S * 1000.0 / response,
            "ritardo_allarme_s": aggregate(delays),
            "quota_calcolo": aggregate(
                {
                    s: None if d is None or response is None else response / 1000.0 / d
                    for s, d in delays.items()
                }
            ),
        }

    return {
        "nome": entry["nome"],
        "famiglia": entry["famiglia"],
        "versione": key,
        "gruppo": group,
        "catena_4_epoche": chain,
        "passi_totali": total_steps,
        "epoche_totali": total_steps / 5524.0,
        "semi": seeds,
        "passo_globale": aggregate({s: v["passo_globale"] for s, v in seeds.items()}),
        "params_adattatore": first["params_adattatore"],
        "params_testa": params_head,
        "costo_passo": {
            s["run_id"]: {
                gpu: None
                if (c := (costs.get(s["architettura"]) or {}).get(gpu)) is None
                else {
                    **asdict(c),
                    "per_run": dict((spread.get(s["architettura"]) or {}).get(gpu) or {}),
                }
                for gpu in GPUS
            }
            for s in chain
        },
        "ore_training_4_epoche": {gpu: chain_hours(chain, costs, gpu) for gpu in GPUS},
        "ore_al_checkpoint": {
            gpu: aggregate({s: v["ore_al_checkpoint"][gpu] for s, v in seeds.items()})
            for gpu in GPUS
        },
        "ore_gpu_consumate_s42": consumed_hours(training_jobs(logs, chain_runs), sacct),
        "memoria_training_stadio_finale": memory(training_jobs(logs, same_arch_runs), sacct),
        "memoria_training_catena": memory(training_jobs(logs, chain_arch_runs), sacct),
        "ritardo_griglia_mediana_s": aggregate(
            {s: v["ritardo_griglia_mediana_s"] for s, v in seeds.items()}
        ),
        "ritardo_griglia_p90_s": aggregate(
            {s: v["ritardo_griglia_p90_s"] for s, v in seeds.items()}
        ),
        "inferenza": deployed,
    }


def _fmt(value: Any, decimals: int = 2) -> str:
    """:return: a table cell, ``—`` when missing."""
    if value is None:
        return "—"
    if decimals == 0:
        return f"{int(round(float(value))):,}".replace(",", " ")
    return f"{float(value):.{decimals}f}"


def _agg(block: Mapping[str, Any], decimals: int = 2) -> str:
    """:return: ``mean ± sd`` (or the single value) of an :func:`aggregate` block."""
    if not block or block.get("media") is None:
        return "—"
    text = _fmt(block["media"], decimals)
    return text if block.get("sd") is None else f"{text} ± {_fmt(block['sd'], decimals)}"


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    """:return: the lines of a Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return lines


def _epochs(record: Mapping[str, Any]) -> str:
    """:return: the training epochs behind the reported version and its chain, e.g. ``3 / 4``."""
    seed = record["semi"][min(record["semi"])]
    return f"{seed['passi_catena_versione'] / 5524.0:g} / {record['epoche_totali']:g}"


def _chain(record: Mapping[str, Any]) -> str:
    """:return: the chain as experiment prefixes with their epochs, e.g. ``e3r 1 → e3m 2``."""
    return " → ".join(
        f"{s['run_id'].split('_', 1)[0]} {s['max_steps'] / 5524.0:g}"
        for s in record["catena_4_epoche"]
    )


def markdown(payload: Mapping[str, Any]) -> str:
    """Render the statistics as the Markdown document of ``--out``.

    :param payload: the JSON payload of :func:`build`.
    :return: the document text.
    """
    models = payload["modelli"]
    agreement = payload["profili"]["accordo"]
    out = [
        "# Cost statistics of the thesis models",
        "",
        f"{len(models)} models: ten trainings in the version selected by the mean of the "
        "UCF/MIVIA Protocol Means at seed 42 (OVAD, guard 30 s), and eight E3-S sequential "
        "combinations. The version is fixed across seeds. E3-SV and E3-SO are excluded "
        "(BRIEF chapter 4, rules 3 and 7). Only deployed inference forms are reported (rule 10).",
        "Seed 42 is reported first; variability is separate. Per-seed values follow the order "
        "42, 67, 104 where available.",
        "",
        "## 1. Selected checkpoint over the four epochs",
        "",
        "Global step = steps inherited by the reported version along its chain (each warm start "
        "or pre-adaptation receipt names the step of its parent) + the step the model selection "
        "kept in it (validation AUROC every 920 steps). One epoch = 5524 steps. A continuation "
        "inherits the LAST step of its parent; E3-S includes its one-epoch source.",
        "",
    ]
    rows = []
    for r in models.values():
        steps = " · ".join(_fmt(v["passo_globale"], 0) for v in r["semi"].values())
        local = " · ".join(_fmt(v["passo_locale"], 0) for v in r["semi"].values())
        auroc = " · ".join(_fmt(v["val_auroc"], 4) for v in r["semi"].values())
        rows.append(
            [
                r["nome"],
                _chain(r),
                _epochs(r),
                str(len(r["semi"])),
                local,
                steps,
                _fmt(r["passi_totali"], 0),
                auroc,
                _fmt(r["params_adattatore"], 0),
                _fmt(r["params_testa"], 0),
            ]
        )
    out += _table(
        [
            "Model",
            "Chain (seed 42, epochs)",
            "Epochs (version / chain)",
            "Seeds",
            "Local step",
            "Global step",
            "Chain steps",
            "Val AUROC",
            "Adapter params",
            "Head params",
        ],
        rows,
    )
    out += [
        "",
        "## 2. Training cost",
        "",
        "Seconds per step: median of the intervals between checkpoints without a validation in "
        "the SLURM logs, pooled over the runs of the same architecture; a validation pass is the "
        "median excess of the intervals containing one. Hours = steps × s/step + validations × "
        "s/validation, on ONE GPU type (never converted). *4 epochs* = the whole chain; *to "
        "checkpoint* = the chain up to the selected step. *Consumed* = elapsed time of the real "
        "seed-42 training jobs (resumes, migrations and final evaluation included, A100 + H100).",
        "",
    ]
    rows = []
    for r in models.values():
        last = r["catena_4_epoche"][-1]["run_id"]
        cost = r["costo_passo"][last]
        consumed = r["ore_gpu_consumate_s42"]
        rows.append(
            [
                r["nome"],
                _fmt((cost["A100"] or {}).get("s_per_step")),
                _fmt((cost["H100"] or {}).get("s_per_step")),
                _fmt(
                    (cost["A100"] or {}).get("validation_s") and cost["A100"]["validation_s"] / 60,
                    1,
                ),
                _fmt(
                    (cost["H100"] or {}).get("validation_s") and cost["H100"]["validation_s"] / 60,
                    1,
                ),
                _fmt(r["ore_training_4_epoche"]["A100"], 1),
                _fmt(r["ore_training_4_epoche"]["H100"], 1),
                _fmt(r["semi"][42]["ore_al_checkpoint"]["A100"], 1),
                _fmt(r["semi"][42]["ore_al_checkpoint"]["H100"], 1),
                _agg(r["ore_al_checkpoint"]["A100"], 1),
                _agg(r["ore_al_checkpoint"]["H100"], 1),
                f"{_fmt(consumed.get('A100'), 1)} + {_fmt(consumed.get('H100'), 1)}",
            ]
        )
    out += _table(
        [
            "Model",
            "s/step A100",
            "s/step H100",
            "validation min A100",
            "validation min H100",
            "4 ep. h A100",
            "4 ep. h H100",
            "to ckpt h A100 s42",
            "to ckpt h H100 s42",
            "to ckpt h A100 mean ± sd",
            "to ckpt h H100 mean ± sd",
            "consumed GPU-h (A100 + H100)",
        ],
        rows,
    )
    out += [
        "",
        "s/step and validation are those of the model's own (last) stage; a sequential model's "
        "chain also contains its one-epoch source, costed with the source architecture.",
        "",
        "## 3. Memory",
        "",
        "Training: peak GPU memory of the real training jobs (`gres/gpumem` of SLURM accounting, "
        "NVML: allocator cache and CUDA context included; not accounted on every node, hence the "
        "job count), micro-batch 2 in bf16, and host RSS (data-loader workers included). "
        "Inference: peak memory allocated by PyTorch in E5b, batch 1, deployed form.",
        "",
    ]
    rows = []
    for r in models.values():
        mem = r["memoria_training_stadio_finale"]
        inf = r["inferenza"]
        rows.append(
            [
                r["nome"],
                _fmt(mem["gpu_picco_gib"]),
                f"{mem['n_job_gpu']}/{mem['n_job']}",
                _fmt(mem["rss_picco_gib"], 1),
                _fmt((inf["A100"]["fusa"] or {}).get("vram_picco_gib"), 3),
            ]
        )
    out += _table(
        [
            "Model",
            "train GPU peak GiB",
            "jobs (with gpumem / all)",
            "train host RSS GiB",
            "inference GiB (deployed)",
        ],
        rows,
    )
    out += [
        "",
        "## 4. Response time and detection delay",
        "",
        f"`alarm delay = grid delay + compute + decoding`. Compute: E5b, batch 1, bf16, deployed "
        f"form (reparam merged; partially merged for the combinations, ADR-064). Decoding: "
        f"{DECODE_MS} ms/window (E0d). Grid delay: median over the 276 test events, intersection "
        f"rule (ADR-061), seed 42 followed by mean ± sd over the available seeds. Real-time factor = "
        f"hop ({HOP_S:g} s) / response.",
        "",
    ]
    rows = []
    for r in models.values():
        a, h = r["inferenza"]["A100"], r["inferenza"]["H100"]
        rows.append(
            [
                r["nome"],
                (a["fusa"] or {}).get("fusione", "—"),
                _fmt((a["fusa"] or {}).get("latenza_ms"), 1),
                _fmt((h["fusa"] or {}).get("latenza_ms"), 1),
                _fmt(a["risposta_ms"], 1),
                _fmt(h["risposta_ms"], 1),
                _fmt(r["semi"][42]["ritardo_griglia_mediana_s"], 3),
                _fmt(r["semi"][42]["ritardo_griglia_p90_s"], 3),
                _fmt(a["ritardo_allarme_s"]["valori"].get("42"), 3),
                _fmt(h["ritardo_allarme_s"]["valori"].get("42"), 3),
                _agg(r["ritardo_griglia_mediana_s"], 3),
                _agg(r["ritardo_griglia_p90_s"], 3),
                _agg(a["ritardo_allarme_s"], 3),
                _agg(h["ritardo_allarme_s"], 3),
                _fmt(a["fattore_realtime"], 1),
            ]
        )
    out += _table(
        [
            "Model",
            "merge",
            "compute ms A100",
            "compute ms H100",
            "response ms A100",
            "response ms H100",
            "grid median s42 s",
            "grid p90 s42 s",
            "alarm s42 A100 s",
            "alarm s42 H100 s",
            "grid median mean ± sd s",
            "grid p90 mean ± sd s",
            "alarm A100 mean ± sd s",
            "alarm H100 mean ± sd s",
            "RTF A100",
        ],
        rows,
    )
    out += ["", "## 5. Provenance and caveats", ""]
    for gpu, block in agreement.items():
        if block.get("scarto_massimo") is not None:
            out.append(
                f"* **{gpu}**: the combination profile `{block['secondo']}` and the reference "
                f"`{block['riferimento']}` share {', '.join(block['varianti'])}; maximum latency "
                f"deviation {block['scarto_massimo'] * 100:.2f}%."
            )
    out += [f"* {note}" for note in payload["avvertenze"]]
    return "\n".join(out) + "\n"


CAVEATS: tuple[str, ...] = (
    "The step cost of an architecture is the median over all its runs; per-run medians agree "
    "within 0.5%, except the two one-epoch probes at seed 42 (31 August, H100): 1.74 and 1.40 "
    "s/step against 1.52 and 1.17 of every later run (`per_run` in the JSON).",
    "The reported version is read from OVAD guard-30 selezione_versioni.json: mean of the "
    "UCF/MIVIA Protocol Means at seed 42, on the TEST sets. Checkpoint selection within each "
    "version remains validation AUROC. The same version is used for every seed.",
    "Seed availability is listed per model. Training costs are pooled per architecture and GPU.",
    "A100 and H100 hours are two measurements, never a conversion: where a GPU never ran an "
    "architecture long enough (5 intervals), its cell is empty.",
    "Training GPU memory is NVML (allocator cache and context included) and comes from the "
    "nodes that account `gres/gpumem`; inference memory is the PyTorch allocated peak. They are "
    "not the same quantity.",
    "The response time is a composition of measures taken on different machines (model on "
    "A100/H100, decoder on an H100 node) without scheduling or contention.",
)


def build(args: argparse.Namespace) -> dict[str, Any]:
    """Compute every statistic.

    :param args: the parsed command line.
    :return: the JSON payload.
    """
    cells = find_cells(args.runs_dir)
    models, chosen, selection_path = selected_models(args, cells)
    logs = TL.read_logs(args.log_slurm)
    pools = architecture_pools(cells)
    costs = costs_by_architecture(logs, pools)
    spread = {
        arch: {gpu: per_run_step_costs(logs, runs, gpu) for gpu in GPUS}
        for arch, runs in pools.items()
    }
    sacct = parse_sacct(args.sacct.read_text(encoding="utf-8")) if args.sacct else {}
    reference = {
        gpu: load_profile_rows(folder) for gpu, folder in zip(GPUS, args.profilo, strict=True)
    }
    combos = {
        gpu: load_profile_rows(folder)
        for gpu, folder in zip(GPUS, args.profilo_combinazioni, strict=True)
    }
    profiles = {gpu: [p for p in (reference[gpu], combos[gpu]) if p] for gpu in GPUS}
    agreement = {
        gpu: {
            "riferimento": reference[gpu].get("run_id"),
            "secondo": combos[gpu].get("run_id"),
            **profile_agreement(reference[gpu], combos[gpu]),
        }
        for gpu in GPUS
    }
    records = {
        key: describe_model(
            key,
            entry,
            runs_dir=args.runs_dir,
            cells=cells,
            logs=logs,
            costs=costs,
            spread=spread,
            sacct=sacct,
            profiles=profiles,
            cache_dir=args.cache_ablation if args.cache_ablation.is_dir() else None,
        )
        for key, entry in models.items()
    }
    return {
        "modelli": records,
        "versioni": {g: c["versione"] for g, c in chosen.items()},
        "profili": {
            "riferimento": {g: p.get("run_id") for g, p in reference.items()},
            "combinazioni": {g: p.get("run_id") for g, p in combos.items()},
            "accordo": agreement,
        },
        "sorgenti": {
            "selezione_versioni": str(selection_path),
            "selezione_versioni_sha256": file_sha256(selection_path),
            "log_slurm": [str(p) for p in args.log_slurm],
            "n_log": len(logs),
            "sacct": None if args.sacct is None else str(args.sacct),
            "n_job_sacct": len(sacct),
        },
        "decodifica_ms": DECODE_MS,
        "hop_s": HOP_S,
        "avvertenze": list(CAVEATS),
    }


def csv_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """:return: one flat row per model, the columns of the Markdown tables."""
    rows = []
    for r in payload["modelli"].values():
        last = r["catena_4_epoche"][-1]["run_id"]
        row: dict[str, Any] = {
            "modello": r["nome"],
            "versione": r["versione"],
            "epoche_catena": r["epoche_totali"],
            "semi": " ".join(map(str, r["semi"])),
            "passo_locale_s42": r["semi"][42]["passo_locale"],
            "passo_globale_s42": r["semi"][42]["passo_globale"],
            "val_auroc_s42": r["semi"][42]["val_auroc"],
            "passo_globale_media": r["passo_globale"]["media"],
            "passo_globale_semi": " ".join(str(v["passo_globale"]) for v in r["semi"].values()),
            "params_adattatore": r["params_adattatore"],
            "params_testa": r["params_testa"],
        }
        for gpu in GPUS:
            cost = r["costo_passo"][last][gpu] or {}
            inf = r["inferenza"][gpu]
            row |= {
                f"s_passo_{gpu}": cost.get("s_per_step"),
                f"s_validazione_{gpu}": cost.get("validation_s"),
                f"ore_4_epoche_{gpu}": r["ore_training_4_epoche"][gpu],
                f"ore_al_checkpoint_{gpu}": r["ore_al_checkpoint"][gpu]["media"],
                f"ore_al_checkpoint_s42_{gpu}": r["semi"][42]["ore_al_checkpoint"][gpu],
                f"calcolo_ms_{gpu}": (inf["fusa"] or {}).get("latenza_ms"),
                f"risposta_ms_{gpu}": inf["risposta_ms"],
                f"ritardo_allarme_s_{gpu}": inf["ritardo_allarme_s"]["media"],
                f"ritardo_allarme_s42_s_{gpu}": inf["ritardo_allarme_s"]["valori"].get("42"),
            }
        row |= {
            "ritardo_griglia_mediana_s": r["ritardo_griglia_mediana_s"]["media"],
            "ritardo_griglia_p90_s": r["ritardo_griglia_p90_s"]["media"],
            "vram_training_gib": r["memoria_training_stadio_finale"]["gpu_picco_gib"],
            "rss_training_gib": r["memoria_training_stadio_finale"]["rss_picco_gib"],
            "vram_inferenza_gib": (r["inferenza"]["A100"]["fusa"] or {}).get("vram_picco_gib"),
            "fusione": (r["inferenza"]["A100"]["fusa"] or {}).get("fusione"),
            "ore_gpu_consumate_A100": r["ore_gpu_consumate_s42"].get("A100"),
            "ore_gpu_consumate_H100": r["ore_gpu_consumate_s42"].get("H100"),
        }
        rows.append(row)
    return rows


def list_jobs(args: argparse.Namespace) -> list[str]:
    """:return: the ids of every training job of the thesis models' architectures."""
    cells = find_cells(args.runs_dir)
    models, _, _ = selected_models(args, cells)
    archs = set()
    for key, entry in models.items():
        group = entry.get("gruppo")
        tail = VERSION_GROUPS[group][-1] if group else key
        archs |= {s["architettura"] for s in ancestry(args.runs_dir, cells[tail][42].name)}
    runs = [
        f.name for k, by_seed in cells.items() if architecture(k) in archs for f in by_seed.values()
    ]
    return sorted(training_jobs(TL.read_logs(args.log_slurm), runs), key=int)


def build_parser() -> argparse.ArgumentParser:
    """:return: the CLI parser."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    p.add_argument(
        "--selezione-versioni",
        type=Path,
        help="published OVAD guard-30 choices; default under --runs-dir",
    )
    p.add_argument(
        "--log-slurm",
        type=Path,
        action="append",
        default=None,
        help="SLURM log folder (repeatable); default ../hpc/log",
    )
    p.add_argument(
        "--sacct",
        type=Path,
        default=None,
        help="`sacct -P` dump of the training jobs (JobIDRaw, Partition, NodeList, State, "
        "ElapsedRaw, MaxRSS, TRESUsageInMax); without it memory stays empty and the consumed "
        "hours come from the logs",
    )
    p.add_argument(
        "--profilo",
        type=Path,
        nargs=2,
        default=[ROOT / "runs/e5b3_efficienza_gpuq", ROOT / "runs/e5b3_efficienza_aiq"],
        metavar=("A100", "H100"),
        help="reference inference profiles, the leaderboards' ones",
    )
    p.add_argument(
        "--profilo-combinazioni",
        type=Path,
        nargs=2,
        default=[ROOT / "runs/e5b4_combo_gpuq", ROOT / "runs/e5b4_combo_aiq"],
        metavar=("A100", "H100"),
        help="profiles with the merged combinations (E5b4, 18 October 2026)",
    )
    p.add_argument("--cache-ablation", type=Path, default=ROOT / "runs/analisi_e3_20260907")
    p.add_argument("--out", type=Path, default=ROOT / "runs/statistiche_modelli")
    p.add_argument(
        "--elenca-job", action="store_true", help="print the training job ids for sacct and exit"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Compute and write the statistics.

    :param argv: command-line arguments.
    :return: the exit code.
    """
    args = build_parser().parse_args(argv)
    setup_logging()
    args.log_slurm = args.log_slurm or [ROOT.parent / "hpc" / "log"]
    if args.elenca_job:
        print("\n".join(list_jobs(args)))
        return 0
    payload = build(args)
    ensure_dir(args.out)
    write_text_atomic(
        args.out / "statistiche_modelli.json",
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
    )
    write_text_atomic(args.out / "statistiche_modelli.md", markdown(payload))
    rows = csv_rows(payload)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_text_atomic(args.out / "statistiche_modelli.csv", buffer.getvalue())
    log.info("%d models written to %s", len(payload["modelli"]), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
