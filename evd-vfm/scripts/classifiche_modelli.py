#!/usr/bin/env python
"""Leaderboards of every model on every metric, per class and per video.

    python scripts/classifiche_modelli.py
    python scripts/classifiche_modelli.py --profilo runs/e5b3_efficienza_gpuq

Writes ``classifiche.json`` (everything), ``classifiche.md`` (readable), ``per_video.csv``
(seed averages), ``per_video_run.csv`` (single runs), ``cross_dataset_mivia.csv`` (the
MIVIA-Crime transfer) and a ``per_video.csv`` in every run. It ranks the models on every
metric and feeds the presentation tables without copying any number by hand (ADR-014).

It does not train, evaluate or use the GPU: it reads artifacts already on disk, and every
recomputed number is compared with the one the run wrote.

Sources:

* ``runs/{run_id}/metrics.json``: the run receipt (AUC, AP, P/R/F1 at the three
  granularities, threshold, per class, budget);
* ``runs/{run_id}/scores.jsonl``: one score per window, the only source of the per-video
  view;
* the intersection-rule latency (ADR-061) and the strata (ADR-060), in ``latenza.json`` /
  ``strata.json`` next to the run or in the ``runs/analisi_e3_20260907/{run_id}.json``
  cache for the ablation cells (same format, different container);
* an inference profile (``efficiency_inference.json``), if present, for the compute time
  per window;
* ``runs/{run_id}__mivia/metrics.json``: the evaluation on the laboratory private test set
  (ADR-066). It becomes a column, never a row; only seed 42 is reported
  (``--semi-mivia``), and the delta is paired on the seed (:func:`compose_mivia`).

One version per model: the versions of the same training extended in continuation
(:data:`VERSION_GROUPS`: one, three and four epochs; three and four for the E3-C
combinations) appear once, under the plain name, chosen by :func:`choose_versions` (on the
test set, not on validation, and declared in ``classifiche.md``); ``--tutte-le-versioni``
keeps them all.

The registry is not used: it has one row per execution, while the artifacts in
``runs/{run_id}/`` are those of the execution that counts.

``trainable_params`` means two things (ADR-057); the budget-parity number is
``adapt.n_trainable`` (``params_adattatore`` here), and ``params_testa`` is derived by
difference. Latency uses the intersection rule (ADR-061): with it, events without a
measurable latency must equal missed events, and the script checks it; the historical
"endpoint inside" rule is reported alongside, flagged.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evd.annotations.schema import TemporalAnnotation  # noqa: E402
from evd.eval import metrics as M  # noqa: E402
from evd.eval import strata as S  # noqa: E402
from evd.exp import training_log as TL  # noqa: E402
from evd.exp.rilettura import reread_run  # noqa: E402
from evd.exp.tracker import git_info  # noqa: E402
from evd.utils.hashing import file_sha256  # noqa: E402
from evd.utils.io import ensure_dir, read_json, write_json, write_text_atomic  # noqa: E402
from evd.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger(__name__)

#: The presentation models, in display order, with names free of internal acronyms. The
#: key is the ``adapt`` field of the ``run_id``, which also separates the two frozen probes.
MODELS: dict[str, dict[str, str]] = {
    "frozen-linear": {"nome": "Linear probe", "famiglia": "frozen"},
    "frozen-attentive": {"nome": "Attentive probe", "famiglia": "frozen"},
    "lora-r8-qkvomlp": {"nome": "LoRA r=8", "famiglia": "reparam"},
    "lora-r16-qkvomlp": {"nome": "LoRA r=16", "famiglia": "reparam"},
    "lora-r32-qkvomlp": {"nome": "LoRA r=32", "famiglia": "reparam"},
    "lora-r64-qkvomlp": {"nome": "LoRA r=64", "famiglia": "reparam"},
    "dora-r32-qkvomlp": {"nome": "DoRA r=32", "famiglia": "reparam"},
    "dora-r8-qkvomlp": {"nome": "DoRA r=8", "famiglia": "reparam"},
    "dora-r16-qkvomlp": {"nome": "DoRA r=16", "famiglia": "reparam"},
    "dora-r64-qkvomlp": {"nome": "DoRA r=64", "famiglia": "reparam"},
    "vptdeep-n8": {"nome": "VPT-deep n=8", "famiglia": "prompt"},
    "vptdeep-n64": {"nome": "VPT-deep n=64", "famiglia": "prompt"},
    "vptdeep-n192": {"nome": "VPT-deep n=192", "famiglia": "prompt"},
    "vptdeep-n576": {"nome": "VPT-deep n=576", "famiglia": "prompt"},
    "vptshallow-n8": {"nome": "VPT-shallow n=8", "famiglia": "prompt"},
    "vptshallow-n64": {"nome": "VPT-shallow n=64", "famiglia": "prompt"},
    "vptshallow-n192": {"nome": "VPT-shallow n=192", "famiglia": "prompt"},
    "vptshallow-n576": {"nome": "VPT-shallow n=576", "famiglia": "prompt"},
    # --- E3-M: the same six models at THREE epochs (F12, ADR-062). The key carries the
    # experiment prefix because the `run_id` has no budget field
    # (:data:`EXTENDED_EXPERIMENTS`), and the display name states the epochs.
    "e3m:frozen-linear": {"nome": "Linear probe (3 ep.)", "famiglia": "frozen"},
    "e3m:frozen-attentive": {"nome": "Attentive probe (3 ep.)", "famiglia": "frozen"},
    "e3m:vptshallow-n8": {"nome": "VPT-shallow n=8 (3 ep.)", "famiglia": "prompt"},
    "e3m:vptdeep-n8": {"nome": "VPT-deep n=8 (3 ep.)", "famiglia": "prompt"},
    "e3m:lora-r32-qkvomlp": {"nome": "LoRA r=32 (3 ep.)", "famiglia": "reparam"},
    "e3m:dora-r32-qkvomlp": {"nome": "DoRA r=32 (3 ep.)", "famiglia": "reparam"},
    # --- E3-C: reparam + prompt combinations trained TOGETHER (F13). A fourth family: filing
    # them under `reparam` or `prompt` would graphically decide the question C7 asks. They are
    # not at parameter parity with the other rows, and this is declared.
    "e3c:lora-r32-vptshallow-n8": {
        "nome": "LoRA r=32 + VPT-shallow n=8",
        "famiglia": "combinazione",
    },
    "e3c:lora-r32-vptdeep-n8": {"nome": "LoRA r=32 + VPT-deep n=8", "famiglia": "combinazione"},
    "e3c:dora-r32-vptshallow-n8": {
        "nome": "DoRA r=32 + VPT-shallow n=8",
        "famiglia": "combinazione",
    },
    "e3c:dora-r32-vptdeep-n8": {"nome": "DoRA r=32 + VPT-deep n=8", "famiglia": "combinazione"},
    # --- E3-S: SEQUENTIAL combinations (F13-bis, ADR-065). Same two families and three-cycle
    # budget as E3-C; the ORDER changes, to separate "training together" from "adding the two
    # priors". The family is `sequenziale`, not `combinazione`. The four `congela` cells carry
    # the SAME `adapt` as the four E3-C cells: without the :data:`EXTENDED_EXPERIMENTS` prefix
    # they would be the same row. `fondi` = the reparam is already merged into the backbone and
    # VPT is trained on top; `congela` = the prompts are pre-trained and frozen and the reparam
    # is trained.
    "e3s:vptshallow-n8-su-lora-r32": {
        "nome": "VPT-shallow n=8 on fused LoRA r=32",
        "famiglia": "sequenziale",
    },
    "e3s:vptdeep-n8-su-lora-r32": {
        "nome": "VPT-deep n=8 on fused LoRA r=32",
        "famiglia": "sequenziale",
    },
    "e3s:vptshallow-n8-su-dora-r32": {
        "nome": "VPT-shallow n=8 on fused DoRA r=32",
        "famiglia": "sequenziale",
    },
    "e3s:vptdeep-n8-su-dora-r32": {
        "nome": "VPT-deep n=8 on fused DoRA r=32",
        "famiglia": "sequenziale",
    },
    "e3s:lora-r32-vptshallow-n8": {
        "nome": "LoRA r=32 on frozen VPT-shallow n=8",
        "famiglia": "sequenziale",
    },
    "e3s:lora-r32-vptdeep-n8": {
        "nome": "LoRA r=32 on frozen VPT-deep n=8",
        "famiglia": "sequenziale",
    },
    "e3s:dora-r32-vptshallow-n8": {
        "nome": "DoRA r=32 on frozen VPT-shallow n=8",
        "famiglia": "sequenziale",
    },
    "e3s:dora-r32-vptdeep-n8": {
        "nome": "DoRA r=32 on frozen VPT-deep n=8",
        "famiglia": "sequenziale",
    },
}

# E3-SV stays distinct from E3-S: the prior checkpoint changes, not the family.
MODELS.update(
    {
        key.replace("e3s:", "e3sv:", 1): {**value, "nome": value["nome"] + " (prior val-best)"}
        for key, value in list(MODELS.items())
        if key.startswith("e3s:")
    }
)


# E3-SO: val-best prior at one epoch, second stage at two; three in total.
MODELS.update(
    {
        "e3so:vptshallow-n8-su-dora-r16": {
            "nome": "VPT-shallow n=8 on fused DoRA r=16 (1+2 ep.)",
            "famiglia": "sequenziale",
        },
        "e3so:dora-r16-vptshallow-n8": {
            "nome": "DoRA r=16 on frozen VPT-shallow n=8 (1+2 ep.)",
            "famiglia": "sequenziale",
        },
        "e3so:vptdeep-n8-su-dora-r16": {
            "nome": "VPT-deep n=8 on fused DoRA r=16 (1+2 ep.)",
            "famiglia": "sequenziale",
        },
        "e3so:dora-r16-vptdeep-n8": {
            "nome": "DoRA r=16 on frozen VPT-deep n=8 (1+2 ep.)",
            "famiglia": "sequenziale",
        },
    }
)

# E3-M4 / E3-C4 (13 October 2026): one more cycle in continuation from E3-M and E3-C, matching
# the total compute of E3-S (1+3 epochs). Same family, the name states the epochs. ⚠️ The
# fourth-epoch cell alone is not "the best over four epochs": that is the maximum validation
# AUROC over the chain of runs, and it is composed outside this file.
MODELS.update(
    {
        **{
            key.replace("e3m:", "e3m4:", 1): {
                **value,
                "nome": value["nome"].replace("(3 ep.)", "(4 ep.)"),
            }
            for key, value in list(MODELS.items())
            if key.startswith("e3m:")
        },
        **{
            key.replace("e3c:", "e3c4:", 1): {**value, "nome": value["nome"] + " (4 ep.)"}
            for key, value in list(MODELS.items())
            if key.startswith("e3c:")
        },
    }
)

#: Which inference-profile variant gives each model its compute time. A model missing
#: here, or a variant missing from the profile, leaves the cell empty: no interpolation.
PROFILE_VARIANT: dict[str, str] = {
    "frozen-linear": "frozen-linear",
    "frozen-attentive": "frozen",
    "lora-r8-qkvomlp": "lora-r8",
    "lora-r16-qkvomlp": "lora-r16",
    "lora-r32-qkvomlp": "lora-r32",
    "lora-r64-qkvomlp": "lora-r64",
    "dora-r32-qkvomlp": "dora-r32",
    "vptdeep-n8": "vptdeep-n8",
    "vptdeep-n64": "vptdeep-n64",
    "vptdeep-n192": "vptdeep-n192",
    "vptdeep-n576": "vptdeep-n576",
    "vptshallow-n8": "vptshallow-n8",
    "vptshallow-n64": "vptshallow-n64",
    "vptshallow-n192": "vptshallow-n192",
    "vptshallow-n576": "vptshallow-n576",
    # E3-M does not change the architecture, only the number of training steps: at inference a
    # three-epoch cell is the same model as its one-epoch cell, so the profile row is the same.
    # E3-C combinations are absent (E5b did not profile them), and so is E3-S (a `fondi` cell
    # is an already merged backbone plus eight tokens, not the `vptshallow-n8` row).
    "e3m:frozen-linear": "frozen-linear",
    "e3m:frozen-attentive": "frozen",
    "e3m:vptshallow-n8": "vptshallow-n8",
    "e3m:vptdeep-n8": "vptdeep-n8",
    "e3m:lora-r32-qkvomlp": "lora-r32",
    "e3m:dora-r32-qkvomlp": "dora-r32",
    # E3-M4: one more cycle, same architecture.
    "e3m4:frozen-linear": "frozen-linear",
    "e3m4:frozen-attentive": "frozen",
    "e3m4:vptshallow-n8": "vptshallow-n8",
    "e3m4:vptdeep-n8": "vptdeep-n8",
    "e3m4:lora-r32-qkvomlp": "lora-r32",
    "e3m4:dora-r32-qkvomlp": "dora-r32",
}

#: Corresponding merged variant, where it exists. Merging is the condition of the claim
#: "reparam costs nothing at inference": without it, reparam is the most expensive family.
MERGED_VARIANT: dict[str, str] = {
    "lora-r8-qkvomlp": "lora-r8-fusa",
    "lora-r16-qkvomlp": "lora-r16-fusa",
    "lora-r32-qkvomlp": "lora-r32-fusa",
    "lora-r64-qkvomlp": "lora-r64-fusa",
    "dora-r32-qkvomlp": "dora-r32-fusa",
    "e3m:lora-r32-qkvomlp": "lora-r32-fusa",
    "e3m:dora-r32-qkvomlp": "dora-r32-fusa",
    "e3m4:lora-r32-qkvomlp": "lora-r32-fusa",
    "e3m4:dora-r32-qkvomlp": "dora-r32-fusa",
    # No row for E3-C: `can_merge()` is false on every composition (ADR-064). A partial merge
    # exists but E5b did not measure it, so the column stays empty.
}

#: Milliseconds per window spent decoding the video, measured with 8 workers (25.10
#: windows/s, augmentation off, 512 windows). A system constant, the same for every model.
DECODE_MS: float = 40.8

#: The experiment prefixes producing thesis numbers. The LR pilots share the ``adapt``
#: field and the seed of the real cells but run on another protocol with a trial budget.
EXPERIMENTS: frozenset[str] = frozenset(
    {"e2c", "e2d", "e3p", "e3r", "e3m", "e3m4", "e3c", "e3c4", "e3s", "e3sv", "e3so"}
)

#: Extended-budget (E3-M, three epochs) and composite (E3-C) blocks. Their key cannot be
#: the ``adapt`` alone: ``e3m_..._lora-r32-qkvomlp_s42`` and ``e3r_..._lora-r32-qkvomlp_s42``
#: share adapter and seed (only the budget differs), so for these experiments the key is
#: ``{exp}:{adapt}``. E3-S is here for the same reason (its ``congela`` cells share the
#: E3-C ``adapt``). Not every E3-S cell has three seeds; the ``n`` column says so.
EXTENDED_EXPERIMENTS: frozenset[str] = frozenset(
    {"e3m", "e3m4", "e3c", "e3c4", "e3s", "e3sv", "e3so"}
)

#: Versions of the SAME training extended in continuation, shortest first: one epoch, then
#: three (E3-M, warm start from the last step), then four (E3-M4); the E3-C combinations
#: at three epochs, then four (E3-C4). The leaderboards report one per group, under the
#: name of the first (decision of 14 October 2026: the added epochs are an extension of the
#: training, and only the best result counts). ``--tutte-le-versioni`` keeps them all.
VERSION_GROUPS: dict[str, tuple[str, ...]] = {
    **{
        base: (base, "e3m:" + base, "e3m4:" + base)
        for base in (
            "frozen-linear",
            "frozen-attentive",
            "vptshallow-n8",
            "vptdeep-n8",
            "lora-r32-qkvomlp",
            "dora-r32-qkvomlp",
        )
    },
    **{
        key: (key, key.replace("e3c:", "e3c4:", 1))
        for key in (
            "e3c:lora-r32-vptshallow-n8",
            "e3c:lora-r32-vptdeep-n8",
            "e3c:dora-r32-vptshallow-n8",
            "e3c:dora-r32-vptdeep-n8",
        )
    },
}

#: Training epochs behind a key, by experiment prefix; a key without a prefix is one epoch.
EPOCHS: dict[str, int] = {"e3m": 3, "e3c": 3, "e3m4": 4, "e3c4": 4}

#: The :func:`choose_versions` criterion, as read in ``classifiche.md``.
VERSION_CRITERION: str = (
    "among the versions of the same training extended in continuation (one, three and four "
    "epochs; three and four for the combinations), the one with the higher seed-42 chunk AUC "
    "on the FS-UCF-Crime **test**; tie: fewer epochs. The version is fixed for every seed. "
    "⚠️ A selection on the test set, not the protocol's validation criterion"
)


def version_epochs(key: str) -> int:
    """:return: the training epochs of a :data:`MODELS` key (1 without a known prefix)."""
    prefix, separator, _ = key.partition(":")
    return EPOCHS.get(prefix, 1) if separator else 1


def choose_versions(cells: Mapping[str, Mapping[int, Path]]) -> dict[str, dict[str, Any]]:
    """Choose the reported version of every :data:`VERSION_GROUPS` group.

    The only place of the rule: leaderboards, per class and chunk ranking read it through
    :func:`active_models`; :mod:`valuta_ovad` applies the same groups with its own metric.

    :param cells: ``{key: {seed: folder}}`` as from :func:`find_cells`.
    :return: ``{group: {"versione": key, "epoche": n, "chunk_auc_s42": {key: auc}}}`` for
        the groups with at least one seed-42 cell; the maximum seed-42 chunk AUC wins, and
        on a tie the version with fewer epochs.
    """
    chosen: dict[str, dict[str, Any]] = {}
    for group, keys in VERSION_GROUPS.items():
        scores = {
            key: float(read_json(cells[key][42] / "metrics.json")["chunk_auc"])
            for key in keys
            if 42 in cells.get(key, {})
        }
        if scores:
            best = max(scores, key=lambda k: (scores[k], -keys.index(k)))
            chosen[group] = {
                "versione": best,
                "epoche": version_epochs(best),
                "chunk_auc_s42": scores,
            }
    return chosen


def reported_models(
    tutte_le_versioni: bool = False, chosen: Mapping[str, str] | None = None
) -> dict[str, dict[str, str]]:
    """Return the :data:`MODELS` entries entering the leaderboards, with their display name.

    :param tutte_le_versioni: ``True`` keeps every version, named with its epochs, i.e.
        :data:`MODELS` as is.
    :param chosen: ``{group: reported key}`` (:func:`choose_versions`); a group missing here
        reports its first version.
    :return: ``{key: {"nome": …, "famiglia": …}}`` in :data:`MODELS` order, with the reported
        version in place of the group's first row and with that row's name.
    :raises ValueError: if two reported entries share a name.
    """
    alternatives = {key for keys in VERSION_GROUPS.values() for key in keys[1:]}
    active: dict[str, dict[str, str]] = {}
    for key, entry in MODELS.items():
        if tutte_le_versioni:
            active[key] = dict(entry)
        elif key in VERSION_GROUPS:
            reported = (chosen or {}).get(key, key)
            active[reported] = {**MODELS[reported], "nome": entry["nome"]}
        elif key not in alternatives:
            active[key] = dict(entry)
    names = [v["nome"] for v in active.values()]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"two reported models with the same name: {', '.join(duplicates)}")
    return active


def active_models(
    cells: Mapping[str, Mapping[int, Path]], tutte_le_versioni: bool = False
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
    """:return: the reported models (:func:`reported_models`) and the version choices
    behind them (:func:`choose_versions`; empty with ``tutte_le_versioni``)."""
    chosen = {} if tutte_le_versioni else choose_versions(cells)
    reported = {group: choice["versione"] for group, choice in chosen.items()}
    return reported_models(tutte_le_versioni, reported), chosen


#: Grid hop in seconds: the budget within which a window must be decoded and evaluated
#: for the system to keep up with the stream.
HOP_S: float = 4.0


def model_key(run_id: str) -> str | None:
    """Translate a ``run_id`` string into a :data:`MODELS` key.

    The only run_id -> model translation of the file, shared by :func:`find_cells` (run
    folders) and :func:`cost_per_step` (SLURM logs), which must agree by construction.

    :param run_id: the run folder name, or the run named in a log.
    :return: the :data:`MODELS` key, or ``None`` if the ``run_id`` is outside the matrix
        (experiment outside :data:`EXPERIMENTS`, other backbone, non-numeric seed, model not
        in :data:`MODELS`).
    """
    if "_vjepa21l_" not in run_id:
        return None
    experiment = run_id.split("_", 1)[0]
    if experiment not in EXPERIMENTS:
        return None
    adapt, _, seed = run_id.split("_vjepa21l_", 1)[1].rpartition("_s")
    if not seed.isdigit():
        return None
    key = f"{experiment}:{adapt}" if experiment in EXTENDED_EXPERIMENTS else adapt
    return key if key in MODELS else None


def _nan(valore: Any) -> bool:
    """:return: ``True`` if the value is a float NaN."""
    return isinstance(valore, float) and math.isnan(valore)


def _float_o_none(valore: Any) -> float | None:
    """:return: the value as ``float``, or ``None`` if missing.

    ``None`` and ``0.0`` differ ("not measured" vs "measured and null").
    """
    return None if valore is None else float(valore)


def _get(block: Any, *keys: str) -> Any:
    """Read a nested path, returning ``None`` if a link is missing.

    :param block: the starting structure.
    :param chiavi: the keys to traverse, in order.
    :return: the value, or ``None`` if the path does not exist.
    """
    current = block
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _missed(cella: Mapping[str, Any]) -> float | None:
    """:return: the annotated events that no alarm intersected."""
    ev = _get(cella["m"], "event_prf")
    if not isinstance(ev, Mapping):
        return None
    return float(int(ev["n_eventi"]) - int(ev["n_eventi_rilevati"]))


#: Leaderboard metrics: key, English slide label, direction (``max`` = higher is better),
#: decimals, and how to extract it from a cell. The direction is part of the definition.
METRICS: tuple[tuple[str, str, str, int, Callable[[dict[str, Any]], Any]], ...] = (
    ("chunk_auc", "Chunk AUC", "max", 4, lambda c: _get(c["m"], "chunk_auc")),
    ("chunk_ap", "Chunk AP", "max", 4, lambda c: _get(c["m"], "chunk_ap")),
    (
        "chunk_auc_anomalous",
        "Chunk AUC (anomalous videos)",
        "max",
        4,
        lambda c: _get(c["m"], "chunk_auc_anomalous"),
    ),
    (
        "chunk_ap_anomalous",
        "Chunk AP (anomalous videos)",
        "max",
        4,
        lambda c: _get(c["m"], "chunk_ap_anomalous"),
    ),
    (
        "chunk_precision",
        "Chunk precision",
        "max",
        4,
        lambda c: _get(c["m"], "chunk_prf", "precision"),
    ),
    ("chunk_recall", "Chunk recall", "max", 4, lambda c: _get(c["m"], "chunk_prf", "recall")),
    ("chunk_f1", "Chunk F1", "max", 4, lambda c: _get(c["m"], "chunk_prf", "f1")),
    (
        "video_precision",
        "Video precision",
        "max",
        4,
        lambda c: _get(c["m"], "video_prf", "precision"),
    ),
    ("video_recall", "Video recall", "max", 4, lambda c: _get(c["m"], "video_prf", "recall")),
    ("video_f1", "Video F1", "max", 4, lambda c: _get(c["m"], "video_prf", "f1")),
    (
        "event_precision",
        "Event precision",
        "max",
        4,
        lambda c: _get(c["m"], "event_prf", "precision"),
    ),
    ("event_recall", "Event recall", "max", 4, lambda c: _get(c["m"], "event_prf", "recall")),
    ("event_f1", "Event F1", "max", 4, lambda c: _get(c["m"], "event_prf", "f1")),
    ("eventi_mancati", "Missed events (of 276)", "min", 1, _missed),
    (
        "latenza_griglia_mediana_s",
        "Grid delay, median (s)",
        "min",
        3,
        lambda c: _get(c["lat"], "mediana_s"),
    ),
    ("latenza_griglia_p90_s", "Grid delay, p90 (s)", "min", 3, lambda c: _get(c["lat"], "p90_s")),
    ("soglia", "Operating threshold", "max", 4, lambda c: _get(c["m"], "threshold", "value")),
    (
        "passo_selezionato",
        "Selected step",
        "max",
        0,
        lambda c: _get(c["m"], "model_selection", "step"),
    ),
    (
        "val_auroc",
        "Validation AUROC at selection",
        "max",
        4,
        lambda c: _get(c["m"], "model_selection", "value"),
    ),
    (
        "params_adattatore",
        "Adapter parameters",
        "min",
        0,
        lambda c: _get(c["m"], "adapt", "n_trainable"),
    ),
)

#: Metrics of the summary table; the others stay in the JSON and in the per-metric section.
SUMMARY_METRICS: tuple[str, ...] = (
    "chunk_auc",
    "chunk_ap",
    "chunk_auc_anomalous",
    "chunk_ap_anomalous",
    "chunk_f1",
    "video_f1",
    "event_f1",
    "latenza_griglia_mediana_s",
)


def aggregate(valori: Mapping[int, Any]) -> dict[str, Any]:
    """Return mean, spread and count over the seeds.

    :param valori: value per seed; ``None`` and ``NaN`` count as missing.
    :return: ``media``, ``sd``, ``n`` and the per-seed values.

    With a single seed ``sd`` is ``None``, never ``0``.
    """
    present = {s: float(v) for s, v in valori.items() if v is not None and not _nan(v)}
    if not present:
        return {"media": None, "sd": None, "n": 0, "valori": {}}
    numbers = [present[s] for s in sorted(present)]
    return {
        "media": statistics.mean(numbers),
        "sd": statistics.stdev(numbers) if len(numbers) > 1 else None,
        "n": len(numbers),
        "valori": {str(s): present[s] for s in sorted(present)},
    }


def find_cells(runs_dir: Path, *, solo: Path | None = None) -> dict[str, dict[int, Path]]:
    """Collect the run folders, per model and seed.

    :param runs_dir: the ``runs/`` folder.
    :param solo: ONE run folder, entering alone and without the filters: for runs outside the
        matrix ``run_id`` grammar, such as the cross-dataset evaluations of
        :mod:`evd.exp.esterno` (``{run_id}__{dataset}``, ADR-066). The filter keeps a cell
        measured on another dataset out of the UCF-Crime leaderboards.
    :return: ``{key: {seed: folder}}``, limited to :data:`MODELS` and :data:`EXPERIMENTS`.
        The key is the ``run_id`` ``adapt``, except for the extended-budget experiments
        (:data:`EXTENDED_EXPERIMENTS`), where it is ``{exp}:{adapt}``.
    :raises ValueError: if two folders claim the same (model, seed), or ``solo`` is not a
        run folder.
    """
    if solo is not None:
        folder = Path(solo)
        if not (folder / "metrics.json").is_file():
            raise ValueError(f"{folder} is not a run folder: metrics.json is missing")
        # The key goes through the SINGLE translation, as in the normal branch below (deriving it
        # from the raw `adapt` lost the :data:`EXTENDED_EXPERIMENTS` prefix, ADR-052). The name is
        # first brought back to the SOURCE run: a cross-dataset evaluation is `{run_id}__{dataset}`
        # (ADR-066).
        source = folder.name.partition("__")[0] or folder.name
        adapt, _, tail = folder.name.partition("_vjepa21l_")
        adapt, _, seed = (tail or folder.name).rpartition("_s")
        digits = "".join(c for c in seed.split("_")[0] if c.isdigit())
        # The fallback on the raw `adapt` remains for runs outside the matrix.
        key = model_key(source) or adapt or folder.name
        return {key: {int(digits) if digits else 0: folder}}

    cells: dict[str, dict[int, Path]] = defaultdict(dict)
    for folder in sorted(runs_dir.iterdir()):
        if not (folder / "metrics.json").is_file():
            continue
        name = folder.name
        # Extended-budget blocks carry the same `adapt` as the one-epoch ones: without the prefix
        # they would be the same row (:func:`model_key`).
        key = model_key(name)
        if key is None:
            continue
        seed = name.rpartition("_s")[2]
        if int(seed) in cells[key]:
            raise ValueError(
                f"two folders for {key} seed {seed}: {cells[key][int(seed)]} and {folder}"
            )
        cells[key][int(seed)] = folder
    return dict(cells)


#: Suffix given by :mod:`evd.exp.esterno` to a cross-dataset evaluation folder:
#: ``{run_id}__{data.id}``, with ``data.id`` = ``mivia`` for MIVIA-Crime.
MIVIA_SUFFIX = "__mivia"


def find_mivia(runs_dir: Path) -> dict[str, dict[int, Path]]:
    """Collect the MIVIA-Crime evaluations, per model and seed.

    :param runs_dir: the ``runs/`` folder.
    :return: ``{key: {seed: folder}}``, in the same form as :func:`find_cells`.
    :raises ValueError: if two folders claim the same (model, seed).

    They become columns, not rows. Since 22 September 2026 the exclusion is structural: an
    external evaluation folder sits inside the source run one
    (``runs/{source}/{source}__mivia``), so :func:`find_cells` never meets it; the name check
    in :func:`model_key` stays as a second line. The name goes back to the source ``run_id``
    and through :func:`model_key` (a single translation, ADR-052).
    """
    found: dict[str, dict[int, Path]] = defaultdict(dict)
    # Look both inside the source run (the right place) and at the top level of `runs/` (where
    # they were before the move).
    candidate = {*runs_dir.glob("*/*" + MIVIA_SUFFIX), *runs_dir.glob("*" + MIVIA_SUFFIX)}
    for folder in sorted(candidate):
        if not (folder / "metrics.json").is_file():
            continue
        source = folder.name[: -len(MIVIA_SUFFIX)]
        key = model_key(source)
        if key is None:
            continue
        seed = int(source.rpartition("_s")[2])
        if seed in found[key]:
            raise ValueError(
                f"two MIVIA evaluations for {key} seed {seed}: {found[key][seed]} and {folder}"
            )
        found[key][seed] = folder
    return dict(found)


def filter_mivia_seeds(
    found: Mapping[str, Mapping[int, Path]], semi: Sequence[int] | None
) -> dict[str, dict[int, Path]]:
    """Keep only the requested seeds of the MIVIA-Crime evaluations.

    Section 6 reports seed 42 only (the laboratory protocol seed, the same for every model);
    E3-S also has seeds 67 and 104 on disk. The discarded folders stay where they are.

    :param trovate: the output of :func:`find_mivia`.
    :param semi: the seeds to keep; ``None`` keeps them all.
    :return: the same form, without models left empty.
    """
    if semi is None:
        return {k: dict(v) for k, v in found.items()}
    kept = set(semi)
    filtered = {k: {s: c for s, c in v.items() if s in kept} for k, v in found.items()}
    return {k: v for k, v in filtered.items() if v}


def run_dir_of(runs_dir: Path, run_id: str) -> Path:
    """Translate a ``run_id`` into the folder that actually contains it.

    Since 22 September 2026 a cross-dataset evaluation lives inside the source run
    (``runs/{source}/{source}__{dataset}``), so plain concatenation gives a missing folder.
    The translation lives here only, next to the folder-discovery functions (ADR-052).

    :param runs_dir: the ``runs/`` folder.
    :param run_id: the run identifier.
    :return: the run folder; if missing in both places, the top-level one (a writer creates
        it there, a reader raises).
    """
    direct = runs_dir / run_id
    if direct.is_dir():
        return direct
    source, separator, _ = run_id.partition("__")
    if separator and source:
        nested = runs_dir / source / run_id
        if nested.is_dir():
            return nested
    return direct


def load_cell(run_dir: Path, cache_dir: Path | None) -> dict[str, Any]:
    """Gather metrics, intersection latency and strata in one place.

    They live in different containers for historical reasons (next to the run, or in the
    ablation cache); the block format is the same.

    :param run_dir: the ``runs/{run_id}/`` folder.
    :param cache_dir: the ablation analysis cache, or ``None``.
    :return: ``m`` (metrics), ``lat`` (intersection latency), ``lat_storica``, ``strata``,
        plus the cell identity and provenance.
    :raises FileNotFoundError: if ``metrics.json`` is missing.
    """
    metrics = read_json(run_dir / "metrics.json")
    provenance: list[str] = ["metrics.json"]

    # Three containers, the first the most authoritative: since September the runner writes the
    # intersection latency and the strata INSIDE `metrics.json`; August cells have the post-hoc
    # recomputation in `latenza.json` / `strata.json` or in the ablation cache.
    latency: Any = metrics.get("detection_latency_intersezione") or None
    strata: Any = metrics.get("strata") or None

    latency_path = run_dir / "latenza.json"
    if latency is None and latency_path.is_file():
        latency = read_json(latency_path).get("detection_latency_intersezione")
        provenance.append("latenza.json")
    strata_path = run_dir / "strata.json"
    if strata is None and strata_path.is_file():
        strata = read_json(strata_path)
        provenance.append("strata.json")

    if (latency is None or strata is None) and cache_dir is not None:
        cache_path = cache_dir / f"{run_dir.name}.json"
        if cache_path.is_file():
            block = read_json(cache_path)
            latency = latency if latency is not None else block.get("latency_intersection")
            strata = strata if strata is not None else block.get("strata")
            provenance.append(f"{cache_dir.name}/{run_dir.name}.json")

    return {
        "run_id": run_dir.name,
        "run_dir": run_dir,
        "seed": metrics.get("seed"),
        "m": metrics,
        "lat": latency or {},
        "lat_storica": metrics.get("detection_latency") or {},
        "strata": strata or {},
        "provenienza": provenance,
    }


VIDEO_MEDIANS = (
    "score_mediano_chunk_anomali",
    "score_mediano_chunk_normali",
    "delta_score_mediano_chunk",
)

VIDEO_METRICS = (
    *VIDEO_MEDIANS,
    "chunk_auc",
    "chunk_ap",
    "chunk_precision",
    "chunk_recall",
    "chunk_f1",
    "chunk_accuracy",
    "tp",
    "fp",
    "fn",
    "tn",
    "n_sopra_soglia",
    "event_precision",
    "event_recall",
    "event_f1",
    "n_eventi",
    "n_eventi_rilevati",
    "n_eventi_mancati",
    "n_eventi_irraggiungibili",
    "n_allarmi",
    "n_allarmi_veri",
    "n_allarmi_falsi",
    "latenza_mediana_s",
    "latenza_p90_s",
    "soglia",
)

VIDEO_STATES = ("chunk_auc_stato", "latenza_stato", "tasso_falsi_positivi_stato")


def chunk_medians(
    anomalous_scores: Sequence[float], normal_scores: Sequence[float]
) -> dict[str, Any]:
    """Return the score medians split by true label, independent of the threshold.

    :param anomalous_scores: scores of the label-1 chunks, TP and FN together.
    :param normal_scores: scores of the label-0 chunks, TN and FP together.
    :return: medians and anomalous-minus-normal difference; None if a set is missing.
    """
    a = statistics.median(anomalous_scores) if anomalous_scores else None
    n = statistics.median(normal_scores) if normal_scores else None
    return dict(
        zip(VIDEO_MEDIANS, (a, n, a - n if a is not None and n is not None else None), strict=True)
    )


def per_video(
    run_dir: Path,
    soglia: float,
    overrides: list[str] | None = None,
    *,
    split: str = "test",
) -> dict[str, dict[str, Any]]:
    """Rebuild the count of every single video from the scores on disk.

    The count is derived from the same rows that produced the global metrics, so it cannot
    diverge from them, as :func:`check_cell` proves.

    :param run_dir: the ``runs/{run_id}/`` folder.
    :param soglia: the run operating threshold, the same for chunks and videos.
    :param overrides: local paths of the data group, without modifying the run.
    :param split: the partition to count, ``test`` (default) or ``val``.
    :return: one entry per video of the requested partition.
    :raises FileNotFoundError: if ``scores.jsonl`` does not exist.
    :raises ValueError: if scores, annotations or metrics do not reproduce the run.

    One partition per call: the file covers test and validation, which have different grids
    (8 s hop instead of 4). The comparison with ``metrics.json`` holds only on test, the
    evaluated partition; on ``val`` only per-video checks and the :func:`rileggi_run` grid
    check apply.
    """
    path = run_dir / "scores.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist: the run did not save its scores")

    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n_finestre": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}
    )
    labels = {}
    score_per_label = defaultdict(lambda: {0: [], 1: []})
    with path.open(encoding="utf-8") as f:
        for row in f:
            record = json.loads(row)
            if record.get("split") != split:
                continue
            entry = counts[record["video_id"]]
            labels[record["window_id"]] = int(record["label"])
            score_per_label[record["video_id"]][int(record["label"])].append(float(record["score"]))
            true_value = int(record["label"]) == 1
            predicted = float(record["score"]) >= soglia
            entry["n_finestre"] += 1
            if true_value and predicted:
                entry["tp"] += 1
            elif predicted:
                entry["fp"] += 1
            elif true_value:
                entry["fn"] += 1
            else:
                entry["tn"] += 1

    result: dict[str, dict[str, Any]] = {}
    for video, entry in counts.items():
        tp, fp, fn, tn = entry["tp"], entry["fp"], entry["fn"], entry["tn"]
        anomalous = (tp + fn) > 0
        precision = (tp / (tp + fp) if (tp + fp) else 0.0) if anomalous else None
        recall = tp / (tp + fn) if (tp + fn) else None
        if precision and recall:
            f1: float | None = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0 if anomalous else None
        # The video-level decision is asymmetric (protocol §4.2): on an anomalous video only a
        # window actually containing the event counts; on a normal one any window above threshold.
        predicted_positive = tp > 0 if anomalous else fp > 0
        result[video] = {
            **chunk_medians(score_per_label[video][1], score_per_label[video][0]),
            "anomalo": anomalous,
            "n_finestre": entry["n_finestre"],
            "n_positive_vere": tp + fn,
            "n_sopra_soglia": tp + fp,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tasso_falsi_positivi": fp / (fp + tn) if (fp + tn) else None,
            "predetto_positivo": bool(predicted_positive),
        }
    read_value = reread_run(run_dir, split=split, overrides=overrides or [])
    class_names = {c.idx: c.name for c in read_value.bundle.classes}
    for video, entry in result.items():
        entry["classi_annotate"] = sorted(
            {class_names[i[2]] for i in read_value.gt[video].intervals}
        )
    metrics = read_json(run_dir / "metrics.json") if split == "test" else {}
    # Pooled AUC/AP are not the mean of the per-video AUC/AP.
    for name, function in (("chunk_auc", M.dataset_chunk_auc), ("chunk_ap", M.dataset_chunk_ap)):
        if not metrics:
            break
        value = function(read_value.scores, read_value.gt)
        if not math.isclose(value, float(metrics[name]), abs_tol=1e-9):
            raise ValueError(f"{run_dir.name}: {name} does not reproduce metrics.json")
    for record in M.chunk_records(read_value.scores, read_value.gt):
        # The grid is checked by the replay; labels must match too.
        if labels.get(record["window_id"]) != record["label"]:
            raise ValueError(f"{run_dir.name}: different label for {record['window_id']}")
    for video_id, entry in result.items():
        scores = {video_id: read_value.scores[video_id]}
        gt = {video_id: read_value.gt[video_id]}
        n_pos = sum(r["label"] for r in M.chunk_records(scores, gt))
        if n_pos != entry["n_positive_vere"]:
            raise ValueError(f"{run_dir.name}/{video_id}: inconsistent labels")
        mixed = 0 < n_pos < entry["n_finestre"]
        ev = M.event_prf(scores, gt, soglia, source="val")
        lat = M.detection_latency(scores, gt, read_value.fps, threshold=soglia, rule="intersezione")
        entry.update(
            {
                "chunk_auc": M.dataset_chunk_auc(scores, gt) if mixed else None,
                # AP is defined (1) even if all chunks are positive; the shared dataset AUC/AP guard
                # excludes this case.
                "chunk_ap": M.dataset_chunk_ap(scores, gt) if mixed else (1.0 if n_pos else None),
                "chunk_precision": entry["precision"],
                "chunk_recall": entry["recall"],
                "chunk_f1": entry["f1"],
                "chunk_accuracy": (entry["tp"] + entry["tn"]) / entry["n_finestre"],
                "event_precision": ev.precision if ev.n_eventi else None,
                "event_recall": ev.recall if ev.n_eventi else None,
                "event_f1": ev.f1 if ev.n_eventi else None,
                "n_eventi": ev.n_eventi,
                "n_eventi_rilevati": ev.n_eventi_rilevati,
                "n_eventi_mancati": ev.n_eventi - ev.n_eventi_rilevati,
                "n_eventi_irraggiungibili": ev.n_eventi_irraggiungibili,
                "n_allarmi": ev.n_allarmi,
                "n_allarmi_veri": ev.n_allarmi_veri,
                "n_allarmi_falsi": ev.n_allarmi - ev.n_allarmi_veri,
                "latenza_mediana_s": None if _nan(lat.mediana_s) else lat.mediana_s,
                "latenza_p90_s": None if _nan(lat.p90_s) else lat.p90_s,
                "soglia": soglia,
                "chunk_auc_stato": "definita"
                if mixed
                else (
                    "non_definita_senza_chunk_negativi"
                    if n_pos
                    else "non_applicabile_video_normale"
                ),
                "latenza_stato": "misurata"
                if ev.n_eventi_rilevati
                else (
                    "non_misurabile_nessun_evento_rilevato"
                    if ev.n_eventi
                    else "non_applicabile_video_normale"
                ),
                "tasso_falsi_positivi_stato": "definito"
                if n_pos < entry["n_finestre"]
                else "non_definito_senza_chunk_negativi",
            }
        )
    for name in (
        "n_eventi",
        "n_eventi_rilevati",
        "n_allarmi",
        "n_allarmi_veri",
        "n_eventi_irraggiungibili",
    ):
        if not metrics:
            break
        if sum(v[name] for v in result.values()) != metrics["event_prf"][name]:
            raise ValueError(f"{run_dir.name}: {name} does not reproduce metrics.json")
    return result


def check_cell(cella: Mapping[str, Any], video: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Compare what the script recomputed with what the run wrote.

    A per-video count that does not recompose the global P/R/F1 means the re-reading used the
    wrong partition or threshold.

    :param cella: the cell loaded by :func:`load_cell`.
    :param video: the per-video count of :func:`per_video`.
    :return: the inconsistencies found; empty if everything matches.
    """
    problems: list[str] = []
    m = cella["m"]
    tp = sum(v["tp"] for v in video.values())
    fp = sum(v["fp"] for v in video.values())
    fn = sum(v["fn"] for v in video.values())
    n_windows = sum(v["n_finestre"] for v in video.values())
    anomalous = sum(1 for v in video.values() if v["anomalo"])

    for name, mine, expected in (
        ("windows", n_windows, m.get("n_chunk_eval")),
        ("video", len(video), m.get("n_eval_videos")),
        ("anomalous videos", anomalous, m.get("n_video_anomalous")),
    ):
        if expected is not None and mine != int(expected):
            problems.append(f"{name}: recomputed {mine} vs {expected}")

    prf = m.get("chunk_prf") or {}
    if prf and (tp + fp) and (tp + fn):
        precision, recall = tp / (tp + fp), tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        for name, my_value, its_value in (
            ("precision", precision, prf.get("precision")),
            ("recall", recall, prf.get("recall")),
            ("f1", f1, prf.get("f1")),
        ):
            if its_value is not None and not math.isclose(my_value, float(its_value), abs_tol=1e-9):
                problems.append(
                    f"chunk {name}: recomputed {my_value:.9f} vs {float(its_value):.9f}"
                )

    # The per-video decision is asymmetric: reproducing it shows that the rule implemented here
    # is the protocol one.
    vprf = m.get("video_prf") or {}
    if vprf:
        predicted = [v for v in video.values() if v["predetto_positivo"]]
        true_items = [v for v in predicted if v["anomalo"]]
        expected_pred = vprf.get("n_pred_pos")
        if expected_pred is not None and len(predicted) != int(expected_pred):
            problems.append(f"videos predicted positive: {len(predicted)} vs {expected_pred}")
        if predicted and anomalous:
            precision, recall = len(true_items) / len(predicted), len(true_items) / anomalous
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            if vprf.get("f1") is not None and not math.isclose(f1, float(vprf["f1"]), abs_tol=1e-9):
                problems.append(f"video F1: recomputed {f1:.9f} vs {float(vprf['f1']):.9f}")

    # ADR-061: with the `intersection` rule, events without a measurable latency MUST equal the
    # missed events, since the predicate is the same.
    lat, ev = cella["lat"], m.get("event_prf") or {}
    if lat and ev:
        missed = int(ev["n_eventi"]) - int(ev["n_eventi_rilevati"])
        without = int(lat.get("n_senza_latenza_misurabile", -1))
        if without != missed:
            problems.append(f"latency: {without} excluded vs {missed} missed events")
    return problems


#: SLURM partition -> GPU, checked against the node name; the single definition lives with
#: the log parser.
PARTITION_GPU: dict[str, str] = TL.PARTITION_GPU

#: How many intervals between checkpoints are needed for the median to be a steady-state
#: cost rather than start-up (a quickly dying resume leaves two intervals that include
#: model construction and warm-up; on ``vptdeep-n8`` seed 104 they gave 16.29 s/step on
#: H100 against 6.19 on A100).
MIN_INTERVALS: int = 5


def cost_per_step(log_dir: Path | None) -> dict[str, dict[str, Any]]:
    """Return the measured cost of a training step, per model and GPU.

    Source: the SLURM logs, where the trainer writes a line at every checkpoint
    (:mod:`evd.exp.training_log`); the median of the intervals between consecutive checkpoints
    divided by the steps between them is the cost of a step (the median, because a validation
    may fall between two checkpoints). The GPU is declared, never converted: a cell run on
    H100 does not become an A100 number. The same run_id appears in several logs (resumes):
    the median over all intervals of every log of a (run, GPU) pair is used.

    :param log_dir: the ``hpc/log`` folder, or ``None`` to measure nothing.
    :return: ``{adapt: {"A100": {...}, "H100": {...}}}``, each block with ``s_per_passo``,
        ``n_intervalli``, ``n_esecuzioni`` and the ``run_id`` values; models without logs are
        absent.
    """
    if log_dir is None or not log_dir.is_dir():
        log.warning("SLURM logs missing: the training cost columns stay empty")
        return {}
    # (adapt, gpu) -> seconds per step of every interval between checkpoints
    raw: dict[tuple[str, str], list[float]] = defaultdict(list)
    sources: dict[tuple[str, str], set[str]] = defaultdict(set)
    executions: dict[tuple[str, str], int] = defaultdict(int)
    for parsed in TL.read_logs([log_dir]):
        segments = [s for s in parsed.segments.values() if s.checkpoints]
        if sum(len(s.checkpoints) for s in segments) < 3:
            continue
        if parsed.gpu is None:
            log.warning(
                "%s: partition %r and node %r disagree on the GPU, log ignored",
                parsed.name,
                parsed.partition,
                parsed.node,
            )
            continue
        for segment in segments:
            adapt = model_key(segment.run_id)
            if adapt is None:
                continue
            # Validations are not separated here: the median absorbs them.
            values = [i.seconds / i.steps for i in segment.intervals(None)]
            if values:
                key = (adapt, parsed.gpu)
                raw[key].extend(values)
                sources[key].add(segment.run_id)
                executions[key] += 1
    outside: dict[str, dict[str, Any]] = defaultdict(dict)
    for (adapt, gpu), values in raw.items():
        if len(values) < MIN_INTERVALS:
            log.warning(
                "%s on %s: only %d intervals between checkpoints, below the %d needed to "
                "tell steady state from start-up; cell left empty",
                adapt,
                gpu,
                len(values),
                MIN_INTERVALS,
            )
            continue
        outside[adapt][gpu] = {
            "s_per_passo": float(statistics.median(values)),
            "n_intervalli": len(values),
            "n_esecuzioni": executions[(adapt, gpu)],
            "run_id": sorted(sources[(adapt, gpu)]),
        }
    return dict(outside)


def compose_training_cost(
    adapt: str, costs: Mapping[str, Mapping[str, Any]], max_steps: int
) -> dict[str, Any]:
    """Turn seconds per step into hours per cell, on both machines.

    :param adapt: the model key in :data:`MODELS`.
    :param costs: the output of :func:`cost_per_step`.
    :param max_steps: the budget steps, equal for every method (what makes hours comparable
        across methods).
    :return: the two columns, the speed-up where both exist, and ``None`` where no execution
        ran on that machine.

    Hours per cell are training only: the epoch validations fall inside the intervals
    discarded by the median.
    """
    row = costs.get(adapt) or {}
    a100 = row.get("A100")
    h100 = row.get("H100")

    def _hours(block: Any) -> float | None:
        if not block:
            return None
        return float(block["s_per_passo"]) * max_steps / 3600.0

    s_a = None if not a100 else float(a100["s_per_passo"])
    s_h = None if not h100 else float(h100["s_per_passo"])
    return {
        "s_per_passo_a100": s_a,
        "s_per_passo_h100": s_h,
        "ore_a100": _hours(a100),
        "ore_h100": _hours(h100),
        "speedup_h100": None if (s_a is None or not s_h) else s_a / s_h,
        "n_intervalli_a100": None if not a100 else a100["n_intervalli"],
        "n_intervalli_h100": None if not h100 else h100["n_intervalli"],
    }


def load_profile(cartella: Path | None) -> dict[str, Any]:
    """Read the inference profile, if present.

    :param cartella: the folder containing ``efficiency_inference.json``.
    :return: ``latenze`` as ``{variant: {batch: ms}}``, plus machine and provenance; empty if
        the profile does not exist.

    The machine is declared: A100 and H100 latencies are not comparable (ratio ~1.73×).
    """
    if cartella is None:
        return {}
    path = cartella / "efficiency_inference.json"
    if not path.is_file():
        log.warning("inference profile missing in %s: the compute columns stay empty", cartella)
        return {}
    block = read_json(path)
    latencies: dict[str, dict[int, float]] = defaultdict(dict)
    variants: dict[str, dict[str, Any]] = {}
    for row in block.get("rows", []):
        variant = str(row["variante"])
        batch = int(row["batch"])
        latencies[variant][batch] = float(row["latency_ms_mean"])
        if batch == 1:
            variants[variant] = {
                "latenza_ms": float(row["latency_ms_mean"]),
                "latenza_ms_sd": _float_o_none(row.get("latency_ms_std")),
                "fondibile": bool(row.get("fondibile")),
                "fusa": bool(row.get("fusa")),
                "extra_tokens": row.get("extra_tokens"),
                "params_adapter": row.get("params_adapter"),
                "params_da_spedire": row.get("params_da_spedire"),
                "fattore_realtime": _float_o_none(row.get("fattore_realtime")),
                "merge_max_abs": _float_o_none(row.get("merge_max_abs")),
                "effetto_adattatore": _float_o_none(row.get("effetto_adattatore")),
            }
    return {
        "gpu": block.get("gpu"),
        "precision": block.get("precision"),
        "geometria": block.get("geometria"),
        "sorgente": str(path),
        "latenze": {k: dict(v) for k, v in latencies.items()},
        # Batch-1 rows per profile variant, not per model: merged rows live here too (they are not
        # trained models).
        "varianti": variants,
    }


def compose_latency(
    modello: str,
    grid_delay_s: float | None,
    profilo: Mapping[str, Any],
    alt_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Sum the three components of the delay with which the system signals an event.

    ``alarm delay = grid delay + compute time + decoding``

    The grid delay is the wait for the window containing the event to end (it depends on the
    hop, not on the model); compute is the forward pass; decoding is a system constant.

    :param modello: the key in :data:`MODELS`.
    :param ritardo_griglia_s: median grid delay, intersection rule.
    :param profilo: the output of :func:`load_profile`, the reference machine.
    :param alt_profile: a second profile on another GPU, or ``None``; its columns go under
        ``altra_gpu``.
    :return: the three separate components, the total, and the two shares.

    Components are reported separately (shortening compute does not shorten the response);
    a variant missing from the profile leaves empty cells; the two machines are never mixed
    in one column (only the grid delay is shared).
    """
    latencies = profilo.get("latenze") or {}
    variant = PROFILE_VARIANT.get(modello)
    compute_ms = (latencies.get(variant) or {}).get(1) if variant else None
    merged = MERGED_VARIANT.get(modello)
    compute_merged_ms = (latencies.get(merged) or {}).get(1) if merged else None
    reference_ms = (latencies.get("frozen") or {}).get(1)

    def _total(calcolo: float | None) -> float | None:
        if calcolo is None or grid_delay_s is None:
            return None
        return grid_delay_s + (calcolo + DECODE_MS) / 1000.0

    def _overhead(calcolo: float | None) -> float | None:
        if calcolo is None or not reference_ms:
            return None
        return 100.0 * (calcolo - reference_ms) / reference_ms

    total_s = _total(compute_ms)
    response_ms = None if compute_ms is None else compute_ms + DECODE_MS
    return {
        "ritardo_griglia_s": grid_delay_s,
        "calcolo_ms": compute_ms,
        "decodifica_ms": None if compute_ms is None else DECODE_MS,
        "risposta_ms": response_ms,
        "totale_s": total_s,
        "quota_calcolo_pct": (
            None
            if total_s is None or response_ms is None
            else 100.0 * (response_ms / 1000.0) / total_s
        ),
        "quota_griglia_pct": (
            None if total_s is None or grid_delay_s is None else 100.0 * grid_delay_s / total_s
        ),
        "margine_su_hop_ms": None if response_ms is None else HOP_S * 1000.0 - response_ms,
        "fattore_realtime": None if not response_ms else HOP_S * 1000.0 / response_ms,
        "overhead_calcolo_pct": _overhead(compute_ms),
        "calcolo_fuso_ms": compute_merged_ms,
        "totale_fuso_s": _total(compute_merged_ms),
        "overhead_calcolo_fuso_pct": _overhead(compute_merged_ms),
        "altra_gpu": (
            None if alt_profile is None else compose_latency(modello, grid_delay_s, alt_profile)
        ),
    }


#: Metrics reported cross-dataset: fewer than :data:`METRICS`, since threshold columns
#: depend on a transferred tau and the event level would use another laboratory annotation.
MIVIA_METRICS: tuple[tuple[str, str, Callable[[Mapping[str, Any]], Any]], ...] = (
    ("chunk_auc", "Chunk AUC", lambda m: _get(m, "chunk_auc")),
    ("chunk_ap", "Chunk AP", lambda m: _get(m, "chunk_ap")),
    ("chunk_auc_anomalous", "Chunk AUC (anom.)", lambda m: _get(m, "chunk_auc_anomalous")),
    ("chunk_ap_anomalous", "Chunk AP (anom.)", lambda m: _get(m, "chunk_ap_anomalous")),
    ("chunk_f1", "Chunk F1", lambda m: _get(m, "chunk_prf", "f1")),
    ("video_f1", "Video F1", lambda m: _get(m, "video_prf", "f1")),
)


def _prevalence(metriche: Mapping[str, Any]) -> float | None:
    """:return: the share of positive chunks, the floor of AP.

    Needed for the lift: APs on datasets with different positive rates are not comparable,
    AP/prevalence is.
    """
    positives = _get(metriche, "chunk_prf", "n_true_pos")
    total = _get(metriche, "n_chunk_eval") or _get(metriche, "n_chunk")
    if not positives or not total:
        return None
    return float(positives) / float(total)


def compose_mivia(
    loaded_ucf: Mapping[int, Mapping[str, Any]], folders: Mapping[int, Path] | None
) -> dict[str, Any] | None:
    """:return: the cross-dataset block of the model, or ``None`` if not measured.

    :param loaded_ucf: the UCF-Crime cells of the model, per seed.
    :param folders: the ``{run_id}__mivia`` folders, per seed.

    The delta is PAIRED on the seed, not computed against the multi-seed UCF-Crime mean. APs
    carry their lift (positives are 30.7% on MIVIA-Crime vs 16.4% on UCF-Crime). ``chunk_f1``
    depends on a transferred tau; ``f1_oracolo`` (Youden on MIVIA-Crime itself, not protocol)
    separates the two causes. ``chunk_auc`` also carries ``medio_fra_dataset``, the plain mean
    of the two chunk AUCs; weighting the datasets equally is a choice.
    """
    if not folders:
        return None
    per_seed: dict[int, Mapping[str, Any]] = {}
    for seed, folder in sorted(folders.items()):
        per_seed[seed] = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))

    metrics: dict[str, Any] = {}
    for key, label, extract in MIVIA_METRICS:
        values = {s: _float_o_none(extract(m)) for s, m in per_seed.items()}
        paired = {
            s: _float_o_none(extract(loaded_ucf[s]["m"]))
            for s in values
            if s in loaded_ucf and loaded_ucf[s].get("m") is not None
        }
        delta = {
            s: (values[s] - paired[s])
            for s in values
            if values.get(s) is not None and paired.get(s) is not None
        }
        metrics[key] = {
            "etichetta": label,
            "mivia": aggregate(values),
            "ucf_stessi_semi": aggregate(paired),
            "delta": aggregate(delta),
        }

    # The cross-dataset mean, used to choose a model that must hold at home AND abroad.
    # Weighting the two datasets equally is a CHOICE; both columns stay next to it. The
    # UCF-Crime term is the one paired on the seed.
    auc = metrics.get("chunk_auc", {})
    at_home = _get(auc, "ucf_stessi_semi", "media")
    outside = _get(auc, "mivia", "media")
    auc["medio_fra_dataset"] = (
        (float(at_home) + float(outside)) / 2
        if at_home is not None and outside is not None
        else None
    )

    first_seed = next(iter(sorted(per_seed)))
    first = per_seed[first_seed]
    prev_mivia = _prevalence(first)
    prev_ucf = _prevalence(loaded_ucf[first_seed]["m"]) if first_seed in loaded_ucf else None
    ap_mivia = _float_o_none(_get(first, "chunk_ap"))
    ap_ucf = (
        _float_o_none(_get(loaded_ucf[first_seed]["m"], "chunk_ap"))
        if first_seed in loaded_ucf
        else None
    )
    return {
        "semi": sorted(per_seed),
        "run_id": {str(s): (folders[s].name) for s in sorted(per_seed)},
        "n_video": _get(first, "n_eval_videos"),
        "n_chunk": _get(first, "n_chunk_eval") or _get(first, "n_chunk"),
        "prevalenza": {"mivia": prev_mivia, "ucf": prev_ucf},
        "lift_ap": {
            "mivia": (ap_mivia / prev_mivia) if ap_mivia and prev_mivia else None,
            "ucf": (ap_ucf / prev_ucf) if ap_ucf and prev_ucf else None,
        },
        "soglia": {
            "valore": _get(first, "threshold", "value"),
            "origine": _get(first, "threshold", "source"),
            "trasferita_da": _get(first, "threshold", "trasferita_da"),
        },
        "f1_oracolo": _float_o_none(_get(first, "soglia_oracolo", "chunk_prf", "f1")),
        "metriche": metrics,
    }


def _num(valore: Any, decimals: int) -> str:
    """:return: the number formatted for a table, or ``—`` if missing."""
    if valore is None or _nan(valore):
        return "—"
    if decimals == 0:
        return f"{int(round(float(valore))):,}".replace(",", " ")
    return f"{float(valore):.{decimals}f}"


def _pct(valore: Any) -> str:
    """:return: a fraction written as a percentage, or ``—`` if missing."""
    if valore is None or _nan(valore):
        return "—"
    return f"{float(valore) * 100:.2f}%"


def _row(celle: Sequence[str]) -> str:
    """:return: a Markdown table row."""
    return "| " + " | ".join(celle) + " |"


def _tabella(header: Sequence[str], righe: Iterable[Sequence[str]]) -> str:
    """:return: a complete Markdown table with separator."""
    parts = [_row(header), _row(["---"] * len(header))]
    parts.extend(_row(r) for r in righe)
    return "\n".join(parts)


def _sort(models: Mapping[str, Any], chiave: str, direction: str) -> list[dict[str, Any]]:
    """:return: the models sorted on a metric, with rank; missing ones last."""
    entry_list = [
        {"modello": v["nome"], "famiglia": v["famiglia"], **v["metriche"][chiave]}
        for v in models.values()
    ]
    con = [v for v in entry_list if v["media"] is not None]
    without = [v for v in entry_list if v["media"] is None]
    con.sort(key=lambda v: v["media"], reverse=(direction == "max"))
    for position, entry in enumerate(con, start=1):
        entry["rango"] = position
    for entry in without:
        entry["rango"] = None
    return con + without


def _per_class(
    models: Mapping[str, Any], loaded_per_model: Mapping[str, Mapping[int, dict[str, Any]]]
) -> dict[str, Any]:
    """Aggregate per anomaly class: AUC and AP from chunks, recall from events.

    :param modelli: the already built models, for the display name.
    :param loaded_per_model: the already loaded cells, per model and seed.
    :return: one entry per class, with ``n_pos`` and one row per model.

    ``n_pos`` always travels next to the numbers (``Abuse`` has six positive chunks out of
    9116).
    """
    raw: dict[str, dict[str, Any]] = {}
    for adapt, loaded in loaded_per_model.items():
        for seed, cell in loaded.items():
            for class_name, block in (_get(cell["m"], "per_class_chunk") or {}).items():
                entry = raw.setdefault(
                    class_name, {"n_pos_chunk": block.get("n_pos"), "modelli": {}}
                )
                row = entry["modelli"].setdefault(
                    adapt, {"auc": {}, "ap": {}, "event_recall": {}, "n_eventi": None}
                )
                row["auc"][seed] = block.get("auc")
                row["ap"][seed] = block.get("ap")
            for class_name, block in (_get(cell["strata"], "evento", "classe") or {}).items():
                entry = raw.setdefault(class_name, {"n_pos_chunk": None, "modelli": {}})
                row = entry["modelli"].setdefault(
                    adapt, {"auc": {}, "ap": {}, "event_recall": {}, "n_eventi": None}
                )
                row["event_recall"][seed] = block.get("recall")
                row["n_eventi"] = block.get("n_eventi")

    result: dict[str, Any] = {}
    for class_name in sorted(raw):
        entry = raw[class_name]
        result[class_name] = {
            "n_pos_chunk": entry["n_pos_chunk"],
            "modelli": {
                models[adapt]["nome"]: {
                    "auc": aggregate(row["auc"]),
                    "ap": aggregate(row["ap"]),
                    "event_recall": aggregate(row["event_recall"]),
                    "n_eventi": row["n_eventi"],
                }
                for adapt, row in entry["modelli"].items()
                if adapt in models
            },
        }
    return result


#: Stratified axes entering the leaderboards, with their metrics. ``precision`` and ``f1``
#: are NOT the run aggregates: false positives are shared across the bands of an axis
#: (`evd.eval.strata`), so bands are compared, not summed.
AXES_TO_AGGREGATE: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "copertura": ("chunk", "n_pos", ("auc", "ap", "precision", "recall", "f1")),
    "durata": ("evento", "n_eventi", ("precision", "recall", "f1", "latenza_mediana_s")),
    "quota_anomala_video": (
        "evento",
        "n_eventi",
        ("precision", "recall", "f1", "latenza_mediana_s"),
    ),
}


def _per_stratum(
    models: Mapping[str, Any], loaded_per_model: Mapping[str, Mapping[int, dict[str, Any]]]
) -> dict[str, Any]:
    """Aggregate the stratified axes over the seeds, per model (ADR-060).

    The source of the "where it fails" tables.

    :param modelli: the already built models, for the display name.
    :param loaded_per_model: the already loaded cells, per model and seed.
    :return: ``{axis: {band: {"n": …, "modelli": {name: {metric: aggregate}}}}}``, with bands
        in the ``strata.json`` order (increasing, not alphabetical).
    """
    result: dict[str, Any] = {}
    for axis, (level, n_field, metrics) in AXES_TO_AGGREGATE.items():
        bands: dict[str, dict[str, Any]] = {}
        for adapt, loaded in loaded_per_model.items():
            if adapt not in models:
                continue
            for seed, cell in loaded.items():
                block = _get(cell["strata"], level, axis) or {}
                for band, row in block.items():
                    entry = bands.setdefault(band, {"n": row.get(n_field), "modelli": {}})
                    per_model = entry["modelli"].setdefault(
                        models[adapt]["nome"], {m: {} for m in metrics}
                    )
                    for metric in metrics:
                        per_model[metric][seed] = row.get(metric)
        result[axis] = {
            band: {
                "n": entry["n"],
                "modelli": {
                    name: {m: aggregate(values) for m, values in rows.items()}
                    for name, rows in entry["modelli"].items()
                },
            }
            for band, entry in bands.items()
        }
    return result


def _aggregated_per_video(
    models: Mapping[str, Any],
    videos_per_model: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """:return: one entry per video, with every model's count aggregated over the seeds."""
    final: dict[str, dict[str, Any]] = {}
    for adapt, video in videos_per_model.items():
        name = models[adapt]["nome"]
        for video_id, per_seed in video.items():
            first = next(iter(per_seed.values()))
            entry = final.setdefault(
                video_id,
                {
                    "anomalo": first["anomalo"],
                    "n_finestre": first["n_finestre"],
                    "n_positive_vere": first["n_positive_vere"],
                    "modelli": {},
                },
            )
            entry["modelli"][name] = {
                **{
                    k: aggregate({int(s): v[k] for s, v in per_seed.items()}) for k in VIDEO_METRICS
                },
                "f1": aggregate({int(s): v["f1"] for s, v in per_seed.items()}),
                "precision": aggregate({int(s): v["precision"] for s, v in per_seed.items()}),
                "recall": aggregate({int(s): v["recall"] for s, v in per_seed.items()}),
                "tasso_falsi_positivi": aggregate(
                    {int(s): v["tasso_falsi_positivi"] for s, v in per_seed.items()}
                ),
                "n_semi_predetto_positivo": sum(
                    1 for v in per_seed.values() if v["predetto_positivo"]
                ),
                "n_semi": len(per_seed),
                "per_seme": per_seed,
            }
    return {k: final[k] for k in sorted(final)}


def _provenance(
    runs_dir: Path,
    cache_dir: Path | None,
    profilo: Mapping[str, Any],
    alt_profile: Mapping[str, Any],
    models: Mapping[str, Any],
    inconsistencies: Mapping[str, Sequence[str]],
    tutte_le_versioni: bool = False,
    chosen: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """:return: where the numbers come from, to make the file verifiable."""
    chosen = chosen or {}
    return {
        # Which version of each extended training is in the table, and why.
        "versioni": {
            "tutte": tutte_le_versioni,
            "riportata": {g: c["versione"] for g, c in chosen.items()},
            "scelte": dict(chosen),
            "criterio": None if tutte_le_versioni else VERSION_CRITERION,
        },
        "git": git_info(),
        "runs_dir": str(runs_dir),
        "cache_ablation": str(cache_dir) if cache_dir else None,
        "profilo_inferenza": profilo.get("sorgente"),
        "gpu_del_profilo": profilo.get("gpu"),
        "precisione_del_profilo": profilo.get("precision"),
        # The second machine is named, or its columns would be numbers without hardware.
        "profilo_inferenza_altra_gpu": alt_profile.get("sorgente"),
        "gpu_del_profilo_altra": alt_profile.get("gpu"),
        "decodifica_ms_per_finestra": DECODE_MS,
        "hop_s": HOP_S,
        "regola_latenza": "intersezione (ADR-061)",
        "n_modelli": len(models),
        "n_celle": sum(len(v["semi"]) for v in models.values()),
        "semi": sorted({s for v in models.values() for s in v["semi"]}),
        "coerenza_verificata": not inconsistencies,
    }


def build_catalog(
    runs_dir: Path,
    cache_dir: Path | None,
    profile_dir: Path | None,
    alt_profile_dir: Path | None = None,
    log_dir: Path | None = None,
    overrides: list[str] | None = None,
    solo: Path | None = None,
    semi_mivia: Sequence[int] | None = None,
    tutte_le_versioni: bool = False,
) -> dict[str, Any]:
    """Build the whole leaderboard structure.

    :param runs_dir: the ``runs/`` folder.
    :param overrides: local paths for re-reading scores and annotations.
    :param cache_dir: the ablation analysis cache, or ``None``.
    :param profile_dir: the inference profile folder, or ``None``.
    :param alt_profile_dir: the folder of a second profile on another GPU, or ``None``.
    :param log_dir: the ``hpc/log`` folder, the only source of the training cost
        (:func:`cost_per_step`).
    :param semi_mivia: the MIVIA-Crime seeds entering section 6 (:func:`filter_mivia_seeds`);
        ``None`` = all those on disk.
    :param tutte_le_versioni: keep every version of the extended trainings instead of the
        chosen one (:func:`active_models`).
    :return: the structure written to ``classifiche.json``.
    """
    cells_per_model = find_cells(runs_dir, solo=solo)
    active, chosen = active_models(cells_per_model, tutte_le_versioni)
    if solo is not None:
        # A single run enters anyway, even if it is the non-reported version.
        active = {k: active.get(k) or MODELS[k] for k in cells_per_model if k in MODELS}
    else:
        missing = [k for k in active if k not in cells_per_model]
        if missing:
            log.warning("models without cells: %s", ", ".join(missing))
    profile = load_profile(profile_dir)
    alt_profile = load_profile(alt_profile_dir)
    step_costs = cost_per_step(log_dir)
    # Cross-dataset evaluations are collected separately and become model COLUMNS, never rows
    # (:func:`find_mivia`). With `--run` nothing is enumerated.
    mivia_per_model = (
        {} if solo is not None else filter_mivia_seeds(find_mivia(runs_dir), semi_mivia)
    )

    models: dict[str, Any] = {}
    loaded_per_model: dict[str, dict[int, dict[str, Any]]] = {}
    videos_per_model: dict[str, dict[str, dict[str, Any]]] = {}
    inconsistencies: dict[str, list[str]] = {}

    for adapt in active:
        cells = cells_per_model.get(adapt)
        if not cells:
            continue
        loaded = {seed: load_cell(d, cache_dir) for seed, d in sorted(cells.items())}
        loaded_per_model[adapt] = loaded

        video_aggregates: dict[str, dict[str, Any]] = defaultdict(dict)
        for seed, cell in loaded.items():
            threshold = _get(cell["m"], "threshold", "value")
            if threshold is None:
                log.warning("%s has no threshold: the per-video view is skipped", cell["run_id"])
                continue
            count = per_video(cell["run_dir"], float(threshold), overrides)
            problems = check_cell(cell, count)
            if problems:
                inconsistencies[cell["run_id"]] = problems
            for video, entry in count.items():
                video_aggregates[video][str(seed)] = entry
            log.info(
                "read %s%s",
                cell["run_id"],
                "" if not problems else f"  ⚠️ {len(problems)} inconsistencies",
            )

        metrics: dict[str, Any] = {}
        for key, label, direction, decimals, extract in METRICS:
            block = aggregate({s: extract(c) for s, c in loaded.items()})
            block.update({"etichetta": label, "verso": direction, "decimali": decimals})
            metrics[key] = block

        first = next(iter(loaded.values()))
        total = _get(first["m"], "trainable_params")
        adapter = _get(first["m"], "adapt", "n_trainable") or 0
        models[adapt] = {
            "nome": active[adapt]["nome"],
            "famiglia": active[adapt]["famiglia"],
            "semi": sorted(loaded),
            "run_id": {str(s): c["run_id"] for s, c in loaded.items()},
            "params_adattatore": adapter,
            # Derived, not read: `trainable_params` means two things depending on the writing path
            # (ADR-057).
            "params_testa": (total - adapter) if isinstance(total, int) else None,
            "params_addestrabili_totali": total,
            "metriche": metrics,
            "latenza": compose_latency(
                adapt,
                metrics["latenza_griglia_mediana_s"]["media"],
                profile,
                alt_profile or None,
            ),
            "latenza_regola_storica": aggregate(
                {s: _get(c["lat_storica"], "mediana_s") for s, c in loaded.items()}
            ),
            "costo_training": compose_training_cost(
                adapt, step_costs, int(_get(first["m"], "max_steps") or 0)
            ),
            "mivia": compose_mivia(loaded, mivia_per_model.get(adapt)),
            "provenienza": sorted({p for c in loaded.values() for p in c["provenienza"]}),
        }
        videos_per_model[adapt] = dict(video_aggregates)

    return {
        "generato": _provenance(
            runs_dir,
            cache_dir,
            profile,
            alt_profile,
            models,
            inconsistencies,
            tutte_le_versioni,
            chosen,
        ),
        "profilo_varianti": profile.get("varianti") or {},
        "profilo_varianti_altra_gpu": alt_profile.get("varianti") or {},
        "modelli": models,
        "classifica_globale": {
            key: _sort(models, key, direction) for key, _, direction, _, _ in METRICS
        },
        "per_classe": _per_class(models, loaded_per_model),
        "strati": _per_stratum(models, loaded_per_model),
        "per_video": _aggregated_per_video(models, videos_per_model),
        "costo_training": [
            {
                "modello": models[a]["nome"],
                "famiglia": models[a]["famiglia"],
                **models[a]["costo_training"],
            }
            for a in models
        ],
        "latenza": [
            {
                "modello": models[a]["nome"],
                "famiglia": models[a]["famiglia"],
                **models[a]["latenza"],
            }
            for a in models
        ],
        # Sorted by MIVIA-Crime chunk AUC, which is NOT the `classifica_globale` order: comparing the
        # two orders is the point of this table.
        "cross_dataset": sorted(
            (
                {
                    "chiave": a,
                    "modello": models[a]["nome"],
                    "famiglia": models[a]["famiglia"],
                    **models[a]["mivia"],
                }
                for a in models
                if models[a].get("mivia")
            ),
            key=lambda r: -(_get(r, "metriche", "chunk_auc", "mivia", "media") or 0.0),
        ),
        "incoerenze": inconsistencies,
    }


def _seeds_of(row: Mapping[str, Any]) -> str:
    """:return: the seeds of a cross-dataset row, such as ``42`` or ``42, 67, 104``."""
    return ", ".join(str(s) for s in row.get("semi") or [])


def _auc_at_seed(row: Mapping[str, Any], seme: int) -> tuple[float | None, float | None]:
    """:return: UCF-Crime and MIVIA-Crime chunk AUC of a row at one seed."""
    auc = _get(row, "metriche", "chunk_auc") or {}
    ucf = (_get(auc, "ucf_stessi_semi", "valori") or {}).get(str(seme))
    outside = (_get(auc, "mivia", "valori") or {}).get(str(seme))
    return ucf, outside


def cross_dataset_section(crossed: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return section 6 of ``classifiche.md``: the MIVIA-Crime transfer.

    Seeds are declared PER ROW: with the default (``--semi-mivia``, seed 42 for everyone) the
    "Seed(s) 42 only" sentence holds; when more seeds are requested a three-seed row carries a
    mean, hence the `Seeds` column and section 6c (ranking by mean AUC at the seed every
    model has), absent with the default.

    :param crossed: the ``cross_dataset`` rows of :func:`build_catalog`.
    :return: the Markdown rows, empty without external evaluations.
    """
    if not crossed:
        return []
    first = crossed[0]
    sets = [tuple(r.get("semi") or ()) for r in crossed]
    uniform = len(set(sets)) == 1
    if uniform:
        over_seeds = (
            f"Seed(s) {_seeds_of(first)} only, so **these are single-seed numbers** next "
            "to multi-seed means, and the delta is paired on the same seed rather than "
            "taken against the UCF mean."
        )
    else:
        count: dict[str, int] = defaultdict(int)
        for r in crossed:
            count[_seeds_of(r)] += 1
        items = "; ".join(f"{n} models at seed(s) {s}" for s, n in sorted(count.items()))
        over_seeds = (
            f"⚠️ **Not every model has the same seeds here** ({items}): see the `Seeds` "
            "column. A row with more seeds carries a **mean**, and its `Mean AUC` is not "
            "comparable with a single-seed one. The delta is always paired on the same "
            "seed. To pick a model, use §6c, which ranks every model at the seed(s) they "
            "all share."
        )
    parts = [
        "## 6. Cross-dataset: MIVIA-Crime",
        "",
        f"The selected weights of each cell, run over the lab's private test set "
        f"({first['n_video']} videos, {first['n_chunk']} chunks) without any training "
        f"or re-tuning. {over_seeds}",
        "",
        "Read the AP through its **lift** (AP / positive-chunk rate), not raw: "
        f"MIVIA-Crime is {_pct(first['prevalenza']['mivia'])} positive chunks against "
        f"{_pct(first['prevalenza']['ucf'])} on UCF-Crime, so a higher AP here is floor, "
        "not skill. `Chunk F1` uses a threshold **transferred** from UCF-Crime and never "
        "re-estimated, so it measures ranking *plus* how well the calibration survives; "
        "`F1 oracle` is Youden computed on MIVIA-Crime itself and is **not protocol** — "
        "it exists only to separate the two.",
        "",
        "`Mean AUC` is the plain average of the two chunk AUCs, the UCF-Crime one taken "
        "**at the same seed**. It is the single number to rank by when the model has to "
        "hold up at home *and* on a domain it has never seen — ⚠️ but weighting the two "
        "datasets equally is a **choice**, not a fact: both terms are right next to it "
        "for anyone who wants to weight them differently.",
        "",
        _tabella(
            [
                "Model",
                "Family",
                *([] if uniform else ["Seeds"]),
                "Chunk AUC",
                "UCF AUC",
                "Δ vs UCF",
                "Mean AUC",
                "AUC (anom.)",
                "Δ vs UCF",
                "AP",
                "AP lift",
                "Chunk F1",
                "F1 oracle",
                "Video F1",
            ],
            [
                [
                    r["modello"],
                    r["famiglia"],
                    *([] if uniform else [_seeds_of(r)]),
                    _num(_get(r, "metriche", "chunk_auc", "mivia", "media"), 4),
                    _num(_get(r, "metriche", "chunk_auc", "ucf_stessi_semi", "media"), 4),
                    _num(_get(r, "metriche", "chunk_auc", "delta", "media"), 4),
                    _num(_get(r, "metriche", "chunk_auc", "medio_fra_dataset"), 4),
                    _num(_get(r, "metriche", "chunk_auc_anomalous", "mivia", "media"), 4),
                    _num(_get(r, "metriche", "chunk_auc_anomalous", "delta", "media"), 4),
                    _num(_get(r, "metriche", "chunk_ap", "mivia", "media"), 4),
                    _num(_get(r, "lift_ap", "mivia"), 2),
                    _num(_get(r, "metriche", "chunk_f1", "mivia", "media"), 4),
                    _num(r.get("f1_oracolo"), 4),
                    _num(_get(r, "metriche", "video_f1", "mivia", "media"), 4),
                ]
                for r in crossed
            ],
        ),
        "",
        "### 6b. Same models, ranked by mean AUC",
        "",
        "The table above is ordered by MIVIA-Crime alone. This one is ordered by the "
        "average of the two"
        + (
            ", which is the order to use when picking one model to ship."
            if uniform
            else " over each row's own seeds — ⚠️ rows with different `Seeds` are not "
            "comparable here; §6c is the ranking to pick a model by."
        ),
        "",
        _tabella(
            [
                "Model",
                "Family",
                *([] if uniform else ["Seeds"]),
                "Mean AUC",
                "UCF AUC",
                "MIVIA AUC",
                "Δ",
            ],
            [
                [
                    r["modello"],
                    r["famiglia"],
                    *([] if uniform else [_seeds_of(r)]),
                    _num(_get(r, "metriche", "chunk_auc", "medio_fra_dataset"), 4),
                    _num(_get(r, "metriche", "chunk_auc", "ucf_stessi_semi", "media"), 4),
                    _num(_get(r, "metriche", "chunk_auc", "mivia", "media"), 4),
                    _num(_get(r, "metriche", "chunk_auc", "delta", "media"), 4),
                ]
                for r in sorted(
                    crossed,
                    key=lambda x: -(_get(x, "metriche", "chunk_auc", "medio_fra_dataset") or 0.0),
                )
            ],
        ),
        "",
    ]
    if uniform:
        return parts
    common = sorted(set.intersection(*(set(s) for s in sets)))
    for seed in common:
        rows = []
        for r in crossed:
            ucf, outside = _auc_at_seed(r, seed)
            mean = (ucf + outside) / 2 if ucf is not None and outside is not None else None
            rows.append((mean, r, ucf, outside))
        rows.sort(key=lambda x: -(x[0] if x[0] is not None else 0.0))
        parts += [
            f"### 6c. Every model at seed {seed}, ranked by mean AUC",
            "",
            f"The same cells for everybody — seed {seed} on both datasets — so every row is "
            "comparable with every other. This is the ranking to pick a model by. ⚠️ It is "
            "still **one** seed: gaps of a few thousandths are inside the seed-to-seed "
            "spread measured on UCF-Crime.",
            "",
            _tabella(
                ["#", "Model", "Family", "Mean AUC", "UCF AUC", "MIVIA AUC", "Δ"],
                [
                    [
                        str(i),
                        r["modello"],
                        r["famiglia"],
                        _num(mean, 4),
                        _num(ucf, 4),
                        _num(outside, 4),
                        _num(outside - ucf if ucf is not None and outside is not None else None, 4),
                    ]
                    for i, (mean, r, ucf, outside) in enumerate(rows, 1)
                ],
            ),
            "",
        ]
    return parts


def markdown(dati: Mapping[str, Any]) -> str:
    """Render the leaderboards in Markdown, in English, ready for slides.

    :param dati: the :func:`build_catalog` structure.
    :return: the whole document.
    """
    g = dati["generato"]
    models = dati["modelli"]
    order = list(models)
    label_of = {k: e for k, e, _, _, _ in METRICS}
    versions = g.get("versioni") or {}
    reported = versions.get("riportata") or {}
    versions_note: list[str] = []
    if reported:
        per_epochs: dict[int, list[str]] = defaultdict(list)
        for base, choice in reported.items():
            per_epochs[version_epochs(choice)].append(MODELS[base]["nome"])
        kept = "; ".join(
            f"{n} epoch{'s' if n > 1 else ''} for {', '.join(names)}"
            for n, names in sorted(per_epochs.items())
        )
        versions_note = [
            "> **One version per model.** The versions of the same training extended in "
            "continuation (one, three and four epochs; three and four for the combinations) "
            f"are reported once, under the plain name: {kept}. The version kept is "
            f"{versions['criterio']}. The other models are at one epoch (ablations) or as "
            "trained (sequential combinations). Every version: `--tutte-le-versioni`.",
            "",
        ]

    parts: list[str] = [
        "# Model leaderboard — all models, all metrics",
        "",
        f"Mean ± sample standard deviation over seeds {g['semi']}; "
        f"{g['n_modelli']} models, {g['n_celle']} cells. "
        f"Latency rule: {g['regola_latenza']}. "
        f"Decoding: {g['decodifica_ms_per_finestra']} ms/window. "
        f"Inference profile: {g.get('gpu_del_profilo') or 'not available'}.",
        "",
        "> The standard deviation over three seeds is a dispersion, not a confidence "
        "interval: the replicas share the same test videos.",
        "",
        *versions_note,
        "## 1. Summary",
        "",
        _tabella(
            ["Model", "Adapter params"] + [label_of[m] for m in SUMMARY_METRICS],
            [
                [models[a]["nome"], _num(models[a]["params_adattatore"], 0)]
                + [
                    _num(models[a]["metriche"][k]["media"], models[a]["metriche"][k]["decimali"])
                    + (
                        ""
                        if models[a]["metriche"][k]["sd"] is None
                        else " ± "
                        + _num(models[a]["metriche"][k]["sd"], models[a]["metriche"][k]["decimali"])
                    )
                    for k in SUMMARY_METRICS
                ]
                for a in order
            ],
        ),
        "",
        "## 2. Ranking, metric by metric",
        "",
    ]

    for key, label, direction, decimals, _ in METRICS:
        parts += [
            f"### {label} ({'higher' if direction == 'max' else 'lower'} is better)",
            "",
            _tabella(
                ["#", "Model", "Mean", "SD", "n"] + [f"seed {s}" for s in g["semi"]],
                [
                    [
                        "—" if v["rango"] is None else str(v["rango"]),
                        v["modello"],
                        _num(v["media"], decimals),
                        _num(v["sd"], decimals),
                        str(v["n"]),
                    ]
                    + [_num(v["valori"].get(str(s)), decimals) for s in g["semi"]]
                    for v in dati["classifica_globale"][key]
                ],
            ),
            "",
        ]

    parts += [
        "## 3. End-to-end alarm latency",
        "",
        "`alarm delay = grid delay + compute time + video decoding`",
        "",
        "The grid delay is how long one waits for the window covering the event to close "
        "and become scorable: it depends on the hop, not on the model. Compute and decoding "
        "are per window. Merged columns apply to reparameterization only.",
        "",
        _tabella(
            [
                "Model",
                "Grid delay (s)",
                "Compute (ms)",
                "Decoding (ms)",
                "Total (s)",
                "Compute share",
                "Real-time",
                "Merged compute (ms)",
                "Merged total (s)",
            ],
            [
                [
                    r["modello"],
                    _num(r["ritardo_griglia_s"], 3),
                    _num(r["calcolo_ms"], 1),
                    _num(r["decodifica_ms"], 1),
                    _num(r["totale_s"], 3),
                    "—" if r["quota_calcolo_pct"] is None else f"{r['quota_calcolo_pct']:.1f}%",
                    "—" if r["fattore_realtime"] is None else f"{r['fattore_realtime']:.1f}×",
                    _num(r["calcolo_fuso_ms"], 1),
                    _num(r["totale_fuso_s"], 3),
                ]
                for r in dati["latenza"]
            ],
        ),
        "",
        "## 4. Per class",
        "",
        "`n_pos` is the number of positive chunks the row rests on. A row with a "
        "single-digit `n_pos` is not a measurement.",
        "",
    ]
    for class_name, entry in dati["per_classe"].items():
        parts += [
            f"### {class_name} — {entry['n_pos_chunk']} positive chunks",
            "",
            _tabella(
                ["Model", "AUC", "AP", "Event recall", "Events"],
                [
                    [
                        name,
                        _num(r["auc"]["media"], 4),
                        _num(r["ap"]["media"], 4),
                        _num(r["event_recall"]["media"], 3),
                        _num(r["n_eventi"], 0),
                    ]
                    for name, r in entry["modelli"].items()
                ],
            ),
            "",
        ]

    video_rows = []
    for a in order:
        name = models[a]["nome"]
        perfect = false_ones = never = 0
        for entry in dati["per_video"].values():
            row = entry["modelli"].get(name)
            if row is None:
                continue
            if entry["anomalo"]:
                if row["f1"]["media"] == 1.0:
                    perfect += 1
                if row["n_semi_predetto_positivo"] == 0:
                    never += 1
            elif row["n_semi_predetto_positivo"] > 0:
                false_ones += 1
        video_rows.append([name, str(perfect), str(false_ones), str(never)])

    parts += [
        "## 5. Per video",
        "",
        f"{len(dati['per_video'])} test videos. The full table — one row per (video, model) — "
        "is in the JSON; CSV rankings include only anomalous videos. Here is the summary.",
        "",
        _tabella(
            [
                "Model",
                "Anomalous videos at F1 = 1.0",
                "Normal videos with a false alarm",
                "Anomalous videos never flagged",
            ],
            video_rows,
        ),
        "",
    ]

    parts += cross_dataset_section(dati.get("cross_dataset") or [])
    return "\n".join(parts)


def write_rankings_csv(dati: Mapping[str, Any], percorso: Path) -> None:
    """Write the anomalous per-video table, one row per (video, model).

    :param dati: the :func:`build_catalog` structure.
    :param percorso: the CSV file to write.
    """
    columns = [
        "video_id",
        "anomalo",
        "n_finestre",
        "n_positive_vere",
        "modello",
        "tasso_falsi_positivi_media",
        "n_semi_predetto_positivo",
        "n_semi",
    ]
    columns += [f"{k}_{s}" for k in VIDEO_METRICS for s in ("media", "sd", "n")]
    columns += list(VIDEO_STATES)
    with percorso.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for video_id, entry in dati["per_video"].items():
            if not entry["anomalo"]:
                continue
            for name, row in entry["modelli"].items():
                writer.writerow(
                    [
                        _csv_value(v)
                        for v in [
                            video_id,
                            int(entry["anomalo"]),
                            entry["n_finestre"],
                            entry["n_positive_vere"],
                            name,
                            row["tasso_falsi_positivi"]["media"],
                            row["n_semi_predetto_positivo"],
                            row["n_semi"],
                            *[row[k][s] for k in VIDEO_METRICS for s in ("media", "sd", "n")],
                            *[
                                ";".join(sorted({v[k] for v in row["per_seme"].values()}))
                                for k in VIDEO_STATES
                            ],
                        ]
                    ]
                )


def write_cross_dataset_csv(dati: Mapping[str, Any], percorso: Path) -> None:
    """Write the MIVIA-Crime table, one row per model.

    :param dati: the :func:`build_catalog` structure.
    :param percorso: the CSV to write; nothing is written if no model has a cross-dataset
        evaluation.

    Every metric carries three columns (MIVIA-Crime value, UCF-Crime value at the same seeds,
    delta), since the comparison is paired.
    """
    rows = dati.get("cross_dataset") or []
    if not rows:
        return
    columns = ["chiave", "modello", "famiglia", "semi", "n_video", "n_chunk"]
    columns += ["prevalenza_mivia", "prevalenza_ucf", "lift_ap_mivia", "lift_ap_ucf"]
    columns += ["soglia", "soglia_origine", "f1_oracolo"]
    # The cross-dataset mean sits among the summary columns: it is the sorting key when
    # choosing a single model.
    columns += ["chunk_auc_medio_fra_dataset"]
    for key, _, _ in MIVIA_METRICS:
        columns += [f"{key}_mivia", f"{key}_ucf_stessi_semi", f"{key}_delta"]
    with percorso.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for row in rows:
            values = [
                row["chiave"],
                row["modello"],
                row["famiglia"],
                ";".join(str(s) for s in row["semi"]),
                row["n_video"],
                row["n_chunk"],
                _get(row, "prevalenza", "mivia"),
                _get(row, "prevalenza", "ucf"),
                _get(row, "lift_ap", "mivia"),
                _get(row, "lift_ap", "ucf"),
                _get(row, "soglia", "valore"),
                _get(row, "soglia", "origine"),
                row.get("f1_oracolo"),
                _get(row, "metriche", "chunk_auc", "medio_fra_dataset"),
            ]
            for key, _, _ in MIVIA_METRICS:
                values += [
                    _get(row, "metriche", key, "mivia", "media"),
                    _get(row, "metriche", key, "ucf_stessi_semi", "media"),
                    _get(row, "metriche", key, "delta", "media"),
                ]
            writer.writerow([_csv_value(v) for v in values])


def write_run_csv(dati: Mapping[str, Any], runs_dir: Path, uscita: Path) -> None:
    """Write per-run leaderboards and the full seed detail.

    :param dati: the already verified leaderboards.
    :param runs_dir: folder containing the original runs.
    :param uscita: folder of the overall ``per_video_run.csv``.
    """
    all_items = []
    for model in dati["modelli"].values():
        for seed, run_id in model["run_id"].items():
            rows = []
            for video_id, video in dati["per_video"].items():
                if not video["anomalo"]:
                    continue
                entry = video["modelli"].get(model["nome"], {}).get("per_seme", {}).get(seed)
                if entry is not None:
                    rows.append(
                        {
                            "run_id": run_id,
                            "modello": model["nome"],
                            "seed": int(seed),
                            "video_id": video_id,
                            **{
                                k: v
                                for k, v in entry.items()
                                if k not in ("precision", "recall", "f1")
                            },
                        }
                    )
            if not rows:
                continue
            # Two groups: anomalous videos by decreasing F1/AP/AUC, normal ones by increasing FPR.
            rows.sort(
                key=lambda r: (
                    not r["anomalo"],
                    -(r["chunk_f1"] or 0) if r["anomalo"] else r["tasso_falsi_positivi"],
                    -(r["chunk_ap"] or 0),
                    -(r["chunk_auc"] or 0),
                    r["video_id"],
                )
            )
            positions = defaultdict(int)
            for row in rows:
                group = "anomali" if row["anomalo"] else "normali"
                positions[group] += 1
                row["gruppo_classifica"] = group
                row["posizione_gruppo"] = positions[group]
            run_dir = run_dir_of(runs_dir, run_id)
            _write_csv_rows(rows, run_dir / "per_video.csv")
            classes_dir = ensure_dir(run_dir / "per_video_classi")
            for class_name in sorted({c for r in rows for c in r["classi_annotate"]}):
                of_class = [r for r in rows if class_name in r["classi_annotate"]]
                # Medians of the whole video, not only of this class's events.
                of_class.sort(
                    key=lambda r: (
                        r["delta_score_mediano_chunk"] is None,
                        -(r["delta_score_mediano_chunk"] or 0),
                        r["video_id"],
                    )
                )
                _write_csv_rows(
                    [
                        {
                            "run_id": r["run_id"],
                            "modello": r["modello"],
                            "seed": r["seed"],
                            "classe": class_name,
                            "video_id": r["video_id"],
                            "n_chunk_anomali": r["n_positive_vere"],
                            "n_chunk_normali": r["n_finestre"] - r["n_positive_vere"],
                            **{k: r[k] for k in VIDEO_MEDIANS},
                        }
                        for r in of_class
                    ],
                    classes_dir / f"{class_name}.csv",
                )
            all_items.extend(rows)
    if all_items:
        _write_csv_rows(all_items, uscita / "per_video_run.csv")


def _write_csv_rows(righe: list[dict[str, Any]], percorso: Path) -> None:
    with percorso.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(righe[0]))
        writer.writeheader()
        writer.writerows({k: _csv_value(v) for k, v in r.items()} for r in righe)


def _csv_value(valore: Any) -> Any:
    if isinstance(valore, (list, tuple)):
        return ";".join(str(v) for v in valore) if valore else "NA"
    return "NA" if valore is None or _nan(valore) else valore


def ranking_chunk(records: Sequence[Mapping[str, Any]], soglia: float) -> list[dict[str, Any]]:
    """Sort the test chunks separately into TP, TN, FP and FN.

    :param records: original ``scores.jsonl`` rows, possibly with other partitions.
    :param soglia: run operating threshold; a score equal to the threshold is positive.
    :return: TP/FP by decreasing score and TN/FN increasing, with the position in the group.
    :raises ValueError: if threshold, scores, labels or chunk identities are invalid.
    """
    if not math.isfinite(soglia) or not 0 <= soglia <= 1:
        raise ValueError("invalid threshold")
    groups = {k: [] for k in ("TP", "TN", "FP", "FN")}
    views = set()
    seen_spans = set()
    anomalous = set()
    for r in records:
        if r.get("split") != "test":
            continue
        score, label = float(r["score"]), r["label"]
        if not math.isfinite(score) or not 0 <= score <= 1 or label not in (0, 1):
            raise ValueError(f"invalid score or label: {r['window_id']}")
        span = (r["video_id"], int(r["start_frame"]), int(r["end_frame"]))
        if span[1] < 0 or span[2] < span[1]:
            raise ValueError(f"invalid span: {r['window_id']}")
        if r["window_id"] in views or span in seen_spans:
            raise ValueError(f"duplicated chunk: {r['window_id']}")
        views.add(r["window_id"])
        seen_spans.add(span)
        pred = int(score >= soglia)
        outcome = ("TP" if label else "FP") if pred else ("FN" if label else "TN")
        if label:
            anomalous.add(r["video_id"])
        groups[outcome].append(
            {
                "esito": outcome,
                "posizione_esito": 0,
                "video_id": r["video_id"],
                "window_id": r["window_id"],
                "split": "test",
                "start_frame": span[1],
                "end_frame": span[2],
                "class_idx": r["class"],
                "label": int(label),
                "predizione": pred,
                "score": score,
                "soglia": soglia,
                "margine_score_soglia": score - soglia,
                "distanza_soglia": abs(score - soglia),
            }
        )
    outside = []
    for outcome, rows in groups.items():
        rows.sort(
            key=lambda r: (
                -r["score"] if outcome in ("TP", "FP") else r["score"],
                r["video_id"],
                r["start_frame"],
                r["end_frame"],
                r["window_id"],
            )
        )
        for position, r in enumerate(rows, 1):
            r["posizione_esito"] = position
            r["video_anomalo"] = r["video_id"] in anomalous
        outside.extend(rows)
    return outside


def export_chunk_ranking(
    runs_dir: Path, out: Path, *, solo: Path | None = None, tutte_le_versioni: bool = False
) -> None:
    """Write the score rankings of every run after the comparison with ``metrics.json``.

    :param runs_dir: folder of the final runs.
    :param out: destination of the count summary and the legend.
    :param tutte_le_versioni: as in :func:`build_catalog`; without it, runs of the
        non-reported versions (:func:`choose_versions`) are neither rewritten nor summarized.
    :raises ValueError: if scores or counts do not reproduce the run metrics.
    """
    found = find_cells(runs_dir, solo=solo)
    active, _ = active_models(found, tutte_le_versioni)
    summary = []
    for adapt, cells in found.items():
        if solo is None and adapt not in active:
            continue
        for seed, run_dir in sorted(cells.items()):
            hashes = {n: file_sha256(run_dir / n) for n in ("metrics.json", "scores.jsonl")}
            m = read_json(run_dir / "metrics.json")
            threshold = float(m["threshold"]["value"])
            with (run_dir / "scores.jsonl").open(encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
            rows = ranking_chunk(records, threshold)
            if not rows:
                raise ValueError(f"{run_dir.name}: no test chunk")
            groups = {e: [r for r in rows if r["esito"] == e] for e in ("TP", "TN", "FP", "FN")}
            tp, tn, fp, fn = (len(groups[e]) for e in ("TP", "TN", "FP", "FN"))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            checks = {
                "precision": (precision, m["chunk_prf"]["precision"]),
                "recall": (recall, m["chunk_prf"]["recall"]),
                "f1": (f1, m["chunk_prf"]["f1"]),
                "n_pred_pos": (tp + fp, m["chunk_prf"]["n_pred_pos"]),
                "n_true_pos": (tp + fn, m["chunk_prf"]["n_true_pos"]),
                "n_chunk": (len(rows), m["n_chunk_eval"]),
                "n_video": (len({r["video_id"] for r in rows}), m["n_eval_videos"]),
                "soglia": (threshold, m["chunk_prf"]["threshold"]),
            }
            for name, (recomputed, historical) in checks.items():
                if not math.isclose(recomputed, historical, rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError(f"{run_dir.name}: {name} does not reproduce metrics.json")
            if any(file_sha256(run_dir / n) != sha for n, sha in hashes.items()):
                raise ValueError(f"{run_dir.name}: artifacts modified while reading")
            model = (active.get(adapt) or MODELS[adapt])["nome"]
            identity = {"run_id": run_dir.name, "modello": model, "seed": seed}
            rows = [{**identity, **r} for r in rows]
            _write_csv_rows(rows, run_dir / "ranking_chunks.csv")
            for outcome in groups:
                with (run_dir / f"ranking_chunks_{outcome}.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(r for r in rows if r["esito"] == outcome)
            count = {
                **identity,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "n_chunk": len(rows),
                "soglia": threshold,
                "csv": str(run_dir / "ranking_chunks.csv"),
            }
            write_json(
                run_dir / "ranking_chunks_provenienza.json",
                {
                    **count,
                    "split": "test",
                    "sha256": hashes,
                    "coerenza_verificata": True,
                    "script_sha256": file_sha256(Path(__file__)),
                    "ordinamento": {
                        "TP": "score_desc",
                        "TN": "score_asc",
                        "FP": "score_desc",
                        "FN": "score_asc",
                    },
                },
            )
            summary.append(count)
            log.info("chunk ranking verified and written: %s", run_dir.name)
    if not summary:
        raise ValueError("no final run available")
    ensure_dir(out)
    _write_csv_rows(summary, out / "ranking_chunks_riepilogo.csv")
    write_text_atomic(
        out / "LEGGIMI_ranking_chunks.md",
        """# Chunk score rankings

Each run contains ranking_chunks.csv and four separate CSV files:
ranking_chunks_TP.csv, ranking_chunks_TN.csv, ranking_chunks_FP.csv,
ranking_chunks_FN.csv. Test only, normal videos included: they also serve to
analyse TN and FP. The threshold is the original one of the run, estimated on
validation. A score equal to the threshold is a positive prediction.

TP: detected anomalous chunks, decreasing score (best positives first).
TN: negative chunks correctly rejected, increasing score (best negatives first).
FP: flagged negative chunks, decreasing score (clearest errors first).
FN: missed anomalous chunks, increasing score (clearest errors first).
The single CSV concatenates the TP, TN, FP, FN groups. posizione_esito restarts
from 1 in each group; ties are ordered by video_id, frame and window_id.

video_id and start_frame/end_frame identify the sequence, with inclusive bounds
in native frames. class_idx is the chunk annotation, -1 on negatives; it is not
a predicted class. video_anomalo also marks the negative chunks inside an
anomalous video. Scores are copied from scores.jsonl, without rounding.

The ranking points to segments to inspect; it does not measure the quality of a
whole video on its own: a video can appear in several groups. Scores are not
calibrated probabilities and rankings are read within a single run.
Recomposed counts and P/R/F1 match metrics.json. Hashes and ordering are in
ranking_chunks_provenienza.json. The sources are not modified.

Regenerate from the repository root:
python scripts/classifiche_modelli.py --solo-ranking-chunk
""",
    )


def metrics_per_class(read_value: Any) -> list[dict[str, Any]]:
    """Compute the per-class binary diagnostics at the three granularities.

    :param read_value: run rebuilt by ``rileggi_run`` on the test partition.
    :return: one row per anomaly class; negatives and false alarms are shared.
    """
    scores, gt = read_value.scores, read_value.gt
    threshold = float(read_value.run.threshold)
    classes = read_value.bundle.classes
    normal_idx = read_value.bundle.normal_idx
    anomalous = tuple(c for c in classes if c.idx != normal_idx)
    ranking = M.per_class_chunk(scores, gt, classes, normal_idx=normal_idx)
    prf = M.per_class_chunk_prf(
        scores, gt, classes, normal_idx=normal_idx, threshold=threshold, source="val"
    )
    scores_a = {v: s for v, s in scores.items() if not gt[v].is_normal}
    ranking_a = M.per_class_chunk(scores_a, gt, classes, normal_idx=normal_idx)
    event_key, event_labels = S.by_event_class(anomalous)
    event = S.event_strata(
        scores,
        gt,
        read_value.fps,
        event_key,
        labels=event_labels,
        threshold=threshold,
        source="val",
        latency_rule="intersezione",
    )
    global_ = M.event_prf(scores, gt, threshold, source="val")
    event_fp = global_.n_allarmi - global_.n_allarmi_veri
    records = M.chunk_records(scores, gt)
    negatives = [r for r in records if not r["label"]]
    fp_chunk = sum(r["score"] >= threshold for r in negatives)
    rows = []
    for class_name in anomalous:
        pos = [r for r in records if r["class"] == class_name.idx]
        tp = sum(r["score"] >= threshold for r in pos)
        fn, tn = len(pos) - tp, len(negatives) - fp_chunk
        # Other events in the same video do not become positives of the class.
        gt_class = {
            v: TemporalAnnotation(v, tuple(i for i in a.intervals if i[2] == class_name.idx))
            for v, a in gt.items()
            if a.is_normal or any(i[2] == class_name.idx for i in a.intervals)
        }
        class_scores = {v: scores[v] for v in gt_class}
        video = M.video_decision_prf(class_scores, gt_class, threshold, source="val")
        decisions = M.video_decision_scores(class_scores, gt_class)
        vtp = sum(s >= threshold and not gt_class[v].is_normal for v, s in decisions.items())
        vfp = sum(s >= threshold and gt_class[v].is_normal for v, s in decisions.items())
        vpos = sum(not a.is_normal for a in gt_class.values())
        vneg = len(gt_class) - vpos
        lat = M.detection_latency(
            scores, gt_class, read_value.fps, threshold=threshold, rule="intersezione"
        )
        ev = event[class_name.name]
        ev_class = M.event_prf(scores, gt_class, threshold, source="val")
        chunk = prf[class_name.name]
        rows.append(
            {
                "classe": class_name.name,
                "class_idx": class_name.idx,
                "split": "test",
                "soglia": threshold,
                "chunk_auc": ranking[class_name.name].auc,
                "chunk_ap": ranking[class_name.name].ap,
                "chunk_auc_anomalous": ranking_a[class_name.name].auc,
                "chunk_ap_anomalous": ranking_a[class_name.name].ap,
                "chunk_precision": chunk.precision,
                "chunk_recall": chunk.recall,
                "chunk_f1": chunk.f1,
                "chunk_accuracy": (tp + tn) / (len(pos) + len(negatives)),
                "chunk_tp": tp,
                "chunk_fp_condivisi": fp_chunk,
                "chunk_fn": fn,
                "chunk_tn_condivisi": tn,
                "chunk_n_positivi": len(pos),
                "chunk_n_negativi_condivisi": len(negatives),
                "chunk_fpr": fp_chunk / len(negatives) if negatives else None,
                "video_precision": video.precision,
                "video_recall": video.recall,
                "video_f1": video.f1,
                "video_accuracy": (vtp + vneg - vfp) / len(gt_class),
                "video_tp": vtp,
                "video_fp_condivisi": vfp,
                "video_fn": vpos - vtp,
                "video_tn_condivisi": vneg - vfp,
                "video_n_positivi": vpos,
                "video_n_normali_condivisi": vneg,
                "video_fpr": vfp / vneg if vneg else None,
                "event_precision": ev.precision,
                "event_recall": ev.recall,
                "event_f1": ev.f1,
                "event_n_eventi": ev.n_eventi,
                "event_n_rilevati": ev.n_rilevati,
                "event_n_mancati": ev.n_eventi - ev.n_rilevati,
                "event_n_irraggiungibili": ev_class.n_eventi_irraggiungibili,
                "event_n_allarmi_veri": ev.n_allarmi_veri,
                "event_n_falsi_allarmi_condivisi": event_fp,
                "event_n_allarmi_globali": global_.n_allarmi,
                "event_latenza_mediana_s": lat.mediana_s,
                "event_latenza_p90_s": lat.p90_s,
                "event_n_senza_latenza": lat.n_senza_latenza_misurabile,
                "event_latenza_stato": "misurata" if ev.n_rilevati else "nessun_evento_rilevato",
                "chunk_regola": "positivi_classe_vs_negativi_globali",
                "video_regola": "eventi_classe_vs_video_normali_decisione_asimmetrica",
                "event_regola": "falsi_allarmi_globali_condivisi_intersezione",
            }
        )
    return rows


def export_per_class(
    runs_dir: Path,
    out: Path,
    overrides: list[str],
    *,
    solo: Path | None = None,
    tutte_le_versioni: bool = False,
) -> None:
    """Recompute and verify every run before writing the per-class CSVs.

    :param runs_dir: folder of the completed E2/E3 runs.
    :param out: destination of the overall table and the legend.
    :param overrides: local paths of the data group.
    :param tutte_le_versioni: as in :func:`build_catalog`; without it, runs of the
        non-reported versions (:func:`choose_versions`) are neither rewritten nor included.
    :raises ValueError: if the re-reading does not reproduce the original metrics.
    """
    found = find_cells(runs_dir, solo=solo)
    active, _ = active_models(found, tutte_le_versioni)
    ready = []
    for adapt, cells in found.items():
        if solo is None and adapt not in active:
            continue
        model = (active.get(adapt) or MODELS[adapt])["nome"]
        for seed, run_dir in sorted(cells.items()):
            read_value = reread_run(run_dir, split="test", overrides=overrides)
            m = read_json(run_dir / "metrics.json")
            rows = metrics_per_class(read_value)
            scores, gt, threshold = read_value.scores, read_value.gt, read_value.run.threshold
            checks = {
                "chunk_auc": (M.dataset_chunk_auc(scores, gt), m["chunk_auc"]),
                "chunk_ap": (M.dataset_chunk_ap(scores, gt), m["chunk_ap"]),
            }
            for name, value in (
                ("chunk_prf", M.dataset_chunk_prf(scores, gt, threshold, source="val")),
                ("video_prf", M.video_decision_prf(scores, gt, threshold, source="val")),
                ("event_prf", M.event_prf(scores, gt, threshold, source="val")),
            ):
                for k, v in value.as_dict().items():
                    if k not in ("source", "threshold") and k in m[name]:
                        checks[f"{name}.{k}"] = (v, m[name][k])
            for row in rows:
                ref = m["per_class_chunk"][row["classe"]]
                for k in ("auc", "ap"):
                    checks[f"{row['classe']}.{k}"] = (row[f"chunk_{k}"], ref[k])
                checks[f"{row['classe']}.n_pos"] = (row["chunk_n_positivi"], ref["n_pos"])
            historical = load_cell(run_dir, runs_dir / "analisi_e3_20260907")["strata"]
            for row in rows:
                ref = historical.get("evento", {}).get("classe", {}).get(row["classe"], {})
                for field, original in {
                    "event_precision": "precision",
                    "event_recall": "recall",
                    "event_f1": "f1",
                    "event_n_eventi": "n_eventi",
                    "event_n_rilevati": "n_rilevati",
                    "event_n_allarmi_veri": "n_allarmi_veri",
                    "event_latenza_mediana_s": "latenza_mediana_s",
                }.items():
                    if original in ref:
                        checks[f"{row['classe']}.{field}"] = (row[field], ref[original])
            checks["somma_eventi"] = (
                sum(r["event_n_eventi"] for r in rows),
                m["event_prf"]["n_eventi"],
            )
            checks["somma_rilevati"] = (
                sum(r["event_n_rilevati"] for r in rows),
                m["event_prf"]["n_eventi_rilevati"],
            )
            for name, (new, old) in checks.items():
                if not (
                    (_nan(new) and _nan(old))
                    or math.isclose(float(new), float(old), rel_tol=1e-9, abs_tol=1e-9)
                ):
                    raise ValueError(f"{run_dir.name}: {name}: {new} != {old}")
            rows = [{"run_id": run_dir.name, "modello": model, "seed": seed, **r} for r in rows]
            provenance = {
                "run_id": run_dir.name,
                "split": "test",
                "threshold": float(threshold),
                "origine": "diagnostica_post_hoc",
                "annotation_source": read_value.bundle.annotation_source,
                "data_path_overrides": overrides,
                "coerenza_verificata": True,
                "script_sha256": file_sha256(Path(__file__)),
                "n_verifiche": len(checks),
                "sha256": {
                    n: file_sha256(run_dir / n)
                    for n in ("metrics.json", "scores.jsonl", "config.yaml")
                },
            }
            ready.append((run_dir, rows, provenance))
            log.info("per class verified %s", run_dir.name)
    if not ready:
        raise ValueError("no final run with available metrics")
    ensure_dir(out)
    all_items = []
    for run_dir, rows, provenance in ready:
        _write_csv_rows(rows, run_dir / "per_class.csv")
        write_json(run_dir / "per_class_provenienza.json", provenance)
        all_items.extend(rows)
    _write_csv_rows(all_items, out / "per_class.csv")
    write_json(out / "per_class_provenienza.json", [p for _, _, p in ready])
    write_text_atomic(
        out / "LEGGIMI_per_class.md",
        """# Binary metrics per annotated class

13 anomalous test classes; one row per class, model and seed. Each run has
per_class.csv and per_class_provenienza.json. The overall CSV concatenates the runs.
Original validation threshold. No training and no new inference.

Chunk: positives of the dominant class of the chunk against all negative test
chunks (ADR-031). The other positive classes are excluded. AUC/AP and P/R/F1 use
the same set. The *_anomalous columns also restrict negatives to anomalous videos.
TP/FN belong to the class; FP/TN and negatives are shared.

Video: videos containing at least one event of the class against all normal
videos. In anomalous videos only an above-threshold chunk intersecting an event of
that class counts; in normal videos any above-threshold chunk does. A video with
events of several classes appears in several rows. The class comes from the
annotated intervals, not from the file or folder name. Per-class diagnostics of
the ADR-047 rule.

Event: strata.json definition (ADR-060). Precision = true alarms of the class /
(true alarms of the class + global false alarms). Recall = detected events /
events of the class. F1 is the harmonic mean. An alarm can touch several classes;
per-class precision/F1 do not add up to the global metrics. Median and p90
latency on detected events only, intersection rule (ADR-061). False alarms do
not become FP events: the denominators are distinct.

NA marks an undefined metric or an unmeasurable latency. Chunk and video precision
without predictions follows the project's 0 convention; event precision without
true or false alarms stays NA, as in strata.json. There is no video (ADR-047) or
event AUC/AP, and no event TN/accuracy. Historical metrics are not rewritten.
Before export, global and per-class AUC/AP, global P/R/F1 and event counts are
checked against metrics.json.
""",
    )
    log.info("written per_class.csv: %d runs, %d rows overall", len(ready), len(all_items))


def build_parser() -> argparse.ArgumentParser:
    """:return: the CLI parser."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--runs-dir", type=Path, default=Path("runs"), help="the runs folder")
    p.add_argument(
        "--run",
        type=Path,
        default=None,
        help="ONE run folder, entering without the MODELS and "
        "EXPERIMENTS filters. For runs outside the matrix run_id grammar (the "
        "cross-dataset evaluations of exp/esterno.py, ADR-066), which otherwise stay "
        "invisible, rightly so: a cell measured on another dataset "
        "inside the UCF-Crime leaderboards would be a plausible and false row. "
        "With --run also use a different --out, or the real leaderboards are overwritten",
    )
    p.add_argument(
        "--cache-ablation",
        type=Path,
        default=Path("runs/analisi_e3_20260907"),
        help="ablation analysis cache, for latency and strata",
    )
    p.add_argument(
        "--profilo",
        type=Path,
        default=Path("runs/e5b3_efficienza_gpuq"),
        help="folder with efficiency_inference.json; the reference machine (A100)",
    )
    p.add_argument(
        "--profilo-altra-gpu",
        type=Path,
        default=Path("runs/e5b3_efficienza_aiq"),
        help="second profile on another GPU (H100): separate columns, never mixed",
    )
    p.add_argument(
        "--log-slurm",
        type=Path,
        default=Path("../hpc/log"),
        help="SLURM log folder, the source of the training step cost",
    )
    p.add_argument(
        "--semi-mivia",
        type=int,
        nargs="+",
        default=[42],
        help="seeds of the MIVIA-Crime evaluations entering section 6. Default: "
        "42 only, on request, the same for every model even where more exist "
        "on disk (E3-S)",
    )
    p.add_argument(
        "--tutte-le-versioni",
        action="store_true",
        help="report every version of the extended trainings (VERSION_GROUPS), each named "
        "with its epochs. Default: one per group, the best seed-42 test chunk AUC "
        "(choose_versions), under the plain name (decision of 14 October 2026)",
    )
    p.add_argument("--out", type=Path, default=Path("runs/classifiche"), help="output folder")
    p.add_argument(
        "--solo-per-classe",
        action="store_true",
        help="recompute and export only per_class.csv for every run and overall",
    )
    p.add_argument(
        "--solo-ranking-chunk",
        action="store_true",
        help="export ranking_chunks.csv and TP/TN/FP/FN for every run",
    )
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="path override of the data group for local re-reading",
    )
    p.add_argument(
        "--consenti-incoerenze",
        action="store_true",
        help="write even if a recomputation does not reproduce metrics.json, and say why you should not",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Build and write the leaderboards.

    :param argv: command-line arguments.
    :return: exit code; ``1`` if a consistency check failed.
    """
    setup_logging()
    args = build_parser().parse_args(argv)
    if args.run is not None and args.out == Path("runs/classifiche"):
        # The `--out` default is the folder of the REAL leaderboards, used by the deck: with a single
        # run inside, `classifiche.json` and `per_video.csv` would be rewritten with a one-row table.
        log.error(
            "--run with the default --out would rewrite %s, which contains the leaderboards of "
            "every model. Pass a dedicated --out (e.g. runs/classifiche_%s)",
            args.out,
            args.run.name.rpartition("__")[2] or "run",
        )
        return 1
    if args.solo_ranking_chunk:
        export_chunk_ranking(
            args.runs_dir, args.out, solo=args.run, tutte_le_versioni=args.tutte_le_versioni
        )
        return 0
    if args.solo_per_classe:
        export_per_class(
            args.runs_dir,
            args.out,
            args.overrides,
            solo=args.run,
            tutte_le_versioni=args.tutte_le_versioni,
        )
        return 0
    cache = args.cache_ablation if args.cache_ablation and args.cache_ablation.is_dir() else None
    profile = args.profilo if args.profilo and args.profilo.is_dir() else None
    alt = args.profilo_altra_gpu
    alt_profile = alt if alt and alt.is_dir() else None

    log_slurm = args.log_slurm if args.log_slurm and args.log_slurm.is_dir() else None
    data = build_catalog(
        args.runs_dir,
        cache,
        profile,
        alt_profile,
        log_slurm,
        args.overrides,
        solo=args.run,
        semi_mivia=args.semi_mivia,
        tutte_le_versioni=args.tutte_le_versioni,
    )
    data["generato"]["data_path_overrides"] = args.overrides
    data["generato"]["semi_mivia"] = list(args.semi_mivia)

    if data["incoerenze"]:
        for run_id, problems in data["incoerenze"].items():
            log.error("%s: %s", run_id, "; ".join(problems))
        if not args.consenti_incoerenze:
            log.error(
                "a recomputation does not reproduce metrics.json: the re-reading is not faithful and the "
                "new numbers would not be credible. Use --consenti-incoerenze only if you know why."
            )
            return 1

    output = ensure_dir(args.out)
    write_json(output / "classifiche.json", data)
    write_text_atomic(output / "classifiche.md", markdown(data))
    write_rankings_csv(data, output / "per_video.csv")
    write_run_csv(data, args.runs_dir, output)
    write_cross_dataset_csv(data, output / "cross_dataset_mivia.csv")
    write_text_atomic(
        output / "LEGGIMI_per_video.md",
        """# Per-video leaderboards

`per_video.csv`: one row per video and model, with mean, sample standard
deviation and number of valid seeds for each metric. Anomalous videos only.
`per_video_run.csv`: all runs, one row per video, model and seed.
Each `runs/<run_id>/per_video.csv` holds the same rows for a single run.

score_mediano_chunk_anomali uses all chunks with true label 1 (TP and FN).
score_mediano_chunk_normali uses all those with label 0 (TN and FP).
delta_score_mediano_chunk = anomalous median - normal median. They do not depend
on the threshold. If a set is empty, its median and the delta are NA.
In the aggregated CSV, *_media and *_sd summarise the medians computed per seed;
scores of different runs are never mixed before taking a median.

Each run also has per_video_classi/<Class>.csv, with these three metrics and the
counts for each video of the class, sorted by decreasing delta (NA last). The
class comes from the annotated events: a multi-class video appears in several
files, always with the medians of the whole video. The per-video CSV files keep
the selection of the 140 anomalous videos; the JSON also holds the normal ones.

Test only; validation threshold of the run. Scores from `scores.jsonl`, geometry
from the protocol sampler, events from the lab annotations. Pooled AUC/AP, chunk
counts, video decisions and event counts are checked against `metrics.json`.
The original run receipts stay untouched. Local paths and provenance are
recorded in `classifiche.json`, in the `generato` block.

In the per-run CSV files, anomalous videos are sorted by decreasing chunk F1, AP
and AUC; ties are broken by video_id. `posizione_gruppo` is not a statistical
rank. The aggregated CSV is sorted by video and model, to ease comparison.

AP is 1 when all chunks are positive: the previous export omitted it by using
the AUC guard. AUC needs both labels: without negatives it is NA, as is the
false positive rate. Precision without predictions is 0 by convention, as in the
project metrics; recall and F1 are 0 when no positive is detected.
NA marks an undefined or unmeasurable value, never a zero. The *_stato columns
explain why. *_n counts the valid seeds; an SD needs at least two seeds. Latency
is NA when no event is detected, and its means exclude those seeds: read *_n and
`latenza_stato`.
The JSON also keeps the normal videos to check the full test set: on normal
videos chunk and event precision/recall/F1 are null; counts, accuracy and FPR
remain, as they are defined even without anomalies.
In the CSV files precision/recall/F1 appear only with the chunk_ prefix; the
historical aliases stay in the JSON for compatibility with existing summaries.
Latency uses the intersection rule (ADR-061), in seconds on detected events only:
always read n_eventi_mancati too. It does not include compute time.
Per-video means do not reconstruct the pooled AUC/AP of the whole test set.
""",
    )
    log.info(
        "written %d leaderboards over %d models (%d cells) in %s",
        len(data["classifica_globale"]),
        data["generato"]["n_modelli"],
        data["generato"]["n_celle"],
        output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
