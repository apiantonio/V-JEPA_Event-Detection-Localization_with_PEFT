"""The official table of the class-routing localization: AP offline and online, FP metric, heads.

One page (decided on 28 October 2026) with a table of the metric columns, rows named after
detector, configuration, box variant and score, the per-class values, and the classification
metrics of the heads. CPU only; reads the campaign folder
(``runs/localization-class-routing-20261021``), the detector runs and the temporal
annotations, and writes ``evaluation/tabella_localizzazione.{json,html}`` (``--output``
elsewhere).

- **Localization** (GT reading ``union`` only): the per-event AP at the IoU thresholds
  0.05:0.95 of ``localization_class_ap.py evaluate --iou-grid fine``, averaged per event over
  ``[0.3:0.05:0.7]`` (main) and ``[0.5:0.05:0.95]`` (COCO), then over the events of a class and
  over the thirteen classes. Offline: ``evaluation/ap2-offline-<stage>`` (every variant), or the
  legacy variants of ``evaluation/ap-rbdc-<stage>`` before it exists; online:
  ``evaluation/ap2-online-<stage>``, every GT frame on every chunk that holds it, with no box
  on the chunks the detector does not flag: their GT boxes count as missed (30 October 2026;
  before, those chunks were left out and an event the detector missed left its class).
- **1 fps and the FP column** (decision of 30 October 2026,
  ``docs/varie/metriche_localizzazione_1fps_fpr_ovad_20261030.md``): AP offline and online and
  the FP metric on one frame per second, the boxes shown on the chunks with ``h(c) >= θ_O``
  (the threshold of the OVAD rows), IG at 32 nodes: the node set ``1fps``
  (``evaluation/f1-n32-*``), the page's default. The FP column is ``FPR_box@TPR(0.90)``, the
  share of the 1 fps frames of the OVAD FP rows (guard 30 s, clean chunks only) still showing
  a box at the threshold that keeps one on 90% of the annotated frames of the TP rows
  (:func:`frame_fpr_cells`); its other points, means, AUROC and ceiling, and the FP metrics
  that came before it (``B^σ_FP@TPR_P``, FAR@TPR90, ...), are in the page's FP tables.
- **FP metrics** (``docs/varie/metrica_fp_localizzazione_20261028.md`` §4): FAR@TPR90, the share
  of false positive chunks (flagged, outside every temporal event) showing at least one box at
  or above the box confidence that keeps 90% of the true positive chunks (flagged, holding GT
  frames); lower is better. The control AUROC of the chunk confidence ``S(c)`` (the highest box
  score of the chunk) between true and false positive chunks, and the same two numbers for the
  detector's binary head on the same chunks, as the reference.
- **Heads**: the binary heads of the three detectors on every test chunk (threshold: Youden on
  validation, transferred to MIVIA-Crime) and the multiclass heads (B1, B2, B3, B4) on the
  annotated test chunks (Normal left out before the argmax, as the routing does). Two ways to
  average, both in the page: per event, then class, then the thirteen classes (accuracy and
  recall; precision, AUC and the F1 of the displayed precision and recall stay per class), and
  the chunk-level standard ones.
- **Ladder** (29 October 2026): the gap between the attentive-probe baseline on the
  frozen V-JEPA 2.1 and our detectors, each with the simplest box rule (global dictionary, one
  box per concept, IG mass) and with its best rule: on the baseline the best class-free row, on
  ours the best row without the B4 classifiers (left out of every comparison). "Best" is the
  highest mean of the two datasets of the offline AP@[0.3:0.05:0.7], the oracle left out;
  chosen on test, exploratory.
- **Comparisons** (approved on 29 October 2026): methods of other paradigms
  (Grad-CAM, IG alone, three VTCD-only methods) against ours in two rows, the class predicted
  by a multiclass head or chosen by the dictionaries (NNLS, no head), on the LoRA+VPT base
  detector; then the ablations of our method, one axis at a time from either row.
- **IG nodes** (28 October 2026): the same table with IG integrated on 16, 32, 64 and
  128 Gauss-Legendre nodes instead of 256 (``scripts/localization_class_ig_nodes.py``, folders
  ``evaluation/ign-n<N>-*``), class macros only, plus a rerun at 256 nodes as the control of
  the GPU nondeterminism, and the integral time per chunk of the benchmark of 28 October.

``--exclude-stage`` leaves a stage out of the page, with the class heads only its rows use
(``--exclude-stage b4`` drops the B4 classifiers everywhere).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tarfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402

from localization_class_plan import CLASSES  # noqa: E402

CAMPAIGN = ROOT / "runs/localization-class-routing-20261021"
#: Stages of the AP, with the detector whose binary head flags their chunks.
STAGES = {
    "campaign": "campaign",
    "b3s": "b3s",
    "b3g": "b3g",
    "b4": "campaign",
    "attentive": "attentive",
    "cmp-ig": "campaign",
    "cmp-other": "campaign",
}
#: Stages whose boxes do not depend on IG: the same rows at every node count.
NO_IG_STAGES = {"cmp-other"}
#: Class heads used only by the rows of one stage, dropped with it by ``--exclude-stage``.
STAGE_SELECTORS = {"b4": ("B4L", "B4V")}
DETECTORS = {
    "campaign": ("LoRA+VPT base", "e3s_ucf_vjepa21l_vptshallow-n8-su-lora-r32_s42"),
    "b3s": ("LoRA+VPT +multiclass (sum)", "lcb3s_ucf_vjepa21l_vptshallow-n8-su-lora-r32_s42"),
    "b3g": (
        "LoRA+VPT +multiclass (GradNorm)",
        "lcb3g_ucf_vjepa21l_vptshallow-n8-su-lora-r32_s42",
    ),
    "attentive": ("Frozen attentive probe", "e2d_ucf_vjepa21l_frozen-attentive_s42"),
}
DATASETS = (("ucf_test", "FS-UCF-Crime"), ("mivia_test", "MIVIA-Crime"))
#: IoU ranges of the official table (decision of 28 October 2026).
RANGES = {"main": (0.3, 0.7), "coco": (0.5, 0.95)}
#: Share of the true positive chunks whose box the FP threshold keeps.
TPR = 0.90
#: TPR targets of the box count ``B_FP@TPR_C(q)`` in percent, and of its discrete mean
#: ``B̄_FP[0.70:0.05:0.95]`` (30 October 2026).
BFP_GRID = (70, 75, 80, 85, 90, 95)
#: Row values of the box count: ``B_FP@TPR_C(0.90)`` and the mean over :data:`BFP_GRID`.
BOX_METRICS = ("bfp90", "bfpm")
#: The mean kept share of false boxes over the confidence thresholds, with no TPR,
#: ``F̄_box^θ`` (``docs/varie/proposte_metriche_fp_localizzazione_20261029.md`` §5): the only
#: FP column of the result tables from 30 October 2026; the other FP metrics move to a table of
#: their own.
FBOX = "fbox"
#: FP values stored as ``{"value"}`` and averaged over the two datasets: the box counts,
#: ``F̄_box^θ``, the AUROC of ``S_C`` between true and false positive chunks, and the false boxes
#: produced per false positive chunk (``N⁻_box / |F|``).
#: The false boxes per false positive chunk, each box filtered by its own score at the threshold
#: that keeps a box on q of the positive planes, ``B^σ_FP@TPR_P`` (30 October 2026):
#: the FP columns of the result tables, the point at 0.90 and the mean over
#: :data:`FALSE_BOX_GRID`; the continuous means over [0.80, 0.95] and [0.70, 0.95], the share of
#: false positive chunks still showing a box and the boxes each shows go to the FP tables.
FALSE_BOX_METRICS = ("bsig90", "bsigm", "bsigm95", "bsigm70", "bsig_far", "bsig_per")
#: Targets of the mean ``B̄^σ_FP[0.80:0.01:0.90]``, in percent. Up to 0.90 and not 0.95: on
#: MIVIA-Crime Grad-CAM and the class-routed VTCD put a box on only 92.7-93.3% of the positive
#: planes, so 0.95 is unreachable for them, and a mean is compared only where every method has it.
FALSE_BOX_GRID = tuple(range(80, 91))
#: The FP metric of the official table from 30 October 2026 (note §5): ``FPR_box@TPR(q)`` at
#: the four points, its continuous means over [0.80, 0.90], [0.80, 0.95] and [0.70, 0.95], the
#: AUROC of the frame scores and the ceiling ``FPR(-inf)``; ``fpr90`` is the table's column.
FPR_METRICS = (
    "fpr70",
    "fpr75",
    "fpr90",
    "fpr80",
    "fpr85",
    "fpr95",
    "fprm8090",
    "fprm8095",
    "fprm7095",
    "fpr_auroc",
    "fpr_ceiling",
)
VALUE_METRICS = (
    *FPR_METRICS,
    *BOX_METRICS,
    FBOX,
    "auroc",
    "nbox",
    "fb90",
    "fbm",
    "auroc_box",
    *FALSE_BOX_METRICS,
    "fn_main",
    "fn_coco",
)
#: The online evaluations of the old rule, which left the detector's false negative chunks out
#: (archived on 30 October 2026): what :func:`gap_split` compares the new rule with.
FN_ARCHIVE = "online-fn-esclusi-20261029"
#: Targets of ``F̄_box,B[0.70:0.05:0.95]``, in percent of the boxes on positive planes.
RETENTION_GRID = (70, 75, 80, 85, 90, 95)
#: Threshold interval of ``F̄_box^θ`` per confidence label (:func:`row_labels`): a share of the
#: chunk's mass lies in ``[0, 1]``, its whole scale, where the metric is the mean share of the
#: false boxes. The interval of a confidence that is not a share is not decided yet (no value).
FBOX_INTERVALS: dict[str, tuple[float, float]] = {}
#: Categories of the box score histograms (``localization_class_ap.BOX_CATEGORIES``).
BOX_CATEGORIES = ("T", "F")
#: Who chooses the class, by configuration prefix.
SELECTORS = {
    "A1": "NNLS-{rule}",
    "A2": "NNLS-{rule}",
    "B1": "MultiClass-Branch",
    "B2": "MultiClass-Full",
    "B3": "MultiClass-Own",
    "B4L": "MultiClass-LoRA",
    "B4V": "MultiClass-LoRA→VPT",
    "Oracle": "Oracle",
}
#: Display names of the class rules of proposal A (the ids stay those of the configurations).
RULES = {
    "somma-grezza": "RawSum",
    "regione-un-voto": "RegionOneVote",
    "token-un-voto": "TokenOneVote",
    "voto-netto": "NetVote",
}
DICTIONARIES = {"A1": "Concepts-References", "A2": "Concepts-AllChunks"}
VARIANTS = {
    "union": "Cumulative union of the concepts",
    "component": "8-conn. components · component conf.",
    "concept-union": "Union of the concept's components",
    "concept-box": "Concept box",
    "component-concept-score": "8-conn. components · concept conf.",
    "largest-component": "Largest 8-conn. component",
    "nms30-component": "8-conn. components + NMS 0.3",
    "nms50-component": "8-conn. components + NMS 0.5",
    "nms70-component": "8-conn. components + NMS 0.7",
    "nms30-concept-box": "Concept box + NMS 0.3",
    "nms50-concept-box": "Concept box + NMS 0.5",
    "nms70-concept-box": "Concept box + NMS 0.7",
    "softnms-component": "8-conn. components + Soft-NMS",
    "softnms-concept-box": "Concept box + Soft-NMS",
    "wbf-component": "8-conn. components + WBF",
    "wbf-concept-box": "Concept box + WBF",
    "component4": "4-conn. components · component conf.",
    "concept-union4": "Union of the concept's 4-conn. components",
    "component4-concept-score": "4-conn. components · concept conf.",
    "largest-component4": "Largest 4-conn. component",
    "region": "SLIC regions · region conf.",
    "region-concept-score": "SLIC regions · concept conf.",
    "region-combos-nms50": "Region combinations + NMS 0.5 (YOLO)",
    "region-combos-nms50-concept": "Region combinations + NMS 0.5 per concept",
}
SCORES = {"mass": "IG mass", "share": "IG share"}
#: The comparison configurations (``localization_compare.py``; stage ``cmp-ig``, IG alone,
#: and ``cmp-other``, without IG): name, box rule, what the ``mass`` score is, and whether the
#: ``share`` score (that confidence over the chunk total) is a share worth a row.
COMPARE_CONFIGS = {
    "IG-tree": ("IG alone", "8-conn. tree + NMS 0.5", "IG mass", True),
    "IG-tree-density": ("IG alone", "8-conn. tree + NMS 0.5", "IG density", False),
    "IG-tree-peak": ("IG alone", "8-conn. tree + NMS 0.5", "IG peak", False),
    "IG-tree-level": ("IG alone", "8-conn. tree + NMS 0.5", "level t", False),
    "IG-tree-persistence": ("IG alone", "8-conn. tree + NMS 0.5", "persistence", False),
    "IG-tree-conn4": ("IG alone", "4-conn. tree + NMS 0.5", "IG mass", True),
    "IG-tree-nms30": ("IG alone", "8-conn. tree + NMS 0.3", "IG mass", True),
    "IG-tree-nms70": ("IG alone", "8-conn. tree + NMS 0.7", "IG mass", True),
    "IG-tree-softnms": ("IG alone", "8-conn. tree + Soft-NMS", "IG mass", True),
    "IG-tree-wbf": ("IG alone", "8-conn. tree + WBF", "IG mass", True),
    "IG-tree-nonms": ("IG alone", "8-conn. tree, every node", "IG mass", True),
    "IG-tree-largest": ("IG alone", "8-conn. tree, the largest per level", "IG mass", True),
    "IG-q100": ("IG alone", "100 quantiles, 8-conn. + NMS 0.5", "IG mass", True),
    "IG-nested": ("IG alone", "nested thresholds 1–20%, 8-conn. + NMS 0.5", "IG mass", True),
    "IG-top5": ("IG alone", "5% of the tokens, 8-conn. + NMS 0.5", "IG mass", True),
    "IG-tree-min01": ("IG alone", "8-conn. tree ≥ 0.1% of the mass + NMS 0.5", "IG mass", True),
    "IG-tree-black": ("IG alone · black baseline", "8-conn. tree + NMS 0.5", "IG mass", True),
    "IG-tree-noise": ("IG alone · noise baseline", "8-conn. tree + NMS 0.5", "IG mass", True),
    "IG-tree-mean": ("IG alone · mean baseline", "8-conn. tree + NMS 0.5", "IG mass", True),
    "IG-tree-blur": ("IG alone · blurred baseline", "8-conn. tree + NMS 0.5", "IG mass", True),
    "IG-tree-abs": ("IG alone · |IG|", "8-conn. tree + NMS 0.5", "|IG| mass", True),
    "GradCAM": ("Grad-CAM", "70% of the mass, one per 8-conn. component", "Grad-CAM mass", True),
    "GradCAM-tree": ("Grad-CAM", "8-conn. tree + NMS 0.5", "Grad-CAM mass", True),
    "CRIS-top10": ("VTCD · CRIS", "10 concepts, one per component", "CRIS importance", False),
    "CRIS-top1": ("VTCD · CRIS", "1 concept (P01), one per component", "CRIS importance", False),
    "VTCD-global": ("VTCD · Concepts-Global", "one box per concept", "enrichment", False),
    "VTCD-global-components": (
        "VTCD · Concepts-Global",
        "one per 8-conn. component",
        "enrichment",
        False,
    ),
    "VTCD-global-tokens": (
        "VTCD · Concepts-Global",
        "one box per concept",
        "token share",
        False,
    ),
    "VTCD-class": (
        "VTCD · MultiClass-Branch→Concepts-AllChunks",
        "one box per concept",
        "enrichment",
        False,
    ),
    "VTCD-class-components": (
        "VTCD · MultiClass-Branch→Concepts-AllChunks",
        "one per 8-conn. component",
        "enrichment",
        False,
    ),
    "VTCD-class-tokens": (
        "VTCD · MultiClass-Branch→Concepts-AllChunks",
        "one box per concept",
        "token share",
        False,
    ),
    "VTCD-class-A1": (
        "VTCD · MultiClass-Branch→Concepts-References",
        "one box per concept",
        "enrichment",
        False,
    ),
}
MULTICLASS = {
    "B1": ("MultiClass-Branch (B1)", "campaign", "test_heads", "B1"),
    "B2": ("MultiClass-Full (B2)", "campaign", "test_heads", "B2"),
    "B3s": ("MultiClass-Own sum (B3)", "b3_sum", "test", "own"),
    "B3g": ("MultiClass-Own GradNorm (B3)", "b3_gradnorm", "test", "own"),
    "B4L": ("MultiClass-LoRA (B4)", "campaign", "test_classifiers", "B4L"),
    "B4V": ("MultiClass-LoRA→VPT (B4)", "campaign", "test_classifiers", "B4V"),
}

#: Who chooses the class, as the page's head filter lists it: key, label, shown by default (the
#: B4 classifiers are hidden until asked).
SELECTOR_FILTER = (
    ("Global", "No class", True),
    ("A", "NNLS (no head)", True),
    ("B1", "MultiClass-Branch", True),
    ("B2", "MultiClass-Full", True),
    ("B3", "MultiClass-Own", True),
    ("B4L", "MultiClass-LoRA", False),
    ("B4V", "MultiClass-LoRA→VPT", False),
    ("Oracle", "Oracle", True),
    ("Compare", "Comparison methods", True),
)


def selector_of(name: str) -> str:
    """:return: the key in :data:`SELECTOR_FILTER` of a configuration or multiclass head."""
    if name in COMPARE_CONFIGS:
        return "Compare"
    head = name.partition("-")[0]
    if head in ("A1", "A2"):
        return "A"
    return "B3" if head.startswith("B3") else head


def configuration_name(name: str) -> str:
    """:return: the explanatory name of a configuration: who chooses the class → dictionary."""
    if name in COMPARE_CONFIGS:
        return COMPARE_CONFIGS[name][0]
    if name == "Global":
        return "No class→Concepts-Global"
    head, _, rest = name.partition("-")
    if head in ("A1", "A2"):
        return f"NNLS-{RULES[rest]}→{DICTIONARIES[head]}"
    return f"{SELECTORS[head]}→{DICTIONARIES[rest]}"


def range_mean(values: np.ndarray, thresholds, low: float, high: float) -> np.ndarray:
    """:return: the per-event mean over the thresholds in ``[low, high]`` (last axis)."""
    picked = [i for i, t in enumerate(thresholds) if low - 1e-9 <= t <= high + 1e-9]
    return values[..., picked].mean(axis=-1)


def class_means(values: np.ndarray, classes_of: list[str]) -> tuple[dict, float | None, int]:
    """Event → class → thirteen-class mean of per-event values, skipping NaN events.

    :return: ``(per_class, macro, events_without_value)``; a class without events is ``None``
        and left out of the macro.
    """
    per_class = {}
    for name in CLASSES:
        rows = [v for v, c in zip(values, classes_of, strict=True) if c == name and np.isfinite(v)]
        per_class[name] = float(np.mean(rows)) if rows else None
    kept = [v for v in per_class.values() if v is not None]
    missing = int(np.sum(~np.isfinite(values)))
    return per_class, (float(np.mean(kept)) if kept else None), missing


def read_ap(folder: Path, dataset: str) -> tuple | None:
    """:return: the axes and the GT-``union`` per-event AP array (configuration × variant ×
    score × event × threshold) of an evaluation folder, ``None`` if it does not exist."""
    axes_path = folder / f"axes_{dataset}.json"
    if not axes_path.exists():
        return None
    axes = json.loads(axes_path.read_text(encoding="utf-8"))
    arrays = np.load(folder / f"metrics_per_event_{dataset}.npz")
    union = axes["gt"].index("union")
    return axes, arrays["ap"][:, :, :, union], arrays


#: The simplest box rule of the ladder: global dictionary, one box per concept, IG mass.
SIMPLE = ("Global", "concept-box", "mass")
#: Steps of the ladder: key, label, AP stages searched, box rule (``simple`` or ``best``),
#: whether the step belongs to the main sequence (the others are shown next to it).
LADDER = (
    ("base-simple", "Baseline · simple algorithm", ("attentive",), "simple", True),
    ("base-best", "Baseline · best without class", ("attentive",), "best", True),
    ("ours-simple", "Ours · simple algorithm", ("campaign",), "simple", True),
    ("ours-best", "Ours · best", ("campaign",), "best", True),
    ("b3s-simple", "B3 sum · simple algorithm", ("b3s",), "simple", False),
    ("b3s-best", "B3 sum · best", ("b3s",), "best", False),
    ("b3g-simple", "B3 GradNorm · simple algorithm", ("b3g",), "simple", False),
    ("b3g-best", "B3 GradNorm · best", ("b3g",), "best", False),
)


#: Node counts of IG the comparison views choose their rows at; 32 is the page's default
#: (decision of 29 October 2026), 256 the campaign's.
NODE_SETS = ("1fps", "32", "256")


def _offline_mean(row: dict) -> float | None:
    """:return: the mean over the two datasets of the offline AP@[0.3:0.05:0.7], if both exist."""
    cell = row["values"].get("mean", {}).get("offline_main")
    return None if cell is None else cell["macro"]


def offline_means(rows: list[dict], sweep: dict | None) -> dict[str, dict[str, float | None]]:
    """:return: per node set of :data:`NODE_SETS` the :func:`_offline_mean` of every row id at
    that IG node count: the campaign's at 256, the sweep's at the others (a row whose boxes do
    not use IG keeps its value; a row the sweep did not recompute has ``None``)."""
    base = {r["id"]: _offline_mean(r) for r in rows}
    out = {"256": base}
    metric = NODE_METRICS.index("offline_main")
    for key in NODE_SETS:
        if key == "256":
            continue
        values = (sweep or {}).get("values", {}).get(key)
        at = {}
        for index, row in enumerate(rows):
            if not row.get("ig", True) and key not in FRESH_SETS:
                at[row["id"]] = base[row["id"]]
            elif values is None or values[index] is None:
                at[row["id"]] = None
            else:
                at[row["id"]] = values[index][len(DATASETS)][metric]
        out[key] = at
    return out


def _best_id(
    rows: list[dict],
    stages,
    score: str,
    means: dict[str, float | None],
    selectors: tuple[str, ...] | None = None,
) -> str | None:
    """:return: the id of the row with the highest offline mean in ``means`` among the rows of
    ``stages`` (and of ``selectors``, when given), the oracle left out (it reads the annotated
    class); ``score="mass"`` restricts the choice to the IG-mass confidence. ``None`` if no
    such row is evaluated."""
    pool = [
        r
        for r in rows
        if r["stage"] in stages
        and (selectors is None or r["selector"] in selectors)
        and r["selector"] != "Oracle"
        and means.get(r["id"]) is not None
        and (score == "any" or r["score"] == SCORES["mass"])
    ]
    return max(pool, key=lambda r: means[r["id"]])["id"] if pool else None


def ladder(rows: list[dict], means: dict) -> dict:
    """The rows of the baseline-to-ours ladder, by node set and row id.

    A ``best`` step takes :func:`_best_id` among the stages of the step (B4 is not one of
    them), at each node set; ``mass`` keeps the false-alarm metric near the head's (both are
    shown). A step whose stage is not evaluated yet has ``None``.

    :param means: :func:`offline_means`.
    :return: ``{"criterion", "simple", "steps"}``; each step ``{"key", "label", "rule",
        "main", "detector", "ids": {node set: {"any", "mass"}}}``.
    """
    steps = []
    for key, label, stages, rule, main in LADDER:
        detector = STAGES[stages[0]]
        ids = {}
        for nodes in NODE_SETS:
            at = means.get(nodes, {})
            ids[nodes] = {}
            for score in ("any", "mass"):
                if rule == "simple":
                    found = f"{stages[0]}|{'|'.join(SIMPLE)}"
                    ids[nodes][score] = found if at.get(found) is not None else None
                else:
                    ids[nodes][score] = _best_id(rows, stages, score, at)
        steps.append(
            {
                "key": key,
                "label": label,
                "rule": rule,
                "main": main,
                "detector": DETECTORS[detector][0],
                "ids": ids,
            }
        )
    return {
        "criterion": "offline AP@[0.3:0.05:0.7], mean of the two datasets, oracle excluded",
        "simple": dict(zip(("configuration", "variant", "score"), SIMPLE, strict=True)),
        "steps": steps,
    }


#: The comparison methods approved on 29 October 2026, in the page's order. ``ig``/``concepts``
#: place a method in the paradigm grid; ``ours`` names the selectors of our two rows, kept apart
#: as the approval asked (class predicted by a head, or chosen by the dictionaries), and
#: ``config`` the comparison configuration of the others; ``cost`` names the parts of
#: :func:`method_seconds`.
COMPARISONS = (
    {
        "key": "gradcam",
        "paradigm": "Neither IG nor concepts",
        "label": "Grad-CAM on the tokens",
        "uses": "gradient of the anomaly logit on the final features; cut at 70% of the mass, "
        "one box per 8-connected component, confidence its mass",
        "ig": False,
        "concepts": False,
        "config": "GradCAM",
        "cost": ("decode", "forward", "gradcam", "box:GradCAM"),
    },
    {
        "key": "ig-only",
        "paradigm": "IG alone",
        "label": "IG, components at every threshold",
        "uses": "the same IG as ours, no concept: every 8-connected component of {IG ≥ t}, "
        "for every t > 0, is a box; NMS 0.5 per frame, confidence the IG mass",
        "ig": True,
        "concepts": False,
        "config": "IG-tree",
        "cost": ("decode", "forward", "baselines", "integral", "box:IG-tree"),
    },
    {
        "key": "cris",
        "paradigm": "Concepts alone",
        "label": "VTCD with the official CRIS",
        "uses": "the 10 VTCD concepts of highest CRIS importance (as in the paper), on the "
        "keys of their heads; one box per component, confidence the importance",
        "ig": False,
        "concepts": True,
        "config": "CRIS-top10",
        "cost": ("decode", "forward", "cris", "box:CRIS-top10"),
    },
    {
        "key": "dict-global",
        "paradigm": "Concepts alone",
        "label": "VTCD, global dictionary directly",
        "uses": "the concepts of the global dictionary, without IG: one box per concept, "
        "confidence the enrichment of the concept in the positive training chunks",
        "ig": False,
        "concepts": True,
        "config": "VTCD-global",
        "cost": ("decode", "forward", "slic", "box:VTCD-global"),
    },
    {
        "key": "dict-class",
        "paradigm": "Concepts alone",
        "label": "VTCD, classification → class dictionary",
        "uses": "MultiClass-Branch chooses the class, then the concepts of its dictionary "
        "(Concepts-AllChunks), without IG: one box per concept, confidence the enrichment",
        "ig": False,
        "concepts": True,
        "config": "VTCD-class",
        "cost": ("decode", "forward", "slic", "head", "box:VTCD-class"),
    },
    {
        "key": "ours-head",
        "paradigm": "IG + VTCD",
        "label": "Ours, class from a head",
        "uses": "a multiclass head of the detector chooses the class; the concepts of its "
        "dictionary are chosen by IG mass",
        "ig": True,
        "concepts": True,
        "ours": ["B1", "B2"],
        "cost": ("decode", "forward", "baselines", "integral", "slic", "head", "box:ours"),
    },
    {
        "key": "ours-dict",
        "paradigm": "IG + VTCD",
        "label": "Ours, class from the dictionaries",
        "uses": "no head: NNLS on the 13 dictionaries chooses the class; the concepts are "
        "chosen by IG mass",
        "ig": True,
        "concepts": True,
        "ours": ["A"],
        "cost": ("decode", "forward", "baselines", "integral", "slic", "box:ours"),
    },
)
#: Ablations of the comparison methods, one axis at a time from the method's configuration
#: (the first value of each axis): axis label, then ``(label, configuration, score)``.
METHOD_ABLATIONS = {
    "ig-only": (
        (
            "Box confidence",
            (
                ("positive IG mass", "IG-tree", "mass"),
                ("share of the chunk's mass", "IG-tree", "share"),
                ("mean density (mass / tokens)", "IG-tree-density", "mass"),
                ("IG peak in the component", "IG-tree-peak", "mass"),
                ("level t at which the component appears", "IG-tree-level", "mass"),
                ("persistence (appearance minus change level)", "IG-tree-persistence", "mass"),
            ),
        ),
        (
            "Connectivity",
            (("8-connected", "IG-tree", "mass"), ("4-connected", "IG-tree-conn4", "mass")),
        ),
        (
            "Nested boxes",
            (
                ("NMS 0.5", "IG-tree", "mass"),
                ("NMS 0.3", "IG-tree-nms30", "mass"),
                ("NMS 0.7", "IG-tree-nms70", "mass"),
                ("Soft-NMS", "IG-tree-softnms", "mass"),
                ("WBF", "IG-tree-wbf", "mass"),
                ("no NMS", "IG-tree-nonms", "mass"),
                ("only the largest component per level", "IG-tree-largest", "mass"),
            ),
        ),
        (
            "Thresholds",
            (
                ("every positive value", "IG-tree", "mass"),
                ("100 quantile levels", "IG-q100", "mass"),
                ("nested thresholds at 1, 2, 3, 5, 10, 20% of the tokens", "IG-nested", "mass"),
                ("fixed share of 5% of the tokens", "IG-top5", "mass"),
            ),
        ),
        (
            "Minimum components",
            (
                ("1 token", "IG-tree", "mass"),
                ("at least 0.1% of the chunk's mass", "IG-tree-min01", "mass"),
            ),
        ),
        (
            "IG map",
            (
                ("signed mean of the 4 baselines, positive part", "IG-tree", "mass"),
                ("black baseline only", "IG-tree-black", "mass"),
                ("noise baseline only", "IG-tree-noise", "mass"),
                ("mean baseline only", "IG-tree-mean", "mass"),
                ("blurred baseline only", "IG-tree-blur", "mass"),
                ("absolute value instead of the positive part", "IG-tree-abs", "mass"),
            ),
        ),
    ),
    "gradcam": (
        (
            "Box",
            (
                ("cut at 70% of the mass, one per component", "GradCAM", "mass"),
                ("component tree + NMS 0.5, as IG alone", "GradCAM-tree", "mass"),
            ),
        ),
        (
            "Box confidence",
            (
                ("Grad-CAM mass", "GradCAM", "mass"),
                ("share of the chunk's mass", "GradCAM", "share"),
            ),
        ),
    ),
    "cris": (
        (
            "Concepts",
            (
                ("the 10 most important", "CRIS-top10", "mass"),
                ("only the most important (P01 of 16 October)", "CRIS-top1", "mass"),
            ),
        ),
    ),
    "dict-global": (
        (
            "Box",
            (
                ("one box per concept", "VTCD-global", "mass"),
                ("one per 8-connected component", "VTCD-global-components", "mass"),
            ),
        ),
        (
            "Concept order",
            (
                ("enrichment in training", "VTCD-global", "mass"),
                ("token share in the chunk", "VTCD-global-tokens", "mass"),
            ),
        ),
    ),
    "dict-class": (
        (
            "Class dictionary",
            (
                ("Concepts-AllChunks (A2)", "VTCD-class", "mass"),
                ("Concepts-References (A1)", "VTCD-class-A1", "mass"),
            ),
        ),
        (
            "Box",
            (
                ("one box per concept", "VTCD-class", "mass"),
                ("one per 8-connected component", "VTCD-class-components", "mass"),
            ),
        ),
        (
            "Concept order",
            (
                ("enrichment in training", "VTCD-class", "mass"),
                ("token share in the chunk", "VTCD-class-tokens", "mass"),
            ),
        ),
    ),
}
#: The comparisons' detector: every method runs on it, so that a gap comes from the method.
COMPARISON_STAGE = "campaign"
#: Medians of the parts of a method's cost per chunk (``localization_compare.py timing``).
COMPARE_TIMING = "fits/compare/timing.json"


def compare_id(configuration: str, score: str = "mass") -> str:
    """:return: the row id of a comparison configuration (its stage, the ``component`` variant
    that reads the method's final boxes, and the score)."""
    stage = "cmp-ig" if configuration.startswith("IG-") else "cmp-other"
    return f"{stage}|{configuration}|component|{score}"


def method_seconds(cost: tuple[str, ...], timing: dict | None, nodes: str) -> float | None:
    """:return: the median seconds per chunk of a method at an IG node count: the sum of the
    medians of its parts in ``timing`` (``integral`` at ``nodes``), ``None`` when one misses."""
    if not timing:
        return None
    total = 0.0
    for part in cost:
        count = SET_NODES.get(nodes, nodes)
        value = timing.get(f"integral_{count}" if part == "integral" else part)
        if value is None:
            return None
        total += value
    return total


def _parts(configuration: str) -> tuple[str, str]:
    """:return: who chooses the class and the dictionary of a configuration id: ``Global`` is
    ``("", "G")``, ``A2-voto-netto`` is ``("voto-netto", "A2")``, ``B1-A2`` is
    ``("B1", "A2")``."""
    if configuration == "Global":
        return "", "G"
    head, _, rest = configuration.partition("-")
    return (rest, head) if head in DICTIONARIES else (head, rest)


def _configuration(chooser: str, dictionary: str) -> str:
    """:return: the configuration id of a chooser and a dictionary, the inverse of
    :func:`_parts` (the global dictionary has no chooser)."""
    if dictionary == "G":
        return "Global"
    return f"{dictionary}-{chooser}" if chooser in RULES else f"{chooser}-{dictionary}"


def ablation_axes(rows: list[dict], reference: str) -> list[dict]:
    """The ablations of one row of ours, one axis at a time with the others held.

    Axes: the dictionary (global, A1, A2, same chooser), who chooses the class (the four NNLS
    rules, B1, B2, the oracle as the upper bound, and B3 on its own detectors, same
    dictionary), the box variant and the confidence. The reference row is in every axis; a
    combination that was not evaluated is left out, and so is an axis with one row.

    :param reference: a row id ``stage|configuration|variant|score``.
    :return: ``[{"key", "label", "rows": [{"id", "label", "note"}]}]``.
    """
    by_id = {r["id"]: r for r in rows}
    stage, configuration, variant, score = reference.split("|")
    chooser, dictionary = _parts(configuration)

    def entry(row_id: str, label: str, note: str | None = None) -> dict | None:
        row = by_id.get(row_id)
        if row is None or _offline_mean(row) is None:
            return None
        return {"id": row_id, "label": label, "note": note}

    dictionaries = [entry(f"{stage}|Global|{variant}|{score}", "Concepts-Global (no class)")]
    if dictionary != "G":
        dictionaries += [
            entry(f"{stage}|{_configuration(chooser, d)}|{variant}|{score}", f"{name} ({d})")
            for d, name in DICTIONARIES.items()
        ]
    choosers = []
    if dictionary != "G":
        for c in [*RULES, "B1", "B2", "Oracle"]:
            conf = _configuration(c, dictionary)
            note = "upper bound" if c == "Oracle" else None
            choosers.append(
                entry(f"{stage}|{conf}|{variant}|{score}", configuration_name(conf), note)
            )
        for b3 in ("b3s", "b3g"):
            conf = f"B3-{dictionary}"
            choosers.append(
                entry(
                    f"{b3}|{conf}|{variant}|{score}",
                    configuration_name(conf),
                    f"other detector: {DETECTORS[STAGES[b3]][0]}",
                )
            )
    variants = [entry(f"{stage}|{configuration}|{v}|{score}", name) for v, name in VARIANTS.items()]
    scores = [entry(f"{stage}|{configuration}|{variant}|{s}", name) for s, name in SCORES.items()]
    axes = (
        ("dictionary", "Concept dictionary", dictionaries),
        ("chooser", "Who chooses the class", choosers),
        ("variant", "Box shape", variants),
        ("score", "Box confidence", scores),
    )
    out = []
    for key, label, found in axes:
        kept = [e for e in found if e is not None]
        if len(kept) > 1:
            out.append({"key": key, "label": label, "rows": kept})
    return out


def comparisons(rows: list[dict], means: dict, timing: dict | None = None) -> dict:
    """The comparison methods, our two rows and the ablations of each method.

    Each of our rows is :func:`_best_id` among the rows of the comparisons' detector whose
    selector the method names (B4 is another stage, so never one of them), at each node set;
    every other method is its configuration (``None`` until evaluated).

    :param means: :func:`offline_means`.
    :param timing: medians of ``fits/compare/timing.json`` (``localization_compare.py
        timing``), for the seconds per chunk.
    :return: ``{"criterion", "detector", "node_sets", "methods", "ablations",
        "method_ablations"}``; ``ids`` and ``ablations`` are by node set and score choice
        (``any``, ``mass``).
    """
    known = {r["id"] for r in rows}
    methods, ablations = [], {}
    for spec in COMPARISONS:
        method = {k: v for k, v in spec.items() if k not in ("ours", "cost")}
        method["seconds"] = {n: method_seconds(spec["cost"], timing, n) for n in NODE_SETS}
        if "ours" in spec:
            method["ids"] = {
                n: {
                    score: _best_id(
                        rows, (COMPARISON_STAGE,), score, means.get(n, {}), tuple(spec["ours"])
                    )
                    for score in ("any", "mass")
                }
                for n in NODE_SETS
            }
            ablations[spec["key"]] = {
                n: {
                    score: ablation_axes(rows, found) if found else []
                    for score, found in method["ids"][n].items()
                }
                for n in NODE_SETS
            }
        else:
            found = compare_id(spec["config"])
            found = found if found in known else None
            method["ids"] = {n: {"any": found, "mass": found} for n in NODE_SETS}
        methods.append(method)
    method_ablations = {
        key: [
            {
                "label": label,
                "rows": [
                    {"id": compare_id(c, s), "label": text, "default": i == 0}
                    for i, (text, c, s) in enumerate(values)
                    if compare_id(c, s) in known
                ],
            }
            for label, values in axes
        ]
        for key, axes in METHOD_ABLATIONS.items()
    }
    return {
        "criterion": "offline AP@[0.3:0.05:0.7], mean of the two datasets, oracle excluded",
        "detector": DETECTORS[STAGES[COMPARISON_STAGE]][0],
        "node_sets": list(NODE_SETS),
        "methods": methods,
        "ablations": ablations,
        "method_ablations": method_ablations,
    }


def far_auroc(true: np.ndarray, false: np.ndarray) -> dict:
    """FAR@TPR90 and AUROC of a confidence between true and false positive chunks.

    ``-inf`` means no box: such a chunk shows nothing, never counts as visible, and ranks below
    every chunk with a box.

    :return: ``{"far", "tpr", "auroc", "true", "false"}``; ``far`` and ``tpr`` are ``None``
        when fewer than 90% of the true positive chunks show a box.
    """
    true, false = np.asarray(true, dtype=np.float64), np.asarray(false, dtype=np.float64)
    out = {"true": int(len(true)), "false": int(len(false)), "far": None, "tpr": None}
    if len(true) and len(false):
        ordered = np.sort(true)[::-1]
        theta = ordered[math.ceil(TPR * len(true)) - 1]
        if np.isfinite(theta):
            out["tpr"] = float(np.mean(true >= theta))
            out["far"] = float(np.mean(np.isfinite(false) & (false >= theta)))
        greater = (true[:, None] > false[None, :]).mean()
        ties = (true[:, None] == false[None, :]).mean()
        out["auroc"] = float(greater + 0.5 * ties)
    else:
        out["auroc"] = None
    return out


def box_counts(true: np.ndarray, false: np.ndarray, boxes: np.ndarray) -> dict:
    """``B_FP@TPR_C(q)`` on :data:`BFP_GRID` and its discrete mean ``B̄_FP[0.70:0.05:0.95]``.

    ``docs/varie/proposte_metriche_fp_localizzazione_20261029.md`` §4–5: the threshold
    ``eta`` on ``S_C`` is the one of FAR@TPR (the ``ceil(q|T|)``-th highest true positive
    score); a false positive chunk at or above it keeps all its ``n(c)`` boxes, and the sum is
    divided by every false positive chunk, filtered ones included. Lower is better; it can
    exceed 1. A target whose threshold is ``-inf`` (fewer than ``q`` of the true positive
    chunks show a box) is unreachable, and so is the mean over a grid with such a target.

    :param true: ``S_C`` of the true positive chunks.
    :param false: ``S_C`` of the false positive chunks.
    :param boxes: ``n(c)`` of the false positive chunks, aligned with ``false``.
    :return: ``{"bfp90": {"value", "tpr"}, "bfpm": {"value", "curve", "tpr"}}``, values
        ``None`` where unavailable (no chunk of a kind, or an unreachable target).
    """
    true, false = np.asarray(true, dtype=np.float64), np.asarray(false, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64)
    curve, tpr = [], []
    for percent in BFP_GRID:
        value = reached = None
        if len(true) and len(false):
            eta = np.sort(true)[::-1][math.ceil(percent / 100 * len(true)) - 1]
            if np.isfinite(eta):
                kept = np.isfinite(false) & (false >= eta)
                value = float(boxes[kept].sum() / len(false))
                reached = float(np.mean(true >= eta))
        curve.append(value)
        tpr.append(reached)
    at = BFP_GRID.index(round(TPR * 100))
    return {
        "bfp90": {"value": curve[at], "tpr": tpr[at]},
        "bfpm": {
            "value": None if None in curve else float(np.mean(curve)),
            "curve": curve,
            "tpr": tpr,
        },
    }


def mean_kept_share(histogram: np.ndarray, edges: np.ndarray, low: float, high: float) -> dict:
    """``F̄_box^θ[low, high]``: the share of false boxes kept, averaged over the thresholds.

    ``docs/varie/proposte_metriche_fp_localizzazione_20261029.md`` §5, the mean of ``F_box``
    over the confidence thresholds, without a TPR: each false box counts ``clip((σ - low)/(high
    - low), 0, 1)``, the part of the threshold interval over which it stays visible, and the sum is
    divided by every false box produced. Lower is better, between 0 and 1. Exact when ``low``
    and ``high`` are bin edges; a bin holding a limit inside is read at its mean score.

    :param histogram: ``2×B`` count and score sum per bin of ``localization_class_ap.py
        box-scores``: bin 0 below ``edges[0]``, bin ``k`` in ``[edges[k-1], edges[k])``.
    :param edges: the ``B - 1`` bin edges.
    :return: ``{"value", "boxes"}``, ``value`` ``None`` without any false box.
    """
    count, total = np.asarray(histogram, dtype=np.float64)
    boxes = float(count.sum())
    if boxes == 0:
        return {"value": None, "boxes": 0}
    lower = np.concatenate(([-np.inf], edges))
    upper = np.concatenate((edges, [np.inf]))
    width = high - low
    kept = np.where(lower >= high, count, 0.0)
    inside = (lower >= low) & (upper <= high)
    kept += np.where(inside, (total - count * low) / width, 0.0)
    split = ~inside & (lower < high) & (upper > low) & (count > 0)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    kept += np.where(split, count * np.clip((mean - low) / width, 0, 1), 0.0)
    return {"value": float(kept.sum() / boxes), "boxes": int(boxes)}


def retention_cells(stored, c: int, v: int, s: int) -> dict:
    """The row values of the box retention (``localization_class_ap.py box-retention``).

    ``F_box@R⁺_box(0.90)``: the share of false boxes kept when the threshold keeps 90% of the
    boxes emitted on positive planes; ``F̄_box,B[0.70:0.05:0.95]``: its discrete mean over
    :data:`RETENTION_GRID`, with the continuous mean over ``[0.70, 0.95]`` beside it; the AUROC
    of the individual box scores, positive against false (``docs/varie/proposte_metriche_fp_
    localizzazione_20261029.md`` §4–5). Lower is better, except the AUROC.

    :param stored: the loaded ``box_retention_<dataset>.npz``.
    :return: ``{"fb90", "fbm", "auroc_box"}`` cells, values ``None`` without boxes of a kind.
    """
    value = lambda x: None if not np.isfinite(x) else float(x)  # noqa: E731
    at = {int(q): i for i, q in enumerate(stored["q"])}
    f, r = stored["f"][c, v, s], stored["r"][c, v, s]
    curve = [value(f[at[q]]) for q in RETENTION_GRID]
    continuous = {tuple(int(x) for x in k): i for i, k in enumerate(stored["intervals"])}
    return {
        "fb90": {"value": value(f[at[90]]), "tpr": value(r[at[90]])},
        "fbm": {
            "value": None if None in curve else float(np.mean(curve)),
            "curve": curve,
            "continuous": value(stored["mean"][c, v, s][continuous[(70, 95)]]),
        },
        "auroc_box": {"value": value(stored["auroc"][c, v, s])},
    }


def false_box_cells(stored, c: int, v: int, s: int) -> dict:
    """The row values of ``B^σ_FP@TPR_P`` (``localization_class_ap.py false-boxes``).

    ``bsig90``: false boxes still shown per false positive chunk at the threshold that keeps a
    box on 90% of the positive planes, every box filtered by its own score; ``bsigm``: its mean
    over :data:`FALSE_BOX_GRID`; ``bsigm95`` and ``bsigm70``: its exact continuous means over
    ``[0.80, 0.95]`` and ``[0.70, 0.95]``;
    ``bsig_far`` and ``bsig_per``: the two factors of ``bsig90``, the share of false positive
    chunks still showing a box and the boxes each of them shows. Lower is better; ``None``
    where the target is unreachable (fewer positive planes with a box than asked).

    :param stored: the loaded ``false_boxes_<dataset>.npz``.
    """
    value = lambda x: None if not np.isfinite(x) else float(x)  # noqa: E731
    at = {int(q): i for i, q in enumerate(stored["q"])}
    means = {tuple(int(x) for x in k): i for i, k in enumerate(stored["intervals"])}
    b, far = stored["b"][c, v, s][at[90]], stored["far"][c, v, s][at[90]]
    mean = stored["mean"][c, v, s]
    planes = int(stored["planes"][c, v, s])
    grid = [value(stored["b"][c, v, s][at[q]]) for q in FALSE_BOX_GRID]
    return {
        "bsig90": {
            "value": value(b),
            "tpr": value(stored["tpr"][c, v, s][at[90]]),
            "planes": planes,
            "planes_with_boxes": int(stored["planes_with_boxes"][c, v, s]),
        },
        "bsigm": {
            "value": None if None in grid else float(np.mean(grid)),
            "grid": list(FALSE_BOX_GRID),
        },
        "bsigm95": {"value": value(mean[means[(80, 95)]])},
        "bsigm70": {"value": value(mean[means[(70, 95)]])},
        "bsig_far": {"value": value(far)},
        "bsig_per": {"value": value(b / far) if np.isfinite(b) and far > 0 else None},
    }


def frame_fpr_cells(stored, c: int, v: int, s: int) -> dict:
    """The row values of ``FPR_box@TPR(q)`` (``localization_class_ap.py frame-fpr``).

    Shares in ``[0, 1]``, lower is better except the AUROC; ``None`` where the target is
    unreachable (fewer covered annotated frames with a box than asked) or a mean meets it.

    :param stored: the loaded ``frame_fpr_<dataset>.npz``.
    :return: ``fpr70`` through ``fpr95`` at steps of five with ``value``, the reached ``tpr`` and
        ``theta``; ``fprm8090``, ``fprm8095``, ``fprm7095``, ``fpr_auroc``, ``fpr_ceiling``.
    """
    value = lambda x: None if not np.isfinite(x) else float(x)  # noqa: E731
    at = {int(q): i for i, q in enumerate(stored["q"])}
    means = {tuple(int(x) for x in k): i for i, k in enumerate(stored["intervals"])}
    fpr, tpr, theta = (stored[k][c, v, s] for k in ("fpr", "tpr", "theta"))
    # Older sealed outputs lack 70/75; recover them exactly from stored frame scores.
    if not {70, 75} <= set(at) and "positive" in stored and "negative" in stored:
        from localization_class_ap import FPR_POINTS, frame_fpr

        computed = frame_fpr(stored["positive"][c, v, s], stored["negative"][c, v, s])
        at = {q: i for i, q in enumerate(FPR_POINTS)}
        fpr, tpr, theta = (computed[k] for k in ("fpr", "tpr", "theta"))
    out = {
        f"fpr{q}": {
            "value": value(fpr[at[q]]),
            "tpr": value(tpr[at[q]]),
            "theta": value(theta[at[q]]),
        }
        for q in (70, 75, 80, 85, 90, 95)
        if q in at
    }
    for low, high in ((80, 90), (80, 95), (70, 95)):
        out[f"fprm{low}{high}"] = {"value": value(stored["mean"][c, v, s][means[(low, high)]])}
    out["fpr_auroc"] = {"value": value(stored["auroc"][c, v, s])}
    out["fpr_ceiling"] = {"value": value(stored["ceiling"][c, v, s])}
    return out


def gap_split(rows: dict, stage: str, dataset: str, offline: tuple, old: tuple) -> None:
    """Split each row's online − offline AP into the part due to the detector's false negatives.

    The old online rule (:data:`FN_ARCHIVE`) showed the GT frames only on the flagged chunks and
    left out an event with none, so on the events it kept, old online − offline is what the
    other chunks holding the same frames change (several chunks and planes per frame instead of
    the one assigned offline); the rest of the gap, ``fn``, is what the false negative chunks
    cost, their GT boxes now missed and the events never flagged at AP 0. Adds to each row's
    cell ``fn_<range>``: ``{"value": fn, "chunks": the rest}``, as fractions like the AP.

    :param offline: :func:`read_ap` of the row's offline folder.
    :param old: :func:`read_ap` of its online folder under the old rule.
    :raises KeyError: if the two folders do not hold the same events.
    """
    off_axes, off_ap, _ = offline
    old_axes, old_ap, _ = old
    position = {e["event_id"]: i for i, e in enumerate(old_axes["event"])}
    order = [position[e["event_id"]] for e in off_axes["event"]]
    classes_of = [e["class"] for e in off_axes["event"]]
    for c, configuration in enumerate(old_axes["configuration"]):
        for v, variant in enumerate(old_axes["variant"]):
            for s, score in enumerate(old_axes["score"]):
                key = f"{stage}|{configuration}|{variant}|{score}"
                cell = rows.get(key, {}).get("values", {}).get(dataset)
                if cell is None or variant not in off_axes["variant"]:
                    continue
                oc = off_axes["configuration"].index(configuration)
                ov, os_ = off_axes["variant"].index(variant), off_axes["score"].index(score)
                for name, (low, high) in RANGES.items():
                    found = [
                        cell.get(f"{m}_{name}", {}).get("macro") for m in ("online", "offline")
                    ]
                    if None in found:
                        continue
                    off = range_mean(off_ap[oc, ov, os_], off_axes["iou_threshold"], low, high)
                    before = range_mean(old_ap[c, v, s], old_axes["iou_threshold"], low, high)
                    before = before[order]
                    kept = np.where(np.isfinite(before), off, np.nan)
                    _, old_macro, _ = class_means(before, classes_of)
                    _, kept_macro, _ = class_means(kept, classes_of)
                    if old_macro is None or kept_macro is None:
                        continue
                    chunks = old_macro - kept_macro
                    cell[f"fn_{name}"] = {"value": found[0] - found[1] - chunks, "chunks": chunks}


def fn_exclusion_penalty(
    rows: dict, stage: str, dataset: str, online: tuple, excluded: tuple
) -> dict:
    """Observed AP minus conditional AP without FN images (31 October 2026).

    Predictions, confidence and matching stay unchanged. Wholly missed events have no
    conditional AP and are excluded from its class mean, explicitly counted here. This
    measures the effect of including FN cases, not potential performance of unseen boxes.
    A conditional macro lacking any class is unavailable rather than averaged over fewer
    classes. Original online/offline metrics are never modified.
    """
    axes, observed, _ = online
    ref_axes, reference, _ = excluded
    for key in ("configuration", "variant", "score", "iou_threshold", "gt"):
        if axes[key] != ref_axes[key]:
            raise ValueError(f"FN reference axes differ: {stage}/{dataset}/{key}")
    events = axes["event"]
    if [(e["event_id"], e["class"]) for e in events] != [
        (e["event_id"], e["class"]) for e in ref_axes["event"]
    ]:
        raise ValueError("FN reference must retain all event identities")
    if any(b["images"] > a["images"] for a, b in zip(events, ref_axes["event"], strict=True)):
        raise ValueError("FN exclusion cannot add images")
    support = {
        "definition": "online_minus_online_excluding_head_fn",
        "events": len(events),
        "events_entirely_missed": sum(
            a["images"] > 0 and b["images"] == 0
            for a, b in zip(events, ref_axes["event"], strict=True)
        ),
        "images_excluded": sum(e["images"] for e in events)
        - sum(e["images"] for e in ref_axes["event"]),
        "images_total": sum(e["images"] for e in events),
    }
    classes_of = [e["class"] for e in events]
    for c, configuration in enumerate(axes["configuration"]):
        for v, variant in enumerate(axes["variant"]):
            for s, score in enumerate(axes["score"]):
                key = f"{stage}|{configuration}|{variant}|{score}"
                cell = rows.get(key, {}).get("values", {}).get(dataset)
                if cell is None:
                    continue
                for name, (low, high) in RANGES.items():
                    per_class, macro, missing = class_means(
                        range_mean(reference[c, v, s], axes["iou_threshold"], low, high),
                        classes_of,
                    )
                    absent = [cl for cl, val in per_class.items() if val is None]
                    actual = cell[f"online_{name}"]["macro"]
                    cell[f"fn_{name}"] = {
                        **support,
                        "value": actual - macro if not absent and actual is not None else None,
                        "reference": macro if not absent else None,
                        "events_without_value": missing,
                        "classes_without_value": absent,
                    }
    return support


def fbox_interval(score: str, label: str) -> tuple[float, float] | None:
    """:return: the threshold interval of ``F̄_box^θ`` for a row's confidence: ``[0, 1]`` for a
    share, :data:`FBOX_INTERVALS` by label otherwise, ``None`` while undecided."""
    return (0.0, 1.0) if score == "share" else FBOX_INTERVALS.get(label)


def head_scores(run_dir: Path) -> dict[tuple[str, str], float]:
    """:return: the binary head score of every test chunk of a detector, both datasets."""
    out = {}
    for dataset, path in (
        ("ucf_test", run_dir / "scores.jsonl"),
        ("mivia_test", run_dir / f"{run_dir.name}__mivia" / "scores.jsonl"),
    ):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("split", "test") == "test":
                out[dataset, row["window_id"]] = float(row["score"])
    return out


def row_labels(configuration: str, variant: str, score: str) -> tuple[str, str] | None:
    """:return: the variant and score labels of a table row, ``None`` for a row the table
    leaves out: the ``share`` score of a comparison configuration whose confidence is not a
    mass (a score over a total that means nothing)."""
    if configuration not in COMPARE_CONFIGS:
        return VARIANTS.get(variant, variant), SCORES[score]
    _, rule, mass, share = COMPARE_CONFIGS[configuration]
    if score == "share":
        return (rule, mass.replace("mass", "share")) if share else None
    return rule, mass


def localization_rows(
    campaign: Path,
    heads: dict,
    prefix: str = "ap2",
    legacy: bool = True,
    exclude: frozenset[str] = frozenset(),
) -> tuple[list[dict], dict, dict]:
    """The localization rows of one set of evaluation folders.

    :param prefix: the folders read are ``evaluation/<prefix>-{offline,online}-<stage>``.
    :param legacy: fall back on ``evaluation/ap-rbdc-<stage>`` for a stage without the
        offline folder (the stored campaign only).
    :param exclude: stages left out.
    :return: the table rows, the FP reference of each detector's head, the provenance.
    """
    rows: dict[str, dict] = {}
    reference: dict = {}
    provenance: dict = {}
    for stage, detector in STAGES.items():
        if stage in exclude:
            continue
        for dataset, _ in DATASETS:
            offline = read_ap(campaign / f"evaluation/{prefix}-offline-{stage}", dataset)
            source = f"{prefix}-offline-{stage}"
            if offline is None and legacy and stage != "b4":
                offline = read_ap(campaign / f"evaluation/ap-rbdc-{stage}", dataset)
                source = f"ap-rbdc-{stage} (variants of 27 October)"
            online = read_ap(campaign / f"evaluation/{prefix}-online-{stage}", dataset)
            provenance.setdefault(stage, {})[dataset] = {
                "offline": source if offline is not None else None,
                "online": f"{prefix}-online-{stage}" if online is not None else None,
            }
            for mode, found in (("offline", offline), ("online", online)):
                if found is None:
                    continue
                axes, ap, arrays = found
                classes_of = [e["class"] for e in axes["event"]]
                for c, configuration in enumerate(axes["configuration"]):
                    for v, variant in enumerate(axes["variant"]):
                        for s, score in enumerate(axes["score"]):
                            key = f"{stage}|{configuration}|{variant}|{score}"
                            label = row_labels(configuration, variant, score)
                            if label is None:
                                continue
                            row = rows.setdefault(
                                key,
                                {
                                    "id": key,
                                    "stage": stage,
                                    "detector": DETECTORS[detector][0],
                                    "configuration": configuration_name(configuration),
                                    "configuration_id": configuration,
                                    "selector": selector_of(configuration),
                                    "variant": label[0],
                                    "variant_id": variant,
                                    "score": label[1],
                                    "new_variant": variant not in LEGACY,
                                    "ig": stage not in NO_IG_STAGES,
                                    "values": {},
                                },
                            )
                            cell = row["values"].setdefault(dataset, {})
                            for name, (low, high) in RANGES.items():
                                per_event = range_mean(
                                    ap[c, v, s], axes["iou_threshold"], low, high
                                )
                                per_class, macro, missing = class_means(per_event, classes_of)
                                cell[f"{mode}_{name}"] = {
                                    "macro": macro,
                                    "per_class": per_class,
                                    "events_without_value": missing,
                                }
                if mode == "online" and "chunk_confidence" in arrays.files:
                    summary = json.loads(
                        (campaign / f"evaluation/{prefix}-online-{stage}/summary.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    chunks = summary["datasets"][dataset]["flagged_chunks"]
                    category = np.array([ch["category"] for ch in chunks])
                    confidence = arrays["chunk_confidence"]
                    # The box counts exist from 30 October 2026: older folders have no B_FP.
                    counts = arrays["chunk_boxes"] if "chunk_boxes" in arrays.files else None
                    scores_path = campaign / f"evaluation/{prefix}-online-{stage}"
                    scores_path = scores_path / f"box_scores_{dataset}.npz"
                    histograms = np.load(scores_path) if scores_path.exists() else None
                    retention_path = scores_path.with_name(f"box_retention_{dataset}.npz")
                    retention = np.load(retention_path) if retention_path.exists() else None
                    false_path = scores_path.with_name(f"false_boxes_{dataset}.npz")
                    false_boxes = np.load(false_path) if false_path.exists() else None
                    false = BOX_CATEGORIES.index("F")
                    for c, configuration in enumerate(axes["configuration"]):
                        for v, variant in enumerate(axes["variant"]):
                            for s, score in enumerate(axes["score"]):
                                key = f"{stage}|{configuration}|{variant}|{score}"
                                if key not in rows:
                                    continue
                                values = confidence[c, v, s]
                                fp = far_auroc(values[category == "T"], values[category == "F"])
                                oracle = configuration.startswith("Oracle")
                                if oracle:
                                    fp = {**fp, "far": None, "auroc": None, "oracle": True}
                                cell = rows[key]["values"][dataset]
                                cell["fp"] = fp
                                cell["auroc"] = {"value": fp["auroc"]}
                                if counts is None:
                                    continue
                                found = box_counts(
                                    values[category == "T"],
                                    values[category == "F"],
                                    counts[c, v, s][category == "F"],
                                )
                                n_false = int(np.sum(category == "F"))
                                found["nbox"] = {
                                    "value": float(counts[c, v, s][category == "F"].sum() / n_false)
                                    if n_false
                                    else None
                                }
                                if histograms is not None:
                                    interval = fbox_interval(score, rows[key]["score"])
                                    histogram = histograms["histogram"][c, v, s, false]
                                    kept = (
                                        mean_kept_share(histogram, histograms["edges"], *interval)
                                        if interval
                                        else {"value": None, "boxes": int(histogram[0].sum())}
                                    )
                                    found[FBOX] = {**kept, "interval": interval}
                                if retention is not None:
                                    found.update(retention_cells(retention, c, v, s))
                                if false_boxes is not None:
                                    found.update(false_box_cells(false_boxes, c, v, s))
                                for name, metric in found.items():
                                    cell[name] = (
                                        {"value": None, "oracle": True} if oracle else metric
                                    )
                    h = np.array([heads[detector][dataset, ch["window_id"]] for ch in chunks])
                    reference.setdefault(detector, {})[dataset] = far_auroc(
                        h[category == "T"], h[category == "F"]
                    ) | {"zone": int(np.sum(category == "Z"))}
                fpr_path = campaign / f"evaluation/{prefix}-online-{stage}"
                fpr_path = fpr_path / f"frame_fpr_{dataset}.npz"
                if mode == "online" and fpr_path.exists():
                    # Raw frame scores are used for added TPR points; decompress each array
                    # once rather than once per configuration/variant/score cell.
                    with np.load(fpr_path) as archive:
                        stored = {key: archive[key] for key in archive.files}
                    info = json.loads(fpr_path.with_suffix(".json").read_text(encoding="utf-8"))
                    provenance[stage][dataset]["frame_fpr"] = {
                        "theta_ovad": info["theta_ovad"],
                        **info["counts"],
                        "extra_frames": len(info["extra_frames"]),
                    }
                    for c, configuration in enumerate(info["configurations"]):
                        for v, variant in enumerate(info["variants"]):
                            for s, score in enumerate(info["scores"]):
                                key = f"{stage}|{configuration}|{variant}|{score}"
                                if key not in rows:
                                    continue
                                cell = rows[key]["values"][dataset]
                                oracle = configuration.startswith("Oracle")
                                for name, metric in frame_fpr_cells(stored, c, v, s).items():
                                    cell[name] = (
                                        {"value": None, "oracle": True} if oracle else metric
                                    )
            old = read_ap(campaign / f"evaluation/{FN_ARCHIVE}/{prefix}-online-{stage}", dataset)
            if offline is not None and online is not None and old is not None:
                gap_split(rows, stage, dataset, offline, old)
            if prefix == "f1-n32" and online is not None:
                fn_folder = campaign / f"evaluation/{prefix}-fn-excluded-{stage}"
                excluded = read_ap(fn_folder, dataset)
                if excluded is not None:
                    summary = json.loads((fn_folder / "summary.json").read_text(encoding="utf-8"))
                    if (
                        summary.get("head_fn") != "excluded"
                        or summary.get("head_gate") != "observed"
                    ):
                        raise ValueError(
                            "Expected conditional FN-excluded AP with observed predictions"
                        )
                    actual_summary = json.loads(
                        (campaign / f"evaluation/{prefix}-online-{stage}/summary.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    for field in (
                        "stage",
                        "mode",
                        "frames",
                        "variants",
                        "gt_modes",
                        "intervals",
                        "ovad_plan",
                    ):
                        if summary[field] != actual_summary[field]:
                            raise ValueError(f"FN reference provenance differs: {field}")
                    provenance[stage][dataset]["fn_exclusion"] = fn_exclusion_penalty(
                        rows, stage, dataset, online, excluded
                    )
    for row in rows.values():
        both = [row["values"].get(d) for d, _ in DATASETS]
        if all(both):
            mean = {}
            for metric in both[0]:
                if metric == "fp":
                    far = [b["fp"]["far"] for b in both if "fp" in b]
                    mean["fp"] = {
                        "far": float(np.mean(far)) if len(far) == 2 and None not in far else None,
                        "auroc": None,
                    }
                    continue
                if metric in VALUE_METRICS:
                    found = [b[metric]["value"] for b in both if metric in b]
                    mean[metric] = {
                        "value": float(np.mean(found))
                        if len(found) == 2 and None not in found
                        else None
                    }
                    if prefix == "f1-n32" and metric in ("fn_main", "fn_coco"):
                        parts = [b.get(metric, {}) for b in both]
                        for count in (
                            "events",
                            "events_entirely_missed",
                            "images_excluded",
                            "images_total",
                        ):
                            mean[metric][count] = sum(p.get(count, 0) for p in parts)
                        refs = [p.get("reference") for p in parts]
                        mean[metric]["reference"] = (
                            float(np.mean(refs)) if None not in refs else None
                        )
                        mean[metric]["definition"] = "online_minus_online_excluding_head_fn"
                        mean[metric]["classes_without_value"] = sorted(
                            {cl for p in parts for cl in p.get("classes_without_value", [])}
                        )
                    if metric == FBOX:
                        mean[metric]["interval"] = both[0][metric].get("interval")
                    if metric == "fbm":
                        found = [b[metric].get("continuous") for b in both if metric in b]
                        mean[metric]["continuous"] = (
                            float(np.mean(found)) if len(found) == 2 and None not in found else None
                        )
                    continue
                if metric not in both[1]:
                    continue
                macros = [b[metric]["macro"] for b in both]
                mean[metric] = {
                    "macro": None if None in macros else float(np.mean(macros)),
                    "per_class": {
                        k: None
                        if None
                        in (both[0][metric]["per_class"][k], both[1][metric]["per_class"][k])
                        else (both[0][metric]["per_class"][k] + both[1][metric]["per_class"][k]) / 2
                        for k in CLASSES
                    },
                }
            row["values"]["mean"] = mean
    return list(rows.values()), reference, provenance


LEGACY = {
    "union",
    "component",
    "concept-union",
    "concept-box",
    "component-concept-score",
    "largest-component",
    "nms30-component",
    "nms50-component",
    "nms70-component",
    "nms30-concept-box",
    "nms50-concept-box",
    "nms70-concept-box",
    "softnms-component",
    "softnms-concept-box",
    "wbf-component",
    "wbf-concept-box",
}


#: The IG node sweep (28 October 2026): key, Gauss-Legendre nodes, folder prefix of
#: ``scripts/localization_class_ig_nodes.py``. The campaign's 256 is the reference; the rerun
#: at 256 measures the GPU nondeterminism alone, on the base detector and B4.
IG_NODES = (
    ("1fps", 32, "f1-n32"),
    ("16", 16, "ign-n16"),
    ("32", 32, "ign-n32"),
    ("64", 64, "ign-n64"),
    ("128", 128, "ign-n128"),
    ("256r", 256, "ign-n256"),
)
#: Node sets that re-evaluate every stage, the ones without IG too, under other definitions:
#: their values never fall back on the campaign's, and they are not an IG node count to compare
#: with 256 (``1fps``: one frame per second and the boxes on ``h(c) >= θ_O``, 30 October 2026).
FRESH_SETS = {"1fps"}
#: The IG node count of a node set (``1fps`` is at 32 nodes).
SET_NODES = {"1fps": "32"}
#: The four AP metrics and the FP metrics, in the order of a node set's compact values.
NODE_METRICS = ("offline_main", "offline_coco", "online_main", "online_coco", "fp", *VALUE_METRICS)


#: The node counts whose per-class AP the page keeps: the default count of results and times
#: (29 October 2026), where the online AP is evaluated (30 October 2026).
PER_CLASS_SETS = ("1fps", "32")
#: The AP metrics with per-class values, in the order of a row's per-class lists.
PER_CLASS_METRICS = ("offline_main", "offline_coco", "online_main", "online_coco")


def metric_value(cell: dict | None, metric: str):
    """:return: the number a table cell shows for ``metric`` (FAR, an FP value or class macro)."""
    field = "far" if metric == "fp" else "value" if metric in VALUE_METRICS else "macro"
    return (cell or {}).get(field)


#: The integral benchmark of 28 October 2026 (H100, job 807665): seconds per chunk at each node
#: count and agreement of the maps with 256 nodes.
IG_BENCH = "timing/ig-bench-20261028/ig_bench.json"


def ig_node_sweep(
    campaign: Path, heads: dict, rows: list[dict], exclude: frozenset[str] = frozenset()
) -> dict | None:
    """The table values at every IG node count of the sweep, aligned with ``rows``.

    The class macros are kept at every count, the per-class values only at the counts of
    :data:`PER_CLASS_SETS` (all of them would make the page several times larger); the oracle
    has no FP value, as in the table.

    :return: ``{"sets", "metrics", "values", "per_class", "bench", "fp_reference"}`` (the last,
        the head's FP reference of each detector from the first set that has it, is for the
        page's reference rows): ``values[key][i]`` is
        ``None`` when row ``i`` has no value at that node count, else one list of
        :data:`NODE_METRICS` per dataset of :data:`DATASETS` and the mean; ``per_class[key][i]``
        likewise, one list per dataset and metric of :data:`PER_CLASS_METRICS` of the 13 class
        values; ``None`` when no set is evaluated.
    """
    keys = [d for d, _ in DATASETS] + ["mean"]

    def compact(row: dict | None) -> list | None:
        if row is None:
            return None
        cells = [row["values"].get(d, {}) for d in keys]
        return [[metric_value(c.get(m), m) for m in NODE_METRICS] for c in cells]

    def classes(row: dict | None) -> list | None:
        if row is None:
            return None
        cells = [row["values"].get(d, {}) for d in keys]
        return [
            [
                [((c.get(m) or {}).get("per_class") or {}).get(k) for k in CLASSES]
                for m in PER_CLASS_METRICS
            ]
            for c in cells
        ]

    sets, values, per_class, references = [], {}, {}, {}
    for key, nodes, prefix in IG_NODES:
        found, reference, provenance = localization_rows(
            campaign, heads, prefix=prefix, legacy=False, exclude=exclude
        )
        if not found:
            continue
        # The head's reference does not depend on IG: any set with an online folder gives it.
        for detector, cells in reference.items():
            references.setdefault(detector, cells)
        by_id = {r["id"]: r for r in found}
        values[key] = [compact(by_id.get(r["id"])) for r in rows]
        if key in PER_CLASS_SETS:
            per_class[key] = [classes(by_id.get(r["id"])) for r in rows]
        sets.append(
            {
                "key": key,
                "nodes": nodes,
                "control": key == "256r",
                "fresh": key in FRESH_SETS,
                "stages": sorted({r["stage"] for r in found}),
                "rows": len(found),
                **({"provenance": provenance} if key in FRESH_SETS else {}),
                **(
                    {
                        "details": {
                            r["id"]: {
                                d: {
                                    m: cell[m]
                                    for m in (*FPR_METRICS, "fn_main", "fn_coco")
                                    if m in cell
                                }
                                for d, cell in r["values"].items()
                            }
                            for r in found
                        }
                    }
                    if key in FRESH_SETS
                    else {}
                ),
            }
        )
    if not sets:
        return None
    bench = None
    path = campaign / IG_BENCH
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        measured = {256: raw["summary"]["campaign"]["median_seconds"]}
        agreement = {}
        for name, cell in raw["summary"].items():
            if name.startswith("nodes") and "median_seconds" in cell:
                nodes = int(name[len("nodes") :])
                measured[nodes] = cell["median_seconds"]
                ensemble = cell["agreement_median"]["ig_ensemble"]
                agreement[nodes] = {
                    "pearson": ensemble["pearson"],
                    "top10_jaccard": ensemble["top10_jaccard"],
                }
        # The integral is linear in the nodes: a least-squares line through the measured
        # points estimates the counts the benchmark did not run (declared as estimates).
        xs, ys = np.array(sorted(measured)), np.array([measured[n] for n in sorted(measured)])
        slope, intercept = np.polyfit(xs, ys, 1)
        seconds = {
            n: {"value": measured.get(n, slope * n + intercept), "estimated": n not in measured}
            for _, n, _ in IG_NODES
        }
        bench = {
            "gpu": raw["gpu"],
            "slurm_job": raw.get("slurm_job"),
            "chunks": raw["summary"]["campaign"]["chunks"],
            "seconds": seconds,
            "agreement": agreement,
        }
    return {
        "sets": sets,
        "metrics": list(NODE_METRICS),
        "values": values,
        "per_class_metrics": list(PER_CLASS_METRICS),
        "per_class": per_class,
        "bench": bench,
        "fp_reference": references,
    }


def intervals_of(data_root: Path, mivia: Path, frames: dict) -> dict:
    """:return: per dataset and video the temporal events ``(start, end, class index)``."""
    from evd.annotations.ucf_lab import MIVIA_CRIME, UcfLabAnnotations

    out = {}
    for dataset, folder, delivery in (
        ("ucf_test", data_root / "annotations/fs_ucf_crime", None),
        ("mivia_test", mivia, MIVIA_CRIME),
    ):
        kwargs = {"num_frames": frames[dataset]}
        if delivery is not None:
            kwargs["delivery"] = delivery
        loaded = UcfLabAnnotations.from_dir(folder, **kwargs).load()
        out[dataset] = {
            video: [tuple(int(x) for x in i[:2]) + (int(i[2]),) for i in loaded[video].intervals]
            for video in frames[dataset]
        }
    return out


def _prf(tp: int, fp: int, fn: int) -> tuple[float | None, float | None]:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    return p, r


def _f1(p, r):
    return None if p is None or r is None or p + r == 0 else 2 * p * r / (p + r)


def _auc(labels, scores):
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels)
    return float(roc_auc_score(labels, scores)) if 0 < labels.sum() < len(labels) else None


def _mean(values):
    kept = [v for v in values if v is not None]
    return float(np.mean(kept)) if kept else None


#: OVAD metrics of the binary heads (``docs/19_protocollo_ovad.md``): the Protocol Mean and its
#: six terms, as the laboratory's evaluator writes them (guard 30 s).
OVAD_METRICS = ("mean", "auc", "auc_a_macro", "row_f1", "row_fpr", "video_f1", "event_recall")


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _float(text: str) -> float | None:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def ovad_per_class(cell: Path) -> dict:
    """Detection of each class's events by a binary head, from one OVAD evaluation cell.

    The head does not choose a class: the values break its detection down by the class of the
    annotated event, on the anomalous videos of that class only (no normal video, so no false
    alarm of a normal video enters). Recall of the events (an OVAD event takes the class of the
    annotated interval it overlaps most), mean AUC_a of the class's videos that have both
    labels, F1 of the judgeable rows of those videos, share of those videos detected.

    :return: per class ``{"event_recall", "auc_a", "row_f1", "video_recall", "events",
        "videos"}``.
    """
    labels = json.loads((cell / "input/labels.json").read_text(encoding="utf-8"))
    events = _rows(cell / "eventi.csv")
    per_video = {r["video_id"]: r for r in _rows(cell / "per_video.csv")}
    rows = _rows(cell / "righe.csv")
    out = {}
    for name in CLASSES:
        videos = {v for v, lab in labels.items() if any(a[2] == name for a in lab["anomalies"])}
        recalled = []
        for e in events:
            if e["video_id"] not in videos or e["judgeable"] != "True":
                continue
            a, b = float(e["start"]), float(e["end"])
            best = max(
                labels[e["video_id"]]["anomalies"],
                key=lambda x, a=a, b=b: min(b, x[1]) - max(a, x[0]),
            )
            if best[2] == name:
                recalled.append(e["detected"] == "True")
        auc = [
            _float(per_video[v]["auc_a_macro"])
            for v in videos
            if v in per_video and int(per_video[v]["n_entries_in_macro"] or 0) > 0
        ]
        judged = [r for r in rows if r["video_id"] in videos and r["judgeable"] == "True"]
        tp = sum(r["label"] == "1" and r["prediction"] == "1" for r in judged)
        fp = sum(r["label"] == "0" and r["prediction"] == "1" for r in judged)
        fn = sum(r["label"] == "1" and r["prediction"] == "0" for r in judged)
        p, r_ = _prf(tp, fp, fn)
        out[name] = {
            "event_recall": _mean([float(x) for x in recalled]),
            "auc_a": _mean(auc),
            "row_f1": _f1(p, r_),
            "video_recall": _mean(
                [float(per_video[v]["video_tp"]) for v in videos if v in per_video]
            ),
            "events": len(recalled),
            "videos": len(videos),
        }
    return out


def ovad_class_means(cell: Path) -> dict[str, float | None]:
    """Protocol Mean of a binary head per event class (30 October 2026).

    The laboratory's evaluator (``ovad_protocol/evaluate_ovad.py``, untouched) on the anomalous
    videos of the class and every normal video of the test set, at the cell's threshold and
    guard: the six terms keep their meaning, false alarms of the normal videos included, which
    the class's videos alone could not give (video precision would be 1 by construction). On
    every video it gives back the cell's Protocol Mean.

    :return: per class the Protocol Mean, ``None`` for a class with no video or no judgeable
        row.
    """
    sys.path.insert(0, str(ROOT / "ovad_protocol"))
    from evaluate_ovad import evaluate_ovad
    from protocol_core.inputs import InputError

    labels = json.loads((cell / "input/labels.json").read_text(encoding="utf-8"))
    scores = json.loads((cell / "input/scores.json").read_text(encoding="utf-8"))
    result = json.loads((cell / "results.json").read_text(encoding="utf-8"))
    theta, guard = result["threshold"]["value"], result["parameters"]["guard"]
    normal = [v for v, lab in labels.items() if not lab["anomalies"]]
    out: dict[str, float | None] = {}
    for name in CLASSES:
        videos = [v for v, lab in labels.items() if any(a[2] == name for a in lab["anomalies"])]
        if not videos:
            out[name] = None
            continue
        keep = videos + normal
        try:
            found = evaluate_ovad(
                {v: labels[v] for v in keep},
                {**scores, "scores": {v: scores["scores"][v] for v in keep}},
                threshold=theta,
                guard=guard,
            )
        except InputError:
            out[name] = None
            continue
        out[name] = float(found["levels"]["anomalies"]["mean"])
    return out


def ovad_binary(campaign: Path) -> dict:
    """:return: per detector the OVAD metrics (guard 30 s) of its binary head on both test
    sets, with the threshold and the per-class detection of :func:`ovad_per_class`."""
    root = campaign / "detection/ovad_guard30/celle"
    out = {}
    for key, (label, run_id) in DETECTORS.items():
        entry = {"label": label}
        for dataset, folder in (("ucf_test", "ucf"), ("mivia_test", "mivia")):
            cell = root / run_id / folder
            result = json.loads((cell / "results.json").read_text(encoding="utf-8"))
            levels = result["levels"]["anomalies"]
            per_class = ovad_per_class(cell)
            for name, value in ovad_class_means(cell).items():
                per_class[name]["protocol_mean"] = value
            entry[dataset] = {
                "metrics": {m: levels[m] for m in OVAD_METRICS},
                "threshold": result["threshold"],
                "rows": levels["n_rows"],
                "events": levels["n_events"],
                "videos_with_event": levels["n_entries_with_event"],
                "per_class": per_class,
            }
        out[key] = entry
    return out


def multiclass_metrics(records: list[dict], intervals: dict) -> dict:
    """Multiclass head metrics on the annotated test chunks of one dataset (13 classes).

    Each chunk goes to the event of its class it overlaps most. ``events``: accuracy and recall
    are the share of an event's chunks given its class, averaged per class and over the
    classes; precision and AUC (one-vs-rest on the softmax of the 13 logits) are per class and
    the F1 joins the displayed precision and recall. ``chunks``: the standard per-class values
    (one-vs-rest accuracy) and the overall accuracy.
    """
    true = np.array([r["true"] for r in records])
    logits = np.array([r["logits"] for r in records], dtype=np.float64)
    predicted = logits.argmax(1)
    shifted = logits - logits.max(1, keepdims=True)
    prob = np.exp(shifted) / np.exp(shifted).sum(1, keepdims=True)
    events = defaultdict(list)
    for i, r in enumerate(records):
        best, key = -1, None
        for e, (a, b, k) in enumerate(intervals[r["video_id"]]):
            overlap = min(b, r["end"]) - max(a, r["start"]) + 1
            if k == r["true"] and overlap > best:
                best, key = overlap, (r["video_id"], e)
        events[key or (r["video_id"], f"chunk{i}")].append(i)
    out = {"events": {"per_class": {}}, "chunks": {"per_class": {}}}
    for c, name in enumerate(CLASSES):
        tp = int(np.sum((predicted == c) & (true == c)))
        fp = int(np.sum((predicted == c) & (true != c)))
        fn = int(np.sum((predicted != c) & (true == c)))
        p, r = _prf(tp, fp, fn)
        auc = _auc(true == c, prob[:, c])
        acc = float(np.mean((predicted == c) == (true == c)))
        per_event = [
            float(np.mean(predicted[idx] == c)) for idx in events.values() if true[idx[0]] == c
        ]
        r_events = _mean(per_event)
        out["chunks"]["per_class"][name] = {
            "accuracy": acc,
            "precision": p,
            "recall": r,
            "f1": _f1(p, r),
            "auc": auc,
        }
        out["events"]["per_class"][name] = {
            "accuracy": r_events,
            "precision": p,
            "recall": r_events,
            "f1": _f1(p, r_events),
            "auc": auc,
            "events": len(per_event),
        }
    for level in ("events", "chunks"):
        out[level]["macro"] = {
            m: _mean([v[m] for v in out[level]["per_class"].values()])
            for m in ("accuracy", "precision", "recall", "f1", "auc")
        }
    out["chunks"]["macro"]["accuracy_overall"] = float(np.mean(predicted == true))
    out["chunks"]["chunks"] = int(len(records))
    return out


def classification(
    campaign: Path,
    runs: Path,
    data_root: Path,
    mivia: Path,
    exclude_selectors: frozenset[str] = frozenset(),
) -> dict:
    """:return: the binary and multiclass head metrics, per dataset and their mean, without the
    multiclass heads of ``exclude_selectors``."""
    scores = {d: {} for d, _ in DATASETS}
    frames = {d: {} for d, _ in DATASETS}
    for key, (_, run_id) in DETECTORS.items():
        run_dir = runs / run_id
        for dataset, path in (
            ("ucf_test", run_dir / "scores.jsonl"),
            ("mivia_test", run_dir / f"{run_id}__mivia" / "scores.jsonl"),
        ):
            rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
            rows = [r for r in rows if r.get("split", "test") == "test"]
            scores[dataset][key] = rows
            for r in rows:
                frames[dataset][r["video_id"]] = max(
                    frames[dataset].get(r["video_id"], 0), int(r["end_frame"]) + 1
                )
    intervals = intervals_of(data_root, mivia, frames)
    binary = ovad_binary(campaign)
    test = {}
    with tarfile.open(campaign / "plan.tar.gz") as plan:
        for row in json.load(plan.extractfile("plan/test.json"))["rows"]:
            test[row["dataset"], row["window_id"]] = row
    multiclass = {}
    with tarfile.open(campaign / "passes.tar.gz") as passes:
        members = [m for m in passes.getmembers() if m.isfile() and m.name.endswith(".json")]
        for key, (label, detector, split, logit) in MULTICLASS.items():
            if selector_of(key) in exclude_selectors:
                continue
            prefix = f"passes/{detector}/{split}/"
            by_dataset = defaultdict(list)
            for member in members:
                if not member.name.startswith(prefix) or member.name.count("/") != 3:
                    continue
                record = json.load(passes.extractfile(member))
                row = test[record["dataset"], record["window_id"]]
                by_dataset[record["dataset"]].append(
                    {
                        "video_id": record["video_id"],
                        "start": row["start"],
                        "end": row["end"],
                        "true": CLASSES.index(record["class_name"]),
                        "logits": record["class_logits"][logit][: len(CLASSES)],
                    }
                )
            if not by_dataset:
                continue
            multiclass[key] = {
                "label": label,
                "selector": selector_of(key),
                **{d: multiclass_metrics(by_dataset[d], intervals[d]) for d, _ in DATASETS},
            }
    return {"binary": binary, "multiclass": multiclass}


def rounded(value, digits: int = 7):
    """:return: ``value`` with every float rounded (a percentage with three decimals needs five
    decimals of the fraction; seven keep the page small and every displayed digit exact)."""
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [rounded(v, digits) for v in value]
    return value


#: The OVAD tables of the state-of-the-art detectors (3 November 2026): seven methods with their
#: official weights, evaluated with the laboratory's protocol on both test sets.
SOTA_TABLES = ROOT / "docs/TABELLE_OVAD.md"
#: Row label of ``docs/TABELLE_OVAD.md`` (an Italian document, read as data) → key of the
#: OVAD results (``results.json``).
SOTA_LABELS = {
    "Media del protocollo": "mean",
    "Righe · AUC": "auc",
    "Righe · AP": "ap",
    "Righe · AUC_a macro": "auc_a_macro",
    "Righe · AP_a macro": "ap_a_macro",
    "Righe · AUC_a micro": "auc_a_micro",
    "Righe · AP_a micro": "ap_a_micro",
    "Righe · Precision": "row_precision",
    "Righe · Recall": "row_recall",
    "Righe · F1": "row_f1",
    "Righe · FPR": "row_fpr",
    "Video · Precision": "video_precision",
    "Video · Recall": "video_recall",
    "Video · F1": "video_f1",
    "Video · FPR": "video_fpr",
    "Eventi · Recall": "event_recall",
    "Soglia θ": "threshold",
    "Righe giudicabili": "n_rows",
    "Righe non giudicabili": "n_unjudgeable_rows",
    "Eventi": "n_events",
    "Eventi rilevati": "n_events_detected",
    "Eventi mai visti": "n_events_unseen",
}
#: Values of those tables that are counts or a threshold, not percentages.
SOTA_RAW = {
    "threshold",
    "n_rows",
    "n_unjudgeable_rows",
    "n_events",
    "n_events_detected",
    "n_events_unseen",
}
#: Section title of each test set in ``docs/TABELLE_OVAD.md``.
SOTA_SECTIONS = {"UCF-Crime test": "ucf_test", "MIVIA-Crime test": "mivia_test"}
#: What each state-of-the-art method reads (``docs/TABELLE_OVAD.md``, «Modelli»).
SOTA_NOTES = {
    "UR-DMU": "I3D features, max over the snippets",
    "PEL4VAD": "I3D features, mean of the top-k snippets",
    "GlanceVAD": "I3D features, max over the snippets",
    "Holmes-VAU": "I3D features, max over the snippets",
    "AnomalyCLIP": "CLIP ViT-B/16 on 512 frames per window, max",
    "VadCLIP": "official CLIP features, Top-K (K = 16)",
    "OVadCLIP": "CALM + Top-K (λtv 0.2, λ1 0.001, K 16)",
}


def sota_tables(path: Path) -> dict:
    """The OVAD metrics of the state-of-the-art detectors, read from ``docs/TABELLE_OVAD.md``.

    Percentages become shares, as the page stores every metric; counts and thresholds stay as
    they are. Every method must have every row on both test sets, and its protocol mean must be
    the mean of its six components within the rounding of the table (two decimals).

    :return: per method and dataset key ``{metric: value}``, in the table's column order.
    :raises ValueError: on a missing value or a protocol mean that does not add up.
    """
    out: dict[str, dict[str, dict]] = {}
    dataset = methods = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            dataset = SOTA_SECTIONS.get(line[3:].strip())
            methods = None
            continue
        if dataset is None or not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [c.strip().strip("*").strip() for c in line.strip().strip("|").split("|")]
        if cells[0] == "Metrica":
            methods = cells[1:]
            continue
        key = SOTA_LABELS.get(cells[0].replace("\\", "").rstrip("* ").strip())
        if methods is None or key is None:
            continue
        for name, text in zip(methods, cells[1:], strict=True):
            number = float(text.strip("*"))
            value = int(number) if key in SOTA_RAW and key != "threshold" else number
            value = value if key in SOTA_RAW else number / 100
            out.setdefault(name, {}).setdefault(dataset, {})[key] = value
    for name, per_dataset in out.items():
        for dataset in SOTA_SECTIONS.values():
            found = per_dataset.get(dataset, {})
            missing = sorted(set(SOTA_LABELS.values()) - set(found))
            if missing:
                raise ValueError(f"{name} {dataset}: no value for {missing}")
            parts = ("auc", "auc_a_macro", "row_f1", "video_f1", "event_recall")
            mean = (sum(found[k] for k in parts) + 1 - found["row_fpr"]) / 6
            if abs(mean - found["mean"]) > 1.1e-4:
                raise ValueError(f"{name} {dataset}: the protocol mean does not add up")
    return out


def sota_comparison(campaign: Path, path: Path = SOTA_TABLES) -> dict | None:
    """The binary heads of our detectors next to the state-of-the-art detectors, OVAD metrics.

    Both sides use the laboratory's protocol (rows of at most 32 s, asking rate 4 s, guard 30 s,
    Youden on the judgeable validation rows of FS-UCF-Crime, transferred to MIVIA-Crime); the
    state-of-the-art methods read 30 s chunks and ours 8 s, so the rows are the same and the
    judgeable ones are not.

    :return: ``{"source", "sha256", "metrics", "methods"}``, each method ``{"key", "label",
        "group", "chunk_s", "note", "values"}``; ``None`` when the tables are not there.
    """
    from evd.utils.hashing import file_sha256

    if not path.exists():
        return None
    tables = sota_tables(path)
    keys = list(SOTA_LABELS.values())
    methods = [
        {
            "key": f"sota:{name}",
            "label": name,
            "group": "sota",
            "chunk_s": 30,
            "note": SOTA_NOTES.get(name, ""),
            "values": tables[name],
        }
        for name in tables
    ]
    root = campaign / "detection/ovad_guard30/celle"
    for key, (label, run_id) in DETECTORS.items():
        values = {}
        for dataset, folder in (("ucf_test", "ucf"), ("mivia_test", "mivia")):
            result = json.loads((root / run_id / folder / "results.json").read_text("utf-8"))
            levels = result["levels"]["anomalies"]
            values[dataset] = {k: levels[k] for k in keys}
        methods.append(
            {
                "key": key,
                "label": label,
                "group": "baseline" if key == "attentive" else "ours",
                "chunk_s": 8,
                "note": run_id,
                "values": values,
            }
        )
    return {
        "source": path.relative_to(ROOT).as_posix(),
        "sha256": file_sha256(path),
        "metrics": keys,
        "methods": methods,
    }


def build(
    campaign: Path,
    runs: Path,
    data_root: Path,
    mivia: Path,
    template: Path,
    output: Path | None = None,
    exclude: frozenset[str] = frozenset(),
) -> Path:
    """Write ``tabella_localizzazione.{json,html}``.

    :param output: destination folder; ``None`` for the campaign's ``evaluation/``.
    :param exclude: stages left out of the page, with the class heads of
        :data:`STAGE_SELECTORS` that only they use.
    :return: the HTML path.
    """
    lost = frozenset(s for stage in exclude for s in STAGE_SELECTORS.get(stage, ()))
    detector_heads = {k: head_scores(runs / run_id) for k, (_, run_id) in DETECTORS.items()}
    rows, reference, provenance = localization_rows(campaign, detector_heads, exclude=exclude)
    sweep = ig_node_sweep(campaign, detector_heads, rows, exclude)
    means = offline_means(rows, sweep)
    timing_path = campaign / COMPARE_TIMING
    timing = (
        json.loads(timing_path.read_text(encoding="utf-8"))["medians"]
        if timing_path.exists()
        else None
    )
    data = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "campaign": campaign.name,
        "datasets": [{"key": k, "label": v} for k, v in DATASETS]
        + [{"key": "mean", "label": "Mean of the two"}],
        "classes": CLASSES,
        "ranges": {k: list(v) for k, v in RANGES.items()},
        "tpr": TPR,
        "fbox_intervals": {k: list(v) for k, v in FBOX_INTERVALS.items()},
        "detectors": {k: v[0] for k, v in DETECTORS.items()},
        "selectors": [
            {"key": k, "label": v, "default": d} for k, v, d in SELECTOR_FILTER if k not in lost
        ],
        "rows": rows,
        "ladder": ladder(rows, means),
        "comparisons": comparisons(rows, means, timing),
        "ig_nodes": sweep,
        "fp_reference": {**(sweep or {}).pop("fp_reference", {}), **reference},
        "provenance": provenance,
        "classification": classification(campaign, runs, data_root, mivia, lost),
        "sota": sota_comparison(campaign),
    }
    target = (output or campaign / "evaluation") / "tabella_localizzazione.json"
    text = json.dumps(rounded(data), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    target.write_text(text, encoding="utf-8")
    page = template.read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", text.replace("</", "<\\/")
    )
    html = target.with_suffix(".html")
    html.write_text(page, encoding="utf-8")
    return html


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "data")
    parser.add_argument("--mivia", type=Path, default=ROOT.parent / "data/mivia_crime")
    parser.add_argument(
        "--template", type=Path, default=ROOT / "scripts/localization_class_table.html"
    )
    parser.add_argument("--output", type=Path, help="folder of the page (default: evaluation/)")
    parser.add_argument(
        "--exclude-stage",
        action="append",
        default=[],
        choices=sorted(STAGES),
        help="leave a stage out of the page (repeatable)",
    )
    args = parser.parse_args()
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
    page = build(
        args.campaign,
        args.runs,
        args.data_root,
        args.mivia,
        args.template,
        args.output,
        frozenset(args.exclude_stage),
    )
    print(page)


if __name__ == "__main__":
    main()
