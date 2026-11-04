"""Offline rendering of saved maps; no model and no torch."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import unicodedata
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import av
import cv2
import numpy as np

from apps.demo.overlay import COLORS
from evd.annotations.ucf_classes import UCF_ANOMALY_CLASSES
from evd.explain.boxes import in_pixel
from evd.explain.project import frame_annotati, offline_assignment, offline_plane
from evd.types import WindowSpec

COLORS = {**COLORS, "mancato": (205, 105, 195)}

#: Concept name in per-class composed movies: a concept map (``vtcd_concept_``) that on
#: every frame comes from the dictionary of its event class (:class:`CompositeCampaign`).
CONCEPT_ROLE = "vtcd_concept_evento"


def map_kind(method: str) -> str:
    """:return: the map quantity, without the selected class."""
    return base_method(method).split("__class_", 1)[0]


def transparent_center(method: str) -> bool:
    """:return: whether the map zero must leave the video visible."""
    return map_kind(method).startswith(("ig_", "anomaly_", "class_"))


def color_map(method: str, signed: bool):
    """:return: RGBA table; coolwarm with zero alpha at the centre for IG and directions."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = plt.get_cmap(_cmap_name(method, signed))
    if not transparent_center(method):
        return cmap
    x = np.linspace(0, 1, 257)
    rgba = cmap(x)
    rgba[:, 3] = abs(2 * x - 1)
    return ListedColormap(rgba, name="coolwarm_transparent")


def blend_map(frame: np.ndarray, heat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """:return: overlay on the video and map on a neutral background, both BGR."""
    if heat.shape[-1] == 4:
        alpha = heat[..., 3:4].astype(np.float32) / 255
        rgb = heat[..., :3]
        overlay_alpha = alpha * OVERLAY_ALPHA
        overlay = frame * (1 - overlay_alpha) + rgb * overlay_alpha
        standalone = np.asarray(COLORS["fondo"]) * (1 - alpha) + rgb * alpha
        return overlay.astype(np.uint8), standalone.astype(np.uint8)
    return cv2.addWeighted(frame, 1 - OVERLAY_ALPHA, heat, OVERLAY_ALPHA, 0), heat


#: Readable title of every map, shared by the movie header and the page summary.
LABELS = {
    "raw_attention": "Raw attention",
    "gradient_activation": "Gradient × Activation",
    "gradient_activation_prenorm": "Gradient × Activation before LayerNorm (control)",
    "attention_gradient": "Attention × Gradient",
    "pca_rgb": "PCA-RGB (representation)",
    "eigencam": "EigenCAM (representation)",
    "cosine": "Cosine similarity (fixed query)",
    "legrad_last": "LeGrad — last layer",
    "legrad_four": "LeGrad — four layers",
    "anomaly_projection": "Anomaly Direction: TCAV-style projection",
    "anomaly_sensitivity": "Anomaly Direction: TCAV-style logit sensitivity",
    "class_projection": "Class direction: TCAV-style projection",
    "class_sensitivity": "Class direction: TCAV-style logit sensitivity",
    "prototype_normal": "Normal prototype: cosine similarity",
    "prototype_anomaly": "Anomalous prototype: cosine similarity",
    "prototype_difference": "Anomaly Prototype Maps: anomaly minus normality",
    # Short titles: level and dictionary configuration are on the row below (`vtcd_family`).
    "vtcd_ids": "VTCD",
    "vtcd_importance": "VTCD: CRIS importance (BCE degradation)",
    **{f"vtcd_concept_{k:02d}": f"VTCD: concept {k}" for k in range(16)},
    # Role of the concept in per-class composed movies: on every frame it is the concept
    # chosen for its event class, and the header names the real one.
    CONCEPT_ROLE: "VTCD: concept of the event class",
    "occlusion_blur": "Spatio-temporal occlusion: blur",
    "occlusion_mean": "Spatio-temporal occlusion: mean colour",
    "occlusion_black": "Spatio-temporal occlusion: black",
    "motion_local_freeze": "Motion: local freeze per volume",
}

#: Integrated Gradients baselines, which change the question and not only the reference.
IG_BASELINE = {
    "ig_blur": "Blur",
    "ig_mean": "Mean colour",
    "ig_black": "Black",
    "ig_noise": "Noise",
    "ig_tn_same_video": "Normal reference",
    "ig_ensemble": "Reference ensemble",
}

#: Names of the eleven input perturbations in headers and summaries.
VIEWS = {
    "occlusion_blur": "blurred occlusion",
    "occlusion_mean": "mean-colour occlusion",
    "occlusion_black": "black occlusion",
    "motion_local_freeze": "local freeze of the volume",
    "vtcd_concept_00_black": "global VTCD concept 0 covered in black",
    # The fraction is in the name: batches computed at different fractions must not share a
    # catalogue row and label.
    "ig_top10_black": "top 10% of IG ensemble tokens covered in black",
    "freeze": "global freeze (no motion)",
    "reverse": "reversed time",
    **{f"permutation_{s}": f"temporal permutation (seed {s})" for s in range(42, 47)},
}

#: Families, in presentation order.
METHOD_FAMILIES = (
    (
        "Raw Attention",
        ("raw_attention",),
        "Shows how much the attentive head weighs the tokens. A high weight alone does not prove a positive contribution to the anomaly score.",
    ),
    (
        "Gradient × Activation",
        ("gradient_activation", "gradient_activation_prenorm"),
        "Combines the feature value with the logit sensitivity. The sign separates positive and negative local contributions.",
    ),
    (
        "Attention × Gradient",
        ("attention_gradient",),
        "Weighs attention by the logit sensitivity: separates tokens supporting or opposing the decision.",
    ),
    (
        "Integrated Gradients",
        ("ig_blur", "ig_mean", "ig_black", "ig_noise", "ig_tn_same_video", "ig_ensemble"),
        "Attributes the logit difference with respect to a reference, integrating gradients along the path between the two features. Variants change the reference.",
    ),
    (
        "LeGrad",
        ("legrad_last", "legrad_four"),
        "Uses the positive part of the attention gradients to localize evidence for the anomaly.",
    ),
    (
        "PCA-RGB",
        ("pca_rgb",),
        "Colours the features with three principal components. Similar colours mean similar representations, not necessarily anomalies.",
    ),
    (
        "EigenCAM",
        ("eigencam",),
        "Projects the features on the dominant principal component. Describes the representation without using the score gradient.",
    ),
    (
        "Cosine similarity",
        ("cosine",),
        "Compares every token with an explicit reference region. Cosine measures feature similarity, not anomaly probability.",
    ),
    (
        "Anomaly Direction",
        ("anomaly_projection", "anomaly_sensitivity"),
        "Learns a normal/anomalous direction from training windows. The two views separate similarity to the direction from sensitivity of the decision.",
    ),
    (
        "Class direction (per-class TCAV)",
        ("class_projection", "class_sensitivity"),
        "The same question asked on the concept the window actually carries (Robbery on a robbery chunk) instead of the generic «anomalous». The direction is chosen window by window and the header states which: on a multi-class video it changes from one chunk to the next, and on a normal chunk it does not exist, so the map is null and says so.",
    ),
    (
        "Anomaly Prototypes",
        ("prototype_difference", "prototype_anomaly", "prototype_normal"),
        "Two centroids learned on the same 52 training windows: the main view is the difference of the two cosines, positive where the region is closer to the anomalous prototype.",
    ),
    (
        "VTCD – final layer",
        ("vtcd_ids", "vtcd_importance", *(f"vtcd_concept_{k:02d}" for k in range(16))),
        "Groups spatio-temporal regions of the final features into concepts. Colours and IDs are fixed only within each dictionary; they do not correspond to annotated classes and are not aligned across configurations.",
    ),
    (
        "Occlusion",
        ("occlusion_blur", "occlusion_mean", "occlusion_black"),
        "Replaces an input volume and reruns the detector. A positive delta means the replacement lowered the original logit. The map averages the deltas of overlapping masks of different areas: it is not a pointwise per-pixel attribution.",
    ),
    (
        "Motion",
        ("motion_local_freeze",),
        "Measures the effect of repeating over time the central frame of each input volume.",
    ),
)

_FAMILY_OF = {metodo: nome for nome, metodi, _ in METHOD_FAMILIES for metodo in metodi}

#: What the colour means: the bar says how much, this row says of what.
MEANINGS = {
    "raw_attention": "bright = the query reads more from here",
    "gradient_activation": "yellow = raises the logit · violet = lowers it",
    "gradient_activation_prenorm": "yellow = raises the logit · violet = lowers it",
    "attention_gradient": "bright = the connection raises the logit",
    "legrad_last": "bright = reading more here would raise the logit",
    "legrad_four": "bright = reading more here would raise the logit",
    "eigencam": "bright = more energy on the dominant component",
    "pca_rgb": "similar colours = similar features, not «more anomalous»",
    "cosine": "red = similar to the query · blue = opposite",
    "ig_": "red = pushes towards the anomaly · blue = towards the reference",
    "anomaly_projection": "red = anomalous side of the learned direction",
    "class_projection": "red = side of THIS class concept",
    "class_sensitivity": "red = raises the logit along the class; blue = lowers it; zero transparent",
    "anomaly_sensitivity": "red = moving along the direction raises the logit",
    "prototype_difference": "red = closer to the anomalous prototype",
    "prototype_anomaly": "red = similar to the anomalous prototype",
    "prototype_normal": "red = similar to the normal prototype",
    "vtcd_ids": "one colour per concept, not a scale",
    "vtcd_importance": "bright = removing the concept costs more",
    "vtcd_concept_": "white = the concept is assigned here",
    "occlusion_": "bright = removing this region lowers the logit",
    "motion_local_": "bright = removing motion here lowers the logit",
}


def base_method(method: str) -> str:
    """:return: the method ``method`` is a perturbed view of, or itself.

    Colormap, sign and colour scale follow the base method: a map and its perturbed
    counterpart measure the same quantity.
    """
    if not method.startswith("pert__"):
        return method
    _, _, rest = method.partition("__")
    _, _, base = rest.partition("__")
    return base or method


def view_of(method: str) -> str:
    """:return: the perturbation the map was computed on, or an empty string."""
    if not method.startswith("pert__"):
        return ""
    _, _, rest = method.partition("__")
    view, _, base = rest.partition("__")
    return view if base else ""


_CONCEPT_UNION = re.compile(r"vtcd_concept_(\d{2}(?:\+\d{2})+)")


def method_concepts(method: str) -> tuple[int, ...] | None:
    """Return the concepts of a ``vtcd_concept_KK`` map or a ``vtcd_concept_05+07`` union.

    Unions (Shoplifting: 5 and 7 as one concept) are not in the campaign npz files: they
    are built by :func:`render_campaign` (``unioni``) as the maximum of the masks.

    :return: the IDs in order, or ``None`` if ``method`` is not a concept map.
    """
    base = map_kind(method)
    union = _CONCEPT_UNION.fullmatch(base)
    if union:
        return tuple(int(k) for k in union[1].split("+"))
    if re.fullmatch(r"vtcd_concept_\d{2}", base):
        return (int(base.rsplit("_", 1)[1]),)
    return None


def method_label(method: str) -> str:
    """:return: a readable title, uniform for all IG and perturbed variants."""
    if _CONCEPT_UNION.fullmatch(method):
        return "VTCD: concepts " + "+".join(str(k) for k in method_concepts(method))
    if "__class_" in method:
        base, selected = method.rsplit("__class_", 1)
        label = "true class" if selected == "true" else UCF_ANOMALY_CLASSES[int(selected)]
        prefix = "Per-class VTCD: " if base.startswith("vtcd_") else ""
        return f"{prefix}{method_label(base)} · {label}"
    view = view_of(method)
    if view:
        return f"{method_label(base_method(method))} — on {VIEWS.get(view, view)}"
    if method.startswith("ig_"):
        return f"Integrated Gradients – «{IG_BASELINE.get(method, method[3:].replace('_', ' '))}»"
    return LABELS.get(method, method)


def family_of(method: str) -> str:
    """:return: the family the method belongs to, for the context row."""
    view = view_of(method)
    if view:
        return "Input perturbations"
    base = base_method(method)
    if base.startswith("ig_"):
        return "Integrated Gradients"
    return _FAMILY_OF.get(base, "Explainability")


def vtcd_family(meta: dict) -> str:
    """Return the context row of a VTCD map: the dictionary that produced it.

    Concept IDs are valid only within a dictionary, defined by three numbers (training
    windows, SLIC regions, concepts), read from the campaign because the collection carries
    several configurations.

    :param meta: the already read ``campaign.json``.
    :return: ``"VTCD – 52 training windows, 32 regions, 8 concepts"``, or the generic family
        if the campaign lacks one of the three numbers.
    """
    vtcd = (meta.get("explanation_settings") or {}).get("vtcd") or {}
    references = (meta.get("reference_diagnostics") or {}).get("references")
    regions = (vtcd.get("slic") or {}).get("n_segments")
    concepts = vtcd.get("n_concepts")
    if not isinstance(references, list) or regions is None or concepts is None:
        return _FAMILY_OF["vtcd_ids"]
    return f"VTCD – {len(references)} training windows, {regions} regions, {concepts} concepts"


def meaning_of(method: str) -> str:
    """:return: a row stating what the colour means, or an empty string."""
    base = map_kind(method)
    for key, text in MEANINGS.items():
        if base == key or (key.endswith("_") and base.startswith(key)):
            return text
    return ""


def model_label(run_id: str) -> str:
    """:return: a readable name, also for synthetic runs without underscore-separated fields."""
    # Before the DoRA case: an E3-S `fondi` cell names the reparam UNDER the prompts
    # (`vptshallow-n8-su-dora-r32`).
    sequential = re.search(r"vpt(shallow|deep)-n(\d+)-su-(lora|dora)-r(\d+)", run_id)
    if sequential:
        kind, n, reparam, rank = sequential.groups()
        name = {"lora": "LoRA", "dora": "DoRA"}[reparam]
        return f"VPT-{kind} n{n} on merged {name} r{rank}"
    if "dora" in run_id:
        return "DoRA r32"
    if "frozen" in run_id:
        return "Frozen backbone"
    return run_id


class _VideoWriter:
    """H.264/yuv420p, readable by the HTML catalogue in browsers."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int]):
        self.container = av.open(str(path), mode="w", options={"movflags": "+faststart"})
        self.stream = self.container.add_stream(
            "libx264", rate=Fraction(str(fps)).limit_denominator(100000)
        )
        self.stream.width, self.stream.height = size
        self.stream.pix_fmt = "yuv420p"
        self.stream.thread_count = 1
        self.stream.options = {"crf": "20", "preset": "veryfast"}

    def isOpened(self):
        return self.container is not None

    def write(self, array):
        for packet in self.stream.encode(av.VideoFrame.from_ndarray(array, format="bgr24")):
            self.container.mux(packet)

    def release(self):
        if self.container is not None:
            try:
                for packet in self.stream.encode():
                    self.container.mux(packet)
            finally:
                self.container.close()
                self.container = None


def covered_segments(assignments: np.ndarray) -> list[tuple[int, int]]:
    """Return the covered inclusive intervals, without filling gaps with invented maps.

    :param assignments: window index per frame; -1 means absence.
    :return: contiguous segments in temporal order.
    """
    covered = np.asarray(assignments) >= 0
    edges = np.diff(np.r_[False, covered, False].astype(np.int8))
    return list(
        zip(
            np.flatnonzero(edges == 1).tolist(),
            (np.flatnonzero(edges == -1) - 1).tolist(),
            strict=True,
        )
    )


def class_of(record: dict, metodo: str) -> str:
    """:return: ``" · class Robbery"`` for a class map, an empty string otherwise.

    A map whose direction changes between windows must say which: otherwise two movies of
    the same method on the same video would describe different concepts with the same
    caption (it happens within one movie on multi-class videos).

    :param record: the ``campaign.json`` window, with its ``checks``.
    :param metodo: the name of the map being drawn.
    """
    if "__class_" in metodo:
        selected = metodo.rsplit("__class_", 1)[1]
        if selected != "true":
            return f" · class {UCF_ANOMALY_CLASSES[int(selected)]}"
    if not metodo.startswith("class_") and not metodo.endswith("__class_true"):
        return ""
    checks = record.get("checks", {})
    index = (
        checks.get("class_direction")
        or checks.get("class_direction_all")
        or checks.get("vtcd_class")
        or {}
    ).get("class_idx", -1)
    if index is None or int(index) < 0 or int(index) >= len(UCF_ANOMALY_CLASSES):
        return " · no class"
    return f" · class {UCF_ANOMALY_CLASSES[int(index)]}"


def _state(values: np.ndarray, record: dict, method: str) -> str:
    if "__class_" in method and record.get("label") == 0:
        return "NORMAL CHUNK: class map not shown"
    if method.startswith("class_") and not np.any(values):
        # A null class map is not a method failure: it is a chunk without a class.
        return (record.get("checks", {}).get("class_direction", {}) or {}).get(
            "stato"
        ) or "NULL MAP: no class for this window"
    if map_kind(method) == "vtcd_ids":
        return f"SINGLE CONCEPT: ID {int(values.flat[0])}" if np.ptp(values) == 0 else ""
    if method.startswith("vtcd_concept_") and not np.any(values):
        return "CONCEPT ABSENT IN THIS WINDOW"
    if not np.any(values):
        return "NULL MAP: no positive contribution" if method.startswith("legrad") else "NULL MAP"
    check = record.get("checks", {}).get(method, {})
    if method.startswith("ig_") and check.get("converged") is False:
        return "IG NOT CONVERGED: diagnostics only"
    if method == "gradient_activation_prenorm":
        return "PRE-LAYERNORM: diagnostic control"
    if np.ptp(values) == 0:
        return "UNIFORM MAP"
    return ""


#: Colormaps chosen per method: ``inferno`` on Attention × Gradient, EigenCAM and VTCD;
#: ``coolwarm`` on Integrated Gradients, cosine similarity, Anomaly Direction and
#: prototypes. Keys ending with ``_`` are prefixes. A signed map may use a sequential
#: colormap, with zero in the middle of the bar; the scale stays symmetric
#: (``_scale_bounds``). Movies rendered before 13 September 2026 use the previous
#: convention (``render_spiegazioni.py --force`` regenerates them).
COLORMAPS = {
    "attention_gradient": "inferno",
    "eigencam": "inferno",
    "vtcd_importance": "inferno",
    "ig_": "coolwarm",
    "cosine": "coolwarm",
    "anomaly_": "coolwarm",
    "class_": "coolwarm",
    "prototype_": "coolwarm",
}
SIGNED_CMAP = "viridis"
UNSIGNED_CMAP = "inferno"


def _cmap_name(method: str, signed: bool) -> str:
    """:return: the colormap of ``method``; the sign decides only the unlisted cases."""
    method = base_method(method)
    for key, name in COLORMAPS.items():
        if method == key or (key.endswith("_") and method.startswith(key)):
            return name
    return SIGNED_CMAP if signed else UNSIGNED_CMAP


def _is_signed(method: str) -> bool:
    method = map_kind(method)
    return method not in (
        "raw_attention",
        "legrad_last",
        "legrad_four",
        "pca_rgb",
        "vtcd_ids",
    ) and not method.startswith("vtcd_concept_")


def _scale_bounds(method: str, all_values: np.ndarray, signed: bool):
    """Return the colormap extremes of ``method`` on the observed values.

    The only place of the criterion (robust percentile, not the absolute maximum), shared by
    the per-campaign scale and the group scale (``compute_group_scales``).

    :param all_values: raw map values, flattened (``pca_rgb``: last dimension with 3
        components, not flattened).
    :return: ``(lo, hi)``, scalars except for ``pca_rgb`` (one value per channel).
    """
    method = map_kind(method)
    if method in ("cosine", "prototype_anomaly", "prototype_normal"):
        return -1.0, 1.0
    if map_kind(method) == "vtcd_ids":
        return 0.0, 7.0
    if method.startswith("vtcd_concept_"):
        return 0.0, 1.0
    if method == "pca_rgb":
        lo, hi = np.percentile(all_values, [1, 99], axis=0)
        return lo, hi
    if signed:
        hi = float(np.percentile(abs(all_values), 99))
        return -hi, hi
    return tuple(np.percentile(all_values, [1, 99]))


def compute_group_scales(folders: list[Path]) -> dict[str, dict]:
    """Return a common per-method scale over a group of campaigns (same model/run).

    Colour becomes comparable ACROSS videos. A campaign without a method (e.g. no TN
    reference found in the split) simply does not contribute to its range.

    :param folders: campaign folders with a complete ``campaign.json``.
    :return: per method, ``{"lo": ..., "hi": ...}`` in the ``render_campaign`` form (scalar,
        except ``pca_rgb`` with 3 components).
    """
    values_by_method: dict[str, list[np.ndarray]] = {}
    for folder in folders:
        meta = json.loads((folder / "campaign.json").read_text(encoding="utf-8"))
        if meta["status"] != "complete":
            continue
        for i in range(len(meta["windows"])):
            with np.load(folder / f"maps_{i:05d}.npz") as archive:
                for method, values in archive.items():
                    # A perturbed view measures the same quantity as its base method: they are accumulated
                    # together.
                    base = map_kind(method)
                    values_by_method.setdefault(base, []).append(
                        values.reshape(-1, 3) if base == "pca_rgb" else values.ravel()
                    )
    scales: dict[str, dict] = {}
    for method, chunks in values_by_method.items():
        all_values = np.concatenate(chunks)
        lo, hi = _scale_bounds(method, all_values, _is_signed(method))
        scales[method] = {"lo": np.asarray(lo).tolist(), "hi": np.asarray(hi).tolist()}
    return scales


def _truncate(testo: str, width: int, font, scala: float, thickness: int = 1) -> str:
    """:return: ``testo`` shortened until it fits in ``larghezza`` px, already in ASCII.

    The transliterated text is measured, since that is what will be drawn.
    """
    testo = _ascii(testo)
    if cv2.getTextSize(testo, font, scala, thickness)[0][0] <= width:
        return testo
    for cut in range(len(testo) - 1, 0, -1):
        short = testo[:cut] + "..."
        if cv2.getTextSize(short, font, scala, thickness)[0][0] <= width:
            return short
    return ""


@dataclass(frozen=True)
class Header:
    """What a movie header must say, without deciding how.

    Separate from drawing because the three movie types show the same fields at different
    widths.

    :param significato: what the colour means, in words.
    :param avviso: diagnostics to highlight, or an empty string.
    :param extra: context row at the bottom of the band (chosen mask, logit delta); a
        warning takes its place.
    :param score_perturbato: score the detector assigns to the alternative input, next to
        the original one, with its own verdict against the same threshold.
    :param chunk_positivo: whether the shown window is truly anomalous according to the
        annotation, or ``None`` when unknown. The TP/FP/FN/TN code printed in each badge is
        derived here from the same rule. ``None`` is not negative: the badge stays without a
        code. With the code, the badge takes the colour of the chunk cell in the timeline
        (:data:`VERDICT_COLORS`).
    :param etichette_legenda: one label per equal bar segment, for categorical legends (VTCD
        IDs), drawn after widening the bar.
    """

    metodo: str
    famiglia: str
    modello: str
    video: str
    dettaglio: str
    significato: str = ""
    score: float | None = None
    soglia: float | None = None
    avviso: str = ""
    extra: str = ""
    legenda: np.ndarray | None = None
    estremi: tuple[str, str] | None = None
    score_perturbato: float | None = None
    etichetta_perturbato: str = "PERTURBED"
    chunk_positivo: bool | None = None
    etichette_legenda: tuple[str, ...] = ()


#: Header and strip geometry. The header holds four rows plus the colour bar; the strip
#: covers the whole video, so its height does not depend on the panels above. The strip
#: grew from 58 to 92 px on 17 September 2026 for the two lanes (chunks with their
#: verdict, and the annotation).
HEADER_H = 142
#: Header of NARROW movies (one panel on a 320x240 video, i.e. most of UCF-Crime): the
#: context rows go below the badge, at full width, and wrap. Wide movies keep
#: :data:`HEADER_H`.
NARROW_HEADER_H = 196
NARROW_WIDTH = 520
STRIP_H = 98
#: Kept under the old name: earlier rendering manifests cite it and
#: ``explanation_advanced`` still measures the total height with it.
_HEADER_H = HEADER_H + STRIP_H
_LEGEND_H = 16
#: Map weight in the overlay on the frame. Not the ADR-063 0.80: here the bare map is in
#: the ``map`` cut, and the overlay must show where the colour falls on the content.
OVERLAY_ALPHA = 0.55

#: Gap between two side-by-side badges, and minimum text that must remain on their left
#: for two badges to be worth drawing.
_GAP_BADGE = 8
_MIN_TEXT = 220
_FONT = cv2.FONT_HERSHEY_DUPLEX
_FONT_MONO = cv2.FONT_HERSHEY_SIMPLEX


#: OpenCV Hershey fonts cover printable ASCII only: any other character silently
#: disappears. Only what ends up in the pixels is transliterated.
_REPLACEMENTS = {
    "·": "-",
    "—": "-",
    "–": "-",
    "…": "...",
    "×": "x",
    "«": '"',
    "»": '"',
    "“": '"',
    "”": '"',
    "’": "'",
    "⚠": "!",
    "️": "",
}


def _ascii(testo: str) -> str:
    """:return: ``testo`` written with only the characters the font can draw."""
    for origin, destination in _REPLACEMENTS.items():
        testo = testo.replace(origin, destination)
    decomposed = unicodedata.normalize("NFKD", testo)
    return "".join(c for c in decomposed if 32 <= ord(c) < 127)


def _text(canvas, testo, origine, scala, color, font=_FONT_MONO, thickness=1):
    cv2.putText(canvas, _ascii(testo), origine, font, scala, color, thickness, cv2.LINE_AA)


def header_height(width: int) -> int:
    """:return: the header height in pixels of a movie ``larghezza`` wide.

    The narrow header is taller, so whoever opens a ``VideoWriter`` or reads a frame body
    goes through here.
    """
    return NARROW_HEADER_H if width < NARROW_WIDTH else HEADER_H


def _wrap_lines(testo: str, width: int, font, scala: float, max_rows: int) -> list[str]:
    """:return: ``testo`` split on spaces into at most ``max_rows`` rows ``larghezza`` wide.

    The last row carries the remainder and is truncated with an ellipsis only if needed.
    """
    rows: list[str] = []
    current = ""
    for word in _ascii(testo).split(" "):
        trial = f"{current} {word}".strip()
        if not current or cv2.getTextSize(trial, font, scala, 1)[0][0] <= width:
            current = trial
        else:
            rows.append(current)
            current = word
    rows.append(current)
    if len(rows) > max_rows:
        rows = [*rows[: max_rows - 1], " ".join(rows[max_rows - 1 :])]
    return [_truncate(row, width, font, scala) for row in rows]


def header(width: int, header_spec: Header) -> np.ndarray:
    """Return the top band: method, provenance, colour meaning, score.

    One or two score badges: the second appears when the header carries
    ``score_perturbato``, in panel order (original left, perturbed right), each with its
    verdict against the threshold; on a band too narrow for both badges and the title, the
    second is dropped. When the campaign knows whether the window is truly anomalous, each
    badge also carries its TP/FP/FN/TN code; the two codes can differ (a perturbation that
    switches the alarm off turns TP into FN), which is what the movie exists to show. Below
    :data:`NARROW_WIDTH` the context rows go below the badge.

    :param larghezza: movie body width, in pixels.
    :return: a BGR band ``(header_height(larghezza), larghezza, 3)``.
    """
    height = header_height(width)
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:] = COLORS["pannello"]
    cv2.line(canvas, (0, height - 1), (width, height - 1), COLORS["linea"], 1)
    margin = 12
    narrow = width < NARROW_WIDTH
    badge_side = 118 if narrow else 156
    # Scores to show, in the order of the panels below: the real input first, then the
    # alternative one. The last stays flush with the right edge.
    badge = [("ANOMALY SCORE", header_spec.score)] if header_spec.score is not None else []
    if header_spec.score_perturbato is not None:
        badge.append((header_spec.etichetta_perturbato, header_spec.score_perturbato))
    frame_box = len(badge) * badge_side + max(0, len(badge) - 1) * _GAP_BADGE
    # An extra badge is not worth a header without a title: if the two boxes leave no room
    # for the text rows, the second is dropped and its number stays in the context row.
    while len(badge) > 1 and width - 2 * margin - frame_box < _MIN_TEXT:
        badge.pop()
        frame_box = len(badge) * badge_side + max(0, len(badge) - 1) * _GAP_BADGE
    useful = max(60, width - 2 * margin - frame_box)
    _text(canvas, "EVD · EXPLAINABILITY", (margin, 18), 0.34, COLORS["tenue"])
    scale = 0.62
    if narrow:
        # The title stays next to the badge: it shrinks before being truncated.
        scale = next(
            (
                s
                for s in (0.48, 0.42, 0.36)
                if cv2.getTextSize(_ascii(header_spec.metodo), _FONT, s, 1)[0][0] <= useful
            ),
            0.36,
        )
    _text(
        canvas,
        _truncate(header_spec.metodo, useful, _FONT, scale),
        (margin, 44),
        scale,
        COLORS["testo"],
        font=_FONT,
    )
    context = (
        f"{header_spec.famiglia} · {header_spec.modello}",
        f"{header_spec.video} · {header_spec.dettaglio}",
    )
    if narrow:
        y = 90
        for text in context:
            for row in _wrap_lines(text, width - 2 * margin, _FONT_MONO, 0.38, 2):
                _text(canvas, row, (margin, y), 0.38, COLORS["tenue"])
                y += 15
    else:
        for text, y in zip(context, (62, 78), strict=True):
            _text(
                canvas,
                _truncate(text, useful, _FONT_MONO, 0.38),
                (margin, y),
                0.38,
                COLORS["tenue"],
            )
    x0 = width - margin - frame_box
    for label, value in badge:
        alarm = header_spec.soglia is not None and value >= header_spec.soglia
        # The code sits INSIDE the badge, top right, not at the end of the threshold row, where
        # two badges would overlap.
        code_name = chunk_verdict(alarm, header_spec.chunk_positivo)
        # With the ground truth the badge takes the colour of the chunk cell in the timeline; TN is
        # dark like its cell, so the text turns light and the border separates it.
        color = (
            COLORS[VERDICT_COLORS[code_name]]
            if code_name
            else (COLORS["allarme"] if alarm else COLORS["calma"])
        )
        ink = COLORS["testo"] if code_name == "TN" else COLORS["fondo"]
        cv2.rectangle(canvas, (x0, 10), (x0 + badge_side, 58), color, -1)
        if code_name == "TN":
            cv2.rectangle(canvas, (x0, 10), (x0 + badge_side, 58), COLORS["tenue"], 1)
        code_space = 0
        if code_name:
            code_width = cv2.getTextSize(code_name, _FONT, 0.46, 1)[0][0]
            code_space = code_width + 8
            _text(
                canvas,
                code_name,
                (x0 + badge_side - 8 - code_width, 27),
                0.46,
                ink,
                font=_FONT,
            )
        _text(
            canvas,
            _truncate(label, badge_side - 12 - code_space, _FONT_MONO, 0.32),
            (x0 + 8, 26),
            0.32,
            ink,
        )
        _text(
            canvas,
            f"{value:.4f}",
            (x0 + 8, 50),
            0.46 if narrow else 0.54,
            ink,
            font=_FONT,
        )
        if header_spec.soglia is not None:
            _text(
                canvas,
                f"threshold {header_spec.soglia:.4f} · {'ABOVE' if alarm else 'below'}",
                (x0 + 8, 72),
                0.32,
                COLORS["tenue"],
            )
        x0 += badge_side + _GAP_BADGE
    if header_spec.legenda is not None and header_spec.estremi is not None:
        bar_height = header_spec.legenda.shape[0]
        available = width - 2 * margin
        bar = cv2.resize(
            header_spec.legenda, (available, bar_height), interpolation=cv2.INTER_LINEAR
        )
        y0 = 146 if narrow else 92
        canvas[y0 : y0 + bar_height, margin : margin + available] = bar
        cv2.rectangle(
            canvas, (margin, y0), (margin + available, y0 + bar_height), COLORS["linea"], 1
        )
        # Category labels go on the ALREADY widened bar (in the narrow bar the resize stretched
        # them).
        n_categories = len(header_spec.etichette_legenda)
        for k, written in enumerate(header_spec.etichette_legenda):
            wide, text_top = cv2.getTextSize(written, _FONT_MONO, 0.33, 1)[0]
            center = margin + int((k + 0.5) * available / n_categories)
            _text(
                canvas,
                written,
                (center - wide // 2, y0 + (bar_height + text_top) // 2),
                0.33,
                (255, 255, 255),
            )
        # Extremes at the two ends of the bar and, in the middle, what the colour means.
        bottom, top = _ascii(header_spec.estremi[0]), _ascii(header_spec.estremi[1])
        row = y0 + bar_height + 14
        wide_low = cv2.getTextSize(bottom, _FONT_MONO, 0.33, 1)[0][0]
        wide_high = cv2.getTextSize(top, _FONT_MONO, 0.33, 1)[0][0]
        _text(canvas, bottom, (margin, row), 0.33, COLORS["tenue"])
        _text(canvas, top, (width - margin - wide_high, row), 0.33, COLORS["tenue"])
        space = available - wide_low - wide_high - 24
        if header_spec.significato and space > 80:
            center = _truncate(header_spec.significato, space, _FONT_MONO, 0.33)
            wide = cv2.getTextSize(center, _FONT_MONO, 0.33, 1)[0][0]
            _text(canvas, center, ((width - wide) // 2, row), 0.33, COLORS["testo"])
    if header_spec.avviso or header_spec.extra:
        _text(
            canvas,
            _truncate(
                header_spec.avviso or header_spec.extra, width - 2 * margin, _FONT_MONO, 0.36
            ),
            (margin, height - 7),
            0.36,
            COLORS["allarme"] if header_spec.avviso else COLORS["tenue"],
        )
    return canvas


#: Colour filling a chunk cell in the lane: the verdict, not just the decision. A TN stays
#: almost invisible, the case that needs no attention.
VERDICT_COLORS = {"TP": "calma", "FP": "allarme", "FN": "mancato", "TN": "linea"}


def chunk_verdict(above_threshold: bool, positive: bool | None) -> str:
    """:return: ``TP``/``FP``/``FN``/``TN``, or an empty string if the truth is unknown.

    Without a label there is no guess: an unlabelled chunk is not a negative.

    :param above_threshold: whether the score reaches the operating threshold.
    :param positivo: whether the window intersects an annotated interval.
    """
    if positive is None:
        return ""
    return ("T" if above_threshold == bool(positive) else "F") + ("P" if above_threshold else "N")


@dataclass(frozen=True)
class Timeline:
    """Everything the strip below the movie must show, derived once.

    Shared by the six renderers of the collection (ADR-052: the conversion is what must be
    factored).

    :param punteggi: score per native frame, projected from the window that covers the frame
        and closes first, whatever the campaign policy (ADR-077); ``0`` where no window
        exists (not drawn).
    :param assegnazioni: per frame, the index of the covering window, ``-1`` if none; it says
        which map is shown, and with ``overlap_positivo`` it may differ from the window of
        ``punteggi``.
    :param spans: the true frame interval of every window. Not the ``assegnazioni`` region:
        windows overlap (4 s hop on an 8 s window) while regions are a partition; drawing
        regions would misplace TP cells. The region ENDS where the span ends (since 20
        September 2026 ``offline_assignment`` gives a frame the window that contains it and
        closes first), so the shown score changes exactly at a chunk end mark.
    :param score: the score of every window, read here rather than from ``punteggi``.
    :param soglia: the operating threshold, for the line and the verdicts.
    :param etichette: per window ``1``, ``0``, or ``None`` if unknown.
    :param verita: annotated intervals, or ``None`` if the ground-truth file is missing;
        ``()`` (a normal video) and ``None`` differ.
    :param classi: the annotated class of each ``verita`` interval, in the same order, or
        ``None`` if the ground-truth report does not carry it.
    """

    punteggi: np.ndarray
    assegnazioni: np.ndarray
    spans: tuple[tuple[int, int], ...] = ()
    score: tuple[float, ...] = ()
    soglia: float | None = None
    etichette: tuple[int | None, ...] = ()
    verita: tuple[tuple[int, int], ...] | None = None
    classi: tuple[str, ...] | None = None

    def event_at(self, frame: int) -> str:
        """:return: the class of the annotated event containing ``frame``, or ``""``.

        Two overlapping events of different classes are both named, in annotation order.
        """
        if not self.verita or not self.classi:
            return ""
        present = (
            c for (a, b), c in zip(self.verita, self.classi, strict=True) if a <= frame <= b and c
        )
        return " + ".join(dict.fromkeys(present))

    @property
    def num_frames(self) -> int:
        """:return: how many frames the strip covers."""
        return int(len(self.punteggi))

    def label_of(self, finestra: int) -> bool | None:
        """:return: whether the window is truly anomalous, or ``None`` if unknown."""
        if finestra < 0 or finestra >= len(self.etichette):
            return None
        value = self.etichette[finestra]
        return None if value is None else bool(value)

    def finestre(self) -> range:
        """:return: the window indices, in temporal order.

        Cell geometry comes from ``_cell``, the only translation of a span into pixels.
        """
        return range(len(self.spans))


@lru_cache(maxsize=4)
def _truth_report(folder: Path) -> tuple[tuple[str, dict], ...]:
    """Return the ground-truth report of the collection containing ``folder``, if any.

    A few levels are walked up instead of requiring a path (the collection is three levels
    above a campaign, ``risultati/gruppo/modello/video``); elsewhere its absence is the right
    answer. Cached: the six renderers of a campaign would otherwise read it six times.

    :param folder: the campaign folder.
    :return: the report entries as pairs, or empty if it does not exist.
    """
    for folder_path in (folder, *list(folder.parents)[:4]):
        path = folder_path / "rapporti/verita.json"
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return tuple((data.get("video") or {}).items())
    return ()


def campaign_truth(folder: Path, records) -> tuple[tuple[int | None, ...], tuple | None]:
    """Return chunk labels and annotated intervals, if the collection carries them.

    ``rapporti/verita.json`` is written by ``scripts/prestazioni_explain.py`` from run scores
    and annotations; rendering builds no dataset (ADR-015). Missing does not mean negative:
    labels stay ``None`` and the strip draws the decision only, saying so in the legend.

    :param folder: the campaign folder.
    :param records: the ``campaign.json`` windows, in their order.
    :return: ``(labels per window, annotated intervals or None)``.
    """
    nothing = (None,) * len(records), None
    if not records:
        return nothing
    data = dict(_truth_report(folder.resolve())).get(records[0]["window"]["video_id"])
    if not data:
        return nothing
    per_id = data.get("finestre") or {}
    labels = tuple(per_id.get(r["window_id"]) for r in records)
    intervals = tuple((int(a), int(b)) for a, b in data.get("intervalli") or ())
    return labels, intervals


def campaign_classes(folder: Path, records) -> tuple[str, ...] | None:
    """:return: the class of every annotated interval, parallel to those of
        :func:`campaign_truth`, or ``None`` if the report does not carry it.

    :raises ValueError: if classes and intervals differ in number (they are paired by
        position).
    """
    if not records:
        return None
    data = dict(_truth_report(folder.resolve())).get(records[0]["window"]["video_id"])
    if not data or data.get("classi") is None:
        return None
    classes = tuple(str(c) for c in data["classi"])
    if len(classes) != len(data.get("intervalli") or ()):
        raise ValueError(f"verita.json: classes and intervals not paired for {folder.name}")
    return classes


def assignment_of(folder: Path, meta: dict, finestre) -> tuple[np.ndarray, dict]:
    """Return which window every frame shows, with the policy the campaign declares.

    The policy travels with the campaign, not with the rendering script (ADR-077): it lives
    in ``explanation_settings.render_policy.assegnazione`` of ``campaign.json``, so every
    renderer (maps, boxes, comparisons) shows the same window on a frame. Missing means
    ``fine``, i.e. campaigns computed before 29 September 2026 render as before.

    :param folder: the campaign folder, where ``verita.json`` is looked up.
    :param meta: the already read ``campaign.json``.
    :param finestre: the :class:`~evd.types.WindowSpec` of the windows, in their order.
    :return: ``(window index per frame, receipt)``; the receipt names the policy and, if
        used, how many annotated frames decided it.
    :raises ValueError: on ``overlap_positivo`` for a video missing from ``verita.json``.
    """
    policy = (
        ((meta.get("explanation_settings") or {}).get("render_policy") or {}).get("assegnazione")
    ) or "fine"
    # The score curve does not follow the policy: it stays on the chunk end (`timeline_of`),
    # and the receipt says so next to the map rule.
    receipt: dict = {"politica": policy, "curva": "fine"}
    intervals = None
    if policy == "overlap_positivo":
        _, intervals = campaign_truth(folder, meta["windows"])
        if intervals is None:
            raise ValueError(
                f"Campaign {folder} requests the 'overlap_positivo' assignment, but "
                "rapporti/verita.json does not carry this video: without annotated intervals "
                "the window to show cannot be decided"
            )
        receipt.update(
            parita="nearest centre, then lower endpoint",
            intervalli=[[int(a), int(b)] for a, b in intervals],
            frame_annotati=int(frame_annotati(intervals, meta["num_frames"]).sum()),
        )
    elif policy == "fine":
        receipt["parita"] = "with equal ends the later start wins (ADR-050)"
    assignments = offline_assignment(
        finestre, meta["num_frames"], politica=policy, intervalli=intervals
    )
    return assignments, receipt


def windows_of(records: list[dict]) -> list[WindowSpec]:
    """:return: the ``campaign.json`` windows, in the form produced by the sampler."""
    return [
        WindowSpec(
            **{
                **r["window"],
                "explicit_frame_indices": tuple(r["window"]["explicit_frame_indices"]),
            }
        )
        for r in records
    ]


def timeline_of(folder: Path, meta, records, assegnazioni) -> Timeline:
    """Return the strip of a campaign: per-frame scores, chunks, threshold and ground truth.

    The curve ALWAYS follows the chunk end, whatever the campaign policy (ADR-077);
    ``assegnazioni`` decides which map is shown, and with it the header and the highlighted
    cell. With ``fine`` the two coincide; with ``overlap_positivo`` the map may come from the
    most annotated chunk while the curve stays on the one closing first.

    :param folder: the campaign folder, where ``verita.json`` is looked up.
    :param meta: the already read ``campaign.json``.
    :param records: the windows, in their order.
    :param assegnazioni: frame -> index of the shown window, from :func:`assignment_of`.
    :return: the timeline to pass to :func:`strip_background`.
    """
    assegnazioni = np.asarray(assegnazioni)
    curve = offline_assignment(windows_of(records), int(assegnazioni.size))
    scores = np.array(
        [records[int(i)]["score"] if int(i) >= 0 else 0.0 for i in curve], dtype=float
    )
    spans = tuple((int(min(r["frame_indices"])), int(max(r["frame_indices"]))) for r in records)
    score = tuple(float(r["score"]) for r in records)
    labels, truth = campaign_truth(folder, records)
    classes = campaign_classes(folder, records)
    return Timeline(
        scores, assegnazioni, spans, score, meta.get("threshold"), labels, truth, classes
    )


def annotated_intervals(line: Timeline, classe: str | None = None) -> tuple[tuple[int, int], ...]:
    """Return the annotated positive intervals receiving boxes, of every class or of one.

    A single class serves per-class VTCD: the box concept is chosen for the dictionary
    class, so its boxes stay on the events of that class.

    :param linea: the campaign timeline.
    :param classe: the annotated class to keep, or ``None`` for all.
    :return: inclusive intervals, empty on a video without events.
    :raises ValueError: if a class is requested but the ground-truth report lacks the
        interval classes.
    """
    if classe is None:
        return tuple(line.verita or ())
    if line.verita is None or line.classi is None:
        raise ValueError(
            f"Boxes of class {classe} only: rapporti/verita.json does not carry the classes "
            "of the intervals"
        )
    return tuple(iv for iv, c in zip(line.verita, line.classi, strict=True) if c == classe)


def classes_per_frame(line: Timeline, num_frames: int) -> tuple[tuple[str, ...], np.ndarray]:
    """Return the class of every frame, for per-class composed movies.

    Inside an annotated event it is the event class (boxes and map come from that class
    dictionary). Outside every event it is the class of the nearest event (frames from its
    edge; on ties, the one starting first): no boxes are drawn there, but the side-by-side
    view still shows a mask, and the nearest event's is the only one continuing without
    jumps. On a single-class video the composed movie equals the single-class one.

    :param linea: the campaign timeline, with intervals and classes.
    :param num_frames: native frames of the video.
    :return: ``(classes in alphabetical order, class index per frame)``.
    :raises ValueError: if the ground truth has no intervals or classes, or two events of
        different classes cover the same frame.
    """
    if not line.verita or not line.classi:
        raise ValueError(
            "Per-class composed movie: rapporti/verita.json does not carry intervals with "
            "their classes"
        )
    classes = tuple(sorted(set(line.classi)))
    intervals = sorted(
        (int(a), min(int(b), num_frames - 1), classes.index(c))
        for (a, b), c in zip(line.verita, line.classi, strict=True)
    )
    index = np.full(num_frames, -1, dtype=np.int32)
    for a, b, k in intervals:
        stroke = index[a : b + 1]
        if np.any((stroke >= 0) & (stroke != k)):
            other = classes[int(stroke[(stroke >= 0) & (stroke != k)][0])]
            raise ValueError(
                f"Events of different classes on the same frames ({other}, {classes[k]}, "
                f"{a}-{b}): a frame would have two dictionaries"
            )
        stroke[:] = k
    outside = np.flatnonzero(index < 0)
    if outside.size:
        # Distance of every frame outside the events from each interval; `argmin` takes the first
        # minimum, i.e. the event starting first.
        distances = np.stack(
            [np.where(outside < a, a - outside, outside - b) for a, b, _ in intervals]
        )
        index[outside] = np.asarray([k for _, _, k in intervals])[np.argmin(distances, axis=0)]
    return classes, index


#: Geometry of the two lanes at the bottom of the strip: row of the first chunk row, cell
#: height, distance between the two staggered rows, row of the annotation lane. Module
#: constants because :func:`strip_background` and :func:`strip` both read them.
_Y_CHUNK, _CELL_THICKNESS, _ROW_STEP, _Y_TRUTH = 56, 7, 10, 76


def _column(frame: int, width: int, n: int) -> int:
    """Return the strip column a frame falls on.

    A single frame -> column map for everything (curve, lanes, chunk end marks, cursor), so
    lanes never appear shifted with respect to the curve.

    :param frame: native frame index, clamped to ``[0, n]``.
    :param larghezza: strip width in pixels.
    :param n: number of video frames, at least 1.
    :return: the column, in ``[0, larghezza]``.
    """
    return int(min(max(frame, 0), n) * width // max(n, 1))


def _cell(line: Timeline, width: int, finestra: int) -> tuple[int, int, int] | None:
    """Return the rectangle of a chunk cell in the lane: ``(x0, x1, y)``.

    :param linea: the video timeline.
    :param larghezza: strip width in pixels.
    :param finestra: window index.
    :return: ``(first column, last column, top row)``, or ``None`` if the window does not
        exist (frames on which the system has not decided yet).
    """
    if finestra < 0 or finestra >= len(line.spans):
        return None
    start, term = line.spans[finestra]
    n = max(line.num_frames, 1)
    x0 = _column(start, width, n)
    x1 = max(_column(term + 1, width, n) - 1, x0)
    return x0, x1, _Y_CHUNK + (finestra % 2) * _ROW_STEP


def strip_background(width: int, line: Timeline) -> np.ndarray:
    """Return the timeline of the whole video, drawn once.

    Three rows, in reading order: the score curve with the threshold, the chunk lane coloured
    by verdict, and the annotation lane (the same two lanes as ``eval.plots.timeline_png``
    and the annotated demonstrator video), plus the chunk grid, which tells which window the
    header refers to. The cursor is added per frame on a copy.

    :param larghezza: movie body width, in pixels.
    :param linea: what to show, from :func:`timeline_of`.
    :return: a BGR band ``(STRIP_H, larghezza, 3)``.
    """
    canvas = np.empty((STRIP_H, width, 3), dtype=np.uint8)
    canvas[:] = COLORS["fondo"]
    top, bottom = 20, 50
    n = max(line.num_frames, 1)
    threshold = line.soglia

    def column(frame: int) -> int:
        return _column(frame, width, n)

    columns = np.clip(np.arange(width) * n // width, 0, n - 1)
    values = np.asarray(line.punteggi, dtype=float)[columns]
    present = (np.asarray(line.assegnazioni) >= 0)[columns]
    if threshold is not None:
        y = int(bottom - (bottom - top) * float(np.clip(threshold, 0, 1)))
        cv2.line(canvas, (0, y), (width, y), COLORS["linea"], 1)
    previous = None
    for x in range(width):
        if not present[x]:
            previous = None
            continue
        y = int(bottom - (bottom - top) * float(np.clip(values[x], 0, 1)))
        above = threshold is not None and values[x] >= threshold
        color = COLORS["allarme"] if above else COLORS["calma"]
        cv2.line(canvas, previous or (x, y), (x, y), color, 1)
        previous = (x, y)

    thickness = _CELL_THICKNESS
    known = any(e is not None for e in line.etichette)
    # Two staggered rows: chunks overlap by half (4 s hop on an 8 s window), so a single row
    # would hide the grid. Two cells of the SAME row touch: on `edp_v2` a window lasts exactly
    # two hops, so windows i and i+2 are contiguous (what is drawn is `window_s`, not
    # `span_s`).
    for window in line.finestre():
        score = line.score[window] if window < len(line.score) else 0.0
        above = threshold is not None and score >= threshold
        code_name = chunk_verdict(above, line.label_of(window))
        name = VERDICT_COLORS[code_name] if code_name else ("allarme" if above else "linea")
        x0, x1, y = _cell(line, width, window)
        cv2.rectangle(canvas, (x0, y), (x1, y + thickness), COLORS[name], -1)
        # One pixel of separation at the start, against the previous cell of the same row.
        cv2.line(canvas, (x0, y), (x0, y + thickness), COLORS["fondo"], 1)
        # The END is marked, not the start: it is when the chunk produces its score (the
        # demonstrator reveals a window when `(end + 1) / fps` has passed). The mark falls exactly
        # where the curve steps, since a window region is `[previous end + 1, end]`.
        cv2.line(canvas, (x1, y - 1), (x1, y + thickness + 1), COLORS["tenue"], 1)
    for start, term in line.verita or ():
        x0 = column(start)
        x1 = max(column(term + 1) - 1, x0)
        cv2.rectangle(canvas, (x0, _Y_TRUTH), (x1, _Y_TRUTH + thickness), COLORS["verita"], -1)

    _text(canvas, "VIDEO SCORE", (4, 12), 0.3, COLORS["tenue"])
    # The legend sits on the FIRST row, next to the title: at the bottom it covered the
    # ground-truth lane.
    _lanes_legend(canvas, width, known, line.verita is not None)
    _text(canvas, f"{n} frame", (width - 66, STRIP_H - 4), 0.3, COLORS["tenue"])
    return canvas


def _lanes_legend(canvas: np.ndarray, width: int, known: bool, verita: bool) -> None:
    """Draw what the colours of the two lanes mean, at the bottom of the strip.

    Entries are dropped from the right when the strip is narrow.
    """
    y = 12
    limit = width - 8
    x = 120
    verdicts = [(s, VERDICT_COLORS[s]) for s in ("TP", "FP", "FN", "TN")]
    decision = [("ABOVE", "allarme"), ("below", "linea")]
    # «end» comes right after the lane title and before the verdicts: it explains the only
    # vertical bar of the grid. Entries drop from the right, so the order is also a priority
    # (on a 320 px strip TN drops too; from 480 px all fit).
    entry_list: list[tuple[str, str | None]] = [("CHUNK", None), ("end", "tenue")]
    entry_list += verdicts if known else decision
    entry_list.append(("TRUTH", "verita") if verita else ("TRUTH n/a", None))
    for name, color in entry_list:
        wide = cv2.getTextSize(_ascii(name), _FONT_MONO, 0.3, 1)[0][0]
        if x + wide + (13 if color else 0) + 10 > limit:
            return
        if color:
            cv2.rectangle(canvas, (x, y - 8), (x + 9, y - 1), COLORS[color], -1)
            x += 13
        _text(canvas, name, (x, y), 0.3, COLORS["tenue"])
        x += wide + 10


def strip(background: np.ndarray, frame: int, line: Timeline) -> np.ndarray:
    """:return: the strip with the cursor and the current chunk cell.

    The cursor also crosses the two lanes, and the cell of the window deciding on this frame
    is highlighted. The cell is chosen with ``assegnazioni``, the same rule that decides
    which window the header describes, so the cursor always lies inside the highlighted
    cell.

    :param background: the strip already drawn by :func:`strip_background`.
    :param frame: the current native frame.
    :param linea: the video timeline, from :func:`timeline_of`.
    """
    canvas = background.copy()
    width = canvas.shape[1]
    n = max(line.num_frames, 1)
    assignments = np.asarray(line.assegnazioni)
    current = int(assignments[frame]) if 0 <= frame < assignments.size else -1
    cell = _cell(line, width, current)
    if cell is not None:
        x0, x1, y = cell
        # The extra pixel on each side is needed: the cell ends at `column(end + 1) - 1` while the
        # cursor on the last frame of the region is at `column(end)`; both can round to the same
        # value, leaving the cursor one pixel outside
        # (`test_il_cursore_non_esce_mai_dal_riquadro_acceso`).
        cv2.rectangle(
            canvas,
            (max(x0 - 1, 0), y - 1),
            (min(x1 + 1, width - 1), y + _CELL_THICKNESS + 1),
            COLORS["testo"],
            1,
        )
    # The same frame -> column map as `strip_background`.
    x = min(_column(frame, width, n), width - 1)
    cv2.line(canvas, (x, 18), (x, STRIP_H - 12), COLORS["testo"], 1)
    label = f"frame {frame}"
    wide = cv2.getTextSize(label, _FONT_MONO, 0.3, 1)[0][0]
    # Right of the cursor while there is room, left otherwise: a label cut by the edge no
    # longer says which frame is shown.
    x_text = x + 5 if x + 5 + wide < width - 70 else max(4, x - wide - 5)
    _text(canvas, label, (x_text, STRIP_H - 4), 0.3, COLORS["testo"])
    return canvas


def compose(
    body: np.ndarray, header_spec: Header, background: np.ndarray | None = None
) -> np.ndarray:
    """Lay out a frame: header, body, strip.

    :param fondo: strip already drawn for this frame, or ``None`` for movies without a
        timeline.
    :return: the complete BGR frame, as wide as ``body``.
    """
    parts = [header(body.shape[1], header_spec), body]
    if background is not None:
        parts.append(background)
    return np.concatenate(parts, axis=0)


def interpolation_of(metodo: str) -> int:
    """Return how the map of ``metodo`` is upscaled from the token grid to pixels.

    A single decision for every renderer. Maps carrying identifiers or regions are upscaled
    in blocks (interpolating a concept ID or an occluded volume edge would invent values);
    continuous maps are smoothed. The answer is given on the base method (a perturbed view
    of IG is still an IG attribution).

    :param metodo: map name, also in the ``pert__view__base`` form.
    :return: the ``cv2.INTER_*`` constant to pass to ``cv2.resize``.
    """
    base = map_kind(metodo)
    in_blocks = base == "vtcd_ids" or base.startswith(
        ("vtcd_concept_", "occlusion_", "motion_local_")
    )
    return cv2.INTER_NEAREST if in_blocks else cv2.INTER_LANCZOS4


#: Bounding-box colours, in order (the first goes to the largest box), from the
#: demonstrator palette.
BOX_PALETTE = (COLORS["allarme"], COLORS["calma"], COLORS["verita"], COLORS["tenue"])

#: Rectangle thickness: two pixels stay visible on 320x240 without eating the box area.
BOX_THICKNESS = 2


def draw_boxes(canvas: np.ndarray, caselle, *, etichette: bool = True) -> np.ndarray:
    """Draw the bounding boxes of a temporal plane on the frame, in place.

    The only author of box drawing, shared by the box movies
    (:mod:`apps.demo.explanation_boxes`) and the third panel of ``side_by_side_bbox``.

    :param canvas: BGR frame ``[H, W, 3]``; box coordinates are scaled to its width and
        height (legitimate because ``edp_v2`` uses ``squash``, ADR-072).
    :param caselle: the plane boxes, already sorted by decreasing area.
    :param etichette: whether to write the box number above each box.
    :return: the same ``canvas``.
    """
    height, width = canvas.shape[:2]
    for index, box in enumerate(caselle):
        x0, y0, x1, y1 = in_pixel(box, width, height)
        color = BOX_PALETTE[index % len(BOX_PALETTE)]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, BOX_THICKNESS)
        if not etichette:
            continue
        # Only the box number, the same as the CSV `box` column (the token area is in the CSV and
        # in boxes.json).
        label = str(index + 1)
        # Above the box if there is room, inside otherwise, and shifted from the right edge as
        # needed to fit entirely.
        wide = cv2.getTextSize(label, _FONT_MONO, 0.35, 1)[0][0]
        y_text = y0 - 4 if y0 >= 14 else min(y1 - 3, height - 3)
        x_text = max(0, min(x0 + 1, width - wide - 1))
        _text(canvas, label, (x_text, y_text), 0.35, color)
    return canvas


#: Cuts with three side-by-side panels, three times the frame width.
#: ``side_by_side_bbox_annotati`` draws boxes only on annotated positive frames.
SIDE_BY_SIDE_CUTS = ("side_by_side", "side_by_side_bbox", "side_by_side_bbox_annotati")

#: Cuts drawing boxes (so they need ``caselle``), and among them those drawing only on
#: annotated positive frames.
BBOX_CUTS = ("side_by_side_bbox", "side_by_side_bbox_annotati")
ANNOTATED_ONLY_CUTS = ("side_by_side_bbox_annotati", "concept_bbox_annotati")

#: TWO-panel cuts: light overlay and bounding boxes, without the original video.
#: ``concept_bbox`` draws on every frame; ``concept_bbox_annotati`` on the whole video but
#: only on annotated positive frames (elsewhere both panels show the clean frame).
#: Produced only for the methods named in ``coppie`` (:func:`render_campaign`).
PAIR_CUTS = ("concept_bbox", "concept_bbox_annotati")


@dataclass(frozen=True)
class CompositeCampaign:
    """One of the campaigns of a per-class composed movie (:func:`render_campaign`).

    A video with events of several classes has one campaign per VTCD dictionary
    (``<video>__<Class>``), with the same windows and scores; the composed movie takes every
    frame from the campaign of its event class (:func:`classes_per_frame`).

    :param folder: the campaign, of the same video and with the same windows as the others.
    :param classe: the annotated class whose frames take their maps from here.
    :param metodi: movie role -> campaign method, also a union; roles are the movie method
        names and must be the same in every campaign of the composition.
    :param unioni: derived methods of the campaign, ``name -> parts``.
    :param box: box label on the frames of this class.
    """

    folder: Path
    classe: str
    metodi: Mapping[str, str]
    unioni: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    box: str = ""


@dataclass(frozen=True)
class _Copies:
    """The windows of a composed movie, repeated once per campaign.

    Window ``w`` of copy ``s`` has index ``s * windows + w``: maps, states and boxes come from
    the copy; header, cell and ground truth from the real window.
    """

    finestre: int
    titoli: tuple[Mapping[str, str], ...]
    famiglie: tuple[str, ...]
    box: tuple[str, ...]


#: Opacity of the light overlay: colour only where the concept is, the rest of the frame
#: is not darkened. On ``vtcd_ids`` (a concept in every cell) the tint is lighter.
CONCEPT_LIGHT_ALPHA = 0.5
ID_LIGHT_ALPHA = 0.38


def light_overlay(
    frame: np.ndarray, piano: np.ndarray, method: str, concept_color, palette_id
) -> np.ndarray:
    """Draw a VTCD concept on the frame without darkening it: tint inside the mask and border.

    Unlike :func:`blend_map`, the frame stays intact outside the concept. The map is upscaled
    in blocks (``INTER_NEAREST``): it is categorical.

    :param frame: BGR frame ``[H, W, 3]`` uint8.
    :param piano: the raw map of the temporal plane, ``[H', W']`` (0/1 for
        ``vtcd_concept_KK``, the ID for ``vtcd_ids``).
    :param method: ``vtcd_concept_KK`` or ``vtcd_ids``.
    :param concept_color: BGR tint of a single concept.
    :param palette_id: ``[K, 3]`` BGR, one colour per ID (the legend ones).
    :return: a new uint8 BGR frame.
    :raises ValueError: if ``method`` is not a VTCD concept map.
    """
    height, width = frame.shape[:2]
    large = cv2.resize(
        np.asarray(piano, np.float32), (width, height), interpolation=cv2.INTER_NEAREST
    )
    out = frame.astype(np.float32)
    base = map_kind(method)
    if base == "vtcd_ids":
        ids = np.clip(np.rint(large).astype(int), 0, len(palette_id) - 1)
        out = out * (1 - ID_LIGHT_ALPHA) + np.asarray(palette_id, np.float32)[ids] * ID_LIGHT_ALPHA
        border = np.zeros(ids.shape, bool)
        border[:, 1:] |= ids[:, 1:] != ids[:, :-1]
        border[1:, :] |= ids[1:, :] != ids[:-1, :]
        out[border] = out[border] * 0.3 + 255 * 0.7
        return out.astype(np.uint8)
    if not base.startswith("vtcd_concept_"):
        raise ValueError(f"Light overlay only for VTCD concept maps: {method}")
    mask = large > 0.5
    color = np.asarray(concept_color, np.float32)
    out[mask] = out[mask] * (1 - CONCEPT_LIGHT_ALPHA) + color * CONCEPT_LIGHT_ALPHA
    out = out.astype(np.uint8)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, tuple(int(c) for c in color), 1)
    return out


def panel_sizes(width: int, height: int, altezza_max: int | None = None) -> tuple[int, int]:
    """Return the panel size of the movies: native, or reduced under a cap.

    Panels follow the video resolution: the three-panel side-by-side view is 960 px wide on
    UCF-Crime (320x240) and 5760 on a 1920x1080 MIVIA-Crime video, beyond what many browser
    decoders accept. The cap scales proportionally; masks and boxes are redrawn on the new
    size, exported coordinates stay native.

    :param larghezza: native width in pixels.
    :param height: native height in pixels.
    :param altezza_max: maximum height, or ``None`` for no cap.
    :return: ``(width, height)``, both even (H.264 requirement).
    :raises ValueError: on a non-positive cap.
    """
    if altezza_max is not None:
        if altezza_max < 2:
            raise ValueError(f"Invalid maximum height: {altezza_max}")
        if height > altezza_max:
            width = max(2, round(width * altezza_max / height))
            height = altezza_max
    return width + width % 2, height + height % 2


def _render_videos(
    meta,
    records,
    windows,
    assignments,
    colors,
    scales,
    states,
    legends,
    tn_notes,
    out,
    tagli,
    line,
    caselle=None,
    coppie=None,
    classe_annotati=None,
    copies: _Copies | None = None,
    altezza_max: int | None = None,
):
    """Write one movie per method and cut, with the common header.

    :param tagli: which cuts to produce: ``side_by_side`` (three panels: clean video,
        overlay, map), ``overlay`` (the overlay alone, full width) and ``map`` (the map
        alone, telling whether a pale map is weak or just covered). The middle panel of
        ``side_by_side`` is the same frame as the ``overlay`` cut. ``side_by_side_bbox``
        replaces the map with the frame showing the mask bounding boxes.
    :param linea: the video timeline, shared by all cuts.
    :param caselle: per method, ``(source label, boxes per window and plane)``; needed by
        ``side_by_side_bbox`` and the pair cuts.
    :param coppie: for the methods also receiving the :data:`PAIR_CUTS`,
        ``method -> (raw maps per window, BGR concept colour, BGR ID palette)``.
    :param classe_annotati: for the :data:`ANNOTATED_ONLY_CUTS`, the only annotated class
        whose intervals receive boxes (:func:`annotated_intervals`); ``None`` for all.
    :param copies: in a per-class composed movie, the windows repeated per campaign
        (:class:`_Copies`); ``None`` outside the composition.
    :param altezza_max: maximum panel height in pixels; taller videos are scaled
        proportionally (even sides). ``None``: native resolution.
    :return: written paths and manifest rows, after decoding every exported file.
    """
    coppie = coppie or {}
    segments = covered_segments(assignments)
    full = segments == [(0, meta["num_frames"] - 1)]
    artifacts, descriptions = [], []
    capture = cv2.VideoCapture(meta["video"])
    if not capture.isOpened():
        raise RuntimeError(f"Unreadable video: {meta['video']}")
    width, height = panel_sizes(
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        altezza_max,
    )
    threshold = meta.get("threshold")
    backgrounds = {
        width_px: strip_background(width_px, line)
        for width_px in {
            width * 3
            if any(t.removesuffix("_plain") in SIDE_BY_SIDE_CUTS for t in tagli)
            else width,
            width * 2
            if coppie and any(t.removesuffix("_plain") in PAIR_CUTS for t in tagli)
            else width,
            width,
        }
    }
    # Annotated positive frames: they decide where the `_annotati` cuts draw.
    annotated = frame_annotati(annotated_intervals(line, classe_annotati), meta["num_frames"])
    # With a class, a frame of an event of another class is annotated positive: the header says
    # why there is no box there.
    outside_event = (
        f"outside the {classe_annotati} events"
        if classe_annotati
        else "frame not annotated positive"
    )
    try:
        for segment, (start, end) in enumerate(segments):
            suffix = "" if full else f"_clip_{segment:03d}_f{start:06d}-{end:06d}"
            writers = []
            expected_black = {}
            try:
                for method in colors:
                    for kind in tagli:
                        base_kind = kind.removesuffix("_plain")
                        if base_kind not in (*SIDE_BY_SIDE_CUTS, *PAIR_CUTS, "overlay", "map"):
                            raise ValueError(f"Unknown video cut: {kind}")
                        if base_kind in PAIR_CUTS and method not in coppie:
                            continue
                        w = (
                            width * 3
                            if base_kind in SIDE_BY_SIDE_CUTS
                            else width * 2
                            if base_kind in PAIR_CUTS
                            else width
                        )
                        plain = kind.endswith("_plain")
                        target = out / "video" / f"{method}{suffix}_{kind}.mp4"
                        writer = _VideoWriter(
                            target,
                            meta["fps"],
                            (w, height if plain else height + header_height(w) + STRIP_H),
                        )
                        writers.append((method, kind, target, writer))
                        expected_black[method, kind] = set()
                        if not writer.isOpened():
                            raise RuntimeError(f"Cannot create {target}")
                # One seek per segment; all methods share the decoding.
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, start):
                    raise RuntimeError(f"Seek to frame {start} not supported")
                cursors = dict.fromkeys(backgrounds)  # recomputed at every frame
                for frame_id in range(start, end + 1):
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(f"Decoding stopped at frame {frame_id}, expected {end}")
                    frame = cv2.resize(frame, (width, height))
                    index = int(assignments[frame_id])
                    if index < 0:
                        raise AssertionError("An exported segment contains a frame without a map")
                    for width_px, background in backgrounds.items():
                        cursors[width_px] = strip(background, frame_id, line)
                    # In a composed movie `index` is the copy of the window in the campaign of the frame class
                    # (`_Copies`).
                    real = index % copies.finestre if copies else index
                    copy = index // copies.finestre if copies else 0
                    for method in colors:
                        plane = offline_plane(windows[index], frame_id, len(colors[method][index]))
                        heat = cv2.resize(
                            colors[method][index][plane],
                            (width, height),
                            interpolation=interpolation_of(method),
                        )
                        unit = (
                            "volume t"
                            if method.startswith(("occlusion_", "motion_local_"))
                            else "tubelet"
                        )
                        warning = states[method][index] or ""
                        note = tn_notes.get(method) or ""
                        class_name = class_of(records[index], method).lstrip(" ·")
                        # Class of the annotated event under the cursor, next to the window.
                        event = line.event_at(frame_id)
                        header_spec = Header(
                            metodo=copies.titoli[copy][method] if copies else method_label(method),
                            famiglia=(
                                copies.famiglie[copy]
                                if copies
                                else vtcd_family(meta)
                                if map_kind(method).startswith("vtcd_")
                                else family_of(method)
                            ),
                            modello=model_label(meta["run_id"]),
                            video=Path(meta["video"]).stem,
                            dettaglio=(
                                (f"{class_name} · " if class_name else "")
                                + f"window {real + 1}/{copies.finestre if copies else len(records)} · "
                                + (f"event {event} · " if event else "")
                                + f"{unit} {plane} · "
                                f"{'whole video' if full else f'clip {start}-{end}'}"
                            ),
                            significato=meaning_of(method),
                            score=float(records[index]["score"]),
                            soglia=threshold,
                            chunk_positivo=line.label_of(real),
                            # The TN reference provenance is not a diagnostic: it stays at the bottom, in grey.
                            avviso=warning,
                            extra=note,
                            legenda=legends.get(method),
                            estremi=scales[method].get("legend_range"),
                            etichette_legenda=tuple(scales[method].get("legend_labels", ())),
                        )
                        # The map OVER the frame, computed once and used by two cuts out of three.
                        overlaid, heat = blend_map(frame, heat)
                        if method in coppie:
                            raw, concept_color, palette_id = coppie[method]
                            light = light_overlay(
                                frame, raw[index][plane], method, concept_color, palette_id
                            )
                        for m, kind, _, writer in writers:
                            if m != method:
                                continue
                            cut_header = header_spec
                            base_kind = kind.removesuffix("_plain")
                            if base_kind == "side_by_side":
                                body_html = np.concatenate((frame, overlaid, heat), axis=1)
                            elif base_kind in BBOX_CUTS:
                                box_label, planes = caselle[method]
                                box_label = copies.box[copy] if copies else box_label
                                outside = (
                                    base_kind in ANNOTATED_ONLY_CUTS and not annotated[frame_id]
                                )
                                of_plane = [] if outside else planes[index][plane]
                                body_html = np.concatenate(
                                    (
                                        frame,
                                        overlaid,
                                        draw_boxes(frame.copy(), of_plane),
                                    ),
                                    axis=1,
                                )
                                if not warning:
                                    cut_header = replace(
                                        header_spec,
                                        extra=f"{outside_event}: no box"
                                        if outside
                                        else f"box: {box_label} - {len(of_plane)} box",
                                    )
                            elif base_kind in PAIR_CUTS:
                                box_label, planes = caselle[method]
                                box_label = copies.box[copy] if copies else box_label
                                of_plane = planes[index][plane]
                                if base_kind == "concept_bbox_annotati" and not annotated[frame_id]:
                                    # Outside the annotated event nothing is proposed: the clean frame in both panels.
                                    body_html = np.concatenate((frame, frame), axis=1)
                                    extra = f"{outside_event}: no map and no box"
                                else:
                                    body_html = np.concatenate(
                                        (light, draw_boxes(frame.copy(), of_plane)), axis=1
                                    )
                                    extra = f"box: {box_label} - {len(of_plane)} box"
                                if not warning:
                                    cut_header = replace(header_spec, extra=extra)
                            elif base_kind == "overlay":
                                body_html = overlaid
                            else:
                                body_html = heat
                            # H.264 may quantize an almost-black fade to black (Shooting004: maximum 6/255 in input);
                            # the margin is limited to 8/255 and recorded.
                            if np.max(body_html) <= 8:
                                expected_black[method, kind].add(frame_id - start)
                            writer.write(
                                body_html
                                if kind.endswith("_plain")
                                else compose(body_html, cut_header, cursors[body_html.shape[1]])
                            )
            finally:
                for _, _, _, writer in writers:
                    writer.release()
            for method, kind, target, _ in writers:
                # Verify the actual decoding of every file, not just the container.
                check = cv2.VideoCapture(str(target))
                decoded = black = unexpected_black = 0
                while True:
                    ok, frame = check.read()
                    if not ok:
                        break
                    decoded += 1
                    top = 0 if kind.endswith("_plain") else header_height(frame.shape[1])
                    body = frame[top : top + height]
                    is_black = np.max(body) <= 3
                    black += int(is_black)
                    unexpected_black += int(
                        is_black and decoded - 1 not in expected_black[method, kind]
                    )
                check.release()
                if decoded != end - start + 1 or unexpected_black:
                    raise RuntimeError(
                        f"Invalid rendering {target}: {decoded} frames, {black} black, "
                        f"{unexpected_black} black frames not present before encoding"
                    )
                artifacts.append(target)
                descriptions.append(
                    {
                        "file": target.relative_to(out).as_posix(),
                        "method": method,
                        "kind": kind,
                        "interface": not kind.endswith("_plain"),
                        "start_frame": start,
                        "end_frame": end,
                        "frames": decoded,
                        "black_frames": black,
                        "expected_black_frames": sorted(expected_black[method, kind]),
                        "expected_black_max_pixel": 8,
                        "unexpected_black_frames": unexpected_black,
                        "coverage": "full_video" if full else "clip",
                        **(
                            {"box": caselle[method][0]}
                            if kind.removesuffix("_plain") in (*BBOX_CUTS, *PAIR_CUTS)
                            else {}
                        ),
                        **(
                            {"solo_annotati": True}
                            if kind.removesuffix("_plain") in ANNOTATED_ONLY_CUTS
                            else {}
                        ),
                        **(
                            {"classe_annotati": classe_annotati}
                            if classe_annotati
                            and kind.removesuffix("_plain") in ANNOTATED_ONLY_CUTS
                            else {}
                        ),
                    }
                )
    finally:
        capture.release()
    return artifacts, descriptions


def _composite_sources(
    folder: Path,
    meta: dict,
    composizione: Sequence[CompositeCampaign],
    selected_methods: tuple[str, ...] | None,
) -> list[tuple[Path, dict[str, str], dict[str, tuple[str, ...]]]]:
    """Return the campaigns of a composed movie, checked against the one of ``folder``.

    Windows are compared by identity and score, not by count: copy ``s`` of window ``w``
    takes its map from ``maps_{w}.npz`` of campaign ``s``.

    :return: ``(folder, role -> method, unions)`` per campaign, in the given order.
    :raises ValueError: if ``folder`` is not among the campaigns, roles or classes differ or
        repeat, ``selected_methods`` differs from the roles, or a campaign has a different
        video, model, windows or scores.
    """
    roles = list(composizione[0].metodi)
    if any(list(c.metodi) != roles for c in composizione):
        raise ValueError("Composed campaigns with different roles: a movie has one method per role")
    if selected_methods is not None and list(selected_methods) != roles:
        raise ValueError(f"In a composed movie the methods are the roles {roles}")
    classes = [c.classe for c in composizione]
    if len(set(classes)) != len(classes):
        raise ValueError(f"Repeated classes in the composition: {classes}")
    folders = [Path(c.folder).resolve() for c in composizione]
    if folder not in folders:
        raise ValueError("The movie campaign must be one of the composed ones")
    expected = [(r["window_id"], float(r["score"])) for r in meta["windows"]]
    sources = []
    for folder_path, campaign in zip(folders, composizione, strict=True):
        other = json.loads((folder_path / "campaign.json").read_text(encoding="utf-8"))
        if other["status"] != "complete":
            raise ValueError(f"Incomplete composed campaign: {folder_path}")
        different = [
            k for k in ("video", "run_id", "num_frames", "fps") if other.get(k) != meta.get(k)
        ]
        if different or [(r["window_id"], float(r["score"])) for r in other["windows"]] != expected:
            raise ValueError(
                f"{folder_path.name} is not the same video as {folder.name}: "
                f"{different or 'different windows or scores'}"
            )
        sources.append((folder_path, dict(campaign.metodi), dict(campaign.unioni)))
    return sources


def _composite_assignment(
    assignments: np.ndarray,
    line: Timeline,
    num_frames: int,
    composizione: Sequence[CompositeCampaign],
) -> tuple[np.ndarray, dict]:
    """Return which copy every frame of a composed movie comes from, and the receipt.

    :param assignments: the real window of every frame (:func:`assignment_of`).
    :return: ``(copy index * windows + window, receipt)``, ``-1`` where there is no window.
    :raises ValueError: if the composition classes differ from those annotated in the video.
    """
    classes, per_frame = classes_per_frame(line, num_frames)
    order = [c.classe for c in composizione]
    if set(order) != set(classes):
        raise ValueError(f"Composition for {sorted(order)}, annotated events of {list(classes)}")
    copy = np.asarray([order.index(c) for c in classes], dtype=np.int64)[per_frame]
    # The number of windows is the campaign one, from the timeline (one span per window).
    windows = len(line.spans)
    composite = np.where(assignments >= 0, assignments + windows * copy, -1)
    receipt = {
        "regola": (
            "every frame takes maps and boxes from the dictionary of its event class; "
            "outside events from that of the nearest event, without boxes"
        ),
        "campagne": [
            {
                "classe": c.classe,
                "campagna": Path(c.folder).name,
                "metodi": dict(c.metodi),
                "frame": int(np.sum(copy == s)),
            }
            for s, c in enumerate(composizione)
        ],
    }
    return composite.astype(assignments.dtype), receipt


def render_campaign(
    folder: Path,
    *,
    videos: bool = True,
    group_scales: dict | None = None,
    tagli: tuple[str, ...] | None = None,
    selected_methods: tuple[str, ...] | None = None,
    output_name: str = "render",
    diagnostic_images: bool | None = None,
    caselle: dict | None = None,
    coppie: tuple[str, ...] = (),
    classe_annotati: str | None = None,
    aggiungi_overlay: bool = True,
    unioni: dict[str, tuple[str, ...]] | None = None,
    composizione: Sequence[CompositeCampaign] = (),
    destinazione: Path | None = None,
    altezza_max: int | None = None,
) -> list[Path]:
    """Regenerate movies, tubelet panels and curves from the saved campaign alone.

    :param folder: directory with ``campaign.json`` and ``maps_*.npz``.
    :param videos: whether to also render the mp4 cuts per method.
    :param selected_methods: explicit subset, to render independent batches.
    :param output_name: publication subfolder; each batch uses its own.
    :param diagnostic_images: override for panels and curves; ``None`` keeps the saved
        policy.
    :param group_scales: per-method scale shared by several campaigns
        (``compute_group_scales``); missing methods fall back to the local scale.
    :param tagli: forces which cuts to produce, ignoring the saved ``render_policy`` (which
        lives in ``campaign.json``, the computation receipt). The overlay is still added by
        default. The ``_plain`` suffix exports content only, without header or timeline.
    :param caselle: per method, ``(label, boxes per window and plane)``
        (:func:`apps.demo.explanation_boxes.campaign_boxes`); required for every drawn method
        when ``tagli`` requests ``side_by_side_bbox``, and for every method of ``coppie``.
    :param coppie: VTCD methods that also receive the :data:`PAIR_CUTS` when ``tagli``
        requests them.
    :param classe_annotati: the ``*_annotati`` cuts draw boxes on this class's intervals only
        (per-class VTCD); ``None`` on all.
    :param aggiungi_overlay: ``False`` renders only the requested ``tagli``, without the
        default overlay (for a batch adding cuts to a page that already has it).
    :param unioni: derived maps ``name -> methods``, the per-window maximum of their maps
        (for merged VTCD concepts such as ``vtcd_concept_05+07``).
    :param composizione: a per-class composed movie: the campaigns of the video, one per
        annotated class, with ``folder`` among them. Every frame takes maps and boxes from the
        campaign of its event class (:func:`classes_per_frame`); movie methods are the roles
        of :attr:`CompositeCampaign.metodi`, and ``caselle`` carries for every role a list of
        ``len(composizione) * windows`` planes. Empty outside the composition.
    :param destinazione: folder of the composed movie; required with ``composizione`` and
        forbidden without.
    :param altezza_max: cap on the movie panel height (:func:`panel_sizes`); ``None`` keeps
        the native resolution.
    :return: produced artifacts, with scales recorded in ``render/scales.json``
        (``shared_across_campaigns`` says whether a scale comes from the group).
    :raises ValueError: on composition campaigns with different windows, scores or video,
        different roles, classes not covering those of the video, or panels and curves
        requested on a composed movie.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = folder.resolve()
    if Path(output_name).name != output_name or output_name in ("", ".", ".."):
        raise ValueError("The rendering name must be a single subfolder")
    meta = json.loads((folder / "campaign.json").read_text(encoding="utf-8"))
    if meta["status"] != "complete":
        raise ValueError("Incomplete campaign: rendering refused")
    records = meta["windows"]
    if not records:
        raise ValueError("No saved window to draw")
    windows = [
        WindowSpec(
            **{
                **r["window"],
                "explicit_frame_indices": tuple(r["window"]["explicit_frame_indices"]),
            }
        )
        for r in records
    ]
    if bool(composizione) != (destinazione is not None):
        raise ValueError("A composed movie requires its destination, and only it does")
    # Outside the composition there is one source (this campaign) and method names are its
    # own; inside, one per class and names are the roles.
    sources = [(folder, None, dict(unioni or {}))]
    if composizione:
        sources = _composite_sources(folder, meta, composizione, selected_methods)
        if diagnostic_images:
            raise ValueError("A composed movie has no panels or curves")
        diagnostic_images = False
    collections = []
    for folder_path, roles, source_unions in sources:
        for i in range(len(records)):
            with np.load(folder_path / f"maps_{i:05d}.npz") as archive:
                names = list(
                    roles.values() if roles is not None else selected_methods or archive.files
                )
                read_items = {m: archive[m] for m in names if m not in source_unions}
                for parts in source_unions.values():
                    read_items.update({m: archive[m] for m in parts if m not in read_items})
            for item_name, parts in source_unions.items():
                read_items[item_name] = np.maximum.reduce([read_items[m] for m in parts])
            if roles is not None:
                collections.append({role: read_items[m] for role, m in roles.items()})
                continue
            collections.append(
                {m: read_items[m] for m in [*names, *(u for u in source_unions if u not in names)]}
            )
    methods = list(collections[0])
    if any(set(c) != set(methods) for c in collections):
        raise ValueError("Methods not uniform across windows")
    if any(not np.isfinite(a).all() for c in collections for a in c.values()):
        raise ValueError("Non-finite maps: rendering refused")
    vtcd_count = int((meta.get("explanation_settings") or {}).get("vtcd", {}).get("n_concepts", 8))
    if "vtcd_ids" in methods and (
        vtcd_count < 1
        or any(np.any((c["vtcd_ids"] < 0) | (c["vtcd_ids"] >= vtcd_count)) for c in collections)
    ):
        raise ValueError("VTCD ID outside the declared dictionary")
    vtcd_colors = plt.get_cmap("tab10" if vtcd_count <= 10 else "tab20")(np.arange(vtcd_count))
    # These files are published for the user and inherit the campaign permissions (mkdtemp
    # imposes 0700). A composed movie is prepared next to its destination, on the same disk,
    # since publication is a rename.
    home = destinazione.resolve().parent if composizione else folder
    home.mkdir(parents=True, exist_ok=True)
    assignments, assignment = assignment_of(folder, meta, windows)
    # The timeline describes the video and its REAL windows: it is built before repeating the
    # windows for the composition.
    line = timeline_of(folder, meta, records, assignments)
    copies = None
    if composizione:
        assignments, assignment["composizione"] = _composite_assignment(
            assignments, line, meta["num_frames"], composizione
        )
        copies = _Copies(
            finestre=len(records),
            titoli=tuple(
                {role_name: method_label(m) for role_name, m in c.metodi.items()}
                for c in composizione
            ),
            famiglie=tuple(f"{vtcd_family(meta)} · dictionary {c.classe}" for c in composizione),
            box=tuple(c.box for c in composizione),
        )
        records = records * len(composizione)
        windows = windows * len(composizione)
    out = home / f".render_pending_{uuid.uuid4().hex}"
    out.mkdir()
    for name in ("video", "pannelli", "curve"):
        (out / name).mkdir()
    artifacts = []
    scales = {}
    colors, states, legends = {}, {}, {}
    # Provenance note of the TN reference: a video without its own TN window uses one from a
    # second, entirely normal video; flagged on ig_tn_same_video, and on ig_ensemble when it
    # includes it.
    tn_reference = meta.get("tn_reference") or {}
    tn_note = ""
    if tn_reference.get("source") == "fallback_normal_video":
        tn_note = f"TN baseline from an external entirely normal video: {tn_reference['video_id']}"
    elif tn_reference.get("source") == "same_video":
        tn_note = "TN baseline: a normal window predicted normal from this same video"
    ensemble_tn_source = (records[0].get("checks", {}).get("ig_ensemble", {}) or {}).get(
        "tn_source"
    )
    tn_notes = dict(meta.get("method_notes", {}))
    if tn_note and "ig_tn_same_video" in methods:
        tn_notes["ig_tn_same_video"] = tn_note
    if tn_note and ensemble_tn_source is not None and "ig_ensemble" in methods:
        tn_notes["ig_ensemble"] = tn_note
    # The policy travels with the campaign, not with the rendering script: whoever computed
    # the campaign knows which treatment it needs.
    policy = (meta.get("explanation_settings") or {}).get("render_policy") or {}
    with_panels = bool(policy.get("pannelli", True))
    with_curves = bool(policy.get("curve", True))
    if diagnostic_images is not None:
        with_panels = with_curves = diagnostic_images
    tagli = tuple(tagli) if tagli else tuple(policy.get("tagli", ("side_by_side", "map")))
    # The overlay is ALWAYS added, even where the policy does not name it (policies saved
    # before this cut existed do not reject it). Excluding it is declared in `render_policy`.
    if "overlay" not in tagli and aggiungi_overlay:
        tagli = (*tagli, "overlay")
    # Perturbed views describe the PERTURBED clip: drawing them over the original video would
    # say something false. They are rendered by `render_perturbed_views`.
    drawable = [m for m in methods if not view_of(m)]
    if any(t.removesuffix("_plain") in BBOX_CUTS for t in tagli):
        # A method without boxes does not fall back on the map: the movie would be called «with
        # bbox» and show something else.
        without = sorted(set(drawable) - set(caselle or {}))
        if without:
            raise ValueError(f"Side-by-side with boxes but no bounding boxes for: {without}")
        for method in drawable:
            if len(caselle[method][1]) != len(records):
                raise ValueError(f"Boxes of {method}: one list per window, {len(records)} expected")
    for method in drawable:
        maps = [c[method] for c in collections]
        signed = _is_signed(method)
        base = map_kind(method)
        shared = group_scales is not None and base in group_scales
        if shared:
            lo = np.asarray(group_scales[base]["lo"])
            hi = np.asarray(group_scales[base]["hi"])
            if lo.ndim == 0:
                lo, hi = float(lo), float(hi)
        else:
            all_values = np.concatenate(
                [m.reshape(-1, 3) if method == "pca_rgb" else m.ravel() for m in maps]
            )
            lo, hi = _scale_bounds(method, all_values, signed)
        scales[method] = {
            "lo": np.asarray(lo).tolist(),
            "hi": np.asarray(hi).tolist(),
            "per_frame_normalization": False,
            "colormap": "coolwarm_transparent"
            if transparent_center(method)
            else _cmap_name(method, signed),
            "alpha": "abs(2 * normalized - 1)" if transparent_center(method) else "constant",
            "constant_plane_display": "neutral gray, numerical values unchanged",
            "shared_across_campaigns": shared,
        }
        low_text = np.array2string(np.asarray(lo), formatter={"float_kind": lambda v: f"{v:.3g}"})
        high_text = np.array2string(np.asarray(hi), formatter={"float_kind": lambda v: f"{v:.3g}"})
        scales[method]["label"] = f"{low_text} .. {high_text}"
        if method == "pca_rgb":
            legends[method] = None
        else:
            scales[method]["legend_range"] = (low_text, high_text)
            gradient = np.linspace(0, 1, 160)
            if map_kind(method) == "vtcd_ids":
                palette = (vtcd_colors[:, :3] * 255).astype(np.uint8)[:, ::-1]
                bar = np.repeat(palette, 32, axis=0)
            else:
                rgba = color_map(method, signed)(gradient)
                rgb = rgba[:, :3] * rgba[:, 3:4] + np.asarray(COLORS["fondo"])[::-1] / 255 * (
                    1 - rgba[:, 3:4]
                )
                bar = (rgb * 255).astype(np.uint8)[:, ::-1]
            legends[method] = np.tile(bar[None], (_LEGEND_H - 4, 1, 1))
            if map_kind(method) == "vtcd_ids":
                # IDs are written in `header`, on the widened bar.
                scales[method]["legend_labels"] = [str(k) for k in range(vtcd_count)]

        def colored(plane, lo=lo, hi=hi, method=method, signed=signed):
            if map_kind(method) == "vtcd_ids":
                if np.all(plane < 0):
                    return np.full((*plane.shape[:2], 3), 160, dtype=np.uint8)
                return (
                    (vtcd_colors[plane.astype(int)][..., :3] * 255)
                    .astype(np.uint8)[..., ::-1]
                    .copy()
                )
            if np.ptp(plane) == 0 and not transparent_center(method):
                return np.full((*plane.shape[:2], 3), 160, dtype=np.uint8)
            normalized = np.clip((plane - lo) / np.maximum(np.asarray(hi) - lo, 1e-12), 0, 1)
            if transparent_center(method) and np.all(np.asarray(hi) == lo):
                normalized = np.full_like(plane, 0.5)
            if method == "pca_rgb":
                return (normalized * 255).astype(np.uint8)[..., ::-1].copy()
            cmap = color_map(method, signed)
            rgba = (cmap(normalized) * 255).astype(np.uint8)
            if transparent_center(method):
                return rgba[..., [2, 1, 0, 3]].copy()
            return rgba[..., [2, 1, 0]].copy()

        colors[method] = [
            np.zeros((*values.shape, 4), dtype=np.uint8)
            if "__class_" in method and records[i].get("label") == 0
            else np.stack([colored(p) for p in values])
            for i, values in enumerate(maps)
        ]
        states[method] = [_state(values, records[i], method) for i, values in enumerate(maps)]
        if method.startswith("anomaly_"):
            states[method] = [
                (s + " | " if s else "") + "DERIVATIVE: numerical check failed"
                if records[i]
                .get("checks", {})
                .get("anomaly_direction", {})
                .get("finite_difference_pass")
                is False
                else s
                for i, s in enumerate(states[method])
            ]
        for i, values in enumerate(maps):
            if with_panels:
                t = values.shape[0]
                cols = min(4, t)
                fig, axes = plt.subplots(
                    math.ceil(t / cols),
                    cols,
                    figsize=(3 * cols, 2.6 * math.ceil(t / cols)),
                    squeeze=False,
                )
                for j, ax in enumerate(axes.flat):
                    ax.axis("off")
                    if j < t:
                        picture = colored(values[j])
                        ax.imshow(
                            picture[..., [2, 1, 0, 3]]
                            if picture.shape[-1] == 4
                            else picture[..., ::-1]
                        )
                        per_plane = len(records[i]["frame_indices"]) // t
                        indices = records[i]["frame_indices"][j * per_plane : (j + 1) * per_plane]
                        unit = (
                            "volume t"
                            if method.startswith(("occlusion_", "motion_local_"))
                            else "tubelet"
                        )
                        ax.set_title(
                            f"{unit} {j}: frames {indices[0]}–{indices[-1]} ({len(indices)} samples)",
                            fontsize=8,
                        )
                fig.suptitle(
                    f"OFFLINE {method} | {records[i]['window_id']} | score {records[i]['score']:.4f}"
                    + (f"\n{states[method][i]}" if states[method][i] else "")
                    + (f"\n{tn_notes[method]}" if method in tn_notes else "")
                )
                if method == "pca_rgb":
                    fig.tight_layout()
                    fig.text(
                        0.5,
                        0.005,
                        f"Global scale: {low_text} .. {high_text}",
                        ha="center",
                        fontsize=8,
                    )
                else:
                    from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize

                    fig.tight_layout(rect=(0, 0.07, 1, 1))
                    bar_axis = fig.add_axes((0.22, 0.038, 0.56, 0.015))
                    if map_kind(method) == "vtcd_ids":
                        cmap = ListedColormap(vtcd_colors)
                        normalizer = BoundaryNorm(np.arange(vtcd_count + 1) - 0.5, vtcd_count)
                    else:
                        cmap = color_map(method, signed)
                        normalizer = Normalize(lo, hi if hi > lo else lo + 1e-12)
                    colorbar = fig.colorbar(
                        plt.cm.ScalarMappable(norm=normalizer, cmap=cmap),
                        cax=bar_axis,
                        orientation="horizontal",
                    )
                    if map_kind(method) == "vtcd_ids":
                        colorbar.set_ticks(np.arange(vtcd_count))
                    colorbar.set_label(
                        "concept ID"
                        if map_kind(method) == "vtcd_ids"
                        else f"Global scale: {low_text} .. {high_text}",
                        fontsize=8,
                    )
                    colorbar.ax.tick_params(labelsize=8)
                target = out / "pannelli" / f"{method}_clip_{i:05d}.png"
                fig.savefig(target, dpi=100)
                plt.close(fig)
                artifacts.append(target)
            if map_kind(method) != "pca_rgb" and with_curves:
                fig, ax = plt.subplots(figsize=(7, 2.8))
                if map_kind(method) == "vtcd_ids":
                    for k in range(vtcd_count):
                        ax.plot(
                            (values == k).mean((1, 2)),
                            color=vtcd_colors[k],
                            label=f"ID {k}",
                        )
                    ax.set_ylim(-0.03, 1.03)
                else:
                    ax.plot(np.maximum(values, 0).sum((1, 2)), label="sum of positive values")
                    ax.plot(np.minimum(values, 0).sum((1, 2)), label="sum of negative values")
                ax.set(
                    xlabel="Tubelet",
                    ylabel="Token fraction"
                    if map_kind(method) == "vtcd_ids"
                    else "Sum of the raw map",
                    title=f"{method} | {records[i]['window_id']}",
                )
                ax.legend(ncol=4 if map_kind(method) == "vtcd_ids" else 1, fontsize=8)
                fig.tight_layout()
                target = out / "curve" / f"{method}_temporal_{i:05d}.png"
                fig.savefig(target, dpi=100)
                plt.close(fig)
                artifacts.append(target)
    descriptions = []
    pair_data = {}
    if any(t.removesuffix("_plain") in PAIR_CUTS for t in tagli):
        # Pair cuts draw the RAW map (ID or 0/1 mask) and the boxes of the same method: without
        # either the movie would lie.
        extraneous = sorted(set(coppie) - set(drawable))
        without_boxes = sorted(set(coppie) - set(caselle or {}))
        if (
            not coppie
            or extraneous
            or without_boxes
            or any(not map_kind(m).startswith(("vtcd_concept_", "vtcd_ids")) for m in coppie)
        ):
            raise ValueError(
                f"Pair cuts: methods {list(coppie)}, not drawable {extraneous}, "
                f"without boxes {without_boxes} (VTCD maps with their boxes are required)"
            )
        palette_id = (vtcd_colors[:, :3] * 255).astype(np.float32)[:, ::-1]
        for method in coppie:
            tint = np.asarray(color_map(method, False)(1.0)[:3]) * 255
            pair_data[method] = ([c[method] for c in collections], tint[::-1], palette_id)
    if videos:
        video_files, descriptions = _render_videos(
            meta,
            records,
            windows,
            assignments,
            colors,
            scales,
            states,
            legends,
            tn_notes,
            out,
            tagli,
            line,
            caselle,
            pair_data,
            classe_annotati,
            copies,
            altezza_max,
        )
        artifacts.extend(video_files)
    (out / "scales.json").write_text(json.dumps(scales, indent=2), encoding="utf-8")
    manifest = {
        "renderer_schema": 4,
        "video_codec": "h264/yuv420p",
        "status": "complete",
        "source_frames": meta["num_frames"],
        "covered_frames": int(np.sum(assignments >= 0)),
        "segments": covered_segments(assignments),
        # Which window gave its plane to every frame: with two policies in circulation, a movie
        # that does not declare it cannot be compared with another (ADR-077).
        "assegnazione": assignment,
        # Methods rendered HERE: perturbed views have their own manifest.
        "methods": drawable,
        "perturbed_methods": [m for m in methods if view_of(m)],
        "map_states": states,
        "tn_reference": tn_reference or None,
        "videos": descriptions,
        "images": [p.relative_to(out).as_posix() for p in artifacts if p.suffix == ".png"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    target = destinazione.resolve() if composizione else folder / output_name
    previous = None
    if target.exists():
        previous = home / f".render_previous_{out.name.removeprefix('.render_pending_')}"
        os.replace(target, previous)
    os.replace(out, target)
    if previous is not None:
        if not previous.resolve().is_relative_to(home) or not previous.name.startswith(
            ".render_previous_"
        ):
            raise RuntimeError("Invalid rendering cleanup path")
        shutil.rmtree(previous)
    return [target / p.relative_to(out) for p in artifacts]
