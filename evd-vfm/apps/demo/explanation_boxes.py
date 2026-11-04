"""Movies with spatial-localization bounding boxes; no model.

A rectangle around the area highlighted by a VTCD concept, only on annotated positive
chunks and frames, presented as a possible automatic annotation technique; the same on
the highest-attribution tokens of ``ig_ensemble``.

Both variants show the same whole movie and differ only in where boxes are drawn:
``intero`` on every covered frame, ``annotati`` only on annotated positive frames (the
viewer must also see where the box does not appear).

The deliverable is the coordinates, not the movie: ``boxes.json`` carries every
rectangle in three systems (token grid, frame fractions, native pixels), plus a
per-frame CSV for the annotated variant.

Boxes are drawn at the native resolution, which is legitimate only because ``edp_v2``
preprocesses with ``squash`` (a resize without crop).

Two styles per movie, rendered in the same pass on the same frames and boxes:
``interfaccia`` (header, timeline and box numbers) and ``pulito`` (native frame and
rectangles only). ``vtcdK`` isolates concept ``K``, since VTCD IDs follow the K-means
order and differ between models (ADR-077); which window gives a frame its plane is
decided by the campaign (:func:`apps.demo.explanation.assignment_of`).
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
from collections.abc import Sequence
from pathlib import Path

import av
import cv2
import numpy as np

from apps.demo.explanation import (
    COLORS,
    STRIP_H,
    Header,
    _VideoWriter,
    annotated_intervals,
    assignment_of,
    compose,
    covered_segments,
    draw_boxes,
    header_height,
    model_label,
    strip,
    strip_background,
    timeline_of,
    windows_of,
)
from apps.demo.explanation_advanced import _previews_of, reuse_manifest
from evd.explain.boxes import (
    Box,
    MergeRule,
    boxes_from_mask,
    concept_mask,
    in_pixel,
    top_fraction_mask,
)
from evd.explain.project import frame_annotati, offline_plane
from evd.types import WindowSpec
from evd.utils.io import write_json, write_text_atomic

#: Mask sources with a fixed name: name -> (map in the npz, label, matching perturbed
#: view). The view is the same mask covered in black, so the box surrounds exactly the
#: area the ablation removes. Any other VTCD concept is requested as ``vtcdK``
#: (:func:`source_of`). The key has no ``map__`` prefix: names are bare inside
#: ``maps_NNNNN.npz``.
SOURCES = {
    "vtcd0": ("vtcd_ids", "VTCD concept 0", "vtcd_concept_00_black"),
    "ig_top10": ("ig_ensemble", "IG ensemble top 10%", "ig_top10_black"),
}

#: The two styles of every movie, in writing order.
STYLES = ("interfaccia", "pulito")

_VTCD = re.compile(r"vtcd(0|[1-9][0-9]*)")
_VTCD_UNION = re.compile(r"vtcd((?:0|[1-9][0-9]*)(?:\+(?:0|[1-9][0-9]*))+)")


def source_of(nome: str) -> tuple[str, str, str, int | tuple[int, ...] | None]:
    """Return map, label, matching view and concept of a mask source.

    ``vtcdK`` isolates concept ``K`` of the campaign dictionary; ``vtcdK+J`` treats several
    concepts as one (Shoplifting, ``vtcd5+7``). The name is canonical (``vtcd3``, not
    ``vtcd03``; ``vtcd5+7``, not ``vtcd7+5``) because it becomes the folder and movie name.

    :param nome: ``ig_top10``, ``vtcdK`` or ``vtcdK+J...``.
    :return: ``(map in the npz, label, view, concept_id or tuple of IDs or None)``.
    :raises ValueError: if the name is not a source, or the union is not canonical.
    """
    union = _VTCD_UNION.fullmatch(nome)
    if union:
        ids = tuple(int(k) for k in union[1].split("+"))
        if list(ids) != sorted(set(ids)):
            raise ValueError(f"Non-canonical union: {nome!r}; IDs must be increasing and distinct")
        text = "+".join(map(str, ids))
        view = "vtcd_concepts_" + "+".join(f"{k:02d}" for k in ids) + "_black"
        return "vtcd_ids", f"VTCD concepts {text}", view, ids
    matching = _VTCD.fullmatch(nome)
    if matching:
        k = int(matching[1])
        return "vtcd_ids", f"VTCD concept {k}", f"vtcd_concept_{k:02d}_black", k
    if nome in SOURCES:
        return (*SOURCES[nome], None)
    raise ValueError(
        f"Unknown source: {nome!r}; allowed 'vtcdK' (K = concept), 'vtcdK+J' and 'ig_top10'"
    )


#: Boxes entirely contained in a larger one on the same plane are removed (since 3
#: October 2026; movies rendered earlier do not change).
REMOVE_CONTAINED = True

#: Fraction of tokens selected by ``ig_top10``: the first non-trivial fraction already
#: declared in ``configs/explain/campaign.yaml`` for deletion/insertion.
IG_FRACTION = 0.10


def _windows(records: list[dict]) -> list[WindowSpec]:
    """:return: the campaign windows, in the form produced by the sampler."""
    return windows_of(records)


def _mask(archivio, sorgente: str) -> tuple[np.ndarray, dict]:
    """Return the token mask of a window, with the receipt of the rule.

    :param archivio: the already open ``maps_NNNNN.npz``.
    :param sorgente: a name accepted by :func:`source_of`.
    :return: ``(boolean mask [T,H,W], receipt)``.
    :raises KeyError: if the map required by the source is not in this campaign.
    """
    bare, _, _, concept = source_of(sorgente)
    # The prefixed form is accepted too: the same array lives bare inside `maps_NNNNN.npz` and
    # with `map__` inside the per-method artifact.
    key = next((c for c in (bare, f"map__{bare}") if c in archivio.files), None)
    if key is None:
        raise KeyError(
            f"The campaign does not carry '{bare}', required by '{sorgente}': "
            f"maps present {sorted(archivio.files)}"
        )
    values = archivio[key]
    if concept is not None:
        # IDs are integers saved as float32: rounding is safer than comparing with `== 0.0`.
        mask = concept_mask(np.rint(values).astype(np.int64), concept_id=concept)
        return mask, {
            "regola": "VTCD assignment, no threshold",
            "concept_id": concept,
            "token_selezionati": int(mask.sum()),
            "total_tokens": int(mask.size),
            "coverage": float(mask.mean()),
        }
    return top_fraction_mask(values, frazione=IG_FRACTION)


def campaign_boxes(
    folder: Path,
    sorgenti: list[str],
    indices: list[int] | None = None,
    fusione: MergeRule | None = None,
    remove_contained: bool = REMOVE_CONTAINED,
) -> dict[str, tuple[list[list[list[Box]]], list[dict | None]]]:
    """Return the boxes of every source, per window and temporal plane.

    The only path from mask to boxes (8-connectivity, minimum area 1 token), shared by the
    box movies and the ``side_by_side_bbox`` comparison. Every ``maps_NNNNN.npz`` is opened
    once for all sources.

    :param folder: campaign carrying the maps.
    :param sorgenti: names accepted by :func:`source_of`.
    :param indici: windows to compute; the others stay empty with a ``None`` receipt. All
        if omitted.
    :param fusione: component merging (:class:`evd.explain.boxes.MergeRule`); ``None`` keeps
        one box per component.
    :param remove_contained: remove boxes contained in a larger one
        (:func:`evd.explain.boxes.without_contained`).
    :return: ``source -> (boxes[window][plane], receipts[window])``.
    :raises KeyError: if the campaign lacks the map required by a source.
    """
    n = len(json.loads((folder / "campaign.json").read_text(encoding="utf-8"))["windows"])
    result = {s: ([[] for _ in range(n)], [None] * n) for s in sorgenti}
    for index in range(n) if indices is None else indices:
        with np.load(folder / f"maps_{index:05d}.npz") as archive:
            for source in sorgenti:
                mask, receipt = _mask(archive, source)
                planes, receipts = result[source]
                planes[index] = boxes_from_mask(
                    mask,
                    connectivity=8,
                    min_area=1,
                    fusione=fusione,
                    remove_contained=remove_contained,
                )
                receipts[index] = receipt
    return result


def _annotated_assignment(
    assegnazioni: np.ndarray, line, num_frames: int, classe: str | None = None
) -> tuple[np.ndarray, dict]:
    """Return the frames to draw on: only those the annotation declares positive.

    Decides where a box appears, not what the movie shows (the movie always covers the
    whole video). Both conditions apply: annotated positive chunk and frame. The chunk
    filter should remove nothing (with any-intersection positivity a window containing an
    annotated frame is positive); its count goes into the receipt.

    :param assegnazioni: output of :func:`apps.demo.explanation.assignment_of`.
    :param linea: the campaign timeline, carrying intervals and labels.
    :param num_frames: native frames of the video.
    :param classe: the only annotated class whose intervals receive boxes (per-class VTCD),
        or ``None`` for all.
    :return: ``(restricted assignment, receipt)``; ``-1`` where nothing is drawn.
    :raises ValueError: if the ground-truth file is missing, or it lacks the classes when
        one is requested.
    """
    if line.verita is None:
        raise ValueError(
            "The 'annotati' variant requires rapporti/verita.json: without it, "
            "'annotated positive frame' is undefined and an empty movie "
            "would look like a video without events"
        )
    intervals = annotated_intervals(line, classe)
    inside = frame_annotati(intervals, num_frames)
    restricted = np.where(inside, assegnazioni, -1)
    discarded_per_chunk = 0
    for index in np.unique(restricted[restricted >= 0]):
        if line.label_of(int(index)) is not True:
            discarded_per_chunk += int(np.sum(restricted == index))
            restricted = np.where(restricted == index, -1, restricted)
    return restricted.astype(np.int32), {
        "intervalli": [[int(a), int(b)] for a, b in intervals],
        **({"classe": classe} if classe else {}),
        "frame_annotati": int(inside.sum()),
        "frame_disegnati": int(np.sum(restricted >= 0)),
        "frame_scartati_per_chunk_non_positivo": discarded_per_chunk,
    }


def _frame(
    preview: np.ndarray,
    width: int,
    height: int,
    caselle: list[Box],
    *,
    etichette: bool = True,
) -> np.ndarray:
    """Return the native frame with its rectangles.

    :param preview: preprocessed sample ``[384, 384, 3]`` BGR.
    :param larghezza: output width, even.
    :param height: output height, even.
    :param caselle: boxes of the current temporal plane, already sorted.
    :param etichette: whether to write the box number above each box (the ``pulito`` style
        does not; colour, thickness and position stay the same).
    :return: BGR frame ready for composition.
    """
    interpolation = (
        cv2.INTER_AREA if width * height < preview.shape[0] * preview.shape[1] else cv2.INTER_LINEAR
    )
    canvas = cv2.resize(preview, (width, height), interpolation=interpolation)
    return draw_boxes(canvas, caselle, etichette=etichette)


def _even(canvas: np.ndarray) -> np.ndarray:
    """:return: the canvas with even width and height, as ``yuv420p`` requires.

    At most one background row and column are added instead of resizing, which would shift
    every box with respect to the exported coordinates.
    """
    height, width = canvas.shape[:2]
    if not (height % 2 or width % 2):
        return canvas
    new = np.full((height + height % 2, width + width % 2, 3), COLORS["fondo"], dtype=np.uint8)
    new[:height, :width] = canvas
    return new


def render_bounding_box(
    folder: Path,
    *,
    sorgente: str,
    variante: str,
    anteprime: Path | None = None,
    native: tuple[int, int] | None = None,
    force: bool = False,
    classe: str | None = None,
) -> dict:
    """Render the bounding-box movies of a campaign, plus their coordinates.

    For every covered segment two movies are written, one per style of :data:`STYLES`:
    ``bbox__{source}__{variant}.mp4`` with header, timeline and labels, and
    ``..._pulito.mp4`` with the native frame and rectangles only.

    :param folder: campaign carrying the source maps and ``campaign.json``.
    :param sorgente: ``vtcdK`` (concept ``K``) or ``ig_top10`` (:func:`source_of`).
    :param variante: ``intero`` (boxes on all covered frames) or ``annotati`` (boxes on
        annotated positive frames only); the movie covers the whole video in both.
    :param anteprime: sibling campaign carrying ``preview_manifest.json``.
    :param nativo: ``(width, height)`` of the source video, from the inventory; without it
        the preprocessed geometry is kept and declared.
    :param force: also redo existing movies.
    :param classe: in the ``annotati`` variant, the only annotated class whose intervals
        receive boxes (per-class VTCD); written to ``boxes.json``.
    :return: the manifest, also written to disk.
    :raises ValueError: on unknown source or variant, a concept outside the campaign
        dictionary, or missing ground truth when needed.
    :raises KeyError: if the campaign lacks the map required by the source.
    """
    _, source_label, view, concept = source_of(sorgente)
    if variante not in ("intero", "annotati"):
        raise ValueError(f"Unknown variant: {variante!r}; allowed 'intero' and 'annotati'")
    if classe is not None and variante != "annotati":
        raise ValueError("The class restricts the annotated frames: it applies only to 'annotati'")
    meta = json.loads((folder / "campaign.json").read_text(encoding="utf-8"))
    _check_concept(folder, meta, sorgente)
    records = meta["windows"]
    windows = _windows(records)
    previews = _previews_of(folder, anteprime or folder)
    assignments, assignment = assignment_of(folder, meta, windows)
    line = timeline_of(folder, meta, records, assignments)
    side = int(meta["preprocess"]["size"])
    width, height = native or (side, side)
    # One folder per (source, variant) pair, each with its `manifest.json`, as expected by
    # `reuse_manifest` and the catalogue.
    target = folder / "bounding_box" / f"{sorgente}__{variante}"
    target.mkdir(parents=True, exist_ok=True)

    # `disegnabili` says where a box appears, `assegnazioni` what is shown: segments come from
    # the latter, so the annotated variant is a whole video, not a cut.
    if variante == "annotati":
        drawable, variant_detail = _annotated_assignment(
            assignments, line, meta["num_frames"], classe
        )
    else:
        drawable = assignments
        variant_detail = {"frame_disegnati": int(np.sum(assignments >= 0))}
    segments = covered_segments(assignments)

    # Masks are computed only for the windows that are drawn (a minority in the annotated
    # variant), which also keeps `boxes.json` honest.
    used = sorted({int(i) for i in np.unique(drawable) if i >= 0})
    planes_per_window, receipts = campaign_boxes(folder, [sorgente], used)[sorgente]
    whole = segments == [(0, meta["num_frames"] - 1)]

    result = {
        "status": "running",
        "videos": [],
        "images": [],
        "sorgente": sorgente,
        "etichetta": source_label,
        "variante": variante,
        "stili": list(STYLES),
        "vista_corrispondente": view,
        "assegnazione": assignment,
        "regola": {
            "connettivita": 8,
            "area_minima_token": 1,
            **({"frazione": IG_FRACTION} if concept is None else {"concept_id": concept}),
            # Since 3 October 2026; a manifest without this row kept them.
            "box_contenuti": "tolti" if REMOVE_CONTAINED else "tenuti",
        },
        **({"classe": classe} if classe else {}),
        "grid": list(records[0]["grid"]),
        "size": side,
        "geometry": meta["preprocess"]["geometry"],
        "nativo": {
            "width": width,
            "height": height,
            "fps": meta["fps"],
            "num_frames": meta["num_frames"],
            "sorgente": "inventario" if native else "preprocessed (no native resolution)",
        },
        "covered_frames": int(np.sum(assignments >= 0)),
        "source_frames": meta["num_frames"],
        "display": (
            "native-resolution frame with one rectangle per connected area "
            "of the mask; the rectangle is the outer edge of the tokens, not a detected object"
        ),
        **variant_detail,
    }
    if not segments or not used:
        # Declared instead of writing a movie without boxes, which would look like a method that
        # found nothing when the filter removed everything (the normal videos end here).
        result["status"] = "complete"
        result["skipped"] = (
            "no annotated interval covered by a window: on a normal video "
            "the 'annotati' variant has nothing to show"
            if variante == "annotati"
            else "no frame covered by a window"
        )
        write_json(target / "manifest.json", result)
        _remove_orphans(target, sorgente, variante, result)
        _update_index(target.parent, sorgente, variante, result)
        return result

    # A movie rendered with another assignment policy has another window's boxes on the same
    # frames: it is not reused, even without `force`.
    previous_manifest = target / "manifest.json"
    same_policy = (
        not previous_manifest.is_file()
        or json.loads(previous_manifest.read_text(encoding="utf-8"))
        .get("assegnazione", {"politica": "fine"})
        .get("politica")
        == assignment["politica"]
    )
    previous_items = reuse_manifest(target, force or not same_policy)
    background = strip_background(width, line)
    for number, (start, end) in enumerate(segments):
        suffix = "" if whole else f"_clip_{number:03d}"
        names = {style: movie_name(sorgente, variante, suffix, style) for style in STYLES}
        # The two styles are reused only TOGETHER: they come from the same pass.
        if all(name in previous_items for name in names.values()):
            result["videos"].extend(previous_items[name] for name in names.values())
            continue
        result["videos"].extend(
            _render(
                {style: target / item_name for style, item_name in names.items()},
                start=start,
                fine=end,
                meta=meta,
                records=records,
                finestre=windows,
                previews=previews,
                assegnazioni=assignments,
                drawable=drawable,
                planes_per_window=planes_per_window,
                receipts=receipts,
                line=line,
                background=background,
                width=width,
                height=height,
                source_label=source_label,
                vista=view,
                variante=variante,
                whole=whole,
                outside=f"outside the {classe} events"
                if classe
                else "frame not annotated positive",
            )
        )
    result["status"] = "complete"
    write_json(target / "manifest.json", result)
    _remove_orphans(target, sorgente, variante, result)
    _write_coordinates(target, result, records, planes_per_window, receipts, width, height)
    if variante == "annotati":
        _write_per_frame(target, records, windows, planes_per_window, drawable, width, height)
    _update_index(target.parent, sorgente, variante, result)
    return result


def _check_concept(folder: Path, meta: dict, sorgente: str) -> None:
    """Stop on a concept beyond the dictionary.

    It would give an empty mask everywhere, and a movie without boxes that looks like a
    concept that never appears.

    :raises ValueError: if the source names a concept the dictionary does not have.
    """
    concept = source_of(sorgente)[3]
    if concept is None:
        return
    vtcd = (meta.get("explanation_settings") or {}).get("vtcd") or {}
    concepts = vtcd.get("n_concepts")
    maximum = max(concept) if isinstance(concept, tuple) else concept
    if concepts is not None and maximum >= int(concepts):
        raise ValueError(
            f"Campaign {folder} has a dictionary of {concepts} concepts: '{sorgente}' does not exist"
        )


#: Box source name in a per-class composed movie: the concept changes with the event
#: class, and each ``boxes.json`` window says which.
COMPOSITE_SOURCE = "vtcd_classe"


def render_composite_bounding_box(
    composizione: Sequence[tuple[str, Path, str]],
    destinazione: Path,
    *,
    anteprime: Path,
    native: tuple[int, int] | None = None,
) -> dict:
    """Render the box movie of a video, with the dictionary of each event class.

    On every annotated frame the boxes come from the concept chosen for the class of its
    event, in that class dictionary. ``annotati`` variant, interface style only.

    Windows repeat once per class: window ``w`` seen from dictionary ``s`` has index
    ``s * windows + w`` in boxes, receipts and export (``boxes.json`` schema 2, ``classe``
    CSV column). The shown frame and its window stay the real ones.

    :param composizione: ``(class, campaign, source)`` per annotated class of the video;
        campaigns share windows and scores.
    :param destinazione: video folder, receiving ``bounding_box/``.
    :param anteprime: campaign carrying ``preview_manifest.json``.
    :param nativo: ``(width, height)`` of the video, from the inventory.
    :return: the manifest, also written to disk.
    :raises ValueError: if the classes differ from the annotated ones, campaigns belong to
        different videos, or a concept is outside a dictionary.
    """
    from apps.demo.explanation import (
        CompositeCampaign,
        _composite_assignment,
        _composite_sources,
    )

    if not composizione:
        raise ValueError("Empty composition: no class to draw")
    base = Path(composizione[0][1]).resolve()
    meta = json.loads((base / "campaign.json").read_text(encoding="utf-8"))
    campaigns = [
        CompositeCampaign(Path(folder), class_name, {"vtcd_ids": "vtcd_ids"}, box=source_of(s)[1])
        for class_name, folder, s in composizione
    ]
    _composite_sources(base, meta, campaigns, None)
    for _, folder, source in composizione:
        data = json.loads((Path(folder) / "campaign.json").read_text(encoding="utf-8"))
        _check_concept(Path(folder), data, source)
    records = meta["windows"]
    real = len(records)
    windows = _windows(records)
    previews = _previews_of(base, anteprime)
    assignments, assignment = assignment_of(base, meta, windows)
    line = timeline_of(base, meta, records, assignments)
    _, assignment["composizione"] = _composite_assignment(
        assignments, line, meta["num_frames"], campaigns
    )
    side = int(meta["preprocess"]["size"])
    width, height = native or (side, side)
    # For each class, the frames of ITS events and the boxes of ITS dictionary; events of
    # different classes do not overlap (checked by `classi_per_frame`).
    drawable = np.full(meta["num_frames"], -1, dtype=np.int64)
    planes_per_window: list[list[list[Box]]] = []
    receipts: list[dict | None] = []
    details = {}
    for copy, (class_name, folder, source) in enumerate(composizione):
        its_items, details[class_name] = _annotated_assignment(
            assignments, line, meta["num_frames"], class_name
        )
        if np.any((its_items >= 0) & (drawable >= 0)):
            raise ValueError(f"Events {class_name} overlap those of another class")
        drawable = np.where(its_items >= 0, its_items + copy * real, drawable)
        used = sorted({int(i) for i in np.unique(its_items) if i >= 0})
        planes, its_receipts = campaign_boxes(Path(folder), [source], used)[source]
        planes_per_window.extend(planes)
        receipts.extend(its_receipts)
    classes_copy = [class_name for class_name, _, _ in composizione for _ in range(real)]
    labels = [source_of(s)[1] for _, _, s in composizione for _ in range(real)]
    target = destinazione / "bounding_box" / f"{COMPOSITE_SOURCE}__annotati"
    target.mkdir(parents=True, exist_ok=True)
    segments = covered_segments(assignments)
    whole = segments == [(0, meta["num_frames"] - 1)]
    result = {
        "status": "running",
        "videos": [],
        "images": [],
        "sorgente": COMPOSITE_SOURCE,
        "etichetta": "concept chosen for the event class",
        "variante": "annotati",
        "stili": ["interfaccia"],
        "vista_corrispondente": None,
        "assegnazione": assignment,
        "regola": {
            "connettivita": 8,
            "area_minima_token": 1,
            "concept_id": {class_name: source_of(s)[3] for class_name, _, s in composizione},
            "box_contenuti": "tolti" if REMOVE_CONTAINED else "tenuti",
        },
        "classi": [class_name for class_name, _, _ in composizione],
        "sorgenti": {class_name: s for class_name, _, s in composizione},
        "campagne": {class_name: Path(c).name for class_name, c, _ in composizione},
        "grid": list(records[0]["grid"]),
        "size": side,
        "geometry": meta["preprocess"]["geometry"],
        "nativo": {
            "width": width,
            "height": height,
            "fps": meta["fps"],
            "num_frames": meta["num_frames"],
            "sorgente": "inventario" if native else "preprocessed (no native resolution)",
        },
        "covered_frames": int(np.sum(assignments >= 0)),
        "source_frames": meta["num_frames"],
        "display": (
            "native-resolution frame with one rectangle per connected area of the "
            "mask of the concept chosen for the event class; the rectangle is the outer "
            "edge of the tokens, not a detected object"
        ),
        "intervalli": [iv for d in details.values() for iv in d["intervalli"]],
        "frame_annotati": sum(d["frame_annotati"] for d in details.values()),
        "frame_disegnati": int(np.sum(drawable >= 0)),
        "frame_scartati_per_chunk_non_positivo": sum(
            d["frame_scartati_per_chunk_non_positivo"] for d in details.values()
        ),
        "per_classe": details,
    }
    if not segments or not np.any(drawable >= 0):
        result["status"] = "complete"
        result["skipped"] = "no annotated interval covered by a window"
        write_json(target / "manifest.json", result)
        _remove_orphans(target, COMPOSITE_SOURCE, "annotati", result)
        return result
    background = strip_background(width, line)
    for number, (start, end) in enumerate(segments):
        suffix = "" if whole else f"_clip_{number:03d}"
        name = movie_name(COMPOSITE_SOURCE, "annotati", suffix, "interfaccia")
        result["videos"].extend(
            _render(
                {"interfaccia": target / name},
                start=start,
                fine=end,
                meta=meta,
                records=records,
                finestre=windows,
                previews=previews,
                assegnazioni=assignments,
                drawable=drawable,
                planes_per_window=planes_per_window,
                receipts=receipts,
                line=line,
                background=background,
                width=width,
                height=height,
                source_label="concept of the event class",
                vista=COMPOSITE_SOURCE,
                variante="annotati",
                whole=whole,
                etichette=labels,
            )
        )
    result["status"] = "complete"
    write_json(target / "manifest.json", result)
    _remove_orphans(target, COMPOSITE_SOURCE, "annotati", result)
    copies = records * len(composizione)
    _write_coordinates(
        target, result, copies, planes_per_window, receipts, width, height, classes_copy
    )
    _write_per_frame(
        target,
        copies,
        windows * len(composizione),
        planes_per_window,
        drawable,
        width,
        height,
        classes_copy,
    )
    return result


def movie_name(sorgente: str, variante: str, suffisso: str, style: str) -> str:
    """:return: the movie name of a style; ``interfaccia`` keeps the historical name."""
    if style not in STYLES:
        raise ValueError(f"Unknown style: {style!r}; allowed {STYLES}")
    tail = "" if style == "interfaccia" else f"__{style}"
    return f"bbox__{sorgente}__{variante}{suffisso}{tail}.mp4"


def _remove_orphans(target: Path, sorgente: str, variante: str, risultato: dict) -> list[str]:
    """Remove the movies of this combination that the manifest no longer names.

    A superseded movie left on disk looks valid. Only names of this ``(source, variant)``
    are removed, after the new manifest is written.

    :return: the removed names.
    """
    alive = {row["file"] for row in risultato["videos"]}
    removed = []
    for path in sorted(target.glob(f"bbox__{sorgente}__{variante}*.mp4")):
        if path.name not in alive:
            path.unlink()
            removed.append(path.name)
    return removed


def _update_index(root: Path, sorgente: str, variante: str, risultato: dict) -> None:
    """Keep ``bounding_box/manifest.json`` (read by the catalogue) up to date.

    Rows of a combination are REPLACED, not appended (a second run with ``--force`` produces
    the same names). The ``base`` field is the source map (``vtcd_ids``, ``ig_ensemble``),
    so the box appears under the method it comes from.

    :param root: the campaign ``bounding_box/`` folder.
    :param sorgente: a name accepted by :func:`source_of`.
    :param variante: ``intero`` or ``annotati``.
    :param risultato: the manifest of the combination just written.
    """
    path = root / "manifest.json"
    index = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.is_file()
        else {"status": "complete", "videos": [], "images": []}
    )
    map_, label, _, _ = source_of(sorgente)
    others = [
        row
        for row in index.get("videos", [])
        if (row.get("sorgente"), row.get("variante")) != (sorgente, variante)
    ]
    new_items = [
        {
            **row,
            "base": map_,
            "sorgente": sorgente,
            "etichetta": label,
            "variante": variante,
            "file": f"{sorgente}__{variante}/{row['file']}",
        }
        for row in risultato["videos"]
    ]
    index["videos"] = sorted(
        others + new_items,
        key=lambda r: (
            r["base"],
            r.get("sorgente", ""),
            r["variante"],
            r.get("stile", "interfaccia"),
            r.get("start_frame", 0),
        ),
    )
    index["status"] = "complete"
    index["display"] = risultato["display"]
    write_json(path, index)


def _render(
    paths: dict[str, Path],
    *,
    start: int,
    fine: int,
    meta: dict,
    records: list[dict],
    finestre: list[WindowSpec],
    previews: list[np.ndarray],
    assegnazioni: np.ndarray,
    drawable: np.ndarray,
    planes_per_window: list[list[list[Box]]],
    # `dict | None`: the same sparse list as `render_bounding_box`, `None` on windows not
    # drawn.
    receipts: list[dict | None],
    line,
    background: np.ndarray,
    width: int,
    height: int,
    source_label: str,
    vista: str,
    variante: str,
    whole: bool,
    outside: str = "frame not annotated positive",
    etichette: Sequence[str] | None = None,
) -> list[dict]:
    """Write the movie of a segment in every style and return its rows.

    The frame comes from ``assegnazioni``, the box only where ``disegnabili`` is valid. A
    single pass for all styles, so they cannot show different boxes on the same frame.

    :param percorsi: style -> movie to write, styles of :data:`STYLES`.
    :param disegnabili: per frame, the index in ``planes_per_window`` of the boxes to draw,
        ``-1`` where nothing is drawn.
    :param etichette: box label per ``planes_per_window`` index, in a composed movie;
        ``None`` uses ``source_label`` everywhere.
    :return: one row per style, with codec and frame count read back from the file.
    :raises ValueError: on an unknown style, or a written movie lacking the expected frames
        or codec.
    """
    if set(paths) - set(STYLES):
        raise ValueError(f"Unknown styles: {sorted(set(paths) - set(STYLES))}; allowed {STYLES}")
    body_height = height + height % 2
    body_width = width + width % 2
    geometries = {
        "interfaccia": (
            body_width,
            body_height + header_height(body_width) + STRIP_H,
        ),
        "pulito": (body_width, body_height),
    }
    in_progress = {style: p.with_name(f".{p.stem}.pending.mp4") for style, p in paths.items()}
    writers = {
        style: _VideoWriter(in_progress[style], meta["fps"], geometries[style]) for style in paths
    }
    temporal_planes = int(records[0]["grid"][0])
    box_count = 0
    try:
        for frame in range(start, fine + 1):
            index = int(assegnazioni[frame])
            window = finestre[index]
            plane = offline_plane(window, frame, temporal_planes)
            drawn = int(drawable[frame])
            draw = drawn >= 0
            boxes = planes_per_window[drawn][plane] if draw else []
            box_count += len(boxes)
            sample = offline_plane(window, frame, len(previews[index]))
            preview = previews[index][sample]
            if "pulito" in writers:
                writers["pulito"].write(
                    _even(_frame(preview, width, height, boxes, etichette=False))
                )
            if "interfaccia" not in writers:
                continue
            body = _even(_frame(preview, width, height, boxes))
            receipt = receipts[drawn] if draw else None
            label = etichette[drawn] if draw and etichette else source_label
            event = line.event_at(frame)
            # Header rows are short on purpose: on a 320x240 video a full sentence is truncated.
            header_spec = Header(
                metodo=f"Box: {label}",
                famiglia="Proposed localization",
                modello=model_label(meta["run_id"]),
                video=Path(meta["video"]).stem,
                dettaglio=(
                    f"win. {index + 1}/{len(records)}"
                    + (f" - event {event}" if event else "")
                    + f" - plane {plane + 1}/{temporal_planes}"
                    + ("" if whole else f" - clip {start}-{fine}")
                ),
                significato="connected area of the map, not a detected object",
                score=float(records[index]["score"]),
                soglia=meta.get("threshold"),
                chunk_positivo=line.label_of(index),
                extra=(
                    f"{len(boxes)} box - {receipt['token_selezionati']} token"
                    f" ({receipt['coverage']:.1%}) - {variante}"
                    if draw
                    else f"{outside}: no box"
                ),
            )
            writers["interfaccia"].write(compose(body, header_spec, strip(background, frame, line)))
    finally:
        for writer in writers.values():
            writer.release()
    rows = []
    for style in STYLES:
        if style not in paths:
            continue
        with av.open(str(in_progress[style])) as container:
            stream = container.streams.video[0]
            codec, pixel_format = stream.codec_context.name, stream.codec_context.pix_fmt
            sizes = (stream.codec_context.width, stream.codec_context.height)
            count = sum(1 for _ in container.decode(video=0))
        if (
            count != fine - start + 1
            or codec != "h264"
            or pixel_format != "yuv420p"
            or sizes != geometries[style]
        ):
            raise ValueError(f"Invalid box movie: {in_progress[style]}")
        os.replace(in_progress[style], paths[style])
        rows.append(
            {
                "method": f"bbox__{vista}",
                "file": paths[style].name,
                "stile": style,
                "frames": count,
                "start_frame": start,
                "end_frame": fine,
                "codec": codec,
                "pixel_format": pixel_format,
                "box_disegnati": box_count,
            }
        )
    return rows


def _write_coordinates(
    target: Path,
    risultato: dict,
    records: list[dict],
    planes_per_window: list[list[list[Box]]],
    receipts: list[dict | None],
    width: int,
    height: int,
    classi: Sequence[str] | None = None,
) -> None:
    """Write the coordinates of every box, per window and temporal plane.

    The deliverable artifact. Tokens say where a box comes from, fractions make it
    resolution-independent, pixels make it drawable. Only drawn windows are listed. In a
    per-class composed movie a window can appear twice (one per dictionary), so the schema
    is 2, with ``classe`` on every window and key ``(window_id, classe)``.

    :param records: the windows, repeated per copy in a composed movie.
    :param classi: in a composed movie, the class of every ``records`` index.
    """
    windows = []
    for index, record in enumerate(records):
        if receipts[index] is None:
            continue
        planes = []
        for plane, boxes in enumerate(planes_per_window[index]):
            if not boxes:
                continue
            planes.append(
                {
                    "t": plane,
                    "box": [
                        {
                            "token": list(c.token),
                            "norm": [round(v, 6) for v in c.norm],
                            "nativo": list(in_pixel(c, width, height)),
                            "area_token": c.area_token,
                            "riempimento": round(c.riempimento, 4),
                        }
                        for c in boxes
                    ],
                }
            )
        windows.append(
            {
                "window_id": record["window_id"],
                **({"classe": classi[index]} if classi else {}),
                "score": float(record["score"]),
                "maschera": receipts[index],
                "piani": planes,
            }
        )
    # Temporal geometry of the REAL windows, once each and in time order: the annotator redoes
    # the plane assignment on it.
    real = len(records) // (len(set(classi)) if classi else 1)
    used = sorted({i % real for i, receipt in enumerate(receipts) if receipt is not None})
    geometries = _windows(records[:real])
    write_json(
        target / "boxes.json",
        {
            "schema": 2 if classi else 1,
            **{k: risultato[k] for k in ("sorgente", "variante", "regola", "grid", "size")},
            # Classes whose events carry boxes: the annotator does not ask for proposals on events of
            # another class.
            **(
                {"classi": list(risultato["classi"])}
                if risultato.get("classi")
                else {"classi": [risultato["classe"]]}
                if risultato.get("classe")
                else {}
            ),
            "geometry": risultato["geometry"],
            "nativo": risultato["nativo"],
            "finestre": windows,
            "temporal_layout": {
                "assignment": risultato["assegnazione"],
                "windows": [
                    {
                        "window_id": geometries[i].window_id,
                        "frame_indices": list(geometries[i].frame_indices),
                    }
                    for i in used
                ],
            },
        },
    )


def _write_per_frame(
    target: Path,
    records: list[dict],
    finestre: list[WindowSpec],
    planes_per_window: list[list[list[Box]]],
    effective: np.ndarray,
    width: int,
    height: int,
    classi: Sequence[str] | None = None,
) -> None:
    """Write one row per ``(frame, box)``: the form in which an annotation is used.

    Only for the ``annotati`` variant.

    :param classi: in a per-class composed movie, the class of every ``records`` index; adds
        the ``classe`` column, which with ``window_id`` selects the ``boxes.json`` window
        (schema 2).
    """
    video_id = records[0]["window"]["video_id"]
    temporal_planes = int(records[0]["grid"][0])
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "video_id",
            "frame",
            "x0",
            "y0",
            "x1",
            "y1",
            "x0n",
            "y0n",
            "x1n",
            "y1n",
            "area_token",
            "riempimento",
            "window_id",
            "piano",
            "score",
            "box",
            *(["classe"] if classi else []),
        ]
    )
    for frame in np.flatnonzero(effective >= 0).tolist():
        index = int(effective[frame])
        plane = offline_plane(finestre[index], frame, temporal_planes)
        for number, box in enumerate(planes_per_window[index][plane], start=1):
            x0, y0, x1, y1 = in_pixel(box, width, height)
            writer.writerow(
                [
                    video_id,
                    frame,
                    x0,
                    y0,
                    x1,
                    y1,
                    *(round(v, 6) for v in box.norm),
                    box.area_token,
                    round(box.riempimento, 4),
                    records[index]["window_id"],
                    plane,
                    round(float(records[index]["score"]), 6),
                    number,
                    *([classi[index]] if classi else []),
                ]
            )
    write_text_atomic(target / "boxes_per_frame.csv", buffer.getvalue())
