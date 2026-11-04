"""Local HTML catalogue of the explanations: coverage, videos, panels and limits."""

from __future__ import annotations

import html
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path

#: Titles, families and colour meanings live in ``apps.demo.explanation``, shared with the
#: movies.
from apps.demo.explanation import (  # noqa: E402
    METHOD_FAMILIES as FAMILIES,
)
from apps.demo.explanation import (
    VIEWS,
    method_concepts,
    method_label,
)
from evd.annotations.ucf_classes import UCF_ANOMALY_CLASSES

STYLE = """
body{font:16px/1.5 system-ui,sans-serif;color:#172737;background:#f4f6f8;margin:0}
/* The column stays NARROW: at 1560 px very long prose lines are hard to read, and
   limiting only paragraphs gives a page whose width changes between sections. Movies
   grow inside the column; the «Full width» checkbox enlarges them further. */
main{max-width:1280px;margin:32px auto;padding:0 24px 48px}h1{font-size:30px;line-height:1.2}
h2{font-size:23px;margin-top:36px}h3{font-size:18px}a{color:#075eaa;text-underline-offset:3px}
.card,details{background:white;border:1px solid #dce2e8;border-radius:10px;padding:20px;margin:16px 0}
.notice{border-left:5px solid #b56d00;background:#fff6e5;padding:14px 18px;margin:18px 0}
.ok{border-left:5px solid #237d50;background:#edf8f1;padding:14px 18px;margin:18px 0}
.muted{color:#566979}.links a{display:inline-block;margin:4px 16px 4px 0}
table{border-collapse:collapse;width:100%;background:white}td,th{text-align:left;vertical-align:top;padding:10px;border-bottom:1px solid #dce2e8}
th{background:#eaf0f5}summary{cursor:pointer;font-weight:650}input{padding:10px;border:1px solid #aab8c5;border-radius:6px;width:min(450px,90%)}
code{overflow-wrap:anywhere}.preview{max-width:100%;max-height:760px}
/* A movie carries a 142 px header and a 98 px strip around its body: a one-panel cut
   on a 320x240 video is 320x480 (taller than wide), a side-by-side cut 960x480. Both
   limits apply and the browser decides which one bites; the height cap is generous
   so the default side-by-side cut fills the column. On a one-panel cut the height
   always bites: `.piena` removes the cap and accepts scrolling. */
video{max-width:100%;max-height:80vh;width:auto;height:auto;
background:#ddd;margin:12px 0;display:block}
video.lettore{max-height:84vh}
video.lettore.piena{max-height:none;width:100%}
label.piena{display:inline-block;margin:10px 0 0;font-size:14px;color:#566979;cursor:pointer}
label.piena input{width:auto;padding:0;border:0;margin-right:6px;vertical-align:middle}
th{cursor:pointer;user-select:none}th[data-verso]{color:#075eaa}
/* The arrows have a DOUBLED backslash: `STYLE` is not a raw string, so `\\2191` written
   with a single backslash would be a Python octal escape. CSS must receive `\\2191`. */
th[data-verso="asc"]::after{content:" \\2191"}th[data-verso="desc"]::after{content:" \\2193"}
td .sotto{display:block;font-size:13px;color:#566979;margin-top:2px}
.griglia{display:grid;gap:5px;margin:10px 0 18px}
.cella{display:flex;flex-direction:column;justify-content:center;align-items:center;aspect-ratio:1;
border:1px solid #cbd4dd;border-radius:6px;text-decoration:none;color:#172737;background:#fff}
.cella:hover{outline:2px solid #075eaa}.cella.vuota{border-style:dashed;opacity:.4}
.cella .delta{font-weight:650;font-size:14px}.cella .tok{font-size:11px;color:#566979}
.cella[data-attivo]{outline:3px solid #075eaa}
a.gioca[data-attivo]{font-weight:700;text-decoration:none;background:#e3eefb;padding:1px 7px;border-radius:5px}
details h4{margin:22px 0 4px;font-size:16px;color:#0b3c61;border-top:1px solid #dce2e8;padding-top:16px}
details h4:first-of-type{border-top:2px solid #c3cfda}
details h5{margin:14px 0 4px;font-size:14px;color:#566979;font-weight:600}
td a{display:inline-block;margin:2px 0}td a+a{margin-left:14px}
/* A `details` inside a table cell: closed it is a single row, which is why the 45 masks
   and the 11 views are no longer sections. */
details.dentro{background:transparent;border:0;border-radius:0;padding:0;margin:2px 0}
details.dentro>summary{font-weight:500;color:#075eaa;text-decoration:underline;text-underline-offset:3px}
details.dentro[open]{display:block;margin:6px 0 10px}
/* The per-class menu sits inside the section of its family: indent and rule separate it
   from the binary variants above it. */
.class-picker{border-top:2px solid #c3cfda;margin-top:26px;padding-top:8px}
.class-picker>h3{margin:12px 0 4px;color:#0b3c61}
.class-picker h4{margin:18px 0 6px;font-size:16px;color:#566979}
.class-picker>label{display:inline-block;margin:6px 0 10px}
"""


#: A click on a movie opens it in the method player instead of a new page (about 180
#: movies per campaign). Works on the mask-grid cells and on every row of the method
#: table; without JavaScript the links still work.
_SCRIPT = """
document.addEventListener('click', function (event) {
  var cell = event.target.closest('a.cella, a.gioca');
  if (!cell) return;
  // Walk up to the `details` that HAS the player: the mask grid and the views sit
  // inside a `details` nested in a table cell, which contains none.
  var block = cell.closest('details');
  var player = null;
  while (block && !player) {
    player = block.querySelector('video.lettore');
    if (!player) block = block.parentElement && block.parentElement.closest('details');
  }
  if (!player) return;
  event.preventDefault();
  block.querySelectorAll('a[data-attivo]').forEach(function (a) {
    a.removeAttribute('data-attivo');
  });
  cell.setAttribute('data-attivo', '');
  player.src = cell.dataset.src;
  player.play().catch(function () {});
});

// «Full width» removes the height cap of the player. The checkbox sits right AFTER its
// video, so the previous sibling is used.
document.addEventListener('change', function (event) {
  var checkbox = event.target.closest('label.piena');
  if (!checkbox) return;
  var player = checkbox.previousElementSibling;
  if (player && player.matches('video.lettore')) {
    player.classList.toggle('piena', event.target.checked);
  }
});

// A click on a header sorts the table by that column. The key is `data-ord` when
// present (a number) and the cell text otherwise. Missing values always go last, in
// both directions: «undefined» is not a small number.
document.addEventListener('click', function (event) {
  var th = event.target.closest('th');
  if (!th) return;
  var table = th.closest('table');
  if (!table) return;
  var index = Array.prototype.indexOf.call(th.parentElement.children, th);
  var direction = th.dataset.verso === 'asc' ? -1 : 1;
  table.querySelectorAll('th[data-verso]').forEach(function (other) {
    delete other.dataset.verso;
  });
  th.dataset.verso = direction === 1 ? 'asc' : 'desc';
  function key(row) {
    var cell = row.cells[index];
    if (!cell) return null;
    var raw = cell.dataset.ord;
    if (raw === undefined) {
      var text = cell.textContent.trim();
      return text === '' || text === '—' ? null : text;
    }
    return raw === '' ? null : parseFloat(raw);
  }
  var rows = Array.prototype.filter.call(table.rows, function (row) {
    return row.cells.length && row.cells[0].tagName === 'TD';
  });
  rows.sort(function (a, b) {
    var x = key(a), y = key(b);
    if (x === null || y === null) return x === y ? 0 : (x === null ? 1 : -1);
    if (typeof x === 'number' && typeof y === 'number') return (x - y) * direction;
    return String(x).localeCompare(String(y)) * direction;
  });
  rows.forEach(function (row) { row.parentElement.appendChild(row); });
});
"""

#: Catalogue names of the localization campaigns after the first one: the key is the
#: `<tag>` of `localizzazione_<tag>_vtcd`.
LOCALIZATION_LABELS = {
    "it2000": "CNMF up to 2000 iterations",
    "eventi17": "17 events per class · 442 references · CNMF up to 2000 iterations",
    "eventi17it10000": "17 events per class · 442 references · CNMF up to 10000 iterations",
}

#: Uniform name of the VTCD localization views:
#: «VTCD localization · VTCD <scope> · <references> · <partition>». The key is the group
#: `<tag>`; scope and normals also decide where the view sits in the index. «First
#: selection» views render the 2 October dictionaries, whose chunks could contain annotated
#: frames of another class of the same continuous event.
_FIRST = "first selection (classes not distinguished)"
VTCD_TITLES = {
    "eventi17": ("global", "one chunk per training event + as many normal ones", True),
    "eventi17it10000": ("global", "one chunk per training event + as many normal ones", True),
    "it2000": ("global", "2 chunks per class + as many normal ones", True),
    "globalenn": ("global", "one chunk per training event", False),
    "classe": ("per class", "one chunk per training event + as many normal ones", True),
    "classenn": ("per class", "one chunk per training event", False),
    "globalennprima": ("global", f"one chunk per training event · {_FIRST}", False),
    "classeprima": (
        "per class",
        f"one chunk per training event + as many normal ones · {_FIRST}",
        True,
    ),
    "classennprima": ("per class", f"one chunk per training event · {_FIRST}", False),
}
#: First-selection views come right after their current version.
FIRST_SELECTION = frozenset({"globalennprima", "classeprima", "classennprima"})
#: Suffix of the ``vtcdclassi_<tag><suffix>_vtcd`` group -> partition in the title.
#: ``_mivia``: the MIVIA-Crime test set with the UCF-Crime dictionaries.
VTCD_CLASS_PARTITIONS = {"": "test", "_val": "validation", "_mivia": "test MIVIA-Crime"}


def _vtcd_title(tag: str, partizione: str) -> str | None:
    """:return: the uniform view name, or ``None`` if the tag is not listed."""
    if tag not in VTCD_TITLES:
        return None
    scope, references, _ = VTCD_TITLES[tag]
    return f"VTCD localization · VTCD {scope} · {references} · {partizione}"


#: The VTCD variants of 2 October 2026 (``scripts/vtcd_classi.py``), in the
#: ``vtcdclassi_<tag>[_val]_vtcd`` groups: section title and opening line. The prefix is
#: not ``localizzazione_`` on purpose: an older catalogue ignores them instead of
#: describing them with another campaign's text.
VTCD_CLASS_LABELS = {
    "classe": (
        "Per-class VTCD · one chunk per training event + as many normal ones",
        "One dictionary per class: one training chunk for every annotated event of the class "
        "(the purest, i.e. without annotated frames of other classes or, failing that, with the "
        "fewest; then maximum overlap and the most central), plus "
        "as many normal chunks. On every "
        "video the dictionary of its event class is applied; a video with events of "
        "two classes has one page per class.",
    ),
    "classenn": (
        "Per-class VTCD without normals · one chunk per training event",
        "The same per-class dictionaries, fitted on the annotated event chunks only, "
        "without normal chunks.",
    ),
    "globalenn": (
        "Global VTCD without normals · 17 events per class · 221 references",
        "One chunk per event (the purest, then overlap and centrality), 17 events for each "
        "of the 13 classes, no normal chunk.",
    ),
    "classeprima": (
        "Per-class VTCD · first selection",
        "First chunk selection (2 October), replaced on 3 October: one training chunk "
        "per annotated event of the class, with maximum overlap with the event and then the most "
        "central, without excluding annotated frames of another class when a continuous "
        "event changes class (mostly Explosion/Arson); plus as many normal chunks. "
        "Only the seven classes where the selection differs from the purest chunk "
        "(Arson, Explosion, Fighting, RoadAccidents, Robbery, Shooting, Vandalism): for the "
        "other six the dictionaries are identical and the pages are in the current view.",
    ),
    "classennprima": (
        "Per-class VTCD without normals · first selection",
        "The same per-class dictionaries of the first selection, fitted on the event chunks "
        "only, without normal chunks, on all 13 classes: for Arson, Explosion, Fighting, "
        "RoadAccidents, Robbery, Shooting and Vandalism the dictionary differs from the "
        "purest-chunk one, for the other six it is identical. This view carries the bounding "
        "boxes of the concept chosen for every class (3 October 2026), only on the class events.",
    ),
    "globalennprima": (
        "Global VTCD without normals · first selection",
        "The 221 event chunks of the global view with normals (17 events per class, maximum "
        "overlap, classes not distinguished: 12 contain frames of another class), without "
        "normal chunks.",
    ),
}


def _page(title: str, body: str) -> str:
    return (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{STYLE}</style><main>{body}</main>"
        f"<script>{_SCRIPT}</script></html>"
    )


def _link(path: str, label: str) -> str:
    return f'<a href="{html.escape(path, quote=True)}">{html.escape(label)}</a>'


VARIANTS = {
    "raw_attention": "Head attention weights on the final layer, averaged over heads.",
    "gradient_activation": "Features after the head normalization multiplied by their gradient.",
    "gradient_activation_prenorm": "Same check before the normalization: an almost null map may come from the LayerNorm symmetry.",
    "attention_gradient": "Signed product of attention weight and logit gradient with respect to that weight.",
    "ig_blur": "Reference obtained by blurring every frame of the clip; diagnostics say whether the integration converges.",
    "ig_mean": "Uniform per-frame reference, with the mean colour of the original frame.",
    "ig_black": "Reference of truly black pixels, transformed with the detector normalization.",
    "ig_noise": "Gaussian-noise reference with a deterministic per-window seed.",
    "ig_tn_same_video": "Reference from a normal window predicted normal. The provenance says whether it belongs to the same video or to an external normal video.",
    "ig_ensemble": "Average of the available black, noise, mean-colour, normal-reference and blur attributions. Blur is included since 22 September 2026; non-converged components stay flagged.",
    "legrad_last": "Uses the last backbone layer.",
    "legrad_four": "Combines four backbone layers, as recorded in the campaign configuration.",
    "pca_rgb": "The first three principal components are mapped to the red, green and blue channels; the transform stays fixed over the video.",
    "eigencam": "A single principal component per window; the display scale stays declared.",
    "cosine": "The reference stays fixed over the clip; the metadata report video, frame and query region. Values close to 1 mean higher similarity.",
    "anomaly_projection": "Projection of the token on the learned direction: describes the representation, without attributing it a causal effect on the decision.",
    "anomaly_sensitivity": "Derivative of the logit along the learned direction, computed with the head in fp32 and checked with finite differences. A TCAV-style adaptation, not a full statistical TCAV test.",
    "vtcd_ids": "Assigns a colour to every ID. The curves show the token fraction occupied by every concept.",
    "vtcd_importance": "CRIS: removes random groups of concepts from the final tokens and measures the degradation of the binary loss. The map reports the importance of the concept assigned to each token.",
    "occlusion_blur": "Every volume is replaced with the corresponding blurred portion.",
    "occlusion_mean": "Every volume is replaced with the mean colour of the frame.",
    "occlusion_black": "Every volume is replaced with RGB black, normalized like the model input.",
    "motion_local_freeze": "Repeats the central frame only inside the selected volume; not equivalent to just reversing the temporal order.",
}


#: Page names of the cuts: a cut without its own label would appear under another's.
CUT_LABELS = {
    "side_by_side": "Side by side (video, overlay, map)",
    "side_by_side_bbox": "Side by side (video, overlay, bbox)",
    "concept_bbox": "Concept and bbox (whole video)",
    "concept_bbox_annotati": "Concept and bbox (annotated frames only)",
    "side_by_side_bbox_annotati": "Side by side (video, overlay, bbox on annotated frames only)",
    "side_by_side_bbox_annotati_plain": "Side by side with bbox, no interface",
    "concept_bbox_annotati_plain": "Concept and bbox, no interface",
    "overlay": "Overlay",
    "map": "Map only",
    "side_by_side_plain": "Side by side, no interface (video, overlay, map)",
    "overlay_plain": "Overlay, no interface",
    "map_plain": "Map only, no interface",
}


def _relative(target: Path, page_folder: Path) -> str:
    return Path(os.path.relpath(target, page_folder)).as_posix()


def _verified_manifest(path: Path) -> dict:
    """Read a rendering manifest and check that the declared files exist.

    A manifest is a declaration, not a proof. A ``running`` manifest is not a failure:
    perturbed views cost 4.392 s per source frame, so long videos stay half-done by
    construction; existing files are kept, declared-but-missing ones dropped, and the
    missing count is returned in ``parziale``. On a complete manifest a missing file stays
    an error (a half-copied folder).

    :param path: ``manifest.json`` of a rendering folder, possibly missing.
    :return: the manifest (with ``parziale`` while rendering is running), or an empty
        manifest if the file does not exist.
    :raises ValueError: if a file declared by a complete manifest is missing.
    """
    if not path.is_file():
        return {"videos": [], "images": []}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    present = [v for v in manifest["videos"] if (path.parent / v["file"]).is_file()]
    images = [n for n in manifest["images"] if (path.parent / n).is_file()]
    declared = len(manifest["videos"]) + len(manifest["images"])
    if len(present) + len(images) < declared or manifest.get("status") != "complete":
        manifest["parziale"] = {
            "dichiarati": declared,
            "presenti": len(present) + len(images),
            "stato": manifest.get("status", "sconosciuto"),
        }
    manifest["videos"], manifest["images"] = present, images
    return manifest


def _mask_cell(row: dict, page_folder: Path, cartella: Path, estremi: tuple[float, float]) -> str:
    """Return a grid cell: the link to a movie and its weight.

    The background is the mask effect rescaled between the minimum and maximum of its
    method, so the grid is itself a map; the number stays written.

    :param row: entry of the ``per_maschera`` manifest.
    :param estremi: minimum and maximum of ``effect_mean`` in the method.
    :return: the HTML fragment of the cell.
    """
    bottom, top = estremi
    quota = 0.0 if top <= bottom else (row["effect_mean"] - bottom) / (top - bottom)
    path = _relative(cartella / row["file"], page_folder)
    title = (
        f"{row['mask']} - {row['volume_tokens']} tokens - "
        f"mean logit delta {row['effect_mean']:+.4f}, max {row['effect_max']:+.4f} - "
        f"perturbed score between {row['perturbed_score_min']:.4f} and {row['perturbed_score_max']:.4f}"
    )
    return (
        f'<a class="cella" style="background:rgba(199,62,29,{quota * 0.72:.3f})" '
        f'href="{html.escape(path, quote=True)}" '
        f'data-src="{html.escape(path, quote=True)}" '
        f'title="{html.escape(title, quote=True)}">'
        f'<span class="delta">{row["effect_mean"]:+.2f}</span>'
        f'<span class="tok">{row["volume_tokens"]}</span></a>'
    )


def _performance(root: Path) -> dict[str, dict]:
    """Return how the detector did on every video, if someone counted it.

    Written by ``scripts/prestazioni_explain.py``; only read here. If missing, the column
    disappears instead of filling with dashes.

    :param root: the explainability collection.
    :return: ``{"model/video": count}``; empty if the report does not exist.
    """
    path = root / "rapporti/prestazioni.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))["video"]


def _number(value: float | None, digits: int = 3) -> str:
    """:return: the number, or ``n/a`` if undefined on this video."""
    return "n/a" if value is None else f"{value:.{digits}f}"


def _performance_cell(entry: dict) -> tuple[str, str]:
    """Return the summary of a row and the numbers that justify it.

    A normal video has no AUC or recall (no positive chunks): the right count is how many
    windows raised an alarm. The same holds for a fully annotated anomalous video, without
    negatives; ``chunk_auc_stato`` says which case applies.

    :param entry: the :func:`_performance` row for this video and model.
    :return: ``(cell text, title with the detail)``.
    """
    above = f"{entry['n_sopra_soglia']}/{entry['n_finestre']} ≥ τ"
    if entry["anomalo"]:
        text = f"AUC {_number(entry['chunk_auc'])} · F1 {_number(entry['chunk_f1'], 2)} · {above}"
    else:
        text = f"normal · {above}" + (" (false alarms)" if entry["fp"] else " (no alarm)")
    # The medians sit in the cell, on a row below: the same reading of the video without a
    # threshold. `get` rather than `[]`: an older report shows «n/a» instead of breaking the
    # catalogue.
    text += (
        f"<SOTTO>medians: anomalous {_number(entry.get('score_mediano_chunk_anomali'))} · "
        f"normal {_number(entry.get('score_mediano_chunk_normali'))}"
    )
    detail = [
        f"TP {entry['tp']} · FP {entry['fp']} · FN {entry['fn']} · TN {entry['tn']}",
        f"{entry['n_positive_vere']} positive windows out of {entry['n_finestre']}",
    ]
    if entry["anomalo"]:
        detail.append(f"detected events {entry['n_eventi_rilevati']}/{entry['n_eventi']}")
        if entry["latenza_mediana_s"] is not None:
            detail.append(f"median latency {entry['latenza_mediana_s']:.1f} s")
        else:
            detail.append(f"latency {entry['latenza_stato']}")
        if entry["chunk_auc"] is None:
            detail.append(f"AUC {entry['chunk_auc_stato']}")
        detail.append(f"AP {_number(entry['chunk_ap'])}")
        detail.append(
            f"precision {_number(entry['chunk_precision'], 2)} · recall {_number(entry['chunk_recall'], 2)}"
        )
    detail.append(
        f"score medians: anomalous {_number(entry.get('score_mediano_chunk_anomali'))}, "
        f"normal {_number(entry.get('score_mediano_chunk_normali'))}, "
        f"difference {_number(entry.get('delta_score_mediano_chunk'))} "
        "(threshold independent)"
    )
    if entry["classi_annotate"]:
        detail.append("classes " + ", ".join(entry["classi_annotate"]))
    detail.append(f"threshold {entry['soglia']:.4f} · partition {entry['split']}")
    return text, " — ".join(detail)


def _order(value: float | None) -> str:
    """:return: the attribute a cell sorts by; empty without a number.

    A cell without a key goes last in both directions: «undefined» is not zero.
    """
    return "" if value is None else f"{float(value):.6f}"


def _performance_td(performance: Mapping[str, dict], modello: str, video: str) -> str:
    """:return: the two cells of the row (performance and median difference)."""
    if not performance:
        return ""
    entry = performance.get(f"{modello}/{video}")
    if not entry:
        return "<td class='muted' colspan='2'>not computed</td>"
    text, detail = _performance_cell(entry)
    first, below = text.split("<SOTTO>")
    delta = entry.get("delta_score_mediano_chunk")
    written = "n/a" if delta is None else f"{delta:+.3f}"
    return (
        f"<td data-ord='{_order(entry['chunk_auc'])}' "
        f"title='{html.escape(detail, quote=True)}'>{html.escape(first)}"
        f"<span class='sotto'>{html.escape(below)}</span></td>"
        f"<td data-ord='{_order(delta)}'>{html.escape(written)}</td>"
    )


def _reader(percorso: str) -> str:
    """:return: a video player with the same look everywhere in the page.

    ``preload="none"``: with ``metadata`` the page made 180 requests on load through the
    tunnel to the cluster and became too slow. The «Full width» checkbox removes the height
    cap, useful for one-panel cuts, which are taller than wide.
    """
    return (
        f'<video class="lettore" controls preload="none" '
        f'src="{html.escape(percorso, quote=True)}"></video>'
        '<label class="piena"><input type="checkbox">Full width '
        "(the movie exceeds the screen height)</label>"
    )


def _play(percorso: str, etichetta: str, title: str = "", attivo: bool = False) -> str:
    """Return a link that changes the movie of the method player.

    It stays a real link with its ``href``: without JavaScript it opens the movie.

    :param percorso: movie path, relative to the page.
    :param etichetta: name of what is about to be shown.
    :param titolo: its numbers, in the ``title``.
    :param attivo: whether it is the movie the player already shows (marked as a click
        would).
    :return: the HTML fragment of the link.
    """
    attributes = f' title="{html.escape(title, quote=True)}"' if title else ""
    if attivo:
        attributes += " data-attivo"
    return (
        f'<a class="gioca" href="{html.escape(percorso, quote=True)}" '
        f'data-src="{html.escape(percorso, quote=True)}"{attributes}>{html.escape(etichetta)}</a>'
    )


def _key(row: dict) -> tuple[int, int] | None:
    """:return: first and last frame of the artifact, or ``None`` if not declared."""
    start, end = row.get("start_frame"), row.get("end_frame")
    if start is None or end is None:
        return None
    return int(start), int(end)


def _window_interval(record: dict) -> tuple[int, int] | None:
    """:return: first and last frame of a campaign window."""
    indices = record.get("frame_indices") or record.get("window", {}).get("explicit_frame_indices")
    end = record.get("window", {}).get("end_frame")
    if not indices or end is None:
        return None
    return int(min(indices)), int(end)


def _add(celle: dict, chiave, fragment: str) -> None:
    """Accumulate a fragment under its interval, without overwriting another."""
    if fragment:
        celle.setdefault(chiave, []).append(fragment)


def _comparison_cells(manifest: dict, cartella: Path, metodo: str, page_folder: Path) -> dict:
    """Return the three-panel comparison of a perturbative method, one entry per window."""
    clip = [v for v in manifest["videos"] if v["method"] == metodo]
    if not clip:
        return {}
    choices = manifest["chosen_masks"][metodo]
    delta = [c["effect"] for c in choices]
    title = (
        "three panels: true input, same input with the volume replaced, map; "
        f"mask {choices[0]['mask']}{'' if len(choices) == 1 else ' and others'}; "
        f"logit delta between {min(delta):+.3f} and {max(delta):+.3f}"
    )
    cells: dict = {}
    for row in clip:
        _add(
            cells,
            _key(row),
            _play(_relative(cartella / row["file"], page_folder), "Perturbed comparison", title),
        )
    return cells


def _ig_baseline_cells(manifest: dict, cartella: Path, metodo: str, page_folder: Path) -> dict:
    """Return the reference the integral of an IG baseline starts from."""
    clip = [v for v in manifest["videos"] if v["method"] == metodo]
    if not clip:
        return {}
    rows = manifest["references"].get(metodo) or []
    scores = [r["reference_score"] for r in rows]
    cut_item = max((r["clipped_fraction"] for r in rows), default=0.0)
    note = ["the attribution integrates the gradient from this reference to the clip"]
    if scores:
        note.append(
            f"the detector gives the reference a score between {min(scores):.4f} and {max(scores):.4f}"
        )
    if rows and rows[0].get("tn_video_id"):
        note.append(f"normal window from {rows[0]['tn_video_id']}")
    if cut_item > 0.001:
        note.append(
            f"{cut_item:.1%} of the reference falls outside the colour range and is clipped"
        )
    cells: dict = {}
    for row in clip:
        _add(
            cells,
            _key(row),
            _play(
                _relative(cartella / row["file"], page_folder),
                "Integral reference",
                " · ".join(note),
            ),
        )
    return cells


def _mask_cells(manifest: dict, cartella: Path, metodo: str, page_folder: Path) -> dict:
    """Return the 45 masks of a perturbative method, inside the row of their window.

    They sit in a ``details``, not in a page section: open they are a map (cell position =
    occluded region position), closed they take one row.
    """
    rows = [v for v in manifest["videos"] if v["method"] == metodo]
    if not rows:
        return {}
    extremes = (min(r["effect_mean"] for r in rows), max(r["effect_mean"] for r in rows))
    cells: dict = {}
    for key in sorted({_key(r) for r in rows}, key=lambda k: (k is None, k)):
        group = [r for r in rows if _key(r) == key]
        per_index = {r["mask_index"]: r for r in group}
        strong = max(group, key=lambda r: r["effect_max"])
        body = (
            f"<details class='dentro'><summary>{len(group)} masks, one per movie</summary>"
            f"<p class='muted'>Mean logit delta between {extremes[0]:+.3f} and {extremes[1]:+.3f}; the "
            "background of each cell is the effect of its mask and the cell position is the "
            "position of the occluded region. A click changes the movie of the player above; the "
            f"strongest is <code>{html.escape(strong['mask'])}</code>.</p>"
        )
        for data in manifest["families"].values():
            present = [k for k in data["indices"] if k in per_index]
            if not present:
                continue
            body += (
                f"<h5>{html.escape(data['label'])} <span class='muted'>({len(present)})</span></h5>"
            )
            if "griglia" in data:
                body += (
                    "<div class='griglia' style='grid-template-columns:repeat("
                    f"{data['colonne']},minmax(56px,84px))'>"
                )
                for grid_row in data["griglia"]:
                    for k in grid_row:
                        body += (
                            _mask_cell(per_index[k], page_folder, cartella, extremes)
                            if k in per_index
                            else "<span class='cella vuota'></span>"
                        )
                body += "</div>"
            else:
                body += "<div class='links'>"
                for k in present:
                    entry = per_index[k]
                    body += _play(
                        _relative(cartella / entry["file"], page_folder),
                        f"{entry['mask']} ({entry['effect_mean']:+.3f})",
                    )
                body += "</div>"
        _add(cells, key, body + "</details>")
    return cells


def _view_cells(manifest: dict, cartella: Path, metodo: str, page_folder: Path) -> dict:
    """Return the maps of the same method recomputed on the perturbed inputs."""
    rows = [v for v in manifest["videos"] if v.get("base") == metodo]
    if not rows:
        return {}
    cells: dict = {}
    for key in sorted({_key(r) for r in rows}, key=lambda k: (k is None, k)):
        group = sorted((r for r in rows if _key(r) == key), key=lambda r: r["view"])
        body = (
            f"<details class='dentro'><summary>{len(group)} views on the perturbed input"
            "</summary><p class='muted'>The clip is no longer the true one: it is what the "
            "detector saw when the score changed, and the map describes the central "
            "panel. The colour scale stays the base method one, or an explanation that "
            "<em>moves</em> could not be told from one that <em>fades</em>.</p>"
            "<div class='links'>"
        )
        for row in group:
            body += _play(
                _relative(cartella / row["file"], page_folder),
                VIEWS.get(row["view"], row["view"]),
            )
        _add(cells, key, body + "</div></details>")
    return cells


def _quad_cells(manifest: dict, cartella: Path, metodo: str, page_folder: Path) -> dict:
    """Return the 2x2 quad putting both inputs and both maps in the same frame.

    Next to the perturbed views, not in their place: the quad also shows where the
    explanation fell before the perturbation.
    """
    rows = [v for v in manifest["videos"] if v.get("base") == metodo]
    if not rows:
        return {}
    cells: dict = {}
    for key in sorted({_key(r) for r in rows}, key=lambda k: (k is None, k)):
        group = sorted((r for r in rows if _key(r) == key), key=lambda r: r["view"])
        body = (
            f"<details class='dentro'><summary>{len(group)} original/perturbed quads"
            "</summary><p class='muted'>Four panels: the original input and its map on top, "
            "the perturbed input and its map below. The colour scale is "
            "<strong>a single one</strong> for both, or a halved explanation would fill the "
            "table like the whole one and an explanation that <em>moves</em> could not be told "
            "from one that <em>fades</em>.</p>"
            "<div class='links'>"
        )
        for row in group:
            body += _play(
                _relative(cartella / row["file"], page_folder),
                VIEWS.get(row["view"], row["view"]),
            )
        _add(cells, key, body + "</div></details>")
    return cells


#: Catalogue name of a bounding-box variant.
_BBOX_VARIANTS = {
    "intero": "whole video",
    "annotati": "whole video, boxes only on annotated positive frames",
}


def _bbox_label(row: dict, righe: list[dict]) -> str:
    """Return the page name of a bounding-box movie.

    The concept is named only when the entry carries more than one; the ``pulito`` style is
    always declared, since without a header the movie does not say what it shows (ADR-077).

    :param row: the row of ``bounding_box/manifest.json``.
    :param righe: all rows of the same map, to know whether there are several concepts.
    :return: the link label.
    """
    text = _BBOX_VARIANTS.get(row["variante"], row["variante"])
    if len({r.get("sorgente") for r in righe}) > 1:
        text = f"{row.get('etichetta') or row.get('sorgente')} · {text}"
    if row.get("stile") == "pulito":
        text += " · no interface"
    return text


def _bbox_cells(manifest: dict, cartella: Path, metodo: str, page_folder: Path) -> dict:
    """Return the bounding boxes derived from the map of this method.

    They sit in the entry of the source map (``vtcd_ids`` for concept 0, ``ig_ensemble`` for
    the top 10% of the attribution), so they can be compared with the heatmap. Not a
    validated localization: UCF-Crime has no spatial ground truth, and the text says so.
    """
    rows = [v for v in manifest["videos"] if v.get("base") == metodo]
    if not rows:
        return {}
    cells: dict = {}
    for key in sorted({_key(r) for r in rows}, key=lambda k: (k is None, k)):
        group = sorted(
            (r for r in rows if _key(r) == key),
            key=lambda r: (r.get("sorgente", ""), r["variante"], r["file"]),
        )
        body = (
            f"<details class='dentro'><summary>{len(group)} movies with bounding boxes"
            "</summary><p class='muted'>One rectangle per <strong>connected</strong> area "
            "of the mask, temporal plane by temporal plane, at the native resolution of the "
            "video. The box is the outer edge of the tokens, not a detected object: it is a "
            "<em>proposal</em> of automatic annotation, and UCF-Crime has no spatial ground "
            "truth to measure it against. The coordinates are next to the movies in "
            "<code>boxes.json</code>.</p><div class='links'>"
        )
        for row in group:
            body += _play(_relative(cartella / row["file"], page_folder), _bbox_label(row, rows))
        _add(cells, key, body + "</div></details>")
    return cells


def _extra_per_method(sources: list[dict], page_folder: Path) -> dict:
    """Return everything visible of a method, indexed by method and window.

    Every artifact is a cell in the row of its window, and the method entry has a single
    player and a single table. Global ablations stay out: they do not belong to a method.

    :param sources: the campaigns of the same video/model.
    :param page_folder: page folder, for relative paths.
    :return: ``{method: {(first, last frame) | None: [HTML fragments]}}``.
    """
    builders = (
        ("confronti", _comparison_cells, lambda m: {v["method"] for v in m["videos"]}),
        ("confronti_ig", _ig_baseline_cells, lambda m: {v["method"] for v in m["videos"]}),
        ("per_maschera", _mask_cells, lambda m: {v["method"] for v in m["videos"]}),
        ("viste_perturbate", _view_cells, lambda m: {v["base"] for v in m["videos"]}),
        ("quadri_concetto", _quad_cells, lambda m: {v["base"] for v in m["videos"]}),
        ("bounding_box", _bbox_cells, lambda m: {v["base"] for v in m["videos"]}),
    )
    blocks: dict = {}
    for entry in sources:
        for name, build_catalog, methods_of in builders:
            path = entry["folder"] / name / "manifest.json"
            if not path.is_file():
                continue
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("status") != "complete":
                # Rendering in progress: existing movies are kept instead of skipping the whole folder (a
                # 15,000-frame campaign needs 19 h of perturbed views against 4 h of wall time).
                manifest["videos"] = [
                    v for v in manifest["videos"] if (path.parent / v["file"]).is_file()
                ]
            if not manifest["videos"]:
                continue
            for method in sorted(methods_of(manifest)):
                produced = build_catalog(manifest, path.parent, method, page_folder)
                for key, fragments in produced.items():
                    blocks.setdefault(method, {}).setdefault(key, []).extend(fragments)
    return blocks


#: Order of the cuts in the table: the side-by-side cut first, since the player loads it
#: by default.
CUT_ORDER = (
    "side_by_side_bbox",
    "concept_bbox_annotati",
    "side_by_side_bbox_annotati",
    "concept_bbox_annotati_plain",
    "side_by_side_bbox_annotati_plain",
    "concept_bbox",
    "side_by_side",
    "overlay",
    "map",
    "side_by_side_plain",
    "overlay_plain",
    "map_plain",
)


def _row_label(chiave: tuple[int, int] | None, source_frame: int) -> str:
    """:return: the table name of an interval that is not a window."""
    if chiave is None:
        return "Undeclared interval"
    if chiave[0] == 0 and chiave[1] >= source_frame - 1:
        return "Whole video"
    return f"Frame {chiave[0]}–{chiave[1]}"


def _method_html(entry: dict, method: str, page_folder: Path, extra: Mapping | None = None) -> str:
    """Return everything visible of a method, in a single player and a single table.

    One row per window and one column per artifact family (cuts, images, perturbed input).
    An artifact belonging to no window (the comparison of a whole video) gets its own
    declared row at the end.

    :param entry: the campaign covering the most frames for this method.
    :param method: the method of the entry.
    :param page_folder: page folder, for relative paths.
    :param extra: ``{interval: [fragments]}`` from :func:`_extra_per_method`.
    :return: the HTML fragment of the entry.
    """
    folder, meta, render = entry["folder"], entry["meta"], entry["render"]
    extra = dict(extra or {})
    description = VARIANTS.get(method, "")
    concepts = method_concepts(method)
    if concepts and len(concepts) == 1:
        description = f"Mask of concept {concepts[0]} only: highlights where it is assigned; its absence in the window is flagged."
    elif concepts:
        description = (
            "Mask of concepts " + "+".join(map(str, concepts)) + " merged as a single "
            "concept: highlights where either is assigned."
        )
    parts = [
        f"<details><summary>{html.escape(method_label(method))}</summary><p>{html.escape(description)}</p>"
    ]
    warnings = sorted({s for s in render["map_states"].get(method, []) if s})
    if warnings:
        parts.append(f"<div class='notice'>{'<br>'.join(map(html.escape, warnings))}</div>")
    if method in meta.get("method_notes", {}):
        parts.append(f"<p class='muted'>{html.escape(meta['method_notes'][method])}</p>")
    full = render["covered_frames"] == render["source_frames"]
    videos = [v for v in render["videos"] if v["method"] == method]
    parts.append(
        f"<p class='muted'>{'Whole video' if full else 'Selected clips'}: "
        f"{len(meta['windows'])} windows, "
        f"{render['covered_frames']:,}/{render['source_frames']:,} frames. "
        "A click on a movie opens it in the player below.</p>"
    )
    # One player per method; it opens on the SIDE-BY-SIDE cut, the only one showing video,
    # overlay and map together and the only one wider than tall.
    first = next(
        (v for v in videos if v["kind"] in ("side_by_side_bbox", "side_by_side")),
        next((v for v in videos if v["kind"] == "overlay"), videos[0] if videos else None),
    )
    if first:
        parts.append(_reader(_relative(folder / "render" / first["file"], page_folder)))
    parts.append(
        "<table><tr><th>Window</th><th>State</th><th>Movies</th><th>Images</th>"
        "<th>Perturbed input</th></tr>"
    )
    # Rows are the UNION of the intervals, not just the windows: on a whole-video campaign the
    # movies cover one `segment` per cut, and matching them by window would make them
    # disappear from the page.
    states = render["map_states"].get(method, [""] * len(meta["windows"]))
    image_paths = {"/".join(Path(name).parts[-2:]): name for name in render["images"]}
    rows: dict = {}
    for i, record in enumerate(meta["windows"]):
        key = _window_interval(record)
        row = rows.setdefault(key, {"filmati": [], "immagini": [], "titolo": ""})
        row["etichetta"] = f"Frame {key[0]}–{key[1]}" if key else str(record["window_id"])
        row["titolo"] = record["window_id"]
        row["stato"] = (states[i] if i < len(states) else "") or "Available"
        row["immagini"] += [
            _link(_relative(folder / "render" / image_paths[item_name], page_folder), label_text)
            for item_name, label_text in (
                (f"pannelli/{method}_clip_{i:05d}.png", "Temporal planes"),
                (f"curve/{method}_temporal_{i:05d}.png", "Temporal curve"),
            )
            if item_name in image_paths
        ]
    for cut in CUT_ORDER:
        for video in videos:
            if video["kind"] != cut:
                continue
            key = _key(video)
            row = rows.setdefault(key, {"filmati": [], "immagini": [], "titolo": ""})
            row.setdefault("etichetta", _row_label(key, render["source_frames"]))
            row.setdefault("stato", "—")
            row["filmati"].append(
                _play(
                    _relative(folder / "render" / video["file"], page_folder),
                    CUT_LABELS.get(video["kind"], video["kind"]),
                    attivo=video is first,
                )
            )
    for key, fragments in extra.items():
        row = rows.setdefault(key, {"filmati": [], "immagini": [], "titolo": ""})
        row.setdefault("etichetta", _row_label(key, render["source_frames"]))
        row.setdefault("stato", "—")
        row["perturbati"] = fragments
    for key in sorted(rows, key=lambda k: (k is None, k)):
        row = rows[key]
        title = f" title='{html.escape(row['titolo'], quote=True)}'" if row["titolo"] else ""
        order = "" if key is None else str(key[0])
        parts.append(
            f"<tr><td data-ord='{order}'{title}>{html.escape(row['etichetta'])}</td>"
            f"<td>{html.escape(row.get('stato', '—'))}</td>"
            f"<td>{' · '.join(row['filmati']) or '—'}</td>"
            f"<td>{' · '.join(row['immagini']) or '—'}</td>"
            f"<td>{' '.join(row.get('perturbati', [])) or '—'}</td></tr>"
        )
    parts.append("</table>")
    parts.append("<p class='links'>")
    for name, label in (
        ("campaign.json", "Provenance and checks"),
        ("render/scales.json", "Colour scales"),
    ):
        if (folder / name).is_file():
            parts.append(_link(_relative(folder / name, page_folder), label))
    parts.append("</p></details>")
    return "".join(parts)


#: Opening line of the menu, per family: the two per-class families need different
#: caveats.
PER_CLASS_NOTES = {
    "Class direction (per-class TCAV)": (
        "On annotated positive chunks: choose the class whose direction to show. The classes "
        "are training references, not predictions of the binary detector; on a normal chunk "
        "the direction does not exist and the map is null."
    ),
    "VTCD – final layer": (
        "On annotated positive chunks: choose the class whose dictionary to show. IDs are "
        "local to each class and are not comparable across classes, nor with those of the "
        "final layer above."
    ),
}

#: Fallback text: a per-class method belonging to no family gets its own menu at the end.
OTHERS_PER_CLASS_NOTE = (
    "On annotated positive chunks: choose the concept to show. The classes are "
    "training references, not predictions of the binary detector."
)


def _class_selector(
    by_method: Mapping[str, dict],
    metodi: set[str],
    page_folder: Path,
    blocks: Mapping,
    note: str,
    real_bases: tuple[str, ...] = (),
    ordine: tuple[str, ...] = (),
) -> tuple[str, set[str]]:
    """Build the «Per-class explanations» menu of a single family.

    The menu lives inside the section of its family, next to its binary variants. The
    ``select`` hooks to ``.class-picker`` rather than to the enclosing ``section``, so two
    menus in the same page never hide each other's entries.

    :param by_method: variant to take the artifacts of every method from.
    :param metodi: the ``*__class_*`` methods of this family only.
    :param page_folder: page folder, for relative paths.
    :param blocks: perturbative artifacts per method.
    :param nota: opening line of the menu, specific to the family.
    :param real_bases: methods without suffix to show under «True class».
    :param ordine: the family methods, in reading order (alphabetical order would put
        `vtcd_concept_00` before `vtcd_ids`); methods outside the list go last, alphabetically.
    :return: the HTML fragment (empty if there is nothing) and the shown methods.
    """
    real_chosen = [m for m in real_bases if m in by_method]
    if not metodi and not real_chosen:
        return "", set()
    choices = [("true", "True class")] + [
        (f"{k:02d}", name) for k, name in enumerate(UCF_ANOMALY_CLASSES)
    ]
    parts = [
        "<div class='class-picker'><h3>Per-class explanations</h3>",
        f"<p>{html.escape(note)}</p>",
        "<label>Class <select onchange=\"this.closest('.class-picker')"
        ".querySelectorAll('[data-class]').forEach(e=>e.hidden=e.dataset.class!==this.value)\">",
    ]
    for value, label in choices:
        parts.append(f"<option value='{value}'>{html.escape(label)}</option>")
    parts.append("</select></label>")

    def key(metodo: str) -> tuple[int, str]:
        base = metodo.rsplit("__class_", 1)[0]
        return (ordine.index(base) if base in ordine else len(ordine), metodo)

    shown: set[str] = set()
    for value, label in choices:
        hidden = " hidden" if value != "true" else ""
        parts.append(f"<div data-class='{value}'{hidden}><h4>{html.escape(label)}</h4>")
        selected = sorted((m for m in metodi if m.endswith(f"__class_{value}")), key=key)
        if value == "true":
            selected = real_chosen + selected
        for method in selected:
            parts.append(_method_html(by_method[method], method, page_folder, blocks.get(method)))
        shown.update(selected)
        parts.append("</div>")
    parts.append("</div>")
    return "".join(parts), shown


#: The per-class VTCD box cuts, in the order of the box section.
_PAGE_BOX_CUTS = ("concept_bbox_annotati", "side_by_side_bbox_annotati")

#: Suffix of the folder of the unified boxes of a video, next to its per-class campaigns
#: (``scripts/vtcd_classi.py``, ``BOX_SUFFIX``).
_BOX_VIDEO = "__box"


def _box_video(cartella: Path) -> dict | None:
    """:return: the summary of the unified boxes of a video, or ``None`` if missing."""
    path = cartella / "manifest.json"
    if not path.is_file():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    return summary if summary.get("status") == "complete" else None


def _box_video_html(cartella: Path, page_folder: Path) -> str:
    """Return the unified bounding boxes of a per-class VTCD video, at the top of the page.

    A single player with the box movies (concept/bbox pair, side by side, boxes only) and
    the two annotations (``boxes.json`` schema 2, ``boxes_per_frame.csv`` with the ``classe``
    column) imported by the annotator. On every event the boxes come from the concept chosen
    for its class, in that class dictionary.

    :param cartella: the ``<video>__box`` folder of the video.
    :param page_folder: page folder, for relative paths.
    :return: the HTML fragment, empty if there are no boxes.
    """
    summary = _box_video(cartella)
    if summary is None:
        return ""
    render = _verified_manifest(cartella / "render_box/manifest.json")
    pairs = [
        v
        for cut in _PAGE_BOX_CUTS
        for v in render.get("videos", [])
        if v["kind"] == cut and v["method"] != "vtcd_ids"
    ]
    pairs += [
        v
        for cut in _PAGE_BOX_CUTS
        for v in render.get("videos", [])
        if v["kind"] == cut and v["method"] == "vtcd_ids"
    ]
    folders = sorted((cartella / "bounding_box").glob("*__annotati/manifest.json"))
    if not pairs or not folders:
        return ""
    box = json.loads(folders[0].read_text(encoding="utf-8"))
    chosen = " · ".join(
        f"{html.escape(class_name)}: {html.escape(method_label(method))}"
        for class_name, method in summary.get("metodi", {}).items()
    )
    body = (
        "<section id='bounding-box'><h2>Bounding boxes</h2><p>One rectangle per connected "
        "area of the mask of the concept chosen for the <strong>event class</strong>, "
        "in that class dictionary, temporal plane by temporal plane, at the native "
        f"resolution and <strong>only on annotated frames</strong>. Concepts: {chosen}. Outside "
        "the events there are no boxes; the side-by-side view shows the mask of the nearest event. "
        "Boxes entirely contained in a larger one are removed; the number on a box is that "
        f"of the CSV <code>box</code> column. Frames with boxes: "
        f"{box.get('frame_disegnati', 0):,} of {box.get('frame_annotati', 0):,} annotated. They "
        "are a proposal to correct with the annotator, not a validated localization.</p>"
    )
    if box.get("skipped"):
        body += f"<div class='notice'>{html.escape(box['skipped'])}</div>"
    body += _reader(_relative(cartella / "render_box" / pairs[0]["file"], page_folder))
    links = [
        _play(
            _relative(cartella / "render_box" / v["file"], page_folder),
            CUT_LABELS.get(v["kind"], v["kind"])
            + (" · ID map" if v["method"] == "vtcd_ids" else ""),
            attivo=v is pairs[0],
        )
        for v in pairs
    ]
    links += [
        _play(_relative(folders[0].parent / v["file"], page_folder), "Boxes only")
        for v in box.get("videos", [])
    ]
    body += "<p class='links'>" + " · ".join(links) + "</p><p class='links'>"
    for name, label in (
        ("boxes.json", "Annotations per window and plane (boxes.json)"),
        ("boxes_per_frame.csv", "Annotations per frame (boxes_per_frame.csv)"),
    ):
        if (folders[0].parent / name).is_file():
            body += _link(_relative(folders[0].parent / name, page_folder), label) + " "
    return body + "</p></section>"


def _dictionary_sections(
    sources: list[dict], page_folder: Path, chosen: Mapping[str, str] | None = None
) -> str:
    """Return the body of a per-class VTCD page: one section per dictionary.

    A video with events of several classes has one campaign per dictionary
    (``<video>__<Class>``), all in the same page. Methods share names across dictionaries,
    so they are not merged by name. Boxes sit at the top of the page (:func:`_box_video_html`).

    :param sources: the campaigns of the video in the view.
    :param page_folder: page folder, for relative paths.
    :param chosen: class -> concept chosen for the boxes, which opens its section.
    :return: the HTML fragment.
    """

    def class_of(entry: dict) -> str:
        return (entry["meta"].get("vtcd_dictionary") or {}).get("class") or ""

    sorted_items = sorted(sources, key=lambda e: (class_of(e), e["row"]["video"]))
    body = ""
    if len(sorted_items) > 1:
        body += (
            "<p><strong>Dictionaries of this video:</strong> "
            + " · ".join(
                f"<a href='#dizionario-{html.escape(class_of(e), quote=True)}'>"
                f"class {html.escape(class_of(e))}</a>"
                for e in sorted_items
            )
            + "</p>"
        )
    for entry in sorted_items:
        dictionary = entry["meta"].get("vtcd_dictionary") or {}
        class_name = class_of(entry)
        which = f"class {class_name}" if class_name else "global"
        references = len((entry["meta"].get("reference_diagnostics") or {}).get("references") or [])
        body += (
            f"<section id='dizionario-{html.escape(class_name or 'globale', quote=True)}'>"
            f"<h2>Dictionary {html.escape(which)}</h2><p>"
            f"{dictionary.get('n_positive', '?')} event chunks + "
            f"{dictionary.get('n_normal', '?')} normal ones · {references} training references. "
            "IDs are valid only for this dictionary.</p>"
        )
        blocks = _extra_per_method([entry], page_folder)
        chosen_value = (chosen or {}).get(class_name)
        # First the box concept, then the ID map, then the other concepts.
        methods = sorted(
            entry["render"]["methods"], key=lambda m: (m != chosen_value, m != "vtcd_ids", m)
        )
        for method in methods:
            body += _method_html(entry, method, page_folder, blocks.get(method))
        body += "</section>"
    return body


def _index_section(
    group: str,
    vista: tuple[str, str, str],
    view_rows: list[dict],
    iterations: int | None,
    column: str,
    performance: Mapping[str, dict],
) -> str:
    """:return: title, opening and table of an index view, under its section."""
    label, _, opening = vista
    details = [f"{len(view_rows)} pages"]
    if iterations:
        details.insert(0, f"CNMF up to {iterations} iterations")
    if any(r.get("box") for r in view_rows):
        details.append("bounding boxes available")
    body = (
        f"<h3 id='{html.escape(group, quote=True)}'>{label} "
        f"<span class='muted'>({' · '.join(details)})</span></h3>"
        f"<p>{html.escape(opening)} <span class='muted'>A click on a header "
        "sorts the table by that column.</span></p>"
        f"<table><tr><th>Model</th><th>Video</th>"
        f"<th>Content</th>{column}<th>Open</th></tr>"
    )
    for row in view_rows:
        search = html.escape(
            f"{label} {row['model']} {row['video']} {' '.join(row['categories'])}".lower(),
            quote=True,
        )
        content = f"{row['windows']} windows · {len(row['methods'])} variants"
        if row["categories"]:
            content += " · " + ", ".join(row["categories"])
        if row.get("classi"):
            content += " · " + ("classes " if len(row["classi"]) > 1 else "class ")
            content += ", ".join(row["classi"])
        if row.get("box"):
            content += " · bounding box"
        body += (
            f"<tr data-search='{search}'><td>{html.escape(row['model'])}</td>"
            f"<td>{html.escape(row['video'])}</td>"
            f"<td data-ord='{row['windows']}'>{html.escape(content)}</td>"
            f"{_performance_td(performance, row['model'], row['video'].split('__', 1)[0])}"
            f"<td>{_link(row['page'], 'Video, maps and curves')}</td></tr>"
        )
    return body + "</table>"


def write_catalog(root: Path) -> dict:
    """Gather the methods into one page per video and model, without moving the artifacts.

    :param root: collection with results, configuration and reports.
    :return: inventory of campaigns and unified pages.
    :raises ValueError: on an incomplete manifest or a declared file that is missing.
    """
    from evd.utils.io import write_json, write_text_atomic

    root = root.resolve()
    categories = {}
    selection = root / "configurazione/development_20260910.json"
    if selection.exists():
        categories = {
            w["window_id"]: w["category"]
            for w in json.loads(selection.read_text(encoding="utf-8"))["windows"]
        }
    performance = _performance(root)
    view_path = root / "configurazione/catalog_view.json"
    view = json.loads(view_path.read_text(encoding="utf-8")) if view_path.exists() else {}
    hidden_models = set(view.get("hidden_models", []))
    inventory, entries = [], []
    for path in sorted((root / "risultati").glob("*/*/*/campaign.json")):
        folder = path.parent
        group, model, video = folder.relative_to(root / "risultati").parts
        if model in hidden_models:
            continue
        meta = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "group": group,
            "model": model,
            "video": video,
            "status": "processing",
            "completed_windows": len(meta["windows"]),
            "expected_windows": len(meta.get("expected_window_ids", meta["windows"])),
        }
        inventory.append(row)
        if meta["status"] != "complete":
            continue
        render_path = folder / "render/manifest.json"
        if not render_path.exists():
            row["status"] = "rendering missing"
            continue
        # The main rendering also goes through the tolerant check: while a copy is running a
        # campaign can have a complete manifest and only part of the movies.
        render = _verified_manifest(render_path)
        if not render.get("videos") and not render.get("images"):
            row["status"] = "rendering missing"
            continue
        global_render = _verified_manifest(folder / "global_motion/manifest.json")
        comparisons = _verified_manifest(folder / "confronti/manifest.json")
        per_mask = _verified_manifest(folder / "per_maschera/manifest.json")
        baseline_ig = _verified_manifest(folder / "confronti_ig/manifest.json")
        views = _verified_manifest(folder / "viste_perturbate/manifest.json")
        quads = _verified_manifest(folder / "quadri_concetto/manifest.json")
        # A rendering still in progress is DECLARED: the page shows what exists and how much is
        # missing.
        partial = {
            name: m["parziale"]
            for name, m in (
                ("filmati", render),
                ("confronti", comparisons),
                ("per_maschera", per_mask),
                ("confronti_ig", baseline_ig),
                ("viste_perturbate", views),
                ("quadri_concetto", quads),
            )
            if "parziale" in m
        }
        cats = sorted(
            {categories[w["window_id"]] for w in meta["windows"] if w["window_id"] in categories}
        )
        row.update(
            status="complete",
            windows=len(meta["windows"]),
            categories=cats,
            coverage="whole video"
            if render["covered_frames"] == render["source_frames"]
            else "selected clips",
            covered_frames=render["covered_frames"],
            source_frames=render["source_frames"],
            videos=sum(
                len(m["videos"])
                for m in (render, global_render, comparisons, per_mask, baseline_ig)
            ),
            images=sum(
                len(m["images"])
                for m in (render, global_render, comparisons, per_mask, baseline_ig)
            ),
            methods=render["methods"],
            page=(folder / "index.html").relative_to(root).as_posix(),
            scientific_status="diagnostic"
            if any(
                "diagnostic" in s.lower() or "fallito" in s.lower()
                for states in render["map_states"].values()
                for s in states
            )
            else "available",
        )
        if partial:
            row["parziali"] = partial
        entries.append(
            {"folder": folder, "meta": meta, "render": render, "row": row, "parziali": partial}
        )
    # Both lists are a view: they do not change the data or geometry of the sources.
    group_view = {
        "sviluppo": "sviluppo",
        "nuovi_sviluppo": "sviluppo",
        "cosine_sviluppo": "sviluppo",
        "volumi_sviluppo": "sviluppo",
        "regressioni": "completi",
        "nuovi_completi": "completi",
        "cosine_completi": "completi",
        "cosine": "completi",
        "volumi_completi": "completi",
        # Campaign of 15 September: the 46 videos of the qualitative selection with the whole
        # perturbative family, and the 279 anomalous val/test videos with the promising methods
        # only.
        "selezione": "selezione",
        "selezione_avanzati": "selezione",
        # Concepts (VTCD and per-class direction) sit in their OWN campaign: same base, same view,
        # a third campaign on the same videos.
        "selezione_concetti": "selezione",
        "selezione_classi": "selezione",
        "selezione_vtcd_r32_k16": "vtcd_r32_k16",
        "selezione_vtcd_r64_k8": "vtcd_r64_k8",
        "selezione_vtcd_r64_k16": "vtcd_r64_k16",
        "corpus": "corpus",
        "corpus_avanzati": "corpus",
        # Localization of 29 September 2026 (ADR-077): base VTCD on the best model of the OVAD
        # ranking, plus the base campaign carrying windows, scores and previews. The concepts of
        # that campaign (CNMF at 500 iterations) left the collection on 30 September 2026: the view
        # stays, and without concepts it produces no pages. The test base also enters the later
        # test views.
        "localizzazione_base": "localizzazione",
        "localizzazione_vtcd": "localizzazione",
    }
    #: For every view: label, canonical group (the campaign hosting the page) and opening line.
    #: The canonical group is always the base campaign.
    views = {
        # The two tuning groups: the trials of 10 September, used to check that maps were drawn;
        # they come last because they are not a result.
        "sviluppo": (
            "Tuning · development windows",
            "sviluppo",
            "The windows chosen on 10 September to tune the methods, on 22 videos. "
            "They served to check that maps were drawn: not a qualitative selection.",
        ),
        "completi": (
            "Tuning · two whole videos",
            "regressioni",
            "Arson016 and Fighting047 in full, the two videos used to test method "
            "regressions before the campaign of 15 September.",
        ),
        "selezione": (
            "Qualitative selection",
            "selezione",
            "The videos of the qualitative analysis of 9 September, in full: promising methods, "
            "perturbative family, perturbed views and deletion/insertion curves.",
        ),
        "vtcd_r32_k16": (
            "VTCD ablation · 32 regions, 16 concepts",
            "selezione_vtcd_r32_k16",
            "Global VTCD concepts only on the updated selection; IDs are valid only for this dictionary.",
        ),
        "vtcd_r64_k8": (
            "VTCD ablation · 64 regions, 8 concepts",
            "selezione_vtcd_r64_k8",
            "Global VTCD concepts only on the updated selection; IDs are valid only for this dictionary.",
        ),
        "vtcd_r64_k16": (
            "VTCD ablation · 64 regions, 16 concepts",
            "selezione_vtcd_r64_k16",
            "Global VTCD concepts only on the updated selection; IDs are valid only for this dictionary.",
        ),
        "localizzazione": (
            "VTCD localization · best model (OVAD, 30 s guard)",
            "localizzazione_vtcd",
            "VTCD in the base configuration (52 training windows, 32 regions, 8 concepts) "
            "on the first model of the OVAD ranking with a 30 s guard. Where two chunks "
            "overlap, the frame shows the plane of the chunk with more annotated "
            "positive frames, and on ties the one with the nearest centre. Concept IDs are valid "
            "only for this dictionary.",
        ),
        "corpus": (
            "Corpus: validation and test",
            "corpus",
            "All anomalous videos of both partitions, in full, with the promising methods: "
            "raw attention, Gradient x Activation, Integrated Gradients, Anomaly Direction, "
            "Anomaly Prototypes.",
        ),
    }
    # Localization campaigns after the first (ADR-077, 30 September addendum) are discovered
    # from the group name: `localizzazione_<tag>` plus `_val` on validation, with `_vtcd` for
    # the concepts and `_base` for the campaign carrying windows and previews. TEST bases stay
    # those of the first campaign (`localizzazione_base`), shared by every test view.
    shared_bases: dict[str, list[str]] = {}
    view_tag: dict[str, str] = {}
    for entry in entries:
        group = entry["row"]["group"]
        match = re.fullmatch(r"localizzazione_([a-z0-9]+)(_val)?_(base|vtcd)", group)
        if match is None or group in group_view and group_view[group] != "localizzazione":
            continue
        view_name = f"localizzazione_{match[1]}{match[2] or ''}"
        group_view[group] = view_name
        base_views = shared_bases.setdefault("localizzazione_base", [])
        if not match[2] and view_name not in base_views:
            base_views.append(view_name)
        if view_name not in views:
            description_text = LOCALIZATION_LABELS.get(match[1], match[1])
            partition = "validation" if match[2] else "test"
            view_tag[view_name] = match[1]
            views[view_name] = (
                _vtcd_title(match[1], partition)
                or f"VTCD localization · {description_text} · {partition}",
                f"{view_name}_vtcd",
                f"VTCD ({442 if match[1].startswith('eventi17') else 52} training windows, 32 regions, 8 "
                f"concepts) on the first model of the OVAD ranking with a 30 s guard, "
                f"{description_text}, on the {partition} videos. Same plane assignment rule as the "
                f"first campaign. Concept IDs are valid only for this "
                f"dictionary.",
            )
    # The per-class and no-normals VTCD variants (2 October 2026): one view per group, one
    # page per (video, dictionary); the canonical group is the group itself. `_mivia`: the same
    # dictionaries, fitted on UCF-Crime, on the MIVIA-Crime test set.
    for entry in entries:
        group = entry["row"]["group"]
        match = re.fullmatch(r"vtcdclassi_([a-z0-9]+)(_val|_mivia)?_vtcd", group)
        if match is None or match[1] not in VTCD_CLASS_LABELS:
            continue
        view_name = group.removesuffix("_vtcd")
        group_view[group] = view_name
        if view_name not in views:
            _, classes_opening = VTCD_CLASS_LABELS[match[1]]
            partition = VTCD_CLASS_PARTITIONS[match[2] or ""]
            view_tag[view_name] = match[1]
            regions = (
                "SLIC regions computed as in «17 events per class» (same model, same "
                "settings), dictionaries fitted on the UCF-Crime training chunks"
                if match[2] == "_mivia"
                else "same SLIC regions as «17 events per class»"
            )
            views[view_name] = (
                _vtcd_title(match[1], partition),
                group,
                f"{classes_opening} 32 regions, 8 concepts, CNMF to convergence (cap 10000 "
                f"iterations), {regions}, on the {partition} videos. Concept "
                f"IDs are valid only for the dictionary of the page.",
            )
    # New groups of the 52/104/208 matrix are discovered from the published manifests; the
    # three views of the first batch keep their historical names.
    vtcd_groups = {}
    vtcd_parent_views = {}
    for entry in entries:
        group = entry["row"]["group"]
        local_match = re.fullmatch(
            r"localizzazione_([a-z0-9]+)_n([1-9]\d*)_r(16|32|64)_k(4|8|16)_vtcd", group
        )
        match = re.fullmatch(r"selezione_vtcd_(?:n([1-9]\d*)_)?r(16|32|64)_k(4|8|16)", group)
        if match is None and local_match is None:
            continue
        if local_match:
            tag = local_match[1]
            n, r, k = map(int, local_match.group(2, 3, 4))
            view_name = f"vtcd_{tag}_n{n}_r{r}_k{k}"
            vtcd_parent_views[group] = f"localizzazione_{tag}"
        else:
            n, r, k = int(match[1] or 52), int(match[2]), int(match[3])
            view_name = f"vtcd_n{n}_r{r}_k{k}"
            vtcd_parent_views[group] = "selezione"
        vtcd_groups[group] = (n, r, k)
        if group not in group_view:
            group_view[group] = view_name
            views[view_name] = (
                f"VTCD ablation · {n} training windows, {r} regions, {k} concepts",
                group,
                "Global VTCD concepts only on the updated selection; "
                "IDs are valid only for this dictionary.",
            )
    bundles = {}
    for entry in entries:
        row = entry["row"]
        view = group_view.get(row["group"])
        if view:
            # Per-class VTCD: one page per VIDEO, with all its dictionaries.
            video = (
                row["video"].split("__", 1)[0] if view.startswith("vtcdclassi_") else row["video"]
            )
            bundles.setdefault((view, row["model"], video), []).append(entry)
        for shared in shared_bases.get(row["group"], []):
            bundles.setdefault((shared, row["model"], row["video"]), []).append(entry)
    # A base alone does not make a localization page: without concepts there is nothing to
    # localize.
    bundles = {
        key: source_list
        for key, source_list in bundles.items()
        if not key[0].startswith("localizzazione")
        or any(e["row"]["group"].endswith("_vtcd") for e in source_list)
    }
    pages = []
    ablations_by_video = {}
    for entry in entries:
        group = entry["row"]["group"]
        if group in vtcd_groups:
            ablations_by_video.setdefault(
                (entry["row"]["model"], entry["row"]["video"]), []
            ).append(entry)
    for (view, model, video), sources in sorted(bundles.items()):
        view_label, canonical_group, opening = views[view]
        canonical = next((e for e in sources if e["row"]["group"] == canonical_group), sources[0])
        page_folder = canonical["folder"]
        by_method = {}
        for entry in sources:
            for method in entry["render"]["methods"]:
                previous = by_method.get(method)
                if (
                    previous is None
                    or entry["render"]["covered_frames"] > previous["render"]["covered_frames"]
                ):
                    by_method[method] = entry
        cats = sorted({c for e in sources for c in e["row"]["categories"]})
        body = f"<p>{_link(_relative(root / 'index.html', page_folder), '← Catalogue')}</p><h1>{html.escape(video)} · {html.escape(model)}</h1><p>{view_label} · {len(by_method)} variants available.</p>"
        ablations = sorted(
            [
                e
                for e in ablations_by_video.get((model, video), [])
                if vtcd_parent_views[e["row"]["group"]] == view
            ],
            key=lambda e: ("_n" not in e["row"]["group"], vtcd_groups[e["row"]["group"]]),
        )
        if model == "dora" and view.startswith("vtcd_"):
            if view.startswith("vtcd_n"):
                baseline = (
                    root / "risultati/selezione_vtcd_n52_r32_k8" / model / video / "index.html"
                )
                label = "Clean baseline: 52 references · 32 regions · 8 concepts"
            else:
                baseline = root / "risultati/selezione" / model / video / "index.html"
                label = "Historical VTCD 32 regions · 8 concepts (outside the clean comparison)"
            if (baseline.parent / "campaign.json").is_file():
                body += f"<p>{_link(_relative(baseline, page_folder), label)}</p>"
        if view.startswith("vtcd_"):
            body += f"<p>{html.escape(opening)} Every concept has the two usual video cuts: side by side and overlay. There are no perturbations or bounding boxes here.</p>"
        elif view == "localizzazione":
            body += f"<p>{html.escape(opening)} Every concept has the two usual video cuts: side by side and overlay. Bounding boxes, when present, sit below the VTCD ID map, in two styles: with and without the interface.</p>"
        elif view.startswith("vtcdclassi_"):
            # MIVIA-Crime renders only the views with the interface.
            styles = (
                "with the interface"
                if view.endswith("_mivia")
                else "with and without the interface"
            )
            body += (
                f"<p>{html.escape(opening)}</p><p>One section per dictionary: a video with "
                "events of several classes has all its dictionaries here, and the IDs of one dictionary "
                "do not hold for another. Every concept has the side-by-side and overlay views, "
                f"{styles}. Bounding boxes, where present, sit at the top: one "
                "per video, with the dictionary of each event class.</p>"
            )
        elif view.startswith("localizzazione_eventi17"):
            body += f"<p>{html.escape(opening)} One chunk per event, 17 events for each of the 13 classes and 221 normal chunks. Every concept has the side-by-side and overlay views, with and without the interface. IDs are valid for this dictionary.</p>"
        elif view.startswith("localizzazione_"):
            body += f"<p>{html.escape(opening)} Every concept has two video cuts: the side-by-side one (video, overlay and the concept bounding boxes over the whole video; for the ID map, the boxes of the concept chosen for localization) and the overlay. Raw attention keeps the side-by-side cut with the map. The box-only movies sit below the VTCD ID map, on the whole video and on annotated positive frames only, in two styles: with and without the interface.</p>"
        else:
            body += f"<p>{html.escape(opening)} Every variant has <strong>one player and one table</strong>: one row per window, and on the row everything about that method that can be watched there (cuts, panels and curves, the comparison with the perturbed input, the reference the integral starts from, the 45 masks and the views). A click opens the movie in the variant player without changing page. Only the global ablations, which belong to no method, remain at the end.</p>"
        body += "<p>Sign and intensity are read on the colour bar of each method. Numeric scales stay constant over time. Concept IDs have discrete colours.</p>"
        # How the detector did on THIS video, at the top of the page: maps explain a decision, and
        # without its outcome a success and an error look the same. Per-class pages are named
        # `<video>__<Class>`: performance belongs to the video, not to the dictionary.
        entry_item = performance.get(f"{model}/{video.split('__', 1)[0]}")
        if entry_item:
            text, detail = _performance_cell(entry_item)
            # Here the two rows share one line: it is a paragraph, not a cell.
            joined = text.replace("<SOTTO>", " · ")
            body += (
                f"<p><strong>Detector performance on this video</strong> "
                f"{html.escape(joined)}"
                f"<br><span class='muted'>{html.escape(detail)}</span></p>"
            )
        # What is missing is declared at the top, with its count.
        incomplete = {}
        for entry in sources:
            for name, data in entry.get("parziali", {}).items():
                summed = incomplete.setdefault(name, {"dichiarati": 0, "presenti": 0})
                summed["dichiarati"] += data["dichiarati"]
                summed["presenti"] += data["presenti"]
        if incomplete:
            entry_list = ", ".join(
                f"{item_name} {d['presenti']}/{d['dichiarati']}"
                for item_name, d in sorted(incomplete.items())
            )
            body += (
                "<div class='notice'>Rendering still in progress on this video: "
                f"{html.escape(entry_list)}. What has been produced so far is shown; the rest "
                "appears when rendering finishes.</div>"
            )
        box_dir = page_folder.parent / f"{video}{_BOX_VIDEO}"
        box_video = _box_video(box_dir) if view.startswith("vtcdclassi_") else None
        if view.startswith("vtcdclassi_"):
            # The unified boxes of the video, then one section per dictionary: methods of two
            # dictionaries share names (`vtcd_concept_04`) and merging them by name would show one
            # concept.
            body += _box_video_html(box_dir, page_folder)
            body += _dictionary_sections(sources, page_folder, (box_video or {}).get("metodi"))
        else:
            blocks = _extra_per_method(sources, page_folder)
            # A method existing ONLY in perturbative artifacts would have blocks and no entry: it is
            # declared instead of disappearing.
            orphans = sorted(set(blocks) - set(by_method))
            shown = set()
            # Per-class maps are read INSIDE the family producing them (TCAV under the class
            # direction, VTCD under the dictionary), split by the base method name.
            class_methods = {m for m in by_method if "__class_" in m}
            per_family: dict[str, set[str]] = {}
            for method_name in class_methods:
                base = method_name.rsplit("__class_", 1)[0]
                family_name = next(
                    (item_name for item_name, methods, _ in FAMILIES if base in methods), ""
                )
                per_family.setdefault(family_name, set()).add(method_name)
            for family, methods, description in FAMILIES:
                is_vtcd = "vtcd_ids" in methods
                # «True class» also hosts the unsuffixed variants of the TCAV family only: the same map on
                # the chunk class.
                real_bases = (
                    ("class_projection", "class_sensitivity")
                    if family.startswith("Class direction")
                    else ()
                )
                selector, shown_items = _class_selector(
                    by_method,
                    per_family.get(family, set()),
                    page_folder,
                    blocks,
                    PER_CLASS_NOTES.get(family, OTHERS_PER_CLASS_NOTE),
                    real_bases,
                    tuple(methods),
                )
                shown.update(shown_items)
                present = [
                    method for method in methods if method in by_method and method not in shown
                ]
                if not present and not selector and not (is_vtcd and ablations):
                    continue
                title = "VTCD" if is_vtcd else family
                section_id = ' id="vtcd"' if is_vtcd else ""
                body += f"<section{section_id}><h2>{html.escape(title)}</h2><p>{html.escape(description)}</p>"
                for method in present:
                    body += _method_html(by_method[method], method, page_folder, blocks.get(method))
                body += selector
                if is_vtcd and ablations:
                    body += (
                        '<div class="vtcd-ablations"><h3>VTCD ablation</h3>'
                        "<p>Configurations of this video, ordered by training windows, "
                        "SLIC regions and concepts. Every dictionary has its own IDs; "
                        "movies open in the players below.</p>"
                    )
                    for entry in ablations:
                        group = entry["row"]["group"]
                        n, r, k = vtcd_groups[group]
                        label = f"{n} training windows · {r} regions · {k} concepts"
                        if group.startswith("selezione_vtcd_") and not group.startswith(
                            "selezione_vtcd_n"
                        ):
                            label = "Historical (outside the clean comparison) · " + label
                        body += (
                            f'<details class="vtcd-config" id="{group_view[group]}">'
                            f"<summary>{html.escape(label)}</summary><p>"
                            + _link(
                                _relative(entry["folder"] / "index.html", page_folder),
                                "Open this configuration in a dedicated page",
                            )
                            + "</p>"
                        )
                        # Never merge by name: concept 0 of two different dictionaries keeps the map, player and
                        # paths of its campaign.
                        for method in methods:
                            if method in entry["render"]["methods"]:
                                body += _method_html(entry, method, page_folder)
                        body += "</details>"
                    body += "</div>"
                body += "</section>"
                shown.update(present)
            # A per-class method belonging to no family gets its own menu at the end.
            selector, shown_items = _class_selector(
                by_method,
                per_family.get("", set()) - shown,
                page_folder,
                blocks,
                OTHERS_PER_CLASS_NOTE,
            )
            if selector:
                body += "<section><h2>Per-class explanations without a family</h2>" + selector
                body += "</section>"
                shown.update(shown_items)
            for method in by_method.keys() - shown:
                body += (
                    f"<section><h2>{html.escape(method_label(method))}</h2>"
                    + _method_html(by_method[method], method, page_folder, blocks.get(method))
                    + "</section>"
                )
            shown_globals = False
            if orphans:
                # A map existing only in perturbed views has no entry of its own: it is named instead of
                # hiding its movies.
                body += (
                    "<section><h2>Artifacts without a matching variant</h2>"
                    "<p class='muted'>These maps appear only in the perturbative artifacts of "
                    "this collection: "
                    + ", ".join(f"<code>{html.escape(m)}</code>" for m in orphans)
                    + ".</p></section>"
                )
            for entry in sources:
                volumes_path = entry["folder"] / "volumes/manifest.json"
                if volumes_path.is_file():
                    volumes = json.loads(volumes_path.read_text(encoding="utf-8"))
                    body += "<h2>Results of the 45 masks</h2><p>Logit delta per single mask and reference; the perturbed area is shown to compare the volumes.</p><div class='links'>"
                    for item in volumes["images"]:
                        body += _link(
                            _relative(volumes_path.parent / item["file"], page_folder),
                            item["label"],
                        )
                    body += (
                        _link(_relative(volumes_path, page_folder), "Numeric values and provenance")
                        + "</div>"
                    )
                motion_path = entry["folder"] / "global_motion/manifest.json"
                if not motion_path.is_file() or shown_globals:
                    continue
                motion = json.loads(motion_path.read_text(encoding="utf-8"))
                if motion.get("status") != "complete":
                    continue
                # Once only: sibling campaigns of the same video/model each have their global ablations;
                # the first complete one is kept.
                shown_globals = True
                body += "<h2>Global motion ablations</h2><p>Original and perturbed side by side, with score and logit delta. Freeze, temporal reversal and five permutations.</p>"
                for method in motion["methods"]:
                    body += f"<details><summary>{html.escape(method)}</summary><div class='links'>"
                    clips = [v for v in motion["videos"] if v["method"] == method]
                    for clip in clips:
                        path = entry["folder"] / "global_motion" / clip["file"]
                        body += _link(
                            _relative(path, page_folder),
                            f"Frame {clip['start_frame']}–{clip['end_frame']}",
                        )
                    body += "</div>"
                    if clips:
                        path = _relative(
                            entry["folder"] / "global_motion" / clips[0]["file"], page_folder
                        )
                        body += f'<video controls preload="none" src="{html.escape(path, quote=True)}"></video>'
                    plot = f"{method}_scores.png"
                    if plot in motion["images"]:
                        body += _link(
                            _relative(motion_path.parent / plot, page_folder),
                            "Original and perturbed score, logit delta",
                        )
                    body += "</details>"
        write_text_atomic(page_folder / "index.html", _page(f"{video} · {model}", body))
        # Old addresses stay navigable and point to the single page.
        for entry in sources:
            if entry["folder"] != page_folder:
                link = _relative(page_folder / "index.html", entry["folder"])
                write_text_atomic(
                    entry["folder"] / "index.html",
                    _page(
                        f"{video} · {model}",
                        f"<h1>{html.escape(video)} · {html.escape(model)}</h1><p>The methods are gathered in the video page.</p>{_link(link, 'Open all methods')}",
                    ),
                )
        pages.append(
            {
                "group": view,
                "model": model,
                "video": video,
                "categories": cats,
                "methods": list(by_method),
                "page": (page_folder / "index.html").relative_to(root).as_posix(),
                "windows": len(canonical["meta"]["windows"]),
                "source_campaigns": [e["row"]["page"] for e in sources],
                "classi": sorted(
                    {
                        c
                        for e in sources
                        if (c := (e["meta"].get("vtcd_dictionary") or {}).get("class"))
                    }
                ),
                "box": box_video is not None
                or any(
                    v["kind"].startswith(("side_by_side_bbox", "concept_bbox"))
                    for e in sources
                    for v in e["render"]["videos"]
                ),
            }
        )
    complete = [r for r in inventory if r["status"] == "complete"]
    totals = {
        "campaigns": len(inventory),
        "rendered": len(complete),
        "videos": sum(r["videos"] for r in complete),
        "images": sum(r["images"] for r in complete),
        "missing": len(inventory) - len(complete),
        "pages": len(pages),
    }
    body = "<h1>Explainability V-JEPA 2.1</h1><p>DoRA r32 and frozen backbone with attentive head, and for localization VPT-shallow n8 on merged LoRA r32 · selected checkpoints, seed 42.</p>"
    body += "<p>Every page gathers the methods of the same video and model, with variant explanations, movies, panels and curves.</p>"
    body += "<label>Filter by model, video or category: <input id='filter' placeholder='e.g. dora, Fighting047, FN'></label>"
    # Label and opening line come from `views`, not rewritten here. The two columns also
    # appear in the table of what is only on the cluster, built after the loop.
    column = "<th>Performance</th><th>Δ medians</th>" if performance else ""
    # VTCD configurations are consulted inside the page of each video. Localization (ADR-077)
    # has its own section, first: it uses another model.
    subsequent = sorted(v for v in views if v.startswith("localizzazione_"))
    # «Tuning» (development, complete) left the index on 2 October 2026: method trials, not
    # results; pages and movies stay on disk. At the top, a table of contents linking every
    # section. Top-level sections: global VTCD, per-class VTCD, then the rest; inside, views
    # with the highest CNMF cap first, then with normals before without, then test before
    # validation.
    view_iterations: dict[str, int] = {}
    for entry in entries:
        perturbed_view = group_view.get(entry["row"]["group"])
        nmf = ((entry["meta"].get("explanation_settings") or {}).get("vtcd") or {}).get("nmf")
        if perturbed_view in view_tag and nmf and entry["row"]["group"].endswith("_vtcd"):
            # The maximum: a view being redone still carries old pages.
            view_iterations[perturbed_view] = max(
                view_iterations.get(perturbed_view, 0), int(nmf.get("max_iter", 0))
            )

    def scope_of(vista: str) -> str:
        return VTCD_TITLES.get(view_tag.get(vista, ""), ("",))[0]

    def view_key(vista: str) -> tuple:
        normals = VTCD_TITLES[view_tag[vista]][2]
        return (
            -view_iterations.get(vista, 0),
            not normals,
            view_tag[vista] in FIRST_SELECTION,
            # MIVIA-Crime right after its UCF-Crime view.
            vista.endswith("_mivia"),
            vista.endswith("_val"),
            vista,
        )

    counted = {v for v in views if scope_of(v)}
    sections = [
        (
            "vtcd-globale",
            "Global VTCD",
            sorted((v for v in counted if scope_of(v) == "global"), key=view_key),
        ),
        (
            "vtcd-per-classe",
            "Per-class VTCD",
            sorted((v for v in counted if scope_of(v) == "per class"), key=view_key),
        ),
        (
            "altre-raccolte",
            "Other collections",
            [*(v for v in subsequent if v not in counted), "localizzazione", "selezione", "corpus"],
        ),
    ]
    header, body, index_entries = body, "", []
    for section_anchor, section_title, groups in sections:
        section_entries = []
        for group in groups:
            if not any(r["group"] == group for r in pages):
                continue
            label = views[group][0]
            n = sum(r["group"] == group for r in pages)
            box = any(r.get("box") for r in pages if r["group"] == group)
            section_entries.append(
                (group, label, f"{n} pages · bounding boxes" if box else f"{n} pages")
            )
        if not section_entries:
            continue
        index_entries.append((section_anchor, section_title, section_entries))
        body += f"<h2 id='{section_anchor}'>{html.escape(section_title)}</h2>"
        for group, _, _ in section_entries:
            body += _index_section(
                group,
                views[group],
                [r for r in pages if r["group"] == group],
                view_iterations.get(group),
                column,
                performance,
            )
    # What is on the cluster and not here is declared: a local-only catalogue would suggest the
    # corpus does not exist.
    elsewhere = root / "rapporti/catalogo_cluster.json"
    if elsewhere.is_file():
        remote = json.loads(elsewhere.read_text(encoding="utf-8"))
        local_items = {(r["group"], r["model"], r["video"]) for r in pages}
        missing = [
            r
            for r in remote.get("pages", [])
            if r["model"] not in hidden_models
            and (r["group"], r["model"], r["video"]) not in local_items
        ]
        if missing:
            index_entries.append(("sul-cluster", "On the cluster, not here", []))
            body += (
                f"<h2 id='sul-cluster'>On the cluster, not here "
                f"<span class='muted'>({len(missing)} pages)</span></h2>"
                "<div class='notice'>The artifacts of these pages are in "
                "<code>$EVD_RUNS_DIR/analisi_explain/campagna_20260910</code> on the cluster: "
                "corpus movies and panels weigh tens of GB and are not replicated here. "
                "The numbers below come from the remote catalogue; to browse them, open "
                "<code>index.html</code> of that copy.</div>"
                f"<table><tr><th>Model</th><th>Video</th><th>Content</th>{column}</tr>"
            )
            for row_data in missing:
                search = html.escape(f"{row_data['model']} {row_data['video']}".lower(), quote=True)
                content = f"{row_data['windows']} windows · {len(row_data['methods'])} variants"
                # Performance is read from the scores, which are HERE, so the column also exists on rows
                # whose movies stayed on the cluster.
                body += (
                    f"<tr data-search='{search}'><td>{html.escape(row_data['model'])}</td>"
                    f"<td>{html.escape(row_data['video'])}</td>"
                    f"<td data-ord='{row_data['windows']}'>{html.escape(content)}</td>"
                    f"{_performance_td(performance, row_data['model'], row_data['video'])}</tr>"
                )
            body += "</table>"
            totals["remote_pages"] = len(missing)
    index = "".join(
        f"<li><a href='#{html.escape(g, quote=True)}'>{html.escape(title)}</a>"
        + (
            "<ul>"
            + "".join(
                f"<li><a href='#{html.escape(v, quote=True)}'>{html.escape(label)}</a> "
                f"<span class='muted'>({html.escape(n)})</span></li>"
                for v, label, n in below
            )
            + "</ul>"
            if below
            else ""
        )
        + "</li>"
        for g, title, below in index_entries
    )
    body = f"{header}<nav id='indice'><h2>Contents</h2><ul>{index}</ul></nav>{body}"
    body += "<p class='muted'>Every page lists the available artifacts and their diagnostics. Perturbation maps report the logit delta with respect to the original on the same compute host.</p>"
    body += (
        "<script>document.getElementById('filter').oninput=function(){"
        "const q=this.value.trim().toLowerCase();"
        "document.querySelectorAll('[data-search]').forEach(r=>r.hidden=!r.dataset.search.includes(q));"
        "}</script>"
    )
    write_text_atomic(root / "index.html", _page("Explainability V-JEPA 2.1", body))
    result = {"totals": totals, "campaigns": inventory, "pages": pages}
    write_json(root / "catalogo.json", result)
    return result
