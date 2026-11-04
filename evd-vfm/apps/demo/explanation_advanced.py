"""Global ablation movies and concept summaries; no model.

Global ablations show the original and the reordered samples with distinct scores; no
heatmap is attributed to a global perturbation.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import av
import cv2
import numpy as np
import yaml

from apps.demo.explanation import (
    _FONT_MONO,
    _LEGEND_H,
    COLORS,
    IG_BASELINE,
    STRIP_H,
    VIEWS,
    Header,
    _is_signed,
    _text,
    _truncate,
    _VideoWriter,
    assignment_of,
    base_method,
    blend_map,
    color_map,
    compose,
    compute_group_scales,
    covered_segments,
    header_height,
    interpolation_of,
    meaning_of,
    method_label,
    model_label,
    render_campaign,
    strip,
    strip_background,
    timeline_of,
    transparent_center,
    view_of,
)
from evd.exp.explanation_campaign import seed_from_window_id
from evd.explain.display import (
    ig_reference,
    window_from_grid,
    window_from_id,
    window_samples,
)
from evd.explain.perturbed import view_name
from evd.explain.project import offline_plane
from evd.types import WindowSpec
from evd.utils.hashing import file_sha256
from evd.utils.io import write_json, write_text_atomic


def requested_methods(root: Path) -> set[str]:
    """Read the methods requested by the active plan, without inferring them from the results."""
    from evd.explain.methods import ADVANCED_METHODS

    path = root / "rapporti/advanced_plan.json"
    if not path.exists():
        return set(ADVANCED_METHODS)
    plan = json.loads(path.read_text(encoding="utf-8"))
    return set(plan["campaigns"][0]["methods"])


def render_global_motion(folder: Path, *, anteprime: Path | None = None) -> dict:
    """Draw the seven global ablations with the per-window scores.

    :param folder: campaign with the global perturbations in its artifacts.
    :param anteprime: campaign to take ``preview_inputs/`` from when it is not the same (the
        sibling campaign on the SAME windows); ``_previews_of`` checks that windows match.
    :return: manifest of MP4 files and panels, after full decoding.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = json.loads((folder / "campaign.json").read_text(encoding="utf-8"))
    records = meta["windows"]
    if "motion_ablation" not in records[0]["checks"]:
        return {"videos": [], "images": []}
    windows = [
        WindowSpec(
            **{
                **r["window"],
                "explicit_frame_indices": tuple(r["window"]["explicit_frame_indices"]),
            }
        )
        for r in records
    ]
    previews = _previews_of(folder, anteprime or folder)
    assignments, _ = assignment_of(folder, meta, windows)
    segments = covered_segments(assignments)
    target = folder / "global_motion"
    target.mkdir(exist_ok=True)
    methods = [r["name"] for r in records[0]["checks"]["motion_ablation"]["globals"]]
    result = {
        "status": "running",
        "videos": [],
        "images": [],
        "methods": methods,
        "covered_frames": int(np.sum(assignments >= 0)),
        "source_frames": meta["num_frames"],
        "display": "preprocessed samples; nearest sample on the original time",
    }
    width = height = int(meta["preprocess"]["size"])
    line = timeline_of(folder, meta, records, assignments)
    for method_index, method in enumerate(methods):
        rows = [r["checks"]["motion_ablation"]["globals"][method_index] for r in records]
        for segment_index, (start, end) in enumerate(segments):
            path = target / f"{method}_segment_{segment_index:03d}.mp4"
            pending = path.with_name(f".{path.stem}.pending.mp4")
            writer = _VideoWriter(
                pending, meta["fps"], (width * 2, height + header_height(width * 2) + STRIP_H)
            )
            background = strip_background(width * 2, line)
            try:
                for frame in range(start, end + 1):
                    i = int(assignments[frame])
                    sample = offline_plane(windows[i], frame, len(previews[i]))
                    row = rows[i]
                    altered_sample = (
                        row.get("center_sample", sample)
                        if method == "freeze"
                        else row["order"][sample]
                    )
                    body = np.concatenate(
                        (previews[i][sample], previews[i][altered_sample]), axis=1
                    )
                    header_spec = Header(
                        metodo=f"Global ablation: {VIEWS.get(method, method)}",
                        famiglia="Input perturbations",
                        modello=model_label(meta["run_id"]),
                        video=Path(meta["video"]).stem,
                        dettaglio=(
                            f"left original - right perturbed - window "
                            f"{i + 1}/{len(records)} - sample {sample}"
                        ),
                        score=float(records[i]["score"]),
                        score_perturbato=float(row["score"]),
                        soglia=meta.get("threshold"),
                        chunk_positivo=line.label_of(i),
                        extra=(
                            f"logit delta {row['logit_difference']:+.4f} - "
                            "preprocessed samples, no motion interpolation"
                        ),
                    )
                    writer.write(compose(body, header_spec, strip(background, frame, line)))
            finally:
                writer.release()
            with av.open(str(pending)) as container:
                stream = container.streams.video[0]
                codec, pixel_format = stream.codec_context.name, stream.codec_context.pix_fmt
                count = sum(1 for _ in container.decode(video=0))
            if count != end - start + 1 or codec != "h264" or pixel_format != "yuv420p":
                raise ValueError(f"Invalid global movie: {pending}")
            os.replace(pending, path)
            result["videos"].append(
                {
                    "method": method,
                    "file": path.name,
                    "frames": count,
                    "start_frame": start,
                    "end_frame": end,
                    "codec": codec,
                    "pixel_format": pixel_format,
                    "sha256": file_sha256(path),
                }
            )
        fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
        times = [w.end_frame / meta["fps"] for w in windows]
        axes[0].plot(times, [r["score"] for r in records], label="Original")
        axes[0].plot(times, [r["score"] for r in rows], label=method)
        axes[0].axhline(meta["threshold"], ls="--", color="gray", label="Detector threshold")
        axes[0].set(ylabel="Score", ylim=(-0.03, 1.03), title=f"{folder.name} | {method}")
        axes[0].legend()
        axes[1].plot(times, [r["logit_difference"] for r in rows], marker="o")
        axes[1].axhline(0, color="gray", lw=0.8)
        axes[1].set(xlabel="Window end in the original video (s)", ylabel="Delta logit")
        fig.tight_layout()
        path = target / f"{method}_scores.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        result["images"].append(path.name)
    result["status"] = "complete"
    write_json(target / "manifest.json", result)
    return result


#: The four methods replacing an input volume: for each, the alternative input can be
#: rebuilt and shown next to the original.
COMPARISON_METHODS = ("occlusion_blur", "occlusion_mean", "occlusion_black", "motion_local_freeze")


def blur_kernel_of(meta: dict) -> int:
    """Return the side of the blur filter the campaign computed with.

    It lives at the top level of ``campaign.json`` (written by ``explanation_campaign``), not
    inside ``explanation_settings``, which only campaigns with a perturbation family carry.
    No fallback value: an invented kernel would rebuild a different IG reference from the
    one the maps were computed on.

    :param meta: the already deserialized ``campaign.json``.
    :return: the odd filter side, as used in the computation.
    :raises KeyError: if the receipt does not declare it in either place.
    """
    if "blur_kernel" in meta:
        return int(meta["blur_kernel"])
    settings = meta.get("explanation_settings") or {}
    if "blur_kernel" in settings:
        return int(settings["blur_kernel"])
    raise KeyError(
        "campaign.json does not declare blur_kernel: the receipt belongs to a campaign "
        "that is too old, and rebuilding the IG references without that value "
        "would produce movies that do not describe the saved maps"
    )


def visual_reference(frames: np.ndarray, fill: str, blur_kernel: int = 31) -> np.ndarray:
    """Rebuild in pixel space the reference the detector actually used.

    A second implementation of ``explain.perturbation.reference_clip`` (which works on the
    normalized tensor), operating on denormalized uint8 BGR samples because ``apps/`` does
    not import torch. ``test_il_riferimento_visivo_riproduce_quello_del_detector`` checks
    that they agree: ``mean`` and ``black`` exactly (the normalization is affine), ``blur``
    up to integer rounding.

    :param frames: preprocessed samples, ``[T, H, W, 3]`` uint8.
    :param fill: ``blur`` (box mean, not Gaussian), ``mean`` (spatial mean colour of the
        sample) or ``black`` (true black).
    :param blur_kernel: odd side of the mean filter, as in the campaign.
    :return: a reference with the same shape as ``frames``.
    :raises ValueError: if the fill cannot be rebuilt in pixel space.
    """
    if fill == "black":
        return np.zeros_like(frames)
    if fill == "mean":
        mean = frames.mean(axis=(1, 2), keepdims=True)
        return np.broadcast_to(np.rint(mean).astype(np.uint8), frames.shape).copy()
    if fill != "blur" or blur_kernel < 1 or blur_kernel % 2 != 1:
        raise ValueError(f"Fill cannot be rebuilt in pixel space: {fill}")
    # torch `avg_pool2d` with "reflect" padding is `cv2.blur` with BORDER_REFLECT_101.
    return np.stack(
        [cv2.blur(f, (blur_kernel, blur_kernel), borderType=cv2.BORDER_REFLECT_101) for f in frames]
    )


def perturbed_clip(
    frames: np.ndarray,
    mask: np.ndarray,
    metodo: str,
    blur_kernel: int = 31,
    *,
    riferimento: np.ndarray | None = None,
) -> np.ndarray:
    """Return the samples with the volume replaced, as the detector saw them.

    :param frames: preprocessed samples ``[T, H, W, 3]`` uint8.
    :param maschera: token mask ``[T', H', W']``, expanded by integer repetition, never by
        interpolation (as in ``volumes.masked_effects``).
    :param metodo: one of :data:`COMPARISON_METHODS`.
    :param riferimento: precomputed :func:`visual_reference` output for this clip and fill.
        It does not depend on the mask, so it is computed once instead of 45 times. Ignored
        by ``motion_local_freeze``, whose reference depends on the mask.
    :return: the alternative clip, same shape as ``frames``.
    :raises ValueError: if the grid does not divide the clip or the method is not
        perturbative.
    """
    if metodo not in COMPARISON_METHODS:
        raise ValueError(f"Method without an alternative input: {metodo}")
    expanded = mask
    for axis, (s, g) in enumerate(zip(frames.shape[:3], mask.shape, strict=True)):
        if s % g:
            raise ValueError("Token grid does not divide the clip pixels")
        expanded = np.repeat(expanded, s // g, axis=axis)
    if metodo == "motion_local_freeze":
        active = np.flatnonzero(expanded.any(axis=(1, 2)))
        center = (int(active[0]) + int(active[-1]) + 1) // 2
        source = np.broadcast_to(frames[center], frames.shape)
    elif riferimento is not None:
        source = riferimento
    else:
        source = visual_reference(frames, metodo.removeprefix("occlusion_"), blur_kernel)
    return np.where(expanded[..., None], source, frames)


def _previews_of(folder: Path, anteprime: Path) -> list[np.ndarray]:
    """Return the preprocessed samples of the campaign, checked against their sha256.

    :param folder: perturbative campaign, without previews.
    :param anteprime: sibling campaign with the previews, on the SAME windows.
    :return: one ``[T, H, W, 3]`` BGR array per window.
    :raises ValueError: if the windows differ between the two campaigns, or a preview was
        altered.
    """
    keys = [
        [
            w["window_id"]
            for w in json.loads((c / "campaign.json").read_text(encoding="utf-8"))["windows"]
        ]
        for c in (folder, anteprime)
    ]
    if keys[0] != keys[1]:
        raise ValueError(f"Previews of different windows: {anteprime} vs {folder}")
    frames = []
    manifest = json.loads((anteprime / "preview_manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["inputs"]:
        path = anteprime / entry["file"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Altered preview: {path}")
        with np.load(path, allow_pickle=False) as archive:
            frames.append(archive["rgb"][..., ::-1].copy())
    return frames


def _lut(metodo: str):
    return color_map(metodo, _is_signed(metodo))


def _bar(metodo: str) -> np.ndarray:
    """:return: the colormap bar of ``metodo``, as in the normal movies."""
    gradient = np.linspace(0, 1, 160)
    rgba = _lut(metodo)(gradient)
    rgb = rgba[:, :3] * rgba[:, 3:4] + np.asarray(COLORS["fondo"])[::-1] / 255 * (1 - rgba[:, 3:4])
    bar = (rgb * 255).astype(np.uint8)[:, ::-1]
    return np.tile(bar[None], (_LEGEND_H - 4, 1, 1))


def _colorize(piano: np.ndarray, lo: float, hi: float, metodo: str) -> np.ndarray:
    """:return: a map plane in BGR, with the common campaign scale."""
    if np.ptp(piano) == 0 and not transparent_center(metodo):
        return np.full((*piano.shape[:2], 3), 160, dtype=np.uint8)
    normalized = np.clip((piano - lo) / max(hi - lo, 1e-12), 0, 1)
    rgba = _lut(metodo)(normalized)
    rgb = rgba[..., :3] * rgba[..., 3:4] + np.asarray(COLORS["fondo"])[::-1] / 255 * (
        1 - rgba[..., 3:4]
    )
    return (rgb * 255).astype(np.uint8)[..., ::-1].copy()


def render_comparisons(folder: Path, *, anteprime: Path) -> dict:
    """Render original, perturbed input and map side by side, with the scores of both sides.

    The middle panel is the true alternative input, rebuilt from the saved mask, and the
    header carries both scores with their difference. Per window the maximum-effect mask
    among the LOCALIZED ones is shown (the smallest family, 16x6x6, 576 tokens): the maximum
    over all 45 would almost always be one of the eleven unions, which removes more. The
    global maximum stays in the manifest, in ``global_maximum``.

    :param folder: campaign with ``artifacts/w*_{occlusion,motion_ablation}.npz``.
    :param anteprime: sibling campaign carrying ``preview_inputs/``, on the same windows.
    :return: manifest of the produced movies, with the chosen mask per window.
    :raises ValueError: if the campaign has no perturbations or artifacts are inconsistent.
    """
    meta = json.loads((folder / "campaign.json").read_text(encoding="utf-8"))
    records = meta["windows"]
    if "occlusion" not in records[0]["checks"]:
        return {"videos": [], "images": [], "methods": []}
    windows = [
        WindowSpec(
            **{
                **r["window"],
                "explicit_frame_indices": tuple(r["window"]["explicit_frame_indices"]),
            }
        )
        for r in records
    ]
    previews = _previews_of(folder, anteprime)
    scale = json.loads((folder / "render/scales.json").read_text(encoding="utf-8"))
    regions = records[0]["checks"]["occlusion"]["regions"]
    # The smallest family is the one with the fewest tokens per test, derived from the
    # artifacts instead of written.
    local_family = min(regions, key=lambda r: r["volume_tokens"])["family"]
    kernel = blur_kernel_of(meta)
    assignments, _ = assignment_of(folder, meta, windows)
    segments = covered_segments(assignments)
    full = segments == [(0, meta["num_frames"] - 1)]
    target = folder / "confronti"
    target.mkdir(exist_ok=True)
    side = int(meta["preprocess"]["size"])
    line = timeline_of(folder, meta, records, assignments)
    result = {
        "status": "running",
        "videos": [],
        "images": [],
        "methods": list(COMPARISON_METHODS),
        "covered_frames": int(np.sum(assignments >= 0)),
        "source_frames": meta["num_frames"],
        "display": "preprocessed 384x384 samples, not the native frame: the model input",
        "mask_choice": (
            "per window, the maximum-effect mask among the localized ones only "
            f"(family {local_family}); the overall maximum is in global_maximum"
        ),
        # Group/model/video and not just the name: the same video exists in four groups.
        "preview_source": "/".join(
            (anteprime.parent.parent.name, anteprime.parent.name, anteprime.name)
        ),
        "chosen_masks": {},
    }
    for method in COMPARISON_METHODS:
        suffix = (
            "local_freeze" if method == "motion_local_freeze" else method.removeprefix("occlusion_")
        )
        artifact = "motion_ablation" if method == "motion_local_freeze" else "occlusion"
        choices, maps = [], []
        localized = np.array([r["family"] == local_family for r in regions])
        for i in range(len(records)):
            with np.load(folder / f"artifacts/w{i:05d}_{artifact}.npz") as archive:
                effects = archive[f"effects_{suffix}"]
                k = int(np.flatnonzero(localized)[np.argmax(effects[localized])])
                g = int(np.argmax(effects))
                choices.append(
                    {
                        "window": i,
                        "mask": regions[k]["name"],
                        "volume_tokens": int(regions[k]["volume_tokens"]),
                        "effect": float(effects[k]),
                        "logit": float(archive[f"logits_{suffix}"][k]),
                        "score": float(archive[f"scores_{suffix}"][k]),
                        "mask_index": k,
                        "global_maximum": {
                            "mask": regions[g]["name"],
                            "volume_tokens": int(regions[g]["volume_tokens"]),
                            "effect": float(effects[g]),
                        },
                    }
                )
                maps.append(archive["masks"][k].copy())
            with np.load(folder / f"maps_{i:05d}.npz") as archive:
                choices[-1]["_piani"] = archive[method].copy()
        result["chosen_masks"][method] = [
            {c: v for c, v in s.items() if not c.startswith("_")} for s in choices
        ]
        lo, hi = float(scale[method]["lo"]), float(scale[method]["hi"])
        bar, extremes = _bar(method), tuple(scale[method]["legend_range"])
        for segment_index, (start, end) in enumerate(segments):
            name = f"{method}" + ("" if full else f"_clip_{segment_index:03d}") + "_confronto.mp4"
            path = target / name
            pending = path.with_name(f".{path.stem}.pending.mp4")
            writer = _VideoWriter(
                pending, meta["fps"], (side * 3, side + header_height(side * 3) + STRIP_H)
            )
            background = strip_background(side * 3, line)
            cache: tuple[int, np.ndarray] | None = None
            try:
                for frame in range(start, end + 1):
                    i = int(assignments[frame])
                    if cache is None or cache[0] != i:
                        cache = (i, perturbed_clip(previews[i], maps[i], method, kernel))
                    sample = offline_plane(windows[i], frame, len(previews[i]))
                    plane = offline_plane(windows[i], frame, len(choices[i]["_piani"]))
                    heat = cv2.resize(
                        _colorize(choices[i]["_piani"][plane], lo, hi, method),
                        (side, side),
                        interpolation=interpolation_of(method),
                    )
                    body = np.concatenate((previews[i][sample], cache[1][sample], heat), axis=1)
                    s = choices[i]
                    header_spec = Header(
                        metodo=method_label(method),
                        famiglia="Input perturbations - original, perturbed, map",
                        modello=model_label(meta["run_id"]),
                        video=Path(meta["video"]).stem,
                        dettaglio=(
                            f"window {i + 1}/{len(records)} - volume t {plane} - "
                            f"{'whole video' if full else f'clip {start}-{end}'}"
                        ),
                        significato=meaning_of(method),
                        score=float(records[i]["score"]),
                        score_perturbato=float(s["score"]),
                        soglia=meta.get("threshold"),
                        chunk_positivo=line.label_of(i),
                        legenda=bar,
                        estremi=extremes,
                        extra=(
                            f"mask {s['mask']} ({s['volume_tokens']} tokens) - "
                            f"delta logit {s['effect']:+.4f}"
                        ),
                    )
                    writer.write(compose(body, header_spec, strip(background, frame, line)))
            finally:
                writer.release()
            with av.open(str(pending)) as container:
                stream = container.streams.video[0]
                codec, pixel_format = stream.codec_context.name, stream.codec_context.pix_fmt
                count = sum(1 for _ in container.decode(video=0))
            if count != end - start + 1 or codec != "h264" or pixel_format != "yuv420p":
                raise ValueError(f"Invalid comparison movie: {pending}")
            os.replace(pending, path)
            result["videos"].append(
                {
                    "method": method,
                    "file": path.name,
                    "frames": count,
                    "start_frame": start,
                    "end_frame": end,
                    "codec": codec,
                    "pixel_format": pixel_format,
                    "sha256": file_sha256(path),
                }
            )
    result["status"] = "complete"
    write_json(target / "manifest.json", result)
    return result


#: Readable label of every mask family. The technical name states the geometry
#: (``16x6x6_s16x4x4`` = 16x6x6-token volume, stride 16x4x4) and stays in the artifacts.
MASK_FAMILIES = {
    "16x6x6_s16x4x4": "localized volumes, 6x6 tokens per frame",
    "16x8x8_s16x6x6": "wide volumes, 8x8 tokens per frame",
    "special": "unions: bands, half frame, diagonals",
}

#: Memory budget of the visual-reference cache of a window (a reference is ~14 MB);
#: beyond it the reference is recomputed rather than risking OOM.
_REFERENCES_CAP = 256 * 1024 * 1024


def reuse_manifest(target: Path, force: bool) -> dict[str, dict]:
    """Return the rows of the previous manifest whose movies are still on disk.

    Makes a folder of hundreds of movies restartable, reusing on the file existence (the
    rule of ``scripts/render_spiegazioni.py``). Changes of colour scale or header propagate
    only with ``force``.

    :param target: folder containing ``manifest.json`` and the movies.
    :param force: when true nothing is reused.
    :return: reusable rows, keyed by their ``file``.
    """
    path = target / "manifest.json"
    if force or not path.is_file():
        return {}
    old = json.loads(path.read_text(encoding="utf-8"))
    return {row["file"]: row for row in old.get("videos", []) if (target / row["file"]).is_file()}


def mask_layout(regions: list[dict]) -> dict:
    """Return how the masks of every family are laid out, read from their boxes.

    Localized families tile the frame, so they have a natural grid used by the page to lay
    out the links (the position of the movie is the position of the occluded region). It
    is read from the ``boxes``, not from the name (as :func:`evd.explain.selection.local_family`).
    Families that do not tile (the ``special`` unions) get the plain list only.

    :param regions: test descriptions, in artifact order.
    :return: per family, ``indici`` and, when it exists, ``righe``, ``colonne`` and
        ``griglia`` (the index matrix in spatial order).
    """
    per_family: dict[str, list[int]] = {}
    for k, region in enumerate(regions):
        per_family.setdefault(region["family"], []).append(k)
    layout: dict[str, dict] = {}
    for family, indices in per_family.items():
        layout[family] = {"indici": indices}
        if any(len(regions[k]["boxes"]) != 1 for k in indices):
            continue
        ys = sorted({regions[k]["boxes"][0][2] for k in indices})
        xs = sorted({regions[k]["boxes"][0][4] for k in indices})
        if len(ys) * len(xs) != len(indices):
            continue
        grid: list[list[int | None]] = [[None] * len(xs) for _ in ys]
        for k in indices:
            frame_box = regions[k]["boxes"][0]
            grid[ys.index(frame_box[2])][xs.index(frame_box[4])] = k
        if any(cell is None for row in grid for cell in row):
            continue
        layout[family].update(righe=len(ys), colonne=len(xs), griglia=grid)
    return layout


_MASK_CONTEXT: dict | None = None


def _render_mask(job: tuple) -> dict:
    """Render one independent movie; only the parent updates the manifest."""
    c = _MASK_CONTEXT
    if c is None:
        raise RuntimeError("Mask context not initialized")
    k, region, start, end, name, relative = job
    folder, meta, records = c["cartella"], c["meta"], c["records"]
    previews, windows, assignments = c["previews"], c["windows"], c["assignments"]
    masks, effects = c["maschere"], c["effetti"]
    masked_scores = c["punteggi_mascherati"]
    method, suffix, kernel = c["metodo"], c["suffisso"], c["kernel"]
    side, threshold, line, background = c["lato"], c["soglia"], c["linea"], c["fondo"]
    full, regions = c["full"], c["regions"]
    references, spent = c["riferimenti"], c["speso"]
    short = region["name"].removeprefix(region["family"] + "_") or region["name"]
    path = folder / name
    pending = path.with_name(f".{path.stem}.pending.mp4")
    writer = _VideoWriter(
        pending, meta["fps"], (side * 2, side + header_height(side * 2) + STRIP_H)
    )
    cache: tuple[int, np.ndarray] | None = None
    try:
        for frame in range(start, end + 1):
            i = int(assignments[frame])
            if cache is None or cache[0] != i:
                visual = references.get(i)
                if visual is None and method != "motion_local_freeze":
                    visual = visual_reference(previews[i], suffix, kernel)
                    if spent + visual.nbytes <= _REFERENCES_CAP:
                        references[i] = visual
                        spent += visual.nbytes
                cache = (
                    i,
                    perturbed_clip(
                        previews[i],
                        masks[i][k],
                        method,
                        kernel,
                        riferimento=visual,
                    ),
                )
            sample = offline_plane(windows[i], frame, len(previews[i]))
            body = np.concatenate((previews[i][sample], cache[1][sample]), axis=1)
            header_spec = Header(
                metodo=method_label(method),
                famiglia=MASK_FAMILIES.get(region["family"], region["family"]),
                modello=model_label(meta["run_id"]),
                video=Path(meta["video"]).stem,
                # The name drops the family prefix, already written in full on the row above.
                dettaglio=(
                    f"mask {k + 1}/{len(regions)} {short} - "
                    f"window {i + 1}/{len(records)} - "
                    f"{'whole video' if full else f'clip {start}-{end}'}"
                ),
                score=float(records[i]["score"]),
                score_perturbato=float(masked_scores[i][k]),
                soglia=threshold,
                chunk_positivo=line.label_of(i),
                extra=(
                    f"{int(region['volume_tokens'])} tokens occluded out of "
                    f"{masks[i][k].size} - delta logit {float(effects[i][k]):+.4f}"
                ),
            )
            writer.write(compose(body, header_spec, strip(background, frame, line)))
    finally:
        writer.release()
    with av.open(str(pending)) as container:
        stream = container.streams.video[0]
        codec = stream.codec_context.name
        pixel_format = stream.codec_context.pix_fmt
        count = sum(1 for _ in container.decode(video=0))
    if count != end - start + 1 or codec != "h264" or pixel_format != "yuv420p":
        raise ValueError(f"Invalid per-mask movie: {pending}")
    os.replace(pending, path)
    column = np.array([row_data[k] for row_data in masked_scores], dtype=float)
    delta = np.array([row_data[k] for row_data in effects], dtype=float)
    row = {
        "method": method,
        "mask_index": k,
        "mask": region["name"],
        "family": region["family"],
        "volume_tokens": int(region["volume_tokens"]),
        "file": relative,
        "frames": count,
        "start_frame": start,
        "end_frame": end,
        "codec": codec,
        "pixel_format": pixel_format,
        "sha256": file_sha256(path),
        "perturbed_score_min": float(column.min()),
        "perturbed_score_max": float(column.max()),
        "effect_mean": float(delta.mean()),
        "effect_max": float(delta.max()),
    }
    c["speso"] = spent
    return row


def render_comparisons_per_mask(
    folder: Path, *, anteprime: Path, force: bool = False, jobs: int = 1
) -> dict:
    """Render one movie for every mask: original on the left, perturbed on the right.

    :func:`render_comparisons` shows the maximum-effect mask per window; this answers "what
    happens when removing this region", for each of the 45 measured masks. Two panels: the
    method map aggregates all masks and would be the same image in all 45 movies.

    :param folder: campaign with ``artifacts/w*_{occlusion,motion_ablation}.npz``.
    :param anteprime: sibling campaign carrying ``preview_inputs/``.
    :param force: also rewrite movies already in the previous manifest.
    :param jobs: processes per movie on Linux; sequential by default.
    :return: manifest of the movies, with masks, families and layout.
    :raises ValueError: if a movie cannot be verified after writing.
    """
    meta = json.loads((folder / "campaign.json").read_text(encoding="utf-8"))
    records = meta["windows"]
    if "occlusion" not in records[0]["checks"]:
        return {"videos": [], "images": [], "methods": []}
    windows = [
        WindowSpec(
            **{
                **r["window"],
                "explicit_frame_indices": tuple(r["window"]["explicit_frame_indices"]),
            }
        )
        for r in records
    ]
    previews = _previews_of(folder, anteprime)
    regions = records[0]["checks"]["occlusion"]["regions"]
    kernel = blur_kernel_of(meta)
    assignments, _ = assignment_of(folder, meta, windows)
    segments = covered_segments(assignments)
    full = segments == [(0, meta["num_frames"] - 1)]
    side = int(meta["preprocess"]["size"])
    threshold = meta.get("threshold")
    line = timeline_of(folder, meta, records, assignments)
    target = folder / "per_maschera"
    target.mkdir(exist_ok=True)
    layout = mask_layout(regions)
    manifest_path = target / "manifest.json"
    previous_items = reuse_manifest(target, force)
    result = {
        "status": "running",
        "videos": [],
        "images": [],
        "methods": list(COMPARISON_METHODS),
        "covered_frames": int(np.sum(assignments >= 0)),
        "source_frames": meta["num_frames"],
        "display": "preprocessed 384x384 samples, not the native frame: the model input",
        "panels": "left original, right perturbed input; no map",
        "mask_choice": (
            f"one mask per movie, all {len(regions)}: no selection, the viewer chooses"
        ),
        "preview_source": "/".join(
            (anteprime.parent.parent.name, anteprime.parent.name, anteprime.name)
        ),
        "families": {
            family: {
                "label": MASK_FAMILIES.get(family, family),
                "indices": data["indici"],
                **{key: value for key, value in data.items() if key != "indici"},
            }
            for family, data in layout.items()
        },
        "masks": [
            {
                "index": k,
                "name": region["name"],
                "family": region["family"],
                "volume_tokens": int(region["volume_tokens"]),
            }
            for k, region in enumerate(regions)
        ],
    }
    for method in COMPARISON_METHODS:
        suffix = (
            "local_freeze" if method == "motion_local_freeze" else method.removeprefix("occlusion_")
        )
        artifact = "motion_ablation" if method == "motion_local_freeze" else "occlusion"
        masks, effects, masked_scores = [], [], []
        for i in range(len(records)):
            with np.load(folder / f"artifacts/w{i:05d}_{artifact}.npz") as archive:
                masks.append(archive["masks"].copy())
                effects.append(archive[f"effects_{suffix}"].copy())
                masked_scores.append(archive[f"scores_{suffix}"].copy())
        folder_path = target / method
        folder_path.mkdir(exist_ok=True)
        background = strip_background(side * 2, line)
        job_list = []
        for k, region in enumerate(regions):
            for segment_index, (start, end) in enumerate(segments):
                name = f"m{k:02d}" + ("" if full else f"_clip_{segment_index:03d}") + ".mp4"
                relative = f"{method}/{name}"
                if relative in previous_items:
                    result["videos"].append(previous_items[relative])
                else:
                    job_list.append((k, region, start, end, name, relative))
        global _MASK_CONTEXT
        _MASK_CONTEXT = {
            "cartella": folder_path,
            "meta": meta,
            "records": records,
            "previews": previews,
            "windows": windows,
            "assignments": assignments,
            "maschere": masks,
            "effetti": effects,
            "punteggi_mascherati": masked_scores,
            "metodo": method,
            "suffisso": suffix,
            "kernel": kernel,
            "lato": side,
            "soglia": threshold,
            "linea": line,
            "fondo": background,
            "full": full,
            "regions": regions,
            "riferimenti": {},
            "speso": 0,
        }

        def record(row):
            result["videos"].append(row)
            write_json(manifest_path, result)

        try:
            if jobs > 1 and len(job_list) > 1 and _fork_available():
                import multiprocessing
                from concurrent.futures import ProcessPoolExecutor, as_completed

                with ProcessPoolExecutor(
                    max_workers=min(jobs, len(job_list)),
                    mp_context=multiprocessing.get_context("fork"),
                ) as pool:
                    futures = [pool.submit(_render_mask, task) for task in job_list]
                    for future in as_completed(futures):
                        record(future.result())
            else:
                for task in job_list:
                    record(_render_mask(task))
        finally:
            _MASK_CONTEXT = None
    result["videos"].sort(key=lambda row: row["file"])
    result["status"] = "complete"
    write_json(manifest_path, result)
    return result


#: IG baselines with a single reference, with their receipt suffix. ``ig_ensemble`` is
#: absent: an average of five attributions has no single reference.
BASELINE_IG = ("blur", "mean", "black", "noise", "tn_same_video")


def _declared_by_sibling(campagna: Path, video_id: str, window_id: str) -> tuple:
    """Return window and video path from a campaign that has that video.

    If the TN reference is itself an explained video, its campaign carries the verbatim
    ``explicit_frame_indices``, the most authoritative source. The canonical group is looked
    up from the model of the requesting campaign, not a hard-coded name.

    :param campagna: the campaign folder being rendered.
    :param video_id: the TN reference video.
    :param window_id: the window id looked for.
    :return: ``(window | None, video paths found)``.
    """
    window = None
    found: list[Path] = []
    root = campagna.parents[3]
    models = dict.fromkeys((campagna.parent.name, "dora", "congelato"))
    groups = ("selezione_classi", "selezione_concetti", "selezione_avanzati", "selezione")
    for group in groups:
        for model in models:
            reference = root / "risultati" / group / model / video_id / "campaign.json"
            if not reference.is_file():
                continue
            declaration = json.loads(reference.read_text(encoding="utf-8"))
            for row in declaration.get("windows", []):
                window_json = row.get("window", row)
                if row.get("window_id", window_json.get("window_id")) == window_id:
                    window = WindowSpec(
                        **{
                            **window_json,
                            "explicit_frame_indices": tuple(window_json["explicit_frame_indices"]),
                        }
                    )
                    break
            video_path = declaration.get("video")
            if video_path and Path(video_path).is_file():
                found.append(Path(video_path))
            if window is not None and found:
                return window, found
    return window, found


def _tn_window(
    window_id: str, video: Path, meta: dict, modello: WindowSpec, campagna: Path | None
) -> WindowSpec:
    """Return the TN reference window, from its video grid when possible.

    The fallback on :func:`evd.explain.display.window_from_id`, shifting the offsets of
    ``modello``, holds only with the same temporal profile (native fps); it is kept as a last
    resort for campaigns without ``source_config.yaml``.

    :param window_id: the TN window id, from the ``ig_tn_same_video`` receipt.
    :param video: the reference video file, already resolved.
    :param meta: ``campaign.json`` of the campaign being rendered.
    :param modello: a campaign window, for the fallback only.
    :param campagna: the campaign folder, where ``source_config.yaml`` lives.
    :return: the window with exactly ``window_id``.
    """
    config = None if campagna is None else campagna / "source_config.yaml"
    if config is not None and config.is_file():
        protocol = (yaml.safe_load(config.read_text(encoding="utf-8")) or {}).get("protocol")
        if protocol:
            return window_from_grid(
                window_id,
                video,
                protocol,
                meta["preprocess"],
                validazione=str(meta.get("split", "test")) == "val",
            )
    return window_from_id(window_id, modello)


def _tn_reference(
    ricevuta: dict,
    modello: WindowSpec,
    meta: dict,
    video_root: Path,
    *,
    campagna: Path | None = None,
) -> np.ndarray:
    """Return the samples of the normal window used by IG as reference.

    The ``window_id`` carries the temporal profile as a non-invertible hash, so the window
    is rebuilt from three sources, by decreasing authority: a sibling campaign that already
    has that video (verbatim indices), the grid of the reference video
    (:func:`evd.explain.display.window_from_grid`), and shifting the offsets of a window of
    the current campaign (which assumes the same native fps).

    :raises FileNotFoundError: if the reference video is missing or ambiguous.
    """
    video_id, window_id = ricevuta["tn_video_id"], ricevuta["tn_window_id"]
    window, found = (None, [])
    if campagna is not None:
        window, found = _declared_by_sibling(campagna, video_id, window_id)
    if not found:
        found = list(video_root.rglob(f"{video_id}.mp4"))
    found = list(dict.fromkeys(p.resolve() for p in found))
    if len(found) > 1:
        # UCF-Crime sometimes contains duplicates of the same file in different folders; with
        # identical content the provenance is unique and rendering is not blocked.
        per_hash = {}
        for path in found:
            per_hash.setdefault(file_sha256(path), path)
        found = list(per_hash.values())
    if len(found) != 1:
        raise FileNotFoundError(
            f"TN reference video '{video_id}' not found "
            f"(or duplicated) under {video_root}: {len(found)} matches"
        )
    if window is None:
        window = _tn_window(window_id, found[0], meta, modello, campagna)
    return window_samples(found[0], window, meta["preprocess"])


def render_ig_comparisons(
    folder: Path, *, anteprime: Path, video_root: Path | None = None, force: bool = False
) -> dict:
    """Render original, reference and map side by side, for every IG baseline.

    An Integrated Gradients attribution integrates the gradient along the segment from a
    reference to the true clip; the middle panel is the reference, and the header carries
    the scores of the clip and of the reference. The reference is recomputed with the
    detector function (:func:`evd.explain.display.ig_reference`), not rewritten. IG maps and
    receipts live in the base campaign, preprocessed samples in the perturbative one;
    ``_previews_of`` checks that windows match.

    :param folder: campaign with ``maps_*.npz`` containing ``ig_*`` and the ``checks.ig_*``
        receipts.
    :param anteprime: sibling campaign with ``preview_inputs/``, same windows.
    :param video_root: root under which to look for the TN reference video; by default the
        folder containing the campaign video folder.
    :param force: also rewrite movies already in the previous manifest.
    :return: manifest of the movies, with the diagnostics of every reference.
    :raises ValueError: if a movie cannot be verified after writing.
    """
    meta = json.loads((folder / "campaign.json").read_text(encoding="utf-8"))
    records = meta["windows"]
    present, without_provenance = [], {}
    for fill in BASELINE_IG:
        receipts = [r.get("checks", {}).get(f"ig_{fill}") for r in records]
        if any(r is None for r in receipts):
            continue
        # `tn_same_video` is the only baseline whose reference is another true clip, known only
        # from the receipt provenance. Campaigns before 11 September 2026 do not record it: the
        # reference is then declared not drawable instead of guessed.
        if fill == "tn_same_video" and any("tn_window_id" not in r for r in receipts):
            without_provenance[f"ig_{fill}"] = (
                "reference not drawable: this campaign does not record which normal "
                "window it used (provenance introduced on 11 September 2026)"
            )
            continue
        present.append(fill)
    if not present:
        return {"videos": [], "images": [], "methods": [], "excluded": without_provenance}
    windows = [
        WindowSpec(
            **{
                **r["window"],
                "explicit_frame_indices": tuple(r["window"]["explicit_frame_indices"]),
            }
        )
        for r in records
    ]
    previews = _previews_of(folder, anteprime)
    scale = json.loads((folder / "render/scales.json").read_text(encoding="utf-8"))
    preprocess = meta["preprocess"]
    mean, deviation = tuple(preprocess["mean"]), tuple(preprocess["std"])
    kernel = blur_kernel_of(meta)
    root = video_root or Path(meta["video"]).parents[1]
    assignments, _ = assignment_of(folder, meta, windows)
    segments = covered_segments(assignments)
    full = segments == [(0, meta["num_frames"] - 1)]
    side = int(preprocess["size"])
    threshold = meta.get("threshold")
    line = timeline_of(folder, meta, records, assignments)
    target = folder / "confronti_ig"
    target.mkdir(exist_ok=True)
    previous_items = reuse_manifest(target, force)
    result = {
        "status": "running",
        "videos": [],
        "images": [],
        "methods": [f"ig_{f}" for f in present],
        "covered_frames": int(np.sum(assignments >= 0)),
        "source_frames": meta["num_frames"],
        "display": "preprocessed 384x384 samples, not the native frame: the model input",
        "panels": "left true clip, centre baseline reference, right IG map",
        "baseline_source": (
            "rebuilt with evd.explain.perturbation.reference_clip, the detector function"
        ),
        "excluded": {
            "ig_ensemble": "average of five attributions: it has no single reference",
            **without_provenance,
        },
        "preview_source": "/".join(
            (anteprime.parent.parent.name, anteprime.parent.name, anteprime.name)
        ),
        "references": {},
    }
    background = strip_background(side * 3, line)
    for fill in present:
        map_name = f"ig_{fill}"
        if map_name not in scale:
            continue
        lo, hi = float(scale[map_name]["lo"]), float(scale[map_name]["hi"])
        bar, extremes = _bar(map_name), tuple(scale[map_name]["legend_range"])
        # Reference and map of every window, computed once.
        references, diagnostics, planes, reference_scores = [], [], [], []
        cache_tn: dict[str, np.ndarray] = {}
        for i, record in enumerate(records):
            receipt = record["checks"][map_name]
            if fill == "tn_same_video":
                key = receipt["tn_window_id"]
                if key not in cache_tn:
                    cache_tn[key] = _tn_reference(receipt, windows[i], meta, root, campagna=folder)
                samples = cache_tn[key]
                diagnostic = {
                    "fill": fill,
                    "clipped_fraction": 0.0,
                    "exact": True,
                    "tn_video_id": receipt["tn_video_id"],
                    "tn_window_id": key,
                    "tn_source": receipt.get("tn_source"),
                }
            else:
                samples, diagnostic = ig_reference(
                    previews[i],
                    fill,
                    mean=mean,
                    std=deviation,
                    blur_kernel=kernel,
                    seed=seed_from_window_id(record["window_id"]),
                )
            references.append(samples)
            diagnostics.append(diagnostic)
            # The reference logit is not measured again: `logit_difference` is F(clip) - F(reference).
            logit = float(record["logit"]) - float(receipt["logit_difference"])
            reference_scores.append(1.0 / (1.0 + math.exp(-logit)))
            with np.load(folder / f"maps_{i:05d}.npz") as archive:
                planes.append(archive[map_name].copy())
        result["references"][map_name] = [
            {**d, "reference_score": s} for d, s in zip(diagnostics, reference_scores, strict=True)
        ]
        for segment_index, (start, end) in enumerate(segments):
            name = map_name + ("" if full else f"_clip_{segment_index:03d}") + ".mp4"
            if name in previous_items:
                result["videos"].append(previous_items[name])
                continue
            path = target / name
            pending = path.with_name(f".{path.stem}.pending.mp4")
            writer = _VideoWriter(
                pending, meta["fps"], (side * 3, side + header_height(side * 3) + STRIP_H)
            )
            try:
                for frame in range(start, end + 1):
                    i = int(assignments[frame])
                    sample = offline_plane(windows[i], frame, len(previews[i]))
                    plane = offline_plane(windows[i], frame, len(planes[i]))
                    heat = cv2.resize(
                        _colorize(planes[i][plane], lo, hi, map_name),
                        (side, side),
                        interpolation=interpolation_of(map_name),
                    )
                    # The TN reference is a different clip: its `sample` is the temporal counterpart, not the
                    # same instant.
                    ref = references[i][min(sample, len(references[i]) - 1)]
                    body = np.concatenate((previews[i][sample], ref, heat), axis=1)
                    d = diagnostics[i]
                    reference_detail = (
                        f"reference: {d['tn_video_id']}"
                        if fill == "tn_same_video"
                        else f"reference {fill}"
                    )
                    writer.write(
                        compose(
                            body,
                            Header(
                                metodo=method_label(map_name),
                                famiglia="Integrated Gradients - clip, reference, map",
                                modello=model_label(meta["run_id"]),
                                video=Path(meta["video"]).stem,
                                dettaglio=(
                                    f"window {i + 1}/{len(records)} - {reference_detail} - "
                                    f"{'whole video' if full else f'clip {start}-{end}'}"
                                ),
                                significato=meaning_of(map_name),
                                score=float(records[i]["score"]),
                                score_perturbato=float(reference_scores[i]),
                                etichetta_perturbato="REFERENCE",
                                soglia=threshold,
                                chunk_positivo=line.label_of(i),
                                legenda=bar,
                                estremi=extremes,
                                extra=(
                                    f"delta logit {float(records[i]['checks'][map_name]['logit_difference']):+.4f}"
                                    + (
                                        f" - {d['clipped_fraction']:.1%} of the reference clipped "
                                        "to be drawable"
                                        if d["clipped_fraction"] > 0.001
                                        else ""
                                    )
                                ),
                            ),
                            strip(background, frame, line),
                        )
                    )
            finally:
                writer.release()
            with av.open(str(pending)) as container:
                stream = container.streams.video[0]
                codec, pixel_format = stream.codec_context.name, stream.codec_context.pix_fmt
                count = sum(1 for _ in container.decode(video=0))
            if count != end - start + 1 or codec != "h264" or pixel_format != "yuv420p":
                raise ValueError(f"Invalid IG movie: {pending}")
            os.replace(pending, path)
            result["videos"].append(
                {
                    "method": map_name,
                    "baseline": fill,
                    "file": name,
                    "frames": count,
                    "start_frame": start,
                    "end_frame": end,
                    "codec": codec,
                    "pixel_format": pixel_format,
                    "sha256": file_sha256(path),
                }
            )
            write_json(target / "manifest.json", result)
    result["status"] = "complete"
    write_json(target / "manifest.json", result)
    return result


def _perturbed_score(ricevuta: dict, vista: str) -> float:
    """Return the score the detector gave to the perturbed input.

    The same quantity has two names: ``perturbed_views`` writes ``score`` and the 20
    September 2026 VTCD0 campaign writes ``perturbed_score``; the first present is read
    (both exist and match bit for bit on occlusion rows). ``or`` is not used: a score may be
    0.0, and on a global view ``perturbed_score`` exists and is ``None``.

    :param ricevuta: the row of ``checks.perturbed_views.views[vista]``.
    :param vista: the view name, for the error.
    :return: the score, in [0, 1].
    :raises KeyError: if the row carries neither field.
    """
    for field in ("score", "perturbed_score"):
        value = ricevuta.get(field)
        if value is not None:
            return float(value)
    raise KeyError(
        f"View '{vista}' does not declare the score of the perturbed input: "
        f"neither 'score' nor 'perturbed_score' among {sorted(ricevuta)}"
    )


def _view_clip(
    frames: np.ndarray, vista: str, ricevuta: dict, mask: np.ndarray | None, kernel: int
) -> np.ndarray:
    """Return the samples as the detector saw them under that perturbation.

    Localized views are rebuilt from the saved mask; global ones reorder the samples with
    the order recorded in the receipt.

    :param ricevuta: the row of ``checks.perturbed_views.views[vista]``.
    :return: the alternative clip, same shape as ``frames``.
    :raises ValueError: on an unknown view, or a missing mask where needed.
    """
    if ricevuta.get("kind") == "locale":
        if mask is None:
            raise ValueError(f"Localized view without a saved mask: {vista}")
        mode = ricevuta.get("mode")
        method = f"occlusion_{mode}" if mode in {"blur", "mean", "black"} else vista
        return perturbed_clip(frames, mask, method, kernel)
    if vista == "freeze":
        center = int(ricevuta["center_sample"])
        return np.broadcast_to(frames[center], frames.shape)
    order = ricevuta.get("order")
    if order is None:
        raise ValueError(f"Global view without a temporal order: {vista}")
    return frames[list(order)]


#: What a perturbed-view movie needs, shared with child processes by fork (a long video
#: needs a few GB, too much for a pipe). Valid only inside
#: :func:`render_perturbed_views`, which clears it when done.
_VIEW_CONTEXT: dict | None = None


def _fork_available() -> bool:
    """:return: whether the system can create processes with ``fork`` (Linux, not Windows)."""
    import multiprocessing

    return "fork" in multiprocessing.get_all_start_methods()


def _render_view(job: tuple[str, int, int, int]) -> dict:
    """Render one perturbed-view movie and return its manifest row.

    Module-level because with several workers it runs in another process, reading the
    context from :data:`_VIEW_CONTEXT`, filled by the parent before creating the pool.

    :param job: ``(name, segment index, first frame, last frame)``.
    :return: the manifest row, with codec and frame count.
    :raises ValueError: if the written movie lacks the expected frames or codec.
    """
    name, segment_index, start, end = job
    ctx = _VIEW_CONTEXT
    assert ctx is not None, "view context not initialized"
    previews, colors, masks = ctx["previews"], ctx["colori"], ctx["maschere"]
    records, windows, assignments = ctx["records"], ctx["windows"], ctx["assignments"]
    meta, scale, legends, background = ctx["meta"], ctx["scale"], ctx["legende"], ctx["fondo"]
    side, threshold, kernel, full = ctx["lato"], ctx["soglia"], ctx["kernel"], ctx["full"]
    line = ctx["linea"]
    target = ctx["target"]

    view, base = view_of(name), base_method(name)
    suffix = "" if full else f"_clip_{segment_index:03d}"
    path = target / f"{name}{suffix}.mp4"
    pending = path.with_name(f".{path.stem}.pending.mp4")
    writer = _VideoWriter(
        pending, meta["fps"], (side * 3, side + header_height(side * 3) + STRIP_H)
    )
    cache: tuple[int, np.ndarray] | None = None
    try:
        for frame in range(start, end + 1):
            i = int(assignments[frame])
            if cache is None or cache[0] != i:
                cache = (
                    i,
                    _view_clip(
                        previews[i],
                        view,
                        records[i]["checks"]["perturbed_views"]["views"][view],
                        masks.get((view, i)),
                        kernel,
                    ),
                )
            sample = offline_plane(windows[i], frame, len(previews[i]))
            plane = offline_plane(windows[i], frame, len(colors[name][i]))
            heat = cv2.resize(
                colors[name][i][plane],
                (side, side),
                # `name` is `pert__view__base`: the rule answers on the base method.
                interpolation=interpolation_of(name),
            )
            body = np.concatenate((previews[i][sample], cache[1][sample], heat), axis=1)
            row = records[i]["checks"]["perturbed_views"]["views"][view]
            header_spec = Header(
                metodo=method_label(name),
                famiglia="Input perturbations - the map describes the input in the centre",
                modello=model_label(meta["run_id"]),
                video=Path(meta["video"]).stem,
                dettaglio=(
                    f"window {i + 1}/{len(records)} - tubelet {plane} - "
                    f"{'whole video' if full else f'clip {start}-{end}'}"
                ),
                significato=meaning_of(base),
                score=float(records[i]["score"]),
                score_perturbato=_perturbed_score(row, view),
                soglia=threshold,
                chunk_positivo=line.label_of(i),
                legenda=legends[name],
                estremi=(f"{scale[name]['lo']:.3g}", f"{scale[name]['hi']:.3g}"),
                extra=(
                    f"delta logit {row['logit_difference']:+.4f}"
                    + (
                        f" - mask {row['mask']} ({row['volume_tokens']} tokens)"
                        if row.get("kind") == "locale"
                        else ""
                    )
                ),
            )
            writer.write(compose(body, header_spec, strip(background, frame, line)))
    finally:
        writer.release()
    with av.open(str(pending)) as container:
        stream = container.streams.video[0]
        codec, pixel_format = stream.codec_context.name, stream.codec_context.pix_fmt
        count = sum(1 for _ in container.decode(video=0))
    if count != end - start + 1 or codec != "h264" or pixel_format != "yuv420p":
        raise ValueError(f"Invalid perturbed-view movie: {pending}")
    os.replace(pending, path)
    return {
        "method": name,
        "view": view,
        "base": base,
        "file": path.name,
        "frames": count,
        "start_frame": start,
        "end_frame": end,
        "codec": codec,
        "pixel_format": pixel_format,
    }


def render_perturbed_views(
    folder: Path,
    *,
    anteprime: Path | None = None,
    group_scales: dict | None = None,
    force: bool = False,
    jobs: int = 1,
) -> dict:
    """Render one three-panel movie per (perturbation, method) pair.

    Perturbed maps describe the PERTURBED clip, so the panels are original · perturbed input
    · map, the second rebuilt from the saved mask or order. Preprocessed samples and the
    already coloured planes of every map stay in memory (about 2.5 GB on a 130-window video
    with 143 variants).

    :param group_scales: per base method scales common to the model; without them the scale
        is computed on this campaign only and colours are not comparable with the sibling.
    :param jobs: how many movies are rendered together (they are independent; serially, a
        15,000-frame video would need 19 h). Requires ``fork`` (Linux); serial on Windows.
    :return: manifest of the movies, with view and method of each.
    :raises ValueError: if the campaign has no perturbed views or artifacts are
        inconsistent.
    """
    import matplotlib

    matplotlib.use("Agg")

    from apps.demo.explanation import _scale_bounds

    meta = json.loads((folder / "campaign.json").read_text(encoding="utf-8"))
    records = meta["windows"]
    receipts = (records[0].get("checks") or {}).get("perturbed_views", {}).get("views")
    if not receipts:
        return {"videos": [], "images": [], "methods": []}
    windows = [
        WindowSpec(
            **{
                **r["window"],
                "explicit_frame_indices": tuple(r["window"]["explicit_frame_indices"]),
            }
        )
        for r in records
    ]
    previews = _previews_of(folder, anteprime or folder)
    assignments, _ = assignment_of(folder, meta, windows)
    segments = covered_segments(assignments)
    full = segments == [(0, meta["num_frames"] - 1)]
    kernel = blur_kernel_of(meta)
    side = int(meta["preprocess"]["size"])
    threshold = meta.get("threshold")
    line = timeline_of(folder, meta, records, assignments)
    # A single pass over the NPZ files: planes are coloured immediately and kept.
    colors: dict[str, list[np.ndarray]] = {}
    masks: dict[tuple[str, int], np.ndarray] = {}
    raw: dict[str, list[np.ndarray]] = {}
    for index in range(len(records)):
        with np.load(folder / f"maps_{index:05d}.npz") as archive:
            for name in archive.files:
                if view_of(name):
                    raw.setdefault(name, []).append(archive[name])
        path = folder / "artifacts" / f"w{index:05d}_perturbed_views.npz"
        if path.exists():
            with np.load(path) as archive:
                for name in archive.files:
                    if name.startswith("mask__"):
                        masks[(name.removeprefix("mask__"), index)] = archive[name]
    if not raw:
        return {"videos": [], "images": [], "methods": []}
    scale, legends = {}, {}
    for name, maps in raw.items():
        base = base_method(name)
        signed = _is_signed(name)
        if group_scales and base in group_scales:
            lo, hi = float(group_scales[base]["lo"]), float(group_scales[base]["hi"])
            shared = True
        else:
            lo, hi = _scale_bounds(name, np.concatenate([m.ravel() for m in maps]), signed)
            lo, hi, shared = float(lo), float(hi), False
        scale[name] = {"lo": lo, "hi": hi, "shared_across_campaigns": shared}
        legends[name] = _bar(name)
        colors[name] = [np.stack([_colorize(p, lo, hi, name) for p in m]) for m in maps]
    raw.clear()
    target = folder / "viste_perturbate"
    target.mkdir(exist_ok=True)
    previous_items = reuse_manifest(target, force)
    result = {
        "status": "running",
        "videos": [],
        "images": [],
        "methods": sorted(colors),
        "views": sorted({view_of(n) for n in colors}),
        "covered_frames": int(np.sum(assignments >= 0)),
        "source_frames": meta["num_frames"],
        "display": "preprocessed samples: left original, centre perturbed input, right map",
        "scales": scale,
    }
    background = strip_background(side * 3, line)
    job_list = []
    for name in sorted(colors):
        for segment_index, (start, end) in enumerate(segments):
            suffix = "" if full else f"_clip_{segment_index:03d}"
            if f"{name}{suffix}.mp4" in previous_items:
                result["videos"].append(previous_items[f"{name}{suffix}.mp4"])
                continue
            job_list.append((name, segment_index, int(start), int(end)))

    # The context is passed by FORK, not by arguments: set in a global BEFORE creating the
    # pool, children see it copy-on-write.
    global _VIEW_CONTEXT
    _VIEW_CONTEXT = {
        "previews": previews,
        "colori": colors,
        "maschere": masks,
        "records": records,
        "windows": windows,
        "assignments": assignments,
        "full": full,
        "kernel": kernel,
        "lato": side,
        "soglia": threshold,
        "linea": line,
        "meta": meta,
        "scale": scale,
        "legende": legends,
        "fondo": background,
        "target": target,
    }

    def _record(row: dict) -> None:
        """Add the movie to the manifest and rewrite it: the resume point."""
        result["videos"].append(row)
        # Written after every movie, not at the end of the loop.
        write_json(target / "manifest.json", result)

    try:
        if jobs > 1 and len(job_list) > 1 and _fork_available():
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(
                max_workers=min(jobs, len(job_list)),
                mp_context=multiprocessing.get_context("fork"),
            ) as pool:
                # `as_completed`, not `map`: the manifest advances as soon as a movie is ready.
                expected = [pool.submit(_render_view, job) for job in job_list]
                for future in as_completed(expected):
                    _record(future.result())
        else:
            for job in job_list:
                _record(_render_view(job))
    finally:
        _VIEW_CONTEXT = None
    result["status"] = "complete"
    write_json(target / "manifest.json", result)
    return result


#: Height of the caption written inside every panel of the quad.
CAPTION_H = 24

#: The four quad captions, in reading order. ``{metodo}`` is the IG baseline short name
#: and ``{vista}`` the perturbation one: columns say what is shown, rows on which input.
QUAD_CAPTIONS = (
    "1 · original input",
    "2 · {metodo} on the original",
    "3 · {vista}",
    "4 · {metodo} on the occluded input",
)


def _caption(panel: np.ndarray, testo: str) -> np.ndarray:
    """:return: the panel with its label on a darkened band at the bottom.

    The band is darkened, not filled: an opaque rectangle would cover the bottom of the shot.
    """
    canvas = panel.copy()
    band = canvas[-CAPTION_H:]
    cv2.addWeighted(band, 0.30, np.zeros_like(band), 0.70, 0.0, dst=band)
    _text(
        canvas,
        _truncate(testo, canvas.shape[1] - 16, _FONT_MONO, 0.42),
        (8, canvas.shape[0] - 8),
        0.42,
        COLORS["testo"],
    )
    return canvas


def _colorize_rgba(piano: np.ndarray, lo: float, hi: float, metodo: str) -> np.ndarray:
    """Return a map plane in BGRA, with the alpha declared by the table.

    The sibling of :func:`_colorize`, which flattens the same colour on the theme neutral
    background; the table choice stays in one place (:func:`_lut`). :func:`_colorize` is not
    rewritten on top of this one: the two quantize at different points, and rewriting would
    change the bytes of movies already on disk.

    :return: a ``[..., 4]`` uint8 array in BGRA order.
    """
    if np.ptp(piano) == 0 and not transparent_center(metodo):
        full = np.full((*piano.shape[:2], 4), 160, dtype=np.uint8)
        full[..., 3] = 255
        return full
    normalized = np.clip((piano - lo) / max(hi - lo, 1e-12), 0, 1)
    rgba = (_lut(metodo)(normalized) * 255).astype(np.uint8)
    return rgba[..., [2, 1, 0, 3]].copy()


#: What a quad needs, shared with child processes by fork (as :data:`_VIEW_CONTEXT`).
#: Valid only inside :func:`render_concept_quads`.
_QUAD_CONTEXT: dict | None = None


def _render_quad(job: tuple[str, int, int, int]) -> dict:
    """Render one four-panel quad and return its manifest row.

    :param job: ``(base method, segment index, first frame, last)``.
    :return: the manifest row, with codec and frame count.
    :raises ValueError: if the written movie lacks the expected frames or codec.
    """
    base, segment_index, start, end = job
    ctx = _QUAD_CONTEXT
    assert ctx is not None, "quad context not initialized"
    previews, masks = ctx["previews"], ctx["maschere"]
    originals, perturbed = ctx["originali"], ctx["perturbate"]
    records, windows, assignments = ctx["records"], ctx["windows"], ctx["assignments"]
    meta, scale, legends, background = ctx["meta"], ctx["scale"], ctx["legende"], ctx["fondo"]
    side, threshold, kernel, full = ctx["lato"], ctx["soglia"], ctx["kernel"], ctx["full"]
    view, line, target = ctx["vista"], ctx["linea"], ctx["target"]

    name = view_name(view, base)
    method_title = IG_BASELINE.get(base, method_label(base))
    view_label = VIEWS.get(view, view)
    captions = [text.format(metodo=method_title, vista=view_label) for text in QUAD_CAPTIONS]
    suffix = "" if full else f"_clip_{segment_index:03d}"
    path = target / f"quadro__{name}{suffix}.mp4"
    pending = path.with_name(f".{path.stem}.pending.mp4")
    writer = _VideoWriter(
        pending, meta["fps"], (side * 2, side * 2 + header_height(side * 2) + STRIP_H)
    )
    cache: tuple[int, np.ndarray] | None = None
    try:
        for frame in range(start, end + 1):
            i = int(assignments[frame])
            row = records[i]["checks"]["perturbed_views"]["views"][view]
            if cache is None or cache[0] != i:
                cache = (
                    i,
                    _view_clip(previews[i], view, row, masks.get((view, i)), kernel),
                )
            sample = offline_plane(windows[i], frame, len(previews[i]))
            original = previews[i][sample]
            masked = cache[1][sample]
            # Both maps are upscaled with the rule of the base method: the same quantity on two
            # inputs.
            interpolation = interpolation_of(base)
            above_original, _ = blend_map(
                original,
                cv2.resize(
                    originals[base][i][offline_plane(windows[i], frame, len(originals[base][i]))],
                    (side, side),
                    interpolation=interpolation,
                ),
            )
            above_masked, _ = blend_map(
                masked,
                cv2.resize(
                    perturbed[name][i][offline_plane(windows[i], frame, len(perturbed[name][i]))],
                    (side, side),
                    interpolation=interpolation,
                ),
            )
            body = np.concatenate(
                (
                    np.concatenate(
                        (
                            _caption(original, captions[0]),
                            _caption(above_original, captions[1]),
                        ),
                        axis=1,
                    ),
                    np.concatenate(
                        (
                            _caption(masked, captions[2]),
                            _caption(above_masked, captions[3]),
                        ),
                        axis=1,
                    ),
                ),
                axis=0,
            )
            header_spec = Header(
                metodo=method_label(base),
                famiglia="Comparison quad - same map and same scale on both inputs",
                modello=model_label(meta["run_id"]),
                video=Path(meta["video"]).stem,
                dettaglio=(
                    f"window {i + 1}/{len(records)} - "
                    f"{'whole video' if full else f'clip {start}-{end}'}"
                ),
                significato=meaning_of(base),
                score=float(records[i]["score"]),
                score_perturbato=_perturbed_score(row, view),
                soglia=threshold,
                chunk_positivo=line.label_of(i),
                legenda=legends[name],
                estremi=(f"{scale[name]['lo']:.3g}", f"{scale[name]['hi']:.3g}"),
                extra=(
                    f"{view_label} - delta logit {row['logit_difference']:+.4f}"
                    + (
                        f" - {row['volume_tokens']} token oscurati"
                        if row.get("volume_tokens") is not None
                        else ""
                    )
                ),
            )
            writer.write(compose(body, header_spec, strip(background, frame, line)))
    finally:
        writer.release()
    with av.open(str(pending)) as container:
        stream = container.streams.video[0]
        codec, pixel_format = stream.codec_context.name, stream.codec_context.pix_fmt
        count = sum(1 for _ in container.decode(video=0))
    if count != end - start + 1 or codec != "h264" or pixel_format != "yuv420p":
        raise ValueError(f"Invalid quad: {pending}")
    os.replace(pending, path)
    return {
        "method": name,
        "view": view,
        "base": base,
        "file": path.name,
        "frames": count,
        "start_frame": start,
        "end_frame": end,
        "codec": codec,
        "pixel_format": pixel_format,
        "panels": captions,
    }


def render_concept_quads(
    folder: Path,
    *,
    base_campaign: Path,
    vista: str,
    anteprime: Path | None = None,
    group_scales: dict | None = None,
    force: bool = False,
    jobs: int = 1,
    only_bases: tuple[str, ...] | None = None,
) -> dict:
    """Render one 2x2 quad per method: the two inputs and the same map on each.

        original               | map on the original
        occluded input         | map on the occluded input

    The question is how much the explanation moved, so both terms share the frame and the
    colour scale (the group base-method scale, ``group_scales``): with two scales a halved
    explanation would fill the colormap like the whole one. Maps on the original live in
    another campaign (``base_campaign``), and windows are checked one by one.

    :param folder: campaign with the perturbed maps and the view receipts.
    :param base_campaign: sibling campaign with the maps on the whole input.
    :param vista: which perturbation is compared, e.g. ``vtcd_concept_00_black``.
    :param anteprime: campaign carrying ``preview_inputs/``; ``folder`` if omitted.
    :param group_scales: per base method scales common to the model.
    :param jobs: how many quads are rendered together; requires ``fork`` (Linux).
    :param only_bases: if given, the only base methods to render (e.g. ``("ig_ensemble",)``).
    :return: manifest of the movies, with the panels declared row by row.
    :raises ValueError: if the view is missing, windows are misaligned or maps are missing.
    """
    import matplotlib

    matplotlib.use("Agg")

    from apps.demo.explanation import _scale_bounds

    meta = json.loads((folder / "campaign.json").read_text(encoding="utf-8"))
    records = meta["windows"]
    base_meta = json.loads((base_campaign / "campaign.json").read_text(encoding="utf-8"))
    if [r["window_id"] for r in records] != [r["window_id"] for r in base_meta["windows"]]:
        raise ValueError(f"Different windows between {base_campaign} and {folder}")
    if not (records[0].get("checks") or {}).get("perturbed_views", {}).get("views", {}).get(vista):
        raise ValueError(f"View '{vista}' missing from campaign {folder}")
    windows = [
        WindowSpec(
            **{
                **r["window"],
                "explicit_frame_indices": tuple(r["window"]["explicit_frame_indices"]),
            }
        )
        for r in records
    ]
    previews = _previews_of(folder, anteprime or folder)
    assignments, _ = assignment_of(folder, meta, windows)
    segments = covered_segments(assignments)
    full = segments == [(0, meta["num_frames"] - 1)]
    kernel = blur_kernel_of(meta)
    side = int(meta["preprocess"]["size"])
    threshold = meta.get("threshold")
    line = timeline_of(folder, meta, records, assignments)
    raw_o: dict[str, list[np.ndarray]] = {}
    raw_p: dict[str, list[np.ndarray]] = {}
    masks: dict[tuple[str, int], np.ndarray] = {}
    for index in range(len(records)):
        with np.load(folder / f"maps_{index:05d}.npz") as archive:
            for name in archive.files:
                if view_of(name) == vista:
                    raw_p.setdefault(name, []).append(archive[name])
        with np.load(base_campaign / f"maps_{index:05d}.npz") as archive:
            for name in archive.files:
                if not view_of(name):
                    raw_o.setdefault(name, []).append(archive[name])
        path = folder / "artifacts" / f"w{index:05d}_perturbed_views.npz"
        if path.exists():
            with np.load(path) as archive:
                for name in archive.files:
                    if name.startswith("mask__"):
                        masks[(name.removeprefix("mask__"), index)] = archive[name]
    bases = sorted(
        b for b in raw_o if view_name(vista, b) in raw_p and (only_bases is None or b in only_bases)
    )
    if not bases:
        raise ValueError(
            f"No method has the map on both inputs in {folder.name}: "
            f"original {sorted(raw_o)}, perturbed {sorted(raw_p)}"
        )
    scale, legends = {}, {}
    originals: dict[str, list[np.ndarray]] = {}
    perturbed: dict[str, list[np.ndarray]] = {}
    for base in bases:
        name = view_name(vista, base)
        signed = _is_signed(base)
        if group_scales and base in group_scales:
            lo, hi = float(group_scales[base]["lo"]), float(group_scales[base]["hi"])
            shared = True
        else:
            # Without the group scale it is estimated on BOTH inputs together.
            values = np.concatenate([m.ravel() for m in raw_o[base] + raw_p[name]])
            lo, hi = _scale_bounds(base, values, signed)
            lo, hi, shared = float(lo), float(hi), False
        scale[name] = {"lo": lo, "hi": hi, "shared_across_campaigns": shared}
        legends[name] = _bar(base)
        originals[base] = [
            np.stack([_colorize_rgba(p, lo, hi, base) for p in m]) for m in raw_o[base]
        ]
        perturbed[name] = [
            np.stack([_colorize_rgba(p, lo, hi, base) for p in m]) for m in raw_p[name]
        ]
    raw_o.clear()
    raw_p.clear()
    target = folder / "quadri_concetto"
    target.mkdir(exist_ok=True)
    previous_items = reuse_manifest(target, force)
    result = {
        "status": "running",
        "videos": [],
        "images": [],
        "view": vista,
        "methods": sorted(view_name(vista, b) for b in bases),
        "bases": bases,
        "covered_frames": int(np.sum(assignments >= 0)),
        "source_frames": meta["num_frames"],
        "base_campaign": base_campaign.name,
        "display": (
            "four panels: original input and its map on top, "
            "perturbed input and its map below; a single colour scale"
        ),
        "scales": scale,
    }
    background = strip_background(side * 2, line)
    job_list = []
    for base in bases:
        name = view_name(vista, base)
        for segment_index, (start, end) in enumerate(segments):
            suffix = "" if full else f"_clip_{segment_index:03d}"
            if f"quadro__{name}{suffix}.mp4" in previous_items:
                result["videos"].append(previous_items[f"quadro__{name}{suffix}.mp4"])
                continue
            job_list.append((base, segment_index, int(start), int(end)))

    global _QUAD_CONTEXT
    _QUAD_CONTEXT = {
        "previews": previews,
        "originali": originals,
        "perturbate": perturbed,
        "maschere": masks,
        "records": records,
        "windows": windows,
        "assignments": assignments,
        "full": full,
        "kernel": kernel,
        "lato": side,
        "soglia": threshold,
        "linea": line,
        "meta": meta,
        "scale": scale,
        "legende": legends,
        "fondo": background,
        "vista": vista,
        "target": target,
    }

    def _record(row: dict) -> None:
        """Add the quad to the manifest and rewrite it: the resume point."""
        result["videos"].append(row)
        write_json(target / "manifest.json", result)

    try:
        if jobs > 1 and len(job_list) > 1 and _fork_available():
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(
                max_workers=min(jobs, len(job_list)),
                mp_context=multiprocessing.get_context("fork"),
            ) as pool:
                expected = [pool.submit(_render_quad, job) for job in job_list]
                for future in as_completed(expected):
                    _record(future.result())
        else:
            for job in job_list:
                _record(_render_quad(job))
    finally:
        _QUAD_CONTEXT = None
    result["status"] = "complete"
    write_json(target / "manifest.json", result)
    return result


def render_extension(root: Path) -> dict:
    """Render the new groups and update the catalogue after the checks.

    :return: actual state per model, counts and per-window aggregated CRIS ranking.
    """
    from apps.demo.explanation_catalog import write_catalog

    report = {"models": {}}
    folders = sorted(p.parent for p in (root / "risultati").glob("nuovi_*/*/*/campaign.json"))
    for model in ("dora", "congelato"):
        selected = [
            p
            for p in folders
            if p.parent.name == model
            and json.loads((p / "campaign.json").read_text(encoding="utf-8"))["status"]
            == "complete"
        ]
        scales = compute_group_scales(selected)
        rows = []
        all_five = 0
        analyzed = 0
        required_methods = requested_methods(root)
        for folder in selected:
            render_campaign(folder, group_scales=scales)
            render_global_motion(folder)
            meta = json.loads((folder / "campaign.json").read_text(encoding="utf-8"))
            for i, row in enumerate(meta["windows"]):
                analyzed += 1
                all_five += int(required_methods.issubset(row["checks"]))
                if "vtcd_final" not in row["checks"]:
                    continue
                with np.load(
                    folder / f"artifacts/w{i:05d}_vtcd_final.npz", allow_pickle=False
                ) as stored:
                    rows.append(
                        {"window_id": row["window_id"], "importance": stored["importance"].tolist()}
                    )
        if rows:
            mean = np.mean([r["importance"] for r in rows], axis=0)
            report["models"][model] = {
                "windows": analyzed,
                "windows_requested_methods": all_five,
                "requested_methods": sorted(required_methods),
                "expected_windows": 75,
                "status": "complete" if all_five == 75 else "partial",
                "cris_mean_loss_degradation": mean.tolist(),
                "cris_ranking": np.argsort(-mean, kind="stable").tolist(),
                "aggregation": "equal weight per analysed window; diagnostics on the displayed sample",
                "windows_detail": rows,
            }
        else:
            report["models"][model] = {
                "windows": analyzed,
                "windows_requested_methods": all_five,
                "requested_methods": sorted(required_methods),
                "expected_windows": 75,
                "status": "complete" if all_five == 75 else "partial" if analyzed else "missing",
            }
    write_json(root / "rapporti/advanced_rendering.json", report)
    readme = root / "README.md"
    start_marker, end_marker = "<!-- advanced-status -->", "<!-- /advanced-status -->"
    original = readme.read_text(encoding="utf-8")
    lines = [
        start_marker,
        "## Requested methods: verified state",
        "",
        "Coverage with rendering: " + ", ".join(sorted(requested_methods(root))),
        "",
    ]
    for model, record in report["models"].items():
        lines.append(
            f"- {model}: {record['windows_requested_methods']}/75 windows ({record['status']})."
        )
    lines += [
        "",
        "The new views are in `risultati/nuovi_sviluppo/` and `risultati/nuovi_completi/`.",
        "Active methods are declared in the plan; earlier raw artifacts remain available.",
        "Fixed references are split by checkpoint in `riferimenti_advanced/`; the common manifest is in `configurazione/advanced_references.json`.",
        "Numerical details, diagnostics and the CRIS ranking are in [advanced_rendering.json](rapporti/advanced_rendering.json).",
        "Global ablations show original/perturbed and distinct scores in `global_motion/`.",
        "Resume: `python scripts/estendi_spiegazioni.py --render --methods "
        + " ".join(sorted(requested_methods(root)))
        + "` from the repository root.",
        "Earlier views remain available and linked from the video pages.",
        end_marker,
    ]
    block = "\n".join(lines)
    if start_marker in original and end_marker in original:
        before, rest = original.split(start_marker, 1)
        _, after = rest.split(end_marker, 1)
        updated = before + block + after
    else:
        updated = original.rstrip() + "\n\n" + block + "\n"
    write_text_atomic(readme, updated)
    write_catalog(root)
    return report


def verify_extension(root: Path, *, require_complete: bool = False) -> dict:
    """Verify coverage, receipts, codecs and links of the actual artifacts.

    :return: the persisted receipt, with missing items and diagnostics kept apart.
    :raises ValueError: on corruption or incompleteness when completion is required.
    """
    required = requested_methods(root)
    campaigns = []
    failures = []
    source_root = root / "sorgenti" if (root / "sorgenti").is_dir() else root / "risultati"
    for group, source_group in (("nuovi_sviluppo", "sviluppo"), ("nuovi_completi", "regressioni")):
        for source in sorted((source_root / source_group).glob("*/*/campaign.json")):
            folder = root / "risultati" / group / source.parent.parent.name / source.parent.name
            source_meta = json.loads(source.read_text(encoding="utf-8"))
            row = {
                "folder": str(folder.relative_to(root)),
                "expected_windows": len(source_meta["windows"]),
                "model": source.parent.parent.name,
                "status": "missing",
                "diagnostics": [],
            }
            campaigns.append(row)
            metadata_path = folder / "campaign.json"
            render_path = folder / "render/manifest.json"
            global_path = folder / "global_motion/manifest.json"
            paths = [metadata_path, render_path]
            if "motion_ablation" in required:
                paths.append(global_path)
            if not all(p.is_file() for p in paths):
                continue
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            if meta["status"] != "complete":
                row["status"] = "running"
                continue
            if not (folder / "index.html").is_file():
                failures.append(f"Missing HTML page: {folder}")
            if [r["window_id"] for r in meta["windows"]] != [
                r["window_id"] for r in source_meta["windows"]
            ]:
                failures.append(f"Different window coverage: {folder}")
                continue
            row["status"] = "complete"
            expected_maps = set()
            for i, record in enumerate(meta["windows"]):
                if not required.issubset(record["checks"]):
                    row["status"] = "partial_methods"
                    continue
                with np.load(folder / f"maps_{i:05d}.npz", allow_pickle=False) as stored_maps:
                    expected_maps.update(stored_maps.files)
                    for method in required:
                        payload = folder / f"artifacts/w{i:05d}_{method}.npz"
                        receipt = json.loads(
                            payload.with_suffix(".json").read_text(encoding="utf-8")
                        )
                        if file_sha256(payload) != receipt["payload_sha256"]:
                            failures.append(f"Invalid checksum: {payload}")
                        with np.load(payload, allow_pickle=False) as arrays:
                            for key in arrays.files:
                                if key.startswith("map__") and not np.array_equal(
                                    arrays[key], stored_maps[key.removeprefix("map__")]
                                ):
                                    failures.append(f"Inconsistent derived map: {payload}:{key}")
                baseline = (
                    meta.get("extension_identity", {})
                    .get("model", {})
                    .get("settings", {})
                    .get("source_baseline", "strict")
                )
                historical_prefix = "source_" if baseline == "current_host" else ""
                if any(
                    record.get(historical_prefix + k) != source_meta["windows"][i][k]
                    for k in ("logit", "score")
                ):
                    failures.append(f"Original logit or score modified: {folder}:{i}")
                if baseline == "current_host":
                    tolerance = meta["extension_identity"]["model"]["settings"]["fidelity_atol"]
                    for check in record["checks"].values():
                        if (
                            check.get("original_logit") != record["logit"]
                            or check.get("source_logit") != record["source_logit"]
                            or check.get("detector_repeat_delta", float("inf")) > tolerance
                        ):
                            failures.append(f"Inconsistent host baseline: {folder}:{i}")
                if (
                    record["checks"].get("anomaly_direction", {}).get("finite_difference_pass")
                    is False
                ):
                    row["diagnostics"].append(f"derivative window {i}")
            render = json.loads(render_path.read_text(encoding="utf-8"))
            global_render = (
                json.loads(global_path.read_text(encoding="utf-8"))
                if "motion_ablation" in required
                else {"videos": [], "methods": [], "covered_frames": meta["num_frames"]}
            )
            # Perturbed views are not drawn over the original video: they have their own manifest and
            # are verified against it.
            normal_maps = {m for m in expected_maps if not view_of(m)}
            perturbed_maps = expected_maps - normal_maps
            policy = (meta.get("explanation_settings") or {}).get("render_policy") or {}
            cuts = tuple(policy.get("tagli", ("side_by_side", "map")))
            # The overlay is ALWAYS added, even where the policy does not name it (policies saved
            # before this cut existed do not reject it). Excluding it is declared in `render_policy`.
            if "overlay" not in cuts:
                cuts = (*cuts, "overlay")
            if set(render["methods"]) != normal_maps:
                failures.append(f"Methods missing from rendering: {folder}")
            views_path = folder / "viste_perturbate/manifest.json"
            if perturbed_maps:
                if not views_path.is_file():
                    failures.append(f"Perturbed views not rendered: {folder}")
                else:
                    render_views = json.loads(views_path.read_text(encoding="utf-8"))
                    if set(render_views["methods"]) != perturbed_maps:
                        failures.append(f"Incomplete perturbed views: {folder}")
            expected_videos = {
                (method, kind, start, end)
                for method in normal_maps
                for kind in cuts
                for start, end in render["segments"]
            }
            actual_videos = {
                (v["method"], v["kind"], v["start_frame"], v["end_frame"]) for v in render["videos"]
            }
            if actual_videos != expected_videos or len(actual_videos) != len(render["videos"]):
                failures.append(f"Missing or duplicated video cuts: {folder}")
            expected_globals = {"freeze", "reverse"} | {
                f"permutation_{seed}" for seed in meta["explanation_settings"]["motion_seeds"]
            }
            if "motion_ablation" not in required:
                expected_globals = set()
            if set(global_render["methods"]) != expected_globals:
                failures.append(f"Missing global ablations: {folder}")
            expected_global_videos = {
                (method, start, end)
                for method in expected_globals
                for start, end in render["segments"]
            }
            actual_global_videos = {
                (v["method"], v["start_frame"], v["end_frame"]) for v in global_render["videos"]
            }
            if actual_global_videos != expected_global_videos:
                failures.append(f"Missing global ablation segments: {folder}")
            if group == "nuovi_completi" and (
                render["covered_frames"] != meta["num_frames"]
                or global_render["covered_frames"] != meta["num_frames"]
            ):
                failures.append(f"Whole video with uncovered frames: {folder}")
            for base, manifest in (
                (folder / "render", render),
                (folder / "global_motion", global_render),
            ):
                for entry in manifest["videos"]:
                    path = base / entry["file"]
                    with av.open(str(path)) as container:
                        stream = container.streams.video[0]
                        if (
                            stream.codec_context.name != "h264"
                            or stream.codec_context.pix_fmt != "yuv420p"
                        ):
                            failures.append(f"Non-compliant codec: {path}")
                    if entry["frames"] != entry["end_frame"] - entry["start_frame"] + 1:
                        failures.append(f"Wrong video count: {path}")
            if (
                meta.get("reference_diagnostics", {}).get("factorization", {}).get("converged")
                is False
            ):
                row["diagnostics"].append("CNMF at the iteration limit")
    pages = [root / "index.html", *(root / "risultati").glob("*/*/*/index.html")]
    checked_links = 0
    for page in pages:
        if not page.exists():
            continue
        for link in re.findall(r'(?:href|src)="([^"]+)"', page.read_text(encoding="utf-8")):
            url = urlsplit(link)
            if url.scheme or not url.path:
                continue
            checked_links += 1
            if not (page.parent / unquote(url.path)).is_file():
                failures.append(f"Missing link: {page}: {link}")
    complete = sum(row["status"] == "complete" for row in campaigns)
    if complete and not (root / "index.html").is_file():
        failures.append(f"Missing HTML catalogue: {root}")
    result = {
        "status": "complete" if complete == len(campaigns) and not failures else "partial",
        "complete_campaigns": complete,
        "expected_campaigns": len(campaigns),
        "checked_links": checked_links,
        "failures": failures,
        "campaigns": campaigns,
    }
    write_json(root / "rapporti/advanced_verification.json", result)
    if failures or (require_complete and result["status"] != "complete"):
        raise ValueError(
            f"Extension check: {complete}/{len(campaigns)} complete campaigns, {len(failures)} errors; see advanced_verification.json"
        )
    return result
