"""Offline explanation campaign: reuses engine, dataset, preprocessing and selected weights.

Intermediates are identified by the checkpoint and never enter the feature store. A
second read of the tokens on disk allows a PCA shared across the video.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from evd.data.dataset import WindowDataset, collate_windows
from evd.exp.demo import RunDemo, live_engine
from evd.explain.attribution import explain_head, integrated_gradients
from evd.explain.methods import ADVANCED_METHODS
from evd.explain.perturbation import perturbation_curve, reference_clip
from evd.explain.project import offline_plane
from evd.explain.representation import StreamingPCA, cosine_similarity, eigencam
from evd.models.backbones.base import encode_multilevel
from evd.utils.io import read_jsonl

METHODS = (
    "raw_attention",
    "gradient_activation",
    "gradient_activation_prenorm",
    "attention_gradient",
    "pca_rgb",
    "eigencam",
    "cosine",
    "ig",
    "legrad_last",
    "legrad_four",
) + ADVANCED_METHODS


def file_sha(path: Path) -> str:
    """:return: SHA256 of the file, read in blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict | list) -> None:
    """:param path: destination; no NaN is serialized."""
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def seed_from_window_id(window_id: str) -> int:
    """Return a deterministic seed for ``fill="noise"``, derived from the window identity.

    Never the global RNG or the execution order (as ``clip_seed`` in ``exp/e0.py``): two
    runs on the same window must produce the same noise.

    :param window_id: window identifier.
    :return: 32-bit integer, stable across runs.
    """
    return int.from_bytes(hashlib.sha256(window_id.encode()).digest()[:4], "big")


#: Alias of the previous private name, still imported by ``exp/explanation_advanced.py``.
_seed_from_window_id = seed_from_window_id


def _resolve_video(video_root: Path, video_id: str) -> Path:
    """Resolve a ``video_id`` to a file under ``video_root``.

    Same criterion as the inventory built by ``scripts/explain_video.py``: a single match on
    the stem, otherwise an error.

    :raises FileNotFoundError: if there is not exactly one match.
    """
    matches = list(video_root.rglob(f"{video_id}.mp4"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Video '{video_id}' not found (or duplicated) under {video_root}: "
            f"{len(matches)} matches"
        )
    return matches[0]


def select_windows(
    rows: list[dict], threshold: float, *, per_category: int = 6, seed: int = 42
) -> dict:
    """Select windows fixed by the DoRA scores before computing maps, rotating over videos.

    :param rows: scores with label, video_id and window_id, already filtered by split.
    :return: up to six windows per category, without duplicates.
    """
    rng = np.random.default_rng(seed)
    categories: dict[str, dict[str, list[dict]]] = {c: {} for c in ("TP", "TN", "FP", "FN")}
    seen = set()
    for row in sorted(rows, key=lambda r: r["window_id"]):
        if row["window_id"] in seen:
            continue
        seen.add(row["window_id"])
        y, p = bool(row["label"]), row["score"] >= threshold
        category = ("TP" if y else "FP") if p else ("FN" if y else "TN")
        categories[category].setdefault(row["video_id"], []).append({**row, "category": category})
    selected = []
    counts = {}
    for category, videos in categories.items():
        keys = sorted(videos)
        rng.shuffle(keys)
        for values in videos.values():
            rng.shuffle(values)
        chosen = []
        while len(chosen) < per_category and any(videos.values()):
            for key in keys:
                if videos[key] and len(chosen) < per_category:
                    chosen.append(videos[key].pop())
        selected.extend(chosen)
        counts[category] = len(chosen)
    return {"seed": seed, "threshold": threshold, "counts": counts, "windows": selected}


#: The five Integrated Gradients baselines, in computation order.
BASELINE_IG = ("blur", "mean", "black", "noise", "tn_same_video")

#: Ensemble components: all five baselines since 22 September 2026. ``blur`` used to be
#: excluded for not converging on one exploratory example, but on the corpus it fails to
#: converge less often than ``noise`` and ``tn_same_video``; its map carries a smaller
#: logit mass, so including it lowers the ensemble magnitude (median ratio 0.945) without
#: changing its shape (median correlation 0.992).
ENSEMBLE_COMPONENTS = ("black", "noise", "mean", "tn_same_video", "blur")


def ensemble_component_names(maps: dict) -> list[str]:
    """:return: which ensemble baselines were computed, in order.

    Eligibility depends only on the map existing, not on this window converging, so the
    method set stays identical across the folder (a condition of :func:`render_campaign`).
    """
    return [fill for fill in ENSEMBLE_COMPONENTS if f"ig_{fill}" in maps]


def ensemble_mean(maps: dict, componenti: list[str]) -> np.ndarray:
    """Return the Expected Gradients map: arithmetic mean of the components.

    Shared by the campaign computation and the TN-only recomputation (ADR-052).

    :param componenti: names without the ``ig_`` prefix, from
        :func:`ensemble_component_names`.
    :return: the averaged map, with the components' shape.
    """
    return np.mean([maps[f"ig_{fill}"] for fill in componenti], axis=0)


def explain_clip(
    scorer,
    clip: torch.Tensor,
    methods: tuple[str, ...],
    *,
    spec,
    ig_kwargs: dict | None = None,
    blur_kernel: int = 31,
    noise_seed: int = 0,
    tn_features=None,
    tn_receipt: dict | None = None,
    ig_fills: tuple[str, ...] = BASELINE_IG,
) -> tuple[dict[str, np.ndarray], dict, object, float, dict]:
    """Compute the maps read from the model on an already preprocessed clip.

    The single place computing attention, gradients and Integrated Gradients, shared by the
    campaign on the original video (:func:`analyze_video`) and the perturbed views
    (:mod:`evd.explain.perturbed`), as ADR-052 requires. Representation methods
    (``pca_rgb``, ``eigencam``, ``cosine``) need a basis or query shared by the whole video
    and live in the caller.

    :param clip: ``B,C,T,H,W`` already preprocessed, on the scorer device.
    :param methods: subset of :data:`METHODS`.
    :param spec: run preprocessing; mean and std are needed by ``black``.
    :param ig_kwargs: Integrated Gradients quadrature parameters.
    :param noise_seed: seed of the ``noise`` baseline, derived from the window.
    :param tn_features: features of the normal reference window, encoded once per campaign;
        ``None`` if none exists, in which case ``ig_tn_same_video`` is declared skipped so
        the method set stays identical across the folder.
    :param tn_receipt: provenance of the TN reference, copied into the receipt.
    :return: ``(maps, checks, feats, logit, timings)``; ``feats`` is the backbone output on
        the clip, reusable without a second forward.
    :raises RuntimeError: if the hooks alter the head logit.
    """
    ig_kwargs = dict(ig_kwargs or {})
    start = time.perf_counter()
    with torch.no_grad(), scorer.autocast():
        if "legrad_four" in methods:
            outputs = scorer.backbone.encode_layers(clip, (20, 21, 22, 23))
            feats = outputs[23]
            # Non-hierarchical blocks are raw in the wrapper: the trained final norm is applied to
            # them too, and declared.
            norm = scorer.backbone.module().norms_block[-1]
            for level in (20, 21, 22):
                outputs[level] = replace(outputs[level], tokens=norm(outputs[level].tokens))
        else:
            outputs = None
            feats = encode_multilevel(scorer.backbone, clip, scorer.levels)
        reference_logit = float(scorer.head(feats).float().item())
    if clip.is_cuda:
        torch.cuda.synchronize()
    forward_s = time.perf_counter() - start
    with scorer.autocast():
        attribution = explain_head(scorer.head, feats)
    maps = {k: v for k, v in attribution.maps.items() if k in methods}
    checks = {
        "attention_fidelity": attribution.fidelity,
        "forward_logit_delta": abs(reference_logit - attribution.logit),
    }
    if checks["forward_logit_delta"] > 1e-5:
        raise RuntimeError(f"Head forward altered by the hooks: {checks}")
    if "legrad_four" in methods:
        level_maps = [attribution.maps["legrad_last"]]
        readouts = {23: attribution.logit}
        for level in (20, 21, 22):
            with scorer.autocast():
                result = explain_head(scorer.head, outputs[level])
            level_maps.append(result.maps["legrad_last"])
            readouts[level] = result.logit
        maps["legrad_four"] = np.mean(level_maps, axis=0)
        checks["intermediate_readouts"] = readouts
        checks["intermediate_normalization"] = "encoder.norms_block[-1]"
        del outputs
    if "ig" in methods:
        for fill in ig_fills:
            if fill == "tn_same_video":
                if tn_features is None:
                    checks["ig_tn_same_video"] = {
                        "skipped": True,
                        "reason": (
                            "no normal window predicted normal, neither in the "
                            "video nor in a fully normal video of the "
                            "split, that is not also a target window "
                            "of this campaign"
                        ),
                    }
                    continue
                base = tn_features
            else:
                if fill == "black":
                    ref_clip = reference_clip(clip, fill, blur_kernel, mean=spec.mean, std=spec.std)
                elif fill == "noise":
                    ref_clip = reference_clip(clip, fill, blur_kernel, seed=noise_seed)
                else:
                    ref_clip = reference_clip(clip, fill, blur_kernel)
                with torch.no_grad(), scorer.autocast():
                    base = encode_multilevel(scorer.backbone, ref_clip, scorer.levels)

            def head_forward(tokens, feats=feats):
                with scorer.autocast():
                    return scorer.head(replace(feats, tokens=tokens, pooled=None))

            values, receipt = integrated_gradients(
                head_forward, feats.tokens, base.tokens, **ig_kwargs
            )
            if fill == "tn_same_video":
                receipt = {**receipt, **(tn_receipt or {})}
            maps[f"ig_{fill}"] = values.sum(-1).reshape(feats.grid).cpu().numpy()
            checks[f"ig_{fill}"] = receipt
            if fill != "tn_same_video":
                del base
        # Expected Gradients (Erion et al. 2019): the mean of the maps of several baselines,
        # rather than IG on an averaged baseline that no real forward produces. Since
        # 22 September 2026 all five components are used (see ENSEMBLE_COMPONENTS). Eligibility
        # depends only on a component being computed, so the method set stays identical across
        # the folder; convergence is declared in ``all_components_converged``.
        ensemble_components = ensemble_component_names(maps)
        if len(ensemble_components) >= 2:
            maps["ig_ensemble"] = ensemble_mean(maps, ensemble_components)
            checks["ig_ensemble"] = {
                "derived": "mean of the available maps (Expected Gradients)",
                "components": ensemble_components,
                "all_components_converged": all(
                    checks.get(f"ig_{fill}", {}).get("converged") for fill in ensemble_components
                ),
                "tn_source": (
                    (tn_receipt or {}).get("tn_source")
                    if "tn_same_video" in ensemble_components
                    else None
                ),
            }
        else:
            checks["ig_ensemble"] = {
                "skipped": True,
                "reason": f"fewer than two baselines available ({ensemble_components})",
            }
    return (
        maps,
        checks,
        feats,
        attribution.logit,
        {"forward_s": forward_s, "explanation_s": time.perf_counter() - start},
    )


def choose_tn_references(
    score_rows: list[dict],
    video_id: str,
    by_id: dict,
    threshold: float,
    video_root: Path | None,
) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    """Choose which normal window is the IG reference, and which replaces it.

    Shared by :func:`analyze_video` and ``scripts/ricalcola_ig_tn.py`` (ADR-052). Rule of
    18 September 2026: the TN reference is the video's own normal, predicted-normal window
    with the lowest score, one per video; videos without one use the run-wide common
    reference.

    :param score_rows: ``scores.jsonl`` rows of the evaluated partition.
    :param by_id: grid windows of this video, by ``window_id``.
    :param threshold: run threshold; below it a window is predicted normal.
    :param video_root: where to look for the common reference video.
    :return: ``(tn_window_id, tn_video_id, tn_source, common_window_id, common_video_id)``.
    :raises ValueError: if there is no own TN reference and ``video_root`` is missing.
    """
    meta_video_id = video_id
    tn_window_id: str | None = None
    tn_video_id: str | None = None
    tn_source: str | None = None
    #: Run-wide common reference, on a fully normal video: used when the video has no TN
    #: window of its own, and by the single window that is the video's own reference.
    tn_common_window_id: str | None = None
    tn_common_video_id: str | None = None
    # TN candidates in the same video (label 0, score below threshold), by increasing score.
    same_video_rows = [
        r
        for r in score_rows
        if r.get("video_id") == meta_video_id
        and r.get("label") == 0
        and r.get("score", 1.0) < threshold
        and r.get("window_id") in by_id
    ]
    tn_candidates = [
        r["window_id"] for r in sorted(same_video_rows, key=lambda r: (r["score"], r["window_id"]))
    ]
    # One TN window per video, the most confidently normal, without excluding target windows
    # (18 September 2026): when the whole video is explained every window is a target. The
    # window that is itself the reference uses the common one.
    tn_window_id = tn_candidates[0] if tn_candidates else None
    if tn_window_id is not None:
        tn_video_id, tn_source = meta_video_id, "same_video"
    # The common reference is always searched: a fully normal video (every window labelled
    # 0) predicted normal, one for the whole run so that maps stay comparable across videos.
    by_video: dict[str, list[dict]] = {}
    for r in score_rows:
        by_video.setdefault(r["video_id"], []).append(r)
    normal_video_ids = {
        video_id
        for video_id, rows_ in by_video.items()
        if rows_ and all(r.get("label") == 0 for r in rows_)
    }
    candidates_anywhere = [
        r
        for r in score_rows
        if r["video_id"] in normal_video_ids and r.get("score", 1.0) < threshold
    ]
    if candidates_anywhere and (video_root is not None or tn_window_id is None):
        # video_root is required only when the common reference is the only one available.
        if video_root is None:
            raise ValueError(
                "No own TN window for this video: video_root is required "
                "to search for one in another video"
            )
        # Only videos actually decodable here (the local machine does not hold the whole dataset).
        local_video_ids = {p.stem for p in video_root.rglob("*.mp4")}
        fallback_candidates = sorted(
            (r for r in candidates_anywhere if r["video_id"] in local_video_ids),
            key=lambda r: (r["score"], r["video_id"], r["window_id"]),
        )
        if fallback_candidates:
            best = fallback_candidates[0]
            tn_common_window_id = best["window_id"]
            tn_common_video_id = best["video_id"]
            if tn_window_id is None:
                # The video has no own TN reference: the common one becomes the reference of all its
                # windows.
                tn_window_id, tn_video_id, tn_source = (
                    tn_common_window_id,
                    tn_common_video_id,
                    "fallback_normal_video",
                )
    return (
        tn_window_id,
        tn_video_id,
        tn_source,
        tn_common_window_id,
        tn_common_video_id,
    )


def analyze_video(
    run: RunDemo,
    video: Path,
    out: Path,
    *,
    methods: tuple[str, ...],
    backbone_checkpoint: Path,
    device: str = "cuda",
    split: str = "test",
    window_ids: tuple[str, ...] = (),
    max_windows: int | None = None,
    query: dict | None = None,
    perturb: bool = False,
    blur_kernel: int = 31,
    video_root: Path | None = None,
    engine=None,
    settings: dict | None = None,
    source_snapshot: dict[str, bytes] | None = None,
) -> dict:
    """Write a new campaign, never overwriting historical maps or scores.

    :param query: window_id, native frame, y0/y1/x0/x1 on the patch grid.
    :param perturb: measure pixel deletion/insertion and five random controls.
    :param video_root: root under which to look for an external TN reference video;
        required only if the current video has no own TN window.
    :return: metadata and actual costs; files are written progressively per window.
    :raises FileExistsError: if the campaign folder already exists.
    :raises ValueError: on unsupported methods, missing selected weights or missing
        windows.
    """
    if not methods or set(methods) - set(METHODS):
        raise ValueError(f"Allowed methods: {METHODS}")
    if set(methods) & set(ADVANCED_METHODS):
        raise ValueError(
            "The new methods require prepared references: use extend_campaign or --extend-campaign in the CLI"
        )
    settings = settings or {}
    perturbation_settings = settings.get("perturbation", {})
    perturbation_kwargs = {
        k: tuple(perturbation_settings[k])
        for k in ("grid", "fractions")
        if k in perturbation_settings
    }
    random_seeds = tuple(perturbation_settings.get("random_seeds", range(42, 47)))
    if len(set(random_seeds)) != 5:
        raise ValueError("The comparison requires five distinct random seeds")
    ig_kwargs = dict(settings.get("ig", {}))
    if "steps" in ig_kwargs:
        ig_kwargs["steps"] = tuple(ig_kwargs["steps"])
    if "cosine" in methods and query is None:
        raise ValueError("Cosine requires an explicit query region in JSON")
    selected = torch.load(run.checkpoint_path, map_location="cpu", weights_only=False)
    selected_step = selected.get("selected_step")
    if selected_step is None or not selected.get("selected_head"):
        raise ValueError("The campaign requires a selected checkpoint, not the last state")
    if run.cfg.get("adapt", {}).get("family", "none") != "none" and not selected.get(
        "selected_adapter"
    ):
        raise ValueError("Selected checkpoint without adapter weights")
    del selected
    if out.exists():
        raise FileExistsError(f"The campaign already exists: choose a new folder: {out}")
    out.mkdir(parents=True)
    source_root = Path(__file__).parents[1]
    source_snapshot = source_snapshot or {
        str(p.relative_to(source_root)): p.read_bytes() for p in source_root.rglob("*.py")
    }
    for relative, contents in source_snapshot.items():
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)
    shutil.copy2(run.dir / "config.yaml", out / "source_config.yaml")
    if engine is None:
        engine, _ = live_engine(
            run, device=device, batch_size=1, backbone_checkpoint=backbone_checkpoint
        )
    engine.sampler = run.sampler(engine.spec, validazione=split == "val")
    scorer = engine.scorer
    meta = engine.decoder.probe(video)
    all_items = engine._windows(video, meta)
    items = all_items
    if window_ids:
        items = [item for item in items if item.window.window_id in window_ids]
        if {item.window.window_id for item in items} != set(window_ids):
            raise ValueError("Requested windows missing from the grid: check video and split")
    if max_windows is not None:
        items = items[:max_windows]
    if not items:
        raise ValueError("No requested window available")
    ds = WindowDataset(items, engine.decoder, engine.spec)
    pca = StreamingPCA()
    records = []
    query_vector = None
    uses_representation = bool(set(methods) & {"pca_rgb", "eigencam", "cosine"})
    by_id = {it.window.window_id: it for it in all_items}
    tn_window_id = tn_video_id = tn_source = None
    tn_common_window_id = tn_common_video_id = None
    if "ig" in methods:
        score_rows = [r for r in read_jsonl(run.score_path) if str(r.get("split", "test")) == split]
        (
            tn_window_id,
            tn_video_id,
            tn_source,
            tn_common_window_id,
            tn_common_video_id,
        ) = choose_tn_references(score_rows, meta.video_id, by_id, run.threshold, video_root)
    tn_clip_cache: dict[str, torch.Tensor] = {}

    def tn_reference_clip(tn_window_id: str | None, tn_video_id: str | None):
        """Return the clip of the given TN window, or ``None`` if there is none.

        Taken from the current video grid, or decoded separately if it belongs to another video
        (resolved under ``video_root``).
        """
        if tn_window_id is None:
            return None
        if tn_window_id not in tn_clip_cache:
            if tn_video_id == meta.video_id:
                tn_item = by_id[tn_window_id]
            else:
                fallback_path = _resolve_video(video_root, tn_video_id)
                fallback_meta = engine.decoder.probe(fallback_path)
                fallback_by_id = {
                    it.window.window_id: it for it in engine._windows(fallback_path, fallback_meta)
                }
                if tn_window_id not in fallback_by_id:
                    raise ValueError(
                        f"Fallback window {tn_window_id} missing from the grid of {tn_video_id}"
                    )
                tn_item = fallback_by_id[tn_window_id]
            tn_ds = WindowDataset([tn_item], engine.decoder, engine.spec)
            tn_batch = collate_windows([tn_ds[0]])
            tn_clip, _, tn_views = scorer.prepare_clips(tn_batch)
            if tn_views is not None:
                raise ValueError("This campaign requires a single view")
            tn_clip_cache[tn_window_id] = tn_clip
        return tn_clip_cache[tn_window_id]

    tn_features_cache: dict[str, object] = {}

    def tn_reference_features(tn_window_id: str | None, tn_video_id: str | None):
        """Return the features of the given TN window, encoded once each.

        The backbone is deterministic on the same host, so encoding once gives the same numbers
        and saves one forward per window.

        :return: ``BackboneOutput`` of the TN clip, or ``None`` if none exists.
        """
        if tn_window_id is None:
            return None
        if tn_window_id not in tn_features_cache:
            clip = tn_reference_clip(tn_window_id, tn_video_id)
            if clip is None:
                tn_features_cache[tn_window_id] = None
            else:
                with torch.no_grad(), scorer.autocast():
                    tn_features_cache[tn_window_id] = encode_multilevel(
                        scorer.backbone, clip, scorer.levels
                    )
        return tn_features_cache[tn_window_id]

    def tn_per_window(window_id: str) -> tuple[str | None, str | None, str | None]:
        """Return which normal reference this window uses.

        The video's own lowest-score TN window, or the common one when the video has none. The
        window that is itself the reference uses the common one (an attribution against itself
        would be identically zero).

        :return: ``(window_id, video_id, source)`` of the reference to use.
        """
        if tn_window_id is not None and window_id == tn_window_id:
            return (
                tn_common_window_id,
                tn_common_video_id,
                ("fallback_normal_video_e_se_stessa" if tn_common_window_id else None),
            )
        return tn_window_id, tn_video_id, tn_source

    metadata = {
        "schema": 1,
        "status": "running",
        "run_id": run.run_id,
        "checkpoint_sha256": file_sha(run.checkpoint_path),
        "selected_step": selected_step,
        "config_sha256": file_sha(run.dir / "config.yaml"),
        "source_sha256": {
            relative: hashlib.sha256(contents).hexdigest()
            for relative, contents in sorted(source_snapshot.items())
        },
        "backbone_sha256": file_sha(backbone_checkpoint),
        "video": str(video.resolve()),
        "video_sha256": file_sha(video),
        "split": split,
        "threshold": run.threshold,
        "threshold_source": run.threshold_source,
        "methods": methods,
        "representation_formulas": {
            "pca_rgb": "centred covariance, basis shared per video",
            "eigencam": "O V1 without centring; dominant coordinate sign positive",
        },
        "precision": scorer.precision,
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "preprocess": asdict(engine.spec),
        "mode": "offline",
        "temporal_projection": "nearest sample, tubelet of the sample; intermediate frames: visual interpolation",
        # Since 20 September 2026 (previously "nearest centre"); older campaigns carry the old
        # string, which describes the frame projection used for drawing, not the saved maps.
        "overlap": "nearest end among the windows containing the frame, ties to the later start",
        "query": query,
        "blur_kernel": blur_kernel,
        "tn_reference": {
            "window_id": tn_window_id,
            "video_id": tn_video_id,
            "source": tn_source,
            # The common reference is declared even when it is not the campaign reference: one window
            # uses it anyway.
            "comune_window_id": tn_common_window_id,
            "comune_video_id": tn_common_video_id,
        },
        "perturbations": perturb,
        "requested_window_ids": window_ids,
        "max_windows": max_windows,
        "explanation_settings": settings,
        "windows": records,
        "fps": meta.fps,
        "num_frames": meta.num_frames,
        "environment": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": device,
            "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None,
        },
    }
    write_json(out / "campaign.json", metadata)
    for i, item in enumerate(items):
        window_start = time.perf_counter()
        batch = collate_windows([ds[i]])
        clips, _, views = scorer.prepare_clips(batch)
        if views is not None:
            raise ValueError("This campaign requires a single view")
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        preparation_s = time.perf_counter() - window_start
        tn_window = tn_per_window(item.window.window_id)
        start = time.perf_counter()
        maps, checks, feats, window_logit, timings = explain_clip(
            scorer,
            clips,
            methods,
            spec=engine.spec,
            ig_kwargs=ig_kwargs,
            blur_kernel=blur_kernel,
            noise_seed=seed_from_window_id(item.window.window_id),
            tn_features=tn_reference_features(*tn_window[:2]),
            tn_receipt={
                "tn_source": tn_window[2],
                "tn_video_id": tn_window[1],
                "tn_window_id": tn_window[0],
            },
        )
        forward_s = timings["forward_s"]
        if uses_representation:
            tokens = feats.tokens[0].float().cpu().numpy()
            np.save(out / f"tokens_{i:05d}.npy", tokens)
            if "pca_rgb" in methods:
                pca.update(tokens)
            if "eigencam" in methods:
                maps["eigencam"] = eigencam(tokens).reshape(feats.grid)
            if query is not None and item.window.window_id == query["window_id"]:
                plane = offline_plane(item.window, int(query["frame"]), feats.grid[0])
                if not (
                    item.window.start_frame <= query["frame"] <= item.window.end_frame
                    and 0 <= query["y0"] < query["y1"] <= feats.grid[1]
                    and 0 <= query["x0"] < query["x1"] <= feats.grid[2]
                ):
                    raise ValueError("Query outside the window or the grid")
                region = tokens.reshape(*feats.grid, -1)[
                    plane, query["y0"] : query["y1"], query["x0"] : query["x1"]
                ]
                query_vector = region.mean((0, 1))
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        explanation_s = time.perf_counter() - start
        curves = []
        if perturb:

            def detector(clip):
                with scorer.autocast():
                    return scorer.head(encode_multilevel(scorer.backbone, clip, scorer.levels))

            # Backward tensors are no longer needed while perturbing pixels.
            for method, relevance in maps.items():
                if method in ("eigencam", "pca_rgb", "cosine"):
                    continue
                print(f"Perturbations {meta.video_id} window {i}: {method}", flush=True)
                signs = (False, True) if np.any(relevance < 0) else (False,)
                for fill in ("blur", "mean"):
                    for opposition in signs:
                        for row in perturbation_curve(
                            detector,
                            clips,
                            relevance,
                            fill=fill,
                            opposition=opposition,
                            blur_kernel=blur_kernel,
                            **perturbation_kwargs,
                        ):
                            curves.append({"method": method, **row})
            for fill in ("blur", "mean"):
                for seed in random_seeds:
                    for row in perturbation_curve(
                        detector,
                        clips,
                        # Only the shape matters (the order is a random permutation), but the map must be one
                        # computed on this window.
                        maps.get("raw_attention", next(iter(maps.values()))),
                        fill=fill,
                        random_seed=seed,
                        blur_kernel=blur_kernel,
                        **perturbation_kwargs,
                    ):
                        curves.append({"method": "random", **row})
        np.savez_compressed(out / f"maps_{i:05d}.npz", **maps)
        write_json(out / f"perturbations_{i:05d}.json", curves)
        record = {
            "window": asdict(item.window),
            "window_id": item.window.window_id,
            "frame_indices": item.window.frame_indices,
            "grid": feats.grid,
            "logit": window_logit,
            "score": float(torch.sigmoid(torch.tensor(window_logit))),
            "checks": checks,
            "map_statistics": {
                method: {
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "positive_mass": float(np.maximum(values, 0).sum()),
                    "negative_mass": float(np.minimum(values, 0).sum()),
                    "null": bool(not values.any()),
                }
                for method, values in maps.items()
            },
            "forward_s": forward_s,
            "preparation_s": preparation_s,
            "explanation_s": explanation_s,
            "total_s": time.perf_counter() - window_start,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0,
        }
        records.append(record)
        write_json(out / "campaign.json", metadata)
        print(
            f"{run.run_id} {meta.video_id} {i + 1}/{len(items)}: {record['total_s']:.2f}s",
            flush=True,
        )
        del feats, clips
    if "cosine" in methods and query_vector is None:
        raise ValueError("The query window is not included in the selection")
    postprocess_start = time.perf_counter()
    basis = pca.basis() if "pca_rgb" in methods else None
    if basis is not None:
        np.savez_compressed(out / "pca_basis.npz", mean=pca.mean, basis=basis, count=pca.count)
    if query_vector is not None:
        np.save(out / "query.npy", query_vector)
    if uses_representation:
        for i, record in enumerate(records):
            tokens = np.load(out / f"tokens_{i:05d}.npy")
            with np.load(out / f"maps_{i:05d}.npz") as stored:
                maps = dict(stored)
            if basis is not None:
                maps["pca_rgb"] = (
                    ((tokens - pca.mean) @ basis).reshape(*record["grid"], 3).astype(np.float32)
                )
            if "cosine" in methods:
                maps["cosine"] = cosine_similarity(tokens, query_vector).reshape(record["grid"])
            np.savez_compressed(out / f"maps_{i:05d}.npz", **maps)
            # Local intermediate: after the projection, raw maps and PCA basis are enough to
            # regenerate the rendering.
            (out / f"tokens_{i:05d}.npy").unlink()
    metadata["status"] = "complete"
    metadata["postprocess_s"] = time.perf_counter() - postprocess_start
    metadata["artifact_bytes"] = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    write_json(out / "campaign.json", metadata)
    return metadata
