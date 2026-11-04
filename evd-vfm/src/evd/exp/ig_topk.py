"""The "top 10% of IG ensemble blacked out" view, and the maps computed on it.

Counterpart of the VTCD0 campaign on the other family: here the tokens ranked highest by
``ig_ensemble`` are blacked out. The question is the same (where does the explanation
move when what supported it is removed) and is shown with the same 2x2 layout
(:func:`apps.demo.explanation_advanced.render_quadri_concetto`).

It writes a new group and extends nothing: the campaign identity includes
``explain/methods.py`` (``exp/explanation_advanced.model_identity``), so a method
cannot be added to an existing folder. The mask has a single author
(:func:`evd.explain.boxes.top_fraction_mask`), shared with the bounding boxes.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from evd.data.dataset import WindowDataset, collate_windows
from evd.exp.demo import RunDemo
from evd.exp.explanation_campaign import (
    _resolve_video,
    choose_tn_references,
    explain_clip,
    file_sha,
    seed_from_window_id,
    write_json,
)
from evd.explain.perturbed import top_attribution_view, view_name
from evd.models.backbones.base import encode_multilevel
from evd.utils.io import read_jsonl

#: View name, also the key the renderer recognizes (``apps/demo/explanation.VISTE``);
#: the fraction is in the name so batches at different fractions do not collide.
VIEW = "ig_top10_black"

#: Map ranking the tokens: ``ig_ensemble``, the mean of all five baselines since
#: 22 September 2026 (ADR-073).
MAP = "ig_ensemble"

#: Methods recomputed on the blacked-out input: those of the base campaign, so every map
#: has its counterpart and the 2x2 layout uses a single scale.
METHODS = ("raw_attention", "gradient_activation", "ig")


def base_map(base: Path, indice: int, window_id: str, records: list[dict]) -> np.ndarray:
    """Return the ``ig_ensemble`` map of the window, taken from the sister campaign.

    The ``window_id`` is checked, not the position (ADR-055).

    :param base: campaign folder with the maps on the whole input.
    :param indice: window position.
    :param window_id: expected identifier.
    :param records: windows of the base campaign ``campaign.json``.
    :return: the ``[T', H', W']`` map.
    :raises ValueError: on misaligned windows.
    :raises KeyError: if the base campaign lacks ``ig_ensemble``.
    """
    if records[indice]["window_id"] != window_id:
        raise ValueError(
            f"Window {indice} misaligned: the base campaign says "
            f"{records[indice]['window_id']}, this one {window_id}"
        )
    with np.load(base / f"maps_{indice:05d}.npz") as archive:
        if MAP not in archive.files:
            raise KeyError(
                f"The base campaign has no '{MAP}' on window {window_id}: "
                f"available maps {sorted(archive.files)}"
            )
        return np.asarray(archive[MAP])


def campagna_ig_topk(
    run: RunDemo,
    video: Path,
    out: Path,
    *,
    base: Path,
    engine,
    backbone_checkpoint: Path,
    frazione: float,
    device: str = "cuda",
    split: str = "test",
    blur_kernel: int = 31,
    settings: dict | None = None,
    video_root: Path | None = None,
    source_snapshot: dict[str, bytes] | None = None,
) -> dict:
    """Compute the view for a video and write a campaign in the usual form.

    The result is readable by ``render_quadri_concetto`` and ``render_viste_perturbate``:
    same NPZ keys, same ``campaign.json`` shape, same per-view receipt.

    :param run: selected run, already open.
    :param video: video file path.
    :param out: new campaign folder.
    :param base: sister campaign carrying ``ig_ensemble`` and defining the windows.
    :param engine: inference engine already built, reused across videos.
    :param frazione: fraction of tokens to black out.
    :return: the written metadata, with actual costs.
    :raises FileExistsError: if the folder already exists.
    :raises ValueError: on misaligned windows or multi-view preprocessing.
    """
    if out.exists():
        raise FileExistsError(f"The campaign already exists: choose a new folder: {out}")
    base_meta = json.loads((base / "campaign.json").read_text(encoding="utf-8"))
    base_records = base_meta["windows"]
    settings = settings or {}
    ig_kwargs = dict(settings.get("ig", {}))
    if "steps" in ig_kwargs:
        ig_kwargs["steps"] = tuple(ig_kwargs["steps"])

    selected = torch.load(run.checkpoint_path, map_location="cpu", weights_only=False)
    selected_step = selected.get("selected_step")
    if selected_step is None or not selected.get("selected_head"):
        raise ValueError("The campaign requires a selected checkpoint, not the last state")
    del selected

    out.mkdir(parents=True)
    source_root = Path(__file__).parents[1]
    source_snapshot = source_snapshot or {
        str(p.relative_to(source_root)): p.read_bytes() for p in source_root.rglob("*.py")
    }
    for relative, content in source_snapshot.items():
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    engine.sampler = run.sampler(engine.spec, validazione=split == "val")
    scorer = engine.scorer
    meta = engine.decoder.probe(video)
    everyone = engine._windows(video, meta)
    expected = [r["window_id"] for r in base_records]
    items = [item for item in everyone if item.window.window_id in set(expected)]
    if [item.window.window_id for item in items] != expected:
        raise ValueError(
            f"The grid of {meta.video_id} does not reproduce the windows of the base campaign: "
            f"{len(items)} vs {len(expected)}"
        )
    dataset = WindowDataset(items, engine.decoder, engine.spec)
    per_id = {it.window.window_id: it for it in everyone}
    score_rows = [r for r in read_jsonl(run.score_path) if str(r.get("split", "test")) == split]
    tn_window_id, tn_video_id, tn_source, tn_common, tn_common_video = choose_tn_references(
        score_rows, meta.video_id, per_id, run.threshold, video_root
    )
    cache_tn: dict[str, object] = {}

    def feature_tn(window_id: str | None, video_id: str | None):
        """Encode the features of the normal reference once.

        The reference is the same as in the base campaign, even on another video; skipping it
        would compare a four-component ensemble with a five-component one.

        :raises ValueError: if the reference window is missing or the preprocessing is
            multi-view.
        """
        if window_id is None:
            return None
        if window_id not in cache_tn:
            if video_id == meta.video_id:
                item_tn = per_id[window_id]
            else:
                path = _resolve_video(video_root, video_id)
                others = {
                    it.window.window_id: it
                    for it in engine._windows(path, engine.decoder.probe(path))
                }
                if window_id not in others:
                    raise ValueError(
                        f"TN reference {window_id} missing from the grid of {video_id}"
                    )
                item_tn = others[window_id]
            tn_dataset = WindowDataset([item_tn], engine.decoder, engine.spec)
            clip_tn, _, tn_views = scorer.prepare_clips(collate_windows([tn_dataset[0]]))
            if tn_views is not None:
                raise ValueError("This campaign requires a single view")
            with torch.no_grad(), scorer.autocast():
                cache_tn[window_id] = encode_multilevel(scorer.backbone, clip_tn, scorer.levels)
        return cache_tn[window_id]

    def tn_of(window_id: str) -> tuple[str | None, str | None, str | None]:
        """Return the reference of this window, with the ``analyze_video`` rule.

        The window that is itself the reference uses the run-wide common reference, as the
        base campaign did.
        """
        if tn_window_id is not None and window_id == tn_window_id:
            return (
                tn_common,
                tn_common_video,
                ("fallback_normal_video_e_se_stessa" if tn_common else None),
            )
        return tn_window_id, tn_video_id, tn_source

    records: list[dict] = []
    metadata = {
        "schema": 1,
        "status": "running",
        "run_id": run.run_id,
        "checkpoint_sha256": file_sha(run.checkpoint_path),
        "selected_step": selected_step,
        "config_sha256": file_sha(run.dir / "config.yaml"),
        "source_sha256": {
            relative: hashlib.sha256(content).hexdigest()
            for relative, content in sorted(source_snapshot.items())
        },
        "backbone_sha256": file_sha(backbone_checkpoint),
        "video": str(video.resolve()),
        "video_sha256": file_sha(video),
        "split": split,
        "threshold": run.threshold,
        "threshold_source": run.threshold_source,
        "methods": list(METHODS),
        "precision": scorer.precision,
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "preprocess": asdict(engine.spec),
        "mode": "offline",
        "temporal_projection": (
            "nearest sample, tubelet of the sample; intermediate frames: visual interpolation"
        ),
        "overlap": ("nearest end among the windows containing the frame, ties to the later start"),
        "query": None,
        "blur_kernel": blur_kernel,
        "tn_reference": {
            "window_id": tn_window_id,
            "video_id": tn_video_id,
            "source": tn_source,
            "comune_window_id": tn_common,
            "comune_video_id": tn_common_video,
        },
        "perturbations": False,
        "requested_window_ids": [],
        "max_windows": None,
        "explanation_settings": settings,
        "vista": {
            "nome": VIEW,
            "source_map": MAP,
            "frazione": frazione,
            "base_campaign": base.name,
            "adaptation": "the maps describe the BLACKED-OUT clip, not the original",
        },
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
    (out / "artifacts").mkdir(exist_ok=True)

    for index, item in enumerate(items):
        start = time.perf_counter()
        clip, _, views = scorer.prepare_clips(collate_windows([dataset[index]]))
        if views is not None:
            raise ValueError("This campaign requires a single view")
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        preparation_s = time.perf_counter() - start

        map_ = base_map(base, index, item.window.window_id, base_records)
        altered, mask, receipt = top_attribution_view(
            clip,
            map_,
            frazione=frazione,
            mean=engine.spec.mean,
            std=engine.spec.std,
            sorgente=MAP,
        )
        begin = time.perf_counter()
        maps, checks, feats, logit, times = explain_clip(
            scorer,
            altered,
            METHODS,
            spec=engine.spec,
            ig_kwargs=ig_kwargs,
            blur_kernel=blur_kernel,
            noise_seed=seed_from_window_id(item.window.window_id),
            tn_features=feature_tn(*tn_of(item.window.window_id)[:2]),
            tn_receipt={
                "tn_source": tn_of(item.window.window_id)[2],
                "tn_video_id": tn_of(item.window.window_id)[1],
                "tn_window_id": tn_of(item.window.window_id)[0],
            },
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        explanation_s = time.perf_counter() - begin

        np.savez_compressed(
            out / f"maps_{index:05d}.npz",
            **{view_name(VIEW, name): value for name, value in maps.items()},
        )
        # The mask lets the renderer rebuild the blacked-out input without recomputing; it is
        # saved under the name `render_quadri_concetto` looks for.
        np.savez_compressed(
            out / "artifacts" / f"w{index:05d}_perturbed_views.npz",
            **{f"mask__{VIEW}": mask},
        )
        original = base_records[index]
        records.append(
            {
                "window": asdict(item.window),
                "window_id": item.window.window_id,
                "frame_indices": item.window.frame_indices,
                "grid": list(feats.grid),
                # Logit and score of the row are those of the whole input, from the base campaign: the
                # reference value shown next to the perturbed one.
                "logit": original["logit"],
                "score": original["score"],
                "checks": {
                    **checks,
                    "perturbed_views": {
                        "views": {
                            VIEW: {
                                **receipt,
                                "logit": logit,
                                "perturbed_score": float(torch.sigmoid(torch.tensor(logit))),
                                "logit_difference": original["logit"] - logit,
                                "original_logit": original["logit"],
                            }
                        },
                        "n_views": 1,
                        "mask_choice": f"top {frazione:.0%} of tokens of {MAP}",
                        "adaptation": "the maps describe the BLACKED-OUT clip, not the original",
                    },
                },
                "map_statistics": {
                    name: {
                        "minimum": float(np.min(value)),
                        "maximum": float(np.max(value)),
                        "mean": float(np.mean(value)),
                    }
                    for name, value in maps.items()
                },
                "forward_s": times["forward_s"],
                "preparation_s": preparation_s,
                "explanation_s": explanation_s,
                "total_s": preparation_s + explanation_s,
                "peak_gpu_bytes": (
                    int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else None
                ),
            }
        )
        metadata["windows"] = records
        write_json(out / "campaign.json", metadata)
        print(
            f"{meta.video_id} window {index + 1}/{len(items)}: "
            f"{receipt['token_selezionati']} tokens blacked out, "
            f"delta logit {records[-1]['checks']['perturbed_views']['views'][VIEW]['logit_difference']:+.4f}",
            flush=True,
        )

    metadata["status"] = "complete"
    write_json(out / "campaign.json", metadata)
    return metadata
