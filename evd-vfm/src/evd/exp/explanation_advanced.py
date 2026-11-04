"""Extend offline campaigns with fixed references and per-method resumption.

Reuses ``live_engine``, ``WindowDataset`` and the run grid. Previous artifacts are
read-only sources; new campaigns keep the schema consumed by the existing renderer.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from evd.data.dataset import WindowDataset, WindowItem, collate_windows
from evd.exp.demo import RunDemo
from evd.exp.explanation_campaign import _resolve_video, _seed_from_window_id, explain_clip
from evd.exp.explanation_store import ArtifactStore, atomic_json, atomic_npz, identity_sha
from evd.explain.concepts import (
    direction_maps,
    fit_binary_references,
    fit_class_references,
    prototype_maps,
)
from evd.explain.methods import ADVANCED_METHODS, HANDLERS, MethodContext
from evd.explain.vtcd import convex_nmf, slic_regions
from evd.infer.engine import InferenceEngine
from evd.models.backbones.base import encode_multilevel
from evd.types import BackboneOutput, PreprocessSpec, WindowSpec
from evd.utils.hashing import file_sha256


def window_from_record(record: dict) -> WindowSpec:
    """:return: the value object of the saved window, with immutable indices."""
    return WindowSpec(
        **{
            **record["window"],
            "explicit_frame_indices": tuple(record["window"]["explicit_frame_indices"]),
        }
    )


def model_identity(run: RunDemo, engine: InferenceEngine, checkpoint: Path, settings: dict) -> dict:
    """Check the selection and identify model, preprocessing and algorithm.

    :return: identity shared by extraction, references and methods.
    :raises ValueError: on a checkpoint without selected weights or multiple levels.
    """
    selected = torch.load(run.checkpoint_path, map_location="cpu", weights_only=True)
    if selected.get("selected_step") is None or not selected.get("selected_head"):
        raise ValueError("The explanation requires the selected weights of a closed run")
    if run.cfg["adapt"].get("family", "none") != "none" and not selected.get("selected_adapter"):
        raise ValueError("The selected adapter weights are missing")
    if len(engine.scorer.levels) != 1:
        raise ValueError("The five methods require a single final level")
    source = Path(__file__).parents[1]
    names = (
        "explain/concepts.py",
        "explain/vtcd.py",
        "explain/volumes.py",
        "explain/methods.py",
        "explain/perturbation.py",
        "exp/explanation_advanced.py",
        "data/preprocess.py",
        "data/dataset.py",
        "exp/demo.py",
        "models/backbones/vjepa21.py",
    )
    return {
        "run_id": run.run_id,
        "selected_step": int(selected["selected_step"]),
        "checkpoint_sha256": file_sha256(run.checkpoint_path),
        "backbone_sha256": file_sha256(checkpoint),
        "config_sha256": file_sha256(run.dir / "config.yaml"),
        "preprocess": json.loads(json.dumps(asdict(engine.spec))),
        "levels": list(engine.scorer.levels),
        "precision": engine.scorer.precision,
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "settings": settings,
        "algorithm_sha256": {
            name: file_sha256(source / name) for name in names if (source / name).exists()
        },
        "environment": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(engine.scorer.device),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        },
    }


def prepare_clip(engine: InferenceEngine, item: WindowItem) -> torch.Tensor:
    """Decode and preprocess with the same path as the detector.

    :return: a B,C,T,H,W clip on the scorer device.
    :raises ValueError: on multi-view preprocessing, outside the campaign.
    """
    dataset = WindowDataset([item], engine.decoder, engine.spec)
    clips, _, views = engine.scorer.prepare_clips(collate_windows([dataset[0]]))
    if views is not None:
        raise ValueError("The campaign requires a single view")
    return clips


def extract_features(engine: InferenceEngine, clip: torch.Tensor) -> tuple[BackboneOutput, float]:
    """:return: final features and original logit, without keeping the backbone graph."""
    with torch.no_grad(), engine.scorer.autocast():
        features = encode_multilevel(engine.scorer.backbone, clip, engine.scorer.levels)
        logit = float(engine.scorer.head(features).float().item())
    return features, logit


def prepare_references(
    engine: InferenceEngine, manifest: dict, out: Path, identity: dict
) -> tuple[dict[str, np.ndarray], dict]:
    """Extract every reference once and freeze separator, prototypes and concepts.

    :return: arrays shared by all windows of the checkpoint, and diagnostics.
    :raises ValueError: on an altered manifest or a reference video differing from the
        inventory.
    """
    store = ArtifactStore(out)
    if (
        identity_sha({key: value for key, value in manifest.items() if key != "sha256"})
        != manifest["sha256"]
    ):
        raise ValueError("Altered reference manifest: recompute the selection")
    reference_identity = {"model": identity, "manifest_sha256": manifest["sha256"]}
    saved = store.load("references", reference_identity)
    if saved is not None:
        return saved
    settings = identity["settings"]
    means, region_descriptors, labels, diagnostics = [], [], [], []
    for index, row in enumerate(manifest["windows"]):
        key = f"training_{index:03d}"
        item_identity = {**reference_identity, "window": row}
        saved = store.load(key, item_identity)
        if saved is None:
            start = time.perf_counter()
            if file_sha256(Path(row["video"])) != row["video_sha256"]:
                raise ValueError(f"Reference video differs from the inventory: {row['video']}")
            clip = prepare_clip(engine, WindowItem(window_from_record(row), row["video"]))
            features, logit = extract_features(engine, clip)
            tokens = features.tokens[0].float().cpu().numpy()
            regions, descriptors, segmentation = slic_regions(
                tokens.reshape(*features.grid, -1), **settings["vtcd"]["slic"]
            )
            arrays = {
                "tokens": tokens,
                "regions": regions,
                "descriptors": descriptors,
                "mean": tokens.mean(0),
            }
            receipt = {
                "grid": features.grid,
                "logit": logit,
                "segmentation": segmentation,
                "elapsed_s": time.perf_counter() - start,
            }
            store.save(key, item_identity, arrays, receipt)
            print(
                f"References {identity['run_id']}: {index + 1}/{len(manifest['windows'])}, {receipt['elapsed_s']:.1f}s",
                flush=True,
            )
            del features, clip
        else:
            arrays, receipt = saved
        means.append(arrays["mean"])
        region_descriptors.append(arrays["descriptors"])
        labels.append(row["label"])
        diagnostics.append(receipt)
    start = time.perf_counter()
    binary, binary_receipt = fit_binary_references(
        np.stack(means), np.asarray(labels), seed=settings["seed"], c=settings["logistic_c"]
    )
    factors, factor_receipt = convex_nmf(
        np.concatenate(region_descriptors),
        n_concepts=settings["vtcd"]["n_concepts"],
        **settings["vtcd"]["nmf"],
    )
    arrays = {
        **binary,
        **factors,
        "reference_descriptors": np.stack(means),
        "labels": np.asarray(labels),
    }
    receipt = {
        "binary": binary_receipt,
        "factorization": factor_receipt,
        "references": diagnostics,
        "fit_s": time.perf_counter() - start,
        "manifest_sha256": manifest["sha256"],
    }
    store.save("references", reference_identity, arrays, receipt)
    return arrays, receipt


def prepare_class_references(
    engine: InferenceEngine, manifest: dict, out: Path, identity: dict
) -> tuple[dict[str, np.ndarray], dict]:
    """Fit one CAV per class from the mean descriptors of the reference windows.

    No SLIC (needed only by VTCD, about 29 s per window), hence a separate function. The
    manifest is a different one from the binary references, whose ``sha256`` must not
    change.

    :param engine: inference engine already built on the checkpoint.
    :param manifest: class reference manifest, already verified.
    :param out: ``ArtifactStore`` folder to save to and resume from.
    :param identity: model identity, as for the binary references.
    :return: per-class directions and fit diagnostics.
    :raises ValueError: on an altered manifest or a differing reference video.
    """
    store = ArtifactStore(out)
    if (
        identity_sha({key: value for key, value in manifest.items() if key != "sha256"})
        != manifest["sha256"]
    ):
        raise ValueError("Altered class reference manifest: recompute the selection")
    reference_identity = {"model": identity, "manifest_sha256": manifest["sha256"], "kind": "class"}
    saved = store.load("class_references", reference_identity)
    if saved is not None:
        return saved
    means, classes, labels, diagnostics = [], [], [], []
    for index, row in enumerate(manifest["windows"]):
        key = f"class_training_{index:04d}"
        item_identity = {**reference_identity, "window": row}
        saved = store.load(key, item_identity)
        if saved is None:
            start = time.perf_counter()
            if file_sha256(Path(row["video"])) != row["video_sha256"]:
                raise ValueError(f"Reference video differs from the inventory: {row['video']}")
            clip = prepare_clip(engine, WindowItem(window_from_record(row), row["video"]))
            features, logit = extract_features(engine, clip)
            arrays = {"mean": features.tokens[0].float().cpu().numpy().mean(0)}
            receipt = {"logit": logit, "elapsed_s": time.perf_counter() - start}
            store.save(key, item_identity, arrays, receipt)
            del features, clip
        else:
            arrays, receipt = saved
        means.append(arrays["mean"])
        classes.append(row["class_idx"])
        labels.append(row["label"])
        diagnostics.append(receipt)
    settings = identity["settings"]
    start = time.perf_counter()
    arrays, receipt = fit_class_references(
        np.stack(means),
        np.asarray(classes),
        np.asarray(labels),
        seed=settings["seed"],
        c=settings["logistic_c"],
    )
    receipt.update(
        references=diagnostics,
        fit_s=time.perf_counter() - start,
        manifest_sha256=manifest["sha256"],
        per_class=manifest.get("per_class"),
    )
    store.save("class_references", reference_identity, arrays, receipt)
    return arrays, receipt


def prepare_vtcd_class_references(
    engine: InferenceEngine, manifest: dict, out: Path, identity: dict
) -> tuple[dict[str, np.ndarray], dict]:
    """Fit a separate VTCD dictionary per class on training data only.

    :param manifest: sealed class references, with split and class_idx.
    :param out: store separate from the CAVs and the global dictionary.
    :param identity: checkpoint and factorization settings.
    :return: C×K×D dictionaries and per-class CNMF diagnostics.
    :raises ValueError: on an altered manifest or references outside training.
    """
    if identity_sha({k: v for k, v in manifest.items() if k != "sha256"}) != manifest["sha256"]:
        raise ValueError("Altered per-class VTCD reference manifest")
    store = ArtifactStore(out)
    reference_identity = {
        "model": identity,
        "manifest_sha256": manifest["sha256"],
        "kind": "vtcd_class",
    }
    saved = store.load("vtcd_class_references", reference_identity)
    if saved is not None:
        return saved
    settings = identity["settings"]["vtcd"]
    descriptors_by_class = {}
    for index, row in enumerate(manifest["windows"]):
        if row["label"] != 1:
            continue
        if row.get("split") != "train":
            raise ValueError("Per-class VTCD requires training references only")
        key = f"vtcd_class_training_{index:04d}"
        item_identity = {**reference_identity, "window": row}
        saved = store.load(key, item_identity)
        if saved is None:
            if file_sha256(Path(row["video"])) != row["video_sha256"]:
                raise ValueError(f"Altered reference video: {row['video']}")
            clip = prepare_clip(engine, WindowItem(window_from_record(row), row["video"]))
            features, logit = extract_features(engine, clip)
            regions, descriptors, segmentation = slic_regions(
                features.tokens[0].float().cpu().numpy().reshape(*features.grid, -1),
                **settings["slic"],
            )
            arrays = {"descriptors": descriptors, "regions": regions}
            receipt = {"logit": logit, "segmentation": segmentation}
            store.save(key, item_identity, arrays, receipt)
            del clip, features
        else:
            arrays, receipt = saved
        descriptors_by_class.setdefault(int(row["class_idx"]), []).append(arrays["descriptors"])
        print(f"Class VTCD: reference {index + 1}/{len(manifest['windows'])}", flush=True)
    dictionaries, checks = [], {}
    indices = sorted(descriptors_by_class)
    for k in indices:
        descriptors = np.concatenate(descriptors_by_class[k])
        fitted, check = convex_nmf(
            descriptors, n_concepts=settings["n_concepts"], **settings["nmf"]
        )
        dictionaries.append(fitted["dictionary"])
        checks[str(k)] = {
            **check,
            "n_windows": len(descriptors_by_class[k]),
            "n_regions": len(descriptors),
        }
    arrays = {
        "vtcd_class_indices": np.asarray(indices),
        "vtcd_class_dictionaries": np.stack(dictionaries),
    }
    receipt = {"per_class": checks, "manifest_sha256": manifest["sha256"], "training_only": True}
    store.save("vtcd_class_references", reference_identity, arrays, receipt)
    return arrays, receipt


def _explainer(engine, source_record, meta, references, *, tn_features, tn_reference):
    """Recompute the promising maps on any clip of this window.

    Used by ``perturbed_views`` through :func:`evd.exp.explanation_campaign.explain_clip`,
    the single place of the computation.

    :param tn_features: already encoded normal reference, for ``ig_tn_same_video``.
    :return: ``clip -> (maps per method, controls, logit)``; maps have the grid shape.
    """
    scorer = engine.scorer
    methods = ("raw_attention", "gradient_activation", "ig")
    ig_kwargs = dict((meta.get("explanation_settings") or {}).get("ig", {}))
    if "steps" in ig_kwargs:
        ig_kwargs["steps"] = tuple(ig_kwargs["steps"])
    seed = _seed_from_window_id(source_record["window_id"])
    blur = int(meta.get("blur_kernel", 31))

    def explain(clip):
        maps, checks, feats, logit, _ = explain_clip(
            scorer,
            clip,
            methods,
            spec=engine.spec,
            ig_kwargs=ig_kwargs,
            blur_kernel=blur,
            noise_seed=seed,
            tn_features=tn_features,
            tn_receipt=tn_reference,
        )
        grid = tuple(feats.grid)
        if "direction" in references:

            def head_fp32(tokens, feats=feats):
                with torch.autocast(device_type=clip.device.type, enabled=False):
                    return scorer.head(replace(feats, tokens=tokens.float(), pooled=None))

            directions, receipt = direction_maps(head_fp32, feats.tokens.detach(), references)
            maps.update({k: v.reshape(grid) for k, v in directions.items()})
            checks["anomaly_direction"] = receipt
            prototypes = prototype_maps(feats.tokens[0].float().cpu().numpy(), references)
            maps.update({k: v.reshape(grid) for k, v in prototypes.items()})
        del feats
        return maps, checks, logit

    return explain


def extend_campaign(
    run: RunDemo,
    engine: InferenceEngine,
    source: Path,
    out: Path,
    *,
    references: dict[str, np.ndarray],
    reference_receipt: dict,
    manifest: dict,
    identity: dict,
    methods: tuple[str, ...] = ADVANCED_METHODS,
    max_new_windows: int | None = None,
    save_features: bool = True,
    video_root: Path | None = None,
) -> dict:
    """Run the registered methods on the exact windows of the source campaign.

    :param max_new_windows: pilot limit; valid results remain reusable.
    :param save_features: whether to write the final tokens of every window (35 MB each);
        an output policy, outside the artifact identity.
    :param video_root: root under which to look for the TN reference video when it belongs
        to another video (``perturbed_views`` only).
    :return: metadata in the renderer schema, with actual status and per-method receipts.
    :raises ValueError: on an incompatible source, geometry or fidelity.
    """
    meta = json.loads((source / "campaign.json").read_text(encoding="utf-8"))
    if meta["status"] != "complete" or not methods or set(methods) - set(ADVANCED_METHODS):
        raise ValueError("Incomplete source or unsupported methods")
    for key in (
        "run_id",
        "checkpoint_sha256",
        "backbone_sha256",
        "selected_step",
        "config_sha256",
        "preprocess",
        "precision",
        "tf32",
    ):
        if meta[key] != identity[key]:
            raise ValueError(f"The source differs from the current model: {key}, {source}")
    if file_sha256(Path(meta["video"])) != meta["video_sha256"]:
        raise ValueError(f"Different video: {meta['video']}")
    engine.sampler = run.sampler(engine.spec, validazione=meta["split"] == "val")
    video_meta = engine.decoder.probe(meta["video"])
    items = {item.window.window_id: item for item in engine._windows(meta["video"], video_meta)}
    tn_cache: list = []

    def tn_features():
        """Return the features of the source campaign normal reference, computed once.

        :return: ``BackboneOutput`` of the TN window, or ``None`` if the source campaign had
            none (``ig_tn_same_video`` is then declared skipped).
        """
        if not tn_cache:
            reference = meta.get("tn_reference") or {}
            window, video_id = reference.get("window_id"), reference.get("video_id")
            if not window:
                tn_cache.append(None)
                return tn_cache[0]
            if window in items:
                item = items[window]
            elif video_root is None:
                raise ValueError(
                    f"The TN reference is in {video_id}, another video: video_root is required"
                )
            else:
                other = _resolve_video(video_root, video_id)
                others = {
                    it.window.window_id: it
                    for it in engine._windows(other, engine.decoder.probe(other))
                }
                item = others[window]
            with torch.no_grad(), engine.scorer.autocast():
                tn_cache.append(
                    encode_multilevel(
                        engine.scorer.backbone, prepare_clip(engine, item), engine.scorer.levels
                    )
                )
        return tn_cache[0]

    source_records = meta["windows"]
    baseline_policy = identity["settings"].get("source_baseline", "strict")
    if baseline_policy not in ("strict", "current_host"):
        raise ValueError("Unknown original-logit policy")
    if any(
        row["window_id"] not in items or items[row["window_id"]].window != window_from_record(row)
        for row in source_records
    ):
        raise ValueError("The saved windows do not match the detector grid")
    store = ArtifactStore(out / "artifacts")
    expected = [r["window_id"] for r in source_records]
    campaign_identity = {
        "model": identity,
        "reference_manifest_sha256": manifest["sha256"],
        "references_sha256": identity_sha(reference_receipt),
        "video_sha256": meta["video_sha256"],
        "split": meta["split"],
        "expected_windows": expected,
    }
    previous = out / "campaign.json"
    if (
        previous.exists()
        and json.loads(previous.read_text(encoding="utf-8")).get("extension_identity")
        != campaign_identity
    ):
        raise ValueError(f"Different campaign identity: {out}")
    result = {
        **meta,
        "schema": 2,
        "status": "running",
        "methods": list(methods),
        "windows": [],
        "extension_identity": campaign_identity,
        "source_campaign": str(source.resolve()),
        "expected_window_ids": expected,
        "reference_diagnostics": reference_receipt,
        "method_notes": {
            "anomaly_projection": "TCAV-style: projection, not attribution; temporal labels",
            "anomaly_sensitivity": "TCAV-style: logit derivative, fp32 head; no TCAV statistical test",
            "prototype_difference": "anomaly cosine minus normality cosine; fixed prototypes from training",
            "vtcd_ids": "final VTCD: IDs independent across models; fixed colours, not annotated classes",
            "vtcd_importance": "final CRIS: BCE degradation with concept groups removed",
            "occlusion_blur": "Delta logit original - perturbed; blur occlusion",
            "occlusion_mean": "Delta logit original - perturbed; mean-colour occlusion",
            "occlusion_black": "Delta logit original - perturbed; normalized black pixel",
            "motion_local_freeze": "Delta logit original - perturbed; freeze of the volume centre",
        },
        "explanation_settings": identity["settings"],
        "tn_reference": None,
        "requested_window_ids": expected,
        "max_windows": None,
        "representation_formulas": {},
        "perturbations": False,
    }
    atomic_json(previous, result)
    rows_by_id = {
        r["window_id"]: r
        for r in (
            json.loads(line) for line in run.score_path.read_text().splitlines() if line.strip()
        )
        if r["split"] == meta["split"]
    }
    new_windows = 0
    for index, source_record in enumerate(source_records):
        row_identity = {**campaign_identity, "window": source_record["window"]}
        saved_methods = {
            method: store.load(f"w{index:05d}_{method}", {**row_identity, "method": method})
            for method in methods
        }
        pending = [m for m, v in saved_methods.items() if v is None]
        if pending and max_new_windows is not None and new_windows >= max_new_windows:
            break
        start = time.perf_counter()
        if pending:
            scorer = engine.scorer
            if str(scorer.device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            clip = prepare_clip(engine, items[source_record["window_id"]])
            feats, original = extract_features(engine, clip)
            delta = abs(original - source_record["logit"])
            if baseline_policy == "strict" and delta > identity["settings"]["fidelity_atol"]:
                raise ValueError(
                    f"The current logit does not reproduce the source: delta={delta}, {source_record['window_id']}"
                )

            def head(tokens, scorer=scorer, feats=feats):
                with scorer.autocast():
                    return scorer.head(replace(feats, tokens=tokens, pooled=None))

            def head_fp32(tokens, scorer=scorer, feats=feats, device_type=clip.device.type):
                with torch.autocast(device_type=device_type, enabled=False):
                    return scorer.head(replace(feats, tokens=tokens.float(), pooled=None))

            def detector(pixels, scorer=scorer):
                with scorer.autocast():
                    return scorer.head(encode_multilevel(scorer.backbone, pixels, scorer.levels))

            with torch.no_grad():
                roundtrip = float(head(feats.tokens).float().item())
                repeated = float(detector(clip).float().item())
            if abs(roundtrip - original) > identity["settings"]["fidelity_atol"]:
                raise ValueError("The head on the savable tokens does not reproduce the detector")
            if abs(repeated - original) > identity["settings"]["fidelity_atol"]:
                raise ValueError("The detector does not reproduce its own logit on the same clip")
            if save_features and store.load(f"w{index:05d}_features", row_identity) is None:
                # Explanation intermediate only, never a detector cache.
                store.save(
                    f"w{index:05d}_features",
                    row_identity,
                    {"tokens": feats.tokens[0].float().cpu().numpy()},
                    {"grid": feats.grid, "logit": original, "fidelity_delta": delta},
                )
            # Artifacts already computed for this window: dependent methods read them from here; the
            # same dictionary is shared across the window.
            siblings = {name: v for name, v in saved_methods.items() if v is not None}
            source_maps = None
            if "deletion_insertion" in pending:
                # Maps of the base campaign (attention, gradients, IG), already on disk and ranked
                # together with the concept maps.
                with np.load(source / f"maps_{index:05d}.npz") as archive:
                    source_maps = dict(archive)
            score_row = rows_by_id[source_record["window_id"]]
            context = MethodContext(
                clip,
                feats.tokens.detach(),
                tuple(feats.grid),
                original,
                int(score_row["label"]),
                detector,
                head,
                head_fp32,
                references,
                identity["settings"],
                tuple(engine.spec.mean),
                tuple(engine.spec.std),
                explain=_explainer(
                    engine,
                    source_record,
                    meta,
                    references,
                    tn_features=tn_features() if "perturbed_views" in pending else None,
                    tn_reference=meta.get("tn_reference") or {},
                ),
                siblings=siblings,
                source_maps=source_maps,
                # The class comes from `scores.jsonl`, i.e. `chunk_records`: the same rule used by the
                # per-class metrics.
                class_idx=int(score_row.get("class", -1)),
            )
            for method in pending:
                tick = time.perf_counter()
                print(f"Starting {run.run_id} {source_record['window_id']} {method}", flush=True)
                arrays, receipt = HANDLERS.get(method)(context)
                receipt.update(
                    elapsed_s=time.perf_counter() - tick,
                    original_logit=original,
                    source_logit=source_record["logit"],
                    source_score=source_record["score"],
                    source_baseline_policy=baseline_policy,
                    detector_repeat_delta=abs(repeated - original),
                    fidelity_delta=delta,
                    head_roundtrip_delta=abs(roundtrip - original),
                    peak_gpu_bytes=torch.cuda.max_memory_allocated() if clip.is_cuda else 0,
                )
                store.save(
                    f"w{index:05d}_{method}", {**row_identity, "method": method}, arrays, receipt
                )
                saved_methods[method] = (arrays, receipt)
                siblings[method] = saved_methods[method]
                print(f"Completato {method}: {receipt['elapsed_s']:.1f}s", flush=True)
            del context, clip, feats
            new_windows += 1
        maps, checks = {}, {}
        for method, (arrays, receipt) in saved_methods.items():
            maps.update(
                {
                    key.removeprefix("map__"): value
                    for key, value in arrays.items()
                    if key.startswith("map__")
                }
            )
            checks[method] = receipt
        atomic_npz(out / f"maps_{index:05d}.npz", maps)
        record = {
            **source_record,
            "source_logit": source_record["logit"],
            "source_score": source_record["score"],
            "logit": next(iter(checks.values()))["original_logit"],
            "score": float(
                torch.sigmoid(torch.tensor(next(iter(checks.values()))["original_logit"]))
            ),
            "checks": checks,
            "label": int(rows_by_id[source_record["window_id"]]["label"]),
            "total_s": sum(c["elapsed_s"] for c in checks.values()),
            "map_statistics": {
                method: {
                    "minimum": float(v.min()),
                    "maximum": float(v.max()),
                    "null": bool(not v.any()),
                }
                for method, v in maps.items()
            },
        }
        result["windows"].append(record)
        atomic_json(previous, result)
        print(
            f"Windows {source.name}: {index + 1}/{len(expected)}, new={bool(pending)}, {time.perf_counter() - start:.1f}s",
            flush=True,
        )
    result["status"] = "complete" if len(result["windows"]) == len(expected) else "running"
    result["all_five_methods_complete"] = result["status"] == "complete" and set(methods) == set(
        ADVANCED_METHODS
    )
    result["new_windows_this_invocation"] = new_windows
    atomic_json(previous, result)
    return result


def prepare_previews(folder: Path) -> None:
    """Save the preprocessed samples to draw ablations without models.

    :param folder: completed campaign; reuses decoder and WindowDataset on CPU.
    :raises ValueError: if the video changed with respect to the campaign.
    """
    from evd.data.decoder import make_decoder

    meta = json.loads((folder / "campaign.json").read_text(encoding="utf-8"))
    if meta["status"] != "complete":
        return
    if file_sha256(Path(meta["video"])) != meta["video_sha256"]:
        raise ValueError("The video to visualize differs from the source")
    spec = PreprocessSpec(**meta["preprocess"])
    decoder = make_decoder("pyav")
    store = ArtifactStore(folder / "preview_inputs")
    manifest = []
    for index, record in enumerate(meta["windows"]):
        key = f"w{index:05d}"
        identity = {
            "video_sha256": meta["video_sha256"],
            "preprocess": meta["preprocess"],
            "window": record["window"],
        }
        if store.load(key, identity) is None:
            item = WindowItem(window_from_record(record), meta["video"])
            clip = WindowDataset([item], decoder, spec)[0]["clip"]
            values = clip.permute(1, 2, 3, 0).numpy()
            rgb = np.rint(
                np.clip(values * np.asarray(spec.std) + np.asarray(spec.mean), 0, 1) * 255
            ).astype(np.uint8)
            store.save(
                key,
                identity,
                {"rgb": rgb},
                {"meaning": "preprocessed samples, affine normalization inverted for display"},
            )
        manifest.append(
            {"file": f"preview_inputs/{key}.npz", "sha256": file_sha256(store.root / f"{key}.npz")}
        )
    atomic_json(folder / "preview_manifest.json", {"inputs": manifest})


def cosine_from_saved_features(source: Path, out: Path, query_campaign: Path) -> dict:
    """Extend the cosine method to the saved windows with an explicit, fixed query.

    :param source: advanced campaign with receipts and final tokens per window.
    :param out: separate folder, preserving existing maps and videos.
    :param query_campaign: original cosine campaign with query.npy and provenance.
    :return: cosine campaign with the same scores and geometry.
    :raises ValueError: on incompatible identities, corrupted features or an inconsistent
        query.
    """
    from evd.explain.project import offline_plane
    from evd.explain.representation import cosine_similarity

    meta = json.loads((source / "campaign.json").read_text(encoding="utf-8"))
    query_meta = json.loads((query_campaign / "campaign.json").read_text(encoding="utf-8"))
    if meta["status"] != "complete" or query_meta["status"] != "complete":
        raise ValueError("Cosine: incomplete source or query")
    keys = (
        "run_id",
        "checkpoint_sha256",
        "backbone_sha256",
        "config_sha256",
        "selected_step",
        "preprocess",
        "precision",
        "tf32",
    )
    if any(meta[key] != query_meta[key] for key in keys):
        raise ValueError("Cosine: the query belongs to a different model or preprocessing")
    query_file = query_campaign / "query.npy"
    query = np.load(query_file, allow_pickle=False)
    query_record = query_meta["query"]
    identity = {
        "model": {key: meta[key] for key in keys},
        "source_campaign_sha256": file_sha256(source / "campaign.json"),
        "query_sha256": file_sha256(query_file),
        "query": query_record,
        "formula": "cosine_similarity(tokens, fixed_query)",
    }
    store = ArtifactStore(out / "artifacts")
    feature_store = ArtifactStore(source / "artifacts")
    result = {
        **meta,
        "status": "running",
        "methods": ["cosine"],
        "windows": [],
        "cosine_identity": identity,
        "query": {
            **query_record,
            "video": query_meta["video"],
            "source_campaign": str(query_campaign.resolve()),
            "scope": "same explicit query for all windows and videos of the model",
        },
        "source_campaign": str(source.resolve()),
        "method_notes": {
            "cosine": f"Fixed query from {Path(query_meta['video']).stem}, frame {query_record['frame']}; similarity, not anomaly probability"
        },
    }
    previous = out / "campaign.json"
    if (
        previous.exists()
        and json.loads(previous.read_text(encoding="utf-8")).get("cosine_identity") != identity
    ):
        raise ValueError("Different cosine identity: choose a new folder")
    atomic_json(previous, result)
    for i, record in enumerate(meta["windows"]):
        start = time.perf_counter()
        key = f"w{i:05d}_features"
        feature_receipt = json.loads(
            (feature_store.root / f"{key}.json").read_text(encoding="utf-8")
        )
        feature_identity = feature_receipt["identity"]
        if (
            feature_identity["window"] != record["window"]
            or feature_identity["model"] != meta["extension_identity"]["model"]
        ):
            raise ValueError("Feature identity differs from the campaign")
        arrays, receipt = feature_store.load(key, feature_identity)
        tokens = arrays["tokens"]
        grid = tuple(receipt["grid"])
        if tokens.shape[0] != int(np.prod(grid)) or tokens.shape[1:] != query.shape:
            raise ValueError("Incompatible grid or query dimension")
        row_identity = {
            **identity,
            "window": record["window"],
            "features_sha256": feature_receipt["payload_sha256"],
        }
        saved = store.load(f"w{i:05d}_cosine", row_identity)
        if saved is None:
            # On the origin window the query must be reproducible from the tokens.
            query_error = None
            if record["window_id"] == query_record["window_id"]:
                plane = offline_plane(window_from_record(record), query_record["frame"], grid[0])
                region = tokens.reshape(*grid, -1)[
                    plane,
                    query_record["y0"] : query_record["y1"],
                    query_record["x0"] : query_record["x1"],
                ]
                query_error = float(np.max(abs(region.mean((0, 1)) - query)))
                if query_error > 1e-6:
                    raise ValueError(f"The saved query is not reproduced: {query_error}")
            values = cosine_similarity(tokens, query).reshape(grid).astype(np.float32)
            checks = {
                "elapsed_s": time.perf_counter() - start,
                "query_reconstruction_max_error": query_error,
                "features_sha256": feature_receipt["payload_sha256"],
                "query_sha256": identity["query_sha256"],
                "new_detector_forwards": 0,
            }
            store.save(f"w{i:05d}_cosine", row_identity, {"map__cosine": values}, checks)
        else:
            values, checks = saved[0]["map__cosine"], saved[1]
        atomic_npz(out / f"maps_{i:05d}.npz", {"cosine": values})
        result["windows"].append(
            {
                **record,
                "checks": {"cosine": checks},
                "total_s": checks["elapsed_s"],
                "map_statistics": {
                    "cosine": {
                        "minimum": float(values.min()),
                        "maximum": float(values.max()),
                        "null": bool(not values.any()),
                    }
                },
            }
        )
        atomic_json(previous, result)
    result["status"] = "complete"
    atomic_json(previous, result)
    return result
