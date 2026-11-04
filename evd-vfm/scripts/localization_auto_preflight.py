"""Audit inputs, validation calibration and GPU readouts without evaluating test GT."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from evd.data.inventory import read_records  # noqa: E402
from evd.eval.scorers import anomaly_from_logits  # noqa: E402
from evd.exp.demo import live_engine, open_run  # noqa: E402
from evd.exp.explanation_advanced import extract_features, prepare_clip  # noqa: E402
from evd.explain.attribution import explain_head  # noqa: E402
from evd.explain.localization import MaskRule, map_mask, native_boxes  # noqa: E402
from evd.protocol.chunks import sampler_for_protocol  # noqa: E402
from evd.utils.hashing import file_sha256  # noqa: E402

#: Detectors the campaign can localize with: run, sha256 of its checkpoint, selected step.
#: The default is the campaign detector; ``dora-r32-s42`` is follow-up F of 17 October;
#: ``e2d-attentive-s42`` is the frozen backbone with the attentive probe (20 October: the
#: model without PEFT, compared with the final proposal).
DETECTORS = {
    "e3s-vptshallow-su-lora-r32-s42": (
        "e3s_ucf_vjepa21l_vptshallow-n8-su-lora-r32_s42",
        "9d1901dfb4d95c361d2aa6258977a8b069a97a7530df467aec056ef562635625",
        12880,
    ),
    "dora-r32-s42": (
        "e3r_ucf_vjepa21l_dora-r32-qkvomlp_s42",
        "bd840d553f08bc36d291b840b7c2d4851c5bfed8baf834f875918704cb4423eb",
        5520,
    ),
    "e2d-attentive-s42": (
        "e2d_ucf_vjepa21l_frozen-attentive_s42",
        "d903645999ab5a0e823cc10996628e71d7bd9350e7cf2db44945b41c4e1f88ad",
        4600,
    ),
}
# Chosen by the job environment (set on the cron/queue line, inherited by sbatch); every
# script then checks the checkpoint sha and the gate against it, so a mixed detector fails.
DETECTOR = os.environ.get("EVD_LOC_DETECTOR", "e3s-vptshallow-su-lora-r32-s42")
if DETECTOR not in DETECTORS:
    raise ValueError(f"Unknown EVD_LOC_DETECTOR {DETECTOR!r}; known: {sorted(DETECTORS)}")
RUN, CHECKPOINT_SHA, SELECTED_STEP = DETECTORS[DETECTOR]
READOUTS = {
    "P21-postnorm": "gradient_activation",
    "P37-last": "legrad_last",
    "P38-postnorm": "gradcam",
    "P38-prenorm": "gradcam_prenorm",
    "P39-postnorm": "layercam",
}


def write(path: Path, value: object) -> None:
    """Publish a complete JSON receipt atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def calibrate(
    scores: dict,
    normal_ids: set[str],
    quantile: float,
    source: str = "checkpoint_selected_validation_scores",
) -> dict:
    """Weighted empirical quantile: equal video weight, equal chunks within video."""
    values = [
        (key, float(value)) for key, value in scores.items() if key.rsplit("@", 1)[0] in normal_ids
    ]
    counts = Counter(key.rsplit("@", 1)[0] for key, _ in values)
    if set(counts) != normal_ids or not values:
        raise ValueError("Selected validation scores do not cover every normal validation video")
    ordered = sorted(values, key=lambda row: (row[1], row[0]))
    cumulative, threshold = 0.0, None
    for key, value in ordered:
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid validation probability")
        cumulative += 1 / (len(counts) * counts[key.rsplit("@", 1)[0]])
        if cumulative >= quantile:
            threshold = value
            break
    if threshold is None:
        raise ValueError("Validation quantile was not reached")
    return {
        "threshold": threshold,
        "quantile": quantile,
        "normal_videos": len(counts),
        "normal_chunks": len(values),
        "source": source,
        "weights": "equal_video_then_equal_chunk",
        "comparison": "strict_greater",
    }


def selected_probabilities(logits: dict) -> dict[str, float]:
    """Convert checkpoint validation logits with the detector's score definition."""
    ids = list(logits)
    tensor = torch.tensor([float(logits[key]) for key in ids], dtype=torch.float32)[:, None]
    if not torch.isfinite(tensor).all():
        raise ValueError("Nonfinite selected validation logits")
    values, _ = anomaly_from_logits(tensor, batch_size=len(ids))
    return dict(zip(ids, values.tolist(), strict=True))


def validation_probabilities(run, checkpoint: dict) -> tuple[dict[str, float], str, str]:
    """Validation probabilities of the selected weights, with their provenance.

    Checkpoints written since ADR-053 carry the selected step's validation logits. Older
    runs (E2d, resumed on 1 September) do not; their ``scores.jsonl`` validation rows were
    scored with the same selected weights, so they are the same quantity.

    :return: ``window_id`` to probability, the gate ``source`` and the stored value type.
    :raises ValueError: if neither source exists.
    """
    if checkpoint.get("selected_scores"):
        return (
            selected_probabilities(checkpoint["selected_scores"]),
            "checkpoint_selected_validation_scores",
            "binary_logit_converted_with_anomaly_from_logits",
        )
    if not run.score_path.exists():
        raise ValueError("No selected validation scores in the checkpoint nor in scores.jsonl")
    from evd.infer.replay import scores_from_disk

    videos = scores_from_disk(run.score_path, split="val")
    probabilities = {
        key: value for video in videos.values() for key, value in video.per_finestra.items()
    }
    return probabilities, "run_scores_jsonl_validation_rows", "probability_as_written"


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    run = open_run(Path(os.environ["EVD_RUNS_DIR"]) / RUN)
    checkpoint_sha = file_sha256(run.checkpoint_path)
    if checkpoint_sha != CHECKPOINT_SHA:
        raise ValueError(
            "The selected detector checkpoint changed; resolve its identity before launch"
        )
    checkpoint = torch.load(run.checkpoint_path, map_location="cpu", weights_only=True)
    adapted = run.cfg["adapt"].get("family", "none") != "none"
    if (
        checkpoint.get("selected_step") != SELECTED_STEP
        or not checkpoint.get("selected_head")
        or (adapted and not checkpoint.get("selected_adapter"))
    ):
        raise ValueError("Selected step/head/adapter are missing or changed")
    selected_scores, score_source, value_type = validation_probabilities(run, checkpoint)
    del checkpoint
    split_path = ROOT / "manifests/edp_v2/ucf_crime/split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    records = read_records(run.cfg["data"]["inventory"])
    ucf = {row.video_id: row for row in records}
    normal_val = set()
    annotation_path = (
        Path(run.cfg["data"]["annotations"]["path"]) / "FS-UCF-Crime_official_annotations.txt"
    )
    for line in annotation_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts and Path(parts[0]).stem in set(split["val"]) and len(parts) == 1:
            normal_val.add(Path(parts[0]).stem)
    fit_start = time.perf_counter()
    gate = calibrate(selected_scores, normal_val, 0.99, score_source)
    gate["fit_seconds"] = time.perf_counter() - fit_start
    gate["checkpoint_sha256"] = checkpoint_sha
    gate["stored_value_type"] = value_type
    gate["historical_detector_validation_cost"] = "shared_detector_cost_not_localizer_fit"
    write(args.output / "gate.json", gate)
    manifests = {}
    coverage = {}
    for name in ("train", "val", "test"):
        rows = []
        for video in split[name]:
            meta = ucf[video].to_meta()
            item = asdict(meta)
            item["path"] = str(Path(run.cfg["data"]["root"]) / ucf[video].relative_path)
            if not Path(item["path"]).is_file():
                raise FileNotFoundError(item["path"])
            rows.append(item)
        manifests[f"ucf_{name}"] = rows
        coverage[f"ucf_{name}"] = len(rows)
        write(args.output / f"inputs-ucf-{name}.json", rows)
    mivia_inventory = Path(os.environ["EVD_DATA_DIR"]) / "mivia_crime_inventory/inventory.csv"
    mivia = []
    for row in read_records(mivia_inventory):
        item = asdict(row.to_meta())
        item["path"] = str(Path(os.environ["EVD_MIVIA_DIR"]) / row.relative_path)
        if not Path(item["path"]).is_file():
            raise FileNotFoundError(item["path"])
        mivia.append(item)
    if len(mivia) != 270 or coverage != {"ucf_train": 1303, "ucf_val": 306, "ucf_test": 290}:
        raise ValueError(f"Unexpected dataset coverage: {coverage}, MIVIA {len(mivia)}")
    coverage["mivia_test"] = len(mivia)
    write(args.output / "inputs-mivia-test.json", mivia)
    write(
        args.output / "input-coverage.json",
        {
            "status": "complete",
            "videos": coverage,
            "split_sha256": file_sha256(split_path),
            "test_manifests_contain_labels": False,
            "metadata_elapsed_seconds": time.perf_counter() - started,
        },
    )
    if args.metadata_only:
        print(json.dumps({"status": "metadata_complete", "coverage": coverage}))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("GPU preflight requires working CUDA; CPU fallback is forbidden")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    load_start = time.perf_counter()
    engine, spec = live_engine(run, device="cuda", batch_size=1, num_workers=0)
    engine.sampler = sampler_for_protocol(spec, run.protocol, validazione=True)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    pilot_ids = sorted(normal_val)[:2] + sorted(set(split["val"]) - normal_val)[:2]
    receipts = []
    for video in pilot_ids:
        meta = ucf[video].to_meta()
        path = Path(run.cfg["data"]["root"]) / ucf[video].relative_path
        items = engine.finestre(path, meta)
        item = items[len(items) // 2]
        torch.cuda.synchronize()
        before = time.perf_counter()
        clip = prepare_clip(engine, item)
        torch.cuda.synchronize()
        decoded = time.perf_counter()
        feats, logit = extract_features(engine, clip)
        torch.cuda.synchronize()
        encoded = time.perf_counter()
        probability = float(torch.sigmoid(torch.tensor(logit)).item())
        expected = float(selected_scores[item.window.window_id])
        replay_delta = abs(probability - expected)
        if replay_delta > 0.011001:
            raise RuntimeError(
                f"Selected validation replay mismatch: {item.window.window_id}: {replay_delta}"
            )
        with engine.scorer.autocast():
            result = explain_head(engine.scorer.head, feats, extra_cams=True)
        torch.cuda.synchronize()
        attributed = time.perf_counter()
        if abs(result.logit - logit) > 1e-5:
            raise RuntimeError("Attribution hooks changed the original logit")
        outputs = {}
        for experiment, readout in READOUTS.items():
            post_start = time.perf_counter()
            mask = map_mask(result.maps[readout], MaskRule())
            boxes = native_boxes(mask, meta.width, meta.height)
            outputs[experiment] = {
                "postprocess_seconds": time.perf_counter() - post_start,
                "map_shape": list(mask.shape),
                "finite": bool(np.isfinite(result.maps[readout]).all()),
                "planes": len(boxes),
                "boxes": sum(map(len, boxes)),
            }
        receipts.append(
            {
                "video": video,
                "window_id": item.window.window_id,
                "replay_absolute_delta": replay_delta,
                "hook_logit_delta": abs(result.logit - logit),
                "attention_reconstruction_error": result.fidelity,
                "decode_preprocess_seconds": decoded - before,
                "encode_score_seconds": encoded - decoded,
                "shared_head_readouts_seconds": attributed - encoded,
                "methods": outputs,
            }
        )
        del clip, feats, result
    receipt = {
        "status": "complete",
        "scope": "validation_only_technical_preflight_no_test_quality",
        "run_id": RUN,
        "checkpoint_sha256": checkpoint_sha,
        "selected_step": SELECTED_STEP,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "precision": engine.scorer.precision,
        "tf32": False,
        "batch_size": 1,
        "threads": 1,
        "load_seconds": load_seconds,
        "first_sample_includes_warmup": True,
        "fit": gate,
        "pilot": receipts,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write(args.output / "gpu-preflight.json", receipt)
    print(json.dumps({"status": "complete", "gpu": receipt["gpu"], "pilot_windows": len(receipts)}))


if __name__ == "__main__":
    main()
