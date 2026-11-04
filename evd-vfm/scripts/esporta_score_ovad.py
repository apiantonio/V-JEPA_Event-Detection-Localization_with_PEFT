#!/usr/bin/env python
"""Convert EDP-v2 scores into the original OVAD format (docs/19).

Reuses ``rileggi_run`` and the sampler: no inference, interpolation or new threshold
here. Explicit extremes preserve the frame rounding.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "ovad_protocol")]

from evaluate_ovad import from_triples  # noqa: E402
from protocol_core.inputs import build_entries  # noqa: E402

from evd.eval import metrics as M  # noqa: E402
from evd.exp.rilettura import data_group, reread_run  # noqa: E402
from evd.utils.hashing import file_sha256  # noqa: E402
from evd.utils.io import read_json, read_jsonl, write_json  # noqa: E402


def convert(read_value: Any, originali: list[dict], expected: set[str]) -> tuple[dict, dict, dict]:
    """Convert a partition, checking coverage, duplicates and geometry.

    :param read_value: ``rileggi_run`` output for the requested partition.
    :param originali: original rows of the requested partition only.
    :param attesi: complete set of videos of the authoritative partition.
    :return: labels, OVAD scores and the receipt of the checks.
    :raises ValueError: if the scores do not cover exactly the expected videos and windows.
    """
    if set(read_value.scores) != expected or not expected:
        raise ValueError(
            f"Coverage of {read_value.split} differs from the split: missing "
            f"{sorted(expected - set(read_value.scores))[:5]}, extra "
            f"{sorted(set(read_value.scores) - expected)[:5]}. Restore the complete scores."
        )
    index = {}
    for r in originali:
        k = (r["video_id"], r["window_id"])
        if k in index:
            raise ValueError(f"Duplicated score {k}: fix the source, do not deduplicate.")
        index[k] = r
    temporal = read_value.run.protocol["temporal"]
    if temporal["window_grid"] != "fixed":
        raise ValueError("The EDP-v2 fixed grid is required; do not reinterpret endpoint windows.")
    labels, scores, used = {}, {}, set()
    for video, windows in read_value.scores.items():
        meta = read_value.bundle.meta(video)
        fps = float(meta.fps)
        labels[video] = {
            "duration": meta.num_frames / fps,
            "anomalies": [
                [a / fps, (b + 1) / fps, read_value.bundle.classes[cl].name]
                for a, b, cl in read_value.gt[video].intervals
            ],
        }
        scores[video] = []
        for w in windows:
            k = (video, w.window.window_id)
            r = index[k]
            start, end = w.window.start_frame, w.window.end_frame
            if (r["start_frame"], r["end_frame"]) != (start, end):
                raise ValueError(f"Inconsistent extremes for {k}: rebuild the provenance.")
            if not math.isfinite(w.anomaly) or float(r["score"]) != w.anomaly:
                raise ValueError(f"Non-finite or altered score for {k}.")
            scores[video].append({"start": start / fps, "end": (end + 1) / fps, "score": w.anomaly})
            used.add(k)
        # The delivered code truncates the timeline at the last chunk: a missing tail must stop the
        # export, not disappear from the denominators.
        if (
            windows[0].window.start_frame != 0
            or windows[-1].window.end_frame != meta.num_frames - 1
        ):
            raise ValueError(f"{video}: the scores do not cover the whole video duration.")
        if any(
            b.window.start_frame > a.window.end_frame + 1
            for a, b in zip(windows, windows[1:], strict=False)
        ):
            raise ValueError(
                f"{video}: gap between windows; inference on the complete grid is required."
            )
    if used != set(index):
        raise ValueError(
            "Extra windows in the scores: replay would ignore them, OVAD rejects them."
        )
    hp = {
        "chunk_size": float(temporal["window_s"]),
        "asking_rate": float(
            temporal.get("val_hop_s", temporal["hop_s"])
            if read_value.split == "val"
            else temporal["hop_s"]
        ),
        "grid": "start",
    }
    packaged = {"hyperparameters": hp, "scores": scores}
    build_entries(from_triples(labels, read_value.split), packaged)
    old = {
        "chunk_auc": M.dataset_chunk_auc(read_value.scores, read_value.gt),
        "chunk_ap": M.dataset_chunk_ap(read_value.scores, read_value.gt),
    }
    delta = {}
    if read_value.split == read_value.run.metrics.get("eval_split", "test"):
        for key, value in old.items():
            delta[key] = abs(value - float(read_value.run.metrics[key]))
            if not math.isfinite(delta[key]) or delta[key] > 1e-9:
                raise ValueError(f"{key}: deviation {delta[key]} from the original metrics.")
    return (
        labels,
        packaged,
        {
            "n_video": len(labels),
            "n_chunk": len(used),
            "split": read_value.split,
            "griglia_verificata": True,
            "copertura_completa": True,
            "metriche_originali_ricalcolate": old,
            "scarti_metriche_originali": delta,
            "annotation_source": read_value.bundle.annotation_source,
        },
    )


def esporta(run: Path, out: Path, *, split: str, overrides: list[str], manifest: Path) -> dict:
    """Write the two OVAD JSON files and the verified provenance of the conversion.

    :param manifest: sealed UCF split; for MIVIA the datasource catalogue applies.
    :return: receipt with the hashes of sources and outputs.
    :raises ValueError: if split, annotations or scores do not match the source.
    """
    read_value = reread_run(run, split=split, overrides=overrides)
    inputs = [run / name for name in ("scores.jsonl", "config.yaml", "metrics.json", "meta.json")]
    if read_value.bundle.dataset_id == "ucf":
        split_data = read_json(manifest)
        if split_data["annotation_source"] != read_value.bundle.annotation_source:
            raise ValueError("Annotations differ from the sealed split: check the delivery.")
        expected = set(split_data[split])
        inputs.append(manifest)
    elif split == "test":
        expected = set(read_value.bundle.query_videos)
    else:
        raise ValueError(
            "The external dataset has no validation: use the one of the source UCF run."
        )
    originals = [r for r in read_jsonl(run / "scores.jsonl") if r.get("split", "test") == split]
    labels, scores, receipt = convert(read_value, originals, expected)
    data = data_group(read_value.run, overrides)
    inputs.append(Path(str(data.inventory)))
    inputs.extend(sorted(Path(str(data.annotations.path)).glob("*.txt")))
    inputs.extend(sorted(Path(str(data.annotations.path)).glob("*.json")))
    write_json(out / "labels.json", labels)
    write_json(out / "scores.json", scores)
    receipt.update(
        {
            "run_id": run.name,
            "dataset": read_value.bundle.dataset_id,
            "hyperparameters": scores["hyperparameters"],
            "sorgenti_sha256": {str(p.resolve()): file_sha256(p) for p in inputs},
            "output_sha256": {
                name: file_sha256(out / name) for name in ("labels.json", "scores.json")
            },
            "conversione": "Inclusive frames [a,b] -> seconds [a/fps,(b+1)/fps); scores unchanged",
        }
    )
    write_json(out / "provenienza.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "manifests/edp_v2/ucf_crime/split.json"
    )
    args = parser.parse_args()
    print(
        esporta(
            args.run, args.out, split=args.split, overrides=args.overrides, manifest=args.manifest
        )
    )


if __name__ == "__main__":
    main()
