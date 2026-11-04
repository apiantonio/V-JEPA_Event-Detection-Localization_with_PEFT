"""Derive P16 from the sealed IG stage and its validation calibration (no GPU).

P16 thresholds the same IG ensemble map as P12–P15 with an absolute value, the 0.99
quantile of the per-chunk maxima on every normal validation chunk (A1). The derived stage
copies window, grid, score and gate of each IG chunk record, applies the threshold to the
saved ``ig_ensemble`` map and charges the IG computation (the P14 record, whose only extra
work is its own threshold) plus the new threshold. The output layout is the evaluator's.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402

from evd.exp.localization_auto import file_sha256, read_json, seal_stage, write_json  # noqa: E402
from evd.explain.localization import MaskRule, map_mask, native_boxes  # noqa: E402

EXPERIMENT = "P16-normal_q99"


def derive(
    ig: Path, calibration: Path, gate: Path, output: Path, manifests: dict[str, Path]
) -> None:
    threshold = read_json(calibration)
    rule = MaskRule("absolute", threshold["threshold"])
    fit = read_json(gate)["fit_seconds"] + threshold["fit_seconds"]
    for dataset, manifest in manifests.items():
        seal = read_json(ig / "operations" / f"seal-{dataset}.json")
        for entry in seal["videos"]:
            source = ig / entry["directory"]
            if file_sha256(source / "complete.json") != entry["complete_sha256"]:
                raise ValueError("IG prediction changed after sealing")
            complete = read_json(source / "complete.json")
            target = output / entry["directory"]
            chunks = []
            for item in complete["chunks"]:
                record = read_json(source / item["path"])
                shared = (
                    record["timing"]["s0_decode_preprocess_seconds"]
                    + record["timing"]["s0_encode_score_seconds"]
                )
                width, height = complete["video"]["width"], complete["video"]["height"]
                # A merged stage (17 October) marks the chunks it localized and charges only
                # those; the gated layout charges S0 on every chunk.
                merged = "localized" in record
                active = record["localized"] if merged else record["gate_passed"]
                if active:
                    start = time.perf_counter()
                    index = Path(item["path"]).stem
                    ensemble = np.load(source / "maps" / f"{index}.npz")["ig_ensemble"]
                    boxes = native_boxes(map_mask(ensemble, rule), width, height)
                    own = (
                        record["methods"]["P14-mass70"]["method_seconds"]
                        + time.perf_counter()
                        - start
                    )
                else:
                    boxes, own = [[] for _ in range(record["grid"][0])], 0.0
                derived = {
                    **{
                        k: record[k]
                        for k in ("window", "window_id", "grid", "score", "gate_passed")
                    },
                    "identity": {
                        "stage": "p16_from_ig",
                        "ig_seal_sha256": file_sha256(ig / "operations" / f"seal-{dataset}.json"),
                        "calibration_sha256": file_sha256(calibration),
                        "threshold": threshold["threshold"],
                    },
                    "methods": {
                        EXPERIMENT: {
                            "boxes": boxes,
                            "method_seconds": own,
                            "inference_compute_seconds": shared + own
                            if active or not merged
                            else 0.0,
                        }
                    },
                    **({"localized": active} if merged else {}),
                }
                path = target / item["path"]
                write_json(path, derived)
                chunks.append({"path": item["path"], "sha256": file_sha256(path)})
            records = [read_json(target / c["path"]) for c in chunks]
            write_json(
                target / "complete.json",
                {
                    "status": "complete",
                    "identity": records[0]["identity"] if records else {},
                    "video": complete["video"],
                    "chunks": chunks,
                    "timing": {
                        EXPERIMENT: {
                            "fit_seconds": fit,
                            "inference_compute_seconds": sum(
                                r["methods"][EXPERIMENT]["inference_compute_seconds"]
                                for r in records
                            ),
                            "inference_chunks": sum(r.get("localized", True) for r in records),
                            "fit_shared_cost_do_not_sum_over_videos": True,
                        }
                    },
                    "prediction_scope": "entire_video_all_protocol_windows",
                },
            )
        seal_stage(output, dataset, [EXPERIMENT], manifest)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--ig", type=Path, required=True, help="sealed IG stage root")
    parser.add_argument("--calibration", type=Path, required=True, help="p16.json")
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--ucf-manifest", type=Path, required=True)
    parser.add_argument("--mivia-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    derive(
        args.ig,
        args.calibration,
        args.gate,
        args.output,
        {"ucf_test": args.ucf_manifest, "mivia_test": args.mivia_manifest},
    )


if __name__ == "__main__":
    main()
