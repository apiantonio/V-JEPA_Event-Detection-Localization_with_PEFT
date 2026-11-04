"""Re-threshold saved maps of sealed stages with other dynamic rules (no GPU, no test GT).

Follow-up of 17 October: the A1 absolute threshold leaves the memory
and learned-localizer maps almost empty, so the same maps are thresholded with the mass-70
rule of P08/P27 (proposal 2); P13 is re-run with other token fractions (proposal 3). Every
rule is computed by the method on its own chunk map, as in the original stages.

Each derived chunk record copies window, grid, score and gate of the source record; its
inference cost is the source method's own work (the map) plus the new threshold, plus S0
decode and encode. The fit drops the A1 calibration the relative rule no longer needs. The
output layout and the seal are the evaluator's (``localization_auto_report.py``).
Generalizes ``localization_auto_derive.py``, which stays as the record of how P16 was made.
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

MASS70 = MaskRule("mass", 0.7)
#: experiment -> (source stage, source method, saved map key, rule, drops the A1 calibration)
RULES = {
    **{
        f"{name}-mass70": ("memory", name, name, MASS70, True)
        for name in (
            "P24-bank25000",
            "P25-contexts5",
            "P26-gaussian1",
            "P28-transitions",
            "P29-slots128",
            "P30-linear",
            "P32-top10",
        )
    },
    "P13-count05": ("ig", "P13-count10", "ig_ensemble", MaskRule("count", 0.05), False),
    "P13-count20": ("ig", "P13-count10", "ig_ensemble", MaskRule("count", 0.2), False),
}
METHODS = list(RULES)
#: Third follow-up of 17 October: P13 below 5% of the tokens, and the follow-up rules for a
#: second detector (``--methods`` picks any of them).
P13_SWEEP = {
    f"P13-count{round(100 * q):02d}": (
        "ig",
        "P13-count10",
        "ig_ensemble",
        MaskRule("count", q),
        False,
    )
    for q in (0.01, 0.02, 0.03)
}
ALL_RULES = {**RULES, **P13_SWEEP}


def a1_calibration_seconds(learn: Path) -> float:
    """:return: validation-normal extraction plus threshold calibration of the learn step."""
    receipt = read_json(learn)
    return receipt["extraction_seconds"]["val_normal"] + receipt["calibration_seconds"]


def derive(
    predictions: Path,
    learn: Path | None,
    manifests: dict[str, Path],
    output: Path,
    rules: dict | None = None,
) -> None:
    """Write and seal the re-thresholded stage for both test datasets.

    :param predictions: root holding the sealed ``memory`` and ``ig`` stages.
    :param learn: ``fits/learn.json``, for the calibration time the new rules do not need;
        may be ``None`` when no rule drops the A1 calibration.
    :param manifests: prediction input manifest per dataset.
    :param output: root of the new stage.
    :param rules: experiment -> rule spec, default ``RULES``.
    :raises ValueError: if a source changed after sealing or its windows disagree, or if a
        rule drops the A1 calibration and ``learn`` is missing.
    """
    rules = RULES if rules is None else rules
    methods_ = list(rules)
    drops = any(spec[4] for spec in rules.values())
    if drops and learn is None:
        raise ValueError("A rule drops the A1 calibration: --learn is required")
    calibration = a1_calibration_seconds(learn) if drops else 0.0
    sources = sorted({spec[0] for spec in rules.values()})
    for dataset, manifest in manifests.items():
        seals = {
            s: read_json(predictions / s / "operations" / f"seal-{dataset}.json") for s in sources
        }
        seal_sha = {
            s: file_sha256(predictions / s / "operations" / f"seal-{dataset}.json") for s in sources
        }
        directories = {s: [v["directory"] for v in seals[s]["videos"]] for s in sources}
        if len({tuple(d) for d in directories.values()}) != 1:
            raise ValueError("Source stages cover different videos")
        identity = {
            "stage": "rethreshold",
            "source_seal_sha256": seal_sha,
            "rules": {
                k: {"source": v[1], "map": v[2], "rule": vars(v[3])} for k, v in rules.items()
            },
        }
        for position, directory in enumerate(directories[sources[0]]):
            completes = {}
            for s in sources:
                entry = seals[s]["videos"][position]
                path = predictions / s / directory / "complete.json"
                if file_sha256(path) != entry["complete_sha256"]:
                    raise ValueError(f"{s} prediction changed after sealing")
                completes[s] = read_json(path)
            reference = completes[sources[0]]
            target = output / directory
            chunks, totals, localized = [], dict.fromkeys(methods_, 0.0), 0
            for item in reference["chunks"]:
                records = {
                    s: read_json(predictions / s / directory / item["path"]) for s in sources
                }
                base = records[sources[0]]
                if any(r["window_id"] != base["window_id"] for r in records.values()):
                    raise ValueError("Source stages disagree on the windows")
                shared = (
                    base["timing"]["s0_decode_preprocess_seconds"]
                    + base["timing"]["s0_encode_score_seconds"]
                )
                width, height = reference["video"]["width"], reference["video"]["height"]
                planes = base["grid"][0]
                maps = {}
                methods = {}
                merged = "localized" in base
                active = base["localized"] if merged else base["gate_passed"]
                localized += active or not merged
                for name, (stage, source, key, rule, _) in rules.items():
                    # A merged stage (17 October) marks the chunks it localized and charges
                    # only those; the gated layout charges S0 on every chunk.
                    if active:
                        if stage not in maps:
                            maps[stage] = np.load(
                                predictions
                                / stage
                                / directory
                                / "maps"
                                / f"{Path(item['path']).stem}.npz"
                            )
                        start = time.perf_counter()
                        boxes = native_boxes(map_mask(maps[stage][key], rule), width, height)
                        own = (
                            records[stage]["methods"][source]["method_seconds"]
                            + time.perf_counter()
                            - start
                        )
                    else:
                        boxes, own = [[] for _ in range(planes)], 0.0
                    cost = shared + own if active or not merged else 0.0
                    methods[name] = {
                        "boxes": boxes,
                        "method_seconds": own,
                        "inference_compute_seconds": cost,
                    }
                    totals[name] += cost
                derived = {
                    **{k: base[k] for k in ("window", "window_id", "grid", "score", "gate_passed")},
                    "identity": identity,
                    "methods": methods,
                    **({"localized": active} if merged else {}),
                }
                path = target / item["path"]
                write_json(path, derived)
                chunks.append({"path": item["path"], "sha256": file_sha256(path)})
            timing = {}
            for name, (stage, source, _, _, drops_a1) in rules.items():
                fit = completes[stage]["timing"][source]["fit_seconds"]
                timing[name] = {
                    "fit_seconds": fit - calibration if drops_a1 else fit,
                    "inference_compute_seconds": totals[name],
                    "inference_chunks": localized,
                    "fit_shared_cost_do_not_sum_over_videos": True,
                }
            write_json(
                target / "complete.json",
                {
                    "status": "complete",
                    "identity": identity,
                    "video": reference["video"],
                    "chunks": chunks,
                    "timing": timing,
                    "prediction_scope": reference.get(
                        "prediction_scope", "entire_video_all_protocol_windows"
                    ),
                },
            )
        seal_stage(output, dataset, methods_, manifest)
        print(f"sealed {dataset}: {len(directories[sources[0]])} videos", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="sealed stages root")
    parser.add_argument("--learn", type=Path, help="fits/learn.json (rules dropping A1)")
    parser.add_argument("--ucf-manifest", type=Path, required=True)
    parser.add_argument("--mivia-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--methods", nargs="+", choices=sorted(ALL_RULES), help="subset of rules (default RULES)"
    )
    args = parser.parse_args()
    derive(
        args.predictions,
        args.learn,
        {"ucf_test": args.ucf_manifest, "mivia_test": args.mivia_manifest},
        args.output,
        None if args.methods is None else {m: ALL_RULES[m] for m in args.methods},
    )


if __name__ == "__main__":
    main()
