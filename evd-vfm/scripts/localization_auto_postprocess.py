"""Post-process the saved boxes of every reduced-plan method (no GPU, no maps, no test GT).

Follow-up D of 17 October: two rules on the 8-connected component
boxes each method already produced, applied plane by plane to all 34 methods of the
reduced plan sealed on the cluster (P16, derived locally and ~0, is left out; with mass70
its map would give P14):

- ``largest``: keep only the largest box of the plane (area of the inclusive native
  rectangle; ties keep the first box in the method's order);
- ``minarea``: drop boxes smaller than 2 x 2 tokens of the 24 x 24 grid, i.e.
  ``4 * (W / 24) * (H / 24)`` native pixels.

The inference cost is the source method's plus the post-processing; the fit is the
source fit. Output layout and seal are the evaluator's.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from evd.exp.localization_auto import (  # noqa: E402
    file_sha256,
    read_json,
    seal_stage,
    video_key,
    write_json,
)

PLAN = ROOT / "configs/localization_auto_20261016_reduced.json"
GRID = 24
MIN_TOKENS = 4
EXCLUDED = {"P16-normal_q99"}


def area(box: list[int]) -> int:
    """:return: pixels of an inclusive native rectangle ``x0, y0, x1, y1``."""
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


def largest(plane: list[list[int]], width: int, height: int) -> list[list[int]]:
    """:return: the single largest box of the plane (empty plane stays empty)."""
    if not plane:
        return []
    best = max(range(len(plane)), key=lambda i: (area(plane[i]), -i))
    return [plane[best]]


def minarea(plane: list[list[int]], width: int, height: int) -> list[list[int]]:
    """:return: the boxes of at least ``MIN_TOKENS`` token cells of native area."""
    floor = MIN_TOKENS * (width / GRID) * (height / GRID)
    return [box for box in plane if area(box) >= floor]


RULES = {"largest": largest, "minarea": minarea}


def stage_methods(plan: dict, only: list[str] | None = None) -> dict[str, list[str]]:
    """:return: the plan methods of each stage sealed on the cluster (``only``: those stages)."""
    return {
        stage: [m for m in methods if m not in EXCLUDED]
        for stage, methods in plan["stages"].items()
        if [m for m in methods if m not in EXCLUDED] and (only is None or stage in only)
    }


def derive(
    predictions: Path,
    manifests: dict[str, Path],
    output: Path,
    plan: dict,
    only: list[str] | None = None,
) -> None:
    """Write and seal the post-processed stage for both test datasets.

    :param only: post-process only these stages of the plan (default: all of them).
    :raises ValueError: if a source changed after sealing or the stages disagree on windows.
    """
    stages = stage_methods(plan, only)
    if only is not None and set(only) - set(stages):
        raise ValueError(f"Stages absent from the plan: {sorted(set(only) - set(stages))}")
    methods = [f"{m}-{r}" for ms in stages.values() for m in ms for r in RULES]
    for dataset, manifest in manifests.items():
        seals = {
            s: read_json(predictions / s / "operations" / f"seal-{dataset}.json") for s in stages
        }
        hashes = {
            s: {v["video_id"]: v["complete_sha256"] for v in seals[s]["videos"]} for s in stages
        }
        identity = {
            "stage": "postprocess",
            "rules": {
                "largest": "largest box per plane",
                "minarea": f"{MIN_TOKENS} of {GRID}x{GRID} tokens",
            },
            "source_seal_sha256": {
                s: file_sha256(predictions / s / "operations" / f"seal-{dataset}.json")
                for s in stages
            },
        }
        for row in read_json(manifest):
            directory = f"{dataset}/{video_key(row['video_id'])}"
            completes = {}
            for s in stages:
                path = predictions / s / directory / "complete.json"
                if file_sha256(path) != hashes[s][row["video_id"]]:
                    raise ValueError(f"{s} prediction changed after sealing")
                completes[s] = read_json(path)
            reference = completes[next(iter(stages))]
            width, height = row["width"], row["height"]
            target = output / directory
            chunks, totals, localized = [], dict.fromkeys(methods, 0.0), 0
            for position, item in enumerate(reference["chunks"]):
                records = {
                    s: read_json(
                        predictions / s / directory / completes[s]["chunks"][position]["path"]
                    )
                    for s in stages
                }
                base = records[next(iter(stages))]
                if any(r["window_id"] != base["window_id"] for r in records.values()):
                    raise ValueError("Stages disagree on the windows")
                derived = {}
                # Merged layout (17 October): only localized chunks are charged and counted.
                active = base.get("localized", True)
                localized += active
                for s, ms in stages.items():
                    for m in ms:
                        source = records[s]["methods"][m]
                        for rule, function in RULES.items():
                            start = time.perf_counter()
                            boxes = [function(p, width, height) for p in source["boxes"]]
                            seconds = (
                                source["inference_compute_seconds"] + time.perf_counter() - start
                                if active
                                else 0.0
                            )
                            derived[f"{m}-{rule}"] = {
                                "boxes": boxes,
                                "inference_compute_seconds": seconds,
                            }
                            totals[f"{m}-{rule}"] += seconds
                path = target / item["path"]
                write_json(
                    path,
                    {
                        **{
                            k: base[k]
                            for k in ("window", "window_id", "grid", "score", "gate_passed")
                        },
                        "identity": identity,
                        "methods": derived,
                        **({"localized": base["localized"]} if "localized" in base else {}),
                    },
                )
                chunks.append({"path": item["path"], "sha256": file_sha256(path)})
            timing = {
                f"{m}-{r}": {
                    "fit_seconds": completes[s]["timing"][m]["fit_seconds"],
                    "inference_compute_seconds": totals[f"{m}-{r}"],
                    "inference_chunks": localized,
                    "fit_shared_cost_do_not_sum_over_videos": True,
                }
                for s, ms in stages.items()
                for m in ms
                for r in RULES
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
        seal_stage(output, dataset, methods, manifest)
        print(f"sealed {dataset}: {len(methods)} methods", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="sealed stages root")
    parser.add_argument("--ucf-manifest", type=Path, required=True)
    parser.add_argument("--mivia-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=PLAN, help="plan naming the stage methods")
    parser.add_argument("--stages", nargs="+", help="stages to post-process (default: all)")
    args = parser.parse_args()
    derive(
        args.predictions,
        {"ucf_test": args.ucf_manifest, "mivia_test": args.mivia_manifest},
        args.output,
        read_json(args.plan),
        args.stages,
    )


if __name__ == "__main__":
    main()
