"""Merge gated and missing-chunk predictions into the 17 October layout (no GPU, no GT).

Decision of 17 October: boxes on exactly the chunks that hold an annotated
spatial plane (``targets.json``, from ``localization_auto_targets.py``), whatever the
detector score. The chunks above the old gate were already computed by the sealed stages
(same code path, same weights); the others were computed by the same stages run with
``--selection`` into a second root. For every stage this writes one sealed stage where:

- a target chunk carries the record of whichever root localized it (gated root first),
  with its map hard-linked; ``localized`` is true;
- every other chunk has no box and costs nothing; ``localized`` is false;
- the timing of a method sums the localized chunks only (``inference_chunks`` counts them),
  and the fit is the source fit.

The output is the evaluator's layout, so derived stages (P16, re-threshold, post-process)
and the report run on it unchanged.
"""

from __future__ import annotations

import argparse
import os
import sys
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

DATASETS = ("ucf_test", "mivia_test")


def _sealed(root: Path, stage: str, dataset: str) -> dict[str, dict]:
    seal = read_json(root / stage / "operations" / f"seal-{dataset}.json")
    return {v["video_id"]: v for v in seal["videos"]}


def _complete(root: Path, stage: str, entry: dict) -> dict:
    path = root / stage / entry["directory"] / "complete.json"
    if file_sha256(path) != entry["complete_sha256"]:
        raise ValueError(f"{stage} prediction changed after sealing")
    return read_json(path)


def merge_stage(
    gated: Path,
    extra: Path,
    stage: str,
    methods: list[str],
    targets: dict,
    manifests: dict[str, Path],
    output: Path,
) -> None:
    """Write and seal one merged stage for both datasets.

    :raises ValueError: on a changed source, disagreeing windows, or a target chunk that
        neither root localized.
    """
    for dataset in DATASETS:
        first, second = _sealed(gated, stage, dataset), _sealed(extra, stage, dataset)
        identity = {
            "stage": f"{stage}_annotated_chunks",
            "rule": "boxes on every chunk holding an annotated spatial plane; no gate",
            "gated_seal_sha256": file_sha256(gated / stage / "operations" / f"seal-{dataset}.json"),
            "extra_seal_sha256": file_sha256(extra / stage / "operations" / f"seal-{dataset}.json"),
        }
        for row in read_json(manifests[dataset]):
            video = row["video_id"]
            wanted = set(targets[dataset].get(video, {}).get("indices", []))
            directory = f"{dataset}/{video_key(video)}"
            a, b = _complete(gated, stage, first[video]), _complete(extra, stage, second[video])
            target = output / stage / directory
            chunks = []
            totals = dict.fromkeys(methods, 0.0)
            count = 0
            for index, (item_a, item_b) in enumerate(zip(a["chunks"], b["chunks"], strict=True)):
                rec_a = read_json(gated / stage / directory / item_a["path"])
                rec_b = read_json(extra / stage / directory / item_b["path"])
                if rec_a["window_id"] != rec_b["window_id"]:
                    raise ValueError(f"Roots disagree on the windows: {video}")
                if index in wanted:
                    if rec_a["gate_passed"]:
                        source, root, item = rec_a, gated, item_a
                    elif rec_b.get("localized"):
                        source, root, item = rec_b, extra, item_b
                    else:
                        raise ValueError(f"Target chunk {index} of {video} was never localized")
                    record = {**source, "localized": True, "localized_by": str(root.name)}
                    record["methods"] = {m: source["methods"][m] for m in methods}
                    stem = Path(item["path"]).stem
                    origin = root / stage / directory / "maps" / f"{stem}.npz"
                    if origin.exists():
                        link = target / "maps" / f"{stem}.npz"
                        link.parent.mkdir(parents=True, exist_ok=True)
                        if link.exists():
                            link.unlink()
                        os.link(origin, link)
                    for m in methods:
                        totals[m] += source["methods"][m]["inference_compute_seconds"]
                    count += 1
                else:
                    planes = rec_a["grid"][0]
                    record = {
                        **{
                            k: rec_a[k]
                            for k in ("window", "window_id", "grid", "score", "gate_passed")
                        },
                        "timing": rec_a.get("timing", {}),
                        "localized": False,
                        "methods": {
                            m: {
                                "boxes": [[] for _ in range(planes)],
                                "method_seconds": 0.0,
                                "inference_compute_seconds": 0.0,
                            }
                            for m in methods
                        },
                    }
                record["merge_identity"] = identity
                path = target / item_a["path"]
                write_json(path, record)
                chunks.append({"path": item_a["path"], "sha256": file_sha256(path)})
            write_json(
                target / "complete.json",
                {
                    "status": "complete",
                    "identity": identity,
                    "video": a["video"],
                    "chunks": chunks,
                    "timing": {
                        m: {
                            "fit_seconds": a["timing"][m]["fit_seconds"],
                            "inference_compute_seconds": totals[m],
                            "inference_chunks": count,
                            "fit_shared_cost_do_not_sum_over_videos": True,
                        }
                        for m in methods
                    },
                    "prediction_scope": "chunks_with_annotated_spatial_planes",
                },
            )
        seal_stage(output / stage, dataset, methods, manifests[dataset])
        print(f"merged {stage} {dataset}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--gated", type=Path, required=True, help="sealed gated stages root")
    parser.add_argument("--extra", type=Path, required=True, help="missing-chunk stages root")
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", required=True)
    parser.add_argument("--ucf-manifest", type=Path, required=True)
    parser.add_argument("--mivia-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan, targets = read_json(args.plan), read_json(args.targets)
    manifests = {"ucf_test": args.ucf_manifest, "mivia_test": args.mivia_manifest}
    for stage in args.stages:
        merge_stage(
            args.gated, args.extra, stage, plan["stages"][stage], targets, manifests, args.output
        )


if __name__ == "__main__":
    main()
