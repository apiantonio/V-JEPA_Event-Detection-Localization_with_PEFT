"""List the chunks that hold annotated spatial planes (local only: reads the GT snapshot).

Decision of 17 October: boxes are produced for every chunk of the test sets that
holds a reviewed spatial annotation, whatever the detector score; the q99 gate no longer
selects the chunks. The chunk of each annotated frame is the one the evaluator assigns
(``offline_assignment`` with the ``fine`` policy, as in ``localization_auto_evaluate``), so
the list covers exactly the frames the metric reads.

The output holds chunk indices and window ids only, never a box or a class: it is the one
GT-derived file that goes to the cluster, so the stages know where to localize.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from evd.exp.localization_auto import read_json, video_key, write_json  # noqa: E402
from evd.explain.project import offline_assignment  # noqa: E402
from evd.types import WindowSpec  # noqa: E402

DATASETS = {"ucf_test": "ucf_crime", "mivia_test": "mivia_crime"}


def targets(snapshot: Path, s0: Path) -> dict:
    """:return: per dataset folder and video id, the sorted chunk indices and window ids."""
    out = {}
    for folder, dataset in DATASETS.items():
        out[folder] = {}
        for truth_path in sorted((snapshot / dataset).glob("*.json")):
            truth = read_json(truth_path)
            target = s0 / folder / video_key(truth["video_id"])
            complete = read_json(target / "complete.json")
            records = [read_json(target / c["path"]) for c in complete["chunks"]]
            windows = [
                WindowSpec(
                    **{
                        **r["window"],
                        "explicit_frame_indices": tuple(r["window"]["explicit_frame_indices"]),
                    }
                )
                for r in records
            ]
            assignment = offline_assignment(windows, truth["num_frames"], politica="fine")
            chosen = set()
            for event in truth["events"]:
                for plane in event["planes"]:
                    for segment in plane["segments"]:
                        for frame in range(segment["start"], segment["end"] + 1):
                            index = int(assignment[frame])
                            if index < 0:
                                raise ValueError(f"Annotated frame without a chunk: {frame}")
                            chosen.add(index)
            indices = sorted(chosen)
            out[folder][truth["video_id"]] = {
                "indices": indices,
                "window_ids": [records[i]["window_id"] for i in indices],
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--s0", type=Path, required=True, help="pulled sealed readout stage")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = targets(args.snapshot, args.s0)
    write_json(args.output, result)
    for folder, videos in result.items():
        gated = 0
        for video_id, entry in videos.items():
            target = args.s0 / folder / video_key(video_id)
            complete = read_json(target / "complete.json")
            gated += sum(
                read_json(target / complete["chunks"][i]["path"])["gate_passed"]
                for i in entry["indices"]
            )
        total = sum(len(v["indices"]) for v in videos.values())
        print(f"{folder}: {len(videos)} videos, {total} target chunks, {gated} already gated")


if __name__ == "__main__":
    main()
