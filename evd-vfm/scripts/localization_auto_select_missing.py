"""Keep the target chunks a detector's gated stages did not localize (no GT, no GPU).

Input: ``targets.json`` (chunk indices and window ids only) and the sealed S0 stage of one
detector. Output: the selection the stages run with ``--selection`` into
``predictions_extra/`` (decision of 17 October). Window ids are checked against S0, so a
target list made on another grid fails instead of selecting the wrong chunks.

With ``--all`` every target is kept whatever the gate: a detector added later (the frozen
attentive probe, 20 October) runs its stages once on every annotated chunk, into
``predictions/``, instead of a gated pass plus an extra one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from evd.exp.localization_auto import read_json, video_key, write_json  # noqa: E402


def missing(targets: dict, s0: Path, keep_all: bool = False) -> dict:
    """:return: per dataset and video, the target indices whose S0 record is below the gate
    (every target index with ``keep_all``)."""
    out = {}
    for dataset, videos in targets.items():
        out[dataset] = {}
        for video, entry in videos.items():
            target = s0 / dataset / video_key(video)
            complete = read_json(target / "complete.json")
            keep = []
            for index, window_id in zip(entry["indices"], entry["window_ids"], strict=True):
                record = read_json(target / complete["chunks"][index]["path"])
                if record["window_id"] != window_id:
                    raise ValueError(f"Target window differs from S0: {video} chunk {index}")
                if keep_all or not record["gate_passed"]:
                    keep.append(index)
            out[dataset][video] = {"indices": keep}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--s0", type=Path, required=True, help="sealed readout stage root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all", action="store_true", help="keep every target, gate ignored")
    args = parser.parse_args()
    result = missing(read_json(args.targets), args.s0, keep_all=args.all)
    write_json(args.output, result)
    for dataset, videos in result.items():
        print(dataset, sum(len(v["indices"]) for v in videos.values()), "chunks to compute")


if __name__ == "__main__":
    main()
