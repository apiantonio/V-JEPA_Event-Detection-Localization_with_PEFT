"""Shared runner of the automatic-localization campaign stages.

Contract: ``docs/varie/localizzazione_automatica_20261016.md``. The readout stage (S0,
``scripts/localization_auto_predict.py``) decoded and encoded every test chunk once and
froze the gate decision. Every later stage reads that sealed stage instead of re-deciding:
all methods therefore share the same gate, and the shared decode/encode cost of each chunk
is taken from the S0 record, so that a method's standalone inference time is
``S0 decode + S0 encode + its own work``. A stage recomputes features only for chunks above
the gate, and writes the layout read by ``scripts/localization_auto_evaluate.py``.

No stage receives annotations, event intervals or classes: inputs are video metadata, the
sealed S0 records and validation/train-derived artifacts.

The S0 script keeps its own loop: its outputs were produced before this module existed and
its identity covers its source; it is not rewritten.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from evd.types import VideoMeta
from evd.utils.hashing import file_sha256

#: Display names of the method families of the campaign plan, in report order.
FAMILY_LABELS = {
    "gradients_attention": "Gradients and attention",
    "integrated_gradients": "Integrated Gradients",
    "concepts": "Concepts",
    "perturbations_surrogates": "Perturbations",
    "memories_prototypes": "Memories and prototypes",
    "learned_localizers": "Learned localizers",
}


def read_json(path: Path) -> Any:
    """:return: the parsed JSON document."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    """Publish a JSON document atomically; NaN is rejected."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def video_key(video_id: str) -> str:
    """:return: the directory name of a video, shared with the S0 stage."""
    return hashlib.sha256(video_id.encode()).hexdigest()[:24]


def in_shard(video_id: str, shard: int, shards: int) -> bool:
    """:return: whether the video belongs to the shard (hash split, as in S0)."""
    return int(video_key(video_id), 16) % shards == shard


def validate_inputs(rows: list[dict]) -> None:
    """Reject manifests that carry anything beyond ``VideoMeta`` fields.

    :raises ValueError: on duplicates, extra fields (e.g. annotations) or invalid geometry.
    """
    allowed = {item.name for item in fields(VideoMeta)}
    if not rows or len({row["video_id"] for row in rows}) != len(rows):
        raise ValueError("Empty manifest or duplicate videos")
    for row in rows:
        if set(row) != allowed:
            raise ValueError(f"Prediction input must contain only VideoMeta fields: {set(row)}")
        if min(row["width"], row["height"], row["fps"], row["num_frames"]) <= 0:
            raise ValueError("Invalid video geometry")


def weighted_quantile(values: dict[str, list[float]], quantile: float) -> float:
    """Empirical quantile with equal weight per video, then equal weight per chunk.

    Same rule as the gate (``localization_auto_preflight.calibrate``), without its
    probability-range check, for thresholds on map values.

    :param values: finite values per video, at least one per video.
    :param quantile: in ``(0, 1)``.
    :return: the smallest value whose cumulative weight reaches ``quantile``.
    :raises ValueError: on empty input, empty videos, non-finite values or a bad quantile.
    """
    if not values or not 0 < quantile < 1 or any(not v for v in values.values()):
        raise ValueError("Weighted quantile needs nonempty videos and a quantile in (0, 1)")
    rows = [
        (float(value), video, index)
        for video, items in values.items()
        for index, value in enumerate(items)
    ]
    if not np.isfinite([row[0] for row in rows]).all():
        raise ValueError("Weighted quantile of non-finite values")
    cumulative = 0.0
    for value, video, _ in sorted(rows):
        cumulative += 1 / (len(values) * len(values[video]))
        if cumulative >= quantile:
            return value
    raise ValueError("Quantile not reached")


def seal_stage(stage_root: Path, dataset: str, methods: list[str], manifest: Path) -> dict:
    """Write the dataset seal once every video of the manifest is complete and intact.

    :param stage_root: root of the stage predictions.
    :param dataset: ``ucf_test`` or ``mivia_test``.
    :param methods: experiments every chunk record and timing must carry.
    :param manifest: the prediction input manifest (VideoMeta rows).
    :return: the seal document.
    :raises ValueError: on a missing video, method, timing or a checksum mismatch.
    """
    entries = []
    for row in read_json(manifest):
        directory = f"{dataset}/{video_key(row['video_id'])}"
        target = stage_root / directory
        complete = read_json(target / "complete.json")
        if (
            complete["status"] != "complete"
            or complete["video"] != row
            or set(complete["timing"]) != set(methods)
        ):
            raise ValueError(f"Incomplete prediction: {row['video_id']}")
        for chunk in complete["chunks"]:
            path = target / chunk["path"]
            if file_sha256(path) != chunk["sha256"]:
                raise ValueError("Prediction checksum mismatch")
            if set(read_json(path)["methods"]) != set(methods):
                raise ValueError("Missing method in a completed chunk")
        entries.append(
            {
                "video_id": row["video_id"],
                "directory": directory,
                "complete_sha256": file_sha256(target / "complete.json"),
            }
        )
    seal = {
        "status": "complete",
        "dataset": "ucf_crime" if dataset == "ucf_test" else "mivia_crime",
        "methods": methods,
        "videos": entries,
        "input_manifest_sha256": file_sha256(manifest),
        "prediction_scope": "all_test_videos_all_protocol_windows",
        "test_quality_evaluated": False,
    }
    write_json(stage_root / "operations" / f"seal-{dataset}.json", seal)
    return seal


@dataclass
class S0Video:
    """Sealed readout-stage records of one video, in window order."""

    records: list[dict]
    complete_sha256: str
    selection: tuple[int, ...] | None = None

    @property
    def gated(self) -> list[int]:
        """:return: indices of the chunks to localize.

        Without a selection, the chunks above the frozen gate (the 16 October campaign);
        with one (decision of 17 October), exactly the selected chunks, whatever the score.
        """
        if self.selection is not None:
            return list(self.selection)
        return [i for i, record in enumerate(self.records) if record["gate_passed"]]


def load_s0_video(
    s0_root: Path, seal: dict, dataset: str, row: dict, selection: list[int] | None = None
) -> S0Video:
    """Read one video of the sealed S0 stage, verifying every checksum.

    :param selection: chunk indices to localize instead of the gate (``None``: the gate).
    :raises ValueError: if the video is absent from the seal, a file changed after sealing,
        or the selection names a chunk the video does not have.
    """
    entry = next((e for e in seal["videos"] if e["video_id"] == row["video_id"]), None)
    if entry is None:
        raise ValueError(f"Video absent from the S0 seal: {row['video_id']}")
    target = s0_root / entry["directory"]
    if file_sha256(target / "complete.json") != entry["complete_sha256"]:
        raise ValueError("S0 prediction changed after sealing")
    complete = read_json(target / "complete.json")
    if complete["video"] != row:
        raise ValueError("S0 video metadata differs from the stage manifest")
    records = []
    for chunk in complete["chunks"]:
        path = target / chunk["path"]
        if file_sha256(path) != chunk["sha256"]:
            raise ValueError("S0 chunk changed after sealing")
        records.append(read_json(path))
    if selection is not None and any(not 0 <= i < len(records) for i in selection):
        raise ValueError(f"Selection outside the video's chunks: {row['video_id']}")
    chosen = None if selection is None else tuple(sorted(set(selection)))
    return S0Video(records, entry["complete_sha256"], chosen)


@dataclass
class MethodOutput:
    """Boxes of one experiment on one chunk, with the method's own compute time."""

    boxes: list[list[list[int]]]
    seconds: float


@dataclass
class ChunkResult:
    """What a stage returns for one chunk above the gate."""

    methods: dict[str, MethodOutput]
    maps: dict[str, np.ndarray] = field(default_factory=dict)
    receipts: dict = field(default_factory=dict)
    logit: float | None = None
    stage_decode_seconds: float = 0.0
    stage_encode_seconds: float = 0.0


class Stage(Protocol):
    """One group of experiments computed in a single pass over the gated chunks."""

    name: str
    methods: tuple[str, ...]

    def parameters(self) -> dict:
        """:return: frozen parameters, part of the stage identity."""

    def fit_seconds(self) -> dict[str, float]:
        """:return: per-experiment fit time (shared fits counted in full for each method)."""

    def begin_video(self, meta: VideoMeta, items: list, s0: S0Video) -> None:
        """Prepare per-video state (e.g. a reference computed once per video)."""

    def chunk(self, meta: VideoMeta, item, index: int) -> ChunkResult:
        """Compute the experiments on one gated chunk."""


def run_stage(
    stage: Stage,
    *,
    rows: list[dict],
    windows: Callable[[VideoMeta], list],
    s0_root: Path,
    dataset: str,
    output: Path,
    shard: int,
    shards: int,
    source_commit: str,
    runtime: dict,
    extra_identity: dict,
    selection: dict[str, list[int]] | None = None,
) -> None:
    """Run a stage on one shard, resuming per chunk and refusing incompatible outputs.

    :param windows: protocol windows of a video (the canonical sampler), in S0 order.
    :param extra_identity: hashes of external inputs (plan, gate, fitted artifacts).
    :param selection: per video id, the chunk indices to localize; ``None`` keeps the gate.
        A video absent from the selection has no chunk to localize.
    :raises ValueError: if the windows differ from the S0 records element by element, or
        if existing outputs carry a different identity.
    """
    validate_inputs(rows)
    s0_seal_path = s0_root / "operations" / f"seal-{dataset}.json"
    s0_seal = read_json(s0_seal_path)
    if s0_seal.get("status") != "complete":
        raise ValueError("The readout stage must be sealed before later stages run")
    identity = {
        "schema": 1,
        "stage": stage.name,
        "dataset": dataset,
        "source_commit": source_commit,
        "methods": list(stage.methods),
        "parameters": stage.parameters(),
        "s0_seal_sha256": file_sha256(s0_seal_path),
        "projection": "fine",
        "test_labels_available": False,
        **extra_identity,
    }
    fit = stage.fit_seconds()
    selected = sorted(
        (row for row in rows if in_shard(row["video_id"], shard, shards)),
        key=lambda row: row["video_id"],
    )
    for number, row in enumerate(selected):
        meta = VideoMeta(**row)
        target = output / dataset / video_key(meta.video_id)
        if (target / "complete.json").exists():
            if read_json(target / "complete.json")["identity"] != identity:
                raise ValueError("Existing video has incompatible provenance")
            continue
        s0 = load_s0_video(
            s0_root,
            s0_seal,
            dataset,
            row,
            None if selection is None else selection.get(row["video_id"], []),
        )
        items = windows(meta)
        ids = [item.window.window_id for item in items]
        if ids != [record["window_id"] for record in s0.records]:
            raise ValueError(f"Windows differ from the S0 records: {meta.video_id}")
        localize = set(s0.gated)
        if localize:
            stage.begin_video(meta, items, s0)
        chunks = []
        for index, (item, record) in enumerate(zip(items, s0.records, strict=True)):
            destination = target / "chunks" / f"{index:06d}.json"
            if destination.exists():
                if read_json(destination)["identity"] != identity:
                    raise ValueError("Existing chunk has incompatible provenance")
            else:
                shared = (
                    record["timing"]["decode_preprocess_seconds"]
                    + record["timing"]["encode_score_seconds"]
                )
                planes = record["grid"][0]
                if index in localize:
                    result = stage.chunk(meta, item, index)
                else:
                    result = ChunkResult(
                        {
                            name: MethodOutput([[] for _ in range(planes)], 0.0)
                            for name in stage.methods
                        }
                    )
                if set(result.methods) != set(stage.methods):
                    raise RuntimeError("A stage returned an incomplete method set")
                maps_sha = None
                if result.maps:
                    map_path = target / "maps" / f"{index:06d}.npz"
                    map_path.parent.mkdir(parents=True, exist_ok=True)
                    with map_path.with_suffix(".npz.tmp").open("wb") as stream:
                        np.savez_compressed(stream, **result.maps)
                    map_path.with_suffix(".npz.tmp").replace(map_path)
                    maps_sha = file_sha256(map_path)
                write_json(
                    destination,
                    {
                        "identity": identity,
                        "window": record["window"],
                        "window_id": record["window_id"],
                        "grid": record["grid"],
                        "score": record["score"],
                        "gate_passed": record["gate_passed"],
                        "localized": index in localize,
                        "s0_logit": record["logit"],
                        "logit": result.logit,
                        "methods": {
                            name: {
                                "boxes": output_.boxes,
                                "method_seconds": output_.seconds,
                                "inference_compute_seconds": shared + output_.seconds,
                            }
                            for name, output_ in result.methods.items()
                        },
                        "timing": {
                            "s0_decode_preprocess_seconds": record["timing"][
                                "decode_preprocess_seconds"
                            ],
                            "s0_encode_score_seconds": record["timing"]["encode_score_seconds"],
                            "stage_decode_seconds": result.stage_decode_seconds,
                            "stage_encode_seconds": result.stage_encode_seconds,
                            "stage_reencoding_not_charged_to_methods": True,
                        },
                        "receipts": result.receipts,
                        "runtime": runtime,
                        "maps_sha256": maps_sha,
                    },
                )
            chunks.append(
                {"path": str(destination.relative_to(target)), "sha256": file_sha256(destination)}
            )
        records = [read_json(target / chunk["path"]) for chunk in chunks]
        timing = {
            name: {
                "fit_seconds": fit[name],
                "inference_compute_seconds": sum(
                    r["methods"][name]["inference_compute_seconds"] for r in records
                ),
                "inference_chunks": len(records),
                "fit_shared_cost_do_not_sum_over_videos": True,
            }
            for name in stage.methods
        }
        write_json(
            target / "complete.json",
            {
                "status": "complete",
                "identity": identity,
                "video": row,
                "chunks": chunks,
                "timing": timing,
                "prediction_scope": "entire_video_all_protocol_windows",
            },
        )
        print(
            json.dumps(
                {
                    "video": meta.video_id,
                    "chunks": len(chunks),
                    "progress": [number + 1, len(selected)],
                }
            ),
            flush=True,
        )
    write_json(
        output / "operations" / f"{dataset}-{stage.name}-shard{shard:02d}.json",
        {
            "status": "complete",
            "identity": identity,
            "shard": shard,
            "shards": shards,
            "videos": [row["video_id"] for row in selected],
            "runtime": runtime,
            "completed_at": time.time(),
        },
    )


def sync_time(cuda: bool) -> float:
    """:return: a wall-clock reading after the device finished queued work."""
    if cuda:
        import torch

        torch.cuda.synchronize()
    return time.perf_counter()
