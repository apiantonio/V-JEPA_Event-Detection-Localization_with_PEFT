"""Training references, isolated from the visualized videos and their duplicates.

Reuses datasource, sealed split and protocol sampler. The sample belongs
to the analysis and does not change the detector training.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from evd.data.windows import CausalWindowSampler
from evd.exp.explanation_store import atomic_json, identity_sha
from evd.exp.sources import DatasetBundle
from evd.protocol.chunks import chunks_for_video
from evd.protocol.split import SplitSpec


def select_references(
    bundle: DatasetBundle,
    split: SplitSpec,
    sampler: CausalWindowSampler,
    excluded: set[str],
    *,
    local_paths: dict[str, Path],
    seed: int = 42,
    per_class: int = 2,
) -> dict:
    """Select anomalous windows per class and as many fully normal ones.

    :param excluded: all videos to visualize; identical hashes are excluded too.
    :param local_paths: videos available on the executing machine.
    :return: deterministic manifest with per-class shortfalls and provenance.
    :raises ValueError: on an incompatible training set or no local binary support.
    """
    if split.annotation_source != bundle.annotation_source:
        raise ValueError("Split and annotations of the references do not match")
    if per_class < 1:
        raise ValueError("The per-class quota must be positive")
    hashes = {bundle.content_sha[v] for v in excluded if bundle.content_sha.get(v)}
    excluded = excluded | {v for v, sha in bundle.content_sha.items() if sha and sha in hashes}
    candidates: dict[int, list] = {c.idx: [] for c in bundle.classes if c.idx != bundle.normal_idx}
    normals = []
    unavailable = []
    rng = np.random.default_rng(seed)
    for video_id in sorted(set(split.train) - excluded):
        if video_id not in local_paths:
            unavailable.append(video_id)
            continue
        ann = bundle.annotations[video_id]
        chunks = chunks_for_video(bundle.inventory[video_id], ann, sampler)
        for chunk in chunks:
            row = {
                "video_id": video_id,
                "video": str(local_paths[video_id].resolve()),
                "video_sha256": bundle.content_sha[video_id],
                "window_id": chunk.window.window_id,
                "window": asdict(chunk.window),
                "label": chunk.label,
                "class_idx": chunk.class_idx,
                "annotated_fraction": chunk.overlap_frames
                / (chunk.window.end_frame - chunk.window.start_frame + 1),
                "split": "train",
            }
            if ann.is_normal:
                normals.append((rng.random(), row))
            elif chunk.label:
                candidates[chunk.class_idx].append((rng.random(), row))
    selected, used, supports = [], set(), {}
    for cls, rows in sorted(candidates.items()):
        chosen = []
        for _, row in sorted(
            rows, key=lambda pair: (-int(pair[1]["annotated_fraction"] == 1), pair[0])
        ):
            if row["video_id"] not in used:
                chosen.append(row)
                used.add(row["video_id"])
            if len(chosen) == per_class:
                break
        selected.extend(chosen)
        name = next(c.name for c in bundle.classes if c.idx == cls)
        supports[name] = {
            "requested": per_class,
            "selected": len(chosen),
            "fully_annotated": sum(r["annotated_fraction"] == 1 for r in chosen),
            "insufficient": len(chosen) < per_class,
        }
    n_positive = len(selected)
    chosen_normal = []
    for _, row in sorted(normals, key=lambda pair: pair[0]):
        if row["video_id"] not in used:
            chosen_normal.append(row)
            used.add(row["video_id"])
        if len(chosen_normal) == n_positive:
            break
    if not n_positive or len(chosen_normal) != n_positive:
        raise ValueError(
            f"Insufficient binary support: {n_positive} anomalous, {len(chosen_normal)} normal"
        )
    selected.extend(chosen_normal)
    result = {
        "schema": 1,
        "seed": seed,
        "per_class": per_class,
        "windows": selected,
        "n_anomaly": n_positive,
        "n_normal": len(chosen_normal),
        "support": supports,
        "excluded_visualization_and_duplicates": sorted(excluded),
        "unavailable_training_videos": unavailable,
        "split_sha256": split.sha256,
        "annotation_source": bundle.annotation_source,
        "labels": "temporali a livello finestra; nessuna annotazione spaziale per token",
        "selection": "training locale, video distinti; prima finestre interamente annotate, poi ordine seminato",
    }
    result["sha256"] = identity_sha(result)
    return result


def save_reference_manifest(path: Path, manifest: dict) -> None:
    """Save, or verify, the fixed manifest shared by the two models.

    :raises ValueError: if an existing manifest describes different references.
    """
    import json

    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if identity_sha(existing) != identity_sha(manifest):
            raise ValueError(f"Different reference manifest: {path}")
    else:
        atomic_json(path, manifest)
