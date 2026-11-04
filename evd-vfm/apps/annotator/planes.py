"""VTCD review units: contiguous stretches assigned to the same window/plane.

Reuses ``offline_assignment``/``offline_plane`` (ADR-077). The historical mapping is
accepted only if it reproduces every CSV row, including the absence of boxes. Without a
source, planes are derived from the protocol with the same rule
(:func:`protocol_units`), so every event has its planes from scratch.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from apps.annotator.geometry import PROTOCOL_PATH, load_protocol
from apps.annotator.project import event_classes, sha, window_index

if TYPE_CHECKING:  # typing only
    from evd.types import WindowSpec

#: Policy used by the base planes to choose the window of a frame: the same as the VTCD
#: sources exported since 29 September 2026 (ADR-077).
PROTOCOL_ASSIGNMENT = "overlap_positivo"


def protocol_windows(video: dict) -> list[WindowSpec]:
    """Return the ``edp_v2`` windows of the video, on the grid of its partition.

    :return: sampler windows in temporal order.
    :raises KeyError: if the protocol lacks the required temporal fields.
    """
    from evd.protocol.chunks import sampler_for_protocol
    from evd.types import VideoMeta

    sampler = sampler_for_protocol(None, load_protocol(), validazione=video.get("split") == "val")
    meta = VideoMeta(
        **{k: video[k] for k in ("video_id", "path", "num_frames", "fps", "width", "height")}
    )
    return list(sampler.windows_for_video(meta))


def protocol_planes() -> int:
    """:return: temporal planes of an ``edp_v2`` window, one per tubelet (16).

    :raises ValueError: if the window samples are not divisible by the tubelet.
    """
    from evd.eval.efficiency import PATCH_T

    frames = int(load_protocol()["temporal"]["num_frames"])
    if frames % PATCH_T:
        raise ValueError("The window samples do not split into whole tubelets.")
    return frames // PATCH_T


def _event_frames(video: dict) -> list[int]:
    """:return: frames inside at least one event, in order; the only domain of the units."""
    last = video["num_frames"] - 1
    frames = set()
    for e in video["events"]:
        frames.update(range(max(0, e["start"]), min(last, e["end"]) + 1))
    return sorted(frames)


def _assign(
    video: dict,
    windows: Sequence[WindowSpec],
    policy: str,
    planes: int,
    classes_of: Callable[[int], list[str]],
) -> dict[int, tuple[int, int]]:
    """Return window and plane of every event frame covered by a window.

    :return: frame -> (window index, plane); frames outside events or uncovered are absent.
    :raises ValueError: on an unknown policy or planes incompatible with the window.
    """
    from evd.explain.project import offline_assignment, offline_plane

    intervals = [[e["start"], e["end"]] for e in video["events"]]
    mapped = offline_assignment(windows, video["num_frames"], politica=policy, intervalli=intervals)
    result = {}
    for frame in _event_frames(video):
        i = int(mapped[frame])
        if i >= 0 and classes_of(frame):
            result[frame] = (i, offline_plane(windows[i], frame, planes))
    return result


def _group(
    video: dict, mapping: dict[int, tuple[str, int]], classes_of: Callable[[int], list[str]]
) -> list[dict]:
    """Merge contiguous event frames with the same window, plane and classes.

    :return: inclusive units; ``known`` is false where the plane is unknown.
    """
    units = []
    for frame in _event_frames(video):
        classes = classes_of(frame)
        if not classes:
            continue
        window_id, plane = mapping.get(frame, (None, None))
        previous = units[-1] if units else None
        if (
            previous
            and previous["end"] + 1 == frame
            and previous["window_id"] == window_id
            and previous["plane"] == plane
            and previous["classes"] == classes
        ):
            previous["end"] = frame
        else:
            units.append(
                {
                    "start": frame,
                    "end": frame,
                    "window_id": window_id,
                    "plane": plane,
                    "classes": classes,
                    "known": plane is not None,
                }
            )
    for unit in units:
        unit["id"] = (
            f"{unit['window_id'] or 'unknown'}:{unit['plane']}:{unit['start']}-{unit['end']}"
        )
        unit["representative"] = unit["start"]
    return units


#: Version of the base-plane algorithm: increment it when the algorithm changes, or the
#: copies saved in the project would remain valid.
PROTOCOL_UNITS_VERSION = 1


def protocol_fingerprint(video: dict) -> str:
    """Return a fingerprint of everything that determines the base planes of a video.

    :return: sha256 of algorithm, protocol, policy and video geometry/events.
    """
    keys = ("video_id", "split", "num_frames", "fps", "events")
    return sha(
        json.dumps(
            {
                "version": PROTOCOL_UNITS_VERSION,
                "protocol_sha256": _protocol_sha(),
                "assignment": PROTOCOL_ASSIGNMENT,
                "video": {k: video.get(k) for k in keys},
            },
            sort_keys=True,
        ).encode()
    )


@lru_cache(maxsize=1)
def _protocol_sha() -> str:
    return sha(PROTOCOL_PATH.read_bytes())


def protocol_units(video: dict) -> dict:
    """Return the base VTCD planes of every event, without source or boxes.

    ``edp_v2`` windows, policy :data:`PROTOCOL_ASSIGNMENT` and one plane per tubelet: the
    split an exported VTCD source gives to the same frames. A video without events has no
    units.

    :return: units as in :func:`temporal_units`, with the provenance of the computation.
    """
    planes = protocol_planes()

    def classes_of(frame: int) -> list[str]:
        return event_classes(video, frame)

    windows = protocol_windows(video)
    assigned = _assign(video, windows, PROTOCOL_ASSIGNMENT, planes, classes_of)
    ids = {i: windows[i].window_id for i in {i for i, _ in assigned.values()}}
    units = _group(video, {f: (ids[i], p) for f, (i, p) in assigned.items()}, classes_of)
    return {
        "units": units,
        "provenance": {
            "mode": "edp_v2_protocol",
            "assignment": PROTOCOL_ASSIGNMENT,
            "planes": planes,
            "protocol_sha256": _protocol_sha(),
            "fingerprint": protocol_fingerprint(video),
        },
        "complete": all(u["known"] for u in units),
    }


def temporal_units(
    video: dict, metadata: dict, frames: dict, *, manifest: dict | None = None
) -> dict:
    """Group frames without interpolating or propagating across unknown gaps.

    :param manifest: original renderer receipt, for sources predating the export of the
        temporal geometry inside ``boxes.json``.
    :return: inclusive units, completeness and provenance of the reconstruction.
    :raises ValueError: if the declared geometry is incompatible with the original boxes.
    """
    from evd.types import WindowSpec

    # A per-class VTCD source (3 October 2026) declares the classes of its events: events of
    # another class in the same video are not its own, and there a missing box is not an
    # empty plane to review.
    own = set(metadata.get("classi") or ())

    def classes_of(frame: int) -> list[str]:
        classes = event_classes(video, frame)
        return classes if not own or own & set(classes) else []

    layout = metadata.get("temporal_layout")
    assignment = (layout or {}).get("assignment") or (manifest or {}).get("assegnazione")
    resolved = None
    provenance = {"mode": "csv_observed"}
    if layout:
        resolved = [
            WindowSpec.from_frame_indices(video["video_id"], w["frame_indices"])
            for w in layout["windows"]
        ]
        if [w.window_id for w in resolved] != [w["window_id"] for w in layout["windows"]]:
            raise ValueError("Window identity differs from the exported temporal indices.")
        provenance = {"mode": "exported_geometry"}
    elif assignment:
        candidates = {w.window_id: w for w in protocol_windows(video)}
        wanted = list(dict.fromkeys(w["window_id"] for w in metadata["finestre"]))
        if all(w in candidates for w in wanted):
            resolved = [candidates[w] for w in wanted]
            provenance = {
                "mode": "verified_edp_v2_reconstruction",
                "protocol_sha256": _protocol_sha(),
                "renderer_manifest": manifest,
            }
    mapping = {}
    if resolved is not None and assignment:
        assigned = _assign(
            video, resolved, assignment["politica"], int(metadata["grid"][0]), classes_of
        )
        # Schema 2: the boxes of a plane also depend on the class of the frame event, since each
        # class has its own dictionary.
        composite = metadata.get("schema") == 2
        planes = {
            key: {p["t"]: p["box"] for p in w["piani"]} for key, w in window_index(metadata).items()
        }
        ids = [w.window_id for w in resolved]
        for frame in _event_frames(video):
            classes = classes_of(frame)
            if not classes:
                continue
            if frame not in assigned:
                if str(frame) in frames:
                    raise ValueError("CSV covered by a window that is not in the geometry.")
                continue
            index, plane = assigned[frame]
            window_id = ids[index]
            if composite and len(set(classes) & own) != 1:
                raise ValueError(f"Frame {frame}: dictionary class not unique {classes}.")
            key = (window_id, next(iter(set(classes) & own)) if composite else None)
            expected = [b["nativo"] for b in planes.get(key, {}).get(plane, [])]
            observed = frames.get(str(frame), [])
            if expected != [b["xyxy"] for b in observed] or any(
                b["window_id"] != window_id or b["plane"] != plane for b in observed
            ):
                raise ValueError(f"Temporal mapping incompatible with the CSV at frame {frame}.")
            mapping[frame] = (window_id, plane)
    else:
        for key, boxes in frames.items():
            identities = {(b["window_id"], b["plane"]) for b in boxes}
            if len(identities) != 1:
                raise ValueError("Several windows/planes in the same CSV frame.")
            mapping[int(key)] = next(iter(identities))
    units = _group(video, mapping, classes_of)
    return {"units": units, "provenance": provenance, "complete": all(u["known"] for u in units)}


def original_manifest(video: dict, metadata: dict, csv_text: str) -> dict | None:
    """Find the receipt only next to the same imported JSON/CSV pair.

    :return: the original manifest or None; no policy is presumed.
    """
    for name in video.get("candidates", []):
        path = Path(name)
        receipt = path.with_name("manifest.json")
        if (
            not receipt.is_file()
            or not path.is_file()
            or not path.with_name("boxes.json").is_file()
        ):
            continue
        if json.loads(
            path.with_name("boxes.json").read_text(encoding="utf-8-sig")
        ) == metadata and path.read_text(encoding="utf-8-sig").replace(
            "\r\n", "\n"
        ) == csv_text.lstrip("﻿").replace("\r\n", "\n"):
            return json.loads(receipt.read_text(encoding="utf-8-sig"))
    return None
