"""Collection import, with per-pair verification and immutable sources.

Reuses ``Project.add_source`` and ``import_proposals`` (a single reading of the VTCD
format). Imports never write human annotations (``docs/20``).
"""

import csv
import io
import json
from pathlib import Path

from apps.annotator.project import Project, import_proposals, sha


def collection_candidates(project: Project) -> list[dict]:
    """:return: registered pairs for all videos, without reading videos or maps."""
    return [
        {
            "video": vid,
            "candidate": i,
            "label": Path(p).parent.name,
            "name": f"{video['dataset']}/{video['video_id']}/{Path(p).parent.name}",
        }
        for vid, video in project.videos.items()
        for i, p in enumerate(video.get("candidates", []))
    ]


def import_pair(project: Project, body: dict, *, commit: bool = False) -> dict:
    """Match and verify a collection pair; optionally import it.

    :param body: registered candidate, or metadata/csv texts and an optional dataset.
    :return: video, source identity and state ready/existing/imported.
    :raises ValueError: if the video is ambiguous or missing, or the format or candidate is
        invalid.
    """
    if "candidate" in body:
        vid, index = body["video"], body["candidate"]
        candidates = project.videos[vid].get("candidates", [])
        if type(index) is not int or not 0 <= index < len(candidates):
            raise ValueError("Candidate not registered in the project.")
        path = Path(candidates[index])
        metadata_text = path.with_name("boxes.json").read_bytes().decode("utf-8")
        csv_text = path.read_bytes().decode("utf-8")
        label = body.get("label") or path.parent.name
    else:
        metadata_text, csv_text = body["metadata"], body["csv"]
        meta = json.loads(metadata_text.lstrip("\ufeff"))
        identities = {w["window_id"].rsplit("@", 1)[0] for w in meta.get("finestre", [])}
        identities.update(
            r["video_id"] for r in csv.DictReader(io.StringIO(csv_text.lstrip("\ufeff")))
        )
        if len(identities) != 1:
            raise ValueError("The pair must identify a single video, even if it contains no box.")
        choices = [
            v["id"]
            for v in project.videos.values()
            if v["video_id"] in identities
            and (not body.get("dataset") or v["dataset"] == body["dataset"])
        ]
        if len(choices) != 1:
            raise ValueError("Missing or ambiguous video: select the correct dataset.")
        vid, label = choices[0], body.get("label", "Collection import")
    if not isinstance(label, str) or not label.strip() or len(label) > 200:
        raise ValueError("Invalid source name (at most 200 characters).")
    sid = sha(metadata_text.encode() + b"\0" + csv_text.encode())
    with project.connect() as con:
        exists = bool(
            con.execute("SELECT 1 FROM sources WHERE video=? AND id=?", (vid, sid)).fetchone()
        )
    if commit:
        project.add_source(vid, metadata_text, csv_text, label)
    else:
        import_proposals(
            project.videos[vid], json.loads(metadata_text.lstrip("\ufeff")), csv_text, sid
        )
    return {
        "video": vid,
        "source_id": sid,
        "label": label,
        "status": "existing" if exists else "imported" if commit else "ready",
    }
