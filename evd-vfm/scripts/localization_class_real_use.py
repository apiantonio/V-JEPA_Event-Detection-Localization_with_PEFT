"""Real-use scope of the class-routing localization: the detector decides the chunks (CPU).

Plan: ``docs/varie/campagna_uso_reale_localizzazione_20261028.md``. Until 27 October 2026 the
boxes were made on the annotated test chunks whatever the detector said; in real use
(decision of 28 October 2026) every chunk of the test videos is seen, and only the chunks the
binary head of the configuration's own detector flags get boxes: ``score >= τ``, with ``τ``
the Youden threshold of that detector's validation and, on MIVIA-Crime, the same threshold
transferred (MIVIA-Crime has no validation, as in ``scripts/valuta_esterno.py``).

``plan`` writes, for every detector, the chunks it flags on the videos in scope and the rows
of those that have no pass record yet (they are not among the annotated chunks of
``plan/test.json``), in the format of ``localization_class_plan.py``, so
``localization_class_pass.py`` runs on them unchanged. It also writes the scope manifests
(the stages are written on these videos only) and the temporal intervals of every video in
scope, from which the evaluators take the frames outside every event (the negative images).
``rehearsal`` keeps only the flagged chunks that already have a record, to rehearse the CPU
steps on the real data while the GPU pass runs.

Scope: every FS-UCF-Crime test video (all 140 anomalous videos have spatial GT); on
MIVIA-Crime the 39 videos with spatial GT and as many normal videos, chosen as the annotated
ones were: by increasing number of base planes
(``apps.annotator.planes.protocol_units``, the whole normal video taken as one event), ties by
name. The scores and thresholds are the detectors' official ones (``scores.jsonl`` and
``metrics.json`` of the run and of its ``__mivia`` external evaluation); the threshold is
recomputed from the validation scores and must equal the stored one. No spatial ground truth
is read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

import yaml  # noqa: E402

from evd.annotations.schema import window_overlap  # noqa: E402
from evd.annotations.ucf_lab import MIVIA_CRIME, UcfLabAnnotations  # noqa: E402
from evd.data.inventory import read_records  # noqa: E402
from evd.eval.metrics import threshold_youden  # noqa: E402
from evd.exp.localization_auto import read_json, write_json  # noqa: E402
from evd.protocol.chunks import sampler_for_protocol  # noqa: E402
from evd.types import VideoMeta  # noqa: E402
from localization_class_plan import row, sealed  # noqa: E402

CONFIG = ROOT / "configs/localization_class_routing_20261021.json"
DATASETS = ("ucf_test", "mivia_test")


def read_scores(path: Path) -> list[dict]:
    """:return: the rows of a ``scores.jsonl``."""
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def detector_scores(run_dir: Path) -> tuple[dict, dict[str, dict[str, float]]]:
    """Threshold and test scores of one detector, on FS-UCF-Crime and MIVIA-Crime.

    :return: ``(threshold, scores)``: the threshold with its provenance and checks, and per
        dataset the score of every test window id.
    :raises ValueError: if the Youden threshold recomputed from the validation scores differs
        from ``metrics.json``, or the MIVIA-Crime evaluation does not carry the same threshold
        transferred from this run.
    """
    run_id = run_dir.name
    stored = read_json(run_dir / "metrics.json")["threshold"]
    rows = read_scores(run_dir / "scores.jsonl")
    val = [r for r in rows if r["split"] == "val"]
    value, source = threshold_youden([r["score"] for r in val], [r["label"] for r in val])
    if stored.get("source") != source or value != stored["value"]:
        raise ValueError(f"{run_id}: Youden on validation gives {value}, metrics.json {stored}")
    external = run_dir / f"{run_id}__mivia"
    transferred = read_json(external / "metrics.json")["threshold"]
    if transferred["value"] != value or transferred.get("trasferita_da") != run_id:
        raise ValueError(f"{external.name}: threshold {transferred} is not the one of {run_id}")
    scores = {
        "ucf_test": {r["window_id"]: r["score"] for r in rows if r["split"] == "test"},
        "mivia_test": {r["window_id"]: r["score"] for r in read_scores(external / "scores.jsonl")},
    }
    threshold = {
        "value": value,
        "rule": "score >= value",
        "source": source,
        "validation_chunks": len(val),
        "recomputed_equals_metrics_json": True,
        "mivia_test": {k: transferred[k] for k in ("value", "source", "trasferita_da")},
    }
    return threshold, scores


def base_planes(meta: dict) -> int:
    """:return: the base planes of a normal video taken as one event, as Forma counts them."""
    from apps.annotator.planes import protocol_units

    video = {
        **meta,
        "split": "test",
        "events": [{"start": 0, "end": meta["num_frames"] - 1, "class_name": "Normal"}],
    }
    return len(protocol_units(video)["units"])


def mivia_scope(manifest: list[dict], annotations: dict, annotated: set[str]) -> dict:
    """The MIVIA-Crime videos in scope: the annotated ones and as many normal videos.

    :return: ``{"videos", "annotated", "normal", "normal_ranking"}``, the normal videos by
        increasing base planes, ties by name; the ranking keeps every normal video.
    :raises ValueError: if an annotated video is missing or has no temporal event.
    """
    names = {m["video_id"] for m in manifest}
    if annotated - names or any(not annotations[v].intervals for v in annotated):
        raise ValueError("An annotated MIVIA-Crime video is missing or has no temporal event")
    ranking = sorted(
        (base_planes(m), m["video_id"])
        for m in manifest
        if not annotations[m["video_id"]].intervals
    )
    normal = [v for _, v in ranking[: len(annotated)]]
    return {
        "videos": sorted(annotated | set(normal)),
        "annotated": sorted(annotated),
        "normal": normal,
        "normal_ranking": [{"video_id": v, "base_planes": n} for n, v in ranking],
    }


def flagged_rows(
    dataset: str,
    manifest: list[dict],
    scores: dict[str, float],
    threshold: float,
    annotations: dict,
    sampler,
    known: set[tuple[str, str]],
) -> tuple[list[str], list[dict]]:
    """Window ids flagged by the detector and rows of the flagged chunks without a record.

    :return: ``(positives, rows)``; ``positives`` in video and temporal order.
    :raises ValueError: if the windows of a video differ from the scored ones.
    """
    positives, rows = [], []
    for m in manifest:
        meta = VideoMeta(
            m["video_id"], m["path"], m["num_frames"], m["fps"], m["width"], m["height"]
        )
        windows = sampler.windows_for_video(meta)
        scored = {w for w in scores if w.split("@", 1)[0] == m["video_id"]}
        if {w.window_id for w in windows} != scored:
            raise ValueError(f"{dataset} {m['video_id']}: protocol windows differ from the scores")
        for index, window in enumerate(windows):
            if scores[window.window_id] < threshold:
                continue
            positives.append(window.window_id)
            if (dataset, window.window_id) in known:
                continue
            count, dominant = window_overlap(annotations[m["video_id"]], *window.frame_span)
            rows.append(
                {
                    **row(dataset, meta, window, count, dominant),
                    "index": index,
                    "path": m["path"],
                    "width": m["width"],
                    "height": m["height"],
                }
            )
    return positives, rows


def plan(args) -> None:
    config = read_json(CONFIG)
    protocol = yaml.safe_load((ROOT / "configs/protocol/edp_v2.yaml").read_text(encoding="utf-8"))
    sampler = sampler_for_protocol(None, protocol, validazione=False)
    manifests = {
        "ucf_test": read_json(args.manifest_ucf),
        "mivia_test": read_json(args.manifest_mivia),
    }
    records = read_records(args.data_root / "ucf_crime_inventory/inventory.csv")
    annotations = {
        "ucf_test": UcfLabAnnotations.from_dir(
            args.data_root / "annotations/fs_ucf_crime",
            num_frames={r.video_id: r.num_frames for r in records},
        ).load(),
        "mivia_test": UcfLabAnnotations.from_dir(
            args.mivia_annotations,
            delivery=MIVIA_CRIME,
            num_frames={m["video_id"]: m["num_frames"] for m in manifests["mivia_test"]},
        ).load(),
    }
    test_plan = read_json(args.test_plan)
    known = {(r["dataset"], r["window_id"]) for r in test_plan["rows"]}
    annotated = {r["video_id"] for r in test_plan["rows"] if r["dataset"] == "mivia_test"}
    scope = {
        "ucf_test": {
            "videos": sorted(m["video_id"] for m in manifests["ucf_test"]),
            "rule": "every test video",
        },
        "mivia_test": {
            **mivia_scope(manifests["mivia_test"], annotations["mivia_test"], annotated),
            "rule": "the videos with spatial GT and as many normal videos by increasing base "
            "planes, ties by name",
        },
    }
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    scoped = {}
    for dataset in DATASETS:
        wanted = set(scope[dataset]["videos"])
        scoped[dataset] = [m for m in manifests[dataset] if m["video_id"] in wanted]
        if len(scoped[dataset]) != len(wanted):
            raise ValueError(f"{dataset}: a video in scope is not in the manifest")
        write_json(out / f"manifest-{dataset}.json", scoped[dataset])
    outside = sorted(k for k in known if k[1].split("@", 1)[0] not in scope[k[0]]["videos"])
    if outside:
        raise ValueError(f"Annotated chunks outside the scope: {outside[:3]}")
    intervals = {
        dataset: {
            m["video_id"]: [list(i) for i in annotations[dataset][m["video_id"]].intervals]
            for m in scoped[dataset]
        }
        for dataset in DATASETS
    }
    write_json(
        out / "intervals.json",
        sealed({"intervals": intervals, "unit": "native frames, ends included"}),
    )
    summary = {}
    for name in args.detectors:
        run_id = config["detectors"][name]
        threshold, scores = detector_scores(args.runs / run_id)
        positives, rows = {}, []
        for dataset in DATASETS:
            flagged, new = flagged_rows(
                dataset,
                scoped[dataset],
                scores[dataset],
                threshold["value"],
                annotations[dataset],
                sampler,
                known,
            )
            positives[dataset] = flagged
            rows += new
        counts = {
            dataset: {
                "videos": len(scoped[dataset]),
                "chunks": sum(
                    len(
                        sampler.windows_for_video(
                            VideoMeta(
                                m["video_id"],
                                m["path"],
                                m["num_frames"],
                                m["fps"],
                                m["width"],
                                m["height"],
                            )
                        )
                    )
                    for m in scoped[dataset]
                ),
                "flagged": len(positives[dataset]),
                "flagged_with_record": sum((dataset, w) in known for w in positives[dataset]),
                "flagged_new": sum(r["dataset"] == dataset for r in rows),
                "annotated_not_flagged": sum(
                    1 for d, w in known if d == dataset and w not in set(positives[dataset])
                ),
            }
            for dataset in DATASETS
        }
        body = {
            "detector": name,
            "run_id": run_id,
            "threshold": threshold,
            "scope": scope,
            "positives": positives,
            "counts": counts,
            "rows": rows,
            "test_plan_sha256": test_plan["sha256"],
            "intervals_sha256": read_json(out / "intervals.json")["sha256"],
        }
        write_json(out / f"{name}.json", sealed(body))
        summary[name] = {"threshold": threshold["value"], "counts": counts, "rows": len(rows)}
    print(
        json.dumps({"mivia_normal": scope["mivia_test"]["normal"], "detectors": summary}, indent=1)
    )


def rehearsal(args) -> None:
    """Restrict a real-use plan to the flagged chunks that already have a record.

    The rehearsal of the CPU steps (stages, evaluations, report) on the real data while the GPU
    pass over the other flagged chunks runs: same scope and intervals, no row to pass. Its
    numbers are not results.
    """
    body = read_json(args.plan)
    known = {(r["dataset"], r["window_id"]) for r in read_json(args.test_plan)["rows"]}
    positives = {d: [w for w in ids if (d, w) in known] for d, ids in body["positives"].items()}
    out = {k: v for k, v in body.items() if k != "sha256"}
    out.update(positives=positives, rows=[], rehearsal_of=body["sha256"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, sealed(out))
    print(json.dumps({d: len(ids) for d, ids in positives.items()}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="step", required=True)
    r = sub.add_parser("rehearsal", help="a plan of the flagged chunks that have a record")
    r.add_argument("--plan", type=Path, required=True, help="a real-use plan of one detector")
    r.add_argument("--test-plan", type=Path, required=True, help="plan/test.json")
    r.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("plan", help="flagged chunks, scope manifests and temporal intervals")
    p.add_argument("--data-root", type=Path, required=True, help="UCF inventory/annotations")
    p.add_argument("--mivia-annotations", type=Path, required=True)
    p.add_argument("--manifest-ucf", type=Path, required=True, help="VideoMeta rows")
    p.add_argument("--manifest-mivia", type=Path, required=True, help="VideoMeta rows")
    p.add_argument("--test-plan", type=Path, required=True, help="plan/test.json")
    p.add_argument("--runs", type=Path, default=Path(os.environ.get("EVD_RUNS_DIR", "runs")))
    p.add_argument(
        "--detectors",
        nargs="+",
        default=["campaign", "b3_sum", "b3_gradnorm"],
        choices=("campaign", "b3_sum", "b3_gradnorm", "attentive"),
    )
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (plan if args.step == "plan" else rehearsal)(args)


if __name__ == "__main__":
    main()
