"""Derive P07 at other IG mass fractions from a sealed VTCD stage with concept maps (no GPU).

P07 keeps the minimal prefix of concepts reaching a share of the chunk's positive IG mass.
A ``VTCDAblationStage`` run with ``mass_modes`` (``vtcd_ig_absolute``, 21 October) saves the
token concept map of every localized chunk (``maps/*.npz`` key ``concept_ids``). With the
IG ensemble map of the IG stage it read, P07 is recomputed at any fraction on CPU, on the
same concept assignment. The derived stage carries:

- ``P07-mass70`` recomputed: it must equal the boxes of the source stage on every chunk
  (same map, same assignment, same rule), otherwise the derivation stops;
- ``P07-massNN`` for every requested fraction.

Each derived method is charged the source ``P07-mass70`` cost (decode, encode, IG, SLIC and
assignment are shared) plus its own prefix and box extraction. Output layout and seals are
the evaluator's (``localization_auto_evaluate``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402

from evd.exp.localization_auto import file_sha256, read_json, seal_stage, write_json  # noqa: E402
from evd.exp.localization_vtcd import CONCEPTS, mass_prefix_concepts  # noqa: E402
from evd.explain.localization import native_boxes  # noqa: E402

SOURCE = "P07-mass70"
STAGE = "vtcd_mass_derived"


def experiment(fraction: float) -> str:
    """:return: the experiment id of P07 at ``fraction`` (``0.55`` → ``P07-mass55``)."""
    percent = round(fraction * 100)
    if not 0 < percent <= 100 or abs(percent - fraction * 100) > 1e-9:
        raise ValueError(f"Fraction must be a whole percentage in (0, 1], not {fraction}")
    return f"P07-mass{percent}"


def _plain(boxes) -> list:
    """:return: boxes as JSON would read them back (lists of ints)."""
    return json.loads(json.dumps(boxes))


def derive_video(
    source: Path, ig: Path, target: Path, complete: dict, fractions: dict[str, float]
) -> None:
    """Write the derived chunk records and ``complete.json`` of one video.

    :param source: the video directory of the sealed VTCD stage.
    :param ig: the video directory of the IG stage the VTCD stage read.
    :param target: the derived video directory.
    :param fractions: experiment id → fraction, ``P07-mass70`` included.
    :raises ValueError: on a changed record, a chunk/IG mismatch or a replay difference.
    """
    width, height = complete["video"]["width"], complete["video"]["height"]
    chunks = []
    for item in complete["chunks"]:
        path = source / item["path"]
        if file_sha256(path) != item["sha256"]:
            raise ValueError(f"Prediction chunk changed after sealing: {path}")
        record = read_json(path)
        original = record["methods"][SOURCE]
        active = record["localized"] if "localized" in record else record["gate_passed"]
        methods = {}
        if active:
            stem = Path(item["path"]).stem
            concept = np.load(source / "maps" / f"{stem}.npz")["concept_ids"]
            ig_record = read_json(ig / "chunks" / f"{stem}.json")
            if ig_record["window_id"] != record["window_id"]:
                raise ValueError(f"IG record of another chunk: {ig / 'chunks' / stem}")
            positive = np.maximum(np.load(ig / "maps" / f"{stem}.npz")["ig_ensemble"], 0)
            for name, fraction in fractions.items():
                start = time.perf_counter()
                mask = mass_prefix_concepts(positive, concept, np.arange(CONCEPTS), fraction)
                boxes = native_boxes(mask, width, height)
                own = time.perf_counter() - start
                methods[name] = {
                    "boxes": boxes,
                    "method_seconds": original["method_seconds"] + own,
                    "inference_compute_seconds": original["inference_compute_seconds"] + own,
                }
            if _plain(methods[SOURCE]["boxes"]) != original["boxes"]:
                raise ValueError(f"P07-mass70 replay differs from the source: {path}")
        else:
            methods = {name: dict(original) for name in fractions}
        derived = {
            **{k: record[k] for k in ("window", "window_id", "grid", "score", "gate_passed")},
            **({"localized": record["localized"]} if "localized" in record else {}),
            "identity": {"stage": STAGE, "fractions": fractions},
            "methods": methods,
        }
        out = target / item["path"]
        write_json(out, derived)
        chunks.append({"path": item["path"], "sha256": file_sha256(out)})
    records = [read_json(target / c["path"]) for c in chunks]
    timing = {
        name: {
            **complete["timing"][SOURCE],
            "inference_compute_seconds": sum(
                r["methods"][name]["inference_compute_seconds"] for r in records
            ),
        }
        for name in fractions
    }
    write_json(
        target / "complete.json",
        {
            "status": "complete",
            "identity": records[0]["identity"] if records else {"stage": STAGE},
            "video": complete["video"],
            "chunks": chunks,
            "timing": timing,
            "prediction_scope": complete.get(
                "prediction_scope", "entire_video_all_protocol_windows"
            ),
        },
    )


def derive(
    vtcd: Path, ig: Path, output: Path, fractions: list[float], manifests: dict[str, Path]
) -> dict:
    """Derive and seal the stage on every dataset of ``manifests``.

    :param vtcd: root of the sealed VTCD stage with concept maps.
    :param ig: root of the IG stage it read (``<dataset>/<video key>/maps``).
    :param output: root of the derived stage.
    :return: the provenance written to ``operations/derivation.json``.
    """
    wanted = {SOURCE: 0.7, **{experiment(f): f for f in fractions}}
    provenance = {"stage": STAGE, "fractions": wanted, "source_seals": {}, "ig": str(ig)}
    for dataset, manifest in manifests.items():
        seal_path = vtcd / "operations" / f"seal-{dataset}.json"
        seal = read_json(seal_path)
        if SOURCE not in seal["methods"]:
            raise ValueError(f"{SOURCE} absent from {seal_path}")
        for entry in seal["videos"]:
            source = vtcd / entry["directory"]
            if file_sha256(source / "complete.json") != entry["complete_sha256"]:
                raise ValueError("Prediction changed after sealing")
            derive_video(
                source,
                ig / entry["directory"],
                output / entry["directory"],
                read_json(source / "complete.json"),
                wanted,
            )
        seal_stage(output, dataset, list(wanted), manifest)
        provenance["source_seals"][dataset] = file_sha256(seal_path)
    write_json(output / "operations" / "derivation.json", provenance)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--vtcd", type=Path, required=True, help="sealed VTCD stage root")
    parser.add_argument("--ig", type=Path, required=True, help="IG stage root it read")
    parser.add_argument("--fractions", type=float, nargs="+", required=True)
    parser.add_argument("--ucf-manifest", type=Path, required=True)
    parser.add_argument("--mivia-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provenance = derive(
        args.vtcd,
        args.ig,
        args.output,
        args.fractions,
        {"ucf_test": args.ucf_manifest, "mivia_test": args.mivia_manifest},
    )
    print(json.dumps(provenance), flush=True)


if __name__ == "__main__":
    main()
