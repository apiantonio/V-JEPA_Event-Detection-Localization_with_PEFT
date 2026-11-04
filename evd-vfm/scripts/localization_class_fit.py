"""Fit the dictionaries of the class-routing localization campaign (CPU).

Plan: ``docs/varie/piano_localizzazione_per_classe_20261021.md`` §5.3 and §6.2. Reads the
region descriptors that ``localization_class_pass.py`` saved for the A1 references and the
pure train chunks of one detector, and writes ``dictionaries.npz`` with:

- ``global_dictionary`` (8 concepts, P07): copied from ``--existing`` (the campaign
  detector's ``vtcd_models.npz``) or fitted on the 208 references (B3 detectors);
- ``a1`` (13 × 8 concepts): copied from ``--existing`` (``class_dictionaries``) or fitted on
  the 16 references of each class;
- ``a2`` (13 × 8 concepts): fitted on all the pure chunks of each class with the halving rule
  (every failure recorded, half of the chunks drawn with seed 42, below 16 chunks the A1
  dictionary); classes whose plan source is ``a1_references`` (Explosion, Shooting) use the
  A1 dictionary unchanged.

With ``--without-a2`` (29 October 2026, the attentive-probe baseline, which runs only the
``global`` configurations) the file holds the global and A1 dictionaries of ``--existing`` and
no ``a2``: no pure-chunk pass is needed.

``fit_receipts.json`` holds every attempt with its seconds and peak memory, for the timing
analysis. Global and A1 fits must converge, as in the 16 October campaign.

Every fitted dictionary is saved as a part under ``--output/parts/`` as soon as it is done,
and a rerun loads the parts instead of fitting again: a fit of the largest classes can take
hours and a ``fatq`` job has four, so a job stopped by its time limit loses one class at most.
``--classes`` restricts the A2 classes fitted by this run (to split them over jobs);
``dictionaries.npz`` is written only when every part exists.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402

from evd.exp.localization_auto import read_json, write_json  # noqa: E402
from evd.exp.localization_class import fit_with_halving, peak_memory_mb  # noqa: E402
from evd.exp.localization_vtcd import fit_dictionary  # noqa: E402
from localization_class_pass import _key, rows_of  # noqa: E402
from localization_class_plan import CLASSES  # noqa: E402


def descriptors_of(root: Path, rows: list[dict]) -> list[np.ndarray]:
    """:return: the saved ``R×D`` region descriptors of each row, in row order.

    :raises FileNotFoundError: if a chunk of the list was not encoded.
    """
    return [np.load(root / f"{_key(row)}.npz")["descriptors"].astype(np.float64) for row in rows]


def fitted_part(
    parts: Path, name: str, identity: dict, compute: Callable[[], tuple[np.ndarray, object]]
) -> tuple[np.ndarray, object]:
    """Load a dictionary already fitted by an earlier run, or fit it and save it.

    The ``.json`` is written after the ``.npz``, so its presence marks a finished part.

    :param parts: folder of the parts.
    :param name: part name (``global``, ``a1-<class>``, ``a2-<class>``).
    :param identity: inputs of the fit; a saved part with another identity is refused.
    :param compute: the fit, ``() -> (dictionary, receipt)``.
    :return: the dictionary and its receipt.
    :raises ValueError: if a saved part has another identity.
    """
    stem = parts / name
    record = stem.with_suffix(".json")
    if record.exists():
        saved = read_json(record)
        if saved["identity"] != {**identity, "part": name}:
            raise ValueError(f"Existing fit part with another identity: {record}")
        return np.load(stem.with_suffix(".npz"))["dictionary"], saved["receipt"]
    dictionary, receipt = compute()
    parts.mkdir(parents=True, exist_ok=True)
    temporary = stem.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, dictionary=np.asarray(dictionary))
    temporary.replace(stem.with_suffix(".npz"))
    write_json(record, {"identity": {**identity, "part": name}, "receipt": receipt})
    return np.asarray(dictionary), receipt


def timed_fit(descriptors: list[np.ndarray]) -> tuple[np.ndarray, dict]:
    """:return: a converged dictionary and its receipt with seconds and peak memory."""
    start = time.perf_counter()
    dictionary, receipt = fit_dictionary(np.concatenate(descriptors))
    return dictionary, {
        "chunks": len(descriptors),
        "iterations": receipt["iterations"],
        "seconds": time.perf_counter() - start,
        "peak_memory_mb": peak_memory_mb(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass-root", type=Path, required=True, help="output of the pass")
    parser.add_argument("--references", type=Path, required=True, help="references.json")
    parser.add_argument("--pure", type=Path, required=True, help="train_pure.json")
    parser.add_argument("--existing", type=Path, help="vtcd_models.npz of the campaign detector")
    parser.add_argument("--classes", nargs="*", choices=CLASSES, help="A2 classes of this run")
    parser.add_argument(
        "--without-a2", action="store_true", help="copy --existing and fit no A2 dictionary"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.without_a2 and (args.existing is None or args.classes):
        parser.error("--without-a2 needs --existing and fits no class")
    references, pure = read_json(args.references), read_json(args.pure)
    parts = args.output / "parts"
    identity = {
        "pass_root": str(args.pass_root),
        "references_sha256": references["sha256"],
        "pure_sha256": pure["sha256"],
        "existing": None if args.existing is None else str(args.existing),
    }
    receipts: dict = {"global": None, "a1": {}, "a2": {}}
    by_class = {c: [r for r in references["rows"] if r["class_name"] == c] for c in CLASSES}
    if args.existing is not None:
        model = np.load(args.existing)
        if [str(c) for c in model["class_names"]] != CLASSES:
            raise ValueError("Existing class dictionaries are in another class order")
        global_dictionary = model["global_dictionary"]
        a1 = np.asarray(model["class_dictionaries"])
        receipts["global"] = receipts["a1"] = {"source": str(args.existing)}
    else:
        root = args.pass_root / "references"
        global_dictionary, receipts["global"] = fitted_part(
            parts,
            "global",
            identity,
            lambda: timed_fit(descriptors_of(root, rows_of(references))),
        )
        a1 = []
        for name in CLASSES:
            dictionary, receipts["a1"][name] = fitted_part(
                parts,
                f"a1-{name}",
                identity,
                lambda name=name: timed_fit(descriptors_of(root, by_class[name])),
            )
            a1.append(dictionary)
        a1 = np.stack(a1)
    selected = set(CLASSES if args.classes is None else args.classes)
    a2, missing = [], []
    for index, name in enumerate([] if args.without_a2 else CLASSES):
        source = pure["sources"][name]["source"]
        if source == "a1_references":
            receipts["a2"][name] = [{"outcome": "a1_dictionary", "reason": "fewer than 16 pure"}]
            a2.append(a1[index])
            continue
        if name not in selected and not (parts / f"a2-{name}.json").exists():
            missing.append(name)
            continue
        dictionary, attempts = fitted_part(
            parts,
            f"a2-{name}",
            identity,
            lambda name=name, index=index: fit_with_halving(
                descriptors_of(args.pass_root / "train_pure", pure["by_class"][name]), a1[index]
            ),
        )
        receipts["a2"][name] = attempts
        a2.append(dictionary)
        print(json.dumps({"class": name, "attempts": attempts}), flush=True)
    if missing:
        print(json.dumps({"status": "partial", "a2_missing": missing}))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = {"global_dictionary": global_dictionary, "a1": a1, "class_names": np.asarray(CLASSES)}
    if not args.without_a2:
        arrays["a2"] = np.stack(a2)
    np.savez(args.output / "dictionaries.npz", **arrays)
    write_json(
        args.output / "fit_receipts.json",
        {
            "receipts": receipts,
            "references_sha256": references["sha256"],
            "pure_sha256": pure["sha256"],
        },
    )
    print(json.dumps({"status": "complete", "output": str(args.output)}))


if __name__ == "__main__":
    main()
