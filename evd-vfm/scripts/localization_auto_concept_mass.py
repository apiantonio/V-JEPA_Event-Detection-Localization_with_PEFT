"""IG mass of the VTCD concepts on train and validation chunks, to choose the P07 fraction.

P07 keeps the minimal prefix of concepts carrying a fixed share of the chunk's positive IG
mass. The share (70%) was fixed a priori, and the 10–90% sweep exists only on test chunks,
where choosing would be selection on the test. This script measures how the positive IG mass
spreads over the K=8 concepts on chunks no evaluation reads:

- ``plan`` (CPU, no decoding): the positive chunks of the frozen A2 train sample (one per
  event, ``train-plan.json``) and every temporally positive chunk of the anomalous
  validation videos on the validation grid, with their dominant class.
- ``compute`` (GPU shard): per chunk, the IG ensemble of the test stage
  (``localization_ig.explain_ig``), SLIC and NNLS assignment on the frozen global dictionary
  (``localization_vtcd``), and per concept the positive and signed IG mass and the token
  count. The ensemble map (fp16) and the token concept map are saved, so other selection
  rules replay on CPU. Resumes per chunk.
- ``report`` (CPU): per split, sorted mass shares, concepts and area selected at each
  fraction, and the concepts whose mass share exceeds their token share. The token count
  kept is also tabulated at every 1% of fraction (``tokens_by_fraction.csv``), with
  histograms of the shares for plotting.

No test file and no spatial ground truth are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402

from evd.exp.localization_auto import in_shard, read_json, write_json  # noqa: E402

FRACTIONS = tuple(round(f, 2) for f in np.arange(0.10, 0.951, 0.05))
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
FINE_FRACTIONS = tuple(round(f / 100, 2) for f in range(1, 101))
HISTOGRAM_BINS = np.linspace(0.0, 1.0, 51)


def build_plan(data_root: Path, train_plan: Path) -> dict:
    """:return: train A2 positives and validation positives, each row tagged with its split."""
    from localization_auto_train_plan import validation_positives

    train = [{**row, "split": "train"} for row in read_json(train_plan)["positive"]]
    by_class, _ = validation_positives(data_root)
    val = [{**row, "split": "val"} for rows in by_class.values() for row in rows]
    val.sort(key=lambda row: (row["video_id"], row["window"]["end_frame"]))
    plan = {
        "train": train,
        "val": val,
        "train_plan_sha256": read_json(train_plan)["sha256"],
        "rule": {
            "train": "positive chunks of the frozen A2 sample (one per train event)",
            "val": "every chunk of an anomalous validation video intersecting an event",
        },
    }
    body = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    plan["sha256"] = hashlib.sha256(body.encode()).hexdigest()
    return plan


def concept_masses(
    ig: np.ndarray, labels: np.ndarray, ids: np.ndarray, concepts: int
) -> dict[str, list]:
    """Positive and signed IG mass and token count of each concept of a chunk.

    :param ig: ``T×H×W`` IG ensemble map.
    :param labels: ``T×H×W`` SLIC region of each token.
    :param ids: concept id of each region.
    :return: three lists of length ``concepts``.
    :raises ValueError: if the map and the regions disagree on the grid, or the map is not
        finite.
    """
    if ig.shape != labels.shape:
        raise ValueError(f"IG map {ig.shape} and SLIC labels {labels.shape} differ")
    if not np.isfinite(ig).all():
        raise ValueError("Non-finite IG map")
    of_token = ids[labels].ravel()
    positive = np.bincount(of_token, weights=np.maximum(ig, 0).ravel(), minlength=concepts)
    signed = np.bincount(of_token, weights=ig.ravel(), minlength=concepts)
    tokens = np.bincount(of_token, minlength=concepts)
    return {
        "positive_mass": positive.tolist(),
        "signed_mass": signed.tolist(),
        "tokens": tokens.tolist(),
    }


def compute(args) -> None:
    import torch

    from evd.data.dataset import WindowItem
    from evd.exp.demo import live_engine, open_run
    from evd.exp.explanation_advanced import prepare_clip, window_from_record
    from evd.exp.localization_ig import IG_BATCH, IG_FILLS, IG_STEPS, IG_TOLERANCE, explain_ig
    from evd.exp.localization_vtcd import CONCEPTS, REGIONS, chunk_concepts, segment
    from evd.utils.hashing import file_sha256
    from localization_auto_preflight import CHECKPOINT_SHA, RUN

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no silent CPU fallback")
    torch.manual_seed(42)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan = read_json(args.plan)
    paths = {}
    for name in ("inputs-ucf-train.json", "inputs-ucf-val.json"):
        paths.update({row["video_id"]: row["path"] for row in read_json(args.preflight / name)})
    run = open_run(Path(os.environ["EVD_RUNS_DIR"]) / RUN)
    if file_sha256(run.checkpoint_path) != CHECKPOINT_SHA:
        raise ValueError("Selected checkpoint identity mismatch")
    dictionary = np.load(args.fits / "vtcd_models.npz")["global_dictionary"]
    identity = {
        "purpose": "p07_concept_mass_train_val",
        "plan_sha256": plan["sha256"],
        "checkpoint_sha256": CHECKPOINT_SHA,
        "vtcd_models_sha256": file_sha256(args.fits / "vtcd_models.npz"),
        "source_commit": args.source_commit,
        "ig": {
            "steps": list(IG_STEPS),
            "fills": list(IG_FILLS),
            "batch": IG_BATCH,
            "tolerance": IG_TOLERANCE,
        },
        "vtcd": {"regions": REGIONS, "concepts": CONCEPTS},
    }
    engine, spec = live_engine(run, device="cuda", batch_size=1, num_workers=0)
    rows = [row for split in ("train", "val") for row in plan[split]]
    mine = sorted(
        (r for r in rows if in_shard(r["video_id"] + r["window_id"], args.shard, args.shards)),
        key=lambda r: (r["split"], r["window_id"]),
    )

    def paths_of(row: dict) -> tuple[Path, Path]:
        key = hashlib.sha256(f"{row['split']}:{row['window_id']}".encode()).hexdigest()[:24]
        return args.output / "chunks" / f"{key}.json", args.output / "maps" / f"{key}.npz"

    def finish(row: dict, encoded: dict, segmented) -> None:
        labels, descriptors, counts, slic_s = segmented
        start = time.perf_counter()
        ids = chunk_concepts(descriptors, counts, dictionary)["ids"]
        masses = concept_masses(encoded["ig"], labels, ids, CONCEPTS)
        record_path, map_path = paths_of(row)
        map_path.parent.mkdir(parents=True, exist_ok=True)
        with map_path.with_suffix(".tmp").open("wb") as stream:
            np.savez_compressed(
                stream,
                ig_ensemble=encoded["ig"].astype(np.float16),
                concept=ids[labels].astype(np.uint8),
            )
        map_path.with_suffix(".tmp").replace(map_path)
        write_json(
            record_path,
            {
                "identity": identity,
                "split": row["split"],
                "video_id": row["video_id"],
                "window_id": row["window_id"],
                "class_name": row["class_name"],
                "start": row["start"],
                "end": row["end"],
                "logit": encoded["logit"],
                **masses,
                "regions": int(len(descriptors)),
                "converged": encoded["converged"],
                "steps": encoded["steps"],
                "seconds": {
                    "decode": encoded["decode_s"],
                    "ig_and_forward": encoded["ig_s"],
                    "slic_cpu": slic_s,
                    "assign": time.perf_counter() - start,
                },
                "gpu": encoded["gpu"],
            },
        )

    pool = ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    )
    pending: list[tuple[dict, dict, object]] = []
    done = 0
    for row in mine:
        record_path, map_path = paths_of(row)
        if record_path.exists() and map_path.exists():
            if read_json(record_path)["identity"] != identity:
                raise ValueError(f"Existing record with another identity: {record_path}")
            continue
        torch.cuda.synchronize()
        start = time.perf_counter()
        clip = prepare_clip(engine, WindowItem(window_from_record(row), paths[row["video_id"]]))
        torch.cuda.synchronize()
        decoded = time.perf_counter()
        maps, checks, feats, logit, _ = explain_ig(engine.scorer, spec, clip, row["window_id"])
        torch.cuda.synchronize()
        tokens = feats.tokens[0].float().cpu().numpy().reshape(*feats.grid, -1)
        encoded = {
            "ig": np.asarray(maps["ig_ensemble"], dtype=np.float32),
            "logit": logit,
            "converged": {f: checks[f"ig_{f}"]["converged"] for f in IG_FILLS},
            "steps": {f: checks[f"ig_{f}"]["steps"] for f in IG_FILLS},
            "decode_s": decoded - start,
            "ig_s": time.perf_counter() - decoded,
            "gpu": torch.cuda.get_device_name(),
        }
        pending.append((row, encoded, pool.submit(segment, tokens)))
        del clip, feats, tokens
        # SLIC runs on CPU while the GPU explains the next chunks; bound the queue (37 MB each).
        while len(pending) > 2 * args.workers or (pending and pending[0][2].done()):
            first_row, first_encoded, future = pending.pop(0)
            finish(first_row, first_encoded, future.result())
            done += 1
            if done % 25 == 0:
                print(json.dumps({"done": done, "of": len(mine)}), flush=True)
    for row, encoded, future in pending:
        finish(row, encoded, future.result())
    pool.shutdown()
    write_json(
        args.output / "operations" / f"shard{args.shard:02d}.json",
        {"status": "complete", "identity": identity, "chunks": len(mine)},
    )


def prefix_selection(masses: np.ndarray, fraction: float) -> np.ndarray:
    """:return: boolean concepts of the P07 prefix (same rule as ``mass_prefix_concepts``)."""
    chosen = np.zeros(len(masses), dtype=bool)
    if masses.sum() <= 0:
        return chosen
    order = np.lexsort((np.arange(len(masses)), -masses))
    cumulative = np.cumsum(masses[order])
    count = int(np.searchsorted(cumulative, fraction * cumulative[-1], side="left")) + 1
    chosen[[k for k in order[:count] if masses[k] > 0]] = True
    return chosen


def chunk_statistics(record: dict) -> dict:
    """:return: the per-chunk quantities the report summarizes."""
    masses = np.asarray(record["positive_mass"], dtype=np.float64)
    tokens = np.asarray(record["tokens"], dtype=np.float64)
    total = masses.sum()
    stats = {"total_positive_mass": float(total), "present": int((tokens > 0).sum())}
    if total <= 0:
        return {**stats, "empty": True}
    shares = masses / total
    area = tokens / tokens.sum()
    order = np.lexsort((np.arange(len(masses)), -masses))
    cumulative = np.cumsum(shares[order])
    enriched = (shares > area) & (tokens > 0)
    selections = {f: prefix_selection(masses, f) for f in FRACTIONS}
    fine = {f: prefix_selection(masses, f) for f in FINE_FRACTIONS}
    # Cumulative share at which the prefix first holds every enriched concept: the fraction a
    # fixed rule would need on this chunk to keep what the parameter-free rule keeps.
    last_enriched = np.flatnonzero(enriched[order])
    stats.update(
        empty=False,
        sorted_shares=shares[order].tolist(),
        cumulative=cumulative.tolist(),
        selected={f: int(chosen.sum()) for f, chosen in selections.items()},
        area={f: float(area[chosen].sum()) for f, chosen in selections.items()},
        fine_selected={f: int(chosen.sum()) for f, chosen in fine.items()},
        fine_tokens={f: int(tokens[chosen].sum()) for f, chosen in fine.items()},
        enriched_count=int(enriched.sum()),
        enriched_mass=float(shares[enriched].sum()),
        enriched_area=float(area[enriched].sum()),
        enriched_equivalent_fraction=float(cumulative[last_enriched.max()])
        if len(last_enriched)
        else 0.0,
    )
    return stats


def quantiles(values: list[float]) -> dict[str, float]:
    """:return: the report quantiles, or an empty dict without values."""
    if not values:
        return {}
    return {f"q{int(q * 100):02d}": float(np.quantile(values, q)) for q in QUANTILES}


def histogram(values: list[float]) -> list[int]:
    """:return: counts over ``HISTOGRAM_BINS`` (50 bins of width 0.02 on [0, 1])."""
    return np.histogram(np.clip(values, 0.0, 1.0), bins=HISTOGRAM_BINS)[0].tolist()


def fine_csv(populations: dict) -> str:
    """:return: tokens and concepts kept at every 1% of fraction, one row per split."""
    ranks = range(1, 9)
    head = ["split", "fraction", "mean_concepts", *(f"share_k{k}" for k in ranks)]
    head += ["mean_tokens", *(f"tokens_q{int(q * 100):02d}" for q in QUANTILES)]
    lines = [",".join(head)]
    for split in ("train", "val"):
        for f, row in populations.get(split, {}).get("fine", {}).items():
            cells = [split, f, f"{row['mean_concepts']:.4f}"]
            cells += [f"{row['concepts_share'].get(str(k), 0.0):.4f}" for k in ranks]
            cells += [f"{row['mean_tokens']:.1f}"]
            cells += [f"{row['tokens'][f'q{int(q * 100):02d}']:.0f}" for q in QUANTILES]
            lines.append(",".join(cells))
    return "\n".join(lines) + "\n"


def summarize(stats: list[dict]) -> dict:
    """:return: the distributions of one population of chunks."""
    full = [s for s in stats if not s["empty"]]
    summary = {
        "chunks": len(stats),
        "empty_positive_mass": len(stats) - len(full),
        "present_concepts": quantiles([s["present"] for s in stats]),
    }
    if not full:
        return summary
    ranks = range(len(full[0]["sorted_shares"]))
    summary["share_of_rank"] = {
        str(rank + 1): quantiles([s["sorted_shares"][rank] for s in full]) for rank in ranks
    }
    summary["cumulative_of_rank"] = {
        str(rank + 1): quantiles([s["cumulative"][rank] for s in full]) for rank in ranks
    }
    summary["by_fraction"] = {
        f"{f:.2f}": {
            "concepts": quantiles([s["selected"][f] for s in full]),
            "mean_concepts": float(np.mean([s["selected"][f] for s in full])),
            "area": quantiles([s["area"][f] for s in full]),
        }
        for f in FRACTIONS
    }
    summary["fine"] = {
        f"{f:.2f}": {
            "tokens": quantiles([s["fine_tokens"][f] for s in full]),
            "mean_tokens": float(np.mean([s["fine_tokens"][f] for s in full])),
            "mean_concepts": float(np.mean([s["fine_selected"][f] for s in full])),
            "concepts_share": {
                str(k): float(np.mean([s["fine_selected"][f] == k for s in full]))
                for k in range(1, len(full[0]["sorted_shares"]) + 1)
            },
        }
        for f in FINE_FRACTIONS
    }
    summary["histograms"] = {
        "edges": HISTOGRAM_BINS.tolist(),
        "top1_share": histogram([s["sorted_shares"][0] for s in full]),
        "top2_cumulative": histogram([s["cumulative"][1] for s in full]),
        "equivalent_fraction": histogram([s["enriched_equivalent_fraction"] for s in full]),
    }
    summary["mass_above_area"] = {
        "concepts": quantiles([s["enriched_count"] for s in full]),
        "mass": quantiles([s["enriched_mass"] for s in full]),
        "area": quantiles([s["enriched_area"] for s in full]),
        "equivalent_fraction": quantiles([s["enriched_equivalent_fraction"] for s in full]),
    }
    return summary


def report(args) -> None:
    records = [read_json(p) for p in sorted((args.output / "chunks").glob("*.json"))]
    plan = read_json(args.plan)
    identities = {json.dumps(r["identity"], sort_keys=True) for r in records}
    expected = len(plan["train"]) + len(plan["val"])
    if len(identities) > 1:
        raise ValueError(f"{len(identities)} identities among the records")
    if len(records) != expected and not args.partial:
        raise ValueError(f"{len(records)} records of {expected}: pass --partial to report anyway")
    stats = defaultdict(list)
    for record in records:
        s = chunk_statistics(record)
        s["video_id"], s["class_name"] = record["video_id"], record["class_name"]
        stats[record["split"]].append(s)
        stats[f"{record['split']}/{record['class_name']}"].append(s)
    converged = sum(all(r["converged"].values()) for r in records)
    out = {
        "records": len(records),
        "expected": expected,
        "identity": json.loads(next(iter(identities))) if identities else None,
        "chunks_all_baselines_converged": converged,
        "fractions": list(FRACTIONS),
        "populations": {name: summarize(group) for name, group in sorted(stats.items())},
    }
    write_json(args.output / "report" / "concept_mass.json", out)
    (args.output / "report" / "concept_mass.md").write_text(markdown(out), encoding="utf-8")
    (args.output / "report" / "tokens_by_fraction.csv").write_text(
        fine_csv(out["populations"]), encoding="utf-8"
    )
    print(json.dumps({k: out[k] for k in ("records", "expected")}), flush=True)


def _iqr(q: dict[str, float], digits: int = 2) -> str:
    """:return: ``median [q25–q75]``, or a dash without values."""
    if not q:
        return "–"
    return f"{q['q50']:.{digits}f} [{q['q25']:.{digits}f}–{q['q75']:.{digits}f}]"


def markdown(out: dict) -> str:
    """:return: the report tables, train and validation side by side."""
    pops = out["populations"]
    splits = [s for s in ("train", "val") if s in pops]
    head = "| | " + " | ".join(splits) + " |\n|---|" + "---|" * len(splits)
    lines = [
        "# Positive IG mass of the VTCD concepts (choice of the P07 fraction)",
        "",
        f"Records {out['records']} of {out['expected']}; all four IG baselines converged on "
        f"{out['chunks_all_baselines_converged']}. Values: median [q25–q75] over chunks.",
        "",
        "## Chunks",
        "",
        head,
    ]
    lines.append("| chunks | " + " | ".join(str(pops[s]["chunks"]) for s in splits) + " |")
    lines.append(
        "| no positive mass | "
        + " | ".join(str(pops[s]["empty_positive_mass"]) for s in splits)
        + " |"
    )
    lines.append(
        "| concepts present | "
        + " | ".join(_iqr(pops[s]["present_concepts"], 0) for s in splits)
        + " |"
    )
    for title, key in (
        ("Share of the k-th concept by mass", "share_of_rank"),
        ("Cumulative share of the first k concepts", "cumulative_of_rank"),
    ):
        lines += ["", f"## {title}", "", head.replace("| |", "| k |", 1)]
        for rank in pops[splits[0]].get(key, {}):
            lines.append(
                f"| {rank} | "
                + " | ".join(_iqr(pops[s].get(key, {}).get(rank, {})) for s in splits)
                + " |"
            )
    lines += [
        "",
        "## P07 at each fraction: concepts kept (median, mean) and token area",
        "",
        "| fraction | "
        + " | ".join(f"{s} concepts | {s} area" for s in splits)
        + " |\n|---|"
        + "---|---|" * len(splits),
    ]
    for f in out["fractions"]:
        cells = []
        for s in splits:
            row = pops[s].get("by_fraction", {}).get(f"{f:.2f}")
            cells += (
                [f"{row['concepts']['q50']:.0f} ({row['mean_concepts']:.2f})", _iqr(row["area"])]
                if row
                else ["–", "–"]
            )
        lines.append(f"| {f:.2f} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Concepts whose mass share exceeds their token share (no fraction to fix)",
        "",
        head,
    ]
    for label, key, digits in (
        ("concepts", "concepts", 0),
        ("mass kept", "mass", 2),
        ("area kept", "area", 2),
        ("equivalent P07 fraction", "equivalent_fraction", 2),
    ):
        lines.append(
            f"| {label} | "
            + " | ".join(
                _iqr(pops[s].get("mass_above_area", {}).get(key, {}), digits) for s in splits
            )
            + " |"
        )
    lines += [
        "",
        "## Per class",
        "",
        "| split / class | chunks | top-1 share | top-2 cumulative | equivalent fraction |",
        "|---|---|---|---|---|",
    ]
    for name, pop in pops.items():
        if "/" not in name:
            continue
        lines.append(
            f"| {name} | {pop['chunks']} | "
            + " | ".join(
                _iqr(q)
                for q in (
                    pop.get("share_of_rank", {}).get("1", {}),
                    pop.get("cumulative_of_rank", {}).get("2", {}),
                    pop.get("mass_above_area", {}).get("equivalent_fraction", {}),
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=("plan", "compute", "report"))
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "data")
    parser.add_argument("--train-plan", type=Path, help="frozen train-plan.json (plan)")
    parser.add_argument("--preflight", type=Path, help="operations/preflight (compute)")
    parser.add_argument("--fits", type=Path, help="fits folder with vtcd_models.npz (compute)")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--workers", type=int, default=6, help="SLIC processes (compute)")
    parser.add_argument("--source-commit", default="")
    parser.add_argument("--partial", action="store_true", help="report on incomplete records")
    args = parser.parse_args()
    if args.action == "plan":
        plan = build_plan(args.data_root, args.train_plan)
        write_json(args.plan, plan)
        print(json.dumps({"train": len(plan["train"]), "val": len(plan["val"])}), flush=True)
    else:
        (compute if args.action == "compute" else report)(args)


if __name__ == "__main__":
    main()
