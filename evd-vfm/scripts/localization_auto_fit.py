"""CPU fits of the automatic-localization campaign from the frozen train sample.

``a2`` reads the first-pass train features (``localization_auto_features.py a2``) and writes
to ``fits/``: the 32 channels of P45/P46 (largest mean spatial variance), the train
standardization and PCA-128 of normal tokens, the binary references of P08 and P27
(``evd.explain.concepts.fit_binary_references``), the P19 normal reference window, the
P24 token bank, the P26 diagonal Gaussian and the P28 transition bank. Each artifact
records its fit time. Only train features are read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402

from evd.exp.localization_auto import read_json, write_json  # noqa: E402
from evd.explain.region_attribution import top_variance_channels  # noqa: E402

SEED = 42
PCA_DIM = 128
BANK = 25_000


def load_a2(features: Path) -> list[dict]:
    """:return: one record per extracted chunk (arrays and receipt)."""
    records = []
    for receipt in sorted((features / "a2" / "receipts").glob("*.json")):
        meta = read_json(receipt)
        with np.load(features / "a2" / "chunks" / f"{receipt.stem}.npz") as arrays:
            records.append({**meta, **{k: arrays[k] for k in arrays.files if k != "tokens"}})
    return records


def quotas(capacity: dict[str, int], total: int) -> dict[str, int]:
    """Split ``total`` as equally as possible over videos, within each video's capacity.

    Water filling: a video with fewer rows than the equal share gives all of them and the
    rest is split again among the others; remainders go to the first videos in name order.

    :raises ValueError: if the videos hold fewer than ``total`` rows altogether.
    """
    if sum(capacity.values()) < total:
        raise ValueError(f"Only {sum(capacity.values())} rows for a sample of {total}")
    quota = dict.fromkeys(capacity, 0)
    left, open_ = total, sorted(capacity)
    while left:
        base, extra = divmod(left, len(open_))
        for i, video in enumerate(list(open_)):
            add = min(base + (i < extra), capacity[video] - quota[video])
            quota[video] += add
            left -= add
        open_ = [v for v in open_ if quota[v] < capacity[v]]
    return quota


def equal_per_video(groups: dict[str, np.ndarray], total: int, seed: int) -> np.ndarray:
    """Draw ``total`` rows with an equal quota per video, capped by what each video holds.

    :param groups: video → rows available.
    :return: stacked sample, videos in name order.
    """
    rng = np.random.default_rng(seed)
    quota = quotas({v: len(rows) for v, rows in groups.items()}, total)
    return np.concatenate(
        [
            groups[v][rng.choice(len(groups[v]), size=quota[v], replace=False)]
            for v in sorted(groups)
        ]
    )


def fit_pca(x: np.ndarray, dim: int) -> dict[str, np.ndarray]:
    """:return: train standardization and the top ``dim`` principal axes (rows)."""
    mean = x.mean(0)
    scale = x.std(0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    z = (x - mean) / scale
    _, values, vectors = np.linalg.svd(z - z.mean(0), full_matrices=False)
    explained = values**2 / (values**2).sum()
    return {
        "mean": mean,
        "scale": scale,
        "components": vectors[:dim],
        "explained": explained[:dim],
    }


def project(x: np.ndarray, pca: dict) -> np.ndarray:
    """:return: standardized tokens on the principal axes."""
    return ((x - pca["mean"]) / pca["scale"]) @ pca["components"].T


def fit_a2(args) -> None:
    from evd.explain.concepts import fit_binary_references

    out = args.output
    records = load_a2(args.features)
    plan = read_json(args.plan)
    expected = {
        r["window_id"] for role in ("positive", "normal", "vtcd_reference") for r in plan[role]
    }
    if {r["row"]["window_id"] for r in records} != expected:
        raise ValueError("The first-pass features do not cover the frozen train sample")
    a2 = [r for r in records if {"positive", "normal"} & set(r["roles"])]
    normal = [r for r in a2 if "normal" in r["roles"]]
    timings = {}

    start = time.perf_counter()
    channels = top_variance_channels(np.stack([r["channel_variance"] for r in a2]), 32)
    timings["channels"] = time.perf_counter() - start
    write_json(
        out / "channels.json",
        {"channels": channels.tolist(), "chunks": len(a2), "fit_seconds": timings["channels"]},
    )

    start = time.perf_counter()
    tokens = np.concatenate([r["pair_tokens"].astype(np.float64) for r in normal])
    pca = fit_pca(tokens, PCA_DIM)
    timings["pca"] = time.perf_counter() - start
    np.savez(out / "pca.npz", **pca)

    start = time.perf_counter()
    labels = np.array([int("positive" in r["roles"]) for r in a2])
    references, receipt = fit_binary_references(
        np.stack([r["mean"] for r in a2]), labels, seed=SEED, c=1.0
    )
    timings["binary_references"] = time.perf_counter() - start
    np.savez(out / "binary_references.npz", **references)
    write_json(
        out / "binary_references.json", {**receipt, "fit_seconds": timings["binary_references"]}
    )

    start = time.perf_counter()
    by_video = defaultdict(list)
    for r in normal:
        mean = r["mean"].astype(np.float64)
        by_video[r["row"]["video_id"]].append((mean / np.linalg.norm(mean), r["row"]))
    center = np.mean([np.mean([m for m, _ in items], 0) for items in by_video.values()], 0)
    candidates = [
        (float(((m - center) ** 2).sum()), row["window_id"], row)
        for items in by_video.values()
        for m, row in items
    ]
    distance, window_id, row = min(candidates, key=lambda c: (c[0], c[1]))
    timings["p19_reference"] = time.perf_counter() - start
    write_json(
        out / "p19_reference.json",
        {
            "row": row,
            "distance": distance,
            "rule": "normalized mean descriptor nearest to the video-weighted mean",
            "fit_seconds": timings["p19_reference"],
        },
    )

    start = time.perf_counter()
    projected = {}
    for r in normal:
        z = project(r["pair_tokens"].astype(np.float64), pca)
        projected.setdefault(r["row"]["video_id"], []).append(z)
    projected = {v: np.concatenate(z) for v, z in projected.items()}
    bank = equal_per_video(projected, BANK, SEED)
    timings["p24_bank"] = time.perf_counter() - start + timings["pca"]
    np.save(out / "p24_bank.npy", bank.astype(np.float32))

    start = time.perf_counter()
    allz = np.concatenate(list(projected.values()))
    gaussian = {"mean": allz.mean(0), "var": np.maximum(allz.var(0), 1e-6)}
    timings["p26_gaussian"] = time.perf_counter() - start + timings["pca"]
    np.savez(out / "p26_gaussian.npz", **gaussian)

    write_json(
        out / "a2-fits.json",
        {
            "fit_seconds": timings,
            "normal_chunks": len(normal),
            "a2_chunks": len(a2),
            "normal_videos": len(projected),
            "pca_explained_128": float(pca["explained"].sum()),
            "plan_sha256": plan["sha256"],
        },
    )
    print(json.dumps({"fit_seconds": timings}), flush=True)


def load_projected(features: Path) -> dict[str, list[dict]]:
    """:return: second-pass records grouped by role (train/val, bag/normal)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for receipt in sorted((features / "project" / "receipts").glob("*.json")):
        meta = read_json(receipt)
        for role in meta["roles"]:
            groups[role].append(
                {**meta, "path": features / "project" / "chunks" / f"{receipt.stem}.npz"}
            )
    return groups


def learn(args) -> None:
    """Second-pass fits, trainings and the A1 calibration of the memory family (GPU job)."""
    import torch

    from evd.exp import localization_memory as lm
    from evd.exp.localization_auto import weighted_quantile

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out, plan = args.output, read_json(args.plan)
    validation = read_json(args.val_plan)
    a2 = [r for r in load_a2(args.features) if "normal" in r["roles"]]
    pca = dict(np.load(out / "pca.npz"))
    groups = load_projected(args.features)
    expected = {
        "train_bag": len(plan["bag"]),
        "train_normal": len(plan["normal"]),
        "val_normal": len(validation["normal"]),
        "val_bag": len(validation["bag"]),
    }
    found = {k: len(groups[k]) for k in expected}
    if found != expected:
        raise ValueError(f"Second pass incomplete: {found} instead of {expected}")
    extraction = {role: sum(r["seconds"] for r in rows) for role, rows in groups.items()}
    a2_extraction = sum(r["seconds"] for r in a2)
    timings = read_json(out / "a2-fits.json")["fit_seconds"]

    start = time.perf_counter()
    by_video = defaultdict(list)
    for r in a2:
        z = project(r["pair_tokens"].astype(np.float64), pca).reshape(-1, 2, PCA_DIM)
        by_video[r["row"]["video_id"]].append(np.hstack([z[:, 0], z[:, 1] - z[:, 0]]))
    pairs = {v: np.concatenate(p) for v, p in by_video.items()}
    pair_pca = fit_pca(np.concatenate(list(pairs.values())), PCA_DIM)
    bank = equal_per_video({v: project(p, pair_pca) for v, p in pairs.items()}, BANK, SEED)
    np.savez(out / "p28_pca.npz", **pair_pca)
    np.save(out / "p28_bank.npy", bank.astype(np.float32))
    timings["p28_transitions"] = time.perf_counter() - start

    start = time.perf_counter()
    contexts = sorted(groups["train_normal"], key=lambda r: r["row"]["window_id"])
    means, tokens = [], []
    for r in contexts:
        with np.load(r["path"]) as arrays:
            means.append(arrays["mean"])
            tokens.append(arrays["tokens_pca"])
    np.savez(out / "p25_contexts.npz", means=np.stack(means), tokens=np.stack(tokens))
    timings["p25_contexts"] = time.perf_counter() - start

    train = torch.as_tensor(
        np.concatenate([project(r["pair_tokens"].astype(np.float64), pca) for r in a2]),
        dtype=torch.float32,
    )
    rng = np.random.default_rng(SEED)
    val_tokens = []
    for r in groups["val_normal"]:
        with np.load(r["path"]) as arrays:
            z = arrays["tokens_pca"]
        val_tokens.append(z[rng.choice(len(z), size=64, replace=False)].astype(np.float32))
    val = torch.as_tensor(np.concatenate(val_tokens))
    receipts = {}
    for name, model in (("p29", lm.MemAE()), ("p30", lm.Student())):
        state, receipts[name] = lm.train_reconstruction(model.to(device), train, val)
        torch.save(state, out / f"{name}.pt")

    label = {r["window_id"]: r["bag_label"] for r in plan["bag"]}
    label.update({r["window_id"]: r["bag_label"] for r in validation["bag"]})

    def bags(rows):
        videos = defaultdict(list)
        for r in rows:
            videos[r["row"]["video_id"]].append(r)
        result = []
        for vid in sorted(videos):
            chunks = []
            for r in sorted(videos[vid], key=lambda r: r["row"]["window_id"]):
                with np.load(r["path"]) as arrays:
                    chunks.append(torch.as_tensor(arrays["tokens_pca"]))
            labels = {label[r["row"]["window_id"]] for r in videos[vid]}
            if len(labels) != 1:
                raise ValueError(f"Inconsistent bag label: {vid}")
            result.append((torch.stack(chunks).to(device), labels.pop()))
        return result

    start = time.perf_counter()
    train_bags, val_bags = bags(groups["train_bag"]), bags(groups["val_bag"])
    loading = time.perf_counter() - start
    state, receipts["p32"] = lm.train_mil(train_bags, val_bags, device)
    receipts["p32"]["loading_seconds"] = loading
    torch.save(state, out / "p32.pt")
    del train_bags, val_bags
    if device == "cuda":
        torch.cuda.empty_cache()

    start = time.perf_counter()
    models = lm.MemoryModels(out, device)
    maxima = {name: defaultdict(list) for name in lm.ABSOLUTE}
    for r in groups["val_normal"]:
        with np.load(r["path"]) as arrays:
            z = torch.as_tensor(arrays["tokens_pca"], dtype=torch.float32, device=device)
            mean = torch.as_tensor(arrays["mean"], device=device)
            grid = tuple(int(g) for g in arrays["grid"])
        for name, values in models.absolute_maps(z, mean, grid).items():
            maxima[name][r["row"]["video_id"]].append(float(values.max()))
    calibration = time.perf_counter() - start
    thresholds = {
        name: {
            "threshold": weighted_quantile(dict(values), 0.99),
            "quantile": 0.99,
            "statistic": "max of the map per chunk",
            "normal_videos": len(values),
            "normal_chunks": sum(map(len, values.values())),
        }
        for name, values in maxima.items()
    }
    write_json(out / "memory_thresholds.json", thresholds)

    shared_calibration = extraction["val_normal"] + calibration
    pca_cost = a2_extraction + timings["pca"]
    fits = {
        "P08-projection": a2_extraction + timings["binary_references"],
        "P27-centroids": a2_extraction + timings["binary_references"],
        "P19-one_representative": a2_extraction + timings["p19_reference"],
        "P24-bank25000": pca_cost + timings["p24_bank"] + shared_calibration,
        "P25-contexts5": pca_cost
        + extraction["train_normal"]
        + timings["p25_contexts"]
        + shared_calibration,
        "P26-gaussian1": pca_cost + timings["p26_gaussian"] + shared_calibration,
        "P28-transitions": pca_cost + timings["p28_transitions"] + shared_calibration,
        "P29-slots128": pca_cost + receipts["p29"]["fit_seconds"] + shared_calibration,
        "P30-linear": pca_cost + receipts["p30"]["fit_seconds"] + shared_calibration,
        "P32-top10": pca_cost
        + extraction["train_bag"]
        + extraction["val_bag"]
        + receipts["p32"]["fit_seconds"]
        + loading
        + shared_calibration,
    }
    write_json(out / "memory_fit_seconds.json", fits)
    manifest = {row["video_id"]: row["path"] for row in read_json(args.train_manifest)}
    reference = read_json(out / "p19_reference.json")["row"]
    write_json(out / "train_paths.json", {reference["video_id"]: manifest[reference["video_id"]]})
    write_json(
        out / "learn.json",
        {
            "receipts": receipts,
            "extraction_seconds": extraction,
            "a2_extraction_seconds": a2_extraction,
            "calibration_seconds": calibration,
            "fit_seconds": fits,
            "thresholds": {k: v["threshold"] for k, v in thresholds.items()},
            "device": device,
        },
    )
    print(json.dumps({"thresholds": {k: v["threshold"] for k, v in thresholds.items()}}))


def fit_vtcd(args) -> None:
    """VTCD dictionaries, A2 assignments and the P05, P06, P10 selections (GPU + CPU pool)."""
    import multiprocessing
    import os
    from concurrent.futures import ProcessPoolExecutor

    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from evd.data.dataset import WindowItem
    from evd.exp import localization_vtcd as lv
    from evd.exp.demo import live_engine, open_run
    from evd.exp.explanation_advanced import extract_features, prepare_clip, window_from_record
    from evd.utils.hashing import file_sha256
    from localization_auto_preflight import CHECKPOINT_SHA, RUN

    out, plan = args.output, read_json(args.plan)
    records = {r["row"]["window_id"]: r for r in load_a2(args.features)}
    pool = ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    )
    timings = {}

    refs = plan["vtcd_reference"]
    start = time.perf_counter()
    ref_tokens = []
    for row in refs:
        with np.load(args.features / "a2" / "chunks" / f"{_key(row['window_id'])}.npz") as arrays:
            grid = tuple(int(g) for g in arrays["grid"])
            ref_tokens.append(arrays["tokens"].astype(np.float32).reshape(*grid, -1))
    ref_segments = list(pool.map(lv.segment, ref_tokens))
    del ref_tokens
    ref_extraction = sum(records[row["window_id"]]["seconds"] for row in refs)
    ref_slic = sum(seg[3] for seg in ref_segments)
    timings["reference_slic_wall"] = time.perf_counter() - start

    start = time.perf_counter()
    dictionary, global_receipt = lv.fit_dictionary(np.concatenate([seg[1] for seg in ref_segments]))
    global_fit = time.perf_counter() - start
    names = sorted({row["class_name"] for row in refs})
    class_dictionaries, class_receipts, class_fit = [], {}, 0.0
    for name in names:
        start = time.perf_counter()
        descriptors = np.concatenate(
            [
                seg[1]
                for seg, row in zip(ref_segments, refs, strict=True)
                if row["class_name"] == name
            ]
        )
        values, class_receipts[name] = lv.fit_dictionary(descriptors)
        class_dictionaries.append(values)
        class_fit += time.perf_counter() - start

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no silent CPU fallback")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    run = open_run(Path(os.environ["EVD_RUNS_DIR"]) / RUN)
    if file_sha256(run.checkpoint_path) != CHECKPOINT_SHA:
        raise ValueError("Selected checkpoint identity mismatch")
    engine, _ = live_engine(run, device="cuda", batch_size=1, num_workers=0)
    paths = {row["video_id"]: row["path"] for row in read_json(args.train_manifest)}
    rows = [(row, 1) for row in plan["positive"]] + [(row, 0) for row in plan["normal"]]
    encode_seconds, futures = 0.0, []
    for row, _ in rows:
        begin = time.perf_counter()
        clip = prepare_clip(engine, WindowItem(window_from_record(row), paths[row["video_id"]]))
        feats, _ = extract_features(engine, clip)
        torch.cuda.synchronize()
        encode_seconds += time.perf_counter() - begin
        futures.append(
            pool.submit(lv.segment, feats.tokens[0].float().cpu().numpy().reshape(*feats.grid, -1))
        )
        del clip, feats
    segments = [f.result() for f in futures]
    pool.shutdown()
    a2_slic = sum(seg[3] for seg in segments)

    start = time.perf_counter()
    assigned = [lv.chunk_concepts(seg[1], seg[2], dictionary) for seg in segments]
    a2_nnls = time.perf_counter() - start
    positive = np.array([flag for _, flag in rows], dtype=bool)
    videos = [row["video_id"] for row, _ in rows]

    start = time.perf_counter()
    presence = np.stack([a["presence"] for a in assigned])
    ratios = lv.enrichment(presence, positive, videos)
    best = int(np.argmax(ratios))
    p05_concept = best if ratios[best] > 0 else -1
    p05_fit = time.perf_counter() - start

    start = time.perf_counter()
    means = np.stack([a["mean"] for a in assigned])
    scaler = StandardScaler().fit(means)
    classifier = LogisticRegression(penalty="l1", C=1.0, solver="liblinear", random_state=SEED)
    classifier.fit(scaler.transform(means), positive.astype(int))
    p06_weights = np.maximum(classifier.coef_[0], 0) / scaler.scale_
    p06_fit = time.perf_counter() - start

    start = time.perf_counter()
    p10_concepts, p10_ratios = [], {}
    for name, values in zip(names, class_dictionaries, strict=True):
        keep = [i for i, (row, flag) in enumerate(rows) if not flag or row["class_name"] == name]
        class_presence = np.stack(
            [lv.chunk_concepts(segments[i][1], segments[i][2], values)["presence"] for i in keep]
        )
        ratio = lv.enrichment(class_presence, positive[keep], [videos[i] for i in keep])
        p10_ratios[name] = ratio.tolist()
        top = int(np.argmax(ratio))
        p10_concepts.append(top if ratio[top] > 0 else -1)
    mean_tokens = np.stack([records[row["window_id"]]["mean"] for row, _ in rows])
    labels = [row["class_name"] if flag else "Normal" for row, flag in rows]
    router_scaler = StandardScaler().fit(mean_tokens)
    router = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    router.fit(router_scaler.transform(mean_tokens), labels)
    p10_fit = time.perf_counter() - start

    np.savez(
        out / "vtcd_models.npz",
        global_dictionary=dictionary,
        class_dictionaries=np.stack(class_dictionaries),
        class_names=np.array(names),
        p05_concept=np.array(p05_concept),
        p06_token_weights=p06_weights,
        p10_concepts=np.array(p10_concepts),
        router_mean=router_scaler.mean_,
        router_scale=router_scaler.scale_,
        router_coef=router.coef_,
        router_intercept=router.intercept_,
        router_classes=np.array(router.classes_),
    )
    dictionary_cost = ref_extraction + ref_slic + global_fit
    assignment_cost = encode_seconds + a2_slic + a2_nnls
    fits = {
        "P04-top1": dictionary_cost,
        "P07-mass70": dictionary_cost,
        "P05-presence": dictionary_cost + assignment_cost + p05_fit,
        "P06-soft": dictionary_cost + assignment_cost + p06_fit,
        "P10-soft": ref_extraction + ref_slic + class_fit + encode_seconds + a2_slic + p10_fit,
    }
    write_json(out / "vtcd_fit_seconds.json", fits)
    write_json(
        out / "vtcd_fit.json",
        {
            "global": global_receipt,
            "classes": class_receipts,
            "references": len(refs),
            "a2_chunks": len(rows),
            "p05_ratios": ratios.tolist(),
            "p05_concept": p05_concept,
            "p06_coefficients_standardized": classifier.coef_[0].tolist(),
            "p10_concepts": dict(zip(names, p10_concepts, strict=True)),
            "p10_ratios": p10_ratios,
            "router_training_accuracy": float(
                router.score(router_scaler.transform(mean_tokens), labels)
            ),
            "fit_seconds": fits,
            "timings": {
                "reference_extraction": ref_extraction,
                "reference_slic_cpu": ref_slic,
                "global_fit": global_fit,
                "class_fit": class_fit,
                "a2_encode": encode_seconds,
                "a2_slic_cpu": a2_slic,
                "a2_nnls": a2_nnls,
                **timings,
            },
        },
    )
    print(json.dumps({"p05": p05_concept, "p10": p10_concepts, "fits": fits}), flush=True)


def _key(window_id: str) -> str:
    """:return: the feature file stem of a window (``localization_auto_features``)."""
    import hashlib

    return hashlib.sha256(window_id.encode()).hexdigest()[:24]


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=("a2", "learn", "vtcd"))
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--val-plan", type=Path)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    {"a2": fit_a2, "learn": learn, "vtcd": fit_vtcd}[args.action](args)


if __name__ == "__main__":
    main()
