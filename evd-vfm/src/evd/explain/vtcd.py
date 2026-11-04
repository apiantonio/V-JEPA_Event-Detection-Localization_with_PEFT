"""VTCD on the final features: 3D SLIC regions, Convex NMF and CRIS.

Adaptation of Kowal et al., CVPR 2024, §3.1-3.2 (https://arxiv.org/html/2401.10831v2):
the dictionary uses the training data only and masks act on the final tokens, not on
all heads and layers. The factorization uses the data-sample form of Ding et al. (2010),
Convex NMF: X ~= H W^T X, H,W >= 0, with W columns normalized to sum one. Choices and
diagnostics: docs/varie/interpretabilita_cinque_metodi.md.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import product

import numpy as np
import torch
from scipy import ndimage
from scipy.optimize import nnls
from sklearn.cluster import KMeans
from torch import Tensor


def slic_regions(
    features: np.ndarray,
    *,
    n_segments: int = 32,
    compactness: float = 0.1,
    max_iter: int = 10,
    min_size_factor: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Propose connected regions in T×H×W×D features, signed ones included.

    :param features: finite volume; RMS feature distance, coordinates in [0,1].
    :param n_segments: requested number, before connection and merging.
    :return: T×H×W labels, R×D mean descriptors in original coordinates, and diagnostics.
        Small components merge only with adjacent neighbours.
    :raises ValueError: on an invalid volume or parameters.
    :raises RuntimeError: if SLIC leaves tokens without a region.
    """
    f = np.asarray(features, dtype=np.float32)
    if f.ndim != 4 or not np.isfinite(f).all() or min(f.shape) < 1:
        raise ValueError("SLIC requires a finite T×H×W×D volume")
    shape = np.asarray(f.shape[:3])
    n = int(np.prod(shape))
    if (
        not 1 <= n_segments <= n
        or compactness <= 0
        or max_iter < 1
        or not 0 <= min_size_factor <= 1
    ):
        raise ValueError("Invalid SLIC parameters")
    raw = f.reshape(n, -1)
    rms = float(np.sqrt(np.mean(raw.astype(np.float64) ** 2)))
    x = raw / max(rms, 1e-12)
    positions = np.stack(
        np.meshgrid(*(np.linspace(0, 1, s) for s in shape), indexing="ij"), -1
    ).reshape(n, 3)
    # Regular grid closest to the requested count, on normalized coordinates.
    choices = list(product(*(range(1, min(int(s), n_segments) + 1) for s in shape)))
    counts = np.array(
        min(choices, key=lambda c: (abs(np.prod(c) - n_segments), max(c) / min(c), c))
    )
    axes = [(np.arange(c) + 0.5) / c for c in counts]
    centers_p = np.asarray(list(product(*axes)))
    indices = np.rint(centers_p * (shape - 1)).astype(int)
    centers = x[np.ravel_multi_index(indices.T, tuple(shape))].copy()
    spacing = 1 / counts
    labels = np.full(n, -1, dtype=np.int32)
    for iteration in range(max_iter):  # noqa: B007 - recorded in the receipt after the loop
        distances = np.full(n, np.inf)
        previous = labels.copy()
        for k, (center, position) in enumerate(zip(centers, centers_p, strict=True)):
            local = np.flatnonzero((abs(positions - position) <= spacing * 2).all(1))
            delta = x[local] - center
            d = np.mean(delta * delta, axis=1) + compactness**2 * np.sum(
                ((positions[local] - position) / spacing) ** 2, axis=1
            )
            wins = d < distances[local]
            distances[local[wins]] = d[wins]
            labels[local[wins]] = k
        if np.any(labels < 0):
            raise RuntimeError("SLIC left tokens without a region")
        for k in range(len(centers)):
            selected = labels == k
            if selected.any():
                centers[k] = x[selected].mean(0)
                centers_p[k] = positions[selected].mean(0)
        if np.array_equal(previous, labels):
            break
    volume = labels.reshape(tuple(shape))
    connected = np.full(tuple(shape), -1, dtype=np.int32)
    count = 0
    for k in np.unique(volume):
        parts, number = ndimage.label(volume == k)
        mask = parts > 0
        connected[mask] = parts[mask] + count - 1
        count += number
    premerge = count
    minimum = max(1, int(min_size_factor * n / n_segments))
    # Every merge crosses a border, so connectivity remains an invariant.
    while True:
        ids, sizes = np.unique(connected, return_counts=True)
        small = [(int(size), int(k)) for k, size in zip(ids, sizes, strict=True) if size < minimum]
        if not small or len(ids) == 1:
            break
        _, k = min(small)
        mask = connected == k
        border = ndimage.binary_dilation(mask) & ~mask
        adjacent, contacts = np.unique(connected[border], return_counts=True)
        target = int(adjacent[np.argmax(contacts)])
        connected[mask] = target
    _, relabeled = np.unique(connected, return_inverse=True)
    labels = relabeled.reshape(tuple(shape)).astype(np.int32)
    descriptors = np.stack([raw[labels.ravel() == k].mean(0) for k in range(int(labels.max()) + 1)])
    return (
        labels,
        descriptors,
        {
            "requested": n_segments,
            "initial": len(centers),
            "connected_before_merge": premerge,
            "actual": len(descriptors),
            "iterations": iteration + 1,
            "converged": bool(np.array_equal(previous, volume.ravel())),
            "compactness": compactness,
            "feature_rms": rms,
            "coordinate_range": [0, 1],
            "feature_distance": "mean squared difference / volume RMS squared",
            "min_size": minimum,
            "region_sizes": np.bincount(labels.ravel()).tolist(),
        },
    )


def convex_nmf(
    descriptors: np.ndarray,
    *,
    n_concepts: int = 8,
    seed: int = 42,
    max_iter: int = 500,
    tolerance: float = 1e-5,
) -> tuple[dict[str, np.ndarray], dict]:
    """Factorize signed descriptors as H @ (W.T @ X).

    :param descriptors: R×D, without ReLU, shifts or clipping.
    :return: K×D dictionary, coefficients H, convex weights W and error history.
    :raises ValueError: on invalid data or fewer regions than concepts.
    """
    x = np.asarray(descriptors, dtype=np.float64)
    if (
        x.ndim != 2
        or len(x) < n_concepts
        or not np.isfinite(x).all()
        or n_concepts < 1
        or max_iter < 1
    ):
        raise ValueError("Convex NMF: at least K finite descriptors are required")
    groups = KMeans(n_clusters=n_concepts, random_state=seed, n_init=1).fit_predict(x)
    h = np.eye(n_concepts)[groups] + 0.01
    w = h / h.sum(0)
    gram = x @ x.T
    positive, negative = np.maximum(gram, 0), np.maximum(-gram, 0)
    total = max(float(np.trace(gram)), 1e-20)
    history = []
    converged = False
    for iteration in range(max_iter):
        kp_w, kn_w = positive @ w, negative @ w
        h *= np.sqrt(
            np.maximum(kp_w + h @ (w.T @ kn_w), 1e-20) / np.maximum(kn_w + h @ (w.T @ kp_w), 1e-20)
        )
        hth = h.T @ h
        w *= np.sqrt(
            np.maximum(positive @ h + negative @ w @ hth, 1e-20)
            / np.maximum(negative @ h + positive @ w @ hth, 1e-20)
        )
        sums = w.sum(0)
        w /= sums
        h *= sums
        if iteration % 10 == 0 or iteration == max_iter - 1:
            dictionary = w.T @ x
            residual = float(np.sum((x - h @ dictionary) ** 2) / total)
            history.append({"iteration": iteration + 1, "relative_squared_error": residual})
            if len(history) > 1 and abs(
                history[-2]["relative_squared_error"] - residual
            ) <= tolerance * max(history[-2]["relative_squared_error"], 1e-12):
                converged = True
                break
    dictionary = w.T @ x
    return {"dictionary": dictionary, "convex_weights": w, "training_coefficients": h}, {
        "objective": "||X - H W^T X||_F^2; H,W>=0; sum_rows(W)=1",
        "seed": seed,
        "iterations": iteration + 1,
        "max_iter": max_iter,
        "tolerance": tolerance,
        "converged": converged,
        "history": history,
        "negative_input_fraction": float(np.mean(x < 0)),
        "convex_sum_max_error": float(np.max(abs(w.sum(0) - 1))),
        "training_concept_counts": np.bincount(h.argmax(1), minlength=n_concepts).tolist(),
        "stable_ids": "deterministic fit order; independent across checkpoints",
    }


def assign_concepts(
    descriptors: np.ndarray, dictionary: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Assign new regions with NNLS on a frozen dictionary.

    :return: dominant ID per region and R×K non-negative coefficients; ties go to the
        lowest ID.
    """
    coefficients = np.stack(
        [nnls(dictionary.T, x, maxiter=100 * len(dictionary))[0] for x in descriptors]
    )
    return coefficients.argmax(1), coefficients


def cris(
    forward: Callable[[Tensor], Tensor],
    tokens: Tensor,
    concept_ids: np.ndarray,
    *,
    label: int,
    n_concepts: int = 8,
    n_masks: int = 256,
    probability: float = 0.5,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], dict]:
    """Mask groups of concepts and measure BCE degradation.

    :param forward: head on the final features, in the same regime as the detector.
    :param label: binary temporal label of the window.
    :return: masks, logits, deltas, losses and importance (Eq. 6 with D=-BCE).
    :raises ValueError: on an invalid label, concept IDs or sampling.
    """
    ids = np.asarray(concept_ids).ravel()
    if (
        label not in (0, 1)
        or len(ids) != tokens.shape[1]
        or np.any((ids < 0) | (ids >= n_concepts))
        or n_masks < 1
        or not 0 < probability < 1
    ):
        raise ValueError("CRIS: invalid label, concepts or sampling")
    masks = np.random.default_rng(seed).random((n_masks, n_concepts)) < probability
    index = torch.as_tensor(ids, device=tokens.device, dtype=torch.long)
    logits = []
    with torch.no_grad():
        original = float(forward(tokens).float().item())
        for mask in masks:
            removed = torch.as_tensor(mask, device=tokens.device)[index]
            logits.append(float(forward(tokens * (~removed)[None, :, None]).float().item()))
    logits = np.asarray(logits, dtype=np.float64)
    losses = np.logaddexp(0, logits) - label * logits
    original_loss = float(np.logaddexp(0, original) - label * original)
    degradation = losses - original_loss
    importance = (masks * degradation[:, None]).mean(0)
    present = np.bincount(ids, minlength=n_concepts) > 0
    # Raw CRIS can assign a value to absent concepts through co-masking; the map only keeps
    # IDs actually present.
    return {
        "masks": masks,
        "perturbed_logits": logits,
        "logit_differences": original - logits,
        "perturbed_scores": 1 / (1 + np.exp(-np.clip(logits, -700, 700))),
        "loss_degradation": degradation,
        "importance": importance,
        "present": present,
    }, {
        "seed": seed,
        "n_masks": n_masks,
        "probability": probability,
        "original_logit": original,
        "original_loss": original_loss,
        "formula": "mean_k[mask_ki * (BCE(perturbed_k,y) - BCE(original,y))]",
        "ranking": np.argsort(-importance, kind="stable").tolist(),
        "absent_concepts": np.flatnonzero(~present).tolist(),
        "removed_counts": masks.sum(0).tolist(),
        "mask_fill": "zeroed final tokens; not black pixels",
    }
