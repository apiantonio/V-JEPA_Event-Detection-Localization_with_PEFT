"""Binary directions and prototypes on the final features (explainability campaign).

TCAV-style adaptation of Kim et al., ICML 2018, §3
(https://proceedings.mlr.press/v80/kim18d.html). The separator describes windows, not
spatial annotations, and is not the full TCAV statistical test. Choices and formulas:
docs/varie/interpretabilita_cinque_metodi.md.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import Tensor

from evd.explain.representation import cosine_similarity


def fit_binary_references(
    descriptors: np.ndarray, labels: np.ndarray, *, seed: int = 42, c: float = 1.0
) -> tuple[dict[str, np.ndarray], dict]:
    """Fit separator and centroids with equal weight per window.

    :param descriptors: W×D, one mean token per training window.
    :param labels: W binary temporal labels; both classes required.
    :return: reusable arrays and fit diagnostics on the training data only.
    :raises ValueError: on empty or non-finite data, or a missing class.
    """
    x = np.asarray(descriptors, dtype=np.float64)
    y = np.asarray(labels)
    if x.ndim != 2 or len(x) != len(y) or set(y.tolist()) != {0, 1}:
        raise ValueError("References: W×D descriptors and two binary classes are required")
    if not np.isfinite(x).all() or c <= 0:
        raise ValueError("Non-finite references or non-positive C")
    scaler = StandardScaler().fit(x)
    standardized = scaler.transform(x)
    classifier = LogisticRegression(C=c, class_weight="balanced", random_state=seed, max_iter=2000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(standardized, y)
    normal = classifier.coef_[0] / scaler.scale_
    # A covector (the separator normal) returns to original coordinates with S^-1, while the
    # path z + eps*w in standardized coordinates becomes x + eps*S*w: both are stored.
    direction = normal / np.linalg.norm(normal)
    arrays = {
        "mean": scaler.mean_,
        "scale": scaler.scale_,
        "coefficient_standardized": classifier.coef_[0],
        "normal_original": normal,
        "direction": direction,
        "standardized_path_original": scaler.scale_ * classifier.coef_[0],
        "intercept_original": np.asarray(classifier.intercept_[0] - scaler.mean_ @ normal),
        "intercept_standardized": classifier.intercept_[0],
        "prototype_normal": x[y == 0].mean(0),
        "prototype_anomaly": x[y == 1].mean(0),
    }
    receipt = {
        "method": "TCAV-style, binary logistic separator; no TCAV statistical test",
        "C": c,
        "class_weight": "balanced",
        "seed": seed,
        "n_normal": int(sum(y == 0)),
        "n_anomaly": int(sum(y == 1)),
        "iterations": classifier.n_iter_.tolist(),
        "converged": not any(issubclass(w.category, ConvergenceWarning) for w in caught),
        "training_accuracy_diagnostic": float(classifier.score(standardized, y)),
        "normal_original": "w_standardized / scale",
        "projection": "(token - training_mean) dot unit_normal_original",
        "sensitivity": "gradient_token(logit) dot unit_normal_original",
        "standardized_path_original": "scale * w_standardized (not used for the map)",
        "prototype_weighting": "equal weight per window; mean of the mean descriptors",
    }
    return arrays, receipt


def fit_class_references(
    descriptors: np.ndarray,
    class_idx: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = 42,
    c: float = 1.0,
) -> tuple[dict[str, np.ndarray], dict]:
    """Fit one CAV per anomaly class: the class against everything else.

    Reuses :func:`fit_binary_references` per class. The negatives are the other anomaly
    classes plus the normal windows; with normals only every CAV would learn "anomalous vs
    normal" again. With very few positives the separator fits them exactly (training
    accuracy 1.0 by construction), so the receipt reports ``n_positive`` next to the
    accuracy.

    :param descriptors: W×D, one mean token per reference window.
    :param class_idx: W class indices; ``-1`` on normal windows.
    :param labels: W binary labels, separating anomalous from normal windows.
    :param seed: seed of the logistic regression.
    :param c: inverse regularization, as in the binary case.
    :return: ``{"direzioni": C×D, "classi": C, "medie": C×D}`` and per-class diagnostics.
    :raises ValueError: if no class has at least one positive example.
    """
    x = np.asarray(descriptors, dtype=np.float64)
    classes = np.asarray(class_idx)
    y = np.asarray(labels)
    if x.ndim != 2 or len(x) != len(classes) or len(x) != len(y):
        raise ValueError(
            "Class references: W×D descriptors, classes and labels of the same W are required"
        )
    present = sorted({int(k) for k in classes[y == 1].tolist()})
    if not present:
        raise ValueError("No anomalous window: class CAVs cannot be learned")
    directions, means, diagnostics = [], [], {}
    for k in present:
        one_vs_rest = (classes == k).astype(int)
        arrays, receipt = fit_binary_references(x, one_vs_rest, seed=seed, c=c)
        directions.append(arrays["direction"])
        means.append(arrays["mean"])
        diagnostics[str(k)] = {
            **{field: receipt[field] for field in ("converged", "iterations")},
            "n_positive": int(one_vs_rest.sum()),
            "n_negative": int(len(one_vs_rest) - one_vs_rest.sum()),
            "training_accuracy_diagnostic": receipt["training_accuracy_diagnostic"],
            "separabile_per_costruzione": int(one_vs_rest.sum()) <= x.shape[1],
        }
    arrays = {
        "class_directions": np.stack(directions),
        "class_indices": np.asarray(present),
        "class_means": np.stack(means),
    }
    receipt = {
        "method": "TCAV-style per class, one versus rest; no TCAV statistical test",
        "negativo": "the other anomaly classes plus the normal windows",
        "classi": present,
        "per_classe": diagnostics,
        "C": c,
        "seed": seed,
    }
    return arrays, receipt


def class_direction_maps(
    forward: Callable[[Tensor], Tensor],
    tokens: Tensor,
    references: dict[str, np.ndarray],
    class_idx: int,
) -> tuple[dict[str, np.ndarray], dict]:
    """Project onto and differentiate along the CAV of this window's class.

    On normal chunks ``class_idx`` is ``-1``: the maps are zero and the receipt says so,
    instead of showing the direction of a class that is not there.

    :param forward: head only, in float32, as in :func:`direction_maps`.
    :param tokens: final features 1×N×D, without the backbone graph.
    :param references: must carry ``class_directions`` and ``class_indices``.
    :param class_idx: chunk class; ``-1`` if it has none.
    :return: the two N maps and the receipt, with the class actually used.
    """
    indices = [int(k) for k in np.asarray(references["class_indices"]).tolist()]
    if class_idx < 0 or class_idx not in indices:
        empty = np.zeros(int(tokens.shape[1]), dtype=np.float32)
        return {"class_projection": empty, "class_sensitivity": empty.copy()}, {
            "class_idx": int(class_idx),
            "usata": None,
            "stato": "NULL MAP: the window has no annotated class"
            if class_idx < 0
            else f"NULL MAP: no CAV for class {class_idx}",
        }
    position = indices.index(class_idx)
    class_references = {
        "direction": np.asarray(references["class_directions"])[position],
        "mean": np.asarray(references["class_means"])[position],
    }
    maps, receipt = direction_maps(forward, tokens, class_references)
    return {
        "class_projection": maps["anomaly_projection"],
        "class_sensitivity": maps["anomaly_sensitivity"],
    }, {**receipt, "class_idx": int(class_idx), "usata": int(class_idx), "stato": ""}


def direction_maps(
    forward: Callable[[Tensor], Tensor], tokens: Tensor, references: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict]:
    """Compute projection and directional derivative of the logit, per token.

    :param forward: head only, in float32 for the finite-difference check.
    :param tokens: final features 1×N×D, without the backbone graph.
    :return: N maps and the derivative check when all tokens move.
    """
    x = tokens.detach().float().requires_grad_(True)
    direction = torch.as_tensor(references["direction"], device=x.device, dtype=x.dtype)
    mean = torch.as_tensor(references["mean"], device=x.device, dtype=x.dtype)
    logit = forward(x).float().sum()
    gradient = torch.autograd.grad(logit, x)[0]
    sensitivity = (gradient * direction).sum(-1)
    derivative = float(sensitivity.sum())
    differences = []
    with torch.no_grad():
        for epsilon in (0.01, 0.03, 0.1):
            finite = float(
                (forward(x + epsilon * direction) - forward(x - epsilon * direction)).sum()
            ) / (2 * epsilon)
            differences.append(
                {"epsilon": epsilon, "finite": finite, "absolute_error": abs(finite - derivative)}
            )
    best = min(differences, key=lambda row: row["absolute_error"])
    return {
        "anomaly_projection": ((x - mean) * direction).sum(-1).detach().cpu().numpy()[0],
        "anomaly_sensitivity": sensitivity.detach().cpu().numpy()[0],
    }, {
        "logit_fp32": float(logit.detach()),
        "directional_derivative": derivative,
        "finite_differences": differences,
        "finite_difference_pass": best["absolute_error"] <= 1e-4 + abs(derivative) * 0.03,
        "head_precision": "fp32; backbone and detector logit stay in the original regime",
    }


def prototype_maps(tokens: np.ndarray, references: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute cosines towards both centroids and their signed difference.

    :param tokens: N×D in original coordinates, without standardization.
    :return: three N maps; a positive difference means closer to the anomalous prototype.
    """
    normal = cosine_similarity(tokens, references["prototype_normal"])
    anomaly = cosine_similarity(tokens, references["prototype_anomaly"])
    return {
        "prototype_normal": normal,
        "prototype_anomaly": anomaly,
        "prototype_difference": anomaly - normal,
    }
