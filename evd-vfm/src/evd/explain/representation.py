"""Feature diagnostics; not attributions of the decision."""

from __future__ import annotations

import numpy as np


class StreamingPCA:
    """Centred covariance accumulated by blocks, without storing all tokens."""

    def __init__(self) -> None:
        self.count = 0
        self.mean: np.ndarray | None = None
        self.scatter: np.ndarray | None = None

    def update(self, tokens: np.ndarray) -> None:
        """:param tokens: N×D matrix of finite features.
        :raises ValueError: if the matrix is empty or not finite.
        """
        x = np.asarray(tokens, dtype=np.float64)
        if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
            raise ValueError("PCA: expected a non-empty matrix of finite features")
        mean = x.mean(0)
        centered = x - mean
        scatter = centered.T @ centered
        if self.count:
            delta = mean - self.mean
            scatter += self.scatter + np.outer(delta, delta) * (
                self.count * len(x) / (self.count + len(x))
            )
            mean = self.mean + delta * len(x) / (self.count + len(x))
        self.count += len(x)
        self.mean, self.scatter = mean, scatter

    def basis(self, components: int = 3) -> np.ndarray:
        """:return: eigenvectors in decreasing order, with the largest-magnitude component
        positive.
        """
        if self.scatter is None:
            raise ValueError("PCA: no feature accumulated")
        _, vectors = np.linalg.eigh(self.scatter)
        basis = vectors[:, -components:][:, ::-1].copy()
        for i in range(basis.shape[1]):
            if basis[np.argmax(abs(basis[:, i])), i] < 0:
                basis[:, i] *= -1
        return basis.astype(np.float32)


def eigencam(tokens: np.ndarray, *, centered: bool = False) -> np.ndarray:
    """Return the first singular projection of a clip, without ReLU or contrast.

    Eigen-CAM (Muhammad and Yeasin, 2020, equations 2-3): O=USV', map=OV1. The paper does
    not subtract the mean; the centred variant replicates
    pytorch-grad-cam/utils/svd_on_activations.py. The sign is made deterministic by the
    dominant coordinate of the eigenvector.

    :param tokens: N×D features.
    :return: signed projection of length N; independent of the classifier.
    :raises ValueError: if the matrix is empty or not finite.
    """
    x = np.asarray(tokens, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("EigenCAM: expected a non-empty matrix of finite features")
    if centered:
        x = x - x.mean(0)
    # The D×D matrix avoids the full N×N SVD and gives the same V1.
    _, vectors = np.linalg.eigh(x.T @ x)
    direction = vectors[:, -1]
    if direction[np.argmax(abs(direction))] < 0:
        direction = -direction
    return (x @ direction).astype(np.float32)


def cosine_similarity(tokens: np.ndarray, query: np.ndarray) -> np.ndarray:
    """:param query: mean of the features in the explicitly chosen region.
    :param tokens: N×D features.
    :return: cosine in [-1, 1]; a zero query is rejected, a zero token gives zero.
    :raises ValueError: if the query is zero or not finite.
    """
    norm = np.linalg.norm(query)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Zero or non-finite query: choose a region with valid features")
    return np.clip(
        tokens @ (query / norm) / np.maximum(np.linalg.norm(tokens, axis=-1), 1e-12), -1, 1
    )
