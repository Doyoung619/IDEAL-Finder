from __future__ import annotations

import numpy as np


def line_coefficients(count: int) -> np.ndarray:
    """Return the uniformly spaced coefficients used by an M-way line query."""
    if count < 2:
        raise ValueError("MVLQ requires at least two query options")
    return np.linspace(-1.0, 1.0, count, dtype=np.float64)


def maximum_variance_direction(
    covariance: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return a deterministic principal eigenvector and its eigenvalue."""
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be a square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("covariance contains NaN or Inf")
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    index = int(np.argmax(eigenvalues))
    direction = eigenvectors[:, index]
    pivot = int(np.argmax(np.abs(direction)))
    if direction[pivot] < 0:
        direction = -direction
    return direction, float(eigenvalues[index])


def maximum_variance_line_queries(
    center: np.ndarray,
    covariance: np.ndarray,
    count: int,
    radius: float,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Build a symmetric, uniformly spaced query set on the maximum-variance line."""
    center_array = np.asarray(center, dtype=np.float64)
    if center_array.ndim != 1:
        raise ValueError("center must be a one-dimensional vector")
    if radius < 0 or alpha <= 0:
        raise ValueError("radius must be non-negative and alpha must be positive")
    direction, eigenvalue = maximum_variance_direction(covariance)
    if direction.shape != center_array.shape:
        raise ValueError("center and covariance dimensions do not match")
    coefficients = line_coefficients(count)
    queries = (
        center_array[None, :]
        + float(alpha) * float(radius) * coefficients[:, None] * direction[None, :]
    )
    return queries, coefficients, direction, eigenvalue
