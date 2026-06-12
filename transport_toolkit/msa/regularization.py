"""Regularized linear solvers for mobility spectrum analysis."""

from __future__ import annotations

import numpy as np


def smoothness_matrix(size: int, order: int = 2) -> np.ndarray:
    """Build a finite-difference smoothness penalty matrix."""

    if size < 3:
        raise ValueError("size must be at least 3")
    if order == 0:
        return np.eye(size)
    if order == 1:
        matrix = np.zeros((size - 1, size), dtype=float)
        for index in range(size - 1):
            matrix[index, index] = -1.0
            matrix[index, index + 1] = 1.0
        return matrix
    if order == 2:
        matrix = np.zeros((size - 2, size), dtype=float)
        for index in range(size - 2):
            matrix[index, index] = 1.0
            matrix[index, index + 1] = -2.0
            matrix[index, index + 2] = 1.0
        return matrix
    raise ValueError("order must be 0, 1, or 2")


def solve_tikhonov(
    kernel: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float = 1e-3,
    smoothness_order: int = 2,
    non_negative: bool = True,
    iterations: int = 80,
) -> np.ndarray:
    """Solve a Tikhonov-regularized inverse problem."""

    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    kernel = np.asarray(kernel, dtype=float)
    target = np.asarray(target, dtype=float)
    if kernel.ndim != 2:
        raise ValueError("kernel must be two-dimensional")
    if target.ndim != 1 or target.size != kernel.shape[0]:
        raise ValueError("target length must equal kernel row count")

    penalty = smoothness_matrix(kernel.shape[1], smoothness_order)
    lhs = kernel.T @ kernel + alpha * (penalty.T @ penalty)
    rhs = kernel.T @ target
    solution = np.linalg.solve(lhs + 1e-14 * np.eye(lhs.shape[0]), rhs)
    if not non_negative:
        return solution

    solution = np.maximum(solution, 0.0)
    gram = kernel.T @ kernel + alpha * (penalty.T @ penalty)
    lipschitz = float(np.linalg.norm(gram, ord=2))
    step = 1.0 / max(lipschitz, 1e-30)
    for _ in range(iterations):
        gradient = gram @ solution - rhs
        solution = np.maximum(solution - step * gradient, 0.0)
    return solution
