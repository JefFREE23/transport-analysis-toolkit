"""Mobility-grid construction."""

from __future__ import annotations

import numpy as np


def make_mobility_grid(
    min_mobility: float = 1e-3,
    max_mobility: float = 10.0,
    points: int = 200,
    *,
    spacing: str = "log",
    include_negative: bool = True,
) -> np.ndarray:
    """Create a positive or signed mobility grid in m^2/Vs."""

    if min_mobility <= 0 or max_mobility <= 0:
        raise ValueError("mobility bounds must be positive")
    if max_mobility <= min_mobility:
        raise ValueError("max_mobility must exceed min_mobility")
    if points < 2:
        raise ValueError("points must be at least 2")
    if spacing == "log":
        positive = np.logspace(np.log10(min_mobility), np.log10(max_mobility), points)
    elif spacing == "linear":
        positive = np.linspace(min_mobility, max_mobility, points)
    else:
        raise ValueError("spacing must be 'log' or 'linear'")
    if not include_negative:
        return positive
    return np.concatenate([-positive[::-1], positive])


def mobility_weights(mobility: np.ndarray) -> np.ndarray:
    """Return trapezoidal integration weights for a mobility grid."""

    mobility = np.asarray(mobility, dtype=float)
    if mobility.ndim != 1 or mobility.size < 2:
        raise ValueError("mobility must be a one-dimensional grid")
    order = np.argsort(mobility)
    sorted_mu = mobility[order]
    weights_sorted = np.empty_like(sorted_mu)
    weights_sorted[1:-1] = 0.5 * (sorted_mu[2:] - sorted_mu[:-2])
    weights_sorted[0] = sorted_mu[1] - sorted_mu[0]
    weights_sorted[-1] = sorted_mu[-1] - sorted_mu[-2]
    weights = np.empty_like(weights_sorted)
    weights[order] = np.abs(weights_sorted)
    return weights
