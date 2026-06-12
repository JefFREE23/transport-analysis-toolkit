"""Validation and array coercion helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def as_float_array(values: Iterable[float], *, name: str) -> np.ndarray:
    """Return a one-dimensional finite float array."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def require_same_shape(*arrays: np.ndarray, names: tuple[str, ...] | None = None) -> None:
    """Validate that all arrays have identical shapes."""

    if not arrays:
        return
    shape = arrays[0].shape
    for index, array in enumerate(arrays[1:], start=1):
        if array.shape != shape:
            labels = names or tuple(f"array {i}" for i in range(len(arrays)))
            raise ValueError(f"{labels[0]} and {labels[index]} must have the same shape")


def validate_geometry(geometry: str) -> str:
    normalized = geometry.lower()
    if normalized not in {"bulk", "sheet"}:
        raise ValueError("geometry must be 'bulk' or 'sheet'")
    return normalized


def validate_convention(convention: str) -> str:
    normalized = convention.lower()
    if normalized not in {"physics", "msa"}:
        raise ValueError("convention must be 'physics' or 'msa'")
    return normalized


def field_mask(field: np.ndarray, field_range: tuple[float, float] | None) -> np.ndarray:
    """Build a boolean mask for an optional absolute field range."""

    if field_range is None:
        return np.ones_like(field, dtype=bool)
    low, high = sorted(field_range)
    return (field >= low) & (field <= high)
