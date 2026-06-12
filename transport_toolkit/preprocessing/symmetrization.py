"""Hall antisymmetrization and longitudinal symmetrization."""

from __future__ import annotations

import numpy as np
import pandas as pd

from transport_toolkit.preprocessing.cleaning import clean_transport_data, remove_offset


def _paired_field_grid(field: np.ndarray, *, tolerance: float) -> np.ndarray:
    magnitudes = np.unique(np.round(np.abs(field) / tolerance).astype(int)) * tolerance
    pairs: list[float] = []
    for magnitude in magnitudes:
        has_positive = np.any(np.isclose(field, magnitude, atol=tolerance, rtol=0.0))
        has_negative = np.any(np.isclose(field, -magnitude, atol=tolerance, rtol=0.0))
        if magnitude == 0 and (has_positive or has_negative):
            pairs.append(0.0)
        elif has_positive and has_negative:
            pairs.append(float(magnitude))
    return np.asarray(sorted(set(pairs)), dtype=float)


def _interp(values: np.ndarray, field: np.ndarray, targets: np.ndarray) -> np.ndarray:
    order = np.argsort(field)
    return np.interp(targets, field[order], values[order])


def symmetrize_arrays(
    field,
    rxx,
    rxy,
    *,
    tolerance: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return non-negative field, symmetrized Rxx, and antisymmetrized Rxy."""

    field_array = np.asarray(field, dtype=float)
    rxx_array = np.asarray(rxx, dtype=float)
    rxy_array = np.asarray(rxy, dtype=float)
    if not (field_array.shape == rxx_array.shape == rxy_array.shape):
        raise ValueError("field, rxx, and rxy must have the same shape")
    if tolerance is None:
        nonzero = np.diff(np.unique(np.sort(field_array)))
        positive_steps = np.abs(nonzero[nonzero != 0])
        tolerance = float(np.min(positive_steps) * 1e-6) if positive_steps.size else 1e-9
        tolerance = max(tolerance, 1e-12)

    grid = _paired_field_grid(field_array, tolerance=tolerance)
    if grid.size == 0:
        raise ValueError("no matching positive/negative field pairs were found")

    rxx_pos = _interp(rxx_array, field_array, grid)
    rxx_neg = _interp(rxx_array, field_array, -grid)
    rxy_pos = _interp(rxy_array, field_array, grid)
    rxy_neg = _interp(rxy_array, field_array, -grid)
    rxx_sym = 0.5 * (rxx_pos + rxx_neg)
    rxy_anti = 0.5 * (rxy_pos - rxy_neg)
    return grid, rxx_sym, rxy_anti


def symmetrize_transport(
    data: pd.DataFrame,
    *,
    field_column: str = "B",
    rxx_column: str = "Rxx",
    rxy_column: str = "Rxy",
    remove_hall_offset: bool = True,
    remove_longitudinal_offset: bool = False,
    tolerance: float | None = None,
) -> pd.DataFrame:
    """Clean and symmetrize a transport DataFrame."""

    cleaned = clean_transport_data(
        data,
        field_column=field_column,
        columns=(field_column, rxx_column, rxy_column),
    )
    field, rxx_sym, rxy_anti = symmetrize_arrays(
        cleaned[field_column].to_numpy(),
        cleaned[rxx_column].to_numpy(),
        cleaned[rxy_column].to_numpy(),
        tolerance=tolerance,
    )
    if remove_hall_offset:
        rxy_anti = remove_offset(rxy_anti, field)
    if remove_longitudinal_offset:
        rxx_sym = remove_offset(rxx_sym, field)

    result = pd.DataFrame({"B": field, "Rxx_sym": rxx_sym, "Rxy_anti": rxy_anti})
    result.attrs.update(data.attrs)
    result.attrs["symmetrization"] = {
        "hall": "antisymmetric",
        "longitudinal": "symmetric",
        "field_points": int(len(result)),
    }
    return result
