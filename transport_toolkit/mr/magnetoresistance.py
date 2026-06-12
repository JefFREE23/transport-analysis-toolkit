"""Magnetoresistance calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from transport_toolkit.utils.results import result_to_dict
from transport_toolkit.utils.validation import as_float_array, require_same_shape


@dataclass(slots=True)
class MagnetoresistanceResult:
    field: np.ndarray
    mr_percent: np.ndarray
    rxx_zero: float
    maximum_mr: float
    field_at_maximum: float
    minimum_mr: float
    field_at_minimum: float

    def to_dict(self):
        return result_to_dict(self)


def estimate_zero_field_value(field: np.ndarray, values: np.ndarray) -> float:
    """Estimate the zero-field value by exact lookup or interpolation."""

    zero_matches = np.isclose(field, 0.0)
    if np.any(zero_matches):
        return float(np.mean(values[zero_matches]))
    order = np.argsort(field)
    field_sorted = field[order]
    values_sorted = values[order]
    if field_sorted[0] > 0 or field_sorted[-1] < 0:
        return float(values[np.argmin(np.abs(field))])
    return float(np.interp(0.0, field_sorted, values_sorted))


def calculate_mr(field, rxx, *, rxx_zero: float | None = None) -> MagnetoresistanceResult:
    """Calculate magnetoresistance percentage."""

    field_array = as_float_array(field, name="field")
    rxx_array = as_float_array(rxx, name="rxx")
    require_same_shape(field_array, rxx_array, names=("field", "rxx"))
    if rxx_zero is None:
        rxx_zero = estimate_zero_field_value(field_array, rxx_array)
    if rxx_zero == 0:
        raise ValueError("zero-field longitudinal resistance must not be zero")
    mr_percent = (rxx_array - rxx_zero) / rxx_zero * 100.0
    max_index = int(np.argmax(mr_percent))
    min_index = int(np.argmin(mr_percent))
    return MagnetoresistanceResult(
        field=field_array,
        mr_percent=mr_percent,
        rxx_zero=float(rxx_zero),
        maximum_mr=float(mr_percent[max_index]),
        field_at_maximum=float(field_array[max_index]),
        minimum_mr=float(mr_percent[min_index]),
        field_at_minimum=float(field_array[min_index]),
    )


def fit_mr_vs_b_squared(field, mr_percent) -> tuple[float, float, float]:
    """Fit MR percent against B squared and return slope, intercept, R squared."""

    field_array = as_float_array(field, name="field")
    mr_array = as_float_array(mr_percent, name="mr_percent")
    require_same_shape(field_array, mr_array, names=("field", "mr_percent"))
    x = field_array**2
    design = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.lstsq(design, mr_array, rcond=None)[0]
    predicted = slope * x + intercept
    ss_res = float(np.sum((mr_array - predicted) ** 2))
    ss_tot = float(np.sum((mr_array - np.mean(mr_array)) ** 2))
    r_squared = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return float(slope), float(intercept), float(r_squared)
