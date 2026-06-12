"""Hall coefficient and carrier-density analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from transport_toolkit.utils.constants import ELEMENTARY_CHARGE
from transport_toolkit.utils.results import result_to_dict
from transport_toolkit.utils.validation import as_float_array, field_mask, require_same_shape, validate_geometry


@dataclass(slots=True)
class HallResult:
    hall_coefficient: float
    carrier_density: float
    slope: float
    intercept: float
    slope_uncertainty: float
    intercept_uncertainty: float
    r_squared: float
    geometry: str
    field_range: tuple[float, float] | None = None

    def to_dict(self):
        return result_to_dict(self)


def linear_fit_with_uncertainty(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float, float]:
    """Fit y = slope*x + intercept and return uncertainties and R squared."""

    if x.size < 2:
        raise ValueError("at least two points are required for a linear fit")
    design = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = slope * x + intercept
    residuals = y - predicted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    if x.size > 2:
        variance = ss_res / (x.size - 2)
        covariance = variance * np.linalg.inv(design.T @ design)
        slope_uncertainty = float(np.sqrt(max(covariance[0, 0], 0.0)))
        intercept_uncertainty = float(np.sqrt(max(covariance[1, 1], 0.0)))
    else:
        slope_uncertainty = 0.0
        intercept_uncertainty = 0.0
    return float(slope), float(intercept), slope_uncertainty, intercept_uncertainty, float(r_squared)


def analyze_hall(
    field,
    rho_xy,
    *,
    geometry: str = "bulk",
    thickness: float | None = None,
    field_range: tuple[float, float] | None = None,
) -> HallResult:
    """Fit a Hall slope and estimate carrier density.

    For ``geometry="bulk"``, ``rho_xy`` is expected to be bulk Hall
    resistivity. For ``geometry="sheet"``, ``rho_xy`` may be sheet Hall
    resistance. If a thickness is supplied for bulk analysis, sheet-like input
    is converted to bulk resistivity before fitting.
    """

    geometry = validate_geometry(geometry)
    field_array = as_float_array(field, name="field")
    rho_array = as_float_array(rho_xy, name="rho_xy")
    require_same_shape(field_array, rho_array, names=("field", "rho_xy"))
    if geometry == "bulk" and thickness is not None:
        if thickness <= 0:
            raise ValueError("thickness must be positive")
        rho_array = rho_array * thickness

    mask = field_mask(field_array, field_range)
    slope, intercept, slope_unc, intercept_unc, r_squared = linear_fit_with_uncertainty(
        field_array[mask],
        rho_array[mask],
    )
    hall_coefficient = slope
    if hall_coefficient == 0:
        carrier_density = np.inf
    else:
        carrier_density = 1.0 / (ELEMENTARY_CHARGE * abs(hall_coefficient))
    return HallResult(
        hall_coefficient=float(hall_coefficient),
        carrier_density=float(carrier_density),
        slope=float(slope),
        intercept=float(intercept),
        slope_uncertainty=float(slope_unc),
        intercept_uncertainty=float(intercept_unc),
        r_squared=float(r_squared),
        geometry=geometry,
        field_range=field_range,
    )
