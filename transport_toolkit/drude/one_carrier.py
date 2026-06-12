"""One-carrier Drude model."""

from __future__ import annotations

from typing import Literal

import numpy as np

from transport_toolkit.utils.constants import ELEMENTARY_CHARGE
from transport_toolkit.utils.results import FitResult
from transport_toolkit.utils.validation import as_float_array, require_same_shape


Carrier = Literal["electron", "hole"]


def carrier_sign(carrier: Carrier) -> int:
    if carrier == "electron":
        return -1
    if carrier == "hole":
        return 1
    raise ValueError("carrier must be 'electron' or 'hole'")


def one_carrier_conductivity(
    field,
    density: float,
    mobility: float,
    *,
    carrier: Carrier = "electron",
) -> tuple[np.ndarray, np.ndarray]:
    """Return sigma_xx and sigma_xy for a one-carrier Drude model."""

    if density < 0:
        raise ValueError("density must be non-negative")
    if mobility < 0:
        raise ValueError("mobility must be non-negative")
    field_array = as_float_array(field, name="field")
    mu_b = mobility * field_array
    denominator = 1.0 + mu_b**2
    prefactor = ELEMENTARY_CHARGE * density * mobility
    sigma_xx = prefactor / denominator
    sigma_xy = carrier_sign(carrier) * prefactor * mu_b / denominator
    return sigma_xx, sigma_xy


def fit_one_carrier(
    field,
    sigma_xx,
    sigma_xy,
    *,
    carrier: Carrier | Literal["auto"] = "auto",
) -> FitResult:
    """Estimate density and mobility from synthetic-like one-carrier data."""

    field_array = as_float_array(field, name="field")
    sxx = as_float_array(sigma_xx, name="sigma_xx")
    sxy = as_float_array(sigma_xy, name="sigma_xy")
    require_same_shape(field_array, sxx, sxy, names=("field", "sigma_xx", "sigma_xy"))

    nonzero = np.abs(field_array) > 1e-15
    if not np.any(nonzero):
        raise ValueError("non-zero field points are required to estimate mobility")
    ratio = sxy[nonzero] / sxx[nonzero]
    slope = float(np.sum(field_array[nonzero] * ratio) / np.sum(field_array[nonzero] ** 2))
    detected = "hole" if slope >= 0 else "electron"
    selected_carrier: Carrier = detected if carrier == "auto" else carrier
    sign = carrier_sign(selected_carrier)
    mobility = abs(slope)
    if mobility == 0:
        raise ValueError("estimated mobility is zero")

    estimates = sxx * (1.0 + (mobility * field_array) ** 2) / (ELEMENTARY_CHARGE * mobility)
    density = float(np.median(estimates[estimates > 0]))
    fit_sxx, fit_sxy = one_carrier_conductivity(field_array, density, mobility, carrier=selected_carrier)
    residual = np.concatenate([sxx - fit_sxx, sxy - fit_sxy])
    residual_norm = float(np.linalg.norm(residual))
    uncertainty_scale = residual_norm / max(np.sqrt(residual.size), 1.0)
    return FitResult(
        parameters={
            "density": float(density),
            "mobility": float(mobility),
            "carrier_sign": float(sign),
        },
        uncertainties={
            "density": float(abs(density) * uncertainty_scale / max(np.mean(np.abs(sxx)), 1e-30)),
            "mobility": float(abs(mobility) * uncertainty_scale / max(np.mean(np.abs(sxy)), 1e-30)),
        },
        residual_norm=residual_norm,
        success=True,
        message=f"carrier interpreted as {selected_carrier}",
    )
