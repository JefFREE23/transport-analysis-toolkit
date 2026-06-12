"""Resistivity and conductivity tensor helpers."""

from __future__ import annotations

import numpy as np

from transport_toolkit.utils.validation import as_float_array, require_same_shape, validate_convention


def resistivity_tensor(rho_xx, rho_xy):
    """Construct the default 2D resistivity tensor.

    The convention is [[rho_xx, -rho_xy], [rho_xy, rho_xx]].
    """

    rho_xx_array = np.asarray(rho_xx, dtype=float)
    rho_xy_array = np.asarray(rho_xy, dtype=float)
    if rho_xx_array.shape != rho_xy_array.shape:
        raise ValueError("rho_xx and rho_xy must have the same shape")
    tensor = np.empty(rho_xx_array.shape + (2, 2), dtype=float)
    tensor[..., 0, 0] = rho_xx_array
    tensor[..., 0, 1] = -rho_xy_array
    tensor[..., 1, 0] = rho_xy_array
    tensor[..., 1, 1] = rho_xx_array
    return tensor


def conductivity_tensor(rho_xx, rho_xy, *, convention: str = "physics") -> tuple[np.ndarray, np.ndarray]:
    """Convert resistivity tensor components to conductivity components."""

    convention = validate_convention(convention)
    rho_xx_array = as_float_array(rho_xx, name="rho_xx") if np.asarray(rho_xx).ndim else np.asarray([rho_xx], dtype=float)
    rho_xy_array = as_float_array(rho_xy, name="rho_xy") if np.asarray(rho_xy).ndim else np.asarray([rho_xy], dtype=float)
    require_same_shape(rho_xx_array, rho_xy_array, names=("rho_xx", "rho_xy"))
    denominator = rho_xx_array**2 + rho_xy_array**2
    if np.any(denominator == 0):
        raise ZeroDivisionError("rho_xx^2 + rho_xy^2 must not be zero")
    sigma_xx = rho_xx_array / denominator
    sigma_xy = -rho_xy_array / denominator
    if convention == "msa":
        sigma_xy = -sigma_xy
    if np.asarray(rho_xx).ndim == 0 and np.asarray(rho_xy).ndim == 0:
        return float(sigma_xx[0]), float(sigma_xy[0])
    return sigma_xx, sigma_xy
