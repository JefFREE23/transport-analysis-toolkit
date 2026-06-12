"""Mobility spectrum analysis workflow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from transport_toolkit.msa.kernel import build_msa_kernel
from transport_toolkit.msa.mobility_grid import make_mobility_grid
from transport_toolkit.msa.regularization import solve_tikhonov
from transport_toolkit.utils.results import result_to_dict
from transport_toolkit.utils.validation import as_float_array, require_same_shape


@dataclass(slots=True)
class MobilitySpectrumResult:
    mobility: np.ndarray
    spectrum: np.ndarray
    fitted_sigma_xx: np.ndarray
    fitted_sigma_xy: np.ndarray
    residual_norm: float
    alpha: float

    def to_dict(self):
        return result_to_dict(self)


def run_msa(
    field,
    sigma_xx,
    sigma_xy,
    *,
    mobility: np.ndarray | None = None,
    alpha: float = 1e-3,
    smoothness_order: int = 2,
    non_negative: bool = True,
) -> MobilitySpectrumResult:
    """Run a regularized mobility spectrum inversion."""

    field_array = as_float_array(field, name="field")
    sxx = as_float_array(sigma_xx, name="sigma_xx")
    sxy = as_float_array(sigma_xy, name="sigma_xy")
    require_same_shape(field_array, sxx, sxy, names=("field", "sigma_xx", "sigma_xy"))
    if mobility is None:
        mobility = make_mobility_grid()
    mobility = as_float_array(mobility, name="mobility")

    kernel, k_xx, k_xy = build_msa_kernel(field_array, mobility)
    target = np.concatenate([sxx, sxy])
    scale = max(float(np.max(np.abs(target))), 1e-30)
    spectrum = solve_tikhonov(
        kernel / scale,
        target / scale,
        alpha=alpha,
        smoothness_order=smoothness_order,
        non_negative=non_negative,
    )
    fitted_sigma_xx = k_xx @ spectrum
    fitted_sigma_xy = k_xy @ spectrum
    residual_norm = float(np.linalg.norm(np.concatenate([fitted_sigma_xx - sxx, fitted_sigma_xy - sxy])))
    return MobilitySpectrumResult(
        mobility=mobility,
        spectrum=spectrum,
        fitted_sigma_xx=fitted_sigma_xx,
        fitted_sigma_xy=fitted_sigma_xy,
        residual_norm=residual_norm,
        alpha=float(alpha),
    )
