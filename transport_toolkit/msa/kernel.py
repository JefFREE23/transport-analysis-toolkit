"""Mobility spectrum kernel construction."""

from __future__ import annotations

import numpy as np

from transport_toolkit.msa.mobility_grid import mobility_weights
from transport_toolkit.utils.validation import as_float_array


def build_msa_kernel(
    field,
    mobility,
    *,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the linear kernel for sigma_xx and sigma_xy.

    The spectrum is represented as conductivity density over mobility. Signed
    mobility encodes electron-like and hole-like Hall response.
    """

    field_array = as_float_array(field, name="field")
    mobility_array = as_float_array(mobility, name="mobility")
    if weights is None:
        weights = mobility_weights(mobility_array)
    else:
        weights = as_float_array(weights, name="weights")
        if weights.shape != mobility_array.shape:
            raise ValueError("weights and mobility must have the same shape")
    mb = np.outer(field_array, mobility_array)
    denominator = 1.0 + mb**2
    k_xx = weights * (1.0 / denominator)
    k_xy = weights * (mb / denominator)
    kernel = np.vstack([k_xx, k_xy])
    return kernel, k_xx, k_xy
