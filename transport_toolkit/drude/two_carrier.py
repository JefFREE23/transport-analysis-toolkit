"""Two-carrier Drude model and fitting."""

from __future__ import annotations

import itertools

import numpy as np

from transport_toolkit.utils.constants import ELEMENTARY_CHARGE
from transport_toolkit.utils.results import FitResult
from transport_toolkit.utils.validation import as_float_array, require_same_shape


PARAMETER_NAMES = ("ne", "nh", "mu_e", "mu_h")


def two_carrier_conductivity(
    field,
    ne: float,
    nh: float,
    mu_e: float,
    mu_h: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sigma_xx and sigma_xy for the electron-hole Drude model."""

    if min(ne, nh, mu_e, mu_h) < 0:
        raise ValueError("densities and mobilities must be non-negative")
    field_array = as_float_array(field, name="field")
    electron = 1.0 + (mu_e * field_array) ** 2
    hole = 1.0 + (mu_h * field_array) ** 2
    sigma_xx = ELEMENTARY_CHARGE * (ne * mu_e / electron + nh * mu_h / hole)
    sigma_xy = ELEMENTARY_CHARGE * field_array * (-ne * mu_e**2 / electron + nh * mu_h**2 / hole)
    return sigma_xx, sigma_xy


def _residual_vector(log_params, field, target, scale):
    params = np.exp(log_params)
    model = np.concatenate(two_carrier_conductivity(field, *params))
    return (model - target) / scale


def _finite_difference_jacobian(log_params, field, target, scale):
    base = _residual_vector(log_params, field, target, scale)
    jacobian = np.empty((base.size, log_params.size), dtype=float)
    for column in range(log_params.size):
        step = 1e-5 * max(abs(log_params[column]), 1.0)
        shifted = log_params.copy()
        shifted[column] += step
        jacobian[:, column] = (_residual_vector(shifted, field, target, scale) - base) / step
    return base, jacobian


def _least_squares_log(
    initial,
    field,
    target,
    scale,
    *,
    bounds: tuple[tuple[float, float, float, float], tuple[float, float, float, float]] | None,
    max_iter: int,
) -> tuple[np.ndarray, float, bool]:
    lower = np.full(4, 1e-300)
    upper = np.full(4, 1e300)
    if bounds is not None:
        lower = np.asarray(bounds[0], dtype=float)
        upper = np.asarray(bounds[1], dtype=float)
        if np.any(lower <= 0) or np.any(upper <= 0) or np.any(upper <= lower):
            raise ValueError("bounds must be positive and upper bounds must exceed lower bounds")
    x = np.log(np.clip(np.asarray(initial, dtype=float), lower, upper))
    lower_log = np.log(lower)
    upper_log = np.log(upper)
    damping = 1e-3
    current = _residual_vector(x, field, target, scale)
    current_norm = float(np.linalg.norm(current))

    for _ in range(max_iter):
        residual, jacobian = _finite_difference_jacobian(x, field, target, scale)
        lhs = jacobian.T @ jacobian + damping * np.eye(4)
        rhs = -jacobian.T @ residual
        try:
            step = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        candidate = np.clip(x + step, lower_log, upper_log)
        candidate_residual = _residual_vector(candidate, field, target, scale)
        candidate_norm = float(np.linalg.norm(candidate_residual))
        if candidate_norm < current_norm:
            x = candidate
            current_norm = candidate_norm
            current = candidate_residual
            damping = max(damping / 5.0, 1e-12)
            if np.linalg.norm(step) < 1e-8:
                return np.exp(x), current_norm, True
        else:
            damping = min(damping * 10.0, 1e12)
    return np.exp(x), current_norm, False


def _default_initial_guesses(field, sigma_xx):
    sigma0 = float(sigma_xx[np.argmin(np.abs(field))])
    sigma0 = max(abs(sigma0), 1e-30)
    mobilities = (0.05, 0.2, 1.0, 5.0)
    guesses = []
    for mu_e, mu_h in itertools.product(mobilities, repeat=2):
        ne = sigma0 / (2.0 * ELEMENTARY_CHARGE * mu_e)
        nh = sigma0 / (2.0 * ELEMENTARY_CHARGE * mu_h)
        guesses.append((ne, nh, mu_e, mu_h))
    return guesses


def fit_two_carrier(
    field,
    sigma_xx,
    sigma_xy,
    *,
    initial_guess: tuple[float, float, float, float] | None = None,
    bounds: tuple[tuple[float, float, float, float], tuple[float, float, float, float]] | None = None,
    max_iter: int = 120,
) -> FitResult:
    """Fit ne, nh, mu_e, and mu_h to conductivity data."""

    field_array = as_float_array(field, name="field")
    sxx = as_float_array(sigma_xx, name="sigma_xx")
    sxy = as_float_array(sigma_xy, name="sigma_xy")
    require_same_shape(field_array, sxx, sxy, names=("field", "sigma_xx", "sigma_xy"))
    target = np.concatenate([sxx, sxy])
    scale = max(float(np.max(np.abs(target))), 1e-30)

    guesses = [initial_guess] if initial_guess is not None else _default_initial_guesses(field_array, sxx)
    best_params: np.ndarray | None = None
    best_norm = np.inf
    best_success = False
    for guess in guesses:
        if guess is None:
            continue
        params, norm, success = _least_squares_log(
            guess,
            field_array,
            target,
            scale,
            bounds=bounds,
            max_iter=max_iter,
        )
        if norm < best_norm:
            best_params = params
            best_norm = norm
            best_success = success
    if best_params is None:
        raise ValueError("no valid initial guess was available")

    model = np.concatenate(two_carrier_conductivity(field_array, *best_params))
    raw_residual_norm = float(np.linalg.norm(model - target))
    residual, jacobian = _finite_difference_jacobian(np.log(best_params), field_array, target, scale)
    covariance = None
    uncertainties: dict[str, float] = {}
    if residual.size > len(best_params):
        try:
            cov_log = np.linalg.pinv(jacobian.T @ jacobian) * (float(np.sum(residual**2)) / (residual.size - 4))
            covariance = np.diag(best_params) @ cov_log @ np.diag(best_params)
            uncertainties = {
                name: float(np.sqrt(max(covariance[index, index], 0.0)))
                for index, name in enumerate(PARAMETER_NAMES)
            }
        except np.linalg.LinAlgError:
            covariance = None
    return FitResult(
        parameters={name: float(value) for name, value in zip(PARAMETER_NAMES, best_params)},
        covariance=covariance,
        uncertainties=uncertainties,
        residual_norm=raw_residual_norm,
        success=best_success or best_norm < 1e-6,
        message="fit completed with NumPy Levenberg-Marquardt fallback",
    )
