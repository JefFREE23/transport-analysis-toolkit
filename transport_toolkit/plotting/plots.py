"""Optional matplotlib plotting helpers."""

from __future__ import annotations

import numpy as np


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("plotting requires matplotlib; install transport-analysis-toolkit[plotting]") from exc
    return plt


def plot_raw_transport(field, rxx, rxy):
    plt = _pyplot()
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(field, rxx, marker="o")
    axes[0].set_ylabel("Rxx")
    axes[1].plot(field, rxy, marker="o")
    axes[1].set_xlabel("B (T)")
    axes[1].set_ylabel("Rxy")
    return fig, axes


def plot_hall_fit(field, rho_xy, result):
    plt = _pyplot()
    field = np.asarray(field, dtype=float)
    rho_xy = np.asarray(rho_xy, dtype=float)
    fig, ax = plt.subplots()
    ax.plot(field, rho_xy, "o", label="data")
    ax.plot(field, result.slope * field + result.intercept, "-", label="linear fit")
    ax.set_xlabel("B (T)")
    ax.set_ylabel("rho_xy or Rxy")
    ax.legend()
    return fig, ax


def plot_mr(field, mr_percent, *, squared: bool = False):
    plt = _pyplot()
    field = np.asarray(field, dtype=float)
    x = field**2 if squared else field
    fig, ax = plt.subplots()
    ax.plot(x, mr_percent, marker="o")
    ax.set_xlabel("B^2 (T^2)" if squared else "B (T)")
    ax.set_ylabel("MR (%)")
    return fig, ax


def plot_conductivity(field, sigma_xx, sigma_xy):
    plt = _pyplot()
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(field, sigma_xx, marker="o")
    axes[0].set_ylabel("sigma_xx")
    axes[1].plot(field, sigma_xy, marker="o")
    axes[1].set_xlabel("B (T)")
    axes[1].set_ylabel("sigma_xy")
    return fig, axes


def plot_drude_fit(field, sigma_xx, sigma_xy, fit_sigma_xx, fit_sigma_xy):
    plt = _pyplot()
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(field, sigma_xx, "o", label="data")
    axes[0].plot(field, fit_sigma_xx, "-", label="fit")
    axes[0].set_ylabel("sigma_xx")
    axes[0].legend()
    axes[1].plot(field, sigma_xy, "o", label="data")
    axes[1].plot(field, fit_sigma_xy, "-", label="fit")
    axes[1].set_xlabel("B (T)")
    axes[1].set_ylabel("sigma_xy")
    axes[1].legend()
    return fig, axes


def plot_mobility_spectrum(mobility, spectrum, peaks=None):
    plt = _pyplot()
    fig, ax = plt.subplots()
    ax.plot(mobility, spectrum)
    ax.axvline(0, color="0.5", linewidth=1)
    if peaks:
        for peak in peaks:
            ax.plot(peak.position, peak.amplitude, "o")
    ax.set_xlabel("mobility (m^2/Vs)")
    ax.set_ylabel("spectrum")
    return fig, ax
