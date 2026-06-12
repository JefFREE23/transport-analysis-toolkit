"""Peak detection for mobility spectra."""

from __future__ import annotations

import numpy as np

from transport_toolkit.utils.results import Peak
from transport_toolkit.utils.validation import as_float_array, require_same_shape


def find_msa_peaks(
    mobility,
    spectrum,
    *,
    min_prominence: float = 0.05,
    min_amplitude: float | None = None,
) -> list[Peak]:
    """Find local maxima in a mobility spectrum."""

    mobility_array = as_float_array(mobility, name="mobility")
    spectrum_array = as_float_array(spectrum, name="spectrum")
    require_same_shape(mobility_array, spectrum_array, names=("mobility", "spectrum"))
    if spectrum_array.size < 3:
        return []
    maximum = float(np.max(spectrum_array))
    if maximum <= 0:
        return []
    floor = maximum * min_prominence if min_amplitude is None else min_amplitude
    peaks: list[Peak] = []
    for index in range(1, spectrum_array.size - 1):
        value = spectrum_array[index]
        if value < floor:
            continue
        if value >= spectrum_array[index - 1] and value >= spectrum_array[index + 1]:
            kind = "hole" if mobility_array[index] > 0 else "electron"
            peaks.append(
                Peak(
                    position=float(mobility_array[index]),
                    amplitude=float(value),
                    index=index,
                    kind=kind,
                )
            )
    peaks.sort(key=lambda peak: peak.amplitude, reverse=True)
    return peaks


def carrier_contributions(mobility, spectrum) -> dict[str, float]:
    """Estimate relative electron-like and hole-like spectral contributions."""

    mobility_array = as_float_array(mobility, name="mobility")
    spectrum_array = as_float_array(spectrum, name="spectrum")
    require_same_shape(mobility_array, spectrum_array, names=("mobility", "spectrum"))
    electron = float(np.trapezoid(spectrum_array[mobility_array < 0], mobility_array[mobility_array < 0]))
    hole = float(np.trapezoid(spectrum_array[mobility_array > 0], mobility_array[mobility_array > 0]))
    total = abs(electron) + abs(hole)
    if total == 0:
        return {"electron": 0.0, "hole": 0.0}
    return {"electron": abs(electron) / total, "hole": abs(hole) / total}
