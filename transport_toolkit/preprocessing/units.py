"""Unit conversion helpers."""

from __future__ import annotations

import numpy as np


FIELD_TO_TESLA = {
    "t": 1.0,
    "tesla": 1.0,
    "mt": 1e-3,
    "g": 1e-4,
    "gauss": 1e-4,
    "oe": 1e-4,
    "oersted": 1e-4,
}

RESISTANCE_TO_OHM = {
    "ohm": 1.0,
    "ohms": 1.0,
    "mohm": 1e-3,
    "milliohm": 1e-3,
    "kohm": 1e3,
}

MOBILITY_TO_M2_PER_VS = {
    "m2/vs": 1.0,
    "m^2/vs": 1.0,
    "cm2/vs": 1e-4,
    "cm^2/vs": 1e-4,
}

DENSITY_TO_M3 = {
    "m^-3": 1.0,
    "1/m3": 1.0,
    "cm^-3": 1e6,
    "1/cm3": 1e6,
}


def _factor(unit: str, table: dict[str, float], quantity: str) -> float:
    key = unit.strip().lower()
    if key not in table:
        raise ValueError(f"unknown {quantity} unit: {unit}")
    return table[key]


def convert_field(values, from_unit: str, to_unit: str = "T") -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values * _factor(from_unit, FIELD_TO_TESLA, "field") / _factor(to_unit, FIELD_TO_TESLA, "field")


def convert_resistance(values, from_unit: str, to_unit: str = "ohm") -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values * _factor(from_unit, RESISTANCE_TO_OHM, "resistance") / _factor(
        to_unit, RESISTANCE_TO_OHM, "resistance"
    )


def convert_mobility(values, from_unit: str, to_unit: str = "m^2/Vs") -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values * _factor(from_unit, MOBILITY_TO_M2_PER_VS, "mobility") / _factor(
        to_unit, MOBILITY_TO_M2_PER_VS, "mobility"
    )


def convert_density(values, from_unit: str, to_unit: str = "m^-3") -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values * _factor(from_unit, DENSITY_TO_M3, "density") / _factor(to_unit, DENSITY_TO_M3, "density")


def resistance_to_resistivity(
    resistance,
    *,
    width: float,
    thickness: float,
    length: float,
) -> np.ndarray:
    """Convert a four-probe resistance to bulk resistivity."""

    resistance = np.asarray(resistance, dtype=float)
    if width <= 0 or thickness <= 0 or length <= 0:
        raise ValueError("sample dimensions must be positive")
    return resistance * width * thickness / length


def sheet_to_resistivity(sheet_resistance, *, thickness: float) -> np.ndarray:
    """Convert sheet resistance to bulk resistivity using sample thickness."""

    if thickness <= 0:
        raise ValueError("thickness must be positive")
    return np.asarray(sheet_resistance, dtype=float) * thickness
