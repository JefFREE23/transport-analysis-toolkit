"""Recover two-carrier Drude parameters from synthetic data."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transport_toolkit.drude.two_carrier import fit_two_carrier, two_carrier_conductivity


field = np.linspace(-5, 5, 51)
truth = (1.2e25, 0.8e25, 0.3, 0.8)
sigma_xx, sigma_xy = two_carrier_conductivity(field, *truth)

result = fit_two_carrier(
    field,
    sigma_xx,
    sigma_xy,
    initial_guess=(1.0e25, 1.0e25, 0.25, 0.7),
)
print(result.to_dict())
