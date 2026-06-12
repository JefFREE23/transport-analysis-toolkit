"""Recover one-carrier Drude parameters from synthetic data."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transport_toolkit.drude.one_carrier import fit_one_carrier, one_carrier_conductivity


field = np.linspace(-4, 4, 41)
sigma_xx, sigma_xy = one_carrier_conductivity(field, density=1.5e25, mobility=0.4, carrier="electron")
result = fit_one_carrier(field, sigma_xx, sigma_xy)
print(result.to_dict())
