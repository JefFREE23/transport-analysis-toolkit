"""Convert resistivity tensor components to conductivity components."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transport_toolkit.conductivity.tensor import conductivity_tensor


field = np.linspace(-5, 5, 11)
rho_xx = 1.0e-5 * (1.0 + 0.01 * field**2)
rho_xy = 2.0e-8 * field

sigma_xx, sigma_xy = conductivity_tensor(rho_xx, rho_xy)
print({"sigma_xx_first": sigma_xx[0], "sigma_xy_first": sigma_xy[0]})
