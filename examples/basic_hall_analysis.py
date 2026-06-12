"""Basic Hall analysis with synthetic data."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transport_toolkit.hall.hall_analysis import analyze_hall


field = np.linspace(-5, 5, 41)
rho_xy = 2.0e-8 * field + 1.0e-10

result = analyze_hall(field, rho_xy, geometry="bulk")
print(result.to_dict())
