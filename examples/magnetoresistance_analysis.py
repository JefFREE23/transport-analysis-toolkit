"""Magnetoresistance analysis with synthetic data."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transport_toolkit.mr.magnetoresistance import calculate_mr, fit_mr_vs_b_squared


field = np.linspace(-9, 9, 37)
rxx = 10.0 * (1.0 + 0.02 * field**2)

result = calculate_mr(field, rxx)
slope, intercept, r_squared = fit_mr_vs_b_squared(field, result.mr_percent)

print(result.to_dict())
print({"mr_vs_b_squared_slope": slope, "intercept": intercept, "r_squared": r_squared})
