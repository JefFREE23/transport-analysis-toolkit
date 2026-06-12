import unittest

import numpy as np

from transport_toolkit.hall.hall_analysis import analyze_hall
from transport_toolkit.utils.constants import ELEMENTARY_CHARGE


class HallAnalysisTests(unittest.TestCase):
    def test_hall_slope_and_density_from_synthetic_data(self):
        field = np.linspace(-5, 5, 21)
        slope = 2.5e-8
        rho_xy = slope * field + 1e-10
        result = analyze_hall(field, rho_xy)
        self.assertAlmostEqual(result.slope, slope, places=18)
        self.assertAlmostEqual(result.carrier_density, 1 / (ELEMENTARY_CHARGE * slope), places=6)
        self.assertGreater(result.r_squared, 0.999999)


if __name__ == "__main__":
    unittest.main()
