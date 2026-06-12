import unittest

import numpy as np

from transport_toolkit.mr.magnetoresistance import calculate_mr, fit_mr_vs_b_squared


class MagnetoresistanceTests(unittest.TestCase):
    def test_mr_formula_and_maximum(self):
        field = np.array([-2, -1, 0, 1, 2], dtype=float)
        rxx = 10 * (1 + 0.05 * field**2)
        result = calculate_mr(field, rxx)
        np.testing.assert_allclose(result.mr_percent, [20, 5, 0, 5, 20])
        self.assertEqual(result.maximum_mr, 20.0)
        self.assertIn(abs(result.field_at_maximum), {2.0})

    def test_mr_b_squared_fit(self):
        field = np.array([-2, -1, 0, 1, 2], dtype=float)
        mr = 3 * field**2 + 1
        slope, intercept, r2 = fit_mr_vs_b_squared(field, mr)
        self.assertAlmostEqual(slope, 3.0)
        self.assertAlmostEqual(intercept, 1.0)
        self.assertAlmostEqual(r2, 1.0)


if __name__ == "__main__":
    unittest.main()
