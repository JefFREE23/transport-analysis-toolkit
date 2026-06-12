import unittest

import numpy as np

from transport_toolkit.drude.two_carrier import fit_two_carrier, two_carrier_conductivity


class TwoCarrierTests(unittest.TestCase):
    def test_two_carrier_model_shapes(self):
        field = np.linspace(-3, 3, 13)
        sigma_xx, sigma_xy = two_carrier_conductivity(field, 1e25, 2e25, 0.2, 0.5)
        self.assertEqual(sigma_xx.shape, field.shape)
        self.assertEqual(sigma_xy.shape, field.shape)

    def test_two_carrier_fit_recovers_synthetic_parameters_with_initial_guess(self):
        field = np.linspace(-5, 5, 51)
        truth = (1.2e25, 0.8e25, 0.3, 0.8)
        sigma_xx, sigma_xy = two_carrier_conductivity(field, *truth)
        result = fit_two_carrier(
            field,
            sigma_xx,
            sigma_xy,
            initial_guess=(1.0e25, 1.0e25, 0.25, 0.7),
            max_iter=160,
        )
        for name, expected in zip(("ne", "nh", "mu_e", "mu_h"), truth):
            self.assertAlmostEqual(result.parameters[name] / expected, 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
