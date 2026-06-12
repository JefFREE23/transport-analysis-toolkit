import unittest

import numpy as np

from transport_toolkit.drude.one_carrier import fit_one_carrier, one_carrier_conductivity


class OneCarrierTests(unittest.TestCase):
    def test_one_carrier_fit_recovers_synthetic_parameters(self):
        field = np.linspace(-4, 4, 41)
        density = 1.5e25
        mobility = 0.4
        sigma_xx, sigma_xy = one_carrier_conductivity(field, density, mobility, carrier="electron")
        result = fit_one_carrier(field, sigma_xx, sigma_xy)
        self.assertAlmostEqual(result.parameters["density"] / density, 1.0, places=10)
        self.assertAlmostEqual(result.parameters["mobility"], mobility, places=10)
        self.assertLess(result.residual_norm, 1e-8)


if __name__ == "__main__":
    unittest.main()
