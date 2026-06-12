import unittest

import numpy as np

from transport_toolkit.preprocessing.units import convert_density, convert_field, resistance_to_resistivity


class UnitTests(unittest.TestCase):
    def test_field_conversion(self):
        np.testing.assert_allclose(convert_field([10000], "gauss", "T"), [1.0])

    def test_density_conversion(self):
        np.testing.assert_allclose(convert_density([1], "cm^-3", "m^-3"), [1e6])

    def test_resistance_to_resistivity(self):
        np.testing.assert_allclose(resistance_to_resistivity([10], width=2, thickness=0.5, length=5), [2.0])


if __name__ == "__main__":
    unittest.main()
