import unittest

import numpy as np

from transport_toolkit.conductivity.tensor import conductivity_tensor, resistivity_tensor


class TensorTests(unittest.TestCase):
    def test_resistivity_tensor_convention(self):
        tensor = resistivity_tensor(2.0, 1.0)
        np.testing.assert_allclose(tensor, [[2.0, -1.0], [1.0, 2.0]])

    def test_conductivity_tensor_physics_sign(self):
        sigma_xx, sigma_xy = conductivity_tensor([2.0], [1.0])
        np.testing.assert_allclose(sigma_xx, [0.4])
        np.testing.assert_allclose(sigma_xy, [-0.2])

    def test_conductivity_tensor_msa_sign(self):
        _, sigma_xy = conductivity_tensor([2.0], [1.0], convention="msa")
        np.testing.assert_allclose(sigma_xy, [0.2])


if __name__ == "__main__":
    unittest.main()
