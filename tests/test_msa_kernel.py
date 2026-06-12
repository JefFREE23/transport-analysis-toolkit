import unittest

import numpy as np

from transport_toolkit.msa.kernel import build_msa_kernel


class MsaKernelTests(unittest.TestCase):
    def test_kernel_dimensions(self):
        field = np.linspace(-2, 2, 5)
        mobility = np.array([-1.0, -0.5, 0.5, 1.0])
        kernel, k_xx, k_xy = build_msa_kernel(field, mobility)
        self.assertEqual(kernel.shape, (10, 4))
        self.assertEqual(k_xx.shape, (5, 4))
        self.assertEqual(k_xy.shape, (5, 4))


if __name__ == "__main__":
    unittest.main()
