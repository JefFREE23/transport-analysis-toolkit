import unittest

import numpy as np

from transport_toolkit.msa.kernel import build_msa_kernel
from transport_toolkit.msa.mobility_spectrum import run_msa


class MsaTests(unittest.TestCase):
    def test_run_msa_returns_expected_shapes(self):
        field = np.linspace(-2, 2, 21)
        mobility = np.linspace(-2, 2, 41)
        mobility = mobility[mobility != 0]
        _, k_xx, k_xy = build_msa_kernel(field, mobility)
        truth = np.exp(-((mobility - 0.8) / 0.2) ** 2)
        sigma_xx = k_xx @ truth
        sigma_xy = k_xy @ truth
        result = run_msa(field, sigma_xx, sigma_xy, mobility=mobility, alpha=1e-4)
        self.assertEqual(result.mobility.shape, mobility.shape)
        self.assertEqual(result.spectrum.shape, mobility.shape)
        self.assertTrue(np.isfinite(result.residual_norm))


if __name__ == "__main__":
    unittest.main()
