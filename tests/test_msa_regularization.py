import unittest

import numpy as np

from transport_toolkit.msa.regularization import solve_tikhonov, smoothness_matrix


class MsaRegularizationTests(unittest.TestCase):
    def test_smoothness_matrix_shape(self):
        self.assertEqual(smoothness_matrix(6, order=2).shape, (4, 6))

    def test_tikhonov_identity_solution(self):
        kernel = np.eye(5)
        target = np.ones(5)
        solution = solve_tikhonov(kernel, target, alpha=0.0, smoothness_order=0, non_negative=True)
        np.testing.assert_allclose(solution, np.ones(5), atol=1e-10)


if __name__ == "__main__":
    unittest.main()
