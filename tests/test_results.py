import unittest

import numpy as np

from transport_toolkit.utils.results import FitResult


class ResultTests(unittest.TestCase):
    def test_fit_result_to_dict_converts_numpy(self):
        result = FitResult(parameters={"x": np.float64(1.5)}, covariance=np.eye(2))
        payload = result.to_dict()
        self.assertEqual(payload["parameters"]["x"], 1.5)
        self.assertEqual(payload["covariance"], [[1.0, 0.0], [0.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
