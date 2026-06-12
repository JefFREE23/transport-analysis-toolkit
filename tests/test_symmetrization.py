import unittest

import numpy as np
import pandas as pd

from transport_toolkit.preprocessing.symmetrization import symmetrize_arrays, symmetrize_transport


class SymmetrizationTests(unittest.TestCase):
    def test_even_and_odd_parts_are_recovered(self):
        field = np.array([-2, -1, 0, 1, 2], dtype=float)
        rxx = 10.0 + field**2 + 0.3 * field
        rxy = 0.1 + 4.0 * field + 0.2 * field**2
        grid, rxx_sym, rxy_anti = symmetrize_arrays(field, rxx, rxy)
        np.testing.assert_allclose(grid, [0, 1, 2])
        np.testing.assert_allclose(rxx_sym, [10, 11, 14])
        np.testing.assert_allclose(rxy_anti, [0, 4, 8])

    def test_dataframe_symmetrization_returns_named_columns(self):
        frame = pd.DataFrame({"B": [-1, 0, 1], "Rxx": [2, 1, 2], "Rxy": [-3, 0, 3]})
        processed = symmetrize_transport(frame)
        self.assertEqual(list(processed.columns), ["B", "Rxx_sym", "Rxy_anti"])


if __name__ == "__main__":
    unittest.main()
