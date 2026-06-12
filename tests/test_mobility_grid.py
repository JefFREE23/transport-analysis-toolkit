import unittest

import numpy as np

from transport_toolkit.msa.mobility_grid import make_mobility_grid, mobility_weights


class MobilityGridTests(unittest.TestCase):
    def test_signed_log_grid_has_negative_and_positive_sides(self):
        grid = make_mobility_grid(0.1, 10, 5, include_negative=True)
        self.assertEqual(len(grid), 10)
        self.assertTrue(np.all(grid[:5] < 0))
        self.assertTrue(np.all(grid[5:] > 0))

    def test_weights_match_grid_shape(self):
        grid = np.array([-2, -1, 1, 2], dtype=float)
        self.assertEqual(mobility_weights(grid).shape, grid.shape)


if __name__ == "__main__":
    unittest.main()
