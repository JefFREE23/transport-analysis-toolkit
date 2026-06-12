import unittest

import numpy as np

from transport_toolkit.msa.peaks import carrier_contributions, find_msa_peaks


class PeakTests(unittest.TestCase):
    def test_peak_detection_finds_electron_and_hole_peaks(self):
        mobility = np.array([-2, -1, -0.5, 0.5, 1, 2], dtype=float)
        spectrum = np.array([0, 4, 1, 0, 5, 0], dtype=float)
        peaks = find_msa_peaks(mobility, spectrum, min_prominence=0.1)
        self.assertEqual({peak.kind for peak in peaks}, {"electron", "hole"})

    def test_contributions_sum_to_one(self):
        mobility = np.array([-2, -1, 1, 2], dtype=float)
        spectrum = np.array([1, 1, 2, 2], dtype=float)
        contributions = carrier_contributions(mobility, spectrum)
        self.assertAlmostEqual(contributions["electron"] + contributions["hole"], 1.0)


if __name__ == "__main__":
    unittest.main()
