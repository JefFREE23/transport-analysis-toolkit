import unittest

import pandas as pd

from transport_toolkit.preprocessing.cleaning import clean_transport_data, remove_offset


class CleaningTests(unittest.TestCase):
    def test_clean_transport_data_sorts_removes_nan_and_averages_duplicates(self):
        frame = pd.DataFrame(
            {
                "B": [1, 0, 1, None, -1],
                "Rxx": [3, 2, 5, 9, 3],
                "Rxy": [1, 0, 3, 9, -1],
            }
        )
        cleaned = clean_transport_data(frame)
        self.assertEqual(cleaned["B"].tolist(), [-1.0, 0.0, 1.0])
        self.assertEqual(float(cleaned.loc[cleaned["B"] == 1.0, "Rxx"].iloc[0]), 4.0)

    def test_remove_offset_uses_nearest_zero_field(self):
        values = remove_offset([3, 5, 7], field=[-1, 0, 1])
        self.assertEqual(values.tolist(), [-2.0, 0.0, 2.0])


if __name__ == "__main__":
    unittest.main()
