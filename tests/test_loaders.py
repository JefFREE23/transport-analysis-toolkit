import tempfile
import unittest
from pathlib import Path

from transport_toolkit.io.loaders import load_data


class LoaderTests(unittest.TestCase):
    def test_load_data_detects_columns_and_delimiter(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("Field_T,Rxx_Ohm,Rxy_Ohm\n-1,10,-0.2\n0,9,0\n1,10,0.2\n", encoding="utf-8")
            frame = load_data(path)
        self.assertEqual(list(frame.columns[:3]), ["B", "Rxx", "Rxy"])
        self.assertEqual(frame.attrs["transport_metadata"]["column_map"]["B"], "Field_T")
        self.assertEqual(len(frame), 3)


if __name__ == "__main__":
    unittest.main()
