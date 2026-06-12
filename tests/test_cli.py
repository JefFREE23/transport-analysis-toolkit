import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from transport_toolkit.cli.main import main


class CliTests(unittest.TestCase):
    def test_hall_command_runs_without_crashing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text(
                "B,Rxx,Rxy\n-1,11,-2\n0,10,0\n1,11,2\n",
                encoding="utf-8",
            )
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                exit_code = main(["--json", "hall", str(path)])
        self.assertEqual(exit_code, 0)
        self.assertIn("carrier_density", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
