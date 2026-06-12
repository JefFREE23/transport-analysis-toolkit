import unittest

from transport_toolkit.plotting.plots import plot_mr


class PlottingTests(unittest.TestCase):
    def test_plotting_is_optional_but_explicit(self):
        try:
            fig, _ = plot_mr([0, 1], [0, 10])
        except RuntimeError as exc:
            self.assertIn("matplotlib", str(exc))
        else:
            fig.canvas.draw()


if __name__ == "__main__":
    unittest.main()
