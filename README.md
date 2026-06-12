# Transport Analysis Toolkit

Transport Analysis Toolkit is a Python package for reproducible analysis of electrical transport measurements in condensed matter physics. It covers the workflow from raw Hall-effect and magnetoresistance data through symmetrization, Hall coefficient extraction, carrier density estimates, resistivity and conductivity tensors, Drude fitting, mobility spectrum analysis, plotting, command-line summaries, and reports.

The package import name uses underscores:

```python
import transport_toolkit
```

## Installation

```bash
python -m pip install -e .
```

For plotting, reports, and SciPy-backed nonlinear fitting:

```bash
python -m pip install -e ".[dev]"
```

## Quick Start

```python
from transport_toolkit.io.loaders import load_data
from transport_toolkit.preprocessing.symmetrization import symmetrize_transport
from transport_toolkit.hall.hall_analysis import analyze_hall
from transport_toolkit.mr.magnetoresistance import calculate_mr

data = load_data("sample.csv")
processed = symmetrize_transport(data)

hall = analyze_hall(
    processed["B"],
    processed["Rxy_anti"],
    geometry="bulk",
)
mr = calculate_mr(processed["B"], processed["Rxx_sym"])

print(hall.carrier_density)
print(mr.maximum_mr)
```

## Math Conventions

Hall data are antisymmetrized as:

```text
Rxy_anti(B) = [Rxy(B) - Rxy(-B)] / 2
```

Longitudinal data are symmetrized as:

```text
Rxx_sym(B) = [Rxx(B) + Rxx(-B)] / 2
```

The default resistivity tensor convention is:

```text
rho = [[rho_xx, -rho_xy],
       [rho_xy,  rho_xx]]
```

The corresponding physics-sign conductivity components are:

```text
sigma_xx = rho_xx / (rho_xx^2 + rho_xy^2)
sigma_xy = -rho_xy / (rho_xx^2 + rho_xy^2)
```

Some mobility-spectrum-analysis references use the opposite transverse sign, so `conductivity_tensor(..., convention="msa")` is also available.

## Synthetic Validation Rule

Do not trust real experimental plots until the one-carrier and two-carrier synthetic tests pass. This toolkit is designed to avoid workflows where a figure looks scientific while the sign convention, axis, peak meaning, or fitting interpretation is unclear.

## Modules

- `transport_toolkit.io`: CSV/TXT loading, delimiter detection, column normalization, and metadata.
- `transport_toolkit.preprocessing`: cleaning, field sorting, duplicate removal, offsets, outliers, unit conversion, and Hall/MR symmetrization.
- `transport_toolkit.hall`: Hall slope, Hall coefficient, carrier density, uncertainty, and R squared.
- `transport_toolkit.mr`: magnetoresistance percentage, maxima, and simple field scaling fits.
- `transport_toolkit.conductivity`: resistivity tensor construction and conductivity tensor conversion.
- `transport_toolkit.drude`: one-carrier and two-carrier conductivity models plus fitting helpers.
- `transport_toolkit.msa`: mobility grids, kernel construction, regularized inversion, and peak extraction.
- `transport_toolkit.plotting`: optional matplotlib visualizations.
- `transport_toolkit.cli`: command-line summaries.
- `transport_toolkit.report`: lightweight HTML report generation.

## Command Line

```bash
transport analyze sample.csv
transport hall sample.csv
transport mr sample.csv
transport tensor sample.csv
transport fit-one-carrier sample.csv
transport fit-two-carrier sample.csv
transport run-msa sample.csv
```

## Testing

```bash
python -m pytest
```

The tests use synthetic data wherever a formula is involved. Core numerical functions avoid requiring SciPy, while SciPy is used automatically for nonlinear fitting when installed.
