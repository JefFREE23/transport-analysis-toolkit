# Drude Models

## One-Carrier Model

```python
from transport_toolkit.drude.one_carrier import one_carrier_conductivity, fit_one_carrier
```

The model returns `sigma_xx` and `sigma_xy` for one electron-like or hole-like band.

## Two-Carrier Model

```python
from transport_toolkit.drude.two_carrier import two_carrier_conductivity, fit_two_carrier
```

The two-carrier model uses:

```text
sigma_xx(B) = e [ne mu_e / (1 + (mu_e B)^2) + nh mu_h / (1 + (mu_h B)^2)]
sigma_xy(B) = e B [-ne mu_e^2 / (1 + (mu_e B)^2) + nh mu_h^2 / (1 + (mu_h B)^2)]
```

For difficult real data, provide an `initial_guess` and bounds. Synthetic data should recover known inputs before fitting experimental measurements.
