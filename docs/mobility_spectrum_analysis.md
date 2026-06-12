# Mobility Spectrum Analysis

Mobility spectrum analysis solves a regularized inverse problem:

```python
from transport_toolkit.msa.mobility_spectrum import run_msa

result = run_msa(B, sigma_xx, sigma_xy)
```

The spectrum is represented as conductivity density over signed mobility. Negative mobility is electron-like and positive mobility is hole-like in the default kernel.

Important options:

- `mobility`: custom mobility grid
- `alpha`: regularization strength
- `smoothness_order`: penalty order
- `non_negative`: clip the spectrum to non-negative values

Use `find_msa_peaks` to identify electron-like and hole-like spectral peaks.
