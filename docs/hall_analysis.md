# Hall Analysis

Use `analyze_hall` after preprocessing:

```python
from transport_toolkit.hall.hall_analysis import analyze_hall

result = analyze_hall(B, rho_xy, geometry="bulk")
```

The result includes:

- Hall coefficient
- carrier density
- fitted slope and intercept
- slope and intercept uncertainty
- R squared
- geometry convention

Use `field_range=(low, high)` to fit only a selected magnetic-field window.
