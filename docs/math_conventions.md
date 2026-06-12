# Math Conventions

## Symmetrization

Hall resistance or resistivity is treated as odd in magnetic field:

```text
Rxy_anti(B) = [Rxy(B) - Rxy(-B)] / 2
```

Longitudinal resistance or resistivity is treated as even in magnetic field:

```text
Rxx_sym(B) = [Rxx(B) + Rxx(-B)] / 2
```

## Hall Coefficient

For bulk data:

```text
R_H = d(rho_xy) / dB
n = 1 / (e |R_H|)
```

For sheet data:

```text
n_s = 1 / (e |dRxy / dB|)
```

## Resistivity And Conductivity Tensors

The default resistivity tensor convention is:

```text
rho = [[rho_xx, -rho_xy],
       [rho_xy,  rho_xx]]
```

The default conductivity components are:

```text
sigma_xx = rho_xx / (rho_xx^2 + rho_xy^2)
sigma_xy = -rho_xy / (rho_xx^2 + rho_xy^2)
```

Use `convention="msa"` when a mobility-spectrum workflow expects the opposite transverse sign.

## Validation Rule

Synthetic one-carrier and two-carrier tests should pass before interpreting real data. This keeps sign conventions and fitted parameters visible and testable.
