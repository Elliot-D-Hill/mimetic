# Linear predictor

The linear predictor is the foundation of every mimetic pipeline.
A design matrix `X` [T, P] is drawn from a standard normal, coefficients
`beta` [P] are sampled the same way, and the linear predictor is their
product: `eta = X @ beta`.

All examples below use a single subject (`N=1`), `T=5` timepoints, and
`P=2` features with `torch.manual_seed(0)`.

## Fixed effects

```python
data = Simulation(1, 5, 2).data
```

```{eval-rst}
.. plot:: _plots/linear_predictor_eta.py
```

```text
# design matrix [T, P]
X:
  [ 1.54, -0.29]
  [-2.18,  0.57]
  [-1.08, -1.40]
  [ 0.40,  0.84]
  [-0.72, -0.40]

# coefficients [P]
beta: [-0.60, 0.18]

# linear predictor [T]
eta:  [-0.97, 1.40, 0.39, -0.09, 0.36]

# observation schedule [T]
time: [0., 1., 2., 3., 4.]
```

## Random effects

Random effects add subject-specific deviations to the linear predictor:
`eta = X @ beta + Z @ gamma`. The default basis matrix `Z` is a
Vandermonde matrix built from centered time — an intercept column plus a
linear slope column.

```python
data = (
    Simulation(1, 5, 2)
    .random_effects(std=[0.5, 0.3])
    .data
)
```

```{eval-rst}
.. plot:: _plots/z_basis.py
```

```text
# Vandermonde basis: intercept + centered slope [T, q]
Z:
  [1., -2.]
  [1., -1.]
  [1.,  0.]
  [1.,  1.]
  [1.,  2.]

# random effect coefficients [q]
gamma: [-0.43, 0.33]

eta (before): [-0.97,  1.40,  0.39, -0.09,  0.36]
eta (after):  [-2.06,  0.64, -0.04, -0.19,  0.59]
```
