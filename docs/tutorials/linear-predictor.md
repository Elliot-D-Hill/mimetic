# Linear predictor

The linear predictor is the foundation of every simulacra pipeline.
A design matrix `X` [T, P] is drawn from a standard normal, coefficients
`beta` [P] are sampled the same way, and the linear predictor is their
product: `eta = X @ beta`.

## Fixed effects

```python
data = Simulation(50, 8, 3).data
```

```text
TensorDict(
    fields={
        X: Tensor([50, 8, 3], ...),
        beta: Tensor([50, 3, 1], ...),
        eta: Tensor([50, 8, 1], ...),
        time: Tensor([50, 8, 1], ...)},
    batch_size=[50])
```

```{eval-rst}
.. plot:: _plots/linear_predictor_eta.py
```

## Random effects

Random effects add subject-specific deviations to the linear predictor:
`eta = X @ beta + Z @ gamma`. The default basis matrix `Z` is a
Vandermonde matrix built from centered time — an intercept column plus a
linear slope column.

```python
data = (
    Simulation(50, 8, 3)
    .random_effects(std=[0.5, 0.3])
    .data
)
```

```text
TensorDict(
    fields={
        X: Tensor([50, 8, 3], ...),
        Z: Tensor([50, 8, 2], ...),
        beta: Tensor([50, 3, 1], ...),
        eta: Tensor([50, 8, 1], ...),
        gamma: Tensor([50, 2, 1], ...),
        time: Tensor([50, 8, 1], ...)},
    batch_size=[50])
```

```{eval-rst}
.. plot:: _plots/random_effects_eta.py
```
