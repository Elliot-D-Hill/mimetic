# Response distributions

After building the linear predictor `eta`, a response family maps it to the
mean `mu` through a link function, then samples an observation `y`.
Each family targets a different data type: continuous, count, binary, or
multi-class.

All examples below use `N=1`, `T=5`, `P=2`, `torch.manual_seed(0)`.

## Gaussian

The identity link passes `eta` through unchanged: `mu = eta`.
Observations are `y = mu + noise`, where noise is drawn from a
multivariate normal with standard deviation `std`.

```python
data = (
    Simulation(1, 5, 2)
    .gaussian(std=0.5)
    .data
)
```

```{eval-rst}
.. plot:: _plots/gaussian_response.py
```

```text
mu:    [-0.97,  1.40,  0.39, -0.09,  0.36]
y:     [-1.40,  1.95, -0.14, -0.03,  0.07]
noise: [-0.43,  0.55, -0.54,  0.06, -0.28]
```

## Poisson

The log link maps `mu = exp(eta)`, ensuring non-negative rates.
Observations are integer counts drawn from a Poisson distribution.

```python
data = (
    Simulation(1, 5, 2)
    .poisson()
    .data
)
```

```{eval-rst}
.. plot:: _plots/poisson_response.py
```

```text
mu: [0.76, 3.27, 0.31, 0.54, 0.38]
y:  [2.,   4.,   1.,   0.,   0.  ]
```

## Bernoulli

The logit link applies `mu = sigmoid(eta + logit(prevalence))`.
The prevalence parameter shifts the decision boundary to control
base rates independently of the predictor's magnitude.

```python
data = (
    Simulation(1, 5, 2)
    .bernoulli(prevalence=0.3)
    .data
)
```

```{eval-rst}
.. plot:: _plots/bernoulli_response.py
```

```text
mu: [0.42, 0.21, 0.39, 0.83, 0.32]
y:  [0.,   0.,   1.,   1.,   1.  ]
```

## Categorical

For multi-class outcomes, `eta` must first be projected from `[N, T, 1]`
to `[N, T, K]` via `.linear(K)`. The softmax link converts logits to class
probabilities, and a single class is sampled per timepoint.

```python
data = (
    Simulation(1, 5, 2)
    .linear(3)
    .categorical()
    .data
)
```

```{eval-rst}
.. plot:: _plots/categorical_response.py
```

```text
# softmax probabilities [T, K]
mu:
  [0.42, 0.06, 0.52]
  [0.06, 0.90, 0.04]
  [0.25, 0.53, 0.23]
  [0.35, 0.29, 0.36]
  [0.25, 0.51, 0.24]

# sampled class index [T]
y: [0, 1, 0, 0, 1]
```

## Ordinal

The ordinal family models ordered categories through cumulative logits
with evenly spaced thresholds. Class probabilities are differences of
adjacent cumulative probabilities, preserving the natural ordering.

```python
data = (
    Simulation(1, 5, 2)
    .ordinal(num_classes=4)
    .data
)
```

```{eval-rst}
.. plot:: _plots/ordinal_response.py
```

```text
# class probabilities [T, K]
mu:
  [0.21, 0.45, 0.28, 0.07]
  [0.06, 0.26, 0.46, 0.23]
  [0.01, 0.07, 0.32, 0.59]
  [0.77, 0.19, 0.03, 0.01]
  [0.22, 0.46, 0.26, 0.06]

y: [1, 0, 3, 0, 2]
```
