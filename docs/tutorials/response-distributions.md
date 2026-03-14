# Response distributions

After building the linear predictor `eta`, a response family maps it to the
mean `mu` through a link function, then samples an observation `y`.
Each family targets a different data type: continuous, count, binary, or
multi-class.

## Gaussian

The identity link passes `eta` through unchanged: `mu = eta`.
Observations are `y = mu + noise`, where noise is drawn from a
multivariate normal with standard deviation `std`.

```python
data = (
    Simulation(50, 8, 3)
    .gaussian(std=0.5)
    .data
)
```

```text
TensorDict(
    fields={
        X: Tensor([50, 8, 3], ...),
        beta: Tensor([50, 3, 1], ...),
        eta: Tensor([50, 8, 1], ...),
        mu: Tensor([50, 8, 1], ...),
        noise: Tensor([50, 8, 1], ...),
        time: Tensor([50, 8, 1], ...),
        y: Tensor([50, 8, 1], ...)},
    batch_size=[50])
```

```{eval-rst}
.. plot:: _plots/gaussian_response.py
```

## Poisson

The log link maps `mu = exp(eta)`, ensuring non-negative rates.
Observations are integer counts drawn from a Poisson distribution.

```python
data = (
    Simulation(200, 8, 3)
    .poisson()
    .data
)
```

```text
TensorDict(
    fields={
        X: Tensor([200, 8, 3], ...),
        beta: Tensor([200, 3, 1], ...),
        eta: Tensor([200, 8, 1], ...),
        mu: Tensor([200, 8, 1], ...),
        time: Tensor([200, 8, 1], ...),
        y: Tensor([200, 8, 1], ...)},
    batch_size=[200])
```

```{eval-rst}
.. plot:: _plots/poisson_response.py
```

## Bernoulli

The logit link applies `mu = sigmoid(eta + logit(prevalence))`.
The prevalence parameter shifts the decision boundary to control
base rates independently of the predictor's magnitude.

```python
data = (
    Simulation(200, 8, 3)
    .bernoulli(prevalence=0.3)
    .data
)
```

```text
TensorDict(
    fields={
        X: Tensor([200, 8, 3], ...),
        beta: Tensor([200, 3, 1], ...),
        eta: Tensor([200, 8, 1], ...),
        mu: Tensor([200, 8, 1], ...),
        time: Tensor([200, 8, 1], ...),
        y: Tensor([200, 8, 1], ...)},
    batch_size=[200])
```

```{eval-rst}
.. plot:: _plots/bernoulli_response.py
```

## Categorical

For multi-class outcomes, `eta` must first be projected from `[N, T, 1]`
to `[N, T, K]` via `.linear(K)`. The softmax link converts logits to class
probabilities, and a single class is sampled per timepoint.

```python
data = (
    Simulation(200, 8, 3)
    .linear(4)
    .categorical()
    .data
)
```

```text
TensorDict(
    fields={
        X: Tensor([200, 8, 3], ...),
        beta: Tensor([200, 3, 1], ...),
        eta: Tensor([200, 8, 4], ...),
        mu: Tensor([200, 8, 4], ...),
        time: Tensor([200, 8, 1], ...),
        y: Tensor([200, 8, 1], ...)},
    batch_size=[200])
```

```{eval-rst}
.. plot:: _plots/categorical_response.py
```

## Ordinal

The ordinal family models ordered categories through cumulative logits
with evenly spaced thresholds. Class probabilities are differences of
adjacent cumulative probabilities, preserving the natural ordering.

```python
data = (
    Simulation(200, 8, 3)
    .ordinal(num_classes=4)
    .data
)
```

```text
TensorDict(
    fields={
        X: Tensor([200, 8, 3], ...),
        beta: Tensor([200, 3, 1], ...),
        eta: Tensor([200, 8, 1], ...),
        mu: Tensor([200, 8, 4], ...),
        time: Tensor([200, 8, 1], ...),
        y: Tensor([200, 8, 1], ...)},
    batch_size=[200])
```

```{eval-rst}
.. plot:: _plots/ordinal_response.py
```
