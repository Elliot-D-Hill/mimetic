# Competing risks

The competing risks pipeline models multiple failure types simultaneously.
At each timepoint, per-risk Weibull failure times are drawn; the risk with
the shortest time wins. This produces a multiclass event encoding over `K`
risks.

The pipeline starts from a K-dimensional predictor (via `.linear(K)`) and
branches into several encoding strategies: first-failure indicators,
suffix-minimum TTE, and discretized interval bins.

## Competing risks

Each column of `eta` [T, K] parameterizes one risk's Weibull scale.
The argmin across risks gives the winning label; `event_mask` is the
one-hot encoding.

```python
data = (
    Simulation(50, 8, 3)
    .linear(4)
    .competing_risks()
    .data
)
```

```text
TensorDict(
    fields={
        ...
        eta: Tensor([50, 8, 4], ...),
        event_mask: Tensor([50, 8, 4], ...),
        failure_times: Tensor([50, 8, 4], ...),
        tokens: Tensor([50, 8, 1], ...)},
    batch_size=[50])
```

```{eval-rst}
.. plot:: _plots/failure_times_heatmap.py
```

## Independent events

An alternative event process where each risk fires independently via
Bernoulli draws. Multiple risks can fire at the same timepoint, producing
a multi-hot `event_mask` instead of one-hot.

```python
data = (
    Simulation(50, 8, 3)
    .linear(4)
    .independent_events(prevalence=0.3)
    .data
)
```

```text
TensorDict(
    fields={
        ...
        eta: Tensor([50, 8, 4], ...),
        event_mask: Tensor([50, 8, 4], ...)},
    batch_size=[50])
```

## Risk indicators

First-failure encoding: the minimum failure time across risks is broadcast
to all K columns, and the indicator marks which risk won.

```python
data = (
    Simulation(50, 8, 3)
    .linear(4)
    .competing_risks()
    .risk_indicators()
    .data
)
```

```text
TensorDict(
    fields={
        ...
        event_time: Tensor([50, 8, 4], ...),
        failure_times: Tensor([50, 8, 4], ...),
        indicator: Tensor([50, 8, 4], ...)},
    batch_size=[50])
```

## Multi-event

The suffix-minimum algorithm computes, for each timepoint, the time until
the next occurrence of each risk. Risks that never fire again within the
observation window receive the horizon ceiling.

```python
data = (
    Simulation(50, 8, 3)
    .linear(4)
    .competing_risks()
    .multi_event(horizon=10.0)
    .data
)
```

```text
TensorDict(
    fields={
        ...
        event_time: Tensor([50, 8, 4], ...),
        indicator: Tensor([50, 8, 4], ...)},
    batch_size=[50])
```

```{eval-rst}
.. plot:: _plots/multi_event_tte.py
```

## Discretize risk

Bins continuous event times into intervals defined by `boundaries`.
Each value is the fractional exposure within that interval, modulated
by the event indicator. This encoding is designed for discrete-time
survival losses (DeepHit-style).

```python
boundaries = torch.tensor([0.0, 2.0, 4.0, 8.0])
data = (
    Simulation(50, 8, 3)
    .linear(4)
    .competing_risks()
    .multi_event(horizon=10.0)
    .discretize(boundaries)
    .data
)
```

```text
TensorDict(
    fields={
        ...
        discrete_event_time: Tensor([50, 8, 4, 3], ...),
        event_time: Tensor([50, 8, 4], ...),
        indicator: Tensor([50, 8, 4], ...)},
    batch_size=[50])
```
