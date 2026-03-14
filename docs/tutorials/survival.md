# Survival analysis

The survival pipeline builds on a response distribution, adding event times,
censoring, and per-timepoint indicators. The chain is:
response -> `event_time` -> `censor_time` -> `survival_indicators`.

## Event time

An exponential event time is sampled per subject, parameterized by the
mean of `eta` across timepoints.

```python
data = (
    Simulation(20, 8, 3)
    .gaussian(std=0.5)
    .event_time()
    .data
)
```

```text
TensorDict(
    fields={
        ...
        event_time: Tensor([20, 1, 1], ...),
        mu: Tensor([20, 8, 1], ...),
        y: Tensor([20, 8, 1], ...)},
    batch_size=[20])
```

## Censor time

A uniform censor time is drawn within the observation window `[0, T-1]`.
The censor time is independent of the event time, reflecting an
administrative censoring mechanism.

```python
data = (
    Simulation(20, 8, 3)
    .gaussian(std=0.5)
    .event_time()
    .censor_time()
    .data
)
```

```text
TensorDict(
    fields={
        ...
        censor_time: Tensor([20, 1, 1], ...),
        event_time: Tensor([20, 1, 1], ...),
        mu: Tensor([20, 8, 1], ...),
        y: Tensor([20, 8, 1], ...)},
    batch_size=[20])
```

## Survival indicators

The observed time is `min(event_time, censor_time)`. The indicator is 1 if
the event was observed before censoring, 0 otherwise. The time-to-event
vector gives each timepoint's distance to the observed time.

```python
data = (
    Simulation(20, 8, 3)
    .gaussian(std=0.5)
    .event_time()
    .censor_time()
    .survival_indicators()
    .data
)
```

```text
TensorDict(
    fields={
        ...
        censor_time: Tensor([20, 1, 1], ...),
        event_time: Tensor([20, 1, 1], ...),
        indicator: Tensor([20, 1, 1], ...),
        observed_time: Tensor([20, 1, 1], ...),
        time_to_event: Tensor([20, 8, 1], ...),
        y: Tensor([20, 8, 1], ...)},
    batch_size=[20])
```

```{eval-rst}
.. plot:: _plots/survival_timeline.py
```
