# Survival analysis

The survival pipeline builds on a response distribution, adding event times,
censoring, and per-timepoint indicators. The chain is:
response -> `event_time` -> `censor_time` -> `survival_indicators`.

All examples below use `N=1`, `T=5`, `P=2`, `torch.manual_seed(0)`.

## Event time

An exponential event time is sampled per subject, parameterized by the
mean of `eta` across timepoints.

```python
data = (
    Simulation(1, 5, 2)
    .gaussian(std=0.5)
    .event_time()
    .data
)
```

```text
event_time: 0.88
```

## Censor time

A uniform censor time is drawn within the observation window `[0, T-1]`.
The censor time is independent of the event time, reflecting an
administrative censoring mechanism.

```python
data = (
    Simulation(1, 5, 2)
    .gaussian(std=0.5)
    .event_time()
    .censor_time()
    .data
)
```

```text
event_time:  1.93
censor_time: 2.79
```

## Survival indicators

The observed time is `min(event_time, censor_time)`. The indicator is 1 if
the event was observed before censoring, 0 otherwise. The time-to-event
vector gives each timepoint's distance to the observed time.

```python
data = (
    Simulation(1, 5, 2)
    .gaussian(std=0.5)
    .event_time()
    .censor_time()
    .survival_indicators()
    .data
)
```

```{eval-rst}
.. plot:: _plots/survival_timeline.py
```

```text
event_time:    0.58
censor_time:   3.19
observed_time: 0.58

# 1 = event observed, 0 = censored
indicator: 1.

# time from each observation to the event
time_to_event: [0.58, -0.42, -1.42, -2.42, -3.42]
```
