# Competing risks

The competing risks pipeline models multiple failure types simultaneously.
At each timepoint, per-risk Weibull failure times are drawn; the risk with
the shortest time wins. This produces a multiclass event encoding over `K`
risks.

The pipeline starts from a K-dimensional predictor (via `.linear(K)`) and
branches into several encoding strategies: first-failure indicators,
suffix-minimum TTE, and discretized interval bins.

All examples below use `N=1`, `T=5`, `K=3`, `torch.manual_seed(0)`.

## Competing risks

Each column of `eta` [T, K] parameterizes one risk's Weibull scale.
The argmin across risks gives the winning label; `event_mask` is the
one-hot encoding.

```python
data = (
    Simulation(1, 5, 2)
    .linear(3)
    .competing_risks()
    .data
)
```

```{eval-rst}
.. plot:: _plots/failure_times_heatmap.py
```

```text
# Weibull draw per risk [T, K]
failure_times:
  [0.62, 0.07, 1.47]
  [0.40, 0.41, 0.23]
  [0.06, 0.69, 0.13]
  [0.60, 0.57, 0.85]
  [0.76, 0.23, 0.86]

# argmin risk index [T]
tokens: [1, 2, 0, 1, 1]

# one-hot: single winner per timepoint [T, K]
event_mask:
  [0, 1, 0]
  [0, 0, 1]
  [1, 0, 0]
  [0, 1, 0]
  [0, 1, 0]
```

## Independent events

An alternative event process where each risk fires independently via
Bernoulli draws. Multiple risks can fire at the same timepoint, producing
a multi-hot `event_mask` instead of one-hot.

```python
data = (
    Simulation(1, 5, 2)
    .linear(3)
    .independent_events(prevalence=0.3)
    .data
)
```

```text
# multi-hot: multiple risks can fire per timepoint [T, K]
# t=0: risks 0 and 2 fire
# t=1: risks 0 and 1 fire
# t=2: nothing fires
event_mask:
  [1, 0, 1]
  [1, 1, 0]
  [0, 0, 0]
  [1, 0, 1]
  [1, 1, 0]
```

## Risk indicators

First-failure encoding: the minimum failure time across risks is broadcast
to all K columns, and the indicator marks which risk won.

```python
data = (
    Simulation(1, 5, 2)
    .linear(3)
    .competing_risks()
    .risk_indicators()
    .data
)
```

```text
# min failure time, broadcast to all K columns [T, K]
event_time:
  [0.02, 0.02, 0.02]
  [0.99, 0.99, 0.99]
  [0.04, 0.04, 0.04]
  [0.03, 0.03, 0.03]
  [0.00, 0.00, 0.00]

# one-hot winner [T, K]
indicator:
  [0., 0., 1.]
  [0., 1., 0.]
  [0., 1., 0.]
  [0., 0., 1.]
  [0., 1., 0.]
```

## Multi-event

The suffix-minimum algorithm computes, for each timepoint, the time until
the next occurrence of each risk. Risks that never fire again within the
observation window receive the horizon ceiling.

```python
data = (
    Simulation(1, 5, 2)
    .linear(3)
    .competing_risks()
    .multi_event(horizon=8.0)
    .data
)
```

```{eval-rst}
.. plot:: _plots/multi_event_tte.py
```

```text
# TTE to next occurrence of each risk [T, K]
event_time:
  [8., 1., 3.]
  [8., 1., 2.]
  [8., 2., 1.]
  [8., 1., 8.]
  [8., 8., 8.]

# 1 = event within horizon [T, K]
indicator:
  [0., 1., 1.]
  [0., 1., 1.]
  [0., 1., 1.]
  [0., 1., 0.]
  [0., 0., 0.]
```

Risk 0 never fires (no timepoint in `event_mask` had risk 0 as the
winner), so its TTE is always at the horizon ceiling.

## Discretize risk

Bins continuous event times into intervals defined by `boundaries`.
Each value is the fractional exposure within that interval, modulated
by the event indicator. This encoding is designed for discrete-time
survival losses (DeepHit-style).

```python
boundaries = torch.tensor([0.0, 2.0, 4.0, 8.0])
data = (
    Simulation(1, 5, 2)
    .linear(3)
    .competing_risks()
    .multi_event(horizon=8.0)
    .discretize(boundaries)
    .data
)
```

```text
event_time:
  [2., 3., 1.]
  [1., 2., 8.]
  [8., 1., 8.]
  [8., 1., 8.]
  [8., 8., 8.]

indicator:
  [1., 1., 1.]
  [1., 1., 0.]
  [0., 1., 0.]
  [0., 1., 0.]
  [0., 0., 0.]

# fractional exposure per interval [T, J]
# bins: [0, 2), [2, 4), [4, 8)

# risk 0 -- event at t=2: full exposure in [0,2), none after
discrete_event_time[0]:
  [1.00, 0.00, 0.00]
  [0.50, 0.00, 0.00]
  [1.00, 1.00, 1.00]
  [1.00, 1.00, 1.00]
  [1.00, 1.00, 1.00]

# risk 1 -- event at t=3: full [0,2), half of [2,4)
discrete_event_time[1]:
  [0.00, 0.50, 0.00]
  [1.00, 0.00, 0.00]
  [0.50, 0.00, 0.00]
  [0.50, 0.00, 0.00]
  [1.00, 1.00, 1.00]

# risk 2 -- event at t=1: half of [0,2)
discrete_event_time[2]:
  [0.50, 0.00, 0.00]
  [1.00, 1.00, 1.00]
  [1.00, 1.00, 1.00]
  [1.00, 1.00, 1.00]
  [1.00, 1.00, 1.00]
```
