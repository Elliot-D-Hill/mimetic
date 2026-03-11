# mimetic

`mimetic` generates synthetic longitudinal datasets for supervised learning and
time-to-event modeling with PyTorch and TensorDict.

It supports:

- regression and classification targets from latent subject structure
- repeated observations with temporal covariance
- grouped data with arbitrary random effects (intercepts, slopes, etc.)
- survival, mixture-cure, competing-risk, and multi-event outcomes
- tokenized sequence views of the same trajectories

## Installation

`mimetic` currently targets Python 3.13+.

Install from this repository with `uv`:

```bash
uv sync
```

For development dependencies:

```bash
uv sync --dev
```

## Quick Start

`Simulation` provides a fluent interface that guides you through the pipeline
with type-safe state transitions — autocomplete shows only the methods valid at
each step.

```python
import torch
from mimetic import AR1Covariance, Simulation

weight = torch.randn(1, 8) * 0.5

# High-level: one chained call
data = (
    Simulation(num_samples=1024)
    .random_effects(hidden_dim=8, stds=[1.0, 0.1])
    .observations(
        num_timepoints=10,
        observed_std=0.25,
        covariance=AR1Covariance(correlation=0.8),
    )
    .mixture_cure(weight=weight, prevalence=0.3, shape=2.0, rate=1.0)
    .tokenize(vocab_size=256)
    .data
)
print(data)
```

The same simulated dataset built incrementally, calling each step:

```python
data = (
    Simulation(num_samples=1024)
    .random_effects(hidden_dim=8, stds=[1.0, 0.1])
    .observations(
        num_timepoints=10,
        observed_std=0.25,
        covariance=AR1Covariance(correlation=0.8),
    )
    .linear_predictor(weight=weight, prevalence=0.3)
    .logistic_output()
    .event_time()
    .mixture_cure_censoring()
    .observation_time(shape=2.0, rate=1.0)
    .censor_time()
    .survival_indicators()
    .tokenize(vocab_size=256)
    .data
)
```

Typical outputs include:

- `id`, `group`
- `gamma` (random effects), `y` (observed trajectories), `U` (design matrix)
- `tokens`
- `eta` (linear predictor), `coefficients`, `intercept`
- `probability` and `label` for classification tasks
- `time`, `event_time`, `censor_time`, `indicator`, `observed_time`,
  `time_to_event` for survival-style tasks

## Supported Tasks

High-level macros on the observations stage compose the full pipeline for common tasks:

- `.logistic()` — binary classification
- `.multiclass()` — multi-class classification
- `.ordinal()` — ordinal regression
- `.survival()` — right-censored time-to-event
- `.mixture_cure()` — mixture cure survival model
- `.competing_risk()` — competing risks with discrete time
- `.multi_event()` — per-event TTE with sliding horizon

## Notes on Structure

- `src/mimetic/simulation.py` — fluent `Simulation` interface
- `src/mimetic/functional.py` — reusable building-block functions
- `src/mimetic/covariance.py` — temporal covariance helpers

## Development

Run tests with:

```bash
uv run pytest
```
