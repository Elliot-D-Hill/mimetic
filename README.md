# mimetic

`mimetic` generates synthetic longitudinal datasets for supervised learning and
time-to-event modeling with PyTorch and TensorDict.

It supports:

- regression and classification targets from latent subject structure
- repeated observations with temporal covariance
- grouped data with random intercepts and slopes
- survival, mixture-cure, competing-risk, and multi-event outcomes
- tokenized sequence views of the same trajectories

The supported Python API is explicit composition through task functions and
pipeline components. There is no monolithic one-call entrypoint.

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

## Supported Tasks

Task functions are available for:

- `linear_data`
- `logistic_data`
- `multiclass_data`
- `ordinal_data`
- `survival_data`
- `mixture_cure_data`
- `competing_risk_data`
- `multi_event_data`

## Quick Start

Build a base `TensorDict`, add shared latent and observed structure, then finish
with a task function.

```python
import torch
from tensordict import TensorDict

from mimetic.covariance import make_covariance
from mimetic.pipeline import add_observed_features, add_random_effects
from mimetic.tasks import logistic_data

num_samples = 1024
num_timepoints = 12
hidden_dim = 8

data = TensorDict(
    {
        "id": torch.arange(num_samples).view(-1, 1, 1),
        "group": torch.zeros(num_samples, 1, 1, dtype=torch.long),
    },
    batch_size=[num_samples],
)

data = add_random_effects(
    data,
    hidden_dim=hidden_dim,
    latent_std=1.0,
    icc=0.2,
    slope_std=0.1,
)

covariance = make_covariance(
    num_timepoints=num_timepoints,
    covariance_type="ar1",
    rho=0.8,
)

data = add_observed_features(
    data,
    num_timepoints=num_timepoints,
    observed_std=0.25,
    covariance=covariance,
)

data = logistic_data(
    data,
    weights=torch.randn(1, hidden_dim) * 0.5,
    prevalence=0.3,
    vocab_size=256,
)

print(data.keys())
print(data["features"].shape)  # [N, T, D]
print(data["tokens"].shape)    # [N, T, 1]
print(data["label"].shape)     # [N, 1, 1]
```

Typical outputs include:

- `id`, `group`
- `latent`, `slope`, `features`
- `tokens`
- `output`
- `probability` and `label` for classification tasks
- `time`, `event_time`, `censor_time`, `indicator`, `observed_time`,
  `time_to_event` for survival-style tasks

## Pipeline Composition

If you want full control, compose the lower-level pipeline functions directly.

```python
import torch
from tensordict import TensorDict

from mimetic.covariance import make_covariance
from mimetic.pipeline import (
    add_linear_output,
    add_logistic_output,
    add_observed_features,
    add_random_effects,
    add_tokens,
)

num_samples = 256
hidden_dim = 4

data = TensorDict(
    {
        "id": torch.arange(num_samples).view(-1, 1, 1),
        "group": torch.zeros(num_samples, 1, 1, dtype=torch.long),
    },
    batch_size=[num_samples],
)

data = add_random_effects(data, hidden_dim=hidden_dim, latent_std=1.0)
covariance = make_covariance(num_timepoints=10, covariance_type="ar1", rho=0.9)
data = add_observed_features(
    data,
    num_timepoints=10,
    observed_std=0.5,
    covariance=covariance,
)
data = add_linear_output(data, weight=torch.randn(1, hidden_dim), bias=0.4)
data = add_logistic_output(data)
data = add_tokens(data, vocab_size=64)
```

This split API is useful when you want to:

- reuse the same latent trajectories across multiple tasks
- control covariance construction explicitly
- integrate only part of the simulation pipeline into another workflow

## Notes On Structure

- `src/mimetic/pipeline.py` contains reusable building blocks
- `src/mimetic/tasks.py` defines task-specific dataset generators
- `src/mimetic/covariance.py` contains temporal covariance helpers

## Development

Run tests with:

```bash
uv run pytest
```
