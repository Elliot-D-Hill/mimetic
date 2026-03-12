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

### Binary classification

```python
from mimetic import Simulation

data = (
    Simulation(num_samples=1024)
    .random_effects(stds=[1.0, 0.1])
    .observations(num_timepoints=10, num_features=8, observed_std=0.25)
    .logistic(prevalence=0.3)
    .data
)
print(data)
```

```text
TensorDict(
    fields={
        U: Tensor(shape=torch.Size([1024, 10, 2]), device=cpu, dtype=torch.float32, is_shared=False),
        X: Tensor(shape=torch.Size([1024, 10, 8]), device=cpu, dtype=torch.float32, is_shared=False),
        eta: Tensor(shape=torch.Size([1024, 10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        gamma: Tensor(shape=torch.Size([1024, 2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        label: Tensor(shape=torch.Size([1024, 10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        probability: Tensor(shape=torch.Size([1024, 10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        time: Tensor(shape=torch.Size([1024, 10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        y: Tensor(shape=torch.Size([1024, 10, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
    batch_size=torch.Size([1024]),
    device=None,
    is_shared=False)
```

### Mixture-cure survival with tokenized sequences

```python
from mimetic import AR1Covariance, Simulation

data = (
    Simulation(num_samples=1024)
    .random_effects(stds=[1.0, 0.1])
    .observations(
        num_timepoints=10,
        num_features=8,
        observed_std=0.25,
        covariance=AR1Covariance(correlation=0.8),
    )
    .logistic(prevalence=0.3)
    .tokenize(vocab_size=256)
    .event_time()
    .mixture_cure()
    .observation_time(shape=2.0, rate=1.0)
    .censor_time()
    .survival_indicators()
    .data
)
print(data)
```

```text
TensorDict(
    fields={
        U: Tensor(shape=torch.Size([1024, 10, 2]), device=cpu, dtype=torch.float32, is_shared=False),
        X: Tensor(shape=torch.Size([1024, 10, 8]), device=cpu, dtype=torch.float32, is_shared=False),
        censor_time: Tensor(shape=torch.Size([1024, 1, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        eta: Tensor(shape=torch.Size([1024, 10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        event_time: Tensor(shape=torch.Size([1024, 1, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        gamma: Tensor(shape=torch.Size([1024, 2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        indicator: Tensor(shape=torch.Size([1024, 1, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        label: Tensor(shape=torch.Size([1024, 10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        observed_time: Tensor(shape=torch.Size([1024, 1, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        probability: Tensor(shape=torch.Size([1024, 10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        time: Tensor(shape=torch.Size([1024, 10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        time_to_event: Tensor(shape=torch.Size([1024, 10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        tokens: Tensor(shape=torch.Size([1024, 10, 1]), device=cpu, dtype=torch.int64, is_shared=False),
        y: Tensor(shape=torch.Size([1024, 10, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
    batch_size=torch.Size([1024]),
    device=None,
    is_shared=False)
```

### Typical outputs

- `gamma` (random effects), `y` (scalar GLMM response), `X` (design matrix), `U` (random-effects design matrix)
- `tokens` (when `.tokenize()` is called, derived from `X`)
- `eta` (linear predictor, η = Xβ + Uγ)
- `probability` and `label` for classification tasks
- `time`, `event_time`, `censor_time`, `indicator`, `observed_time`,
  `time_to_event` for survival tasks

## Pipeline stages

After `.observations()`, the pipeline branches by task:

| Stage | Methods | Description |
| ------- | --------- | ------------- |
| Observed | `.logistic()`, `.multiclass()`, `.ordinal()`, `.event_time()`, `.tokenize()`, `.observation_time()` | Trajectories ready for labeling or tokenization |
| Tokenized | `.logistic()`, `.multiclass()`, `.ordinal()`, `.event_time()` | Tokenized trajectories ready for labeling |
| Labeled | `.event_time()`, `.tokenize()` | Classification labels assigned |
| EventTime | `.observation_time()`, `.censor_time()` | Survival event times generated |
| LabeledEventTime | `.mixture_cure()`, `.observation_time()`, `.censor_time()` | Survival from labeled path (enables cure fraction) |
| Censored | `.survival_indicators()` | Administrative censoring applied |

## Notes on Structure

- `src/mimetic/simulation.py` — fluent `Simulation` interface
- `src/mimetic/functional.py` — reusable building-block functions
- `src/mimetic/covariance.py` — temporal covariance helpers

## Development

Run tests with:

```bash
uv run pytest
```
