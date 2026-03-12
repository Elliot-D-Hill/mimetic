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

`Simulation` provides a fluent interface for building synthetic datasets.
Construction generates fixed-effects observations (y = Xβ + ε); call
`.random_effects()` to upgrade to a mixed model (y = Xβ + Uγ + ε).

### Binary classification

```python
from mimetic import Simulation

data = (
    Simulation(num_samples=1024, num_timepoints=10, num_features=8, std=0.25)
    .random_effects(stds=[1.0, 0.1])
    .logistic(prevalence=0.3)
    .data
)
print(data)
```

```text
TensorDict(
    fields={
        U: Tensor(shape=torch.Size([1024, 10, 2]), ...),
        X: Tensor(shape=torch.Size([1024, 10, 8]), ...),
        beta: Tensor(shape=torch.Size([1024, 8, 1]), ...),
        eta: Tensor(shape=torch.Size([1024, 10, 1]), ...),
        gamma: Tensor(shape=torch.Size([1024, 2, 1]), ...),
        label: Tensor(shape=torch.Size([1024, 10, 1]), ...),
        probability: Tensor(shape=torch.Size([1024, 10, 1]), ...),
        time: Tensor(shape=torch.Size([1024, 10, 1]), ...),
        y: Tensor(shape=torch.Size([1024, 10, 1]), ...)},
    batch_size=torch.Size([1024]), ...)
```

### Mixture-cure survival with tokenized sequences

```python
from mimetic import AR1Covariance, Simulation

data = (
    Simulation(
        num_samples=1024,
        num_timepoints=10,
        num_features=8,
        std=0.25,
        covariance=AR1Covariance(correlation=0.8),
    )
    .random_effects(stds=[1.0, 0.1])
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
        U: Tensor(shape=torch.Size([1024, 10, 2]), ...),
        X: Tensor(shape=torch.Size([1024, 10, 8]), ...),
        beta: Tensor(shape=torch.Size([1024, 8, 1]), ...),
        censor_time: Tensor(shape=torch.Size([1024, 1, 1]), ...),
        eta: Tensor(shape=torch.Size([1024, 10, 1]), ...),
        event_time: Tensor(shape=torch.Size([1024, 1, 1]), ...),
        gamma: Tensor(shape=torch.Size([1024, 2, 1]), ...),
        indicator: Tensor(shape=torch.Size([1024, 1, 1]), ...),
        label: Tensor(shape=torch.Size([1024, 10, 1]), ...),
        observed_time: Tensor(shape=torch.Size([1024, 1, 1]), ...),
        probability: Tensor(shape=torch.Size([1024, 10, 1]), ...),
        time: Tensor(shape=torch.Size([1024, 10, 1]), ...),
        time_to_event: Tensor(shape=torch.Size([1024, 10, 1]), ...),
        tokens: Tensor(shape=torch.Size([1024, 10, 1]), ...),
        y: Tensor(shape=torch.Size([1024, 10, 1]), ...)},
    batch_size=torch.Size([1024]), ...)
```

### Typical outputs

- `y` (response), `X` (design matrix), `beta` (fixed-effects coefficients), `eta` (linear predictor, Xβ or Xβ + Uγ)
- `gamma` (random effects) and `U` (random-effects design matrix) — present only after `.random_effects()`
- `tokens` (when `.tokenize()` is called, derived from `X`)
- `probability` and `label` for classification tasks
- `time`, `event_time`, `censor_time`, `indicator`, `observed_time`,
  `time_to_event` for survival tasks

## Pipeline stages

`Simulation()` creates observations immediately. From there, optionally add
`.random_effects()`, then branch by task:

| Stage | Methods | Description |
| ------- | --------- | ------------- |
| Simulation | `.random_effects()`, `.observation_time()`, `.logistic()`, `.multiclass()`, `.ordinal()`, `.event_time()`, `.tokenize()` | Observations ready for optional random effects, labeling, or tokenization |
| Tokenized | `.logistic()`, `.multiclass()`, `.ordinal()`, `.event_time()` | Tokenized trajectories ready for labeling |
| Labeled | `.event_time()`, `.tokenize()` | Classification labels assigned |
| EventTime | `.observation_time()`, `.censor_time()` | Survival event times generated |
| LabeledEventTime | `.mixture_cure()`, `.observation_time()`, `.censor_time()` | Survival from labeled path (enables cure fraction) |
| Censored | `.survival_indicators()` | Administrative censoring applied |

## Notes on Structure

- `src/mimetic/simulation.py` — fluent `Simulation` interface
- `src/mimetic/functional.py` — reusable building-block functions
- `src/mimetic/covariance.py` — covariance construction (random-effects and residual)

## Development

Run tests with:

```bash
uv run pytest
```
