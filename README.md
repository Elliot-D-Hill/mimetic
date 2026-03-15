# Simulacra

`simulacra` generates synthetic longitudinal datasets for supervised learning and
time-to-event modeling with PyTorch and TensorDict.

It supports:

- regression, classification, and count targets from latent subject structure
- repeated observations with temporal covariance (Gaussian family)
- grouped data with arbitrary random effects (intercepts, slopes, etc.)
- survival, mixture-cure, competing-risk, and multi-event outcomes
- tokenized sequence views of the same trajectories

## Installation

`simulacra` currently targets Python 3.13+.

Install from this repository with `uv`:

```bash
uv sync
```

For development dependencies:

```bash
uv sync --dev
```

## Quick Start

The `Simulation` object provides a fluent interface for building synthetic datasets.
Construction builds the linear predictor (eta = Xβ); call `.random_effects()`
to upgrade to a mixed model (eta = Xβ + Zγ). Then choose a response
distribution: `.gaussian()`, `.bernoulli()`, `.poisson()`, etc.

### Gaussian regression

```python
from simulacra import Simulation

data = (
    Simulation(num_samples=1024, num_timepoints=10, num_features=8)
    .random_effects(std=[1.0, 0.1])
    .gaussian(std=0.25)
    .data
)
print(data)
```

```text
TensorDict(
    fields={
        Z: Tensor(shape=torch.Size([1024, 10, 2]), ...),
        X: Tensor(shape=torch.Size([1024, 10, 8]), ...),
        beta: Tensor(shape=torch.Size([1024, 8, 1]), ...),
        eta: Tensor(shape=torch.Size([1024, 10, 1]), ...),
        gamma: Tensor(shape=torch.Size([1024, 2, 1]), ...),
        mu: Tensor(shape=torch.Size([1024, 10, 1]), ...),
        noise: Tensor(shape=torch.Size([1024, 10, 1]), ...),
        time: Tensor(shape=torch.Size([1024, 10, 1]), ...),
        y: Tensor(shape=torch.Size([1024, 10, 1]), ...)},
    batch_size=torch.Size([1024]), ...)
```

### Binary classification

```python
from simulacra import Simulation

data = (
    Simulation(num_samples=1024, num_timepoints=10, num_features=8)
    .random_effects(std=[1.0, 0.1])
    .bernoulli(prevalence=0.3)
    .data
)
print(data)
```

```text
TensorDict(
    fields={
        Z: Tensor(shape=torch.Size([1024, 10, 2]), ...),
        X: Tensor(shape=torch.Size([1024, 10, 8]), ...),
        beta: Tensor(shape=torch.Size([1024, 8, 1]), ...),
        eta: Tensor(shape=torch.Size([1024, 10, 1]), ...),
        gamma: Tensor(shape=torch.Size([1024, 2, 1]), ...),
        mu: Tensor(shape=torch.Size([1024, 10, 1]), ...),
        time: Tensor(shape=torch.Size([1024, 10, 1]), ...),
        y: Tensor(shape=torch.Size([1024, 10, 1]), ...)},
    batch_size=torch.Size([1024]), ...)
```

### Mixture-cure survival with tokenized sequences

```python
from simulacra import Simulation

data = (
    Simulation(
        num_samples=1024,
        num_timepoints=10,
        num_features=8,
    )
    .random_effects(std=[1.0, 0.1])
    .bernoulli(prevalence=0.3)
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
        Z: Tensor(shape=torch.Size([1024, 10, 2]), ...),
        X: Tensor(shape=torch.Size([1024, 10, 8]), ...),
        beta: Tensor(shape=torch.Size([1024, 8, 1]), ...),
        censor_time: Tensor(shape=torch.Size([1024, 1, 1]), ...),
        eta: Tensor(shape=torch.Size([1024, 10, 1]), ...),
        event_time: Tensor(shape=torch.Size([1024, 1, 1]), ...),
        gamma: Tensor(shape=torch.Size([1024, 2, 1]), ...),
        indicator: Tensor(shape=torch.Size([1024, 1, 1]), ...),
        mu: Tensor(shape=torch.Size([1024, 10, 1]), ...),
        observed_time: Tensor(shape=torch.Size([1024, 1, 1]), ...),
        time: Tensor(shape=torch.Size([1024, 10, 1]), ...),
        time_to_event: Tensor(shape=torch.Size([1024, 10, 1]), ...),
        tokens: Tensor(shape=torch.Size([1024, 10, 1]), ...),
        y: Tensor(shape=torch.Size([1024, 10, 1]), ...)},
    batch_size=torch.Size([1024]), ...)
```

### Typical outputs

- `eta` (linear predictor = Xβ or Xβ + Uγ), `X` (design matrix), `beta` (fixed-effects coefficients)
- `gamma` (random effects) and `Z` (random-effects design matrix) — present only after `.random_effects()`
- `y` (response — continuous for Gaussian, counts for Poisson, binary for Bernoulli, categorical label for Categorical/Ordinal)
- `mu` (conditional mean h(eta) — identity for Gaussian, exp for Poisson, sigmoid for Bernoulli, softmax for Categorical)
- `noise` (residual error, Gaussian only)
- `tokens` (when `.tokenize()` is called, derived from `X`)
- `time`, `event_time`, `censor_time`, `indicator`, `observed_time`,
  `time_to_event` for survival tasks

## Pipeline stages

`Simulation()` builds the linear predictor. From there, optionally add
`.random_effects()` and predictor transforms, then choose a response
distribution to sample observations:

| Stage | Methods | Description |
| ------- | --------- | ------------- |
| Simulation | `.random_effects()`, `.activation()`, `.linear()`, `.mlp()`, `.observation_time()`, `.gaussian()`, `.poisson()`, `.bernoulli()`, `.categorical()`, `.ordinal()` | Linear predictor ready for transforms and response distribution |
| Response | `.tokenize()`, `.event_time()` | Response sampled (continuous families) |
| DiscreteResponse | `.tokenize()`, `.event_time()` | Response sampled (discrete families); enables mixture-cure via event time |
| Tokenized | `.event_time()` | Tokenized trajectories |
| EventTime | `.observation_time()`, `.censor_time()` | Survival event times generated |
| DiscreteEventTime | `.mixture_cure()`, `.observation_time()`, `.censor_time()` | Discrete event time (enables cure fraction) |
| Censored | `.survival_indicators()` | Administrative censoring applied |

## Notes on Structure

- `src/simulacra/simulation.py` — fluent `Simulation` interface
- `src/simulacra/functional.py` — reusable building-block functions
- `src/simulacra/covariance.py` — covariance construction (random-effects and residual)
- `src/simulacra/survival.py` — survival analysis functions
- `src/simulacra/states.py` — TypedDict state definitions for the pipeline

## Development

Run tests with:

```bash
uv run pytest
```
