from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.distributions as dist
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor

# =============================================================================
# Covariance Functions
# =============================================================================


def isotropic_covariance(size: int) -> Tensor:
    """Identity covariance matrix (no temporal correlation)."""
    return torch.eye(size)


def ar1_covariance(rho: float, num_timepoints: int) -> Tensor:
    """AR(1) covariance with potentially irregular spacing.

    Computes Σ[i,j] = ρ^|i - j|

    Args:
        rho: Autocorrelation parameter in (0, 1).
        num_timepoints: Number of time points (size of covariance matrix).
    """
    grid = torch.arange(num_timepoints, dtype=torch.float32)
    diff = torch.abs(grid.unsqueeze(0) - grid.unsqueeze(1))
    return rho**diff


def lkj_covariance(eta: float, size: int) -> Tensor:
    """Sample correlation matrix from LKJ distribution.

    Args:
        eta: Concentration parameter. eta=1 gives uniform over correlation matrices.
        size: Dimension of the correlation matrix.
    """
    L = dist.LKJCholesky(size, concentration=eta).sample()
    return L @ L.T


def make_covariance(
    num_timepoints: int,
    covariance_type: Literal["isotropic", "ar1", "lkj"] = "isotropic",
    rho: float = 0.9,
    eta: float = 1.0,
) -> Tensor:
    """Factory function for covariance matrices.

    Args:
        num_timepoints: Number of time points (size of covariance matrix).
        covariance_type: Type of covariance structure.
        rho: AR(1) autocorrelation parameter (used if covariance_type="ar1").
        eta: LKJ concentration parameter (used if covariance_type="lkj").
    """
    if covariance_type == "isotropic":
        return isotropic_covariance(num_timepoints)
    elif covariance_type == "ar1":
        return ar1_covariance(rho, num_timepoints)
    elif covariance_type == "lkj":
        return lkj_covariance(eta, num_timepoints)
    else:
        raise ValueError(f"Unknown covariance_type: {covariance_type}")


# =============================================================================
# Feature Generation
# =============================================================================


def make_parameters(parameters: int | list[float], scale: float) -> Tensor:
    match parameters:
        case int():
            mean = torch.randn(parameters)
        case list():
            mean = torch.tensor(parameters)
    return mean.unsqueeze(0) * scale


def make_latent_features(
    hidden_dim: int, num_samples: int, latent_std: float
) -> Tensor:
    zero = torch.zeros(hidden_dim)
    covariance = torch.eye(hidden_dim) * latent_std**2
    mvn_i = dist.MultivariateNormal(zero, covariance)
    features = mvn_i.sample((num_samples,))
    return features


def make_observed_features(
    latent: Tensor, num_timepoints: int, observed_std: float, covariance: Tensor
) -> Tensor:
    """Generate observed features.

    Args:
        latent: Latent features [N, D].
        num_timepoints: Number of time points T.
        observed_std: Standard deviation for observation noise.
        covariance: [T, T] correlation matrix.

    Returns:
        Observed features [N, T, D].
    """
    # Scale correlation to covariance
    covariance = covariance * observed_std**2
    num_samples, hidden_dim = latent.shape
    # Sample noise with temporal correlation: [N, D, T]
    mvn = dist.MultivariateNormal(torch.zeros(num_timepoints), covariance)
    noise = mvn.sample((num_samples, hidden_dim))  # [N, D, T]
    noise = noise.permute(0, 2, 1)  # [N, T, D]
    # Add latent mean (broadcast across time)
    features = latent.unsqueeze(1) + noise  # [N, 1, D] + [N, T, D] = [N, T, D]
    return features


def tokenize_features(
    features: Tensor, vocab_size: int, concentration: float = 1.0
) -> Tensor:
    hidden_dim = features.size(-1)
    weight = torch.randn(vocab_size, hidden_dim)
    logits = F.linear(features, weight)
    prior = torch.ones(vocab_size) * concentration
    dirichlet = dist.Dirichlet(prior)
    skew = dirichlet.sample(logits.shape[:-1]).log()
    probs = F.softmax(logits + skew, dim=-1)
    tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(
        features.shape[:-1]
    )
    return tokens


# =============================================================================
# Task-Specific Data Generation
# =============================================================================


def make_linear_output(input: Tensor, weight: Tensor, bias: float) -> Tensor:
    event_rate = input.new_tensor(bias)
    event_rate = torch.log(event_rate / (1 - event_rate))
    return F.linear(input=input, weight=weight, bias=event_rate)


def make_linear_data(
    latent: Tensor,
    features: Tensor,
    weights: Tensor,
    prevalence: float,
    vocab_size: int = 1000,
    concentration: float = 1.0,
) -> TensorDict:
    """Generate linear regression data.

    Args:
        latent: Pre-built latent features [N, D].
        features: Pre-built observed features [N, T, D].
        weights: Pre-built weight matrix [1, D].
        prevalence: Base rate for the outcome.
        vocab_size: Number of token types.
        concentration: Dirichlet concentration for token sampling.
    """
    num_samples = latent.size(0)
    output = make_linear_output(latent, weights, bias=prevalence)
    tokens = tokenize_features(features, vocab_size, concentration)
    ids = torch.arange(num_samples).view(-1, 1, 1)
    return TensorDict(
        {
            "id": ids,
            "latent": latent.unsqueeze(1),
            "features": features,
            "output": output.unsqueeze(1),
            "tokens": tokens.unsqueeze(-1),
        },
        batch_size=num_samples,
    )


def make_logistic_data(
    latent: Tensor,
    features: Tensor,
    weights: Tensor,
    prevalence: float,
    vocab_size: int = 1000,
    concentration: float = 1.0,
) -> TensorDict:
    """Generate logistic regression data."""
    data = make_linear_data(
        latent, features, weights, prevalence, vocab_size, concentration
    )
    probability = torch.sigmoid(data["output"])
    label = torch.bernoulli(probability)
    data["probability"] = probability
    data["label"] = label
    return data


def make_time(
    num_samples: int, num_timepoints: int, gamma_shape: float, gamma_rate: float
) -> Tensor:
    gamma = dist.Gamma(gamma_shape, gamma_rate)
    time_intervals = gamma.sample((num_samples, num_timepoints))
    time = time_intervals.cumsum(dim=1)
    return time.unsqueeze(-1)  # [N, T, 1]


def make_censor_time(time: Tensor) -> Tensor:
    min_time = torch.amin(time, dim=1, keepdim=True)
    max_time = torch.amax(time, dim=1, keepdim=True)
    censor_time = dist.Uniform(min_time, max_time).sample()
    return censor_time  # [N, 1, 1]


def make_event_time(data: TensorDict) -> TensorDict:
    rate = torch.exp(data["output"])
    data["event_time"] = dist.Exponential(rate=rate).sample()
    return data


def make_survival_data(
    data: TensorDict,
    num_samples: int,
    num_timepoints: int,
    gamma_shape: float,
    gamma_rate: float,
) -> TensorDict:
    time = make_time(num_samples, num_timepoints, gamma_shape, gamma_rate)
    censor_time = make_censor_time(time=time)
    event_time = data["event_time"]
    indicator = (event_time < censor_time).float()
    observed_time = torch.minimum(censor_time, event_time)
    time_to_event = observed_time - time
    data.update(
        {
            "time": time,
            "censor_time": censor_time,
            "indicator": indicator,
            "observed_time": observed_time,
            "time_to_event": time_to_event,
        }
    )
    return data


def survival_analysis_data(
    latent: Tensor,
    features: Tensor,
    weights: Tensor,
    prevalence: float,
    gamma_shape: float,
    gamma_rate: float,
    vocab_size: int = 1000,
    concentration: float = 1.0,
) -> TensorDict:
    """Generate survival analysis data."""
    num_samples, num_timepoints = features.shape[0], features.shape[1]
    data = make_linear_data(
        latent, features, weights, prevalence, vocab_size, concentration
    )
    data = make_event_time(data=data)
    return make_survival_data(
        data, num_samples, num_timepoints, gamma_shape, gamma_rate
    )


def mixture_cure_data(
    latent: Tensor,
    features: Tensor,
    weights: Tensor,
    prevalence: float,
    gamma_shape: float,
    gamma_rate: float,
    vocab_size: int = 1000,
    concentration: float = 1.0,
) -> TensorDict:
    """Generate mixture cure model data."""
    num_samples, num_timepoints = features.shape[0], features.shape[1]
    data = make_logistic_data(
        latent, features, weights, prevalence, vocab_size, concentration
    )
    data = make_event_time(data=data)
    data["event_time"][data["label"] == 0] = torch.inf
    survival_data = make_survival_data(
        data, num_samples, num_timepoints, gamma_shape, gamma_rate
    )
    data.update(survival_data)
    return data


def make_competing_risks_events(
    features: Tensor, vocab_size: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Generate competing risks TTE data using Weibull distributions.

    For each position, samples potential failure times for all event types.
    The observed event is the one with minimum failure time.

    Args:
        features: Input features [N, T, D].
        vocab_size: Number of event types (K).

    Returns:
        tokens: Observed event type [N, T] (argmin of failure times).
        tte: Inter-event TTE [N, T] (min of failure times).
        failure_times: All potential failure times [N, T, K] for multi-label training.
    """
    # Project features to Weibull log-parameters
    weight_alpha = torch.randn(vocab_size, features.size(-1))
    weight_beta = torch.randn(vocab_size, features.size(-1))
    log_alpha = F.linear(features, weight_alpha)
    log_beta = F.linear(features, weight_beta)
    # Ensure positive parameters
    # TODO why 0.1? Should we allow smaller values? Maybe add a separate scale parameter?
    alpha = F.softplus(log_alpha) + 0.1  # scale > 0
    beta = F.softplus(log_beta) + 0.1  # shape > 0
    # Sample failure times for all event types
    weibull = dist.Weibull(alpha, beta)
    failure_times = weibull.rsample()  # [N, T, K]
    # Observed event is the minimum
    tte, tokens = failure_times.min(dim=-1)
    return tokens, tte, failure_times


def _base_tte_data(latent: Tensor, features: Tensor, vocab_size: int) -> TensorDict:
    """Shared base generation for TTE simulation."""
    num_samples = latent.size(0)
    tokens, tte, failure_times = make_competing_risks_events(features, vocab_size)
    time = tte.cumsum(dim=1)
    data = TensorDict(
        {
            "id": torch.arange(num_samples).view(-1, 1, 1),
            "latent": latent.unsqueeze(1),
            "features": features,
            "tokens": tokens.unsqueeze(-1),
            "event_time": tte.unsqueeze(-1),
            "time": time.unsqueeze(-1),
            "failure_times": failure_times,
        },
        batch_size=num_samples,
    )
    return data


def discretize_event_time(
    event_time: Tensor, indicator: Tensor, boundaries: Tensor
) -> Tensor:
    interval_start = boundaries[:-1].view(*([1] * event_time.dim()), -1)
    interval_end = boundaries[1:].view(*([1] * event_time.dim()), -1)
    interval_width = interval_end - interval_start
    event_time = event_time.unsqueeze(-1)
    exposure = ((event_time - interval_start) / interval_width).clamp(0, 1)
    in_interval = (event_time > interval_start) & (event_time <= interval_end)
    indicator = indicator.unsqueeze(-1).to(exposure.dtype)
    return indicator * (exposure * in_interval) + (1.0 - indicator) * exposure


def competing_risk_data(
    latent: Tensor,
    features: Tensor,
    vocab_size: int,
    boundaries: Tensor | None = None,
) -> TensorDict:
    """Generate competing risks TTE data for self-supervised pretraining."""
    data = _base_tte_data(latent, features, vocab_size)
    dtype = data["event_time"].dtype
    indicator = F.one_hot(data["tokens"], num_classes=vocab_size).to(dtype)
    data["indicator"] = indicator
    data["event_time"] = data["event_time"].unsqueeze(-1).expand(-1, -1, vocab_size)
    if boundaries is not None:
        data["discrete_event_time"] = discretize_event_time(
            data["event_time"], indicator, boundaries
        )
    return data


def next_event_times(tokens: Tensor, time: Tensor, vocab_size: int) -> Tensor:
    time = time.squeeze(-1) if time.dim() == 3 else time
    tokens = tokens.squeeze(-1) if tokens.dim() == 3 else tokens
    event_mask = F.one_hot(tokens, num_classes=vocab_size).to(torch.bool)
    event_times = time.unsqueeze(-1).expand(-1, -1, vocab_size)
    event_times = event_times.masked_fill(~event_mask, torch.inf)
    # Suffix min, then shift forward by 1 to get strictly next occurrence
    event_times_reversed = torch.flip(event_times, dims=[1])
    suffix_min = torch.flip(torch.cummin(event_times_reversed, dim=1).values, dims=[1])
    next_time = F.pad(suffix_min[:, 1:], (0, 0, 0, 1), value=torch.inf)
    time_to_event = next_time - time.unsqueeze(-1)
    return time_to_event


def multi_event_data(
    latent: Tensor,
    features: Tensor,
    vocab_size: int,
    boundaries: Tensor | None = None,
) -> TensorDict:
    """Generate per-event TTE data with a sliding horizon window."""
    data = _base_tte_data(latent, features, vocab_size)
    time = data["time"].squeeze(-1)
    event_time = next_event_times(data["tokens"], time, vocab_size)
    horizon = (
        boundaries[-1]
        if boundaries is not None
        else event_time[event_time.isfinite()].max()
    )
    data["event_time"] = event_time.clamp(max=horizon)
    data["indicator"] = (event_time <= horizon).to(data["event_time"].dtype)
    if boundaries is not None:
        data["discrete_event_time"] = discretize_event_time(
            data["event_time"], data["indicator"], boundaries
        )
    return data


# =============================================================================
# Utilities
# =============================================================================


def expand_constants(td: TensorDict, num_timepoints: int) -> TensorDict:
    """Broadcast tensors with singleton time dimension to num_timepoints."""
    for key, value in td.items():
        if value.dim() < 2 or value.size(1) != 1:
            continue
        expanded_shape = (value.size(0), num_timepoints, *value.shape[2:])
        td[key] = value.expand(*expanded_shape)
    return td


@dataclass
class SimulationConfig:
    task: Literal[
        "linear",
        "logistic",
        "survival",
        "mixture",
        "competing_risk",
        "multi_event",
    ]
    num_samples: int
    num_timepoints: int
    parameters: int | list[float]
    scale: float
    latent_std: float
    observed_std: float
    path: Path
    # Task-specific (optional with defaults)
    prevalence: float = 0.5
    gamma_shape: float = 1.0
    gamma_rate: float = 1.0
    vocab_size: int = 1000
    concentration: float = 1.0  # Dirichlet concentration for token sampling
    tte_boundaries: list[float] | None = None
    # Covariance options
    covariance_type: Literal["isotropic", "ar1", "lkj"] = "isotropic"
    rho: float = 0.9  # AR(1) autocorrelation
    eta: float = 1.0  # LKJ concentration


def simulate(cfg: SimulationConfig):
    """Main simulation entry point with centralized component construction."""

    weights = make_parameters(cfg.parameters, cfg.scale)
    latent = make_latent_features(
        hidden_dim=weights.size(-1),
        num_samples=cfg.num_samples,
        latent_std=cfg.latent_std,
    )
    covariance = make_covariance(
        num_timepoints=cfg.num_timepoints,
        covariance_type=cfg.covariance_type,
        rho=cfg.rho,
        eta=cfg.eta,
    )
    features = make_observed_features(
        latent=latent,
        num_timepoints=cfg.num_timepoints,
        observed_std=cfg.observed_std,
        covariance=covariance,
    )

    boundaries = None
    if cfg.task in {"competing_risk", "multi_event"} and cfg.tte_boundaries is not None:
        boundaries = torch.tensor(
            cfg.tte_boundaries, device=features.device, dtype=features.dtype
        )

    if cfg.task == "linear":
        data = make_linear_data(
            latent, features, weights, cfg.prevalence, cfg.vocab_size, cfg.concentration
        )
    elif cfg.task == "logistic":
        data = make_logistic_data(
            latent, features, weights, cfg.prevalence, cfg.vocab_size, cfg.concentration
        )
    elif cfg.task == "survival":
        data = survival_analysis_data(
            latent,
            features,
            weights,
            cfg.prevalence,
            cfg.gamma_shape,
            cfg.gamma_rate,
            cfg.vocab_size,
            cfg.concentration,
        )
    elif cfg.task == "mixture":
        data = mixture_cure_data(
            latent,
            features,
            weights,
            cfg.prevalence,
            cfg.gamma_shape,
            cfg.gamma_rate,
            cfg.vocab_size,
            cfg.concentration,
        )
    elif cfg.task == "competing_risk":
        data = competing_risk_data(
            latent,
            features,
            cfg.vocab_size,
            boundaries=boundaries,
        )
    elif cfg.task == "multi_event":
        data = multi_event_data(
            latent,
            features,
            cfg.vocab_size,
            boundaries=boundaries,
        )
    else:
        raise ValueError(
            f"Unknown task: {cfg.task}. Choose from "
            "'linear', 'logistic', 'survival', 'mixture', 'competing_risk', "
            "or 'multi_event'."
        )

    if "label" in data.keys():
        print("Label prevalence:", f"{data['label'].mean().item():.3f}")
    if "indicator" in data.keys():
        print("Indicator prevalence:", f"{data['indicator'].mean().item():.3f}")

    data = expand_constants(td=data, num_timepoints=cfg.num_timepoints)
    return data
