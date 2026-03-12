from collections.abc import Sequence

import torch
import torch.distributions as dist
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor, arange, randn

from .covariance import random_effects_covariance
from .states import (
    CensoredState,
    EffectsState,
    EventTimeState,
    LabeledState,
    ObservedState,
    SurvivalState,
    TokenizedState,
)

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def random_effects(
    num_samples: int, stds: Sequence[float], correlation: Tensor | float = 0.0
) -> EffectsState:
    """Sample random effects from N(0, Q) (Fahrmeir et al., Eq. 7.11).

    Args:
        num_samples: Number of subjects N.
        stds: Standard deviations for each random effect; len determines q.
        correlation: Off-diagonal correlation. Float gives compound symmetry,
            Tensor gives a user-provided [q, q] correlation matrix.
    """
    q = len(stds)
    Q = random_effects_covariance(stds, correlation)  # [q, q]
    mvn = dist.MultivariateNormal(torch.zeros(q), Q)
    samples = mvn.sample((num_samples,))  # [N, q]
    gamma = samples.unsqueeze(-1)  # [N, q, 1]
    return EffectsState(gamma=gamma)


def observations(
    state: EffectsState,
    num_timepoints: int,
    num_features: int,
    observed_std: float,
    covariance: Tensor,
) -> ObservedState:
    """Generate GLMM observations y_i = X_i*beta + U_i*gamma_i + eps_i (Fahrmeir Box 7.1).

    Also stores eta = X*beta + U*gamma (the linear predictor before noise).

    Args:
        num_timepoints: Number of time points T.
        num_features: Number of design matrix features p. X is always generated.
        observed_std: Residual standard deviation sigma.
        covariance: Residual correlation matrix Sigma [T, T].
    """
    gamma = state["gamma"]  # [N, q, 1]
    num_samples, q, _ = gamma.shape
    t = arange(num_timepoints, dtype=torch.float32)  # [T]
    U = torch.vander(t, N=q, increasing=True)  # [T, q]
    U = U.unsqueeze(0).expand(num_samples, -1, -1)  # [N, T, q]
    random_effects = torch.bmm(U, gamma)  # [N, T, 1]
    X = randn(num_samples, num_timepoints, num_features)  # [N, T, p]
    beta = randn(num_samples, num_features, 1)  # [N, p, 1]
    fixed_effects = torch.bmm(X, beta)  # [N, T, 1]
    eta = fixed_effects + random_effects  # [N, T, 1]
    scaled_cov = covariance * observed_std**2
    mvn = dist.MultivariateNormal(torch.zeros(num_timepoints), scaled_cov)
    noise = mvn.sample((num_samples, 1))  # [N, 1, T]
    noise = noise.permute(0, 2, 1)  # [N, T, 1]
    y = eta + noise  # [N, T, 1]
    time_values = arange(num_timepoints, dtype=torch.float32).view(1, -1, 1)
    time = time_values.expand(num_samples, -1, -1)  # [N, T, 1]
    return ObservedState(**state, y=y, U=U, time=time, eta=eta, X=X, beta=beta)


def tokens(
    state: ObservedState, vocab_size: int, concentration: float = 1.0
) -> TokenizedState:
    """Tokenize design matrix via Dirichlet-skewed softmax.

    Produces 'tokens' [N, T, 1] from 'X' [N, T, p].
    """
    tokens = _compute_tokens(state["X"], vocab_size, concentration)
    return TokenizedState(**state, tokens=tokens)


def logistic_output(state: ObservedState, prevalence: float = 0.5) -> LabeledState:
    """Compute logistic probability and binary label from linear predictor.

    Reads 'eta' [N, T, 1]. Produces 'probability' [N, T, 1], 'label' [N, T, 1].

    Args:
        prevalence: Base rate used to shift eta before applying sigmoid.
    """
    shift = torch.logit(torch.tensor(prevalence))
    probability = torch.sigmoid(state["eta"] + shift)  # [N, T, 1]
    label = torch.bernoulli(probability)  # [N, T, 1]
    return LabeledState(**state, probability=probability, label=label)


def multiclass_output(state: ObservedState) -> LabeledState:
    """Compute multiclass probability and class label from linear predictor.

    Reads 'eta' [N, T, 1]. Produces 'probability' [N, T, K], 'label' [N, T, 1].
    """
    probability = torch.softmax(state["eta"], dim=-1)
    label = dist.Categorical(probs=probability).sample().unsqueeze(-1)
    return LabeledState(**state, probability=probability, label=label)


def ordinal_output(
    state: ObservedState, num_classes: int, start: float = -2.0, end: float = 2.0
) -> LabeledState:
    """Compute ordinal probability and label via cumulative logit model.

    Reads 'eta' [N, T, 1]. Produces 'probability' [N, T, K], 'label' [N, T, 1].
    """
    eta = state["eta"]  # [N, T, 1]
    thresholds = torch.linspace(start, end, num_classes - 1)  # [K-1]
    cumulative = torch.sigmoid(thresholds - eta)  # [N, T, K-1]
    ones = torch.ones_like(eta)
    cumulative = torch.cat([cumulative, ones], dim=-1)  # [N, T, K]
    zeros = torch.zeros_like(eta)
    cumulative_shifted = torch.cat([zeros, cumulative[..., :-1]], dim=-1)  # [N, T, K]
    probs = cumulative - cumulative_shifted  # [N, T, K]
    label = dist.Categorical(probs=probs).sample().unsqueeze(-1)  # [N, T, 1]
    return LabeledState(**state, probability=probs, label=label)


def replace_observation_time(
    state: ObservedState, shape: float, rate: float
) -> ObservedState:
    """Replace observation times with Gamma-distributed intervals.

    Produces 'time' [N, T, 1].
    """
    num_samples, num_timepoints = state["y"].shape[0], state["y"].shape[1]
    gamma_dist = dist.Gamma(shape, rate)
    time_intervals = gamma_dist.sample((num_samples, num_timepoints, 1))  # [N, T, 1]
    time = time_intervals.cumsum(dim=1)  # [N, T, 1]
    result = state.copy()
    result["time"] = time
    return result


# ---------------------------------------------------------------------------
# Survival pipeline
# ---------------------------------------------------------------------------


def event_time(state: ObservedState) -> EventTimeState:
    """Sample event time from exponential distribution.

    Aggregates eta to subject level: mean over timepoints -> single event time.
    """
    subject_eta = state["eta"].mean(dim=1, keepdim=True)  # [N, 1, 1]
    rate = torch.exp(subject_eta)
    sampled = dist.Exponential(rate=rate).sample()  # [N, 1, 1]
    return EventTimeState(**state, event_time=sampled)


def mixture_cure_censoring(state: EventTimeState) -> EventTimeState:
    """Set event_time to infinity for cured individuals (label == 0 at all timepoints).

    Reads 'label' [N, T, 1], 'event_time' [N, 1, 1]. Modifies 'event_time'.
    """
    label = state.get("label")
    assert label is not None
    cured = (label == 0).all(dim=1, keepdim=True)  # [N, 1, 1]
    event_time = state["event_time"].clone()  # [N, 1, 1]
    event_time[cured.expand_as(event_time)] = torch.inf
    result = state.copy()
    result["event_time"] = event_time
    return result


def replace_survival_observation_time(
    state: EventTimeState, shape: float, rate: float
) -> EventTimeState:
    """Replace observation times with Gamma-distributed intervals for survival data."""
    num_samples, num_timepoints = state["y"].shape[0], state["y"].shape[1]
    gamma_dist = dist.Gamma(shape, rate)
    time_intervals = gamma_dist.sample((num_samples, num_timepoints, 1))  # [N, T, 1]
    time = time_intervals.cumsum(dim=1)  # [N, T, 1]
    result = state.copy()
    result["time"] = time
    return result


def censor_time(state: EventTimeState) -> CensoredState:
    """Sample uniform censor time between min and max observation time.

    Reads 'time' [N, T, 1]. Produces 'censor_time' [N, 1, 1].
    """
    time = state["time"]
    min_time = torch.amin(time, dim=1, keepdim=True)
    max_time = torch.amax(time, dim=1, keepdim=True)
    ct = dist.Uniform(min_time, max_time).sample()
    return CensoredState(**state, censor_time=ct)


def survival_indicators(state: CensoredState) -> SurvivalState:
    """Compute survival analysis indicators from event, censor, and observation times.

    Produces 'indicator' [N, 1, 1], 'observed_time' [N, 1, 1], 'time_to_event' [N, T, 1].
    """
    indicator = (state["event_time"] < state["censor_time"]).float()  # [N, 1, 1]
    observed_time = torch.minimum(
        state["censor_time"], state["event_time"]
    )  # [N, 1, 1]
    time_to_event = observed_time - state["time"]  # [N, T, 1]
    return SurvivalState(
        **state,
        indicator=indicator,
        observed_time=observed_time,
        time_to_event=time_to_event,
    )


# ---------------------------------------------------------------------------
# Competing risks (TensorDict utilities — complex branching, less common)
# ---------------------------------------------------------------------------


def competing_risks_events(data: TensorDict, vocab_size: int) -> TensorDict:
    """Generate competing risks events via Weibull distributions.

    Reads: 'X' [N, T, p].
    Writes: 'tokens' [N, T, 1], 'event_time' [N, T, 1], 'time' [N, T, 1],
            'failure_times' [N, T, K].
    """
    features: Tensor = data["X"]
    weight_alpha = randn(vocab_size, features.shape[-1])
    weight_beta = randn(vocab_size, features.shape[-1])
    log_alpha = F.linear(features, weight_alpha)
    log_beta = F.linear(features, weight_beta)
    eps = 1e-6
    alpha = F.softplus(log_alpha) + eps
    beta = F.softplus(log_beta) + eps
    weibull = dist.Weibull(alpha, beta)
    failure_times = weibull.rsample()  # [N, T, K]
    event_time, tokens = failure_times.min(dim=-1, keepdim=True)  # [N, T, 1]
    data["tokens"] = tokens
    data["event_time"] = event_time
    data["time"] = event_time.cumsum(dim=1)
    data["failure_times"] = failure_times
    return data


def discrete_event_time(data: TensorDict, boundaries: Tensor) -> TensorDict:
    """Discretize continuous event times into interval bins.

    Reads: 'event_time', 'indicator'.
    Writes: 'discrete_event_time'.
    """
    event_time: Tensor = data["event_time"]
    indicator: Tensor = data["indicator"]
    interval_start = boundaries[:-1].view(*([1] * event_time.dim()), -1)
    interval_end = boundaries[1:].view(*([1] * event_time.dim()), -1)
    interval_width = interval_end - interval_start
    event_time = event_time.unsqueeze(-1)
    exposure: Tensor = ((event_time - interval_start) / interval_width).clamp(0, 1)
    in_interval: Tensor = (event_time > interval_start) & (event_time <= interval_end)
    indicator = indicator.unsqueeze(-1).to(exposure.dtype)
    event_duration: Tensor = indicator * (exposure * in_interval)
    censored_duration: Tensor = (1.0 - indicator) * exposure
    data["discrete_event_time"] = event_duration + censored_duration
    return data


def competing_risk_indicators(data: TensorDict, vocab_size: int) -> TensorDict:
    """Compute one-hot indicators and expand event times for competing risks.

    Reads: 'tokens' [N, T, 1], 'event_time' [N, T, 1].
    Writes: 'indicator' [N, T, K], 'event_time' [N, T, K].
    """
    tensor_dtype = data["event_time"].dtype
    tokens: Tensor = data["tokens"].squeeze(-1)
    indicator = F.one_hot(tokens, num_classes=vocab_size)
    data["indicator"] = indicator.to(tensor_dtype)
    data["event_time"] = data["event_time"].expand(-1, -1, vocab_size)
    return data


def multi_event_times(
    data: TensorDict, vocab_size: int, horizon: float | Tensor
) -> TensorDict:
    """Compute per-event TTE with a sliding horizon window.

    Reads: 'tokens' [N, T, 1], 'time' [N, T, 1].
    Writes: 'event_time' [N, T, K], 'indicator' [N, T, K].
    """
    time: Tensor = data["time"]
    tokens: Tensor = data["tokens"].squeeze(-1)
    event_mask = F.one_hot(tokens, num_classes=vocab_size).to(torch.bool)
    event_times: Tensor = time.expand(-1, -1, vocab_size)
    event_times = event_times.masked_fill(~event_mask, torch.inf)
    event_times_reversed = torch.flip(event_times, dims=[1])
    suffix_min = torch.flip(torch.cummin(event_times_reversed, dim=1).values, dims=[1])
    next_time = F.pad(suffix_min[:, 1:], (0, 0, 0, 1), value=torch.inf)
    event_time: Tensor = next_time - time
    if horizon is None:
        horizon = event_time[event_time.isfinite()].max()
    data["event_time"] = event_time.clamp(max=horizon)
    data["indicator"] = (event_time <= horizon).to(data["event_time"].dtype)
    return data


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_tokens(X: Tensor, vocab_size: int, concentration: float) -> Tensor:
    """Compute token IDs from design matrix via Dirichlet-skewed softmax."""
    p = X.shape[-1]
    weight = randn(vocab_size, p)  # [K, p]
    logits = F.linear(X, weight)  # [N, T, K]
    prior = torch.ones(vocab_size) * concentration  # [K]
    dirichlet = dist.Dirichlet(prior)
    skew = dirichlet.sample(logits.shape[:-1]).log()  # [N, T, K]
    probs = F.softmax(logits + skew, dim=-1)  # [N, T, K]
    flat = torch.multinomial(probs.view(-1, probs.size(-1)), 1)  # [N*T, 1]
    return flat.view(*X.shape[:-1], 1)  # [N, T, 1]
