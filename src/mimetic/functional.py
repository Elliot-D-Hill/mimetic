from collections.abc import Sequence

import torch
import torch.distributions as dist
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor, arange, randn


def random_effects(
    data: TensorDict,
    hidden_dim: int,
    stds: Sequence[float],
    correlation: Tensor | float = 0.0,
) -> TensorDict:
    """Sample random effects γ_i from N(0, Q) (Fahrmeir et al., Eq. 7.11).

    When correlation is zero, samples each random effect independently.
    Otherwise, samples jointly from the q×q covariance Q.

    Reads: 'id' [N, 1, 1].
    Writes: 'gamma' [N, q, D].

    Args:
        hidden_dim: Dimensionality D of latent features.
        stds: Standard deviations for each random effect; len determines q.
        correlation: Off-diagonal correlation. Float gives compound symmetry,
            Tensor gives a user-provided [q, q] correlation matrix.
    """
    num_samples: int = data["id"].shape[0]
    q = len(stds)
    if isinstance(correlation, float) and correlation == 0.0:
        gamma = torch.cat(
            [randn(num_samples, 1, hidden_dim) * std for std in stds], dim=1
        )  # [N, q, D]
    else:
        from .covariance import make_random_effects_covariance

        Q = make_random_effects_covariance(stds, correlation)
        mvn = dist.MultivariateNormal(torch.zeros(q), Q)
        samples = mvn.sample((num_samples, hidden_dim))  # [N, D, q]
        gamma = samples.permute(0, 2, 1)  # [N, q, D]
    data["gamma"] = gamma
    return data


def observations(
    data: TensorDict,
    num_timepoints: int,
    observed_std: float,
    covariance: Tensor,
    num_fixed_effects: int = 0,
) -> TensorDict:
    """Generate GLMM observations y_i = X_i·β + U_i·γ_i + ε_i (Fahrmeir Box 7.1).

    Reads: 'gamma' [N, q, D].
    Writes: 'y' [N, T, D], 'U' [N, T, q], 'time' [N, T, 1].
        When num_fixed_effects > 0: 'X' [N, T, p], 'beta' [N, p, D].

    Args:
        num_timepoints: Number of time points T.
        observed_std: Residual standard deviation σ.
        covariance: Residual correlation matrix Σ [T, T].
        num_fixed_effects: Number of fixed-effect predictors p. Zero means no
            fixed effects (y = Uγ + ε).
    """
    gamma: Tensor = data["gamma"]  # [N, q, D]
    num_samples, q, hidden_dim = gamma.shape
    # U: polynomial basis [1, t, t², ..., t^(q-1)]
    t = arange(num_timepoints, dtype=torch.float32)  # [T]
    U = torch.stack([t**k for k in range(q)], dim=1)  # [T, q]
    U = U.unsqueeze(0).expand(num_samples, -1, -1)  # [N, T, q]
    random = torch.bmm(U, gamma)  # [N, T, D]
    # Fixed effects
    if num_fixed_effects > 0:
        p = num_fixed_effects
        X = randn(num_samples, num_timepoints, p)
        beta = randn(num_samples, p, hidden_dim)
        fixed = torch.bmm(X, beta)  # [N, T, D]
        data["X"] = X
        data["beta"] = beta
    else:
        fixed = 0.0
    # Residual noise
    scaled_cov = covariance * observed_std**2
    mvn = dist.MultivariateNormal(torch.zeros(num_timepoints), scaled_cov)
    noise = mvn.sample((num_samples, hidden_dim))  # [N, D, T]
    noise = noise.permute(0, 2, 1)  # [N, T, D]
    data["y"] = fixed + random + noise  # [N, T, D]
    data["U"] = U
    time_vals = arange(num_timepoints, dtype=torch.float32).view(1, -1, 1)
    data["time"] = time_vals.expand(num_samples, -1, -1)  # [N, T, 1]
    return data


def tokens(data: TensorDict, vocab_size: int, concentration: float = 1.0) -> TensorDict:
    """Tokenize observed features via Dirichlet-skewed softmax.

    Reads: 'y' [N, T, D].
    Writes: 'tokens' [N, T, 1].
    """
    features = data["y"]
    hidden_dim = features.shape[-1]
    weight = randn(vocab_size, hidden_dim)
    logits = F.linear(features, weight)
    prior = torch.ones(vocab_size) * concentration
    dirichlet = dist.Dirichlet(prior)
    skew = dirichlet.sample(logits.shape[:-1]).log()
    probs = F.softmax(logits + skew, dim=-1)
    tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1)  # [N*T, 1]
    data["tokens"] = tokens.view(*features.shape[:-1], 1)  # [N, T, 1]
    return data


def linear_predictor(data: TensorDict, weight: Tensor, prevalence: float) -> TensorDict:
    """Compute linear predictor η from random intercept (Fahrmeir Box 7.1).

    Reads: 'gamma' [N, q, D].
    Writes: 'eta' [N, 1, K], 'coefficients' [N, K, D], 'intercept' [N, 1, 1].
    """
    event_rate = weight.new_tensor(prevalence)
    event_rate = torch.log(event_rate / (1 - event_rate))
    output = F.linear(input=data["gamma"][:, 0:1, :], weight=weight, bias=event_rate)
    data["eta"] = output  # [N, 1, K]
    num_samples = data.batch_size[0]
    data["coefficients"] = weight.unsqueeze(0).expand(num_samples, -1, -1)  # [N, K, D]
    data["intercept"] = event_rate.view(1, 1, 1).expand(
        num_samples, -1, -1
    )  # [N, 1, 1]
    return data


def logistic_output(data: TensorDict) -> TensorDict:
    """Compute logistic probability and binary label from linear predictor.

    Reads: 'eta' [N, 1, K].
    Writes: 'probability' [N, 1, K], 'label' [N, 1, K].
    """
    probability = torch.sigmoid(data["eta"])
    data["probability"] = probability
    data["label"] = torch.bernoulli(probability)
    return data


def multiclass_output(data: TensorDict) -> TensorDict:
    """Compute multiclass probability and class label from linear predictor.

    Reads: 'eta' [N, 1, K].
    Writes: 'probability' [N, 1, K], 'label' [N, 1, 1].
    """
    probability = torch.softmax(data["eta"], dim=-1)
    data["probability"] = probability
    data["label"] = dist.Categorical(probs=probability).sample().unsqueeze(-1)
    return data


def ordinal_output(data: TensorDict, num_classes: int) -> TensorDict:
    """Compute ordinal probability and label via cumulative logit model.

    Reads: 'eta' [N, 1, 1].
    Writes: 'probability' [N, 1, K], 'label' [N, 1, 1].
    """
    output: Tensor = data["eta"]  # [N, 1, 1]
    # K-1 evenly spaced thresholds centered at 0
    thresholds = torch.linspace(-2, 2, num_classes - 1)  # [K-1]
    # Cumulative P(Y <= k) = sigmoid(threshold_k - output)
    cumulative = torch.sigmoid(thresholds - output)  # [N, 1, K-1]
    ones = torch.ones_like(output)
    cumulative = torch.cat([cumulative, ones], dim=-1)  # [N, 1, K]
    zeros = torch.zeros_like(output)
    cumulative_shifted = torch.cat([zeros, cumulative[..., :-1]], dim=-1)  # [N, 1, K]
    probs = cumulative - cumulative_shifted  # [N, 1, K]
    data["probability"] = probs
    data["label"] = dist.Categorical(probs=probs).sample().unsqueeze(-1)  # [N, 1, 1]
    return data


def event_time(data: TensorDict) -> TensorDict:
    """Sample event time from exponential distribution.

    Reads: 'eta'.
    Writes: 'event_time'.
    """
    rate = torch.exp(data["eta"])
    data["event_time"] = dist.Exponential(rate=rate).sample()
    return data


def mixture_cure_censoring(data: TensorDict) -> TensorDict:
    """Set event_time to infinity for cured individuals (label == 0).

    Reads: 'label', 'event_time'.
    Writes: modifies 'event_time' in-place.
    """
    data["event_time"][data["label"] == 0] = torch.inf
    return data


def observation_time(data: TensorDict, shape: float, rate: float) -> TensorDict:
    """Sample observation times from Gamma-distributed intervals.

    Reads: 'y' (for shape inference).
    Writes: 'time' [N, T, 1].
    """
    num_samples: int = data["y"].shape[0]
    num_timepoints: int = data["y"].shape[1]
    gamma_dist = dist.Gamma(shape, rate)
    time_intervals = gamma_dist.sample((num_samples, num_timepoints, 1))  # [N, T, 1]
    data["time"] = time_intervals.cumsum(dim=1)  # [N, T, 1]
    return data


def censor_time(data: TensorDict) -> TensorDict:
    """Sample uniform censor time between min and max observation time.

    Reads: 'time' [N, T, 1].
    Writes: 'censor_time' [N, 1, 1].
    """
    time: Tensor = data["time"]
    min_time = torch.amin(time, dim=1, keepdim=True)
    max_time = torch.amax(time, dim=1, keepdim=True)
    data["censor_time"] = dist.Uniform(min_time, max_time).sample()
    return data


def survival_indicators(data: TensorDict) -> TensorDict:
    """Compute survival analysis indicators from event, censor, and observation times.

    Reads: 'event_time', 'censor_time', 'time'.
    Writes: 'indicator', 'observed_time', 'time_to_event'.
    """
    event_time: Tensor = data["event_time"]
    censor_time: Tensor = data["censor_time"]
    time: Tensor = data["time"]
    indicator: Tensor = (event_time < censor_time).float()
    observed_time = torch.minimum(censor_time, event_time)
    data["indicator"] = indicator
    data["observed_time"] = observed_time
    data["time_to_event"] = observed_time - time
    return data


def competing_risks_events(data: TensorDict, vocab_size: int) -> TensorDict:
    """Generate competing risks events via Weibull distributions.

    For each position, samples potential failure times for all event types
    from feature-conditioned Weibull distributions. The observed event is
    the one with minimum failure time.

    Reads: 'y' [N, T, D].
    Writes: 'tokens' [N, T, 1], 'event_time' [N, T, 1], 'time' [N, T, 1],
            'failure_times' [N, T, K].
    """
    features: Tensor = data["y"]
    # Project features to Weibull log-parameters
    weight_alpha = randn(vocab_size, features.shape[-1])
    weight_beta = randn(vocab_size, features.shape[-1])
    log_alpha = F.linear(features, weight_alpha)
    log_beta = F.linear(features, weight_beta)
    eps = 1e-6  # Ensure positive parameters
    alpha = F.softplus(log_alpha) + eps  # scale > 0
    beta = F.softplus(log_beta) + eps  # shape > 0
    # Sample failure times for all event types
    weibull = dist.Weibull(alpha, beta)
    failure_times = weibull.rsample()  # [N, T, K]
    # Observed event is the minimum
    tte, tokens = failure_times.min(dim=-1, keepdim=True)  # [N, T, 1]
    data["tokens"] = tokens  # [N, T, 1]
    data["event_time"] = tte  # [N, T, 1]
    data["time"] = tte.cumsum(dim=1)  # [N, T, 1]
    data["failure_times"] = failure_times  # [N, T, K]
    return data


def discrete_event_time(data: TensorDict, boundaries: Tensor) -> TensorDict:
    """Discretize continuous event times into interval bins.

    Reads: 'event_time', 'indicator'.
    Writes: 'discrete_event_time'.
    """
    # B = len(boundaries) - 1 intervals; leading dims are [N,1,1] or [N,T,K]
    event_time: Tensor = data["event_time"]  # [*, K]
    indicator: Tensor = data["indicator"]  # [*, K]
    interval_start = boundaries[:-1].view(*([1] * event_time.dim()), -1)  # [1,..,1, B]
    interval_end = boundaries[1:].view(*([1] * event_time.dim()), -1)  # [1,..,1, B]
    interval_width = interval_end - interval_start  # [1,..,1, B]
    event_time = event_time.unsqueeze(-1)  # [*, K, 1]
    exposure: Tensor = ((event_time - interval_start) / interval_width).clamp(
        0, 1
    )  # [*, K, B]
    in_interval: Tensor = (event_time > interval_start) & (
        event_time <= interval_end
    )  # [*, K, B]
    indicator = indicator.unsqueeze(-1).to(exposure.dtype)  # [*, K, 1]
    event_duration: Tensor = indicator * (exposure * in_interval)  # [*, K, B]
    censored_duration: Tensor = (1.0 - indicator) * exposure  # [*, K, B]
    data["discrete_event_time"] = event_duration + censored_duration  # [*, K, B]
    return data


def competing_risk_indicators(data: TensorDict, vocab_size: int) -> TensorDict:
    """Compute one-hot indicators and expand event times for competing risks.

    Reads: 'tokens' [N, T, 1], 'event_time' [N, T, 1].
    Writes: 'indicator' [N, T, K], 'event_time' [N, T, K].
    """
    tensor_dtype = data["event_time"].dtype
    tokens: Tensor = data["tokens"].squeeze(-1)  # [N, T]
    indicator = F.one_hot(tokens, num_classes=vocab_size)  # [N, T, K]
    data["indicator"] = indicator.to(tensor_dtype)
    # Same observed time for all K; competing risks observe only the winning event
    data["event_time"] = data["event_time"].expand(-1, -1, vocab_size)  # [N, T, K]
    return data


def multi_event_times(
    data: TensorDict, vocab_size: int, horizon: float | Tensor
) -> TensorDict:
    """Compute per-event TTE with a sliding horizon window.

    For each event type, finds the next occurrence time via suffix-min
    over the observation timeline, then clips to a horizon.

    Reads: 'tokens' [N, T, 1], 'time' [N, T, 1].
    Writes: 'event_time' [N, T, K], 'indicator' [N, T, K].
    """
    time: Tensor = data["time"]  # [N, T, 1]
    tokens: Tensor = data["tokens"].squeeze(-1)  # [N, T] (required by F.one_hot)
    event_mask = F.one_hot(tokens, num_classes=vocab_size).to(torch.bool)
    event_times: Tensor = time.expand(-1, -1, vocab_size)  # [N, T, 1] -> [N, T, K]
    event_times = event_times.masked_fill(~event_mask, torch.inf)
    # Suffix min, then shift forward by 1 to get strictly next occurrence
    event_times_reversed = torch.flip(event_times, dims=[1])
    suffix_min = torch.flip(torch.cummin(event_times_reversed, dim=1).values, dims=[1])
    next_time = F.pad(suffix_min[:, 1:], (0, 0, 0, 1), value=torch.inf)
    event_time: Tensor = next_time - time  # [N, T, K] - [N, T, 1] broadcasts
    if horizon is None:
        horizon = event_time[event_time.isfinite()].max()
    data["event_time"] = event_time.clamp(max=horizon)
    data["indicator"] = (event_time <= horizon).to(data["event_time"].dtype)
    return data
