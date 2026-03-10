import torch
import torch.distributions as dist
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor


def add_latent_features(
    data: TensorDict, hidden_dim: int, latent_std: float
) -> TensorDict:
    """Sample latent features from a multivariate normal.

    Reads: batch_size (for num_samples).
    Writes: 'latent' [N, 1, D].
    """
    num_samples = data["id"].shape[0]
    zero = torch.zeros(hidden_dim)
    covariance = torch.eye(hidden_dim) * latent_std**2
    mvn = dist.MultivariateNormal(zero, covariance)
    data["latent"] = mvn.sample((num_samples, 1))  # [N, 1, D]
    return data


def add_observed_features(
    data: TensorDict, num_timepoints: int, observed_std: float, covariance: Tensor
) -> TensorDict:
    """Generate observed features with temporal correlation.

    Reads: 'latent' [N, 1, D].
    Writes: 'features' [N, T, D].
    """
    latent = data["latent"]  # [N, 1, D]
    covariance = covariance * observed_std**2
    num_samples, _, hidden_dim = latent.shape
    mvn = dist.MultivariateNormal(torch.zeros(num_timepoints), covariance)
    noise = mvn.sample((num_samples, hidden_dim))  # [N, D, T]
    noise = noise.permute(0, 2, 1)  # [N, T, D]
    data["features"] = latent + noise  # [N, 1, D] + [N, T, D] -> [N, T, D]
    return data


def add_tokens(
    data: TensorDict, vocab_size: int, concentration: float = 1.0
) -> TensorDict:
    """Tokenize observed features via Dirichlet-skewed softmax.

    Reads: 'features' [N, T, D].
    Writes: 'tokens' [N, T, 1].
    """
    features = data["features"]
    hidden_dim = features.shape[-1]
    weight = torch.randn(vocab_size, hidden_dim)
    logits = F.linear(features, weight)
    prior = torch.ones(vocab_size) * concentration
    dirichlet = dist.Dirichlet(prior)
    skew = dirichlet.sample(logits.shape[:-1]).log()
    probs = F.softmax(logits + skew, dim=-1)
    tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1)  # [N*T, 1]
    data["tokens"] = tokens.view(*features.shape[:-1], 1)  # [N, T, 1]
    return data


def add_linear_output(data: TensorDict, weight: Tensor, bias: float) -> TensorDict:
    """Compute linear output from latent features.

    Reads: 'latent' [N, 1, D].
    Writes: 'output' [N, 1, K] where K = weight.size(0).
    """
    event_rate = weight.new_tensor(bias)
    event_rate = torch.log(event_rate / (1 - event_rate))
    output = F.linear(input=data["latent"], weight=weight, bias=event_rate)
    data["output"] = output  # [N, 1, 1]
    return data


def add_logistic_output(data: TensorDict) -> TensorDict:
    """Compute logistic probability and binary label from linear output.

    Reads: 'output' [N, 1, K].
    Writes: 'probability' [N, 1, K], 'label' [N, 1, K].
    """
    probability = torch.sigmoid(data["output"])
    data["probability"] = probability
    data["label"] = torch.bernoulli(probability)
    return data


def add_multiclass_output(data: TensorDict) -> TensorDict:
    """Compute multiclass probability and class label from linear output.

    Reads: 'output' [N, 1, K].
    Writes: 'probability' [N, 1, K], 'label' [N, 1, 1].
    """
    probability = torch.softmax(data["output"], dim=-1)
    data["probability"] = probability
    data["label"] = dist.Categorical(probs=probability).sample().unsqueeze(-1)
    return data


def add_event_time(data: TensorDict) -> TensorDict:
    """Sample event time from exponential distribution.

    Reads: 'output'.
    Writes: 'event_time'.
    """
    rate = torch.exp(data["output"])
    data["event_time"] = dist.Exponential(rate=rate).sample()
    return data


def add_mixture_cure_censoring(data: TensorDict) -> TensorDict:
    """Set event_time to infinity for cured individuals (label == 0).

    Reads: 'label', 'event_time'.
    Writes: modifies 'event_time' in-place.
    """
    data["event_time"][data["label"] == 0] = torch.inf
    return data


def add_time(data: TensorDict, gamma_shape: float, gamma_rate: float) -> TensorDict:
    """Sample observation times from Gamma-distributed intervals.

    Reads: 'features' (for shape inference).
    Writes: 'time' [N, T, 1].
    """
    num_samples = data["features"].shape[0]
    num_timepoints = data["features"].shape[1]
    gamma = dist.Gamma(gamma_shape, gamma_rate)
    time_intervals = gamma.sample((num_samples, num_timepoints, 1))  # [N, T, 1]
    data["time"] = time_intervals.cumsum(dim=1)  # [N, T, 1]
    return data


def add_censor_time(data: TensorDict) -> TensorDict:
    """Sample uniform censor time between min and max observation time.

    Reads: 'time' [N, T, 1].
    Writes: 'censor_time' [N, 1, 1].
    """
    time = data["time"]
    min_time = torch.amin(time, dim=1, keepdim=True)
    max_time = torch.amax(time, dim=1, keepdim=True)
    data["censor_time"] = dist.Uniform(min_time, max_time).sample()
    return data


def add_survival_indicators(data: TensorDict) -> TensorDict:
    """Compute survival analysis indicators from event, censor, and observation times.

    Reads: 'event_time', 'censor_time', 'time'.
    Writes: 'indicator', 'observed_time', 'time_to_event'.
    """
    event_time = data["event_time"]
    censor_time = data["censor_time"]
    time = data["time"]
    indicator = (event_time < censor_time).float()
    observed_time = torch.minimum(censor_time, event_time)
    data["indicator"] = indicator
    data["observed_time"] = observed_time
    data["time_to_event"] = observed_time - time
    return data


def add_competing_risks_events(data: TensorDict, vocab_size: int) -> TensorDict:
    """Generate competing risks events via Weibull distributions.

    For each position, samples potential failure times for all event types
    from feature-conditioned Weibull distributions. The observed event is
    the one with minimum failure time.

    Reads: 'features' [N, T, D].
    Writes: 'tokens' [N, T, 1], 'event_time' [N, T, 1], 'time' [N, T, 1],
            'failure_times' [N, T, K].
    """
    features = data["features"]
    # Project features to Weibull log-parameters
    weight_alpha = torch.randn(vocab_size, features.shape[-1])
    weight_beta = torch.randn(vocab_size, features.shape[-1])
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


def add_discrete_event_time(data: TensorDict, boundaries: Tensor) -> TensorDict:
    """Discretize continuous event times into interval bins.

    Reads: 'event_time', 'indicator'.
    Writes: 'discrete_event_time'.
    """
    event_time = data["event_time"]
    indicator = data["indicator"]
    interval_start = boundaries[:-1].view(*([1] * event_time.dim()), -1)
    interval_end = boundaries[1:].view(*([1] * event_time.dim()), -1)
    interval_width = interval_end - interval_start
    event_time = event_time.unsqueeze(-1)
    exposure = ((event_time - interval_start) / interval_width).clamp(0, 1)
    in_interval = (event_time > interval_start) & (event_time <= interval_end)
    indicator = indicator.unsqueeze(-1).to(exposure.dtype)
    event_duration = indicator * (exposure * in_interval)
    censored_duration = (1.0 - indicator) * exposure
    data["discrete_event_time"] = event_duration + censored_duration
    return data


def add_competing_risk_indicators(data: TensorDict, vocab_size: int) -> TensorDict:
    """Compute one-hot indicators and expand event times for competing risks.

    Reads: 'tokens' [N, T, 1], 'event_time' [N, T, 1].
    Writes: 'indicator' [N, T, K], 'event_time' [N, T, K].
    """
    dtype = data["event_time"].dtype
    tokens = data["tokens"].squeeze(-1)  # [N, T]
    indicator = F.one_hot(tokens, num_classes=vocab_size).to(dtype)  # [N, T, K]
    data["indicator"] = indicator
    data["event_time"] = data["event_time"].expand(-1, -1, vocab_size)  # [N, T, K]
    return data


def add_multi_event_times(
    data: TensorDict, vocab_size: int, horizon: float | Tensor
) -> TensorDict:
    """Compute per-event TTE with a sliding horizon window.

    For each event type, finds the next occurrence time via suffix-min
    over the observation timeline, then clips to a horizon.

    Reads: 'tokens' [N, T, 1], 'time' [N, T, 1].
    Writes: 'event_time' [N, T, K], 'indicator' [N, T, K].
    """
    time = data["time"]  # [N, T, 1]
    tokens = data["tokens"].squeeze(-1)  # [N, T] (required by F.one_hot)
    event_mask = F.one_hot(tokens, num_classes=vocab_size).to(torch.bool)
    event_times = time.expand(-1, -1, vocab_size)  # [N, T, 1] -> [N, T, K]
    event_times = event_times.masked_fill(~event_mask, torch.inf)
    # Suffix min, then shift forward by 1 to get strictly next occurrence
    event_times_reversed = torch.flip(event_times, dims=[1])
    suffix_min = torch.flip(torch.cummin(event_times_reversed, dim=1).values, dims=[1])
    next_time = F.pad(suffix_min[:, 1:], (0, 0, 0, 1), value=torch.inf)
    event_time = next_time - time  # [N, T, K] - [N, T, 1] broadcasts
    if horizon is None:
        horizon = event_time[event_time.isfinite()].max()
    data["event_time"] = event_time.clamp(max=horizon)
    data["indicator"] = (event_time <= horizon).to(data["event_time"].dtype)
    return data
