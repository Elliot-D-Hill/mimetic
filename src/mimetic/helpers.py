import torch
import torch.distributions as dist
import torch.nn.functional as F
from torch import Tensor


def make_parameters(parameters: int | list[float], scale: float) -> Tensor:
    match parameters:
        case int():
            mean = torch.randn(parameters)
        case list():
            mean = torch.tensor(parameters)
    return mean.unsqueeze(0) * scale


def make_linear_output(input: Tensor, weight: Tensor, bias: float) -> Tensor:
    event_rate = input.new_tensor(bias)
    event_rate = torch.log(event_rate / (1 - event_rate))
    return F.linear(input=input, weight=weight, bias=event_rate)


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
