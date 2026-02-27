from tensordict import TensorDict
from torch import Tensor

from .pipeline import (
    add_censor_time,
    add_competing_risk_indicators,
    add_competing_risks_events,
    add_discrete_event_time,
    add_event_time,
    add_linear_output,
    add_logistic_output,
    add_mixture_cure_censoring,
    add_multi_event_times,
    add_survival_indicators,
    add_time,
    add_tokens,
)


def linear_data(
    data: TensorDict,
    weights: Tensor,
    prevalence: float,
    vocab_size: int = 1000,
    concentration: float = 1.0,
) -> TensorDict:
    """Generate linear regression data."""
    data = add_linear_output(data, weights, prevalence)
    data = add_tokens(data, vocab_size, concentration)
    return data


def logistic_data(
    data: TensorDict,
    weights: Tensor,
    prevalence: float,
    vocab_size: int = 1000,
    concentration: float = 1.0,
) -> TensorDict:
    """Generate logistic regression data."""
    data = linear_data(data, weights, prevalence, vocab_size, concentration)
    data = add_logistic_output(data)
    return data


def survival_data(
    data: TensorDict,
    weights: Tensor,
    prevalence: float,
    gamma_shape: float,
    gamma_rate: float,
    vocab_size: int = 1000,
    concentration: float = 1.0,
) -> TensorDict:
    """Generate survival analysis data."""
    data = linear_data(data, weights, prevalence, vocab_size, concentration)
    data = add_event_time(data)
    data = add_time(data, gamma_shape, gamma_rate)
    data = add_censor_time(data)
    data = add_survival_indicators(data)
    return data


def mixture_cure_data(
    data: TensorDict,
    weights: Tensor,
    prevalence: float,
    gamma_shape: float,
    gamma_rate: float,
    vocab_size: int = 1000,
    concentration: float = 1.0,
) -> TensorDict:
    """Generate mixture cure model data."""
    data = logistic_data(data, weights, prevalence, vocab_size, concentration)
    data = add_event_time(data)
    data = add_mixture_cure_censoring(data)
    data = add_time(data, gamma_shape, gamma_rate)
    data = add_censor_time(data)
    data = add_survival_indicators(data)
    return data


def competing_risk_data(
    data: TensorDict,
    vocab_size: int,
    boundaries: Tensor | None = None,
) -> TensorDict:
    """Generate competing risks TTE data for self-supervised pretraining."""
    data = add_competing_risks_events(data, vocab_size)
    data = add_competing_risk_indicators(data, vocab_size)
    if boundaries is not None:
        data = add_discrete_event_time(data, boundaries)
    return data


def multi_event_data(
    data: TensorDict,
    vocab_size: int,
    boundaries: Tensor | None = None,
) -> TensorDict:
    """Generate per-event TTE data with a sliding horizon window."""
    data = add_competing_risks_events(data, vocab_size)
    horizon = boundaries[-1] if boundaries is not None else None
    data = add_multi_event_times(data, vocab_size, horizon)
    if boundaries is not None:
        data = add_discrete_event_time(data, boundaries)
    return data
