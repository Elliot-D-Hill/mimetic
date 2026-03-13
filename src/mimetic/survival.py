import torch
import torch.distributions as dist

import torch.nn.functional as F
from torch import Tensor

from .states import (
    CensoredState,
    CompetingRisksState,
    DiscreteRiskState,
    EventTimeState,
    ObservedState,
    PredictorState,
    RiskIndicatorState,
    SurvivalState,
)


def event_time(state: ObservedState) -> EventTimeState:
    """Sample event time from exponential distribution.

    Aggregates eta to subject level: mean over timepoints -> single event time.
    """
    subject_eta = state["eta"].mean(dim=1, keepdim=True)  # [N, 1, 1]
    rate = torch.exp(subject_eta)
    sampled = dist.Exponential(rate=rate).sample()  # [N, 1, 1]
    return EventTimeState(**state, event_time=sampled)


def mixture_cure_censoring(state: EventTimeState) -> EventTimeState:
    """Set event_time to infinity for cured individuals (y == 0 at all timepoints).

    Reads 'y' [N, T, 1], 'event_time' [N, 1, 1]. Modifies 'event_time'.
    """
    y = state["y"]
    cured = (y == 0).all(dim=1, keepdim=True)  # [N, 1, 1]
    event_time = state["event_time"].clone()  # [N, 1, 1]
    event_time[cured.expand_as(event_time)] = torch.inf
    result = state.copy()
    result["event_time"] = event_time
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
# Competing risks
# ---------------------------------------------------------------------------


def competing_risks(
    state: PredictorState, shape: float | Tensor = 1.0
) -> CompetingRisksState:
    """Sample per-risk failure times from Weibull(scale, shape).

    Each column of eta [N, T, K] parameterizes one risk's Weibull scale.

    Args
    ----
    shape
        Weibull shape parameter; 1.0 gives Exponential.
    """
    eps = 1e-6
    scale = F.softplus(state["eta"]) + eps  # [N, T, K]
    failure_times = dist.Weibull(scale, shape).rsample()  # [N, T, K]
    tokens = failure_times.argmin(dim=-1, keepdim=True)  # [N, T, 1]
    return CompetingRisksState(**state, failure_times=failure_times, tokens=tokens)


def risk_indicators(state: CompetingRisksState) -> RiskIndicatorState:
    """One-hot encode the winning risk and broadcast min failure time to [N, T, K]."""
    failure_times = state["failure_times"]  # [N, T, K]
    K = failure_times.shape[-1]
    tokens = state["tokens"].squeeze(-1)  # [N, T]
    indicator = F.one_hot(tokens, num_classes=K).to(failure_times.dtype)  # [N, T, K]
    event_time = failure_times.min(dim=-1, keepdim=True).values.expand(
        -1, -1, K
    )  # [N, T, K]
    return RiskIndicatorState(**state, indicator=indicator, event_time=event_time)


def multi_event(
    state: CompetingRisksState, horizon: float | Tensor | None = None
) -> RiskIndicatorState:
    """Compute per-risk TTE via suffix-minimum over the observation schedule.

    Args
    ----
    horizon
        Clamp ceiling for event times; inferred from data if omitted.
    """
    time = state["time"]  # [N, T, 1]
    K = state["failure_times"].shape[-1]
    tokens = state["tokens"].squeeze(-1)  # [N, T]
    event_mask = F.one_hot(tokens, num_classes=K).to(torch.bool)  # [N, T, K]
    event_times = time.expand(-1, -1, K)  # [N, T, K]
    event_times = event_times.masked_fill(~event_mask, torch.inf)
    reversed_ = torch.flip(event_times, dims=[1])
    suffix_min = torch.flip(torch.cummin(reversed_, dim=1).values, dims=[1])
    next_time = F.pad(suffix_min[:, 1:], (0, 0, 0, 1), value=torch.inf)
    et = next_time - time  # [N, T, K]
    if horizon is None:
        horizon = et[et.isfinite()].max()
    et = et.clamp(max=horizon)
    indicator = (et < horizon).to(et.dtype)  # [N, T, K]
    return RiskIndicatorState(**state, event_time=et, indicator=indicator)


def discretize_risk(state: RiskIndicatorState, boundaries: Tensor) -> DiscreteRiskState:
    """Discretize continuous event times into interval bins.

    Args
    ----
    boundaries
        Monotonic bin edges [J+1]; produces J intervals.
    """
    et = state["event_time"]  # [N, T, K]
    indicator = state["indicator"]  # [N, T, K]
    interval_start = boundaries[:-1].view(*([1] * et.dim()), -1)  # [1, 1, 1, J]
    interval_end = boundaries[1:].view(*([1] * et.dim()), -1)
    interval_width = interval_end - interval_start
    et_expanded = et.unsqueeze(-1)  # [N, T, K, 1]
    exposure = ((et_expanded - interval_start) / interval_width).clamp(0, 1)
    in_interval = (et_expanded > interval_start) & (et_expanded <= interval_end)
    ind = indicator.unsqueeze(-1).to(exposure.dtype)  # [N, T, K, 1]
    event_duration = ind * (exposure * in_interval)
    censored_duration = (1.0 - ind) * exposure
    return DiscreteRiskState(
        **state,
        discrete_event_time=event_duration + censored_duration,  # [N, T, K, J]
    )
