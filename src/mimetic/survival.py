import torch
import torch.distributions as dist

from .states import CensoredState, EventTimeState, ObservedState, SurvivalState


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
