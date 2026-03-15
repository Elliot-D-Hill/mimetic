"""Survival analysis and competing-risks simulation primitives."""

import torch
import torch.distributions as dist
import torch.nn.functional as F
from torch import Tensor

from .states import (
    CensoredState,
    CompetingRisksState,
    DiscreteRiskState,
    EventProcessState,
    EventTimeState,
    ObservedState,
    PredictorState,
    RiskIndicatorState,
    SurvivalState,
)
from .types import PositiveFloat, UnitInterval


def event_time(state: ObservedState) -> EventTimeState:
    """
    Sample event time from an exponential distribution.

    Aggregates ``eta`` to subject level via mean over timepoints, then
    samples a single event time per subject.

    Parameters
    ----------
    state : ObservedState
        Must contain ``eta`` [N, T, 1].

    Returns
    -------
    EventTimeState
        Adds ``event_time`` [N, 1, 1].

    See Also
    --------
    censor_time : Next step in the survival pipeline.
    competing_risks : Alternative multi-risk event model.

    Notes
    -----
    .. math:: t^* \\sim \\text{Exp}(\\exp(\\bar{\\eta}))

    Examples
    --------
    >>> import torch
    >>> from simulacra import linear_predictor, gaussian, event_time
    >>> obs = gaussian(linear_predictor(2, 3, 4), 1.0, torch.eye(3))
    >>> result = event_time(obs)
    >>> result["event_time"].shape
    torch.Size([2, 1, 1])
    """
    subject_eta = state["eta"].mean(dim=1, keepdim=True)  # [N, 1, 1]
    rate = torch.exp(subject_eta)
    sampled = dist.Exponential(rate=rate).sample()  # [N, 1, 1]
    return EventTimeState(**state, event_time=sampled)


def mixture_cure_censoring(state: EventTimeState) -> EventTimeState:
    """
    Set event time to infinity for cured subjects (y == 0 everywhere).

    Parameters
    ----------
    state : EventTimeState
        Must contain ``y`` [N, T, 1] and ``event_time`` [N, 1, 1].

    Returns
    -------
    EventTimeState
        Same state with ``event_time`` set to inf where cured.

    See Also
    --------
    event_time : Produce the initial event times.
    censor_time : Next step after cure adjustment.

    Examples
    --------
    >>> import torch
    >>> from simulacra import linear_predictor, bernoulli, event_time
    >>> from simulacra.survival import mixture_cure_censoring
    >>> obs = bernoulli(linear_predictor(2, 3, 4), prevalence=0.01)
    >>> result = mixture_cure_censoring(event_time(obs))
    >>> result["event_time"].shape
    torch.Size([2, 1, 1])
    """
    y = state["y"]
    cured = (y == 0).all(dim=1, keepdim=True)  # [N, 1, 1]
    event_time = state["event_time"].clone()  # [N, 1, 1]
    event_time[cured.expand_as(event_time)] = torch.inf
    result = state.copy()
    result["event_time"] = event_time
    return result


def censor_time(state: EventTimeState) -> CensoredState:
    """
    Sample uniform censor time within the observation window.

    Parameters
    ----------
    state : EventTimeState
        Must contain ``time`` [N, T, 1].

    Returns
    -------
    CensoredState
        Adds ``censor_time`` [N, 1, 1].

    See Also
    --------
    event_time : Preceding step.
    survival_indicators : Next step.

    Notes
    -----
    .. math:: c \\sim \\text{Uniform}(t_{\\min}, t_{\\max})

    Examples
    --------
    >>> import torch
    >>> from simulacra import linear_predictor, gaussian, event_time, censor_time
    >>> obs = gaussian(linear_predictor(2, 3, 4), 1.0, torch.eye(3))
    >>> result = censor_time(event_time(obs))
    >>> result["censor_time"].shape
    torch.Size([2, 1, 1])
    """
    time = state["time"]
    min_time = torch.amin(time, dim=1, keepdim=True)
    max_time = torch.amax(time, dim=1, keepdim=True)
    ct = dist.Uniform(min_time, max_time).sample()
    return CensoredState(**state, censor_time=ct)


def survival_indicators(state: CensoredState) -> SurvivalState:
    """
    Compute survival indicators from event and censor times.

    Parameters
    ----------
    state : CensoredState
        Must contain ``event_time`` [N, 1, 1], ``censor_time`` [N, 1, 1],
        and ``time`` [N, T, 1].

    Returns
    -------
    SurvivalState
        Adds ``indicator`` [N, 1, 1], ``observed_time`` [N, 1, 1],
        ``time_to_event`` [N, T, 1].

    See Also
    --------
    censor_time : Preceding step.
    event_time : Produce event times.

    Notes
    -----
    .. math::
        \\delta = \\mathbb{1}(t^* < c), \\quad
        \\tilde{t} = \\min(t^*, c)

    Examples
    --------
    >>> import torch
    >>> from simulacra import linear_predictor, gaussian
    >>> from simulacra import event_time, censor_time, survival_indicators
    >>> obs = gaussian(linear_predictor(2, 3, 4), 1.0, torch.eye(3))
    >>> result = survival_indicators(censor_time(event_time(obs)))
    >>> result["indicator"].shape
    torch.Size([2, 1, 1])
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
# Event processes
# ---------------------------------------------------------------------------


def independent_events(
    state: PredictorState, prevalence: UnitInterval = 0.1
) -> EventProcessState:
    """
    Generate a multilabel event mask via independent Bernoulli draws.

    Each risk column in ``eta`` [N, T, K] is converted to an event
    probability using a logit shift (same pattern as :func:`bernoulli`).
    Multiple risks can fire at the same timepoint.

    Parameters
    ----------
    state : PredictorState
        Must contain ``eta`` [N, T, K].
    prevalence : float
        Base rate per risk; shifts logits before sigmoid.

    Returns
    -------
    EventProcessState
        Adds ``event_mask`` [N, T, K] (boolean, multi-hot).

    See Also
    --------
    competing_risks : Single-winner Weibull alternative.
    multi_event : Downstream suffix-minimum TTE encoding.

    Notes
    -----
    .. math:: p_k = \\sigma(\\eta_k + \\text{logit}(\\text{prevalence})),
              \\quad m_k \\sim \\text{Bernoulli}(p_k)

    Examples
    --------
    >>> from simulacra import linear_predictor, linear, independent_events
    >>> state = linear(linear_predictor(2, 3, 4), out_features=3)
    >>> result = independent_events(state)
    >>> result["event_mask"].shape
    torch.Size([2, 3, 3])
    """
    shift = torch.logit(torch.tensor(prevalence))
    event_prob = torch.sigmoid(state["eta"] + shift)  # [N, T, K]
    event_mask = torch.bernoulli(event_prob).bool()  # [N, T, K]
    return EventProcessState(**state, event_mask=event_mask)


# ---------------------------------------------------------------------------
# Competing risks
# ---------------------------------------------------------------------------


def competing_risks(
    state: PredictorState, shape: PositiveFloat | Tensor = 1.0
) -> CompetingRisksState:
    """
    Sample per-risk failure times from Weibull distributions.

    Each column of ``eta`` [N, T, K] parameterizes one risk's Weibull scale.

    Parameters
    ----------
    state : PredictorState
        Must contain ``eta`` [N, T, K].
    shape : float or Tensor
        Weibull shape parameter; 1.0 reduces to exponential.

    Returns
    -------
    CompetingRisksState
        Adds ``failure_times`` [N, T, K] and ``tokens`` [N, T, 1]
        (argmin risk index).

    See Also
    --------
    risk_indicators : First-failure encoding.
    multi_event : Suffix-minimum encoding.

    Notes
    -----
    .. math:: t_k \\sim \\text{Weibull}(\\text{softplus}(\\eta_k),\\, \\alpha)

    Examples
    --------
    >>> from simulacra import linear_predictor, linear, competing_risks
    >>> state = linear(linear_predictor(2, 3, 4), out_features=3)
    >>> result = competing_risks(state)
    >>> result["failure_times"].shape
    torch.Size([2, 3, 3])
    """
    eps = 1e-6
    scale = F.softplus(state["eta"]) + eps  # [N, T, K]
    failure_times = dist.Weibull(scale, shape).rsample()  # [N, T, K]
    tokens = failure_times.argmin(dim=-1, keepdim=True)  # [N, T, 1]
    K = failure_times.shape[-1]
    event_mask = F.one_hot(tokens.squeeze(-1), num_classes=K).to(
        torch.bool
    )  # [N, T, K]
    return CompetingRisksState(
        **state, failure_times=failure_times, tokens=tokens, event_mask=event_mask
    )


def risk_indicators(state: CompetingRisksState) -> RiskIndicatorState:
    """
    One-hot encode the winning risk and broadcast event time to [N, T, K].

    Parameters
    ----------
    state : CompetingRisksState
        Must contain ``failure_times`` [N, T, K] and ``tokens`` [N, T, 1].

    Returns
    -------
    RiskIndicatorState
        Adds ``indicator`` [N, T, K] and ``event_time`` [N, T, K].

    See Also
    --------
    competing_risks : Preceding step.
    multi_event : Alternative suffix-minimum encoding.
    discretize_risk : Discretize into interval bins.

    Examples
    --------
    >>> from simulacra import linear_predictor, linear, competing_risks
    >>> from simulacra import risk_indicators
    >>> state = linear(linear_predictor(2, 3, 4), out_features=3)
    >>> result = risk_indicators(competing_risks(state))
    >>> result["indicator"].shape
    torch.Size([2, 3, 3])
    """
    failure_times = state["failure_times"]  # [N, T, K]
    K = failure_times.shape[-1]
    tokens = state["tokens"].squeeze(-1)  # [N, T]
    indicator = F.one_hot(tokens, num_classes=K).to(failure_times.dtype)  # [N, T, K]
    event_time = failure_times.min(dim=-1, keepdim=True).values.expand(
        -1, -1, K
    )  # [N, T, K]
    return RiskIndicatorState(**state, indicator=indicator, event_time=event_time)


def multi_event(
    state: EventProcessState, horizon: float | Tensor | None = None
) -> RiskIndicatorState:
    """
    Compute per-risk TTE via suffix-minimum over the observation schedule.

    The event mask can come from either ``competing_risks`` (single-winner,
    multiclass) or ``independent_events`` (multi-hot, multilabel).  The
    suffix-minimum algorithm is the same in both cases.

    Parameters
    ----------
    state : EventProcessState
        Must contain ``event_mask`` [N, T, K] and ``time`` [N, T, 1].
    horizon : float or Tensor or None
        Clamp ceiling for event times; inferred from data if omitted.

    Returns
    -------
    RiskIndicatorState
        Adds ``event_time`` [N, T, K] and ``indicator`` [N, T, K].

    See Also
    --------
    competing_risks : Single-winner event process.
    independent_events : Multilabel event process.
    risk_indicators : Alternative first-failure encoding (competing risks only).
    discretize_risk : Discretize into interval bins.

    Examples
    --------
    >>> from simulacra import linear_predictor, linear, competing_risks, multi_event
    >>> state = linear(linear_predictor(2, 3, 4), out_features=3)
    >>> result = multi_event(competing_risks(state))
    >>> result["event_time"].shape
    torch.Size([2, 3, 3])
    """
    time = state["time"]  # [N, T, 1]
    event_mask = state["event_mask"]  # [N, T, K]
    K = event_mask.shape[-1]
    event_times = time.expand(-1, -1, K)  # [N, T, K]
    event_times = event_times.masked_fill(~event_mask, torch.inf)
    reversed_ = torch.flip(event_times, dims=[1])
    suffix_min = torch.flip(torch.cummin(reversed_, dim=1).values, dims=[1])
    next_time = F.pad(suffix_min[:, 1:], (0, 0, 0, 1), value=torch.inf)
    event_time = next_time - time  # [N, T, K]
    if horizon is None:
        horizon = event_time[event_time.isfinite()].max()
    event_time = event_time.clamp(max=horizon)
    indicator = (event_time < horizon).to(event_time.dtype)  # [N, T, K]
    return RiskIndicatorState(**state, event_time=event_time, indicator=indicator)


def discretize_risk(state: RiskIndicatorState, boundaries: Tensor) -> DiscreteRiskState:
    """
    Discretize continuous event times into interval bins.

    The combined encoding is designed for discrete failure time NLL losses
    (DeepHit-style).  For piecewise exponential losses (MOTOR/PEANN-style),
    use the raw ``event_time`` and ``indicator`` tensors to construct the
    piecewise-exponential inputs separately.

    Parameters
    ----------
    state : RiskIndicatorState
        Must contain ``event_time`` [N, T, K] and ``indicator`` [N, T, K].
    boundaries : Tensor
        Monotonic bin edges [J+1]; produces J intervals.

    Returns
    -------
    DiscreteRiskState
        Adds ``discrete_event_time`` [N, T, K, J].

    See Also
    --------
    risk_indicators : Preceding step (first-failure).
    multi_event : Preceding step (suffix-minimum).

    Examples
    --------
    >>> import torch
    >>> from simulacra import linear_predictor, linear, competing_risks
    >>> from simulacra import risk_indicators, discretize_risk
    >>> state = linear(linear_predictor(2, 3, 4), out_features=3)
    >>> result = discretize_risk(
    ...     risk_indicators(competing_risks(state)),
    ...     boundaries=torch.linspace(0, 5, 6),
    ... )
    >>> result["discrete_event_time"].shape
    torch.Size([2, 3, 3, 5])
    """
    event_time = state["event_time"]  # [N, T, K]
    indicator = state["indicator"]  # [N, T, K]
    interval_start = boundaries[:-1].view(*([1] * event_time.dim()), -1)  # [1,..,1, J]
    interval_end = boundaries[1:].view(*([1] * event_time.dim()), -1)
    interval_width = interval_end - interval_start
    event_time_expanded = event_time.unsqueeze(-1)  # [N, T, K, 1]
    exposure = ((event_time_expanded - interval_start) / interval_width).clamp(0, 1)
    in_interval = (event_time_expanded > interval_start) & (
        event_time_expanded <= interval_end
    )
    ind = indicator.unsqueeze(-1).to(exposure.dtype)  # [N, T, K, 1]
    event_duration = ind * (exposure * in_interval)
    censored_duration = (1.0 - ind) * exposure
    return DiscreteRiskState(
        **state,
        discrete_event_time=event_duration + censored_duration,  # [N, T, K, J]
    )
