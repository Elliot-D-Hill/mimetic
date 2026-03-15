"""TypedDict state definitions for the simulation pipeline."""

from typing import NotRequired, TypedDict, TypeGuard

from torch import Tensor

# ---------------------------------------------------------------------------
# Main pipeline stages
# ---------------------------------------------------------------------------


class PredictorState(TypedDict):
    """
    State after constructing the linear predictor eta = X*beta.

    See Also
    --------
    linear_predictor : Construct this state.
    random_effects : Upgrade to GLMM.
    """

    eta: Tensor  # [N, T, 1]
    time: Tensor  # [N, T, 1]
    X: Tensor  # [N, T, p]
    beta: Tensor  # [N, p, 1]
    gamma: NotRequired[Tensor]  # [N, q, 1]
    Z: NotRequired[Tensor]  # [N, T, q]


class MixedEffectsState(TypedDict):
    """
    PredictorState after random effects have been added.

    Standalone TypedDict (not inheriting from PredictorState) because
    pyright forbids overriding NotRequired -> Required in subclasses.

    See Also
    --------
    random_effects : Produce this state.
    """

    eta: Tensor  # [N, T, 1]
    time: Tensor  # [N, T, 1]
    X: Tensor  # [N, T, p]
    beta: Tensor  # [N, p, 1]
    gamma: Tensor  # [N, q, 1]
    Z: Tensor  # [N, T, q]


def has_random_effects(state: PredictorState) -> TypeGuard[MixedEffectsState]:
    """
    Narrow a PredictorState to MixedEffectsState.

    Parameters
    ----------
    state : PredictorState
        State to check.

    Returns
    -------
    TypeGuard[MixedEffectsState]
        True when ``gamma`` is present.
    """
    return "gamma" in state


class ObservedState(PredictorState):
    """
    State after sampling a response y from a distribution family.

    See Also
    --------
    gaussian, poisson, bernoulli, categorical, ordinal :
        Response distributions that produce this state.
    """

    y: Tensor  # [N, T, 1]
    mu: Tensor  # [N, T, 1] or [N, T, K]
    noise: NotRequired[Tensor]  # [N, T, 1]


class GaussianObservedState(TypedDict):
    """
    ObservedState after Gaussian sampling -- noise is always present.

    Standalone TypedDict (not inheriting from ObservedState) because
    pyright forbids overriding NotRequired -> Required in subclasses.

    See Also
    --------
    gaussian : Produce this state.
    """

    eta: Tensor  # [N, T, 1]
    time: Tensor  # [N, T, 1]
    X: Tensor  # [N, T, p]
    beta: Tensor  # [N, p, 1]
    gamma: NotRequired[Tensor]  # [N, q, 1]
    Z: NotRequired[Tensor]  # [N, T, q]
    y: Tensor  # [N, T, 1]
    mu: Tensor  # [N, T, 1]
    noise: Tensor  # [N, T, 1]


def has_noise(state: ObservedState) -> TypeGuard[GaussianObservedState]:
    """
    Narrow an ObservedState to GaussianObservedState.

    Parameters
    ----------
    state : ObservedState
        State to check.

    Returns
    -------
    TypeGuard[GaussianObservedState]
        True when ``noise`` is present.
    """
    return "noise" in state


class TokenizedState(ObservedState):
    """
    State after discretizing the design matrix into token IDs.

    See Also
    --------
    tokens : Produce this state.
    """

    tokens: Tensor  # [N, T, 1]


# ---------------------------------------------------------------------------
# Survival extension stages
# ---------------------------------------------------------------------------


class EventTimeState(ObservedState):
    """
    State after sampling a subject-level event time.

    See Also
    --------
    event_time : Produce this state.
    """

    event_time: Tensor  # [N, 1, 1]
    tokens: NotRequired[Tensor]  # [N, T, 1]


class CensoredState(EventTimeState):
    """
    State after sampling a censoring time.

    See Also
    --------
    censor_time : Produce this state.
    """

    censor_time: Tensor  # [N, 1, 1]


class SurvivalState(CensoredState):
    """
    State after computing survival indicators from event and censor times.

    See Also
    --------
    survival_indicators : Produce this state.
    """

    indicator: Tensor  # [N, 1, 1]
    observed_time: Tensor  # [N, 1, 1]
    time_to_event: Tensor  # [N, T, 1]


# ---------------------------------------------------------------------------
# Competing risks extension stages
# ---------------------------------------------------------------------------


class EventProcessState(PredictorState):
    """
    State after generating a per-risk event mask.

    The mask can come from either :func:`independent_events`
    (multi-hot, multilabel) or :func:`competing_risks`
    (one-hot, multiclass).

    See Also
    --------
    independent_events : Multilabel Bernoulli event process.
    competing_risks : Single-winner Weibull event process.
    """

    event_mask: Tensor  # [N, T, K]


class CompetingRisksState(EventProcessState):
    """
    State after sampling per-risk failure times from Weibull distributions.

    See Also
    --------
    competing_risks : Produce this state.
    """

    failure_times: Tensor  # [N, T, K]
    tokens: Tensor  # [N, T, 1]


class RiskIndicatorState(EventProcessState):
    """
    State after encoding risk indicators and event times per risk.

    See Also
    --------
    risk_indicators : Produce via first-failure encoding.
    multi_event : Produce via suffix-minimum encoding.
    """

    event_time: Tensor  # [N, T, K]
    indicator: Tensor  # [N, T, K]
    failure_times: NotRequired[Tensor]  # [N, T, K] — present from competing_risks path
    tokens: NotRequired[Tensor]  # [N, T, 1] — present from competing_risks path


class FullRiskIndicatorState(TypedDict):
    """
    RiskIndicatorState from the competing-risks path.

    Standalone TypedDict (not inheriting from RiskIndicatorState) because
    pyright forbids overriding NotRequired -> Required in subclasses.

    See Also
    --------
    risk_indicators : Produce this state from competing_risks output.
    """

    eta: Tensor  # [N, T, K]
    time: Tensor  # [N, T, 1]
    X: Tensor  # [N, T, p]
    beta: Tensor  # [N, p, 1]
    gamma: NotRequired[Tensor]  # [N, q, 1]
    Z: NotRequired[Tensor]  # [N, T, q]
    event_mask: Tensor  # [N, T, K]
    event_time: Tensor  # [N, T, K]
    indicator: Tensor  # [N, T, K]
    failure_times: Tensor  # [N, T, K]
    tokens: Tensor  # [N, T, 1]


def has_failure_times(state: RiskIndicatorState) -> TypeGuard[FullRiskIndicatorState]:
    """
    Narrow a RiskIndicatorState to FullRiskIndicatorState.

    Parameters
    ----------
    state : RiskIndicatorState
        State to check.

    Returns
    -------
    TypeGuard[FullRiskIndicatorState]
        True when ``failure_times`` is present.
    """
    return "failure_times" in state


class DiscreteRiskState(RiskIndicatorState):
    """
    State after discretizing continuous event times into interval bins.

    See Also
    --------
    discretize_risk : Produce this state.
    """

    discrete_event_time: Tensor  # [N, T, K, J]
