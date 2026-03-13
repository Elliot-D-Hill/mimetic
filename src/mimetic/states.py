from typing import NotRequired, TypedDict

from torch import Tensor

# ---------------------------------------------------------------------------
# Main pipeline stages
# ---------------------------------------------------------------------------


class PredictorState(TypedDict):
    """State after constructing the linear predictor eta = X*beta.

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
    U: NotRequired[Tensor]  # [N, T, q]


class ObservedState(PredictorState):
    """State after sampling a response y from a distribution family.

    See Also
    --------
    gaussian, poisson, bernoulli, categorical, ordinal :
        Response distributions that produce this state.
    """

    y: Tensor  # [N, T, 1]
    mu: Tensor  # [N, T, 1] or [N, T, K]
    noise: NotRequired[Tensor]  # [N, T, 1]


class TokenizedState(ObservedState):
    """State after discretizing the design matrix into token IDs.

    See Also
    --------
    tokens : Produce this state.
    """

    tokens: Tensor  # [N, T, 1]


# ---------------------------------------------------------------------------
# Survival extension stages
# ---------------------------------------------------------------------------


class EventTimeState(ObservedState):
    """State after sampling a subject-level event time.

    See Also
    --------
    event_time : Produce this state.
    """

    event_time: Tensor  # [N, 1, 1]
    tokens: NotRequired[Tensor]  # [N, T, 1]


class CensoredState(EventTimeState):
    """State after sampling a censoring time.

    See Also
    --------
    censor_time : Produce this state.
    """

    censor_time: Tensor  # [N, 1, 1]


class SurvivalState(CensoredState):
    """State after computing survival indicators from event and censor times.

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


class CompetingRisksState(PredictorState):
    """State after sampling per-risk failure times from Weibull distributions.

    See Also
    --------
    competing_risks : Produce this state.
    """

    failure_times: Tensor  # [N, T, K]
    tokens: Tensor  # [N, T, 1]


class RiskIndicatorState(CompetingRisksState):
    """State after encoding risk indicators and event times per risk.

    See Also
    --------
    risk_indicators : Produce via first-failure encoding.
    multi_event : Produce via suffix-minimum encoding.
    """

    event_time: Tensor  # [N, T, K]
    indicator: Tensor  # [N, T, K]


class DiscreteRiskState(RiskIndicatorState):
    """State after discretizing continuous event times into interval bins.

    See Also
    --------
    discretize_risk : Produce this state.
    """

    discrete_event_time: Tensor  # [N, T, K, J]
