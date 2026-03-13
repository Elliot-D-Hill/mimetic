from typing import NotRequired, TypedDict

from torch import Tensor

# ---------------------------------------------------------------------------
# Main pipeline stages
# ---------------------------------------------------------------------------


class PredictorState(TypedDict):
    eta: Tensor  # [N, T, 1]
    time: Tensor  # [N, T, 1]
    X: Tensor  # [N, T, p]
    beta: Tensor  # [N, p, 1]
    gamma: NotRequired[Tensor]  # [N, q, 1]
    U: NotRequired[Tensor]  # [N, T, q]


class ObservedState(PredictorState):
    y: Tensor  # [N, T, 1]
    mu: Tensor  # [N, T, 1] or [N, T, K]
    noise: NotRequired[Tensor]  # [N, T, 1]


class TokenizedState(ObservedState):
    tokens: Tensor  # [N, T, 1]


# ---------------------------------------------------------------------------
# Survival extension stages
# ---------------------------------------------------------------------------


class EventTimeState(ObservedState):
    event_time: Tensor  # [N, 1, 1]
    tokens: NotRequired[Tensor]  # [N, T, 1]


class CensoredState(EventTimeState):
    censor_time: Tensor  # [N, 1, 1]


class SurvivalState(CensoredState):
    indicator: Tensor  # [N, 1, 1]
    observed_time: Tensor  # [N, 1, 1]
    time_to_event: Tensor  # [N, T, 1]


# ---------------------------------------------------------------------------
# Competing risks extension stages
# ---------------------------------------------------------------------------


class CompetingRisksState(PredictorState):
    failure_times: Tensor  # [N, T, K]
    tokens: Tensor  # [N, T, 1]


class RiskIndicatorState(CompetingRisksState):
    event_time: Tensor  # [N, T, K]
    indicator: Tensor  # [N, T, K]


class DiscreteRiskState(RiskIndicatorState):
    discrete_event_time: Tensor  # [N, T, K, J]
