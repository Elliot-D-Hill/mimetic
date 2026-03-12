from typing import NotRequired, TypedDict

from torch import Tensor

# ---------------------------------------------------------------------------
# Main pipeline stages
# ---------------------------------------------------------------------------


class EffectsState(TypedDict):
    gamma: Tensor  # [N, q, 1]


class ObservedState(EffectsState):
    y: Tensor  # [N, T, 1]
    U: Tensor  # [N, T, q]
    time: Tensor  # [N, T, 1]
    eta: Tensor  # [N, T, 1]
    X: Tensor  # [N, T, p]
    beta: Tensor  # [N, p, 1]


class TokenizedState(ObservedState):
    tokens: Tensor  # [N, T, 1]


class LabeledState(ObservedState):
    probability: Tensor  # [N, T, 1] or [N, T, K]
    label: Tensor  # [N, T, 1]
    tokens: NotRequired[Tensor]  # [N, T, 1]


# ---------------------------------------------------------------------------
# Survival extension stages
# ---------------------------------------------------------------------------


class EventTimeState(ObservedState):
    event_time: Tensor  # [N, 1, 1]
    tokens: NotRequired[Tensor]  # [N, T, 1]
    probability: NotRequired[Tensor]  # [N, T, 1]
    label: NotRequired[Tensor]  # [N, T, 1]


class CensoredState(EventTimeState):
    censor_time: Tensor  # [N, 1, 1]


class SurvivalState(CensoredState):
    indicator: Tensor  # [N, 1, 1]
    observed_time: Tensor  # [N, 1, 1]
    time_to_event: Tensor  # [N, T, 1]
