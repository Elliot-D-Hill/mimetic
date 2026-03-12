from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from tensordict import TensorDict
from torch import Tensor

from .covariance import ResidualCovarianceSpec, residual_covariance
from .functional import (
    censor_time,
    event_time,
    logistic_output,
    mixture_cure_censoring,
    multiclass_output,
    observations,
    ordinal_output,
    random_effects,
    replace_observation_time,
    replace_survival_observation_time,
    survival_indicators,
    tokens,
)
from .states import (
    CensoredState,
    EffectsState,
    EventTimeState,
    LabeledState,
    ObservedState,
    SurvivalState,
    TokenizedState,
)

ObservationCovariance = Tensor | ResidualCovarianceSpec


def _to_tensordict(state: Mapping[str, Any]) -> TensorDict:
    first = next(iter(state.values()))
    return TensorDict(dict(state), batch_size=[first.shape[0]])


# ---------------------------------------------------------------------------
# Main pipeline steps
# ---------------------------------------------------------------------------


class Simulation:
    """Entry point for building synthetic datasets.

    Parameters
    ----------
    num_samples
        Number of samples (subjects) to simulate.
    """

    def __init__(self, num_samples: int) -> None:
        self._num_samples = num_samples

    @property
    def data(self) -> TensorDict:
        return TensorDict({}, batch_size=[self._num_samples])

    def random_effects(
        self, stds: Sequence[float], correlation: Tensor | float = 0.0
    ) -> EffectsStep:
        return EffectsStep(random_effects(self._num_samples, stds, correlation))


@dataclass(frozen=True)
class EffectsStep:
    state: EffectsState

    @property
    def data(self) -> TensorDict:
        return _to_tensordict(self.state)

    def observations(
        self,
        num_timepoints: int,
        num_features: int,
        observed_std: float,
        *,
        covariance: ObservationCovariance | None = None,
    ) -> ObservedStep:
        covariance_matrix = (
            covariance
            if isinstance(covariance, Tensor)
            else residual_covariance(num_timepoints, covariance)
        )
        return ObservedStep(
            observations(
                self.state,
                num_timepoints,
                num_features,
                observed_std,
                covariance_matrix,
            )
        )


@dataclass(frozen=True)
class ObservedStep:
    state: ObservedState

    @property
    def data(self) -> TensorDict:
        return _to_tensordict(self.state)

    def tokenize(
        self, vocab_size: int = 1000, concentration: float = 1.0
    ) -> TokenizedStep:
        return TokenizedStep(tokens(self.state, vocab_size, concentration))

    def logistic(self, prevalence: float = 0.5) -> LabeledStep:
        return LabeledStep(logistic_output(self.state, prevalence))

    def multiclass(self) -> LabeledStep:
        return LabeledStep(multiclass_output(self.state))

    def ordinal(self, num_classes: int) -> LabeledStep:
        return LabeledStep(ordinal_output(self.state, num_classes))

    def event_time(self) -> EventTimeStep:
        return EventTimeStep(event_time(self.state))

    def observation_time(self, shape: float, rate: float) -> ObservedStep:
        return ObservedStep(replace_observation_time(self.state, shape, rate))


@dataclass(frozen=True)
class TokenizedStep:
    state: TokenizedState

    @property
    def data(self) -> TensorDict:
        return _to_tensordict(self.state)

    def logistic(self, prevalence: float = 0.5) -> LabeledStep:
        return LabeledStep(logistic_output(self.state, prevalence))

    def multiclass(self) -> LabeledStep:
        return LabeledStep(multiclass_output(self.state))

    def ordinal(self, num_classes: int) -> LabeledStep:
        return LabeledStep(ordinal_output(self.state, num_classes))

    def event_time(self) -> EventTimeStep:
        return EventTimeStep(event_time(self.state))


@dataclass(frozen=True)
class LabeledStep:
    state: LabeledState

    @property
    def data(self) -> TensorDict:
        return _to_tensordict(self.state)

    def tokenize(
        self, vocab_size: int = 1000, concentration: float = 1.0
    ) -> LabeledStep:
        tokenized = tokens(self.state, vocab_size, concentration)
        result = self.state.copy()
        result["tokens"] = tokenized["tokens"]
        return LabeledStep(result)

    def event_time(self) -> LabeledEventTimeStep:
        return LabeledEventTimeStep(event_time(self.state))


# ---------------------------------------------------------------------------
# Survival steps
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventTimeStep:
    state: EventTimeState

    @property
    def data(self) -> TensorDict:
        return _to_tensordict(self.state)

    def observation_time(self, shape: float, rate: float) -> EventTimeStep:
        return EventTimeStep(replace_survival_observation_time(self.state, shape, rate))

    def censor_time(self) -> CensoredStep:
        return CensoredStep(censor_time(self.state))


@dataclass(frozen=True)
class LabeledEventTimeStep:
    state: EventTimeState

    @property
    def data(self) -> TensorDict:
        return _to_tensordict(self.state)

    def mixture_cure(self) -> EventTimeStep:
        return EventTimeStep(mixture_cure_censoring(self.state))

    def observation_time(self, shape: float, rate: float) -> LabeledEventTimeStep:
        return LabeledEventTimeStep(
            replace_survival_observation_time(self.state, shape, rate)
        )

    def censor_time(self) -> CensoredStep:
        return CensoredStep(censor_time(self.state))


@dataclass(frozen=True)
class CensoredStep:
    state: CensoredState

    @property
    def data(self) -> TensorDict:
        return _to_tensordict(self.state)

    def survival_indicators(self) -> SurvivalStep:
        return SurvivalStep(survival_indicators(self.state))


@dataclass(frozen=True)
class SurvivalStep:
    state: SurvivalState

    @property
    def data(self) -> TensorDict:
        return _to_tensordict(self.state)
