from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch
from tensordict import TensorDict
from torch import Tensor

from .covariance import ResidualCovarianceSpec, make_residual_covariance
from .functional import (
    censor_time,
    competing_risk_indicators,
    competing_risks_events,
    discrete_event_time,
    event_time,
    linear_predictor,
    logistic_output,
    mixture_cure_censoring,
    multi_event_times,
    multiclass_output,
    observation_time,
    observations,
    ordinal_output,
    random_effects,
    survival_indicators,
    tokens,
)

ObservationCovariance = Tensor | ResidualCovarianceSpec

# -- Public typestate protocols --


class _HasData(Protocol):
    @property
    def data(self) -> TensorDict: ...


class Complete(_HasData, Protocol):
    """Terminal stage of the fluent interface."""


class _CanTokenize(_HasData, Protocol):
    def tokenize(
        self, vocab_size: int = 1000, concentration: float = 1.0
    ) -> Complete: ...


class HasCompetingIndicators(_HasData, Protocol):
    """Stage with competing-risk indicators and discretization available."""

    def discrete_event_time(self, boundaries: Tensor) -> Complete: ...


class HasSurvival(_CanTokenize, Protocol):
    """Stage with survival indicators available."""

    def discrete_event_time(self, boundaries: Tensor) -> Complete: ...


class HasCensored(_CanTokenize, Protocol):
    """Stage with censor times available."""

    def survival_indicators(self) -> HasSurvival: ...


class HasEventTimeAndTime(_CanTokenize, Protocol):
    """Stage with event and observation times available."""

    def censor_time(self) -> HasCensored: ...


class HasEventTime(_CanTokenize, Protocol):
    """Stage with event times available."""

    def observation_time(self, shape: float, rate: float) -> HasEventTimeAndTime: ...


class HasClassificationEvent(_CanTokenize, Protocol):
    """Stage with classification output and event times available."""

    def mixture_cure_censoring(self) -> HasEventTime: ...

    def observation_time(self, shape: float, rate: float) -> HasEventTimeAndTime: ...


class HasClassification(_CanTokenize, Protocol):
    """Stage with classification outputs available."""

    def event_time(self) -> HasClassificationEvent: ...


class HasCompetingEvents(_HasData, Protocol):
    """Stage with competing-risk events available."""

    def competing_risk_indicators(self, vocab_size: int) -> HasCompetingIndicators: ...

    def multi_event_times(
        self, vocab_size: int, horizon: float | Tensor
    ) -> HasCompetingIndicators: ...


class HasEta(_CanTokenize, Protocol):
    """Stage with the linear predictor available."""

    def logistic_output(self) -> HasClassification: ...

    def multiclass_output(self) -> HasClassification: ...

    def ordinal_output(self, num_classes: int) -> HasClassification: ...

    def event_time(self) -> HasEventTime: ...


class HasObservations(_CanTokenize, Protocol):
    """Stage with observed trajectories available."""

    def linear_predictor(self, weight: Tensor, prevalence: float) -> HasEta: ...

    def competing_risks_events(self, vocab_size: int) -> HasCompetingEvents: ...

    def logistic(self, weight: Tensor, prevalence: float) -> HasClassification: ...

    def multiclass(self, weight: Tensor, prevalence: float) -> HasClassification: ...

    def ordinal(
        self, weight: Tensor, prevalence: float, num_classes: int
    ) -> HasClassification: ...

    def survival(
        self, weight: Tensor, prevalence: float, shape: float, rate: float
    ) -> HasSurvival: ...

    def mixture_cure(
        self, weight: Tensor, prevalence: float, shape: float, rate: float
    ) -> HasSurvival: ...

    def competing_risk(self, vocab_size: int, boundaries: Tensor) -> Complete: ...

    def multi_event(self, vocab_size: int, boundaries: Tensor) -> Complete: ...


class HasRandomEffects(_HasData, Protocol):
    """Stage with random effects configured."""

    def observations(
        self,
        num_timepoints: int,
        observed_std: float,
        *,
        covariance: ObservationCovariance | None = None,
        num_fixed_effects: int = 0,
    ) -> HasObservations: ...


class _BaseStage:
    def __init__(self, data: TensorDict) -> None:
        self._data = data

    @property
    def data(self) -> TensorDict:
        return self._data


class _CompleteStage(_BaseStage):
    pass


class _TokenizableStage(_BaseStage):
    def tokenize(
        self, vocab_size: int = 1000, concentration: float = 1.0
    ) -> _CompleteStage:
        tokens(self._data, vocab_size, concentration)
        return _CompleteStage(self._data)


class _CompetingIndicatorsStage(_BaseStage):
    def discrete_event_time(self, boundaries: Tensor) -> _CompleteStage:
        discrete_event_time(self._data, boundaries)
        return _CompleteStage(self._data)


class _SurvivalStage(_TokenizableStage):
    def discrete_event_time(self, boundaries: Tensor) -> _CompleteStage:
        discrete_event_time(self._data, boundaries)
        return _CompleteStage(self._data)


class _CensoredStage(_TokenizableStage):
    def survival_indicators(self) -> _SurvivalStage:
        survival_indicators(self._data)
        return _SurvivalStage(self._data)


class _EventTimeAndTimeStage(_TokenizableStage):
    def censor_time(self) -> _CensoredStage:
        censor_time(self._data)
        return _CensoredStage(self._data)


class _ObservationTimeStage(_TokenizableStage):
    def observation_time(self, shape: float, rate: float) -> _EventTimeAndTimeStage:
        observation_time(self._data, shape, rate)
        return _EventTimeAndTimeStage(self._data)


class _EventTimeStage(_ObservationTimeStage):
    pass


class _ClassificationEventStage(_ObservationTimeStage):
    def mixture_cure_censoring(self) -> _EventTimeStage:
        mixture_cure_censoring(self._data)
        return _EventTimeStage(self._data)


class _ClassificationStage(_TokenizableStage):
    def event_time(self) -> _ClassificationEventStage:
        event_time(self._data)
        return _ClassificationEventStage(self._data)


class _CompetingEventsStage(_BaseStage):
    def competing_risk_indicators(self, vocab_size: int) -> _CompetingIndicatorsStage:
        competing_risk_indicators(self._data, vocab_size)
        return _CompetingIndicatorsStage(self._data)

    def multi_event_times(
        self, vocab_size: int, horizon: float | Tensor
    ) -> _CompetingIndicatorsStage:
        multi_event_times(self._data, vocab_size, horizon)
        return _CompetingIndicatorsStage(self._data)


class _EtaStage(_TokenizableStage):
    def logistic_output(self) -> _ClassificationStage:
        logistic_output(self._data)
        return _ClassificationStage(self._data)

    def multiclass_output(self) -> _ClassificationStage:
        multiclass_output(self._data)
        return _ClassificationStage(self._data)

    def ordinal_output(self, num_classes: int) -> _ClassificationStage:
        ordinal_output(self._data, num_classes)
        return _ClassificationStage(self._data)

    def event_time(self) -> _EventTimeStage:
        event_time(self._data)
        return _EventTimeStage(self._data)


class _ObservationsStage(_TokenizableStage):
    def linear_predictor(self, weight: Tensor, prevalence: float) -> _EtaStage:
        linear_predictor(self._data, weight, prevalence)
        return _EtaStage(self._data)

    def competing_risks_events(self, vocab_size: int) -> _CompetingEventsStage:
        competing_risks_events(self._data, vocab_size)
        return _CompetingEventsStage(self._data)

    def logistic(self, weight: Tensor, prevalence: float) -> _ClassificationStage:
        return self.linear_predictor(weight, prevalence).logistic_output()

    def multiclass(self, weight: Tensor, prevalence: float) -> _ClassificationStage:
        return self.linear_predictor(weight, prevalence).multiclass_output()

    def ordinal(
        self, weight: Tensor, prevalence: float, num_classes: int
    ) -> _ClassificationStage:
        return self.linear_predictor(weight, prevalence).ordinal_output(num_classes)

    def survival(
        self, weight: Tensor, prevalence: float, shape: float, rate: float
    ) -> _SurvivalStage:
        return (
            self.linear_predictor(weight, prevalence)
            .event_time()
            .observation_time(shape, rate)
            .censor_time()
            .survival_indicators()
        )

    def mixture_cure(
        self, weight: Tensor, prevalence: float, shape: float, rate: float
    ) -> _SurvivalStage:
        return (
            self.linear_predictor(weight, prevalence)
            .logistic_output()
            .event_time()
            .mixture_cure_censoring()
            .observation_time(shape, rate)
            .censor_time()
            .survival_indicators()
        )

    def competing_risk(self, vocab_size: int, boundaries: Tensor) -> _CompleteStage:
        return (
            self.competing_risks_events(vocab_size)
            .competing_risk_indicators(vocab_size)
            .discrete_event_time(boundaries)
        )

    def multi_event(self, vocab_size: int, boundaries: Tensor) -> _CompleteStage:
        return (
            self.competing_risks_events(vocab_size)
            .multi_event_times(vocab_size, boundaries[-1])
            .discrete_event_time(boundaries)
        )


class _RandomEffectsStage(_BaseStage):
    def observations(
        self,
        num_timepoints: int,
        observed_std: float,
        *,
        covariance: ObservationCovariance | None = None,
        num_fixed_effects: int = 0,
    ) -> _ObservationsStage:
        covariance_matrix = (
            covariance
            if isinstance(covariance, Tensor)
            else make_residual_covariance(num_timepoints, covariance)
        )
        observations(
            self._data,
            num_timepoints,
            observed_std,
            covariance_matrix,
            num_fixed_effects,
        )
        return _ObservationsStage(self._data)


class Simulation:
    """Fluent interface for building synthetic datasets.

    Parameters
    ----------
    num_samples
        Number of samples (subjects) to simulate.
    """

    def __init__(self, num_samples: int) -> None:
        self._data = TensorDict(
            {
                "id": torch.arange(num_samples).view(-1, 1, 1),
                "group": torch.zeros(num_samples, 1, 1, dtype=torch.long),
            },
            batch_size=[num_samples],
        )

    @property
    def data(self) -> TensorDict:
        return self._data

    def random_effects(
        self, hidden_dim: int, stds: Sequence[float], correlation: Tensor | float = 0.0
    ) -> HasRandomEffects:
        random_effects(self._data, hidden_dim, stds, correlation)
        return _RandomEffectsStage(self._data)
