from __future__ import annotations

from typing import Literal

import torch
from tensordict import TensorDict
from torch import Tensor

from .covariance import make_residual_covariance
from .functional import (
    censor_time,
    competing_risk_indicators,
    competing_risks_events,
    discrete_event_time,
    event_time,
    linear_output,
    logistic_output,
    mixture_cure_censoring,
    multi_event_times,
    multiclass_output,
    observation_time,
    observed_features,
    ordinal_output,
    random_effects,
    survival_indicators,
    tokens,
)

# -- Leaf states (defined first for readability, order doesn't matter with PEP 563) --


class _Complete:
    """Terminal state — only ``.data`` is available."""

    def __init__(self, data: TensorDict) -> None:
        self._data = data

    @property
    def data(self) -> TensorDict:
        return self._data


class _HasCompetingIndicators:
    """Has competing risk indicators; needs discretization."""

    def __init__(self, data: TensorDict) -> None:
        self._data = data

    @property
    def data(self) -> TensorDict:
        return self._data

    def discrete_event_time(self, boundaries: Tensor) -> _Complete:
        discrete_event_time(self._data, boundaries)
        return _Complete(self._data)


class _HasSurvival:
    """Has survival indicators; may discretise or tokenize."""

    def __init__(self, data: TensorDict) -> None:
        self._data = data

    @property
    def data(self) -> TensorDict:
        return self._data

    def discrete_event_time(self, boundaries: Tensor) -> _Complete:
        discrete_event_time(self._data, boundaries)
        return _Complete(self._data)

    def tokenize(self, vocab_size: int = 1000, concentration: float = 1.0) -> _Complete:
        tokens(self._data, vocab_size, concentration)
        return _Complete(self._data)


# -- Survival chain (bottom-up) --


class _HasCensored:
    """Has censor time; needs survival indicators."""

    def __init__(self, data: TensorDict) -> None:
        self._data = data

    @property
    def data(self) -> TensorDict:
        return self._data

    def survival_indicators(self) -> _HasSurvival:
        survival_indicators(self._data)
        return _HasSurvival(self._data)

    def tokenize(self, vocab_size: int = 1000, concentration: float = 1.0) -> _Complete:
        tokens(self._data, vocab_size, concentration)
        return _Complete(self._data)


class _HasEventTimeAndTime:
    """Has event time and observation time; needs censoring."""

    def __init__(self, data: TensorDict) -> None:
        self._data = data

    @property
    def data(self) -> TensorDict:
        return self._data

    def censor_time(self) -> _HasCensored:
        censor_time(self._data)
        return _HasCensored(self._data)

    def tokenize(self, vocab_size: int = 1000, concentration: float = 1.0) -> _Complete:
        tokens(self._data, vocab_size, concentration)
        return _Complete(self._data)


class _HasEventTime:
    """Has event time; needs observation time."""

    def __init__(self, data: TensorDict) -> None:
        self._data = data

    @property
    def data(self) -> TensorDict:
        return self._data

    def observation_time(self, shape: float, rate: float) -> _HasEventTimeAndTime:
        observation_time(self._data, shape, rate)
        return _HasEventTimeAndTime(self._data)

    def tokenize(self, vocab_size: int = 1000, concentration: float = 1.0) -> _Complete:
        tokens(self._data, vocab_size, concentration)
        return _Complete(self._data)


# -- Classification chain --


class _HasClassificationEvent:
    """Has classification output and event time; may apply mixture cure or observe."""

    def __init__(self, data: TensorDict) -> None:
        self._data = data

    @property
    def data(self) -> TensorDict:
        return self._data

    def mixture_cure_censoring(self) -> _HasEventTime:
        mixture_cure_censoring(self._data)
        return _HasEventTime(self._data)

    def observation_time(self, shape: float, rate: float) -> _HasEventTimeAndTime:
        observation_time(self._data, shape, rate)
        return _HasEventTimeAndTime(self._data)

    def tokenize(self, vocab_size: int = 1000, concentration: float = 1.0) -> _Complete:
        tokens(self._data, vocab_size, concentration)
        return _Complete(self._data)


class _HasClassification:
    """Has classification output (logistic/multiclass/ordinal)."""

    def __init__(self, data: TensorDict) -> None:
        self._data = data

    @property
    def data(self) -> TensorDict:
        return self._data

    def event_time(self) -> _HasClassificationEvent:
        event_time(self._data)
        return _HasClassificationEvent(self._data)

    def tokenize(self, vocab_size: int = 1000, concentration: float = 1.0) -> _Complete:
        tokens(self._data, vocab_size, concentration)
        return _Complete(self._data)


# -- Competing risks chain --


class _HasCompetingEvents:
    """Has competing risks events; needs indicators or multi-event times."""

    def __init__(self, data: TensorDict) -> None:
        self._data = data

    @property
    def data(self) -> TensorDict:
        return self._data

    def competing_risk_indicators(self, vocab_size: int) -> _HasCompetingIndicators:
        competing_risk_indicators(self._data, vocab_size)
        return _HasCompetingIndicators(self._data)

    def multi_event_times(
        self, vocab_size: int, horizon: float | Tensor
    ) -> _HasCompetingIndicators:
        multi_event_times(self._data, vocab_size, horizon)
        return _HasCompetingIndicators(self._data)


# -- Output --


class _HasOutput:
    """Has linear output; may add classification, event time, or tokenize."""

    def __init__(self, data: TensorDict) -> None:
        self._data = data

    @property
    def data(self) -> TensorDict:
        return self._data

    def logistic_output(self) -> _HasClassification:
        logistic_output(self._data)
        return _HasClassification(self._data)

    def multiclass_output(self) -> _HasClassification:
        multiclass_output(self._data)
        return _HasClassification(self._data)

    def ordinal_output(self, num_classes: int) -> _HasClassification:
        ordinal_output(self._data, num_classes)
        return _HasClassification(self._data)

    def event_time(self) -> _HasEventTime:
        event_time(self._data)
        return _HasEventTime(self._data)

    def tokenize(self, vocab_size: int = 1000, concentration: float = 1.0) -> _Complete:
        tokens(self._data, vocab_size, concentration)
        return _Complete(self._data)


# -- Features --


class _HasFeatures:
    """Has observed features; entry point for outcomes and macros."""

    def __init__(self, data: TensorDict) -> None:
        self._data = data

    @property
    def data(self) -> TensorDict:
        return self._data

    # -- Low-level steps --

    def linear_output(self, weight: Tensor, prevalence: float) -> _HasOutput:
        linear_output(self._data, weight, prevalence)
        return _HasOutput(self._data)

    def competing_risks_events(self, vocab_size: int) -> _HasCompetingEvents:
        competing_risks_events(self._data, vocab_size)
        return _HasCompetingEvents(self._data)

    def tokenize(self, vocab_size: int = 1000, concentration: float = 1.0) -> _Complete:
        tokens(self._data, vocab_size, concentration)
        return _Complete(self._data)

    # -- High-level macros --

    def linear(self, weight: Tensor, prevalence: float) -> _HasOutput:
        return self.linear_output(weight, prevalence)

    def logistic(self, weight: Tensor, prevalence: float) -> _HasClassification:
        return self.linear_output(weight, prevalence).logistic_output()

    def multiclass(self, weight: Tensor, prevalence: float) -> _HasClassification:
        return self.linear_output(weight, prevalence).multiclass_output()

    def ordinal(
        self, weight: Tensor, prevalence: float, num_classes: int
    ) -> _HasClassification:
        return self.linear_output(weight, prevalence).ordinal_output(num_classes)

    def survival(
        self, weight: Tensor, prevalence: float, shape: float, rate: float
    ) -> _HasSurvival:
        return (
            self.linear_output(weight, prevalence)
            .event_time()
            .observation_time(shape, rate)
            .censor_time()
            .survival_indicators()
        )

    def mixture_cure(
        self, weight: Tensor, prevalence: float, shape: float, rate: float
    ) -> _HasSurvival:
        return (
            self.linear_output(weight, prevalence)
            .logistic_output()
            .event_time()
            .mixture_cure_censoring()
            .observation_time(shape, rate)
            .censor_time()
            .survival_indicators()
        )

    def competing_risk(self, vocab_size: int, boundaries: Tensor) -> _Complete:
        return (
            self.competing_risks_events(vocab_size)
            .competing_risk_indicators(vocab_size)
            .discrete_event_time(boundaries)
        )

    def multi_event(self, vocab_size: int, boundaries: Tensor) -> _Complete:
        return (
            self.competing_risks_events(vocab_size)
            .multi_event_times(vocab_size, boundaries[-1])
            .discrete_event_time(boundaries)
        )


# -- Setup --


class _HasRandomEffects:
    """Has random effects; needs observed features."""

    def __init__(self, data: TensorDict) -> None:
        self._data = data
        self._covariance_type: Literal["isotropic", "ar1", "lkj"] = "isotropic"
        self._covariance_kwargs: dict = {}

    @property
    def data(self) -> TensorDict:
        return self._data

    def covariance_ar1(self, correlation: float = 0.9) -> _HasRandomEffects:
        self._covariance_type = "ar1"
        self._covariance_kwargs = {"correlation": correlation}
        return self

    def covariance_isotropic(self) -> _HasRandomEffects:
        self._covariance_type = "isotropic"
        self._covariance_kwargs = {}
        return self

    def covariance_lkj(self, concentration: float = 1.0) -> _HasRandomEffects:
        self._covariance_type = "lkj"
        self._covariance_kwargs = {"concentration": concentration}
        return self

    def observed_features(
        self, num_timepoints: int, observed_std: float
    ) -> _HasFeatures:
        covariance = make_residual_covariance(
            num_timepoints, self._covariance_type, **self._covariance_kwargs
        )
        observed_features(self._data, num_timepoints, observed_std, covariance)
        return _HasFeatures(self._data)


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
        self,
        hidden_dim: int,
        intercept_std: float,
        slope_std: float = 0.0,
        correlation: float = 0.0,
    ) -> _HasRandomEffects:
        random_effects(self._data, hidden_dim, intercept_std, slope_std, correlation)
        return _HasRandomEffects(self._data)
