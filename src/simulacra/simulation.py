from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor

from .covariance import ResidualCovarianceSpec, residual_covariance
from .functional import (
    activation,
    bernoulli,
    categorical,
    gaussian,
    linear,
    linear_predictor,
    mlp,
    ordinal,
    poisson,
    random_effects,
    tokens,
)
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
    TokenizedState,
)
from .survival import (
    censor_time,
    competing_risks,
    discretize_risk,
    event_time,
    independent_events,
    mixture_cure_censoring,
    multi_event,
    risk_indicators,
    survival_indicators,
)

ObservationCovariance = Tensor | ResidualCovarianceSpec


def _to_tensordict(state: Mapping[str, Any]) -> TensorDict:
    first = next(iter(state.values()))
    return TensorDict(dict(state), batch_size=[first.shape[0]])


class _Step[T: Mapping[str, Any]]:
    """Base for pipeline steps providing .data export."""

    state: T

    @property
    def data(self) -> TensorDict:
        """Export state as a TensorDict."""
        return _to_tensordict(self.state)


class Simulation:
    """Entry point for building synthetic datasets.

    Generates the linear predictor eta = X*beta on construction.
    Call `.random_effects()` to upgrade to a GLMM: eta = X*beta + Z*gamma.
    Then choose a response distribution (`.gaussian()`, `.bernoulli()`, etc.).

    Parameters
    ----------
    num_samples
        Number of samples (subjects) to simulate.
    num_timepoints
        Number of time points T.
    num_features
        Number of design matrix features p.
    X
        Optional design matrix [N, T, p]; random if omitted.
    beta
        Optional coefficients [N, p, 1]; random if omitted.
    time
        Optional observation schedule [N, T, 1]; arange(T) if omitted.

    See Also
    --------
    linear_predictor : Functional equivalent.

    Examples
    --------
    >>> sim = Simulation(10, 5, 3)
    >>> sim.data["eta"].shape
    torch.Size([10, 5, 1])
    """

    def __init__(
        self,
        num_samples: int,
        num_timepoints: int,
        num_features: int,
        *,
        X: Tensor | None = None,
        beta: Tensor | None = None,
        time: Tensor | None = None,
    ) -> None:
        self.state: PredictorState = linear_predictor(
            num_samples, num_timepoints, num_features, X=X, beta=beta, time=time
        )

    @classmethod
    def _from_state(cls, state: PredictorState) -> Simulation:
        sim = object.__new__(cls)
        sim.state = state
        return sim

    @property
    def data(self) -> TensorDict:
        """Export state as a TensorDict."""
        return _to_tensordict(self.state)

    def random_effects(
        self,
        std: Sequence[float] | Tensor | float,
        correlation: Tensor | float = 0.0,
        Z: Tensor | None = None,
        gamma: Tensor | None = None,
    ) -> Simulation:
        """Add random effects to the linear predictor.

        See Also
        --------
        random_effects : Functional equivalent.
        """
        return Simulation._from_state(
            random_effects(self.state, std, correlation, Z=Z, gamma=gamma)
        )

    def activation(self, fn: Callable[[Tensor], Tensor]) -> Simulation:
        """Apply a nonlinear activation to eta.

        See Also
        --------
        activation : Functional equivalent.
        """
        return Simulation._from_state(activation(self.state, fn))

    def linear(self, out_features: int, weight: Tensor | None = None) -> Simulation:
        """Apply a random linear projection to eta.

        See Also
        --------
        linear : Functional equivalent.
        """
        return Simulation._from_state(linear(self.state, out_features, weight))

    def mlp(
        self,
        hidden_features: int,
        fn: Callable[[Tensor], Tensor] = F.relu,
        out_features: int | None = None,
    ) -> Simulation:
        """Apply a single hidden-layer MLP to eta.

        See Also
        --------
        mlp : Functional equivalent.
        """
        return Simulation._from_state(
            mlp(self.state, hidden_features, fn, out_features)
        )

    # --- Event processes (branch from PredictorState) ---

    def competing_risks(self, shape: float | Tensor = 1.0) -> CompetingRisksStep:
        """Sample per-risk Weibull failure times.

        See Also
        --------
        competing_risks : Functional equivalent.
        """
        return CompetingRisksStep(competing_risks(self.state, shape))

    def independent_events(self, prevalence: float = 0.1) -> IndependentEventsStep:
        """Generate multilabel event mask via independent Bernoulli draws.

        See Also
        --------
        independent_events : Functional equivalent.
        """
        return IndependentEventsStep(independent_events(self.state, prevalence))

    # --- Response distributions (choose one) ---

    def gaussian(
        self, std: float, *, covariance: ObservationCovariance | None = None
    ) -> ResponseStep:
        """Sample Gaussian response (identity link).

        See Also
        --------
        gaussian : Functional equivalent.
        """
        num_timepoints = self.state["eta"].shape[1]
        covariance_matrix = (
            covariance
            if isinstance(covariance, Tensor)
            else residual_covariance(num_timepoints, covariance)
        )
        return ResponseStep(gaussian(self.state, std, covariance_matrix))

    def poisson(self) -> ResponseStep:
        """Sample Poisson response (log link).

        See Also
        --------
        poisson : Functional equivalent.
        """
        return ResponseStep(poisson(self.state))

    def bernoulli(self, prevalence: float = 0.5) -> DiscreteResponseStep:
        """Sample Bernoulli response (logit link).

        See Also
        --------
        bernoulli : Functional equivalent.
        """
        return DiscreteResponseStep(bernoulli(self.state, prevalence))

    def categorical(self) -> DiscreteResponseStep:
        """Sample categorical response (softmax link).

        See Also
        --------
        categorical : Functional equivalent.
        """
        return DiscreteResponseStep(categorical(self.state))

    def ordinal(self, num_classes: int) -> DiscreteResponseStep:
        """Sample ordinal response (cumulative logit).

        See Also
        --------
        ordinal : Functional equivalent.
        """
        return DiscreteResponseStep(ordinal(self.state, num_classes))


@dataclass(frozen=True)
class ResponseStep(_Step[ObservedState]):
    """Observed response ready for tokenization or survival analysis."""

    state: ObservedState

    def tokenize(
        self, vocab_size: int = 1000, concentration: float = 1.0
    ) -> TokenizedStep:
        """Tokenize the design matrix.

        See Also
        --------
        tokens : Functional equivalent.
        """
        return TokenizedStep(tokens(self.state, vocab_size, concentration))

    def event_time(self) -> EventTimeStep:
        """Sample an exponential event time.

        See Also
        --------
        event_time : Functional equivalent.
        """
        return EventTimeStep(event_time(self.state))


@dataclass(frozen=True)
class DiscreteResponseStep(_Step[ObservedState]):
    """Discrete observed response ready for tokenization or survival analysis."""

    state: ObservedState

    def tokenize(
        self, vocab_size: int = 1000, concentration: float = 1.0
    ) -> DiscreteResponseStep:
        """Tokenize the design matrix.

        See Also
        --------
        tokens : Functional equivalent.
        """
        return DiscreteResponseStep(tokens(self.state, vocab_size, concentration))

    def event_time(self) -> DiscreteEventTimeStep:
        """Sample an exponential event time.

        See Also
        --------
        event_time : Functional equivalent.
        """
        return DiscreteEventTimeStep(event_time(self.state))


@dataclass(frozen=True)
class TokenizedStep(_Step[TokenizedState]):
    """Tokenized response ready for survival analysis."""

    state: TokenizedState

    def event_time(self) -> EventTimeStep:
        """Sample an exponential event time.

        See Also
        --------
        event_time : Functional equivalent.
        """
        return EventTimeStep(event_time(self.state))


@dataclass(frozen=True)
class EventTimeStep(_Step[EventTimeState]):
    """Event time ready for censoring."""

    state: EventTimeState

    def censor_time(self) -> CensoredStep:
        """Sample a uniform censor time.

        See Also
        --------
        censor_time : Functional equivalent.
        """
        return CensoredStep(censor_time(self.state))


@dataclass(frozen=True)
class DiscreteEventTimeStep(EventTimeStep):
    """Discrete event time with optional mixture-cure adjustment."""

    def mixture_cure(self) -> EventTimeStep:
        """Set event time to infinity for cured subjects.

        See Also
        --------
        mixture_cure_censoring : Functional equivalent.
        """
        return EventTimeStep(mixture_cure_censoring(self.state))


@dataclass(frozen=True)
class CensoredStep(_Step[CensoredState]):
    """Censored event ready for survival indicator computation."""

    state: CensoredState

    def survival_indicators(self) -> SurvivalStep:
        """Compute event and censoring indicators.

        See Also
        --------
        survival_indicators : Functional equivalent.
        """
        return SurvivalStep(survival_indicators(self.state))


@dataclass(frozen=True)
class SurvivalStep(_Step[SurvivalState]):
    """Terminal survival state with event indicators."""

    state: SurvivalState


# ---------------------------------------------------------------------------
# Event process steps
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndependentEventsStep(_Step[EventProcessState]):
    """Multilabel event mask ready for suffix-minimum encoding."""

    state: EventProcessState

    def multi_event(self, horizon: float | Tensor | None = None) -> RiskIndicatorStep:
        """Compute per-risk TTE via suffix-minimum.

        See Also
        --------
        multi_event : Functional equivalent.
        """
        return RiskIndicatorStep(multi_event(self.state, horizon))


@dataclass(frozen=True)
class CompetingRisksStep(_Step[CompetingRisksState]):
    """Competing risks ready for indicator encoding."""

    state: CompetingRisksState

    def risk_indicators(self) -> RiskIndicatorStep:
        """One-hot encode the winning risk.

        See Also
        --------
        risk_indicators : Functional equivalent.
        """
        return RiskIndicatorStep(risk_indicators(self.state))

    def multi_event(self, horizon: float | Tensor | None = None) -> RiskIndicatorStep:
        """Compute per-risk TTE via suffix-minimum.

        See Also
        --------
        multi_event : Functional equivalent.
        """
        return RiskIndicatorStep(multi_event(self.state, horizon))


@dataclass(frozen=True)
class RiskIndicatorStep(_Step[RiskIndicatorState]):
    """Risk indicators ready for discretization."""

    state: RiskIndicatorState

    def discretize(self, boundaries: Tensor) -> DiscreteRiskStep:
        """Discretize event times into interval bins.

        See Also
        --------
        discretize_risk : Functional equivalent.
        """
        return DiscreteRiskStep(discretize_risk(self.state, boundaries))


@dataclass(frozen=True)
class DiscreteRiskStep(_Step[DiscreteRiskState]):
    """Terminal discrete competing-risks state."""

    state: DiscreteRiskState
