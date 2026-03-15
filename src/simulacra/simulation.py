"""
Fluent builder API for composing synthetic dataset pipelines.

Each ``Simulation`` method returns a typed step object, constraining which
operations are legal at each stage and making the pipeline self-documenting.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch.nn.functional as F
from beartype import beartype
from tensordict import TensorDict
from torch import Tensor

from .covariance import ResidualCovarianceSpec, residual_covariance
from .functional import (
    activation,
    bernoulli,
    beta_response,
    binomial,
    categorical,
    gamma_response,
    gaussian,
    linear,
    linear_predictor,
    log_normal,
    mlp,
    multinomial,
    negative_binomial,
    offset,
    ordinal,
    poisson,
    random_effects,
    tokens,
    zero_inflated_negative_binomial,
    zero_inflated_poisson,
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
from .types import Correlation, PositiveFloat, PositiveInt, UnitInterval

ObservationCovariance = Tensor | ResidualCovarianceSpec


def _to_tensordict(state: Mapping[str, Any]) -> TensorDict:
    """
    Pack a state mapping into a batched TensorDict.

    Parameters
    ----------
    state : Mapping[str, Any]
        Mapping of tensor names to tensors; all tensors must share the same
        leading batch dimension.

    Returns
    -------
    TensorDict
        Batched TensorDict with ``batch_size=[N]``.
    """
    first = next(iter(state.values()))
    return TensorDict(state, batch_size=[first.shape[0]])


class _Step[T: Mapping[str, Any]]:
    """Base for pipeline steps providing .data export."""

    state: T

    @property
    def data(self) -> TensorDict:
        """
        Export state as a TensorDict.

        Returns
        -------
        TensorDict
            All state tensors packed into a batched TensorDict.
        """
        return _to_tensordict(self.state)


class Simulation(_Step[PredictorState]):
    """
    Entry point for building synthetic datasets.

    Generates the linear predictor eta = X*beta on construction.
    Call `.random_effects()` to upgrade to a GLMM: eta = X*beta + Z*gamma.
    Then choose a response distribution (`.gaussian()`, `.bernoulli()`, etc.).

    Parameters
    ----------
    num_samples : int
        Number of samples (subjects) to simulate.
    num_timepoints : int
        Number of time points T.
    num_features : int
        Number of design matrix features p.
    X : Tensor, optional
        Design matrix [N, T, p]; random if omitted.
    beta : Tensor, optional
        Coefficients [N, p, 1]; random if omitted.
    time : Tensor, optional
        Observation schedule [N, T, 1]; arange(T) if omitted.

    See Also
    --------
    linear_predictor : Functional equivalent.

    Examples
    --------
    >>> sim = Simulation(10, 5, 3)
    >>> sim.data["eta"].shape
    torch.Size([10, 5, 1])
    """

    @beartype
    def __init__(
        self,
        num_samples: PositiveInt,
        num_timepoints: PositiveInt,
        num_features: PositiveInt,
        *,
        X: Tensor | None = None,
        beta: Tensor | None = None,
        time: Tensor | None = None,
    ) -> None:
        """
        Initialize the simulation by computing the linear predictor.

        Parameters
        ----------
        num_samples : int
            Number of samples (subjects) to simulate.
        num_timepoints : int
            Number of time points T.
        num_features : int
            Number of design matrix features p.
        X : Tensor, optional
            Design matrix [N, T, p]; random if omitted.
        beta : Tensor, optional
            Coefficients [N, p, 1]; random if omitted.
        time : Tensor, optional
            Observation schedule [N, T, 1]; arange(T) if omitted.
        """
        self.state: PredictorState = linear_predictor(
            num_samples, num_timepoints, num_features, X=X, beta=beta, time=time
        )

    @classmethod
    def _from_state(cls, state: PredictorState) -> Simulation:
        """
        Construct a Simulation directly from an existing PredictorState.

        Parameters
        ----------
        state : PredictorState
            Pre-built predictor state to wrap.

        Returns
        -------
        Simulation
            New instance backed by the given state.
        """
        sim = object.__new__(cls)
        sim.state = state
        return sim

    @beartype
    def random_effects(
        self,
        std: Sequence[PositiveFloat] | Tensor | PositiveFloat,
        correlation: Tensor | Correlation = 0.0,
        Z: Tensor | None = None,
        gamma: Tensor | None = None,
    ) -> Simulation:
        """
        Add random effects to the linear predictor.

        Chaining multiple calls concatenates gamma and Z along the q
        dimension, building composite random-effects tensors for
        multiple grouping factors.

        Parameters
        ----------
        std : Sequence[float] or Tensor or float
            Standard deviations of the random effects; one per grouping factor.
        correlation : Tensor or float, optional
            Correlation among random effects; scalar or matrix. Default 0.0.
        Z : Tensor, optional
            Random-effects design matrix [N, T, q]; random if omitted.
        gamma : Tensor, optional
            Random-effects coefficients [N, q, 1]; drawn from the implied
            covariance if omitted.

        Returns
        -------
        Simulation
            Self, for method chaining.

        See Also
        --------
        random_effects : Functional equivalent.
        """
        return Simulation._from_state(
            random_effects(self.state, std, correlation, Z=Z, gamma=gamma)
        )

    def offset(self, log_exposure: Tensor) -> Simulation:
        """
        Add a log-exposure offset to eta.

        Parameters
        ----------
        log_exposure : Tensor
            Log-transformed exposure; broadcastable to eta's shape.

        Returns
        -------
        Simulation
            Self, for method chaining.

        See Also
        --------
        offset : Functional equivalent.
        """
        return Simulation._from_state(offset(self.state, log_exposure))

    def activation(self, fn: Callable[[Tensor], Tensor]) -> Simulation:
        """
        Apply a nonlinear activation to eta.

        Parameters
        ----------
        fn : Callable[[Tensor], Tensor]
            Elementwise activation function, e.g. ``torch.sigmoid``.

        Returns
        -------
        Simulation
            Self, for method chaining.

        See Also
        --------
        activation : Functional equivalent.
        """
        return Simulation._from_state(activation(self.state, fn))

    def linear(self, out_features: int, weight: Tensor | None = None) -> Simulation:
        """
        Apply a random linear projection to eta.

        Parameters
        ----------
        out_features : int
            Output dimension of the projection.
        weight : Tensor, optional
            Projection weight matrix [out_features, in_features]; random if omitted.

        Returns
        -------
        Simulation
            Self, for method chaining.

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
        """
        Apply a single hidden-layer MLP to eta.

        Parameters
        ----------
        hidden_features : int
            Width of the single hidden layer.
        fn : Callable[[Tensor], Tensor], optional
            Hidden-layer activation. Default ``torch.nn.functional.relu``.
        out_features : int, optional
            Output dimension; defaults to the input dimension if omitted.

        Returns
        -------
        Simulation
            Self, for method chaining.

        See Also
        --------
        mlp : Functional equivalent.
        """
        return Simulation._from_state(
            mlp(self.state, hidden_features, fn, out_features)
        )

    # --- Event processes (branch from PredictorState) ---

    @beartype
    def competing_risks(
        self, shape: PositiveFloat | Tensor = 1.0
    ) -> CompetingRisksStep:
        """
        Sample per-risk Weibull failure times.

        Parameters
        ----------
        shape : float or Tensor, optional
            Weibull shape parameter; scalar or per-risk tensor. Default 1.0
            (exponential).

        Returns
        -------
        CompetingRisksStep
            Next pipeline step.

        See Also
        --------
        competing_risks : Functional equivalent.
        """
        return CompetingRisksStep(competing_risks(self.state, shape))

    @beartype
    def independent_events(
        self, prevalence: UnitInterval = 0.1
    ) -> IndependentEventsStep:
        """
        Generate multilabel event mask via independent Bernoulli draws.

        Parameters
        ----------
        prevalence : float, optional
            Marginal probability of each event occurring. Default 0.1.

        Returns
        -------
        IndependentEventsStep
            Next pipeline step.

        See Also
        --------
        independent_events : Functional equivalent.
        """
        return IndependentEventsStep(independent_events(self.state, prevalence))

    # --- Response distributions (choose one) ---

    @beartype
    def gaussian(
        self, std: PositiveFloat, *, covariance: ObservationCovariance | None = None
    ) -> ResponseStep:
        """
        Sample Gaussian response (identity link).

        Parameters
        ----------
        std : float
            Observation noise standard deviation.
        covariance : Tensor or ResidualCovarianceSpec, optional
            Full covariance matrix [T, T] or a spec for residual covariance;
            diagonal ``std**2 * I`` if omitted.

        Returns
        -------
        ResponseStep
            Next pipeline step.

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
        """
        Sample Poisson response (log link).

        Returns
        -------
        ResponseStep
            Next pipeline step.

        See Also
        --------
        poisson : Functional equivalent.
        """
        return ResponseStep(poisson(self.state))

    @beartype
    def bernoulli(self, prevalence: UnitInterval = 0.5) -> DiscreteResponseStep:
        """
        Sample Bernoulli response (logit link).

        Parameters
        ----------
        prevalence : float, optional
            Marginal probability of the positive class. Default 0.5.

        Returns
        -------
        DiscreteResponseStep
            Next pipeline step.

        See Also
        --------
        bernoulli : Functional equivalent.
        """
        return DiscreteResponseStep(bernoulli(self.state, prevalence))

    def categorical(self) -> DiscreteResponseStep:
        """
        Sample categorical response (softmax link).

        Returns
        -------
        DiscreteResponseStep
            Next pipeline step.

        See Also
        --------
        categorical : Functional equivalent.
        """
        return DiscreteResponseStep(categorical(self.state))

    @beartype
    def ordinal(self, num_classes: PositiveInt) -> DiscreteResponseStep:
        """
        Sample ordinal response (cumulative logit).

        Parameters
        ----------
        num_classes : int
            Number of ordinal categories.

        Returns
        -------
        DiscreteResponseStep
            Next pipeline step.

        See Also
        --------
        ordinal : Functional equivalent.
        """
        return DiscreteResponseStep(ordinal(self.state, num_classes))

    @beartype
    def binomial(
        self, num_trials: PositiveInt, prevalence: UnitInterval = 0.5
    ) -> ResponseStep:
        """
        Sample binomial response (logit link).

        Parameters
        ----------
        num_trials : int
            Number of Bernoulli trials per observation.
        prevalence : float, optional
            Base rate for the logit shift. Default 0.5.

        Returns
        -------
        ResponseStep
            Next pipeline step.

        See Also
        --------
        binomial : Functional equivalent.
        """
        return ResponseStep(binomial(self.state, num_trials, prevalence))

    @beartype
    def multinomial(self, num_trials: PositiveInt) -> ResponseStep:
        """
        Sample multinomial response (softmax link).

        Parameters
        ----------
        num_trials : int
            Number of draws per observation.

        Returns
        -------
        ResponseStep
            Next pipeline step.

        See Also
        --------
        multinomial : Functional equivalent.
        """
        return ResponseStep(multinomial(self.state, num_trials))

    @beartype
    def negative_binomial(self, concentration: PositiveFloat) -> ResponseStep:
        """
        Sample negative binomial response (log link).

        Parameters
        ----------
        concentration : float
            Dispersion parameter.

        Returns
        -------
        ResponseStep
            Next pipeline step.

        See Also
        --------
        negative_binomial : Functional equivalent.
        """
        return ResponseStep(negative_binomial(self.state, concentration))

    @beartype
    def zero_inflated_poisson(self, inflation: UnitInterval) -> ResponseStep:
        """
        Sample zero-inflated Poisson response (log link).

        Parameters
        ----------
        inflation : float
            Probability of structural zero, in [0, 1).

        Returns
        -------
        ResponseStep
            Next pipeline step.

        See Also
        --------
        zero_inflated_poisson : Functional equivalent.
        """
        return ResponseStep(zero_inflated_poisson(self.state, inflation))

    @beartype
    def zero_inflated_negative_binomial(
        self, inflation: UnitInterval, concentration: PositiveFloat
    ) -> ResponseStep:
        """
        Sample zero-inflated negative binomial response (log link).

        Parameters
        ----------
        inflation : float
            Probability of structural zero, in [0, 1).
        concentration : float
            Dispersion parameter.

        Returns
        -------
        ResponseStep
            Next pipeline step.

        See Also
        --------
        zero_inflated_negative_binomial : Functional equivalent.
        """
        return ResponseStep(
            zero_inflated_negative_binomial(self.state, inflation, concentration)
        )

    @beartype
    def gamma(self, concentration: PositiveFloat) -> ResponseStep:
        """
        Sample Gamma response (log link).

        Parameters
        ----------
        concentration : float
            Shape parameter.

        Returns
        -------
        ResponseStep
            Next pipeline step.

        See Also
        --------
        gamma_response : Functional equivalent.
        """
        return ResponseStep(gamma_response(self.state, concentration))

    @beartype
    def beta(self, precision: PositiveFloat) -> ResponseStep:
        """
        Sample Beta response (logit link).

        Parameters
        ----------
        precision : float
            Precision parameter.

        Returns
        -------
        ResponseStep
            Next pipeline step.

        See Also
        --------
        beta_response : Functional equivalent.
        """
        return ResponseStep(beta_response(self.state, precision))

    @beartype
    def log_normal(self, std: PositiveFloat) -> ResponseStep:
        """
        Sample log-normal response (log link).

        Parameters
        ----------
        std : float
            Standard deviation of the underlying normal distribution.

        Returns
        -------
        ResponseStep
            Next pipeline step.

        See Also
        --------
        log_normal : Functional equivalent.
        """
        return ResponseStep(log_normal(self.state, std))


@dataclass(frozen=True)
class ResponseStep(_Step[ObservedState]):
    """Observed response ready for tokenization or survival analysis."""

    state: ObservedState

    def tokenize(
        self, vocab_size: int = 1000, concentration: float = 1.0
    ) -> TokenizedStep:
        """
        Tokenize the design matrix.

        Parameters
        ----------
        vocab_size : int, optional
            Number of token types. Default 1000.
        concentration : float, optional
            Dirichlet concentration parameter for token probabilities. Default 1.0.

        Returns
        -------
        TokenizedStep
            Next pipeline step.

        See Also
        --------
        tokens : Functional equivalent.
        """
        return TokenizedStep(tokens(self.state, vocab_size, concentration))

    def event_time(self) -> EventTimeStep:
        """
        Sample an exponential event time.

        Returns
        -------
        EventTimeStep
            Next pipeline step.

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
        """
        Tokenize the design matrix.

        Parameters
        ----------
        vocab_size : int, optional
            Number of token types. Default 1000.
        concentration : float, optional
            Dirichlet concentration parameter for token probabilities. Default 1.0.

        Returns
        -------
        DiscreteResponseStep
            Next pipeline step.

        See Also
        --------
        tokens : Functional equivalent.
        """
        return DiscreteResponseStep(tokens(self.state, vocab_size, concentration))

    def event_time(self) -> DiscreteEventTimeStep:
        """
        Sample an exponential event time.

        Returns
        -------
        DiscreteEventTimeStep
            Next pipeline step.

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
        """
        Sample an exponential event time.

        Returns
        -------
        EventTimeStep
            Next pipeline step.

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
        """
        Sample a uniform censor time.

        Returns
        -------
        CensoredStep
            Next pipeline step.

        See Also
        --------
        censor_time : Functional equivalent.
        """
        return CensoredStep(censor_time(self.state))


@dataclass(frozen=True)
class DiscreteEventTimeStep(EventTimeStep):
    """Discrete event time with optional mixture-cure adjustment."""

    def mixture_cure(self) -> EventTimeStep:
        """
        Set event time to infinity for cured subjects.

        Returns
        -------
        EventTimeStep
            Next pipeline step.

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
        """
        Compute event and censoring indicators.

        Returns
        -------
        SurvivalStep
            Next pipeline step.

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
        """
        Compute per-risk TTE via suffix-minimum.

        Parameters
        ----------
        horizon : float or Tensor, optional
            Maximum follow-up time; events beyond it are censored. Uses the
            maximum observed time if omitted.

        Returns
        -------
        RiskIndicatorStep
            Next pipeline step.

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
        """
        One-hot encode the winning risk.

        Returns
        -------
        RiskIndicatorStep
            Next pipeline step.

        See Also
        --------
        risk_indicators : Functional equivalent.
        """
        return RiskIndicatorStep(risk_indicators(self.state))

    def multi_event(self, horizon: float | Tensor | None = None) -> RiskIndicatorStep:
        """
        Compute per-risk TTE via suffix-minimum.

        Parameters
        ----------
        horizon : float or Tensor, optional
            Maximum follow-up time; events beyond it are censored. Uses the
            maximum observed time if omitted.

        Returns
        -------
        RiskIndicatorStep
            Next pipeline step.

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
        """
        Discretize event times into interval bins.

        Parameters
        ----------
        boundaries : Tensor
            Sorted 1-D tensor of interval boundaries defining the time bins.

        Returns
        -------
        DiscreteRiskStep
            Next pipeline step.

        See Also
        --------
        discretize_risk : Functional equivalent.
        """
        return DiscreteRiskStep(discretize_risk(self.state, boundaries))


@dataclass(frozen=True)
class DiscreteRiskStep(_Step[DiscreteRiskState]):
    """Terminal discrete competing-risks state."""

    state: DiscreteRiskState
