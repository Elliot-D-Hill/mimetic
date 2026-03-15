from itertools import pairwise

import pytest
import torch

from simulacra import (
    AR1Covariance,
    CensoredState,
    CensoredStep,
    CompetingRisksState,
    CompetingRisksStep,
    DiscreteEventTimeStep,
    DiscreteResponseStep,
    DiscreteRiskState,
    DiscreteRiskStep,
    EventProcessState,
    EventTimeState,
    EventTimeStep,
    IndependentEventsStep,
    ObservedState,
    PredictorState,
    ResponseStep,
    RiskIndicatorState,
    RiskIndicatorStep,
    Simulation,
    SurvivalState,
    SurvivalStep,
    TokenizedState,
    TokenizedStep,
    activation,
    bernoulli,
    beta_response,
    binomial,
    categorical,
    censor_time,
    competing_risks,
    discretize_risk,
    event_time,
    gamma_response,
    gaussian,
    independent_events,
    linear,
    linear_predictor,
    log_normal,
    mixture_cure_censoring,
    mlp,
    multi_event,
    multinomial,
    negative_binomial,
    ordinal,
    poisson,
    random_effects,
    residual_covariance,
    risk_indicators,
    survival_indicators,
    tokens,
    zero_inflated_negative_binomial,
    zero_inflated_poisson,
)


def test_simulation_bernoulli_with_tokens() -> None:
    """Fluent chain with tokenization produces correct shapes.

    Tests: bernoulli -> tokenize chain on DiscreteResponseStep.
    """
    num_samples = 16
    num_timepoints = 6
    num_features = 4
    data = (
        Simulation(num_samples, num_timepoints, num_features)
        .random_effects(std=[1.0, 0.05])
        .bernoulli(prevalence=0.3)
        .tokenize(vocab_size=32)
        .data
    )
    assert data["y"].shape == (num_samples, num_timepoints, 1)
    assert data["X"].shape == (num_samples, num_timepoints, num_features)
    assert data["tokens"].shape == (num_samples, num_timepoints, 1)
    assert data["mu"].shape == (num_samples, num_timepoints, 1)


def test_simulation_bernoulli_without_tokens() -> None:
    """Bernoulli without .tokenize() produces mu but no tokens key.

    Tests: mu is present, tokens is absent.
    """
    num_samples = 16
    num_timepoints = 6
    num_features = 4
    data = (
        Simulation(num_samples, num_timepoints, num_features)
        .random_effects(std=[1.0, 0.05])
        .bernoulli(prevalence=0.3)
        .data
    )
    assert data["mu"].shape == (num_samples, num_timepoints, 1)
    assert "tokens" not in data.keys()


def test_simulation_survival_smoke() -> None:
    """Gaussian survival chain produces expected keys and shapes.

    Tests: gaussian -> event_time -> censor_time -> survival_indicators.
    """
    num_samples = 16
    num_timepoints = 6
    num_features = 4
    data = (
        Simulation(num_samples, num_timepoints, num_features)
        .random_effects(std=[1.0])
        .gaussian(0.25, covariance=AR1Covariance(correlation=0.8))
        .event_time()
        .censor_time()
        .survival_indicators()
        .data
    )
    assert data["indicator"].shape == (num_samples, 1, 1)
    assert data["observed_time"].shape == (num_samples, 1, 1)


def test_simulation_custom_pipeline() -> None:
    """Bernoulli with tokenization produces correct shapes via chaining.

    Tests: bernoulli -> tokenize end-to-end.
    """
    num_samples = 16
    num_timepoints = 6
    num_features = 4
    data = (
        Simulation(num_samples, num_timepoints, num_features)
        .random_effects(std=[1.0])
        .bernoulli(prevalence=0.3)
        .tokenize(vocab_size=32)
        .data
    )
    assert data["mu"].shape == (num_samples, num_timepoints, 1)
    assert data["y"].shape == (num_samples, num_timepoints, 1)
    assert data["tokens"].shape == (num_samples, num_timepoints, 1)


def test_simulation_step_by_step_survival() -> None:
    """Full survival chain across all typestate transitions.

    Tests: gaussian family into survival pipeline.
    """
    num_samples = 16
    num_timepoints = 6
    num_features = 4
    data = (
        Simulation(num_samples, num_timepoints, num_features)
        .random_effects(std=[1.0])
        .gaussian(0.25, covariance=AR1Covariance(correlation=0.8))
        .event_time()
        .censor_time()
        .survival_indicators()
        .data
    )
    assert data["indicator"].shape == (num_samples, 1, 1)
    assert data["observed_time"].shape == (num_samples, 1, 1)


def test_simulation_step_by_step_mixture_cure() -> None:
    """Mixture cure chain with discrete family and cure censoring.

    Tests: bernoulli -> event_time -> mixture_cure -> survival full chain.
    """
    num_samples = 16
    num_timepoints = 6
    num_features = 4
    data = (
        Simulation(num_samples, num_timepoints, num_features)
        .random_effects(std=[1.0])
        .bernoulli(prevalence=0.3)
        .event_time()
        .mixture_cure()
        .censor_time()
        .survival_indicators()
        .data
    )
    assert data["indicator"].shape == (num_samples, 1, 1)
    assert data["observed_time"].shape == (num_samples, 1, 1)
    assert data["y"].shape == (num_samples, num_timepoints, 1)


def test_simulation_chained_random_effects() -> None:
    """Fluent chain with two random_effects calls produces concatenated tensors.

    Tests: chaining .random_effects() twice gives gamma [N, q1+q2, 1]
    and Z [N, T, q1+q2].
    """
    num_samples = 16
    num_timepoints = 6
    num_features = 4
    q1, q2 = 2, 1
    data = (
        Simulation(num_samples, num_timepoints, num_features)
        .random_effects(std=[1.0, 0.5])
        .random_effects(std=[0.3])
        .data
    )
    assert data["gamma"].shape == (num_samples, q1 + q2, 1)
    assert data["Z"].shape == (num_samples, num_timepoints, q1 + q2)
    assert data["eta"].shape == (num_samples, num_timepoints, 1)


def test_simulation_correlated_random_effects_shapes() -> None:
    """Joint Q sampling preserves output shapes.

    Tests: correlated random effects produce correct gamma and eta shapes.
    """
    num_samples = 64
    num_timepoints = 6
    num_features = 4
    data = (
        Simulation(num_samples, num_timepoints, num_features)
        .random_effects(std=[1.0, 0.5], correlation=0.5)
        .data
    )
    assert data["gamma"].shape == (num_samples, 2, 1)
    assert data["eta"].shape == (num_samples, num_timepoints, 1)
    assert data["X"].shape == (num_samples, num_timepoints, num_features)


def test_simulation_positive_intercept_slope_correlation() -> None:
    """High positive correlation produces positively correlated intercepts and slopes.

    Tests: empirical gamma correlation reflects specified Q correlation.
    """
    torch.manual_seed(42)
    num_samples = 4096
    data = (
        Simulation(num_samples, 4, 1)
        .random_effects(std=[1.0, 1.0], correlation=0.8)
        .data
    )
    intercepts = data["gamma"][:, 0, :].squeeze()  # [N]
    slopes = data["gamma"][:, 1, :].squeeze()  # [N]
    empirical_corr = torch.corrcoef(torch.stack([intercepts, slopes]))[0, 1]
    assert empirical_corr > 0.5, f"Expected positive correlation, got {empirical_corr}"


def test_simulation_zero_correlation_independence() -> None:
    """Zero correlation produces near-zero empirical intercept-slope correlation.

    Tests: independent gamma components are uncorrelated.
    """
    torch.manual_seed(42)
    num_samples = 4096
    data = (
        Simulation(num_samples, 4, 1)
        .random_effects(std=[1.0, 1.0], correlation=0.0)
        .data
    )
    intercepts = data["gamma"][:, 0, :].squeeze()
    slopes = data["gamma"][:, 1, :].squeeze()
    empirical_corr = torch.corrcoef(torch.stack([intercepts, slopes]))[0, 1]
    assert abs(empirical_corr) < 0.1, (
        f"Expected near-zero correlation, got {empirical_corr}"
    )


def test_time_tensor_exists_after_construction() -> None:
    """Simulation() stores a regular time grid as data['time'].

    Tests: time grid is [0, 1, ..., T-1] expanded to [N, T, 1].
    """
    num_samples = 16
    num_timepoints = 6
    data = Simulation(num_samples, num_timepoints, 4).data
    assert data["time"].shape == (num_samples, num_timepoints, 1)
    expected = torch.arange(num_timepoints, dtype=torch.float32).view(1, -1, 1)
    assert torch.allclose(data["time"], expected.expand(num_samples, -1, -1))


def test_simulation_data_accessible_at_any_stage() -> None:
    """The .data property is available immediately after construction.

    Tests: PredictorState is accessible before choosing a family.
    """
    sim = Simulation(8, 4, 2)
    assert sim.data.batch_size == torch.Size([8])


def test_simulation_without_random_effects() -> None:
    """Pipeline works without .random_effects(); gamma and Z are absent.

    Tests: minimal PredictorState has eta, X, beta but no gamma, Z.
    """
    num_samples = 16
    num_timepoints = 6
    num_features = 4
    data = Simulation(num_samples, num_timepoints, num_features).data
    assert data["eta"].shape == (num_samples, num_timepoints, 1)
    assert data["X"].shape == (num_samples, num_timepoints, num_features)
    assert data["beta"].shape == (num_samples, num_features, 1)
    assert "gamma" not in data.keys()
    assert "Z" not in data.keys()


def test_general_q3_auto_Z() -> None:
    """q=3 random effects auto-generates polynomial Z and produces correct shapes.

    Tests: Vandermonde basis for q=3 gives correct gamma, eta, X, Z shapes.
    """
    num_samples = 32
    num_timepoints = 8
    num_features = 4
    q = 3
    data = (
        Simulation(num_samples, num_timepoints, num_features)
        .random_effects(std=[1.0, 0.3, 0.05])
        .data
    )
    assert data["gamma"].shape == (num_samples, q, 1)
    assert data["eta"].shape == (num_samples, num_timepoints, 1)
    assert data["X"].shape == (num_samples, num_timepoints, num_features)
    assert data["Z"].shape == (num_samples, num_timepoints, q)


def test_activation_applies_transform() -> None:
    """activation(torch.relu) zeroes negative eta and chains to bernoulli.

    Tests: nonlinearity applied to eta, downstream family chaining.
    """
    torch.manual_seed(0)
    num_samples = 64
    num_timepoints = 6
    num_features = 4
    sim = Simulation(num_samples, num_timepoints, num_features)
    step = sim.activation(torch.relu)
    data = step.data
    assert data["eta"].shape == (num_samples, num_timepoints, 1)
    assert (data["eta"] >= 0).all()
    labeled = step.bernoulli(0.3).data
    assert labeled["mu"].shape == (num_samples, num_timepoints, 1)


def test_linear_changes_eta_dimension() -> None:
    """linear(8) projects eta to [N, T, 8].

    Tests: eta dimension change, no y at PredictorState level.
    """
    torch.manual_seed(0)
    num_samples = 32
    num_timepoints = 6
    num_features = 4
    data = Simulation(num_samples, num_timepoints, num_features).linear(8).data
    assert data["eta"].shape == (num_samples, num_timepoints, 8)
    assert "y" not in data.keys()


def test_mlp_shapes() -> None:
    """mlp(32) applies hidden layer and projects back to input dimension.

    Tests: eta returns to input dim [N, T, 1] after MLP.
    """
    torch.manual_seed(0)
    num_samples = 32
    num_timepoints = 6
    num_features = 4
    data = Simulation(num_samples, num_timepoints, num_features).mlp(32).data
    assert data["eta"].shape == (num_samples, num_timepoints, 1)


def test_mlp_chains_to_bernoulli() -> None:
    """mlp(32) -> bernoulli(0.3) produces mu and y with correct shapes.

    Tests: end-to-end MLP + classification chain.
    """
    torch.manual_seed(0)
    num_samples = 32
    num_timepoints = 6
    num_features = 4
    data = (
        Simulation(num_samples, num_timepoints, num_features)
        .mlp(32, torch.relu)
        .bernoulli(0.3)
        .data
    )
    assert data["mu"].shape == (num_samples, num_timepoints, 1)
    assert data["y"].shape == (num_samples, num_timepoints, 1)


def test_user_supplied_X_and_beta() -> None:
    """User-supplied X and beta are used as-is; eta = X @ beta exactly.

    Tests: override path for fixed-effects tensors, verifiable eta.
    """
    torch.manual_seed(0)
    num_samples = 8
    num_timepoints = 4
    num_features = 3
    X = torch.ones(num_samples, num_timepoints, num_features)
    beta = torch.full((num_samples, num_features, 1), 0.5)
    data = Simulation(num_samples, num_timepoints, num_features, X=X, beta=beta).data
    assert torch.equal(data["X"], X)
    assert torch.equal(data["beta"], beta)
    assert torch.allclose(
        data["eta"], torch.full((num_samples, num_timepoints, 1), 1.5)
    )


def test_user_supplied_Z_and_gamma() -> None:
    """User-supplied Z and gamma are used as-is in random effects.

    Tests: override path for random-effects tensors, tensors stored unchanged.
    """
    torch.manual_seed(0)
    num_samples = 8
    num_timepoints = 4
    q = 2
    Z = torch.ones(num_samples, num_timepoints, q)
    gamma = torch.full((num_samples, q, 1), 0.25)
    data = (
        Simulation(num_samples, num_timepoints, 3)
        .random_effects(std=[1.0, 0.5], Z=Z, gamma=gamma)
        .data
    )
    assert torch.equal(data["Z"], Z)
    assert torch.equal(data["gamma"], gamma)


def test_predictor_X_and_beta_shapes() -> None:
    """Linear predictor with random effects produces correct X and beta shapes.

    Tests: X, beta, eta shapes after random_effects.
    """
    num_samples = 32
    num_timepoints = 8
    num_features = 3
    data = (
        Simulation(num_samples, num_timepoints, num_features)
        .random_effects(std=[1.0, 0.1])
        .data
    )
    assert data["eta"].shape == (num_samples, num_timepoints, 1)
    assert data["X"].shape == (num_samples, num_timepoints, num_features)
    assert data["beta"].shape == (num_samples, num_features, 1)


# ---------------------------------------------------------------------------
# Family-specific smoke tests
# ---------------------------------------------------------------------------


def test_gaussian_family_smoke() -> None:
    """Gaussian family: y continuous, mu = eta, noise present.

    Tests: identity link produces mu equal to eta, noise key exists.
    """
    torch.manual_seed(0)
    num_samples = 100
    num_timepoints = 10
    num_features = 5
    data = Simulation(num_samples, num_timepoints, num_features).gaussian(std=0.5).data
    expected_shape = (num_samples, num_timepoints, 1)
    assert data["y"].shape == expected_shape
    assert data["mu"].shape == expected_shape
    assert data["noise"].shape == expected_shape
    assert torch.allclose(data["y"], data["mu"] + data["noise"])


def test_poisson_family_smoke() -> None:
    """Poisson family: y non-negative integers, mu = exp(eta).

    Tests: log link, non-negative integer response.
    """
    torch.manual_seed(0)
    num_samples = 100
    num_timepoints = 10
    num_features = 5
    data = (
        Simulation(num_samples, num_timepoints, num_features)
        .random_effects(std=[0.2])
        .poisson()
        .data
    )
    expected_shape = (num_samples, num_timepoints, 1)
    assert data["y"].shape == expected_shape
    assert data["mu"].shape == expected_shape
    assert (data["y"] >= 0).all()
    assert (data["y"] == data["y"].floor()).all()
    assert "noise" not in data.keys()


def test_bernoulli_family_smoke() -> None:
    """Bernoulli family: y in {0, 1}, mu = sigmoid(eta).

    Tests: logit link, binary response, mu in (0, 1).
    """
    torch.manual_seed(0)
    num_samples = 100
    num_timepoints = 10
    num_features = 5
    data = Simulation(num_samples, num_timepoints, num_features).bernoulli().data
    expected_shape = (num_samples, num_timepoints, 1)
    assert data["y"].shape == expected_shape
    assert data["mu"].shape == expected_shape
    assert set(data["y"].unique().tolist()).issubset({0.0, 1.0})
    assert (data["mu"] > 0).all() and (data["mu"] < 1).all()
    assert "noise" not in data.keys()


def test_binomial_family_smoke() -> None:
    """Binomial family: y counts bounded by num_trials, mu via logit link.

    Tests: logit link, count response bounded by num_trials.
    """
    torch.manual_seed(0)
    num_samples = 100
    num_timepoints = 10
    num_features = 5
    num_trials = 10
    data = (
        Simulation(num_samples, num_timepoints, num_features)
        .binomial(num_trials, 0.3)
        .data
    )
    expected_shape = (num_samples, num_timepoints, 1)
    assert data["y"].shape == expected_shape
    assert data["mu"].shape == expected_shape
    assert (data["y"] >= 0).all()
    assert (data["y"] <= num_trials).all()
    assert (data["y"] == data["y"].floor()).all()


def test_multinomial_family_smoke() -> None:
    """Multinomial family: y count vectors summing to num_trials, mu = softmax(eta).

    Tests: softmax link, count vectors.
    """
    torch.manual_seed(0)
    num_samples = 100
    num_timepoints = 10
    num_features = 5
    K = 4
    num_trials = 10
    data = (
        Simulation(num_samples, num_timepoints, num_features)
        .linear(K)
        .multinomial(num_trials)
        .data
    )
    expected_shape = (num_samples, num_timepoints, K)
    assert data["y"].shape == expected_shape
    assert data["mu"].shape == expected_shape
    assert (data["y"] >= 0).all()
    sums = data["y"].sum(dim=-1)
    assert torch.equal(sums, torch.full_like(sums, num_trials))


def test_negative_binomial_family_smoke() -> None:
    """Negative binomial family: y non-negative integers, mu = exp(eta).

    Tests: log link, non-negative integer response.
    """
    torch.manual_seed(0)
    num_samples = 100
    num_timepoints = 10
    num_features = 5
    data = (
        Simulation(num_samples, num_timepoints, num_features)
        .negative_binomial(2.0)
        .data
    )
    expected_shape = (num_samples, num_timepoints, 1)
    assert data["y"].shape == expected_shape
    assert data["mu"].shape == expected_shape
    assert (data["y"] >= 0).all()
    assert (data["y"] == data["y"].floor()).all()


def test_gamma_family_smoke() -> None:
    """Gamma family: y positive continuous, mu = exp(eta).

    Tests: log link, positive continuous response.
    """
    torch.manual_seed(0)
    num_samples = 100
    num_timepoints = 10
    num_features = 5
    data = Simulation(num_samples, num_timepoints, num_features).gamma(2.0).data
    expected_shape = (num_samples, num_timepoints, 1)
    assert data["y"].shape == expected_shape
    assert data["mu"].shape == expected_shape
    assert (data["y"] > 0).all()


def test_beta_family_smoke() -> None:
    """Beta family: y in (0,1), mu = sigmoid(eta).

    Tests: logit link, bounded continuous response.
    """
    torch.manual_seed(0)
    num_samples = 100
    num_timepoints = 10
    num_features = 5
    data = Simulation(num_samples, num_timepoints, num_features).beta(5.0).data
    expected_shape = (num_samples, num_timepoints, 1)
    assert data["y"].shape == expected_shape
    assert data["mu"].shape == expected_shape
    assert (data["y"] > 0).all() and (data["y"] < 1).all()
    assert (data["mu"] > 0).all() and (data["mu"] < 1).all()


def test_log_normal_family_smoke() -> None:
    """Log-normal family: y positive continuous, mu = exp(eta + std^2/2).

    Tests: log link, positive continuous response.
    """
    torch.manual_seed(0)
    num_samples = 100
    num_timepoints = 10
    num_features = 5
    data = Simulation(num_samples, num_timepoints, num_features).log_normal(0.5).data
    expected_shape = (num_samples, num_timepoints, 1)
    assert data["y"].shape == expected_shape
    assert data["mu"].shape == expected_shape
    assert (data["y"] > 0).all()


def test_offset_preserves_shapes() -> None:
    """Offset in a fluent chain preserves all tensor shapes.

    Tests: offset applied before a count response does not alter output dimensions.
    """
    torch.manual_seed(0)
    num_samples = 100
    num_timepoints = 10
    num_features = 5
    expected_shape = (num_samples, num_timepoints, 1)
    log_exposure = torch.ones(expected_shape)
    data = (
        Simulation(num_samples, num_timepoints, num_features)
        .offset(log_exposure)
        .poisson()
        .data
    )
    assert data["y"].shape == expected_shape
    assert data["mu"].shape == expected_shape
    assert data["eta"].shape == expected_shape


def test_zero_inflated_poisson_family_smoke() -> None:
    """Zero-inflated Poisson family: y non-negative integers, mu = exp(eta).

    Tests: log link, non-negative integer response with excess zeros.
    """
    torch.manual_seed(0)
    num_samples = 100
    num_timepoints = 10
    num_features = 5
    data = (
        Simulation(num_samples, num_timepoints, num_features)
        .zero_inflated_poisson(0.3)
        .data
    )
    expected_shape = (num_samples, num_timepoints, 1)
    assert data["y"].shape == expected_shape
    assert data["mu"].shape == expected_shape
    assert (data["y"] >= 0).all()
    assert (data["y"] == data["y"].floor()).all()
    assert "noise" not in data.keys()


def test_zero_inflated_negative_binomial_family_smoke() -> None:
    """Zero-inflated NB family: y non-negative integers, mu = exp(eta).

    Tests: log link, non-negative integer response with excess zeros.
    """
    torch.manual_seed(0)
    num_samples = 100
    num_timepoints = 10
    num_features = 5
    data = (
        Simulation(num_samples, num_timepoints, num_features)
        .zero_inflated_negative_binomial(0.3, 2.0)
        .data
    )
    expected_shape = (num_samples, num_timepoints, 1)
    assert data["y"].shape == expected_shape
    assert data["mu"].shape == expected_shape
    assert (data["y"] >= 0).all()
    assert (data["y"] == data["y"].floor()).all()
    assert "noise" not in data.keys()


# ---------------------------------------------------------------------------
# Typestate machine property tests
# ---------------------------------------------------------------------------

N, T, P = 8, 4, 3
BOUNDARIES = torch.tensor([0.0, 1.0, 2.0, 5.0])


def _domain_methods(cls: type) -> set[str]:
    """Return public non-dunder callable names defined directly on cls."""
    return {
        name
        for name, obj in vars(cls).items()
        if not name.startswith("_") and callable(obj)
    }


def _assert_states_equal(
    fluent: dict[str, torch.Tensor], functional: dict[str, torch.Tensor]
) -> None:
    """Assert identical keys and element-wise equal tensors."""
    assert set(fluent.keys()) == set(functional.keys())
    for key in fluent:
        assert torch.equal(fluent[key], functional[key]), f"Mismatch on key '{key}'"


_STATE_CONTRACTS: dict[type, type] = {
    Simulation: PredictorState,
    ResponseStep: ObservedState,
    DiscreteResponseStep: ObservedState,
    TokenizedStep: TokenizedState,
    EventTimeStep: EventTimeState,
    DiscreteEventTimeStep: EventTimeState,
    CensoredStep: CensoredState,
    SurvivalStep: SurvivalState,
    IndependentEventsStep: EventProcessState,
    CompetingRisksStep: CompetingRisksState,
    RiskIndicatorStep: RiskIndicatorState,
    DiscreteRiskStep: DiscreteRiskState,
}

PIPELINE_PATHS = [
    (
        "gaussian_survival",
        [
            lambda s: s.gaussian(0.5),
            lambda s: s.event_time(),
            lambda s: s.censor_time(),
            lambda s: s.survival_indicators(),
        ],
    ),
    (
        "poisson_survival",
        [
            lambda s: s.poisson(),
            lambda s: s.event_time(),
            lambda s: s.censor_time(),
            lambda s: s.survival_indicators(),
        ],
    ),
    (
        "bernoulli_survival",
        [
            lambda s: s.bernoulli(0.3),
            lambda s: s.event_time(),
            lambda s: s.censor_time(),
            lambda s: s.survival_indicators(),
        ],
    ),
    (
        "bernoulli_mixture_cure",
        [
            lambda s: s.bernoulli(0.3),
            lambda s: s.event_time(),
            lambda s: s.mixture_cure(),
            lambda s: s.censor_time(),
            lambda s: s.survival_indicators(),
        ],
    ),
    (
        "gaussian_tokenized",
        [
            lambda s: s.gaussian(0.5),
            lambda s: s.tokenize(32),
            lambda s: s.event_time(),
            lambda s: s.censor_time(),
            lambda s: s.survival_indicators(),
        ],
    ),
    (
        "categorical_survival",
        [
            lambda s: s.linear(3),
            lambda s: s.categorical(),
            lambda s: s.event_time(),
            lambda s: s.censor_time(),
            lambda s: s.survival_indicators(),
        ],
    ),
    (
        "ordinal_survival",
        [
            lambda s: s.ordinal(4),
            lambda s: s.event_time(),
            lambda s: s.censor_time(),
            lambda s: s.survival_indicators(),
        ],
    ),
    (
        "cr_risk_indicators",
        [
            lambda s: s.linear(3),
            lambda s: s.competing_risks(),
            lambda s: s.risk_indicators(),
            lambda s: s.discretize(BOUNDARIES),
        ],
    ),
    (
        "cr_multi_event",
        [
            lambda s: s.linear(3),
            lambda s: s.competing_risks(),
            lambda s: s.multi_event(10.0),
            lambda s: s.discretize(BOUNDARIES),
        ],
    ),
    (
        "ie_multi_event",
        [
            lambda s: s.linear(3),
            lambda s: s.independent_events(0.3),
            lambda s: s.multi_event(10.0),
            lambda s: s.discretize(BOUNDARIES),
        ],
    ),
    (
        "binomial_survival",
        [
            lambda s: s.binomial(5, 0.3),
            lambda s: s.event_time(),
            lambda s: s.censor_time(),
            lambda s: s.survival_indicators(),
        ],
    ),
    (
        "multinomial_survival",
        [
            lambda s: s.linear(3),
            lambda s: s.multinomial(5),
            lambda s: s.event_time(),
            lambda s: s.censor_time(),
            lambda s: s.survival_indicators(),
        ],
    ),
    (
        "negative_binomial_survival",
        [
            lambda s: s.negative_binomial(2.0),
            lambda s: s.event_time(),
            lambda s: s.censor_time(),
            lambda s: s.survival_indicators(),
        ],
    ),
    (
        "gamma_survival",
        [
            lambda s: s.gamma(2.0),
            lambda s: s.event_time(),
            lambda s: s.censor_time(),
            lambda s: s.survival_indicators(),
        ],
    ),
    (
        "beta_survival",
        [
            lambda s: s.beta(5.0),
            lambda s: s.event_time(),
            lambda s: s.censor_time(),
            lambda s: s.survival_indicators(),
        ],
    ),
    (
        "log_normal_survival",
        [
            lambda s: s.log_normal(0.5),
            lambda s: s.event_time(),
            lambda s: s.censor_time(),
            lambda s: s.survival_indicators(),
        ],
    ),
    (
        "zero_inflated_poisson_survival",
        [
            lambda s: s.zero_inflated_poisson(0.3),
            lambda s: s.event_time(),
            lambda s: s.censor_time(),
            lambda s: s.survival_indicators(),
        ],
    ),
    (
        "zero_inflated_negative_binomial_survival",
        [
            lambda s: s.zero_inflated_negative_binomial(0.3, 2.0),
            lambda s: s.event_time(),
            lambda s: s.censor_time(),
            lambda s: s.survival_indicators(),
        ],
    ),
]


def _walk_pipeline(transitions: list) -> list:
    """Apply transitions sequentially from Simulation(N, T, P)."""
    steps: list[Simulation] = [Simulation(N, T, P)]
    for transition in transitions:
        steps.append(transition(steps[-1]))
    return steps


def _bisim_case(fluent_fn, functional_fn):
    """Build a bisimulation case that seeds before each call chain."""

    def case():
        torch.manual_seed(0)
        fluent = fluent_fn()
        torch.manual_seed(0)
        functional = functional_fn()
        return fluent, functional

    return case


BISIMULATION_CASES = [
    (
        "init",
        _bisim_case(
            lambda: Simulation(N, T, P).state, lambda: linear_predictor(N, T, P)
        ),
    ),
    (
        "random_effects",
        _bisim_case(
            lambda: Simulation(N, T, P).random_effects([0.5]).state,
            lambda: random_effects(linear_predictor(N, T, P), [0.5]),
        ),
    ),
    (
        "activation",
        _bisim_case(
            lambda: Simulation(N, T, P).activation(torch.relu).state,
            lambda: activation(linear_predictor(N, T, P), torch.relu),
        ),
    ),
    (
        "linear",
        _bisim_case(
            lambda: Simulation(N, T, P).linear(5).state,
            lambda: linear(linear_predictor(N, T, P), 5),
        ),
    ),
    (
        "mlp",
        _bisim_case(
            lambda: Simulation(N, T, P).mlp(16).state,
            lambda: mlp(linear_predictor(N, T, P), 16),
        ),
    ),
    (
        "gaussian",
        _bisim_case(
            lambda: Simulation(N, T, P).gaussian(0.5).state,
            lambda: gaussian(linear_predictor(N, T, P), 0.5, residual_covariance(T)),
        ),
    ),
    (
        "poisson",
        _bisim_case(
            lambda: Simulation(N, T, P).poisson().state,
            lambda: poisson(linear_predictor(N, T, P)),
        ),
    ),
    (
        "bernoulli",
        _bisim_case(
            lambda: Simulation(N, T, P).bernoulli(0.3).state,
            lambda: bernoulli(linear_predictor(N, T, P), 0.3),
        ),
    ),
    (
        "categorical",
        _bisim_case(
            lambda: Simulation(N, T, P).linear(3).categorical().state,
            lambda: categorical(linear(linear_predictor(N, T, P), 3)),
        ),
    ),
    (
        "ordinal",
        _bisim_case(
            lambda: Simulation(N, T, P).ordinal(4).state,
            lambda: ordinal(linear_predictor(N, T, P), 4),
        ),
    ),
    (
        "tokenize",
        _bisim_case(
            lambda: Simulation(N, T, P).gaussian(0.5).tokenize(32).state,
            lambda: tokens(
                gaussian(linear_predictor(N, T, P), 0.5, residual_covariance(T)), 32
            ),
        ),
    ),
    (
        "event_time",
        _bisim_case(
            lambda: Simulation(N, T, P).gaussian(0.5).event_time().state,
            lambda: event_time(
                gaussian(linear_predictor(N, T, P), 0.5, residual_covariance(T))
            ),
        ),
    ),
    (
        "censor_time",
        _bisim_case(
            lambda: Simulation(N, T, P).gaussian(0.5).event_time().censor_time().state,
            lambda: censor_time(
                event_time(
                    gaussian(linear_predictor(N, T, P), 0.5, residual_covariance(T))
                )
            ),
        ),
    ),
    (
        "survival_indicators",
        _bisim_case(
            lambda: (
                Simulation(N, T, P)
                .gaussian(0.5)
                .event_time()
                .censor_time()
                .survival_indicators()
                .state
            ),
            lambda: survival_indicators(
                censor_time(
                    event_time(
                        gaussian(linear_predictor(N, T, P), 0.5, residual_covariance(T))
                    )
                )
            ),
        ),
    ),
    (
        "mixture_cure",
        _bisim_case(
            lambda: (
                Simulation(N, T, P).bernoulli(0.3).event_time().mixture_cure().state
            ),
            lambda: mixture_cure_censoring(
                event_time(bernoulli(linear_predictor(N, T, P), 0.3))
            ),
        ),
    ),
    (
        "competing_risks",
        _bisim_case(
            lambda: Simulation(N, T, P).linear(3).competing_risks().state,
            lambda: competing_risks(linear(linear_predictor(N, T, P), 3)),
        ),
    ),
    (
        "independent_events",
        _bisim_case(
            lambda: Simulation(N, T, P).linear(3).independent_events(0.3).state,
            lambda: independent_events(linear(linear_predictor(N, T, P), 3), 0.3),
        ),
    ),
    (
        "risk_indicators",
        _bisim_case(
            lambda: (
                Simulation(N, T, P).linear(3).competing_risks().risk_indicators().state
            ),
            lambda: risk_indicators(
                competing_risks(linear(linear_predictor(N, T, P), 3))
            ),
        ),
    ),
    (
        "multi_event",
        _bisim_case(
            lambda: (
                Simulation(N, T, P).linear(3).competing_risks().multi_event(10.0).state
            ),
            lambda: multi_event(
                competing_risks(linear(linear_predictor(N, T, P), 3)), 10.0
            ),
        ),
    ),
    (
        "discretize",
        _bisim_case(
            lambda: (
                Simulation(N, T, P)
                .linear(3)
                .competing_risks()
                .risk_indicators()
                .discretize(BOUNDARIES)
                .state
            ),
            lambda: discretize_risk(
                risk_indicators(competing_risks(linear(linear_predictor(N, T, P), 3))),
                BOUNDARIES,
            ),
        ),
    ),
    (
        "binomial",
        _bisim_case(
            lambda: Simulation(N, T, P).binomial(5, 0.3).state,
            lambda: binomial(linear_predictor(N, T, P), 5, 0.3),
        ),
    ),
    (
        "multinomial",
        _bisim_case(
            lambda: Simulation(N, T, P).linear(3).multinomial(5).state,
            lambda: multinomial(linear(linear_predictor(N, T, P), 3), 5),
        ),
    ),
    (
        "negative_binomial",
        _bisim_case(
            lambda: Simulation(N, T, P).negative_binomial(2.0).state,
            lambda: negative_binomial(linear_predictor(N, T, P), 2.0),
        ),
    ),
    (
        "gamma",
        _bisim_case(
            lambda: Simulation(N, T, P).gamma(2.0).state,
            lambda: gamma_response(linear_predictor(N, T, P), 2.0),
        ),
    ),
    (
        "beta",
        _bisim_case(
            lambda: Simulation(N, T, P).beta(5.0).state,
            lambda: beta_response(linear_predictor(N, T, P), 5.0),
        ),
    ),
    (
        "log_normal",
        _bisim_case(
            lambda: Simulation(N, T, P).log_normal(0.5).state,
            lambda: log_normal(linear_predictor(N, T, P), 0.5),
        ),
    ),
    (
        "zero_inflated_poisson",
        _bisim_case(
            lambda: Simulation(N, T, P).zero_inflated_poisson(0.3).state,
            lambda: zero_inflated_poisson(linear_predictor(N, T, P), 0.3),
        ),
    ),
    (
        "zero_inflated_negative_binomial",
        _bisim_case(
            lambda: Simulation(N, T, P).zero_inflated_negative_binomial(0.3, 2.0).state,
            lambda: zero_inflated_negative_binomial(
                linear_predictor(N, T, P), 0.3, 2.0
            ),
        ),
    ),
]

REACHABILITY_CASES: list[tuple[type, object]] = [
    (Simulation, lambda: Simulation(N, T, P)),
    (ResponseStep, lambda: Simulation(N, T, P).gaussian(0.5)),
    (DiscreteResponseStep, lambda: Simulation(N, T, P).bernoulli(0.3)),
    (TokenizedStep, lambda: Simulation(N, T, P).gaussian(0.5).tokenize(32)),
    (EventTimeStep, lambda: Simulation(N, T, P).gaussian(0.5).event_time()),
    (DiscreteEventTimeStep, lambda: Simulation(N, T, P).bernoulli(0.3).event_time()),
    (
        CensoredStep,
        lambda: Simulation(N, T, P).gaussian(0.5).event_time().censor_time(),
    ),
    (
        SurvivalStep,
        lambda: (
            Simulation(N, T, P)
            .gaussian(0.5)
            .event_time()
            .censor_time()
            .survival_indicators()
        ),
    ),
    (CompetingRisksStep, lambda: Simulation(N, T, P).linear(3).competing_risks()),
    (
        IndependentEventsStep,
        lambda: Simulation(N, T, P).linear(3).independent_events(0.3),
    ),
    (
        RiskIndicatorStep,
        lambda: Simulation(N, T, P).linear(3).competing_risks().risk_indicators(),
    ),
    (
        DiscreteRiskStep,
        lambda: (
            Simulation(N, T, P)
            .linear(3)
            .competing_risks()
            .risk_indicators()
            .discretize(BOUNDARIES)
        ),
    ),
]


# ---------------------------------------------------------------------------
# 1. State monotonicity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path_id, transitions", PIPELINE_PATHS, ids=[p[0] for p in PIPELINE_PATHS]
)
def test_monotonicity_keys_only_grow(path_id: str, transitions: list) -> None:
    """State keys never shrink across transitions."""
    steps = _walk_pipeline(transitions)
    for prev, curr in pairwise(steps):
        assert set(prev.state.keys()) <= set(curr.state.keys())


# ---------------------------------------------------------------------------
# 2. Postcondition contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path_id, transitions", PIPELINE_PATHS, ids=[p[0] for p in PIPELINE_PATHS]
)
def test_postcondition_required_keys_present(path_id: str, transitions: list) -> None:
    """Each step's state satisfies its TypedDict required keys."""
    steps = _walk_pipeline(transitions)
    for step in steps:
        contract = _STATE_CONTRACTS[type(step)]
        assert contract.__required_keys__ <= set(step.state.keys())


# ---------------------------------------------------------------------------
# 3. Functional-fluent bisimulation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case_id, builder", BISIMULATION_CASES, ids=[c[0] for c in BISIMULATION_CASES]
)
def test_bisimulation_fluent_equals_functional(case_id: str, builder) -> None:
    """Fluent API produces identical state to functional API under same seed."""
    fluent, functional = builder()
    _assert_states_equal(fluent, functional)


# ---------------------------------------------------------------------------
# 4. Reachability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expected_type, builder",
    REACHABILITY_CASES,
    ids=[cls.__name__ for cls, _ in REACHABILITY_CASES],
)
def test_reachability_every_step_type_reachable(expected_type: type, builder) -> None:
    """Every Step class is reachable via the fluent API."""
    assert isinstance(builder(), expected_type)


# ---------------------------------------------------------------------------
# 5. Path structural equivalence
# ---------------------------------------------------------------------------


def test_path_equivalence_survival_required_keys() -> None:
    """All families converge to the same SurvivalState required keys."""
    terminals = [
        Simulation(N, T, P)
        .gaussian(0.5)
        .event_time()
        .censor_time()
        .survival_indicators(),
        Simulation(N, T, P).poisson().event_time().censor_time().survival_indicators(),
        Simulation(N, T, P)
        .bernoulli(0.3)
        .event_time()
        .censor_time()
        .survival_indicators(),
        Simulation(N, T, P)
        .linear(3)
        .categorical()
        .event_time()
        .censor_time()
        .survival_indicators(),
        Simulation(N, T, P).ordinal(4).event_time().censor_time().survival_indicators(),
        Simulation(N, T, P)
        .binomial(5, 0.3)
        .event_time()
        .censor_time()
        .survival_indicators(),
        Simulation(N, T, P)
        .linear(3)
        .multinomial(5)
        .event_time()
        .censor_time()
        .survival_indicators(),
        Simulation(N, T, P)
        .negative_binomial(2.0)
        .event_time()
        .censor_time()
        .survival_indicators(),
        Simulation(N, T, P).gamma(2.0).event_time().censor_time().survival_indicators(),
        Simulation(N, T, P).beta(5.0).event_time().censor_time().survival_indicators(),
        Simulation(N, T, P)
        .log_normal(0.5)
        .event_time()
        .censor_time()
        .survival_indicators(),
        Simulation(N, T, P)
        .zero_inflated_poisson(0.3)
        .event_time()
        .censor_time()
        .survival_indicators(),
        Simulation(N, T, P)
        .zero_inflated_negative_binomial(0.3, 2.0)
        .event_time()
        .censor_time()
        .survival_indicators(),
    ]
    for terminal in terminals:
        assert SurvivalState.__required_keys__ <= set(terminal.state.keys())


def test_path_equivalence_discrete_risk_required_keys() -> None:
    """All event processes converge to the same DiscreteRiskState required keys."""
    terminals = [
        Simulation(N, T, P)
        .linear(3)
        .competing_risks()
        .risk_indicators()
        .discretize(BOUNDARIES),
        Simulation(N, T, P)
        .linear(3)
        .competing_risks()
        .multi_event(10.0)
        .discretize(BOUNDARIES),
        Simulation(N, T, P)
        .linear(3)
        .independent_events(0.3)
        .multi_event(10.0)
        .discretize(BOUNDARIES),
    ]
    for terminal in terminals:
        assert DiscreteRiskState.__required_keys__ <= set(terminal.state.keys())


# ---------------------------------------------------------------------------
# 6. Terminal absorption
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [SurvivalStep, DiscreteRiskStep])
def test_terminal_absorption_no_transition_methods(cls: type) -> None:
    """Terminal steps expose no public transition methods."""
    assert _domain_methods(cls) == set()


@pytest.mark.parametrize(
    "cls",
    [
        ResponseStep,
        DiscreteResponseStep,
        TokenizedStep,
        EventTimeStep,
        DiscreteEventTimeStep,
        CensoredStep,
        CompetingRisksStep,
        IndependentEventsStep,
        RiskIndicatorStep,
    ],
    ids=lambda cls: cls.__name__,
)
def test_nonterminal_has_transition_methods(cls: type) -> None:
    """Non-terminal steps expose at least one transition method."""
    assert len(_domain_methods(cls)) > 0


# ---------------------------------------------------------------------------
# 7. Invariant preservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path_id, transitions", PIPELINE_PATHS, ids=[p[0] for p in PIPELINE_PATHS]
)
def test_batch_dimension_consistent_across_all_keys(
    path_id: str, transitions: list
) -> None:
    """Every tensor in the state has shape[0] == N at every stage."""
    steps = _walk_pipeline(transitions)
    for step in steps:
        for key, tensor in step.state.items():
            assert tensor.shape[0] == N, (
                f"Key '{key}' has shape[0]={tensor.shape[0]}, expected {N}"
            )
