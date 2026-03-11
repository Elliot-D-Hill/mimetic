from typing import Any, cast, get_type_hints

import pytest
import torch

from mimetic import AR1Covariance, Simulation
from mimetic.simulation import (
    Complete,
    HasClassification,
    HasEta,
    HasObservations,
    HasRandomEffects,
    HasSurvival,
)


def test_simulation_logistic_with_tokens() -> None:
    """Fluent chain with tokenization produces same shapes as free-function pipeline."""
    num_samples = 16
    num_timepoints = 6
    hidden_dim = 4
    data = (
        Simulation(num_samples)
        .random_effects(hidden_dim=hidden_dim, stds=[1.0, 0.05])
        .observations(
            num_timepoints=num_timepoints,
            observed_std=0.25,
            covariance=AR1Covariance(correlation=0.8),
        )
        .logistic(weight=torch.randn(1, hidden_dim), prevalence=0.3)
        .tokenize(vocab_size=32)
        .data
    )
    assert data["y"].shape == (num_samples, num_timepoints, hidden_dim)
    assert data["tokens"].shape == (num_samples, num_timepoints, 1)
    assert data["probability"].shape == (num_samples, 1, 1)
    assert data["label"].shape == (num_samples, 1, 1)


def test_simulation_logistic_without_tokens() -> None:
    """Task without .tokenize() produces outcome but no tokens key."""
    num_samples = 16
    num_timepoints = 6
    hidden_dim = 4
    data = (
        Simulation(num_samples)
        .random_effects(hidden_dim=hidden_dim, stds=[1.0, 0.05])
        .observations(
            num_timepoints=num_timepoints,
            observed_std=0.25,
            covariance=AR1Covariance(correlation=0.8),
        )
        .logistic(weight=torch.randn(1, hidden_dim), prevalence=0.3)
        .data
    )
    assert data["probability"].shape == (num_samples, 1, 1)
    assert "tokens" not in data.keys()


def test_simulation_survival_smoke() -> None:
    """Survival task chain produces expected keys and shapes."""
    num_samples = 16
    num_timepoints = 6
    hidden_dim = 4
    data = (
        Simulation(num_samples)
        .random_effects(hidden_dim=hidden_dim, stds=[1.0])
        .observations(
            num_timepoints=num_timepoints,
            observed_std=0.25,
            covariance=AR1Covariance(correlation=0.8),
        )
        .survival(
            weight=torch.randn(1, hidden_dim), prevalence=0.3, shape=2.0, rate=1.0
        )
        .data
    )
    assert data["indicator"].shape == (num_samples, 1, 1)
    assert data["observed_time"].shape == (num_samples, 1, 1)


def test_simulation_custom_pipeline() -> None:
    """Low-level pipeline methods compose freely via chaining."""
    num_samples = 16
    num_timepoints = 6
    hidden_dim = 4
    data = (
        Simulation(num_samples)
        .random_effects(hidden_dim=hidden_dim, stds=[1.0])
        .observations(
            num_timepoints=num_timepoints,
            observed_std=0.25,
            covariance=AR1Covariance(correlation=0.8),
        )
        .linear_predictor(weight=torch.randn(1, hidden_dim), prevalence=0.3)
        .logistic_output()
        .tokenize(vocab_size=32)
        .data
    )
    assert data["probability"].shape == (num_samples, 1, 1)
    assert data["label"].shape == (num_samples, 1, 1)
    assert data["tokens"].shape == (num_samples, num_timepoints, 1)


def test_simulation_step_by_step_survival() -> None:
    """Low-level survival chain across all typestate transitions."""
    num_samples = 16
    num_timepoints = 6
    hidden_dim = 4
    data = (
        Simulation(num_samples)
        .random_effects(hidden_dim=hidden_dim, stds=[1.0])
        .observations(
            num_timepoints=num_timepoints,
            observed_std=0.25,
            covariance=AR1Covariance(correlation=0.8),
        )
        .linear_predictor(weight=torch.randn(1, hidden_dim), prevalence=0.3)
        .event_time()
        .observation_time(shape=2.0, rate=1.0)
        .censor_time()
        .survival_indicators()
        .tokenize(vocab_size=32)
        .data
    )
    assert data["indicator"].shape == (num_samples, 1, 1)
    assert data["observed_time"].shape == (num_samples, 1, 1)
    assert data["tokens"].shape == (num_samples, num_timepoints, 1)


def test_simulation_step_by_step_mixture_cure() -> None:
    """Low-level mixture cure chain with classification and cure censoring."""
    num_samples = 16
    num_timepoints = 6
    hidden_dim = 4
    data = (
        Simulation(num_samples)
        .random_effects(hidden_dim=hidden_dim, stds=[1.0])
        .observations(
            num_timepoints=num_timepoints,
            observed_std=0.25,
            covariance=AR1Covariance(correlation=0.8),
        )
        .linear_predictor(weight=torch.randn(1, hidden_dim), prevalence=0.3)
        .logistic_output()
        .event_time()
        .mixture_cure_censoring()
        .observation_time(shape=2.0, rate=1.0)
        .censor_time()
        .survival_indicators()
        .data
    )
    assert data["indicator"].shape == (num_samples, 1, 1)
    assert data["observed_time"].shape == (num_samples, 1, 1)
    assert data["label"].shape == (num_samples, 1, 1)


def test_simulation_correlated_random_effects_shapes() -> None:
    """Joint Q sampling preserves output shapes."""
    num_samples = 64
    num_timepoints = 6
    hidden_dim = 4
    data = (
        Simulation(num_samples)
        .random_effects(hidden_dim=hidden_dim, stds=[1.0, 0.5], correlation=0.5)
        .observations(num_timepoints=num_timepoints, observed_std=0.25)
        .data
    )
    assert data["gamma"].shape == (num_samples, 2, hidden_dim)
    assert data["y"].shape == (num_samples, num_timepoints, hidden_dim)


def test_simulation_positive_intercept_slope_correlation() -> None:
    """High positive correlation produces positively correlated intercepts and slopes."""
    torch.manual_seed(42)
    num_samples = 4096
    hidden_dim = 1
    data = (
        Simulation(num_samples)
        .random_effects(hidden_dim=hidden_dim, stds=[1.0, 1.0], correlation=0.8)
        .observations(num_timepoints=4, observed_std=0.1)
        .data
    )
    intercepts = data["gamma"][:, 0, :].squeeze()  # [N]
    slopes = data["gamma"][:, 1, :].squeeze()  # [N]
    empirical_corr = torch.corrcoef(torch.stack([intercepts, slopes]))[0, 1]
    assert empirical_corr > 0.5, f"Expected positive correlation, got {empirical_corr}"


def test_simulation_zero_correlation_independence() -> None:
    """Zero correlation produces near-zero empirical intercept-slope correlation."""
    torch.manual_seed(42)
    num_samples = 4096
    hidden_dim = 1
    data = (
        Simulation(num_samples)
        .random_effects(hidden_dim=hidden_dim, stds=[1.0, 1.0], correlation=0.0)
        .observations(num_timepoints=4, observed_std=0.1)
        .data
    )
    intercepts = data["gamma"][:, 0, :].squeeze()
    slopes = data["gamma"][:, 1, :].squeeze()
    empirical_corr = torch.corrcoef(torch.stack([intercepts, slopes]))[0, 1]
    assert abs(empirical_corr) < 0.1, (
        f"Expected near-zero correlation, got {empirical_corr}"
    )


def test_time_tensor_exists_after_observations() -> None:
    """observations() stores a regular time grid as data['time']."""
    num_samples = 16
    num_timepoints = 6
    hidden_dim = 4
    data = (
        Simulation(num_samples)
        .random_effects(hidden_dim=hidden_dim, stds=[1.0])
        .observations(num_timepoints=num_timepoints, observed_std=0.25)
        .data
    )
    assert data["time"].shape == (num_samples, num_timepoints, 1)
    expected = torch.arange(num_timepoints, dtype=torch.float32).view(1, -1, 1)
    assert torch.allclose(data["time"], expected.expand(num_samples, -1, -1))


def test_simulation_data_accessible_at_any_stage() -> None:
    """The .data property is available before any task method."""
    sim = Simulation(8)
    assert sim.data["id"].shape == (8, 1, 1)


def test_simulation_uses_public_typestate_protocols_in_annotations() -> None:
    """Public method annotations expose Protocol stages instead of private wrappers."""
    assert get_type_hints(Simulation.random_effects)["return"] is HasRandomEffects
    assert get_type_hints(HasRandomEffects.observations)["return"] is HasObservations
    assert get_type_hints(HasObservations.linear_predictor)["return"] is HasEta
    assert get_type_hints(HasObservations.survival)["return"] is HasSurvival
    assert get_type_hints(HasEta.logistic_output)["return"] is HasClassification
    assert get_type_hints(HasSurvival.tokenize)["return"] is Complete


def test_simulation_invalid_runtime_step_raises_clear_error() -> None:
    """Calling a hidden-by-protocol method from the wrong runtime stage fails fast."""
    stage = cast(Any, Simulation(8).random_effects(hidden_dim=4, stds=[1.0]))
    with pytest.raises(AttributeError, match="linear_predictor"):
        stage.linear_predictor(weight=torch.randn(1, 4), prevalence=0.3)


def test_general_q3_auto_U() -> None:
    """q=3 random effects auto-generates polynomial U and produces correct shapes."""
    num_samples = 32
    num_timepoints = 8
    hidden_dim = 4
    q = 3
    data = (
        Simulation(num_samples)
        .random_effects(hidden_dim=hidden_dim, stds=[1.0, 0.3, 0.05])
        .observations(num_timepoints=num_timepoints, observed_std=0.25)
        .data
    )
    assert data["gamma"].shape == (num_samples, q, hidden_dim)
    assert data["y"].shape == (num_samples, num_timepoints, hidden_dim)
    assert data["U"].shape == (num_samples, num_timepoints, q)


def test_fixed_effects_X_beta() -> None:
    """y = Xβ + Uγ + ε with num_fixed_effects produces correct shapes."""
    num_samples = 32
    num_timepoints = 8
    hidden_dim = 4
    p = 3
    data = (
        Simulation(num_samples)
        .random_effects(hidden_dim=hidden_dim, stds=[1.0, 0.1])
        .observations(
            num_timepoints=num_timepoints, observed_std=0.25, num_fixed_effects=p
        )
        .data
    )
    assert data["y"].shape == (num_samples, num_timepoints, hidden_dim)
    assert data["X"].shape == (num_samples, num_timepoints, p)
    assert data["beta"].shape == (num_samples, p, hidden_dim)
