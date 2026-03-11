import torch

from mimetic import Simulation


def test_simulation_logistic_with_tokens() -> None:
    """Fluent chain with tokenization produces same shapes as free-function pipeline."""
    num_samples = 16
    num_timepoints = 6
    hidden_dim = 4
    data = (
        Simulation(num_samples)
        .random_effects(hidden_dim=hidden_dim, latent_std=1.0, slope_std=0.05)
        .covariance_ar1(correlation=0.8)
        .observed_features(num_timepoints=num_timepoints, observed_std=0.25)
        .logistic(weight=torch.randn(1, hidden_dim), prevalence=0.3)
        .tokenize(vocab_size=32)
        .data
    )
    assert data["features"].shape == (num_samples, num_timepoints, hidden_dim)
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
        .random_effects(hidden_dim=hidden_dim, latent_std=1.0, slope_std=0.05)
        .covariance_ar1(correlation=0.8)
        .observed_features(num_timepoints=num_timepoints, observed_std=0.25)
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
        .random_effects(hidden_dim=hidden_dim, latent_std=1.0)
        .covariance_ar1(correlation=0.8)
        .observed_features(num_timepoints=num_timepoints, observed_std=0.25)
        .survival(
            weight=torch.randn(1, hidden_dim),
            prevalence=0.3,
            shape=2.0,
            rate=1.0,
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
        .random_effects(hidden_dim=hidden_dim, latent_std=1.0)
        .covariance_ar1(correlation=0.8)
        .observed_features(num_timepoints=num_timepoints, observed_std=0.25)
        .linear_output(weight=torch.randn(1, hidden_dim), prevalence=0.3)
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
        .random_effects(hidden_dim=hidden_dim, latent_std=1.0)
        .covariance_ar1(correlation=0.8)
        .observed_features(num_timepoints=num_timepoints, observed_std=0.25)
        .linear_output(weight=torch.randn(1, hidden_dim), prevalence=0.3)
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
        .random_effects(hidden_dim=hidden_dim, latent_std=1.0)
        .covariance_ar1(correlation=0.8)
        .observed_features(num_timepoints=num_timepoints, observed_std=0.25)
        .linear_output(weight=torch.randn(1, hidden_dim), prevalence=0.3)
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


def test_simulation_data_accessible_at_any_stage() -> None:
    """The .data property is available before any task method."""
    sim = Simulation(8)
    assert sim.data["id"].shape == (8, 1, 1)
