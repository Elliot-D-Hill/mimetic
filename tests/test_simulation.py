import torch

from mimetic import Simulation


def test_simulation_logistic_with_tokens() -> None:
    """Fluent chain with tokenization produces same shapes as free-function pipeline."""
    num_samples = 16
    num_timepoints = 6
    hidden_dim = 4
    data = (
        Simulation(num_samples)
        .random_effects(hidden_dim=hidden_dim, intercept_std=1.0, slope_std=0.05)
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
        .random_effects(hidden_dim=hidden_dim, intercept_std=1.0, slope_std=0.05)
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
        .random_effects(hidden_dim=hidden_dim, intercept_std=1.0)
        .covariance_ar1(correlation=0.8)
        .observed_features(num_timepoints=num_timepoints, observed_std=0.25)
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
        .random_effects(hidden_dim=hidden_dim, intercept_std=1.0)
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
        .random_effects(hidden_dim=hidden_dim, intercept_std=1.0)
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
        .random_effects(hidden_dim=hidden_dim, intercept_std=1.0)
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


def test_simulation_correlated_random_effects_shapes() -> None:
    """Joint Q sampling preserves output shapes."""
    num_samples = 64
    num_timepoints = 6
    hidden_dim = 4
    data = (
        Simulation(num_samples)
        .random_effects(
            hidden_dim=hidden_dim, intercept_std=1.0, slope_std=0.5, correlation=0.5
        )
        .observed_features(num_timepoints=num_timepoints, observed_std=0.25)
        .data
    )
    assert data["random_intercept"].shape == (num_samples, 1, hidden_dim)
    assert data["random_slope"].shape == (num_samples, 1, hidden_dim)
    assert data["features"].shape == (num_samples, num_timepoints, hidden_dim)


def test_simulation_positive_intercept_slope_correlation() -> None:
    """High positive correlation produces positively correlated intercepts and slopes."""
    torch.manual_seed(42)
    num_samples = 4096
    hidden_dim = 1
    data = (
        Simulation(num_samples)
        .random_effects(
            hidden_dim=hidden_dim, intercept_std=1.0, slope_std=1.0, correlation=0.8
        )
        .observed_features(num_timepoints=4, observed_std=0.1)
        .data
    )
    intercepts = data["random_intercept"].squeeze()  # [N]
    slopes = data["random_slope"].squeeze()  # [N]
    empirical_corr = torch.corrcoef(torch.stack([intercepts, slopes]))[0, 1]
    assert empirical_corr > 0.5, f"Expected positive correlation, got {empirical_corr}"


def test_simulation_zero_correlation_independence() -> None:
    """Zero correlation produces near-zero empirical intercept-slope correlation."""
    torch.manual_seed(42)
    num_samples = 4096
    hidden_dim = 1
    data = (
        Simulation(num_samples)
        .random_effects(
            hidden_dim=hidden_dim, intercept_std=1.0, slope_std=1.0, correlation=0.0
        )
        .observed_features(num_timepoints=4, observed_std=0.1)
        .data
    )
    intercepts = data["random_intercept"].squeeze()
    slopes = data["random_slope"].squeeze()
    empirical_corr = torch.corrcoef(torch.stack([intercepts, slopes]))[0, 1]
    assert abs(empirical_corr) < 0.1, (
        f"Expected near-zero correlation, got {empirical_corr}"
    )


def test_time_tensor_exists_after_observed_features() -> None:
    """observed_features stores a regular time grid as data['time']."""
    num_samples = 16
    num_timepoints = 6
    hidden_dim = 4
    data = (
        Simulation(num_samples)
        .random_effects(hidden_dim=hidden_dim, intercept_std=1.0)
        .observed_features(num_timepoints=num_timepoints, observed_std=0.25)
        .data
    )
    assert data["time"].shape == (num_samples, num_timepoints, 1)
    # All subjects share the same regular grid
    expected = torch.arange(num_timepoints, dtype=torch.float32).view(1, -1, 1)
    assert torch.allclose(data["time"], expected.expand(num_samples, -1, -1))


def test_simulation_data_accessible_at_any_stage() -> None:
    """The .data property is available before any task method."""
    sim = Simulation(8)
    assert sim.data["id"].shape == (8, 1, 1)
