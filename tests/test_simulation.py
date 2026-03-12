import torch

from mimetic import AR1Covariance, Simulation


def test_simulation_logistic_with_tokens() -> None:
    """Fluent chain with tokenization produces same shapes as free-function pipeline."""
    num_samples = 16
    num_timepoints = 6
    num_features = 4
    data = (
        Simulation(
            num_samples,
            num_timepoints,
            num_features,
            0.25,
            covariance=AR1Covariance(correlation=0.8),
        )
        .random_effects(stds=[1.0, 0.05])
        .logistic(prevalence=0.3)
        .tokenize(vocab_size=32)
        .data
    )
    assert data["y"].shape == (num_samples, num_timepoints, 1)
    assert data["X"].shape == (num_samples, num_timepoints, num_features)
    assert data["tokens"].shape == (num_samples, num_timepoints, 1)
    assert data["probability"].shape == (num_samples, num_timepoints, 1)
    assert data["label"].shape == (num_samples, num_timepoints, 1)


def test_simulation_logistic_without_tokens() -> None:
    """Task without .tokenize() produces outcome but no tokens key."""
    num_samples = 16
    num_timepoints = 6
    num_features = 4
    data = (
        Simulation(
            num_samples,
            num_timepoints,
            num_features,
            0.25,
            covariance=AR1Covariance(correlation=0.8),
        )
        .random_effects(stds=[1.0, 0.05])
        .logistic(prevalence=0.3)
        .data
    )
    assert data["probability"].shape == (num_samples, num_timepoints, 1)
    assert "tokens" not in data.keys()


def test_simulation_survival_smoke() -> None:
    """Survival task chain produces expected keys and shapes."""
    num_samples = 16
    num_timepoints = 6
    num_features = 4
    data = (
        Simulation(
            num_samples,
            num_timepoints,
            num_features,
            0.25,
            covariance=AR1Covariance(correlation=0.8),
        )
        .random_effects(stds=[1.0])
        .event_time()
        .observation_time(shape=2.0, rate=1.0)
        .censor_time()
        .survival_indicators()
        .data
    )
    assert data["indicator"].shape == (num_samples, 1, 1)
    assert data["observed_time"].shape == (num_samples, 1, 1)


def test_simulation_custom_pipeline() -> None:
    """Low-level pipeline methods compose freely via chaining."""
    num_samples = 16
    num_timepoints = 6
    num_features = 4
    data = (
        Simulation(
            num_samples,
            num_timepoints,
            num_features,
            0.25,
            covariance=AR1Covariance(correlation=0.8),
        )
        .random_effects(stds=[1.0])
        .logistic(prevalence=0.3)
        .tokenize(vocab_size=32)
        .data
    )
    assert data["probability"].shape == (num_samples, num_timepoints, 1)
    assert data["label"].shape == (num_samples, num_timepoints, 1)
    assert data["tokens"].shape == (num_samples, num_timepoints, 1)


def test_simulation_step_by_step_survival() -> None:
    """Low-level survival chain across all typestate transitions."""
    num_samples = 16
    num_timepoints = 6
    num_features = 4
    data = (
        Simulation(
            num_samples,
            num_timepoints,
            num_features,
            0.25,
            covariance=AR1Covariance(correlation=0.8),
        )
        .random_effects(stds=[1.0])
        .event_time()
        .observation_time(shape=2.0, rate=1.0)
        .censor_time()
        .survival_indicators()
        .data
    )
    assert data["indicator"].shape == (num_samples, 1, 1)
    assert data["observed_time"].shape == (num_samples, 1, 1)


def test_simulation_step_by_step_mixture_cure() -> None:
    """Low-level mixture cure chain with classification and cure censoring."""
    num_samples = 16
    num_timepoints = 6
    num_features = 4
    data = (
        Simulation(
            num_samples,
            num_timepoints,
            num_features,
            0.25,
            covariance=AR1Covariance(correlation=0.8),
        )
        .random_effects(stds=[1.0])
        .logistic(prevalence=0.3)
        .event_time()
        .mixture_cure()
        .observation_time(shape=2.0, rate=1.0)
        .censor_time()
        .survival_indicators()
        .data
    )
    assert data["indicator"].shape == (num_samples, 1, 1)
    assert data["observed_time"].shape == (num_samples, 1, 1)
    assert data["label"].shape == (num_samples, num_timepoints, 1)


def test_simulation_correlated_random_effects_shapes() -> None:
    """Joint Q sampling preserves output shapes."""
    num_samples = 64
    num_timepoints = 6
    num_features = 4
    data = (
        Simulation(num_samples, num_timepoints, num_features, 0.25)
        .random_effects(stds=[1.0, 0.5], correlation=0.5)
        .data
    )
    assert data["gamma"].shape == (num_samples, 2, 1)
    assert data["y"].shape == (num_samples, num_timepoints, 1)
    assert data["X"].shape == (num_samples, num_timepoints, num_features)


def test_simulation_positive_intercept_slope_correlation() -> None:
    """High positive correlation produces positively correlated intercepts and slopes."""
    torch.manual_seed(42)
    num_samples = 4096
    data = (
        Simulation(num_samples, 4, 1, 0.1)
        .random_effects(stds=[1.0, 1.0], correlation=0.8)
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
    data = (
        Simulation(num_samples, 4, 1, 0.1)
        .random_effects(stds=[1.0, 1.0], correlation=0.0)
        .data
    )
    intercepts = data["gamma"][:, 0, :].squeeze()
    slopes = data["gamma"][:, 1, :].squeeze()
    empirical_corr = torch.corrcoef(torch.stack([intercepts, slopes]))[0, 1]
    assert abs(empirical_corr) < 0.1, (
        f"Expected near-zero correlation, got {empirical_corr}"
    )


def test_time_tensor_exists_after_observations() -> None:
    """Simulation() stores a regular time grid as data['time']."""
    num_samples = 16
    num_timepoints = 6
    data = Simulation(num_samples, num_timepoints, 4, 0.25).data
    assert data["time"].shape == (num_samples, num_timepoints, 1)
    expected = torch.arange(num_timepoints, dtype=torch.float32).view(1, -1, 1)
    assert torch.allclose(data["time"], expected.expand(num_samples, -1, -1))


def test_simulation_data_accessible_at_any_stage() -> None:
    """The .data property is available immediately after construction."""
    sim = Simulation(8, 4, 2, 0.25)
    assert sim.data.batch_size == torch.Size([8])


def test_simulation_without_random_effects() -> None:
    """Pipeline works without .random_effects(); gamma and U are absent."""
    num_samples = 16
    num_timepoints = 6
    num_features = 4
    data = Simulation(num_samples, num_timepoints, num_features, 0.25).data
    assert data["y"].shape == (num_samples, num_timepoints, 1)
    assert data["X"].shape == (num_samples, num_timepoints, num_features)
    assert data["beta"].shape == (num_samples, num_features, 1)
    assert "gamma" not in data.keys()
    assert "U" not in data.keys()


def test_general_q3_auto_U() -> None:
    """q=3 random effects auto-generates polynomial U and produces correct shapes."""
    num_samples = 32
    num_timepoints = 8
    num_features = 4
    q = 3
    data = (
        Simulation(num_samples, num_timepoints, num_features, 0.25)
        .random_effects(stds=[1.0, 0.3, 0.05])
        .data
    )
    assert data["gamma"].shape == (num_samples, q, 1)
    assert data["y"].shape == (num_samples, num_timepoints, 1)
    assert data["X"].shape == (num_samples, num_timepoints, num_features)
    assert data["U"].shape == (num_samples, num_timepoints, q)


def test_observations_X_and_beta_shapes() -> None:
    """y = Xβ + ε always generates X and beta with correct shapes."""
    num_samples = 32
    num_timepoints = 8
    num_features = 3
    data = (
        Simulation(num_samples, num_timepoints, num_features, 0.25)
        .random_effects(stds=[1.0, 0.1])
        .data
    )
    assert data["y"].shape == (num_samples, num_timepoints, 1)
    assert data["X"].shape == (num_samples, num_timepoints, num_features)
    assert data["beta"].shape == (num_samples, num_features, 1)
