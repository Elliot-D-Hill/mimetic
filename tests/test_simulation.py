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


def test_activation_preserves_noise_and_applies_transform() -> None:
    """activation(torch.relu) zeroes negative eta, preserves noise, and chains to logistic.

    Tests: nonlinearity applied to eta, explicit noise tensor preserved, downstream chaining.
    """
    torch.manual_seed(0)
    num_samples = 64
    num_timepoints = 6
    num_features = 4
    sim = Simulation(num_samples, num_timepoints, num_features, 0.25)
    noise_before = sim.data["noise"]
    step = sim.activation(torch.relu)
    data = step.data
    assert data["eta"].shape == (num_samples, num_timepoints, 1)
    assert (data["eta"] >= 0).all()
    assert torch.allclose(data["noise"], noise_before)
    assert torch.allclose(data["y"], data["eta"] + data["noise"])
    labeled = step.logistic(0.3).data
    assert labeled["probability"].shape == (num_samples, num_timepoints, 1)


def test_linear_changes_eta_dimension() -> None:
    """linear(8) projects eta to [N,T,8], y broadcasts noise [N,T,1] to [N,T,8].

    Tests: eta dimension change, y shape matches, noise tensor unchanged.
    """
    torch.manual_seed(0)
    num_samples = 32
    num_timepoints = 6
    num_features = 4
    sim = Simulation(num_samples, num_timepoints, num_features, 0.25)
    noise_before = sim.data["noise"]
    data = sim.linear(8).data
    assert data["eta"].shape == (num_samples, num_timepoints, 8)
    assert data["y"].shape == (num_samples, num_timepoints, 8)
    assert data["noise"].shape == (num_samples, num_timepoints, 1)
    assert torch.allclose(data["noise"], noise_before)


def test_mlp_shapes_and_noise_preservation() -> None:
    """mlp(32) applies hidden layer and projects back, preserving noise tensor.

    Tests: eta returns to input dim [N,T,1], noise invariant, y = eta + noise.
    """
    torch.manual_seed(0)
    num_samples = 32
    num_timepoints = 6
    num_features = 4
    sim = Simulation(num_samples, num_timepoints, num_features, 0.25)
    noise_before = sim.data["noise"]
    data = sim.mlp(32).data
    assert data["eta"].shape == (num_samples, num_timepoints, 1)
    assert data["noise"].shape == (num_samples, num_timepoints, 1)
    assert torch.allclose(data["noise"], noise_before)
    assert torch.allclose(data["y"], data["eta"] + data["noise"])


def test_mlp_chains_to_logistic() -> None:
    """mlp(32) → logistic(0.3) produces probability and label with correct shapes.

    Tests: end-to-end MLP + classification chain.
    """
    torch.manual_seed(0)
    num_samples = 32
    num_timepoints = 6
    num_features = 4
    data = (
        Simulation(num_samples, num_timepoints, num_features, 0.25)
        .mlp(32, torch.relu)
        .logistic(0.3)
        .data
    )
    assert data["probability"].shape == (num_samples, num_timepoints, 1)
    assert data["label"].shape == (num_samples, num_timepoints, 1)


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
    data = Simulation(
        num_samples, num_timepoints, num_features, 0.25, X=X, beta=beta
    ).data
    assert torch.equal(data["X"], X)
    assert torch.equal(data["beta"], beta)
    assert torch.allclose(
        data["eta"], torch.full((num_samples, num_timepoints, 1), 1.5)
    )


def test_user_supplied_U_and_gamma() -> None:
    """User-supplied U and gamma are used as-is in random effects.

    Tests: override path for random-effects tensors, tensors stored unchanged.
    """
    torch.manual_seed(0)
    num_samples = 8
    num_timepoints = 4
    q = 2
    U = torch.ones(num_samples, num_timepoints, q)
    gamma = torch.full((num_samples, q, 1), 0.25)
    data = (
        Simulation(num_samples, num_timepoints, 3, 0.25)
        .random_effects(stds=[1.0, 0.5], U=U, gamma=gamma)
        .data
    )
    assert torch.equal(data["U"], U)
    assert torch.equal(data["gamma"], gamma)


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
