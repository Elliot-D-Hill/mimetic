import torch

from simulacra import AR1Covariance, Simulation


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
    assert data["y"].shape == (num_samples, num_timepoints, 1)
    assert data["mu"].shape == (num_samples, num_timepoints, 1)
    assert data["noise"].shape == (num_samples, num_timepoints, 1)
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
    assert data["y"].shape == (num_samples, num_timepoints, 1)
    assert data["mu"].shape == (num_samples, num_timepoints, 1)
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
    assert data["y"].shape == (num_samples, num_timepoints, 1)
    assert data["mu"].shape == (num_samples, num_timepoints, 1)
    assert set(data["y"].unique().tolist()).issubset({0.0, 1.0})
    assert (data["mu"] > 0).all() and (data["mu"] < 1).all()
    assert "noise" not in data.keys()
