import torch

from simulacra.functional import (
    activation,
    bernoulli,
    categorical,
    gaussian,
    linear,
    linear_predictor,
    mlp,
    observation_time,
    ordinal,
    poisson,
    random_effects,
    tokens,
)


def test_task_pipeline_smoke() -> None:
    """Free-function pipeline: linear_predictor -> random_effects -> bernoulli -> tokens.

    Tests: end-to-end functional pipeline produces correct shapes for all keys.
    """
    num_samples = 16
    num_timepoints = 6
    num_features = 4

    state = linear_predictor(num_samples, num_timepoints, num_features)
    state = random_effects(state, std=[1.0, 0.05])
    observed = bernoulli(state, prevalence=0.3)
    data = tokens(observed, vocab_size=32)

    assert data["y"].shape == (num_samples, num_timepoints, 1)
    assert data["X"].shape == (num_samples, num_timepoints, num_features)
    assert data["mu"].shape == (num_samples, num_timepoints, 1)
    tok = data.get("tokens")
    assert tok is not None
    assert tok.shape == (num_samples, num_timepoints, 1)


# ---------------------------------------------------------------------------
# linear_predictor
# ---------------------------------------------------------------------------


def test_linear_predictor_shapes() -> None:
    """eta [N,T,1], time [N,T,1], X [N,T,p], beta [N,p,1].

    Tests: shape contract for all returned tensors.
    """
    torch.manual_seed(0)
    N, T, p = 4, 5, 3
    state = linear_predictor(N, T, p)
    assert state["eta"].shape == (N, T, 1)
    assert state["time"].shape == (N, T, 1)
    assert state["X"].shape == (N, T, p)
    assert state["beta"].shape == (N, p, 1)


def test_linear_predictor_eta_equals_X_bmm_beta() -> None:
    """eta == bmm(X, beta) from the returned state.

    Tests: definitional invariant — eta is exactly X @ beta.
    """
    torch.manual_seed(1)
    state = linear_predictor(4, 5, 3)
    expected = torch.bmm(state["X"], state["beta"])
    assert torch.equal(state["eta"], expected)


def test_linear_predictor_user_supplied_X_beta() -> None:
    """User-supplied X, beta stored unchanged; eta = X @ beta.

    Tests: closed-form oracle with known inputs and expected output.
    """
    X = torch.ones(2, 3, 4)
    beta = torch.full((2, 4, 1), 0.5)
    state = linear_predictor(2, 3, 4, X=X, beta=beta)
    assert torch.equal(state["X"], X)
    assert torch.equal(state["beta"], beta)
    expected_eta = torch.full((2, 3, 1), 2.0)
    assert torch.equal(state["eta"], expected_eta)


def test_linear_predictor_default_time_grid() -> None:
    """Default time is arange(T) broadcast to [N,T,1].

    Tests: closed-form oracle for default time grid.
    """
    N, T = 3, 4
    state = linear_predictor(N, T, 2)
    expected = torch.arange(T, dtype=torch.float32).view(1, T, 1).expand(N, T, 1)
    assert torch.equal(state["time"], expected)


def test_linear_predictor_user_supplied_time() -> None:
    """User-supplied time stored unchanged.

    Tests: closed-form oracle — user time passes through.
    """
    time = torch.tensor([[[1.0], [3.0], [7.0]]])
    state = linear_predictor(1, 3, 2, time=time)
    assert torch.equal(state["time"], time)


def test_linear_predictor_single_timepoint() -> None:
    """T=1 produces valid shapes.

    Tests: degenerate case with a single observation per subject.
    """
    torch.manual_seed(2)
    state = linear_predictor(4, 1, 3)
    assert state["eta"].shape == (4, 1, 1)
    assert state["X"].shape == (4, 1, 3)


def test_linear_predictor_single_feature() -> None:
    """p=1 produces eta [N,T,1].

    Tests: degenerate case with a single predictor.
    """
    torch.manual_seed(3)
    state = linear_predictor(4, 5, 1)
    assert state["eta"].shape == (4, 5, 1)
    assert state["beta"].shape == (4, 1, 1)


# ---------------------------------------------------------------------------
# gaussian
# ---------------------------------------------------------------------------


def test_gaussian_y_equals_mu_plus_noise() -> None:
    """y == mu + noise exactly.

    Tests: definitional invariant — Gaussian response is mean plus noise.
    """
    torch.manual_seed(10)
    state = linear_predictor(4, 5, 3)
    result = gaussian(state, std=1.0, covariance=torch.eye(5))
    assert torch.equal(result["y"], result["mu"] + result["noise"])  # pyright: ignore[reportTypedDictNotRequiredAccess]


def test_gaussian_mu_equals_eta() -> None:
    """mu == eta (identity link, cloned).

    Tests: definitional invariant — identity link means mu is eta.
    """
    torch.manual_seed(11)
    state = linear_predictor(4, 5, 3)
    eta_before = state["eta"].clone()
    result = gaussian(state, std=1.0, covariance=torch.eye(5))
    assert torch.equal(result["mu"], eta_before)


def test_gaussian_shapes() -> None:
    """y, mu, noise all [N,T,1].

    Tests: shape contract for Gaussian response.
    """
    torch.manual_seed(12)
    N, T = 4, 5
    state = linear_predictor(N, T, 3)
    result = gaussian(state, std=1.0, covariance=torch.eye(T))
    assert result["y"].shape == (N, T, 1)
    assert result["mu"].shape == (N, T, 1)
    assert result["noise"].shape == (N, T, 1)  # pyright: ignore[reportTypedDictNotRequiredAccess]


def test_gaussian_preserves_predictor_keys() -> None:
    """X, beta, time, eta present in output.

    Tests: shape contract — response step preserves upstream keys.
    """
    torch.manual_seed(13)
    state = linear_predictor(4, 5, 3)
    result = gaussian(state, std=1.0, covariance=torch.eye(5))
    for key in ("X", "beta", "time", "eta"):
        assert key in result


def test_gaussian_noise_mean_converges() -> None:
    """mean(noise) converges to 0 at large N.

    Tests: large-sample convergence of noise mean.
    """
    torch.manual_seed(14)
    N = 4096
    X = torch.zeros(N, 1, 1)
    beta = torch.zeros(N, 1, 1)
    state = linear_predictor(N, 1, 1, X=X, beta=beta)
    result = gaussian(state, std=1.0, covariance=torch.eye(1))
    noise_mean = result["noise"].mean().item()  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert abs(noise_mean) < 0.1


def test_gaussian_noise_variance_converges() -> None:
    """var(noise) converges to std**2 at large N.

    Tests: large-sample convergence of noise variance.
    """
    torch.manual_seed(15)
    N = 4096
    std = 2.0
    X = torch.zeros(N, 1, 1)
    beta = torch.zeros(N, 1, 1)
    state = linear_predictor(N, 1, 1, X=X, beta=beta)
    result = gaussian(state, std=std, covariance=torch.eye(1))
    noise_var = result["noise"].var().item()  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert abs(noise_var - std**2) < 0.5


# ---------------------------------------------------------------------------
# poisson
# ---------------------------------------------------------------------------


def test_poisson_mu_equals_exp_eta() -> None:
    """mu == exp(eta) exactly.

    Tests: definitional invariant — log link inverts to exp.
    """
    torch.manual_seed(20)
    state = linear_predictor(4, 5, 3)
    result = poisson(state)
    assert torch.equal(result["mu"], torch.exp(state["eta"]))


def test_poisson_y_nonneg_integer() -> None:
    """y >= 0 and y == floor(y).

    Tests: distribution support — Poisson outcomes are non-negative integers.
    """
    torch.manual_seed(21)
    state = linear_predictor(8, 5, 3)
    result = poisson(state)
    y = result["y"]
    assert (y >= 0).all()
    assert torch.equal(y, y.floor())


def test_poisson_shapes() -> None:
    """y, mu [N,T,1]; noise absent.

    Tests: shape contract for Poisson response.
    """
    torch.manual_seed(22)
    N, T = 4, 5
    state = linear_predictor(N, T, 3)
    result = poisson(state)
    assert result["y"].shape == (N, T, 1)
    assert result["mu"].shape == (N, T, 1)
    assert "noise" not in result


def test_poisson_zero_eta_mean_near_one() -> None:
    """eta=0 implies mu=1; mean(y) converges to 1.0 at large N.

    Tests: large-sample convergence for Poisson(1).
    """
    torch.manual_seed(23)
    N = 4096
    X = torch.zeros(N, 1, 1)
    beta = torch.zeros(N, 1, 1)
    state = linear_predictor(N, 1, 1, X=X, beta=beta)
    result = poisson(state)
    assert torch.equal(result["mu"], torch.ones(N, 1, 1))
    assert abs(result["y"].mean().item() - 1.0) < 0.1


def test_poisson_equidispersion() -> None:
    """mean(y) approximately equals var(y) at large N (Poisson property).

    Tests: theorem-derived — Poisson equidispersion.
    """
    torch.manual_seed(24)
    N = 4096
    X = torch.zeros(N, 1, 1)
    beta = torch.zeros(N, 1, 1)
    state = linear_predictor(N, 1, 1, X=X, beta=beta)
    result = poisson(state)
    y = result["y"]
    assert abs(y.mean().item() - y.var().item()) < 0.15


def test_poisson_jensen_inequality() -> None:
    """mean(y) >= exp(mean(eta)) at large N (Jensen's inequality).

    Tests: theorem-derived — convexity of exp implies E[exp(X)] >= exp(E[X]).
    """
    torch.manual_seed(25)
    N = 4096
    state = linear_predictor(N, 1, 3)
    result = poisson(state)
    y_mean = result["y"].mean().item()
    exp_eta_mean = torch.exp(state["eta"].mean()).item()
    assert y_mean >= exp_eta_mean - 0.1


# ---------------------------------------------------------------------------
# bernoulli
# ---------------------------------------------------------------------------


def test_bernoulli_mu_equals_sigmoid_shifted_eta() -> None:
    """mu == sigmoid(eta + logit(prevalence)).

    Tests: definitional invariant — logit link with prevalence shift.
    """
    torch.manual_seed(30)
    state = linear_predictor(4, 5, 3)
    prevalence = 0.3
    result = bernoulli(state, prevalence=prevalence)
    shift = torch.logit(torch.tensor(prevalence))
    expected_mu = torch.sigmoid(state["eta"] + shift)
    assert torch.allclose(result["mu"], expected_mu)


def test_bernoulli_y_binary() -> None:
    """y contains only {0.0, 1.0}.

    Tests: distribution support — Bernoulli outcomes are binary.
    """
    torch.manual_seed(31)
    state = linear_predictor(16, 5, 3)
    result = bernoulli(state)
    y = result["y"]
    assert ((y == 0.0) | (y == 1.0)).all()


def test_bernoulli_mu_bounded() -> None:
    """0 < mu < 1 everywhere.

    Tests: distribution support — probabilities are strictly in (0, 1).
    """
    torch.manual_seed(32)
    state = linear_predictor(16, 5, 3)
    result = bernoulli(state)
    mu = result["mu"]
    assert (mu > 0).all()
    assert (mu < 1).all()


def test_bernoulli_shapes() -> None:
    """y, mu [N,T,1]; noise absent.

    Tests: shape contract for Bernoulli response.
    """
    torch.manual_seed(33)
    N, T = 4, 5
    state = linear_predictor(N, T, 3)
    result = bernoulli(state)
    assert result["y"].shape == (N, T, 1)
    assert result["mu"].shape == (N, T, 1)
    assert "noise" not in result


def test_bernoulli_prevalence_half_eta_zero() -> None:
    """eta=0, prevalence=0.5 implies mu=0.5 exactly.

    Tests: degenerate case — logit(0.5)=0, so sigmoid(0+0)=0.5.
    """
    X = torch.zeros(2, 3, 1)
    beta = torch.zeros(2, 1, 1)
    state = linear_predictor(2, 3, 1, X=X, beta=beta)
    result = bernoulli(state, prevalence=0.5)
    expected_mu = torch.full((2, 3, 1), 0.5)
    assert torch.equal(result["mu"], expected_mu)


def test_bernoulli_large_sample_mean() -> None:
    """eta=0, prevalence=0.3: mean(y) converges to 0.3 at large N.

    Tests: large-sample convergence of Bernoulli mean.
    """
    torch.manual_seed(35)
    N = 4096
    X = torch.zeros(N, 1, 1)
    beta = torch.zeros(N, 1, 1)
    state = linear_predictor(N, 1, 1, X=X, beta=beta)
    result = bernoulli(state, prevalence=0.3)
    assert abs(result["y"].mean().item() - 0.3) < 0.05


def test_bernoulli_cauchy_schwarz_mu_bound() -> None:
    """mu in (0,1) implies Var(y) <= 0.25 (Bernoulli max variance at p=0.5).

    Tests: theorem-derived — Bernoulli variance p(1-p) is maximized at 0.25.
    """
    torch.manual_seed(36)
    N = 4096
    state = linear_predictor(N, 1, 3)
    result = bernoulli(state)
    y_var = result["y"].var().item()
    assert y_var <= 0.25 + 0.01


# ---------------------------------------------------------------------------
# categorical
# ---------------------------------------------------------------------------


def test_categorical_mu_sums_to_one() -> None:
    """mu.sum(dim=-1) approximately equals 1.0.

    Tests: theorem-derived — total probability axiom.
    """
    torch.manual_seed(40)
    K = 5
    state = linear_predictor(4, 3, 2)
    state = linear(state, out_features=K)
    result = categorical(state)
    sums = result["mu"].sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums))


def test_categorical_mu_nonneg() -> None:
    """mu >= 0 everywhere.

    Tests: distribution support — probabilities are non-negative.
    """
    torch.manual_seed(41)
    K = 5
    state = linear(linear_predictor(8, 3, 2), out_features=K)
    result = categorical(state)
    assert (result["mu"] >= 0).all()


def test_categorical_y_in_range() -> None:
    """0 <= y < K.

    Tests: distribution support — category indices in valid range.
    """
    torch.manual_seed(42)
    K = 5
    state = linear(linear_predictor(16, 3, 2), out_features=K)
    result = categorical(state)
    y = result["y"]
    assert (y >= 0).all()
    assert (y < K).all()


def test_categorical_y_integer() -> None:
    """y == floor(y).

    Tests: distribution support — category indices are integers.
    """
    torch.manual_seed(43)
    K = 5
    state = linear(linear_predictor(16, 3, 2), out_features=K)
    result = categorical(state)
    y = result["y"].float()
    assert torch.equal(y, y.floor())


def test_categorical_shapes() -> None:
    """y [N,T,1], mu [N,T,K].

    Tests: shape contract for categorical response.
    """
    torch.manual_seed(44)
    N, T, K = 4, 3, 5
    state = linear(linear_predictor(N, T, 2), out_features=K)
    result = categorical(state)
    assert result["y"].shape == (N, T, 1)
    assert result["mu"].shape == (N, T, K)


def test_categorical_k2_binary() -> None:
    """K=2 produces y in {0, 1}.

    Tests: degenerate case — binary categorical equivalent.
    """
    torch.manual_seed(45)
    state = linear(linear_predictor(16, 3, 2), out_features=2)
    result = categorical(state)
    y = result["y"]
    assert ((y == 0) | (y == 1)).all()


def test_categorical_softmax_large_logits() -> None:
    """eta = 100 * randn produces mu that still sums to 1 (no NaN).

    Tests: numerical stability under extreme logits.
    """
    torch.manual_seed(46)
    K = 5
    state = linear(linear_predictor(8, 3, 2), out_features=K)
    state["eta"] = state["eta"] * 100
    result = categorical(state)
    mu = result["mu"]
    assert not torch.isnan(mu).any()
    sums = mu.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums))


# ---------------------------------------------------------------------------
# ordinal
# ---------------------------------------------------------------------------


def test_ordinal_mu_sums_to_one() -> None:
    """mu.sum(dim=-1) approximately equals 1.0.

    Tests: theorem-derived — total probability axiom.
    """
    torch.manual_seed(50)
    state = linear_predictor(4, 3, 2)
    result = ordinal(state, num_classes=4)
    sums = result["mu"].sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums))


def test_ordinal_mu_nonneg() -> None:
    """mu >= 0 everywhere.

    Tests: distribution support — probabilities are non-negative.
    """
    torch.manual_seed(51)
    state = linear_predictor(8, 3, 2)
    result = ordinal(state, num_classes=4)
    assert (result["mu"] >= 0).all()


def test_ordinal_cumulative_nondecreasing() -> None:
    """cumsum(mu, dim=-1) is non-decreasing.

    Tests: theorem-derived — CDF monotonicity.
    """
    torch.manual_seed(52)
    state = linear_predictor(8, 3, 2)
    result = ordinal(state, num_classes=5)
    cdf = result["mu"].cumsum(dim=-1)
    diffs = cdf[..., 1:] - cdf[..., :-1]
    assert (diffs >= -1e-6).all()


def test_ordinal_y_in_range() -> None:
    """0 <= y < K.

    Tests: distribution support — ordinal category indices in valid range.
    """
    torch.manual_seed(53)
    K = 4
    state = linear_predictor(16, 3, 2)
    result = ordinal(state, num_classes=K)
    y = result["y"]
    assert (y >= 0).all()
    assert (y < K).all()


def test_ordinal_shapes() -> None:
    """y [N,T,1], mu [N,T,K].

    Tests: shape contract for ordinal response.
    """
    torch.manual_seed(54)
    N, T, K = 4, 3, 5
    state = linear_predictor(N, T, 2)
    result = ordinal(state, num_classes=K)
    assert result["y"].shape == (N, T, 1)
    assert result["mu"].shape == (N, T, K)


def test_ordinal_eta_zero_closed_form() -> None:
    """eta=0, K=3, start=-1, end=1 produces mu matching REPL values.

    Tests: closed-form oracle — [0.2689, 0.4621, 0.2689].
    """
    X = torch.zeros(1, 1, 1)
    beta = torch.zeros(1, 1, 1)
    state = linear_predictor(1, 1, 1, X=X, beta=beta)
    result = ordinal(state, num_classes=3, start=-1.0, end=1.0)
    expected = torch.tensor([[[0.2689, 0.4621, 0.2689]]])
    assert torch.allclose(result["mu"], expected, atol=1e-4)


def test_ordinal_large_positive_eta_concentrates_high() -> None:
    """Large positive eta pushes mass onto the highest class.

    Tests: limiting case — extreme positive shift.
    """
    X = torch.full((1, 1, 1), 100.0)
    beta = torch.ones(1, 1, 1)
    state = linear_predictor(1, 1, 1, X=X, beta=beta)
    result = ordinal(state, num_classes=4)
    mu = result["mu"]
    # highest class should have nearly all the mass
    assert mu[0, 0, -1].item() > 0.99


def test_ordinal_large_negative_eta_concentrates_low() -> None:
    """Large negative eta pushes mass onto the lowest class.

    Tests: limiting case — extreme negative shift.
    """
    X = torch.full((1, 1, 1), -100.0)
    beta = torch.ones(1, 1, 1)
    state = linear_predictor(1, 1, 1, X=X, beta=beta)
    result = ordinal(state, num_classes=4)
    mu = result["mu"]
    # lowest class should have nearly all the mass
    assert mu[0, 0, 0].item() > 0.99


def test_ordinal_k2_binary() -> None:
    """K=2 produces y in {0, 1} and mu sums to 1.

    Tests: degenerate case — binary ordinal.
    """
    torch.manual_seed(58)
    state = linear_predictor(16, 3, 2)
    result = ordinal(state, num_classes=2)
    y = result["y"]
    assert ((y == 0) | (y == 1)).all()
    sums = result["mu"].sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums))


# ---------------------------------------------------------------------------
# random_effects
# ---------------------------------------------------------------------------


def test_random_effects_eta_update() -> None:
    """new_eta == old_eta + bmm(Z, gamma).

    Tests: definitional invariant — random effects add Z @ gamma to eta.
    """
    torch.manual_seed(60)
    state = linear_predictor(4, 5, 3)
    old_eta = state["eta"].clone()
    result = random_effects(state, std=[1.0, 0.5])
    expected = old_eta + torch.bmm(result["Z"], result["gamma"])  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert torch.allclose(result["eta"], expected)


def test_random_effects_shapes() -> None:
    """gamma [N,q,1], Z [N,T,q], eta [N,T,1].

    Tests: shape contract for random effects.
    """
    torch.manual_seed(61)
    N, T, q = 4, 5, 2
    state = linear_predictor(N, T, 3)
    result = random_effects(state, std=[1.0, 0.5])
    assert result["gamma"].shape == (N, q, 1)  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert result["Z"].shape == (N, T, q)  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert result["eta"].shape == (N, T, 1)


def test_random_effects_vandermonde_ones_column() -> None:
    """Z[:,:,0] == 1.0 (t^0 = 1 in the Vandermonde basis).

    Tests: closed-form oracle — first column of polynomial basis.
    """
    torch.manual_seed(62)
    state = linear_predictor(4, 5, 3)
    result = random_effects(state, std=[1.0, 0.5])
    ones_col = result["Z"][:, :, 0]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert torch.allclose(ones_col, torch.ones_like(ones_col))


def test_random_effects_vandermonde_centered() -> None:
    """Z[:,:,1] has mean 0 per subject (centered time).

    Tests: algebraic invariant — time centering zeroes the mean.
    """
    torch.manual_seed(63)
    state = linear_predictor(4, 5, 3)
    result = random_effects(state, std=[1.0, 0.5])
    linear_col = result["Z"][:, :, 1]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    means = linear_col.mean(dim=1)  # [N]
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-6)


def test_random_effects_user_supplied_Z_gamma() -> None:
    """User-supplied Z, gamma stored unchanged; eta updated correctly.

    Tests: closed-form oracle with known random-effects design.
    """
    N, T, q = 2, 3, 1
    state = linear_predictor(N, T, 2)
    old_eta = state["eta"].clone()
    Z = torch.ones(N, T, q)
    gamma = torch.full((N, q, 1), 0.5)
    result = random_effects(state, std=[1.0], Z=Z, gamma=gamma)
    assert torch.equal(result["Z"], Z)  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert torch.equal(result["gamma"], gamma)  # pyright: ignore[reportTypedDictNotRequiredAccess]
    expected_eta = old_eta + torch.full((N, T, 1), 0.5)
    assert torch.equal(result["eta"], expected_eta)


def test_random_effects_single_effect() -> None:
    """q=1 produces gamma [N,1,1] and Z [N,T,1].

    Tests: degenerate case — single random intercept.
    """
    torch.manual_seed(65)
    N, T = 4, 5
    state = linear_predictor(N, T, 3)
    result = random_effects(state, std=[1.0])
    assert result["gamma"].shape == (N, 1, 1)  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert result["Z"].shape == (N, T, 1)  # pyright: ignore[reportTypedDictNotRequiredAccess]


def test_random_effects_preserves_X_beta() -> None:
    """X and beta unchanged after adding random effects.

    Tests: algebraic invariant — random effects do not mutate fixed-effects.
    """
    torch.manual_seed(66)
    state = linear_predictor(4, 5, 3)
    X_before = state["X"].clone()
    beta_before = state["beta"].clone()
    result = random_effects(state, std=[1.0, 0.5])
    assert torch.equal(result["X"], X_before)
    assert torch.equal(result["beta"], beta_before)


def test_random_effects_tower_property() -> None:
    """E[E[Y|group]] approximately equals E[Y] (tower property / law of iterated expectations).

    Tests: theorem-derived — the tower property holds for intercept-only
    random effects passed through a Gaussian response.
    N=4096, T=1 so each subject is one group.
    """
    torch.manual_seed(67)
    N = 4096
    X = torch.zeros(N, 1, 1)
    beta = torch.zeros(N, 1, 1)
    state = linear_predictor(N, 1, 1, X=X, beta=beta)
    state = random_effects(state, std=[1.0])
    result = gaussian(state, std=0.5, covariance=torch.eye(1))
    grand_mean = result["y"].mean().item()
    # with zero fixed effects, grand mean should be near 0
    assert abs(grand_mean) < 0.1


def test_random_effects_law_of_total_variance() -> None:
    """Var(y) approximately equals sigma_re^2 + sigma_noise^2 (law of total variance).

    Tests: theorem-derived — variance decomposes into between-group
    (random intercept) and within-group (residual noise) components.
    Theoretical: 1.0^2 + 0.5^2 = 1.25.
    """
    torch.manual_seed(68)
    N = 10000
    X = torch.zeros(N, 1, 1)
    beta = torch.zeros(N, 1, 1)
    state = linear_predictor(N, 1, 1, X=X, beta=beta)
    state = random_effects(state, std=[1.0])
    result = gaussian(state, std=0.5, covariance=torch.eye(1))
    total_var = result["y"].var().item()
    assert abs(total_var - 1.25) < 0.15


# ---------------------------------------------------------------------------
# activation
# ---------------------------------------------------------------------------


def test_activation_relu_zeroes_negatives() -> None:
    """After relu, eta >= 0.

    Tests: algebraic invariant — relu clamps negatives to zero.
    """
    torch.manual_seed(70)
    state = linear_predictor(4, 5, 3)
    result = activation(state, torch.relu)
    assert (result["eta"] >= 0).all()


def test_activation_identity_fn() -> None:
    """fn = identity leaves eta unchanged.

    Tests: degenerate case — no-op activation.
    """
    torch.manual_seed(71)
    state = linear_predictor(4, 5, 3)
    eta_before = state["eta"].clone()
    result = activation(state, lambda x: x)
    assert torch.equal(result["eta"], eta_before)


def test_activation_preserves_other_keys() -> None:
    """X, beta, time unchanged after activation.

    Tests: algebraic invariant — activation only modifies eta.
    """
    torch.manual_seed(72)
    state = linear_predictor(4, 5, 3)
    X_before = state["X"].clone()
    beta_before = state["beta"].clone()
    time_before = state["time"].clone()
    result = activation(state, torch.relu)
    assert torch.equal(result["X"], X_before)
    assert torch.equal(result["beta"], beta_before)
    assert torch.equal(result["time"], time_before)


def test_activation_tanh_bounded() -> None:
    """After tanh, -1 <= eta <= 1.

    Tests: range constraint — tanh squashes to [-1, 1].
    """
    torch.manual_seed(73)
    state = linear_predictor(8, 5, 3)
    result = activation(state, torch.tanh)
    assert (result["eta"] >= -1.0).all()
    assert (result["eta"] <= 1.0).all()


# ---------------------------------------------------------------------------
# linear
# ---------------------------------------------------------------------------


def test_linear_output_dimension() -> None:
    """eta last dim == out_features after linear projection.

    Tests: shape contract — linear changes final dimension.
    """
    torch.manual_seed(80)
    state = linear_predictor(4, 5, 3)
    result = linear(state, out_features=7)
    assert result["eta"].shape[2] == 7


def test_linear_user_supplied_weight() -> None:
    """eta = old_eta @ W exactly for known W.

    Tests: closed-form oracle with user-supplied weight matrix.
    """
    torch.manual_seed(81)
    state = linear_predictor(4, 5, 3)
    old_eta = state["eta"].clone()
    W = torch.eye(1, 2)  # [1, 2] — first column is identity, second is zero
    result = linear(state, out_features=2, weight=W)
    expected = old_eta @ W
    assert torch.equal(result["eta"], expected)


def test_linear_preserves_other_keys() -> None:
    """X, beta, time unchanged after linear projection.

    Tests: algebraic invariant — linear only modifies eta.
    """
    torch.manual_seed(82)
    state = linear_predictor(4, 5, 3)
    X_before = state["X"].clone()
    beta_before = state["beta"].clone()
    time_before = state["time"].clone()
    result = linear(state, out_features=7)
    assert torch.equal(result["X"], X_before)
    assert torch.equal(result["beta"], beta_before)
    assert torch.equal(result["time"], time_before)


def test_linear_does_not_mutate_input() -> None:
    """Original state eta unchanged after linear projection.

    Tests: algebraic invariant — linear returns a new state, not a mutation.
    """
    torch.manual_seed(83)
    state = linear_predictor(4, 5, 3)
    eta_before = state["eta"].clone()
    linear(state, out_features=7)
    assert torch.equal(state["eta"], eta_before)


def test_linear_out_one() -> None:
    """out_features=1 produces eta [N,T,1].

    Tests: degenerate case — projecting to scalar.
    """
    torch.manual_seed(84)
    N, T = 4, 5
    state = linear(linear_predictor(N, T, 3), out_features=5)
    result = linear(state, out_features=1)
    assert result["eta"].shape == (N, T, 1)


# ---------------------------------------------------------------------------
# mlp
# ---------------------------------------------------------------------------


def test_mlp_default_preserves_dimension() -> None:
    """Default out_features restores input eta last dim.

    Tests: shape contract — MLP preserves dimension by default.
    """
    torch.manual_seed(90)
    N, T = 4, 5
    state = linear_predictor(N, T, 3)
    result = mlp(state, hidden_features=8)
    assert result["eta"].shape[2] == state["eta"].shape[2]


def test_mlp_custom_out_features() -> None:
    """Explicit out_features changes eta last dim.

    Tests: shape contract — MLP respects custom output dimension.
    """
    torch.manual_seed(91)
    state = linear_predictor(4, 5, 3)
    result = mlp(state, hidden_features=8, out_features=6)
    assert result["eta"].shape[2] == 6


def test_mlp_preserves_other_keys() -> None:
    """X, beta, time unchanged after MLP.

    Tests: algebraic invariant — MLP only modifies eta.
    """
    torch.manual_seed(92)
    state = linear_predictor(4, 5, 3)
    X_before = state["X"].clone()
    beta_before = state["beta"].clone()
    time_before = state["time"].clone()
    result = mlp(state, hidden_features=8)
    assert torch.equal(result["X"], X_before)
    assert torch.equal(result["beta"], beta_before)
    assert torch.equal(result["time"], time_before)


# ---------------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------------


def test_tokens_shape() -> None:
    """tokens [N,T,1].

    Tests: shape contract for tokenized output.
    """
    torch.manual_seed(100)
    N, T = 4, 5
    state = linear_predictor(N, T, 3)
    observed = gaussian(state, std=1.0, covariance=torch.eye(T))
    result = tokens(observed, vocab_size=32)
    assert result["tokens"].shape == (N, T, 1)


def test_tokens_in_range() -> None:
    """0 <= tokens < vocab_size.

    Tests: distribution support — token IDs in valid range.
    """
    torch.manual_seed(101)
    vocab_size = 32
    state = linear_predictor(8, 5, 3)
    observed = gaussian(state, std=1.0, covariance=torch.eye(5))
    result = tokens(observed, vocab_size=vocab_size)
    tok = result["tokens"]
    assert (tok >= 0).all()
    assert (tok < vocab_size).all()


def test_tokens_integer() -> None:
    """Token values are integers.

    Tests: distribution support — discrete token IDs.
    """
    torch.manual_seed(102)
    state = linear_predictor(8, 5, 3)
    observed = gaussian(state, std=1.0, covariance=torch.eye(5))
    result = tokens(observed, vocab_size=32)
    tok = result["tokens"].float()
    assert torch.equal(tok, tok.floor())


def test_tokens_preserves_observed_keys() -> None:
    """y, mu, X, beta, eta, time present in tokenized output.

    Tests: shape contract — tokenization preserves all upstream keys.
    """
    torch.manual_seed(103)
    state = linear_predictor(4, 5, 3)
    observed = gaussian(state, std=1.0, covariance=torch.eye(5))
    result = tokens(observed, vocab_size=32)
    for key in ("y", "mu", "X", "beta", "eta", "time"):
        assert key in result


# ---------------------------------------------------------------------------
# observation_time
# ---------------------------------------------------------------------------


def test_observation_time_strictly_increasing() -> None:
    """time[:,t+1,:] > time[:,t,:] for all t.

    Tests: theorem-derived — cumulative sum of positive intervals is monotone.
    """
    torch.manual_seed(110)
    state = linear_predictor(8, 10, 3)
    result = observation_time(state, shape=2.0, rate=1.0)
    time = result["time"]
    diffs = time[:, 1:, :] - time[:, :-1, :]
    assert (diffs > 0).all()


def test_observation_time_positive() -> None:
    """All time values > 0.

    Tests: distribution support — Gamma intervals are strictly positive.
    """
    torch.manual_seed(111)
    state = linear_predictor(8, 10, 3)
    result = observation_time(state, shape=2.0, rate=1.0)
    assert (result["time"] > 0).all()


def test_observation_time_shape() -> None:
    """time [N,T,1].

    Tests: shape contract for observation time.
    """
    torch.manual_seed(112)
    N, T = 4, 5
    state = linear_predictor(N, T, 3)
    result = observation_time(state, shape=2.0, rate=1.0)
    assert result["time"].shape == (N, T, 1)


def test_observation_time_mean_interval_converges() -> None:
    """Mean interval converges to shape/rate at large N.

    Tests: large-sample convergence — Gamma(a, b) has mean a/b.
    """
    torch.manual_seed(113)
    N = 4096
    shape_param, rate_param = 3.0, 2.0
    state = linear_predictor(N, 5, 1)
    result = observation_time(state, shape=shape_param, rate=rate_param)
    time = result["time"]
    first_intervals = time[:, 0, 0]  # first cumsum step = first interval
    mean_interval = first_intervals.mean().item()
    expected = shape_param / rate_param
    assert abs(mean_interval - expected) < 0.1


def test_observation_time_preserves_eta() -> None:
    """eta unchanged after replacing observation times.

    Tests: algebraic invariant — observation_time only modifies time.
    """
    torch.manual_seed(114)
    state = linear_predictor(4, 5, 3)
    eta_before = state["eta"].clone()
    result = observation_time(state, shape=2.0, rate=1.0)
    assert torch.equal(result["eta"], eta_before)
