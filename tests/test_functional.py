import torch

from simulacra.functional import (
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
    observation_time,
    offset,
    ordinal,
    poisson,
    random_effects,
    tokens,
    zero_inflated_negative_binomial,
    zero_inflated_poisson,
)
from simulacra.states import has_noise, has_random_effects


def test_task_pipeline_smoke() -> None:
    """Free-function pipeline: predictor -> random_effects -> bernoulli -> tokens.

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
    assert data["tokens"].shape == (num_samples, num_timepoints, 1)


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
    assert has_noise(result)
    assert torch.equal(result["y"], result["mu"] + result["noise"])


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
    assert has_noise(result)
    assert result["noise"].shape == (N, T, 1)


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
    assert has_noise(result)
    assert abs(result["noise"].mean().item()) < 0.1


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
    assert has_noise(result)
    assert abs(result["noise"].var().item() - std**2) < 0.5


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
# zero_inflated_poisson
# ---------------------------------------------------------------------------


def test_zero_inflated_poisson_shapes() -> None:
    """y, mu [N,T,1]; no noise key.

    Tests: shape contract for zero-inflated Poisson response.
    """
    torch.manual_seed(260)
    N, T = 4, 5
    state = linear_predictor(N, T, 3)
    result = zero_inflated_poisson(state, inflation=0.3)
    assert result["y"].shape == (N, T, 1)
    assert result["mu"].shape == (N, T, 1)
    assert "noise" not in result


def test_zero_inflated_poisson_mu_equals_exp_eta() -> None:
    """mu == exp(eta) regardless of inflation.

    Tests: definitional invariant — log link inverts to exp, unaffected by mixture.
    """
    torch.manual_seed(261)
    state = linear_predictor(4, 5, 3)
    result = zero_inflated_poisson(state, inflation=0.5)
    assert torch.allclose(result["mu"], torch.exp(state["eta"]))


def test_zero_inflated_poisson_y_nonneg_integer() -> None:
    """y >= 0 and y == floor(y).

    Tests: distribution support — ZIP counts are non-negative integers.
    """
    torch.manual_seed(262)
    state = linear_predictor(8, 5, 3)
    result = zero_inflated_poisson(state, inflation=0.3)
    y = result["y"]
    assert (y >= 0).all()
    assert torch.equal(y, y.floor())


def test_zero_inflated_poisson_more_zeros_than_poisson() -> None:
    """ZIP zero fraction > Poisson zero fraction.

    Tests: excess-zero property — inflation adds structural zeros beyond Poisson zeros.
    """
    torch.manual_seed(263)
    N = 8192
    X = torch.ones(N, 1, 1)
    beta = torch.zeros(N, 1, 1)
    state = linear_predictor(N, 1, 1, X=X, beta=beta)
    inflated_y = zero_inflated_poisson(state, inflation=0.3)["y"]
    torch.manual_seed(263)
    poisson_y = poisson(state)["y"]
    inflated_zeros = (inflated_y == 0).float().mean().item()
    poisson_zeros = (poisson_y == 0).float().mean().item()
    assert inflated_zeros > poisson_zeros


def test_zero_inflated_poisson_inflation_zero_recovers_poisson() -> None:
    """inflation=0 → zero fraction matches Poisson P(0) = e^{-1}.

    Tests: degenerate case — zero inflation removes the mixture component.
    """
    torch.manual_seed(264)
    N = 8192
    X = torch.ones(N, 1, 1)
    beta = torch.zeros(N, 1, 1)
    state = linear_predictor(N, 1, 1, X=X, beta=beta)
    result = zero_inflated_poisson(state, inflation=0.0)
    zero_frac = (result["y"] == 0).float().mean().item()
    import math

    assert abs(zero_frac - math.exp(-1)) < 0.03


def test_zero_inflated_poisson_high_inflation_nearly_all_zeros() -> None:
    """inflation=0.99 → >98% zeros.

    Tests: extreme case — nearly all observations are structural zeros.
    """
    torch.manual_seed(265)
    N = 8192
    X = torch.ones(N, 1, 1)
    beta = torch.zeros(N, 1, 1)
    state = linear_predictor(N, 1, 1, X=X, beta=beta)
    result = zero_inflated_poisson(state, inflation=0.99)
    zero_frac = (result["y"] == 0).float().mean().item()
    assert zero_frac > 0.98


def test_zero_inflated_poisson_mean_converges() -> None:
    """sample mean → (1-pi)*lambda.

    Tests: large-sample convergence.
    E[Y] = (1 - 0.3) * exp(0.5) = 0.7 * 1.6487 ≈ 1.154.
    """
    torch.manual_seed(266)
    N = 8192
    X = torch.ones(N, 1, 1)
    beta = torch.full((N, 1, 1), 0.5)
    state = linear_predictor(N, 1, 1, X=X, beta=beta)
    result = zero_inflated_poisson(state, inflation=0.3)
    assert abs(result["y"].mean().item() - 1.154) < 0.1


def test_zero_inflated_poisson_overdispersed() -> None:
    """Var(Y) > E[Y] for inflation > 0.

    Tests: overdispersion — zero inflation induces variance exceeding the mean.
    """
    torch.manual_seed(267)
    N = 8192
    X = torch.ones(N, 1, 1)
    beta = torch.full((N, 1, 1), 0.5)
    state = linear_predictor(N, 1, 1, X=X, beta=beta)
    result = zero_inflated_poisson(state, inflation=0.3)
    y = result["y"]
    assert y.var().item() > y.mean().item()


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


def test_categorical_mu_equals_softmax() -> None:
    """mu == softmax(eta, dim=-1) exactly.

    Tests: definitional invariant — softmax link for categorical.
    """
    torch.manual_seed(47)
    K = 5
    state = linear(linear_predictor(8, 3, 2), out_features=K)
    result = categorical(state)
    expected = torch.softmax(state["eta"], dim=-1)
    assert torch.allclose(result["mu"], expected)


def test_categorical_softmax_shift_invariance() -> None:
    """softmax(eta + c) == softmax(eta) for any constant c.

    Tests: theorem-derived — softmax is translation-invariant.
    """
    torch.manual_seed(48)
    K = 5
    state = linear(linear_predictor(8, 3, 2), out_features=K)
    mu_original = categorical(state)["mu"]
    state["eta"] = state["eta"] + 42.0
    mu_shifted = categorical(state)["mu"]
    assert torch.allclose(mu_original, mu_shifted)


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
    assert has_random_effects(result)
    expected = old_eta + torch.bmm(result["Z"], result["gamma"])
    assert torch.allclose(result["eta"], expected)


def test_random_effects_shapes() -> None:
    """gamma [N,q,1], Z [N,T,q], eta [N,T,1].

    Tests: shape contract for random effects.
    """
    torch.manual_seed(61)
    N, T, q = 4, 5, 2
    state = linear_predictor(N, T, 3)
    result = random_effects(state, std=[1.0, 0.5])
    assert has_random_effects(result)
    assert result["gamma"].shape == (N, q, 1)
    assert result["Z"].shape == (N, T, q)
    assert result["eta"].shape == (N, T, 1)


def test_random_effects_vandermonde_ones_column() -> None:
    """Z[:,:,0] == 1.0 (t^0 = 1 in the Vandermonde basis).

    Tests: closed-form oracle — first column of polynomial basis.
    """
    torch.manual_seed(62)
    state = linear_predictor(4, 5, 3)
    result = random_effects(state, std=[1.0, 0.5])
    assert has_random_effects(result)
    ones_col = result["Z"][:, :, 0]
    assert torch.allclose(ones_col, torch.ones_like(ones_col))


def test_random_effects_vandermonde_centered() -> None:
    """Z[:,:,1] has mean 0 per subject (centered time).

    Tests: algebraic invariant — time centering zeroes the mean.
    """
    torch.manual_seed(63)
    state = linear_predictor(4, 5, 3)
    result = random_effects(state, std=[1.0, 0.5])
    assert has_random_effects(result)
    linear_col = result["Z"][:, :, 1]
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
    assert has_random_effects(result)
    assert torch.equal(result["Z"], Z)
    assert torch.equal(result["gamma"], gamma)
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
    assert has_random_effects(result)
    assert result["gamma"].shape == (N, 1, 1)
    assert result["Z"].shape == (N, T, 1)


def test_random_effects_chained_shapes() -> None:
    """Two chained calls produce gamma [N, q1+q2, 1] and Z [N, T, q1+q2].

    Tests: shape contract — chaining concatenates along q dimension.
    """
    torch.manual_seed(70)
    N, T = 4, 5
    q1, q2 = 2, 3
    state = linear_predictor(N, T, 3)
    state = random_effects(state, std=[1.0, 0.5])
    state = random_effects(state, std=[0.3, 0.2, 0.1])
    assert has_random_effects(state)
    assert state["gamma"].shape == (N, q1 + q2, 1)
    assert state["Z"].shape == (N, T, q1 + q2)
    assert state["eta"].shape == (N, T, 1)


def test_random_effects_chained_eta_equals_sum() -> None:
    """eta_final == eta_base + Z1@gamma1 + Z2@gamma2.

    Tests: definitional invariant — each call adds its own contribution.
    """
    torch.manual_seed(71)
    N, T = 4, 5
    state = linear_predictor(N, T, 3)
    eta_base = state["eta"].clone()
    state1 = random_effects(state, std=[1.0, 0.5])
    state2 = random_effects(state1, std=[0.3])
    assert has_random_effects(state1)
    assert has_random_effects(state2)
    gamma2 = state2["gamma"][:, 2:, :]
    Z2 = state2["Z"][:, :, 2:]
    expected = (
        eta_base + torch.bmm(state1["Z"], state1["gamma"]) + torch.bmm(Z2, gamma2)
    )
    assert torch.allclose(state2["eta"], expected)


def test_random_effects_chained_Z_gamma_product() -> None:
    """[Z1|Z2] @ [gamma1; gamma2] == Z1@gamma1 + Z2@gamma2.

    Tests: algebraic invariant — concatenated product equals sum of products.
    """
    torch.manual_seed(72)
    N, T = 4, 5
    state = linear_predictor(N, T, 3)
    state1 = random_effects(state, std=[1.0])
    state2 = random_effects(state1, std=[0.5, 0.2])
    assert has_random_effects(state1)
    assert has_random_effects(state2)
    Z2 = state2["Z"][:, :, 1:]
    gamma2 = state2["gamma"][:, 1:, :]
    product = torch.bmm(state2["Z"], state2["gamma"])
    sum_of_products = torch.bmm(state1["Z"], state1["gamma"]) + torch.bmm(Z2, gamma2)
    assert torch.allclose(product, sum_of_products)


def test_random_effects_single_call_unchanged() -> None:
    """Single call produces same shapes as before chaining support.

    Tests: backward compatibility — single call behavior is unchanged.
    """
    torch.manual_seed(73)
    N, T, q = 4, 5, 2
    state = linear_predictor(N, T, 3)
    result = random_effects(state, std=[1.0, 0.5])
    assert has_random_effects(result)
    assert result["gamma"].shape == (N, q, 1)
    assert result["Z"].shape == (N, T, q)
    assert result["eta"].shape == (N, T, 1)


def test_random_effects_chained_user_supplied() -> None:
    """Mixed auto + user-supplied Z/gamma concatenates correctly.

    Tests: chaining with user-supplied tensors in second call.
    """
    N, T = 2, 3
    state = linear_predictor(N, T, 2)
    state = random_effects(state, std=[1.0])
    Z_user = torch.ones(N, T, 1)
    gamma_user = torch.full((N, 1, 1), 0.5)
    state = random_effects(state, std=[1.0], Z=Z_user, gamma=gamma_user)
    assert has_random_effects(state)
    assert state["gamma"].shape == (N, 2, 1)
    assert state["Z"].shape == (N, T, 2)
    assert torch.equal(state["gamma"][:, 1:, :], gamma_user)
    assert torch.equal(state["Z"][:, :, 1:], Z_user)


def test_random_effects_tower_property() -> None:
    """E[E[Y|group]] ~ E[Y] (tower property / law of iterated expectations).

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
# offset
# ---------------------------------------------------------------------------


def test_offset_adds_to_eta() -> None:
    """eta_new == eta_old + log_exposure exactly.

    Tests: algebraic identity — offset is pure addition.
    """
    torch.manual_seed(74)
    state = linear_predictor(4, 5, 3)
    eta_before = state["eta"].clone()
    log_exposure = torch.ones(4, 5, 1) * 0.5
    result = offset(state, log_exposure)
    assert torch.allclose(result["eta"], eta_before + log_exposure)


def test_offset_zero_is_identity() -> None:
    """log_exposure = 0 leaves eta unchanged.

    Tests: degenerate case — zero offset is a no-op.
    """
    torch.manual_seed(75)
    state = linear_predictor(4, 5, 3)
    eta_before = state["eta"].clone()
    result = offset(state, torch.zeros(4, 5, 1))
    assert torch.equal(result["eta"], eta_before)


def test_offset_broadcasts_scalar() -> None:
    """A scalar log_exposure broadcasts to all elements.

    Tests: broadcasting — scalar offset shifts every eta element equally.
    """
    torch.manual_seed(76)
    state = linear_predictor(4, 5, 3)
    eta_before = state["eta"].clone()
    log_exposure = torch.tensor(2.0)
    result = offset(state, log_exposure)
    assert torch.allclose(result["eta"], eta_before + 2.0)


def test_offset_preserves_other_keys() -> None:
    """X, beta, time unchanged after offset.

    Tests: isolation — offset only modifies eta.
    """
    torch.manual_seed(77)
    state = linear_predictor(4, 5, 3)
    result = offset(state, torch.ones(4, 5, 1))
    assert torch.equal(result["X"], state["X"])
    assert torch.equal(result["beta"], state["beta"])
    assert torch.equal(result["time"], state["time"])


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


# ---------------------------------------------------------------------------
# binomial
# ---------------------------------------------------------------------------


def test_binomial_shapes() -> None:
    """y, mu [N,T,1].

    Tests: shape contract for binomial response.
    """
    torch.manual_seed(200)
    N, T = 4, 5
    state = linear_predictor(N, T, 3)
    result = binomial(state, num_trials=10)
    assert result["y"].shape == (N, T, 1)
    assert result["mu"].shape == (N, T, 1)


def test_binomial_mu_equals_sigmoid_shifted_eta() -> None:
    """mu == sigmoid(eta + logit(prevalence)).

    Tests: definitional invariant — logit link with prevalence shift.
    """
    torch.manual_seed(201)
    state = linear_predictor(4, 5, 3)
    prevalence = 0.3
    result = binomial(state, num_trials=10, prevalence=prevalence)
    shift = torch.logit(torch.tensor(prevalence))
    expected_mu = torch.sigmoid(state["eta"] + shift)
    assert torch.allclose(result["mu"], expected_mu)


def test_binomial_y_bounded_integer() -> None:
    """0 <= y <= num_trials and y == floor(y).

    Tests: distribution support — binomial counts bounded by trials.
    """
    torch.manual_seed(202)
    num_trials = 10
    state = linear_predictor(16, 5, 3)
    result = binomial(state, num_trials=num_trials)
    y = result["y"]
    assert (y >= 0).all()
    assert (y <= num_trials).all()
    assert torch.equal(y, y.floor())


def test_binomial_n1_recovers_bernoulli() -> None:
    """num_trials=1 produces same mu as bernoulli and binary y.

    Tests: degenerate case — binomial(1) is bernoulli.
    """
    torch.manual_seed(203)
    state = linear_predictor(16, 5, 3)
    result_bin = binomial(state, num_trials=1, prevalence=0.3)
    result_bern = bernoulli(state, prevalence=0.3)
    assert torch.equal(result_bin["mu"], result_bern["mu"])
    assert ((result_bin["y"] == 0) | (result_bin["y"] == 1)).all()


def test_binomial_mean_converges() -> None:
    """mean(y) converges to n * mu at large N.

    Tests: large-sample convergence of binomial mean.
    mu = sigmoid(0.5 + logit(0.3)) = sigmoid(-0.347) = 0.414, n*mu = 4.14.
    """
    torch.manual_seed(204)
    N = 4096
    num_trials = 10
    X = torch.ones(N, 1, 1)
    beta = torch.full((N, 1, 1), 0.5)
    state = linear_predictor(N, 1, 1, X=X, beta=beta)
    result = binomial(state, num_trials=num_trials, prevalence=0.3)
    assert abs(result["y"].mean().item() - 4.14) < 0.5


def test_binomial_variance_converges() -> None:
    """var(y) converges to n*p*(1-p).

    Tests: theorem-derived — binomial variance formula.
    mu = sigmoid(0.5 + logit(0.3)) = 0.414, Var = 10*0.414*0.586 = 2.43.
    """
    torch.manual_seed(205)
    N = 4096
    num_trials = 10
    X = torch.ones(N, 1, 1)
    beta = torch.full((N, 1, 1), 0.5)
    state = linear_predictor(N, 1, 1, X=X, beta=beta)
    result = binomial(state, num_trials=num_trials, prevalence=0.3)
    assert abs(result["y"].var().item() - 2.43) < 0.5


def test_binomial_extreme_eta_saturates() -> None:
    """Large positive eta drives y toward n; large negative toward 0.

    Tests: stability — saturated probabilities produce boundary counts.
    """
    num_trials = 10
    N = 16
    beta = torch.ones(N, 1, 1)
    state_pos = linear_predictor(N, 1, 1, X=torch.full((N, 1, 1), 20.0), beta=beta)
    state_neg = linear_predictor(N, 1, 1, X=torch.full((N, 1, 1), -20.0), beta=beta)
    result_pos = binomial(state_pos, num_trials=num_trials)
    result_neg = binomial(state_neg, num_trials=num_trials)
    assert (result_pos["y"] == num_trials).all()
    assert (result_neg["y"] == 0).all()


# ---------------------------------------------------------------------------
# multinomial
# ---------------------------------------------------------------------------


def test_multinomial_shapes() -> None:
    """y [N,T,K], mu [N,T,K].

    Tests: shape contract for multinomial response.
    """
    torch.manual_seed(210)
    N, T, K = 4, 3, 5
    state = linear(linear_predictor(N, T, 2), out_features=K)
    result = multinomial(state, num_trials=10)
    assert result["y"].shape == (N, T, K)
    assert result["mu"].shape == (N, T, K)


def test_multinomial_mu_equals_softmax() -> None:
    """mu == softmax(eta, dim=-1).

    Tests: definitional invariant — softmax link for multinomial.
    """
    torch.manual_seed(211)
    K = 5
    state = linear(linear_predictor(8, 3, 2), out_features=K)
    result = multinomial(state, num_trials=10)
    expected = torch.softmax(state["eta"], dim=-1)
    assert torch.allclose(result["mu"], expected)


def test_multinomial_y_nonneg_integer_sums_to_n() -> None:
    """y >= 0, integer, and sums to num_trials along K.

    Tests: distribution support — count vector with fixed total.
    """
    torch.manual_seed(212)
    num_trials = 10
    K = 5
    state = linear(linear_predictor(16, 3, 2), out_features=K)
    result = multinomial(state, num_trials=num_trials)
    y = result["y"]
    assert (y >= 0).all()
    assert torch.equal(y, y.floor())
    sums = y.sum(dim=-1)
    assert torch.equal(sums, torch.full_like(sums, num_trials))


def test_multinomial_n1_one_hot() -> None:
    """num_trials=1 produces one-hot count vectors.

    Tests: degenerate case — single draw yields exactly one count.
    """
    torch.manual_seed(213)
    K = 5
    state = linear(linear_predictor(16, 3, 2), out_features=K)
    result = multinomial(state, num_trials=1)
    y = result["y"]
    assert torch.equal(y.sum(dim=-1), torch.ones_like(y.sum(dim=-1)))
    assert ((y == 0) | (y == 1)).all()


def test_multinomial_mean_converges() -> None:
    """Asymmetric eta: first-category mean exceeds others.

    Tests: large-sample convergence with non-uniform softmax.
    softmax([1,0,0]) = [0.576, 0.212, 0.212].
    n=20 -> expected means [11.52, 4.24, 4.24].
    """
    torch.manual_seed(214)
    N = 4096
    num_trials = 20
    K = 3
    state = linear(linear_predictor(N, 1, 2), out_features=K)
    eta = torch.zeros(N, 1, K)
    eta[:, :, 0] = 1.0
    state["eta"] = eta
    result = multinomial(state, num_trials=num_trials)
    means = result["y"].float().mean(dim=0).squeeze()  # [K]
    assert abs(means[0].item() - 11.52) < 0.5
    assert abs(means[1].item() - 4.24) < 0.5


def test_multinomial_mu_sums_to_one() -> None:
    """mu.sum(dim=-1) approximately equals 1.0.

    Tests: theorem-derived — total probability axiom.
    """
    torch.manual_seed(215)
    K = 5
    state = linear(linear_predictor(4, 3, 2), out_features=K)
    result = multinomial(state, num_trials=10)
    sums = result["mu"].sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums))


def test_multinomial_large_logits_finite() -> None:
    """Extreme logits produce finite mu and y.

    Tests: numerical stability under extreme logits.
    """
    torch.manual_seed(216)
    K = 5
    state = linear(linear_predictor(8, 3, 2), out_features=K)
    state["eta"] = state["eta"] * 100
    result = multinomial(state, num_trials=10)
    assert not torch.isnan(result["mu"]).any()
    assert torch.isfinite(result["y"]).all()


# ---------------------------------------------------------------------------
# negative_binomial
# ---------------------------------------------------------------------------


def test_negative_binomial_shapes() -> None:
    """y, mu [N,T,1].

    Tests: shape contract for negative binomial response.
    """
    torch.manual_seed(220)
    N, T = 4, 5
    state = linear_predictor(N, T, 3)
    result = negative_binomial(state, concentration=2.0)
    assert result["y"].shape == (N, T, 1)
    assert result["mu"].shape == (N, T, 1)


def test_negative_binomial_mu_equals_exp_eta() -> None:
    """mu == exp(eta).

    Tests: definitional invariant — log link inverts to exp.
    """
    torch.manual_seed(221)
    state = linear_predictor(4, 5, 3)
    result = negative_binomial(state, concentration=2.0)
    assert torch.allclose(result["mu"], torch.exp(state["eta"]))


def test_negative_binomial_y_nonneg_integer() -> None:
    """y >= 0 and y == floor(y).

    Tests: distribution support — negative binomial counts are non-negative integers.
    """
    torch.manual_seed(222)
    state = linear_predictor(8, 5, 3)
    result = negative_binomial(state, concentration=2.0)
    y = result["y"]
    assert (y >= 0).all()
    assert torch.equal(y, y.floor())


def test_negative_binomial_large_r_approaches_poisson() -> None:
    """Large concentration: var(y) converges to mean(y) (equidispersion).

    Tests: degenerate case — NegBin(r -> inf) has Poisson-like variance.
    Var = mu + mu^2/r; as r -> inf, Var -> mu = mean.
    """
    torch.manual_seed(223)
    N = 4096
    X = torch.ones(N, 1, 1)
    beta = torch.full((N, 1, 1), 0.5)
    state = linear_predictor(N, 1, 1, X=X, beta=beta)
    result = negative_binomial(state, concentration=10000.0)
    y = result["y"]
    assert abs(y.var().item() - y.mean().item()) < 0.3


def test_negative_binomial_mean_converges() -> None:
    """mean(y) converges to mu at large N.

    Tests: large-sample convergence of negative binomial mean.
    mu = exp(0.5) = 1.65.
    """
    torch.manual_seed(224)
    N = 4096
    X = torch.ones(N, 1, 1)
    beta = torch.full((N, 1, 1), 0.5)
    state = linear_predictor(N, 1, 1, X=X, beta=beta)
    result = negative_binomial(state, concentration=5.0)
    assert abs(result["y"].mean().item() - 1.65) < 0.2


def test_negative_binomial_variance_formula() -> None:
    """var(y) converges to mu + mu^2/r.

    Tests: theorem-derived — NegBin variance validates concentration.
    mu = exp(0.5) = 1.65, r = 2, Var = 1.65 + 1.65^2/2 = 3.01.
    """
    torch.manual_seed(225)
    N = 4096
    X = torch.ones(N, 1, 1)
    beta = torch.full((N, 1, 1), 0.5)
    state = linear_predictor(N, 1, 1, X=X, beta=beta)
    result = negative_binomial(state, concentration=2.0)
    assert abs(result["y"].var().item() - 3.01) < 0.5


def test_negative_binomial_extreme_eta_finite() -> None:
    """Large eta (mu ~ 5e8) produces finite results.

    Tests: numerical stability — exp(20) pushes into large-mu regime.
    """
    torch.manual_seed(226)
    state = linear_predictor(4, 5, 3)
    state["eta"] = torch.full_like(state["eta"], 20.0)
    result = negative_binomial(state, concentration=2.0)
    assert torch.isfinite(result["mu"]).all()
    assert torch.isfinite(result["y"]).all()


# ---------------------------------------------------------------------------
# zero_inflated_negative_binomial
# ---------------------------------------------------------------------------


def test_zero_inflated_negative_binomial_shapes() -> None:
    """y, mu [N,T,1]; no noise key.

    Tests: shape contract for zero-inflated negative binomial response.
    """
    torch.manual_seed(270)
    N, T = 4, 5
    state = linear_predictor(N, T, 3)
    result = zero_inflated_negative_binomial(state, inflation=0.3, concentration=2.0)
    assert result["y"].shape == (N, T, 1)
    assert result["mu"].shape == (N, T, 1)
    assert "noise" not in result


def test_zero_inflated_negative_binomial_mu_equals_exp_eta() -> None:
    """mu == exp(eta) regardless of inflation.

    Tests: definitional invariant — log link inverts to exp, unaffected by mixture.
    """
    torch.manual_seed(271)
    state = linear_predictor(4, 5, 3)
    result = zero_inflated_negative_binomial(state, inflation=0.5, concentration=2.0)
    assert torch.allclose(result["mu"], torch.exp(state["eta"]))


def test_zero_inflated_negative_binomial_y_nonneg_integer() -> None:
    """y >= 0 and y == floor(y).

    Tests: distribution support — ZINB counts are non-negative integers.
    """
    torch.manual_seed(272)
    state = linear_predictor(8, 5, 3)
    result = zero_inflated_negative_binomial(state, inflation=0.3, concentration=2.0)
    y = result["y"]
    assert (y >= 0).all()
    assert torch.equal(y, y.floor())


def test_zero_inflated_negative_binomial_more_zeros_than_nb() -> None:
    """ZINB zero fraction > NB zero fraction.

    Tests: excess-zero property — inflation adds structural zeros beyond NB zeros.
    """
    torch.manual_seed(273)
    N = 8192
    X = torch.ones(N, 1, 1)
    beta = torch.zeros(N, 1, 1)
    state = linear_predictor(N, 1, 1, X=X, beta=beta)
    inflated_y = zero_inflated_negative_binomial(
        state, inflation=0.3, concentration=2.0
    )["y"]
    torch.manual_seed(273)
    baseline_y = negative_binomial(state, concentration=2.0)["y"]
    inflated_zeros = (inflated_y == 0).float().mean().item()
    baseline_zeros = (baseline_y == 0).float().mean().item()
    assert inflated_zeros > baseline_zeros


def test_zero_inflated_negative_binomial_inflation_zero_recovers_nb() -> None:
    """inflation=0 → zero fraction matches NB P(0) = (r/(r+mu))^r.

    Tests: degenerate case — zero inflation removes the mixture component.
    mu = 1, r = 2, P(0) = (2/3)^2 ≈ 0.444.
    """
    torch.manual_seed(274)
    N = 8192
    X = torch.ones(N, 1, 1)
    beta = torch.zeros(N, 1, 1)
    state = linear_predictor(N, 1, 1, X=X, beta=beta)
    result = zero_inflated_negative_binomial(state, inflation=0.0, concentration=2.0)
    zero_frac = (result["y"] == 0).float().mean().item()
    expected = (2.0 / 3.0) ** 2
    assert abs(zero_frac - expected) < 0.03


def test_zero_inflated_negative_binomial_high_inflation_nearly_all_zeros() -> None:
    """inflation=0.99 → >98% zeros.

    Tests: extreme case — nearly all observations are structural zeros.
    """
    torch.manual_seed(275)
    N = 8192
    X = torch.ones(N, 1, 1)
    beta = torch.zeros(N, 1, 1)
    state = linear_predictor(N, 1, 1, X=X, beta=beta)
    result = zero_inflated_negative_binomial(state, inflation=0.99, concentration=2.0)
    zero_frac = (result["y"] == 0).float().mean().item()
    assert zero_frac > 0.98


def test_zero_inflated_negative_binomial_mean_converges() -> None:
    """sample mean → (1-pi)*mu.

    Tests: large-sample convergence.
    E[Y] = (1 - 0.3) * exp(0.5) = 0.7 * 1.6487 ≈ 1.154.
    """
    torch.manual_seed(276)
    N = 8192
    X = torch.ones(N, 1, 1)
    beta = torch.full((N, 1, 1), 0.5)
    state = linear_predictor(N, 1, 1, X=X, beta=beta)
    result = zero_inflated_negative_binomial(state, inflation=0.3, concentration=5.0)
    assert abs(result["y"].mean().item() - 1.154) < 0.15


def test_zero_inflated_negative_binomial_more_variance_than_nb() -> None:
    """ZINB variance > NB variance.

    Tests: zero inflation adds variance beyond the NB baseline.
    Uses large mu (eta=2) so the pi*(1-pi)*mu^2 term dominates.
    """
    torch.manual_seed(277)
    N = 8192
    X = torch.ones(N, 1, 1)
    beta = torch.full((N, 1, 1), 2.0)
    state = linear_predictor(N, 1, 1, X=X, beta=beta)
    inflated_y = zero_inflated_negative_binomial(
        state, inflation=0.3, concentration=10.0
    )["y"]
    torch.manual_seed(277)
    baseline_y = negative_binomial(state, concentration=10.0)["y"]
    assert inflated_y.var().item() > baseline_y.var().item()


# ---------------------------------------------------------------------------
# gamma_response
# ---------------------------------------------------------------------------


def test_gamma_response_shapes() -> None:
    """y, mu [N,T,1].

    Tests: shape contract for Gamma response.
    """
    torch.manual_seed(230)
    N, T = 4, 5
    state = linear_predictor(N, T, 3)
    result = gamma_response(state, concentration=2.0)
    assert result["y"].shape == (N, T, 1)
    assert result["mu"].shape == (N, T, 1)


def test_gamma_response_mu_equals_exp_eta() -> None:
    """mu == exp(eta).

    Tests: definitional invariant — log link inverts to exp.
    """
    torch.manual_seed(231)
    state = linear_predictor(4, 5, 3)
    result = gamma_response(state, concentration=2.0)
    assert torch.allclose(result["mu"], torch.exp(state["eta"]))


def test_gamma_response_y_positive() -> None:
    """y > 0 everywhere.

    Tests: distribution support — Gamma values are strictly positive.
    """
    torch.manual_seed(232)
    state = linear_predictor(8, 5, 3)
    result = gamma_response(state, concentration=2.0)
    assert (result["y"] > 0).all()


def test_gamma_response_eta_zero_mu_one() -> None:
    """eta=0 implies mu = exp(0) = 1.

    Tests: degenerate case — zero linear predictor.
    """
    X = torch.zeros(2, 3, 1)
    beta = torch.zeros(2, 1, 1)
    state = linear_predictor(2, 3, 1, X=X, beta=beta)
    result = gamma_response(state, concentration=2.0)
    assert torch.allclose(result["mu"], torch.ones(2, 3, 1))


def test_gamma_response_mean_converges() -> None:
    """mean(y) converges to mu at large N.

    Tests: large-sample convergence of Gamma mean. mu = exp(0.5) = 1.65.
    """
    torch.manual_seed(234)
    N = 4096
    X = torch.ones(N, 1, 1)
    beta = torch.full((N, 1, 1), 0.5)
    state = linear_predictor(N, 1, 1, X=X, beta=beta)
    result = gamma_response(state, concentration=2.0)
    assert abs(result["y"].mean().item() - 1.65) < 0.15


def test_gamma_response_variance_formula() -> None:
    """var(y) converges to mu^2 / concentration.

    Tests: theorem-derived — Gamma variance = mu^2 / alpha.
    mu = exp(0.5) = 1.65, Var = 1.65^2 / 2 = 1.36.
    """
    torch.manual_seed(235)
    N = 4096
    concentration = 2.0
    X = torch.ones(N, 1, 1)
    beta = torch.full((N, 1, 1), 0.5)
    state = linear_predictor(N, 1, 1, X=X, beta=beta)
    result = gamma_response(state, concentration=concentration)
    assert abs(result["y"].var().item() - 1.36) < 0.3


def test_gamma_response_extreme_eta_finite() -> None:
    """Large eta (mu ~ 5e8) produces finite results.

    Tests: numerical stability — exp(20) pushes into large-mu regime.
    """
    torch.manual_seed(236)
    state = linear_predictor(4, 5, 3)
    state["eta"] = torch.full_like(state["eta"], 20.0)
    result = gamma_response(state, concentration=2.0)
    assert torch.isfinite(result["mu"]).all()
    assert torch.isfinite(result["y"]).all()


# ---------------------------------------------------------------------------
# beta_response
# ---------------------------------------------------------------------------


def test_beta_response_shapes() -> None:
    """y, mu [N,T,1].

    Tests: shape contract for Beta response.
    """
    torch.manual_seed(240)
    N, T = 4, 5
    state = linear_predictor(N, T, 3)
    result = beta_response(state, precision=5.0)
    assert result["y"].shape == (N, T, 1)
    assert result["mu"].shape == (N, T, 1)


def test_beta_response_mu_equals_sigmoid_eta() -> None:
    """mu == sigmoid(eta).

    Tests: definitional invariant — logit link.
    """
    torch.manual_seed(241)
    state = linear_predictor(4, 5, 3)
    result = beta_response(state, precision=5.0)
    assert torch.allclose(result["mu"], torch.sigmoid(state["eta"]))


def test_beta_response_y_bounded() -> None:
    """0 < y < 1 and 0 < mu < 1.

    Tests: distribution support — Beta values in open unit interval.
    """
    torch.manual_seed(242)
    state = linear_predictor(8, 5, 3)
    result = beta_response(state, precision=5.0)
    assert (result["y"] > 0).all()
    assert (result["y"] < 1).all()
    assert (result["mu"] > 0).all()
    assert (result["mu"] < 1).all()


def test_beta_response_eta_zero_mu_half() -> None:
    """eta=0 implies mu = sigmoid(0) = 0.5.

    Tests: degenerate case — zero linear predictor.
    """
    X = torch.zeros(2, 3, 1)
    beta = torch.zeros(2, 1, 1)
    state = linear_predictor(2, 3, 1, X=X, beta=beta)
    result = beta_response(state, precision=5.0)
    assert torch.allclose(result["mu"], torch.full((2, 3, 1), 0.5))


def test_beta_response_mean_converges() -> None:
    """mean(y) converges to mu at large N.

    Tests: large-sample convergence of Beta mean.
    mu = sigmoid(0.5) = 0.622.
    """
    torch.manual_seed(244)
    N = 4096
    X = torch.ones(N, 1, 1)
    beta = torch.full((N, 1, 1), 0.5)
    state = linear_predictor(N, 1, 1, X=X, beta=beta)
    result = beta_response(state, precision=10.0)
    assert abs(result["y"].mean().item() - 0.622) < 0.05


def test_beta_response_variance_formula() -> None:
    """var(y) converges to mu*(1-mu)/(phi+1).

    Tests: theorem-derived — Beta variance validates precision.
    mu = sigmoid(0.5) = 0.622, phi = 5, Var = 0.622*0.378/6 = 0.039.
    """
    torch.manual_seed(245)
    N = 4096
    X = torch.ones(N, 1, 1)
    beta = torch.full((N, 1, 1), 0.5)
    state = linear_predictor(N, 1, 1, X=X, beta=beta)
    result = beta_response(state, precision=5.0)
    assert abs(result["y"].var().item() - 0.039) < 0.01


def test_beta_response_extreme_eta_saturates() -> None:
    """Extreme eta drives y toward support boundaries; results stay finite.

    Tests: stability — saturated mu produces boundary-concentrated y.
    """
    N = 16
    beta = torch.ones(N, 1, 1)
    state_pos = linear_predictor(N, 1, 1, X=torch.full((N, 1, 1), 100.0), beta=beta)
    state_neg = linear_predictor(N, 1, 1, X=torch.full((N, 1, 1), -100.0), beta=beta)
    result_pos = beta_response(state_pos, precision=5.0)
    result_neg = beta_response(state_neg, precision=5.0)
    assert torch.isfinite(result_pos["y"]).all()
    assert torch.isfinite(result_neg["y"]).all()
    assert (result_pos["y"] > 0.9).all()
    assert (result_neg["y"] < 0.1).all()


# ---------------------------------------------------------------------------
# log_normal
# ---------------------------------------------------------------------------


def test_log_normal_shapes() -> None:
    """y, mu [N,T,1].

    Tests: shape contract for log-normal response.
    """
    torch.manual_seed(250)
    N, T = 4, 5
    state = linear_predictor(N, T, 3)
    result = log_normal(state, std=0.5)
    assert result["y"].shape == (N, T, 1)
    assert result["mu"].shape == (N, T, 1)


def test_log_normal_mu_equals_exp_eta_plus_half_var() -> None:
    """mu == exp(eta + std^2/2).

    Tests: definitional invariant — log-normal mean formula.
    """
    torch.manual_seed(251)
    std = 0.5
    state = linear_predictor(4, 5, 3)
    result = log_normal(state, std=std)
    expected = torch.exp(state["eta"] + std**2 / 2)
    assert torch.allclose(result["mu"], expected)


def test_log_normal_y_positive() -> None:
    """y > 0 everywhere.

    Tests: distribution support — log-normal values are strictly positive.
    """
    torch.manual_seed(252)
    state = linear_predictor(8, 5, 3)
    result = log_normal(state, std=0.5)
    assert (result["y"] > 0).all()


def test_log_normal_std_zero_deterministic() -> None:
    """std near 0 produces y approximately exp(eta).

    Tests: degenerate case — zero variance implies deterministic output.
    """
    torch.manual_seed(253)
    state = linear_predictor(4, 5, 3)
    result = log_normal(state, std=1e-6)
    expected = torch.exp(state["eta"])
    assert torch.allclose(result["y"], expected, atol=1e-3)


def test_log_normal_mean_converges() -> None:
    """mean(y) converges to mu at large N.

    Tests: large-sample convergence of log-normal mean.
    mu = exp(0.5 + 0.125) = exp(0.625) = 1.87.
    """
    torch.manual_seed(254)
    N = 4096
    X = torch.ones(N, 1, 1)
    beta = torch.full((N, 1, 1), 0.5)
    state = linear_predictor(N, 1, 1, X=X, beta=beta)
    result = log_normal(state, std=0.5)
    assert abs(result["y"].mean().item() - 1.87) < 0.2


def test_log_normal_log_y_normal() -> None:
    """log(y) ~ Normal(eta, std^2) with non-zero eta.

    Tests: theorem-derived — log of log-normal is normal.
    eta = 0.5, so log(y) should have mean 0.5 and var 0.25.
    """
    torch.manual_seed(255)
    N = 4096
    X = torch.ones(N, 1, 1)
    beta = torch.full((N, 1, 1), 0.5)
    state = linear_predictor(N, 1, 1, X=X, beta=beta)
    result = log_normal(state, std=0.5)
    log_y = torch.log(result["y"])
    assert abs(log_y.mean().item() - 0.5) < 0.1
    assert abs(log_y.var().item() - 0.25) < 0.1


def test_log_normal_extreme_eta_finite() -> None:
    """Large eta (mu ~ 3e8) produces finite results.

    Tests: numerical stability — exp(20) pushes into large-mu regime.
    """
    torch.manual_seed(256)
    state = linear_predictor(4, 5, 3)
    state["eta"] = torch.full_like(state["eta"], 20.0)
    result = log_normal(state, std=0.5)
    assert torch.isfinite(result["mu"]).all()
    assert torch.isfinite(result["y"]).all()
