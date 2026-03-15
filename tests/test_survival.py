import pytest
import torch

from simulacra.functional import (
    bernoulli,
    gaussian,
    linear,
    linear_predictor,
    random_effects,
)
from simulacra.states import (
    CompetingRisksState,
    EventProcessState,
    ObservedState,
    RiskIndicatorState,
    has_failure_times,
)
from simulacra.survival import (
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

N = 16
T = 6
P = 4
K = 3


def _gaussian_state() -> ObservedState:
    return gaussian(linear_predictor(N, T, P), std=1.0, covariance=torch.eye(T))


def _competing_risks_state() -> CompetingRisksState:
    return competing_risks(linear(linear_predictor(N, T, P), out_features=K))


# ---------------------------------------------------------------------------
# event_time
# ---------------------------------------------------------------------------


def test_event_time_shapes() -> None:
    """event_time [N, 1, 1].

    Tests: shape contract — subject-level event time collapses T to 1.
    """
    state = _gaussian_state()
    result = event_time(state)
    assert result["event_time"].shape == (N, 1, 1)


def test_event_time_positive() -> None:
    """event_time > 0 everywhere.

    Tests: distribution support — Exponential is strictly positive.
    """
    state = _gaussian_state()
    result = event_time(state)
    assert (result["event_time"] > 0).all()


def test_event_time_mean_converges_eta_zero() -> None:
    """eta=0 implies rate=1; mean event time converges to 1.0.

    Tests: large-sample convergence — Exp(1) has mean 1.
    """
    torch.manual_seed(3)
    n = 4096
    X = torch.zeros(n, 1, 1)
    beta = torch.zeros(n, 1, 1)
    state = gaussian(
        linear_predictor(n, 1, 1, X=X, beta=beta), std=0.1, covariance=torch.eye(1)
    )
    result = event_time(state)
    mean_et = result["event_time"].mean().item()
    assert abs(mean_et - 1.0) < 0.15


def test_event_time_compound_mean() -> None:
    """With random intercept η ~ N(0, σ²), E[Exp(exp(η))] = exp(σ²/2).

    Tests: CAS-derived oracle — normal MGF gives compound Exponential mean.
    """
    torch.manual_seed(9)
    n = 8192
    sigma = 1.0
    X = torch.zeros(n, 1, 1)
    beta = torch.zeros(n, 1, 1)
    state = random_effects(linear_predictor(n, 1, 1, X=X, beta=beta), std=[sigma])
    state = gaussian(state, std=0.01, covariance=torch.eye(1))
    result = event_time(state)
    empirical_mean = result["event_time"].mean().item()
    # SymPy: E[exp(-η)] for η ~ N(0, σ²) = exp(σ²/2) via MGF
    expected_mean = torch.exp(torch.tensor(sigma**2 / 2)).item()  # exp(0.5) ≈ 1.6487
    assert abs(empirical_mean - expected_mean) / expected_mean < 0.15


def test_event_time_higher_eta_shorter_times() -> None:
    """Higher eta produces larger rate, hence shorter event times on average.

    Tests: monotonicity — exp(eta) increases rate, decreasing expected time.
    """
    torch.manual_seed(4)
    n = 2048
    X_low = torch.full((n, 1, 1), -1.0)
    X_high = torch.full((n, 1, 1), 1.0)
    beta = torch.ones(n, 1, 1)
    state_low = gaussian(
        linear_predictor(n, 1, 1, X=X_low, beta=beta), std=0.01, covariance=torch.eye(1)
    )
    state_high = gaussian(
        linear_predictor(n, 1, 1, X=X_high, beta=beta),
        std=0.01,
        covariance=torch.eye(1),
    )
    event_time_low = event_time(state_low)["event_time"].mean().item()
    event_time_high = event_time(state_high)["event_time"].mean().item()
    assert event_time_low > event_time_high


def test_event_time_finite() -> None:
    """event_time contains no NaN or Inf.

    Tests: numerical stability — no degenerate samples.
    """
    state = _gaussian_state()
    result = event_time(state)
    assert result["event_time"].isfinite().all()


# ---------------------------------------------------------------------------
# mixture_cure_censoring
# ---------------------------------------------------------------------------


def test_mixture_cure_cured_subjects_get_inf() -> None:
    """Cured subjects (y==0 everywhere) have event_time == inf.

    Tests: definitional invariant — cure model sets event time to infinity.
    """
    n = 4
    X = torch.zeros(n, T, 1)
    beta = torch.zeros(n, 1, 1)
    state = bernoulli(linear_predictor(n, T, 1, X=X, beta=beta), prevalence=0.01)
    state = event_time(state)
    result = mixture_cure_censoring(state)
    cured = (state["y"] == 0).all(dim=1, keepdim=True)  # [N, 1, 1]
    if cured.any():
        assert (
            result["event_time"][cured.expand_as(result["event_time"])] == torch.inf
        ).all()


def test_mixture_cure_uncured_unchanged() -> None:
    """Non-cured subjects keep their original event_time.

    Tests: definitional invariant — only cured subjects are modified.
    """
    n = 64
    state = bernoulli(linear_predictor(n, T, P), prevalence=0.99)
    state = event_time(state)
    original_et = state["event_time"].clone()
    result = mixture_cure_censoring(state)
    uncured = ~(state["y"] == 0).all(dim=1, keepdim=True)  # [N, 1, 1]
    if uncured.any():
        mask = uncured.expand_as(original_et)
        assert torch.equal(result["event_time"][mask], original_et[mask])


def test_mixture_cure_shape_unchanged() -> None:
    """Output event_time has same shape as input.

    Tests: shape contract — mixture cure does not change dimensionality.
    """
    state = bernoulli(linear_predictor(N, T, P), prevalence=0.5)
    state = event_time(state)
    result = mixture_cure_censoring(state)
    assert result["event_time"].shape == state["event_time"].shape


def test_mixture_cure_all_cured() -> None:
    """When y==0 everywhere, all event times are inf.

    Tests: degenerate case — entire cohort is cured.
    """
    n = 8
    X = torch.zeros(n, T, 1)
    beta = torch.full((n, 1, 1), -100.0)
    state = bernoulli(linear_predictor(n, T, 1, X=X, beta=beta), prevalence=0.001)
    state = event_time(state)
    # Force y to 0 to guarantee all cured
    state["y"] = torch.zeros(n, T, 1)
    result = mixture_cure_censoring(state)
    assert (result["event_time"] == torch.inf).all()


def test_mixture_cure_all_uncured() -> None:
    """When every subject has at least one y==1, no event times are inf.

    Tests: degenerate case — no cured subjects.
    """
    n = 8
    state = bernoulli(linear_predictor(n, T, P), prevalence=0.5)
    state = event_time(state)
    # Force at least one 1 per subject
    state["y"][:, 0, :] = 1.0
    original_et = state["event_time"].clone()
    result = mixture_cure_censoring(state)
    assert torch.equal(result["event_time"], original_et)


# ---------------------------------------------------------------------------
# censor_time
# ---------------------------------------------------------------------------


def test_censor_time_shapes() -> None:
    """censor_time [N, 1, 1].

    Tests: shape contract.
    """
    state = _gaussian_state()
    result = censor_time(event_time(state))
    assert result["censor_time"].shape == (N, 1, 1)


def test_censor_time_within_time_range() -> None:
    """censor_time in [min(time), max(time)] per subject.

    Tests: distribution support — Uniform bounds.
    """
    state = _gaussian_state()
    result = censor_time(event_time(state))
    time = result["time"]
    min_time = torch.amin(time, dim=1, keepdim=True)
    max_time = torch.amax(time, dim=1, keepdim=True)
    assert (result["censor_time"] >= min_time).all()
    assert (result["censor_time"] <= max_time).all()


def test_censor_time_mean_converges() -> None:
    """Mean censor_time converges to midpoint of [min(time), max(time)].

    Tests: large-sample convergence — Uniform mean = (a+b)/2.
    """
    torch.manual_seed(23)
    n = 4096
    state = gaussian(linear_predictor(n, 5, 2), std=1.0, covariance=torch.eye(5))
    result = censor_time(event_time(state))
    time = result["time"]
    min_time = torch.amin(time, dim=1, keepdim=True)
    max_time = torch.amax(time, dim=1, keepdim=True)
    midpoint = ((min_time + max_time) / 2).mean().item()
    empirical = result["censor_time"].mean().item()
    assert abs(empirical - midpoint) < 0.15


# ---------------------------------------------------------------------------
# survival_indicators
# ---------------------------------------------------------------------------


def test_survival_indicators_shapes() -> None:
    """indicator [N,1,1], observed_time [N,1,1], time_to_event [N,T,1].

    Tests: shape contract for all survival indicator outputs.
    """
    state = _gaussian_state()
    result = survival_indicators(censor_time(event_time(state)))
    assert result["indicator"].shape == (N, 1, 1)
    assert result["observed_time"].shape == (N, 1, 1)
    assert result["time_to_event"].shape == (N, T, 1)


def test_survival_indicator_binary() -> None:
    """indicator in {0.0, 1.0}.

    Tests: distribution support — event indicator is binary.
    """
    state = _gaussian_state()
    result = survival_indicators(censor_time(event_time(state)))
    ind = result["indicator"]
    assert ((ind == 0.0) | (ind == 1.0)).all()


def test_survival_observed_time_is_minimum() -> None:
    """observed_time == min(event_time, censor_time).

    Tests: definitional invariant — observed time is the minimum.
    """
    state = _gaussian_state()
    result = survival_indicators(censor_time(event_time(state)))
    expected = torch.minimum(result["event_time"], result["censor_time"])
    assert torch.equal(result["observed_time"], expected)


def test_survival_indicator_matches_comparison() -> None:
    """indicator == (event_time < censor_time).float().

    Tests: definitional invariant — indicator encodes which came first.
    """
    state = _gaussian_state()
    result = survival_indicators(censor_time(event_time(state)))
    expected = (result["event_time"] < result["censor_time"]).float()
    assert torch.equal(result["indicator"], expected)


def test_survival_time_to_event_definition() -> None:
    """time_to_event == observed_time - time.

    Tests: definitional invariant — time_to_event is the difference.
    """
    state = _gaussian_state()
    result = survival_indicators(censor_time(event_time(state)))
    expected = result["observed_time"] - result["time"]
    assert torch.equal(result["time_to_event"], expected)


def test_survival_observed_time_nonneg() -> None:
    """observed_time >= 0 (both event_time and censor_time are positive).

    Tests: distribution support — times are non-negative.
    """
    state = _gaussian_state()
    result = survival_indicators(censor_time(event_time(state)))
    assert (result["observed_time"] >= 0).all()


def test_survival_observed_time_leq_event_and_censor() -> None:
    """observed_time <= event_time and observed_time <= censor_time.

    Tests: theorem-derived — min(a, b) <= a and min(a, b) <= b.
    """
    state = _gaussian_state()
    result = survival_indicators(censor_time(event_time(state)))
    assert (result["observed_time"] <= result["event_time"]).all()
    assert (result["observed_time"] <= result["censor_time"]).all()


def test_survival_pipeline_end_to_end() -> None:
    """Full chain: predictor -> gaussian -> event/censor_time -> survival.

    Tests: integration — all keys present with consistent shapes.
    """
    state = linear_predictor(N, T, P)
    state = gaussian(state, std=0.5, covariance=torch.eye(T))
    state = event_time(state)
    state = censor_time(state)
    result = survival_indicators(state)
    assert result["indicator"].shape == (N, 1, 1)
    assert result["observed_time"].shape == (N, 1, 1)
    assert result["time_to_event"].shape == (N, T, 1)
    for key in ("X", "beta", "eta", "y", "mu", "time", "event_time", "censor_time"):
        assert key in result


def test_survival_indicator_event_implies_event_is_observed_time() -> None:
    """When indicator==1, observed_time == event_time.

    Tests: definitional invariant — uncensored subjects have observed_time = event_time.
    """
    state = gaussian(linear_predictor(64, T, P), std=1.0, covariance=torch.eye(T))
    result = survival_indicators(censor_time(event_time(state)))
    events = result["indicator"] == 1.0
    if events.any():
        assert torch.equal(
            result["observed_time"][events], result["event_time"][events]
        )


def test_survival_indicator_censored_implies_censor_is_observed_time() -> None:
    """When indicator==0, observed_time == censor_time.

    Tests: definitional invariant — censored subjects have observed_time = censor_time.
    """
    state = gaussian(linear_predictor(64, T, P), std=1.0, covariance=torch.eye(T))
    result = survival_indicators(censor_time(event_time(state)))
    censored = result["indicator"] == 0.0
    if censored.any():
        assert torch.equal(
            result["observed_time"][censored], result["censor_time"][censored]
        )


# ---------------------------------------------------------------------------
# competing_risks (functional)
# ---------------------------------------------------------------------------


def test_competing_risks_shapes() -> None:
    """failure_times [N, T, K], tokens [N, T, 1], event_mask [N, T, K].

    Tests: shape contract for competing risks.
    """
    result = _competing_risks_state()
    assert result["failure_times"].shape == (N, T, K)
    assert result["tokens"].shape == (N, T, 1)
    assert result["event_mask"].shape == (N, T, K)


def test_competing_risks_failure_times_positive() -> None:
    """failure_times > 0 everywhere.

    Tests: distribution support — Weibull is strictly positive.
    """
    result = _competing_risks_state()
    assert (result["failure_times"] > 0).all()


def test_competing_risks_tokens_equal_argmin() -> None:
    """tokens == failure_times.argmin(dim=-1, keepdim=True).

    Tests: definitional invariant — winning risk is the one that fails first.
    """
    result = _competing_risks_state()
    expected = result["failure_times"].argmin(dim=-1, keepdim=True)
    assert torch.equal(result["tokens"], expected)


def test_competing_risks_event_mask_one_hot() -> None:
    """event_mask is one-hot along K (exactly one risk fires per timepoint).

    Tests: algebraic invariant — competing risks are mutually exclusive.
    """
    result = _competing_risks_state()
    assert result["event_mask"].dtype == torch.bool
    assert (result["event_mask"].sum(dim=-1) == 1).all()


def test_competing_risks_shape_one_is_exponential() -> None:
    """shape=1 Weibull: mean failure time converges to 1/rate for known eta.

    Tests: degenerate case — Weibull(scale, 1) = Exponential(1/scale).
    """
    torch.manual_seed(44)
    n = 4096
    X = torch.zeros(n, 1, 1)
    beta = torch.zeros(n, 1, 1)
    state = linear(linear_predictor(n, 1, 1, X=X, beta=beta), out_features=1)
    result = competing_risks(state, shape=1.0)
    # eta ≈ 0 → softplus(0) ≈ 0.693 → Exp(0.693) mean ≈ 1/0.693 ≈ 1.443
    scale = torch.nn.functional.softplus(torch.tensor(0.0)).item() + 1e-6
    expected_mean = scale  # Weibull(scale, 1) has mean = scale * Gamma(1 + 1/1) = scale
    empirical_mean = result["failure_times"].mean().item()
    assert abs(empirical_mean - expected_mean) < 0.1


# ---------------------------------------------------------------------------
# risk_indicators (functional)
# ---------------------------------------------------------------------------


def test_risk_indicators_one_hot() -> None:
    """indicator sums to 1.0 along K.

    Tests: algebraic invariant — one-hot encoding of winning risk.
    """
    state = _competing_risks_state()
    result = risk_indicators(state)
    assert result["indicator"].shape == (N, T, K)
    assert torch.allclose(result["indicator"].sum(dim=-1), torch.ones(N, T))


def test_risk_indicators_event_time_equals_min() -> None:
    """event_time[:,:,k] == min(failure_times) for all k.

    Tests: definitional invariant — event time is the first failure, broadcast.
    """
    state = _competing_risks_state()
    result = risk_indicators(state)
    assert has_failure_times(result)
    min_ft = result["failure_times"].min(dim=-1, keepdim=True)[0]
    assert torch.allclose(result["event_time"], min_ft.expand(-1, -1, K))


def test_risk_indicators_event_time_positive() -> None:
    """event_time > 0.

    Tests: distribution support — derived from positive failure_times.
    """
    state = _competing_risks_state()
    result = risk_indicators(state)
    assert (result["event_time"] > 0).all()


def test_risk_indicators_event_time_leq_all_failure_times() -> None:
    """event_time[:,:,k] <= failure_times[:,:,k] for all k.

    Tests: theorem-derived — min(a1,...,aK) <= ak.
    """
    result = risk_indicators(_competing_risks_state())
    assert has_failure_times(result)
    assert (result["event_time"] <= result["failure_times"]).all()


# ---------------------------------------------------------------------------
# independent_events (functional)
# ---------------------------------------------------------------------------


def test_independent_events_shapes() -> None:
    """event_mask [N, T, K] boolean.

    Tests: shape contract and dtype.
    """
    result = independent_events(linear(linear_predictor(N, T, P), out_features=K))
    assert result["event_mask"].shape == (N, T, K)
    assert result["event_mask"].dtype == torch.bool


def test_independent_events_multilabel() -> None:
    """With high prevalence, at least some timepoints have >1 active risk.

    Tests: multilabel property — events are independent, not mutually exclusive.
    """
    state = linear(linear_predictor(64, 20, P), out_features=K)
    result = independent_events(state, prevalence=0.5)
    events_per_tp = result["event_mask"].sum(dim=-1)
    assert (events_per_tp > 1).any()


def test_independent_events_prevalence_effect() -> None:
    """Lower prevalence produces fewer events on average.

    Tests: monotonicity — prevalence controls base event rate.
    """
    torch.manual_seed(62)
    state = linear(linear_predictor(64, 20, P), out_features=K)
    low = independent_events(state, prevalence=0.05)["event_mask"].float().mean()
    torch.manual_seed(62)
    state = linear(linear_predictor(64, 20, P), out_features=K)
    high = independent_events(state, prevalence=0.5)["event_mask"].float().mean()
    assert low < high


def test_independent_events_prevalence_half_eta_zero() -> None:
    """eta=0, prevalence=0.5: mean event rate converges to 0.5.

    Tests: large-sample convergence — logit(0.5)=0, sigmoid(0)=0.5.
    """
    torch.manual_seed(63)
    n = 4096
    X = torch.zeros(n, 1, 1)
    beta = torch.zeros(n, 1, 1)
    state = linear(linear_predictor(n, 1, 1, X=X, beta=beta), out_features=1)
    result = independent_events(state, prevalence=0.5)
    rate = result["event_mask"].float().mean().item()
    assert abs(rate - 0.5) < 0.05


# ---------------------------------------------------------------------------
# multi_event (functional)
# ---------------------------------------------------------------------------


def test_multi_event_shapes() -> None:
    """event_time [N, T, K], indicator [N, T, K].

    Tests: shape contract for multi-event encoding.
    """
    state = _competing_risks_state()
    result = multi_event(state, horizon=10.0)
    assert result["event_time"].shape == (N, T, K)
    assert result["indicator"].shape == (N, T, K)


def test_multi_event_horizon_clamp() -> None:
    """event_time <= horizon.

    Tests: numerical constraint — horizon ceiling is respected.
    """
    horizon = 5.0
    state = _competing_risks_state()
    result = multi_event(state, horizon=horizon)
    assert result["event_time"].max().item() <= horizon


def test_multi_event_indicator_binary() -> None:
    """indicator in {0.0, 1.0}.

    Tests: distribution support — event indicators are binary.
    """
    state = _competing_risks_state()
    result = multi_event(state, horizon=10.0)
    ind = result["indicator"]
    assert ((ind == 0.0) | (ind == 1.0)).all()


def test_multi_event_nonneg() -> None:
    """event_time >= 0.

    Tests: distribution support — time-to-event is non-negative.
    """
    state = _competing_risks_state()
    result = multi_event(state, horizon=10.0)
    assert (result["event_time"] >= 0).all()


def test_multi_event_works_with_independent_events() -> None:
    """multi_event accepts independent_events input (multilabel path).

    Tests: integration — multi_event is polymorphic over event process.
    """
    state = independent_events(
        linear(linear_predictor(N, T, P), out_features=K), prevalence=0.3
    )
    result = multi_event(state, horizon=10.0)
    assert result["event_time"].shape == (N, T, K)
    assert result["indicator"].shape == (N, T, K)


def test_multi_event_indicator_zero_at_horizon() -> None:
    """indicator == 0 where event_time == horizon (censored at boundary).

    Tests: definitional invariant — events at horizon are treated as censored.
    """
    horizon = 5.0
    state = _competing_risks_state()
    result = multi_event(state, horizon=horizon)
    at_horizon = result["event_time"] == horizon
    if at_horizon.any():
        assert (result["indicator"][at_horizon] == 0.0).all()


# ---------------------------------------------------------------------------
# discretize_risk (functional)
# ---------------------------------------------------------------------------


def test_discretize_risk_shapes() -> None:
    """discrete_event_time [N, T, K, J] where J = len(boundaries) - 1.

    Tests: shape contract for discretization.
    """
    boundaries = torch.tensor([0.0, 1.0, 2.0, 5.0, 10.0])
    J = len(boundaries) - 1
    state = risk_indicators(_competing_risks_state())
    result = discretize_risk(state, boundaries)
    assert result["discrete_event_time"].shape == (N, T, K, J)


def test_discretize_risk_values_bounded() -> None:
    """discrete_event_time values in [0, 1].

    Tests: distribution support — fractional exposure is bounded.
    """
    boundaries = torch.tensor([0.0, 1.0, 2.0, 5.0])
    state = risk_indicators(_competing_risks_state())
    result = discretize_risk(state, boundaries)
    det = result["discrete_event_time"]
    assert (det >= 0.0).all()
    assert (det <= 1.0).all()


def test_discretize_risk_single_interval() -> None:
    """J=1 produces discrete_event_time [N, T, K, 1].

    Tests: degenerate case — single bin.
    """
    boundaries = torch.tensor([0.0, 10.0])
    state = risk_indicators(_competing_risks_state())
    result = discretize_risk(state, boundaries)
    assert result["discrete_event_time"].shape == (N, T, K, 1)


def test_discretize_risk_full_chain() -> None:
    """predictor -> linear -> competing_risks -> risk_indicators -> discretize.

    Tests: integration — full functional chain produces all expected keys.
    """
    boundaries = torch.tensor([0.0, 0.5, 1.0, 2.0])
    state = linear_predictor(N, T, P)
    state = linear(state, out_features=K)
    state = competing_risks(state)
    state = risk_indicators(state)
    result = discretize_risk(state, boundaries)
    for key in ("failure_times", "indicator", "event_time", "discrete_event_time"):
        assert key in result
    assert result["discrete_event_time"].shape == (N, T, K, 3)


# ---------------------------------------------------------------------------
# Reference cross-validation (best practice #6)
# ---------------------------------------------------------------------------


def test_event_time_quantiles_vs_scipy() -> None:
    """eta=0 → Exp(1). Empirical quantiles match Exp(1) reference values.

    Tests: reference cross-validation — ppf values from scipy.stats.expon.
    """
    torch.manual_seed(8)
    n = 4096
    X = torch.zeros(n, 1, 1)
    beta = torch.zeros(n, 1, 1)
    state = gaussian(
        linear_predictor(n, 1, 1, X=X, beta=beta), std=0.01, covariance=torch.eye(1)
    )
    result = event_time(state)
    event_times = result["event_time"].squeeze()
    empirical = torch.quantile(event_times, torch.tensor([0.25, 0.50, 0.75]))
    # scipy.stats.expon.ppf([0.25, 0.5, 0.75])
    reference = torch.tensor([0.2877, 0.6931, 1.3863])
    assert torch.allclose(empirical, reference, atol=0.1)


def test_competing_risks_weibull_quantiles_vs_scipy() -> None:
    """K=1, shape=2, eta=0 → Weibull(softplus(0), 2). Quantiles match reference.

    Tests: reference cross-validation — ppf values from scipy.stats.weibull_min.
    """
    torch.manual_seed(46)
    n = 4096
    X = torch.zeros(n, 1, 1)
    beta = torch.zeros(n, 1, 1)
    state = linear(linear_predictor(n, 1, 1, X=X, beta=beta), out_features=1)
    result = competing_risks(state, shape=2.0)
    ft = result["failure_times"].squeeze()
    empirical = torch.quantile(ft, torch.tensor([0.25, 0.50, 0.75]))
    # Weibull_min(c=2, scale=softplus(0)+1e-6 ≈ 0.6931472)
    # ppf(q) = scale * (-ln(1-q))^(1/c)
    reference = torch.tensor([0.3718, 0.5770, 0.8160])
    assert torch.allclose(empirical, reference, atol=0.1)


# ---------------------------------------------------------------------------
# Numerical stability (best practice #9)
# ---------------------------------------------------------------------------


def test_event_time_large_negative_eta() -> None:
    """eta=-20 → near-zero rate → large but finite event times.

    Tests: numerical stability — extreme negative eta doesn't produce NaN/Inf.
    """
    n = 64
    X = torch.full((n, 1, 1), -20.0)
    beta = torch.ones(n, 1, 1)
    state = gaussian(
        linear_predictor(n, 1, 1, X=X, beta=beta), std=0.01, covariance=torch.eye(1)
    )
    result = event_time(state)
    event_times = result["event_time"]
    assert event_times.isfinite().all()
    assert (event_times > 0).all()


def test_event_time_large_positive_eta() -> None:
    """eta=+20 → huge rate → tiny but positive event times.

    Tests: numerical stability — extreme positive eta doesn't produce zeros.
    """
    n = 64
    X = torch.full((n, 1, 1), 20.0)
    beta = torch.ones(n, 1, 1)
    state = gaussian(
        linear_predictor(n, 1, 1, X=X, beta=beta), std=0.01, covariance=torch.eye(1)
    )
    result = event_time(state)
    event_times = result["event_time"]
    assert event_times.isfinite().all()
    assert (event_times > 0).all()


# ---------------------------------------------------------------------------
# Degenerate cases (best practice #3)
# ---------------------------------------------------------------------------


def test_competing_risks_K1_tokens_all_zero() -> None:
    """K=1 → only one risk → tokens always 0.

    Tests: degenerate case — single risk always wins.
    """
    result = competing_risks(linear(linear_predictor(N, T, P), out_features=1))
    assert (result["tokens"] == 0).all()


def test_risk_indicators_K1_indicator_all_ones() -> None:
    """K=1 → indicator always 1.0 (the sole risk always wins).

    Tests: degenerate case — single risk yields all-ones indicator.
    """
    state = competing_risks(linear(linear_predictor(N, T, P), out_features=1))
    result = risk_indicators(state)
    assert (result["indicator"] == 1.0).all()


def test_independent_events_prevalence_near_zero() -> None:
    """prevalence ≈ 0 → event rate < 0.01.

    Tests: degenerate case — near-zero prevalence suppresses events.
    """
    torch.manual_seed(64)
    n = 4096
    X = torch.zeros(n, 1, 1)
    beta = torch.zeros(n, 1, 1)
    state = linear(linear_predictor(n, 1, 1, X=X, beta=beta), out_features=1)
    result = independent_events(state, prevalence=1e-6)
    rate = result["event_mask"].float().mean().item()
    assert rate < 0.01


def test_independent_events_prevalence_near_one() -> None:
    """prevalence ≈ 1 → event rate > 0.99.

    Tests: degenerate case — near-one prevalence fires almost always.
    """
    torch.manual_seed(65)
    n = 4096
    X = torch.zeros(n, 1, 1)
    beta = torch.zeros(n, 1, 1)
    state = linear(linear_predictor(n, 1, 1, X=X, beta=beta), out_features=1)
    result = independent_events(state, prevalence=1 - 1e-6)
    rate = result["event_mask"].float().mean().item()
    assert rate > 0.99


# ---------------------------------------------------------------------------
# Closed-form / hand-crafted oracles (best practice #1)
# ---------------------------------------------------------------------------


def test_multi_event_suffix_minimum_oracle() -> None:
    """Hand-traced suffix-minimum TTE for N=1, T=4, K=2.

    Setup: time=[0,1,2,3], risk0 fires at t=1, risk1 fires at t=2.
    Manual trace of suffix-minimum yields known event_time and indicator.

    Tests: closed-form oracle — exact expected output from manual algorithm trace.
    """
    time = torch.tensor([[[0], [1], [2], [3]]], dtype=torch.float)  # [1, 4, 1]
    event_mask = torch.tensor(
        [[[False, False], [True, False], [False, True], [False, False]]]
    )  # [1, 4, 2]
    state = EventProcessState(
        time=time,
        event_mask=event_mask,
        eta=torch.zeros(1, 4, 2),
        X=torch.zeros(1, 4, 1),
        beta=torch.zeros(1, 1, 1),
    )
    result = multi_event(state, horizon=10.0)
    expected_et = torch.tensor([[[1.0, 2.0], [10.0, 1.0], [10.0, 10.0], [10.0, 10.0]]])
    expected_ind = torch.tensor([[[1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]])
    assert torch.equal(result["event_time"], expected_et)
    assert torch.equal(result["indicator"], expected_ind)


def test_discretize_risk_hand_oracle() -> None:
    """Hand-computed discretization for event_time=2.5, boundaries=[0,1,2,3,5].

    Event (indicator=1): only the containing interval gets fractional exposure.
    Censored (indicator=0): full exposure up to the event time.

    Tests: closed-form oracle — exact expected output from manual formula trace.
    """
    boundaries = torch.tensor([0.0, 1.0, 2.0, 3.0, 5.0])
    # Event case
    state_event = RiskIndicatorState(
        event_time=torch.tensor([[[2.5]]]),  # [1, 1, 1]
        indicator=torch.tensor([[[1.0]]]),  # [1, 1, 1]
        failure_times=torch.tensor([[[2.5]]]),
        tokens=torch.tensor([[[0]]]),
        event_mask=torch.tensor([[[True]]]),
        eta=torch.zeros(1, 1, 1),
        time=torch.zeros(1, 1, 1),
        X=torch.zeros(1, 1, 1),
        beta=torch.zeros(1, 1, 1),
    )
    result_event = discretize_risk(state_event, boundaries)
    expected_event = torch.tensor([[[[0.0, 0.0, 0.5, 0.0]]]])
    assert torch.allclose(result_event["discrete_event_time"], expected_event)
    # Censored case
    state_censored = RiskIndicatorState(
        event_time=torch.tensor([[[2.5]]]),
        indicator=torch.tensor([[[0.0]]]),
        failure_times=torch.tensor([[[2.5]]]),
        tokens=torch.tensor([[[0]]]),
        event_mask=torch.tensor([[[True]]]),
        eta=torch.zeros(1, 1, 1),
        time=torch.zeros(1, 1, 1),
        X=torch.zeros(1, 1, 1),
        beta=torch.zeros(1, 1, 1),
    )
    result_censored = discretize_risk(state_censored, boundaries)
    expected_censored = torch.tensor([[[[1.0, 1.0, 0.5, 0.0]]]])
    assert torch.allclose(result_censored["discrete_event_time"], expected_censored)


# ---------------------------------------------------------------------------
# Untested code path
# ---------------------------------------------------------------------------


def test_multi_event_horizon_none_inferred() -> None:
    """horizon=None auto-infers from data. Result is finite; indicator=0 at max.

    Tests: untested code path — horizon=None branch in multi_event.
    """
    state = _competing_risks_state()
    result = multi_event(state, horizon=None)
    assert result["event_time"].isfinite().all()
    max_et = result["event_time"].max()
    at_max = result["event_time"] == max_et
    if at_max.any():
        assert (result["indicator"][at_max] == 0.0).all()


# ---------------------------------------------------------------------------
# Algebraic invariant (best practice #2)
# ---------------------------------------------------------------------------


def test_discretize_risk_exposure_monotonic() -> None:
    """For censored entries, exposure is non-increasing across intervals.

    Tests: algebraic invariant — censored exposure decreases monotonically.
    """
    boundaries = torch.tensor([0.0, 1.0, 2.0, 3.0, 5.0, 10.0])
    state = risk_indicators(_competing_risks_state())
    # Force all censored
    state["indicator"] = torch.zeros_like(state["indicator"])
    result = discretize_risk(state, boundaries)
    det = result["discrete_event_time"]  # [N, T, K, J]
    diffs = det[..., 1:] - det[..., :-1]
    assert (diffs <= 1e-6).all()


# ---------------------------------------------------------------------------
# Parametrization (best practice #10)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,t,k", [(4, 1, 1), (8, 10, 2), (16, 6, 5)])
def test_competing_risks_shapes_parametrized(n: int, t: int, k: int) -> None:
    """Competing risks shapes match (N, T, K) across configurations.

    Tests: parametrized shape contract.
    """
    result = competing_risks(linear(linear_predictor(n, t, P), out_features=k))
    assert result["failure_times"].shape == (n, t, k)
    assert result["tokens"].shape == (n, t, 1)
    assert result["event_mask"].shape == (n, t, k)


@pytest.mark.parametrize("n,t,k", [(4, 1, 2), (8, 10, 1), (16, 6, 5)])
def test_multi_event_shapes_parametrized(n: int, t: int, k: int) -> None:
    """Multi-event shapes match (N, T, K) across configurations.

    Tests: parametrized shape contract.
    """
    state = competing_risks(linear(linear_predictor(n, t, P), out_features=k))
    result = multi_event(state, horizon=10.0)
    assert result["event_time"].shape == (n, t, k)
    assert result["indicator"].shape == (n, t, k)
