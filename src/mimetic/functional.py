from collections.abc import Callable, Sequence

import torch
import torch.distributions as dist
import torch.nn.functional as F
from torch import Tensor, arange, randn

from .covariance import random_effects_covariance
from .states import ObservedState, PredictorState, TokenizedState


def linear_predictor(
    num_samples: int,
    num_timepoints: int,
    num_features: int,
    X: Tensor | None = None,
    beta: Tensor | None = None,
    time: Tensor | None = None,
) -> PredictorState:
    """Build the linear predictor eta = X*beta (Fahrmeir Box 7.1).

    Args
    ----
    num_samples
        Number of subjects N.
    num_timepoints
        Number of time points T.
    num_features
        Number of design matrix features p.
    X
        Optional design matrix [N, T, p]; random if omitted.
    beta
        Optional coefficients [N, p, 1]; random if omitted.
    time
        Optional observation schedule [N, T, 1]; arange(T) if omitted.
    """
    if X is None:
        X = randn(num_samples, num_timepoints, num_features)  # [N, T, p]
    if beta is None:
        beta = randn(num_samples, num_features, 1)  # [N, p, 1]
    eta = torch.bmm(X, beta)  # [N, T, 1]
    if time is None:
        time_values = arange(num_timepoints, dtype=torch.float32).view(1, -1, 1)
        time = time_values.expand(num_samples, -1, -1)  # [N, T, 1]
    return PredictorState(eta=eta, time=time, X=X, beta=beta)


# ---------------------------------------------------------------------------
# Response distributions (family functions)
# ---------------------------------------------------------------------------


def gaussian(state: PredictorState, std: float, covariance: Tensor) -> ObservedState:
    """Sample Gaussian response y = eta + epsilon (identity link).

    Args
    ----
    std
        Residual standard deviation sigma.
    covariance
        Residual correlation matrix Sigma [T, T].
    """
    eta = state["eta"]  # [N, T, 1]
    num_samples, num_timepoints = eta.shape[0], eta.shape[1]
    mu = eta.clone()  # [N, T, 1]
    scaled_cov = covariance * std**2
    mvn = dist.MultivariateNormal(torch.zeros(num_timepoints), scaled_cov)
    noise = mvn.sample((num_samples, 1))  # [N, 1, T]
    noise = noise.permute(0, 2, 1)  # [N, T, 1]
    y = mu + noise  # [N, T, 1]
    return ObservedState(**state, y=y, mu=mu, noise=noise)


def poisson(state: PredictorState) -> ObservedState:
    """Sample Poisson response y ~ Poisson(exp(eta)) (log link)."""
    mu = torch.exp(state["eta"])  # [N, T, 1]
    y = torch.poisson(mu)  # [N, T, 1]
    return ObservedState(**state, y=y, mu=mu)


def bernoulli(state: PredictorState, prevalence: float = 0.5) -> ObservedState:
    """Sample Bernoulli response from linear predictor (logit link).

    Args
    ----
    prevalence
        Base rate used to shift eta before applying sigmoid.
    """
    shift = torch.logit(torch.tensor(prevalence))
    mu = torch.sigmoid(state["eta"] + shift)  # [N, T, 1]
    y = torch.bernoulli(mu)  # [N, T, 1]
    return ObservedState(**state, y=y, mu=mu)


def categorical(state: PredictorState) -> ObservedState:
    """Sample categorical response from linear predictor (softmax link)."""
    mu = torch.softmax(state["eta"], dim=-1)  # [N, T, K]
    y = dist.Categorical(probs=mu).sample().unsqueeze(-1)  # [N, T, 1]
    return ObservedState(**state, y=y, mu=mu)


def ordinal(
    state: PredictorState, num_classes: int, start: float = -2.0, end: float = 2.0
) -> ObservedState:
    """Sample ordinal response via cumulative logit model."""
    eta = state["eta"]  # [N, T, 1]
    thresholds = torch.linspace(start, end, num_classes - 1)  # [K-1]
    cumulative = torch.sigmoid(thresholds - eta)  # [N, T, K-1]
    ones = torch.ones_like(eta)
    cumulative = torch.cat([cumulative, ones], dim=-1)  # [N, T, K]
    zeros = torch.zeros_like(eta)
    cumulative_shifted = torch.cat([zeros, cumulative[..., :-1]], dim=-1)  # [N, T, K]
    mu = cumulative - cumulative_shifted  # [N, T, K]
    y = dist.Categorical(probs=mu).sample().unsqueeze(-1)  # [N, T, 1]
    return ObservedState(**state, y=y, mu=mu)


# ---------------------------------------------------------------------------
# Predictor transforms (pre-family)
# ---------------------------------------------------------------------------


def random_effects(
    state: PredictorState,
    std: Sequence[float] | Tensor | float,
    correlation: Tensor | float = 0.0,
    U: Tensor | None = None,
    gamma: Tensor | None = None,
) -> PredictorState:
    """Add random effects U*gamma to the linear predictor (Fahrmeir et al., Eq. 7.11).

    Upgrades eta = X*beta to eta = X*beta + U*gamma.
    U is a Vandermonde (polynomial) basis.

    Args
    ----
    std
        Standard deviations for each random effect; len determines q.
    correlation
        Off-diagonal correlation. Scalar gives compound symmetry,
        Tensor gives a user-provided [q, q] correlation matrix.
    U
        Optional design matrix [N, T, q]; Vandermonde basis if omitted.
    gamma
        Optional coefficients [N, q, 1]; sampled from MVN(0, Q) if omitted.
    """
    num_samples, num_timepoints, _ = state["eta"].shape
    s = torch.atleast_1d(torch.as_tensor(std, dtype=torch.float32))
    q = len(s)
    if gamma is None:
        Q = random_effects_covariance(std, correlation)  # [q, q]
        mvn = dist.MultivariateNormal(torch.zeros(q), Q)
        gamma = mvn.sample((num_samples,)).unsqueeze(-1)  # [N, q, 1]
    if U is None:
        t = state["time"].squeeze(-1)  # [N, T]
        t_centered = t - t.mean(dim=1, keepdim=True)  # [N, T]
        powers = arange(q, dtype=torch.float32)  # [q]
        U = t_centered.unsqueeze(-1) ** powers  # [N, T, q]
    random_effect = torch.bmm(U, gamma)  # [N, T, 1]
    result = state.copy()
    result["eta"] = state["eta"] + random_effect  # [N, T, 1]
    result["gamma"] = gamma
    result["U"] = U
    return result


def activation[T: PredictorState](state: T, fn: Callable[[Tensor], Tensor]) -> T:
    """Apply a nonlinear activation to the linear predictor.

    Args
    ----
    fn
        Elementwise activation (e.g. torch.relu, torch.tanh).
    """
    result = state.copy()
    result["eta"] = fn(state["eta"])
    return result


def linear[T: PredictorState](
    state: T, out_features: int, weight: Tensor | None = None
) -> T:
    """Apply a random linear projection to the linear predictor.

    Transforms eta [N, T, in] -> eta [N, T, out] via eta @ W.

    Args
    ----
    out_features
        Output dimension.
    weight
        Optional [in, out] weight matrix; random if omitted.
    """
    result = state.copy()
    eta = state["eta"]  # [N, T, in]
    if weight is None:
        weight = randn(eta.shape[-1], out_features)  # [in, out]
    result["eta"] = eta @ weight  # [N, T, out]
    return result


def mlp[T: PredictorState](
    state: T,
    hidden_features: int,
    fn: Callable[[Tensor], Tensor] = F.relu,
    out_features: int | None = None,
) -> T:
    """Apply a single hidden-layer MLP: linear -> activation -> linear.

    Args
    ----
    hidden_features
        Hidden layer dimension.
    fn
        Activation function between layers.
    out_features
        Output dimension; defaults to input dimension.
    """
    if out_features is None:
        out_features = state["eta"].shape[-1]
    state = activation(linear(state, hidden_features), fn)
    return linear(state, out_features)


# ---------------------------------------------------------------------------
# Post-family transforms
# ---------------------------------------------------------------------------


def tokens(
    state: ObservedState, vocab_size: int, concentration: float = 1.0
) -> TokenizedState:
    """Tokenize design matrix via Dirichlet-skewed softmax.

    Produces 'tokens' [N, T, 1] from 'X' [N, T, p].
    """
    X = state["X"]  # [N, T, p]
    p = X.shape[-1]
    weight = randn(vocab_size, p)  # [K, p]
    logits = F.linear(X, weight)  # [N, T, K]
    prior = torch.ones(vocab_size) * concentration  # [K]
    skew = dist.Dirichlet(prior).sample(logits.shape[:-1]).log()  # [N, T, K]
    probs = F.softmax(logits + skew, dim=-1)  # [N, T, K]
    flat = torch.multinomial(probs.view(-1, probs.size(-1)), 1)  # [N*T, 1]
    token_ids = flat.view(*X.shape[:-1], 1)  # [N, T, 1]
    return TokenizedState(**state, tokens=token_ids)


def observation_time[T: PredictorState](state: T, shape: float, rate: float) -> T:
    """Generate Gamma-distributed observation intervals.

    Produces 'time' [N, T, 1].
    """
    num_samples, num_timepoints = state["eta"].shape[0], state["eta"].shape[1]
    gamma_dist = dist.Gamma(shape, rate)
    time_intervals = gamma_dist.sample((num_samples, num_timepoints, 1))  # [N, T, 1]
    time = time_intervals.cumsum(dim=1)  # [N, T, 1]
    result = state.copy()
    result["time"] = time
    return result
