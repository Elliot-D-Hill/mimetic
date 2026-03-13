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
    """Build the linear predictor eta = X*beta.

    Parameters
    ----------
    num_samples
        Number of subjects N.
    num_timepoints
        Number of time points T.
    num_features
        Number of design matrix features p.
    X
        Design matrix [N, T, p]; standard normal if omitted.
    beta
        Coefficients [N, p, 1]; standard normal if omitted.
    time
        Observation schedule [N, T, 1]; arange(T) if omitted.

    Returns
    -------
    PredictorState
        State with keys ``eta`` [N, T, 1], ``time`` [N, T, 1],
        ``X`` [N, T, p], ``beta`` [N, p, 1].

    See Also
    --------
    random_effects : Upgrade to GLMM.
    gaussian, poisson, bernoulli : Response distributions.

    Notes
    -----
    Implements Fahrmeir et al. [1]_, Box 7.1:

    .. math:: \\eta = X \\beta

    References
    ----------
    .. [1] Fahrmeir, L., Kneib, T., Lang, S., & Marx, B. (2013).
       *Regression*. Springer.

    Examples
    --------
    >>> from mimetic import linear_predictor
    >>> state = linear_predictor(2, 3, 4)
    >>> state["eta"].shape
    torch.Size([2, 3, 1])
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

    Parameters
    ----------
    state
        Must contain ``eta`` [N, T, 1].
    std
        Residual standard deviation sigma.
    covariance
        Residual correlation matrix [T, T].

    Returns
    -------
    ObservedState
        Adds ``y`` [N, T, 1], ``mu`` [N, T, 1], ``noise`` [N, T, 1].

    See Also
    --------
    poisson : Log link.
    bernoulli : Logit link.
    categorical : Softmax link.
    ordinal : Cumulative logit.

    Notes
    -----
    .. math:: y = \\eta + \\varepsilon, \\quad
              \\varepsilon \\sim N(0,\\, \\sigma^2 \\Sigma)

    Examples
    --------
    >>> import torch
    >>> from mimetic import linear_predictor, gaussian
    >>> state = linear_predictor(2, 3, 4)
    >>> result = gaussian(state, std=1.0, covariance=torch.eye(3))
    >>> result["y"].shape
    torch.Size([2, 3, 1])
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
    """Sample Poisson response y ~ Poisson(exp(eta)) (log link).

    Parameters
    ----------
    state
        Must contain ``eta`` [N, T, 1].

    Returns
    -------
    ObservedState
        Adds ``y`` [N, T, 1], ``mu`` [N, T, 1].

    See Also
    --------
    gaussian : Identity link.
    bernoulli : Logit link.

    Notes
    -----
    .. math:: y \\sim \\text{Poisson}(\\exp(\\eta))

    Examples
    --------
    >>> from mimetic import linear_predictor, poisson
    >>> state = linear_predictor(2, 3, 4)
    >>> result = poisson(state)
    >>> result["y"].shape
    torch.Size([2, 3, 1])
    """
    mu = torch.exp(state["eta"])  # [N, T, 1]
    y = torch.poisson(mu)  # [N, T, 1]
    return ObservedState(**state, y=y, mu=mu)


def bernoulli(state: PredictorState, prevalence: float = 0.5) -> ObservedState:
    """Sample Bernoulli response from linear predictor (logit link).

    Parameters
    ----------
    state
        Must contain ``eta`` [N, T, 1].
    prevalence
        Base rate used to shift eta before applying sigmoid.

    Returns
    -------
    ObservedState
        Adds ``y`` [N, T, 1], ``mu`` [N, T, 1].

    See Also
    --------
    gaussian : Identity link.
    poisson : Log link.
    categorical : Softmax link.

    Notes
    -----
    .. math:: \\mu = \\sigma(\\eta + \\text{logit}(p)), \\quad
              y \\sim \\text{Bernoulli}(\\mu)

    Examples
    --------
    >>> from mimetic import linear_predictor, bernoulli
    >>> state = linear_predictor(2, 3, 4)
    >>> result = bernoulli(state, prevalence=0.3)
    >>> result["y"].shape
    torch.Size([2, 3, 1])
    """
    shift = torch.logit(torch.tensor(prevalence))
    mu = torch.sigmoid(state["eta"] + shift)  # [N, T, 1]
    y = torch.bernoulli(mu)  # [N, T, 1]
    return ObservedState(**state, y=y, mu=mu)


def categorical(state: PredictorState) -> ObservedState:
    """Sample categorical response from linear predictor (softmax link).

    Parameters
    ----------
    state
        Must contain ``eta`` [N, T, K] with K > 1 classes.

    Returns
    -------
    ObservedState
        Adds ``y`` [N, T, 1], ``mu`` [N, T, K].

    See Also
    --------
    ordinal : Cumulative logit for ordered categories.
    bernoulli : Binary case.

    Notes
    -----
    .. math:: \\mu = \\text{softmax}(\\eta), \\quad
              y \\sim \\text{Categorical}(\\mu)

    Examples
    --------
    >>> from mimetic import linear_predictor, linear, categorical
    >>> state = linear(linear_predictor(2, 3, 4), out_features=5)
    >>> result = categorical(state)
    >>> result["y"].shape
    torch.Size([2, 3, 1])
    """
    mu = torch.softmax(state["eta"], dim=-1)  # [N, T, K]
    y = dist.Categorical(probs=mu).sample().unsqueeze(-1)  # [N, T, 1]
    return ObservedState(**state, y=y, mu=mu)


def ordinal(
    state: PredictorState, num_classes: int, start: float = -2.0, end: float = 2.0
) -> ObservedState:
    """Sample ordinal response via cumulative logit model.

    Parameters
    ----------
    state
        Must contain ``eta`` [N, T, 1].
    num_classes
        Number of ordinal categories K.
    start
        Lower bound for evenly spaced thresholds.
    end
        Upper bound for evenly spaced thresholds.

    Returns
    -------
    ObservedState
        Adds ``y`` [N, T, 1], ``mu`` [N, T, K].

    See Also
    --------
    categorical : Unordered categories.
    bernoulli : Binary case.

    Notes
    -----
    .. math:: P(y \\leq k) = \\sigma(\\theta_k - \\eta), \\quad
              k = 1, \\ldots, K-1

    Examples
    --------
    >>> from mimetic import linear_predictor, ordinal
    >>> state = linear_predictor(2, 3, 4)
    >>> result = ordinal(state, num_classes=4)
    >>> result["mu"].shape
    torch.Size([2, 3, 4])
    """
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
    """Add random effects U*gamma to the linear predictor.

    Upgrades eta = X*beta to eta = X*beta + U*gamma where U is a
    Vandermonde (polynomial) basis by default.

    Parameters
    ----------
    state
        Must contain ``eta`` [N, T, 1] and ``time`` [N, T, 1].
    std
        Standard deviations for each random effect; length determines q.
    correlation
        Off-diagonal correlation. Scalar gives compound symmetry;
        matrix gives a user-provided [q, q] correlation matrix.
    U
        Random-effects design matrix [N, T, q]; Vandermonde basis if omitted.
    gamma
        Random-effects coefficients [N, q, 1]; sampled from MVN(0, Q)
        if omitted.

    Returns
    -------
    PredictorState
        Updates ``eta`` [N, T, 1] and adds ``gamma`` [N, q, 1],
        ``U`` [N, T, q].

    See Also
    --------
    linear_predictor : Build the fixed-effects predictor.
    random_effects_covariance : Build the covariance Q = S R S.

    Notes
    -----
    Implements Fahrmeir et al. [1]_, Eq. 7.11:

    .. math:: \\eta = X \\beta + U \\gamma, \\quad \\gamma \\sim N(0, Q)

    References
    ----------
    .. [1] Fahrmeir, L., Kneib, T., Lang, S., & Marx, B. (2013).
       *Regression*. Springer.

    Examples
    --------
    >>> from mimetic import linear_predictor, random_effects
    >>> state = random_effects(linear_predictor(2, 3, 4), std=[0.5, 1.0])
    >>> state["gamma"].shape
    torch.Size([2, 2, 1])
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

    Parameters
    ----------
    state
        Must contain ``eta``.
    fn
        Elementwise activation (e.g. ``torch.relu``, ``torch.tanh``).

    Returns
    -------
    T
        Same state type with transformed ``eta``.

    See Also
    --------
    linear : Linear projection.
    mlp : Composed linear-activation-linear.

    Examples
    --------
    >>> import torch
    >>> from mimetic import linear_predictor, activation
    >>> state = activation(linear_predictor(2, 3, 4), torch.relu)
    >>> state["eta"].shape
    torch.Size([2, 3, 1])
    """
    result = state.copy()
    result["eta"] = fn(state["eta"])
    return result


def linear[T: PredictorState](
    state: T, out_features: int, weight: Tensor | None = None
) -> T:
    """Apply a random linear projection to the linear predictor.

    Parameters
    ----------
    state
        Must contain ``eta`` [N, T, in_features].
    out_features
        Output dimension.
    weight
        Projection matrix [in_features, out_features]; random if omitted.

    Returns
    -------
    T
        Same state type with ``eta`` reshaped to [N, T, out_features].

    See Also
    --------
    activation : Nonlinear transform.
    mlp : Composed linear-activation-linear.

    Notes
    -----
    .. math:: \\eta' = \\eta \\, W

    Examples
    --------
    >>> from mimetic import linear_predictor, linear
    >>> state = linear(linear_predictor(2, 3, 4), out_features=5)
    >>> state["eta"].shape
    torch.Size([2, 3, 5])
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

    Parameters
    ----------
    state
        Must contain ``eta`` [N, T, in_features].
    hidden_features
        Hidden layer dimension.
    fn
        Activation function between layers.
    out_features
        Output dimension; defaults to input dimension.

    Returns
    -------
    T
        Same state type with transformed ``eta``.

    See Also
    --------
    linear : Single linear projection.
    activation : Nonlinear transform.

    Examples
    --------
    >>> from mimetic import linear_predictor, mlp
    >>> state = mlp(linear_predictor(2, 3, 4), hidden_features=8)
    >>> state["eta"].shape
    torch.Size([2, 3, 1])
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
    """Tokenize the design matrix via Dirichlet-skewed softmax.

    Parameters
    ----------
    state
        Must contain ``X`` [N, T, p].
    vocab_size
        Number of discrete token classes K.
    concentration
        Dirichlet concentration for the softmax skew.

    Returns
    -------
    TokenizedState
        Adds ``tokens`` [N, T, 1].

    See Also
    --------
    gaussian, poisson : Response distributions that precede tokenization.

    Examples
    --------
    >>> import torch
    >>> from mimetic import linear_predictor, gaussian, tokens
    >>> state = gaussian(linear_predictor(2, 3, 4), 1.0, torch.eye(3))
    >>> result = tokens(state, vocab_size=100)
    >>> result["tokens"].shape
    torch.Size([2, 3, 1])
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
    """Replace observation times with Gamma-distributed intervals.

    Parameters
    ----------
    state
        Must contain ``eta`` [N, T, ...].
    shape
        Gamma distribution shape parameter.
    rate
        Gamma distribution rate parameter.

    Returns
    -------
    T
        Same state type with ``time`` [N, T, 1] replaced by cumulative
        Gamma intervals.

    See Also
    --------
    linear_predictor : Default evenly spaced time grid.

    Notes
    -----
    .. math:: \\Delta t_i \\sim \\text{Gamma}(\\alpha, \\beta), \\quad
              t_i = \\sum_{j \\leq i} \\Delta t_j

    Examples
    --------
    >>> from mimetic import linear_predictor, observation_time
    >>> state = observation_time(linear_predictor(2, 3, 4), shape=2.0, rate=1.0)
    >>> state["time"].shape
    torch.Size([2, 3, 1])
    """
    num_samples, num_timepoints = state["eta"].shape[0], state["eta"].shape[1]
    gamma_dist = dist.Gamma(shape, rate)
    time_intervals = gamma_dist.sample((num_samples, num_timepoints, 1))  # [N, T, 1]
    time = time_intervals.cumsum(dim=1)  # [N, T, 1]
    result = state.copy()
    result["time"] = time
    return result
