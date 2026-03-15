"""
Functional building blocks for composing longitudinal data simulations.
"""

from collections.abc import Callable, Sequence

import torch
import torch.distributions as dist
import torch.nn.functional as F
from torch import Tensor, arange, randn

from .covariance import random_effects_covariance
from .states import ObservedState, PredictorState, TokenizedState, has_random_effects
from .types import Correlation, PositiveFloat, PositiveInt, UnitInterval


def linear_predictor(
    num_samples: PositiveInt,
    num_timepoints: PositiveInt,
    num_features: PositiveInt,
    X: Tensor | None = None,
    beta: Tensor | None = None,
    time: Tensor | None = None,
) -> PredictorState:
    """
    Build the linear predictor eta = X*beta.

    Parameters
    ----------
    num_samples : int
        Number of subjects N.
    num_timepoints : int
        Number of time points T.
    num_features : int
        Number of design matrix features p.
    X : Tensor or None
        Design matrix [N, T, p]; standard normal if omitted.
    beta : Tensor or None
        Coefficients [N, p, 1]; standard normal if omitted.
    time : Tensor or None
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
    >>> from simulacra import linear_predictor
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


def gaussian(
    state: PredictorState, std: PositiveFloat, covariance: Tensor
) -> ObservedState:
    """
    Sample Gaussian response y = eta + epsilon (identity link).

    Parameters
    ----------
    state : PredictorState
        Must contain ``eta`` [N, T, 1].
    std : float
        Residual standard deviation sigma.
    covariance : Tensor
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
    >>> from simulacra import linear_predictor, gaussian
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
    """
    Sample Poisson response y ~ Poisson(exp(eta)) (log link).

    Parameters
    ----------
    state : PredictorState
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
    >>> from simulacra import linear_predictor, poisson
    >>> state = linear_predictor(2, 3, 4)
    >>> result = poisson(state)
    >>> result["y"].shape
    torch.Size([2, 3, 1])
    """
    mu = torch.exp(state["eta"])  # [N, T, 1]
    y = torch.poisson(mu)  # [N, T, 1]
    return ObservedState(**state, y=y, mu=mu)


def bernoulli(state: PredictorState, prevalence: UnitInterval = 0.5) -> ObservedState:
    """
    Sample Bernoulli response from linear predictor (logit link).

    Parameters
    ----------
    state : PredictorState
        Must contain ``eta`` [N, T, 1].
    prevalence : float
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
    >>> from simulacra import linear_predictor, bernoulli
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
    """
    Sample categorical response from linear predictor (softmax link).

    Parameters
    ----------
    state : PredictorState
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
    >>> from simulacra import linear_predictor, linear, categorical
    >>> state = linear(linear_predictor(2, 3, 4), out_features=5)
    >>> result = categorical(state)
    >>> result["y"].shape
    torch.Size([2, 3, 1])
    """
    mu = torch.softmax(state["eta"], dim=-1)  # [N, T, K]
    y = dist.Categorical(probs=mu).sample().unsqueeze(-1)  # [N, T, 1]
    return ObservedState(**state, y=y, mu=mu)


def ordinal(
    state: PredictorState,
    num_classes: PositiveInt,
    start: float = -2.0,
    end: float = 2.0,
) -> ObservedState:
    """
    Sample ordinal response via cumulative logit model.

    Parameters
    ----------
    state : PredictorState
        Must contain ``eta`` [N, T, 1].
    num_classes : int
        Number of ordinal categories K.
    start : float
        Lower bound for evenly spaced thresholds.
    end : float
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
    >>> from simulacra import linear_predictor, ordinal
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


def binomial(
    state: PredictorState, num_trials: PositiveInt, prevalence: UnitInterval = 0.5
) -> ObservedState:
    """
    Sample binomial response from linear predictor (logit link).

    Parameters
    ----------
    state : PredictorState
        Must contain ``eta`` [N, T, 1].
    num_trials : int
        Number of Bernoulli trials per observation.
    prevalence : float
        Base rate used to shift eta before applying sigmoid.

    Returns
    -------
    ObservedState
        Adds ``y`` [N, T, 1], ``mu`` [N, T, 1].

    See Also
    --------
    bernoulli : Special case with num_trials=1.
    multinomial : Multivariate generalization.

    Notes
    -----
    .. math:: \\mu = \\sigma(\\eta + \\text{logit}(p)), \\quad
              y \\sim \\text{Binomial}(n, \\mu)

    Examples
    --------
    >>> from simulacra import linear_predictor, binomial
    >>> state = linear_predictor(2, 3, 4)
    >>> result = binomial(state, num_trials=10, prevalence=0.3)
    >>> result["y"].shape
    torch.Size([2, 3, 1])
    """
    shift = torch.logit(torch.tensor(prevalence))
    mu = torch.sigmoid(state["eta"] + shift)  # [N, T, 1]
    y = dist.Binomial(num_trials, mu).sample()  # [N, T, 1]
    return ObservedState(**state, y=y, mu=mu)


def multinomial(state: PredictorState, num_trials: PositiveInt) -> ObservedState:
    """
    Sample multinomial response from linear predictor (softmax link).

    Parameters
    ----------
    state : PredictorState
        Must contain ``eta`` [N, T, K] with K > 1 classes.
    num_trials : int
        Number of draws per observation.

    Returns
    -------
    ObservedState
        Adds ``y`` [N, T, K], ``mu`` [N, T, K].

    See Also
    --------
    categorical : Special case with num_trials=1 returning class index.
    binomial : Univariate generalization of Bernoulli.

    Notes
    -----
    .. math:: \\mu = \\text{softmax}(\\eta), \\quad
              y \\sim \\text{Multinomial}(n, \\mu)

    Examples
    --------
    >>> from simulacra import linear_predictor, linear, multinomial
    >>> state = linear(linear_predictor(2, 3, 4), out_features=5)
    >>> result = multinomial(state, num_trials=10)
    >>> result["y"].shape
    torch.Size([2, 3, 5])
    """
    mu = torch.softmax(state["eta"], dim=-1)  # [N, T, K]
    y = dist.Multinomial(num_trials, probs=mu).sample()  # [N, T, K]
    return ObservedState(**state, y=y, mu=mu)


def zero_inflated_poisson(
    state: PredictorState, inflation: UnitInterval
) -> ObservedState:
    """
    Sample zero-inflated Poisson response (log link).

    Parameters
    ----------
    state : PredictorState
        Must contain ``eta`` [N, T, 1].
    inflation : float
        Probability of structural zero, in [0, 1).

    Returns
    -------
    ObservedState
        Adds ``y`` [N, T, 1], ``mu`` [N, T, 1].

    See Also
    --------
    poisson : Non-inflated counterpart.
    zero_inflated_negative_binomial : Overdispersed zero-inflated counts.

    Notes
    -----
    .. math:: P(Y=0) = \\pi + (1 - \\pi) e^{-\\mu}, \\quad
              P(Y=y) = (1 - \\pi) \\, \\text{Poisson}(y; \\mu), \\; y > 0

    where :math:`\\mu = \\exp(\\eta)` is the base distribution mean.

    Examples
    --------
    >>> from simulacra import linear_predictor, zero_inflated_poisson
    >>> state = linear_predictor(2, 3, 4)
    >>> result = zero_inflated_poisson(state, inflation=0.3)
    >>> result["y"].shape
    torch.Size([2, 3, 1])
    """
    mu = torch.exp(state["eta"])  # [N, T, 1]
    mask = torch.bernoulli(torch.full_like(mu, 1 - inflation))
    y = mask * torch.poisson(mu)  # [N, T, 1]
    return ObservedState(**state, y=y, mu=mu)


def negative_binomial(
    state: PredictorState, concentration: PositiveFloat
) -> ObservedState:
    """
    Sample negative binomial response from linear predictor (log link).

    Parameters
    ----------
    state : PredictorState
        Must contain ``eta`` [N, T, 1].
    concentration : float
        Dispersion parameter (larger values approach Poisson).

    Returns
    -------
    ObservedState
        Adds ``y`` [N, T, 1], ``mu`` [N, T, 1].

    See Also
    --------
    poisson : Limiting case as concentration approaches infinity.

    Notes
    -----
    .. math:: \\mu = \\exp(\\eta), \\quad
              p = \\frac{\\mu}{\\mu + r}, \\quad
              y \\sim \\text{NegBin}(r, p)

    Examples
    --------
    >>> from simulacra import linear_predictor, negative_binomial
    >>> state = linear_predictor(2, 3, 4)
    >>> result = negative_binomial(state, concentration=2.0)
    >>> result["y"].shape
    torch.Size([2, 3, 1])
    """
    mu = torch.exp(state["eta"])  # [N, T, 1]
    probs = (mu / (mu + concentration)).clamp(max=1 - 1e-6)  # [N, T, 1]
    y = dist.NegativeBinomial(concentration, probs=probs).sample()  # [N, T, 1]
    return ObservedState(**state, y=y, mu=mu)


def zero_inflated_negative_binomial(
    state: PredictorState, inflation: UnitInterval, concentration: PositiveFloat
) -> ObservedState:
    """
    Sample zero-inflated negative binomial response (log link).

    Parameters
    ----------
    state : PredictorState
        Must contain ``eta`` [N, T, 1].
    inflation : float
        Probability of structural zero, in [0, 1).
    concentration : float
        Dispersion parameter (larger values approach zero-inflated Poisson).

    Returns
    -------
    ObservedState
        Adds ``y`` [N, T, 1], ``mu`` [N, T, 1].

    See Also
    --------
    negative_binomial : Non-inflated counterpart.
    zero_inflated_poisson : Equidispersed zero-inflated counts.

    Notes
    -----
    .. math:: P(Y=0) = \\pi + (1 - \\pi) \\left(\\frac{r}{r + \\mu}\\right)^r,
              \\quad
              P(Y=y) = (1 - \\pi) \\, \\text{NegBin}(y; r, p), \\; y > 0

    where :math:`\\mu = \\exp(\\eta)` is the base distribution mean.

    Examples
    --------
    >>> from simulacra import linear_predictor, zero_inflated_negative_binomial
    >>> state = linear_predictor(2, 3, 4)
    >>> result = zero_inflated_negative_binomial(
    ...     state, inflation=0.3, concentration=2.0
    ... )
    >>> result["y"].shape
    torch.Size([2, 3, 1])
    """
    mu = torch.exp(state["eta"])  # [N, T, 1]
    mask = torch.bernoulli(torch.full_like(mu, 1 - inflation))
    probs = (mu / (mu + concentration)).clamp(max=1 - 1e-6)  # [N, T, 1]
    y = mask * dist.NegativeBinomial(concentration, probs=probs).sample()  # [N, T, 1]
    return ObservedState(**state, y=y, mu=mu)


def gamma_response(
    state: PredictorState, concentration: PositiveFloat
) -> ObservedState:
    """
    Sample Gamma response from linear predictor (log link).

    Parameters
    ----------
    state : PredictorState
        Must contain ``eta`` [N, T, 1].
    concentration : float
        Shape parameter of the Gamma distribution.

    Returns
    -------
    ObservedState
        Adds ``y`` [N, T, 1], ``mu`` [N, T, 1].

    See Also
    --------
    log_normal : Alternative for positive continuous data.
    poisson : Count data with log link.

    Notes
    -----
    .. math:: \\mu = \\exp(\\eta), \\quad
              \\text{rate} = \\frac{\\alpha}{\\mu}, \\quad
              y \\sim \\text{Gamma}(\\alpha, \\text{rate})

    Named ``gamma_response`` to avoid collision with the random-effects
    coefficient key ``gamma`` in state dictionaries.

    Examples
    --------
    >>> from simulacra import linear_predictor, gamma_response
    >>> state = linear_predictor(2, 3, 4)
    >>> result = gamma_response(state, concentration=2.0)
    >>> result["y"].shape
    torch.Size([2, 3, 1])
    """
    mu = torch.exp(state["eta"])  # [N, T, 1]
    rate = concentration / mu  # [N, T, 1]
    y = dist.Gamma(concentration, rate).sample()  # [N, T, 1]
    return ObservedState(**state, y=y, mu=mu)


def beta_response(state: PredictorState, precision: PositiveFloat) -> ObservedState:
    """
    Sample Beta response from linear predictor (logit link).

    Parameters
    ----------
    state : PredictorState
        Must contain ``eta`` [N, T, 1].
    precision : float
        Precision parameter phi; larger values concentrate y around mu.

    Returns
    -------
    ObservedState
        Adds ``y`` [N, T, 1], ``mu`` [N, T, 1].

    See Also
    --------
    bernoulli : Binary special case of the logit link.

    Notes
    -----
    .. math:: \\mu = \\sigma(\\eta), \\quad
              \\alpha = \\mu \\phi, \\quad
              \\beta = (1 - \\mu) \\phi, \\quad
              y \\sim \\text{Beta}(\\alpha, \\beta)

    Named ``beta_response`` to avoid collision with the fixed-effects
    coefficient key ``beta`` in state dictionaries.

    Examples
    --------
    >>> from simulacra import linear_predictor, beta_response
    >>> state = linear_predictor(2, 3, 4)
    >>> result = beta_response(state, precision=5.0)
    >>> result["y"].shape
    torch.Size([2, 3, 1])
    """
    eps = 1e-6
    mu = torch.sigmoid(state["eta"]).clamp(eps, 1 - eps)  # [N, T, 1]
    alpha = mu * precision  # [N, T, 1]
    beta_param = (1 - mu) * precision  # [N, T, 1]
    y = dist.Beta(alpha, beta_param).sample()  # [N, T, 1]
    return ObservedState(**state, y=y, mu=mu)


def log_normal(state: PredictorState, std: PositiveFloat) -> ObservedState:
    """
    Sample log-normal response from linear predictor (log link).

    Parameters
    ----------
    state : PredictorState
        Must contain ``eta`` [N, T, 1].
    std : float
        Standard deviation of the underlying normal distribution.

    Returns
    -------
    ObservedState
        Adds ``y`` [N, T, 1], ``mu`` [N, T, 1].

    See Also
    --------
    gaussian : Normal response with identity link.
    gamma_response : Alternative for positive continuous data.

    Notes
    -----
    .. math:: y \\sim \\text{LogNormal}(\\eta, \\sigma), \\quad
              \\mu = \\exp(\\eta + \\sigma^2 / 2)

    Examples
    --------
    >>> from simulacra import linear_predictor, log_normal
    >>> state = linear_predictor(2, 3, 4)
    >>> result = log_normal(state, std=0.5)
    >>> result["y"].shape
    torch.Size([2, 3, 1])
    """
    eta = state["eta"]  # [N, T, 1]
    mu = torch.exp(eta + std**2 / 2)  # [N, T, 1]
    y = torch.exp(eta + std * randn(eta.shape))  # [N, T, 1]
    return ObservedState(**state, y=y, mu=mu)


# ---------------------------------------------------------------------------
# Predictor transforms (pre-family)
# ---------------------------------------------------------------------------


def random_effects(
    state: PredictorState,
    std: Sequence[PositiveFloat] | Tensor | PositiveFloat,
    correlation: Tensor | Correlation = 0.0,
    Z: Tensor | None = None,
    gamma: Tensor | None = None,
) -> PredictorState:
    """
    Add random effects Z*gamma to the linear predictor.

    Upgrades eta = X*beta to eta = X*beta + Z*gamma where Z is a
    Vandermonde (polynomial) basis by default.

    Chaining multiple calls concatenates gamma and Z along the q
    dimension, producing the composite random-effects tensors for
    all grouping factors. Each call contributes its own Z*gamma to
    eta independently, so there is no double-counting.

    Parameters
    ----------
    state : PredictorState
        Must contain ``eta`` [N, T, 1] and ``time`` [N, T, 1].
    std : Sequence[float] or Tensor or float
        Standard deviations for each random effect; length determines q.
    correlation : Tensor or float
        Off-diagonal correlation. Scalar gives compound symmetry;
        matrix gives a user-provided [q, q] correlation matrix.
    Z : Tensor or None
        Random-effects design matrix [N, T, q]; Vandermonde basis if omitted.
    gamma : Tensor or None
        Random-effects coefficients [N, q, 1]; sampled from MVN(0, Q)
        if omitted.

    Returns
    -------
    PredictorState
        Updates ``eta`` [N, T, 1] and adds ``gamma`` [N, q, 1],
        ``Z`` [N, T, q].

    See Also
    --------
    linear_predictor : Build the fixed-effects predictor.
    random_effects_covariance : Build the covariance Q = S R S.

    Notes
    -----
    Implements Laird & Ware (1982):

    .. math:: \\eta = X \\beta + Z \\gamma, \\quad \\gamma \\sim N(0, Q)

    References
    ----------
    .. [1] Laird, N. M., & Ware, J. H. (1982). Random-effects models for
       longitudinal data. *Biometrics*, 38(4), 963-974.

    Examples
    --------
    >>> from simulacra import linear_predictor, random_effects
    >>> state = random_effects(linear_predictor(2, 3, 4), std=[0.5, 1.0])
    >>> state["gamma"].shape
    torch.Size([2, 2, 1])

    Chaining two calls concatenates along q:

    >>> state = random_effects(linear_predictor(2, 3, 4), std=[0.5])
    >>> state = random_effects(state, std=[1.0, 0.2])
    >>> state["gamma"].shape
    torch.Size([2, 3, 1])
    """
    num_samples = state["eta"].shape[0]
    s = torch.atleast_1d(torch.as_tensor(std, dtype=torch.float32))
    q = len(s)
    if gamma is None:
        Q = random_effects_covariance(std, correlation)  # [q, q]
        mvn = dist.MultivariateNormal(torch.zeros(q), Q)
        gamma = mvn.sample((num_samples,)).unsqueeze(-1)  # [N, q, 1]
    if Z is None:
        t = state["time"].squeeze(-1)  # [N, T]
        t_centered = t - t.mean(dim=1, keepdim=True)  # [N, T]
        powers = arange(q, dtype=torch.float32)  # [q]
        Z = t_centered.unsqueeze(-1) ** powers  # [N, T, q]
    random_effect = torch.bmm(Z, gamma)  # [N, T, 1]
    result = state.copy()
    result["eta"] = state["eta"] + random_effect  # [N, T, 1]
    if has_random_effects(state):
        gamma = torch.cat([state["gamma"], gamma], dim=1)  # [N, q_prev+q_new, 1]
        Z = torch.cat([state["Z"], Z], dim=2)  # [N, T, q_prev+q_new]
    result["gamma"] = gamma
    result["Z"] = Z
    return result


def offset[T: PredictorState](state: T, log_exposure: Tensor) -> T:
    """
    Add a log-exposure offset to the linear predictor.

    Standard adjustment for rate modeling with count distributions:
    the offset enters the linear predictor additively so that
    ``exp(eta)`` models a rate rather than a raw count.

    Parameters
    ----------
    state : T
        Must contain ``eta``.
    log_exposure : Tensor
        Log-transformed exposure; broadcastable to ``eta``'s shape.

    Returns
    -------
    T
        Same state type with adjusted ``eta``.

    See Also
    --------
    activation : Nonlinear transform on eta.
    poisson : Count response with log link.
    negative_binomial : Overdispersed count response with log link.

    Notes
    -----
    .. math:: \\eta' = \\eta + \\log(\\text{exposure})

    Examples
    --------
    >>> import torch
    >>> from simulacra import linear_predictor, offset
    >>> exposure = torch.ones(2, 3, 1)
    >>> state = offset(linear_predictor(2, 3, 4), torch.log(exposure))
    >>> state["eta"].shape
    torch.Size([2, 3, 1])
    """
    result = state.copy()
    result["eta"] = state["eta"] + log_exposure
    return result


def activation[T: PredictorState](state: T, fn: Callable[[Tensor], Tensor]) -> T:
    """
    Apply a nonlinear activation to the linear predictor.

    Parameters
    ----------
    state : T
        Must contain ``eta``.
    fn : Callable[[Tensor], Tensor]
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
    >>> from simulacra import linear_predictor, activation
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
    """
    Apply a random linear projection to the linear predictor.

    Parameters
    ----------
    state : T
        Must contain ``eta`` [N, T, in_features].
    out_features : int
        Output dimension.
    weight : Tensor or None
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
    >>> from simulacra import linear_predictor, linear
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
    """
    Apply a single hidden-layer MLP: linear -> activation -> linear.

    Parameters
    ----------
    state : T
        Must contain ``eta`` [N, T, in_features].
    hidden_features : int
        Hidden layer dimension.
    fn : Callable[[Tensor], Tensor]
        Activation function between layers.
    out_features : int or None
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
    >>> from simulacra import linear_predictor, mlp
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
    state: ObservedState, vocab_size: PositiveInt, concentration: PositiveFloat = 1.0
) -> TokenizedState:
    """
    Tokenize the design matrix via Dirichlet-skewed softmax.

    Parameters
    ----------
    state : ObservedState
        Must contain ``X`` [N, T, p].
    vocab_size : int
        Number of discrete token classes K.
    concentration : float
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
    >>> from simulacra import linear_predictor, gaussian, tokens
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


def observation_time[T: PredictorState](
    state: T, shape: PositiveFloat, rate: PositiveFloat
) -> T:
    """
    Replace observation times with Gamma-distributed intervals.

    Parameters
    ----------
    state : T
        Must contain ``eta`` [N, T, ...].
    shape : float
        Gamma distribution shape parameter.
    rate : float
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
    >>> from simulacra import linear_predictor, observation_time
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
