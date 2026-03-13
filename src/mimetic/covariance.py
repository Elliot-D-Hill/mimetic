from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import torch
import torch.distributions as dist
from torch import Tensor


@dataclass(frozen=True)
class IsotropicCovariance:
    """Identity residual covariance Sigma = I.

    See Also
    --------
    residual_covariance : Dispatch to structure-specific builders.
    isotropic_covariance : Build the matrix.
    """


@dataclass(frozen=True)
class AR1Covariance:
    """AR(1) residual covariance Sigma[j,k] = rho^|j-k|.

    Parameters
    ----------
    correlation
        Autocorrelation parameter in (0, 1).

    See Also
    --------
    residual_covariance : Dispatch to structure-specific builders.
    ar1_covariance : Build the matrix.
    """

    correlation: float = 0.9


@dataclass(frozen=True)
class LKJCovariance:
    """Unstructured correlation matrix sampled from an LKJ distribution.

    Parameters
    ----------
    concentration
        LKJ concentration; 1.0 gives uniform over correlation matrices.

    See Also
    --------
    residual_covariance : Dispatch to structure-specific builders.
    lkj_covariance : Build the matrix.
    """

    concentration: float = 1.0


ResidualCovarianceSpec: TypeAlias = IsotropicCovariance | AR1Covariance | LKJCovariance


def isotropic_covariance(num_timepoints: int) -> Tensor:
    """Build identity covariance Sigma = I (no temporal correlation).

    Parameters
    ----------
    num_timepoints
        Number of time points T.

    Returns
    -------
    Tensor
        Identity matrix [T, T].

    See Also
    --------
    ar1_covariance : Autoregressive structure.
    lkj_covariance : Unstructured correlation.

    Examples
    --------
    >>> from mimetic.covariance import isotropic_covariance
    >>> isotropic_covariance(3).shape
    torch.Size([3, 3])
    """
    return torch.eye(num_timepoints)  # [T, T]


def ar1_covariance(correlation: float, num_timepoints: int) -> Tensor:
    """Build AR(1) residual covariance Sigma[j,k] = rho^|j-k|.

    Parameters
    ----------
    correlation
        Autocorrelation parameter in (0, 1).
    num_timepoints
        Number of time points T.

    Returns
    -------
    Tensor
        Covariance matrix [T, T].

    See Also
    --------
    isotropic_covariance : Identity (rho = 0).
    lkj_covariance : Unstructured correlation.

    Notes
    -----
    Implements index-based spacing (Fahrmeir et al. [1]_, Section 7.1.3).
    For irregular time grids, pass actual time differences instead.

    References
    ----------
    .. [1] Fahrmeir, L., Kneib, T., Lang, S., & Marx, B. (2013).
       *Regression*. Springer.

    Examples
    --------
    >>> from mimetic.covariance import ar1_covariance
    >>> ar1_covariance(0.9, 3).shape
    torch.Size([3, 3])
    """
    grid = torch.arange(num_timepoints, dtype=torch.float32)  # [T]
    diff = torch.abs(grid.unsqueeze(0) - grid.unsqueeze(1))  # [T, T]
    return correlation**diff  # [T, T]


def lkj_covariance(concentration: float, num_timepoints: int) -> Tensor:
    """Sample an unstructured correlation matrix from the LKJ distribution.

    Parameters
    ----------
    concentration
        LKJ concentration; 1.0 gives uniform over correlation matrices.
    num_timepoints
        Dimension of the correlation matrix T.

    Returns
    -------
    Tensor
        Correlation matrix [T, T].

    See Also
    --------
    isotropic_covariance : Identity structure.
    ar1_covariance : Autoregressive structure.

    Examples
    --------
    >>> from mimetic.covariance import lkj_covariance
    >>> lkj_covariance(1.0, 3).shape
    torch.Size([3, 3])
    """
    lkj = dist.LKJCholesky(num_timepoints, concentration=concentration)
    L: Tensor = lkj.sample()  # [T, T]
    return L @ L.T  # [T, T]


def random_effects_covariance(
    std: Sequence[float] | Tensor | float, correlation: Tensor | float = 0.0
) -> Tensor:
    """Build the random-effects covariance Q = S R S.

    Parameters
    ----------
    std
        Standard deviations for each random effect; length determines q.
    correlation
        Off-diagonal correlation. Scalar gives compound symmetry
        R = I(1-rho) + J*rho; matrix gives a user-provided [q, q]
        correlation matrix.

    Returns
    -------
    Tensor
        Covariance matrix [q, q].

    See Also
    --------
    random_effects : Add U*gamma to the linear predictor using this covariance.

    Notes
    -----
    Implements Fahrmeir et al. [1]_, Eq. 7.11:

    .. math:: Q = S \\, R \\, S

    where S = diag(sigma_1, ..., sigma_q) and R is the correlation matrix.

    References
    ----------
    .. [1] Fahrmeir, L., Kneib, T., Lang, S., & Marx, B. (2013).
       *Regression*. Springer.

    Examples
    --------
    >>> from mimetic.covariance import random_effects_covariance
    >>> random_effects_covariance([0.5, 1.0]).shape
    torch.Size([2, 2])
    """
    s: Tensor = torch.atleast_1d(torch.as_tensor(std, dtype=torch.float32))
    q = s.shape[0]
    S = torch.diag(s)  # [q, q]
    correlation = torch.as_tensor(correlation, dtype=torch.float32)
    if correlation.ndim == 0:
        R = torch.eye(q) * (1 - correlation) + correlation  # [q, q]
    else:
        R = correlation  # [q, q]
    return S @ R @ S  # [q, q]


def residual_covariance(
    num_timepoints: int, covariance: ResidualCovarianceSpec | None = None
) -> Tensor:
    """Build the within-subject residual covariance Sigma.

    Dispatches to structure-specific builders based on the covariance
    specification.

    Parameters
    ----------
    num_timepoints
        Number of time points T.
    covariance
        Covariance specification; defaults to isotropic (identity).

    Returns
    -------
    Tensor
        Covariance matrix [T, T].

    See Also
    --------
    isotropic_covariance : Identity structure.
    ar1_covariance : Autoregressive structure.
    lkj_covariance : Unstructured correlation.

    Notes
    -----
    Constructs the per-subject error covariance Sigma (Fahrmeir et al. [1]_,
    Eq. 7.21), not the random-effects covariance Q.

    References
    ----------
    .. [1] Fahrmeir, L., Kneib, T., Lang, S., & Marx, B. (2013).
       *Regression*. Springer.

    Examples
    --------
    >>> from mimetic.covariance import residual_covariance, AR1Covariance
    >>> residual_covariance(3, AR1Covariance(0.8)).shape
    torch.Size([3, 3])
    """
    if covariance is None or isinstance(covariance, IsotropicCovariance):
        return isotropic_covariance(num_timepoints)
    if isinstance(covariance, AR1Covariance):
        return ar1_covariance(covariance.correlation, num_timepoints)
    if isinstance(covariance, LKJCovariance):
        return lkj_covariance(covariance.concentration, num_timepoints)
    raise TypeError(f"Unsupported covariance specification: {type(covariance)!r}")
