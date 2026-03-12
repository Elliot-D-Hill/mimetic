from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import torch
import torch.distributions as dist
from torch import Tensor


@dataclass(frozen=True)
class IsotropicCovariance:
    """Identity residual covariance Σ = I."""


@dataclass(frozen=True)
class AR1Covariance:
    """AR(1) residual covariance specification."""

    correlation: float = 0.9


@dataclass(frozen=True)
class LKJCovariance:
    """Unstructured LKJ residual covariance specification."""

    concentration: float = 1.0


ResidualCovarianceSpec: TypeAlias = IsotropicCovariance | AR1Covariance | LKJCovariance


def isotropic_covariance(size: int) -> Tensor:
    """Identity covariance Σ = I (no temporal correlation)."""
    return torch.eye(size)  # [size, size]


def ar1_covariance(correlation: float, num_timepoints: int) -> Tensor:
    """AR(1) residual covariance Σ[j,k] = ρ^|j − k| (Fahrmeir et al., §7.1.3).

    Uses index-based spacing; for irregular time grids, pass actual time
    differences instead (see MIXED_MODEL_REFACTOR_NOTES.md, Invariant 3).

    Args:
        correlation: Autocorrelation parameter in (0, 1).
        num_timepoints: Number of time points (size of covariance matrix).
    """
    grid = torch.arange(num_timepoints, dtype=torch.float32)  # [T]
    diff = torch.abs(grid.unsqueeze(0) - grid.unsqueeze(1))  # [T, T]
    return correlation**diff  # [T, T]


def lkj_covariance(concentration: float, size: int) -> Tensor:
    """Sample an unstructured correlation matrix from LKJ distribution.

    Args:
        concentration: Concentration parameter. concentration=1 gives uniform over correlation matrices.
        size: Dimension of the correlation matrix.
    """
    L = dist.LKJCholesky(size, concentration=concentration).sample()  # [size, size]
    return L @ L.T  # [size, size]


def random_effects_covariance(
    stds: Sequence[float], correlation: Tensor | float = 0.0
) -> Tensor:
    """Build the random-effects covariance Q = S R S (Fahrmeir et al., Eq. 7.11).

    Args:
        stds: Standard deviations for each random effect; len determines q.
        correlation: Off-diagonal correlation. Float gives compound symmetry
            R = I(1−ρ) + J·ρ. Tensor gives a user-provided [q, q] correlation matrix.
    """
    q = len(stds)
    S = torch.diag(torch.tensor(stds, dtype=torch.float32))  # [q, q]
    if isinstance(correlation, Tensor):
        R = correlation  # [q, q]
    else:
        R = torch.eye(q) * (1 - correlation) + torch.full((q, q), correlation)  # [q, q]
    return S @ R @ S  # [q, q]


def residual_covariance(
    num_timepoints: int, covariance: ResidualCovarianceSpec | None = None
) -> Tensor:
    """Build the within-subject residual covariance Σ (Fahrmeir et al., Eq. 7.21).

    This constructs the per-subject error covariance structure, not the
    random-effects covariance Q.

    Args:
        num_timepoints: Number of time points (size of covariance matrix).
        covariance: Residual covariance specification. Defaults to isotropic.
    """
    if covariance is None or isinstance(covariance, IsotropicCovariance):
        return isotropic_covariance(num_timepoints)
    if isinstance(covariance, AR1Covariance):
        return ar1_covariance(covariance.correlation, num_timepoints)
    if isinstance(covariance, LKJCovariance):
        return lkj_covariance(covariance.concentration, num_timepoints)
    raise TypeError(f"Unsupported covariance specification: {type(covariance)!r}")
