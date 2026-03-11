from typing import Literal

import torch
import torch.distributions as dist
from torch import Tensor


def isotropic_covariance(size: int) -> Tensor:
    """Identity covariance Σ = I (no temporal correlation)."""
    return torch.eye(size)


def ar1_covariance(correlation: float, num_timepoints: int) -> Tensor:
    """AR(1) residual covariance Σ[j,k] = ρ^|j − k| (Fahrmeir et al., §7.1.3).

    Uses index-based spacing; for irregular time grids, pass actual time
    differences instead (see MIXED_MODEL_REFACTOR_NOTES.md, Invariant 3).

    Args:
        correlation: Autocorrelation parameter in (0, 1).
        num_timepoints: Number of time points (size of covariance matrix).
    """
    grid = torch.arange(num_timepoints, dtype=torch.float32)
    diff = torch.abs(grid.unsqueeze(0) - grid.unsqueeze(1))
    return correlation**diff


def lkj_covariance(concentration: float, size: int) -> Tensor:
    """Sample an unstructured correlation matrix from LKJ distribution.

    Args:
        concentration: Concentration parameter. concentration=1 gives uniform over correlation matrices.
        size: Dimension of the correlation matrix.
    """
    L = dist.LKJCholesky(size, concentration=concentration).sample()
    return L @ L.T


def make_random_effects_covariance(
    intercept_std: float, slope_std: float = 0.0, correlation: float = 0.0
) -> Tensor:
    """Build the random-effects covariance Q (Fahrmeir et al., Eq. 7.11).

    Constructs Q = [[τ₀², τ₀₁], [τ₀₁, τ₁²]] where τ₀₁ = ρ·τ₀·τ₁.

    Args:
        intercept_std: Standard deviation of random intercepts (τ₀).
        slope_std: Standard deviation of random slopes (τ₁).
        correlation: Correlation between intercepts and slopes (ρ).
    """
    if correlation != 0.0 and slope_std == 0.0:
        raise ValueError(
            "correlation requires slope_std > 0 (no slope to correlate with)"
        )
    if slope_std == 0.0:
        return torch.tensor([[intercept_std**2]])
    tau_01 = correlation * intercept_std * slope_std
    return torch.tensor([[intercept_std**2, tau_01], [tau_01, slope_std**2]])


def make_residual_covariance(
    num_timepoints: int,
    covariance_type: Literal["isotropic", "ar1", "lkj"] = "isotropic",
    correlation: float = 0.9,
    concentration: float = 1.0,
) -> Tensor:
    """Build the within-subject residual covariance Σ (Fahrmeir et al., Eq. 7.21).

    This constructs the per-subject error covariance structure, not the
    random-effects covariance Q.

    Args:
        num_timepoints: Number of time points (size of covariance matrix).
        covariance_type: Type of covariance structure.
        correlation: AR(1) autocorrelation parameter (used if covariance_type="ar1").
        concentration: LKJ concentration parameter (used if covariance_type="lkj").
    """
    if covariance_type == "isotropic":
        return isotropic_covariance(num_timepoints)
    elif covariance_type == "ar1":
        return ar1_covariance(correlation, num_timepoints)
    elif covariance_type == "lkj":
        return lkj_covariance(concentration, num_timepoints)
    else:
        raise ValueError(f"Unknown covariance_type: {covariance_type}")
