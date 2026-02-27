from typing import Literal

import torch
import torch.distributions as dist
from torch import Tensor


def isotropic_covariance(size: int) -> Tensor:
    """Identity covariance matrix (no temporal correlation)."""
    return torch.eye(size)


def ar1_covariance(rho: float, num_timepoints: int) -> Tensor:
    """AR(1) covariance with potentially irregular spacing.

    Computes Σ[i,j] = ρ^|i - j|

    Args:
        rho: Autocorrelation parameter in (0, 1).
        num_timepoints: Number of time points (size of covariance matrix).
    """
    grid = torch.arange(num_timepoints, dtype=torch.float32)
    diff = torch.abs(grid.unsqueeze(0) - grid.unsqueeze(1))
    return rho**diff


def lkj_covariance(eta: float, size: int) -> Tensor:
    """Sample correlation matrix from LKJ distribution.

    Args:
        eta: Concentration parameter. eta=1 gives uniform over correlation matrices.
        size: Dimension of the correlation matrix.
    """
    L = dist.LKJCholesky(size, concentration=eta).sample()
    return L @ L.T


def make_covariance(
    num_timepoints: int,
    covariance_type: Literal["isotropic", "ar1", "lkj"] = "isotropic",
    rho: float = 0.9,
    eta: float = 1.0,
) -> Tensor:
    """Factory function for covariance matrices.

    Args:
        num_timepoints: Number of time points (size of covariance matrix).
        covariance_type: Type of covariance structure.
        rho: AR(1) autocorrelation parameter (used if covariance_type="ar1").
        eta: LKJ concentration parameter (used if covariance_type="lkj").
    """
    if covariance_type == "isotropic":
        return isotropic_covariance(num_timepoints)
    elif covariance_type == "ar1":
        return ar1_covariance(rho, num_timepoints)
    elif covariance_type == "lkj":
        return lkj_covariance(eta, num_timepoints)
    else:
        raise ValueError(f"Unknown covariance_type: {covariance_type}")
