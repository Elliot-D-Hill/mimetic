import pytest
import torch

from mimetic.covariance import (
    AR1Covariance,
    IsotropicCovariance,
    LKJCovariance,
    ar1_covariance,
    isotropic_covariance,
    lkj_covariance,
    random_effects_covariance,
    residual_covariance,
)

# ── isotropic_covariance ─────────────────────────────────────────────


def test_isotropic_covariance_is_identity() -> None:
    """Tests: isotropic_covariance(4) equals torch.eye(4).
    Why: canonical closed-form oracle for the simplest covariance builder.
    """
    assert torch.equal(isotropic_covariance(4), torch.eye(4))


def test_isotropic_covariance_single_timepoint() -> None:
    """Tests: T=1 returns [[1.0]].
    Why: degenerate scalar case must still return a 2-D tensor.
    """
    assert torch.equal(isotropic_covariance(1), torch.ones(1, 1))


# ── ar1_covariance ───────────────────────────────────────────────────


def test_ar1_covariance_closed_form() -> None:
    """Tests: rho=0.5, T=4 matches REPL-verified matrix.
    Why: closed-form oracle guards against regression in the exponent grid.
    """
    expected = torch.tensor(
        [
            [1.0000, 0.5000, 0.2500, 0.1250],
            [0.5000, 1.0000, 0.5000, 0.2500],
            [0.2500, 0.5000, 1.0000, 0.5000],
            [0.1250, 0.2500, 0.5000, 1.0000],
        ]
    )
    result = ar1_covariance(0.5, 4)
    assert torch.allclose(result, expected)


def test_ar1_covariance_symmetric() -> None:
    """Tests: Sigma == Sigma.T.
    Why: AR(1) matrices are symmetric by construction; ensures no index bugs.
    """
    sigma = ar1_covariance(0.7, 5)
    assert torch.equal(sigma, sigma.T)


def test_ar1_covariance_unit_diagonal() -> None:
    """Tests: diagonal entries are all 1.0.
    Why: rho^0 = 1 for any rho, so the diagonal must be ones.
    """
    sigma = ar1_covariance(0.8, 6)
    assert torch.equal(sigma.diag(), torch.ones(6))


def test_ar1_covariance_positive_semidefinite() -> None:
    """Tests: all eigenvalues >= 0.
    Why: covariance matrices must be PSD; catches sign errors.
    """
    sigma = ar1_covariance(0.6, 5)
    eigenvalues = torch.linalg.eigvalsh(sigma)
    assert (eigenvalues >= -1e-6).all()


def test_ar1_covariance_rho_zero_is_identity() -> None:
    """Tests: rho=0 produces identity.
    Why: degenerate case where all off-diagonals vanish (0^k=0 for k>0).
    """
    assert torch.equal(ar1_covariance(0.0, 4), torch.eye(4))


def test_ar1_covariance_rho_one_is_ones() -> None:
    """Tests: rho=1 produces all-ones matrix.
    Why: degenerate case where 1^|j-k|=1 everywhere.
    """
    assert torch.equal(ar1_covariance(1.0, 3), torch.ones(3, 3))


def test_ar1_covariance_single_timepoint() -> None:
    """Tests: T=1 returns [[1.0]] regardless of rho.
    Why: degenerate scalar dimension; rho is irrelevant.
    """
    assert torch.equal(ar1_covariance(0.9, 1), torch.ones(1, 1))


# ── lkj_covariance ──────────────────────────────────────────────────


def test_lkj_covariance_shape() -> None:
    """Tests: output shape is [T, T] for T=5.
    Why: shape contract for the stochastic builder.
    """
    torch.manual_seed(42)
    R = lkj_covariance(1.0, 5)
    assert R.shape[0] == 5
    assert R.shape[1] == 5


def test_lkj_covariance_symmetric() -> None:
    """Tests: R == R.T.
    Why: correlation matrices are symmetric; L@L.T should preserve this.
    """
    torch.manual_seed(42)
    R = lkj_covariance(1.0, 4)
    assert torch.allclose(R, R.T)


def test_lkj_covariance_unit_diagonal() -> None:
    """Tests: diagonal entries are all 1.0.
    Why: correlation matrices have unit diagonal by definition.
    """
    torch.manual_seed(42)
    R = lkj_covariance(1.0, 4)
    assert torch.allclose(R.diag(), torch.ones(4))


def test_lkj_covariance_positive_semidefinite() -> None:
    """Tests: all eigenvalues >= 0.
    Why: product L@L.T is PSD by construction; validates no numerical issues.
    """
    torch.manual_seed(42)
    R = lkj_covariance(1.0, 5)
    eigenvalues = torch.linalg.eigvalsh(R)
    assert (eigenvalues >= -1e-6).all()


def test_lkj_covariance_entries_bounded() -> None:
    """Tests: all entries in [-1, 1].
    Why: correlation matrix entries must lie in this range.
    """
    torch.manual_seed(42)
    R = lkj_covariance(1.0, 5)
    assert (R >= -1.0 - 1e-6).all()
    assert (R <= 1.0 + 1e-6).all()


def test_lkj_covariance_high_concentration_near_identity() -> None:
    """Tests: concentration=1000 produces off-diagonals near zero.
    Why: high concentration concentrates the LKJ distribution around identity.
    """
    torch.manual_seed(42)
    R = lkj_covariance(1000.0, 4)
    off_diag = R - torch.eye(4)
    assert torch.allclose(off_diag, torch.zeros(4, 4), atol=0.1)


# ── random_effects_covariance ───────────────────────────────────────


def test_random_effects_covariance_closed_form() -> None:
    """Tests: std=[2,3], corr=0.5 produces [[4,3],[3,9]].
    Why: REPL-verified Q = S R S closed-form oracle.
    """
    expected = torch.tensor([[4.0, 3.0], [3.0, 9.0]])
    result = random_effects_covariance([2.0, 3.0], correlation=0.5)
    assert torch.allclose(result, expected)


def test_random_effects_covariance_zero_correlation_diagonal() -> None:
    """Tests: std=[0.5,1.0], corr=0 produces diag(0.25, 1.0).
    Why: zero correlation makes R=I, so Q = S^2 is diagonal.
    """
    expected = torch.tensor([[0.25, 0.0], [0.0, 1.0]])
    result = random_effects_covariance([0.5, 1.0], correlation=0.0)
    assert torch.allclose(result, expected)


def test_random_effects_covariance_matrix_correlation() -> None:
    """Tests: std=[2,1], R=[[1,0.3],[0.3,1]] produces [[4,0.6],[0.6,1]].
    Why: user-supplied correlation matrix path.
    """
    R = torch.tensor([[1.0, 0.3], [0.3, 1.0]])
    expected = torch.tensor([[4.0, 0.6], [0.6, 1.0]])
    result = random_effects_covariance([2.0, 1.0], correlation=R)
    assert torch.allclose(result, expected)


def test_random_effects_covariance_symmetric() -> None:
    """Tests: Q == Q.T.
    Why: Q = S R S is symmetric when R is symmetric.
    """
    Q = random_effects_covariance([1.0, 2.0, 0.5], correlation=0.3)
    assert torch.allclose(Q, Q.T)


def test_random_effects_covariance_positive_semidefinite() -> None:
    """Tests: all eigenvalues >= 0.
    Why: covariance matrices must be PSD.
    """
    Q = random_effects_covariance([1.0, 2.0, 0.5], correlation=0.3)
    eigenvalues = torch.linalg.eigvalsh(Q)
    assert (eigenvalues >= -1e-6).all()


def test_random_effects_covariance_identity_std_is_R() -> None:
    """Tests: S=I (std=[1,1,1]) makes Q equal to R.
    Why: degenerate case where S drops out of Q = S R S.
    """
    Q = random_effects_covariance([1.0, 1.0, 1.0], correlation=0.4)
    q = 3
    R = torch.eye(q) * (1 - 0.4) + 0.4
    assert torch.allclose(Q, R)


def test_random_effects_covariance_scalar_std() -> None:
    """Tests: float std=2.0 produces [[4.0]].
    Why: degenerate single-effect case with scalar input.
    """
    Q = random_effects_covariance(2.0)
    assert torch.allclose(Q, torch.tensor([[4.0]]))


# ── residual_covariance ─────────────────────────────────────────────


def test_residual_covariance_none_is_identity() -> None:
    """Tests: covariance=None falls back to identity.
    Why: default/None path must produce isotropic covariance.
    """
    assert torch.equal(residual_covariance(3), torch.eye(3))


def test_residual_covariance_isotropic_is_identity() -> None:
    """Tests: IsotropicCovariance() dispatches to identity.
    Why: explicit isotropic spec must match the None default.
    """
    assert torch.equal(residual_covariance(3, IsotropicCovariance()), torch.eye(3))


def test_residual_covariance_ar1_matches_direct() -> None:
    """Tests: AR1Covariance(0.5) matches ar1_covariance(0.5, T).
    Why: dispatcher must pass through to the direct builder without alteration.
    """
    direct = ar1_covariance(0.5, 4)
    dispatched = residual_covariance(4, AR1Covariance(0.5))
    assert torch.equal(direct, dispatched)


def test_residual_covariance_unsupported_type_raises() -> None:
    """Tests: non-spec type raises TypeError.
    Why: error path for unsupported covariance specifications.
    """
    with pytest.raises(TypeError, match="Unsupported covariance"):
        residual_covariance(3, "invalid")  # type: ignore[arg-type]


# ── cross-cutting ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "spec",
    [None, IsotropicCovariance(), AR1Covariance(0.6), LKJCovariance(2.0)],
    ids=["none", "isotropic", "ar1", "lkj"],
)
def test_residual_covariance_always_square(spec: object) -> None:
    """Tests: all spec types produce [T, T] output.
    Why: shape contract must hold across all dispatch branches.
    """
    torch.manual_seed(99)
    T = 4
    result = residual_covariance(T, spec)  # type: ignore[arg-type]
    assert result.shape[0] == T
    assert result.shape[1] == T
