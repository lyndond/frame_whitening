from typing import Tuple, Optional

import numpy as np
import numpy.typing as npt
from scipy.linalg import fractional_matrix_power


def randcov2(
    cond_max: float = 3.0, l_max: float = 2
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return 2D covariance matrix with condition number less than cond_max and its cholesky factor"""
    Q, _ = np.linalg.qr(np.random.randn(2, 2))
    D = np.diag((np.random.uniform(1, cond_max), 1)) * np.random.rand() * l_max
    C = Q @ D @ Q.T
    L = np.linalg.cholesky(C)
    return C, L


def randcovn(
    N: int, cond_max: float = 3.0, l_max: float = 2
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return ND covariance matrix with condition number less than cond_max and its cholesky factor"""
    Q, _ = np.linalg.qr(np.random.randn(N, N))
    D = np.diag(np.arange(1, N + 1)) * cond_max * np.random.rand() * l_max
    C = Q @ D @ Q.T
    L = np.linalg.cholesky(C)
    return C, L


def normalize_cov(C: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    sigma = np.sqrt(np.diag(C))
    C_hat = C/np.outer(sigma, sigma)
    return C_hat


def psd_sqrt(C: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Computes PSD square root"""
    Csqrt = fractional_matrix_power(C, 0.5)
    return Csqrt


def sample_x(
    Lxx: npt.NDArray[np.float64], n_samples: int = 1, rng: Optional[np.random.Generator] = None
) -> npt.NDArray[np.float64]:
    """Takes cholesky L to colour n_samples of white noise"""
    if rng is None:
        rng = np.random.default_rng()
    n = Lxx.shape[0]
    return Lxx @ rng.standard_normal((n, n_samples))
