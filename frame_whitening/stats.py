import numpy as np
import numpy.typing as npt
from typing import Tuple
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


def bures_dist(C1: npt.NDArray[np.float64], C2: npt.NDArray[np.float64]) -> float:
    """Computes Bures distance between two covariance matrices"""

    C2_sqrt = psd_sqrt(C2)
    d2 = np.trace(C1 + C2 - 2 * psd_sqrt(C2_sqrt @ C1 @ C2_sqrt))
    return np.sqrt(d2)


def bures_fidelity(C1: npt.NDArray[np.float64], C2: npt.NDArray[np.float64]) -> float:
    """arccos of sqrt fidelity gives an angle.
    https://en.wikipedia.org/wiki/Bures_metric#Bures_distance
    """
    C2_sqrt = psd_sqrt(C2)
    F = np.trace(psd_sqrt(C2_sqrt @ C1 @ C2_sqrt)) ** 2
    return F


def psd_sqrt(C: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Computes PSD square root"""
    Csqrt = fractional_matrix_power(C, 0.5)
    return Csqrt


def sample_x(
    Lxx: npt.NDArray[np.float64], n_samples: int = 1
) -> npt.NDArray[np.float64]:
    """Takes cholesky L to colour n_samples of white noise"""
    n = Lxx.shape[0]
    return Lxx @ np.random.randn(n, n_samples)
