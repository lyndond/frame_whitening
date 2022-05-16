import numpy as np
from typing import Tuple
from scipy.linalg import fractional_matrix_power


def randcov2(cond_max: float = 3.0, l_max: float = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Return 2D covariance matrix with condition number less than cond_max and its cholesky factor"""
    Q, _ = np.linalg.qr(np.random.randn(2, 2))
    D = np.diag((np.random.uniform(1, cond_max), 1)) * np.random.rand() * l_max
    C = Q @ D @ Q.T
    L = np.linalg.cholesky(C)
    return C, L


def geometric_mean(C1: np.ndarray, C2: np.ndarray, t: float = 1 / 2) -> np.ndarray:
    # t-distance geodesic matrix between two PSD matrices
    raise NotImplementedError
    assert 0 <= t <= 1


def psd_sqrt(C):
    """Computes PSD square root"""
    Csqrt = fractional_matrix_power(C, 0.5)
    return Csqrt


def sample_x(Lxx: np.ndarray, n_samples: int = 1) -> np.ndarray:
    """Takes cholesky L to colour n_samples of white noise"""
    n = Lxx.shape[0]
    return Lxx @ np.random.randn(n, n_samples)


def orthogonal2(th: float) -> np.ndarray:
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return rot
