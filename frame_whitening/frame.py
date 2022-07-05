import numpy as np
from scipy.linalg import fractional_matrix_power
from typing import Tuple, Optional


def normalize_frame(W: np.ndarray, axis: int = 0) -> np.ndarray:
    """Normalize the columns of W to unit length"""
    W0 = W / np.linalg.norm(W, axis=axis, keepdims=True)
    return W0


def compute_g_opt(C: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Compute w using homog representation"""
    G2 = (W.T @ W) ** 2
    s = np.diag(W.T @ C @ W)
    g = np.linalg.solve(G2, s)
    return g


def get_mercedes_frame(parseval: bool = False, jitter: bool = False) -> np.ndarray:
    """Makes 2D Mercedes Benz frame,
    or equiangular 2D frame with k vectors, and optional added angular jitter, and optionally Parseval"""
    k = 3
    thetas = np.arange(0, 2 * np.pi, 2 * np.pi / k) + (
        jitter * np.random.rand() * np.pi
    )
    benz = np.stack([np.cos(thetas), np.sin(thetas)])
    basis = benz
    basis /= np.linalg.norm(basis, axis=0, keepdims=True)
    if parseval:
        basis *= np.sqrt(2 / k)  # make parseval
    return basis


def parsevalize(W: np.ndarray) -> np.ndarray:
    """Ellipsoidal Parseval transformation of arbitrary frame"""
    S = W @ W.T
    S12 = fractional_matrix_power(S, -0.5)
    R2 = S12 @ W  # turn parseval
    return R2


def squared_gram(W: np.ndarray) -> np.ndarray:
    gram = W.T @ W
    return gram**2


def frame_distance(W: np.ndarray, C1: np.ndarray, C2: np.ndarray) -> np.ndarray:
    """Computes d(C1, C2) = (w1-w2)' * (G * G) * (w1-w2) quadratic form distance in w-space for given R
    This is equivalent to computing the Frobenius norm between the two covariance matrices.
    """

    Gram2 = squared_gram(W)

    g1 = compute_g_opt(C1, W)
    g2 = compute_g_opt(C2, W)

    dist_sq = (g1 - g2) @ Gram2 @ (g1 - g2)
    dist = np.sqrt(dist_sq)
    return dist


def get_equiangular_3d() -> np.ndarray:
    """Returns six vectors in R3 whose pair-wise angles are equal
    Redmond Existence and construction of real-valued equiangular tight frames (2009) PhD Thesis
    """

    return np.array(
        [
            [1, 0, 0],
            [1 / np.sqrt(5), 2 / np.sqrt(5), 0],
            [1 / np.sqrt(5), 0.1 * (5 - np.sqrt(5)), np.sqrt(0.1 * (5 + np.sqrt(5)))],
            [1 / np.sqrt(5), 0.1 * (5 - np.sqrt(5)), -np.sqrt(0.1 * (5 + np.sqrt(5)))],
            [1 / np.sqrt(5), 0.1 * (-5 - np.sqrt(5)), np.sqrt(0.1 * (5 - np.sqrt(5)))],
            [1 / np.sqrt(5), 0.1 * (-5 - np.sqrt(5)), -np.sqrt(0.1 * (5 - np.sqrt(5)))],
        ]
    ).T  # / np.sqrt(2)


def get_rotation_matrix3d(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Returns a 3D rotation matrix for values alpha (x), beta (y), gamma (z) in radians."""
    yaw = np.array(
        [
            [np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1],
        ]
    )
    pitch = np.array(
        [[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]]
    )

    roll = np.array(
        [
            [1, 0, 0],
            [0, np.cos(gamma), -np.sin(gamma)],
            [0, np.sin(gamma), np.cos(gamma)],
        ]
    )
    return yaw @ pitch @ roll


def rot2(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def frame_svd(
    A: np.ndarray, X: np.ndarray = None, Z: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random decomposition of A into two frames and diagonal with all positive entries
    A = X @ diag(y) @ Z.T
    """

    n, m = A.shape
    nm = n * m

    if X is None:
        X = np.random.randn(n, nm)
        X = X / np.linalg.norm(X, axis=0, keepdims=True)
    if Z is None:
        Z = np.random.randn(m, nm)
        Z = Z / np.linalg.norm(Z, axis=0, keepdims=True)

    XX = X.T @ X
    ZZ = Z.T @ Z

    XZ = XX * ZZ  # element-wise

    s = np.diag(X.T @ A @ Z)

    y = np.linalg.solve(XZ, s)

    neg_ind = np.argwhere(y < np.zeros(1))

    # ensure y is all positive by flipping sign of X
    y = np.abs(y)
    X[:, neg_ind] *= -1.0

    return X, y, Z


def get_grassmannian(
    n: int,
    m: int,
    niter: int = 100,
    fract_shrink: float = 0.8,
    shrink_fact: float = 0.9,
    A_init: Optional[np.ndarray] = None,
    expand: bool = False,
):
    """Sample a tight frame with minimal mutual coherence, ie. the angle
    between any pair of column is the same, and the smallest it can be.

    Approximate iterative algorithm to sample grassmannian mtx (tightest frame,
    smallest possible mutual coherence).

    Parameters
    ----------
    n: Dimension of the ambient space
    m: Number of vectors
    niter: Number of iterations to run.
    fract_shrink: TODO(lyndo).
    shrink_fact: TODO(lyndo).
    A_init: Initial matrix (optional).

    Returns
    -------
    W : Tight frame of shape [n, m], with normalized columns
    G : Gram matrix.
    Res : 'optimal mu', 'mean mu', 'obtained mu'.

    Notes
    -----
    From Pierre-Etienne Fiquet May 17 2022
    """
    # assert m <= (np.minimum(n * (n + 1) // 2, (m - n) * (m - n + 1) // 2))

    if A_init is None:
        W = np.random.randn(n, m)
    else:
        assert (n, m) == A_init.shape
        W = A_init

    # normalize columns
    W = normalize_frame(W)
    G = W.T @ W
    if m > n:
        mu = np.sqrt((m - n) / n / (m - 1))

    Res = np.zeros((niter, 3))
    for i in range(niter):
        # 1- shrink high inner products
        gg = np.sort(np.abs(G).flatten())
        idx, idy = np.where(
            (np.abs(G) > gg[int(fract_shrink * (m**2 - m))]) & (np.abs(G - 1) > 1e-6)
        )
        G[idx, idy] *= shrink_fact

        # 1b- expand near 0 products
        if expand:
            idx, idy = np.where(
                (np.abs(G) < gg[int((1 - fract_shrink) * (m**2 - m))])
            )
            G[idx, idy] /= shrink_fact

        # 2- reduce rank back to n
        U, s, Vh = np.linalg.svd(G)
        s[n:] *= 0
        G = U @ np.diag(s) @ Vh

        # 3- normalize cols
        G = np.diag(1 / np.sqrt(np.diag(G))) @ G @ np.diag(1 / np.sqrt(np.diag(G)))

        # status
        gg = np.sort(np.abs(G).flatten())
        idx, idy = np.where(
            (np.abs(G) > gg[int(fract_shrink * (m**2 - m))]) & (np.abs(G - 1) > 1e-6)
        )
        GG = np.abs(G[idx, idy])
        g_shape = GG.shape[0]
        Res[i, :] = [mu, np.mean(GG), np.max(GG - np.eye(g_shape))]

    U, s, Vh = np.linalg.svd(G)
    W = np.diag(np.sqrt(s[:n])) @ U[:, :n].T
    W = normalize_frame(W)
    return W, G, Res
