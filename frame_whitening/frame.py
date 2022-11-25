import itertools
from typing import Tuple, Optional

import numpy as np
import numpy.typing as npt
from scipy.linalg import fractional_matrix_power


def normalize_frame(
    W: npt.NDArray[np.float64], axis: int = 0
) -> npt.NDArray[np.float64]:
    """Normalize the columns of W to unit length"""
    W0 = W / np.linalg.norm(W, axis=axis, keepdims=True)
    return W0


def compute_g_opt(
    C: npt.NDArray[np.float64], W: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute w using homog representation"""
    G2 = (W.T @ W) ** 2
    s = np.diag(W.T @ C @ W)
    g = np.linalg.solve(G2, s)
    return g


def get_mercedes_frame(
    parseval: bool = False, jitter: bool = False
) -> npt.NDArray[np.float64]:
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


def parsevalize(W: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Ellipsoidal Parseval transformation of arbitrary frame"""
    S = W @ W.T
    S12 = fractional_matrix_power(S, -0.5)
    R2 = S12 @ W  # turn parseval
    return R2


def squared_gram(W: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    gram = W.T @ W
    return gram**2


def frame_distance(
    W: npt.NDArray[np.float64], C1: npt.NDArray[np.float64], C2: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Computes d(C1, C2) = (w1-w2)' * (G * G) * (w1-w2) quadratic form distance in w-space for given R
    This is equivalent to computing the Frobenius norm between the two covariance matrices.
    """

    Gram2 = squared_gram(W)

    g1 = compute_g_opt(C1, W)
    g2 = compute_g_opt(C2, W)

    dist_sq = (g1 - g2) @ Gram2 @ (g1 - g2)
    dist = np.sqrt(dist_sq)
    return dist


def get_equiangular_3d() -> npt.NDArray[np.float64]:
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


def get_rotation_matrix3d(alpha: float, beta: float, gamma: float) -> npt.NDArray:
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


def rot2(theta: float) -> npt.NDArray[np.float64]:
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def frame_svd(
    A: npt.NDArray[np.float64],
    X: Optional[npt.NDArray[np.float64]] = None,
    Z: Optional[npt.NDArray[np.float64]] = None,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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

    XX = X.T @ X  # type: ignore
    ZZ = Z.T @ Z  # type: ignore

    XZ = XX * ZZ  # element-wise

    s = np.diag(X.T @ A @ Z)  # type: ignore

    y = np.linalg.solve(XZ, s)

    neg_ind = np.argwhere(y < np.zeros(1))

    # ensure y is all positive by flipping sign of X
    y = np.abs(y)
    X[:, neg_ind] *= -1.0  # type: ignore

    return X, y, Z  # type: ignore


def get_conv_frame(n: int, m: int, h: int, w: int) -> npt.NDArray[np.float64]:
    """Returns a specific convolutional frame that connects all pairs within a window.

    Assumes the nxm image is vectorized, and the filter/neighbourhood size is hxw.

    Parameters
    ----------
    n: Height of image.
    m: Width of image.
    h: Height of convolutional filter.
    w: Width of convolutional filter.

    Returns
    -------
    W: Convolutional frame.
    """

    nm = n * m
    # hw = h * w
    
    # singly-connected interneurons
    WI = np.eye(nm)
    
    # doubly-connected (paired) interneurons
    idx_set = set()
    WP = []
    normalize = lambda x: x / np.linalg.norm(x)

    all_indices = list(itertools.product(range(n), range(m)))
    for i, j in all_indices:
        for k, l in all_indices:

            # get flat indices
            ij_flat = np.ravel_multi_index((i, j), (n, m))
            kl_flat = np.ravel_multi_index((k, l), (n, m))
            if (
                (i, j) != (k, l) and
                (ij_flat, kl_flat) not in idx_set and
                (kl_flat, ij_flat) not in idx_set and
                np.abs(i - k) < h and
                np.abs(j - l) < w
            ):
                idx_set.add((ij_flat, kl_flat))
                # pair (i, j) and (k, l) via flattened indices, and normalize
                wp = np.zeros(nm)
                wp[ij_flat] = 1
                wp[kl_flat] = 1
                WP.append(normalize(wp))

    WP = np.stack(WP, 0)
    
    W = np.concatenate([WI, WP], 0).T
    return W


def get_grassmannian(
    n: int,
    m: int,
    niter: int = 100,
    fract_shrink: float = 0.8,
    shrink_fact: float = 0.9,
    A_init: Optional[npt.NDArray[np.float64]] = None,
    expand: bool = False,
):
    """
    TODO(lyndo): replace this with new version.
    Sample a tight frame with minimal mutual coherence, ie. the angle
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
        Res[i, :] = [mu, np.mean(GG), np.max(GG - np.eye(g_shape))]  # type: ignore

    U, s, Vh = np.linalg.svd(G)
    W = np.diag(np.sqrt(s[:n])) @ U[:, :n].T
    W = normalize_frame(W)
    return W, G, Res
