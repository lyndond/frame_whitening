import numpy as np
from scipy.linalg import fractional_matrix_power
from typing import Tuple
import matplotlib.pyplot as plt
from tqdm import trange


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


def get_rotation_matrix3d(alpha: float, beta: float, gamma: float):
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


def get_near_etf(
    n: int, k: int, s=None, thresh=0.5, tight_iter=0, num_iter=None, show=False
):
    """Generates nearly equiangular tight frame (Tsiligianii et al. 2017)
    Parameters
    ----------
    n: int
        Dimensionality of vector space
    k: int
        Number of frame vectors
    s: np.ndarray, optional
        Target eigenspectrum of signature matrix of frame. Should have two unique values.
    thresh: float, optional
        Used to slow the alternating projection algorithm, leads to better result when used. Must be +ve.
    tight_iter: int, optional
        Choose whether or not to take additional step to turn frame into tight frame using canonical elllipsoidal
        transform. Must be >=0.
    num_iter: int, optional
        Number of iterations the equiangular signature matrix step will take.
    show: bool
        Whether or not to show the signature matrix spectrum and histogram of absolute correlations of frame.
    Returns
    -------
    R: np.ndarray
        Nearly tight, nearly equiangular frame with Size(n,k)
    """
    if num_iter is None:
        num_iter = 2 * k

    if s is None:
        s = np.ones(k)
        s[: k // 2] = 10
        s[k // 2 :] = -10

    Sigma = np.diag(s)  # target spectrum

    welch_bound = np.sqrt((k - n) / (n * (k - 1)))
    R0 = np.random.randn(n, k)
    G = R0.T @ R0
    Qk = np.sign(G - np.diag(np.diag(G)))  # signature matrix
    if show:
        Q0 = Qk.copy()

    for _ in trange(num_iter, mininterval=1, desc="Qk"):
        _, P = np.linalg.eigh(Qk)
        Qk = P @ Sigma @ P.T  # replace with desired spectrum
        Qk = 1 / 2 * (Qk.T + Qk)  # symmetrize

        # separate diagonal and offdiagonal
        diag = np.diag(Qk).copy()
        off = Qk - np.diag(diag)

        # threshold diagonal to zero
        diag[np.abs(diag) < thresh] = 0
        Qk[np.diag_indices(k)] = diag

        # threshold offdiag to sign
        signum = np.abs(1 - np.abs(off)) < thresh
        Qk[signum] = np.sign(off[signum])

    # final step: turn into valid signature matrix
    Qk[np.diag_indices(k)] = 0
    Qk = np.sign(Qk)

    # reconstruct Gram matrix
    G = np.eye(k) + welch_bound * Qk
    U, s1, Vt = np.linalg.svd(G, hermitian=True)
    G = (
        U[:, :n] @ np.diag(s1[:n]) @ Vt[:n, :]
    )  # reduce rank to n so it is valid frame Gramian
    G = (
        np.diag(np.diag(G) ** -0.5) @ G @ np.diag(np.diag(G) ** -0.5)
    )  # normalize so it corresponds to unit norm frame

    # construct the frame
    _, s2, Vt = np.linalg.svd(G)  # symmetric
    S = np.zeros((n, k))
    np.fill_diagonal(S, np.sqrt(s2[:n]))
    R = S @ Vt

    # tighten the frame
    for _ in range(tight_iter):
        U, S, Vt = np.linalg.svd(R @ R.T)
        R = np.sqrt(k / n) * (U @ np.diag(S**-0.5) @ Vt) @ R
        R = R / np.linalg.norm(R, axis=0)

    if show:
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(np.linalg.eigh(Q0)[0], label="Init")
        ax[0].plot(np.linalg.eigh(Qk)[0], label="Converged")
        ax[0].plot(np.diag(Sigma)[::-1], "--k", label="Target")
        ax[0].set(title="Spectrum")
        ax[0].legend()

        ti = np.tril_indices(k, k=-1)
        abs_corr = (np.abs(R.T @ R))[ti]
        ax[1].hist(abs_corr, 100)
        ax[1].set(
            xlim=(0, 1),
            xlabel="Abs correlation",
            title=f"opt lower bound = {welch_bound:.2f}",
        )
        fig.tight_layout()

    return R


def frame_svd(
    A: np.ndarray, X: np.ndarray = None, Z: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random decomposition of A into two frames and diagonal with all positive entries"""

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

    XZ = XX * ZZ  # hadamard

    s = np.diag(X.T @ A @ Z)

    y = np.linalg.solve(XZ, s)

    neg_ind = np.argwhere(y < np.zeros(1))

    # ensure y is all positive by flipping sign of X
    y = np.abs(y)
    X[:, neg_ind] *= -1.0

    return X, y, Z


def get_grassmanian(
    n, m, niter=100, fract_shrink=0.8, shrink_fact=0.9, A_init=None, expand=False
):
    """Sample a tight frame with minimal mutual coherence, ie. the angle
    between any pair of column is the same, and the smallest it can be.

    Approximate iterative algorithm to sample grassmannian mtx (tightest frame,
    smallest possible mutual coherence).

    Parameters
    ----------
    n:
        dimension of the ambient space
    m:
        number of vectors
    niter:
    fract_shrink:
    shrink_fact:
    A_init:

    Returns
    -------
    W :
        tight frame of shape [n, m], with normalized columns
    G :
        Gram matrix
        G = A.T @ A
        abs(Gij) = sqrt((m-n)/n(m-1))
    Res :
        'optimal mu', 'mean mu', 'obtained mu'

    Notes
    -----
    From Pierre-Etienne Fiquet May 17 2022
    """
    assert m <= (np.minimum(n * (n + 1) / 2, (m - n) * (m - n + 1) / 2))

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
