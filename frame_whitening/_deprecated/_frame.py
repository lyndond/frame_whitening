"""Dead code.
Note: This code has no guarantee of working and is only maintained here for posterity.
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


def orthogonal2(th: float) -> np.ndarray:
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return rot


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
