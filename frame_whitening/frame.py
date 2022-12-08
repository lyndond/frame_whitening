import itertools
from typing import Tuple, Optional, List

import numpy as np
import numpy.typing as npt
from scipy.linalg import fractional_matrix_power
import scipy.optimize


def normalize_frame(
    W: npt.NDArray[np.float64], axis: int = 0
) -> npt.NDArray[np.float64]:
    """Normalize the columns of W to unit length"""
    W0 = W / np.linalg.norm(W, axis=axis, keepdims=True)
    return W0


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
    assert n >= 1 and m >= 1
    assert n >= h and m >= w


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


def get_rand_untf(
    m: int, 
    n: int,
    rng: Optional[np.random.Generator] = None,
) -> npt.NDArray[np.float64]:
    """Generate random unit norm tight frame of size mxn."""

    if rng is None:
        rng = np.random.default_rng()

    # generate random tight frame
    D = rng.standard_normal((m, n))
    Q, _ = np.linalg.qr(D.T)
    D = Q.T * np.sqrt(n/m)  # rows have correct norm

    atom_norms = np.sum(D*D, axis=0)
    for i in range(n-1):
        if atom_norms[i] != 1:  # do not normalize if already normalized
            s1 = np.sign(atom_norms[i] - 1)
            j = i + 1
            while np.sign(atom_norms[j]-1) == s1:  # find atom w norm on other side of 1
                j = j + 1

            #compute tangent of rot angle
            an1 = atom_norms[i]
            an2 = atom_norms[j]
            cp = D[:, i].T @ D[:, j]
            t = (cp + np.sign(cp)*np.sqrt(cp*cp - (an1-1)*(an2-1))) / (an2-1)
            # compute rot
            c = 1 / np.sqrt(1+t*t)
            s = t*c
            # new atoms and updated norm
            D[:, [i, j]] = D[:, [i, j]] @ np.array([[c, s], [-s, c]])
            atom_norms[j] = an1 + an2 - 1

    return D


def get_grassmannian(
    m: int, 
    n: int, 
    rng: Optional[np.random.Generator] = None,
    method: str = "highs",
) -> Tuple[npt.NDArray[np.float64], List[float]]:
    """Generates an m x n Grassmannian frame."""
    if rng is None:
        rng = np.random.default_rng()

    F = get_rand_untf(m, n, rng)

    obj = lambda M: np.max(np.abs(M.T @ M - np.eye(n)))
    obj_coh = []
    obj_coh.append(obj(F))

    for _ in range(100):
        tmp = obj(F)
        for k in range(n):
            ind = [i for i in range(n) if i != k]

            W = F[:, ind]
            y = F[:, k]
            W2 = np.concatenate((W.T, -W.T), axis=0)
            b0 = np.ones((2 * (n-1)))

            # solve linear program x = argmin_x y^T x s.t. W2 @ x <= 1
            x = scipy.optimize.linprog(-y, A_ub=W2, b_ub=b0, method=method)
            x = x.x
            F[:,k] = x / np.linalg.norm(x)
        
        tmp1 = obj(F)

        if np.abs(tmp-tmp1) / np.abs(tmp) < 1E-3:
            break
        obj_coh.append(tmp1)
    return F, obj_coh
