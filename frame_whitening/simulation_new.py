import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional, List
import scipy as sp

from frame_whitening import stats


def simulate(
    cholesky_list: List[npt.NDArray[np.float64]],
    W: npt.NDArray[np.float64],
    batch_size: int = 64,
    n_batch: int = 1024,
    lr_g: float = 5e-3,
    g0: Optional[npt.NDArray[np.float64]] = None,
    online: bool = True,
    clamp: bool = False,
    seed: Optional[float] = None,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Simulate data from a given model."""
    if seed is not None:
        np.random.seed(seed)

    # initialize random set of gains g
    N, K = W.shape
    g = g0 if g0 is not None else np.ones(K)

    g_all = []
    g_last = []
    errors = []
    variances_all = []

    # run simulation with minibatches
    Ixx = np.eye(N)
    for Lxx in cholesky_list:
        Cxx = Lxx @ Lxx.T
        for _ in range(n_batch):
            G = np.diag(g)
            WGW = W @ G @ W.T
            M = np.linalg.inv(Ixx + WGW)

            if online:
                x = stats.sample_x(Lxx, batch_size)  # draw a sample of x
                y = np.linalg.inv(Ixx + WGW) @ x
                z = W.T @ y
                variances = z**2
                dg = z**2 - 1
                g = g + lr_g * np.mean(dg, -1)  # gradient descent
            else:
                Cyy = M @ Cxx @ M.T
                Czz = W.T @ Cyy @ W
                variances = np.diag(Czz)
                dg = variances - 1
                g = g + lr_g * dg  # gradient descent

            # project g to be positive
            g = np.clip(g, 0, np.inf) if clamp else g
            error = np.linalg.norm(M @ Cxx @ M.T - Ixx) ** 2 / N**2
            errors.append(error)
            g_all.append(g)
        variances_all.append(variances)
        g_last.append(g)

    g_last = np.stack(g_last, 0)
    g_all = np.stack(g_all, 0)
    return g_last, g_all, errors, variances  # type: ignore


def get_g_opt(W, Cxx):
    N, K = W.shape
    Ixx = np.eye(N)
    gram_sq_inv = np.linalg.inv((W.T @ W) ** 2)
    Cxx_12 = sp.linalg.sqrtm(Cxx)
    g_opt = gram_sq_inv @ np.diag(W.T @ (Cxx_12 - Ixx) @ W)
    return g_opt
