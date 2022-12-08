from typing import Tuple, Optional, List

import numpy as np
import numpy.typing as npt
import scipy as sp
from tqdm import tqdm

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
    alpha: float = 1.,
    save_every: int = 1,
    seed: Optional[float] = None,
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Simulate data from a given model."""
    if seed is not None:
        np.random.seed(seed)
    assert alpha >= 0., "alpha must be non-negative"

    # initialize random set of gains g
    N, K = W.shape
    g = g0 if g0 is not None else np.ones(K)

    g_all = [g]
    g_last = []
    errors = []
    variances_all = []

    n_contexts = len(cholesky_list)
    total_steps = n_batch * n_contexts

    pbar = tqdm(total=total_steps)

    Ixx = np.eye(N)
    for Lxx in cholesky_list:
        Cxx = Lxx @ Lxx.T
        for step in range(n_batch):
            pbar.update(1)
            # G = np.diag(g)
            # WGW = W @ G @ W.T
            WGW = W @ (g[:, None] * W.T)  # equiv to W@diag(g)@W.T
            M = np.linalg.inv(alpha*Ixx + WGW)

            if online: # run simulation with minibatches
                x = stats.sample_x(Lxx, batch_size)  # draw a sample of x
                y = np.linalg.inv(Ixx + WGW) @ x
                z = W.T @ y
                variances = z**2
                dg = z**2 - 1
                g = g + lr_g * np.mean(dg, -1)  # gradient descent
            else:
                # compute diag(W.T@Cyy@W) efficiently
                Cyy = M @ Cxx @ M.T
                tmp = Cyy @ W
                # Czz = W.T @ Cyy @ W
                # variances = np.diag(Czz)
                variances = np.array([w @ t for w, t in zip(W.T, tmp.T)])
                dg = variances - 1
                g = g + lr_g * dg  # gradient descent

            # project g to be positive
            g = np.clip(g, 0, np.inf) if clamp else g
            if step % save_every == 0:
                error = np.linalg.norm(M @ Cxx @ M.T - Ixx) ** 2 / N**2
                errors.append(error)
                if np.allclose(g_all[-1], g):
                    pbar.set_description(f"Converged at step {step}")
                    break
                elif np.any(np.abs(g) > 200):
                    pbar.set_description(f"Diverged.")
                    break
                else:
                    g_all.append(g)
        variances_all.append(variances)
        g_last.append(g)

    g_last = np.stack(g_last, 0)
    g_all = np.stack(g_all, 0)
    return g_last, g_all, errors, variances  # type: ignore


def get_g_opt(
    W: npt.NDArray[np.float64], 
    Cxx: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute optimal G."""
    N, K = W.shape
    assert K == N * (N + 1) // 2, "W must have K = N(N+1)/2 columns."
    Ixx = np.eye(N)
    gram_sq_inv = np.linalg.inv((W.T @ W) ** 2)
    Cxx_12 = sp.linalg.sqrtm(Cxx)
    g_opt = gram_sq_inv @ np.diag(W.T @ (Cxx_12 - Ixx) @ W)
    return g_opt
