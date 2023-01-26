"""Anonymized code implementing frame whitening algorithms.

This simplified version of the algorithm assumes that the covariance is known.
"""

from collections.abc import Sequence
from typing import Tuple, Optional

import numpy as np
import numpy.typing as npt
import scipy.linalg
from tqdm import tqdm


def adapt_covariance(
    Cxx_list: Sequence[npt.NDArray[np.float64]],
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
    verbose: bool = True,
    break_on_convergence: bool = True,
    break_on_divergence: bool = True,
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Adapt the gains g to whiten the data with covariance matrix Cxx. 
    
    In offline mode, the covariance matrix is whitened directly. In online mode, its 
    Cholesky factor is computed, and drawn samples are used to whiten the data.

    Parameters
    ----------
    Cxx_list : Tuple or List of (N, N) covariances, with length `n_contexts`.
    W : (N, K) frame.
    batch_size : Size of batches to draw. Only valid in online mode.
    n_batch : Number of batches/steps run algorithm. Valid for both online and offline.
    lr_g : Learning rate (step size) for gains gradient ascent.
    g0 : Initial gains. If None, use ones.
    online : If True, run in online mode (True). If False, run in offline mode.
    clamp : If True, clamp gains to be non-negative, i.e. projected gradient ascent.
    alpha : Constant multiplier for Identity leak term in y dynamics. 
    save_every : Save gains, errors, etc. every `save_every` steps.
    seed: Random seed. Only valid in online mode when samples are being drawn.
    verbose: If True, show progress bar. 
    break_on_convergence: If True, breaks if converges.
    break_on_divergence: If True, breaks if diverges.

    Returns
    -------
    g_last : (n_contexts, K) gains. One set per Cxx in Cxx_list.
    g_all : (n_batch * n_contexts, K) gains at each step.
    errors : (n_batch * n_contexts,) errors at each step.
    variances : (n_batch * n_contexts, K) variances at each step.
    """

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

    n_contexts = len(Cxx_list)
    total_steps = n_batch * n_contexts

    if verbose:
        pbar = tqdm(total=total_steps)

    Ixx = np.eye(N)
    for Cxx in Cxx_list:
        if online:
            Lxx = np.linalg.cholesky(Cxx)

        for step in range(n_batch):
            if verbose:
                pbar.update(1)

            WGW = W @ (g[:, None] * W.T)  # equiv to W@diag(g)@W.T
            M = np.linalg.inv(alpha*Ixx + WGW)

            if online: # run simulation with minibatches
                x = sample_x(Lxx, batch_size)  # draw a sample of x
                y = np.linalg.inv(Ixx + WGW) @ x
                z = W.T @ y
                variances = z**2
                dg = z**2 - 1
                g = g + lr_g * np.mean(dg, -1)  # gradient ascent
            else:
                # compute diag(W.T@Cyy@W) efficiently
                Cyy = M @ Cxx @ M.T
                tmp = Cyy @ W
                variances = np.array([w @ t for w, t in zip(W.T, tmp.T)])
                dg = variances - 1
                g = g + lr_g * dg  # gradient ascent

            # project g to be positive
            g = np.clip(g, 0, np.inf) if clamp else g
            if step % save_every == 0:
                error = np.linalg.norm(M @ Cxx @ M.T - Ixx) ** 2 / N**2
                errors.append(error)
                if np.allclose(g_all[-1], g) and break_on_convergence:
                    if verbose:
                        pbar.set_description(f"Converged.")
                    break
                elif np.any(np.abs(g) > 200) and break_on_divergence:
                    if verbose:
                        pbar.set_description(f"Diverged.")
                    break
                else:
                    g_all.append(g)
        variances_all.append(variances)
        g_last.append(g)

    g_last = np.stack(g_last, 0)
    g_all = np.stack(g_all, 0)
    return g_last, g_all, errors, variances  # type: ignore


def sample_x(
    Lxx: npt.NDArray[np.float64], n_samples: int = 1
) -> npt.NDArray[np.float64]:
    """Takes cholesky L to colour n_samples of white noise"""
    n = Lxx.shape[0]
    return Lxx @ np.random.randn(n, n_samples)


def get_g_opt(
    W: npt.NDArray[np.float64], 
    Cxx: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute optimal G."""
    N, K = W.shape
    assert K == N * (N + 1) // 2, "W must have K = N(N+1)/2 columns."
    Ixx = np.eye(N)
    gram_sq_inv = np.linalg.inv((W.T @ W) ** 2)
    Cxx_12 = scipy.linalg.sqrtm(Cxx)
    g_opt = gram_sq_inv @ np.diag(W.T @ (Cxx_12 - Ixx) @ W)
    return g_opt
