from collections.abc import Sequence
from typing import Tuple, Optional, List

import numpy as np
import numpy.typing as npt
import scipy as sp
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
    seed: Optional[int] = None,
    verbose: bool = True,
    break_on_convergence: bool = True,
    break_on_divergence: bool = True,
    error_type: str = "fro",
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
    online : If True, run in online mode. If False, run in offline mode.
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

    rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)
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
                pbar.update(1)  # type: ignore

            WGW = W @ (g[:, None] * W.T)  # equiv to W@diag(g)@W.T
            M = np.linalg.inv(alpha*Ixx + WGW)

            if online: # run simulation with minibatches

                # draw a sample of x
                x = Lxx @ rng.standard_normal((N, batch_size))
                y = np.linalg.inv(Ixx + WGW) @ x
                z = W.T @ y
                variances = z**2
                dg = variances - 1
                g = g + lr_g * np.mean(dg, -1)  # stochastic gradient ascent

            else:  # offline mode
                Cyy = M @ Cxx @ M.T

                # compute diag(W.T@Cyy@W) efficiently
                tmp = Cyy @ W
                variances = np.array([w @ t for w, t in zip(W.T, tmp.T)])
                dg = variances - 1
                g = g + lr_g * dg  # gradient ascent

            # project g to be positive
            g = np.clip(g, 0, np.inf) if clamp else g
            if step % save_every == 0:
                # error = np.linalg.norm(M @ Cxx @ M.T - Ixx) ** 2 / N**2
                error = compute_error(M, Cxx, clamp, error_type)
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


def compute_error(
        M: npt.NDArray[np.float64], 
        Cxx: npt.NDArray[np.float64], 
        clamp: bool, 
        error_type: str = 'fro'
    ) -> float:
    assert error_type in ['fro', 'spectral', 'operator']
    Cyy = M @ Cxx @ M.T
    N = Cxx.shape[0]

    if error_type == 'fro':
        err_diag = np.diag(Cyy) - 1
        err_off_diag = Cyy[np.triu_indices_from(Cyy, k=1)] 
        error = 1/(N**2) * (np.sum(err_diag ** 2) + 2 * np.sum(err_off_diag ** 2))
    elif error_type == 'spectral':
        eigvals = np.linalg.eigvalsh(Cyy)
        diff = eigvals - 1
        diff = np.clip(diff, 0, np.inf) if clamp else diff
        error = 1/N * np.sum(diff**2)
    else:  # operator norm
        eigvals = np.linalg.eigvalsh(Cyy)
        opnorm = np.max(eigvals)
        error = opnorm - 1
        
    return error


def get_g_opt(
    W: npt.NDArray[np.float64], 
    Cxx: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute optimal G."""
    N, K = W.shape
    # assert K == N * (N + 1) // 2, "W must have K = N(N+1)/2 columns."
    Ixx = np.eye(N)
    gram_sq_inv = np.linalg.inv((W.T @ W) ** 2)
    Cxx_12 = sp.linalg.sqrtm(Cxx)
    g_opt = gram_sq_inv @ np.diag(W.T @ (Cxx_12 - Ixx) @ W)
    return g_opt
