import numpy as np
from typing import Tuple, Callable, Optional
from .types import *
from .stats import sample_x


def get_opt_funcs(func_type: str) -> Tuple[Callable, Callable]:
    if func_type == POLYNOMIAL:
        get_y = get_y_poly
        get_dg = get_dg_poly
    elif func_type == EXPONENTIAL:
        get_y = get_y_exp
        get_dg = get_dg_exp

    return get_y, get_dg


def get_y_poly(g, W, x, alpha=0):
    """Compute steady-state y for a given x and g and alpha
    f(g) = g^{alpha+1}
    """
    G = np.diag(g ** (alpha + 1))
    y = np.linalg.inv(W @ G @ W.T) @ x
    return y, G


def get_dg_poly(g, W, y, alpha=0):
    """Compute gradient of objective wrt g when f(g) = g^{alpha+1}."""
    w0 = np.sum(W**2, axis=0)
    z = W.T @ y
    dv = (z**2).mean(axis=-1) - w0
    dg = -(alpha + 1) * (g**alpha) * dv
    return dg


def get_y_exp(g, W, x):
    G = np.diag(np.exp(g))
    y = np.linalg.inv(W @ G @ W.T) @ x
    return y, G


def get_dg_exp(g, W, y):
    """Compute gradient of objective wrt g when f(g) = exp(g)."""
    w0 = np.sum(W**2, axis=0)
    z = W.T @ y
    dv = (z**2).mean(axis=-1) - w0
    dg = -np.exp(g) * dv
    return dg


def simulate(
    cholesky_list: Tuple[np.ndarray, ...],
    W: np.ndarray,
    get_y: Callable,
    get_dg: Callable,
    batch_size: int = 64,
    n_batch: int = 1024,
    lr_g: float = 5e-3,
    g0: Optional[np.ndarray] = None,
    seed: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if seed is not None:
        np.random.seed(seed)

    # initialize random set of gains g
    n, k = W.shape
    if g0 is not None:
        g = g0

    g_all = []
    g_last = []
    errors = []
    responses = []
    # run simulation with minibatches
    Ixx = np.eye(n)
    for Lxx in cholesky_list:
        Cxx = Lxx @ Lxx.T
        for _ in range(n_batch):
            x = sample_x(Lxx, batch_size)  # draw a sample of x

            y, G = get_y(g, W, x)
            dg = get_dg(g, W, y)
            g = g - lr_g * dg  # gradient descent
            M = np.linalg.inv(W @ G @ W.T)
            error = np.trace(np.abs(M @ Cxx @ M.T - Ixx)) / n
            errors.append(error)
            z = W.T @ y
            responses.append((x.mean(-1), y.mean(-1), z.mean(-1)))
            g_all.append(g)
    g_last = g
    g_all = np.stack(g_all, 0)
    return g_last, g_all, errors
