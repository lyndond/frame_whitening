import numpy as np
import numpy.typing as npt
from typing import Tuple, Callable, Optional

from frame_whitening.types import FuncType
from frame_whitening import stats
from scipy import optimize


def get_opt_funcs(
    func_type: FuncType,
) -> Tuple[
    Callable[..., Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    Callable[..., npt.NDArray[np.float64]],
]:
    """Returns optimization functions depending on f(g) function type.

    Parameters
    ----------
    func_type: Must be one of FuncType.POWER, FuncType.EXPONENTIAL, or FuncType.G_EXPONENTIAL.

    Returns
    -------
    get_y: Function that returns steady-state y and G (with possible nonlinear activation)
     for a given W, x, and g.
    get_dg: Function that returns gradient of objective wrt g.
    """
    if func_type == FuncType.POWER:
        get_y = _get_y_pow
        get_dg = _get_dg_pow
    elif func_type == FuncType.EXPONENTIAL:
        get_y = _get_y_exp
        get_dg = _get_dg_exp
    elif func_type == FuncType.G_EXPONENTIAL:
        get_y = _get_y_g_exp
        get_dg = _get_dg_g_exp
    else:
        raise ValueError(f"func_type must be one of {FuncType.__members__}")

    return get_y, get_dg


def _get_y_pow(
    g: npt.NDArray[np.float64],
    W: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    alpha: float = 0,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute steady-state y for a given x and g and alpha
    f(g) = g^{alpha+1}
    """
    G = np.diag(g ** (alpha + 1))
    y = np.linalg.inv(W @ G @ W.T) @ x
    return y, G


def _get_dg_pow(
    g: npt.NDArray[np.float64],
    W: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    alpha: float = 0,
    beta: float = 0,
) -> npt.NDArray[np.float64]:
    """Compute gradient of objective wrt g when f(g) = g^{alpha+1}."""
    w0 = np.sum(W**2, axis=0)
    z = W.T @ y
    dv = (z**2).mean(axis=-1) - w0
    dg = -(alpha + 1) * (g**alpha) * dv + beta * np.sign(g ** (1 + alpha))
    return dg


def _get_y_exp(
    g: npt.NDArray[np.float64], W: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute steady-state y for a given W, x, and g."""
    G = np.diag(np.exp(g))
    y = np.linalg.inv(W @ G @ W.T) @ x
    return y, G


def _get_dg_exp(
    g: npt.NDArray[np.float64],
    W: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    beta: float = 0,
) -> npt.NDArray[np.float64]:
    """Compute gradient of objective wrt g when f(g) = exp(g)."""
    w0 = np.sum(W**2, axis=0)
    z = W.T @ y
    dv = (z**2).mean(axis=-1) - w0
    dg = -np.exp(g) * dv + beta * np.sign(np.exp(g))
    return dg


def _get_y_g_exp(
    g: npt.NDArray[np.float64], W: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute steady-state y for a given W, x and g."""
    G = np.diag(g * np.exp(g))
    y = np.linalg.inv(W @ G @ W.T) @ x
    return y, G


def _get_dg_g_exp(
    g: npt.NDArray[np.float64],
    W: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    beta: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Compute gradient of objective wrt g when f(g) = gexp(g)."""
    w0 = np.sum(W**2, axis=0)
    z = W.T @ y
    dv = (z**2).mean(axis=-1) - w0
    deriv = np.exp(g) * (g + 1)
    dg = -deriv * dv + beta * np.sign(g * np.exp(g))
    return dg


def init_g_const(
    const: float, k: int, func_type: FuncType, alpha: Optional[float] = None
) -> npt.NDArray[np.float64]:
    """Initialize g with a constant value. Assumes positive constant."""
    assert const >= 0, "g0 must be non-negative"
    g0 = np.ones(k) * const
    if func_type == FuncType.POWER:
        # solve f(g) = g^(alpha+1) = const
        assert alpha is not None, "alpha must be specified for power function"
        g0 = np.float_power(g0, 1 / (alpha + 1))

    elif func_type == FuncType.EXPONENTIAL:
        # solve f(g) = exp(g) = const
        assert alpha is None, "alpha must be None for exponential function"
        g0 = np.log(g0)

    elif func_type == FuncType.G_EXPONENTIAL:
        # need to numerically solve for f(g) = g*exp(g) = const
        assert alpha is None, "alpha must be None for g-exponential function"
        func = lambda x: x * np.exp(x) - const
        g0 = np.ones(k) * optimize.fsolve(func, [1])[0]

    return g0


def simulate(
    cholesky_list: Tuple[npt.NDArray[np.float64], ...],
    W: npt.NDArray[np.float64],
    get_y: Callable[..., Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    get_dg: Callable[..., npt.NDArray[np.float64]],
    batch_size: int = 64,
    n_batch: int = 1024,
    lr_g: float = 5e-3,
    g0: Optional[npt.NDArray[np.float64]] = None,
    seed: Optional[float] = None,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Simulate data from a given model.

    Parameters
    ----------
    cholesky_list: List of Cxx Cholesky factors to sample with.
    W: Frame.
    get_y: Function to compute y from g, W, and x.
    get_dg: Function to compute gradient of objective wrt g.
    batch_size: Number of samples to draw per batch.
    n_batch: Number of batches to simulate.
    lr_g: Learning rate for g.
    g0: Initial value for g.
    seed: Seed for random number generator.

    Returns
    -------
    g_last: Last value of g for each matrix at end of each n_batch.
    g_all: All values of g throughout simulation.
    errros: Error of Cyy compared to Identity matrix. Computed using trace(|Cyy - I|)/N.
    """
    if seed is not None:
        np.random.seed(seed)

    # initialize random set of gains g
    N, K = W.shape
    g = g0 if g0 is not None else np.ones(K)

    g_all = []
    g_last = []
    errors = []
    responses = []

    # run simulation with minibatches
    Ixx = np.eye(N)
    for Lxx in cholesky_list:
        Cxx = Lxx @ Lxx.T
        for _ in range(n_batch):
            x = stats.sample_x(Lxx, batch_size)  # draw a sample of x

            y, G = get_y(g, W, x)
            M = np.linalg.inv(W @ G @ W.T)
            error = np.trace(np.abs(M @ Cxx @ M.T - Ixx)) / N
            errors.append(error)

            dg = get_dg(g, W, y)
            g = g - lr_g * dg  # gradient descent
            z = W.T @ y
            responses.append((x.mean(-1), y.mean(-1), z.mean(-1)))
            g_all.append(g)
        g_last.append(g)

    g_last = np.stack(g_last, 0)
    g_all = np.stack(g_all, 0)
    return g_last, g_all, errors  # type: ignore
