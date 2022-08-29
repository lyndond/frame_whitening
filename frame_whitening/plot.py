import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def plot_frame2d(
    R: npt.NDArray[np.float64], ax=None, plot_line: bool = False, **kwargs
) -> None:
    """Plots 2D frame vectors, optionally plot the axes along which they lie"""
    assert R.shape[0] == 2 and R.shape[1] > 2
    if ax is None:
        ax = plt
    for i in range(R.shape[1]):
        if i > 0 and "label" in kwargs:
            kwargs.pop("label")
        ax.plot([0, R[0, i]], [0, R[1, i]], "-o", **kwargs)

    # [ax.plot([0, R[0, i]], [0, R[1, i]], "-o", **kwargs) for i in range(R.shape[1])]
    x = np.linspace(-2, 2, 10)
    G2 = (R.T @ R) ** 2
    G2 = np.tril(G2, k=-1)
    G2[np.isclose(G2, 0)] = np.inf
    ind = np.unravel_index(G2.argmin(), G2.shape)
    if plot_line:
        [
            ax.plot(x, (R[1, i] / R[0, i]) * x, "--", color="r" if i in ind else "k")
            for i in range(R.shape[1])
        ]


def plot_ellipse(
    C: npt.NDArray[np.float64], n_pts: int = 20, ax=None, **kwargs
) -> None:
    """Plots 2D 1-stdev ellipse according to covariance matrix C"""
    assert C.shape == (2, 2)
    thetas = np.linspace(0, 2 * np.pi, n_pts)
    dots = np.stack([np.cos(thetas), np.sin(thetas)]) * 2
    E, V = np.linalg.eigh(C)
    ellipse = V @ np.diag(np.sqrt(E)) @ dots
    if ax is None:
        ax = plt
    ax.plot(ellipse[0, :], ellipse[1, :], **kwargs)
