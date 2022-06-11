import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from frame_whitening.types import *
import frame_whitening.simulation as fwsim


func_type = FuncType.POWER
get_y, get_dg = fwsim.get_opt_funcs(func_type)

alphas = (0, 1)

fwsim.simulate()

cols = sns.color_palette("mako", len(alphas))
with sns.plotting_context("paper", font_scale=1):
    sns.set_style("dark")
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
    M0 = np.linalg.inv(W @ np.diag(all_g_last[0]) @ W.T)
    M1 = np.linalg.inv(W @ np.diag(all_g_last[1] ** 2) @ W.T)
    fwplt.plot_ellipse(Cxx0, ax=ax, **{"color": "k", "linewidth": 3, "label": r"Init"})
    fwplt.plot_ellipse(
        M0 @ Cxx0 @ M0,
        ax=ax,
        **{"color": cols[0], "linewidth": 3, "label": r"Whitened, $\alpha=0$"},
    )
    fwplt.plot_ellipse(
        M1 @ Cxx0 @ M1,
        ax=ax,
        **{"color": cols[1], "linewidth": 3, "label": r"Whitened, $\alpha=1$"},
    )
    fwplt.plot_ellipse(
        np.eye(n),
        ax=ax,
        **{"color": "w", "linewidth": 2, "linestyle": "--", "label": "Identity"},
    )

    fwplt.plot_frame2d(W, ax=ax)
    ax.legend(title=r"${\bf C}_{yy}$")

    ax.axis("square")
    ax.set(xlim=(-4, 4), ylim=(-4, 4), ylabel=r"$y_1$", xlabel=r"$y_0$")
    sns.despine()


with sns.plotting_context("paper", font_scale=1.5):
    sns.set_style("white")
    fig, ax = plt.subplots(1, 2, figsize=(10, 6), dpi=300, sharex="all", sharey="all")
    ax[0].plot(all_gs[0])
    ax[0].hlines(g_opt, 0, n_batch, linestyle="--")

    i = 1
    ax[1].plot(all_gs[i] ** (alphas[i] + 1))
    ax[1].hlines(g_opt, 0, n_batch, linestyle="--")

    ax[0].set(ylabel=r"$g^{\alpha + 1}$", title=r"$\alpha=0$", xlabel="Iter")
    ax[1].set(title=r"$\alpha=1$")
    sns.despine()
