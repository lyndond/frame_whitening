#%%
"""Test nonlinearities on gain vector
fog.py (f of g)

This uses f(g) = g^{alpha + 1} or f(g)=exp(g) as the nonlinearity on g, as a means to prevent 
negative values.

June 06, 2022
Seems to increase in convergence speed for with increasing alpha
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial

import frame_whitening as fw
import frame_whitening.plot as fwplt
from frame_whitening.types import *
import frame_whitening.simulation as fwsim

#%%

np.random.seed(420)
n_contexts = 2
n, k = 2, 3
batch_size = 256
n_batch = 20000
lr_g = 5e-3

# V, _ = np.linalg.qr(np.random.randn(n, n))
# Cxx0 = V @ np.diag([3.5, 1]) @ V.T * 0.1
Q = fw.rot2(np.deg2rad(45))
kappa = 8
Cxx0 = Q @ np.diag([kappa, 1]) @ Q.T * 1 / (np.sqrt(kappa))

cholesky_list = [np.linalg.cholesky(C) for C in [Cxx0]]
W = fw.get_mercedes_frame()
# tx = np.deg2rad(30)
# W = np.concatenate([W, np.array([[np.cos(tx)], [np.sin(tx)]])], -1)
# W = fw.normalize_frame(np.random.randn(n, 4))

alphas = [0.0, 1.0, 2.0, 3.0, 5.0, 6.0]

all_gs = []
all_g_last = []

func_type = "EXPONENTIAL"

with sns.plotting_context("paper", font_scale=1.5):
    sns.set_style("white")
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=300)
    cols = sns.color_palette("mako", len(alphas))
    N, K = W.shape
    g0 = np.log(np.ones(K) * 0.1)

    get_y, get_dg = fwsim.get_opt_funcs(EXPONENTIAL)
    sim_resp = fwsim.simulate(
        cholesky_list,
        W,
        get_y,
        get_dg,
        batch_size,
        n_batch,
        lr_g,
        g0=g0,
    )
    g_last, g_all, errors = sim_resp
    all_gs.append(g_all)
    all_g_last.append(g_last)
    label = r"$\exp({\bf g})$"
    ax.plot(errors, color="C3", label=label, linewidth=2, alpha=0.7)

    get_y, get_dg = fwsim.get_opt_funcs(POLYNOMIAL)
    for i, alpha in enumerate(alphas):
        g0 = np.float_power(np.ones(K) * 0.1, 1 / (alpha + 1))

        sim_resp = fwsim.simulate(
            cholesky_list,
            W,
            partial(get_y, alpha=alpha),
            partial(get_dg, alpha=alpha),
            batch_size,
            n_batch,
            lr_g,
            g0=g0,
        )
        g_last, g_all, errors = sim_resp
        all_gs.append(g_all)
        all_g_last.append(g_last)
        label = r"${\bf g}^" + f"{alpha+1:.0f}" + r"$"
        ax.plot(errors, color=cols[i], label=label, linewidth=2, alpha=0.5)

    ax.set(
        yscale="log",
        xscale="log",
        ylim=(1e-4, 1e2),
        xlim=(1, n_batch),
        xlabel="iteration",
        ylabel=r"Error: $\frac{1}{N}$ Tr($\vert {\bf C}_{yy} - {\bf I} \vert$)",
        title=f"$\eta_g=${lr_g:.2e}, batch_size={batch_size}",
    )
    ax.legend(title=r"$f({\bf g})$")
    sns.despine()

#%%

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

#%%
Cxx_sqrt = fw.psd_sqrt(Cxx0)
g_opt = fw.compute_g_opt(Cxx_sqrt, W)

print(r"optimal $g^{\alpha+1}$", g_opt)


#%%
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

#%%

Lxx = cholesky_list[0]
Cxx = Lxx @ Lxx.T
i = 1
gg = all_g_last[i] ** (i + 1)
M = np.linalg.inv(W @ np.diag(gg) @ W.T)
M @ Cxx @ M
