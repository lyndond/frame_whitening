#%%
""" Modifies objective function to get sparse gains.

June 2022
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial

import frame_whitening as fw
import frame_whitening.plot as fwplt
from frame_whitening.types import *
import frame_whitening.simulation as fwsim

#%%  setup
np.random.seed(420)

# V, _ = np.linalg.qr(np.random.randn(n, n))
# Cxx0 = V @ np.diag([3.5, 1]) @ V.T * 0.1
Q = fw.rot2(np.deg2rad(45))
kappa = 8
Cxx0 = Q @ np.diag([kappa, 1]) @ Q.T * 1 / (np.sqrt(kappa))

cholesky_list = [np.linalg.cholesky(C) for C in [Cxx0]]
# W = fw.get_mercedes_frame()
# tx = np.deg2rad(30)
# W = np.concatenate([W, np.array([[np.cos(tx)], [np.sin(tx)]])], -1)
N, K = 2, 6
W = np.random.randn(N, K)
W = fw.normalize_frame(W)
print("Frame dims:", W.shape, " (more than necessary)")

get_y = fwsim.get_y_exp


def get_dg_exp_beta(g, W, y, beta):
    """Compute gradient of objective wrt g when f(g) = exp(g) with an L1
    penalty on g.

    \Delta g = -exp(g)(z^2-1) - beta sgn(g)
    """
    w0 = np.sum(W**2, axis=0)
    z = W.T @ y
    dv = (z**2).mean(axis=-1) - w0
    dg = -np.exp(g) * dv + beta * np.sign(np.exp(g))
    return dg


#%%  run sim

batch_size = 1024
n_batch = 5000
lr_g = 5e-1
beta = 0.005
get_dg = partial(get_dg_exp_beta, beta=beta)
g0 = np.log(np.ones(K) * 0.1)

all_gs = []
all_g_last = []
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

with sns.plotting_context("paper", font_scale=1.5):
    sns.set_style("white")
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=300)
    N, K = W.shape
    g0 = np.log(np.ones(K) * 0.1)

    label = r"$\exp({\bf g})$"
    ax.plot(errors, color="C3", label=label, linewidth=2, alpha=0.7)

    ax.set(
        yscale="log",
        xscale="log",
        ylim=(1e-4, 1e2),
        xlim=(1, n_batch),
        xlabel="iteration",
        ylabel=r"Error: $\frac{1}{N}$ Tr($\vert {\bf C}_{yy} - {\bf I} \vert$)",
        title=f"$\eta_g=${lr_g:.2e}, batch_size={batch_size}, $\\beta=${beta:.2e}",
    )
    ax.legend(title=r"$f({\bf g})$")
    sns.despine()

with sns.plotting_context("paper", font_scale=1.5):
    sns.set_style("white")
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=300)
    N, K = W.shape
    g0 = np.log(np.ones(K) * 0.1)

    label = r"$\exp({\bf g})$"
    ax.plot(np.exp(g_all), label=label, linewidth=2, alpha=0.7)
    ax.hlines(0, 0, n_batch, linestyle="--", color="k")

    ax.set(
        xscale="log",
        xlim=(1, n_batch),
        xlabel="Iteration",
        ylabel=r"$g_i$",
        title=f"$\eta_g=${lr_g:.2e}, batch_size={batch_size}, $\\beta=${beta:.2e}",
    )
    sns.despine()

# %%

M = np.linalg.inv(W @ np.diag(np.exp(g_last)) @ W.T)

print("C init\n", Cxx0)
print("C whitened\n", M @ Cxx0 @ M.T)

#%%
