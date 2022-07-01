#%%
"""Convolutional whitening, take 3.

From David's note.

Weights are shared across z's.
June 30, 2022
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp

#%% make W_hat

np.random.seed(1)

# construct a convolutional filter bank (plotted below)
w_vec = np.random.randn(4)
W_hat_transpose = sp.linalg.toeplitz(w_vec)  # make circulant matrix
W_hat_transpose = W_hat_transpose[:, :-1]  # drop last column
# remove corners
W_hat_transpose[-1][0] = 0
W_hat_transpose[0][-1] = 0
W_hat = W_hat_transpose.T  # flip it

# generate Cxx
N = 3
# construct a local banded Cxx
Cxx = np.array(
    (
        [2.0, -1, 0.5],
        [-1, 2.0, -1],
        [0.5, -1, 2.0],
    )
)

# random Cxx
# A = np.random.randn(N, N)
# Cxx = A @ A.T

T = 512  # batch size
n_batch = int(2**13)
norm_w_sq = np.diag(W_hat.T @ W_hat)
eta_g = 1e-3
g = np.ones(N + 1)  # init g
all_g = []
error = []

# run simulation
for _ in tqdm(range(n_batch)):
    x = A @ np.random.randn(N, T)
    M = W_hat @ np.diag(g) @ W_hat.T
    M_inv = np.linalg.inv(M)
    y = M_inv @ x

    z = W_hat.T @ y

    dg = np.mean(z**2, axis=1) - norm_w_sq
    g = g + eta_g * dg
    all_g.append(g)
    error.append(np.linalg.norm(np.cov(y) - np.eye(N)) ** 2)

#%%
def heatmap(data, vlims, ax, cmap="icefire"):
    """Helper function to plot heatmaps"""
    sns.heatmap(data, cmap=cmap, vmin=vlims[0], vmax=vlims[1], ax=ax)


vlims = (-5, 5)

# compute M = (W G W.T)^-1 and Cyy
M_inv = np.linalg.inv(M)
Cyy = M_inv @ Cxx @ M_inv

with sns.plotting_context("paper", font_scale=1.5):
    fig, ax = plt.subplots(1, 5, figsize=(20, 4), dpi=200)
    ax[0].plot(error)
    ax[1].plot(all_g)
    heatmap(W_hat.T, (-2, 2), ax[2])
    heatmap(Cxx, vlims, ax[3])
    heatmap(Cyy, vlims, ax[4])

    ax[0].set(
        title=f"batch_size:{T}; $\eta_g$:{eta_g:.2e}",
        xlabel="Iter",
        ylabel=r"$\vert C_{yy}- I\vert^2_F$",
        yscale="log",
    )
    ax[1].set(title=f"", xlabel="Iter", ylabel="g")
    ax[2].set(title=r"$\hatW^\top$", xlabel="", ylabel="")
    ax[3].set(title=r"$C_{xx}$")
    ax[4].set(title=r"$C_{yy}$")
    sns.despine()
    fig.tight_layout()
