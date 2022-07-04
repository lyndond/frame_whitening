#%% 
"""
Overcomplete whitening
**Lyndon Duong and David Lipshutz**
May 2022
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
from tqdm import tqdm
import frame_whitening as fw
import os
print(os.getcwd())

# create high-dimensional embedding matrix W
W = fw.get_mercedes_frame()


#%% simulate
def simulate(Lxx, W, batch_size=64, n_batch=1024, lr_g=5E-3, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # initialize random set of gains g
    n, k = W.shape
    g = np.random.rand(k)

    g_all = []

    # run simulation with minibatches
    for _ in range(n_batch):
        X = fw.sample_x(Lxx, batch_size)  # draw a sample of x

        # steady-state of y
        y = np.linalg.inv(W @ np.diag(g) @ W.T) @ X

        # compute z and descend gradient of L(g,y) wrt g
        z =  W.T @ y
        # dg = 1/batch_size * np.diag(z @ z.T) - np.diag(W @ W.T)
        dg = np.mean(z ** 2, -1) - 1.  # more efficient, local gradient
        g = g + lr_g * dg # gradient descent

        g_all.append(g)

    return g, g_all

# run simulation
n, k = 2, 3
Cxx, Lxx = fw.randcov2()

batch_size = 256
n_batch = 1024
lr_g = 5E-3
g, g_all = simulate(Lxx, W, batch_size, n_batch, lr_g)

# converged steady-state y samples and Cyy
X = fw.sample_x(Lxx, 1024)
Y = np.linalg.inv(W @ np.diag(g) @ W.T) @ X
Cyy = np.cov(Y)

print("final g: \n", g)
print("cov Y: \n", Cyy)
print("cov X: \n", Cxx)
cond_x = np.linalg.cond(Cxx)
cond_y = np.linalg.cond(Cyy)
print("Condition number of Cxx: \n", cond_x)
print("Condition number of Cyy: (should now be closer to 1) \n", cond_y)

#%% what should g be?

Csqrt = fw.psd_sqrt(Cxx)
g_opt = fw.compute_g_opt(Csqrt, W)

print("simulated g: ", g)
print("true g: ", g_opt)
print("ratio g/g_opt: ", g/g_opt)

#%% plots

gs = np.stack(g_all, -1)
with sns.plotting_context("paper"):
    fig, ax = plt.subplots(1, 1, figsize=(12,6))
    [ax.plot(gs[i], label=f"g{i}", linewidth=2) for i in range(k)]
    ax.hlines(g_opt, 0, n_batch, linestyle="--", label="optimal")
    ax.set(xlabel="iter", ylabel="g", title="dynamics of g")
    sns.despine()
    ax.legend()

    fig, ax = plt.subplots(1, 2, sharex="all", sharey="all")
    xlim = (-6, 6)
    ylim = xlim
    ax[0].scatter(*X)
    ax[1].scatter(*Y)
    ax[0].axis("square")
    ax[1].axis("square")
    ax[0].set(title=f"X (condition number = {cond_x:.2f})", xlim=xlim, ylim=ylim)
    ax[1].set(title=f"Y (condition number = {cond_y:.2f})", xlim=xlim, ylim=ylim)
    sns.despine()
    fig.tight_layout()

