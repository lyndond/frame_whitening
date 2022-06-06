#%%
"""Test the effect of varying alpha
June 06, 2022
Seems to increase in convergence speed for with increasing alpha
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import frame_whitening as fw
import frame_whitening.plot as fwplt

#%%
def get_y_ss(g, W, x, alpha=0):
    """Compute steady-state y for a given x and g and alpha"""
    G = np.diag(g ** (alpha + 1))
    y = np.linalg.inv(W @ G @ W.T) @ x
    return y


def get_dg(g, W, y, alpha=0):
    w0 = np.sum(W**2, axis=0)
    z = W.T @ y
    dv = (z**2).mean(axis=-1) - w0
    dg = -(alpha + 1) * (g**alpha) * dv
    # dg = -(0 + 1) * (g**alpha) * dv
    return dg


def simulate(
    cholesky_list,
    W,
    batch_size=64,
    n_batch=1024,
    lr_g=5e-3,
    alpha=0,
    g0=None,
    seed=None,
):
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
            x = fw.sample_x(Lxx, batch_size)  # draw a sample of x

            y_ss = get_y_ss(g, W, x, alpha=alpha)
            dg = get_dg(g, W, y_ss, alpha=alpha)
            g = g - lr_g * dg  # gradient descent
            G = np.diag(g ** (alpha + 1))
            M = np.linalg.inv(W @ G @ W.T)
            error = np.trace(np.abs(M @ Cxx @ M.T - Ixx)) / n
            errors.append(error)
            z = W.T @ y_ss
            responses.append((x.mean(-1), y_ss.mean(-1), z.mean(-1)))
            g_all.append(g)
    g_last = g
    g_all = np.stack(g_all, 0)
    return g_last, g_all, errors


#%%

np.random.seed(420)
n_contexts = 2
n, k = 2, 3
batch_size = 256
n_batch = 20000
lr_g = 1e-4

# V, _ = np.linalg.qr(np.random.randn(n, n))
# Cxx0 = V @ np.diag([3.5, 1]) @ V.T * 0.1
Q = fw.rot2(np.deg2rad(45))
Cxx0 = Q @ np.diag([20, 1]) @ Q.T * 1.0

cholesky_list = [np.linalg.cholesky(C) for C in [Cxx0]]
W = fw.get_mercedes_frame()
# tx = np.deg2rad(30)
# W = np.concatenate([W, np.array([[np.cos(tx)], [np.sin(tx)]])], -1)
# W = fw.normalize_frame(np.random.randn(n, 4))

alphas = [0.0, 1.0, 2.0, 3.0, 5.0, 6.0]
# alphas = [2.0]
# alphas = [0.0, 1.0]
all_gs = []
all_g_last = []
with sns.plotting_context("paper", font_scale=1.5):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=150)
    cols = sns.color_palette("mako", len(alphas))
    N, K = W.shape
    # g0 = np.random.randn(K)
    for i, alpha in enumerate(alphas):
        g0 = np.float_power(np.ones(K) * 0.1, 1 / (alpha + 1))

        sim_resp = simulate(
            cholesky_list,
            W,
            batch_size,
            n_batch,
            lr_g,
            alpha=alpha,
            g0=g0,
        )
        g_last, g_all, errors = sim_resp
        all_gs.append(g_all)
        all_g_last.append(g_last)
        ax.plot(errors, color=cols[i], label=f"{alpha}", linewidth=2)

    ax.set(
        yscale="log",
        # ylim=(1e-4, 1e0),
        xlim=(0, n_batch),
        xlabel="iteration",
        ylabel=r"$\frac{1}{N}$ Tr($\vert {\bf C}_{yy} - {\bf I} \vert$)",
        title=f"$\eta_g=${lr_g:.2e}, batch_size={batch_size}",
    )
    ax.legend(title=r"$\alpha$")
    sns.despine()

fig, ax = plt.subplots(1, 1)
fwplt.plot_ellipse(Cxx0, ax=ax)
fwplt.plot_frame2d(W, ax=ax)
ax.axis("square")
ax.set(xlim=(-4, 4), ylim=(-4, 4), ylabel=r"$y_1$", xlabel=r"$y_0$")
sns.despine()

#%%
Cxx_sqrt = fw.psd_sqrt(Cxx0)
g_opt = fw.compute_g_opt(Cxx_sqrt, W)

print(r"optimal $g^{\alpha+1}$", g_opt)


#%%
with sns.plotting_context("paper", font_scale=1.5):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=150, sharex="all", sharey="all")
    ax[0].plot(all_gs[0])
    # ax[0].hlines(g_opt, 0, n_batch, linestyle="--")

    i = 1
    ax[1].plot(all_gs[i] ** (alphas[i] + 1))
    # ax[1].hlines(g_opt, 0, n_batch, linestyle="--")

    ax[0].set(ylabel=r"$g^{\alpha + 1}$", title=r"$\alpha=0$")
    ax[1].set(title=r"$\alpha=1$")
    sns.despine()

#%%

Lxx = cholesky_list[0]
Cxx = Lxx @ Lxx.T
i = 1
gg = all_g_last[i] ** (i + 1)
M = np.linalg.inv(W @ np.diag(gg) @ W.T)
M @ Cxx @ M
