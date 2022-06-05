#%%
"""
Enforce positive g's.
Assume G is G*G and re-derive as before.

June 2022

TODO: I messed up the math. need to fix.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import frame_whitening as fw
import frame_whitening.plot as fwplt

#%%
def simulate(
    cholesky_list,
    W,
    batch_size=64,
    n_batch=1024,
    lr_g=5e-3,
    g0=None,
    enforce_positive=False,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)

    # initialize random set of gains g
    n, k = W.shape
    if g0 is not None:
        g = g0
    else:
        g = fw.compute_g_opt(np.eye(n), W)  # initialize Parseval frame

    g_all = []
    g_last = []
    # run simulation with minibatches
    responses = []
    errors = []
    Ixx = np.eye(n)
    w_norm2 = np.sum(W**2, axis=0)
    for Lxx in cholesky_list:
        Cxx = Lxx @ Lxx.T
        for _ in range(n_batch):
            x = fw.sample_x(Lxx, batch_size)  # draw a sample of x

            # steady-state of y
            G = np.diag(g)
            if enforce_positive:
                G = G**2

            # compute z and descend gradient of L(g,y) wrt g
            y = np.linalg.inv(W @ G @ W.T) @ x
            z = W.T @ y

            dg = np.mean(z**2, -1) - w_norm2  # more efficient, local gradient
            if enforce_positive:
                dg *= 2 * g

            g = g + lr_g * dg  # gradient descent
            M = np.linalg.inv(W @ G @ W.T)
            error = np.trace(np.abs(M @ Cxx @ M.T - Ixx)) / n
            errors.append(error)
            responses.append((x.mean(-1), y.mean(-1), z.mean(-1)))
            g_all.append(g)
    g_last.append(g)
    g_all = np.stack(g_all, 0)
    return g_last, g_all, errors, responses


#%%

np.random.seed(420)
num_switches = 2
n, k = 2, 3
batch_size = 16
n_batch = 10000
lr_g = 3e-2

# V, _ = np.linalg.qr(np.random.randn(n, n))
# Cxx0 = V @ np.diag([3.5, 1]) @ V.T * 0.1
Q = fw.rot2(np.deg2rad(30))
Cxx0 = Q @ np.diag([20.0, 1]) @ Q.T * 1.0

V, _ = np.linalg.qr(np.random.randn(n, n))
V = np.eye(n)
Cxx1 = V @ np.diag([100.0, 1]) @ V.T * 0.5
cholesky_list = [np.linalg.cholesky(C) for C in (Cxx0, Cxx1)]
W = fw.get_mercedes_frame()
# W = fw.normalize_frame(np.random.randn(n, k))

fig, ax = plt.subplots(1, 1)
fwplt.plot_ellipse(Cxx0, ax=ax)
fwplt.plot_frame2d(W, ax=ax, plot_line=True)
ax.axis("square")
ax.set(xlim=(-4, 4), ylim=(-4, 4))


# g0 = np.array([0.01, 1.2, 0.25])
g0 = np.ones(k)
sim_resp = simulate(
    cholesky_list,
    W,
    batch_size,
    n_batch,
    lr_g,
    g0=g0,
    enforce_positive=True,
)
g_last, g_all, errors, responses = sim_resp

fig, ax = plt.subplots(1, 1)
ax.plot(errors)
ax.set(yscale="log", ylim=(1e-4, 1e1))

#%%

fig, ax = plt.subplots(1, 1)
ax.plot(g_all)

for i, Lxx in enumerate(cholesky_list):
    Cxx = Lxx @ Lxx.T
    Csqrt = fw.psd_sqrt(Cxx)
    g_opt = fw.compute_g_opt(Csqrt, W)
    print(g_opt)
    ax.hlines(
        np.sqrt(g_opt), i * n_batch, (i * n_batch) + n_batch, linestyle="--", alpha=0.5
    )

ax.set(yscale="linear", ylim=(-1, 2))
sns.despine()

#%%
