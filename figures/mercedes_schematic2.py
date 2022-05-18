#%%
import numpy as np
import matplotlib.pyplot as plt
import frame_whitening as fw
import frame_whitening.plot as fwplt
import seaborn as sns


#%% simulate
def simulate(Lxx, W, batch_size=64, n_batch=1024, lr_g=5e-3, g0=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # initialize random set of gains g
    n, k = W.shape
    if g0 is not None:
        g = g0
    else:
        g = np.random.rand(k)

    g_all = []
    error = []
    # run simulation with minibatches
    responses = []
    for _ in range(n_batch):
        x = fw.sample_x(Lxx, batch_size)  # draw a sample of x

        # steady-state of y
        y = np.linalg.inv(W @ np.diag(g) @ W.T) @ x

        # compute z and descend gradient of L(g,y) wrt g
        z = W.T @ y
        # dg = 1/batch_size * np.diag(z @ z.T) - np.diag(W @ W.T)
        dg = np.mean(z**2, -1) - 1.0  # more efficient, local gradient
        g = g + lr_g * dg  # gradient descent

        # Cyy = np.cov(y)
        responses.append((x.mean(-1), y.mean(-1), z.mean(-1)))
        g_all.append(g)
    error = np.array(error)
    g_all = np.stack(g_all, 0)
    return g, g_all, error, responses


#%%

np.random.seed(420)
num_switches = 2
n, k = 2, 3
batch_size = 1
n_batch = 500
lr_g = 5e-3
# Cxx, Lxx = fw.randcov2()
V, _ = np.linalg.qr(np.random.randn(n, n))
Cxx = V @ np.diag([3, 1]) @ V.T * np.random.rand() * 2
Lxx = np.linalg.cholesky(Cxx)
W = fw.get_mercedes_frame()

# g0 = np.random.rand(k) * 1
g0 = np.array([0.01, 1.2, 0.25])
g, g_all, error, responses = simulate(Lxx, W, batch_size, n_batch, lr_g, g0=g0)

#%%
plt.plot(g_all)

#%%
X = np.squeeze(np.array([x for x, _, _ in responses]))
Y = np.squeeze(np.array([y for _, y, _ in responses]))
Z = np.squeeze(np.array([z for _, _, z in responses]))


def moving_average(x, w):
    y = np.zeros_like(x)
    k = y.shape[-1]
    for i in range(k):
        y[:, i] = np.convolve(x[:, i], np.ones(w), "same") / w

    return y


t_skip = 1
X0 = X[::t_skip, :]
Y0 = Y[::t_skip, :]
Y0 = Y[::t_skip, :]

sz2 = Z**2
sz2 = moving_average(sz2, 25)
sz2 = sz2[::t_skip, :]
T = len(X0)

cols = sns.color_palette("husl", 3)
with sns.plotting_context("paper", font_scale=1.5):
    fig, ax = plt.subplots(5, 1, figsize=(5, 10), sharex="all", dpi=300)
    ax[0].scatter(range(T), X0[:, 0], s=1)
    ax[1].scatter(range(T), X0[:, 1], s=1)
    [ax[2].plot(range(T), sz2[:, i], color=cols[i], linewidth=3) for i in range(k)]
    ax[2].hlines(1, 0, T, linestyle="--", color="k")

    ax[-2].scatter(range(T), Y0[:, 0], s=1, color="k")
    ax[-1].scatter(range(T), Y0[:, 1], s=1, color="k")

    ax[0].set(ylabel=r"$x_0$")  # , xlim=(0, 1000))
    ax[1].set(ylabel=r"$x_1$")
    ax[2].set(ylabel=r"$\sigma^2_z$", ylim=(1e-1, 5), yscale="linear")
    ax[-2].set(ylabel=r"$y_0$")
    ax[-1].set(ylabel=r"$y_1$", xlabel="Step")

    sns.despine()

# %%


def get_Cyy(Cxx, W, g):
    M = np.linalg.inv(W @ np.diag(g) @ W.T)
    Cyy = M @ Cxx @ M
    return Cyy


t0, t1, t2 = 0, 75, 300
Cyy0 = get_Cyy(Cxx, W, g_all[t0])
Cyy1 = get_Cyy(Cxx, W, g_all[t1])
Cyy2 = get_Cyy(Cxx, W, g_all[t2])

with sns.plotting_context("paper", font_scale=1.5):
    fig, ax = plt.subplots(1, 3, figsize=(6, 3), sharex="all", sharey="all")
    In = np.eye(2)
    I_kwargs = {"linestyle": "--", "color": "C3"}
    Cyy_kwargs = {"linewidth": 2, "color": "k"}
    fwplt.plot_ellipse(Cyy0, ax=ax[0], **Cyy_kwargs)
    fwplt.plot_ellipse(In, ax=ax[0], **I_kwargs)

    fwplt.plot_ellipse(Cyy1, ax=ax[1], **Cyy_kwargs)
    fwplt.plot_ellipse(In, ax=ax[1], **I_kwargs)

    fwplt.plot_ellipse(Cyy2, ax=ax[2], **Cyy_kwargs)
    fwplt.plot_ellipse(In, ax=ax[2], **I_kwargs)

    ax[0].axis("square")
    ax[1].axis("square")
    ax[2].axis("square")
    lim_sq = 10
    ax[0].set(xlim=(-lim_sq, lim_sq), ylim=(-lim_sq, lim_sq), title="t=0")
    ax[1].set(title=f"t={t1}")
    ax[2].set(title=f"t=end")

    sns.despine()
