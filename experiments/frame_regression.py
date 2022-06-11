#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import frame_whitening as fw
import frame_whitening.plot as fwplt

#%%

np.random.seed(42069)

M, N = 2, 5
K = N * M

Cxx, Lxx = fw.randcovn(M)

W = fw.normalize_frame(np.random.randn(K, M).T)
V = fw.normalize_frame(np.random.randn(N, K))

batch_size = 512
g = np.ones(K) * 0.1

eta_g = 5e-3
E = []
G = []
A = np.random.randn(N, M)
errors = []
for _ in range(10000):
    xt = fw.sample_x(Lxx, batch_size)
    yt = A @ xt

    zt = W.T @ xt
    et = yt - V @ np.diag(g) @ zt

    dg = -2 * (zt * (V.T @ et)).mean(-1)

    g = g - eta_g * dg
    G.append(g)
    E.append(np.linalg.norm(et))
    error = np.linalg.norm(V @ np.diag(g) @ W.T - A)
    errors.append(error)


with sns.plotting_context("paper", font_scale=1):
    fig, ax = plt.subplots(1, 1)
    errors = np.stack(errors)
    ax.plot(errors)
    ax.set(xlabel="iteration", ylabel="error", title="error")
    sns.despine()
