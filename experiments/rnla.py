#%%
"""
A randomized linear algebra approach to Frame whitening
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import frame_whitening as fw
import frame_whitening.plot as fwplt
from tqdm import tqdm
import pandas as pd

#%%


def simulate(Lxx, W, eta_g, n_batch, batch_size):
    n, k = W.shape
    g = np.ones(k)
    error = []
    g = np.ones(k)
    Inn = np.eye(n)
    for _ in range(n_batch):
        X = fw.sample_x(Lxx, batch_size)
        Y = np.linalg.solve(W @ np.diag(g) @ W.T, X)
        Z = W.T @ Y
        dg = np.mean(Z**2, -1) - 1.0
        g += eta_g * dg
        Cyy = np.cov(Y)
        err_sq = np.linalg.norm(Inn - Cyy) ** 2
        error.append(err_sq)
    fro_d2 = np.array(error) / n**2
    return fro_d2


df_sim = pd.DataFrame(
    columns=["n", "k", "n_batch", "batch_size", "kappa0", "error"],
    dtype=(np.float64),
)
for n in range(13, 15, 1):
    # n = 2
    k = n * (n + 1) // 2

    print(f"n = {n}, k = {k}")

    n_batch = 1024
    batch_size = 512

    all_error = []

    n_repeats = 10
    pbar = tqdm(range(n_repeats))
    V, _ = np.linalg.qr(np.random.randn(n, n))
    # s = np.random.rand(n) * 3 + alpha
    s = np.linspace(1, 5, n) + np.random.randn(n) * 0.1
    Cxx = V @ np.diag(s) @ V.T
    Lxx = np.linalg.cholesky(Cxx)
    kappa0 = np.linalg.cond(Cxx)

    for _ in pbar:

        if n == 2:
            W = fw.get_mercedes_frame(False, True)
        # elif n == 3:
        # W = fw.get_equiangular_3d()
        else:
            W, G, res = fw.get_grassmanian(n, k, niter=400)
            pbar.set_postfix({"W_init": True})

        err = simulate(Lxx, W, 5e-3, n_batch, batch_size)
        df_sim = pd.concat(
            [
                df_sim,
                pd.DataFrame(
                    [
                        {
                            "n": n,
                            "k": k,
                            "n_batch": n_batch,
                            "batch_size": batch_size,
                            "kappa0": kappa0,
                            "error": float(err[-200:].mean()),
                        }
                    ],
                ),
            ],
            ignore_index=True,
        )
        all_error.append(err)
#%%
err_to_plot = np.stack(all_error, -1)
fig, ax = plt.subplots(1, 1)
ax.plot(err_to_plot)
ax.set(yscale="log")

#%%

fig, ax = plt.subplots(1, 1)
sns.scatterplot(x="n", y="error", data=df_sim, ax=ax)
ax.set(ylim=(1e-4, 1e-0), yscale="log")


#%%
plt.plot(err)
