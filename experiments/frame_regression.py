#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import frame_whitening as fw
import frame_whitening.plot as fwplt

#%%

M, N = 2, 2
K = N * M

Cxx, Lxx = fw.randcov2()
Cyy, Lyy = fw.randcovn(N)

W = fw.normalize_frame(np.random.randn(K, M).T)
V = fw.normalize_frame(np.random.randn(N, K))

#%%
batch_size = 256
g = np.ones(K) * 0.1

eta_g = 5e-4
E = []
G = []

for _ in range(100):
    xt = fw.sample_x(Lxx, batch_size)
    yt = fw.sample_x(Lyy, batch_size)
    # G = np.diag(g)

    zt = W.T @ xt
    et = yt - V @ np.diag(g) @ zt

    dg = 2 * (zt * (V.T @ et)).mean(-1)

    g = g - eta_g * dg
    G.append(g)
    E.append(et.mean(-1))

E = np.stack(E, -1)
plt.plot(E.T)
