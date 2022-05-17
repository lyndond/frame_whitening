#%%
"""
A randomized linear algebra approach to Frame whitening
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import frame_whitening as fw
import frame_whitening.plot as fwplt

#%%

n = 2
k = n * (n + 1) // 2
Cxx, Lxx = fw.randcov2()

V, _ = np.linalg.qr(np.random.randn(n, n))
s = np.random.rand(n) + 1.0
Cxx = V @ np.diag(s) @ V.T
Lxx = np.linalg.cholesky(Cxx)

W = fw.normalize_frame(np.random.randn(n, k))
# W = fw.get_near_etf(n, k, s=None, tight_iter=10, num_iter=1000, show=True)

# W = fw.get_equiangular_3d()

eta_g = 5e-2
error = []
g = np.ones(k)
dist = []
Inn = np.eye(n)
for _ in range(1000):
    X = fw.sample_x(Lxx, 256)
    Y = np.linalg.solve(W @ np.diag(g) @ W.T, X)
    Z = W.T @ Y
    dg = np.mean(Z**2, -1) - 1.0
    g += eta_g * dg
    Cyy = np.cov(Y)
    error.append(np.linalg.norm(Inn - Cyy) ** 2)

fro_d2 = np.array(error)

plt.plot(fro_d2 / n)

# %%
fig, ax = plt.subplots(1, 2, sharex="all", sharey="all")
px = ax[0].imshow(Cxx)
py = ax[1].imshow((np.cov(Y) - np.eye(n)) ** 2)
plt.colorbar(px)
plt.colorbar(py)

#%%





#%%

n = 10
k = n * (n + 1) // 2
print(k)
W, G, res = design_grassmannian(
    n,
    k,
    niter=500,
    fract_shrink=0.8,
    shrink_fact=0.9,
)


#%%

plt.imshow(G, aspect="auto")
plt.colorbar()
#%%
plt.plot(res)
