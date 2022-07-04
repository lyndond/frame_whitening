#%%
"""
Trying to break symmetry in WGW.T -> W_b @ G @ W_f.T

Doesn't seem to work?
May 18 2022
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import frame_whitening as fw
import frame_whitening.plot as fwplt

#%%
n, k = 2, 3
Wf = np.random.randn(n, k)
Wf = fw.normalize_frame(Wf)

Wb = np.random.randn(n, k)
# Wb = fw.parsevalize(Wf)
Wb = fw.normalize_frame(Wb)


#%% simulate
def simulate(Lxx, Wb, Wf, batch_size=64, n_batch=1024, lr_g=5e-3, g0=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # initialize random set of gains g
    n, k = Wf.shape
    if g0 is not None:
        g = g0
    else:
        g = np.random.rand(k)

    g_all = []
    # run simulation with minibatches
    sigma0 = np.diag(Wf.T @ Wb)
    for _ in range(n_batch):
        x = fw.sample_x(Lxx, batch_size)  # draw a sample of x

        # steady-state of y
        G = np.diag(g)
        M1 = Wb @ G @ Wf.T
        y = np.linalg.inv(M1 + M1.T) @ x

        # compute z and descend gradient of L(g,y) wrt g
        z0 = 1 / batch_size * np.diag(Wf.T @ y @ y.T @ Wb)

        dg = z0 - sigma0  # more efficient, local gradient
        g = g + lr_g * dg  # gradient descent

        g_all.append(g)
    g_all = np.stack(g_all, 0)
    return g, g_all


Cxx, Lxx = fw.randcov2()
batch_size = 512
n_batch = 4096
g, g_all, *_ = simulate(Lxx, Wb, Wf, batch_size, n_batch, lr_g=5e-3)

plt.plot(g_all)
# %%

M = np.linalg.inv(Wb @ np.diag(g) @ Wf.T)
M @ Cxx @ M.T
