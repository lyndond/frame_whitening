#%%
import numpy as np
import torch
import torch.nn.functional as F
from functools import partial

#%%
# deterministic initialization
np.random.seed(0)
torch.manual_seed(0)

N, K = 2, 3

# create 1D convolutional filters
# set them to be the Mercedes frame along channel dimension
theta = np.arange(0, np.pi, np.pi / 3)
W = np.array([[np.cos(theta)], [np.sin(theta)]])
weights = torch.from_numpy(W).float()

# reshape to convolutional filter bank
# num_output_chan x num_input_chan x height x width
# where height and width are both 1 here.
weights = weights.view(K, N, 1, 1).to(device)

batch_size, chan, height, width = 128, N, 5, 5
shape = (batch_size, chan, height, width)

# convenience functions to do convolutional W and W.T ops
WT = partial(F.conv2d, weight=weights)
W = partial(F.conv_transpose2d, weight=weights)

# Make Cxx and its Cholesky factor Lxx
Q, _ = np.linalg.qr(np.random.randn(2, 2))
cond_max = 5.0
D = np.diag((np.random.uniform(1, cond_max), 1)) * np.random.rand() * 2
Cxx = Q @ D @ Q.T
Lxx = np.linalg.cholesky(Cxx)
Lxx = torch.from_numpy(Lxx).float().to(device)

# init g, z, y
g = torch.ones(1, K, height, width, device=device) * 0.1
z = torch.randn(1, K, height, width, device=device) * 0.1
y = torch.randn(1, N, height, width, device=device) * 0.1

eta_y, eta_g = 1e-4, 1e-3

n_iters = 1000
pbar = tqdm(range(n_iters))
E = []
for _ in pbar:  # update gains
    # draw x and colour it with Lxx, cholesky decomp of Cxx
    x0 = torch.randn(shape).to(device)
    x = torch.einsum("ck, bkhw -> bchw", Lxx, x0)

    for _ in range(1000):  # let y achieve steady-state
        dy = x - W(g * z)
        y = y + eta_y * dy

    z = WT(y)
    z2 = z.pow(2)
    dvar = z2.mean(axis=0, keepdim=True) - 1.0

    dg = -dvar
    g = g - eta_g * dg

    # avg sq diff from unity for ALL K * batch_size * height * width activations
    error = dvar.pow(2).mean()
    pbar.set_postfix({"Error proxy: mean((z^2 - 1)^2)": error.item()})
    E.append(error)

#%%
