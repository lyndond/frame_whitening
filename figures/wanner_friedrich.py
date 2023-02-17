#%%
import copy
from pathlib import Path
import os
import multiprocessing
from typing import Any, Dict
print(f'num cpus: {multiprocessing.cpu_count()}')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as scio
import seaborn as sns
#%%

def load_matfile(filename: str) -> Dict:
    def parse_mat(element: Any):
        # lists (1D cell arrays usually) or numpy arrays as well
        if element.__class__ == np.ndarray and element.dtype == np.object_ and len(element.shape) > 0:
            return [parse_mat(entry) for entry in element]

        # matlab struct
        # if element.__class__ == scio.matlab.mio5_params.mat_struct:
        if element.__class__ == scio.matlab.mat_struct:
            return {fn: parse_mat(getattr(element, fn)) for fn in element._fieldnames}

        # regular numeric matrix, or a scalar
        return element

    mat = scio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    dict_output = dict()

    # not considering the '__header__', '__version__', '__globals__'
    for key, value in mat.items():
        if not key.startswith('_'):
            dict_output[key] = parse_mat(mat[key])
    return dict_output


data0 = load_matfile('data/wanner_friedrich/Q_Cascaded_reduced.mat')
data0 = data0['Qred']

#%%

def process_cell_info(cell_info_array: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame.from_dict({
    'ID': cell_info_array[:,0].astype(int),
    'row_idx': cell_info_array[:,1].astype(int) - 1,
    'cell_type': cell_info_array[:,2].astype(int),
    'microcluster': cell_info_array[:,3].astype(int),
    'GlomID': cell_info_array[:,4].astype(int),
    'length': cell_info_array[:,5].astype(float),
    })

def process_data(data0: Dict) -> Dict:
    data = copy.deepcopy(data0)
    id_keys = (
        'INidsAW', 
        'MCinds', 
        'INinds', 
        'takeodors',
        )

    # matlab indices start at 1, python at 0
    for key in id_keys:
        data[key] = data[key] - 1
    
    data.pop('Cellinfo')
    for key in ('Cellinfo2', 'CellinfoMC', 'CellinfoIN'):
        data[key] = process_cell_info(data[key])

    return data

print(data0.keys())

data = process_data(data0)
info = data['Cellinfo2']

#%%

def plot_responses(data, cells, cmap='mako', vmin=0., vmax=None, dpi=100):
    assert cells in ('MC', 'IN')
    to_plot = data[f'DMot{cells}']
    fig, ax = plt.subplots(2, 4, figsize=(16, 8), sharex='all', sharey='all', dpi=dpi)
    ax = ax.ravel()
    cmap = sns.color_palette(cmap, as_cmap=True)
    cmap.set_bad('k')

    if vmax is None:
        vmax = np.nanmax(to_plot)

    for stim in range(8):
        ax[stim].imshow(to_plot[...,stim], aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        odor = data['odorlist2'][stim]
        # odor = data['odorlist'][odor]
        ax[stim].set(title=f'{odor}', ylabel=f'{cells} neuron', xlabel='Time')
    sns.despine()
    fig.tight_layout()


plot_responses(data, 'MC', cmap='mako')
plot_responses(data, 'IN', cmap='rocket')


idx_in = data['INinds']
idx_mc = data['MCinds']
y = data['DMotAllbin'][idx_mc][..., data['takeodors']]
z = data['DMotAllbin'][idx_in][..., data['takeodors']]

y = data['DMotAlltr1bin2'][idx_mc]
z = data['DMotAlltr1bin2'][idx_in]

y = data['DMotMC']
z = data['DMotIN']

def nanmean_normalized(x):
    mu = np.nanmean(x, axis=0)
    mu = mu - np.min(mu, axis=0)
    mu = np.mean(mu, axis=1)
    mu /= np.max(mu)
    return mu

mu_y = nanmean_normalized(y)
mu_z = nanmean_normalized(z)

# time = data['time2'] * data['tbin']
time = data['time0']
fig, ax = plt.subplots(1,1, figsize=(4, 4), sharex='all', sharey='all')
cols_cells = sns.color_palette('Set1', n_colors=2)
ax.plot(time, mu_y, '.-', label='MC', color=cols_cells[1], lw=2)
ax.plot(time, mu_z, '.-', label='IN', color=cols_cells[0], lw=2)
t1 = time[data['twinstretch1']]
t2 = time[data['twinstretch2']]
# plot shaded rectangle during (t1[0], t1[-1]) and (t2[0], t2[-1])

from matplotlib.patches import Rectangle
ylim = (0, 1.05)
ax.add_patch(Rectangle((t1[0], 0), t1[-1] - t1[0], ylim[1], facecolor='k', alpha=0.2))
ax.add_patch(Rectangle((t2[0], 0), t2[-1] - t2[0], ylim[1], facecolor='k', alpha=0.2))

ax.set(ylim=ylim, xlim=(-2.2, 4.2), yticks=(0, .5, 1.), 
xticks=(-2, 0, 2, 4))
ax.legend()
sns.despine()

# number of nan rows in y and z
print(f'Num valid MC: {np.sum(~np.isnan(y), axis=0).min()}/{y.shape[0]}')
print(f'Num valid IN: {np.sum(~np.isnan(z), axis=0).min()}/{z.shape[0]}')
#%%

def id_to_idx(id_list, info):
    return info['row_idx'][info['ID'].isin(id_list)].values

import itertools
def get_w_directed(pre_id, post_id, data):
    allowable_id = data['NeuronIDs1003']
    # make sure all pre_id and post_id are in allowable_id
    assert np.all(np.isin(pre_id, allowable_id))
    assert np.all(np.isin(post_id, allowable_id))
    
    W = []
    m, n = len(pre_id), len(post_id)
    for pre, post in itertools.product(pre_id, post_id):
        pre_idx = np.where(allowable_id == pre)[0][0]
        post_idx = np.where(allowable_id == post)[0][0]
        W.append(data['W'][pre_idx][post_idx])
    
    return np.array(W).reshape((m, n)).T
    

Wmc2in = get_w_directed(data['CellinfoMC']['ID'], data['CellinfoIN']['ID'], data)
Win2mc = get_w_directed(data['CellinfoIN']['ID'], data['CellinfoMC']['ID'], data)

def normalize_columns(x):
    norm = np.linalg.norm(x, ord=2, axis=0, keepdims=True)
    norm[np.where(np.isclose(norm, 0.))] = 1.
    return x / norm

Wf = normalize_columns(Wmc2in.T)
Wb = normalize_columns(Win2mc)

with sns.plotting_context('paper'):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=600)
    vmin, vmax = 0, np.max(np.abs([Wf, Wb]))
    ax[0].imshow(Wf, aspect='auto', cmap='mako', vmin=vmin, vmax=vmax)
    ax[1].imshow(Wb, aspect='auto', cmap='mako', vmin=vmin, vmax=vmax)
    ax[0].set(title=r'${\bf W}_{IN \leftarrow MC}^\top$', ylabel='MC', xlabel='IN')
    ax[1].set(title=r'${\bf W}_{MC \leftarrow IN}$', ylabel='MC', xlabel='IN')

    sns.despine()
    fig.tight_layout()

#%%
print((Wf.sum(0)>0).sum())
print((Wb.sum(0)>0).sum())

#%%

# get n_odors x n_neurons responses, averaging within time window

def get_responses_within_time_window(x, time_window):
    return x[:, time_window[0]:time_window[1]].mean(axis=1)

def get_windowed_responses(data, cells="MC"):
    assert cells in ("MC", "IN")
    R = dict()

    t1 = data['twinstretch1']
    R[(1, 0)] = get_responses_within_time_window(data[f'DMot{cells}'], t1)
    R[(1, 1)] = get_responses_within_time_window(data[f'DMot{cells}tr1'], t1)
    R[(1, 2)] = get_responses_within_time_window(data[f'DMot{cells}tr2'], t1)

    t2 = data['twinstretch2']
    R[(2, 0)] = get_responses_within_time_window(data[f'DMot{cells}'], t2)
    R[(2, 1)] = get_responses_within_time_window(data[f'DMot{cells}tr1'], t2)
    R[(2, 2)] = get_responses_within_time_window(data[f'DMot{cells}tr2'], t2)

    return R

def plot_windowed_responses(cells="MC", cmap='mako'):
    windows_trials = [(1, 1), (1, 2), (1, 0), (2, 1), (2, 2), (2, 0)]
    R = get_windowed_responses(data, cells=cells)
    fig, ax = plt.subplots(2, 3, figsize=(12, 6), sharex='all', sharey='all')
    ax = ax.flatten()
    for i, window_trial in enumerate(windows_trials):
        ax[i].imshow(R[window_trial], aspect='auto', cmap=cmap, interpolation='none')

    odors = data['odorlist2']
    ax[0].set(xticks=range(len(odors)), xticklabels=odors)
    ax[0].set(title=f'{cells} Trial 1', ylabel=f'Time bin 1\n{cells}')
    ax[1].set(title=f'{cells} Trial 2')
    ax[2].set(title=f'{cells} Trial Avg')
    ax[3].set(ylabel=f'Time bin 2\n{cells}')
    sns.despine()
    fig.tight_layout()

plot_windowed_responses(cells="MC", cmap='mako')
plot_windowed_responses(cells="IN", cmap='rocket')

#%% compute odor x odor covariance

def cov2corr(cov):
    return cov / np.sqrt(np.diag(cov)[:, None] @ np.diag(cov)[None, :])

def plot_time_bin_covs(data, cells, correlation=True, cmap='viridis'):
    assert cells in ("MC", "IN")
    odors = data['odorlist2']

    R = get_windowed_responses(data, cells=cells)
    cov1 = np.cov(R[(1, 0)], rowvar=False)
    cov2 = np.cov(R[(2, 0)], rowvar=False)

    if correlation:
        cov1 = cov2corr(cov1)
        cov2 = cov2corr(cov2)

    vmax = np.max(np.abs([cov1, cov2]))
    vmin =  -vmax if not correlation else 0
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=300, sharex='all', sharey='all')
    sns.heatmap(cov1, square=True, cmap=cmap, vmin=vmin, vmax=vmax, cbar=False, ax=ax[0])
    sns.heatmap(cov2, square=True, cmap=cmap, vmin=vmin, vmax=vmax, cbar=False, ax=ax[1])
    # center tick labels
    ax[0].set_xticks(np.arange(len(odors)) + 0.5, minor=False)
    ax[0].set(title=f'Time bin 1 Odor x Odor {cells} CorrMat', 
        xticks=np.arange(len(odors))+.5, xticklabels=odors,
        yticks=np.arange(len(odors))+.5, yticklabels=odors)
    ax[1].set(title=f'Time bin 2 Odor x Odor {cells} CorrMat ')

plot_time_bin_covs(data, 'MC', correlation=True, cmap='viridis')
plot_time_bin_covs(data, 'IN', correlation=True, cmap='viridis')


#%%

# make response matrices that concatenate each time step and trial and odor

def get_response_matrices(data, cells="MC"):
    assert cells in ("MC", "IN")
    t1, t2 = data['twinstretch1'], data['twinstretch2']

    # trials 1 and 2
    D1 = data[f'DMot{cells}tr1'].copy()
    D2 = data[f'DMot{cells}tr2'].copy()
    D = np.stack([D1, D2], axis=-1)
    n_neurons = D.shape[0]

    # window 1
    X1 = D[:, t1[0]:t1[-1]+1]
    X1 = X1.reshape((n_neurons, -1))

    # window2
    X2 = D[:, t2[0]:t2[-1]+1]
    X2 = X2.reshape((n_neurons, -1))

    return X1, X2


def participation_ratio(d):
    l1 = np.linalg.norm(d, ord=1) 
    l2 = np.linalg.norm(d, ord=2) 
    return (l1**2) / (l2 ** 2)


cells = "MC"
Y1, Y2 = get_response_matrices(data, cells=cells)

# covariances during timebin 1 and 2
C1 = np.cov(Y1, rowvar=True)
C2 = np.cov(Y2, rowvar=True)
correlation = True

if correlation:
    vmax = np.max(np.abs([cov2corr(C1), cov2corr(C2)]))
    vmin = -vmax
else:
    vmax = np.max(np.abs([C1, C2]))
    vmin = -vmax
cmap = 'icefire'

spectrum_clip = Y1.shape[1]
spectrum_clip = 32

d1 = np.linalg.eigh(C1)[0][::-1]
d2 = np.linalg.eigh(C2)[0][::-1]

cols = sns.color_palette('Set2', 2)
# vmin = 0
with sns.plotting_context('talk'):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), dpi=600)
    if correlation:
        ax[0].imshow(cov2corr(C1), cmap=cmap, vmin=vmin, vmax=vmax)
        ax[1].imshow(cov2corr(C2), cmap=cmap, vmin=vmin, vmax=vmax)

    else:
        ax[0].imshow((C1), cmap=cmap)
        ax[1].imshow((C2), cmap=cmap)

    eigen_idx = np.arange(1, spectrum_clip+1)
    ax[2].plot(eigen_idx, d1[:spectrum_clip], '.-', label='Time bin 1', color=cols[0], lw=2)
    ax[2].plot(eigen_idx, d2[:spectrum_clip], '.-', label='Time bin 2', color=cols[1], lw=2)

    ax[0].set(xlabel=f'{cells} Neuron', ylabel=f'{cells} Neuron', title=r'Time bin 1 $\hat{\bf C}$')
    ax[1].set(xlabel=f'{cells} Neuron', ylabel=f'{cells} Neuron', title=r'Time bin 2 $\hat{\bf C}$')
    ax[2].set(yscale='linear', xlabel='Eigenvalue index', ylabel='Eigenvalue', title=r'Eigenvalues of ${\bf C}$',
        ylim=(0, None)
    )
    ax[2].legend()
    sns.despine()
    fig.tight_layout()
print(participation_ratio(d1))
print(participation_ratio(d2))

#%%    
from tqdm import tqdm

def bootstrapped_participation_ratio(data, cells, n_dims, n_bootstraps=1000):
    """Experiment to compute PR of covariance matrix of reduced population."""
    rng = np.random.default_rng(0)
    p1 = []    
    p2 = []
    cells = 'MC'
    for _ in tqdm(range(n_bootstraps)):
        # sample rows of X1 and X2 without replacement
        X1, X2 = get_response_matrices(data, cells=cells)
        n_neurons, _ = X1.shape
        idx = rng.choice(n_neurons, size=n_dims, replace=False)
        X1, X2 = X1[idx, :], X2[idx, :]

        C1 = np.cov(X1, rowvar=True)
        C2 = np.cov(X2, rowvar=True)
        d1 = np.linalg.eigh(C1)[0][::-1]
        d2 = np.linalg.eigh(C2)[0][::-1]
        p1.append(participation_ratio(d1))
        p2.append(participation_ratio(d2))
    return np.array(p1), np.array(p2)

n_dims = 20
n_bootstraps = 5000
p1, p2 = bootstrapped_participation_ratio(data, cells="MC", n_dims=n_dims, n_bootstraps=n_bootstraps)

# compute p value
dp = np.sort(p2-p1)
pval = np.searchsorted(dp, 0) / len(dp)
with sns.plotting_context('talk'):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=300)
    ax.hist(p2-p1, bins=50, alpha=0.5, label=f'p={pval:.3f}')
    ax.set(xlabel=r'$\pi_2 - \pi_1$', ylabel='Count', title=f'Bootstrapped P.R. of {n_dims} {cells} neurons')
    ax.legend()
    sns.despine()
    fig.tight_layout()


#%%
(Wmc2in.sum(0)>0).sum()
(Win2mc.sum(0)>0).sum()