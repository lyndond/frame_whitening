#%%
import copy
import itertools
import multiprocessing
import os
from typing import Any, Dict
print(os.getcwd())
print(f'num cpus: {multiprocessing.cpu_count()}')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.io as scio
import scipy.stats
import seaborn as sns
from tqdm import tqdm
#%%

def load_matfile(filename: str) -> Dict:
    def parse_mat(element: Any):
        # lists (1D cell arrays usually) or numpy arrays as well
        if element.__class__ == np.ndarray and element.dtype == np.object_ and len(element.shape) > 0:
            return [parse_mat(entry) for entry in element]

        # matlab struct
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
    
    for window in ('twinstretch1', 'twinstretch2'):
        data[window] = data[window] - 1
    
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
    to_plot = data[f'DMot{cells}'].copy()
    fig, ax = plt.subplots(2, 4, figsize=(16, 8), sharex='all', sharey='all', dpi=dpi)
    ax = ax.ravel()
    cmap = sns.color_palette(cmap, as_cmap=True)
    cmap.set_bad('k')

    if vmax is None:
        vmax = np.nanmax(to_plot)

    time = data['time0']
    for stim in range(8):
        ax[stim].imshow(to_plot[...,stim], aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        odor = data['odorlist2'][stim]
        ax[stim].set(title=f'{odor}', ylabel=f'{cells} neuron', xlabel='Time',
        xticks=np.arange(0, len(time), 100), xticklabels=np.round(time[::100], 1),
        xlim=(20, 70)
        )
                     
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
with sns.plotting_context('paper'):
    time = data['time0']
    fig, ax = plt.subplots(1,1, figsize=(4, 4), sharex='all', sharey='all', dpi=300)
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

    ax.set(ylim=ylim, 
    xlim=(-2.2, 4.2), yticks=(0, .5, 1.), 
    xticks=(-2, 0, 2, 4),
    title='Aligned odour response', xlabel='Time (s)', ylabel='Normalized response'
    )
    ax.legend()
    sns.despine()

# number of nan rows in y and z
print(f'Num valid MC: {np.sum(~np.isnan(y), axis=0).min()}/{y.shape[0]}')
print(f'Num valid IN: {np.sum(~np.isnan(z), axis=0).min()}/{z.shape[0]}')
#%%

def id_to_idx(id_list, info):
    return info['row_idx'][info['ID'].isin(id_list)].values

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

data['CellinfoMC']
# data['W']
print(data.keys())
print(data['DMotAllbin'].shape)
idx_mc = data['CellinfoMC']['row_idx'].values
idx_in = data['CellinfoIN']['row_idx'].values

fig, ax = plt.subplots(3, 4, figsize=(10, 10), dpi=100, sharex='all', sharey='all')
ax = ax.flatten()
for i in range(11):
    tmp_mc = data['DMotAll'][idx_mc][...,i] - data['DMotAll'][idx_mc][...,-1].mean(-1, keepdims=True)
    tmp_in = data['DMotAll'][idx_in][...,i] - data['DMotAll'][idx_in][...,-1].mean(-1, keepdims=True)

    tmp_mc = data['DMotAll'][idx_mc][...,i] - np.nanmean(data['DMotAll'][idx_mc][...,1], 0, keepdims=True)
    tmp_in = data['DMotAll'][idx_in][...,i] - np.nanmean(data['DMotAll'][idx_in][...,1], 0, keepdims=True)

    # ax[i].imshow(tmp , aspect='auto', cmap='icefire')
    ax[i].plot(data['time0'], np.nanmean(tmp_mc, 0), 'C0.-', lw=2)
    ax[i].plot(data['time0'], np.nanmean(tmp_in, 0), 'C1.-', lw=2)
    # add rectangles at timebin 1 and 2

    t1 = data['time0'][data['twinstretch1']]
    t2 = data['time0'][data['twinstretch2']]
    # plot shaded rectangle during (t1[0], t1[-1]) and (t2[0], t2[-1])
    ax[i].add_patch(Rectangle((t1[0], 0), t1[-1] - t1[0], .4, facecolor='k', alpha=0.2))
    ax[i].add_patch(Rectangle((t2[0], 0), t2[-1] - t2[0], .4, facecolor='k', alpha=0.2))
    ax[i].set(title=f'{data["odorlist"][i]}', xlim=(-2, 4), ylim=(0, .4))
    fig.tight_layout()


#%%

def normalize_columns(x):
    norm = np.linalg.norm(x, ord=2, axis=0, keepdims=True)
    norm[np.where(np.isclose(norm, 0.))] = 1.
    return x / norm

Wf = normalize_columns(Wmc2in.T)
Wb = normalize_columns(Win2mc)

# Wf = Wmc2in.T
# Wb = Win2mc


with sns.plotting_context('paper'):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=600)
    vmin, vmax = 0, np.max(np.abs([Wf, Wb]))
    ax[0].imshow(Wf, aspect='auto', cmap='mako', vmin=vmin, vmax=vmax)
    ax[1].imshow(Wb, aspect='auto', cmap='mako', vmin=vmin, vmax=vmax)
    ax[0].set(title=r'${\bf W}_{IN \leftarrow MC}^\top$', ylabel='MC', xlabel='IN')
    ax[1].set(title=r'${\bf W}_{MC \leftarrow IN}$', ylabel='MC', xlabel='IN')

    sns.despine()
    fig.tight_layout()


def permutation_test_symmetry(Wf, Wb, n_perm=1000, seed=42069):
    # frob norm between two matrices
    d = np.linalg.norm(Wf - Wb)

    rng = np.random.default_rng(seed)
    d_perm = []
    for _ in range(n_perm):
        idx_rows = rng.permutation(Wb.shape[0])
        idx_cols = rng.permutation(Wb.shape[1])
        Wb_perm = Wb[idx_rows][:, idx_cols]
        d_perm.append(np.linalg.norm(Wf - Wb_perm))

    p = 1 - np.sum(d_perm > d) / n_perm

    _, ax = plt.subplots(1, 1, figsize=(3, 2), dpi=100)
    ax.hist(d_perm, bins=50, density=False, label='Null')
    ax.axvline(d, color='r', ls='--', lw=2, label='Observed')
    ax.set(xlabel=r'$\Vert{\bf W}_f^\top - {\bf W}_b\Vert_{F}$', ylabel='Count', 
           title=f'Perm test symmetry: p={p:.3f}')
    ax.legend()
    sns.despine()

permutation_test_symmetry(Wf, Wb)

#%%
# get n_odors x n_neurons responses, averaging within time window

def get_responses_within_time_window(x, time_window):
    return x[:, time_window[0]:time_window[1]+1].mean(axis=1)

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

def get_response_matrices(data, cells="MC", trials=(1, 2)):
    assert cells in ("MC", "IN")
    t1, t2 = data['twinstretch1'], data['twinstretch2']

    # trials 1 and 2
    D = []
    for trial in trials:
        D.append(data[f'DMot{cells}tr{trial}'].copy())
    D = np.stack(D, axis=-1)
    D = np.squeeze(D)
    n_neurons = D.shape[0]

    # window 1
    X1 = D[:, t1[0]:t1[-1]+1]#.mean(axis=1, keepdims=True)
    X1 = X1.reshape((n_neurons, -1))

    # window2
    X2 = D[:, t2[0]:t2[-1]+1]#.mean(axis=1, keepdims=True)
    X2 = X2.reshape((n_neurons, -1))

    return X1, X2

Y1, Y2 = get_response_matrices(data, cells="MC", trials=(1, 2))
Y11, Y12 = get_response_matrices(data, cells="MC", trials=(1, ))
Y21, Y22 = get_response_matrices(data, cells="MC", trials=(2, ))

#%%

def participation_ratio(X):
    C = np.cov(X, rowvar=True)
    d, _ = np.linalg.eigh(C)
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
    ax[2].set(yscale='log', xlabel='Eigenvalue index', ylabel='Eigenvalue', title=r'Eigenvalues of ${\bf C}$',
        ylim=(1E-3, None)
    )
    ax[2].legend()
    sns.despine()
    fig.tight_layout()
print(participation_ratio(Y1))
print(participation_ratio(Y2))

#%%

def permutation_test(X, Y, n_permutations=1000, statistic=np.mean, seed=42069):
    rng = np.random.default_rng(seed)

    # compute test statistic
    test_stat = statistic(Y - X)

    # compute null distribution
    null_dist = np.zeros((n_permutations,))
    for i in range(n_permutations):
        # permute labels
        idx0 = rng.permutation(len(X)) 
        
        # compute statistic
        null_dist[i] = statistic(Y - (X[idx0]))

    # compute p-value
    p = np.sum(np.abs(test_stat) > np.abs(null_dist)) / n_permutations

    return p, test_stat, null_dist

GZ1, GZ2 = get_response_matrices(data, cells="IN", trials=(1, 2))

#%%
def logvar(X, axis=1, eps=1E-6):
    return (X.var(axis=1) + eps)

with sns.plotting_context('paper'):
    Z1 = Wf.T @ Y1
    Z2 = Wf.T @ Y2
    Z1 = GZ1
    Z2 = GZ2

    # lims = (-6, 0)
    lims = (0, .3)

    eps = 1E-6
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=200)
    ax[0].plot(logvar(Z1), logvar(Z2), '.k', label='Trial 1&2')
    ax[0].plot(lims, lims, 'k--')
    ax[0].set(xlabel='Time bin 1 Z variance', ylabel='Time bin 2 Z variance', yscale='linear', xscale='linear',
    xlim=lims, ylim=lims,
    )

    ax[1].hist(logvar(Z1), bins=50, alpha=0.5, density=True)
    ax[1].hist(logvar(Z2), bins=50, alpha=0.5, density=True)
    sns.kdeplot(logvar(Z1), ax=ax[1], color='C0', lw=2, label='Time bin 1')
    sns.kdeplot(logvar(Z2), ax=ax[1], color='C1', lw=2, label='Time bin 2')
    ax[1].set(xlabel='Z variance', ylabel='Density', yscale='linear', xscale='linear', xlim=lims)
    ax[1].legend()

    iqr1 = scipy.stats.iqr(logvar(Z1))
    iqr2 = scipy.stats.iqr(logvar(Z2))

    # add text
    ax[1].text(.75, 0.8, f'IQR: {iqr1:.2f}', transform=ax[1].transAxes, va='top', color="C0")
    ax[1].text(.75, 0.75, f'IQR: {iqr2:.2f}', transform=ax[1].transAxes, va='top', color="C1")

    sns.despine()
    fig.tight_layout

def permutation_test_iqr_logvar(X, Y, n_perm=1000, seed=0):
    rng = np.random.default_rng(seed)
    logvar1 = logvar(X)
    logvar2 = logvar(Y)

    iqr1 = scipy.stats.iqr(logvar1)
    iqr2 = scipy.stats.iqr(logvar2)
    diff = iqr1 - iqr2

    iqr_diffs = []

    all_iqr = np.concatenate([logvar1, logvar2])
    n = len(all_iqr)
    for _ in range(n_perm):
        idx = rng.permutation(n)
        all_iqr = all_iqr[idx]

        iqr_diffs.append(
            scipy.stats.iqr(all_iqr[:n//2]) - scipy.stats.iqr(all_iqr[n//2:])
            )
    
    pval = np.sum(np.abs(iqr_diffs) > np.abs(diff)) / n_perm
    pval = 1 - np.sum(diff>iqr_diffs) / n_perm
    _, ax = plt.subplots(1, 1, figsize=(3,3), dpi=100)
    ax.hist(iqr_diffs, bins=50, alpha=0.5, label='Null distribution')
    ax.axvline(diff, color='r', lw=2, label='Observed')
    ax.set(xlabel='IQR difference', ylabel='Count')
    # put pval in text
    ax.text(0.05, 0.95, f'p-value: {pval:.2f}', transform=ax.transAxes, va='top', ha='left', color="r")

    ax.legend(loc='lower left')
    sns.despine()

    return ax 

ax_inputs = permutation_test_iqr_logvar(Z1, Z2, n_perm=10_000, seed=0)
ax_outputs = permutation_test_iqr_logvar(GZ1, GZ2, n_perm=10_000, seed=0)

ax_inputs.set(title=r'Input variance: IQR time 2 - IQR time 1')
ax_outputs.set(title=r'Output variance: IQR time 2 - IQR time 1')

#%%
fig, ax = plt.subplots(1, 1)
tt = logvar(GZ2)-logvar(GZ1)
ax.hist(tt, bins=50, alpha=0.5, color='k', density=True)
sns.kdeplot(tt, ax=ax, color='k', lw=2, label='Time bin 1')
ax.set(xlabel='Output var time 2 - Output var time 1', ylabel='Density', yscale='linear', xscale='linear')

#%%
np.linalg.norm(Wf, axis=0)

#%%
def moment_k_cycles(Y: np.ndarray, k: int) -> np.ndarray:
    """Unbiased estimator of k moments of population spectrum.

    Kong and Valiant, "Spectrum estimation from samples", Annals of Stats. 2017.
    """
    assert k > 0, "k must be positive integer."

    d, n = Y.shape
    A = Y.T @ Y  # n x n 
    
    G = np.triu(A, k=1)  # upper tri of A, diag and lower tri set to zero
    Gi = np.eye(A.shape[0])
    moments = []
    for _ in range(1, k+1):
        nchoosek = sp.special.comb(n, _)
        moments.append(np.trace(Gi @ A) / (d * nchoosek))
        Gi = Gi @ G
    return np.array(moments)

def participation_ratio2(X):
    """Ratio of sum of eigenvalues squared to the sum of squared eigenvalues."""
    d, _ = X.shape
    moments = moment_k_cycles(X, 2)
    moments = moments * d
    return  moments[0]**2 / moments[1]

print(participation_ratio2(Y1), participation_ratio2(Y2))
print(participation_ratio(Y1), participation_ratio(Y2))
moment_k_cycles(Y1, 2)
moment_k_cycles(Y2, 2)

#%%    

def permutation_participation_ratio(data, cells, n_perms=1000):
    """Experiment to compute PR of covariance matrix of reduced population."""
    rng = np.random.default_rng(0)
    cells = 'MC'
    X1, X2 = get_response_matrices(data, cells=cells)
    _, n_samples = X1.shape

    # compute participation ratio of original data
    pr01 = participation_ratio2(X1)
    pr02 = participation_ratio2(X2)
    dp0 = pr02 - pr01
    dp = []
    # permutation test
    X = np.concatenate([X1, X2], axis=1)  # concatenate all observations
    for _ in tqdm(range(n_perms)):
        # sample columns of X1 and X2 without replacement
        idx = rng.permutation(2*n_samples)
        idx1, idx2 = idx[:n_samples], idx[n_samples:]
        X1, X2 = X[:, idx1], X[:, idx2]
        dp.append(participation_ratio2(X2) - participation_ratio2(X1))

    return dp0, np.array(dp)


n_perms = 5000
dp0, dp = permutation_participation_ratio(data, cells="MC", n_perms=n_perms)

#compute p value
pval = 1 - np.searchsorted(dp, dp0) / len(dp)
with sns.plotting_context('talk'):
    fig, ax = plt.subplots(1, 1, figsize=(15, 5), dpi=300)
    ax.vlines(dp0, 0, 100, label='Observed', color='k', lw=2)
    ax.hist(dp, bins=50, alpha=0.5, label=f'Null distribution: p={pval:.3f}')
    ax.set(xlabel=r'$\pi$', ylabel='Count', title=r'Bootstrapped $\Delta \pi$ ' f'of N neurons')
    ax.set(xlabel=r'$\pi_2 - \pi_1$')
    ax.legend()
    sns.despine()
    fig.tight_layout()


#%%

# solve for g
# Wb @ diag(g) @ Wf.T @ Y2 = Y2 - Y1
