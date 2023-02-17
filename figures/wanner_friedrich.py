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
    
    cell_info_array = data['Cellinfo2']
    data.pop('Cellinfo')
    data['Cellinfo2'] = pd.DataFrame.from_dict({
    'ID': cell_info_array[:,0].astype(int),
    'row_idx': cell_info_array[:,1].astype(int) - 1,
    'cell_type': cell_info_array[:,2].astype(int),
    'microcluster': cell_info_array[:,3].astype(int),
    'GlomID': cell_info_array[:,4].astype(int),
    'length': cell_info_array[:,5].astype(float),
    })

    return data

print(data0.keys())

data = process_data(data0)
info = data['Cellinfo2']

#%%

fig, ax = plt.subplots(4, 2, figsize=(10, 10), sharex='all', sharey='all')
ax = ax.ravel()

to_plot = data['DMotAlltr1bin2']
cmap = sns.color_palette('mako', as_cmap=True)
cmap.set_bad('k')
idx = data['MCinds']
for stim in range(8):
    ax[stim].imshow(to_plot[idx][...,stim], aspect='auto', cmap=cmap)
    odor = data['takeodors'][stim]
    odor = data['odorlist'][odor]
    ax[stim].set(title=f'{odor}', ylabel='Neuron', xlabel='Time')
sns.despine()
fig.tight_layout()

idx_in = data['INinds']
idx_mc = data['MCinds']
y = data['DMotAllbin'][idx_mc][..., data['takeodors']]
z = data['DMotAllbin'][idx_in][..., data['takeodors']]

y = data['DMotAlltr1bin2'][idx_mc]
z = data['DMotAlltr1bin2'][idx_in]

def nanmean_normalized(x):
    mu = np.nanmean(x, axis=0)
    mu = mu - np.min(mu, axis=0)
    mu = np.mean(mu, axis=1)
    mu /= np.max(mu)
    return mu

# convolve 1D signals with a gaussian of width 10 time bins
mu_y = nanmean_normalized(y)
mu_z = nanmean_normalized(z)

# time = data['time2'] * data['tbin']
time = data['time0']
fig, ax = plt.subplots(1,1, figsize=(4, 4), sharex='all', sharey='all')
ax.plot(time, mu_y, label='MC', color='k', lw=2)
ax.plot(time, mu_z, label='IN', color='C4', lw=2)
t1 = time[data['twinstretch1']]
t2 = time[data['twinstretch2']]
# plot shaded rectangle during (t1[0], t1[-1]) and (t2[0], t2[-1])

from matplotlib.patches import Rectangle
ax.add_patch(Rectangle((t1[0], 0), t1[-1] - t1[0], 1, facecolor='k', alpha=0.2))
ax.add_patch(Rectangle((t2[0], 0), t2[-1] - t2[0], 1, facecolor='k', alpha=0.2))

ax.set(ylim=(0, 1), xlim=(-2.2, 4.2), yticks=(0, .5, 1.), 
xticks=(-2, 0, 2, 4))
ax.legend()
sns.despine()

# number of nan rows in y and z

print(f'{np.sum(~np.isnan(y), axis=0).min()}/{y.shape[0]}')
print(f'{np.sum(~np.isnan(z), axis=0).min()}/{z.shape[0]}')
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
        pre_idx = np.where(data['NeuronIDs1003'] == pre)[0][0]
        post_idx = np.where(data['NeuronIDs1003'] == post)[0][0]
        W.append(data['W'][pre_idx, post_idx])
    
    return np.array(W).reshape((m, n)).T
    

Wmc2in = get_w_directed(data['MCindsID'], data['INindsID'], data)
Win2mc = get_w_directed(data['INindsID'], data['MCindsID'], data)

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