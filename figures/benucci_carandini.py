#%%
import copy
import multiprocessing
import os
from typing import Any, Dict, Optional
from pathlib import Path

print(os.getcwd())
print(f'num cpus: {multiprocessing.cpu_count()}')

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp
import scipy.io as scio
from scipy.io import loadmat
import scipy.stats
import seaborn as sns
from sklearn import preprocessing
from tqdm import tqdm

#%%

def get_benucci_data(
    data_path=Path("data/benucci_carandini/BenucciTuningData.mat"),
    subset: str = "all",
):
    data_v1 = loadmat(data_path)
    all_probs = 0.01 * np.array([50, 35, 35, 50, 50, 50, 50, 40, 40, 30, 30.0])
    assert subset in ["all", "low", "high"]
    if subset == "all":
        prob_idx = range(len(all_probs))
    elif subset == "low":
        prob_idx = [1, 2, 7, 8, 9, 10]  # 30-40% adaptor prob
    else:  # high
        prob_idx = [0, 3, 4, 5, 6]  # 50% adaptor prob

    tc_unadapted = []
    tc_adapted = []

    for i in prob_idx:
        tc_unadapted.append(data_v1["resps"][i][0][..., 0])
        tc_adapted.append(data_v1["resps"][i][0][..., 1])

    all_tc_unadapted = np.stack(tc_unadapted, -1)
    all_tc_adapted = np.stack(tc_adapted, -1)
    tuning_curves = np.stack((all_tc_unadapted, all_tc_adapted), -1)

    oris_stims = np.linspace(-90, 90, 21) * np.pi / 180
    oris_units = np.linspace(-90, 90, 13) * np.pi / 180
    return tuning_curves, oris_stims, oris_units, all_probs[prob_idx]

def normalize_pdf(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Normalize probability distribution"""
    assert p.ndim == 1 or p.shape[1] == 1, "assumes p is flat vec or col vec"
    return p / np.sum(p)

def get_benucci_prob(
    prob_adaptor: float, n_stim_bins: int = 21, peak: float = 0.5
) -> npt.NDArray[np.float64]:
    uniform = (1 - prob_adaptor) / (n_stim_bins - 3)
    p = np.ones(n_stim_bins) * uniform

    sides = 1 - peak
    p[n_stim_bins // 2] = prob_adaptor * peak
    p[n_stim_bins // 2 - 1] = prob_adaptor * sides / 2
    p[n_stim_bins // 2 + 1] = prob_adaptor * sides / 2
    p = normalize_pdf(p)
    return p


def get_idx(oris: npt.NDArray[np.float64], oris_data: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
    idx = [np.argmin((oris - oris_data[i]) ** 2) for i in range(len(oris_data))]
    idx = np.array(idx)
    return idx

#%% stats

def get_oris(
    k: int, oris_extra: Optional[npt.NDArray[np.float64]] = None
) -> npt.NDArray[np.float64]:
    """Return centered orientation labels
    Parameters
    ----------
    k: number of steps between [0, pi)
    oris_extra: k+1 steps between 0 and pi; Defaults to [0, pi]
    Returns
    -------
    oris: Tensor of orientations
    """

    if oris_extra is None:
        oris_extra = np.linspace(0, np.pi, k + 1)
    oris = oris_extra[:k]
    center = int(k // 2)
    ori_center = oris[center]
    oris = oris - ori_center + np.pi / 2  # makes center exactly pi/2

    return oris


def weighted_outer(r: npt.NDArray[np.float64], p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute expected pair-wise response products"""
    C = r @ np.diag(np.squeeze(p)) @ r.T  # compute response products
    return C


def cov(r: npt.NDArray[np.float64], p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute expected covariance matrix"""
    mu = r @ p
    C = weighted_outer(r, p) - np.outer(mu, mu)
    return C


def corr(r: npt.NDArray[np.float64], p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute expected correlation matrix"""
    C = cov(r, p)
    inv_std = np.diag(1 / np.sqrt(np.diag(C)))
    C0 = inv_std @ C @ inv_std
    return C0


def corr_matrix(
    X: npt.NDArray[np.float64], p: npt.NDArray[np.float64] = None, var: npt.NDArray[np.float64] = None
) -> npt.NDArray[np.float64]:
    """Covariance scaled by variance from possibly different data"""

    _, n_inputs = X.shape
    if p is None:
        p = np.ones(n_inputs) / n_inputs
    mu = (X @ p)[:, np.newaxis]
    outer_mean = mu @ mu.T
    cov = X @ np.diag(p) @ X.T - outer_mean

    if var is None:
        inv_std = np.diag(1 / np.sqrt(np.diag(cov)))
    else:
        inv_std = np.diag(1 / np.sqrt(var))
    corr_mat = inv_std @ cov @ inv_std
    return corr_mat

def normalize_fr(X: npt.NDArray[np.float64], Z: npt.NDArray[np.float64] = None) -> npt.NDArray[np.float64]:
    """Rescale baseline and max of X, optionally relative to Z"""
    if Z is None:
        Z = X
    z_min = np.min(Z, axis=1, keepdims=True)
    Y = X - z_min  # baseline subtract each unit
    z_max = (Z - z_min).max(axis=1, keepdims=True)  # compute amplitude of each unit
    Y /= z_max  # scale by amplitude of each unit
    return Y

#%%

def plot_covariances(
    resps_data: npt.NDArray[np.float64],
    resps_model: npt.NDArray[np.float64],
    prob_data: npt.NDArray[np.float64],
    prob_model: npt.NDArray[np.float64],
):

    all_tc_unadapted, all_tc_adapted = resps_data[..., 0], resps_data[..., 1]
    tc_model, tc_model_adapt = resps_model[..., 0], resps_model[..., 1]

    mean_data = all_tc_unadapted.mean(-1)
    mean_data_adapt = all_tc_adapted.mean(-1)

    tc_model = tc_model.mean(-1)
    tc_model_adapt = tc_model_adapt.mean(-1)

    # fmt: off
    # covariances
    with sns.plotting_context("paper", font_scale=1.):
        fig, ax = plt.subplots(
            2, 3, sharex="all", sharey="all", figsize=(8, 5),  # dpi=300
        )
        cbar = False
        cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
        clims0 = (-1, 1.0)
        cmap = "icefire"
        # shared params across all heatmaps
        kwargs = {
            "cmap": cmap,
            "vmin": clims0[0],
            "vmax": clims0[1],
            "square": True,
            "cbar": cbar,
        }

        # data
        sns.heatmap(
            corr_matrix(mean_data, None, var=np.diag(np.cov(mean_data))),
            ax=ax[0, 0], **kwargs)

        sns.heatmap(
            corr_matrix(mean_data, prob_data, var=np.diag(np.cov(mean_data))),
            ax=ax[0, 1], **kwargs)

        sns.heatmap(
            corr_matrix(mean_data_adapt, p=prob_data, var=np.diag(np.cov(mean_data))),
            ax=ax[0, 2], **kwargs,
            xticklabels=[-90] + 5 * [""] + [0] + 5 * [""] + [90],
            yticklabels=[-90] + 5 * [""] + [0] + 5 * [""] + [90])

        # model
        sns.heatmap(
            corr_matrix(tc_model, None, var=np.diag(np.cov(tc_model))),
            ax=ax[1, 0], **kwargs,)

        sns.heatmap(
            corr_matrix(tc_model, prob_model, var=np.diag(np.cov(tc_model))),
            ax=ax[1, 1], **kwargs,)

        kwargs.update({"cbar": True})
        sns.heatmap(
            corr_matrix(
                tc_model_adapt, p=prob_model, var=np.diag(np.cov(tc_model))
            ),
            ax=ax[1, 2],
            **kwargs,
            xticklabels=[-90] + 5 * [""] + [0] + 5 * [""] + [90],
            yticklabels=[-90] + 5 * [""] + [0] + 5 * [""] + [90],
            cbar_ax=cbar_ax,
            cbar_kws={
                "orientation": "vertical",
                "ticks": (-1, 0, 1),
                "label": "Normalized covariance",
            },
        )
        ax[0, 0].set(ylabel="Data \n Preferred ori (deg)", title="Uniform")
        ax[1, 0].set(ylabel="Model \n Preferred ori (deg)", xlabel="Preferred ori (deg)")
        ax[0, 1].set(title="Biased - No adapt")
        ax[0, 2].set(title="Biased - Adapt")
        sns.despine(fig)
    # fmt: on
    return 

# %%
def process_responses(subset, normalize_data=True):

    resps_data, _, _, all_probs = get_benucci_data(subset=subset)

    if normalize_data:
        x = resps_data[..., 1]
        x0 = resps_data[..., 0]
        num_sessions = x.shape[-1]
        X0 = np.stack(
            [normalize_fr(x0[..., i], x0[..., i]) for i in range(num_sessions)],
            -1,
        )
        X = np.stack(
            [normalize_fr(x[..., i], x0[..., i]) for i in range(num_sessions)], -1
        )
        resps_data = np.stack([X0, X], -1)

    prob21 = get_benucci_prob(np.mean(all_probs), 21)

    return resps_data, prob21, all_probs

resps_data, prob21, all_probs = process_responses(subset="all")
oris_stims_deg = np.rad2deg(oris_stims)
oris_units_deg = np.rad2deg(oris_units)

# n_inputs = 511
# ori_inputs = get_oris(n_inputs)
# ori_inputs_deg = np.rad2deg(ori_inputs) - 90.0
# idx_inputs = get_idx(ori_inputs_deg, oris_stims_deg)
# idx_units = get_idx(ori_prefs_deg, oris_units_deg)
# prob21_model = prob_s[idx_inputs, 0]
# prob21_model /= prob21_model.sum()
prob0 = np.ones_like(prob21) / len(prob21)

all_tc_unadapted, all_tc_adapted = resps_data[..., 0], resps_data[..., 1]

X0 = all_tc_unadapted.mean(-1)
X1 = all_tc_adapted.mean(-1)
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

def heatmap(X, prob, ax, **kwargs):
    C = cov(X, prob)
    vmax = np.abs(C).max()
    sns.heatmap(C, ax=ax, vmin=-vmax, vmax=vmax, cmap="icefire", **kwargs)

heatmap(X0, prob0, ax[0], square=True, cbar=False)
heatmap(X0, prob21, ax[1], square=True, cbar=False)
heatmap(X1, prob21, ax[2], square=True, cbar=False)

# %%