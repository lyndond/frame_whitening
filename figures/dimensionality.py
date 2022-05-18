#%%
"""
Plots the effect of using Grassmannian vs Random Normal frame.
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os

print(os.getcwd())
#%%

# df_sim = pd.read_csv("outputs/2022_05_17/23_20_43/dim_experiment_results.csv")
# df_sim.frame[df_sim.frame == "GRASSMAN"] = "Grassmann"  # spelling mistake
# df_sim.frame[df_sim.frame == "RANDN"] = "Normal"

df_sim = pd.read_csv("outputs/2022_05_18/10_03_22/dim_experiment_results.csv")
df_sim.rename(columns={"frame": "Frame"}, inplace=True)

error = df_sim.error
df_sim.error_trace = df_sim.error_trace.apply(lambda x: np.sqrt((x) ** 2))
#%%

with sns.plotting_context("paper", font_scale=1.5):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=300)
    sns.stripplot(
        x="n",
        y="error_fro",
        data=df_sim,
        hue="Frame",
        palette="Set1",
        ax=ax,
        alpha=1.0,
        **{"edgecolor": "none"},
    )

    # Axis
    n = max(df_sim.n)
    N = np.arange(0, n, 5)
    K = (N + 5) * ((N + 5) + 1) // 2
    xticklabels = [f"{n+5} ({k})" for n, k in zip(N, K)]
    ax.set(
        ylabel=r"$\frac{1}{N^2}\parallel{\bf I - C}_{yy} \parallel_F^2$",
        # ylabel=r"$\parallel \frac{1}{N}\operatorname{Tr}({\bf C}_{yy} - {\bf I})\parallel$",
        # ylabel=r"\mathcal{B}({\bf I, \bf C}_{yy})",
        xlabel=r"dim$\bf y$ (dim$\bf z$)",
        ylim=(1e-4, 1e1),
        xticks=N,
        xticklabels=xticklabels,
        yscale="log",
    )
    sns.despine()

#%%
fig.savefig(
    "figures/fig_outputs/dim_experiment.png",
    dpi=300,
    bbox_inches="tight",
    transparent=True,
)

#%%


error = (
    df_sim["error"]
    .apply(
        lambda x: np.fromstring(
            x.replace("\n", "").replace("[", "").replace("]", "").replace("  ", " "),
            sep=" ",
        )
    )
    .values
)


#%%
