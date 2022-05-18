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
import submitit
import time
import pathlib
from pathlib import Path

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


def simulate_many(n, eta_g, n_repeats, n_batch, batch_size, frame):
    k = n * (n + 1) // 2
    print(f"n = {n}, k = {k}")

    all_error = []

    pbar = tqdm(range(n_repeats))
    V, _ = np.linalg.qr(np.random.randn(n, n))
    s = np.linspace(1, 5, n) + np.random.randn(n) * 0.1
    Cxx = V @ np.diag(s) @ V.T
    Lxx = np.linalg.cholesky(Cxx)
    kappa0 = np.linalg.cond(Cxx)
    df_sim = pd.DataFrame(
        columns=["n", "k", "n_batch", "batch_size", "kappa0", "error", "frame"],
    )
    for _ in pbar:
        if frame == "GRASSMAN":
            W, G, res = fw.get_grassmanian(n, k, niter=400)
        elif frame == "RANDN":
            W = np.random.randn(n, k)
            W = fw.normalize_frame(W)

        pbar.set_postfix({"W_init": True})

        err = simulate(Lxx, W, eta_g, n_batch, batch_size)
        df_tmp = pd.DataFrame(
            [
                {
                    "n": n,
                    "k": k,
                    "n_batch": n_batch,
                    "batch_size": batch_size,
                    "kappa0": float(kappa0),
                    "error": float(err[-200:].mean()),
                    "frame": frame,
                }
            ],
        )
        df_sim = df_sim.append([df_sim, df_tmp], ignore_index=True)
        all_error.append(err)

    return df_sim


def submit():
    timestamp = pd.Timestamp.now().strftime("%Y%m%d/%H_%M_%S")
    output_path = Path(f"outputs/{timestamp}")
    output_path.mkdir(parents=True, exist_ok=True)
    df_sim = pd.DataFrame()
    all_n = np.arange(5, 25)
    n_batch = 2000
    batch_size = 512
    n_repeats = 20
    eta_g = 5e-3

    print(f"running jobs into {output_path}")

    print("settin slurm")
    # set up slurm
    slurm_path = f"slurm/dimensionality/{timestamp}/%j"
    executor = submitit.AutoExecutor(folder=slurm_path)
    executor.update_parameters(
        mem_gb=64,
        timeout_min=1200,
        cpus_per_task=1,
        tasks_per_node=1,
        slurm_partition="ccn",
        slurm_array_parallelism=128,
    )

    jobs = []
    with executor.batch():
        for frame in ["GRASSMAN", "RANDN"]:
            for n in all_n:
                job = executor.submit(
                    simulate_many,
                    n,
                    eta_g,
                    n_repeats,
                    n_batch,
                    batch_size,
                    frame,
                )
                jobs.append(job)
    n_jobs = len(jobs)
    print(f"you are running {n_jobs} jobs 😅")

    print(f"waiting for jobs to finish, then consolidate")
    pbar = tqdm(total=n_jobs)
    prev_finished, num_finished = 0, 0
    idx_finished = set({})

    df_sim = pd.DataFrame(
        columns=["n", "k", "n_batch", "batch_size", "kappa0", "error", "frame"],
    )
    while num_finished != n_jobs:
        for j, job in enumerate(jobs):
            if job.done() and j not in idx_finished:
                idx_finished.add(j)
                df_out = job.result()
                df_sim = pd.concat([df_sim, df_out], ignore_index=True)
        num_finished = len(idx_finished)
        if num_finished > prev_finished:
            pbar.update(num_finished - prev_finished)
            prev_finished = num_finished
        time.sleep(5.0)

    df_sim.to_csv(output_path / "dim_experiment_results.csv", index=False)


# with sns.plotting_context("paper", font_scale=1.5):
#     fig, ax = plt.subplots(1, 1, dpi=300)
#     sns.scatterplot(x="n", y="error", data=df_sim, ax=ax)
#     ax.set(ylim=(1e-4, 1e-0), yscale="log", xlabel="N", ylabel="Error")
#     sns.despine()
# fig.savefig("figures/dim_experiment.png")
# df_sim.to_csv("experiments/dim_experiment.csv")

# #%%
# err_to_plot = np.stack(all_error, -1)
# fig, ax = plt.subplots(1, 1)
# ax.plot(err_to_plot)
# ax.set(yscale="log")


if __name__ == "__main__":
    submit()
