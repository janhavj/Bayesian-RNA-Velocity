from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(42)

NOISE_SD = 0.5
N_DRAWS = 2000

TRUE_PARAMETERS = np.array([5.0, 0.8, 0.4])

# --------------------------------------------------
# 2. Load observations and posterior samples
# --------------------------------------------------

data = pd.read_csv(
    DATA_DIR / "synthetic_rna_observations.csv"
)

posterior = pd.read_csv(
    RESULTS_DIR / "posterior_samples.csv"
)

times = data["time"].to_numpy()

observations = data[
    ["unspliced", "spliced"]
].to_numpy()

parameters = posterior[
    ["alpha", "beta", "gamma"]
].to_numpy()

# --------------------------------------------------
# 3. Analytical RNA kinetic model
# --------------------------------------------------

from rna_kinetics import simulate


# --------------------------------------------------
# 4. Select posterior draws
# --------------------------------------------------

indices = rng.choice(
    len(parameters),
    size=N_DRAWS,
    replace=False
)

selected_parameters = parameters[indices]

dense_times = np.linspace(
    times.min(),
    times.max(),
    200
)

# --------------------------------------------------
# 5. Generate posterior trajectories
# --------------------------------------------------

latent_draws = np.array([
    simulate(theta, dense_times)
    for theta in selected_parameters
])

predictive_draws = (
    latent_draws
    + rng.normal(
        0,
        NOISE_SD,
        size=latent_draws.shape
    )
)

latent_mean = latent_draws.mean(axis=0)

latent_lower, latent_upper = np.percentile(
    latent_draws,
    [2.5, 97.5],
    axis=0
)

predictive_lower, predictive_upper = np.percentile(
    predictive_draws,
    [2.5, 97.5],
    axis=0
)

true_trajectory = simulate(
    TRUE_PARAMETERS,
    dense_times
)

# --------------------------------------------------
# 6. Predictions at observed time points
# --------------------------------------------------

observed_time_draws = np.array([
    simulate(theta, times)
    for theta in selected_parameters
])

observed_predictive_draws = (
    observed_time_draws
    + rng.normal(
        0,
        NOISE_SD,
        size=observed_time_draws.shape
    )
)

observed_mean = observed_time_draws.mean(axis=0)

obs_lower, obs_upper = np.percentile(
    observed_predictive_draws,
    [2.5, 97.5],
    axis=0
)

true_observed = simulate(
    TRUE_PARAMETERS,
    times
)

# --------------------------------------------------
# 7. Calculate evaluation metrics
# --------------------------------------------------

metrics = []

for i, species in enumerate(
    ["unspliced", "spliced"]
):

    residuals = (
        observations[:, i] - observed_mean[:, i]
    )

    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))

    coverage = np.mean(
        (observations[:, i] >= obs_lower[:, i])
        & (observations[:, i] <= obs_upper[:, i])
    )

    true_rmse = np.sqrt(
        np.mean(
            (
                true_observed[:, i]
                - observed_mean[:, i]
            ) ** 2
        )
    )

    latent_coverage = np.mean(
        (true_trajectory[:, i] >= latent_lower[:, i])
        & (true_trajectory[:, i] <= latent_upper[:, i])
    )

    metrics.append({
        "rna_species": species,
        "observed_rmse": rmse,
        "observed_mae": mae,
        "ground_truth_rmse": true_rmse,
        "predictive_coverage": coverage,
        "latent_trajectory_coverage": latent_coverage
    })

metrics_df = pd.DataFrame(metrics)

metrics_df.to_csv(
    RESULTS_DIR / "posterior_predictive_metrics.csv",
    index=False
)

# --------------------------------------------------
# 8. Save predicted trajectories
# --------------------------------------------------

trajectory_rows = []

for i, species in enumerate(
    ["unspliced", "spliced"]
):

    for j, time in enumerate(dense_times):

        trajectory_rows.append({
            "time": time,
            "rna_species": species,
            "ground_truth": true_trajectory[j, i],
            "posterior_mean": latent_mean[j, i],
            "latent_lower": latent_lower[j, i],
            "latent_upper": latent_upper[j, i],
            "predictive_lower": predictive_lower[j, i],
            "predictive_upper": predictive_upper[j, i]
        })

pd.DataFrame(trajectory_rows).to_csv(
    RESULTS_DIR / "posterior_predictive_trajectories.csv",
    index=False
)

# --------------------------------------------------
# 9. Posterior predictive visualisation
# --------------------------------------------------

fig, axes = plt.subplots(
    1, 2,
    figsize=(14, 5),
    sharex=True
)

species_names = ["Unspliced RNA", "Spliced RNA"]
colours = ["tab:blue", "tab:orange"]

for i, ax in enumerate(axes):

    colour = colours[i]

    ax.fill_between(
        dense_times,
        predictive_lower[:, i],
        predictive_upper[:, i],
        color=colour,
        alpha=0.12,
        label="95% predictive interval"
    )

    ax.fill_between(
        dense_times,
        latent_lower[:, i],
        latent_upper[:, i],
        color=colour,
        alpha=0.35,
        label="95% latent credible interval"
    )

    ax.plot(
        dense_times,
        latent_mean[:, i],
        color=colour,
        linewidth=2,
        label="Posterior mean"
    )

    ax.plot(
        dense_times,
        true_trajectory[:, i],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Ground truth"
    )

    ax.scatter(
        times,
        observations[:, i],
        color=colour,
        edgecolor="black",
        linewidth=0.4,
        s=30,
        label="Observations"
    )

    ax.set_title(species_names[i])
    ax.set_xlabel("Time (arbitrary units)")
    ax.set_ylabel("RNA abundance (arbitrary units)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

fig.suptitle(
    "Bayesian Posterior Predictive RNA Dynamics"
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "posterior_predictive_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

# --------------------------------------------------
# 10. Residual analysis
# --------------------------------------------------

residuals = observations - observed_mean

fig, ax = plt.subplots(figsize=(9, 5))

for i, species in enumerate(species_names):

    ax.scatter(
        times,
        residuals[:, i],
        color=colours[i],
        label=species,
        alpha=0.8
    )

ax.axhline(
    0,
    color="black",
    linestyle="--"
)

ax.axhline(
    1.96 * NOISE_SD,
    color="grey",
    linestyle=":"
)

ax.axhline(
    -1.96 * NOISE_SD,
    color="grey",
    linestyle=":"
)

ax.set_xlabel("Time (arbitrary units)")
ax.set_ylabel("Observed - predicted abundance")
ax.set_title("Posterior Mean Prediction Residuals")
ax.legend()
ax.grid(alpha=0.2)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "posterior_predictive_residuals.png",
    dpi=300
)

plt.close(fig)

# --------------------------------------------------
# 11. Report results
# --------------------------------------------------

print("\nPosterior Predictive Analysis")
print("--------------------------------------")
print(f"Posterior draws: {N_DRAWS}")
print(f"Observed time points: {len(times)}")
print(f"Prediction time points: {len(dense_times)}")

print("\nEvaluation metrics:")
print(metrics_df.round(4).to_string(index=False))

print("\nPosterior predictive analysis completed.")
print("Results saved to:", RESULTS_DIR)
print("Figures saved to:", FIGURE_DIR)
