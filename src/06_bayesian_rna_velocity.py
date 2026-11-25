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

N_DRAWS = 2000

TRUE_PARAMETERS = np.array([5.0, 0.8, 0.4])

times = np.linspace(0, 20, 400)

# --------------------------------------------------
# 2. Load posterior samples
# --------------------------------------------------

posterior = pd.read_csv(
    RESULTS_DIR / "posterior_samples.csv"
)

parameters = posterior[
    ["alpha", "beta", "gamma"]
].to_numpy()

indices = rng.choice(
    len(parameters),
    size=N_DRAWS,
    replace=False
)

selected_parameters = parameters[indices]

# --------------------------------------------------
# 3. Analytical RNA kinetic model
# --------------------------------------------------

from rna_kinetics import simulate


# --------------------------------------------------
# 4. Calculate RNA velocities
# --------------------------------------------------

from rna_kinetics import calculate_velocity


# --------------------------------------------------
# 5. Generate posterior velocity distributions
# --------------------------------------------------

velocity_draws = np.array([
    calculate_velocity(theta, times)
    for theta in selected_parameters
])

velocity_mean = velocity_draws.mean(axis=0)

velocity_lower, velocity_upper = np.percentile(
    velocity_draws,
    [2.5, 97.5],
    axis=0
)

true_velocity = calculate_velocity(
    TRUE_PARAMETERS,
    times
)

# --------------------------------------------------
# 6. Calculate evaluation metrics
# --------------------------------------------------

metrics = []

for i, species in enumerate(
    ["unspliced", "spliced"]
):

    errors = (
        velocity_mean[:, i]
        - true_velocity[:, i]
    )

    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))

    coverage = np.mean(
        (true_velocity[:, i] >= velocity_lower[:, i])
        & (true_velocity[:, i] <= velocity_upper[:, i])
    )

    metrics.append({
        "rna_species": species,
        "velocity_rmse": rmse,
        "velocity_mae": mae,
        "pointwise_credible_interval_coverage": coverage,
        "final_true_velocity": true_velocity[-1, i],
        "final_posterior_mean_velocity": velocity_mean[-1, i]
    })

metrics_df = pd.DataFrame(metrics)

metrics_df.to_csv(
    RESULTS_DIR / "rna_velocity_metrics.csv",
    index=False
)

# --------------------------------------------------
# 7. Save velocity trajectories
# --------------------------------------------------

rows = []

for i, species in enumerate(
    ["unspliced", "spliced"]
):

    for j, time in enumerate(times):

        rows.append({
            "time": time,
            "rna_species": species,
            "ground_truth_velocity": true_velocity[j, i],
            "posterior_mean_velocity": velocity_mean[j, i],
            "credible_lower": velocity_lower[j, i],
            "credible_upper": velocity_upper[j, i]
        })

pd.DataFrame(rows).to_csv(
    RESULTS_DIR / "rna_velocity_trajectories.csv",
    index=False
)

# --------------------------------------------------
# 8. Peak spliced RNA velocity
# --------------------------------------------------

# Analytical peak time for positive kinetic rates

from rna_kinetics import peak_time


peak_times = np.array([
    peak_time(theta)
    for theta in selected_parameters
])

peak_velocities = np.array([
    calculate_velocity(
        theta,
        np.array([t_peak])
    )[0, 1]
    for theta, t_peak in zip(
        selected_parameters,
        peak_times
    )
])

true_peak_time = peak_time(TRUE_PARAMETERS)

true_peak_velocity = calculate_velocity(
    TRUE_PARAMETERS,
    np.array([true_peak_time])
)[0, 1]

peak_rows = []

for name, samples, truth in [
    (
        "peak_spliced_velocity",
        peak_velocities,
        true_peak_velocity
    ),
    (
        "peak_spliced_velocity_time",
        peak_times,
        true_peak_time
    )
]:

    lower, median, upper = np.percentile(
        samples,
        [2.5, 50, 97.5]
    )

    peak_rows.append({
        "quantity": name,
        "ground_truth": truth,
        "posterior_mean": samples.mean(),
        "posterior_median": median,
        "credible_lower": lower,
        "credible_upper": upper,
        "ground_truth_covered": lower <= truth <= upper
    })

peak_df = pd.DataFrame(peak_rows)

peak_df.to_csv(
    RESULTS_DIR / "rna_velocity_peak_summary.csv",
    index=False
)

# --------------------------------------------------
# 9. RNA velocity visualisation
# --------------------------------------------------

fig, axes = plt.subplots(
    1, 2,
    figsize=(13, 5),
    sharex=True
)

species_names = [
    "Unspliced RNA velocity",
    "Spliced RNA velocity"
]

colours = ["tab:blue", "tab:orange"]

for i, ax in enumerate(axes):

    ax.fill_between(
        times,
        velocity_lower[:, i],
        velocity_upper[:, i],
        color=colours[i],
        alpha=0.25,
        label="95% credible interval"
    )

    ax.plot(
        times,
        velocity_mean[:, i],
        color=colours[i],
        linewidth=2,
        label="Posterior mean"
    )

    ax.plot(
        times,
        true_velocity[:, i],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Ground truth"
    )

    ax.axhline(
        0,
        color="grey",
        linestyle=":"
    )

    ax.set_title(species_names[i])
    ax.set_xlabel("Time (arbitrary units)")
    ax.set_ylabel("RNA abundance change per unit time")
    ax.legend()
    ax.grid(alpha=0.2)

fig.suptitle(
    "Bayesian RNA Velocity and Uncertainty"
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "bayesian_rna_velocity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

# --------------------------------------------------
# 10. RNA phase portrait
# --------------------------------------------------

trajectory_draws = np.array([
    simulate(theta, times)
    for theta in selected_parameters
])

mean_trajectory = trajectory_draws.mean(axis=0)

true_trajectory = simulate(
    TRUE_PARAMETERS,
    times
)

data = pd.read_csv(
    DATA_DIR / "synthetic_rna_observations.csv"
)

fig, ax = plt.subplots(figsize=(8, 6))

# Show a subset of posterior trajectories

for trajectory in trajectory_draws[:100]:

    ax.plot(
        trajectory[:, 1],
        trajectory[:, 0],
        color="tab:blue",
        alpha=0.035
    )

ax.plot(
    true_trajectory[:, 1],
    true_trajectory[:, 0],
    color="black",
    linestyle="--",
    linewidth=2,
    label="Ground truth"
)

ax.plot(
    mean_trajectory[:, 1],
    mean_trajectory[:, 0],
    color="tab:blue",
    linewidth=2,
    label="Posterior mean"
)

ax.scatter(
    data["spliced"],
    data["unspliced"],
    color="tab:orange",
    alpha=0.7,
    label="Synthetic observations"
)

# Direction of motion in phase space

arrow_indices = np.array([
    5, 15, 30, 50, 80, 120
])

ax.quiver(
    mean_trajectory[arrow_indices, 1],
    mean_trajectory[arrow_indices, 0],
    velocity_mean[arrow_indices, 1],
    velocity_mean[arrow_indices, 0],
    angles="xy",
    scale_units="xy",
    scale=5,
    color="tab:green",
    width=0.005,
    label="Velocity direction"
)

ax.scatter(
    TRUE_PARAMETERS[0] / TRUE_PARAMETERS[2],
    TRUE_PARAMETERS[0] / TRUE_PARAMETERS[1],
    marker="*",
    s=200,
    color="red",
    label="Theoretical steady state"
)

ax.set_xlabel("Spliced RNA abundance")
ax.set_ylabel("Unspliced RNA abundance")
ax.set_title("Bayesian RNA Kinetic Phase Portrait")
ax.legend(fontsize=9)
ax.grid(alpha=0.2)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "rna_velocity_phase_portrait.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

# --------------------------------------------------
# 11. Peak velocity posterior distributions
# --------------------------------------------------

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 4.5)
)

for ax, samples, truth, title in [
    (
        axes[0],
        peak_velocities,
        true_peak_velocity,
        "Peak spliced RNA velocity"
    ),
    (
        axes[1],
        peak_times,
        true_peak_time,
        "Time of peak velocity"
    )
]:

    lower, upper = np.percentile(
        samples,
        [2.5, 97.5]
    )

    ax.hist(
        samples,
        bins=40,
        density=True,
        alpha=0.65,
        color="tab:blue"
    )

    ax.axvline(
        truth,
        color="red",
        linestyle="--",
        label="Ground truth"
    )

    ax.axvline(
        samples.mean(),
        color="black",
        label="Posterior mean"
    )

    ax.axvspan(
        lower,
        upper,
        alpha=0.15,
        color="tab:blue",
        label="95% credible interval"
    )

    ax.set_title(title)
    ax.set_ylabel("Posterior density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "rna_velocity_peak_posterior.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

# --------------------------------------------------
# 12. Report results
# --------------------------------------------------

print("\nBayesian RNA Velocity Estimation")
print("--------------------------------------")

print(f"Posterior draws: {N_DRAWS}")
print(f"Time points: {len(times)}")

print("\nVelocity recovery metrics:")
print(metrics_df.round(6).to_string(index=False))

print("\nPeak velocity posterior summary:")
print(peak_df.round(6).to_string(index=False))

print("\nBayesian RNA velocity estimation completed.")
print("Results saved to:", RESULTS_DIR)
print("Figures saved to:", FIGURE_DIR)
