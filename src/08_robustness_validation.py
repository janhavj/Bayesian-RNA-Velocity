from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from tqdm import tqdm

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

BASELINE = np.array([5.0, 0.8, 0.4])
PARAMETERS = ["alpha", "beta", "gamma"]

N_REPLICATES = 30
MASTER_SEED = 42

EVALUATION_TIMES = np.linspace(0, 20, 400)

# --------------------------------------------------
# 2. Analytical kinetic model
# --------------------------------------------------

from rna_kinetics import simulate


from rna_kinetics import velocity


# --------------------------------------------------
# 3. Experimental scenarios
# --------------------------------------------------

scenarios = []


def add_scenario(
    group,
    name,
    theta=BASELINE,
    noise=0.5,
    n=25,
    sampling="uniform"
):

    scenarios.append({
        "group": group,
        "scenario": name,
        "theta": np.array(theta, dtype=float),
        "noise": noise,
        "n": n,
        "sampling": sampling
    })


# Observation noise

for noise in [0.1, 0.5, 1.0, 2.0]:

    add_scenario(
        "noise",
        f"noise_{noise}",
        noise=noise
    )


# Sample size

for n in [10, 25, 50, 100]:

    add_scenario(
        "sample_size",
        f"n_{n}",
        n=n
    )


# Kinetic regimes

regimes = {
    "baseline": [5.0, 0.8, 0.4],
    "fast_splicing": [5.0, 1.6, 0.4],
    "slow_splicing": [5.0, 0.4, 0.4],
    "fast_degradation": [5.0, 0.8, 0.8],
    "slow_degradation": [5.0, 0.8, 0.2]
}

for name, theta in regimes.items():

    add_scenario(
        "kinetic_regime",
        name,
        theta=theta
    )


# Sampling strategies

for sampling in ["uniform", "early", "late"]:

    add_scenario(
        "sampling",
        sampling,
        sampling=sampling
    )


# --------------------------------------------------
# 4. Observation sampling
# --------------------------------------------------

from rna_inference import observation_times


# --------------------------------------------------
# 5. Parameter estimation
# --------------------------------------------------

from rna_inference import fit_model


# --------------------------------------------------
# 6. Independent simulation experiments
# --------------------------------------------------

rows = []

seed_sequence = np.random.SeedSequence(MASTER_SEED)

seeds = seed_sequence.spawn(
    len(scenarios) * N_REPLICATES
)

experiment_index = 0

print("Independent Synthetic Robustness Analysis")
print("-----------------------------------------")
print(f"Scenarios: {len(scenarios)}")
print(f"Replicates per scenario: {N_REPLICATES}")
print(f"Total experiments: {len(seeds)}")

for scenario in tqdm(scenarios, desc="Scenarios"):

    theta = scenario["theta"]
    noise = scenario["noise"]

    t = observation_times(
        scenario["n"],
        scenario["sampling"]
    )

    true_observations = simulate(theta, t)

    true_trajectory = simulate(
        theta,
        EVALUATION_TIMES
    )

    true_velocity = velocity(
        theta,
        true_trajectory
    )

    for replicate in range(N_REPLICATES):

        rng = np.random.default_rng(
            seeds[experiment_index]
        )

        experiment_index += 1

        observations = (
            true_observations
            + rng.normal(
                0,
                noise,
                size=true_observations.shape
            )
        )

        (
            estimated,
            lower,
            upper,
            condition,
            rank,
            success,
            valid_uncertainty,
            at_boundary
        ) = fit_model(t, observations, noise)

        predicted = simulate(
            estimated,
            EVALUATION_TIMES
        )

        predicted_velocity = velocity(
            estimated,
            predicted
        )

        abundance_rmse = np.sqrt(
            np.mean(
                (predicted - true_trajectory) ** 2,
                axis=0
            )
        )

        velocity_rmse = np.sqrt(
            np.mean(
                (predicted_velocity - true_velocity) ** 2,
                axis=0
            )
        )

        row = {
            "group": scenario["group"],
            "scenario": scenario["scenario"],
            "replicate": replicate + 1,
            "noise_sd": noise,
            "n_observations": scenario["n"],
            "sampling": scenario["sampling"],
            "optimisation_success": success,
            "at_boundary": at_boundary,
            "valid_uncertainty": valid_uncertainty,
            "fim_rank": rank,
            "fim_condition_number": condition,
            "unspliced_abundance_rmse": abundance_rmse[0],
            "spliced_abundance_rmse": abundance_rmse[1],
            "unspliced_velocity_rmse": velocity_rmse[0],
            "spliced_velocity_rmse": velocity_rmse[1]
        }

        for j, name in enumerate(PARAMETERS):

            row[f"{name}_true"] = theta[j]

            row[f"{name}_estimated"] = estimated[j]

            row[f"{name}_relative_error_percent"] = (
                abs(estimated[j] - theta[j])
                / theta[j] * 100
            )

            row[f"{name}_ci_lower"] = lower[j]
            row[f"{name}_ci_upper"] = upper[j]

            row[f"{name}_covered"] = (
                float(lower[j] <= theta[j] <= upper[j])
                if valid_uncertainty
                else np.nan
            )

        row["mean_parameter_error_percent"] = np.mean([
            row[f"{name}_relative_error_percent"]
            for name in PARAMETERS
        ])

        rows.append(row)


# --------------------------------------------------
# 7. Save individual experiments
# --------------------------------------------------

df = pd.DataFrame(rows)

df.to_csv(
    RESULTS / "robustness_individual_experiments.csv",
    index=False
)

# Exclude failed optimisations from recovery summaries.
# Retain all runs in the individual results file.

successful = df[
    df["optimisation_success"]
].copy()

# --------------------------------------------------
# 8. Aggregate results
# --------------------------------------------------

summary = successful.groupby(
    ["group", "scenario"],
    sort=False
).agg(
    successful_replicates=(
        "replicate", "count"
    ),
    median_parameter_error_percent=(
        "mean_parameter_error_percent", "median"
    ),
    median_alpha_error_percent=(
        "alpha_relative_error_percent", "median"
    ),
    median_beta_error_percent=(
        "beta_relative_error_percent", "median"
    ),
    median_gamma_error_percent=(
        "gamma_relative_error_percent", "median"
    ),
    median_unspliced_velocity_rmse=(
        "unspliced_velocity_rmse", "median"
    ),
    median_spliced_velocity_rmse=(
        "spliced_velocity_rmse", "median"
    ),
    median_unspliced_abundance_rmse=(
        "unspliced_abundance_rmse", "median"
    ),
    median_spliced_abundance_rmse=(
        "spliced_abundance_rmse", "median"
    ),
    median_fim_condition_number=(
        "fim_condition_number", "median"
    ),
    alpha_coverage=("alpha_covered", "mean"),
    beta_coverage=("beta_covered", "mean"),
    gamma_coverage=("gamma_covered", "mean")
).reset_index()

diagnostics = df.groupby(
    ["group", "scenario"],
    sort=False
).agg(
    optimisation_success_rate=(
        "optimisation_success", "mean"
    ),
    valid_uncertainty_rate=(
        "valid_uncertainty", "mean"
    ),
    boundary_rate=(
        "at_boundary", "mean"
    ),
    total_replicates=("replicate", "count")
).reset_index()

summary = diagnostics.merge(
    summary,
    on=["group", "scenario"],
    how="left"
)

summary.to_csv(
    RESULTS / "robustness_scenario_summary.csv",
    index=False
)

# --------------------------------------------------
# 9. Parameter recovery boxplots
# --------------------------------------------------

groups = [
    "noise",
    "sample_size",
    "kinetic_regime",
    "sampling"
]

titles = [
    "Observation noise",
    "Number of observations",
    "Kinetic regime",
    "Sampling strategy"
]

fig, axes = plt.subplots(
    2, 2,
    figsize=(14, 10)
)

for ax, group, title in zip(
    axes.ravel(), groups, titles
):

    subset = successful[
        successful["group"] == group
    ]

    labels = subset["scenario"].unique()

    values = [
        subset.loc[
            subset["scenario"] == label,
            "mean_parameter_error_percent"
        ].to_numpy()
        for label in labels
    ]

    ax.boxplot(values, tick_labels=labels)

    ax.set_title(title)
    ax.set_ylabel("Mean relative parameter error (%)")

    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.2)

fig.suptitle(
    "Robustness of RNA Kinetic Parameter Recovery"
)

fig.tight_layout()

fig.savefig(
    FIGURES / "robustness_parameter_recovery.png",
    dpi=300
)

plt.close(fig)

# --------------------------------------------------
# 10. RNA velocity robustness
# --------------------------------------------------

fig, axes = plt.subplots(
    2, 2,
    figsize=(14, 10)
)

for ax, group, title in zip(
    axes.ravel(), groups, titles
):

    subset = summary[
        summary["group"] == group
    ]

    x = np.arange(len(subset))
    width = 0.35

    ax.bar(
        x - width / 2,
        subset["median_unspliced_velocity_rmse"],
        width,
        label="Unspliced RNA"
    )

    ax.bar(
        x + width / 2,
        subset["median_spliced_velocity_rmse"],
        width,
        label="Spliced RNA"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        subset["scenario"],
        rotation=25,
        ha="right"
    )

    ax.set_title(title)
    ax.set_ylabel("Median velocity RMSE")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

fig.suptitle(
    "RNA Velocity Recovery Across Experimental Conditions"
)

fig.tight_layout()

fig.savefig(
    FIGURES / "robustness_velocity_rmse.png",
    dpi=300
)

plt.close(fig)

# --------------------------------------------------
# 11. Confidence interval coverage
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(summary))
width = 0.25

for j, name in enumerate(PARAMETERS):

    ax.bar(
        x + (j - 1) * width,
        summary[f"{name}_coverage"],
        width,
        label=name
    )

ax.axhline(
    0.95,
    color="black",
    linestyle="--",
    label="Nominal 95% coverage"
)

ax.set_xticks(x)

ax.set_xticklabels(
    summary["scenario"],
    rotation=45,
    ha="right"
)

ax.set_ylim(0, 1.05)
ax.set_ylabel("Empirical coverage")
ax.set_title("Fisher-Based Confidence Interval Coverage")
ax.legend(ncol=4)

fig.tight_layout()

fig.savefig(
    FIGURES / "robustness_interval_coverage.png",
    dpi=300
)

plt.close(fig)

# --------------------------------------------------
# 12. Identifiability across conditions
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 6))

conditions = summary[
    "median_fim_condition_number"
].replace([np.inf, -np.inf], np.nan)

ax.bar(
    np.arange(len(summary)),
    conditions
)

ax.set_yscale("log")

ax.set_xticks(np.arange(len(summary)))

ax.set_xticklabels(
    summary["scenario"],
    rotation=45,
    ha="right"
)

ax.set_ylabel("Median FIM condition number")
ax.set_title("Parameter Identifiability Across Scenarios")
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()

fig.savefig(
    FIGURES / "robustness_identifiability.png",
    dpi=300
)

plt.close(fig)

# --------------------------------------------------
# 13. Parameter recovery heatmap
# --------------------------------------------------

matrix = summary[[
    "median_alpha_error_percent",
    "median_beta_error_percent",
    "median_gamma_error_percent"
]].to_numpy()

fig, ax = plt.subplots(figsize=(8, 10))

image = ax.imshow(
    matrix,
    cmap="YlOrRd",
    aspect="auto"
)

ax.set_xticks(range(3))
ax.set_xticklabels(PARAMETERS)

ax.set_yticks(range(len(summary)))

ax.set_yticklabels([
    f"{g}: {s}"
    for g, s in zip(
        summary["group"],
        summary["scenario"]
    )
])

for i in range(matrix.shape[0]):

    for j in range(matrix.shape[1]):

        ax.text(
            j,
            i,
            f"{matrix[i, j]:.2f}%",
            ha="center",
            va="center"
        )

fig.colorbar(
    image,
    ax=ax,
    label="Median relative error (%)"
)

ax.set_title(
    "Kinetic Parameter Recovery Across Scenarios"
)

fig.tight_layout()

fig.savefig(
    FIGURES / "robustness_parameter_heatmap.png",
    dpi=300
)

plt.close(fig)

# --------------------------------------------------
# 14. Sampling design visualisation
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 4))

for i, sampling in enumerate(
    ["uniform", "early", "late"]
):

    t = observation_times(25, sampling)

    ax.scatter(
        t,
        np.full(len(t), i),
        s=30
    )

ax.set_yticks([0, 1, 2])

ax.set_yticklabels([
    "Uniform",
    "Early-time enriched",
    "Late-time enriched"
])

ax.set_xlabel("Observation time")
ax.set_title("Comparison of Temporal Sampling Designs")
ax.grid(axis="x", alpha=0.2)

fig.tight_layout()

fig.savefig(
    FIGURES / "robustness_sampling_designs.png",
    dpi=300
)

plt.close(fig)

# --------------------------------------------------
# 15. Final report
# --------------------------------------------------

print("\nRobustness Scenario Summary")
print("---------------------------")

columns = [
    "group",
    "scenario",
    "median_parameter_error_percent",
    "median_unspliced_velocity_rmse",
    "median_spliced_velocity_rmse",
    "median_fim_condition_number",
    "optimisation_success_rate",
    "valid_uncertainty_rate"
]

print(
    summary[columns].round(4).to_string(
        index=False
    )
)

print("\nConfidence interval coverage:")

print(
    summary[[
        "group",
        "scenario",
        "alpha_coverage",
        "beta_coverage",
        "gamma_coverage"
    ]].round(4).to_string(index=False)
)

print("\nOverall diagnostics:")

print(f"Total experiments: {len(df)}")

print(
    "Optimisation success rate:",
    df["optimisation_success"].mean()
)

print(
    "Valid uncertainty rate:",
    df["valid_uncertainty"].mean()
)

print(
    "Boundary solution rate:",
    df["at_boundary"].mean()
)

print("\nRobustness validation completed.")
print("Results saved to:", RESULTS)
print("Figures saved to:", FIGURES)
