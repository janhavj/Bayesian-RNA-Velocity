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

PARAMETER_NAMES = ["alpha", "beta", "gamma"]

TRUE_PARAMETERS = np.array([5.0, 0.8, 0.4])

NOISE_SD = 0.5
RELATIVE_STEP = 1e-4

PERTURBATIONS = [-0.20, -0.10, 0.10, 0.20]

times = np.linspace(0, 20, 400)

# --------------------------------------------------
# 2. Analytical RNA kinetic model
# --------------------------------------------------

from rna_kinetics import simulate


baseline = simulate(TRUE_PARAMETERS, times)

# --------------------------------------------------
# 3. Parameter perturbation analysis
# --------------------------------------------------

perturbation_rows = []

fig, axes = plt.subplots(
    3, 2,
    figsize=(13, 12),
    sharex=True
)

colours = [
    "tab:blue",
    "tab:cyan",
    "tab:orange",
    "tab:red"
]

for j, name in enumerate(PARAMETER_NAMES):

    for i, species in enumerate(
        ["unspliced", "spliced"]
    ):

        ax = axes[j, i]

        ax.plot(
            times,
            baseline[:, i],
            color="black",
            linestyle="--",
            linewidth=2,
            label="Baseline"
        )

        for change, colour in zip(
            PERTURBATIONS,
            colours
        ):

            theta = TRUE_PARAMETERS.copy()

            theta[j] *= 1 + change

            trajectory = simulate(theta, times)

            ax.plot(
                times,
                trajectory[:, i],
                color=colour,
                label=f"{change * 100:+.0f}%"
            )

            rmse = np.sqrt(
                np.mean(
                    (
                        trajectory[:, i]
                        - baseline[:, i]
                    ) ** 2
                )
            )

            final_change = (
                (
                    trajectory[-1, i]
                    - baseline[-1, i]
                )
                / baseline[-1, i]
                * 100
            )

            perturbation_rows.append({
                "parameter": name,
                "perturbation_percent": change * 100,
                "rna_species": species,
                "perturbed_parameter": theta[j],
                "trajectory_rmse": rmse,
                "final_abundance_change_percent": final_change
            })

        ax.set_title(f"{name}: {species} RNA")
        ax.set_ylabel("RNA abundance")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

axes[-1, 0].set_xlabel("Time")
axes[-1, 1].set_xlabel("Time")

fig.suptitle(
    "RNA Kinetic Parameter Perturbation Analysis"
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "parameter_perturbation_analysis.png",
    dpi=300
)

plt.close(fig)

perturbation_df = pd.DataFrame(perturbation_rows)

perturbation_df.to_csv(
    RESULTS_DIR / "parameter_perturbation_results.csv",
    index=False
)

# --------------------------------------------------
# 4. Calculate local sensitivity derivatives
# --------------------------------------------------

from rna_kinetics import calculate_jacobian as shared_jacobian

def calculate_jacobian(theta, t):
    return shared_jacobian(
        theta, t, relative_step=RELATIVE_STEP
    )


jacobian = calculate_jacobian(
    TRUE_PARAMETERS,
    times
)

# --------------------------------------------------
# 5. Normalised local sensitivities
# --------------------------------------------------

normalised = np.full_like(
    jacobian,
    np.nan
)

for j in range(3):

    np.divide(
        jacobian[:, :, j] * TRUE_PARAMETERS[j],
        baseline,
        out=normalised[:, :, j],
        where=np.abs(baseline) > 1e-10
    )

sensitivity_rows = []

for j, name in enumerate(PARAMETER_NAMES):

    for i, species in enumerate(
        ["unspliced", "spliced"]
    ):

        for k, time in enumerate(times):

            sensitivity_rows.append({
                "time": time,
                "parameter": name,
                "rna_species": species,
                "absolute_sensitivity": jacobian[k, i, j],
                "normalised_sensitivity": normalised[k, i, j]
            })

pd.DataFrame(sensitivity_rows).to_csv(
    RESULTS_DIR / "local_sensitivity_trajectories.csv",
    index=False
)

# --------------------------------------------------
# 6. Plot time-dependent sensitivities
# --------------------------------------------------

fig, axes = plt.subplots(
    1, 2,
    figsize=(13, 5),
    sharex=True
)

for i, species in enumerate(
    ["Unspliced RNA", "Spliced RNA"]
):

    for j, name in enumerate(PARAMETER_NAMES):

        axes[i].plot(
            times,
            normalised[:, i, j],
            linewidth=2,
            label=name
        )

    axes[i].axhline(
        0,
        color="black",
        linestyle="--"
    )

    axes[i].set_title(species)
    axes[i].set_xlabel("Time")
    axes[i].set_ylabel("Normalised sensitivity")
    axes[i].legend()
    axes[i].grid(alpha=0.2)

fig.suptitle(
    "Time-Dependent Normalised Parameter Sensitivity"
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "local_sensitivity_analysis.png",
    dpi=300
)

plt.close(fig)

# --------------------------------------------------
# 7. Sensitivity summary
# --------------------------------------------------

summary_rows = []

for j, name in enumerate(PARAMETER_NAMES):

    for i, species in enumerate(
        ["unspliced", "spliced"]
    ):

        values = normalised[:, i, j]

        summary_rows.append({
            "parameter": name,
            "rna_species": species,
            "mean_absolute_sensitivity": np.nanmean(
                np.abs(values)
            ),
            "maximum_absolute_sensitivity": np.nanmax(
                np.abs(values)
            ),
            "final_time_sensitivity": values[-1]
        })

sensitivity_summary = pd.DataFrame(summary_rows)

sensitivity_summary.to_csv(
    RESULTS_DIR / "sensitivity_summary.csv",
    index=False
)

# --------------------------------------------------
# 8. Analytical steady-state sensitivities
# --------------------------------------------------

# u* = alpha / beta
# s* = alpha / gamma

steady_state = np.array([
    [1.0, -1.0, 0.0],
    [1.0, 0.0, -1.0]
])

steady_df = pd.DataFrame(
    steady_state,
    index=["unspliced", "spliced"],
    columns=PARAMETER_NAMES
)

steady_df.to_csv(
    RESULTS_DIR / "steady_state_sensitivity.csv"
)

# --------------------------------------------------
# 9. Fisher information matrix
# --------------------------------------------------

data = pd.read_csv(
    DATA_DIR / "synthetic_rna_observations.csv"
)

observed_times = data["time"].to_numpy()

observed_jacobian = calculate_jacobian(
    TRUE_PARAMETERS,
    observed_times
)

J = observed_jacobian.reshape(-1, 3)

FIM = J.T @ J / NOISE_SD ** 2

# Log-parameter scaling reduces unit dependence

J_scaled = J * TRUE_PARAMETERS[np.newaxis, :]

FIM_scaled = (
    J_scaled.T @ J_scaled
    / NOISE_SD ** 2
)

fim_df = pd.DataFrame(
    FIM,
    index=PARAMETER_NAMES,
    columns=PARAMETER_NAMES
)

fim_df.to_csv(
    RESULTS_DIR / "fisher_information_matrix.csv"
)

pd.DataFrame(
    FIM_scaled,
    index=PARAMETER_NAMES,
    columns=PARAMETER_NAMES
).to_csv(
    RESULTS_DIR / "scaled_fisher_information_matrix.csv"
)

# --------------------------------------------------
# 10. Identifiability diagnostics
# --------------------------------------------------

eigenvalues = np.linalg.eigvalsh(FIM_scaled)

rank = np.linalg.matrix_rank(FIM_scaled)

condition_number = np.linalg.cond(FIM_scaled)

unscaled_condition = np.linalg.cond(FIM)

identifiability_df = pd.DataFrame([{
    "number_of_parameters": 3,
    "fim_rank": rank,
    "smallest_scaled_eigenvalue": eigenvalues.min(),
    "largest_scaled_eigenvalue": eigenvalues.max(),
    "scaled_condition_number": condition_number,
    "unscaled_condition_number": unscaled_condition
}])

identifiability_df.to_csv(
    RESULTS_DIR / "identifiability_diagnostics.csv",
    index=False
)

if rank < 3:

    raise RuntimeError(
        "Fisher information matrix is rank deficient."
    )

# --------------------------------------------------
# 11. Fisher-based parameter uncertainty
# --------------------------------------------------

covariance = np.linalg.inv(FIM)

standard_errors = np.sqrt(
    np.diag(covariance)
)

correlation = covariance / np.outer(
    standard_errors,
    standard_errors
)

fim_uncertainty = pd.DataFrame({
    "parameter": PARAMETER_NAMES,
    "ground_truth": TRUE_PARAMETERS,
    "fisher_standard_error": standard_errors,
    "relative_standard_error_percent": (
        standard_errors / TRUE_PARAMETERS * 100
    )
})

fim_uncertainty.to_csv(
    RESULTS_DIR / "fisher_parameter_uncertainty.csv",
    index=False
)

# --------------------------------------------------
# 12. Compare Fisher and Bayesian correlations
# --------------------------------------------------

posterior = pd.read_csv(
    RESULTS_DIR / "posterior_samples.csv"
)

posterior_corr = posterior[
    PARAMETER_NAMES
].corr().to_numpy()

pd.DataFrame(
    correlation,
    index=PARAMETER_NAMES,
    columns=PARAMETER_NAMES
).to_csv(
    RESULTS_DIR / "fisher_parameter_correlations.csv"
)

# --------------------------------------------------
# 13. Sensitivity heatmaps
# --------------------------------------------------

fig, axes = plt.subplots(
    1, 2,
    figsize=(11, 4.5)
)

mean_sensitivity = np.nanmean(
    np.abs(normalised),
    axis=0
)

for ax, matrix, title, cmap, vmin, vmax in [
    (
        axes[0],
        mean_sensitivity,
        "Mean absolute normalised sensitivity",
        "viridis",
        0,
        max(1.0, mean_sensitivity.max())
    ),
    (
        axes[1],
        steady_state,
        "Analytical steady-state sensitivity",
        "coolwarm",
        -1,
        1
    )
]:

    image = ax.imshow(
        matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto"
    )

    ax.set_xticks(range(3))
    ax.set_xticklabels(PARAMETER_NAMES)

    ax.set_yticks(range(2))
    ax.set_yticklabels(
        ["Unspliced", "Spliced"]
    )

    ax.set_title(title)

    for i in range(2):

        for j in range(3):

            ax.text(
                j, i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
                color="white",
                bbox={
                    "facecolor": "black",
                    "alpha": 0.35,
                    "edgecolor": "none"
                }
            )

    fig.colorbar(image, ax=ax)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "sensitivity_heatmaps.png",
    dpi=300
)

plt.close(fig)

# --------------------------------------------------
# 14. Parameter correlation comparison
# --------------------------------------------------

fig, axes = plt.subplots(
    1, 2,
    figsize=(10, 4.5)
)

for ax, matrix, title in [
    (
        axes[0],
        correlation,
        "Fisher-based correlations"
    ),
    (
        axes[1],
        posterior_corr,
        "Bayesian posterior correlations"
    )
]:

    image = ax.imshow(
        matrix,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    ax.set_xticks(range(3))
    ax.set_xticklabels(PARAMETER_NAMES)

    ax.set_yticks(range(3))
    ax.set_yticklabels(PARAMETER_NAMES)

    ax.set_title(title)

    for i in range(3):

        for j in range(3):

            ax.text(
                j, i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center"
            )

    fig.colorbar(image, ax=ax)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "parameter_correlation_comparison.png",
    dpi=300
)

plt.close(fig)

# --------------------------------------------------
# 15. Fisher information eigenvalue spectrum
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 5))

ax.bar(
    ["Eigenvalue 1", "Eigenvalue 2", "Eigenvalue 3"],
    eigenvalues
)

ax.set_yscale("log")
ax.set_ylabel("Eigenvalue (log scale)")
ax.set_title("Scaled Fisher Information Spectrum")
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "fisher_information_eigenvalues.png",
    dpi=300
)

plt.close(fig)

# --------------------------------------------------
# 16. Report results
# --------------------------------------------------

print("\nParameter Sensitivity Analysis")
print("--------------------------------------")

print(
    sensitivity_summary.round(4).to_string(
        index=False
    )
)

print("\nAnalytical steady-state sensitivities:")
print(steady_df)

print("\nFisher Information Matrix:")
print(fim_df.round(4))

print("\nIdentifiability diagnostics:")
print(f"FIM rank: {rank}/3")
print(f"Scaled eigenvalues: {eigenvalues.round(4)}")
print(f"Scaled condition number: {condition_number:.4f}")
print(f"Unscaled condition number: {unscaled_condition:.4f}")

print("\nFisher-based parameter uncertainty:")
print(
    fim_uncertainty.round(4).to_string(
        index=False
    )
)

print("\nFisher-based correlations:")
print(np.round(correlation, 4))

print("\nBayesian posterior correlations:")
print(np.round(posterior_corr, 4))

print("\nSensitivity and identifiability analysis completed.")
print("Results saved to:", RESULTS_DIR)
print("Figures saved to:", FIGURE_DIR)
