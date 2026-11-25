from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import emcee
import corner

# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)
np.random.seed(SEED)

PARAMETER_NAMES = ["alpha", "beta", "gamma"]
TRUE_PARAMETERS = np.array([5.0, 0.8, 0.4])

NOISE_SD = 0.5

# --------------------------------------------------
# 2. Load synthetic observations
# --------------------------------------------------

data = pd.read_csv(
    DATA_DIR / "synthetic_rna_observations.csv"
)

times = data["time"].to_numpy()

observations = data[
    ["unspliced", "spliced"]
].to_numpy()

# --------------------------------------------------
# 3. Analytical RNA kinetic model
# --------------------------------------------------

from rna_kinetics import simulate


# --------------------------------------------------
# 4. Prior distribution
# --------------------------------------------------

from rna_bayesian import log_prior


# --------------------------------------------------
# 5. Gaussian log-likelihood
# --------------------------------------------------

from rna_bayesian import log_likelihood as shared_log_likelihood

def log_likelihood(theta):
    return shared_log_likelihood(
        theta, times, observations, NOISE_SD
    )


# --------------------------------------------------
# 6. Log-posterior
# --------------------------------------------------

from rna_bayesian import log_posterior as shared_log_posterior

def log_posterior(theta):
    return shared_log_posterior(
        theta, times, observations, NOISE_SD
    )


# --------------------------------------------------
# 7. Initialise MCMC walkers
# --------------------------------------------------

N_DIM = 3
N_WALKERS = 32
N_STEPS = 5000
BURN_IN = 1000

initial_guess = np.array([3.0, 0.5, 0.2])

initial_positions = (
    initial_guess
    + rng.normal(
        0,
        0.01,
        size=(N_WALKERS, N_DIM)
    )
)

sampler = emcee.EnsembleSampler(
    N_WALKERS,
    N_DIM,
    log_posterior
)

# --------------------------------------------------
# 8. Run MCMC
# --------------------------------------------------

print("Starting Bayesian parameter inference...")
print(f"Walkers: {N_WALKERS}")
print(f"Steps per walker: {N_STEPS}")

sampler.run_mcmc(
    initial_positions,
    N_STEPS,
    progress=True
)

# --------------------------------------------------
# 9. Extract posterior samples
# --------------------------------------------------

chain = sampler.get_chain()

posterior = sampler.get_chain(
    discard=BURN_IN,
    flat=True
)

np.save(
    RESULTS_DIR / "mcmc_full_chain.npy",
    chain
)

posterior_df = pd.DataFrame(
    posterior,
    columns=PARAMETER_NAMES
)

posterior_df.to_csv(
    RESULTS_DIR / "posterior_samples.csv",
    index=False
)

# --------------------------------------------------
# 10. Posterior summary statistics
# --------------------------------------------------

summary_rows = []

for i, name in enumerate(PARAMETER_NAMES):

    samples = posterior[:, i]

    lower, median, upper = np.percentile(
        samples,
        [2.5, 50, 97.5]
    )

    summary_rows.append({
        "parameter": name,
        "true_value": TRUE_PARAMETERS[i],
        "posterior_mean": np.mean(samples),
        "posterior_median": median,
        "posterior_sd": np.std(samples, ddof=1),
        "ci_2.5": lower,
        "ci_97.5": upper,
        "true_value_covered": (
            lower <= TRUE_PARAMETERS[i] <= upper
        )
    })

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    RESULTS_DIR / "bayesian_parameter_summary.csv",
    index=False
)

print("\nBayesian parameter summary:")
print(summary.round(4).to_string(index=False))

# --------------------------------------------------
# 11. MCMC trace plots
# --------------------------------------------------

fig, axes = plt.subplots(
    N_DIM,
    1,
    figsize=(11, 8),
    sharex=True
)

for i, name in enumerate(PARAMETER_NAMES):

    axes[i].plot(
        chain[:, :, i],
        alpha=0.25,
        linewidth=0.5
    )

    axes[i].axhline(
        TRUE_PARAMETERS[i],
        color="red",
        linestyle="--",
        label="Ground truth"
    )

    axes[i].axvline(
        BURN_IN,
        color="black",
        linestyle=":"
    )

    axes[i].set_ylabel(name)
    axes[i].legend(loc="upper right")

axes[-1].set_xlabel("MCMC iteration")

fig.suptitle("MCMC Trace Plots")
fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "mcmc_trace_plots.png",
    dpi=300
)

plt.close(fig)

# --------------------------------------------------
# 12. Posterior corner plot
# --------------------------------------------------

fig = corner.corner(
    posterior,
    labels=PARAMETER_NAMES,
    truths=TRUE_PARAMETERS,
    quantiles=[0.025, 0.5, 0.975],
    show_titles=True,
    title_fmt=".3f"
)

fig.savefig(
    FIGURE_DIR / "posterior_corner_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

# --------------------------------------------------
# 13. Initial sampling diagnostics
# --------------------------------------------------

print(
    "\nMean acceptance fraction:",
    round(np.mean(sampler.acceptance_fraction), 4)
)

print("Retained posterior samples:", len(posterior))

try:
    tau = sampler.get_autocorr_time(
        discard=BURN_IN
    )

    print("Integrated autocorrelation times:")
    print(dict(zip(PARAMETER_NAMES, tau.round(2))))

    ess = (
        (N_STEPS - BURN_IN) * N_WALKERS / tau
    )

    print("Approximate effective sample sizes:")
    print(dict(zip(PARAMETER_NAMES, ess.round(1))))

except emcee.autocorr.AutocorrError as error:
    print("Autocorrelation warning:")
    print(error)

print("\nBayesian parameter inference completed.")
print("Results saved to:", RESULTS_DIR)
print("Figures saved to:", FIGURE_DIR)
