
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


from rna_kinetics import rna_kinetics


def simulate(parameters, times):
    alpha, beta, gamma = parameters

    solution = solve_ivp(
        rna_kinetics,
        (0, float(np.max(times))),
        [0.0, 0.0],
        t_eval=times,
        args=(alpha, beta, gamma),
        rtol=1e-8,
        atol=1e-10
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    return solution.y.T


# Load the synthetic observations
data = pd.read_csv(
    DATA_DIR / "synthetic_rna_observations.csv"
)

times = data["time"].to_numpy()

observations = data[
    ["unspliced", "spliced"]
].to_numpy()


# Residuals for least-squares optimisation
def residuals(parameters):
    predictions = simulate(parameters, times)
    return (predictions - observations).ravel()


# Initial parameter guesses
initial_guess = [3.0, 0.5, 0.2]

result = least_squares(
    residuals,
    x0=initial_guess,
    bounds=(
        [0.001, 0.001, 0.001],
        [20.0, 5.0, 5.0]
    )
)

if not result.success:
    raise RuntimeError(result.message)

estimated_parameters = result.x

print("\nMaximum-Likelihood Parameter Estimates")
print("--------------------------------------")

names = ["alpha", "beta", "gamma"]
true_values = [5.0, 0.8, 0.4]

for name, true, estimated in zip(
    names, true_values, estimated_parameters
):
    print(
        f"{name}: true={true:.3f}, "
        f"estimated={estimated:.3f}"
    )


# Generate fitted trajectories
fitted = simulate(estimated_parameters, times)

plt.figure(figsize=(9, 5))

plt.scatter(
    times, observations[:, 0],
    label="Observed unspliced",
    alpha=0.6
)

plt.scatter(
    times, observations[:, 1],
    label="Observed spliced",
    alpha=0.6
)

plt.plot(
    times, fitted[:, 0],
    label="Fitted unspliced",
    linewidth=2
)

plt.plot(
    times, fitted[:, 1],
    label="Fitted spliced",
    linewidth=2
)

plt.xlabel("Time (arbitrary units)")
plt.ylabel("RNA abundance (arbitrary units)")
plt.title("Maximum-Likelihood Fit of RNA Kinetic Model")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "rna_parameter_fit.png",
    dpi=300
)

plt.close()

print("\nParameter estimation completed.")
print("Figure saved: figures/rna_parameter_fit.png")
