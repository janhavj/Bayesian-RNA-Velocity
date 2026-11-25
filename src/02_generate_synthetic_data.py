
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path

# Reproducibility
rng = np.random.default_rng(42)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"

DATA_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)


from rna_kinetics import rna_kinetics


# Ground-truth parameters
TRUE_ALPHA = 5.0
TRUE_BETA = 0.8
TRUE_GAMMA = 0.4

# Synthetic observation times
time_points = np.linspace(0, 20, 25)

solution = solve_ivp(
    rna_kinetics,
    t_span=(0, 20),
    y0=[0.0, 0.0],
    t_eval=time_points,
    args=(TRUE_ALPHA, TRUE_BETA, TRUE_GAMMA),
    rtol=1e-8,
    atol=1e-10
)

if not solution.success:
    raise RuntimeError(solution.message)

# Simulate measurement noise
noise_sd = 0.5

observed_u = solution.y[0] + rng.normal(
    0, noise_sd, len(time_points)
)

observed_s = solution.y[1] + rng.normal(
    0, noise_sd, len(time_points)
)

# Save synthetic observations
data = pd.DataFrame({
    "time": time_points,
    "unspliced": observed_u,
    "spliced": observed_s
})

output_path = DATA_DIR / "synthetic_rna_observations.csv"
data.to_csv(output_path, index=False)

# Visualise ground truth and observations
plt.figure(figsize=(9, 5))

plt.plot(
    time_points, solution.y[0],
    label="True unspliced RNA",
    color="tab:blue"
)

plt.plot(
    time_points, solution.y[1],
    label="True spliced RNA",
    color="tab:orange"
)

plt.scatter(
    time_points, observed_u,
    label="Observed unspliced RNA",
    color="tab:blue",
    marker="o"
)

plt.scatter(
    time_points, observed_s,
    label="Observed spliced RNA",
    color="tab:orange",
    marker="x"
)

plt.xlabel("Time (arbitrary units)")
plt.ylabel("RNA abundance (arbitrary units)")
plt.title("Synthetic RNA Kinetic Observations")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

figure_path = FIGURE_DIR / "synthetic_rna_observations.png"
plt.savefig(figure_path, dpi=300)
plt.show()

print(f"Dataset saved: {output_path}")
print(f"Figure saved: {figure_path}")
print(f"Observations: {len(data)}")
print(data.head())
