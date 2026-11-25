
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path


# --------------------------------------------------
# 1. Define the RNA kinetic model
# --------------------------------------------------

from rna_kinetics import rna_kinetics


# --------------------------------------------------
# 2. Define model parameters
# --------------------------------------------------

alpha = 5.0   # Transcription rate
beta = 0.8    # Splicing rate
gamma = 0.4   # Degradation rate

initial_conditions = [0.0, 0.0]

time_points = np.linspace(0, 20, 200)


# --------------------------------------------------
# 3. Solve the ODE system
# --------------------------------------------------

solution = solve_ivp(
    fun=rna_kinetics,
    t_span=(0, 20),
    y0=initial_conditions,
    t_eval=time_points,
    args=(alpha, beta, gamma),
    method="RK45",
    rtol=1e-8,
    atol=1e-10
)

if not solution.success:
    raise RuntimeError(solution.message)

unspliced = solution.y[0]
spliced = solution.y[1]


# --------------------------------------------------
# 4. Plot the RNA trajectories
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    solution.t,
    unspliced,
    label="Unspliced RNA",
    linewidth=2.5
)

plt.plot(
    solution.t,
    spliced,
    label="Spliced RNA",
    linewidth=2.5
)

plt.xlabel("Time (arbitrary units)")
plt.ylabel("RNA abundance (arbitrary units)")
plt.title("Simulated RNA Transcription and Splicing Dynamics")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()


# --------------------------------------------------
# 5. Save the figure
# --------------------------------------------------

output_dir = Path(__file__).resolve().parents[1] / "figures"
output_dir.mkdir(exist_ok=True)

figure_path = output_dir / "rna_kinetics_simulation.png"

plt.savefig(figure_path, dpi=300)
plt.show()

print("Simulation completed successfully.")
print(f"Figure saved to: {figure_path}")
print(f"Final unspliced RNA: {unspliced[-1]:.3f}")
print(f"Final spliced RNA: {spliced[-1]:.3f}")
