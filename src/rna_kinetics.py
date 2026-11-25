"""Reusable mathematical functions for RNA kinetic modelling."""

import numpy as np

PARAMETER_NAMES = ("alpha", "beta", "gamma")
RELATIVE_STEP = 1e-4


def validate_parameters(theta):
    """Validate positive, finite kinetic parameters."""

    theta = np.asarray(theta, dtype=float)

    if theta.shape != (3,):
        raise ValueError(
            "Expected three parameters: alpha, beta, gamma."
        )

    if not np.all(np.isfinite(theta)):
        raise ValueError("Parameters must be finite.")

    if np.any(theta <= 0):
        raise ValueError("Parameters must be positive.")

    return theta


def rna_kinetics(t, y, alpha, beta, gamma):
    """Governing RNA kinetic ODE system."""

    u, s = y

    du = alpha - beta * u
    ds = beta * u - gamma * s

    return np.array([du, ds])


def simulate(theta, t):
    """Analytical RNA trajectories with zero initial abundance."""

    alpha, beta, gamma = validate_parameters(theta)

    t = np.asarray(t, dtype=float)

    if t.ndim != 1:
        raise ValueError("Time must be a one-dimensional array.")

    if not np.all(np.isfinite(t)) or np.any(t < 0):
        raise ValueError("Time must be finite and non-negative.")

    u = -alpha / beta * np.expm1(-beta * t)

    if np.isclose(beta, gamma, rtol=1e-7, atol=0):

        s = (
            -alpha / gamma * np.expm1(-gamma * t)
            - alpha * t * np.exp(-gamma * t)
        )

    else:

        s = (
            -alpha / gamma * np.expm1(-gamma * t)
            - alpha
            * (np.exp(-beta * t) - np.exp(-gamma * t))
            / (gamma - beta)
        )

    return np.column_stack([u, s])


def velocity(theta, trajectory):
    """Calculate RNA velocity from RNA abundances."""

    alpha, beta, gamma = validate_parameters(theta)

    trajectory = np.asarray(trajectory, dtype=float)

    if trajectory.ndim != 2 or trajectory.shape[1] != 2:
        raise ValueError("Trajectory must have shape (n, 2).")

    u = trajectory[:, 0]
    s = trajectory[:, 1]

    du = alpha - beta * u
    ds = beta * u - gamma * s

    return np.column_stack([du, ds])


def calculate_velocity(theta, t):
    """Calculate RNA velocity at specified observation times."""

    return velocity(theta, simulate(theta, t))


def calculate_jacobian(theta, t, relative_step=RELATIVE_STEP):
    """Calculate parameter sensitivities using central differences."""

    theta = validate_parameters(theta)
    t = np.asarray(t, dtype=float)

    if not np.isfinite(relative_step) or not 0 < relative_step < 1:
        raise ValueError("Relative step must be between 0 and 1.")

    jacobian = np.zeros((len(t), 2, len(theta)))

    for j in range(len(theta)):

        step = theta[j] * relative_step

        theta_plus = theta.copy()
        theta_minus = theta.copy()

        theta_plus[j] += step
        theta_minus[j] -= step

        jacobian[:, :, j] = (
            simulate(theta_plus, t)
            - simulate(theta_minus, t)
        ) / (2 * step)

    return jacobian


def peak_time(theta):
    """Analytical time of maximum spliced RNA velocity."""

    alpha, beta, gamma = validate_parameters(theta)

    if np.isclose(beta, gamma, rtol=1e-7, atol=0):
        return 1.0 / beta

    return np.log(beta / gamma) / (beta - gamma)
