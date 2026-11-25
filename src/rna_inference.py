"""Reusable parameter inference and sampling functions."""

import numpy as np

from scipy.optimize import least_squares
from rna_kinetics import simulate

def observation_times(n, sampling):

    q = np.linspace(0, 1, n)

    if sampling == "uniform":
        return 20 * q

    if sampling == "early":
        return 20 * q ** 2

    if sampling == "late":
        return 20 * (1 - (1 - q) ** 2)

    raise ValueError("Unknown sampling strategy")

def fit_model(t, observations, noise):

    def residuals(log_theta):

        theta = np.exp(log_theta)

        predicted = simulate(theta, t)

        return (
            (predicted - observations) / noise
        ).ravel()

    # Fixed starting point, independent of ground truth

    initial = np.log([3.0, 0.5, 0.3])

    fit = least_squares(
        residuals,
        initial,
        bounds=(
            np.log([0.001, 0.001, 0.001]),
            np.log([100.0, 20.0, 20.0])
        ),
        max_nfev=1000,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10
    )

    estimated = np.exp(fit.x)

    # Jacobian is already scaled by observation noise
    # and calculated in log-parameter space.

    fim = fit.jac.T @ fit.jac

    rank = np.linalg.matrix_rank(fim)

    condition = np.linalg.cond(fim)

    lower = np.full(3, np.nan)
    upper = np.full(3, np.nan)

    at_boundary = np.any(fit.active_mask != 0)

    valid_uncertainty = (
        fit.success
        and not at_boundary
        and rank == 3
        and np.isfinite(condition)
        and condition < 1e12
    )

    if valid_uncertainty:

        covariance = np.linalg.inv(fim)

        log_se = np.sqrt(
            np.maximum(np.diag(covariance), 0)
        )

        lower = np.exp(
            np.clip(fit.x - 1.96 * log_se, -700, 700)
        )

        upper = np.exp(
            np.clip(fit.x + 1.96 * log_se, -700, 700)
        )

    return (
        estimated,
        lower,
        upper,
        condition,
        rank,
        fit.success,
        valid_uncertainty,
        at_boundary
    )
