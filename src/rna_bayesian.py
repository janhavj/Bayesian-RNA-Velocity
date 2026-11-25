"""Reusable Bayesian probability functions for RNA kinetics."""

import numpy as np

from rna_kinetics import simulate


PRIOR_LOWER = np.array([0.001, 0.001, 0.001])
PRIOR_UPPER = np.array([20.0, 5.0, 5.0])


def log_prior(theta):
    """Unnormalised bounded uniform log-prior."""

    theta = np.asarray(theta, dtype=float)

    if theta.shape != (3,):
        return -np.inf

    if (
        np.all(np.isfinite(theta))
        and np.all(theta > PRIOR_LOWER)
        and np.all(theta < PRIOR_UPPER)
    ):
        return 0.0

    return -np.inf


def log_likelihood(theta, times, observations, noise_sd=0.5):
    """Independent Gaussian observation log-likelihood."""

    if not np.isfinite(noise_sd) or noise_sd <= 0:
        raise ValueError("Noise SD must be finite and positive.")

    predictions = simulate(theta, times)

    observations = np.asarray(observations, dtype=float)

    if observations.shape != predictions.shape:
        raise ValueError("Observation and prediction shapes differ.")

    if not np.all(np.isfinite(observations)):
        raise ValueError("Observations must be finite.")

    residuals = observations - predictions

    return float(
        -0.5 * np.sum(
            (residuals / noise_sd) ** 2
            + np.log(2 * np.pi * noise_sd ** 2)
        )
    )


def log_posterior(theta, times, observations, noise_sd=0.5):
    """Unnormalised Bayesian log-posterior."""

    lp = log_prior(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(
        theta,
        times,
        observations,
        noise_sd
    )
