"""Unit tests for Bayesian RNA kinetic inference."""

import numpy as np
import pytest

from scipy.stats import norm

from rna_kinetics import simulate

from rna_bayesian import (
    log_prior,
    log_likelihood,
    log_posterior
)

THETA = np.array([5.0, 0.8, 0.4])
TIMES = np.linspace(0, 20, 25)
NOISE = 0.5

OBSERVATIONS = simulate(THETA, TIMES)


# --------------------------------------------------
# 1. Prior distribution
# --------------------------------------------------

def test_valid_prior():

    assert log_prior(THETA) == 0.0


@pytest.mark.parametrize(
    "theta",
    [
        [1.0, 0.1, 0.1],
        [10.0, 2.0, 2.0],
        [19.9, 4.9, 4.9]
    ]
)
def test_uniform_prior(theta):

    assert log_prior(theta) == 0.0


@pytest.mark.parametrize(
    "theta",
    [
        [-5.0, 0.8, 0.4],
        [5.0, -0.8, 0.4],
        [5.0, 0.8, -0.4],
        [0.001, 0.8, 0.4],
        [20.0, 0.8, 0.4],
        [5.0, 0.001, 0.4],
        [5.0, 5.0, 0.4],
        [5.0, 0.8, 0.001],
        [5.0, 0.8, 5.0],
        [np.nan, 0.8, 0.4],
        [5.0, np.inf, 0.4],
        [5.0, 0.8]
    ]
)
def test_invalid_prior(theta):

    assert np.isneginf(log_prior(theta))


# --------------------------------------------------
# 2. Gaussian likelihood
# --------------------------------------------------

def test_finite_likelihood():

    result = log_likelihood(
        THETA, TIMES, OBSERVATIONS, NOISE
    )

    assert np.isfinite(result)


def test_likelihood_at_exact_fit():

    expected = (
        -0.5
        * OBSERVATIONS.size
        * np.log(2 * np.pi * NOISE ** 2)
    )

    result = log_likelihood(
        THETA, TIMES, OBSERVATIONS, NOISE
    )

    assert result == pytest.approx(expected)


def test_likelihood_against_scipy():

    rng = np.random.default_rng(42)

    observations = OBSERVATIONS + rng.normal(
        0, NOISE, OBSERVATIONS.shape
    )

    expected = np.sum(
        norm.logpdf(
            observations,
            loc=OBSERVATIONS,
            scale=NOISE
        )
    )

    result = log_likelihood(
        THETA, TIMES, observations, NOISE
    )

    assert result == pytest.approx(expected)


def test_true_parameters_preferred():

    incorrect_theta = [3.0, 0.5, 0.2]

    true_ll = log_likelihood(
        THETA, TIMES, OBSERVATIONS, NOISE
    )

    incorrect_ll = log_likelihood(
        incorrect_theta,
        TIMES,
        OBSERVATIONS,
        NOISE
    )

    assert true_ll > incorrect_ll


def test_likelihood_residual_penalty():

    shifted = OBSERVATIONS + 1.0

    exact_ll = log_likelihood(
        THETA, TIMES, OBSERVATIONS, NOISE
    )

    shifted_ll = log_likelihood(
        THETA, TIMES, shifted, NOISE
    )

    expected_penalty = (
        OBSERVATIONS.size / (2 * NOISE ** 2)
    )

    assert exact_ll - shifted_ll == pytest.approx(
        expected_penalty
    )


# --------------------------------------------------
# 3. Observation noise
# --------------------------------------------------

@pytest.mark.parametrize(
    "noise",
    [0.1, 0.5, 1.0, 2.0]
)
def test_valid_noise_levels(noise):

    result = log_likelihood(
        THETA, TIMES, OBSERVATIONS, noise
    )

    assert np.isfinite(result)


@pytest.mark.parametrize(
    "noise",
    [0.0, -0.5, np.nan, np.inf]
)
def test_invalid_noise_levels(noise):

    with pytest.raises(ValueError):

        log_likelihood(
            THETA, TIMES, OBSERVATIONS, noise
        )


def test_noise_reduces_residual_penalty():

    shifted = OBSERVATIONS + 1.0

    def penalty(noise):

        exact_ll = log_likelihood(
            THETA, TIMES, OBSERVATIONS, noise
        )

        shifted_ll = log_likelihood(
            THETA, TIMES, shifted, noise
        )

        return exact_ll - shifted_ll

    assert penalty(0.1) > penalty(1.0)


# --------------------------------------------------
# 4. Posterior distribution
# --------------------------------------------------

def test_posterior_decomposition():

    expected = (
        log_prior(THETA)
        + log_likelihood(
            THETA, TIMES, OBSERVATIONS, NOISE
        )
    )

    result = log_posterior(
        THETA, TIMES, OBSERVATIONS, NOISE
    )

    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "theta",
    [
        [-5.0, 0.8, 0.4],
        [25.0, 0.8, 0.4],
        [5.0, 6.0, 0.4],
        [5.0, 0.8, 6.0],
        [np.nan, 0.8, 0.4]
    ]
)
def test_posterior_rejects_invalid_parameters(theta):

    result = log_posterior(
        theta, TIMES, OBSERVATIONS, NOISE
    )

    assert np.isneginf(result)


def test_posterior_prefers_true_parameters():

    true_lp = log_posterior(
        THETA, TIMES, OBSERVATIONS, NOISE
    )

    incorrect_lp = log_posterior(
        [3.0, 0.5, 0.2],
        TIMES,
        OBSERVATIONS,
        NOISE
    )

    assert true_lp > incorrect_lp


# --------------------------------------------------
# 5. Observation validation
# --------------------------------------------------

def test_invalid_observation_shape():

    with pytest.raises(ValueError):

        log_likelihood(
            THETA,
            TIMES,
            OBSERVATIONS[:, 0],
            NOISE
        )


@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf]
)
def test_nonfinite_observations(invalid_value):

    observations = OBSERVATIONS.copy()

    observations[0, 0] = invalid_value

    with pytest.raises(ValueError):

        log_likelihood(
            THETA, TIMES, observations, NOISE
        )


# --------------------------------------------------
# 6. Reproducibility
# --------------------------------------------------

def test_posterior_reproducibility():

    result1 = log_posterior(
        THETA, TIMES, OBSERVATIONS, NOISE
    )

    result2 = log_posterior(
        THETA, TIMES, OBSERVATIONS, NOISE
    )

    assert result1 == result2


def test_default_noise_consistency():

    implicit = log_posterior(
        THETA, TIMES, OBSERVATIONS
    )

    explicit = log_posterior(
        THETA, TIMES, OBSERVATIONS, 0.5
    )

    assert implicit == explicit
