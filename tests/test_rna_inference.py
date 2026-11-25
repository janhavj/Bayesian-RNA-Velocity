"""Tests for parameter inference and experimental design."""

import numpy as np
import pytest

from scipy.integrate import solve_ivp

from rna_kinetics import simulate, calculate_jacobian
from rna_inference import observation_times, fit_model

THETA = np.array([5.0, 0.8, 0.4])


# --------------------------------------------------
# 1. Independent numerical observations
# --------------------------------------------------

def numerical_observations(theta, t):

    def rhs(time, y):

        alpha, beta, gamma = theta
        u, s = y

        return [
            alpha - beta * u,
            beta * u - gamma * s
        ]

    solution = solve_ivp(
        rhs,
        (0, 20),
        [0.0, 0.0],
        t_eval=t,
        rtol=1e-11,
        atol=1e-13
    )

    assert solution.success

    return solution.y.T


@pytest.fixture(scope="module")
def baseline_fit():

    t = observation_times(50, "uniform")

    observations = numerical_observations(
        THETA, t
    )

    return fit_model(t, observations, 0.5)


# --------------------------------------------------
# 2. Experimental sampling designs
# --------------------------------------------------

@pytest.mark.parametrize(
    "sampling",
    ["uniform", "early", "late"]
)
def test_sampling_dimensions(sampling):

    t = observation_times(25, sampling)

    assert len(t) == 25


@pytest.mark.parametrize(
    "sampling",
    ["uniform", "early", "late"]
)
def test_sampling_endpoints(sampling):

    t = observation_times(25, sampling)

    assert t[0] == pytest.approx(0.0)
    assert t[-1] == pytest.approx(20.0)


@pytest.mark.parametrize(
    "sampling",
    ["uniform", "early", "late"]
)
def test_sampling_monotonicity(sampling):

    t = observation_times(25, sampling)

    assert np.all(np.diff(t) > 0)


def test_uniform_spacing():

    t = observation_times(25, "uniform")

    np.testing.assert_allclose(
        np.diff(t),
        20 / 24
    )


def test_early_sampling_density():

    t = observation_times(25, "early")

    assert np.diff(t)[0] < np.diff(t)[-1]


def test_late_sampling_density():

    t = observation_times(25, "late")

    assert np.diff(t)[0] > np.diff(t)[-1]


def test_invalid_sampling_strategy():

    with pytest.raises(ValueError):
        observation_times(25, "invalid")


# --------------------------------------------------
# 3. Parameter recovery
# --------------------------------------------------

@pytest.mark.parametrize(
    "theta",
    [
        [5.0, 0.8, 0.4],
        [5.0, 1.6, 0.4],
        [5.0, 0.4, 0.4],
        [5.0, 0.8, 0.8],
        [5.0, 0.8, 0.2]
    ]
)
def test_parameter_recovery(theta):

    t = observation_times(50, "uniform")

    observations = numerical_observations(
        theta, t
    )

    result = fit_model(
        t, observations, 0.5
    )

    estimated = result[0]

    assert result[5]
    assert not result[7]

    np.testing.assert_allclose(
        estimated,
        theta,
        rtol=1e-5,
        atol=1e-7
    )


@pytest.mark.parametrize(
    "sampling",
    ["uniform", "early", "late"]
)
def test_recovery_across_sampling_designs(sampling):

    t = observation_times(50, sampling)

    observations = numerical_observations(
        THETA, t
    )

    result = fit_model(
        t, observations, 0.5
    )

    assert result[5]

    np.testing.assert_allclose(
        result[0],
        THETA,
        rtol=1e-5
    )


# --------------------------------------------------
# 4. Optimisation diagnostics
# --------------------------------------------------

def test_optimisation_success(baseline_fit):

    assert baseline_fit[5]


def test_positive_parameters(baseline_fit):

    assert np.all(baseline_fit[0] > 0)


def test_no_boundary_solution(baseline_fit):

    assert not baseline_fit[7]


# --------------------------------------------------
# 5. Uncertainty estimates
# --------------------------------------------------

def test_valid_uncertainty(baseline_fit):

    assert baseline_fit[6]


def test_finite_confidence_intervals(baseline_fit):

    assert np.all(np.isfinite(baseline_fit[1]))
    assert np.all(np.isfinite(baseline_fit[2]))


def test_confidence_interval_order(baseline_fit):

    estimated, lower, upper = baseline_fit[:3]

    assert np.all(lower < estimated)
    assert np.all(estimated < upper)


def test_confidence_interval_contains_truth(baseline_fit):

    lower = baseline_fit[1]
    upper = baseline_fit[2]

    assert np.all(lower < THETA)
    assert np.all(THETA < upper)


# --------------------------------------------------
# 6. Fisher information and identifiability
# --------------------------------------------------

def test_fisher_information_rank(baseline_fit):

    assert baseline_fit[4] == 3


def test_fisher_information_condition(baseline_fit):

    condition = baseline_fit[3]

    assert np.isfinite(condition)
    assert condition < 1e12


def test_fisher_information_symmetry():

    t = observation_times(50, "uniform")

    jacobian = calculate_jacobian(THETA, t)

    # Convert to log-parameter sensitivities
    # and scale by observation noise.

    J = (jacobian * THETA).reshape(-1, 3) / 0.5

    fim = J.T @ J

    np.testing.assert_allclose(
        fim,
        fim.T,
        atol=1e-12
    )


def test_fisher_information_positive_definite():

    t = observation_times(50, "uniform")

    jacobian = calculate_jacobian(THETA, t)

    J = (jacobian * THETA).reshape(-1, 3) / 0.5

    fim = J.T @ J

    eigenvalues = np.linalg.eigvalsh(fim)

    assert np.all(eigenvalues > 0)


def test_rank_deficient_experiment():

    t = np.zeros(10)

    observations = simulate(THETA, t)

    result = fit_model(t, observations, 0.5)

    assert result[4] < 3
    assert not result[6]

    assert np.all(np.isnan(result[1]))
    assert np.all(np.isnan(result[2]))


# --------------------------------------------------
# 7. Reproducibility
# --------------------------------------------------

def test_seeded_observation_reproducibility():

    t = observation_times(50, "uniform")

    expected = simulate(THETA, t)

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    observations1 = expected + rng1.normal(
        0, 0.5, expected.shape
    )

    observations2 = expected + rng2.normal(
        0, 0.5, expected.shape
    )

    np.testing.assert_array_equal(
        observations1,
        observations2
    )


def test_optimisation_reproducibility():

    t = observation_times(50, "uniform")

    rng = np.random.default_rng(42)

    expected = simulate(THETA, t)

    observations = expected + rng.normal(
        0, 0.5, expected.shape
    )

    result1 = fit_model(t, observations, 0.5)
    result2 = fit_model(t, observations, 0.5)

    np.testing.assert_allclose(
        result1[0],
        result2[0],
        rtol=1e-12
    )
