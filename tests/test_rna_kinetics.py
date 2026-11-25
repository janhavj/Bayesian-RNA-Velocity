"""Mathematical validation of the RNA kinetic model."""

import numpy as np
import pytest

from scipy.integrate import solve_ivp

from rna_kinetics import (
    validate_parameters,
    rna_kinetics,
    simulate,
    velocity,
    calculate_velocity,
    calculate_jacobian,
    peak_time
)

THETA = np.array([5.0, 0.8, 0.4])
TIMES = np.linspace(0, 20, 400)


# --------------------------------------------------
# 1. Parameter validation
# --------------------------------------------------

def test_valid_parameters():

    np.testing.assert_allclose(
        validate_parameters(THETA),
        THETA
    )


@pytest.mark.parametrize(
    "theta",
    [
        [-5.0, 0.8, 0.4],
        [5.0, 0.0, 0.4],
        [5.0, 0.8, -0.4],
        [5.0, 0.8],
        [np.nan, 0.8, 0.4],
        [5.0, np.inf, 0.4]
    ]
)
def test_invalid_parameters(theta):

    with pytest.raises(ValueError):
        validate_parameters(theta)


# --------------------------------------------------
# 2. Initial conditions
# --------------------------------------------------

def test_initial_abundance():

    result = simulate(THETA, np.array([0.0]))

    np.testing.assert_allclose(
        result[0],
        [0.0, 0.0],
        atol=1e-12
    )


def test_initial_velocity():

    result = calculate_velocity(
        THETA,
        np.array([0.0])
    )

    np.testing.assert_allclose(
        result[0],
        [5.0, 0.0],
        atol=1e-12
    )


# --------------------------------------------------
# 3. Output dimensions
# --------------------------------------------------

def test_trajectory_shape():

    assert simulate(THETA, TIMES).shape == (400, 2)


def test_velocity_shape():

    assert calculate_velocity(
        THETA, TIMES
    ).shape == (400, 2)


def test_jacobian_shape():

    assert calculate_jacobian(
        THETA, TIMES
    ).shape == (400, 2, 3)


# --------------------------------------------------
# 4. Biological constraints
# --------------------------------------------------

def test_nonnegative_abundance():

    trajectory = simulate(THETA, TIMES)

    assert np.all(trajectory >= -1e-12)


def test_monotonic_abundance():

    trajectory = simulate(THETA, TIMES)

    assert np.all(
        np.diff(trajectory, axis=0) >= -1e-12
    )


# --------------------------------------------------
# 5. Steady-state behaviour
# --------------------------------------------------

def test_steady_state_abundance():

    alpha, beta, gamma = THETA

    expected = [
        alpha / beta,
        alpha / gamma
    ]

    result = simulate(
        THETA,
        np.array([100.0])
    )

    np.testing.assert_allclose(
        result[0],
        expected,
        rtol=1e-8
    )


def test_steady_state_velocity():

    result = calculate_velocity(
        THETA,
        np.array([100.0])
    )

    np.testing.assert_allclose(
        result[0],
        [0.0, 0.0],
        atol=1e-10
    )


# --------------------------------------------------
# 6. Independent numerical ODE verification
# --------------------------------------------------

@pytest.mark.parametrize(
    "theta",
    [
        [5.0, 0.8, 0.4],
        [5.0, 1.6, 0.4],
        [5.0, 0.4, 0.4],
        [5.0, 0.8, 0.8],
        [5.0, 0.8, 0.2],
        [5.0, 0.4, 0.40000001],
        [5.0, 0.4, 0.400001]
    ]
)
def test_analytical_vs_numerical_ode(theta):

    def independent_rhs(t, y):

        alpha, beta, gamma = theta
        u, s = y

        return [
            alpha - beta * u,
            beta * u - gamma * s
        ]

    numerical = solve_ivp(
        independent_rhs,
        (TIMES[0], TIMES[-1]),
        [0.0, 0.0],
        t_eval=TIMES,
        rtol=1e-10,
        atol=1e-12
    )

    assert numerical.success

    analytical = simulate(theta, TIMES)

    np.testing.assert_allclose(
        analytical,
        numerical.y.T,
        rtol=1e-6,
        atol=1e-7
    )


# --------------------------------------------------
# 7. Velocity verification
# --------------------------------------------------

def test_velocity_consistency():

    trajectory = simulate(THETA, TIMES)

    np.testing.assert_allclose(
        velocity(THETA, trajectory),
        calculate_velocity(THETA, TIMES),
        atol=1e-12
    )


def test_velocity_against_ode():

    trajectory = simulate(THETA, TIMES)

    expected = np.array([
        rna_kinetics(t, y, *THETA)
        for t, y in zip(TIMES, trajectory)
    ])

    np.testing.assert_allclose(
        calculate_velocity(THETA, TIMES),
        expected,
        atol=1e-12
    )


def test_velocity_against_numerical_derivative():

    t = np.linspace(0.1, 20, 400)
    h = 1e-5

    numerical_derivative = (
        simulate(THETA, t + h)
        - simulate(THETA, t - h)
    ) / (2 * h)

    analytical_velocity = calculate_velocity(
        THETA, t
    )

    np.testing.assert_allclose(
        analytical_velocity,
        numerical_derivative,
        rtol=1e-5,
        atol=1e-7
    )


# --------------------------------------------------
# 8. Peak RNA velocity
# --------------------------------------------------

def test_peak_velocity_time():

    expected = np.log(2) / 0.4

    assert peak_time(THETA) == pytest.approx(
        expected,
        rel=1e-10
    )


def test_peak_velocity_magnitude():

    t_peak = peak_time(THETA)

    result = calculate_velocity(
        THETA,
        np.array([t_peak])
    )

    assert result[0, 1] == pytest.approx(
        2.5,
        rel=1e-10
    )


def test_equal_rate_peak_time():

    theta = [5.0, 0.4, 0.4]

    assert peak_time(theta) == pytest.approx(2.5)


def test_peak_is_local_maximum():

    t_peak = peak_time(THETA)

    times = np.array([
        t_peak - 0.01,
        t_peak,
        t_peak + 0.01
    ])

    v = calculate_velocity(THETA, times)[:, 1]

    assert v[1] > v[0]
    assert v[1] > v[2]


# --------------------------------------------------
# 9. Parameter sensitivity
# --------------------------------------------------

def test_transcription_sensitivity():

    trajectory = simulate(THETA, TIMES)

    jacobian = calculate_jacobian(
        THETA, TIMES
    )

    expected = trajectory / THETA[0]

    np.testing.assert_allclose(
        jacobian[:, :, 0],
        expected,
        rtol=1e-6,
        atol=1e-8
    )


def test_unspliced_degradation_sensitivity():

    jacobian = calculate_jacobian(
        THETA, TIMES
    )

    np.testing.assert_allclose(
        jacobian[:, 0, 2],
        0.0,
        atol=1e-10
    )


def test_steady_state_normalised_sensitivity():

    t = np.array([100.0])

    trajectory = simulate(THETA, t)

    jacobian = calculate_jacobian(
        THETA, t
    )

    normalised = (
        jacobian[0] * THETA
        / trajectory[0, :, None]
    )

    expected = np.array([
        [1.0, -1.0, 0.0],
        [1.0, 0.0, -1.0]
    ])

    np.testing.assert_allclose(
        normalised,
        expected,
        atol=1e-6
    )


# --------------------------------------------------
# 10. Invalid inputs
# --------------------------------------------------

def test_negative_time():

    with pytest.raises(ValueError):
        simulate(THETA, [-1.0, 0.0])


def test_nonfinite_time():

    with pytest.raises(ValueError):
        simulate(THETA, [0.0, np.nan])


def test_invalid_trajectory_shape():

    with pytest.raises(ValueError):
        velocity(THETA, [1.0, 2.0])


def test_invalid_finite_difference_step():

    with pytest.raises(ValueError):
        calculate_jacobian(
            THETA,
            TIMES,
            relative_step=0
        )


def test_invalid_time_dimensions():

    with pytest.raises(ValueError):
        simulate(
            THETA,
            np.array([[0.0, 1.0], [2.0, 3.0]])
        )
