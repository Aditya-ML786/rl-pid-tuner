import numpy as np
import pytest
from scipy import signal

from rl_pid_tuner import Plant


def test_dc_motor_coefficients():
    p = Plant.dc_motor_position()
    assert np.allclose(p.num, [0.01])
    assert np.allclose(p.den, [0.005, 0.06, 0.1001, 0.0])


def test_dc_motor_poles():
    poles = np.sort(Plant.dc_motor_position().poles.real)
    assert np.allclose(poles, [-10.0, -2.0, 0.0], atol=1e-2) or np.allclose(
        poles, [-10.0, -2.0, 0.0], atol=0.05)


def test_dc_gain():
    assert np.isclose(Plant.first_order(K=2.0, tau=0.5).dc_gain, 2.0)
    assert np.isclose(Plant.second_order(wn=2, zeta=0.5, K=3).dc_gain, 3.0)
    assert Plant.dc_motor_position().dc_gain == float("inf")


def test_state_space_matches_transfer_function():
    p = Plant.second_order(wn=3.0, zeta=0.4)
    t = np.linspace(0, 5, 500)
    _, y_tf = signal.step((p.num, p.den), T=t)
    _, y_ss = signal.step((p.A, p.B, p.C, p.D), T=t)
    assert np.allclose(y_tf, y_ss, atol=1e-6)


def test_from_ss_roundtrip():
    p = Plant.second_order(wn=2.0, zeta=0.5)
    q = Plant.from_ss(p.A, p.B, p.C, p.D)
    assert np.allclose(q.num, p.num) and np.allclose(q.den, p.den)


def test_time_scale():
    assert np.isclose(Plant.first_order(tau=0.5).time_scale(), 0.5)
    assert np.isclose(Plant.dc_motor_position().time_scale(), 0.5, rtol=0.05)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        Plant.from_tf([1, 0, 0], [1, 1])            # improper
    with pytest.raises(ValueError):
        Plant.from_tf([[1], [2]], [1, 1])           # not 1-D
    with pytest.raises(ValueError):
        Plant.from_tf([1], [1, np.nan])             # NaN
    with pytest.raises(ValueError):
        Plant.from_tf([0], [1, 1])                  # zero numerator
    with pytest.raises(ValueError):
        Plant.from_tf([1], [1])                     # no dynamics
    with pytest.raises(ValueError):                 # MIMO (2 outputs)
        Plant.from_ss(np.eye(2) * -1, np.ones((2, 1)), np.eye(2), 0)


def test_perturbed_keeps_integrator_and_is_reproducible():
    p = Plant.dc_motor_position()
    a = p.perturbed(0.2, np.random.default_rng(1))
    b = p.perturbed(0.2, np.random.default_rng(1))
    assert a.den[-1] == 0.0                         # integrator kept
    assert np.allclose(a.den, b.den) and np.allclose(a.num, b.num)
    assert np.all(np.abs(a.den[:-1] / p.den[:-1] - 1) <= 0.2 + 1e-12)
    assert np.allclose(p.perturbed(0.0).den, p.den)


def test_discretize_and_simulate_response_first_order():
    p = Plant.first_order(K=1.0, tau=1.0)
    dt = 0.01
    Ad, Bd, Cd, Dd = p.discretize(dt)
    assert Ad.shape == (1, 1) and Bd.shape == (1, 1)
    assert Cd.shape == (1, 1) and Dd.shape == (1, 1)

    p.reset()
    n_steps = 200
    ys = np.array([p.simulate_response(1.0, dt=dt) for _ in range(n_steps)])
    t = np.arange(n_steps) * dt
    analytic = 1.0 - np.exp(-t)
    assert np.allclose(ys, analytic, atol=1e-6)

