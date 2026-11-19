import numpy as np
import pytest

from rl_pid_tuner import PIDController, PIDGains


def test_p_i_d_terms_individually():
    dt = 0.1

    # Pure P
    p = PIDController(2.5, 0.0, 0.0)
    assert np.isclose(p.compute(4.0, dt=dt), 10.0)
    assert np.isclose(p.compute(-1.0, dt=dt), -2.5)

    # Pure I: integral accumulates error * dt
    i = PIDController(0.0, 3.0, 0.0)
    assert np.isclose(i.compute(2.0, dt=dt), 3.0 * (2.0 * dt))
    assert np.isclose(i.compute(2.0, dt=dt), 3.0 * (4.0 * dt))

    # Pure D (unfiltered, on error): derivative = (e - prev_e) / dt
    d = PIDController(0.0, 0.0, 0.5)
    assert np.isclose(d.compute(1.0, dt=dt), 0.5 * (1.0 - 0.0) / dt)
    assert np.isclose(d.compute(1.0, dt=dt), 0.0)


def test_derivative_on_measurement_avoids_setpoint_kick():
    dt = 0.01
    # Setpoint jumps from 0 to 1 while plant measurement is still 0.
    d_err = PIDController(0.0, 0.0, 1.0, derivative_on_measurement=False)
    d_meas = PIDController(0.0, 0.0, 1.0, derivative_on_measurement=True)

    u_err = d_err.compute(error=1.0, dt=dt, measurement=0.0)
    u_meas = d_meas.compute(error=1.0, dt=dt, measurement=0.0)

    # Derivative-on-error spikes to 1.0 / 0.01 = 100; derivative-on-measurement stays 0.
    assert np.isclose(u_err, 100.0)
    assert np.isclose(u_meas, 0.0)


def test_derivative_filter_reduces_spike():
    dt = 0.01
    unfiltered = PIDController(0.0, 0.0, 1.0, derivative_filter_n=None)
    filtered = PIDController(0.0, 0.0, 1.0, derivative_filter_n=20.0)

    u_raw = unfiltered.compute(error=1.0, dt=dt)
    u_filt = filtered.compute(error=1.0, dt=dt)

    assert u_filt > 0.0
    assert u_filt < u_raw


def test_output_limits_and_anti_windup_clamping():
    dt = 0.1
    ctrl_aw = PIDController(1.0, 5.0, 0.0, u_limits=(-2.0, 2.0), anti_windup=True)
    ctrl_no_aw = PIDController(1.0, 5.0, 0.0, u_limits=(-2.0, 2.0), anti_windup=False)

    # Drive both with large positive error so output saturates at +2.0.
    for _ in range(10):
        u_aw = ctrl_aw.compute(5.0, dt=dt)
        u_no = ctrl_no_aw.compute(5.0, dt=dt)
        assert np.isclose(u_aw, 2.0)
        assert np.isclose(u_no, 2.0)

    # Clamping froze the integral once saturated, whereas non-anti-windup kept growing.
    assert ctrl_aw.integral < ctrl_no_aw.integral


def test_reset_set_gains_and_validation():
    c = PIDController(1.0, 2.0, 3.0)
    c.compute(1.0, dt=0.05)
    c.reset()
    assert np.isclose(c.integral, 0.0)
    assert np.isclose(c.prev_error, 0.0)
    c.set_gains(4.0, 5.0, 6.0)
    assert (c.kp, c.ki, c.kd) == (4.0, 5.0, 6.0)
    assert PIDGains(1.0, 2.0, 3.0).as_tuple() == (1.0, 2.0, 3.0)

    with pytest.raises(ValueError):
        PIDController(1.0, 0.0, 0.0, derivative_filter_n=-5.0)
    with pytest.raises(ValueError):
        PIDController(1.0, 0.0, 0.0, u_limits=(5.0, 1.0))
