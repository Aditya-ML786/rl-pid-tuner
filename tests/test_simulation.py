import numpy as np

from rl_pid_tuner import (
    PIDGains,
    Plant,
    is_closed_loop_stable,
    simulate,
)


def test_p_only_final_value_first_order():
    plant = Plant.first_order(K=1.0, tau=1.0)
    kp = 4.0
    resp = simulate(plant, (kp, 0.0, 0.0), setpoint=1.0, dt=0.005, t_end=10.0)
    assert resp.stable
    expected = kp / (1.0 + kp)
    assert np.isclose(resp.y[-1], expected, rtol=1e-2)


def test_pi_zero_steady_state_error():
    plant = Plant.first_order(K=1.0, tau=1.0)
    resp = simulate(plant, PIDGains(kp=2.0, ki=3.0, kd=0.0), setpoint=1.0, dt=0.005, t_end=10.0)
    assert resp.stable
    assert np.isclose(resp.y[-1], 1.0, atol=1e-3)
    assert np.isclose(resp.error[-1], 0.0, atol=1e-3)


def test_anti_windup_reduces_overshoot_under_saturation():
    plant = Plant.first_order(K=1.0, tau=1.0)
    gains = (4.0, 8.0, 0.0)
    resp_aw = simulate(
        plant, gains, setpoint=1.0, u_limits=(-2.0, 2.0), anti_windup=True, dt=0.01, t_end=8.0
    )
    resp_no_aw = simulate(
        plant, gains, setpoint=1.0, u_limits=(-2.0, 2.0), anti_windup=False, dt=0.01, t_end=8.0
    )
    assert resp_aw.stable and resp_no_aw.stable
    assert resp_aw.y.max() < resp_no_aw.y.max()
    assert np.all(resp_aw.u <= 2.0 + 1e-12) and np.all(resp_aw.u >= -2.0 - 1e-12)


def test_is_closed_loop_stable_dc_motor_ku():
    motor = Plant.dc_motor_position()
    # Analytic Routh-Hurwitz ultimate gain Ku = 120.12 for P-only control.
    assert is_closed_loop_stable(motor, (50.0, 0.0, 0.0)) is True
    assert is_closed_loop_stable(motor, (150.0, 0.0, 0.0)) is False
    # Filtered ZN gains should also be stable.
    assert is_closed_loop_stable(motor, (72.07, 102.65, 12.65), derivative_filter_n=50.0) is True


def test_divergence_guard_handles_unstable_gains():
    motor = Plant.dc_motor_position()
    resp = simulate(motor, (-10.0, 0.0, 0.0), setpoint=1.0, dt=0.01, t_end=5.0, y_max=10.0)
    assert resp.stable is False
    assert resp.reason == "diverged"
    assert np.all(np.isfinite(resp.y))


def test_check_stability_first_short_circuits():
    motor = Plant.dc_motor_position()
    resp = simulate(motor, (150.0, 0.0, 0.0), check_stability_first=True)
    assert resp.stable is False
    assert resp.reason == "unstable_poles"
    assert len(resp.t) == 1


def test_load_disturbance_recovery_with_integral():
    plant = Plant.first_order(K=1.0, tau=1.0)
    resp = simulate(
        plant,
        (3.0, 4.0, 0.0),
        setpoint=1.0,
        disturbance=(3.0, -0.5),
        dt=0.01,
        t_end=10.0,
    )
    assert resp.stable
    # Output dips after t=3.0s disturbance, then integral action brings it back to setpoint.
    dip = resp.y[(resp.t >= 3.0) & (resp.t <= 5.0)].min()
    assert dip < 0.98
    assert np.isclose(resp.y[-1], 1.0, atol=1e-2)
