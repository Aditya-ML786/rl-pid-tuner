import numpy as np
from scipy import signal

from rl_pid_tuner import Plant, simulate
from rl_pid_tuner.baselines import ziegler_nichols
from rl_pid_tuner.metrics import StepMetrics, beats, compare, compute_metrics, stability_margins
from rl_pid_tuner.simulation import Response


def _resp(t, y, sp=1.0, u=None):
    u = np.ones_like(t) if u is None else u
    return Response(t=t, y=y, u=u, e=sp - y, setpoint=sp, stable=True)


def test_golden_dc_motor_zn_numbers():
    """The ZN-tuned ideal PID on the DC motor must reproduce the numbers in the project notes."""
    motor = Plant.dc_motor_position()
    zn = ziegler_nichols(motor)
    resp = simulate(motor, zn.gains(), dt=0.001, t_end=10.0)   # ideal (unfiltered) PID
    m = compute_metrics(resp)
    assert np.isclose(m.overshoot_pct, 61.8, rtol=0.05)
    assert np.isclose(m.settling_time, 5.01, rtol=0.05)
    assert np.isclose(m.rise_time, 0.29, rtol=0.05)
    assert abs(m.sse) < 0.01


def test_first_order_metrics_match_analytic():
    t = np.linspace(0, 10, 10001)
    m = compute_metrics(_resp(t, 1 - np.exp(-t)))
    assert np.isclose(m.overshoot_pct, 0.0)
    assert np.isclose(m.rise_time, np.log(9.0), rtol=1e-2)          # 10 % -> 90 %
    assert np.isclose(m.settling_time, np.log(50.0), rtol=1e-2)     # 2 % band
    assert m.settled and abs(m.sse) < 1e-3


def test_overshoot_matches_second_order_formula():
    zeta = 0.3
    p = Plant.second_order(wn=1.0, zeta=zeta)
    t = np.linspace(0, 40, 4001)
    _, y = signal.step((p.num, p.den), T=t)
    expected = 100 * np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2))
    assert np.isclose(compute_metrics(_resp(t, y)).overshoot_pct, expected, rtol=0.02)


def test_not_settled_is_flagged():
    t = np.linspace(0, 10, 1001)
    m = compute_metrics(_resp(t, 1 - np.exp(-t / 20.0)))             # far too slow
    assert not m.settled and np.isclose(m.settling_time, t[-1])


def test_unstable_response_gives_unstable_metrics():
    r = simulate(Plant.first_order(), (-5.0, 0.0, 0.0), dt=0.01, t_end=5.0, y_max=10.0)
    assert compute_metrics(r).stable is False


def test_margins_known_plants():
    # L = 1/(s(s+1)): crossover at w = 0.786 -> PM = 180 - 90 - atan(w) = 51.8 deg, no GM.
    gm, pm = stability_margins(Plant.from_tf([1], [1, 1, 0]), (1.0, 0.0, 0.0))
    assert np.isinf(gm) and np.isclose(pm, 51.83, atol=0.5)
    # L = 1/(s+1)^3: phase = -180 at w = sqrt(3) where |L| = 1/8 -> GM = 8.
    gm, pm = stability_margins(Plant.from_tf([1], [1, 3, 3, 1]), (1.0, 0.0, 0.0))
    assert np.isclose(gm, 8.0, rtol=0.02)


def test_disturbance_metrics_use_post_disturbance_window():
    p = Plant.first_order()
    r = simulate(p, (3.0, 4.0, 0.0), disturbance=(10.0, -0.5), dt=0.01, t_end=20.0)
    m = compute_metrics(r)
    assert m.dist_peak > 0.05 and m.dist_iae > 0.0
    assert m.t_window <= 10.0 + 1e-9                  # step metrics stop at the disturbance
    assert abs(m.sse) < 0.01


def test_beats_and_compare():
    good = StepMetrics(stable=True, overshoot_pct=5, settling_time=1.0, sse=0.0)
    bad = StepMetrics(stable=True, overshoot_pct=60, settling_time=5.0, sse=0.0)
    unstable = StepMetrics(stable=False)
    assert beats(good, bad) and not beats(bad, good)
    assert beats(good, unstable) and not beats(unstable, good)
    table = compare({"RL": good, "ZN": bad, "X": unstable})
    assert "RL" in table and "ZN" in table and "--" in table
