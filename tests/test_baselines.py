import numpy as np
import pytest

from rl_pid_tuner import Plant, is_closed_loop_stable
from rl_pid_tuner.baselines import (
    heuristic_gains,
    optimize_gains,
    reference_gains,
    simc,
    ziegler_nichols,
)
from rl_pid_tuner.spec import get_spec, make_sim_settings, run_trial


def test_zn_dc_motor_matches_notes():
    zn = ziegler_nichols(Plant.dc_motor_position())
    assert np.isclose(zn.ku, 120.12, rtol=1e-3)
    assert np.isclose(zn.tu, 1.404, rtol=1e-2)
    assert np.allclose(zn.gains(), (72.07, 102.65, 12.65), rtol=5e-3)


def test_zn_raises_when_no_phase_crossover():
    with pytest.raises(ValueError):
        ziegler_nichols(Plant.first_order())
    with pytest.raises(ValueError):
        ziegler_nichols(Plant.second_order())


def test_simc_first_order_closed_form():
    # K=2, tau=50, default tau_c = tau/4 = 12.5 -> Kc = tau/(K tau_c) = 2, Ti = min(tau, 4 tau_c) = 50
    kp, ki, kd = simc(Plant.first_order(K=2.0, tau=50.0))
    assert np.isclose(kp, 2.0) and np.isclose(ki, 0.04) and np.isclose(kd, 0.0)


@pytest.mark.parametrize("plant", [
    Plant.dc_motor_position(),
    Plant.first_order(2.0, 50.0),
    Plant.second_order(2.0, 0.1),
    Plant.second_order(1.0, 1.0),
    Plant.from_tf([1.0], [5.0, 1.0, 0.0]),
])
def test_simc_gives_stable_loop(plant):
    gains = simc(plant)
    assert all(g >= 0 for g in gains)
    assert is_closed_loop_stable(plant, gains)


def test_simc_rejects_unsupported_plants():
    with pytest.raises(ValueError):
        simc(Plant.from_tf([1.0, 1.0], [1.0, 3.0, 2.0]))     # has a zero
    with pytest.raises(ValueError):
        simc(Plant.from_tf([1.0], [1.0, -1.0]))              # unstable


def test_reference_gains_methods():
    gains, name = reference_gains(Plant.dc_motor_position())
    assert name == "simc" and gains[0] > 0
    _, name = reference_gains(Plant.dc_motor_position(), "zn")
    assert name == "zn"
    _, name = reference_gains(Plant.first_order(), "zn")      # ZN impossible -> falls back
    assert name == "heuristic"
    assert all(np.isfinite(heuristic_gains(Plant.first_order())))


def test_optimizer_improves_on_simc():
    plant = Plant.second_order(1.0, 1.0)
    spec = get_spec("balanced")
    st = make_sim_settings(plant, spec)
    _, _, c_simc = run_trial(plant, simc(plant), spec, st)
    gains, c_opt = optimize_gains(plant, spec, st, max_evals=150)
    assert c_opt < c_simc
    assert is_closed_loop_stable(plant, gains)
