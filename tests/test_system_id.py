import numpy as np
import pytest

from rl_pid_tuner import Plant
from rl_pid_tuner.system_id import SystemIdentification, identify, _step_first_order, _step_integrator_first_order, _step_second_order


def _noisy(y, rng, rel=0.005):
    return y + rng.normal(0.0, rel * np.max(np.abs(y)), size=y.shape)


def test_first_order_recovery():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 20, 400)
    plant, rep = identify(t, _noisy(_step_first_order(t, 2.5, 3.0), rng))
    assert rep.model == "first_order" and rep.r2 > 0.98
    assert np.isclose(rep.params["K"], 2.5, rtol=0.05) and np.isclose(rep.params["tau"], 3.0, rtol=0.05)
    assert np.isclose(plant.dc_gain, 2.5, rtol=0.05)


def test_underdamped_second_order_recovery():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 12, 600)
    _, rep = identify(t, _noisy(_step_second_order(t, 1.5, 3.0, 0.4), rng))
    assert rep.model == "second_order" and rep.r2 > 0.98
    assert np.isclose(rep.params["K"], 1.5, rtol=0.05)
    assert np.isclose(rep.params["wn"], 3.0, rtol=0.05)
    assert np.isclose(rep.params["zeta"], 0.4, rtol=0.05)


def test_overdamped_second_order_fits_well():
    rng = np.random.default_rng(2)
    t = np.linspace(0, 15, 500)
    _, rep = identify(t, _noisy(_step_second_order(t, 1.0, 2.0, 1.5), rng), model="second_order")
    assert rep.r2 > 0.98 and np.isclose(rep.params["K"], 1.0, rtol=0.05)


def test_integrator_first_order_recovery():
    rng = np.random.default_rng(3)
    t = np.linspace(0, 10, 500)
    _, rep = identify(t, _noisy(_step_integrator_first_order(t, 0.8, 0.5), rng))
    assert rep.model == "integrator_first_order" and rep.r2 > 0.98
    assert np.isclose(rep.params["K"], 0.8, rtol=0.05) and np.isclose(rep.params["tau"], 0.5, rtol=0.05)


def test_u_step_and_offset_are_handled():
    t = np.linspace(0, 20, 300)
    y = 5.0 + 2.0 * _step_first_order(t, 2.5, 3.0)          # step of size 2, output starts at 5
    _, rep = identify(t, y, u_step=2.0)
    assert np.isclose(rep.params["K"], 2.5, rtol=0.02)


def test_identified_plant_is_usable_and_wrapper():
    t = np.linspace(0, 20, 300)
    ident = SystemIdentification()
    plant = ident.identify_plant(t, _step_first_order(t, 1.0, 2.0))
    assert isinstance(plant, Plant) and "s" in ident.plant_transfer_function
    assert ident.report.model == "first_order"


def test_bad_inputs_raise():
    t = np.linspace(0, 1, 50)
    with pytest.raises(ValueError):
        identify(t[:5], np.ones(5))
    with pytest.raises(ValueError):
        identify(t, np.zeros(50))
    with pytest.raises(ValueError):
        identify(t, np.ones(50), model="cubic")
    with pytest.raises(ValueError):
        identify(t[::-1], np.ones(50))
