import numpy as np
import pytest

from rl_pid_tuner import Plant
from rl_pid_tuner.metrics import StepMetrics
from rl_pid_tuner.spec import (
    PRESETS,
    UNSTABLE_COST,
    cost,
    get_spec,
    make_sim_settings,
    meets_spec,
    run_trial,
)


def _m(**kw):
    base = dict(stable=True, overshoot_pct=0.0, rise_time=1.0, settling_time=2.0, settled=True,
                sse=0.0, iae=1.0, ise=1.0, u_peak=1.0, u_tv=2.0, t_window=10.0)
    base.update(kw)
    return StepMetrics(**base)


def test_presets_and_get_spec():
    assert {"fast", "balanced", "no_overshoot", "robust"} <= set(PRESETS)
    assert get_spec(None).name == "balanced"
    s = get_spec({"preset": "fast", "max_overshoot_pct": 7.0, "u_limits": [-12, 12]})
    assert s.max_overshoot_pct == 7.0 and s.u_limits == (-12, 12)
    with pytest.raises(ValueError):
        get_spec("nope")


def test_cost_ordering_and_unstable():
    spec = get_spec("balanced")
    assert cost(StepMetrics(stable=False), spec) == UNSTABLE_COST
    assert cost(_m(overshoot_pct=40.0), spec) > cost(_m(overshoot_pct=2.0), spec)
    assert cost(_m(settling_time=8.0), spec) > cost(_m(settling_time=1.0), spec)
    assert cost(_m(sse=0.5), spec) > cost(_m(sse=0.0), spec)
    assert 0.0 <= cost(_m(), spec) < UNSTABLE_COST


def test_overshoot_limit_adds_penalty():
    spec = get_spec("balanced")                 # max_overshoot_pct = 10
    inside = cost(_m(overshoot_pct=10.0), spec)
    outside = cost(_m(overshoot_pct=30.0), spec)
    assert outside - inside > spec.penalty * 0.19    # penalty alone is 5 * 0.20


def test_meets_spec():
    spec = get_spec("balanced")
    assert meets_spec(_m(overshoot_pct=5.0, phase_margin_deg=60.0), spec)
    assert not meets_spec(_m(overshoot_pct=50.0), spec)
    assert not meets_spec(_m(settled=False), spec)
    assert not meets_spec(StepMetrics(stable=False), spec)


def test_sim_settings_and_default_limits():
    plant = Plant.dc_motor_position()
    spec = get_spec("balanced")
    st = make_sim_settings(plant, spec)
    assert st.dt > 0 and st.t_end > 0 and np.isclose(st.derivative_filter_n, 1.0 / st.dt)
    assert st.t_disturbance is not None and st.d_amplitude > 0
    assert st.u_limits is not None and st.extra["default_u_limits"] is True
    own = make_sim_settings(plant, get_spec({"u_limits": (-12, 12)}))
    assert own.u_limits == (-12, 12) and own.extra["default_u_limits"] is False


def test_run_trial_unstable_short_circuits():
    plant = Plant.dc_motor_position()
    spec = get_spec("balanced")
    resp, m, c = run_trial(plant, (500.0, 0.0, 0.0), spec, make_sim_settings(plant, spec))
    assert resp is None and not m.stable and c == UNSTABLE_COST
