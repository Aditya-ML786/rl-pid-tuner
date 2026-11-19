"""Tuning specification (what 'good' means), shared simulation settings, and the cost function (F6).

The cost defined here is *the* quantity every method minimises: the RL reward is ``-cost``, the
numerical optimizer minimises ``cost``, and the baselines are ranked with it. Using one cost keeps
the comparison fair.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np

from .metrics import StepMetrics, compute_metrics
from .plant import Plant
from .simulation import Response, choose_sim_time_and_dt, is_closed_loop_stable, simulate

UNSTABLE_COST = 100.0


@dataclass
class TuningSpec:
    """Weights (what you care about) and limits (what you require) for a tuning run.

    Every term is normalised to a unit-free number (e.g. settling time / window length), so the
    same preset behaves sensibly on a fast motor and on a slow oven.
    """

    name: str = "custom"
    # --- weights (higher = care more) ---
    w_overshoot: float = 1.0
    w_settling: float = 1.0
    w_rise: float = 0.2
    w_sse: float = 2.0
    w_iae: float = 0.5
    w_effort: float = 0.05
    w_disturbance: float = 0.5
    w_margin: float = 0.0
    # --- hard-ish requirements: violating them adds ``penalty * excess`` to the cost ---
    max_overshoot_pct: float | None = 10.0
    max_settling_time: float | None = None   # seconds
    min_phase_margin_deg: float | None = 45.0
    penalty: float = 5.0
    # --- simulation setup ---
    setpoint: float = 1.0
    disturbance: bool = True                 # inject a load step to measure disturbance rejection
    u_limits: tuple[float | None, float | None] | None = None  # actuator limits, e.g. (-12, 12)

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS: dict[str, TuningSpec] = {
    # Speed first; tolerate more overshoot.
    "fast": TuningSpec("fast", w_overshoot=0.3, w_settling=2.0, w_rise=1.0, max_overshoot_pct=25.0,
                       min_phase_margin_deg=30.0),
    # Sensible all-round default.
    "balanced": TuningSpec("balanced"),
    # Smooth, almost no overshoot (slower).
    "no_overshoot": TuningSpec("no_overshoot", w_overshoot=4.0, w_settling=0.7, w_rise=0.1,
                               max_overshoot_pct=2.0),
    # Large stability margins and gentle actuator use.
    "robust": TuningSpec("robust", w_overshoot=1.0, w_settling=0.5, w_effort=0.2, w_margin=2.0,
                         max_overshoot_pct=5.0, min_phase_margin_deg=60.0),
}


def get_spec(spec: str | dict | TuningSpec | None) -> TuningSpec:
    """Accept a preset name, a dict of overrides (optionally with ``preset``), or a spec."""
    if spec is None:
        return replace(PRESETS["balanced"])
    if isinstance(spec, TuningSpec):
        return spec
    if isinstance(spec, str):
        if spec not in PRESETS:
            raise ValueError(f"Unknown spec preset '{spec}'. Choose from {sorted(PRESETS)}.")
        return replace(PRESETS[spec])
    if isinstance(spec, dict):
        d = dict(spec)
        base = replace(PRESETS[d.pop("preset", "balanced")])
        if "u_limits" in d and d["u_limits"] is not None:
            d["u_limits"] = tuple(d["u_limits"])
        return replace(base, **d)
    raise TypeError("spec must be a preset name, dict, or TuningSpec.")


# --------------------------------------------------------------------------------------
# Shared simulation settings
# --------------------------------------------------------------------------------------
@dataclass
class SimSettings:
    """Timing/filter settings shared by *all* methods so comparisons are apples-to-apples."""

    dt: float
    t_end: float
    derivative_filter_n: float
    t_disturbance: float | None = None
    d_amplitude: float = 0.0
    setpoint: float = 1.0
    u_limits: tuple[float | None, float | None] | None = None
    extra: dict = field(default_factory=dict)


def disturbance_amplitude(plant: Plant, setpoint: float, scale: float = 0.2) -> float:
    """A representative load step: ~20 % of the control effort needed to hold the setpoint."""
    g0 = plant.dc_gain
    if np.isfinite(g0) and abs(g0) > 1e-12:
        return scale * abs(setpoint) / abs(g0)
    if len(plant.den) >= 2 and abs(plant.den[-2]) > 1e-12:     # integrating plant: velocity gain
        kv = plant.num[-1] / plant.den[-2]
        if abs(kv) > 1e-12:
            return scale * abs(setpoint) / (abs(kv) * plant.time_scale())
    return scale * abs(setpoint)


def make_sim_settings(plant: Plant, spec: TuningSpec, dt: float | None = None,
                      t_end: float | None = None, window_factor: float = 20.0) -> SimSettings:
    """Pick dt / horizon / derivative filter from the (nominal) plant's own time scales."""
    if t_end is None:
        t_end = float(np.clip(window_factor * plant.time_scale(), 4.0, 400.0))
    t_end, dt = choose_sim_time_and_dt(plant, dt=dt, t_end=t_end)
    t_dist = 0.6 * t_end if spec.disturbance else None
    amp = disturbance_amplitude(plant, spec.setpoint) if spec.disturbance else 0.0
    # Actuator limits: use the user's (spec.u_limits) or, by default, +/-10x the effort needed to
    # hold the setpoint. Without *some* limit, "faster" is always better and the tuner would
    # chase absurd gains. Set spec.u_limits to your real hardware limits (e.g. (-12, 12)).
    u_limits = spec.u_limits
    default_limits = u_limits is None
    if default_limits:
        lim = 10.0 * disturbance_amplitude(plant, spec.setpoint, scale=1.0)
        u_limits = (-lim, lim)
    # Derivative filter at the sampling bandwidth: removes sample-to-sample jitter, keeps damping.
    return SimSettings(dt=dt, t_end=t_end, derivative_filter_n=1.0 / dt, t_disturbance=t_dist,
                       d_amplitude=amp, setpoint=spec.setpoint, u_limits=u_limits,
                       extra={"default_u_limits": default_limits})


def run_trial(plant: Plant, gains, spec: TuningSpec, settings: SimSettings,
              pre_check: bool = True) -> tuple[Response | None, StepMetrics, float]:
    """Simulate ``gains`` on ``plant`` and return ``(response, metrics, cost)``.

    ``pre_check`` rejects gains whose continuous closed loop has RHP poles *before* simulating
    (cheap, and saves a full simulation on hopeless candidates).
    """
    g = tuple(float(v) for v in gains[:3]) if not isinstance(gains, dict) else gains
    if pre_check and not is_closed_loop_stable(plant, g, derivative_filter_n=settings.derivative_filter_n):
        m = StepMetrics(stable=False)
        return None, m, UNSTABLE_COST
    disturbance = None
    if settings.t_disturbance is not None:
        disturbance = (settings.t_disturbance, settings.d_amplitude)
    resp = simulate(plant, g, setpoint=settings.setpoint, disturbance=disturbance,
                    u_limits=settings.u_limits, dt=settings.dt, t_end=settings.t_end,
                    derivative_filter_n=settings.derivative_filter_n)
    m = compute_metrics(resp, plant, g, settings.derivative_filter_n)
    return resp, m, cost(m, spec)


# --------------------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------------------
def cost_terms(m: StepMetrics, spec: TuningSpec) -> dict[str, float]:
    """The individual, normalised cost terms (each capped so one bad term cannot dominate)."""
    sp = abs(spec.setpoint) if abs(spec.setpoint) > 1e-12 else 1.0
    tw = max(m.t_window, 1e-9)
    pm = m.phase_margin_deg
    gm = m.gain_margin
    pm_term = 0.0 if not np.isfinite(pm) else float(np.clip((60.0 - pm) / 60.0, 0.0, 1.0))
    gm_term = 0.0 if not np.isfinite(gm) else float(np.clip((3.0 - gm) / 3.0, 0.0, 1.0))
    return {
        "overshoot": min(m.overshoot_pct / 100.0, 5.0),
        "settling": min(m.settling_time / tw, 1.0),
        "rise": min(m.rise_time / tw, 1.0),
        "sse": min(abs(m.sse) / sp, 1.0),
        "iae": min(m.iae / (sp * tw), 1.0),
        "effort": min(m.u_tv / (m.u_peak + 1e-9) / 10.0, 5.0),
        "disturbance": min(0.5 * m.dist_peak / sp + 0.5 * m.dist_iae / (sp * tw), 5.0),
        "margin": 0.5 * (pm_term + gm_term),
    }


def cost(m: StepMetrics, spec: TuningSpec) -> float:
    """Scalar cost to minimise (RL reward = -cost). Unstable responses cost ``UNSTABLE_COST``."""
    if not m.stable:
        return UNSTABLE_COST
    t = cost_terms(m, spec)
    c = (spec.w_overshoot * t["overshoot"] + spec.w_settling * t["settling"] + spec.w_rise * t["rise"]
         + spec.w_sse * t["sse"] + spec.w_iae * t["iae"] + spec.w_effort * t["effort"]
         + spec.w_disturbance * t["disturbance"] + spec.w_margin * t["margin"])
    # Requirement violations: extra, proportional penalties.
    if spec.max_overshoot_pct is not None:
        c += spec.penalty * max(0.0, (m.overshoot_pct - spec.max_overshoot_pct) / 100.0)
    if spec.max_settling_time is not None:
        over = (m.settling_time - spec.max_settling_time) / spec.max_settling_time
        c += spec.penalty * float(np.clip(over, 0.0, 5.0))
    if spec.min_phase_margin_deg is not None and np.isfinite(m.phase_margin_deg):
        c += spec.penalty * max(0.0, (spec.min_phase_margin_deg - m.phase_margin_deg) / spec.min_phase_margin_deg)
    return float(min(c, UNSTABLE_COST - 1.0))


def meets_spec(m: StepMetrics, spec: TuningSpec) -> bool:
    """True if every *requirement* in the spec (overshoot / settling / margin / sse) is met."""
    if not m.stable:
        return False
    ok = abs(m.sse) <= 0.02 * max(abs(spec.setpoint), 1e-12) and m.settled
    if spec.max_overshoot_pct is not None:
        ok = ok and m.overshoot_pct <= spec.max_overshoot_pct
    if spec.max_settling_time is not None:
        ok = ok and m.settling_time <= spec.max_settling_time
    if spec.min_phase_margin_deg is not None and np.isfinite(m.phase_margin_deg):
        ok = ok and m.phase_margin_deg >= spec.min_phase_margin_deg
    return bool(ok)
