"""Classical and numerical baselines: Ziegler-Nichols, SIMC, and a Nelder-Mead optimizer (F9).

These are what the RL tuner is compared against. They also provide the *reference gains* the RL
environment starts from.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize

from .plant import Plant
from .spec import SimSettings, TuningSpec, UNSTABLE_COST, run_trial


# --------------------------------------------------------------------------------------
# Ziegler-Nichols (ultimate-gain method)
# --------------------------------------------------------------------------------------
@dataclass
class ZNResult:
    kp: float
    ki: float
    kd: float
    ku: float     # ultimate gain: P-only gain at which the loop just oscillates forever
    tu: float     # ultimate period of that oscillation (s)
    wu: float     # ultimate frequency (rad/s) = 2*pi/tu

    def gains(self) -> tuple[float, float, float]:
        return (self.kp, self.ki, self.kd)


def ziegler_nichols(plant: Plant) -> ZNResult:
    """Classic ZN PID gains from the plant's -180 degree phase crossover.

    Instead of sweeping gains in time-domain simulations, we use the equivalent frequency-domain
    fact: a P-only loop oscillates forever at the frequency w_u where G(jw) is real and negative,
    and the gain that makes it exactly marginal is Ku = -1 / Re(G(j w_u)). The crossover is first
    *bracketed* on a log grid (sign change of Im G) and then refined with ``brentq``, which is
    guaranteed to converge once bracketed. Raises ``ValueError`` if the plant never reaches
    -180 degrees (e.g. 1st-order plants), in which case ZN does not exist.
    """
    num, den = plant.num, plant.den
    w_ref = 1.0 / np.sqrt(plant.time_scale() * plant.fastest_time_constant())
    w = w_ref * np.logspace(-4, 4, 4000)

    def g(wv):
        s = 1j * wv
        return np.polyval(num, s) / np.polyval(den, s)

    gv = g(w)
    brackets = np.flatnonzero((gv.imag[:-1] * gv.imag[1:] <= 0.0) & (gv.real[:-1] < 0.0))
    for i in brackets:
        lo, hi = w[i], w[i + 1]
        flo, fhi = g(lo).imag, g(hi).imag
        wu = lo if flo == 0.0 else (hi if fhi == 0.0 else optimize.brentq(lambda x: g(x).imag, lo, hi, xtol=1e-12))
        re = g(wu).real
        if re < 0.0:
            ku = -1.0 / re
            tu = 2.0 * np.pi / wu
            kp = 0.6 * ku
            return ZNResult(kp, 1.2 * ku / tu, 0.075 * ku * tu, float(ku), float(tu), float(wu))
    raise ValueError(
        "Ziegler-Nichols is not defined for this plant (no -180 degree phase crossover). "
        "Use the SIMC baseline instead."
    )


# --------------------------------------------------------------------------------------
# SIMC (Skogestad) model-based rule
# --------------------------------------------------------------------------------------
def _series_to_parallel(kc: float, ti: float, td: float) -> tuple[float, float, float]:
    """Convert series PID Kc(1+1/(Ti s))(1+Td s) into parallel gains (Kp, Ki, Kd)."""
    return (kc * (1.0 + td / ti), kc / ti, kc * td)


def simc(plant: Plant, tau_c: float | None = None) -> tuple[float, float, float]:
    """SIMC PID gains for stable (or integrating) all-pole plants, no dead time needed.

    Idea: pick the desired closed-loop speed ``tau_c`` (smaller = faster/more aggressive), then
    the rule gives gains in closed form. Higher-order plants are first reduced to a first/second
    order lag plus an *effective delay* using Skogestad's half-rule. Default ``tau_c`` is twice
    the effective delay (or a quarter of the dominant time constant if there is none).
    """
    if len(plant.num) > 1:
        raise ValueError("SIMC baseline supports all-pole plants (constant numerator) only.")
    poles = plant.poles
    n_int = int(np.sum(np.abs(poles) < 1e-9))
    rest = poles[np.abs(poles) >= 1e-9]
    if np.any(rest.real >= 0):
        raise ValueError("SIMC baseline needs a stable plant (apart from one integrator).")
    if n_int > 1:
        raise ValueError("SIMC baseline supports at most one integrator.")
    complex_pair = bool(np.any(np.abs(rest.imag) > 1e-9))

    # ---- stable, underdamped second-order: IMC-PID with lambda = tau_c ----
    if complex_pair:
        if n_int or len(rest) != 2:
            raise ValueError("SIMC baseline: complex poles supported for 2nd-order plants only.")
        k = plant.dc_gain
        wn = float(np.sqrt(np.prod(np.abs(rest))))
        zeta = float(-rest[0].real / wn)
        lam = tau_c if tau_c is not None else 0.25 / (zeta * wn)
        return (2 * zeta / (wn * k * lam), 1.0 / (k * lam), 1.0 / (wn**2 * k * lam))

    lags = np.sort(1.0 / np.abs(rest.real))[::-1]   # time constants, largest first

    if n_int == 0:                                   # self-regulating plant
        k = plant.dc_gain
        if abs(k) < 1e-12:
            raise ValueError("Plant has zero DC gain; SIMC baseline not applicable.")
        tau1 = lags[0]
        tau2 = 0.0
        theta = 0.0
        if len(lags) == 2:
            tau2 = lags[1]
        elif len(lags) >= 3:                         # half-rule
            tau2 = lags[1] + lags[2] / 2.0
            theta = lags[2] / 2.0 + float(np.sum(lags[3:]))
        tc = tau_c if tau_c is not None else (2.0 * theta if theta > 0 else tau1 / 4.0)
        kc = tau1 / (k * (tc + theta))
        ti = min(tau1, 4.0 * (tc + theta))
        return _series_to_parallel(kc, ti, tau2)

    # integrating plant: G = K' / (s * prod(tau_i s + 1))
    kv = plant.num[-1] / plant.den[-2]
    tau_lag, theta = 0.0, 0.0
    if len(lags) >= 1:
        tau_lag = lags[0]
    if len(lags) == 2:
        tau_lag, theta = lags[0] + lags[1] / 2.0, lags[1] / 2.0
    elif len(lags) >= 3:
        tau_lag, theta = lags[0] + lags[1] / 2.0, lags[1] / 2.0 + float(np.sum(lags[2:]))
    dom = tau_lag if tau_lag > 0 else plant.time_scale()
    tc = tau_c if tau_c is not None else (2.0 * theta if theta > 0 else dom / 3.0)
    kc = 1.0 / (kv * (tc + theta))
    ti = 4.0 * (tc + theta)
    return _series_to_parallel(kc, ti, tau_lag)


def heuristic_gains(plant: Plant) -> tuple[float, float, float]:
    """Last-resort conservative starting gains (used if neither ZN nor SIMC apply)."""
    g0 = plant.dc_gain
    tau = plant.time_scale()
    if np.isfinite(g0) and abs(g0) > 1e-12:
        kp = 0.5 / abs(g0)
    else:
        kv = plant.num[-1] / plant.den[-2] if abs(plant.den[-2]) > 1e-12 else 1.0
        kp = 0.5 / (abs(kv) * tau) if abs(kv) > 1e-12 else 1.0
    return (kp, kp / (2.0 * tau), 0.0)


def reference_gains(plant: Plant, method: str = "auto") -> tuple[tuple[float, float, float], str]:
    """Starting/reference gains ``(kp, ki, kd)`` and the name of the method used.

    ``auto`` tries SIMC (gentle), then ZN, then a heuristic. The RL environment expresses its
    gains *relative* to these, which is what lets one policy work on plants of any scale.
    """
    order = {"auto": ["simc", "zn", "heuristic"], "simc": ["simc"], "zn": ["zn"],
             "heuristic": ["heuristic"]}[method]
    for name in order:
        try:
            if name == "simc":
                gains = simc(plant)
            elif name == "zn":
                gains = ziegler_nichols(plant).gains()
            else:
                gains = heuristic_gains(plant)
        except ValueError:
            continue
        if all(np.isfinite(gains)) and gains[0] > 0:
            return tuple(float(v) for v in gains), name
    return heuristic_gains(plant), "heuristic"


# --------------------------------------------------------------------------------------
# Plain numerical optimizer baseline ("is RL even worth it?")
# --------------------------------------------------------------------------------------
def optimize_gains(plant: Plant, spec: TuningSpec, settings: SimSettings, x0=None,
                   max_evals: int = 300) -> tuple[tuple[float, float, float], float]:
    """Nelder-Mead on log10(gains), minimising the same cost the RL agent sees.

    Returns ``(gains, best_cost)``. No learning: this is the honest 'just optimise it' baseline.
    """
    if x0 is None:
        x0, _ = reference_gains(plant)
    floor = 1e-6
    start = np.log10(np.maximum(np.asarray(x0, dtype=float), floor))
    start[1:] = np.maximum(start[1:], np.log10(max(x0[0], floor)) - 4.0)   # keep Ki/Kd off the floor

    best = {"cost": np.inf, "gains": tuple(x0)}

    def objective(z):
        z = np.clip(z, -8.0, 8.0)
        gains = tuple(10.0 ** z)
        _, _, c = run_trial(plant, gains, spec, settings)
        if c < best["cost"]:
            best["cost"], best["gains"] = c, gains
        return c

    optimize.minimize(objective, start, method="Nelder-Mead",
                      options={"maxfev": max_evals, "xatol": 1e-2, "fatol": 1e-4,
                               "initial_simplex": _simplex(start)})
    return best["gains"], float(best["cost"])


def _simplex(start: np.ndarray, step: float = 0.3) -> np.ndarray:
    """Initial Nelder-Mead simplex: +/- ``step`` decades along each gain."""
    pts = [start]
    for i in range(len(start)):
        p = start.copy()
        p[i] += step
        pts.append(p)
    return np.array(pts)


def design_baselines(plant: Plant, spec: TuningSpec, settings: SimSettings,
                     include_optimizer: bool = True, max_evals: int = 300) -> dict[str, tuple]:
    """Gains from every applicable classical method on ``plant``: ``zn`` / ``simc`` / ``optimizer``.

    Methods that do not apply to the plant (e.g. ZN on a 1st-order plant) are simply left out.
    """
    out: dict[str, tuple] = {}
    for name, fn in (("zn", lambda p: ziegler_nichols(p).gains()), ("simc", simc)):
        try:
            out[name] = tuple(float(v) for v in fn(plant))
        except ValueError:
            pass
    if include_optimizer:
        out["optimizer"] = tuple(float(v) for v in optimize_gains(plant, spec, settings, max_evals=max_evals)[0])
    return out


__all__ = ["ZNResult", "ziegler_nichols", "simc", "heuristic_gains", "reference_gains",
           "optimize_gains", "design_baselines", "UNSTABLE_COST"]
