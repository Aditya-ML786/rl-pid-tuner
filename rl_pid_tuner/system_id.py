"""System identification: fit a linear plant model to measured step-response data (F2).

You apply a step of size ``u_step`` to the real system, record the output, and call
:func:`identify`. The result is a :class:`~rl_pid_tuner.plant.Plant` you can tune like any other.

Supported model families (all linear, no dead time in v1):

* ``first_order``            G = K / (tau s + 1)            (heater, RC circuit, ...)
* ``second_order``           G = K wn^2 / (s^2 + 2 zeta wn s + wn^2)   (over- or under-damped)
* ``integrator_first_order`` G = K / (s (tau s + 1))        (motor position, tank level, ...)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize

from .plant import Plant

MODELS = ("first_order", "integrator_first_order", "second_order")   # simplest first


# --------------------------------------------------------------------------------------
# Analytic unit-step responses of each model family
# --------------------------------------------------------------------------------------
def _step_first_order(t, K, tau):
    return K * (1.0 - np.exp(-t / tau))


def _step_integrator_first_order(t, K, tau):
    # inverse Laplace of K / (s^2 (tau s + 1))
    return K * (t - tau * (1.0 - np.exp(-t / tau)))


def _step_second_order(t, K, wn, zeta):
    if abs(zeta - 1.0) < 1e-4:                       # critically damped
        return K * (1.0 - np.exp(-wn * t) * (1.0 + wn * t))
    if zeta < 1.0:                                   # underdamped: decaying oscillation
        wd = wn * np.sqrt(1.0 - zeta**2)
        return K * (1.0 - np.exp(-zeta * wn * t) * (np.cos(wd * t) + zeta / np.sqrt(1.0 - zeta**2) * np.sin(wd * t)))
    r = wn * np.sqrt(zeta**2 - 1.0)                  # overdamped: two real poles s1, s2
    s1, s2 = -zeta * wn + r, -zeta * wn - r
    return K * (1.0 + (s2 * np.exp(s1 * t) - s1 * np.exp(s2 * t)) / (s1 - s2))


@dataclass
class FitReport:
    """How good the fit is. ``r2`` close to 1 means the model explains the data well."""

    model: str
    params: dict
    r2: float
    rmse: float
    candidates: dict = field(default_factory=dict)   # model name -> R^2 (when model='auto')
    warning: str | None = None


def _fit(model: str, t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares fit from several starting points; returns (params, sse).

    K is fitted directly; time constants / frequencies are fitted in log space so they stay > 0.
    The *last* parameter is a free baseline offset: subtracting a single noisy first sample would
    otherwise bias the fit, so the offset is estimated jointly from all the data.
    """
    n_tail = max(len(y) // 10, 2)
    k_final = float(np.mean(y[-n_tail:]))
    span = float(t[-1])
    sign = 1.0 if k_final >= 0 else -1.0

    if model == "first_order":
        target = 0.632 * k_final
        idx = np.flatnonzero(y * sign >= target * sign)
        tau0 = float(t[idx[0]]) if idx.size and t[idx[0]] > 0 else span / 3.0
        starts = [[k_final, np.log(tau0 * f), 0.0] for f in (1.0, 0.3, 3.0)]
        fun = lambda p: _step_first_order(t, p[0], np.exp(p[1])) + p[2] - y
    elif model == "integrator_first_order":
        slope, intercept = np.polyfit(t[-n_tail:], y[-n_tail:], 1)
        tau0 = -intercept / slope if slope != 0 and -intercept / slope > 0 else span / 10.0
        starts = [[slope, np.log(tau0 * f), 0.0] for f in (1.0, 0.3, 3.0)]
        fun = lambda p: _step_integrator_first_order(t, p[0], np.exp(p[1])) + p[2] - y
    elif model == "second_order":
        idx = np.flatnonzero(y * sign >= 0.632 * k_final * sign)
        t63 = float(t[idx[0]]) if idx.size and t[idx[0]] > 0 else span / 3.0
        starts = [[k_final, np.log(f / t63), np.log(z), 0.0] for f in (1.5, 4.0) for z in (0.2, 0.6, 1.0, 2.5)]
        fun = lambda p: _step_second_order(t, p[0], np.exp(p[1]), np.exp(p[2])) + p[3] - y
    else:
        raise ValueError(f"Unknown model '{model}'. Choose from {MODELS + ('auto',)}.")

    best_p, best_sse = None, np.inf
    for x0 in starts:
        try:
            sol = optimize.least_squares(fun, x0, method="trf", max_nfev=200)
        except (ValueError, FloatingPointError):
            continue
        sse = float(np.sum(sol.fun**2))
        if np.isfinite(sse) and sse < best_sse:
            best_p, best_sse = sol.x, sse
    if best_p is None:
        raise RuntimeError(f"Could not fit model '{model}' to the data.")
    return best_p, best_sse


def _to_plant(model: str, p: np.ndarray, name: str) -> tuple[Plant, dict]:
    if model == "first_order":
        K, tau = float(p[0]), float(np.exp(p[1]))
        return Plant([K], [tau, 1.0], name=name), {"K": K, "tau": tau}
    if model == "integrator_first_order":
        K, tau = float(p[0]), float(np.exp(p[1]))
        return Plant([K], [tau, 1.0, 0.0], name=name), {"K": K, "tau": tau}
    K, wn, zeta = float(p[0]), float(np.exp(p[1])), float(np.exp(p[2]))
    return Plant([K * wn**2], [1.0, 2 * zeta * wn, wn**2], name=name), {"K": K, "wn": wn, "zeta": zeta}


def identify(t, y, u_step: float = 1.0, model: str = "auto", r2_tolerance: float = 0.002) -> tuple[Plant, FitReport]:
    """Fit a linear plant to step-response data.

    Args:
        t, y: time stamps (s) and measured output. The output is assumed to start at rest, so the
            initial value ``y[0]`` is subtracted as the baseline.
        u_step: size of the input step that produced ``y`` (so gains come out per unit input).
        model: one of ``first_order | integrator_first_order | second_order | auto``. ``auto``
            fits all and picks the *simplest* model whose R^2 is within ``r2_tolerance`` of the best.
    Returns:
        ``(plant, report)``.
    """
    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if t.shape != y.shape or len(t) < 10:
        raise ValueError("t and y must have the same length (>= 10 samples).")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(y))):
        raise ValueError("t and y must be finite.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing.")
    if u_step == 0 or not np.isfinite(u_step):
        raise ValueError("u_step must be a non-zero finite number.")

    t = t - t[0]
    y = (y - y[0]) / float(u_step)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot < 1e-18:
        raise ValueError("Output does not change; cannot identify a plant from flat data.")

    names = list(MODELS) if model == "auto" else [model]
    fits = {}
    for m in names:
        p, sse = _fit(m, t, y)
        fits[m] = (p, sse, 1.0 - sse / ss_tot)
    r2s = {m: v[2] for m, v in fits.items()}
    if model == "auto":
        # A first-order fit whose time constant is longer than half the record is not trustworthy:
        # over a short window it can mimic a ramp (an integrator) with an absurdly large tau.
        eligible = [m for m in MODELS if not (m == "first_order" and np.exp(fits[m][0][1]) > t[-1] / 2.0)]
        best = max(r2s[m] for m in eligible)
        chosen = next(m for m in eligible if r2s[m] >= best - r2_tolerance)
    else:
        chosen = model
    p, sse, r2 = fits[chosen]
    plant, params = _to_plant(chosen, p, name=f"identified_{chosen}")
    warn = "Poor fit (R^2 < 0.9): the plant may need dead time or a higher-order model." if r2 < 0.9 else None
    return plant, FitReport(chosen, params, float(r2), float(np.sqrt(sse / len(t))), r2s, warn)


class SystemIdentification:
    """Thin object wrapper around :func:`identify` (mirrors the SysML ``identifyPlant`` operation)."""

    def __init__(self, model: str = "auto"):
        self.model = model
        self.plant_transfer_function: str = ""    # human-readable TF of the last fit
        self.report: FitReport | None = None

    def identify_plant(self, t, y, u_step: float = 1.0) -> Plant:
        plant, self.report = identify(t, y, u_step, self.model)
        self.plant_transfer_function = plant.tf_string
        return plant
