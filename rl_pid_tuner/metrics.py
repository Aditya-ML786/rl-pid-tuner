"""Step-response performance metrics and stability margins (feature F5)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .plant import Plant
from .simulation import Response, _controller_tf, _extract_gains

# Large finite stand-in used instead of inf/NaN so metrics stay JSON- and RL-friendly.
_BIG = 1e6


@dataclass
class StepMetrics:
    """Performance numbers for one closed-loop simulation.

    * overshoot_pct : peak above the setpoint, in % of the setpoint (0 if never above).
    * rise_time     : time to go from 10 % to 90 % of the setpoint.
    * settling_time : last time the output was outside the +/- band around the setpoint.
    * sse           : steady-state error (setpoint - final output) at the end of the step window.
    * iae / ise     : integral of |e| and e^2 over the step window (overall tracking quality).
    * u_peak, u_tv  : peak |u| and total variation of u (how hard / how jittery the actuator works).
    * dist_iae/peak : error integral / peak after a load disturbance (disturbance rejection).
    * gain_margin / phase_margin_deg : robustness (how much extra gain / lag before instability).
    """

    stable: bool
    overshoot_pct: float = 0.0
    rise_time: float = _BIG
    settling_time: float = _BIG
    settled: bool = False
    sse: float = _BIG
    iae: float = _BIG
    ise: float = _BIG
    u_peak: float = 0.0
    u_tv: float = 0.0
    dist_iae: float = 0.0
    dist_peak: float = 0.0
    gain_margin: float = float("inf")       # linear factor (>1 is stable)
    phase_margin_deg: float = float("nan")  # NaN if |L| never crosses 1
    t_window: float = 0.0                   # length of the step-response window (s)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("extra", None)
        # JSON cannot hold inf/NaN reliably; encode them as None.
        return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in d.items()}


def _trapz(y: np.ndarray, t: np.ndarray) -> float:
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(t))) if len(t) > 1 else 0.0


def compute_metrics(
    resp: Response,
    plant: Plant | None = None,
    gains=None,
    derivative_filter_n: float | None = None,
    band: float = 0.02,
) -> StepMetrics:
    """Compute :class:`StepMetrics` from a :class:`Response` (optionally with margins).

    If the response contains a load disturbance, the setpoint metrics (overshoot, rise,
    settling, sse, IAE) use only the part *before* it; disturbance metrics use the part after.
    """
    if not resp.stable or len(resp.t) < 3:
        return StepMetrics(stable=False, u_peak=float(np.max(np.abs(resp.u))) if len(resp.u) else 0.0)

    t, y, u, e, sp = resp.t, resp.y, resp.u, resp.e, resp.setpoint
    n_win = len(t)
    if resp.t_disturbance is not None:
        n_win = max(int(np.searchsorted(t, resp.t_disturbance, side="left")), 3)
    tw, yw, ew = t[:n_win], y[:n_win], e[:n_win]
    scale = abs(sp) if abs(sp) > 1e-12 else 1.0

    # Overshoot: how far the peak goes beyond the final target, relative to the target size.
    peak = yw.max() if sp >= 0 else yw.min()
    overshoot = max(0.0, (peak - sp) / scale * 100.0) if sp >= 0 else max(0.0, (sp - peak) / scale * 100.0)

    # Rise time (10 % -> 90 % of setpoint); only defined if the output actually gets to 90 %.
    frac = yw / sp if abs(sp) > 1e-12 else yw
    above10, above90 = np.flatnonzero(frac >= 0.1), np.flatnonzero(frac >= 0.9)
    rise = float(tw[above90[0]] - tw[above10[0]]) if above10.size and above90.size else _BIG

    # Settling time: the last instant the output is outside the +/- band, i.e. it stays inside after that.
    outside = np.flatnonzero(np.abs(ew) > band * scale)
    if outside.size == 0:
        settling, settled = 0.0, True
    elif outside[-1] >= n_win - 1:
        settling, settled = float(tw[-1]), False        # still outside at the end of the window
    else:
        settling, settled = float(tw[outside[-1] + 1]), True

    # Steady-state error: average over the last 5 % of the window to ignore tiny ripples.
    tail = max(n_win // 20, 1)
    sse = float(np.mean(ew[-tail:]))

    m = StepMetrics(
        stable=True,
        overshoot_pct=float(overshoot),
        rise_time=rise,
        settling_time=settling,
        settled=bool(settled),
        sse=sse,
        iae=_trapz(np.abs(ew), tw),
        ise=_trapz(ew**2, tw),
        u_peak=float(np.max(np.abs(u))),
        u_tv=float(np.sum(np.abs(np.diff(u)))),
        t_window=float(tw[-1]),
    )

    if resp.t_disturbance is not None and n_win < len(t) - 2:
        ed = np.abs(e[n_win:])
        m.dist_iae = _trapz(ed, t[n_win:])
        m.dist_peak = float(ed.max())

    if plant is not None and gains is not None:
        m.gain_margin, m.phase_margin_deg = stability_margins(plant, gains, derivative_filter_n)
    return m


def stability_margins(plant: Plant, gains, derivative_filter_n: float | None = None) -> tuple[float, float]:
    """Gain margin (linear) and phase margin (degrees) of the open loop L(s) = C(s) P(s).

    * Gain margin  : how much the loop gain can grow before hitting the -180 deg phase crossing.
    * Phase margin : 180 deg + phase of L at the frequency where |L| = 1.
    Typical healthy values: GM > 2 (6 dB) and PM > 45 deg.

    The phase is built factor-by-factor from poles/zeros so it is continuous (no 360 deg jumps),
    which matters for plants with integrators.
    """
    kp, ki, kd, gains_n = _extract_gains(gains)
    n_filt = derivative_filter_n if derivative_filter_n is not None else gains_n
    c_num, c_den = _controller_tf(kp, ki, kd, n_filt)
    num = np.polymul(plant.num, c_num)
    den = np.polymul(plant.den, c_den)
    num = num[np.flatnonzero(np.abs(num) > 0)[0]:] if np.any(num) else np.array([0.0])
    if not np.any(num) or not np.all(np.isfinite(num)) or not np.all(np.isfinite(den)):
        return float("inf"), float("nan")
    k_gain = num[0] / den[0]
    zeros, poles = np.roots(num), np.roots(den)

    # Frequency grid centred on the plant's own time scales (6 decades wide).
    w_ref = 1.0 / np.sqrt(plant.time_scale() * plant.fastest_time_constant())
    w = w_ref * np.logspace(-3, 3, 2500)
    jw = 1j * w
    mag = np.abs(k_gain) * np.ones_like(w)
    phase = np.full_like(w, 0.0 if k_gain > 0 else np.pi)
    for z in zeros:
        mag = mag * np.abs(jw - z)
        phase = phase + np.angle(jw - z)
    for p in poles:
        mag = mag / np.maximum(np.abs(jw - p), 1e-300)
        phase = phase - np.angle(jw - p)
    phase_deg = np.degrees(phase)

    # Gain margin: first frequency where the phase falls through -180 deg.
    gm = float("inf")
    cross = np.flatnonzero((phase_deg[:-1] + 180.0 > 0) & (phase_deg[1:] + 180.0 <= 0))
    if cross.size:
        i = cross[0]
        f = (phase_deg[i] + 180.0) / (phase_deg[i] - phase_deg[i + 1])
        mag_c = np.exp(np.log(mag[i]) + f * (np.log(mag[i + 1]) - np.log(mag[i])))
        gm = float(1.0 / mag_c) if mag_c > 0 else float("inf")

    # Phase margin: first frequency where |L| falls through 1 (gain crossover).
    pm = float("nan")
    gc = np.flatnonzero((mag[:-1] >= 1.0) & (mag[1:] < 1.0))
    if gc.size:
        i = gc[0]
        f = (np.log(mag[i])) / (np.log(mag[i]) - np.log(mag[i + 1]))
        pm = float(180.0 + phase_deg[i] + f * (phase_deg[i + 1] - phase_deg[i]))
    return gm, pm


def beats(candidate: StepMetrics, baseline: StepMetrics, sse_tol: float = 0.02) -> bool:
    """True if ``candidate`` is at least as good as ``baseline`` on overshoot, settling and sse.

    This is the 'RL matches or outperforms the baseline' test (REQ-P-04): an unstable
    candidate never wins, and an unstable baseline is beaten by any stable candidate.
    """
    if not candidate.stable:
        return False
    if not baseline.stable:
        return True
    return bool(
        candidate.overshoot_pct <= baseline.overshoot_pct + 1e-9
        and candidate.settling_time <= baseline.settling_time + 1e-9
        and abs(candidate.sse) <= max(abs(baseline.sse), sse_tol)
    )


def compare(results: dict[str, StepMetrics]) -> str:
    """Return a plain-text side-by-side table of several methods' metrics."""
    cols = [
        ("OS %", lambda m: m.overshoot_pct, "{:8.2f}"),
        ("rise s", lambda m: m.rise_time, "{:8.3f}"),
        ("settle s", lambda m: m.settling_time, "{:8.3f}"),
        ("sse", lambda m: m.sse, "{:9.4f}"),
        ("IAE", lambda m: m.iae, "{:9.3f}"),
        ("GM", lambda m: m.gain_margin, "{:7.2f}"),
        ("PM deg", lambda m: m.phase_margin_deg, "{:8.1f}"),
    ]
    head = f"{'method':<12}{'stable':>7}" + "".join(f"{c[0]:>10}" for c in cols)
    lines = [head, "-" * len(head)]
    for name, m in results.items():
        if m.stable:
            vals = "".join(f"{c[2].format(c[1](m)):>10}" for c in cols)
        else:
            vals = "".join(f"{'--':>10}" for _ in cols)
        lines.append(f"{name:<12}{str(m.stable):>7}{vals}")
    return "\n".join(lines)
