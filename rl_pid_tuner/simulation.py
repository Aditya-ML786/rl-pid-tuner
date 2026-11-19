"""Closed-loop SISO linear simulation and analytic pole-stability checks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pid_controller import PIDController, PIDGains
from .plant import Plant


@dataclass
class Response:
    """Time-domain traces and status produced by :func:`simulate`."""

    t: np.ndarray
    y: np.ndarray
    u: np.ndarray
    e: np.ndarray
    setpoint: float
    stable: bool
    reason: str = "ok"
    # Load-disturbance info (None when no disturbance was injected); used by metrics.
    t_disturbance: float | None = None
    d_amplitude: float = 0.0

    @property
    def error(self) -> np.ndarray:
        return self.e


def _extract_gains(gains) -> tuple[float, float, float, float | None]:
    """Normalize tuple, dict, PIDGains, or PIDController into (kp, ki, kd, N)."""
    if isinstance(gains, PIDController):
        return (gains.k_p, gains.k_i, gains.k_d, gains.derivative_filter_n)
    if isinstance(gains, PIDGains):
        return (gains.kp, gains.ki, gains.kd, gains.derivative_filter_n)
    if isinstance(gains, dict):
        kp = gains.get("kp", gains.get("k_p"))
        ki = gains.get("ki", gains.get("k_i"))
        kd = gains.get("kd", gains.get("k_d"))
        if kp is None or ki is None or kd is None:
            raise ValueError("Gain dict must contain kp/ki/kd (or k_p/k_i/k_d).")
        return (float(kp), float(ki), float(kd), gains.get("derivative_filter_n"))
    arr = np.asarray(gains, dtype=float).ravel()
    if arr.size != 3:
        raise ValueError("gains must have 3 elements: (kp, ki, kd).")
    return (float(arr[0]), float(arr[1]), float(arr[2]), None)


def _trim_poly(p: np.ndarray) -> np.ndarray:
    nz = np.flatnonzero(np.abs(p) > 0.0)
    return p[nz[0]:] if nz.size else np.array([0.0])


def _controller_tf(
    kp: float, ki: float, kd: float, derivative_filter_n: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Build C(s) = N_c(s) / D_c(s) without introducing artificial poles at s = 0."""
    if derivative_filter_n is None:
        if ki == 0.0:
            c_num = np.array([kd, kp], dtype=float)
            c_den = np.array([1.0], dtype=float)
        else:
            c_num = np.array([kd, kp, ki], dtype=float)
            c_den = np.array([1.0, 0.0], dtype=float)
    else:
        n = float(derivative_filter_n)
        if n <= 0.0:
            raise ValueError("derivative_filter_n must be > 0.")
        if ki == 0.0:
            if kd == 0.0:
                c_num = np.array([kp], dtype=float)
                c_den = np.array([1.0], dtype=float)
            else:
                # C(s) = Kp + Kd * N*s / (s + N)
                c_num = np.array([kp + kd * n, kp * n], dtype=float)
                c_den = np.array([1.0, n], dtype=float)
        else:
            # C(s) = Kp + Ki/s + Kd * N*s / (s + N)
            c_num = np.array([kp + kd * n, kp * n + ki, ki * n], dtype=float)
            c_den = np.array([1.0, n, 0.0], dtype=float)
    return _trim_poly(c_num), _trim_poly(c_den)


def is_closed_loop_stable(
    plant: Plant,
    gains,
    derivative_filter_n: float | None = None,
    tol: float = 1e-9,
) -> bool:
    """Analytically check whether the continuous-time closed-loop poles are strictly in the LHP.

    A linear feedback loop ``L(s) = C(s) * P(s)`` has closed-loop transfer function
    ``T(s) = L(s) / (1 + L(s))``. Its poles are the roots of the characteristic
    polynomial ``D_p(s)*D_c(s) + N_p(s)*N_c(s) = 0``. If *every* root has
    ``Re(p) < -tol``, any disturbance decays exponentially (stable). If any root
    has ``Re(p) >= 0``, the response will oscillate forever or blow up.
    """
    kp, ki, kd, gains_n = _extract_gains(gains)
    if not (np.isfinite(kp) and np.isfinite(ki) and np.isfinite(kd)):
        return False
    eff_n = derivative_filter_n if derivative_filter_n is not None else gains_n
    c_num, c_den = _controller_tf(kp, ki, kd, eff_n)

    # Characteristic polynomial: D_p(s)*D_c(s) + N_p(s)*N_c(s)
    char_poly = np.polyadd(np.polymul(plant.den, c_den), np.polymul(plant.num, c_num))
    char_poly = _trim_poly(char_poly)
    if char_poly.size < 2 or not np.all(np.isfinite(char_poly)):
        return False
    poles = np.roots(char_poly)
    if poles.size == 0 or not np.all(np.isfinite(poles)):
        return False
    return bool(np.all(poles.real < -tol))


def choose_sim_time_and_dt(
    plant: Plant, dt: float | None = None, t_end: float | None = None
) -> tuple[float, float]:
    """Choose sensible simulation duration and timestep from the plant's poles."""
    tau_slow = plant.time_scale()
    tau_fast = plant.fastest_time_constant()
    sim_t_end = float(np.clip(10.0 * tau_slow, 2.0, 200.0)) if t_end is None else float(t_end)
    if sim_t_end <= 0.0:
        raise ValueError("t_end must be > 0.")
    if dt is None:
        # Sample ~10x faster than the fastest pole: accurate enough, and keeps RL training fast.
        sim_dt = float(max(min(tau_fast / 10.0, sim_t_end / 400.0), 1e-4))
        sim_dt = min(sim_dt, sim_t_end / 10.0)
    else:
        sim_dt = float(dt)
    if sim_dt <= 0.0:
        raise ValueError("dt must be > 0.")
    return sim_t_end, sim_dt


def simulate(
    plant: Plant,
    gains,
    setpoint: float = 1.0,
    disturbance: float | tuple[float, float] | None = None,
    noise_std: float = 0.0,
    u_limits: tuple[float | None, float | None] | None = None,
    dt: float | None = None,
    t_end: float | None = None,
    *,
    derivative_filter_n: float | None = None,
    derivative_on_measurement: bool = False,
    anti_windup: bool = True,
    y_max: float = 1e4,
    check_stability_first: bool = False,
    rng: np.random.Generator | None = None,
) -> Response:
    """Run a closed-loop discrete-time (ZOH) step simulation of ``plant`` with ``gains``.

    Note: when ``noise_std > 0``, ``e`` records the *true* noiseless tracking error
    ``setpoint - y_clean`` so metrics judge real plant behavior while the controller
    sees the noisy measurement ``y_clean + noise``.
    """
    kp, ki, kd, gains_n = _extract_gains(gains)
    eff_n = derivative_filter_n if derivative_filter_n is not None else gains_n
    sp = float(setpoint)

    if not (np.isfinite(kp) and np.isfinite(ki) and np.isfinite(kd) and np.isfinite(sp)):
        return Response(
            t=np.array([0.0]),
            y=np.array([0.0]),
            u=np.array([0.0]),
            e=np.array([sp if np.isfinite(sp) else 0.0]),
            setpoint=sp,
            stable=False,
            reason="non_finite_inputs",
        )

    if check_stability_first and not is_closed_loop_stable(
        plant, (kp, ki, kd), derivative_filter_n=eff_n
    ):
        return Response(
            t=np.array([0.0]),
            y=np.array([0.0]),
            u=np.array([0.0]),
            e=np.array([sp]),
            setpoint=sp,
            stable=False,
            reason="unstable_poles",
        )

    sim_t_end, sim_dt = choose_sim_time_and_dt(plant, dt=dt, t_end=t_end)
    n_steps = max(int(round(sim_t_end / sim_dt)) + 1, 2)
    t = np.arange(n_steps, dtype=float) * sim_dt

    # Parse optional load disturbance injected at the plant input (u_plant = u_ctrl + d).
    if disturbance is None:
        t_dist, d_amp = float("inf"), 0.0
    elif isinstance(disturbance, (tuple, list)):
        if len(disturbance) != 2:
            raise ValueError("disturbance tuple must be (t_start, amplitude).")
        t_dist, d_amp = float(disturbance[0]), float(disturbance[1])
    else:
        t_dist, d_amp = 0.5 * sim_t_end, float(disturbance)

    # Pre-compute everything the hot loop needs as plain Python floats/lists: for tiny state
    # vectors (n = 1..4) this is ~3x faster than calling NumPy on every one of thousands of steps.
    Ad, Bd, Cd, Dd = plant.discretize(sim_dt)
    A_rows = Ad.tolist()
    B_col = Bd[:, 0].tolist()
    C_row = Cd[0].tolist()
    dd_scalar = float(Dd[0, 0])
    x = [0.0] * len(B_col)  # Local state, so the Plant object is never mutated.

    pid = PIDController(
        kp,
        ki,
        kd,
        derivative_filter_n=eff_n,
        derivative_on_measurement=derivative_on_measurement,
        u_limits=u_limits,
        anti_windup=anti_windup,
        dt=sim_dt,
    )

    # Index of the first sample at which the load disturbance is active.
    k_dist = n_steps if not np.isfinite(t_dist) else max(int(np.ceil(t_dist / sim_dt - 1e-9)), 0)

    has_noise = noise_std > 0.0
    if has_noise:
        rng = rng or np.random.default_rng()
        noise = (rng.normal(0.0, noise_std, size=n_steps)).tolist()

    y_list: list[float] = []
    u_list: list[float] = []
    e_list: list[float] = []
    compute = pid.compute
    t_dist_out = float(t_dist) if np.isfinite(t_dist) else None

    def _result(stable: bool, reason: str) -> Response:
        end = len(y_list)
        return Response(
            t=t[:end],
            y=np.asarray(y_list),
            u=np.asarray(u_list),
            e=np.asarray(e_list),
            setpoint=sp,
            stable=stable,
            reason=reason,
            t_disturbance=t_dist_out,
            d_amplitude=d_amp if t_dist_out is not None else 0.0,
        )

    for k in range(n_steps):
        # Sensor reads the plant output y = C x (+ D u below for non-strictly-proper plants).
        y_sensor = sum(c * xi for c, xi in zip(C_row, x))
        y_meas = y_sensor + noise[k] if has_noise else y_sensor

        # Controller sees the (possibly noisy) measurement; the disturbance enters at the plant input.
        u_ctrl = compute(sp - y_meas, dt=sim_dt, measurement=y_meas)
        u_plant = u_ctrl + (d_amp if k >= k_dist else 0.0)

        y_actual = y_sensor + dd_scalar * u_plant
        y_list.append(y_actual)
        u_list.append(u_ctrl)
        e_list.append(sp - y_actual)

        # Divergence guard: NaN fails every comparison, so `not (... <= ...)` also catches NaN.
        if not (abs(y_actual) <= y_max) or not (abs(u_ctrl) < 1e12):
            return _result(False, "diverged")

        # State update x[k+1] = Ad x[k] + Bd u (exact ZOH step).
        x = [sum(a * xi for a, xi in zip(row, x)) + b * u_plant for row, b in zip(A_rows, B_col)]

    return _result(True, "ok")
