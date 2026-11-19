"""Realistic discrete-time PID controller with filtering and anti-windup."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PIDGains:
    """Structured container for parallel PID controller gains.

    The parallel (ideal) PID form sums three independent terms:
    ``u(t) = Kp * e(t) + Ki * integral(e) + Kd * d/dt(...)``.
    """

    kp: float
    ki: float
    kd: float
    derivative_filter_n: float | None = None

    def as_tuple(self) -> tuple[float, float, float]:
        return (float(self.kp), float(self.ki), float(self.kd))


class PIDController:
    """Discrete-time parallel PID controller.

    Beginner-friendly control notes:
    * **P (Proportional):** reacts to the *present* error ``e = setpoint - y``.
      Larger ``Kp`` speeds up response, but too much causes overshoot/oscillation.
    * **I (Integral):** accumulates *past* error over time so steady-state error
      vanishes even against constant disturbances or offsets.
    * **D (Derivative):** reacts to the *rate of change* (future trend), adding
      damping to reduce overshoot—like a shock absorber.
    * **Derivative filter (``derivative_filter_n``):** raw finite differences
      amplify high-frequency sensor noise. A 1st-order low-pass filter
      ``N / (s + N)`` (typically ``N = 10..20`` rad/s or higher) rolls off
      high-frequency noise while keeping useful damping.
    * **Derivative on measurement:** during a sudden step change in setpoint,
      ``e`` jumps instantaneously and ``de/dt`` creates a huge "derivative kick"
      spike. Differentiating ``-y`` instead of ``e`` avoids that spike while
      giving identical damping once the setpoint is constant.
    * **Anti-windup (clamping):** real actuators have physical limits (e.g.,
      ``[-12, 12]`` V). If the output is saturated and the integrator keeps
      accumulating error, it "winds up" and causes huge overshoot when returning
      to the linear range. Clamping freezes integration when saturated and the
      error is still pushing in the saturated direction.
    """

    def __init__(
        self,
        k_p: float = 0.0,
        k_i: float = 0.0,
        k_d: float = 0.0,
        *,
        derivative_filter_n: float | None = None,
        derivative_on_measurement: bool = False,
        u_limits: tuple[float | None, float | None] | None = None,
        anti_windup: bool = True,
        dt: float = 0.01,
    ) -> None:
        if derivative_filter_n is not None and derivative_filter_n <= 0:
            raise ValueError("derivative_filter_n must be > 0 when provided.")
        if dt <= 0:
            raise ValueError("dt must be > 0.")
        if u_limits is not None:
            u_min, u_max = u_limits
            if u_min is not None and u_max is not None and u_min > u_max:
                raise ValueError("u_limits must satisfy u_min <= u_max.")

        self.k_p = float(k_p)
        self.k_i = float(k_i)
        self.k_d = float(k_d)
        self.derivative_filter_n = (
            None if derivative_filter_n is None else float(derivative_filter_n)
        )
        self.derivative_on_measurement = bool(derivative_on_measurement)
        self.u_limits = u_limits
        self.anti_windup = bool(anti_windup)
        self.dt = float(dt)

        # Internal controller state (reset between episodes/simulations).
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_measurement: float | None = None
        self.d_state = 0.0

    @property
    def kp(self) -> float:
        return self.k_p

    @property
    def ki(self) -> float:
        return self.k_i

    @property
    def kd(self) -> float:
        return self.k_d

    def set_gains(self, k_p: float, k_i: float, k_d: float) -> None:
        """Update (Kp, Ki, Kd) gains without clearing accumulated state."""
        self.k_p = float(k_p)
        self.k_i = float(k_i)
        self.k_d = float(k_d)

    def reset(self) -> None:
        """Clear integral, derivative memory, and filter state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_measurement = None
        self.d_state = 0.0

    def compute(
        self,
        error: float,
        dt: float | None = None,
        measurement: float | None = None,
    ) -> float:
        """Compute one control step from ``error = setpoint - measurement``."""
        step_dt = self.dt if dt is None else float(dt)
        if step_dt <= 0.0:
            raise ValueError("dt must be > 0.")
        err = float(error)

        # ---- Derivative term ------------------------------------------
        if self.derivative_on_measurement:
            if measurement is None:
                raise ValueError(
                    "measurement must be provided when derivative_on_measurement=True."
                )
            meas = float(measurement)
            if self.prev_measurement is None:
                # First sample after reset has no prior measurement -> zero derivative
                # avoids an artificial spike on startup.
                raw_derivative = 0.0
            else:
                # Because error = setpoint - y, for constant setpoint: de/dt = -dy/dt.
                raw_derivative = -(meas - self.prev_measurement) / step_dt
            self.prev_measurement = meas
        else:
            raw_derivative = (err - self.prev_error) / step_dt

        self.prev_error = err

        if self.derivative_filter_n is not None:
            # Backward-Euler discretization of first-order low-pass filter N / (s + N):
            # d_filt[k] = alpha * d_filt[k-1] + (1 - alpha) * raw_derivative[k]
            alpha = 1.0 / (1.0 + self.derivative_filter_n * step_dt)
            self.d_state = alpha * self.d_state + (1.0 - alpha) * raw_derivative
            d_term = self.k_d * self.d_state
        else:
            d_term = self.k_d * raw_derivative

        # ---- Proportional + Integral + Clamping Anti-Windup -----------
        p_term = self.k_p * err
        prev_integral = self.integral
        candidate_integral = prev_integral + err * step_dt
        u_unsat = p_term + self.k_i * candidate_integral + d_term

        if self.u_limits is not None:
            u_min, u_max = self.u_limits
            hit_high = (u_max is not None) and (u_unsat > u_max)
            hit_low = (u_min is not None) and (u_unsat < u_min)

            # Clamping anti-windup: freeze the integrator if saturated AND the error
            # tries to push the actuator even deeper into saturation.
            if (
                self.anti_windup
                and self.k_i != 0.0
                and ((hit_high and err > 0.0) or (hit_low and err < 0.0))
            ):
                self.integral = prev_integral
                u_unsat = p_term + self.k_i * self.integral + d_term
            else:
                self.integral = candidate_integral

            u_out = u_unsat
            if u_min is not None and u_out < u_min:
                u_out = float(u_min)
            if u_max is not None and u_out > u_max:
                u_out = float(u_max)
            return float(u_out)

        self.integral = candidate_integral
        return float(u_unsat)

    def compute_control_signal(
        self,
        error: float,
        dt: float | None = None,
        measurement: float | None = None,
    ) -> float:
        """Alias for :meth:`compute` matching the SysML ``computeControlSignal`` operation."""
        return self.compute(error, dt=dt, measurement=measurement)
