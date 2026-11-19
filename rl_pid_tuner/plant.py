"""Linear SISO plant models (transfer function / state space)."""
from __future__ import annotations

import numpy as np
from scipy import signal


def _trim(p: np.ndarray) -> np.ndarray:
    """Strip leading zeros from a polynomial coefficient array.

    NumPy/SciPy represent polynomials in descending powers of ``s``,
    for example ``[a_n, ..., a_1, a_0]`` for ``a_n s^n + ... + a_1 s + a_0``.
    """
    nz = np.flatnonzero(p)
    if nz.size == 0:
        return np.array([0.0])
    return p[nz[0]:]


class Plant:
    """A continuous-time, linear, Single-Input Single-Output (SISO) plant.

    Build it with :meth:`from_tf` or :meth:`from_ss`. Internally the plant is
    always stored in both forms:

    * **Transfer function** ``G(s) = N(s) / D(s)`` using descending powers of
      ``s`` (so ``den[-1]`` is always the constant ``s^0`` term).
    * **State-space model** ``dx/dt = A x + B u``, ``y = C x + D u``, which
      lets us advance the plant one discrete timestep at a time.
    """

    def __init__(self, num, den, name: str = "plant"):
        num = np.atleast_1d(np.asarray(num, dtype=float))
        den = np.atleast_1d(np.asarray(den, dtype=float))
        if num.ndim != 1 or den.ndim != 1:
            raise ValueError("num and den must be 1-D (SISO plants only).")
        if not (np.all(np.isfinite(num)) and np.all(np.isfinite(den))):
            raise ValueError("num and den must contain only finite numbers.")
        num, den = _trim(num), _trim(den)
        if not np.any(den):
            raise ValueError("Denominator must not be zero.")
        if not np.any(num):
            raise ValueError("Numerator must not be zero.")
        # Physical causality requires deg(N) <= deg(D) ("proper" transfer function);
        # otherwise the plant would act as an ideal differentiator with infinite high-freq gain.
        if len(num) > len(den):
            raise ValueError("Improper transfer function (numerator order > denominator order).")
        if len(den) < 2:
            raise ValueError("Plant must have dynamics (denominator order >= 1).")

        self.num, self.den, self.name = num, den, name
        # Convert transfer function N(s)/D(s) into continuous-time state-space (A, B, C, D).
        self.A, self.B, self.C, self.D = signal.tf2ss(num, den)

        # Cache Zero-Order Hold (ZOH) discretizations keyed by timestep dt so repeated
        # RL rollouts do not pay matrix-exponential cost on every episode.
        self._zoh_cache: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        # Internal state vector x[k] used by stateful single-step simulation.
        self._state = np.zeros((self.A.shape[0], 1), dtype=float)

    # ---- constructors -------------------------------------------------
    @classmethod
    def from_tf(cls, num, den, name: str = "plant") -> "Plant":
        return cls(num, den, name)

    @classmethod
    def from_ss(cls, A, B, C, D=0.0, name: str = "plant") -> "Plant":
        A = np.atleast_2d(np.asarray(A, dtype=float))
        B = np.asarray(B, dtype=float)
        C = np.asarray(C, dtype=float)
        D = np.asarray(D, dtype=float)
        n = A.shape[0]
        if A.ndim != 2 or A.shape[1] != n:
            raise ValueError("A must be square.")
        B, C = B.reshape(-1, B.shape[-1] if B.ndim > 1 else 1), np.atleast_2d(C)
        if B.shape != (n, 1) or C.shape != (1, n) or D.size != 1:
            raise ValueError("Only SISO state-space models are supported (B: n x 1, C: 1 x n, D scalar).")
        num, den = signal.ss2tf(A, B, C, D.reshape(1, 1))
        return cls(np.squeeze(num), den, name)

    # ---- built-in examples --------------------------------------------
    @classmethod
    def first_order(cls, K: float = 1.0, tau: float = 1.0) -> "Plant":
        """First-order lag G(s) = K / (tau*s + 1) (e.g., thermal heater or simple RC filter)."""
        return cls([K], [tau, 1.0], name="first_order")

    @classmethod
    def second_order(cls, wn: float = 1.0, zeta: float = 0.7, K: float = 1.0) -> "Plant":
        """Canonical 2nd-order system G(s) = K*wn^2 / (s^2 + 2*zeta*wn*s + wn^2)."""
        return cls([K * wn**2], [1.0, 2 * zeta * wn, wn**2], name="second_order")

    @classmethod
    def dc_motor_position(cls, J=0.01, b=0.1, K=0.01, R=1.0, L=0.5) -> "Plant":
        """theta(s)/V(s) = K / (s((Js+b)(Ls+R)+K^2)) (Michigan CTMS example values)."""
        den = np.polymul([J, b], [L, R])
        den = np.polyadd(den, [K**2])
        den = np.polymul(den, [1.0, 0.0])  # Extra 1/s integrates angular velocity omega -> position theta.
        return cls([K], den, name="dc_motor_position")

    # ---- properties ---------------------------------------------------
    @property
    def order(self) -> int:
        return len(self.den) - 1

    @property
    def poles(self) -> np.ndarray:
        """Roots of D(s). Real part < 0 means decaying modes; root at 0 is an integrator."""
        return np.roots(self.den)

    @property
    def dc_gain(self) -> float:
        """Steady-state gain G(0) = N(0)/D(0). Returns ``inf`` if the plant has an integrator (D(0)=0)."""
        # Because coefficients are in descending powers of s, index -1 is the s^0 constant term.
        if abs(self.den[-1]) < 1e-12:
            return float("inf")
        return float(self.num[-1] / self.den[-1])

    @property
    def tf_string(self) -> str:
        """Human-readable transfer function, e.g. ``(0.01) / (0.005 s^3 + 0.06 s^2 + 0.1001 s)``."""
        def poly(c):
            n = len(c) - 1
            terms = []
            for i, a in enumerate(c):
                p = n - i
                if a == 0:
                    continue
                terms.append(f"{a:.6g}" + ("" if p == 0 else (" s" if p == 1 else f" s^{p}")))
            return " + ".join(terms) if terms else "0"
        return f"({poly(self.num)}) / ({poly(self.den)})"

    def time_scale(self) -> float:
        """Slowest non-integrator time constant tau = 1/|Re(p)| in seconds (1.0 if none)."""
        re = np.abs(self.poles.real)
        re = re[re > 1e-9]  # Ignore pure integrators (pole at s = 0) when picking simulation length.
        return float(1.0 / re.min()) if re.size else 1.0

    def fastest_time_constant(self) -> float:
        """Fastest non-integrator time constant tau = 1/|Re(p)| in seconds (used to pick dt)."""
        re = np.abs(self.poles.real)
        re = re[re > 1e-9]
        return float(1.0 / re.max()) if re.size else 1.0

    # ---- discretization & single-step simulation ----------------------
    def discretize(self, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert continuous (A, B, C, D) to exact Zero-Order Hold (ZOH) discrete matrices.

        ZOH assumes the control input u(t) is held constant over each sample interval
        [k*dt, (k+1)*dt), which matches digital/computer control and stays numerically exact
        for linear plants where forward Euler can diverge.
        """
        dt_f = float(dt)
        if not np.isfinite(dt_f) or dt_f <= 0.0:
            raise ValueError("dt must be a positive finite float.")
        cached = self._zoh_cache.get(dt_f)
        if cached is not None:
            return cached
        Ad, Bd, Cd, Dd, _ = signal.cont2discrete(
            (self.A, self.B, self.C, self.D), dt_f, method="zoh"
        )
        mats = (
            np.asarray(Ad, dtype=float),
            np.asarray(Bd, dtype=float),
            np.asarray(Cd, dtype=float),
            np.asarray(Dd, dtype=float),
        )
        self._zoh_cache[dt_f] = mats
        return mats

    def reset(self) -> None:
        """Reset internal state vector x to zero before a new simulation episode."""
        self._state = np.zeros((self.A.shape[0], 1), dtype=float)

    def simulate_response(self, u: float, dt: float = 0.01) -> float:
        """Advance the plant by one ZOH sample with input ``u`` and return output ``y[k]``."""
        Ad, Bd, Cd, Dd = self.discretize(dt)
        u_val = float(u)
        y = float((Cd @ self._state)[0, 0] + Dd[0, 0] * u_val)
        self._state = Ad @ self._state + Bd * u_val
        return y

    # ---- uncertainty --------------------------------------------------
    def perturbed(self, rel: float = 0.2, rng: np.random.Generator | None = None) -> "Plant":
        """Scale each non-zero coefficient by a factor in [1-rel, 1+rel].

        Zero coefficients stay zero, so integrators are preserved.
        """
        if rel < 0:
            raise ValueError("rel must be >= 0.")
        rng = rng or np.random.default_rng()

        def jitter(p):
            f = 1.0 + rng.uniform(-rel, rel, size=p.shape)
            return np.where(p == 0, 0.0, p * f)

        return Plant(jitter(self.num), jitter(self.den), name=self.name + "_perturbed")

    def __repr__(self) -> str:
        return f"Plant({self.name}, num={self.num.tolist()}, den={self.den.tolist()})"


def random_second_order(rng: np.random.Generator) -> Plant:
    """Sample a plant from a family of 2nd-order systems (used to train one reusable policy).

    Natural frequency and gain are log-uniform in [0.5, 5]; damping ratio is uniform in [0.3, 1.5]
    (so both oscillatory and sluggish plants appear).
    """
    wn = 10.0 ** rng.uniform(-0.3, 0.7)
    zeta = rng.uniform(0.3, 1.5)
    gain = 10.0 ** rng.uniform(-0.3, 0.7)
    return Plant.second_order(wn=wn, zeta=zeta, K=gain)
