"""Gymnasium environment for RL-based PID tuning (REQ: Gymnasium-compatible interface).

One *episode* is a short tuning session on one plant (or one plant family member). One *step* is:

    1. the agent proposes a small change to the gains (action),
    2. the environment simulates a full closed-loop step response with the new gains,
    3. the response is scored by the shared cost (reward = -cost).

So an episode ends with a single fixed ``(Kp, Ki, Kd)`` set that can be deployed and compared with
Ziegler-Nichols - unlike changing gains every simulation timestep.

Gains are expressed *relative to reference gains* in log10 space:  ``K = K_ref * 10 ** z``.
This makes the problem scale-free: the same policy can work on a motor needing Kp ~ 100 and on an
oven needing Kp ~ 2.
"""
from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .baselines import reference_gains
from .metrics import StepMetrics
from .plant import Plant, random_second_order
from .spec import SimSettings, TuningSpec, cost_terms, get_spec, make_sim_settings, run_trial

OBS_DIM = 14
TERMS = ("overshoot", "settling", "rise", "sse", "iae", "effort", "disturbance")


def metrics_to_obs_part(m: StepMetrics, spec: TuningSpec) -> np.ndarray:
    """Normalised metrics -> 10 numbers: 7 cost terms, phase margin, gain margin, stable flag."""
    if not m.stable:
        return np.zeros(10, dtype=np.float64)
    t = cost_terms(m, spec)
    pm = m.phase_margin_deg
    gm = m.gain_margin
    pm_n = 1.0 if not np.isfinite(pm) else float(np.clip(pm, -90.0, 120.0) / 90.0)
    gm_n = 1.0 if not np.isfinite(gm) or gm <= 0 else float(np.clip(20.0 * np.log10(gm), -20.0, 40.0) / 40.0)
    return np.array([t[k] for k in TERMS] + [pm_n, gm_n, 1.0], dtype=np.float64)


class PIDTuningEnv(gym.Env):
    """Tune ``(Kp, Ki, Kd)`` for a linear SISO plant by repeated small refinements.

    Args:
        plant: the plant to tune (or give ``plant_sampler`` instead).
        plant_sampler: ``f(rng) -> Plant``; a new plant each episode, so one policy learns to tune a
            whole *family* of plants (enables reuse on unseen plants).
        spec: tuning specification (preset name, dict, or :class:`TuningSpec`).
        max_steps: refinement steps per episode.
        uncertainty: relative plant-parameter uncertainty (e.g. 0.2 = +/-20 %).
        n_robust: if > 1, each step is scored on the nominal plant plus ``n_robust - 1`` perturbed
            copies (average), so the gains found are *robust*. If 1 and ``uncertainty > 0``, a
            different perturbed plant is drawn each episode (domain randomisation).
        k_ref: ``"auto" | "simc" | "zn" | "heuristic"`` or an explicit ``(kp, ki, kd)``.
        max_log_step: largest change of log10(gain) per step (0.25 = up to x1.8 per step).
        z_limit: gains may move at most this many decades away from ``k_ref``.
        init_noise: random start offset (decades) for training diversity; 0 for evaluation.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        plant: Plant | None = None,
        plant_sampler: Callable[[np.random.Generator], Plant] | None = None,
        spec="balanced",
        max_steps: int = 10,
        uncertainty: float = 0.0,
        n_robust: int = 1,
        k_ref="auto",
        max_log_step: float = 0.25,
        z_limit: float = 2.0,
        init_noise: float = 0.3,
        dt: float | None = None,
        t_end: float | None = None,
        window_factor: float = 20.0,
    ):
        super().__init__()
        if (plant is None) == (plant_sampler is None):
            raise ValueError("Provide exactly one of `plant` or `plant_sampler`.")
        if n_robust < 1 or max_steps < 1:
            raise ValueError("n_robust and max_steps must be >= 1.")
        self.fixed_plant, self.plant_sampler = plant, plant_sampler
        self.tuning_spec = get_spec(spec)
        self.max_steps, self.uncertainty, self.n_robust = int(max_steps), float(uncertainty), int(n_robust)
        self.k_ref_mode = k_ref
        self.max_log_step, self.z_limit, self.init_noise = float(max_log_step), float(z_limit), float(init_noise)
        self._dt, self._t_end, self._window_factor = dt, t_end, window_factor

        # Float32 everywhere: SB3/PyTorch expect it and Gymnasium's checker is strict about it.
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(OBS_DIM,), dtype=np.float32)

        self._fixed_cache: tuple | None = None
        self.plants: list[Plant] = []
        self.settings: SimSettings | None = None
        self.k_ref = np.ones(3)
        self.k_ref_method = ""
        self.z = np.zeros(3)
        self.steps = 0
        self.best_cost, self.best_gains = np.inf, None
        self.last_metrics: StepMetrics | None = None

    # ------------------------------------------------------------------ helpers
    def _prepare_plants(self) -> None:
        """Choose this episode's plant(s), simulation settings and reference gains."""
        rng = self.np_random
        nominal = self.plant_sampler(rng) if self.plant_sampler is not None else self.fixed_plant
        if self.n_robust > 1:
            plants = [nominal] + [nominal.perturbed(self.uncertainty, rng) for _ in range(self.n_robust - 1)]
            ref_plant = nominal
        elif self.uncertainty > 0:
            plants = [nominal.perturbed(self.uncertainty, rng)]
            ref_plant = plants[0]
        else:
            plants, ref_plant = [nominal], nominal

        if self.fixed_plant is not None and self._fixed_cache is not None:
            settings, k_ref, method = self._fixed_cache
        else:
            settings = make_sim_settings(nominal, self.tuning_spec, self._dt, self._t_end, self._window_factor)
            if isinstance(self.k_ref_mode, str):
                k_ref, method = reference_gains(ref_plant, self.k_ref_mode)
            else:
                k_ref, method = tuple(float(v) for v in self.k_ref_mode), "user"
            if self.fixed_plant is not None and self.uncertainty == 0:
                self._fixed_cache = (settings, k_ref, method)
        self.plants, self.settings = plants, settings
        self.k_ref = np.maximum(np.asarray(k_ref, dtype=float), 1e-9)
        self.k_ref_method = method

    def gains_from_z(self, z: np.ndarray) -> np.ndarray:
        return self.k_ref * 10.0 ** np.asarray(z, dtype=float)

    def _evaluate(self, z: np.ndarray):
        """Simulate gains(z) on every plant; return (obs10, cost, all_stable, first_metrics)."""
        gains = self.gains_from_z(z)
        parts, costs, stable = [], [], True
        first = None
        for p in self.plants:
            _, m, c = run_trial(p, gains, self.tuning_spec, self.settings)
            first = first or m
            stable = stable and m.stable
            parts.append(metrics_to_obs_part(m, self.tuning_spec))
            costs.append(c)
        part = np.mean(parts, axis=0) if stable else np.zeros(10)
        return part, float(np.mean(costs)), stable, first

    def _obs(self, part: np.ndarray) -> np.ndarray:
        left = 1.0 - self.steps / self.max_steps
        obs = np.concatenate([part, self.z / self.z_limit, [left]])
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    def _info(self, cost: float, stable: bool) -> dict:
        gains = tuple(float(g) for g in self.gains_from_z(self.z))
        return {
            "gains": gains,
            "cost": cost,
            "stable": stable,
            "best_gains": None if self.best_gains is None else tuple(float(g) for g in self.best_gains),
            "best_cost": float(self.best_cost),
            "k_ref": tuple(float(g) for g in self.k_ref),
            "k_ref_method": self.k_ref_method,
            "metrics": None if self.last_metrics is None else self.last_metrics.to_dict(),
        }

    def _track_best(self, z: np.ndarray, cost: float) -> None:
        if cost < self.best_cost:
            self.best_cost, self.best_gains = cost, self.gains_from_z(z).copy()

    # ------------------------------------------------------------------ Gymnasium API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._prepare_plants()
        self.steps = 0
        self.best_cost, self.best_gains = np.inf, None
        z = np.clip(self.np_random.uniform(-1.0, 1.0, size=3) * self.init_noise, -self.z_limit, self.z_limit)
        if options and "z" in options:
            z = np.clip(np.asarray(options["z"], dtype=float), -self.z_limit, self.z_limit)
        part, cost, stable, metrics = self._evaluate(z)
        # Reference gains can be unstable on a perturbed plant: back off until the start is stable.
        for _ in range(12):
            if stable:
                break
            z = np.clip(z - 0.3, -self.z_limit, self.z_limit)
            part, cost, stable, metrics = self._evaluate(z)
        self.z, self.last_metrics = z, metrics
        if stable:
            self._track_best(z, cost)
        return self._obs(part), self._info(cost, stable)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(3), -1.0, 1.0)
        self.z = np.clip(self.z + a * self.max_log_step, -self.z_limit, self.z_limit)
        self.steps += 1
        part, cost, stable, metrics = self._evaluate(self.z)
        self.last_metrics = metrics
        if stable:
            self._track_best(self.z, cost)
            reward, terminated = -cost, False
        else:
            reward, terminated = -100.0, True            # unstable / diverged: end the episode
        truncated = (not terminated) and self.steps >= self.max_steps
        return self._obs(part), float(reward), bool(terminated), bool(truncated), self._info(cost, stable)


# ----------------------------------------------------------------------------------------
# Building environments from plain config dicts (used by train.py / evaluate.py / tuner.py)
# ----------------------------------------------------------------------------------------
def plant_from_config(cfg: dict | None) -> tuple[Plant | None, Callable | None]:
    """Return ``(plant, sampler)`` from a config like ``{"name": "dc_motor_position"}``,
    ``{"tf": {"num": [1], "den": [1, 1]}}`` or ``{"family": "second_order"}``."""
    cfg = dict(cfg or {"name": "dc_motor_position"})
    if "family" in cfg:
        if cfg["family"] != "second_order":
            raise ValueError("Only the 'second_order' plant family exists in v1.")
        return None, random_second_order
    if "tf" in cfg:
        return Plant.from_tf(cfg["tf"]["num"], cfg["tf"]["den"], name=cfg.get("label", "custom")), None
    name = cfg.get("name", "dc_motor_position")
    args = {k: v for k, v in cfg.items() if k != "name"}
    factories = {
        "dc_motor_position": Plant.dc_motor_position,
        "dc_motor": Plant.dc_motor_position,
        "first_order": Plant.first_order,
        "second_order": Plant.second_order,
    }
    if name not in factories:
        raise ValueError(f"Unknown plant '{name}'. Choose from {sorted(factories)} or give a 'tf'.")
    return factories[name](**args), None


def env_from_config(cfg: dict, seed: int | None = None, eval_mode: bool = False) -> PIDTuningEnv:
    """Build a :class:`PIDTuningEnv` from the ``plant``/``spec``/``env`` sections of a config.

    ``eval_mode`` removes the random start offset so evaluation episodes are repeatable.
    """
    plant, sampler = plant_from_config(cfg.get("plant"))
    env_cfg = dict(cfg.get("env", {}))
    if eval_mode:
        env_cfg["init_noise"] = 0.0
    env = PIDTuningEnv(plant=plant, plant_sampler=sampler, spec=cfg.get("spec", "balanced"),
                       uncertainty=cfg.get("uncertainty", 0.0), **env_cfg)
    if seed is not None:
        env.reset(seed=seed)
    return env
