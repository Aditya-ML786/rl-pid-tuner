"""Thin wrapper around Stable-Baselines3 (SAC / PPO) for the PID-tuning environment (F7, F10).

Needs the optional training extras:  ``pip install -e .[train]``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SUPPORTED = ("SAC", "PPO")

# Sensible defaults for this problem (small continuous-control task, short episodes).
DEFAULT_KWARGS = {
    "SAC": dict(learning_rate=3e-4, batch_size=256, buffer_size=100_000, learning_starts=500,
                gamma=0.9, ent_coef="auto", train_freq=1, gradient_steps=1,
                policy_kwargs=dict(net_arch=[128, 128])),
    "PPO": dict(learning_rate=3e-4, n_steps=512, batch_size=128, n_epochs=10, gamma=0.9,
                ent_coef=0.0, policy_kwargs=dict(net_arch=[128, 128])),
}


def _sb3():
    try:
        import stable_baselines3 as sb3
    except ImportError as exc:                       # pragma: no cover
        raise ImportError("Training needs Stable-Baselines3: pip install -e .[train]") from exc
    return sb3


class RLAgent:
    """An RL policy that proposes gain refinements ``[dKp, dKi, dKd]`` from an observation.

    Mirrors the SysML block: ``select_action(observation)`` picks an action; ``train`` runs the
    learning loop (SB3's ``learn`` is where ``updatePolicy(reward)`` happens, thousands of times).
    """

    def __init__(self, env=None, algorithm: str = "SAC", seed: int = 0, device: str = "auto",
                 tensorboard_log: str | None = None, model=None, **algo_kwargs):
        algorithm = algorithm.upper()
        if algorithm not in SUPPORTED:
            raise ValueError(f"algorithm must be one of {SUPPORTED}")
        self.algorithm = algorithm
        if model is not None:                        # used by load()
            self.model = model
            return
        sb3 = _sb3()
        kwargs = {**DEFAULT_KWARGS[algorithm], **algo_kwargs}
        self.model = getattr(sb3, algorithm)("MlpPolicy", env, seed=seed, device=device, verbose=0,
                                             tensorboard_log=tensorboard_log, **kwargs)

    # ---- learning --------------------------------------------------------------
    def train(self, total_timesteps: int, callback=None, reset_num_timesteps: bool = True,
              progress_bar: bool = False) -> "RLAgent":
        self.model.learn(total_timesteps=int(total_timesteps), callback=callback,
                         reset_num_timesteps=reset_num_timesteps, progress_bar=progress_bar,
                         tb_log_name=self.algorithm)
        return self

    @property
    def episode_count(self) -> int:
        """Number of training episodes finished so far (SAC tracks it; PPO estimated)."""
        n = getattr(self.model, "_episode_num", None)
        return int(n) if n is not None else 0

    # ---- inference -------------------------------------------------------------
    def select_action(self, observation, deterministic: bool = True) -> np.ndarray:
        action, _ = self.model.predict(np.asarray(observation, dtype=np.float32), deterministic=deterministic)
        return np.asarray(action, dtype=np.float32)

    def tune(self, env, n_episodes: int = 1, deterministic: bool = True) -> tuple[tuple, float, dict]:
        """Run the policy on ``env`` and return ``(best_gains, best_cost, last_info)``.

        With a deterministic policy and ``init_noise=0`` one episode is enough; extra episodes
        (with a random start) can only find better gains.
        """
        best_gains, best_cost, info = None, np.inf, {}
        for _ in range(n_episodes):
            obs, info = env.reset()
            done = False
            while not done:
                obs, _, term, trunc, info = env.step(self.select_action(obs, deterministic))
                done = term or trunc
            if info.get("best_gains") is not None and info["best_cost"] < best_cost:
                best_gains, best_cost = info["best_gains"], info["best_cost"]
        return best_gains, float(best_cost), info

    # ---- persistence -----------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    @classmethod
    def load(cls, path: str | Path, env=None, algorithm: str | None = None, device: str = "auto") -> "RLAgent":
        sb3 = _sb3()
        errors = []
        for name in ([algorithm.upper()] if algorithm else list(SUPPORTED)):
            try:
                return cls(model=getattr(sb3, name).load(str(path), env=env, device=device), algorithm=name)
            except Exception as exc:                 # wrong algorithm class -> try the next one
                errors.append(f"{name}: {exc}")
        raise RuntimeError(f"Could not load model '{path}': " + " | ".join(errors))
