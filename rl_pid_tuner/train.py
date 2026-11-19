"""Training entry point:  ``python -m rl_pid_tuner.train --config configs/dc_motor.yaml --seed 1``

*You* run the full training locally (see docs/TRAINING_GUIDE.md); the AI only smoke-tests it.
Everything for one run lands in ``runs/<name>_<timestamp>/``:

    best_model.zip   best agent seen during periodic evaluation (use this one)
    final_model.zip  agent at the end of training
    run_summary.json numbers to paste back for the final README results
    config.yaml      exact config used (needed for --resume / evaluate)
    tb/              TensorBoard logs (only if tensorboard is installed)
    checkpoints/     periodic snapshots
    eval_logs/       evaluations.npz written by the evaluation callback
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .env import env_from_config

DEFAULTS: dict = {
    "name": "run",
    "algorithm": "SAC",
    "total_steps": 200_000,
    "seed": 0,
    "device": "auto",           # "cpu" is usually just as fast for this small network
    "n_envs": 1,                # keep 1-4: each env already runs a full simulation per step
    "vec_env": "dummy",         # "subproc" = multiprocessing; NOT recommended on Windows
    "output_dir": "runs",
    "plant": {"name": "dc_motor_position"},
    "spec": "balanced",
    "uncertainty": 0.0,
    "env": {"max_steps": 10},
    "algo": {},                  # extra SB3 kwargs, e.g. {learning_rate: 0.001}
    "eval": {"freq": 5000, "episodes": 5},
    "checkpoint_freq": 25_000,
}


def _deep_merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path | None) -> dict:
    """Read a YAML config and fill in defaults."""
    import yaml

    user = yaml.safe_load(Path(path).read_text(encoding="utf-8")) if path else {}
    return _deep_merge(DEFAULTS, user or {})


def _make_env(cfg: dict, seed: int):
    """Top-level (picklable) env factory - required if SubprocVecEnv is ever used on Windows."""
    from stable_baselines3.common.monitor import Monitor

    return Monitor(env_from_config(cfg, seed=seed))


def _vec_env(cfg: dict, seed: int):
    from functools import partial

    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    fns = [partial(_make_env, cfg, seed + i) for i in range(int(cfg["n_envs"]))]
    if cfg["vec_env"] == "subproc" and len(fns) > 1:
        return SubprocVecEnv(fns, start_method="spawn")
    return DummyVecEnv(fns)                          # single process: safe everywhere


def _metrics_logger_cls():
    from stable_baselines3.common.callbacks import BaseCallback

    class MetricsLogger(BaseCallback):
        """Log physically meaningful curves (overshoot, settling time, cost) to TensorBoard."""

        def __init__(self, every: int = 500):
            super().__init__()
            self.every, self.buf = every, {"overshoot": [], "settling_time": [], "cost": []}

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                m = info.get("metrics")
                if m and m.get("stable"):
                    self.buf["overshoot"].append(m["overshoot_pct"])
                    self.buf["settling_time"].append(m["settling_time"])
                    self.buf["cost"].append(info["cost"])
            if self.n_calls % self.every == 0:
                for k, v in self.buf.items():
                    if v:
                        self.logger.record(f"custom/{k}", float(np.mean(v)))
                    v.clear()
            return True

    return MetricsLogger


def _bench_sims_per_sec(cfg: dict, seed: int, n: int = 15) -> float:
    env = env_from_config(cfg, seed=seed)
    env.reset(seed=seed)
    t0, rng = time.perf_counter(), np.random.default_rng(seed)
    count = 0
    for _ in range(n):
        _, _, term, trunc, _ = env.step(rng.uniform(-1, 1, 3).astype(np.float32))
        count += max(env.n_robust, 1)
        if term or trunc:
            env.reset()
    return count / (time.perf_counter() - t0)


def train(cfg: dict, resume: str | Path | None = None, run_dir: str | Path | None = None) -> Path:
    """Run training for ``cfg`` and return the run directory."""
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor

    from .agent import RLAgent

    seed, n_envs = int(cfg["seed"]), int(cfg["n_envs"])
    if resume:
        run_dir = Path(resume)
    elif run_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path(cfg["output_dir"]) / f"{cfg['name']}_{stamp}"
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    import yaml

    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    tb = str(run_dir / "tb") if importlib.util.find_spec("tensorboard") else None
    if tb is None:
        print("[info] tensorboard is not installed -> training curves are not logged (pip install tensorboard).")

    sims_per_sec = _bench_sims_per_sec(cfg, seed)
    print(f"[info] simulator speed: ~{sims_per_sec:.0f} sims/s -> {cfg['total_steps']} steps "
          f"~ {cfg['total_steps'] * max(cfg['env'].get('n_robust', 1), 1) / sims_per_sec / 60:.1f} min (+ network updates)")

    env = _vec_env(cfg, seed)
    eval_env = Monitor(env_from_config(cfg, seed=seed + 10_000, eval_mode=True))
    callbacks = CallbackList([
        EvalCallback(eval_env, best_model_save_path=str(run_dir), log_path=str(run_dir / "eval_logs"),
                     eval_freq=max(int(cfg["eval"]["freq"]) // n_envs, 1),
                     n_eval_episodes=int(cfg["eval"]["episodes"]), deterministic=True),
        CheckpointCallback(save_freq=max(int(cfg["checkpoint_freq"]) // n_envs, 1),
                           save_path=str(run_dir / "checkpoints"), name_prefix="ckpt"),
        _metrics_logger_cls()(),
    ])

    if resume:
        src = run_dir / "final_model.zip"
        if not src.exists():
            ckpts = sorted((run_dir / "checkpoints").glob("ckpt_*.zip"), key=lambda p: p.stat().st_mtime)
            if not ckpts:
                raise FileNotFoundError(f"Nothing to resume from in {run_dir}")
            src = ckpts[-1]
        print(f"[info] resuming from {src}")
        agent = RLAgent.load(src, env=env, algorithm=cfg["algorithm"], device=cfg["device"])
    else:
        agent = RLAgent(env, cfg["algorithm"], seed=seed, device=cfg["device"], tensorboard_log=tb, **cfg["algo"])

    t0 = time.perf_counter()
    interrupted = False
    try:
        agent.train(int(cfg["total_steps"]), callback=callbacks, reset_num_timesteps=not resume)
    except KeyboardInterrupt:                        # Ctrl+C is safe: best_model.zip already exists
        interrupted = True
        print("\n[info] interrupted - saving what we have.")
    wall = time.perf_counter() - t0

    agent.save(run_dir / "final_model")
    if not (run_dir / "best_model.zip").exists():    # no evaluation happened (very short run)
        agent.save(run_dir / "best_model")
    summary = summarize_run(cfg, run_dir, agent, wall, sims_per_sec, interrupted)
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] {run_dir}\n       best eval reward: {summary['best_eval_reward']}  "
          f"tuned gains: {summary['tuned_gains']}")
    return run_dir


def summarize_run(cfg: dict, run_dir: Path, agent, wall: float, sims_per_sec: float, interrupted: bool) -> dict:
    """Collect the numbers needed for the README results (and REQ-P-02 / REQ-P-03)."""
    from .agent import RLAgent

    best_reward, steps_to_best = None, None
    ev = run_dir / "eval_logs" / "evaluations.npz"
    if ev.exists():
        data = np.load(ev)
        mean_rewards = data["results"].mean(axis=1)
        i = int(np.argmax(mean_rewards))
        best_reward, steps_to_best = float(mean_rewards[i]), int(data["timesteps"][i])

    best_agent = RLAgent.load(run_dir / "best_model.zip", algorithm=cfg["algorithm"])
    env = env_from_config(cfg, seed=cfg["seed"] + 20_000, eval_mode=True)
    gains, cost, info = best_agent.tune(env, n_episodes=1)
    max_steps = int(cfg["env"].get("max_steps", 10))
    return {
        "name": cfg["name"], "algorithm": cfg["algorithm"], "seed": cfg["seed"],
        "total_steps": int(cfg["total_steps"]), "interrupted": interrupted,
        "wall_clock_s": round(wall, 1), "sims_per_sec_estimate": round(sims_per_sec, 1),
        "training_episodes_approx": int(cfg["total_steps"]) // max_steps,
        "best_eval_reward": best_reward, "steps_to_best_eval": steps_to_best,
        "episodes_to_best_eval": None if steps_to_best is None else steps_to_best // max_steps,
        "tuned_gains": None if gains is None else dict(zip(("kp", "ki", "kd"), map(float, gains))),
        "tuned_cost": cost, "k_ref": info.get("k_ref"), "k_ref_method": info.get("k_ref_method"),
        "files": {"best_model": "best_model.zip", "final_model": "final_model.zip"},
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train an RL PID tuner (SAC/PPO).")
    ap.add_argument("--config", help="YAML config, e.g. configs/dc_motor.yaml")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--total-steps", type=int)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--n-envs", type=int)
    ap.add_argument("--output-dir")
    ap.add_argument("--run-dir", help="exact folder for this run (default: runs/<name>_<time>)")
    ap.add_argument("--resume", help="existing run folder to continue training in")
    args = ap.parse_args(argv)

    if not args.config and not args.resume:
        ap.error("give --config (new run) or --resume <run_folder>")
    cfg = load_config(Path(args.resume) / "config.yaml" if (args.resume and not args.config) else args.config)
    for key, val in (("seed", args.seed), ("total_steps", args.total_steps), ("device", args.device),
                     ("n_envs", args.n_envs), ("output_dir", args.output_dir)):
        if val is not None:
            cfg[key] = val
    train(cfg, resume=args.resume, run_dir=args.run_dir)


if __name__ == "__main__":      # required on Windows if multiprocessing is ever enabled
    main()
