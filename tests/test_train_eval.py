from pathlib import Path

import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_pid_tuner import Plant
from rl_pid_tuner.agent import RLAgent
from rl_pid_tuner.env import PIDTuningEnv
from rl_pid_tuner.evaluate import evaluate
from rl_pid_tuner.train import DEFAULTS, _deep_merge, load_config, train

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize(
    "filename",
    ["smoke.yaml", "dc_motor.yaml", "dc_motor_robust.yaml", "second_order.yaml"],
)
def test_all_yaml_configs_load(filename):
    cfg = load_config(CONFIGS_DIR / filename)
    for key in ("algorithm", "total_steps", "plant", "spec"):
        assert key in cfg
    assert cfg["algorithm"] in ("SAC", "PPO")
    assert cfg["total_steps"] > 0


def test_agent_save_load_and_select_action(tmp_path):
    env = DummyVecEnv([lambda: PIDTuningEnv(plant=Plant.first_order(), max_steps=3)])
    agent = RLAgent(env, algorithm="SAC", seed=0, device="cpu", learning_starts=10, batch_size=16)
    agent.train(total_timesteps=30)

    model_path = tmp_path / "model.zip"
    agent.save(model_path)
    assert model_path.exists()

    reloaded = RLAgent.load(model_path, algorithm="SAC", device="cpu")
    raw_env = PIDTuningEnv(plant=Plant.first_order(), max_steps=3)
    obs, _ = raw_env.reset(seed=0)
    action = reloaded.select_action(obs, deterministic=True)
    assert action.shape == (3,)
    assert action.dtype == np.float32
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_train_and_evaluate_smoke(tmp_path):
    cfg = _deep_merge(
        DEFAULTS,
        {
            "name": "pytest_smoke",
            "algorithm": "SAC",
            "total_steps": 150,
            "seed": 0,
            "device": "cpu",
            "n_envs": 1,
            "vec_env": "dummy",
            "plant": {"name": "first_order", "K": 1.0, "tau": 1.0},
            "spec": "balanced",
            "env": {"max_steps": 3},
            "algo": {"learning_starts": 20, "batch_size": 16},
            "eval": {"freq": 100, "episodes": 1},
            "checkpoint_freq": 150,
        },
    )
    run_dir = tmp_path / "run"
    out_dir = train(cfg, run_dir=run_dir)
    assert out_dir == run_dir
    for fname in ("best_model.zip", "final_model.zip", "config.yaml", "run_summary.json"):
        assert (run_dir / fname).exists()

    import json

    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert np.isfinite(summary["wall_clock_s"])
    assert summary["tuned_gains"] is not None

    report = evaluate(
        run_dir / "best_model.zip",
        cfg,
        n_perturbed=2,
        uncertainty=0.1,
        seed=0,
        include_optimizer=False,
        out_dir=run_dir,
    )
    assert (run_dir / "eval_report.json").exists()
    assert (run_dir / "rl_vs_baselines.png").exists()
    assert "summary" in report and "verdict" in report
    assert "rl" in report["summary"]
