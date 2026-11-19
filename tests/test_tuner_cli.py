import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_pid_tuner import PIDAutoTuner, Plant, TuningResult, cli
from rl_pid_tuner.agent import RLAgent
from rl_pid_tuner.env import PIDTuningEnv


def test_tuner_optimizer_and_json_roundtrip(tmp_path):
    plant = Plant.first_order(K=1.0, tau=2.0)
    tuner = PIDAutoTuner(plant, spec="balanced")
    result = tuner.tune(method="optimizer", out_dir=tmp_path, verbose=False)

    assert result.method == "optimizer"
    assert result.metrics["stable"] is True
    assert (tmp_path / "tuning_result.json").exists()
    assert (tmp_path / "step_response.png").exists()

    summary_text = result.summary()
    assert "Kp =" in summary_text and "Tuned PID" in summary_text

    # Round-trip from JSON string (tests Windows OSError guard) and from file Path
    from_str = TuningResult.from_json(result.to_json())
    from_file = TuningResult.from_json(tmp_path / "tuning_result.json")
    for loaded in (from_str, from_file):
        assert np.isclose(loaded.kp, result.kp)
        assert np.isclose(loaded.ki, result.ki)
        assert np.isclose(loaded.kd, result.kd)
        assert np.isclose(loaded.cost, result.cost)


def test_tuner_from_step_data():
    true_plant = Plant.first_order(K=2.0, tau=3.0)
    t = np.linspace(0.0, 20.0, 400)
    dt = float(t[1] - t[0])
    true_plant.reset()
    y = np.array([true_plant.simulate_response(1.0, dt=dt) for _ in t])

    tuner = PIDAutoTuner(step_data=(t, y), spec="balanced")
    result = tuner.tune(method="optimizer", verbose=False)
    assert result.identification is not None
    assert result.identification["r2"] > 0.98
    assert result.metrics["stable"] is True


def test_tuner_rl_with_saved_model(tmp_path):
    plant = Plant.first_order(K=1.0, tau=1.0)
    vec = DummyVecEnv([lambda: PIDTuningEnv(plant=plant, max_steps=3)])
    agent = RLAgent(vec, algorithm="SAC", seed=0, device="cpu", learning_starts=10, batch_size=16)
    agent.train(total_timesteps=30)
    model_path = tmp_path / "m.zip"
    agent.save(model_path)

    result = PIDAutoTuner(plant, spec="balanced").tune(model=model_path, verbose=False)
    assert result.method == "rl"
    assert np.isfinite(result.cost)
    assert result.metrics["stable"] is True


def test_cli_help_and_tune(tmp_path):
    with pytest.raises(SystemExit) as exc1:
        cli.main(["--help"])
    assert exc1.value.code == 0

    with pytest.raises(SystemExit) as exc2:
        cli.main(["tune", "--help"])
    assert exc2.value.code == 0

    cli.main(["tune", "--tf", "1", "2 1", "--method", "optimizer", "--out-dir", str(tmp_path)])
    out_json = tmp_path / "tuning_result.json"
    assert out_json.exists()
    loaded = TuningResult.from_json(out_json)
    assert loaded.metrics["stable"] is True
