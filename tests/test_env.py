import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rl_pid_tuner import Plant
from rl_pid_tuner.env import OBS_DIM, PIDTuningEnv, env_from_config, plant_from_config
from rl_pid_tuner.plant import random_second_order


def _env(**kw):
    return PIDTuningEnv(plant=Plant.dc_motor_position(), **kw)


def test_check_env_passes():
    check_env(_env(max_steps=4), skip_render_check=True)
    check_env(PIDTuningEnv(plant_sampler=random_second_order, max_steps=4), skip_render_check=True)


def test_spaces_are_float32():
    env = _env()
    obs, info = env.reset(seed=0)
    assert obs.dtype == np.float32 and obs.shape == (OBS_DIM,)
    assert env.action_space.dtype == np.float32 and env.observation_space.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert {"gains", "best_gains", "cost", "k_ref_method"} <= set(info)


def test_random_episodes_have_no_nan():
    env = PIDTuningEnv(plant_sampler=random_second_order, max_steps=10, uncertainty=0.2)
    rng = np.random.default_rng(0)
    for ep in range(30):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, size=3).astype(np.float32))
            assert np.all(np.isfinite(obs)) and np.isfinite(r)
            done = term or trunc


def test_unstable_gains_terminate_with_penalty():
    env = _env(init_noise=0.0, max_steps=50, z_limit=4.0, max_log_step=1.0, k_ref=(72.0, 100.0, 12.0))
    env.reset(seed=0)
    for _ in range(50):                                  # keep pushing the gains up
        obs, r, term, trunc, info = env.step(np.ones(3, dtype=np.float32))
        if term:
            break
    assert term and r == -100.0 and not info["stable"]


def test_episode_truncates_at_max_steps():
    env = _env(max_steps=3, init_noise=0.0)
    env.reset(seed=0)
    flags = [env.step(np.zeros(3, dtype=np.float32)) for _ in range(3)]
    assert [f[3] for f in flags] == [False, False, True]
    assert all(not f[2] for f in flags)


def test_same_seed_same_episode():
    def run():
        env = PIDTuningEnv(plant_sampler=random_second_order, max_steps=3, uncertainty=0.1)
        obs, _ = env.reset(seed=123)
        out = [obs]
        for a in ([0.3, -0.2, 0.1], [0.0, 0.5, -0.5], [-1, 1, 0]):
            out.append(env.step(np.array(a, dtype=np.float32))[0])
        return np.array(out)
    assert np.array_equal(run(), run())


def test_best_gains_tracking_and_reuse_scale_free():
    env = _env(max_steps=5, init_noise=0.0)
    env.reset(seed=0)
    for _ in range(5):
        _, _, _, _, info = env.step(np.array([0.2, 0.0, 0.0], dtype=np.float32))
    assert info["best_gains"] is not None and info["best_cost"] <= info["cost"] + 1e-12
    # The same relative action space works for a plant of very different scale.
    small = PIDTuningEnv(plant=Plant.first_order(K=0.01, tau=1000.0), max_steps=2)
    small.reset(seed=0)
    assert small.step(np.zeros(3, dtype=np.float32))[0].shape == (OBS_DIM,)


def test_robust_mode_scores_multiple_plants():
    env = PIDTuningEnv(plant=Plant.dc_motor_position(), n_robust=3, uncertainty=0.2, max_steps=2, init_noise=0.0)
    env.reset(seed=0)
    assert len(env.plants) == 3
    _, r, _, _, _ = env.step(np.zeros(3, dtype=np.float32))
    assert np.isfinite(r)


def test_constructor_validation_and_config_builders():
    with pytest.raises(ValueError):
        PIDTuningEnv()
    with pytest.raises(ValueError):
        PIDTuningEnv(plant=Plant.first_order(), plant_sampler=random_second_order)
    plant, sampler = plant_from_config({"name": "first_order", "K": 2.0, "tau": 5.0})
    assert sampler is None and np.isclose(plant.dc_gain, 2.0)
    _, sampler = plant_from_config({"family": "second_order"})
    assert sampler is random_second_order
    with pytest.raises(ValueError):
        plant_from_config({"name": "nope"})
    env = env_from_config({"plant": {"name": "dc_motor"}, "spec": "fast", "env": {"max_steps": 3}}, seed=0, eval_mode=True)
    assert env.init_noise == 0.0 and env.max_steps == 3
