"""Identify a plant from measured step-response data, then tune it.

We pretend a 'real' motor-position system produced noisy step data, identify a linear model from
it, and tune that model. No training is needed (uses the optimizer); pass a model path to use RL.

    python examples/identify_and_tune.py [runs/<run>/best_model.zip]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_pid_tuner import Plant
from rl_pid_tuner.system_id import identify
from rl_pid_tuner.tuner import PIDAutoTuner


def main() -> None:
    # --- 1. "measure" the real system: apply a 1 V step and log the output with sensor noise ---
    rng = np.random.default_rng(0)
    real = Plant.from_tf([0.8], [0.5, 1.0, 0.0])          # unknown to the tuner: K/(s(tau s+1))
    t = np.linspace(0, 10, 500)
    real.reset()
    y = np.array([real.simulate_response(1.0, dt=t[1]) for _ in t])
    y = y + rng.normal(0, 0.005 * y.max(), size=y.shape)

    # --- 2. identify a linear model from the data ---
    plant, report = identify(t, y, u_step=1.0)
    print(f"Identified model: {report.model}  params={report.params}  R^2={report.r2:.4f}")
    print(f"Transfer function: {plant.tf_string}\n")

    # --- 3. tune the identified model ---
    tuner = PIDAutoTuner(step_data=(t, y), spec="balanced")
    model = sys.argv[1] if len(sys.argv) > 1 else None
    tuner.tune(model=model, method="rl" if model else "optimizer", out_dir="results/identify_and_tune")


if __name__ == "__main__":
    main()
