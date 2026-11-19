"""Quickstart: tune the DC motor from the project notes with the RL tuner.

    python examples/quickstart.py                       # trains a tiny agent first (~1-2 min)
    python examples/quickstart.py runs/<run>/best_model.zip   # use a model you trained yourself
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_pid_tuner import Plant
from rl_pid_tuner.tuner import PIDAutoTuner


def main() -> None:
    plant = Plant.dc_motor_position()                 # theta/V = 0.01 / (0.005 s^3 + 0.06 s^2 + 0.1001 s)
    tuner = PIDAutoTuner(plant, spec="balanced")      # presets: fast | balanced | no_overshoot | robust
    model = sys.argv[1] if len(sys.argv) > 1 else None
    result = tuner.tune(model=model, train_steps=0 if model else 3000, out_dir="results/quickstart")
    print("\nStructured output (REQ-I-02):")
    print(result.to_json()[:400], "...")


if __name__ == "__main__":
    main()
