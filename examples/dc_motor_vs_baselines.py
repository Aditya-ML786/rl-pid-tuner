"""Main experiment: RL vs Ziegler-Nichols vs SIMC vs optimizer on the DC motor.

    python examples/dc_motor_vs_baselines.py runs/<run>/best_model.zip

Prints the comparison tables and writes eval_report.json + rl_vs_baselines.png into ``results/``.
(Same as ``python -m rl_pid_tuner.evaluate --model ... --plant dc_motor``.)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_pid_tuner.evaluate import evaluate
from rl_pid_tuner.train import load_config


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: python examples/dc_motor_vs_baselines.py <path/to/best_model.zip> [config.yaml]")
    cfg = load_config(sys.argv[2] if len(sys.argv) > 2 else "configs/dc_motor.yaml")
    evaluate(sys.argv[1], cfg, n_perturbed=20, uncertainty=0.2, out_dir="results/dc_motor")


if __name__ == "__main__":
    main()
