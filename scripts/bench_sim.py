"""Micro-benchmark for closed-loop simulation throughput (pure NumPy/SciPy)."""
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_pid_tuner import Plant, simulate


def main() -> None:
    motor = Plant.dc_motor_position()
    gains = (72.07, 102.65, 12.65)
    n_runs = 500

    # Warm up ZOH discretization cache once before timing.
    simulate(motor, gains, dt=0.005, t_end=5.0)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        resp = simulate(motor, gains, dt=0.005, t_end=5.0)
        assert resp.stable
    elapsed = time.perf_counter() - t0
    print(f"Completed {n_runs} DC-motor sims in {elapsed:.3f} s ({n_runs / elapsed:.1f} sims/s)")


if __name__ == "__main__":
    main()
