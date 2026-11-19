# rl-pid-tuner

Model-free Reinforcement Learning PID auto-tuner for Python (Gymnasium + Stable-Baselines3).

Classical PID tuning rules (Ziegler-Nichols, manual trial-and-error) rely on narrow assumptions or
single objectives, while commercial autotuners sit behind MATLAB paywalls. `rl-pid-tuner` tunes
parallel PID gains `(Kp, Ki, Kd)` directly in simulation against user-defined performance specs,
and compares the result honestly against Ziegler-Nichols, SIMC, and a numerical Nelder-Mead
optimizer.

---

## v1 Scope

- **SISO only** (Single-Input, Single-Output)
- **Linear plant models only** (continuous transfer function, state-space, or identified 1st/2nd-order step models; no dead time)
- **Offline simulation only** (Zero-Order Hold discrete simulation with optional actuator limits, anti-windup, derivative filter, load disturbances, and measurement noise)
- **Out of scope in v1:** nonlinear plant dynamics, MIMO systems, cascade PID, real hardware-in-the-loop, GUI.

---

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[train,plot,dev]"
pytest
```

Core dependencies (`numpy`, `scipy`) handle plant modeling, simulation, system identification, metrics, and classical/optimizer baselines; `[train]` adds `gymnasium`, `stable-baselines3`, `torch`, `tensorboard`, and `pyyaml`.

---

## Python Quickstart

### 1. Tune a known plant (DC motor benchmark)

```python
from rl_pid_tuner import PIDAutoTuner, Plant

plant = Plant.dc_motor_position()          # 0.01 / (0.005 s^3 + 0.06 s^2 + 0.1001 s)
tuner = PIDAutoTuner(plant, spec="balanced")

# Use a pretrained model (recommended) or train briefly on the fly (train_steps=3000)
result = tuner.tune(model="runs/<run_folder>/best_model.zip", out_dir="results/quickstart")
print(result.summary())
print(result.to_json())
```

### 2. Identify a plant from step-response data, then tune it

```python
import numpy as np
from rl_pid_tuner import PIDAutoTuner, Plant

# Example synthetic step test
real = Plant.from_tf([0.8], [0.5, 1.0, 0.0])
t = np.linspace(0, 10, 500)
real.reset()
y = np.array([real.simulate_response(1.0, dt=t[1]) for _ in t])

tuner = PIDAutoTuner(step_data=(t, y), spec="balanced")
result = tuner.tune(method="optimizer", out_dir="results/identified")
```

### Tuning Spec Presets (`spec=...`)
- `"fast"` — prioritizes settling speed and rise time
- `"balanced"` — balanced overshoot (≤ 10%), settling time, SSE, and phase margin (≥ 45°)
- `"no_overshoot"` — suppresses overshoot (≤ 2%)
- `"robust"` — emphasizes gain/phase margins (≥ 60° PM) and gentle actuator effort
- Or pass a custom `TuningSpec` / dictionary override (including `u_limits=(-12, 12)`).

---

## Command-Line Interface

After `pip install -e .` (or via `python -m rl_pid_tuner.cli`):

```powershell
# Tune a plant transfer function using a trained RL policy
rl-pid-tuner tune --tf "0.01" "0.005 0.06 0.1001 0" --spec balanced --model runs/<run_folder>/best_model.zip --out-dir results/cli

# Train an RL agent from a YAML config
rl-pid-tuner train --config configs/dc_motor.yaml --seed 1

# Evaluate a trained policy vs ZN, SIMC, and Nelder-Mead on nominal + perturbed plants
rl-pid-tuner evaluate --model runs/<run_folder>/best_model.zip --plant dc_motor
```

---

## Local RL Training

Full RL training runs locally on your machine using the YAML configs in `configs/` (`smoke.yaml`, `dc_motor.yaml`, `dc_motor_robust.yaml`, `second_order.yaml`). See **[docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)** for step-by-step setup, TensorBoard monitoring, and evaluation instructions.

---

## Benchmark Results

### DC Motor Position Control (`configs/dc_motor.yaml`, SAC, 200,000 steps, seed 1)

Trained on CPU for 3,284.7 s (~54.7 min, ~20,000 episodes); best evaluation reward **-5.512** reached at 190,000 steps (~19,000 episodes). Saved artifacts: [dc_motor_run_summary.json](docs/results/dc_motor_run_summary.json) and [dc_motor_eval_report.json](docs/results/dc_motor_eval_report.json).

![RL vs baselines on nominal DC motor](docs/results/dc_motor_rl_vs_baselines.png)

#### Nominal Plant Metrics (`0.01 / (0.005 s^3 + 0.06 s^2 + 0.1001 s)`, `balanced` spec)

| Method | Gains `(Kp, Ki, Kd)` | Overshoot (%) | Rise (s) | Settling (s) | SSE | IAE | GM | PM (deg) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Ziegler-Nichols | `(72.07, 102.65, 12.65)` | 65.34 | 0.320 | 4.991 | 0.0121 | 1.041 | 25.44 | 22.0 |
| SIMC | `(127.79, 111.17, 36.65)` | 16.08 | 0.300 | 2.941 | -0.0001 | 0.528 | 12.27 | 41.1 |
| Nelder-Mead Optimizer | `(105.85, 50.53, 31.85)` | 10.06 | 0.350 | 4.351 | -0.0080 | 0.557 | 14.44 | 45.0 |
| **RL (SAC, 200k steps)** | **`(92.31, 1.69, 28.17)`** | **0.58** | **0.410** | **0.670** | **-0.0052** | **0.361** | **16.43** | **48.0** |

#### Robustness Over 21 Plants (Nominal + 20 Perturbed Copies at ±20% Parameter Uncertainty)

| Method | Nominal Cost | Robust Mean Cost | Worst Cost | Stable (%) | Meets Spec (%) |
|---|---:|---:|---:|---:|---:|
| Ziegler-Nichols | 6.964 | 7.117 | 8.627 | 100% | 0% |
| SIMC | 1.460 | 1.592 | 3.137 | 100% | 0% |
| Nelder-Mead Optimizer | 0.925 | 1.218 | 2.570 | 100% | 5% |
| **RL (SAC, 200k steps)** | **0.198** | **0.344** | **1.455** | **100%** | **81%** |

#### Honest Observations
- **Clear win on nominal & ±20% perturbed plants:** RL outperformed Ziegler-Nichols, SIMC, and Nelder-Mead on both nominal cost (`0.198` vs `0.925`–`6.964`) and robust mean cost (`0.344` vs `1.218`–`7.117`), and satisfied the `balanced` spec (overshoot ≤ 10%, phase margin ≥ 45°, settled within ±2%) on **81%** of the 21 test plants versus 0% (ZN/SIMC) and 5% (Nelder-Mead).
- **Load-disturbance trade-off:** Because RL chose a low integral gain (`Ki = 1.69`), post-disturbance recovery after the load step at `t = 6 s` settles more slowly than higher-`Ki` baselines (visible in the step-response plot). Increasing `w_disturbance` in `TuningSpec` prioritizes tighter load-disturbance rejection.
- **Scope of this table:** These are single-seed (`seed=1`) results on the DC motor position plant. Multi-seed runs, `configs/dc_motor_robust.yaml`, and reusable plant-family policies (`configs/second_order.yaml`) can be added using [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md).
