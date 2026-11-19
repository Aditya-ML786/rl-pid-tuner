# rl-pid-tuner v1: Build Plan (rev 2, based on what the tuner does)

**What changed in rev 2**
- The library's features now come from **what an RL-based PID auto-tuner should actually do**. The SysML model is not the source of requirements; it can be updated to match the code later.
- **Training is split:** the AI writes the training code and runs only a tiny smoke test. **You run the full training on your gaming laptop** using [TRAINING_GUIDE.md](file:///C:/Users/Aditya%20Deshpande/.gemini/antigravity/brain/ab5124d2-2c09-451f-915b-d08505eba7eb/TRAINING_GUIDE.md). That guide gets copied into the repo in Phase 5.

> [!IMPORTANT]
> **v1 scope limits (every phase stays inside these):** SISO · linear plant models (TF / state space) · simulation only · Gymnasium + Stable-Baselines3 · Python.
> **Not in v1:** nonlinear plants, MIMO, cascade PID, dead time, real hardware, GUI, MBSE/Papyrus edits.
> Actuator saturation **is allowed** (it's a controller-side limit and is part of any real PID).

---

## 1. What the tuner does (feature list)

| # | Feature | What the user gets |
|---|---|---|
| F1 | **Plant input** | Give a transfer function or state-space model; built-in examples (1st/2nd order, DC motor). Bad input (MIMO, improper, NaN) is rejected with a clear error. |
| F2 | **Plant from data** | Give measured step-response data and get a fitted linear model (1st order, 2nd order, integrator + 1st order) with fit quality (R²). |
| F3 | **Realistic PID** | Parallel PID with filtered derivative (N), derivative-on-measurement (no "derivative kick"), output limits, and anti-windup. These are the same settings you'd use on real hardware. |
| F4 | **Fast closed-loop simulation** | Setpoint step **and** load-disturbance step, optional sensor noise. Unstable gain sets are rejected *before* simulating, using a closed-loop pole check. |
| F5 | **Performance analysis** | Overshoot, rise time, settling time, steady-state error, IAE/ISE, control effort and smoothness, disturbance-rejection IAE/peak, gain and phase margin. |
| F6 | **Tuning specification** | You say what "good" means: presets `fast`, `balanced`, `no_overshoot`, `robust`, or custom weights and limits (e.g. OS ≤ 10%, Ts ≤ 2 s, u within ±12 V). This becomes the RL reward. |
| F7 | **RL tuning** | A SAC (or PPO) agent refines the gains step by step, starting from a classical first guess. It returns the single best `(Kp, Ki, Kd)` set (plus the filter N it used). |
| F8 | **Robust tuning** | Optional plant uncertainty (e.g. ±20% on each coefficient) during training, so the gains still work when the real plant differs from the model. This is where RL clearly beats one-shot formulas. |
| F9 | **Baselines & honest comparison** | Ziegler-Nichols (ultimate gain), a model-based rule (SIMC/IMC-style, for 1st/2nd-order plants), and a plain numerical optimizer (Nelder-Mead on the same cost) as a "is RL worth it?" check. Side-by-side table + plot. |
| F10 | **Train once, reuse** | Save the trained policy, load it later, tune the same plant again (or a similar one) in seconds without retraining. Optional short fine-tune. |
| F11 | **Outputs** | `TuningResult` → JSON (gains, metrics, baseline comparison, specs met yes/no, training info), plots, and a one-command CLI. |

---

## 2. Core design (how RL fits in)

| Item | Choice | Why |
|---|---|---|
| RL step | **One step = one full closed-loop simulation** with the proposed gains. Episode = up to 10 refinement steps. | Produces one fixed `(Kp, Ki, Kd)` set you can deploy and compare. |
| Gain scaling | Gains expressed **relative to a reference set** `K_ref` (from a classical rule), in log space: `K = K_ref · 10^z`, with z clipped to ±2 decades | Works for any plant size (DC motor Kp ≈ 72 or a plant needing 0.1). It also makes a policy reusable across plants (F10). |
| Action | `Δz ∈ [-1, 1]³ × max_log_step` (default 0.25 decades) | Small, safe refinements |
| Observation (float32, normalized) | Last-sim metrics (OS, Ts/T_ref, Tr/T_ref, sse, IAE, effort, disturbance IAE, margins) + current `z` + steps-left fraction | Tells the agent how the last attempt went and where it is now |
| Reward | `−cost(metrics, spec)`. Cost is a weighted sum + penalties for violated spec limits. Unstable → −100 and episode ends. | Optimizes exactly what the user asked for (F6) |
| Time scale | `dt` and sim time picked automatically from the plant's poles (overridable) | Keeps sims short and accurate for any plant |
| Speed | Pure **NumPy/SciPy** discretized state-space loop (or `scipy.signal.lfilter`). No Numba in v1 (avoids Python 3.14 / NumPy dependency conflicts). | Linear SISO sims (~200–2,000 steps) are fast in pure NumPy without risking `torch`/`SB3` dependency breakage. |

---

## 3. Repo structure (target)

```
rl-pid-tuner/
├── rl_pid_tuner/
│   ├── plant.py          # F1: Plant, examples, random/uncertain plant generator
│   ├── system_id.py      # F2
│   ├── pid_controller.py # F3
│   ├── simulation.py     # F4 (pure NumPy/SciPy discretized kernel)
│   ├── metrics.py        # F5
│   ├── baselines.py      # F9: ZN, SIMC/IMC, optimizer baseline
│   ├── spec.py           # F6: TuningSpec + presets + cost/reward
│   ├── env.py            # PIDTuningEnv (Gymnasium)
│   ├── agent.py          # SB3 wrapper: build, train, save/load
│   ├── train.py          # training entry point (config-driven)  ← YOU run full training
│   ├── evaluate.py       # evaluate a trained model vs baselines
│   ├── tuner.py          # F7/F10/F11: PIDAutoTuner, TuningResult
│   ├── plotting.py
│   └── cli.py            # `rl-pid-tuner tune|train|evaluate`
├── configs/  smoke.yaml  dc_motor.yaml  dc_motor_robust.yaml  second_order.yaml
├── examples/ quickstart.py  dc_motor_vs_baselines.py  identify_and_tune.py
├── tests/    test_<module>.py
├── docs/     TRAINING_GUIDE.md
├── pyproject.toml  requirements.txt  README.md  LICENSE  .gitignore
```

---

## 4. Phases

Engineering guardrails (apply to all phases):
- **Float comparisons:** always use `np.isclose` or `pytest.approx` in tests (never exact `==` on floats). Keep RL smoke tests focused on invariants (shapes, no NaNs, serialization) rather than convergence targets.
- **No Numba in v1:** use pure NumPy/SciPy only (Python 3.14 compatibility).
- **Windows-safe Gym/SB3:** standardize all Gymnasium spaces/outputs to `np.float32`, clip observations to bounds, and default to `DummyVecEnv` (only use `SubprocVecEnv` if explicitly configured inside `if __name__ == "__main__":`).

Rules for every phase: read only what's listed, create only what's listed, **end with `pytest` green**, no long training runs (smoke tests ≤ 2,000 steps).

### Phase 0: Scaffold + Plant (F1)
- **Create:** package skeleton, `pyproject.toml` (extras: `[train]` = SB3, torch, tensorboard, pyyaml; `[plot]` = matplotlib), `requirements.txt`, `plant.py`, `tests/test_plant.py`.
- `Plant.from_tf`, `Plant.from_ss`, validation, `poles`, `dc_gain`, `time_scale()`, examples `first_order`, `second_order`, `dc_motor_position(J,b,K,R,L)`, `perturbed(rel=0.2, rng)` for uncertainty.
- **Accept:** DC motor = `0.01/(0.005s³+0.06s²+0.1001s)`. Poles of the DC motor ≈ {0, −2, −10}. MIMO/improper input raises `ValueError`.

### Phase 1: PID controller + simulator (F3, F4)
- **Create:** `pid_controller.py`, `simulation.py`, tests, `scripts/bench_sim.py`.
- Discrete PID (Tustin/backward-Euler), filtered D, D-on-measurement option, output limits, anti-windup (clamping), `reset`, `set_gains`.
- `simulate(plant, gains, setpoint=1, disturbance=None, noise_std=0, u_limits=None, dt=auto, t_end=auto) → Response(t, y, u, e, stable)`.
- `is_closed_loop_stable(plant, gains)`: an analytic check that runs before simulating.
- **Accept:** P-only on `1/(s+1)` matches the analytic final value `Kp/(1+Kp)`. Anti-windup test shows less overshoot than without it under saturation. Benchmark prints sims/s (pure NumPy/SciPy).

### Phase 2: Metrics + baselines (F5, F9)
- **Create:** `metrics.py`, `baselines.py`, `plotting.py`, tests.
- `StepMetrics` (all of F5), `compare(results: dict[str, StepMetrics]) → table`.
- `ziegler_nichols(plant)`: finds Ku and Tu via **bracketed frequency-domain phase crossover** (`scipy.signal.freqresp` on a log-spaced grid + `scipy.optimize.brentq` on $\text{Im}(G(j\omega))=0$ where $\text{Re}(G(j\omega))<0$); raises a clear `ValueError` if no $-180^\circ$ crossover exists (e.g. 1st-order plants). `simc(plant)` for 1st/2nd-order plants. `optimize_gains(plant, spec)` with Nelder-Mead over log-gains.
- **Accept (golden test, verified earlier):** DC motor Ku ≈ 120.12, Tu ≈ 1.404 s, gains ≈ (72.07, 102.65, 12.65). An unfiltered ideal PID sim gives **OS ≈ 61.8%, Ts(2%) ≈ 5.01 s, Tr ≈ 0.29 s** (±5%). Margins match `scipy`/hand calculation for a known plant.

### Phase 3: System identification (F2)
- **Create:** `system_id.py`, tests, `examples/identify_and_tune.py` (identification half only).
- `identify(t, y, u_step, model="auto"|"first_order"|"second_order"|"integrator_first_order") → (Plant, fit_report)`. `auto` picks the model with the best R².
- **Accept:** recover known parameters within 5% from noisy synthetic data, R² > 0.98.

### Phase 4: Spec + Gymnasium environment
- **Create:** `spec.py`, `env.py`, tests.
- `TuningSpec` (presets + custom), `cost(metrics, spec)`.
- `PIDTuningEnv(plant | plant_sampler, spec, k_ref="auto", max_steps=10, uncertainty=0.0, ...)`. A `plant_sampler` lets one policy be trained across a family of plants.
- **Accept:** `gymnasium.utils.env_checker.check_env` passes (spaces and returned obs/actions explicitly `np.float32`, obs clipped to bounds). 200 random episodes give no NaN. Unstable gains → −100 and terminated. Same seed → identical episode.

### Phase 5: Training code + training guide (you run full training)
- **Create:** `agent.py`, `train.py`, `evaluate.py`, `configs/*.yaml`, `docs/TRAINING_GUIDE.md` (from the artifact), tests (smoke only).
- `train.py --config X.yaml`: vectorized envs (`DummyVecEnv` by default on Windows; `SubprocVecEnv` only if explicitly enabled under `if __name__ == "__main__":` with a top-level picklable factory), TensorBoard logging, `EvalCallback` (keeps `best_model.zip`), periodic checkpoints, `--resume`, seed, device auto/cpu, writes `run_summary.json`.
- `evaluate.py --model ... --plant dc_motor`: RL vs ZN vs SIMC vs optimizer on nominal **and** perturbed plants → `eval_report.json` + plot.
- **Accept:** `train.py --config configs/smoke.yaml` (~2k steps) finishes in < 2 min and produces all output files. Evaluation runs on the smoke model.
- **Then:** you run `configs/dc_motor.yaml` (and optionally `dc_motor_robust.yaml`) locally following the guide.

### Phase 6: Tuner API + CLI + README (F7, F10, F11)
- **Create:** `tuner.py`, `cli.py`, `examples/quickstart.py`, `examples/dc_motor_vs_baselines.py`, README update.
- `PIDAutoTuner(plant=..., spec="balanced").tune(model=None|path, train_steps=...)` → `TuningResult` (JSON, plot, `summary()`).
- CLI: `rl-pid-tuner tune --tf "0.01" "0.005 0.06 0.1001 0" --spec balanced --model runs/.../best_model.zip`.
- **Accept:** quickstart runs end to end with the smoke model. JSON round-trips. CLI `--help` works.

### Phase 7: Results (after your local training)
- You paste back `run_summary.json` + `eval_report.json`. The AI updates the README results table and plot, and writes down honestly whether RL beat ZN / SIMC / optimizer on the nominal and perturbed plants.

---

## 5. Handoff prompt (paste at the start of each phase)

```
Project: rl-pid-tuner v1. Read docs/implementation_plan.md. Do ONLY Phase <k>.
Read only: <files from earlier phases you need>. Create/modify only the Phase <k> files.
Stay inside the v1 scope limits. No long training runs (smoke ≤ 2,000 steps).
Finish with pytest (all old + new tests green) and summarize the public API you added.
```
