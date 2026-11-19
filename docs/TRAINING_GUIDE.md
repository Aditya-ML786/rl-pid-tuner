# Training Guide: rl-pid-tuner (run this on your laptop)

> [!NOTE]
> The commands below are what the Phase 5 code will provide. They work once Phases 0–5 are done. This file gets copied to `docs/TRAINING_GUIDE.md` in the repo.

## 1. What "training" means here (2-minute version)

- The **agent** (SAC) is a small neural network. It looks at how the last gain set performed (overshoot, settling time, …) and proposes a better one.
- Each proposal is tested by a **full closed-loop simulation**. The agent gets a **reward**: higher when the response meets your spec (`balanced`, `fast`, …).
- Over many thousands of attempts, it learns which gain changes improve the response. **No dataset is needed**; the simulator produces all the data.
- Output of training: `best_model.zip` (the trained agent). The tuner uses it to produce your final `(Kp, Ki, Kd)`.

## 2. Hardware notes for a gaming laptop

| Part | Matters? | Why |
|---|---|---|
| **CPU cores** | **Most** | Simulations run on the CPU. More cores means more parallel environments (`n_envs`). |
| GPU (NVIDIA) | Little | The network is tiny (2×256). For SAC, the CPU is often as fast or faster. You can try `device: cuda` for comparison; it's optional. |
| RAM | Low | < 4 GB is enough |
| Power/cooling | Yes | Plug in the charger, set Windows power mode to **Best performance**, and use the laptop's performance/turbo fan mode. |

## 3. One-time setup (PowerShell, in the repo folder)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[train,plot]"
pytest                                  # everything should pass
```
Optional GPU check: `python -c "import torch; print(torch.cuda.is_available())"`.
If you want CUDA, install the matching PyTorch build from pytorch.org. Not required.

## 4. Step 1: smoke test (~1–2 min)

```powershell
python -m rl_pid_tuner.train --config configs/smoke.yaml
```
Should finish without errors and create `runs/smoke_<date>/` with `best_model.zip`, `run_summary.json`, `tb/`. If this fails, stop and send the error to the AI.

## 5. Step 2: real training

```powershell
python -m rl_pid_tuner.train --config configs/dc_motor.yaml --seed 1
```
- Default budget: ~200k steps (tune with `--total-steps`). Rough time is **10–60 min**, depending on your CPU. The script prints sims/sec at the start so you can estimate.
- Optional robust version (gains that tolerate ±20% plant error):
  `python -m rl_pid_tuner.train --config configs/dc_motor_robust.yaml --seed 1`
- For trustworthy results, repeat with `--seed 2` and `--seed 3` (RL results vary between seeds).

## 6. Watching training (TensorBoard)

In a **second** PowerShell window:
```powershell
.\.venv\Scripts\Activate.ps1
tensorboard --logdir runs
```
Open http://localhost:6006.

| Curve | Healthy looks like |
|---|---|
| `eval/mean_reward` | **The main one.** Rises, then flattens. Flat for a long stretch means training is done. |
| `rollout/ep_rew_mean` | Noisy but trending up |
| `custom/overshoot`, `custom/settling_time` | Trending down toward your spec |
| `train/ent_coef` | Decreases over time (less random exploration) |
| `train/actor_loss`, `critic_loss` | Bounce around; no NaN. Don't over-read these. |

**When to stop:** `eval/mean_reward` hasn't improved for ~25% of the run, or the spec is met. `Ctrl+C` is safe: the best model is already saved.
**Resume:** `python -m rl_pid_tuner.train --config configs/dc_motor.yaml --resume runs/<run_folder>`

## 7. Step 3: evaluate

```powershell
python -m rl_pid_tuner.evaluate --model runs/<run_folder>/best_model.zip --plant dc_motor
```
This compares RL vs Ziegler-Nichols vs SIMC vs a plain optimizer on the nominal and perturbed plants. It writes `eval_report.json` and `rl_vs_baselines.png`.

## 8. Troubleshooting

| Symptom | Likely cause → fix |
|---|---|
| Reward stuck at −100 | Most attempts unstable → lower `max_log_step` or `z_limit` in the config |
| Reward flat from the start | Too few steps, or learning rate too low → let it run longer; try `learning_rate: 1e-3` |
| NaN error | Should be caught by guards. Send the traceback to the AI. |
| Very slow (< 200 sims/s) | CPU bottleneck or `n_envs` too high → use `DummyVecEnv` (`n_envs: 1` or `2`) and close background apps |
| RL worse than optimizer baseline | Normal for a single fixed plant. RL's advantage shows in the robust and reuse cases. Report it honestly. |

## 9. What to send back to the AI (Phase 7)

From each run folder: `run_summary.json`, `eval_report.json`, `rl_vs_baselines.png`, plus a screenshot of the `eval/mean_reward` curve. **Don't** send `best_model.zip` (not needed).
