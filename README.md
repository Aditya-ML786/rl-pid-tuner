# rl-pid-tuner
Model-free RL-based PID auto-tuner for Python.

Classical tuning methods (Ziegler-Nichols, manual tuning) work well when you 
have a clean transfer function and a single objective. They get harder to 
apply when:
1. You're balancing multiple competing objectives (overshoot vs settling 
   time vs control effort), which classical formulas don't optimize for directly.
2. You want to tune in simulation first, before running trial-and-error on 
   real hardware.

RL replaces manual iteration with an automated search in simulation.

v1 scope: SISO, linear plants only, simulated (no real hardware loop yet). 
Nonlinear plants and cascade PID are explicitly out of scope until this 
pipeline works end to end.

Status: under active development. PIDController implemented; PIDEnv and 
training loop in progress.