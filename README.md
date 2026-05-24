# rl-pid-tuner
Model-free RL-based PID auto-tuner for Python. No MATLAB, no clean plant model required.

This is a python library which automatically tune PID controller gains(Kp,Ki,Kd) using reinforcement learning
The problems it solves are: 
1.Solving by Use Ziegler-Nichols or similar methods which requires plant model to be linear time invarient which in real system does not happen.
2.Dependance on matlab
3.Tuning in trial and error basis which requires significant expertise and it's a slow process

This project is for anyone who is interested in controls systems and want to build systems without expensive tooling or clean mathematical plant
Status: Under active development

Planned features:
- RL-based PID gain tuning from step response data
- Black-box plant support (no transfer function needed)
- installable via pip
