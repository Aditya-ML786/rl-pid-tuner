"""rl-pid-tuner: model-free RL-based PID auto-tuner (v1: SISO, linear, simulated)."""
from .baselines import (
    ZNResult,
    design_baselines,
    optimize_gains,
    reference_gains,
    simc,
    ziegler_nichols,
)
from .metrics import StepMetrics, compare, compute_metrics, stability_margins
from .pid_controller import PIDController, PIDGains
from .plant import Plant
from .simulation import (
    Response,
    choose_sim_time_and_dt,
    is_closed_loop_stable,
    simulate,
)
from .spec import TuningSpec, get_spec
from .system_id import FitReport, SystemIdentification, identify
from .tuner import PIDAutoTuner, TuningResult

__all__ = [
    "Plant",
    "PIDController",
    "PIDGains",
    "Response",
    "simulate",
    "is_closed_loop_stable",
    "choose_sim_time_and_dt",
    "StepMetrics",
    "compute_metrics",
    "stability_margins",
    "compare",
    "TuningSpec",
    "get_spec",
    "ZNResult",
    "ziegler_nichols",
    "simc",
    "reference_gains",
    "optimize_gains",
    "design_baselines",
    "FitReport",
    "SystemIdentification",
    "identify",
    "PIDAutoTuner",
    "TuningResult",
]
__version__ = "0.1.0"
