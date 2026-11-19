"""High-level API: give a plant (or step-response data) and a spec, get tuned PID gains (F7, F10, F11)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .baselines import design_baselines
from .metrics import beats, compare
from .plant import Plant
from .spec import TuningSpec, get_spec, make_sim_settings, meets_spec, run_trial
from .system_id import FitReport, identify


@dataclass
class TuningResult:
    """Everything about one tuning run; ``to_json`` gives the structured gain output (REQ-I-02)."""

    kp: float
    ki: float
    kd: float
    derivative_filter_n: float          # filter coefficient (rad/s) the gains were tuned with
    method: str                         # "rl" | "optimizer"
    plant: str                          # human-readable transfer function
    spec: dict
    metrics: dict
    cost: float
    meets_spec: bool
    baselines: dict = field(default_factory=dict)   # name -> {gains, metrics, cost}
    beats_zn: bool | None = None        # None if ZN does not exist for this plant
    beats_simc: bool | None = None
    training: dict = field(default_factory=dict)
    identification: dict | None = None

    def gains(self) -> tuple[float, float, float]:
        return (self.kp, self.ki, self.kd)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path | None = None) -> str:
        text = json.dumps(self.to_dict(), indent=2)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, text_or_path: str | Path) -> "TuningResult":
        if isinstance(text_or_path, Path):
            text = text_or_path.read_text(encoding="utf-8")
        else:
            raw = str(text_or_path)
            if raw.lstrip().startswith("{"):
                text = raw
            else:
                try:
                    p = Path(raw)
                    text = p.read_text(encoding="utf-8") if p.exists() else raw
                except OSError:
                    text = raw
        return cls(**json.loads(text))

    def summary(self) -> str:
        m = self.metrics
        lines = [
            f"Tuned PID ({self.method}) for {self.plant}",
            f"  Kp = {self.kp:.5g}   Ki = {self.ki:.5g}   Kd = {self.kd:.5g}   (derivative filter N = {self.derivative_filter_n:.4g} rad/s)",
            f"  overshoot {m.get('overshoot_pct')}%  settling {m.get('settling_time')} s  sse {m.get('sse')}  "
            f"phase margin {m.get('phase_margin_deg')} deg",
            f"  meets spec '{self.spec.get('name')}': {self.meets_spec}   cost {self.cost:.3f}",
        ]
        for name, key in (("Ziegler-Nichols", "zn"), ("SIMC", "simc")):
            if key in self.baselines:
                b = self.baselines[key]
                beat = self.beats_zn if key == "zn" else self.beats_simc
                lines.append(f"  vs {name}: cost {b['cost']:.3f} -> {'better or equal' if beat else 'NOT better'}")
        if self.beats_zn is None:
            lines.append("  vs Ziegler-Nichols: not defined for this plant (no -180 degree phase crossover)")
        return "\n".join(lines)


class PIDAutoTuner:
    """Tune PID gains for a linear SISO plant.

    Args:
        plant: a :class:`Plant`, or ``None`` if you pass ``step_data``.
        step_data: ``(t, y)`` measured step response; the plant is identified from it first.
        spec: preset name (``fast | balanced | no_overshoot | robust``), dict, or :class:`TuningSpec`.
        u_step: size of the input step used for ``step_data``.
        uncertainty: relative plant uncertainty used while tuning (robust tuning).
        env_options: extra :class:`~rl_pid_tuner.env.PIDTuningEnv` options (e.g. ``n_robust=4``).
    """

    def __init__(self, plant: Plant | None = None, step_data=None, spec="balanced", u_step: float = 1.0,
                 uncertainty: float = 0.0, env_options: dict | None = None, seed: int = 0):
        if (plant is None) == (step_data is None):
            raise ValueError("Provide exactly one of `plant` or `step_data=(t, y)`.")
        self.spec: TuningSpec = get_spec(spec)
        self.uncertainty, self.env_options, self.seed = float(uncertainty), dict(env_options or {}), seed
        self.fit_report: FitReport | None = None
        if plant is None:
            plant, self.fit_report = identify(step_data[0], step_data[1], u_step=u_step)
        self.plant = plant

    # ------------------------------------------------------------------
    def _make_env(self):
        from .env import PIDTuningEnv

        opts = {**self.env_options, "init_noise": 0.0}
        return PIDTuningEnv(plant=self.plant, spec=self.spec, uncertainty=self.uncertainty, **opts)

    def _get_agent(self, model, train_steps: int, verbose: bool):
        from .agent import RLAgent

        if model is not None and not isinstance(model, (str, Path)):
            return model, {"source": "agent object"}
        if model is not None:
            return RLAgent.load(model), {"source": str(model), "train_steps": 0}
        if train_steps and train_steps > 0:
            import time

            from stable_baselines3.common.vec_env import DummyVecEnv

            opts = {**self.env_options}
            from .env import PIDTuningEnv

            vec = DummyVecEnv([lambda: PIDTuningEnv(plant=self.plant, spec=self.spec,
                                                    uncertainty=self.uncertainty, **opts)])
            agent = RLAgent(vec, "SAC", seed=self.seed, device="cpu", learning_starts=min(500, train_steps // 4),
                            batch_size=64)
            t0 = time.perf_counter()
            if verbose:
                print(f"Training SAC for {train_steps} steps on this plant ...")
            agent.train(train_steps)
            return agent, {"source": "trained on the fly", "train_steps": int(train_steps),
                           "train_seconds": round(time.perf_counter() - t0, 1)}
        raise ValueError("RL tuning needs a trained model (model='.../best_model.zip') or train_steps > 0. "
                         "Use method='optimizer' for a no-training alternative.")

    def tune(self, model=None, train_steps: int = 0, method: str = "rl", out_dir: str | Path | None = None,
             verbose: bool = True) -> TuningResult:
        """Return a :class:`TuningResult` (and save JSON + plot into ``out_dir`` if given)."""
        if method not in ("rl", "optimizer"):
            raise ValueError("method must be 'rl' or 'optimizer'.")
        settings = make_sim_settings(self.plant, self.spec)
        base_gains = design_baselines(self.plant, self.spec, settings, include_optimizer=True)

        training: dict = {}
        used = method
        gains = None
        if method == "rl":
            agent, training = self._get_agent(model, train_steps, verbose)
            gains, _, info = agent.tune(self._make_env(), n_episodes=1)
            training["k_ref_method"] = info.get("k_ref_method")
            training["episodes_trained"] = agent.episode_count
            if gains is None:
                used, gains = "optimizer", base_gains["optimizer"]
                training["note"] = "RL found no stable gains; fell back to the optimizer."
        else:
            gains = base_gains["optimizer"]

        resp, m, c = run_trial(self.plant, gains, self.spec, settings)
        baselines, responses = {}, {used.upper(): resp}
        base_metrics = {}
        for name, g in base_gains.items():
            r, bm, bc = run_trial(self.plant, g, self.spec, settings)
            base_metrics[name] = bm
            baselines[name] = {"gains": [float(v) for v in g], "metrics": bm.to_dict(), "cost": float(bc)}
            responses[name.upper()] = r
        result = TuningResult(
            kp=float(gains[0]), ki=float(gains[1]), kd=float(gains[2]),
            derivative_filter_n=float(settings.derivative_filter_n), method=used,
            plant=self.plant.tf_string, spec=self.spec.to_dict(), metrics=m.to_dict(), cost=float(c),
            meets_spec=meets_spec(m, self.spec), baselines=baselines,
            beats_zn=beats(m, base_metrics["zn"]) if "zn" in base_metrics else None,
            beats_simc=beats(m, base_metrics["simc"]) if "simc" in base_metrics else None,
            training=training,
            identification=None if self.fit_report is None else
            {"model": self.fit_report.model, "params": self.fit_report.params, "r2": self.fit_report.r2},
        )
        if verbose:
            print(result.summary())
            table = {used.upper(): m, **{k.upper(): v for k, v in base_metrics.items()}}
            print("\n" + compare(table))
        if out_dir is not None:
            out = Path(out_dir)
            result.to_json(out / "tuning_result.json")
            try:
                from .plotting import plot_step_responses

                plot_step_responses({k: v for k, v in responses.items() if v is not None},
                                    out / "step_response.png", title="Closed-loop step response")
            except ImportError:
                pass
        return result
