"""Evaluate a trained RL tuner against Ziegler-Nichols, SIMC and a numerical optimizer.

    python -m rl_pid_tuner.evaluate --model runs/<run>/best_model.zip --plant dc_motor

Fair-comparison protocol:
  * every method *designs* its gains on the nominal plant, using the same simulation settings and
    the same cost;
  * the gains are then *tested* on the nominal plant and on ``--n-perturbed`` plants whose
    parameters are randomly changed by up to ``--uncertainty`` (robustness to model error).
Results are reported honestly: lower cost is better, and the report says whether RL actually won.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .agent import RLAgent
from .baselines import design_baselines
from .env import PIDTuningEnv, plant_from_config
from .metrics import compare
from .plotting import plot_step_responses
from .spec import get_spec, make_sim_settings, meets_spec, run_trial

METHODS = ("zn", "simc", "optimizer", "rl")


def _resolve_config(args) -> dict:
    from .train import DEFAULTS, _deep_merge, load_config

    model_dir = Path(args.model).parent
    if args.config:
        cfg = load_config(args.config)
    elif (model_dir / "config.yaml").exists() and not args.plant:
        cfg = load_config(model_dir / "config.yaml")
    else:
        cfg = _deep_merge(DEFAULTS, {})
    if args.plant:
        cfg["plant"] = {"name": args.plant}
    if args.spec:
        cfg["spec"] = args.spec
    return cfg


def evaluate(model_path, cfg: dict, n_perturbed: int = 20, uncertainty: float = 0.2, n_design: int = 5,
             seed: int = 0, include_optimizer: bool = True, out_dir=None) -> dict:
    """Run the comparison and return the report dict (also saved if ``out_dir`` is given)."""
    rng = np.random.default_rng(seed)
    plant, sampler = plant_from_config(cfg.get("plant"))
    design_plants = [plant] if plant is not None else [sampler(rng) for _ in range(n_design)]
    n_test = n_perturbed if plant is not None else max(n_perturbed // 4, 5)
    spec = get_spec(cfg.get("spec", "balanced"))
    env_opts = {**cfg.get("env", {}), "init_noise": 0.0}
    agent = RLAgent.load(model_path, algorithm=cfg.get("algorithm"))

    per_method: dict[str, list[dict]] = {m: [] for m in METHODS}
    first_nominal: dict = {}
    first_responses: dict = {}
    gains_first: dict = {}
    for di, p in enumerate(design_plants):
        settings = make_sim_settings(p, spec, window_factor=env_opts.get("window_factor", 20.0))
        gains = design_baselines(p, spec, settings, include_optimizer=include_optimizer)
        env = PIDTuningEnv(plant=p, spec=spec, uncertainty=cfg.get("uncertainty", 0.0), **env_opts)
        env.reset(seed=seed + di)
        rl_gains, _, _ = agent.tune(env, n_episodes=1)
        if rl_gains is not None:
            gains["rl"] = rl_gains
        tests = [p] + [p.perturbed(uncertainty, rng) for _ in range(n_test)]
        for method, g in gains.items():
            costs, stable, meets = [], [], []
            for ti, tp in enumerate(tests):
                resp, m, c = run_trial(tp, g, spec, settings)
                costs.append(c)
                stable.append(m.stable)
                meets.append(meets_spec(m, spec))
                if ti == 0:
                    nominal = (m, c, resp)
            per_method[method].append({
                "gains": [float(v) for v in g], "nominal_cost": float(nominal[1]),
                "robust_mean_cost": float(np.mean(costs)), "robust_worst_cost": float(np.max(costs)),
                "stable_fraction": float(np.mean(stable)), "meets_spec_fraction": float(np.mean(meets)),
                "nominal_metrics": nominal[0].to_dict(),
            })
            if di == 0:
                first_nominal[method], first_responses[method], gains_first[method] = nominal[0], nominal[2], g

    summary = {}
    for method, rows in per_method.items():
        if rows:
            summary[method] = {k: float(np.mean([r[k] for r in rows])) for k in
                               ("nominal_cost", "robust_mean_cost", "robust_worst_cost", "stable_fraction", "meets_spec_fraction")}
            summary[method]["gains_first_plant"] = [float(v) for v in gains_first[method]]
    verdict = {}
    if "rl" in summary:
        for base in ("zn", "simc", "optimizer"):
            if base in summary:
                verdict[f"rl_vs_{base}"] = {
                    "rl_better_nominal": bool(summary["rl"]["nominal_cost"] < summary[base]["nominal_cost"]),
                    "rl_better_robust": bool(summary["rl"]["robust_mean_cost"] < summary[base]["robust_mean_cost"]),
                }
    report = {"plant": cfg.get("plant"), "spec": spec.to_dict(), "uncertainty_tested": uncertainty,
              "n_design_plants": len(design_plants), "n_test_plants_each": n_test + 1,
              "lower_cost_is_better": True, "summary": summary, "verdict": verdict}

    print("\nNominal-plant metrics (first design plant):")
    print(compare(first_nominal))
    print("\nCost over nominal + perturbed plants (lower is better):")
    print(f"{'method':<12}{'nominal':>10}{'robust mean':>13}{'worst':>10}{'stable %':>10}{'meets spec %':>14}")
    for method, s in summary.items():
        print(f"{method:<12}{s['nominal_cost']:>10.3f}{s['robust_mean_cost']:>13.3f}{s['robust_worst_cost']:>10.3f}"
              f"{100 * s['stable_fraction']:>10.0f}{100 * s['meets_spec_fraction']:>14.0f}")

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        try:
            plot_step_responses({m.upper(): r for m, r in first_responses.items()},
                                out / "rl_vs_baselines.png", title="RL vs baselines (nominal plant)")
        except ImportError:
            print("[info] matplotlib not installed -> plot skipped (pip install matplotlib).")
        print(f"\nSaved: {out / 'eval_report.json'}")
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Compare a trained RL tuner with ZN / SIMC / optimizer.")
    ap.add_argument("--model", required=True, help="path to best_model.zip")
    ap.add_argument("--plant", help="dc_motor | first_order | second_order (default: from run config)")
    ap.add_argument("--spec", help="preset name (default: from run config)")
    ap.add_argument("--config", help="YAML config (overrides the run's config.yaml)")
    ap.add_argument("--out-dir", help="where to write eval_report.json / png (default: model folder)")
    ap.add_argument("--n-perturbed", type=int, default=20)
    ap.add_argument("--uncertainty", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-optimizer", action="store_true", help="skip the (slowish) Nelder-Mead baseline")
    args = ap.parse_args(argv)
    cfg = _resolve_config(args)
    evaluate(args.model, cfg, n_perturbed=args.n_perturbed, uncertainty=args.uncertainty, seed=args.seed,
             include_optimizer=not args.no_optimizer, out_dir=args.out_dir or Path(args.model).parent)


if __name__ == "__main__":
    main()
