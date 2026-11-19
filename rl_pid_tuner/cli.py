"""Command line interface:  ``rl-pid-tuner tune | train | evaluate``  (also ``python -m rl_pid_tuner.cli``)."""
from __future__ import annotations

import argparse
import sys

from .plant import Plant


def _coeffs(text: str) -> list[float]:
    return [float(v) for v in text.replace(",", " ").split()]


def _plant_from_args(args) -> Plant:
    if args.tf:
        return Plant.from_tf(_coeffs(args.tf[0]), _coeffs(args.tf[1]))
    from .env import plant_from_config

    plant, _ = plant_from_config({"name": args.plant or "dc_motor_position"})
    return plant


def _tune(argv: list[str]) -> None:
    from .tuner import PIDAutoTuner

    ap = argparse.ArgumentParser(prog="rl-pid-tuner tune", description="Tune PID gains for a linear SISO plant.")
    ap.add_argument("--tf", nargs=2, metavar=("NUM", "DEN"),
                    help='transfer function coefficients, e.g. --tf "0.01" "0.005 0.06 0.1001 0"')
    ap.add_argument("--plant", help="built-in plant: dc_motor | first_order | second_order (default dc_motor)")
    ap.add_argument("--spec", default="balanced", help="fast | balanced | no_overshoot | robust")
    ap.add_argument("--model", help="trained agent (best_model.zip) for method=rl")
    ap.add_argument("--train-steps", type=int, default=0, help="train a quick agent on this plant instead")
    ap.add_argument("--method", choices=["rl", "optimizer"], default=None,
                    help="default: rl if --model/--train-steps is given, else optimizer")
    ap.add_argument("--uncertainty", type=float, default=0.0)
    ap.add_argument("--out-dir", help="write tuning_result.json and step_response.png here")
    args = ap.parse_args(argv)
    method = args.method or ("rl" if (args.model or args.train_steps) else "optimizer")
    tuner = PIDAutoTuner(_plant_from_args(args), spec=args.spec, uncertainty=args.uncertainty)
    tuner.tune(model=args.model, train_steps=args.train_steps, method=method, out_dir=args.out_dir)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmds = {"tune": _tune}
    if argv and argv[0] == "train":
        from .train import main as train_main

        return train_main(argv[1:])
    if argv and argv[0] == "evaluate":
        from .evaluate import main as eval_main

        return eval_main(argv[1:])
    if argv and argv[0] in cmds:
        return cmds[argv[0]](argv[1:])
    ap = argparse.ArgumentParser(prog="rl-pid-tuner", description="RL-based PID auto-tuner (SISO, linear, simulated).")
    ap.add_argument("command", choices=["tune", "train", "evaluate"], help="what to do (add --help after it for options)")
    ap.parse_args(argv or ["--help"])


if __name__ == "__main__":
    main()
