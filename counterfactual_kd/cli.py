"""
Command-line interface for Counterfactual-KD's config-driven runner.

Sub-commands:

* ``run <config.yaml>``   — execute the pipeline end-to-end.
* ``validate <config.yaml>`` — parse + validate a config, don't execute.
* ``show <config.yaml>``  — print the matrix expansion (conditions) so
                             you can eyeball what a config will run.

Keeping these at parity with the Experiment API means users can script
the same operations from Python by constructing ``ExperimentConfig`` and
calling ``Experiment(cfg).run(...)`` directly.
"""

from __future__ import annotations

import argparse
import sys

from counterfactual_kd.config import load_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m counterfactual_kd",
                                description="Counterfactual-KD config-driven pipeline.")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run a config (stage0 → stage1 → stage2)")
    run.add_argument("config", help="Path to YAML config.")
    run.add_argument("--stages", nargs="+", default=None,
                     choices=["stage0", "stage1", "stage2"],
                     help="Subset of stages to run. Default: all three.")
    run.add_argument("--run-id", default=None,
                     help="Custom run id. Default: <date>_<config-name>.")
    run.add_argument("--resume", action="store_true",
                     help="Allow writing into an existing run directory.")
    run.add_argument("--no-skip", action="store_true",
                     help="Redo every condition, even ones that have outputs on disk.")

    val = sub.add_parser("validate", help="Parse + validate a config (no execution)")
    val.add_argument("config")

    show = sub.add_parser("show", help="Expand a config to a matrix listing")
    show.add_argument("config")
    show.add_argument("--limit", type=int, default=10,
                      help="Max number of conditions to print (default: 10).")

    return p


def cmd_run(args) -> int:
    from counterfactual_kd.experiment import Experiment
    cfg = load_config(args.config)
    exp = Experiment(cfg, run_id=args.run_id)
    manifest = exp.run(
        stages=args.stages,
        skip_existing=not args.no_skip,
        resume=args.resume,
    )
    print(f"\nRun complete: {exp.run_dir}")
    print(f"Manifest: {exp.run_dir}/manifest.json")
    for stage_name, summary in manifest["stages"].items():
        print(f"  {stage_name:<7} {summary}")

    if not manifest.get("ok", True):
        n_failures = sum(s.get("failures", 0) for s in manifest["stages"].values())
        print(
            f"FAILED: {n_failures} condition(s) had errors — see stage output above",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_validate(args) -> int:
    from counterfactual_kd.registry import method_spec
    cfg = load_config(args.config)
    print(f"Config name: {cfg.name}")
    print(f"Conditions:  {cfg.total_conditions()}")
    print(f"Methods:     {cfg.methods}")

    # KD-family axes — only print if any method needs pairs.
    if any(method_spec(m).needs_pairs for m in cfg.methods):
        n_teachers = len(set(t for t, _ in cfg.pairs))
        print(f"KD axes:     {n_teachers} teachers × {len(cfg.pairs)} pairs × "
              f"{len(cfg.datasets)} datasets × {len(cfg.kd_types)} kd_types × "
              f"{len(cfg.seeds)} seeds × {len(cfg.attacks)} attacks")

    # FT-family axes — only print if any method skips Stage 0.
    if any(not method_spec(m).needs_pairs for m in cfg.methods):
        ft_methods = [m for m in cfg.methods if not method_spec(m).needs_pairs]
        print(f"FT axes:     {len(ft_methods)} methods ({ft_methods}) × "
              f"{len(cfg.base_models)} base_models × "
              f"{len(cfg.datasets)} datasets × "
              f"{len(cfg.seeds)} seeds × {len(cfg.attacks)} attacks")

    print("OK.")
    return 0


def cmd_show(args) -> int:
    cfg = load_config(args.config)
    conds = cfg.conditions()
    print(f"{cfg.name}: {len(conds)} conditions")
    for i, c in enumerate(conds[: args.limit]):
        print(f"  [{i:3d}] {c}")
    if len(conds) > args.limit:
        print(f"  ... ({len(conds) - args.limit} more)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dispatch = {"run": cmd_run, "validate": cmd_validate, "show": cmd_show}
    code = dispatch[args.cmd](args)
    if code:
        # Belt-and-suspenders: some entry points invoke ``main()`` for its
        # side effects without checking/propagating the return value (e.g.
        # ``python -m counterfactual_kd`` just calls ``main()``). Raising
        # here guarantees a failing run still exits the process non-zero
        # instead of silently reporting success. No-op on the success path.
        sys.exit(code)
    return code


if __name__ == "__main__":
    sys.exit(main())
