#!/usr/bin/env python
"""counterfactual_kd command-line interface.

Core (numpy/stdlib only, no training, no openbackdoor):
    python run.py verdict --preds results/cells/<cell>.preds.json

Reproduction (needs `pip install -e .[repro]`; teachers need `[teacher]`):
    python run.py dashboard --runs-base ./results/runs --out ./results/dashboard/dashboard_rope.csv
    python run.py tables    --csv ./results/dashboard/dashboard_rope.csv
    python run.py repro     --config configs/example_verdict.yaml

Only the verdict path is imported at module scope; repro/dashboard imports are
deferred so a core-only install never needs torch/openbackdoor.
"""
import argparse
import json
import sys


def _to_hits(d):
    """Return (obs_hits, null_hits) as 0/1 vectors from a cell preds file.

    Accepts either pre-binarized {"obs_hits","null_hits"} or raw predicted labels
    {"preds_obs","preds_null","target_label"} (hit = predicted == target).
    """
    if "obs_hits" in d and "null_hits" in d:
        return list(d["obs_hits"]), list(d["null_hits"])
    tgt = d.get("target_label", 1)
    obs = [1 if p == tgt else 0 for p in d["preds_obs"]]
    nul = [1 if p == tgt else 0 for p in d["preds_null"]]
    return obs, nul


def cmd_verdict(args):
    from counterfactual_kd.metrics.cohens_h import rope_verdict

    d = json.load(open(args.preds))
    obs, nul = _to_hits(d)
    r = rope_verdict(obs, nul, R=args.bootstrap, seed=args.seed, rope=args.rope)
    cell = d.get("cell", {})
    k = r["p_trig"] - r["c_trig"]
    if args.json:
        print(json.dumps({**({"cell": cell} if cell else {}), **r, "k": k}, indent=2))
        return
    if cell:
        print("cell:        " + " ".join(f"{x}={cell[x]}" for x in cell))
    print(f"n:           {r['n']}")
    print(f"P_trig:      {r['p_trig']:.3f}   (poisoned-teacher student, triggered ASR)")
    print(f"C_trig:      {r['c_trig']:.3f}   (clean-teacher student, lexical baseline)")
    print(f"k = P-C:     {k:+.3f}   (matched, teacher-attributable difference)")
    print(f"Cohen's h:   {r['h']:+.3f}")
    print(f"95% CI(h):   [{r['hdi_lo']:+.3f}, {r['hdi_hi']:+.3f}]")
    print(f"P_transfer:  {r['p_transfer']:.3f}   (Pr_boot[|h| > {args.rope}])")
    print(f"VERDICT:     {r['verdict']}")
    exp = d.get("expected", {})
    if exp.get("verdict") and exp["verdict"] != r["verdict"]:
        print(f"  WARNING: expected verdict {exp['verdict']!r} != computed {r['verdict']!r}", file=sys.stderr)


def cmd_dashboard(args):
    from counterfactual_kd.dashboard import build_rope_dashboard as b
    raise SystemExit(b.build(out=args.out, runs_base=args.runs_base))


def cmd_tables(args):
    from counterfactual_kd.dashboard import emit_v17_tables as t
    t.build(csv=args.csv, out=args.out)


def cmd_repro(args):
    try:
        from counterfactual_kd.cli import main as cli_main  # pulls torch/transformers via the pipeline
    except ImportError as e:
        sys.exit(
            f"repro pipeline unavailable ({e}).\n"
            "Install the reproduction extras:  pip install -e .[repro]\n"
            "To also retrain poisoned teachers:  pip install -e .[repro,teacher]"
        )
    argv = ["run", args.config]
    if args.run_id:
        argv += ["--run-id", args.run_id]
    if args.stages:
        argv += ["--stages", *args.stages]
    raise SystemExit(cli_main(argv))


def main(argv=None):
    p = argparse.ArgumentParser(prog="counterfactual_kd", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verdict", help="ROPE verdict from a per-cell preds file (core; no training)")
    v.add_argument("--preds", required=True, help="path to a *.preds.json cell file")
    v.add_argument("--bootstrap", type=int, default=5000, help="paired bootstrap resamples")
    v.add_argument("--rope", type=float, default=0.2, help="ROPE half-width on Cohen's h")
    v.add_argument("--seed", type=int, default=42)
    v.add_argument("--json", action="store_true", help="emit JSON")
    v.set_defaults(func=cmd_verdict)

    d = sub.add_parser("dashboard", help="rebuild the ROPE dashboard CSV from your own runs")
    d.add_argument("--runs-base", default=".", help="base dir prefixed to the run globs")
    d.add_argument("--out", default=None, help="output CSV (default: bundled results/dashboard/)")
    d.set_defaults(func=cmd_dashboard)

    t = sub.add_parser("tables", help="emit section tables from a dashboard CSV")
    t.add_argument("--csv", default=None, help="dashboard CSV (default: bundled results/dashboard/)")
    t.add_argument("--out", default=None, help="write markdown here (default: stdout)")
    t.set_defaults(func=cmd_tables)

    r = sub.add_parser("repro", help="run the 4-condition matched measurement pipeline (needs [repro])")
    r.add_argument("--config", required=True)
    r.add_argument("--run-id", default=None)
    r.add_argument("--stages", nargs="+", default=None, choices=["stage0", "stage1", "stage2"])
    r.set_defaults(func=cmd_repro)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
