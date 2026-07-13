#!/usr/bin/env python3
"""Emit dashboard tables (paper markdown) from a ROPE dashboard CSV.

Importable: nothing runs at import time. Call ``build(csv, out)`` or use the CLI
(``counterfactual_kd tables`` / ``python -m counterfactual_kd.dashboard.emit_v17_tables``).
Verdict = h + ROPE only; k / McNemar p / P_transfer are descriptive.
"""
from __future__ import annotations

import csv
import os
import statistics

FAM = {"bert": "BERT", "gpt2": "GPT-2", "Qwen3": "Qwen3"}
DSET = {"sst2": "SST-2", "agnews": "AGNews"}
ATK = {"atba": "ATBA", "badnets": "BadNets", "addsent": "AddSent"}
FAM_ORD = {"bert": 0, "gpt2": 1, "Qwen3": 2}

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CSV = os.path.join(_PKG_ROOT, "results", "dashboard", "dashboard_rope.csv")

HEADER = ("| Condition | $P_{\\text{trig}}$ | $C_{\\text{trig}}$ | $k$ | "
          "Cohen $h$ | 95% HDI($h$) | McNemar $p$ | $P_{\\text{tr}}$ | Verdict |")
SEP = "|---|---:|---:|---:|---:|---|---:|---:|---|"


def _f(r, k):
    return float(r[k])


def _row_line(r):
    cond = f"{ATK.get(r['attack'], r['attack'])} / {FAM.get(r['family'], r['family'])} / {DSET.get(r['dataset'], r['dataset'])} / {r['kd_type']}"
    hdi = f"[{_f(r,'hdi_lo'):+.3f}, {_f(r,'hdi_hi'):+.3f}]"
    p = _f(r, "mcnemar_p")
    pstr = "<0.001" if p < 0.001 else f"{p:.3f}"
    return (f"| {cond} | {_f(r,'P_trig'):.3f} | {_f(r,'C_trig'):.3f} | "
            f"{_f(r,'k'):+.3f} | {_f(r,'h'):+.3f} | {hdi} | {pstr} | "
            f"{_f(r,'P_transfer'):.2f} | {r['verdict']} |")


def _block(title, sub):
    sub = sorted(sub, key=lambda r: (r["attack"], FAM_ORD.get(r["family"], 9), r["dataset"], r["kd_type"]))
    return [f"#### {title}", "", HEADER, SEP] + [_row_line(r) for r in sub] + [""]


def _summ(label, sub):
    from collections import Counter
    ks = [_f(r, "k") for r in sub]
    hs = [abs(_f(r, "h")) for r in sub]
    vc = Counter(r["verdict"] for r in sub)
    vstr = ", ".join(f"{v}×{c}" for v, c in vc.most_common())
    return (f"- **{label}** (n={len(sub)}): mean $k$={statistics.mean(ks):+.3f}, "
            f"mean $|h|$={statistics.mean(hs):.3f} — {vstr}")


def build(csv=None, out=None):
    """Render the section tables from a dashboard CSV. Returns the markdown string;
    writes it to ``out`` if given, else prints it."""
    import csv as _csvmod
    csv_path = csv or DEFAULT_CSV
    rows = list(_csvmod.DictReader(open(csv_path)))

    L = ["# Dashboard Tables (auto-generated from dashboard_rope.csv)\n",
         "> verdict = h + ROPE only; k / p / P_tr are descriptive. ROPE=[-0.2,+0.2], B=5000.\n",
         "## ATBA re-measurement\n"]
    order = [("atba_paperhp", "BERT paper-hp (full corpus)"),
             ("atba_fullcorpus", "GPT-2 full corpus (lr=5e-5, warmup=0)"),
             ("atba_subset3054", "GPT-2 paper-match (3054 subset)")]
    for corp, title in order:
        sub = [r for r in rows if r["group"] == "atba" and r["corpus"] == corp]
        if sub:
            L += _block(title, sub)
    atba = [r for r in rows if r["group"] == "atba"]
    if atba:
        L.append("**ATBA summary**")
        for corp, title in order:
            s = [r for r in atba if r["corpus"] == corp]
            if s:
                L.append(_summ(title, s))
        L.append("")

    L.append("## Corpus-poisoning paired ablation\n")
    for corp, title in [("clean", "Clean corpus"), ("poison10", "10% poisoned corpus")]:
        sub = [r for r in rows if r["group"] == "ablation" and r["corpus"] == corp]
        if sub:
            L += _block(title, sub)
    cl = [r for r in rows if r["group"] == "ablation" and r["corpus"] == "clean"]
    po = [r for r in rows if r["group"] == "ablation" and r["corpus"] == "poison10"]
    if cl and po:
        L += ["**Paired summary**", _summ("Clean", cl), _summ("10% poisoned", po)]
        gap = min(_f(r, "k") for r in po) - max(abs(_f(r, "k")) for r in cl)
        L.append(f"- **Separation**: poison min $k$={min(_f(r,'k') for r in po):.3f} "
                 f"> clean max $|k|$={max(abs(_f(r,'k')) for r in cl):.3f} → gap={gap:.3f}")

    text = "\n".join(L)
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        open(out, "w").write(text)
        print(f"[done] -> {out}  ({len(L)} lines)")
    else:
        print(text)
    return text


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Emit dashboard tables from a ROPE dashboard CSV.")
    ap.add_argument("--csv", default=None, help=f"dashboard CSV (default: {DEFAULT_CSV})")
    ap.add_argument("--out", default=None, help="write markdown here (default: stdout)")
    a = ap.parse_args(argv)
    build(csv=a.csv, out=a.out)


if __name__ == "__main__":
    main()
