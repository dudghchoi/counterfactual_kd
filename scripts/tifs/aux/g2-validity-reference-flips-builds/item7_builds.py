#!/usr/bin/env python3
"""A11 item 7 (빌드별 요인 설계): the corpus x teacher factorial per teacher
build. Post hoc auxiliary table (docs/tifs_addendum.md A11); it does not change
any primary verdict.

For each teacher build b (t42 = the seed-42 grid and battery; t43, t44 = the
A/A runs), BadNets trigger, logit KD, corpus seed 42, student seed 42:
  P1, C1 = the build's poisoned / clean-teacher students on the BadNets 10%
           contaminated corpus (t42: tifs_g_<fam>_badnets10_s42; A/A:
           tifs_aa_<fam>_badnets10_t<b>)
  P0, C0 = the same build's students on the clean corpus (t42: the battery
           tifs_b_<fam>_<ds>; A/A: tifs_aa_<fam>_clean_t<b>)
  R1, R0 = the CE-only students (no teacher) on the contaminated / clean
           corpus, from the t42 grid / battery; shared by every build because
           they do not depend on the teacher.
Tango-based contrasts (frozen paired_contrast, delta in {0.02, 0.05, 0.10}):
P1-C1, P1-R1, C1-R1, P0-C0, P0-R0, C0-R0, R1-R0. The teacher x corpus
interaction (P1-C1)-(P0-C0) and the frozen companions (C1-R1, P1-R1, R1-R0)
come from the frozen analyze.interaction_table (joint input bootstrap of
analysis.tifs.joint_bootstrap, R = 5000, seed 0, as analyze.main runs it),
called on the build's cells relabelled as one grid block.

Outputs: builds_factorial.csv (one row per family x dataset x build) and
builds_factorial_spread.csv (per family x dataset: the spread across builds).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # no __pycache__ inside the repository

import statistics  # noqa: E402

from g2common import A, OFFICIAL, TOLS, VKEYS, cli, state, tango, write  # noqa: E402

CONTRASTS = (("P1-C1", "P1", "C1"), ("P1-R1", "P1", "R1"), ("C1-R1", "C1", "R1"),
             ("P0-C0", "P0", "C0"), ("P0-R0", "P0", "R0"), ("C0-R0", "C0", "R0"), ("R1-R0", "R1", "R0"))
BOOT = ("teacher_x_corpus", "suppression_C1_R1", "amplification_P1_R1", "corpus_effect_R1_R0")
R_BOOT = 5000


def find(cells, **kw):
    out = [c for c in cells if all(c.get(k) == v for k, v in kw.items())]
    if len(out) > 1:
        raise ValueError(f"ambiguous {kw}: {[c['run'] for c in out]}")
    return out[0] if out else None


def main():
    cli(__doc__.split("\n\n")[0])
    st = state()
    cells = [c for c in st.cells if not c["single"]]
    rows, spread = [], []
    for fam, ds in (("bert", "sst2"), ("bert", "agnews"), ("qwen3", "sst2"), ("qwen3", "agnews")):
        r1 = st.refs.get(("G", fam, ds, "badnets", "badnets10", 42))
        r0 = st.refs.get(("G", fam, ds, "badnets", "clean", 42))
        if r1 is None or r0 is None:
            continue
        common = dict(fam=fam, dataset=ds, attack="badnets", kd="logit", student_seed=42)
        builds = {42: (find(cells, block="G", corpus="badnets10", teacher_seed=42, **common),
                       find(cells, block="G", corpus="clean", teacher_seed=42, **common))}
        for t in (43, 44):
            c1 = find(cells, block="AA", corpus="badnets10", teacher_seed=t, **common)
            c0 = find(cells, block="AA", corpus="clean", teacher_seed=t, **common)
            if c1 is not None:
                builds[t] = (c1, c0)
        per_build = {}
        for t, (c1, c0) in sorted(builds.items()):
            if c1 is None:
                continue
            vec = {"P1": c1["P"], "C1": c1["C"], "R1": r1["P"]}
            if c0 is not None:
                vec.update(P0=c0["P"], C0=c0["C"], R0=r0["P"])
            row = {"fam": fam, "dataset": ds, "attack": "badnets", "kd": "logit", "build": f"t{t}",
                   "teacher_seed": t, "corpus_seed": 42, "student_seed": 42, "precision": OFFICIAL,
                   "run_contaminated": c1["run"], "run_clean": c0["run"] if c0 else None,
                   "run_R1": r1["run"], "run_R0": r0["run"], "n": len(c1["P"]),
                   **{f"rate_{k}": sum(v) / len(v) for k, v in vec.items()},
                   "teacher_asr": c1["teacher_asr"], "teacher_asr_null": c1["teacher_asr_null"],
                   "teacher_ca": c1["teacher_ca"], "teacher_ca_null": c1["teacher_ca_null"],
                   "teacher_qualified": A.qualified(c1)}
            for name, a, b in CONTRASTS:
                if a in vec and b in vec:
                    flat, _ = tango(vec[a], vec[b])
                    row.update({f"{name}": flat["k"], f"{name}_lo": flat["lo"], f"{name}_hi": flat["hi"],
                                **{f"{name}_{v}": flat[v] for v in VKEYS}})
            if c0 is not None:
                pseudo = [dict(c1, block="G"), dict(c0, block="G"), r1, r0]
                inter = A.interaction_table(pseudo, R_BOOT)
                if len(inter) != 1:
                    raise ValueError(f"interaction_table returned {len(inter)} rows for {fam}/{ds}/t{t}")
                it = inter[0]
                for k in BOOT:
                    row.update({f"boot_{k}": it[k], f"boot_{k}_lo": it[f"{k}_lo"], f"boot_{k}_hi": it[f"{k}_hi"]})
                # frozen factorial_table on the same cells (0.05 verdicts) as a cross-check
                fac = {(f["corpus"]): f for f in A.factorial_table(pseudo)}
                for corpus, tag in (("badnets10", "1"), ("clean", "0")):
                    f = fac.get(corpus)
                    if f is not None:
                        row[f"factorial_check_P{tag}-R{tag}"] = f["P-R"]
                        row[f"factorial_check_C{tag}-R{tag}"] = f["C-R"]
            row["interaction_method"] = ("frozen analyze.interaction_table (joint bootstrap, R = 5000, seed 0)"
                                         if c0 is not None else "no clean-corpus run of this build")
            rows.append(row)
            per_build[t] = row
        if len(per_build) >= 2:
            sp = {"fam": fam, "dataset": ds, "builds": ",".join(f"t{t}" for t in sorted(per_build)),
                  "n_builds": len(per_build)}
            for name in [c[0] for c in CONTRASTS] + [f"boot_{k}" for k in BOOT]:
                vals = [per_build[t].get(name) for t in sorted(per_build)]
                if any(v is None for v in vals):
                    continue
                sp.update({f"{name}@t{t}": per_build[t][name] for t in sorted(per_build)})
                sp.update({f"{name}_min": min(vals), f"{name}_max": max(vals), f"{name}_range": max(vals) - min(vals),
                           f"{name}_sd": statistics.stdev(vals)})
                if not name.startswith("boot_"):
                    for v in VKEYS:
                        vs = [per_build[t][f"{name}_{v}"] for t in sorted(per_build)]
                        sp[f"{name}_{v}_by_build"] = "/".join(vs)
                        sp[f"{name}_{v}_agree"] = len(set(vs)) == 1
            spread.append(sp)
    write(rows, "builds_factorial.csv")
    write(spread, "builds_factorial_spread.csv")
    for r in rows:
        print(r["fam"], r["dataset"], r["build"], {k: round(r[k], 4) for k in ("rate_P1", "rate_C1", "rate_R1", "P1-C1", "P1-R1", "C1-R1") if k in r},
              {k: round(r[k], 4) for k in ("rate_P0", "rate_C0", "rate_R0", "P0-C0", "P0-R0", "C0-R0", "boot_teacher_x_corpus", "boot_teacher_x_corpus_lo", "boot_teacher_x_corpus_hi") if k in r})


if __name__ == "__main__":
    main()
