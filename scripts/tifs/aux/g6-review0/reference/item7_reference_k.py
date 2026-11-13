#!/usr/bin/env python3
"""A14 item 7: per-reference k on every BERT battery trigger (SST-2 and AG News, logit KD).

A14 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

P = the battery's seed-42 P student of each attack (poisoned teacher t42), CPU fp32. References:
  t42 twin    the trainer-matched clean student of the cell (clean-base, or clean-sos / clean-ep
              for SOS / EP; the registered reference of the frozen k)
  t42 base    the battery's clean-base student (base trainer), where the twin is not clean-base
  t<k> C'     the clean-base logit C' student of every clean reference build scored by
              score.py refs: the A/A builds t43, t44 (tifs_aa_bert_clean_t43/_t44) and, once
              they finish, the A13 G2 builds t45, t46 (tifs_x13_bert_ref_t45/_t46)
all CPU fp32 on the same triggered inputs (cross_score.triggered_loader; input sha256 checked).
Per (trigger, reference): Tango k = P - C_ref and the frozen transported interval (SE^2 =
SE_Tango^2 + s2_pool, the family's logit pool and df), verdicts at delta in {0.02, 0.05, 0.10}.

A cell is 'reference-dependent' at delta when its transported verdict is not the same for every
reference of the set {t42 twin} + {t<k> C'} (the A3 rule; 'EQUIVALENT changes with the
reference' is flagged separately); the frozen primary verdict is shown beside it and is not
changed. An A3.3-style interval per trigger adds the between-build variance of the base-trainer
references' hit rates (t42 base or twin, t43, t44[, t45, t46]; df J - 1) to the frozen primary
SE (k ± t sqrt(SE^2 + s2_ref), Satterthwaite df), as A3.3 does for the ATBA phrase.

A reference build is used only if its clean teacher and its C' student pass the functional
floor (Stage 2 clean accuracy > majority share + 0.10; A7/A8/A13 G2). SST-2 also gets
seed-pooled rows P_s - C'_s(t43) over seeds 42-44 (the s43/s44 C' are the A2 replicate's
clean-base students of the same t43 clean teacher).

To add the A13 G2 references later: `run_all.sh` (score.py refs picks up every finished
reference run, this script reads every hits/ref_*.json).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import glob  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
from collections import defaultdict  # noqa: E402

import common as K  # noqa: E402

_FILL = {}
REF_FILE = re.compile(r"^ref_(?P<run>tifs_[a-z0-9_]+)\.(?P<ds>sst2|agnews)\.json$")


def stage2_function(run, ds):
    """(teacher clean acc, student clean acc, source) from the run's Stage 2 results (logit)."""
    for p in sorted(glob.glob(os.path.join(K.RUNS, run, "artifacts", "results", f"*_{ds}_*.json"))):
        if p.endswith(".preds.json"):
            continue
        r = K.load_json(p)
        if r.get("kd_type") == "logit" and r.get("dataset") == ds:
            return r.get("teacher_ca_null"), r.get("ca_clean"), os.path.relpath(p, K.ROOT)
    return None, None, None


def load_references(out, ds):
    """{label: {run, teacher_seed, key, hits: {trigger: vec}, included, ...}} for ``ds``."""
    refs = {}
    floor = K.AD.MAJORITY[ds] + K.AD.FUNCTION_MARGIN
    for p in sorted(glob.glob(os.path.join(K.hits_dir(out), "ref_*.json"))):
        m = REF_FILE.match(os.path.basename(p))
        if m is None or m["ds"] != ds:
            continue
        X = K.load_json(p)
        (key, meta), = [(k, v) for k, v in X["students"].items()
                        if v["tag"] == "clean-base" and v["role"] == "clean"]
        missing = [a for a in K.ATTACKS if key not in X["hits"].get(a, {})]
        t_ca, s_ca, src = stage2_function(m["run"], ds)
        if missing:
            status = f"incomplete scoring (missing {','.join(missing)})"
        elif t_ca is None or s_ca is None:
            status = "function unchecked (no Stage 2 result yet)"
        elif t_ca <= floor or s_ca <= floor:
            status = "excluded: below the functional floor"
        else:
            status = "included"
        refs[f"t{meta['teacher_seed']}"] = {
            "run": m["run"], "teacher_seed": meta["teacher_seed"], "key": key,
            "model_sha256": meta["model_sha256"], "file": os.path.relpath(p, K.ROOT),
            "hits": {a: X["hits"][a][key] for a in K.ATTACKS if a not in missing},
            "inputs": X["inputs"], "teacher_ca": t_ca, "student_ca": s_ca, "function_source": src,
            "floor": floor, "status": status, "included": status in ("included",)}
    return dict(sorted(refs.items(), key=lambda kv: kv[1]["teacher_seed"]))


def battery_hits(out, ds, kd, atk, tag, role, seed=42):
    """CPU fp32 hits of a battery student on ``atk``'s inputs: cross_hits, else hits/fill."""
    X = K.load_cross(ds)
    key = f"{kd}/{tag}/{role}/s{seed}"
    if key in X["hits"].get(atk, {}):
        return X["hits"][atk][key], f"cross_hits_bert_{ds}.json", X["inputs"][atk]["sha256"]
    fp = os.path.join(K.hits_dir(out), f"fill_bert_{ds}.json")
    if os.path.exists(fp):
        if fp not in _FILL:
            _FILL[fp] = K.load_json(fp)
        F = _FILL[fp]
        run = f"tifs_b_bert_{ds}" + ("" if seed == 42 else f"_s{seed}")
        k2 = f"{run}|{key}"
        if k2 in F["hits"].get(atk, {}):
            return F["hits"][atk][k2], f"hits/fill_bert_{ds}.json", F["inputs"][atk]["sha256"]
    return None, None, None


def main():
    out = K.out_dir()
    st = K.state()
    primary = {(r["dataset"], r["kd"], r["attack"]): r for r in st.primary if r["fam"] == "bert"}
    # What the frozen tables already show (so that only new reference dependence is called new):
    # primary.csv robust_verdict@0.05 (A/A between-teacher-seed variance; analyze.primary_table)
    # and the frozen A3.3 table for the ATBA phrase (reference_atba.csv reference_dependent).
    prim_csv = {(r["dataset"], r["kd"], r["attack"]): r for r in K.read_csv(os.path.join(K.REPORT, "primary.csv"))
                if r["fam"] == "bert"}
    a33_frozen = {(r["dataset"], r["kd"]): r["reference_dependent"]
                  for r in K.read_csv(os.path.join(K.ADD, "reference_atba.csv"))}
    rows, summ, runs_rows, sp_rows, checks = [], [], [], [], []
    for ds in ("sst2", "agnews"):
        refs = load_references(out, ds)
        for lab, rf in refs.items():
            runs_rows.append({"dataset": ds, "reference": lab, **{k: v for k, v in rf.items()
                                                                  if k not in ("hits", "inputs")}})
        _, s2p, dfp = K.pool(st, ds)
        for kd in ("logit", "feature"):
            for atk in K.ATTACKS:
                P, psrc, psha = battery_hits(out, ds, kd, atk, K.P_TAG[ds][atk], "poisoned")
                if P is None:
                    continue
                twin, tsrc, tsha = battery_hits(out, ds, kd, atk, K.C_TAG[atk], "clean")
                cand = [("t42 twin", 42, f"{kd}/{K.C_TAG[atk]}", twin, tsrc, tsha, True)]
                if K.C_TAG[atk] != "clean-base" or kd != "logit":
                    base, bsrc, bsha = battery_hits(out, ds, "logit", atk, "clean-base", "clean")
                    if base is not None:
                        cand.append(("t42 base", 42, "logit/clean-base", base, bsrc, bsha, True))
                for lab, rf in refs.items():
                    if atk in rf["hits"]:
                        cand.append((f"{lab} C'", rf["teacher_seed"], "logit/clean-base",
                                     rf["hits"][atk], rf["file"], rf["inputs"][atk]["sha256"],
                                     rf["included"]))
                pr = primary.get((ds, kd, atk), {})
                per = {}
                for lab, tseed, model, C, src, sha, inc in cand:
                    if C is None:
                        continue
                    checks.append({"check": f"inputs sha256 {ds}/{atk}: P vs {lab}",
                                   "ok": sha == psha, "detail": f"{psha[:12]} vs {sha[:12]}"})
                    pc, tr = K.transported(P, C, s2p, dfp)
                    row = {"dataset": ds, "kd": kd, "trigger": atk, "reference": lab,
                           "reference_teacher_seed": tseed, "reference_model": model,
                           "reference_included": inc,
                           "trainer_differs_from_twin": (model.endswith("clean-base")
                                                         and K.C_TAG[atk] != "clean-base"),
                           "kd_differs": not model.startswith(kd),
                           "P_source": psrc, "reference_source": src, "precision": "CPU fp32",
                           "n": pc["n"], "P_rate": pc["a"], "C_ref_rate": pc["b"],
                           "tango_k": pc["k"], "tango_lo": pc["lo"], "tango_hi": pc["hi"],
                           **{f"tango_verdict@{t:g}": pc[f"verdict@{t:g}"] for t in K.TOLS},
                           "transported_se": tr["se"], "transported_df": dfp,
                           "transported_lo": tr["lo"], "transported_hi": tr["hi"],
                           **{f"transported_verdict@{t:g}": tr[f"verdict@{t:g}"] for t in K.TOLS},
                           **{f"transported_p_up@{t:g}": tr[f"p_up@{t:g}"] for t in K.TOLS},
                           **{f"transported_p_pos@{t:g}": tr[f"p_pos@{t:g}"] for t in K.TOLS},
                           "primary_method": pr.get("method"),
                           **{f"primary_verdict@{t:g}": pr.get(f"verdict@{t:g}") for t in K.TOLS}}
                    rows.append(row)
                    per[lab] = row
                # Summary row of the cell.
                use = [r for lab, r in per.items() if lab == "t42 twin"
                       or (lab.endswith("C'") and r["reference_included"])]
                s = {"dataset": ds, "kd": kd, "trigger": atk,
                     "references": ",".join(r["reference"] for r in use), "n_references": len(use),
                     "primary_method": pr.get("method"), "primary_k": pr.get("k"),
                     "primary_lo": pr.get("lo"), "primary_hi": pr.get("hi")}
                for r in use:
                    tag = r["reference"].split()[0]
                    s[f"k_{tag}"] = r["tango_k"]
                    s[f"C_{tag}"] = r["C_ref_rate"]
                s["P_rate_s42"] = use[0]["P_rate"] if use else None
                s["k_min"] = min(r["tango_k"] for r in use)
                s["k_max"] = max(r["tango_k"] for r in use)
                for t in K.TOLS:
                    vs = [r[f"transported_verdict@{t:g}"] for r in use]
                    s[f"verdicts@{t:g}"] = "/".join(v[:3] for v in vs)
                    s[f"reference_dependent@{t:g}"] = len(set(vs)) > 1
                    s[f"equivalent_changes@{t:g}"] = ("EQUIVALENT" in vs) and len(set(vs)) > 1
                    s[f"primary_verdict@{t:g}"] = pr.get(f"verdict@{t:g}")
                    s[f"differs_from_primary@{t:g}"] = any(v != pr.get(f"verdict@{t:g}") for v in vs)
                s["reference_dependent"] = ",".join(f"{t:g}" for t in K.TOLS if s[f"reference_dependent@{t:g}"])
                s["equivalent_changes"] = ",".join(f"{t:g}" for t in K.TOLS if s[f"equivalent_changes@{t:g}"])
                # A3.3-style: base-trainer references' hit-rate variance added to the primary SE.
                base_rates = [per[l]["C_ref_rate"] for l in per
                              if (l == "t42 base") or (l == "t42 twin" and "t42 base" not in per)
                              or (l.endswith("C'") and per[l]["reference_included"])]
                s["J_base_trainer_refs"] = len(base_rates)
                if len(base_rates) >= 2 and pr.get("se"):
                    s2r, dfr = K.var(base_rates), len(base_rates) - 1
                    se2 = pr["se"] ** 2
                    df = (se2 + s2r) ** 2 / (se2 ** 2 / pr["df"] + s2r ** 2 / dfr)
                    rob = K.t_contrast(pr["k"], (se2 + s2r) ** 0.5, df, K.TOLS)
                    chg = [t for t in K.TOLS if rob[f"verdict@{t:g}"] != pr[f"verdict@{t:g}"]]
                    pc_ = prim_csv.get((ds, kd, atk), {})
                    known = set()
                    if atk == "atba" and (ds, kd) in a33_frozen:
                        known = {float(x) for x in a33_frozen[(ds, kd)].split(",") if x}
                        src_known = "reference_atba.csv, frozen A3.3"
                    elif pc_.get("robust_verdict@0.05") and pc_["robust_verdict@0.05"] != pc_.get("verdict@0.05"):
                        known = {0.05}
                        src_known = "primary.csv robust_verdict@0.05"
                    else:
                        src_known = ""
                    s.update(s_ref=s2r ** 0.5, df_ref=dfr, A33style_lo=rob["lo"], A33style_hi=rob["hi"],
                             A33style_df=df, **K.verdicts(rob, "A33style_"),
                             A33style_reference_dependent=",".join(f"{t:g}" for t in chg),
                             **{"primary_robust_verdict@0.05": pc_.get("robust_verdict@0.05")},
                             A33style_already_in_frozen=",".join(f"{t:g}" for t in chg if t in known),
                             A33style_frozen_source=src_known if any(t in known for t in chg) else "",
                             A33style_new_at=",".join(f"{t:g}" for t in chg if t not in known))
                elif pr.get("method") == "structural":
                    s["A33style_note"] = "structural null: primary is seed-42 Tango, no SE to add to"
                summ.append(s)
        # SST-2: seed-pooled P_s - C'_s(t43), seeds 42-44.
        if ds == "sst2" and "t43" in refs:
            fp = os.path.join(K.hits_dir(out), "fill_bert_sst2.json")
            F = K.load_json(fp) if os.path.exists(fp) else {"hits": {}}
            for atk in K.ATTACKS:
                Ps = {s: battery_hits(out, ds, "logit", atk, K.P_TAG[ds][atk], "poisoned", s)[0]
                      for s in (42, 43, 44)}
                Cp = {42: refs["t43"]["hits"].get(atk)}
                for s in (43, 44):
                    Cp[s] = F["hits"].get(atk, {}).get(f"tifs_rep_bert_atba_t43_s{s}|logit/clean-base/clean/s{s}")
                Ct = {s: battery_hits(out, ds, "logit", atk, K.C_TAG[atk], "clean", s)[0] for s in (42, 43, 44)}
                seeds = [s for s in (42, 43, 44) if Ps[s] is not None and Cp[s] is not None]
                if len(seeds) < 2:
                    continue
                for lab, C in (("t42 twin", Ct), ("t43 C' (A/A s42 + A2 replicate s43, s44)", Cp)):
                    ss = [s for s in seeds if C[s] is not None]
                    if len(ss) < 2:
                        continue
                    res = K.pooled_and_own({s: K.diff(Ps[s], C[s]) for s in ss}, s2p, dfp)
                    sp_rows.append({"dataset": ds, "kd": "logit", "trigger": atk, "reference": lab,
                                    "seeds": ",".join(map(str, ss)), "k": res["k"],
                                    **{f"k_s{s}": x for s, x in zip(ss, res["ks"])},
                                    "lo": res["pooled"]["lo"], "hi": res["pooled"]["hi"],
                                    **K.verdicts(res["pooled"]), "own_lo": res["own"]["lo"],
                                    "own_hi": res["own"]["hi"], **K.verdicts(res["own"], "own_"),
                                    **{f"primary_verdict@{t:g}": primary.get((ds, "logit", atk), {}).get(f"verdict@{t:g}")
                                       for t in K.TOLS}})
            # Reference dependence on the seed-pooled basis (t43 vs the t42 twin, same seeds).
            by_atk = {}
            for r in sp_rows:
                by_atk.setdefault(r["trigger"], []).append(r)
            for atk, rs in by_atk.items():
                for t in K.TOLS:
                    vs = {r[f"verdict@{t:g}"] for r in rs}
                    for r in rs:
                        r[f"reference_dependent@{t:g}"] = len(vs) > 1
                        r[f"equivalent_changes@{t:g}"] = "EQUIVALENT" in vs and len(vs) > 1
            # Determinism: the A/A t43 C' and the A2 replicate s42 C' share weights.
            k42 = "tifs_rep_bert_atba_t43_s42|logit/clean-base/clean/s42"
            for atk in K.ATTACKS:
                a, b = refs["t43"]["hits"].get(atk), F["hits"].get(atk, {}).get(k42)
                if a is not None and b is not None:
                    checks.append({"check": f"A/A t43 C' == A2 replicate s42 C' on {atk} (CPU fp32)",
                                   "ok": a == b, "detail": f"{sum(x != y for x, y in zip(a, b))} of {len(a)} differ"})
    K.write_csv(rows, os.path.join(out, "reference_k_all_triggers.csv"), """
        One row per (dataset, KD, trigger, reference): k = P(t42, seed 42) - C_ref on the
        trigger's inputs, CPU fp32. Reference 't42 twin' = the cell's trainer-matched clean
        student (registered reference), 't42 base' = the battery clean-base student where the
        twin is not it, 't<k> C'' = clean-base logit student of clean reference build t<k>
        (A/A t43/t44; A13 G2 t45/t46 when scored). tango_* = frozen paired_contrast;
        transported_* = frozen one-seed rule (SE^2 = SE_Tango^2 + s2_pool, family logit pool
        and df). 'feature' rows use logit-KD references beside their own twin (kd_differs).
        reference_included = False if the build failed the functional floor or is unchecked.
        primary_* = the frozen primary row (seed-pooled / transported / structural), unchanged.""")
    K.write_csv(summ, os.path.join(out, "reference_k_summary.csv"), """
        One row per (dataset, KD, trigger): Tango k against each included reference (t42 twin
        and the C' builds), transported verdicts across references ('/'-joined in reference
        order), reference_dependent@delta = the verdict is not the same for every reference,
        equivalent_changes@delta = EQUIVALENT under some references only, differs_from_primary
        = some reference verdict differs from the frozen primary verdict (different seed basis:
        the primary is seed-pooled where it has seeds 42-44). A33style_* = frozen primary k with
        the between-build variance of the base-trainer references' hit rates on the trigger
        inputs (t42 base or twin, t43, t44, ...; df J-1) added to the primary SE, Satterthwaite
        df (the A3.3 formula applied to every trigger); A33style_reference_dependent = deltas
        where it differs from the frozen primary verdict; A33style_already_in_frozen = those of
        them a frozen table already shows (primary.csv robust_verdict@0.05 differs from
        verdict@0.05, or for ATBA the frozen A3.3 reference_atba.csv reference_dependent);
        A33style_frozen_source names that table; A33style_new_at = the rest. AG News 'feature' BadNets/AddSent/SynBkd/SOS use the
        hits/fill_bert_agnews.json feature re-scores.""")
    K.write_csv(sp_rows, os.path.join(out, "reference_k_seedpooled_t43.csv"), """
        BERT SST-2 logit, seeds 42-44: k = mean_s (P_s - C_s) against the t42 twin and against
        the t43 clean build's C' students (s42 = A/A tifs_aa_bert_clean_t43 student, identical
        weights to the A2 replicate s42 C'; s43, s44 = A2 replicate clean-base students), CPU
        fp32; frozen seed-pooled rule (BERT SST-2 logit pool) and own variance (df r-1);
        reference_dependent@delta = the two references give different pooled verdicts,
        equivalent_changes@delta = EQUIVALENT under one of them only.""")
    K.write_csv(runs_rows, os.path.join(out, "reference_builds.csv"), """
        One row per scored clean reference build and dataset: run, teacher seed, the scored C'
        student's checkpoint sha256, its hits file, the functional check (Stage 2 clean
        accuracy of the clean teacher and of the C' student against majority share + 0.10) and
        whether the build enters the reference sets.""")
    K.write_csv(checks, os.path.join(out, "check_item7.csv"), """
        Self-checks of item 7: every per-reference contrast pairs vectors built from the same
        input texts (sha256 of the triggered inputs), and the A/A t43 C' equals the A2
        replicate seed-42 C' (same weights) on every trigger.""")
    print(f"item 7: {len(rows)} reference rows, {len(summ)} cells, {len(sp_rows)} seed-pooled rows, "
          f"{sum(not c['ok'] for c in checks)} failed checks of {len(checks)}")


if __name__ == "__main__":
    main()
