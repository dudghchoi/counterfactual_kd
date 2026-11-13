#!/usr/bin/env python3
"""A13 G2: clean reference teachers t45, t46 (BERT-large, base trainer, SST-2 and AG News).

A13 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

Registered rule (docs/tifs_addendum.md A13 G2):
  * Functional check (A7): a new clean teacher whose Stage 2 clean accuracy is at or below the
    majority-class share + 0.10 leaves the reference set (addendum.MAJORITY, FUNCTION_MARGIN); the
    fact is reported. The clean teacher's target rate on the untriggered non-target inputs is shown
    beside it (descriptive, as A7 showed for RIPPLe).
  * A3.3 recomputed with 5 references (t42-t46, df 4) for ATBA, SST-2 and AG News, logit and 'feature':
    the frozen addendum.reference_atba_table, unchanged, on the frozen state whose cell list also
    holds the A13 G2 reference cells (read by the frozen loader as A/A-style cells; their seed-42 C'
    student's ATBA-phrase control is the reference hit rate, exactly as for the A/A t43/t44 runs).
    Sentence per delta: ref_lo > delta -> "깨끗한 기준들을 평균해도 teacher 몫이 δ를 넘는다";
    ref_hi < delta -> "기준들을 평균하면 δ 아래"; else -> the reference-dependent wording stays, citing df 4.
  * Per-reference k for every BERT trigger: the A14 item 7 script (g6-review0/reference/
    item7_reference_k.py, unchanged) run with --out results/tifs/addendum/aux/g7-a13/item7 after
    score.py refs scored t45 / t46 there (CPU fp32); compared here with the A14 three-reference result.
  * The BadNets P' students of the same builds are A/A auxiliary material only (frozen analyze.aa_table);
    no decision uses them.
Descriptive (A15 E1 bears on it, not re-analysed): the OpenBackdoor-selected epoch of every reference
build, parsed from the Stage 0 logs for t45 / t46 and read from the A14 item 11 table for t42-t44.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os  # noqa: E402
import re  # noqa: E402

import common as K  # noqa: E402

A, AD, TOLS = K.A, K.AD, K.TOLS
S_ABOVE = "깨끗한 기준들을 평균해도 teacher 몫이 δ를 넘는다"
S_ABOVE_EN = "Even averaged over the clean references, the teacher share exceeds delta."
S_BELOW = "기준들을 평균하면 δ 아래"
S_BELOW_EN = "Averaged over the references, below delta."
S_DEP = "reference-dependent 문구를 유지하고 df 4를 근거로 든다"
S_DEP_EN = "Keep the reference-dependent wording, citing the reference df of 4."
OB_LINE = re.compile(r"\[OB-adapter\]\s+(?P<kind>\w+)\s+\|\s+(?P<attack>\w+)\s+\|\s+(?P<model>\S+)\s+\|\s+"
                     r"(?P<ds>\w+)\s+\|\s+seed=(?P<seed>\d+)")
DEV_LINE = re.compile(r"accuracy on (?P<split>dev-clean|dev-poison): (?P<acc>[0-9.]+)")


class _Shim:
    """The attributes addendum.reference_atba_table reads (groups, prim, cells)."""

    def __init__(self, st, extra_cells):
        self.groups, self.prim, self.cells = st.groups, st.prim, list(st.cells) + list(extra_cells)


def parse_teacher_log(path):
    """[(kind, attack, ds, seed, [(dev_clean, dev_poison) per epoch])] from an OB Stage 0 log."""
    segs, cur = [], None
    with open(path, errors="replace") as f:
        for line in f:
            m = OB_LINE.search(line)
            if m:
                cur = {"kind": m["kind"], "attack": m["attack"], "ds": m["ds"], "seed": int(m["seed"]),
                       "epochs": [], "pending": {}}
                segs.append(cur)
                continue
            m = DEV_LINE.search(line)
            if m and cur is not None:
                cur["pending"][m["split"]] = float(m["acc"])
                if "dev-clean" in cur["pending"] and "dev-poison" in cur["pending"]:
                    cur["epochs"].append((cur["pending"]["dev-clean"], cur["pending"]["dev-poison"]))
                    cur["pending"] = {}
    return segs


def ob_select(epochs):
    """OpenBackdoor ckpt 'best': the first epoch whose mean(dev-clean, dev-poison) exceeds all earlier."""
    best, sel = -1.0, None
    for i, (a, h) in enumerate(epochs, 1):
        s = (a + h) / 2
        if s > best:
            best, sel = s, i
    return sel


def run(out):
    st = K.state()
    x = [c for c in K.load_x13() if c["group"] == "G2"]
    checks = []
    # ── functional check ──
    fun_rows, include = [], {}
    for c in sorted(x, key=lambda c: (c["dataset"], c["teacher_seed"])):
        ds, t = c["dataset"], c["teacher_seed"]
        floor = K.floor_of(ds)
        ok_t = c["teacher_ca_null"] is not None and c["teacher_ca_null"] > floor
        ok_s = c["ca_c"] is not None and c["ca_c"] > floor
        include[(ds, t)] = ok_t and ok_s
        fun_rows.append({
            "dataset": ds, "reference": f"t{t}", "run": c["run"], "teacher": "clean (base trainer, BadNets dev split)",
            "clean_teacher_clean_acc": c["teacher_ca_null"], "majority": AD.MAJORITY[ds], "floor": floor,
            "clean_teacher_functional": ok_t,
            "clean_teacher_target_rate_untriggered_nontarget": K.mean(c["TCc"]) if c["TCc"] else None,
            "clean_teacher_target_rate_badnets_triggered": K.mean(c["TC"]) if c["TC"] else None,
            "Cprime_student_clean_acc": c["ca_c"], "Cprime_student_functional": ok_s,
            "aux_Pprime_badnets_teacher_clean_acc": c["teacher_ca"], "aux_Pprime_teacher_asr": c["teacher_asr"],
            "aux_Pprime_student_clean_acc": c["ca_p"],
            "enters_reference_set": ok_t and ok_s,
            "note": "" if ok_t and ok_s else "excluded from the reference set (A7 / A8 floor)"})
    K.write_csv(fun_rows, os.path.join(out, "g2_function.csv"), """
        A13 G2 functional check, one row per new clean reference build and dataset (official Stage 2,
        GPU bf16). clean_teacher_clean_acc = the clean teacher's Stage 2 clean test accuracy
        (teacher_ca_null); floor = majority-class share of the Stage 2 test set (SST-2 912/1821, AG News
        1900/7600) + 0.10 (A7; addendum.MAJORITY + FUNCTION_MARGIN); functional = accuracy > floor.
        clean_teacher_target_rate_* = the clean teacher's target-label rate on the untriggered non-target
        inputs and on the BadNets-triggered ones (descriptive). Cprime_student_clean_acc = the seed-42
        logit C' student's clean accuracy against the same floor (A8). aux_Pprime_* = the BadNets
        poisoned teacher and P' student of the same build (A/A auxiliary, not used for any decision).""")

    # ── selection epochs (descriptive; A15 E1) ──
    ep_rows = []
    sel_a14 = K.read_csv(os.path.join(K.G6CORPUS, "selection_builds.csv"))
    for r in sel_a14:
        if r["fam"] == "bert" and r["attack"] in ("badnets", "atba") and r["teacher_seed"] in ("42", "43", "44") \
                and (r["kind"] == "clean" or r["attack"] == "atba" or r["teacher_seed"] == "42"):
            ep_rows.append({"dataset": r["dataset"], "teacher": r["teacher"], "kind": r["kind"],
                            "attack": r["attack"], "teacher_seed": int(r["teacher_seed"]),
                            "selected_epoch": int(r["selected_epoch"]), "acc_only_epoch": r["acc_only_epoch"],
                            "last_epoch": r["last_epoch"],
                            "source": "A14 item 11 (g6-review0/corpus/selection_builds.csv)"})
    for t in (45, 46):
        log = os.path.join(K.ROOT, "logs", "tifs", f"tifs_x13_bert_ref_t{t}_teachers.log")
        for seg in parse_teacher_log(log):
            accs = [a for a, _ in seg["epochs"]]
            ds = {"sst2": "sst2", "agnews": "agnews"}.get(seg["ds"], seg["ds"])
            ep_rows.append({"dataset": ds, "teacher": f"ob_{'' if seg['kind'] == 'clean' else seg['attack'] + '_'}"
                                                     f"bert-large_{'sst-2' if ds == 'sst2' else ds}_seed{seg['seed']}",
                            "kind": seg["kind"], "attack": seg["attack"], "teacher_seed": seg["seed"],
                            "selected_epoch": ob_select(seg["epochs"]),
                            "acc_only_epoch": accs.index(max(accs)) + 1 if accs else None,
                            "last_epoch": len(seg["epochs"]),
                            "dev_clean_by_epoch": "/".join(f"{a:.4f}" for a, _ in seg["epochs"]),
                            "dev_poison_by_epoch": "/".join(f"{h:.4f}" for _, h in seg["epochs"]),
                            "source": K.rel(log) + " (OB rule recomputed: first epoch with the highest "
                                                   "mean(dev-clean, dev-poison))"})
    ep_rows.sort(key=lambda r: (r["dataset"], r["kind"] != "poisoned", r["teacher_seed"], r["attack"]))
    K.write_csv(ep_rows, os.path.join(out, "g2_reference_epochs.csv"), """
        Descriptive (A15 E1 bears on G2; not re-analysed with A15 data): the epoch at which each BERT-large
        reference build of the ATBA A3.3 reading was saved by the OpenBackdoor 'best' rule (first epoch with
        the highest mean of dev-clean accuracy and dev-poison hit; clean twins use the BadNets dev-poison
        split), with the ATBA poisoned teachers and the BadNets poisoned teacher t42 (the G3 / G4 pair) for
        comparison. t42-t44 from the A14 item 11 table; t45/t46
        recomputed here from the logged dev numbers of their Stage 0 log (the BadNets poisoned t45/t46 rows are
        the auxiliary P' teachers).""")

    # ── A3.3 with 5 references ──
    inc_cells = [c for c in x if include[(c["dataset"], c["teacher_seed"])]]
    exc = sorted(f"{c['dataset']}/t{c['teacher_seed']}" for c in x if not include[(c["dataset"], c["teacher_seed"])])
    frozen3 = {(r["dataset"], r["kd"]): r for r in AD.reference_atba_table(st) if "kd" in r}
    rows5 = AD.reference_atba_table(_Shim(st, inc_cells))
    out_rows, sentences = [], {}
    for r in rows5:
        if "kd" not in r:
            continue
        f3 = frozen3[(r["dataset"], r["kd"])]
        row = {"dataset": r["dataset"], "kd": r["kd"],
               **{k: v for k, v in r.items() if k.startswith("C_t")}, "n_references": r["n_references"],
               "s_ref": r["s_ref"], "df_ref": r["df_ref"], "k": r["k"], "lo": r["lo"], "hi": r["hi"],
               **{f"primary_verdict@{t:g}": r[f"verdict@{t:g}"] for t in TOLS},
               "ref5_lo": r["ref_lo"], "ref5_hi": r["ref_hi"], "ref5_df": r["ref_df"],
               **{f"ref5_verdict@{t:g}": r[f"ref_verdict@{t:g}"] for t in TOLS},
               "ref5_reference_dependent": r["reference_dependent"],
               "ref3_s_ref": f3["s_ref"], "ref3_df_ref": f3["df_ref"], "ref3_lo": f3["ref_lo"], "ref3_hi": f3["ref_hi"],
               "ref3_df": f3["ref_df"], **{f"ref3_verdict@{t:g}": f3[f"ref_verdict@{t:g}"] for t in TOLS},
               "excluded_references": ";".join(exc) or "none", "note": r.get("note", "")}
        for t in TOLS:
            if r["ref_lo"] > t:
                s, e = S_ABOVE, S_ABOVE_EN
            elif r["ref_hi"] < t:
                s, e = S_BELOW, S_BELOW_EN
            else:
                s, e = S_DEP, S_DEP_EN
            row[f"sentence@{t:g}"] = s
            row[f"sentence_en@{t:g}"] = e
            sentences[(r["dataset"], r["kd"], t)] = s
        out_rows.append(row)
    checks.append({"check": "G2: the 5-reference A3.3 rows use 5 references with df_ref 4 (no exclusion)",
                   "expected": "n_references 5, df_ref 4 in every row",
                   "got": ";".join(f"{r['dataset']}/{r['kd']}:{r['n_references']}/{r['df_ref']}" for r in out_rows),
                   "ok": all(r["n_references"] == 5 and r["df_ref"] == 4 for r in out_rows)})
    checks.append({"check": "G2: addendum.reference_atba_table on the frozen state = results/tifs/addendum/"
                            "reference_atba.csv (3 references; part of the gate)",
                   "expected": "4 rows", "got": f"{len(frozen3)} rows", "ok": len(frozen3) == 4})
    K.write_csv(out_rows, os.path.join(out, "g2_reference_atba5.csv"), """
        A13 G2: A3.3 recomputed with the five clean references t42-t46 (seed-42 logit students; C_t42 = the
        battery's clean-base C student on the ATBA-triggered inputs, C_t43..C_t46 = the C' students of the
        clean reference builds on the ATBA-phrase control, official GPU bf16). s_ref = SD of the five
        reference hit rates (df_ref = 4); ref5_* = k +- t sqrt(SE^2 + s2_ref) with the Satterthwaite df of
        the primary SE (pool df) and s2_ref (df 4), by the frozen addendum.reference_atba_table; k, lo, hi,
        primary_verdict = the frozen primary row (unchanged); ref3_* = the frozen three-reference A3.3
        (reference_atba.csv). 'feature' uses the logit-KD references (as A3.3). sentence@delta = the A13 G2
        sentence the registered rule selects at that delta (ref5_lo > delta: above; ref5_hi < delta: below;
        else reference-dependent).""")

    # ── auxiliary A/A contrasts of the new builds (no decision) ──
    aa = [r for r in A.aa_table(list(st.cells) + list(x)) if r["teacher_seed"] in (45, 46)]
    for r in aa:
        r["status"] = "A/A auxiliary (A13 G2): not used for any decision"
    K.write_csv(aa, os.path.join(out, "g2_aa_auxiliary.csv"), """
        A/A auxiliary contrasts of the A13 G2 builds by the frozen analyze.aa_table (seed-42 battery cell
        against the t45/t46 run on the same BadNets-triggered inputs): C-C', P-C', P'-C' (the auxiliary
        BadNets P' student), P-P', and C-C'@atba (seed-42 clean student on the ATBA inputs vs C' on the
        ATBA-phrase control). Tango intervals. Reported only; A13 G2 uses no P' student for a decision.""")
    K.write_csv(checks, os.path.join(out, "check_g2.csv"), "Self-checks of the G2 script.")
    return {"function": fun_rows, "epochs": ep_rows, "atba5": out_rows, "sentences": {f"{a}/{b}@{t:g}": s for (a, b, t), s in sentences.items()},
            "excluded": exc, "checks": checks}


def item7_compare(out):
    """A14 item 7 with t45 / t46 (results/tifs/addendum/aux/g7-a13/item7) against the A14 result."""
    d7 = os.path.join(out, "item7")
    new = K.read_csv(os.path.join(d7, "reference_k_summary.csv"))
    old = {(r["dataset"], r["kd"], r["trigger"]): r for r in K.read_csv(os.path.join(K.G6REF, "reference_k_summary.csv"))}
    builds = K.read_csv(os.path.join(d7, "reference_builds.csv"))
    rows = []
    for r in new:
        o = old[(r["dataset"], r["kd"], r["trigger"])]
        row = {"dataset": r["dataset"], "kd": r["kd"], "trigger": r["trigger"], "references_5": r["references"],
               "n_references_5": r["n_references"], "references_3": o["references"],
               "primary_method": r["primary_method"], "primary_k": r["primary_k"],
               **{f"k_{t}": r.get(f"k_{t}") for t in ("t42", "t43", "t44", "t45", "t46")},
               **{f"C_{t}": r.get(f"C_{t}") for t in ("t42", "t43", "t44", "t45", "t46")},
               "k_min_5": r["k_min"], "k_max_5": r["k_max"], "k_min_3": o["k_min"], "k_max_3": o["k_max"]}
        for t in TOLS:
            row[f"verdicts_5@{t:g}"] = r[f"verdicts@{t:g}"]
            row[f"verdicts_3@{t:g}"] = o[f"verdicts@{t:g}"]
            row[f"primary_verdict@{t:g}"] = r[f"primary_verdict@{t:g}"]
        for flag in ("reference_dependent", "equivalent_changes"):
            n5 = set(x for x in r[flag].split(",") if x)
            n3 = set(x for x in o[flag].split(",") if x)
            row[f"{flag}_5"] = ",".join(sorted(n5, key=float))
            row[f"{flag}_3"] = ",".join(sorted(n3, key=float))
            row[f"{flag}_new_with_t45_t46"] = ",".join(sorted(n5 - n3, key=float))
            row[f"{flag}_dropped"] = ",".join(sorted(n3 - n5, key=float))
        row.update({"A33style_J_5": r.get("J_base_trainer_refs"), "A33style_df_ref_5": r.get("df_ref"),
                    "A33style_lo_5": r.get("A33style_lo"), "A33style_hi_5": r.get("A33style_hi"),
                    **{f"A33style_verdict_5@{t:g}": r.get(f"A33style_verdict@{t:g}") for t in TOLS},
                    "A33style_lo_3": o.get("A33style_lo"), "A33style_hi_3": o.get("A33style_hi"),
                    **{f"A33style_verdict_3@{t:g}": o.get(f"A33style_verdict@{t:g}") for t in TOLS},
                    "A33style_reference_dependent_5": r.get("A33style_reference_dependent"),
                    "A33style_reference_dependent_3": o.get("A33style_reference_dependent")})
        rows.append(row)
    K.write_csv(rows, os.path.join(out, "g2_item7_compare.csv"), """
        A14 item 7 (per-reference k for every BERT primary trigger, seed-42 P against each clean reference,
        CPU fp32, frozen transported rule) extended with the A13 G2 references t45/t46: *_5 = the item-7
        script (g6-review0/reference/item7_reference_k.py, unchanged) on results/tifs/addendum/aux/g7-a13/item7
        (references t42 twin + t43..t46 C'); *_3 = the A14 result (g6-review0/reference/reference_k_summary.csv;
        t42 twin, t43, t44). k_t<k> = Tango k = P - C_ref; C_t<k> = the reference's hit rate; verdicts_* =
        transported verdict per reference ('/'-joined); reference_dependent = deltas where the verdict is
        not the same for every reference; equivalent_changes = deltas where EQUIVALENT holds under some
        references only (the A13 G2 / A3 'reference-dependent' label); *_new_with_t45_t46 / *_dropped = the
        change from three to five references. A33style_* = frozen primary k with the base-trainer references'
        hit-rate variance added (df J-1), as item 7 defines it.""")
    # Self-check: overlapping per-reference rows identical to the A14 run.
    a_new = K.read_csv(os.path.join(d7, "reference_k_all_triggers.csv"))
    a_old = K.read_csv(os.path.join(K.G6REF, "reference_k_all_triggers.csv"))
    key = ("dataset", "kd", "trigger", "reference")
    idx = {tuple(r[c] for c in key): r for r in a_new}
    n_bad, notes = 0, []
    for o in a_old:
        r = idx.get(tuple(o[c] for c in key))
        if r is None:
            n_bad += 1
            notes.append("/".join(o[c] for c in key) + " missing")
            continue
        for c, v in o.items():
            # *_source columns name the hits file; the copies live in this group's item7/hits folder.
            got = os.path.basename(r.get(c) or "") if c.endswith("_source") else r.get(c)
            want = os.path.basename(v) if c.endswith("_source") else v
            if got != want:
                n_bad += 1
                notes.append(f"{'/'.join(o[c] for c in key)}:{c}")
    checks = [{"check": "item 7: every A14 per-reference row (t42 twin, t42 base, t43, t44) reproduced exactly "
                        "in the 5-reference run (every column; *_source compared by file name, the hit files "
                        "being byte-identical copies)", "expected": f"{len(a_old)} rows, 0 mismatches",
               "got": f"{n_bad} mismatches", "ok": n_bad == 0, "note": "; ".join(notes[:5])},
              {"check": "item 7: rows added are exactly the t45 C' and t46 C' references",
               "expected": f"{len(a_old)} + 2 x {len(rows)}", "got": f"{len(a_new)}",
               "ok": len(a_new) == len(a_old) + 2 * len(rows)
               and {r['reference'] for r in a_new} - {r['reference'] for r in a_old} == {"t45 C'", "t46 C'"}},
              {"check": "item 7: t45 and t46 enter the reference sets on both datasets (functional floor)",
               "expected": "4 builds included", "got": ";".join(f"{b['dataset']}/{b['reference']}:{b['status']}" for b in builds),
               "ok": sum(b["status"] == "included" and b["reference"] in ("t45", "t46") for b in builds) == 4}]
    c7 = K.read_csv(os.path.join(d7, "check_item7.csv"))
    checks.append({"check": "item 7: the script's own checks (input sha256 per pair, t43 determinism) all pass",
                   "expected": f"{len(c7)} ok", "got": f"{sum(r['ok'] == 'True' for r in c7)} ok",
                   "ok": all(r["ok"] == "True" for r in c7)})
    K.write_csv(checks, os.path.join(out, "check_g2_item7.csv"), "Self-checks of the item-7 extension.")
    return {"rows": rows, "checks": checks, "builds": builds}


def main():
    out = K.out_dir()
    s = run(out)
    for r in s["atba5"]:
        print(f"  {r['dataset']}/{r['kd']}: 5 refs s_ref {r['s_ref']:.4f} -> [{r['ref5_lo']:+.4f}, {r['ref5_hi']:+.4f}] "
              f"df {r['ref5_df']:.2f}; 3 refs [{r['ref3_lo']:+.4f}, {r['ref3_hi']:+.4f}]; "
              + " ".join(f"{t:g}:{r[f'ref5_verdict@{t:g}'][:3]}" for t in TOLS))
    if "--with-item7" in sys.argv:
        c = item7_compare(out)
        s["item7_checks"] = c["checks"]
        for r in c["rows"]:
            if r["reference_dependent_new_with_t45_t46"] or r["equivalent_changes_new_with_t45_t46"]:
                print(f"  item7 {r['dataset']}/{r['kd']}/{r['trigger']}: new ref-dep {r['reference_dependent_new_with_t45_t46']}"
                      f" new EQU-changes {r['equivalent_changes_new_with_t45_t46']}")
    K.dump_json(s, os.path.join(out, "logs", "g2_summary.json"))


if __name__ == "__main__":
    main()
