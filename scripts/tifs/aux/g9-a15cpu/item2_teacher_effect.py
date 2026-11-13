#!/usr/bin/env python3
"""A15 CPU item: teacher-level attributable effect k_T and the transmitted share k / k_T per primary cell.

A15 post hoc CPU analysis; changes no primary verdict, battery decision or the paper-wide Holm.

k_T = ASR(P teacher) - ASR(C teacher) on the cell's triggered non-target inputs, from the official teacher
hit vectors stored with the Stage 2 predictions (teacher_p_trig, teacher_c_trig; GPU bf16, the same inputs
as the students). The teacher pair is fixed (teacher seed 42), so its vectors must be identical at every
student seed and for logit and 'feature' KD (checked). k is the frozen primary k (primary.csv). The
interval of k_T is the Tango paired interval (input sampling only: the teachers are fixed models). The
share k / k_T is shown with [lo / k_T, hi / k_T] from the frozen primary interval, i.e. k_T taken as fixed.
Also shown: the teachers' untriggered target-rate difference (teacher_p_clean - teacher_c_clean on the
untriggered non-target inputs, the teacher analogue of delta_b), raw ASR / teacher ASR, and the gate's
teacher_asr / teacher_asr_null from the Stage 2 result files (must equal the vector means).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os  # noqa: E402

import common as K  # noqa: E402

TOLS = K.TOLS


def main():
    out = K.out_dir()
    st = K.state()
    bv = K.battery_vectors(st)
    rows, checks = [], []
    teach = {}
    for r in st.primary:
        fam, ds, kd, atk = r["fam"], r["dataset"], r["kd"], r["attack"]
        seeds = [int(s) for s in r["seeds"].split(",")]
        vs = [bv[(fam, ds, kd, atk, s)] for s in seeds]
        TP, TC, TPc, TCc = vs[0]["TP"], vs[0]["TC"], vs[0]["TPc"], vs[0]["TCc"]
        same = all(v["TP"] == TP and v["TC"] == TC and v["TPc"] == TPc and v["TCc"] == TCc for v in vs)
        checks.append({"check": f"teacher vectors identical across student seeds {K.cell_id(r)}",
                       "expected": True, "got": same, "ok": same})
        prev = teach.setdefault((fam, ds, atk), (TP, TC, TPc, TCc))
        if prev != (TP, TC, TPc, TCc):
            checks.append({"check": f"teacher vectors identical across KD types {fam}/{ds}/{atk}",
                           "expected": True, "got": False, "ok": False})
        pc = K.paired_contrast(TP, TC, TOLS)
        kT = pc["k"]
        asr_ok = abs(K.mean(TP) - r["teacher_asr"]) < 1e-12 and abs(K.mean(TC) - r["teacher_asr_null"]) < 1e-12
        checks.append({"check": f"mean(teacher_p_trig), mean(teacher_c_trig) == teacher_asr, teacher_asr_null "
                                f"(Stage 2 result file) {K.cell_id(r)}",
                       "expected": True, "got": asr_ok, "ok": asr_ok})
        db_T = K.mean(TPc) - K.mean(TCc) if TPc is not None and TCc is not None else None
        row = {"fam": fam, "dataset": ds, "kd": kd, "attack": atk, "method": r["method"],
               "seeds": r["seeds"], "n": len(TP),
               "teacher_P_asr": K.mean(TP), "teacher_C_asr": K.mean(TC),
               "k_T": kT, "k_T_lo": pc["lo"], "k_T_hi": pc["hi"],
               "teacher_n10": pc["n10"], "teacher_n01": pc["n01"],
               "teacher_untriggered_P_minus_C": db_T,
               "raw_asr": r["raw_asr"], "C_trig": r["C_trig"],
               "k": r["k"], "k_lo": r["lo"], "k_hi": r["hi"],
               **{f"verdict@{t:g}": r[f"verdict@{t:g}"] for t in TOLS},
               "k_over_kT": r["k"] / kT if kT else None,
               "k_over_kT_lo": r["lo"] / kT if kT else None,
               "k_over_kT_hi": r["hi"] / kT if kT else None,
               "raw_asr_over_teacher_asr": r["raw_asr"] / K.mean(TP) if K.mean(TP) else None,
               "C_trig_over_teacher_C_asr": r["C_trig"] / K.mean(TC) if K.mean(TC) else None,
               "teacher_gate_asr_ge_0.90": r["teacher_asr"] >= 0.90,
               "kT_below_0.90": kT < 0.90,
               "structural_null": r["structural_null"]}
        rows.append(row)
    K.write_csv(rows, os.path.join(out, "teacher_effect.csv"), """
        One row per primary cell (31). teacher_P_asr / teacher_C_asr = target rate of the P / C teacher on
        the cell's n triggered non-target inputs (official Stage 2 teacher vectors teacher_p_trig /
        teacher_c_trig, GPU bf16; identical at every student seed); k_T = teacher_P_asr - teacher_C_asr
        (teacher-level attributable effect, A15) with its Tango paired 95% interval (input sampling only)
        and discordant counts n10 (P hit, C not) / n01; teacher_untriggered_P_minus_C = the teachers'
        target-rate difference on the untriggered non-target inputs; k, k_lo, k_hi, verdicts = frozen
        primary.csv; k_over_kT = k / k_T (share of the teacher-level effect seen in the students) with
        [k_lo / k_T, k_hi / k_T] (k_T taken as fixed); raw_asr_over_teacher_asr = P student ASR / P teacher
        ASR; teacher_gate_asr_ge_0.90 = the frozen gate's ASR criterion (raw P-teacher ASR); kT_below_0.90
        flags cells whose teacher-level effect is below the gate's 0.90 although the raw ASR passed.""")
    K.write_csv(checks, os.path.join(out, "check_item2.csv"), """
        Item 2 self-checks: teacher vectors identical across student seeds and KD types; the vector means
        equal the teacher_asr / teacher_asr_null of the Stage 2 result files.""")
    print(f"item 2: {len(rows)} cells; checks {'ok' if all(c['ok'] for c in checks) else 'FAILED'}")


if __name__ == "__main__":
    main()
