#!/usr/bin/env python3
"""CPU fp32 re-scoring of the epoch-matched clean students for A15-b2 (A14 item 7 with references at each P's
epoch).

A15 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

A14 item 7 compares each BERT SST-2 battery P student (seed 42, CPU fp32 hits of
results/tifs/addendum/cross_hits_bert_sst2.json) with every clean reference's C' student on the same
triggered inputs. A15-b2 repeats it with the references at the P teacher's epoch e(P) (ATBA, AddSent: e1;
BadNets: e4; SynBkd: e5). This script scores, with the registered A14 scoring engine
(scripts/tifs/aux/g6-review0/reference/score.py ``Scorer``, imported unchanged: frozen
cross_score.triggered_loader inputs, the frozen evaluator's _probe_model, fp32 on CPU), only on the triggers
whose P epoch is the student's epoch:
  b1   tifs_x15_b1_bert_sst2_e<e>_s42  clean-base logit and 'feature' C_e students (the t42 reference at e)
       -> <out>/hits_em/b1_<run>.<kd>.sst2.json
  b2   tifs_x15_b2_bert_ref_t<t>_e<e>  clean-base logit C' students, t in 43-46, e in {1, 4, 5} except t43 e1
       (the frozen C'(t43) is epoch 1: its A14 hits are reused) -> <out>/hits_em/ref_<run>.sst2.json
A run is scored only when its training finished (runs/tifs/_status/<run>.ok) and its checkpoint exists.
Determinism check (--recheck): the frozen C'(t43) student is re-scored on BadNets and ATBA and must equal its
A14 hits (g6-review0/reference/hits/ref_tifs_aa_bert_clean_t43.sst2.json) -> hits_em/recheck_t43.sst2.json.
Output files have the cross_score format and are resumable (a student already scored with the same checkpoint
sha256 is skipped).

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 nice -n 10 python score_em.py [--recheck] [--only-recheck]
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
G6REF_CODE = os.path.join(ROOT, "scripts", "tifs", "aux", "g6-review0", "reference")
EPOCH_ATTACKS = {1: ("addsent", "atba"), 4: ("badnets",), 5: ("synbkd",)}
DEFAULT_RUNS = os.path.join(ROOT, "runs", "tifs")
DEFAULT_OUT = os.path.join(ROOT, "results", "tifs", "addendum", "aux", "g8-a15")


def plan(runs):
    """[(file name, run, kd, key, student path or None, probes, reason)] of every scoring job."""
    jobs = []
    for e, atks in EPOCH_ATTACKS.items():
        run = f"tifs_x15_b1_bert_sst2_e{e}_s42"
        for kd in ("logit", "feature"):
            jobs.append((f"b1_{run}.{kd}.sst2.json", run, kd, f"{run}|{kd}/clean-base/clean/s42",
                         *student(runs, run, kd), atks))
    for t in (43, 44, 45, 46):
        for e, atks in EPOCH_ATTACKS.items():
            if (t, e) == (43, 1):
                continue
            run = f"tifs_x15_b2_bert_ref_t{t}_e{e}"
            jobs.append((f"ref_{run}.sst2.json", run, "logit", f"{run}|logit/clean-base/clean/s42",
                         *student(runs, run, "logit"), atks))
    return jobs


def student(runs, run, kd):
    """(path, reason): the run's clean-base student of ``kd`` on SST-2 when its training finished."""
    if not os.path.exists(os.path.join(runs, "_status", f"{run}.ok")):
        return None, "training not finished (no _status/<run>.ok)"
    cand = sorted(glob.glob(os.path.join(runs, run, "artifacts", "students", f"clean-base_{kd}_*_sst2_*",
                                         "clean_student")))
    cand = [c for c in cand if os.path.exists(os.path.join(c, "model.safetensors"))]
    if len(cand) != 1:
        return None, f"{len(cand)} clean-base {kd} checkpoints"
    return cand[0], "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=DEFAULT_RUNS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "8")))
    ap.add_argument("--recheck", action="store_true", help="also re-score the frozen C'(t43) (determinism)")
    ap.add_argument("--only-recheck", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="list the jobs and their status only")
    a = ap.parse_args()
    hits = os.path.join(a.out, "hits_em")
    jobs = plan(a.runs)
    for name, run, kd, key, path, reason, atks in jobs:
        have = os.path.exists(os.path.join(hits, name))
        print(f"  {name:62s} {','.join(atks):14s} {'have file' if have else ''} {reason}")
    if a.dry_run:
        return 0
    sys.path.insert(1, G6REF_CODE)
    import score as S  # noqa: E402  (scripts/tifs/aux/g6-review0/reference/score.py; imported unchanged)
    sc = S.Scorer("sst2", a.threads)
    os.makedirs(hits, exist_ok=True)
    if a.recheck or a.only_recheck:
        src = os.path.join(ROOT, "results", "tifs", "addendum", "aux", "g6-review0", "reference", "hits",
                           "ref_tifs_aa_bert_clean_t43.sst2.json")
        X = json.load(open(src))
        (k43, m43), = [(k, v) for k, v in X["students"].items() if v["tag"] == "clean-base"]
        path = os.path.join(ROOT, m43["path"])
        out_path = os.path.join(hits, "recheck_t43.sst2.json")
        sc.run([{"key": k43, "run": m43["run"], "tag": "clean-base", "role": "clean", "kd": "logit", "seed": 42,
                 "teacher_seed": 43, "dataset": "sst2", "path": path}], ["badnets", "atba"], out_path,
               {"kind": "A15-b2 determinism re-check of the frozen C'(t43) (A14 hits)"})
        Y = json.load(open(out_path))
        same = all(Y["hits"][p][k43] == X["hits"][p][k43] for p in ("badnets", "atba"))
        same_in = all(Y["inputs"][p]["sha256"] == X["inputs"][p]["sha256"] for p in ("badnets", "atba"))
        print(f"recheck C'(t43): hits identical to the A14 file: {same}; inputs identical: {same_in}")
        if not (same and same_in):
            return 2
    if a.only_recheck:
        return 0
    for name, run, kd, key, path, reason, atks in jobs:
        if path is None:
            print(f"skip {name}: {reason}")
            continue
        meta = json.load(open(os.path.join(path, "kd_meta.json")))
        sc.run([{"key": key, "run": run, "tag": "clean-base", "role": "clean", "kd": kd,
                 "seed": int(meta.get("student_seed", 42)), "teacher_seed": int(meta.get("teacher_seed", -1)),
                 "teacher_kind": meta.get("teacher_kind"), "dataset": "sst2", "path": path}],
               list(atks), os.path.join(hits, name),
               {"kind": "A15-b2 epoch-matched clean student (CPU fp32)", "run": run})
    return 0


if __name__ == "__main__":
    sys.exit(main())
