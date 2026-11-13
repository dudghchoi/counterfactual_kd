#!/usr/bin/env python3
"""A13 'common': the A8 student floor for every A13 student, exclusions, and the job / cut record.

A13 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

* Student floor (A8 rule, A13 'common'): a student whose Stage 2 clean accuracy is at or below the
  majority-class share + 0.10 takes its seed out of the judgement. Every A13 student (P / C of the logit
  cells, R of the CE cells, the auxiliary P' of the G2 builds), plus the official seed-42 students the G1
  pool uses (tifs_b_qwen3_agnews), against addendum.MAJORITY + FUNCTION_MARGIN.
* Jobs: runs/tifs/_status/events.jsonl (read only): every tifs_x13_* job, its 'running' and 'ok'
  events, attempts, against the A13 deadline 10/27 18:00 KST (09:00 UTC) and the registered cut order.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import os  # noqa: E402
from collections import defaultdict  # noqa: E402

import common as K  # noqa: E402


def run(out):
    st = K.state()
    x = K.load_x13()
    rows = K.student_rows(x)
    s42 = [dict(c, group="G1 (seed 42, official)") for c in st.cells
           if (c["fam"], c["dataset"], c["kd"]) == ("qwen3", "agnews", "logit") and c["block"] == "G"
           and c["corpus"] == "clean" and c["seed"] == 42 and not c["single"]]
    rows += K.student_rows(s42)
    K.write_csv(rows, os.path.join(out, "common_student_floor.csv"), """
        A8 student floor applied to every A13 student (A13 'common'), one row per student: Stage 2 clean test
        accuracy (official GPU bf16; P = ca_poisoned, C = ca_clean of a logit cell, R = the CE-only student;
        in G2 'P' is the auxiliary BadNets P' student) against floor = majority-class share of the test set
        (SST-2 912/1821, AG News 1900/7600) + 0.10; functional = accuracy > floor. A non-functional student
        takes its seed out of the group's judgement. The official seed-42 students of the G1 pool
        (tifs_b_qwen3_agnews) are listed as well.""")
    # ── jobs ──
    ev = defaultdict(list)
    notes = []
    with open(os.path.join(K.RUNS, "_status", "events.jsonl")) as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            # A13 jobs only: tifs_x13_<run>[__eval|_teachers]. A15 jobs that re-score A13 students
            # (e.g. tifs_x15_ws__tifs_x13_*) start with tifs_x15_ and are not A13 jobs.
            if e.get("job", "").startswith("tifs_x13_"):
                ev[e["job"]].append(e)
            elif e.get("state") == "note" and ("A13" in e.get("note", "") or "x13" in e.get("note", "")):
                notes.append(e)
    job_rows = []
    for job, es in sorted(ev.items()):
        starts = [e for e in es if e.get("state") == "running"]
        oks = [e for e in es if e.get("state") == "ok"]
        others = sorted({e.get("state") for e in es} - {"running", "ok"})
        last_ok = max((e["time"] for e in oks), default="")
        job_rows.append({"job": job, "run": es[0].get("run", ""), "attempts": len(starts),
                         "first_start_utc": min((e["time"] for e in starts), default=""),
                         "ok_utc": last_ok, "other_states": ",".join(others),
                         "status_file_ok": os.path.exists(os.path.join(K.RUNS, "_status", f"{job}.ok")),
                         "before_deadline": bool(last_ok) and last_ok <= K.DEADLINE_UTC,
                         "lane": ",".join(sorted({e.get("lane", "") for e in starts})),
                         "gpu": ",".join(sorted({str(e.get("gpu", "")) for e in starts}))})
    K.write_csv(job_rows, os.path.join(out, "common_jobs.csv"), """
        One row per A13 GPU job (runs/tifs/_status/events.jsonl, times UTC as logged; read only): attempts =
        number of 'running' events, ok_utc = the 'ok' event, status_file_ok = runs/tifs/_status/<job>.ok exists,
        before_deadline = finished before the A13 deadline 10/27 18:00 KST (09:00 UTC). A13 cut order (not
        needed when every job is ok): G4 s44, G4 s43, G3 s44, G3 s43, G2 t46, G1 s44.""")
    runs = sorted({c["run"] for c in x})
    expected = sorted(K.EXPECTED_RUNS)
    summary = {
        "n_jobs": len(job_rows), "n_ok": sum(bool(r["ok_utc"]) and r["status_file_ok"] for r in job_rows),
        "n_attempts_gt1": sum(r["attempts"] > 1 for r in job_rows),
        "n_other_states": sum(bool(r["other_states"]) for r in job_rows),
        "last_ok_utc": max(r["ok_utc"] for r in job_rows), "all_before_deadline": all(r["before_deadline"] for r in job_rows),
        "runs_with_cells": len(runs), "runs_expected": len(expected), "runs_match": runs == expected,
        "n_cells": len(x), "n_students": len([r for r in rows if not r["group"].startswith("G1 (seed 42")]),
        "n_students_below_floor": sum(not r["functional"] for r in rows),
        "excluded": sorted({r["run"] for r in rows if not r["functional"]}),
        "cut": [], "notes": [{"time": e["time"], "job": e["job"], "note": e["note"]} for e in notes]}
    checks = [{"check": "jobs: 52 A13 GPU jobs (25 student-training runs, 25 evaluations, 2 teacher builds), each one "
                        "attempt, each ok with its status file, before the deadline; none cut",
               "expected": "52 / 52 ok, 0 retries, 0 other states, all before 2026-10-27 09:00 UTC",
               "got": f"{summary['n_ok']} / {summary['n_jobs']} ok, {summary['n_attempts_gt1']} retried, "
                      f"{summary['n_other_states']} other states, last ok {summary['last_ok_utc']}",
               "ok": summary["n_jobs"] == 52 and summary["n_ok"] == 52 and summary["n_attempts_gt1"] == 0
               and summary["n_other_states"] == 0 and summary["all_before_deadline"]},
              {"check": "cells: the frozen loader reads predictions from exactly the 25 registered A13 runs",
               "expected": "25 runs, 38 cells", "got": f"{len(runs)} runs, {len(x)} cells",
               "ok": summary["runs_match"] and len(x) == 38},
              {"check": "floor: every A13 student (and the G1 seed-42 students) above majority + 0.10",
               "expected": "0 below", "got": f"{summary['n_students_below_floor']} below of {len(rows)}",
               "ok": summary["n_students_below_floor"] == 0}]
    K.write_csv(checks, os.path.join(out, "check_common.csv"), "Self-checks of the common A13 rules.")
    return {"summary": summary, "checks": checks, "students": rows, "jobs": job_rows}


def main():
    out = K.out_dir()
    s = run(out)
    print("  ", {k: v for k, v in s["summary"].items() if k != "notes"})
    K.dump_json({"summary": s["summary"], "checks": s["checks"]}, os.path.join(out, "logs", "common_summary.json"))


if __name__ == "__main__":
    main()
