#!/usr/bin/env python3
"""A15 job inventory and input listing (the --dry-run of run_all.sh). Reads file NAMES and status markers only:
no A15 output is opened.

A15 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

Jobs: queue v9 (scripts/tifs/queue_plan.a15_parts, b3 disabled), one row per job id with its status marker in
runs/tifs/_status (<id>.ok / .failed / .blocked / .running; none = pending) and a classification:
  done; failed; cut (a hand-written .blocked marker while every dependency is done: the 10/27 amendment's cut
  method, order e2 -> e -> d); blocked by a dependency (a dependency failed / was cut / is blocked); running or
  not started at analysis time.
Inputs: per part, every file the analysis reads (Stage 2 prediction + result files of the A15 runs, from the
tags of plan_check.json; the ws re-scores; twin checks; the c gate; score_em.py hits), present or missing.
Files of a job that is not terminal-ok (no <run>__eval.ok / tifs_x15_ws__<run>.ok marker) are not read by the
analysis: 'present but job not terminal-ok: excluded' (column analysis_input of inventory.csv, job_ok of
inventory_inputs.csv).

The analysis is registered to run once, when every A15 job is terminal or at the deadline (10/29 14:00 KST =
05:00 UTC; amendment item 5). Without --force this script exits with status 4 when a job is still running or
pending and the deadline has not passed, so run_all.sh stops before any A15 output is read.

    python inventory.py --dry-run          # print only
    python inventory.py --check-ready      # print, write nothing, exit 4 if too early (run_all.sh's first step)
    python inventory.py                    # write inventory.csv, inventory_inputs.csv (exit 4 if too early)
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import datetime  # noqa: E402
import glob  # noqa: E402
import os  # noqa: E402

import a15lib as L  # noqa: E402

PLAN = os.path.join(L.ROOT, "scripts", "tifs", "aux", "g8-a15", "plan_check.json")
EXTERNAL = {"tifs_x15_go", "tifs_rule_frozen"}


def job_rows(runs):
    parts, Q = L.a15_jobs()
    status = {}
    for p, jobs in parts.items():
        for j in jobs:
            status[j["id"]] = L.status_of(runs, j["id"])
    rows = []
    for p, jobs in parts.items():
        for j in jobs:
            s = status[j["id"]]
            deps = [d for d in j.get("after", []) if d not in EXTERNAL and d != getattr(Q, "RULE_FROZEN", "")]
            dep_st = {d: status.get(d, L.status_of(runs, d)) for d in deps}
            if s == "ok":
                cls = "done"
            elif s == "failed":
                cls = "failed"
            elif s == "blocked":
                bad = {d: v for d, v in dep_st.items() if v != "ok"}
                cls = ("cut (hand-written .blocked marker; every dependency done)" if not bad else
                       "blocked by a dependency: " + ",".join(f"{d}={v}" for d, v in bad.items()))
            elif s == "running":
                cls = "running at analysis time"
            else:
                cls = "not started at analysis time"
            kind = ("evaluation" if j["id"].endswith("__eval") else "command" if "cmd" in j else
                    "+".join(j.get("stages", [])))
            rows.append({"part": p, "job": j["id"], "kind": kind, "status": s, "classification": cls,
                         "after": ",".join(deps), "terminal": s in ("ok", "failed", "blocked"),
                         "analysis_input": analysis_input(runs, j["id"], s)})
    return rows


def analysis_input(runs, job_id, status):
    """For an evaluation job (<run>__eval) and an A15-a re-score job (tifs_x15_ws__<run>): whether the analysis
    reads its files (file names and markers only)."""
    if job_id.endswith("__eval"):
        files = glob.glob(os.path.join(runs, job_id[: -len("__eval")], "artifacts", "results", "*.preds.json"))
    elif job_id.startswith("tifs_x15_ws__"):
        files = glob.glob(os.path.join(runs, "tifs_x15_ws", job_id[len("tifs_x15_ws__"):], "*.preds.json"))
    else:
        return ""
    if not files:
        return "no files"
    return "used (job ok)" if status == "ok" else L.EXCLUDED_TEXT


def input_rows(runs, out, hits_dir):
    plan = L.load_json(PLAN)
    rows = []
    for r in plan:
        part = L.part_of(r["run"])
        if part in (None, "b3"):
            continue
        d = os.path.join(runs, r["run"], "artifacts", "results")
        for ext in (".preds.json", ".json"):
            p = os.path.join(d, r["tag"] + ext)
            rows.append({"part": part, "run": r["run"], "kind": "Stage 2 " + ("predictions" if ext != ".json" else "result"),
                         "file": L.rel(p), "present": os.path.exists(p), "job_ok": L.eval_ok(runs, r["run"])})
    for run in L.WS_RUNS + tuple(L.C_RUN.format(s=s) for s in L.SEEDS):
        part = "c" if run.startswith("tifs_x15_c_") else "a"
        if run.startswith("tifs_x15_"):
            src_tags = [rr["tag"] for rr in plan if rr["run"] == run]
        else:
            src_tags = [os.path.basename(p)[: -len(".preds.json")] for p in
                        sorted(glob.glob(os.path.join(runs, run, "artifacts", "results", "*.preds.json")))]
        for tag in src_tags:
            for mode in ("ws", "official"):
                p = os.path.join(runs, "tifs_x15_ws", run, f"{tag}.{mode}.preds.json")
                rows.append({"part": part, "run": run, "kind": f"ws re-score ({mode})", "file": L.rel(p),
                             "present": os.path.exists(p), "job_ok": L.ws_ok(runs, run)})
    for t in ("ob_bert-large_sst-2_seed42", "ob_gpt2-xl_sst-2_seed42", "ob_Qwen3-1.7B_sst-2_seed42"):
        p = os.path.join(runs, "tifs_x15_twincheck", f"{t}.json")
        rows.append({"part": "b1", "run": "tifs_x15_twincheck", "kind": "twin check", "file": L.rel(p),
                     "present": os.path.exists(p)})
    for t in (43, 44, 45, 46):
        p = os.path.join(runs, "tifs_x15_twincheck", f"ob_bert-large_sst-2_seed{t}.json")
        rows.append({"part": "b2", "run": "tifs_x15_twincheck", "kind": "twin check", "file": L.rel(p),
                     "present": os.path.exists(p)})
    p = os.path.join(runs, "tifs_x15_c_gate", "gate.json")
    rows.append({"part": "c", "run": "tifs_x15_c_gate", "kind": "teacher gate", "file": L.rel(p),
                 "present": os.path.exists(p)})
    import score_em  # noqa: E402
    for name, run, kd, key, path, reason, atks in score_em.plan(runs):
        p = os.path.join(hits_dir, name)
        rows.append({"part": "b2", "run": run, "kind": f"CPU fp32 hits ({kd}; score_em.py)", "file": L.rel(p),
                     "present": os.path.exists(p), "note": "checkpoint " + reason})
    return rows


def main():
    ns = L.args(extra=[("--dry-run", {"action": "store_true"}), ("--force", {"action": "store_true"}),
                       ("--check-ready", {"action": "store_true",
                                          "help": "write nothing; exit 4 if too early (unless --force)"}),
                       ("--hits-dir", {"default": None}), ("--now", {"default": None,
                                                                    "help": "UTC time 'YYYY-mm-dd HH:MM:SS' (tests)"})])
    hits_dir = ns.hits_dir or os.path.join(ns.out, "hits_em")
    jobs = job_rows(ns.runs)
    inputs = input_rows(ns.runs, ns.out, hits_dir)
    now = (datetime.datetime.strptime(ns.now, "%Y-%m-%d %H:%M:%S") if ns.now else
           datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None))
    deadline = datetime.datetime.strptime(L.DEADLINE_UTC[:19], "%Y-%m-%d %H:%M:%S")
    open_jobs = [j for j in jobs if not j["terminal"]]
    print(f"A15 inventory at {now:%Y-%m-%d %H:%M:%S} UTC (deadline {L.DEADLINE_UTC}; cut order "
          f"{' -> '.join(L.CUT_ORDER)}); runs = {L.rel(ns.runs)}")
    for part in ("a", "b1", "b2", "c", "d", "e", "e2"):
        js = [j for j in jobs if j["part"] == part]
        cnt = {}
        for j in js:
            cnt[j["classification"].split(":")[0].split(" (")[0]] = cnt.get(j["classification"].split(":")[0].split(" (")[0], 0) + 1
        ins = [r for r in inputs if r["part"] == part]
        miss = [r for r in ins if not r["present"]]
        print(f"  part {part:3s}: jobs {len(js):3d} {cnt}; inputs {len(ins) - len(miss)}/{len(ins)} present")
        for j in js:
            if j["classification"] != "done":
                print(f"      job {j['job']}: {j['classification']}")
        for r in miss[:8]:
            print(f"      missing {r['kind']}: {r['file']}")
        if len(miss) > 8:
            print(f"      ... {len(miss) - 8} more missing")
    exc = [j for j in jobs if j["analysis_input"] == L.EXCLUDED_TEXT]
    print(f"  inputs {L.EXCLUDED_TEXT}: " + (", ".join(f"{j['job']} ({j['status']})" for j in exc) if exc else "none"))
    print("  b3: disabled (A15_B3_ENABLED = False); its registered condition is computed by part_b1.py "
          "(b3_condition.csv); nothing is enabled by the analysis")
    too_early = bool(open_jobs) and now < deadline
    if too_early:
        print(f"  NOT READY: {len(open_jobs)} A15 jobs are running or pending and the deadline has not passed; "
              "the analysis is registered to run once, when A15 is done or at the deadline (amendment item 5)")
    if ns.dry_run:
        return 0
    if ns.check_ready:
        return 4 if (too_early and not ns.force) else 0
    if too_early and not ns.force:
        print("inventory: stopping (exit 4); use --force to override", file=sys.stderr)
        return 4
    for j in jobs:
        j["analysis_time_utc"] = f"{now:%Y-%m-%d %H:%M:%S}"
    L.write_csv(jobs, os.path.join(ns.out, "inventory.csv"), """
        A15 jobs of queue v9 (scripts/tifs/queue_plan.a15_parts; b3 disabled), one row per job: kind, status
        marker in runs/tifs/_status (ok / failed / blocked / running / pending = no marker), classification
        (done; failed; cut = hand-written .blocked marker with every dependency done, the 10/27 amendment's cut
        method, order e2 -> e -> d; blocked by a dependency; running / not started at analysis time),
        dependencies, terminal = ok, failed or blocked; analysis_input (evaluation and ws re-score jobs) = used (job
        ok) / no files / present but job not terminal-ok: excluded (files of a running, failed or cut job are
        never read).""")
    L.write_csv(inputs, os.path.join(ns.out, "inventory_inputs.csv"), """
        Every file the A15 analysis reads (A15 runs' Stage 2 prediction and result files by the tags of
        scripts/tifs/aux/g8-a15/plan_check.json; ws re-scores; twin checks; the A15-c gate; CPU fp32 hits of
        score_em.py), present or missing at analysis time (file names only); job_ok = the job that writes the file
        is terminal-ok (Stage 2: <run>__eval.ok; ws: tifs_x15_ws__<run>.ok); a present file with job_ok False is
        not read.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
