#!/usr/bin/env python3
"""A12 compute-cost record of the TIFS runs (docs/tifs_addendum.md A12; descriptive, post hoc).

It changes no primary verdict, battery decision or the paper-wide Holm. CPU only and
read-only: it reads existing records, trains and evaluates nothing, and writes its tables
to results/tifs/addendum/aux/g5-cost/ (or --out).

Inputs (read only; paths relative to the repository root)
  runs/tifs/_status/events.jsonl      job state changes: time (UTC), lane, gpu, pid, hours, exit
  runs/tifs/_status/*.ok|failed|blocked  lane markers (time-zone check only)
  scripts/tifs/queue_plan.py          job definitions (imported; the import only reads the
                                      .running markers, of which none exist now)
  configs/tifs/<run>.yaml, configs/tifs/_dropped/<run>.yaml   teacher model (family), datasets
  logs/tifs/<job>.log                 one command header per attempt, stage summaries,
                                      per-teacher / per-student / per-cell progress lines
  runs/tifs/<run>/manifest.json       record of each run's last pipeline invocation
  runs/tifs/<run>/artifacts/students/*/*/kd_meta.json      student train_time_sec
  runs/tifs/<run>/artifacts/results/*.json, *.preds.json   Stage 2 cells (file times)
  results/tifs/teachers/{clean,poisoned}/*/ob_meta.json, REUSED.json   teacher train_time_sec
  results/tifs/addendum/gpu0_eval_determinism.json         GPU model; the A9 re-evaluation
  results/tifs/addendum/aux/g1-exposure-controls-tokens/trigger_tokens_meta.json  tokenizers
  docs/tifs_addendum.md               A10-a result times in KST (time-zone check)

Outputs
  cost_jobs.csv         one row per job attempt
  cost_summary.csv      successful (status ok) wall hours by stage x family x dataset x block
  cost_attempts.csv     hours of failed, requeued, yielded and stopped attempts
  cost_by_gpu.csv       per GPU: sum of attempt hours and the union of busy intervals
  cost_units.csv        one row per teacher build, KD student and evaluated cell
  cost_per_unit.csv     medians and ranges of the unit times by family and dataset
  hardware_software.json  GPU, host and software versions as recorded
  selfcheck.csv, provenance.csv, README.md

Run with CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 (run_all.sh does this).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # before any local import: no __pycache__ in the repository

import argparse  # noqa: E402
import ast  # noqa: E402
import csv  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import importlib.metadata as importlib_metadata  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from collections import Counter, OrderedDict, defaultdict  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402

if not sys.flags.dont_write_bytecode:
    raise SystemExit("run with PYTHONDONTWRITEBYTECODE=1 (nothing may be written inside the repo)")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only; nothing here imports torch

import yaml  # noqa: E402

GROUP = "g5-cost"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
RUNS = os.path.join(ROOT, "runs", "tifs")
STATUS = os.path.join(RUNS, "_status")
EVENTS = os.path.join(STATUS, "events.jsonl")
LOGS = os.path.join(ROOT, "logs", "tifs")
CONFIGS = os.path.join(ROOT, "configs", "tifs")
TEACHERS = os.path.join(ROOT, "results", "tifs", "teachers")
ADDENDUM = os.path.join(ROOT, "results", "tifs", "addendum")
AUX_OUT_ROOT = os.path.join(ADDENDUM, "aux")
OUT_DEFAULT = os.path.join(AUX_OUT_ROOT, GROUP)
DETERMINISM = os.path.join(ADDENDUM, "gpu0_eval_determinism.json")
TOKENS_META = os.path.join(AUX_OUT_ROOT, "g1-exposure-controls-tokens", "trigger_tokens_meta.json")
ADDENDUM_DOC = os.path.join(ROOT, "docs", "tifs_addendum.md")
PY_ENV = "/data/Youngho/envs/blade"

UTC = timezone.utc
KST = timezone(timedelta(hours=9))

# Frozen files (freeze_20261021T092041Z); recorded in provenance.csv, checked in selfcheck.csv.
FROZEN_SHA256 = {
    "docs/tifs_plan.md": "bc9dfd8f87f54d778a47e07b7d49de2c68af5b3df5458729fa4e7bb947f88713",
    "scripts/tifs/analyze.py": "1f70bbf4453bd83a5ec5f9858343549e89485627acb63f62d5e0e95dd5f09ef8",
    "counterfactual_kd/analysis/tifs.py": "cc0e05451bee502d888ab52260eb4576aab056dbe89a7c41fdce07db247dc9ce",
    "results/tifs/structural_nulls.json": "a6dc5091d33b0c355a34136b65e28f29163864aeb0a4d1c060afd0c33656a9b0",
}

# --- vocabularies -------------------------------------------------------------------------

STAGES = ("teacher build", "KD students", "evaluation", "other")
B_PRIMARY = "primary battery (seed 42-44)"
B_A1 = "A1"
B_A2 = "A2"
B_AA = "A/A"
B_PC = "positive control"
B_G = "contaminated corpus (G)"
B_A8 = "A8 hidden"
B_A10A = "A10-a"
B_RIPPLE = "RIPPLe (dropped)"
B_OTHER = "other"
BLOCKS = (B_PRIMARY, B_A1, B_A2, B_AA, B_PC, B_G, B_A8, B_A10A, B_RIPPLE, B_OTHER)
STATUSES = ("ok", "failed", "requeued", "yielded", "stopped")
FAMILY_OF_MODEL = {"bert-large-uncased": "bert", "bert-base-uncased": "bert", "gpt2-xl": "gpt2",
                   "gpt2": "gpt2", "Qwen/Qwen3-1.7B": "qwen3", "Qwen/Qwen3-0.6B": "qwen3"}
DATASETS = ("sst2", "agnews")
DS_NAME = {"sst2": "SST-2", "agnews": "AG News"}
HOURS_TOL = 0.005 + 3 / 3600   # event 'hours' are rounded to 0.01 h; plus a few seconds of bookkeeping
FILE_SLACK = 5.0               # seconds a file may be written after the closing event

# --- small helpers ------------------------------------------------------------------------


def rel(path):
    return os.path.relpath(path, ROOT)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_time(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC).timestamp()


def utc(ts):
    return "" if ts is None else datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d %H:%M:%S")


def kst(ts):
    return "" if ts is None else datetime.fromtimestamp(ts, KST).strftime("%Y-%m-%d %H:%M:%S")


def mtime(path):
    return os.stat(path).st_mtime


def r4(x):
    return "" if x is None else round(float(x), 4)


def write_csv(rows, path):
    if not rows:
        raise SystemExit(f"no rows for {path}")
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def union(iv):
    """Union of [s, e] intervals, sorted and merged."""
    out = []
    for s, e in sorted(iv):
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def length(iv):
    return sum(e - s for s, e in iv)


def overlap(merged, s, e):
    return sum(max(0.0, min(e, b) - max(s, a)) for a, b in merged)


def concurrency(iv):
    """Seconds spent at each number of concurrently running intervals."""
    pts = sorted([(s, 1) for s, e in iv if e > s] + [(e, -1) for s, e in iv if e > s],
                 key=lambda p: (p[0], p[1]))
    out, level, last = Counter(), 0, None
    for t, d in pts:
        if last is not None and t > last and level > 0:
            out[level] += t - last
        level += d
        last = t
    return out


def out_dir():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=OUT_DEFAULT,
                    help=f"output folder (default {rel(OUT_DEFAULT)})")
    out = os.path.abspath(ap.parse_args().out)
    inside = os.path.commonpath([out, ROOT]) == ROOT
    if inside and os.path.commonpath([out, AUX_OUT_ROOT]) != AUX_OUT_ROOT:
        raise SystemExit(f"refusing to write inside the repository outside {rel(AUX_OUT_ROOT)}/: {out}")
    for g in ("g1", "g2", "g3", "g4"):
        if inside and os.path.basename(out).startswith(g + "-"):
            raise SystemExit(f"refusing to write into an A11 folder: {out}")
    os.makedirs(out, exist_ok=True)
    return out


# --- inputs -------------------------------------------------------------------------------


def load_events():
    events = []
    with open(EVENTS) as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            e = json.loads(line)
            e["_line"] = i
            e["_t"] = parse_time(e["time"])
            events.append(e)
    return events


def load_queue_plan():
    """Import scripts/tifs/queue_plan.py. Its module code only defines job lists and reads
    runs/tifs/_status/<job>.running markers (_still_running); it writes nothing."""
    sys.path.insert(0, os.path.join(ROOT, "scripts", "tifs"))
    try:
        import queue_plan as qp  # noqa: E402
    finally:
        sys.path.pop(0)
    if not os.path.samefile(os.path.dirname(qp.__file__), os.path.join(ROOT, "scripts", "tifs")):
        raise SystemExit(f"queue_plan not loaded from this repository: {qp.__file__}")
    jobs = OrderedDict()
    for name, pool in qp.POOLS.items():
        for j in pool["jobs"]:
            jobs.setdefault(j["id"], dict(j, where=[]))["where"].append(f"pool {name}")
    for name, lane in qp.LANES.items():
        for j in lane.get("jobs", []):
            jobs.setdefault(j["id"], dict(j, where=[]))["where"].append(f"lane {name}")
    return qp, jobs


def load_config(run_id):
    for d in (CONFIGS, os.path.join(CONFIGS, "_dropped")):
        p = os.path.join(d, f"{run_id}.yaml")
        if os.path.exists(p):
            with open(p) as f:
                return yaml.safe_load(f), p
    return None, None


HDR_RE = re.compile(r"^=== (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (.*) \(GPU (\S+)\)\s*$")
STOP_RE = re.compile(r"^=== (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) stopped, GPU (\S+) in use: (.*)$")
TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3}")
OB_RE = re.compile(r"^\[OB-adapter\]\s+(poisoned|clean) \| (\S+)\s*\| (\S+) \| (\S+) \| seed=(\d+)\s*$")
OB_SKIP_RE = re.compile(r"^\[OB-adapter\] SKIP (.+?)\s*$")
S0_FAIL_RE = re.compile(r"^\[Stage0\] FAILED (\{.*?\}): (.*)$")
KD_SKIP_RE = re.compile(r"^\[KD-adapter\] SKIP (.+?)\s*$")
KD_TRAIN_RE = re.compile(r"^\[KD-adapter\] (\w+) \| (\w+) teacher → student\s*$")
KD_STEP_RE = re.compile(r"KD Epoch \d+/\d+ \[\s*[\d.]+%\] step \d+/(\d+)")
S2_RE = re.compile(r"^\[Stage2\] (.+?)\s*$")
SUMMARY_RE = re.compile(r"^  (stage\d)  (\{.*\})\s*$")
DS_IN_NAME_RE = re.compile(r"_(sst2|agnews)_seed\d+")


def parse_log(job):
    """Split logs/tifs/<job>.log into one segment per lane start ('=== <time> <cmd> (GPU n)')."""
    path = os.path.join(LOGS, f"{job}.log")
    if not os.path.exists(path):
        return path, []
    segs = []
    with open(path, errors="replace") as f:     # universal newlines also split tqdm's '\r'
        for line in f:
            line = line.rstrip("\n")
            m = STOP_RE.match(line)
            if m and segs:
                segs[-1]["stopped"] = (parse_time(m.group(1)), m.group(3))
                continue
            m = HDR_RE.match(line)
            if m:
                cmd = m.group(2)
                stages = []
                if "--stages" in cmd:
                    stages = cmd.split("--stages", 1)[1].split()
                segs.append({"t": parse_time(m.group(1)), "cmd": cmd, "gpu": m.group(3),
                             "stages": stages, "ob": [], "ob_skip": [], "s0_fail": [],
                             "kd_skip": [], "kd_train": [], "kd_steps": [], "s2": [],
                             "summaries": [], "run_complete": False, "failed_line": None,
                             "last_ts": None, "stopped": None, "path": path})
                continue
            if not segs:
                continue
            s = segs[-1]
            for ts in TS_RE.findall(line):
                s["last_ts"] = parse_time(ts)
            m = OB_RE.match(line)
            if m:
                s["ob"].append(tuple(m.groups()))
                continue
            m = OB_SKIP_RE.match(line)
            if m:
                s["ob_skip"].append(m.group(1))
                continue
            m = S0_FAIL_RE.match(line)
            if m:
                s["s0_fail"].append((ast.literal_eval(m.group(1)), m.group(2)[:160]))
                continue
            m = KD_SKIP_RE.match(line)
            if m:
                s["kd_skip"].append(m.group(1))
                continue
            m = KD_TRAIN_RE.match(line)
            if m:
                s["kd_train"].append(m.groups())
                s["kd_steps"].append(None)
                continue
            m = KD_STEP_RE.search(line)
            if m and s["kd_steps"]:
                s["kd_steps"][-1] = int(m.group(1))
                continue
            m = S2_RE.match(line)
            if m:
                s["s2"].append(m.group(1))
                continue
            m = SUMMARY_RE.match(line)
            if m:
                try:
                    s["summaries"].append((m.group(1), ast.literal_eval(m.group(2))))
                except (ValueError, SyntaxError):
                    s["summaries"].append((m.group(1), {"unparsed": m.group(2)[:200]}))
                continue
            if line.startswith("Run complete:"):
                s["run_complete"] = True
            elif line.startswith("FAILED:"):
                s["failed_line"] = line.strip()
    for i, s in enumerate(segs):
        s["last_segment"] = i == len(segs) - 1
    return path, segs


def load_teachers():
    out = []
    reused = json.load(open(os.path.join(TEACHERS, "REUSED.json")))
    reused_links = set(reused.get("links", []))
    for kind in ("poisoned", "clean"):
        for d in sorted(glob.glob(os.path.join(TEACHERS, kind, "*"))):
            meta_p = os.path.join(d, "ob_meta.json")
            m = json.load(open(meta_p))
            name = f"{kind}/{os.path.basename(d)}"
            out.append({
                "name": name, "dir": d, "real": os.path.realpath(d), "kind": kind,
                "link": os.path.islink(d), "reused_tier1": name in reused_links,
                "type": m.get("type"), "attack": m.get("attack"), "trainer": m.get("trainer"),
                "model": m.get("model_path"), "family": FAMILY_OF_MODEL.get(m.get("model_path"), "?"),
                "dataset": m.get("dataset"), "seed": m.get("seed"),
                "train_sec": float(m["train_time_sec"]) if m.get("train_time_sec") is not None else None,
                "end": mtime(meta_p), "meta": meta_p,
            })
    return out, reused


def load_students():
    out = []
    for mp in sorted(glob.glob(os.path.join(RUNS, "tifs_*", "artifacts", "students", "*", "*", "kd_meta.json"))):
        d = os.path.dirname(mp)
        m = json.load(open(mp))
        run = mp.split(os.sep)[-6]
        teacher = None
        if m.get("teacher_ref"):
            teacher = os.path.realpath(os.path.normpath(os.path.join(d, m["teacher_ref"])))
        role = {"poisoned_student": "P", "clean_student": "C", "ce_student": "R"}.get(os.path.basename(d), "?")
        out.append({
            "name": f"{os.path.basename(os.path.dirname(d))}/{os.path.basename(d)}", "run": run, "dir": d,
            "role": role, "kd": m.get("kd_type"), "family": FAMILY_OF_MODEL.get(m.get("student_model"), "?"),
            "dataset": m.get("dataset"), "seed": m.get("student_seed", m.get("seed")),
            "teacher_seed": m.get("teacher_seed"), "teacher_kind": m.get("teacher_kind"),
            "teacher_real": teacher, "corpus_attack": m.get("corpus_attack"),
            "train_sec": float(m["train_time_sec"]) if m.get("train_time_sec") is not None else None,
            "end": mtime(mp), "meta": mp, "tf32": m.get("tf32"), "teacher_dtype": m.get("teacher_dtype"),
            "student_dtype": m.get("student_dtype"),
        })
    return out


def load_cells():
    out = []
    for pp in sorted(glob.glob(os.path.join(RUNS, "tifs_*", "artifacts", "results", "*.preds.json"))):
        jp = pp[: -len(".preds.json")] + ".json"
        m = json.load(open(jp)) if os.path.exists(jp) else {}
        run = pp.split(os.sep)[-4]
        end = max(mtime(pp), mtime(jp)) if os.path.exists(jp) else mtime(pp)
        out.append({
            "name": os.path.basename(pp)[: -len(".preds.json")], "run": run, "path": pp,
            "family": str(m.get("model_family", "?")).lower(), "dataset": m.get("dataset", "?"),
            "kd": m.get("kd_type", "?"), "attack": m.get("attack_method", "?"),
            "seed": m.get("student_seed", m.get("seed")), "teacher_seed": m.get("teacher_seed"),
            "end": end, "has_json": os.path.exists(jp),
        })
    return out


def load_manifests():
    out = {}
    for mp in sorted(glob.glob(os.path.join(RUNS, "tifs_*", "manifest.json"))):
        m = json.load(open(mp))
        out[mp.split(os.sep)[-2]] = (m, mp)
    return out


# --- attempts -----------------------------------------------------------------------------

LANE_TERMINAL = ("ok", "failed")
NOTE_CLOSE = ("stopped", "dropped")


def build_attempts(events):
    attempts, open_by_job, closed_by_job = [], {}, defaultdict(list)
    other = defaultdict(list)          # non-attempt events by job
    for e in events:
        job, st = e["job"], e["state"]
        if st == "running":
            prev = open_by_job.pop(job, None)
            if prev is not None:       # the lane that started it was replaced before it finished
                prev["next_event"] = e
                closed_by_job[job].append(prev)
            a = {"job": job, "start": e["_t"], "start_line": e["_line"], "lane": e.get("lane", ""),
                 "gpu": str(e.get("gpu", "")), "pid": e.get("pid", ""), "host": e.get("host", ""),
                 "run": e.get("run") or (job[: -len("__eval")] if job.endswith("__eval") else job),
                 "mem_gb": e.get("mem_gb", ""), "close": None, "next_event": None,
                 "after_notes": [], "source": "events.jsonl"}
            open_by_job[job] = a
            attempts.append(a)
        elif st in LANE_TERMINAL or st in NOTE_CLOSE or (st == "requeued" and "hours" in e):
            a = open_by_job.pop(job, None)
            if a is None:
                other[job].append(e)   # a marker without a running attempt (tifs_rule_frozen, tifs_a10b_go)
                continue
            a["close"] = e
            closed_by_job[job].append(a)
        elif st == "requeued":         # a note written by hand after a failed or stopped attempt
            if closed_by_job[job]:
                closed_by_job[job][-1]["after_notes"].append(e)
            other[job].append(e)
        else:
            other[job].append(e)
    if open_by_job:
        raise SystemExit(f"attempts without an end: {sorted(open_by_job)}")
    seen = Counter()
    for a in attempts:
        seen[a["job"]] += 1
        a["attempt"] = seen[a["job"]]
    for a in attempts:
        a["n_attempts_job"] = seen[a["job"]]
    return attempts, other


def attach_segments(attempts):
    logs = {}
    for a in attempts:
        if a["job"] not in logs:
            logs[a["job"]] = parse_log(a["job"])
        path, segs = logs[a["job"]]
        cands = [s for s in segs if abs(s["t"] - a["start"]) <= 5]
        a["log"] = rel(path) if os.path.exists(path) else ""
        a["seg"] = cands[0] if len(cands) == 1 else None
        a["seg_matches"] = len(cands)
    return logs


def closed_ends(attempts):
    """End time from the closing event. Attempts without one get the next start of the same
    job as a provisional end (an upper bound), refined by final_ends() once units are known."""
    for a in attempts:
        c = a["close"]
        a["provisional"] = c is None
        if c is not None:
            a["end"] = c["_t"]
            if c["state"] in LANE_TERMINAL or c["state"] == "requeued":
                a["end_source"] = "event (lane)"
                continue
            # Closed by a note written by hand after the process was stopped. The job log's last
            # write is the process's last output; use it when it falls inside the attempt.
            a["end_source"] = f"'{c['state']}' event written by hand"
            seg = a["seg"]
            if seg and seg["last_segment"] and os.path.exists(seg["path"]):
                last_write = mtime(seg["path"])
                if a["start"] < last_write <= c["_t"]:
                    a["end"] = last_write
                    a["end_source"] = (f"last write to its job log (the process's last output), "
                                       f"{c['_t'] - last_write:.0f} s before the '{c['state']}' event "
                                       f"written by hand at {utc(c['_t'])} UTC")
        else:
            a["end"] = a["next_event"]["_t"]


def final_ends(attempts, manifests):
    """No closing event: the lane that started the job was replaced while the pipeline ran on,
    and the next lane start of the same job bounds the end. Use the newest record of the
    attempt's own activity before that: its log segment, the files it wrote (attributed
    units) and the run's manifest if this invocation wrote it."""
    for a in attempts:
        if a["provisional"]:
            upper = a["end"]
            ev = []
            seg = a["seg"]
            if seg and seg["last_ts"] and a["start"] <= seg["last_ts"] <= upper:
                ev.append((seg["last_ts"], "last time stamp in its log segment"))
            for u in a["teachers"] + a["students"] + a["cells"]:
                ev.append((u["end"], "newest file it wrote"))
            man = manifests.get(a["run"])
            if man:
                st, en = (datetime.fromisoformat(man[0][k]).replace(tzinfo=UTC).timestamp()
                          for k in ("started", "ended"))
                if a["start"] <= st <= upper and en <= upper:
                    ev.append((en, "run manifest 'ended' written by this invocation"))
            if not ev:
                raise SystemExit(f"no end evidence for {a['job']} attempt {a['attempt']}")
            a["end"], src = max(ev)
            a["end_source"] = (f"no closing event (lane replaced while the pipeline ran); {src}; "
                               f"before the next start {utc(upper)} UTC")
        a["wall_h"] = (a["end"] - a["start"]) / 3600.0


def status_of(a, qp):
    """Normalised status, raw state, exit, reason, note."""
    c = a["close"]
    seg = a["seg"] or {}
    if c is None:
        summ = dict(seg.get("summaries", []))
        stage_ok = all(v.get("ok", False) for v in summ.values()) if summ else False
        if seg.get("run_complete") and stage_ok:
            return "ok", "none", "", "", ("no closing event: the lane process was replaced while the "
                                          "pipeline ran; the job log shows 'Run complete' with ok=True")
        if seg.get("run_complete"):
            n = sum(v.get("failures", 0) for v in summ.values())
            fails = "; ".join(f"{d.get('attack')} {d.get('model')} {d.get('dataset')}: {err[:60]}"
                              for d, err in seg.get("s0_fail", []))
            return "failed", "none", "", f"{n} condition(s) failed: {fails}", (
                "no closing event: the lane process was replaced while the pipeline ran; the job "
                f"log shows 'Run complete' with failures ({seg.get('failed_line') or ''})")
        return "stopped", "none", "", "", "no closing event and no 'Run complete' in the log"
    st, ex = c["state"], c.get("exit", "")
    reason = c.get("reason") or c.get("note") or ""
    if st == "ok":
        return "ok", st, ex, reason, ""
    if st == "failed":
        if a["job"] == qp.PC_ALPHA1_GO["id"] and ex == 3:
            return "ok", st, ex, reason, ("exit 3 is the registered 'not needed' outcome of the "
                                          "pre-registered alpha=1 decision (queue_plan.PC_ALPHA1_GO): "
                                          "the lane marks it failed so that tifs_pc_bert_kl10a1 is "
                                          "blocked; a decision, not a compute failure")
        notes = " ".join(n.get("note", "") for n in a["after_notes"])
        if ex == -15 and "by hand" in notes:
            return "stopped", st, ex, notes, "SIGTERM by hand (exit -15); requeued by hand"
        return "failed", st, ex, notes or reason, ("requeued by hand: " + notes) if notes else ""
    if st == "requeued":
        if reason.startswith("yielded GPU"):
            return "yielded", st, ex, reason, ("exclusive GPU 1 lane stopped the job because another "
                                               "user's process appeared (A0.1)")
        return "requeued", st, ex, reason, ""
    if st in NOTE_CLOSE:
        return "stopped", st, ex, reason, f"closed by a hand-written '{st}' event"
    return "?", st, ex, reason, "unknown closing state"


def classify(a, cfg):
    """stage, family, dataset, block, block detail, uses_gpu."""
    job, run = a["job"], a["run"]
    seg = a["seg"] or {}
    cmd = seg.get("cmd", "")
    is_pipeline = "counterfactual_kd run" in cmd
    family = "-"
    datasets = []
    if cfg:
        fam = {FAMILY_OF_MODEL.get(p[0], "?") for p in cfg.get("pairs", [])}
        family = "+".join(sorted(fam))
        datasets = [d for d in DATASETS if d in cfg.get("datasets", [])]
    stages = seg.get("stages", [])
    if not is_pipeline and a["source"] == "events.jsonl":
        stage = "other"
    elif stages == ["stage0"]:
        stage = "teacher build"
    elif stages == ["stage1"]:
        stage = "KD students"
    elif stages == ["stage2"]:
        stage = "evaluation"
    elif seg.get("kd_train") or seg.get("kd_skip"):
        stage = "KD students"       # v1 job without --stages: Stage 0 skipped, stopped in Stage 1
    else:
        stage = "?"
    # datasets this attempt actually worked on (teacher builds: the teachers it tried)
    if stage == "teacher build" and seg.get("ob"):
        datasets = [d for d in DATASETS if d in {o[3] for o in seg["ob"]}]
    dataset = "+".join(datasets) if datasets else "-"
    detail = ""
    if re.match(r"^tifs_b_bert_agnews_s4[34](__eval)?$", job):
        block, detail = B_A1, "BERT AG News student seeds 43/44 (logit)"
    elif re.match(r"^tifs_b_", run):
        block, detail = B_PRIMARY, "battery run"
    elif re.match(r"^tifs_(t|ts|tb)_", job):
        if job == "tifs_t_gpt2" and datasets == ["agnews"]:
            block, detail = B_OTHER, ("GPT-2 AG News teachers; the GPT-2 AG News battery was "
                                      "dropped in v2 (event 'dropped' 10/21 01:31 UTC)")
        else:
            block, detail = B_PRIMARY, ("seed-42 teachers, also used by the contaminated-corpus, "
                                        "positive-control, A1, A8 and A10-a runs")
    elif re.match(r"^tifs_(tr|br)_", run):
        block, detail = B_RIPPLE, "RIPPLe SST-2 teachers; students never run (A7)"
    elif run.startswith("tifs_rep_bert_atba_agnews_"):
        block, detail = B_OTHER, "A10-b (not run; blocked by its gate)"
    elif run.startswith("tifs_rep_"):
        block, detail = B_A2, "ATBA teacher-seed-43 replicate"
    elif run.startswith("tifs_aa_"):
        block, detail = B_AA, ("A/A teachers (the t43 clean teacher is also A2's)"
                               if stage == "teacher build" else "A/A students")
    elif job == "tifs_pc_kl10a1_go":
        block, detail = B_PC, "pre-registered alpha=1 go decision (CPU step in a GPU 0 pool slot)"
    elif run.startswith("tifs_pc_"):
        block, detail = B_PC, "label-kept trigger exposure"
    elif run == "tifs_g_bert_clean_s42":
        block, detail = B_OTHER, ("v1 main-grid run (clean corpus) on GPU 1, stopped 10/21 by the "
                                  "author's GPU 1 training ban; dropped in v2; its five finished "
                                  "SST-2 students were kept and reused by tifs_b_bert_sst2")
    elif run.startswith("tifs_g_"):
        block, detail = B_G, "label-poisoned 10% KD corpus"
    elif run.startswith("tifs_hid_"):
        block, detail = B_A8, "TinyBERT-style hidden-state KD"
    elif run.startswith("tifs_rcc_"):
        block, detail = B_A10A, "CE-coefficient check (R_B)"
    elif job == "tifs_synbkd_table":
        block, detail = B_OTHER, ("queue step that checks the SynBkd paraphrase table (built on CPU "
                                  "before the queue started; CPU step in a GPU 0 lane)")
        family, dataset = "-", "sst2+agnews"
    elif a["source"] != "events.jsonl":
        block, detail = B_OTHER, a.get("detail", "")
    else:
        block, detail = "?", "unclassified"
    uses_gpu = "yes" if (is_pipeline or a["source"] != "events.jsonl") else "no"
    return stage, family, dataset, datasets, block, detail, uses_gpu


# --- units --------------------------------------------------------------------------------


def key_label(k):
    """Readable label of a teacher key (kind, attack, model, dataset, seed)."""
    kind, attack, model, ds, seed = k
    what = "clean (base trainer)" if (kind == "clean" and attack == "badnets") else f"{kind} {attack}"
    return f"{what} {model} {ds} seed {seed}"


def teacher_key(t):
    return ("poisoned" if t["type"] == "poisoned_teacher" else "clean", t["attack"], t["model"],
            t["dataset"], int(t["seed"]))


def attribute_units(attempts, teachers, students, cells):
    """Assign every teacher, student and cell to the attempt that wrote it."""
    problems = []
    by_key = {teacher_key(t): t for t in teachers if not t["reused_tier1"]}
    for t in teachers:
        t["attempt"] = None
    for a in attempts:
        a["teachers"], a["students"], a["cells"] = [], [], []
        a["teacher_failed"], a["teacher_inprogress"], a["ob_outcomes"] = [], None, []
    # teachers: the attempt whose log tried to build it and whose window holds ob_meta.json
    for a in attempts:
        seg = a["seg"]
        if not seg or not seg["ob"]:
            continue
        failed = {(d["kind"], d["attack"], d["model"], d["dataset"], int(d["seed"]))
                  for d, _ in seg["s0_fail"]}
        for kind, attack, model, ds, seed in seg["ob"]:
            key = (kind, attack, model, ds, int(seed))
            t = by_key.get(key)
            if key in failed:
                a["teacher_failed"].append(key)
                a["ob_outcomes"].append((key, "failed"))
            elif t is not None and a["start"] <= t["end"] <= a["end"] + FILE_SLACK:
                if t["attempt"] is not None:
                    problems.append(f"teacher {t['name']} claimed twice")
                t["attempt"] = a
                a["teachers"].append(t)
                a["ob_outcomes"].append((key, "built"))
            else:
                a["teacher_inprogress"] = key
                a["ob_outcomes"].append((key, "in progress"))
    for t in teachers:
        if t["reused_tier1"]:
            continue
        if t["attempt"] is None:
            problems.append(f"teacher {t['name']} not attributed")
    # students: the KD attempt whose window holds kd_meta.json (same run first)
    kd_attempts = [a for a in attempts if a["stage"] == "KD students"]
    for s in students:
        c = [a for a in kd_attempts if a["start"] <= s["end"] <= a["end"] + FILE_SLACK]
        same = [a for a in c if a["run"] == s["run"]]
        pick = same if same else c
        if len(pick) != 1:
            problems.append(f"student {s['run']}/{s['name']}: {len(pick)} candidate attempts")
            s["attempt"] = None
            continue
        s["attempt"] = pick[0]
        s["moved"] = not same
        pick[0]["students"].append(s)
    # cells: an evaluation attempt of the run's own eval job whose window holds the files
    ev_attempts = defaultdict(list)
    for a in attempts:
        if a["stage"] == "evaluation":
            ev_attempts[a["run"]].append(a)
    for c in cells:
        cand = [a for a in ev_attempts.get(c["run"], []) if a["start"] <= c["end"] <= a["end"] + FILE_SLACK]
        if len(cand) != 1:
            problems.append(f"cell {c['run']}/{c['name']}: {len(cand)} candidate attempts")
            c["attempt"] = None
            continue
        c["attempt"] = cand[0]
        cand[0]["cells"].append(c)
    # per-cell duration: time since the previous cell of the same attempt (or its start)
    for a in attempts:
        prev = a["start"]
        for i, c in enumerate(sorted(a["cells"], key=lambda x: x["end"])):
            c["dur_sec"] = c["end"] - prev
            c["first_in_attempt"] = i == 0
            prev = c["end"]
    return problems


def mark_evaluated(students, cells):
    """A student counts as evaluated if a Stage 2 cell of its run used it: for a P student the
    cell of the same name; for a C or R student a cell with its KD objective, dataset and seed."""
    by_run = defaultdict(list)
    for c in cells:
        by_run[c["run"]].append(c)
    for s in students:
        cs = by_run.get(s["run"], [])
        stem = s["name"].split("/")[0]
        if s["role"] == "P":
            s["evaluated"] = any(c["name"] == stem for c in cs)
        else:
            s["evaluated"] = any(c["kd"] == s["kd"] and c["dataset"] == s["dataset"]
                                 and str(c["seed"]) == str(s["seed"]) for c in cs)


def unit_interval(u, kind):
    if kind == "cell":
        return u["end"] - u["dur_sec"], u["end"]
    return u["end"] - (u["train_sec"] or 0.0), u["end"]


def apportion(a):
    """Split an attempt's wall hours by dataset in proportion to the recorded unit times it
    wrote; a non-ok attempt's tail after its last unit goes to the unit in progress."""
    w = Counter()
    for t in a["teachers"]:
        w[t["dataset"]] += t["train_sec"] or 0.0
    for s in a["students"]:
        w[s["dataset"]] += s["train_sec"] or 0.0
    for c in a["cells"]:
        w[c["dataset"]] += c["dur_sec"]
    basis = "recorded unit times" if w else ""
    if a["status"] != "ok":
        ends = [u["end"] for u in a["teachers"] + a["students"] + a["cells"]]
        tail = a["end"] - (max(ends) if ends else a["start"])
        ds = inprogress_dataset(a)
        if tail > 0:
            if ds:
                w[ds] += tail
                basis = (basis + " + " if basis else "") + f"time after the last unit -> {ds} (unit in progress, from the log)"
            else:
                w["unattributed"] += tail
                basis = (basis + " + " if basis else "") + "time after the last unit unattributed"
    if not w and len(a["datasets"]) == 1:
        w[a["datasets"][0]] = 1.0
        basis = "single dataset"
    if not w:
        for d in a["datasets"] or ["unattributed"]:
            w[d] = 1.0
        basis = "no unit recorded; equal split over the job's datasets"
    tot = sum(w.values())
    out = {d: a["wall_h"] * w.get(d, 0.0) / tot for d in DATASETS + ("unattributed",)}
    return out, basis


def inprogress_dataset(a):
    """Dataset of the unit the attempt was working on after its last finished unit."""
    seg = a["seg"] or {}
    if a["stage"] == "teacher build" and a["ob_outcomes"]:
        last_built = max((i for i, (_, o) in enumerate(a["ob_outcomes"]) if o == "built"), default=-1)
        rest = [k for k, o in a["ob_outcomes"][last_built + 1:] if o != "built"]
        return rest[-1][3] if rest else None
    if a["stage"] == "evaluation" and seg.get("s2"):
        written = {c["name"] for c in a["cells"]}
        rest = [n for n in seg["s2"] if n not in written]
        if rest:
            m = DS_IN_NAME_RE.search(rest[-1])
            return m.group(1) if m else None
    if a["stage"] == "KD students" and seg.get("kd_steps"):
        # steps per epoch of the student in progress against those of the finished students
        steps = seg["kd_steps"]
        done = sorted(a["students"], key=lambda s: s["end"])
        if len(steps) == len(done) + 1 and len(a["datasets"]) == 2:
            known = {steps[i]: s["dataset"] for i, s in enumerate(done)}
            cur = steps[-1]
            if cur in known:
                return known[cur]
            seen = set(known.values())
            if len(seen) == 1:
                other = [d for d in a["datasets"] if d not in seen]
                return other[0] if len(other) == 1 else None
    return None


# --- main ---------------------------------------------------------------------------------


def main():
    t_start = time.time()
    out = out_dir()
    checks = []

    def check(name, item, value, limit, ok, note=""):
        checks.append({"check": name, "item": item, "value": value, "limit": limit,
                       "ok": "yes" if ok else "NO", "note": note})

    events = load_events()
    qp, qjobs = load_queue_plan()
    attempts, other_events = build_attempts(events)
    logs = attach_segments(attempts)
    teachers, reused = load_teachers()
    students = load_students()
    cells = load_cells()
    manifests = load_manifests()
    determinism = json.load(open(DETERMINISM))

    # The A9 GPU 0 re-evaluation ran outside the lanes (scratch output, not in events.jsonl).
    rc = determinism["recheck"]
    m = re.search(r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2})-(\d{2}:\d{2}) UTC.*?([\d.]+) s", rc)
    a9_start = parse_time(f"{m.group(1)} {m.group(2)}:00")
    a9_sec = float(m.group(4))
    n_a9_cells = sum(1 for f in determinism["files"] if f["file"].endswith(".preds.json"))
    a9 = {"job": f"A9 GPU 0 re-evaluation of {determinism['run']}", "start": a9_start,
          "end": a9_start + a9_sec, "end_source": f"{rel(DETERMINISM)} 'recheck': {rc}",
          "lane": "by hand (outside the lanes)", "gpu": "0", "pid": "", "host": "",
          "run": determinism["run"], "mem_gb": "", "close": None, "next_event": None,
          "after_notes": [], "source": rel(DETERMINISM), "attempt": 1, "n_attempts_job": 1,
          "log": "", "seg": None, "seg_matches": 0,
          "detail": ("A9 determinism check before evaluating on GPU 0: Stage 2 of "
                     f"{determinism['run']} into a scratch run directory ({n_a9_cells} cells; "
                     "output not kept under runs/)")}

    closed_ends(attempts)
    for a in attempts:
        a["status"], a["event_state"], a["exit"], a["reason"], a["note"] = status_of(a, qp)
        cfg, cfg_path = load_config(a["run"])
        a["cfg_path"] = rel(cfg_path) if cfg_path else ""
        (a["stage"], a["family"], a["dataset"], a["datasets"], a["block"], a["block_detail"],
         a["uses_gpu"]) = classify(a, cfg)
    a9.update(status="ok", event_state="none (not a lane job)", exit="", reason="", note="",
              stage="evaluation", family="bert", dataset="sst2", datasets=["sst2"], block=B_OTHER,
              block_detail=a9["detail"], uses_gpu="yes", cfg_path="", provisional=False)
    problems = attribute_units(attempts, teachers, students, cells)
    mark_evaluated(students, cells)
    final_ends(attempts, manifests)
    a9["wall_h"] = a9_sec / 3600.0
    a9.update(teachers=[], students=[], cells=[], teacher_failed=[], teacher_inprogress=None,
              ob_outcomes=[], n_cells_external=n_a9_cells)
    all_attempts = attempts + [a9]

    # used-by counts for teachers
    used = Counter(s["teacher_real"] for s in students if s["teacher_real"])
    for t in teachers:
        t["used_by"] = used.get(t["real"], 0)

    # shared fraction: time another of our attempts ran on the same GPU
    by_gpu = defaultdict(list)
    for a in all_attempts:
        if a["uses_gpu"] == "yes":
            by_gpu[a["gpu"]].append(a)

    def shared_frac(a, s, e):
        if e <= s or a["uses_gpu"] != "yes":
            return ""
        others = union([(b["start"], b["end"]) for b in by_gpu[a["gpu"]] if b is not a])
        return overlap(others, s, e) / (e - s)

    for a in all_attempts:
        a["shared_frac"] = shared_frac(a, a["start"], a["end"])
        a["apport"], a["apport_basis"] = apportion(a) if a is not a9 else (
            {"sst2": a9["wall_h"], "agnews": 0.0, "unattributed": 0.0}, "single dataset")

    # ------------------------------------------------------------------ cost_jobs.csv
    def units_of(a):
        if a["stage"] == "teacher build":
            return "teachers", len(a["teachers"])
        if a["stage"] == "KD students":
            return "students", len(a["students"])
        if a["stage"] == "evaluation":
            return "cells", len(a["cells"]) if a is not a9 else n_a9_cells
        return "", ""

    def unit_hours(a):
        return (sum(t["train_sec"] or 0 for t in a["teachers"]) + sum(s["train_sec"] or 0 for s in a["students"])
                + sum(c["dur_sec"] for c in a["cells"])) / 3600.0

    job_rows = []
    for a in sorted(all_attempts, key=lambda x: (x["start"], x["job"])):
        seg = a["seg"] or {}
        unit, n = units_of(a)
        summ = dict(seg.get("summaries", []))
        n_cond = ""
        for k in ("stage2", "stage1"):
            if k in summ:
                n_cond = summ[k].get("conditions_evaluated", summ[k].get("student_pairs", ""))
        n_skip = len(seg.get("ob_skip", [])) + len(seg.get("kd_skip", []))
        kept = ""
        if a["status"] != "ok":
            parts = []
            if a["teachers"]:
                parts.append(f"{len(a['teachers'])} teacher(s) kept: " + ", ".join(
                    f"{t['name']} (used by {t['used_by']} students)" for t in a["teachers"]))
            if a["students"]:
                parts.append(f"{len(a['students'])} student(s) kept in runs/tifs/"
                             + ",".join(sorted({s['run'] for s in a['students']})))
            if a["cells"]:
                parts.append(f"{len(a['cells'])} cell(s) kept")
            if a["teacher_failed"]:
                parts.append("teacher build(s) failed: " + ", ".join(key_label(k) for k in a["teacher_failed"]))
            if a["teacher_inprogress"]:
                parts.append("in progress when it ended: " + key_label(a["teacher_inprogress"]))
            if a["stage"] in ("KD students", "evaluation"):
                ds = inprogress_dataset(a)
                if ds:
                    parts.append(f"an {DS_NAME[ds]} {'student' if a['stage'] == 'KD students' else 'cell'} was in progress when it ended (from the log)")
            kept = "; ".join(parts) or "nothing kept"
        a["kept"] = kept
        job_rows.append({
            "job": a["job"], "attempt": a["attempt"], "attempts_of_job": a["n_attempts_job"],
            "run": a["run"], "stage": a["stage"], "family": a["family"], "dataset": a["dataset"],
            "block": a["block"], "block_detail": a["block_detail"], "status": a["status"],
            "event_state": a["event_state"], "exit": a["exit"], "gpu": a["gpu"], "lane": a["lane"],
            "uses_gpu": a["uses_gpu"], "start_utc": utc(a["start"]), "end_utc": utc(a["end"]),
            "start_kst": kst(a["start"]), "end_kst": kst(a["end"]), "wall_hours": r4(a["wall_h"]),
            "hours_event": a["close"].get("hours", "") if a["close"] else "",
            "end_source": a["end_source"], "units": unit, "n_written": n,
            "n_skipped_in_log": n_skip if a["seg"] else "", "n_conditions_in_log": n_cond,
            "unit_hours_recorded": r4(unit_hours(a)) if a is not a9 else "",
            "shared_frac": r4(a["shared_frac"]) if a["shared_frac"] != "" else "",
            "hours_sst2": r4(a["apport"]["sst2"]), "hours_agnews": r4(a["apport"]["agnews"]),
            "hours_unattributed": r4(a["apport"]["unattributed"]), "apportion_basis": a["apport_basis"],
            "mem_gb_declared": a["mem_gb"], "pid": a["pid"], "host": a["host"],
            "source": a["source"], "log": a["log"], "config": a["cfg_path"],
            "reason": a["reason"], "kept_outputs": kept, "note": a["note"],
        })
    write_csv(job_rows, os.path.join(out, "cost_jobs.csv"))

    # ------------------------------------------------------------------ cost_summary.csv
    ok = [a for a in all_attempts if a["status"] == "ok"]

    def agg(rows_, **keys):
        r = dict(keys)
        r["n_attempts"] = len(rows_)
        r["n_jobs"] = len({a["job"] for a in rows_})
        r["wall_hours"] = sum(a["wall_h"] for a in rows_)
        r["gpu0_hours"] = sum(a["wall_h"] for a in rows_ if a["gpu"] == "0" and a["uses_gpu"] == "yes")
        r["gpu1_hours"] = sum(a["wall_h"] for a in rows_ if a["gpu"] == "1" and a["uses_gpu"] == "yes")
        r["cpu_step_hours"] = sum(a["wall_h"] for a in rows_ if a["uses_gpu"] != "yes")
        for d in DATASETS + ("unattributed",):
            r[f"hours_{d}"] = sum(a["apport"][d] for a in rows_)
        r["n_teachers"] = sum(len(a["teachers"]) for a in rows_)
        r["n_students"] = sum(len(a["students"]) for a in rows_)
        r["n_cells"] = sum(len(a["cells"]) for a in rows_) + sum(n_a9_cells for a in rows_ if a is a9)
        return r

    def rnd(r):
        return {k: (r4(v) if isinstance(v, float) else v) for k, v in r.items()}

    summ_rows = []
    groups = defaultdict(list)
    for a in ok:
        groups[(STAGES.index(a["stage"]), a["family"], a["dataset"], BLOCKS.index(a["block"]))].append(a)
    for (si, fam, ds, bi), rows_ in sorted(groups.items()):
        summ_rows.append(agg(rows_, row_type="detail", stage=STAGES[si], family=fam, dataset=ds, block=BLOCKS[bi]))
    for si, stg in enumerate(STAGES):
        fams = sorted({a["family"] for a in ok if a["stage"] == stg})
        for fam in fams:
            rows_ = [a for a in ok if a["stage"] == stg and a["family"] == fam]
            summ_rows.append(agg(rows_, row_type="stage x family", stage=stg, family=fam, dataset="all", block="all"))
        rows_ = [a for a in ok if a["stage"] == stg]
        if rows_:
            summ_rows.append(agg(rows_, row_type="stage", stage=stg, family="all", dataset="all", block="all"))
    for fam in sorted({a["family"] for a in ok}):
        summ_rows.append(agg([a for a in ok if a["family"] == fam], row_type="family", stage="all",
                             family=fam, dataset="all", block="all"))
    for bi, blk in enumerate(BLOCKS):
        rows_ = [a for a in ok if a["block"] == blk]
        if rows_:
            summ_rows.append(agg(rows_, row_type="block", stage="all", family="all", dataset="all", block=blk))
    summ_rows.append(agg(ok, row_type="overall (status ok)", stage="all", family="all", dataset="all", block="all"))
    summ_rows.append(agg(all_attempts, row_type="overall (all attempts = ok + cost_attempts.csv)",
                         stage="all", family="all", dataset="all", block="all"))
    # The same margins over all attempts (ok + failed, yielded, stopped): a failed attempt can
    # leave used outputs (tifs_t_gpt2 #1 built the GPT-2 SST-2 teachers), so 'ok' alone
    # understates some stage x family cells.
    for stg in STAGES:
        for fam in sorted({a["family"] for a in all_attempts if a["stage"] == stg}):
            summ_rows.append(agg([a for a in all_attempts if a["stage"] == stg and a["family"] == fam],
                                 row_type="stage x family (all attempts)", stage=stg, family=fam,
                                 dataset="all", block="all"))
        rows_ = [a for a in all_attempts if a["stage"] == stg]
        if rows_:
            summ_rows.append(agg(rows_, row_type="stage (all attempts)", stage=stg, family="all",
                                 dataset="all", block="all"))
    for bi, blk in enumerate(BLOCKS):
        rows_ = [a for a in all_attempts if a["block"] == blk]
        if rows_:
            summ_rows.append(agg(rows_, row_type="block (all attempts)", stage="all", family="all",
                                 dataset="all", block=blk))
    write_csv([rnd(r) for r in summ_rows], os.path.join(out, "cost_summary.csv"))

    # ------------------------------------------------------------------ cost_attempts.csv
    bad = [a for a in all_attempts if a["status"] != "ok"]

    def kept_split(a):
        """Recorded unit hours of the artifacts a non-ok attempt left: later used / never used.
        A teacher is used if some student distilled from it; a student if a Stage 2 cell of
        its run evaluated it; every kept cell is part of the final Stage 2 output."""
        used_h = (sum(t["train_sec"] or 0 for t in a["teachers"] if t["used_by"])
                  + sum(s["train_sec"] or 0 for s in a["students"] if s["evaluated"])
                  + sum(c["dur_sec"] for c in a["cells"])) / 3600.0
        unused_h = (sum(t["train_sec"] or 0 for t in a["teachers"] if not t["used_by"])
                    + sum(s["train_sec"] or 0 for s in a["students"] if not s["evaluated"])) / 3600.0
        return used_h, unused_h

    att_rows = []
    for a in sorted(bad, key=lambda x: x["start"]):
        used_h, unused_h = kept_split(a)
        used_down = []
        if a["teachers"]:
            used_down.append(f"teachers used by students: {sum(1 for t in a['teachers'] if t['used_by'])}/{len(a['teachers'])}")
        if a["students"]:
            used_down.append(f"students evaluated in Stage 2: {sum(1 for s in a['students'] if s['evaluated'])}/{len(a['students'])}")
        if a["cells"]:
            used_down.append(f"cells kept in the final Stage 2 output: {len(a['cells'])}")
        att_rows.append({
            "row_type": "attempt", "job": a["job"], "attempt": a["attempt"], "stage": a["stage"],
            "family": a["family"], "dataset": a["dataset"], "block": a["block"], "status": a["status"],
            "event_state": a["event_state"], "exit": a["exit"], "gpu": a["gpu"], "lane": a["lane"],
            "start_utc": utc(a["start"]), "end_utc": utc(a["end"]), "wall_hours": r4(a["wall_h"]),
            "reason": a["reason"], "kept_outputs": a["kept"], "kept_used_unit_hours": r4(used_h),
            "kept_unused_unit_hours": r4(unused_h), "kept_used_downstream": "; ".join(used_down),
            "note": a["note"],
        })
    for st in STATUSES[1:]:
        rows_ = [a for a in bad if a["status"] == st]
        for g in sorted({a["gpu"] for a in rows_}):
            rg = [a for a in rows_ if a["gpu"] == g]
            att_rows.append({"row_type": "total by status x GPU", "status": st, "gpu": g,
                             "attempt": len(rg), "wall_hours": r4(sum(a["wall_h"] for a in rg))})
        if rows_:
            att_rows.append({"row_type": "total by status", "status": st, "gpu": "all",
                             "attempt": len(rows_), "wall_hours": r4(sum(a["wall_h"] for a in rows_))})
    kept_used_total = sum(kept_split(a)[0] for a in bad)
    kept_unused_total = sum(kept_split(a)[1] for a in bad)
    att_rows.append({"row_type": "total", "status": "all non-ok", "gpu": "all", "attempt": len(bad),
                     "wall_hours": r4(sum(a["wall_h"] for a in bad)),
                     "kept_used_unit_hours": r4(kept_used_total), "kept_unused_unit_hours": r4(kept_unused_total)})
    write_csv(att_rows, os.path.join(out, "cost_attempts.csv"))

    # ------------------------------------------------------------------ cost_by_gpu.csv
    gpu_model = {}
    for g in determinism.get("gpus", []):
        idx, name = [x.strip() for x in g.split(",", 1)]
        gpu_model[idx] = name
    gpu_rows = []
    tot = Counter()
    for g in sorted(by_gpu):
        rows_ = by_gpu[g]
        iv_all = [(a["start"], a["end"]) for a in rows_]
        iv_ok = [(a["start"], a["end"]) for a in rows_ if a["status"] == "ok"]
        u_all, u_ok = union(iv_all), union(iv_ok)
        conc = concurrency(iv_all)
        first, last = min(s for s, _ in iv_all), max(e for _, e in iv_all)
        r = {"gpu": g, "gpu_model": gpu_model.get(g, "?"), "attempts_all": len(rows_),
             "attempts_ok": sum(1 for a in rows_ if a["status"] == "ok"),
             "sum_hours_all": sum(a["wall_h"] for a in rows_), "union_hours_all": length(u_all) / 3600,
             "overlap_hours_all": max(0.0, sum(a["wall_h"] for a in rows_) - length(u_all) / 3600),
             "sum_hours_ok": sum(a["wall_h"] for a in rows_ if a["status"] == "ok"),
             "union_hours_ok": length(u_ok) / 3600,
             "overlap_hours_ok": max(0.0, sum(a["wall_h"] for a in rows_ if a["status"] == "ok") - length(u_ok) / 3600),
             "hours_1_job": conc.get(1, 0) / 3600, "hours_2_jobs": conc.get(2, 0) / 3600,
             "hours_3_jobs": sum(v for k, v in conc.items() if k >= 3) / 3600,
             "max_concurrent": max(conc) if conc else 0,
             "first_start_utc": utc(first), "last_end_utc": utc(last), "span_hours": (last - first) / 3600,
             "idle_hours_in_span": (last - first - length(u_all)) / 3600,
             "busy_fraction_of_span": length(u_all) / (last - first)}
        for stg, key in zip(STAGES, ("teacher_build", "kd", "eval", "other")):
            r[f"sum_hours_{key}"] = sum(a["wall_h"] for a in rows_ if a["stage"] == stg)
        lanes_here = OrderedDict()
        for a in sorted(rows_, key=lambda x: x["start"]):
            la = lanes_here.setdefault(a["lane"], [a["start"], a["end"], 0])
            la[1] = max(la[1], a["end"])
            la[2] += 1
        r["lanes"] = "; ".join(f"{la}: {n} attempts {utc(s0)[5:16]}-{utc(e0)[5:16]} UTC"
                               for la, (s0, e0, n) in lanes_here.items())
        r["note"] = ("static lanes gpu0a, gpu0b (v1-v2), then the two-slot pool gpu0p1/gpu0p2 with a 78 GB "
                     "memory budget (queue v3-v7); the A9 re-evaluation ran by hand beside two pool jobs"
                     if g == "0" else
                     "teacher build and the v1 KD run until the author's 10/21 training ban; then Stage 2 "
                     "only; from the lane restart of 10/21 15:34 UTC in exclusive mode (never beside "
                     "another process, A0.1)")
        gpu_rows.append(r)
        for k, v in r.items():
            if isinstance(v, (int, float)) and k not in ("max_concurrent", "busy_fraction_of_span",
                                                         "span_hours", "idle_hours_in_span"):
                tot[k] += v
    tot_row = {"gpu": "all (GPU 0 + GPU 1)", "gpu_model": "; ".join(sorted(set(gpu_model.values())))}
    for k in gpu_rows[0]:
        if k in tot:
            tot_row[k] = tot[k]
    tot_row["note"] = ("union_hours = device-busy hours (the per-GPU unions added); sum_hours = GPU "
                       "hours as attempt wall hours, an upper bound of device occupancy (A12). GPU 2 "
                       "(" + gpu_model.get("2", "?") + ") ran no TIFS job.")
    gpu_rows.append(tot_row)
    cpu_steps = [a for a in all_attempts if a["uses_gpu"] != "yes"]
    gpu_rows.append({"gpu": "CPU steps in GPU lanes", "gpu_model": "-", "attempts_all": len(cpu_steps),
                     "attempts_ok": sum(1 for a in cpu_steps if a["status"] == "ok"),
                     "sum_hours_all": sum(a["wall_h"] for a in cpu_steps),
                     "note": ", ".join(sorted({a["job"] for a in cpu_steps})) + " (no GPU use; held a lane slot)"})
    write_csv([rnd(r) for r in gpu_rows], os.path.join(out, "cost_by_gpu.csv"))

    # ------------------------------------------------------------------ cost_units.csv
    unit_rows = []
    for t in sorted(teachers, key=lambda x: x["end"]):
        a = t["attempt"]
        s0, e0 = unit_interval(t, "teacher")
        sf = shared_frac(a, s0, e0) if a else ""
        t["shared_frac"] = sf
        unit_rows.append({
            "unit": "teacher", "name": t["name"], "run": a["run"] if a else "", "family": t["family"],
            "dataset": t["dataset"], "kind_or_role": t["kind"], "trainer_or_kd": t["trainer"],
            "attack": t["attack"], "seed": t["seed"], "teacher_seed": t["seed"],
            "hours": r4((t["train_sec"] or 0) / 3600), "hours_source": "ob_meta.json train_time_sec",
            "end_utc": utc(t["end"]), "job": a["job"] if a else "", "attempt": a["attempt"] if a else "",
            "attempt_status": a["status"] if a else "", "gpu": a["gpu"] if a else "",
            "shared_frac": r4(sf) if sf != "" else "", "used_by_students": t["used_by"],
            "in_window": "no" if t["reused_tier1"] else "yes",
            "note": ("tier-1 build reused by link (" + os.path.relpath(t["real"], ROOT) + "); built "
                     + datetime.fromtimestamp(t["end"], UTC).strftime("%Y-%m-%d") + ", outside events.jsonl; "
                     "not in the totals") if t["reused_tier1"] else "",
        })
    for s in sorted(students, key=lambda x: x["end"]):
        a = s["attempt"]
        s0, e0 = unit_interval(s, "student")
        sf = shared_frac(a, s0, e0) if a else ""
        s["shared_frac"] = sf
        hours = (s["train_sec"] / 3600) if s["train_sec"] is not None else (
            a["wall_h"] / max(1, len(a["students"])) if a else None)
        unit_rows.append({
            "unit": "student", "name": s["name"], "run": s["run"], "family": s["family"],
            "dataset": s["dataset"], "kind_or_role": s["role"], "trainer_or_kd": s["kd"],
            "attack": s["name"].split("_")[0] if s["role"] == "P" else ("" if s["role"] == "R" else s["name"].split("_")[0]),
            "seed": s["seed"], "teacher_seed": s["teacher_seed"], "hours": r4(hours),
            "hours_source": "kd_meta.json train_time_sec" if s["train_sec"] is not None else "job hours / students",
            "end_utc": utc(s["end"]), "job": a["job"] if a else "", "attempt": a["attempt"] if a else "",
            "attempt_status": a["status"] if a else "", "gpu": a["gpu"] if a else "",
            "shared_frac": r4(sf) if sf != "" else "", "used_by_students": "", "in_window": "yes",
            "evaluated_in_stage2": "yes" if s["evaluated"] else "no",
            "note": (f"trained by {a['job']} (v1) and reused by {s['run']}" if s.get("moved") else ""),
        })
    for c in sorted(cells, key=lambda x: x["end"]):
        a = c["attempt"]
        s0, e0 = unit_interval(c, "cell")
        sf = shared_frac(a, s0, e0) if a else ""
        c["shared_frac"] = sf
        unit_rows.append({
            "unit": "cell", "name": c["name"], "run": c["run"], "family": c["family"],
            "dataset": c["dataset"], "kind_or_role": "P,C or R vs trigger + controls", "trainer_or_kd": c["kd"],
            "attack": c["attack"], "seed": c["seed"], "teacher_seed": c["teacher_seed"],
            "hours": r4(c["dur_sec"] / 3600),
            "hours_source": ("prediction-file time minus the previous cell's (first cell of an attempt: "
                             "minus the attempt start, so it includes start-up)"),
            "end_utc": utc(c["end"]), "job": a["job"] if a else "", "attempt": a["attempt"] if a else "",
            "attempt_status": a["status"] if a else "", "gpu": a["gpu"] if a else "",
            "shared_frac": r4(sf) if sf != "" else "", "used_by_students": "",
            "in_window": "yes", "note": "first cell of its attempt" if c["first_in_attempt"] else "",
        })
    write_csv(unit_rows, os.path.join(out, "cost_units.csv"))

    # ------------------------------------------------------------------ cost_per_unit.csv
    def summary(unit, fam, ds, group, vals, sfs, source, note="", with_sum=True):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        sfs = [x for x in sfs if x != ""]
        return {"unit": unit, "family": fam, "dataset": ds, "group": group, "n": len(vals),
                "median_h": r4(statistics.median(vals)), "min_h": r4(min(vals)), "max_h": r4(max(vals)),
                "mean_h": r4(statistics.fmean(vals)), "sum_h": r4(sum(vals)) if with_sum else "",
                "median_shared_frac": r4(statistics.median(sfs)) if sfs else "", "hours_source": source,
                "note": note}

    pu = []
    tb = [t for t in teachers if not t["reused_tier1"]]
    src_t = "ob_meta.json train_time_sec (OpenBackdoor training incl. per-epoch dev evaluation)"
    for fam in sorted({t["family"] for t in tb}):
        for ds in DATASETS:
            sel = [t for t in tb if t["family"] == fam and t["dataset"] == ds]
            if not sel:
                continue
            pu.append(summary("teacher build", fam, ds, "all built in the window",
                              [t["train_sec"] / 3600 for t in sel], [t["shared_frac"] for t in sel], src_t))
            for tr in sorted({t["trainer"] for t in sel}):
                s2 = [t for t in sel if t["trainer"] == tr]
                if len(s2) != len(sel):
                    pu.append(summary("teacher build", fam, ds, f"trainer={tr}", [t["train_sec"] / 3600 for t in s2],
                                      [t["shared_frac"] for t in s2], src_t))
            for g in ("0", "1"):
                s2 = [t for t in sel if t["attempt"] and t["attempt"]["gpu"] == g]
                if s2 and len(s2) != len(sel):
                    pu.append(summary("teacher build", fam, ds, f"GPU {g}", [t["train_sec"] / 3600 for t in s2],
                                      [t["shared_frac"] for t in s2], src_t))
    for ds in DATASETS:
        sel = [t for t in teachers if t["reused_tier1"] and t["dataset"] == ds]
        if sel:
            pu.append(summary("teacher build", "bert", ds, "tier-1 reuse (EP, SOS pairs; built 2026-07; not in totals)",
                              [t["train_sec"] / 3600 for t in sel], [], src_t,
                              "outside events.jsonl; GPU and contention not recorded"))
    src_s = "kd_meta.json train_time_sec (teacher + student load, 5 KD epochs, save)"
    for fam in sorted({s["family"] for s in students}):
        for ds in DATASETS:
            sel = [s for s in students if s["family"] == fam and s["dataset"] == ds]
            if not sel:
                continue
            pu.append(summary("KD student", fam, ds, "all", [s["train_sec"] / 3600 for s in sel],
                              [s["shared_frac"] for s in sel], src_s))
            for kd in ("logit", "feature", "hidden", "ce"):
                s2 = [s for s in sel if s["kd"] == kd]
                if s2 and len(s2) != len(sel):
                    pu.append(summary("KD student", fam, ds, f"kd={kd}", [s["train_sec"] / 3600 for s in s2],
                                      [s["shared_frac"] for s in s2], src_s))
            for g in ("0", "1"):
                s2 = [s for s in sel if s["attempt"] and s["attempt"]["gpu"] == g]
                if s2 and len(s2) != len(sel):
                    pu.append(summary("KD student", fam, ds, f"GPU {g}", [s["train_sec"] / 3600 for s in s2],
                                      [s["shared_frac"] for s in s2], src_s))
    src_c = "prediction-file times (successive cells of one attempt)"
    okc = [c for c in cells if c["attempt"] and c["attempt"]["status"] == "ok"]
    for fam in sorted({c["family"] for c in okc}):
        for ds in DATASETS:
            sel = [c for c in okc if c["family"] == fam and c["dataset"] == ds]
            if not sel:
                continue
            pu.append(summary("evaluated cell", fam, ds, "all (ok attempts)", [c["dur_sec"] / 3600 for c in sel],
                              [c["shared_frac"] for c in sel], src_c, "first cells include start-up"))
            s2 = [c for c in sel if not c["first_in_attempt"]]
            pu.append(summary("evaluated cell", fam, ds, "excluding the first cell of each attempt",
                              [c["dur_sec"] / 3600 for c in s2], [c["shared_frac"] for c in s2], src_c))
            for g in ("0", "1"):
                s3 = [c for c in s2 if c["attempt"]["gpu"] == g]
                if s3 and len(s3) != len(s2):
                    pu.append(summary("evaluated cell", fam, ds, f"excluding first, GPU {g}",
                                      [c["dur_sec"] / 3600 for c in s3], [c["shared_frac"] for c in s3], src_c))
    pu = [r for r in pu if r]
    # job-level cross-check: wall hours / units for each successful attempt
    for stg, unit in (("teacher build", "teacher build"), ("KD students", "KD student"), ("evaluation", "evaluated cell")):
        vals = []
        for a in attempts:
            if a["status"] == "ok" and a["stage"] == stg:
                n = len(a["teachers"]) if stg == "teacher build" else len(a["students"]) if stg == "KD students" else len(a["cells"])
                if n:
                    vals.append(a["wall_h"] / n)
        pu.append(summary(unit, "all", "all", "job wall hours / units written (ok attempts with units)",
                          vals, [], "cost_jobs.csv wall_hours / n_written", "one value per attempt; mixes datasets",
                          with_sum=False))
    write_csv(pu, os.path.join(out, "cost_per_unit.csv"))

    # ------------------------------------------------------------------ hardware_software.json
    tf_versions = Counter()
    tf_by_family = defaultdict(Counter)
    for t in teachers:
        p = os.path.join(t["dir"], "config.json")
        v = json.load(open(p)).get("transformers_version")
        tf_versions[v] += 1
        tf_by_family[f"teacher {t['family']}"][v] += 1
    for s in students:
        p = os.path.join(s["dir"], "config.json")
        v = json.load(open(p)).get("transformers_version")
        tf_versions[v] += 1
        tf_by_family[f"student {s['family']}"][v] += 1
    torch_logs = Counter()
    interp = Counter()
    py_minor = Counter()        # run-time Python minor version: site-packages paths in warnings
    py_minor_logs = set()
    for job, (path, segs) in logs.items():
        if not os.path.exists(path):
            continue
        with open(path, errors="replace") as f:
            for line in f:
                for v in re.findall(r"PyTorch version (\d[\w.+]*) available", line):
                    torch_logs[v] += 1
                for v in re.findall(re.escape(PY_ENV) + r"/lib/python(\d+\.\d+)/site-packages/", line):
                    py_minor[v] += 1
                    py_minor_logs.add(rel(path))
        for s in segs:
            interp[s["cmd"].split()[0]] += 1
    tok_meta = json.load(open(TOKENS_META)) if os.path.exists(TOKENS_META) else {}
    env_now = {"python_executable": f"{PY_ENV}/bin/python",
               "python": sys.version.split()[0] if sys.executable.startswith(PY_ENV) else None}
    for p in ("torch", "transformers", "tokenizers", "datasets", "accelerate", "safetensors", "numpy", "scipy", "PyYAML"):
        try:
            env_now[p] = importlib_metadata.version(p)
        except importlib_metadata.PackageNotFoundError:
            env_now[p] = None
    torch_cuda = None
    for vp in glob.glob(os.path.join(PY_ENV, "lib", "python*", "site-packages", "torch", "version.py")):
        mm = re.search(r"^cuda\s*(?::\s*\S+\s*)?=\s*['\"]([^'\"]+)['\"]", open(vp).read(), re.M)
        torch_cuda = mm.group(1) if mm else None
    oom = ""
    tm = manifests.get("tifs_t_gpt2")
    if tm:
        for ff in tm[0].get("stages", {}).get("stage0", {}).get("first_failures", []):
            mm = re.search(r"GPU 0 has a total capacity of ([\d.]+ GiB)", ff.get("error", ""))
            if mm:
                oom = mm.group(1)
    precision = Counter((s["tf32"], s["teacher_dtype"], s["student_dtype"]) for s in students)
    eval_dtypes = Counter()
    for run in {c["run"] for c in cells}:
        cfg, _ = load_config(run)
        eval_dtypes[(cfg or {}).get("stage2", {}).get("eval_dtype")] += 1
    hosts = Counter(e.get("host") for e in events if e.get("host"))
    lanes = Counter((e.get("lane"), str(e.get("gpu"))) for e in events if e["state"] == "running")
    hw = OrderedDict([
        ("status", "A12 descriptive record (post hoc); changes no primary verdict, battery decision or the paper-wide Holm"),
        ("gpus", OrderedDict([
            ("recorded", determinism.get("gpus")),
            ("source", rel(DETERMINISM) + " 'gpus' (nvidia-smi listing at the A9 check, 2026-10-23)"),
            ("used_by_tifs_jobs", sorted({a["gpu"] for a in all_attempts if a["uses_gpu"] == "yes"})),
            ("gpu0_total_capacity_reported_by_pytorch", oom),
            ("gpu0_capacity_source", "runs/tifs/tifs_t_gpt2/manifest.json stage0 first_failures (CUDA OOM message)"),
            ("note", "GPU 2 belonged to another user (docs/tifs_plan.md: 'GPU2는 사용 불가(다른 사용자)') and ran no "
                     "TIFS job (no events.jsonl attempt names it); GPU 1 was shared with another user's jobs (the "
                     f"exclusive lane yielded to them, A0.1, A9). GPU memory: PyTorch reports {oom} total capacity "
                     f"on GPU 0; queue_plan.py budgets {qp.POOLS['gpu0']['capacity_gb']} GB for the GPU 0 pool."),
        ])),
        ("host", OrderedDict([("recorded", dict(hosts)), ("source", "events.jsonl 'host' of running events")])),
        ("lanes", [{"lane": la, "gpu": g, "attempts": n} for (la, g), n in sorted(lanes.items())]),
        ("software_recorded_at_run_time", OrderedDict([
            ("transformers", OrderedDict([("versions", dict(tf_versions)),
                                          ("by_checkpoint_kind", {k: dict(v) for k, v in sorted(tf_by_family.items())}),
                                          ("source", "config.json 'transformers_version' of every teacher and student checkpoint")])),
            ("torch", OrderedDict([("versions", dict(torch_logs)),
                                   ("source", "job logs: 'PyTorch version X available' (datasets library, printed at Stage 0 start)"),
                                   ("n_log_lines", sum(torch_logs.values()))])),
            ("tokenizers", OrderedDict([("version", tok_meta.get("tokenizers_version")),
                                        ("source", rel(TOKENS_META) + " (A11 g1, CPU, " + str(tok_meta.get("snapshot_utc")) + "); not a training-time record")])),
            ("python", OrderedDict([("interpreter", dict(interp)),
                                    ("version", None),
                                    ("minor_version_in_job_logs", dict(py_minor)),
                                    ("minor_version_logs", sorted(py_minor_logs)),
                                    ("source", "interpreter: job log headers ('=== <time> <command> (GPU n)'). The full "
                                               "version is not recorded in run metadata; the minor version appears in "
                                               "site-packages paths of library warnings printed by the jobs "
                                               f"('{PY_ENV}/lib/python3.x/site-packages/...'). The patch version is "
                                               "only in environment_now.")])),
            ("openbackdoor_path", qp.OB_PATH),
        ])),
        ("environment_now", OrderedDict([
            ("queried_utc", datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")),
            ("packages", env_now), ("torch_cuda_build", torch_cuda),
            ("source", "importlib.metadata of " + PY_ENV + " and its torch/version.py; the environment as it is "
                       "now, not a run-time record (it matches the recorded transformers and torch versions)"),
        ])),
        ("precision", OrderedDict([
            ("students_tf32_teacher_dtype_student_dtype", {f"tf32={k[0]}, teacher={k[1]}, student={k[2]}": v for k, v in precision.items()}),
            ("stage2_eval_dtype_by_run", {str(k): v for k, v in eval_dtypes.items()}),
            ("teachers_tf32", dict(Counter(str(json.load(open(t["meta"])).get("tf32")) for t in teachers))),
            ("source", "kd_meta.json, ob_meta.json, configs/tifs/<run>.yaml stage2.eval_dtype"),
        ])),
    ])
    with open(os.path.join(out, "hardware_software.json"), "w") as f:
        json.dump(hw, f, indent=1, default=str)
        f.write("\n")

    # ------------------------------------------------------------------ self-checks
    # (1) events: every running event in one attempt; closing events used once
    n_running = sum(1 for e in events if e["state"] == "running")
    check("events", "running events = lane attempts", len(attempts), n_running, len(attempts) == n_running)
    closes = Counter(a["close"]["_line"] for a in attempts if a["close"])
    check("events", "closing events used once", max(closes.values()) if closes else 0, 1, all(v == 1 for v in closes.values()))
    nterm = [a for a in attempts if a["close"] is None]
    check("events", "attempts without a closing event", len(nterm), "explained", True,
          "; ".join(f"{a['job']} #{a['attempt']}: {a['end_source']}" for a in nterm))
    # (2) event hours vs end - start
    worst = 0.0
    for a in attempts:
        if a["close"] is not None and "hours" in a["close"]:
            d = abs(float(a["close"]["hours"]) - a["wall_h"])
            worst = max(worst, d)
            check("hours", f"{a['job']} #{a['attempt']}", r4(a["wall_h"]), f"event hours {a['close']['hours']} +/- {HOURS_TOL:.4f}",
                  d <= HOURS_TOL, f"|diff| = {d:.4f} h")
    check("hours", "max |event hours - (end - start)|", r4(worst), r4(HOURS_TOL), worst <= HOURS_TOL)
    # (3) log segments
    for a in attempts:
        if a["seg_matches"] != 1:
            check("logs", f"{a['job']} #{a['attempt']}", a["seg_matches"], 1, False, "log segment not found")
    check("logs", "attempts matched to exactly one log segment", sum(1 for a in attempts if a["seg_matches"] == 1),
          len(attempts), all(a["seg_matches"] == 1 for a in attempts))
    n_segs = sum(len(s) for _, s in logs.values())
    check("logs", "log segments = attempts", n_segs, len(attempts), n_segs == len(attempts))
    for a in attempts:
        if a["seg"] and a["seg"]["gpu"] != a["gpu"]:
            check("logs", f"{a['job']} #{a['attempt']} GPU", a["seg"]["gpu"], a["gpu"], False)
    check("logs", "log header GPU = event GPU", sum(1 for a in attempts if a["seg"] and a["seg"]["gpu"] == a["gpu"]),
          len(attempts), all(a["seg"] and a["seg"]["gpu"] == a["gpu"] for a in attempts))
    # (4) train_time_sec vs wall hours, per attempt and per run
    for a in sorted(attempts, key=lambda x: x["start"]):
        if a["stage"] == "KD students" and a["students"]:
            ratio = sum(s["train_sec"] for s in a["students"]) / 3600 / a["wall_h"]
            check("train_time_sec / wall (KD attempt)", f"{a['job']} #{a['attempt']} ({a['status']}, {len(a['students'])} students)",
                  r4(ratio), "<= 1.01", ratio <= 1.01, f"wall {a['wall_h']:.3f} h")
        if a["stage"] == "teacher build" and a["teachers"]:
            ratio = sum(t["train_sec"] for t in a["teachers"]) / 3600 / a["wall_h"]
            check("train_time_sec / wall (teacher attempt)", f"{a['job']} #{a['attempt']} ({a['status']}, {len(a['teachers'])} teachers)",
                  r4(ratio), "<= 1.01", ratio <= 1.01, f"wall {a['wall_h']:.3f} h")
        if a["stage"] == "evaluation" and a["seg"]:
            summ = dict(a["seg"]["summaries"])
            if "stage2" in summ:
                ratio = summ["stage2"]["elapsed_sec"] / 3600 / a["wall_h"]
                check("Stage 2 elapsed_sec / wall (eval attempt)", f"{a['job']} #{a['attempt']}", r4(ratio), "0.80-1.01",
                      0.8 <= ratio <= 1.01, f"wall {a['wall_h']:.4f} h, {len(a['cells'])} cells written")
    runs_kd = sorted({s["run"] for s in students})
    for run in runs_kd:
        ss = [s for s in students if s["run"] == run]
        aa = [a for a in attempts if a["stage"] == "KD students" and a["run"] == run]
        wall = sum(a["wall_h"] for a in aa)
        ratio = sum(s["train_sec"] for s in ss) / 3600 / wall if wall else float("inf")
        moved = [s for s in ss if s.get("moved")]
        check("train_time_sec / wall (per run)", f"{run} ({len(ss)} students, {len(aa)} KD attempt(s) of the job)",
              r4(ratio), "<= 1.01 unless students came from another job", ratio <= 1.01 or bool(moved),
              (f"{len(moved)} students were trained by {moved[0]['attempt']['job']} and moved into this run" if moved else ""))
    # (5) units attributed
    check("units", "students (kd_meta.json)", len(students), 184, len(students) == 184)
    check("units", "students attributed to one KD attempt", sum(1 for s in students if s.get("attempt")), len(students),
          all(s.get("attempt") for s in students))
    check("units", "students with a Stage 2 cell of their run", sum(1 for s in students if s["evaluated"]), len(students),
          all(s["evaluated"] for s in students))
    check("units", "Stage 2 prediction files", len(cells), 146, len(cells) == 146)
    check("units", "cells attributed to one evaluation attempt of their run", sum(1 for c in cells if c.get("attempt")), len(cells),
          all(c.get("attempt") for c in cells))
    nbuilt = sum(1 for t in teachers if not t["reused_tier1"])
    check("units", "teacher directories (built in the window + tier-1 links)", f"{nbuilt} + {len(teachers) - nbuilt}",
          f"{len(teachers)} = 35 + 8", nbuilt == 35 and len(teachers) == 43)
    check("units", "built teachers attributed to one teacher-build attempt", sum(1 for t in teachers if t["attempt"]), nbuilt,
          all(t["attempt"] for t in teachers if not t["reused_tier1"]))
    first_ev = min(e["_t"] for e in events)
    check("units", "tier-1 teachers written before the first event", sum(1 for t in teachers if t["reused_tier1"] and t["end"] < first_ev),
          len(teachers) - nbuilt, all(t["end"] < first_ev for t in teachers if t["reused_tier1"]),
          f"REUSED.json links: {len(reused.get('links', []))}")
    check("units", "attribution problems", len(problems), 0, not problems, "; ".join(problems[:10]))
    for a in attempts:
        if a["stage"] == "evaluation" and a["seg"]:
            summ = dict(a["seg"]["summaries"])
            if "stage2" in summ:
                n_cond = summ["stage2"].get("conditions_evaluated")
                done_before = sum(1 for c in cells if c["run"] == a["run"] and c["end"] < a["start"])
                check("units", f"{a['job']} #{a['attempt']}: cells written + written earlier = conditions_evaluated",
                      f"{len(a['cells'])} + {done_before}", n_cond, len(a["cells"]) + done_before == n_cond)
    ycells = sum(len(a["cells"]) for a in attempts if a["status"] == "yielded")
    check("units", "cells written by yielded attempts (kept: Stage 2 skips finished cells)", ycells, "reported", True,
          "; ".join(f"{a['job']} #{a['attempt']}: {len(a['cells'])}" for a in attempts if a["status"] == "yielded" and a["cells"]))
    unused = [t["name"] for t in teachers if not t["reused_tier1"] and not t["used_by"]]
    check("units", "built teachers no student used", len(unused), "reported", True, ", ".join(unused))
    # (6) classification
    for a in all_attempts:
        good = a["stage"] in STAGES and a["block"] in BLOCKS and a["status"] in STATUSES
        if not good or a["block_detail"] == "unclassified":
            check("classification", f"{a['job']} #{a['attempt']}", f"{a['stage']} / {a['block']} / {a['status']}", "classified", False)
    okc_ = [a for a in all_attempts if a["status"] == "ok"]
    check("classification", "ok attempts with a stage and a block", sum(1 for a in okc_ if a["stage"] in STAGES and a["block"] in BLOCKS),
          len(okc_), all(a["stage"] in STAGES and a["block"] in BLOCKS for a in okc_))
    check("classification", "all attempts classified (stage, block, status)",
          sum(1 for a in all_attempts if a["stage"] in STAGES and a["block"] in BLOCKS and a["status"] in STATUSES),
          len(all_attempts), all(a["stage"] in STAGES and a["block"] in BLOCKS and a["status"] in STATUSES for a in all_attempts))
    for a in all_attempts:
        if a["block"] == B_OTHER:
            check("classification", f"'other': {a['job']} #{a['attempt']}", a["stage"], "explained", True, a["block_detail"])
    # (7) queue plan vs events
    ev_jobs = {e["job"] for e in events}
    missing = [j for j in qjobs if j not in ev_jobs]
    check("queue_plan", "v7 jobs that appear in events.jsonl", len(qjobs) - len(missing), len(qjobs), not missing, ", ".join(missing))
    not_in_plan = sorted({a["job"] for a in attempts} - set(qjobs))
    check("queue_plan", "attempted jobs not in queue_plan v7", len(not_in_plan), "pre-v3 jobs only", set(not_in_plan) <= {
        "tifs_t_bert", "tifs_t_gpt2", "tifs_ts_bert", "tifs_g_bert_clean_s42", "tifs_synbkd_table"}, ", ".join(not_in_plan))
    for a in attempts:
        qj = qjobs.get(a["job"])
        if qj and qj.get("stages") and a["seg"] and a["seg"]["stages"] and qj["stages"] != a["seg"]["stages"]:
            check("queue_plan", f"{a['job']} #{a['attempt']} stages", " ".join(a["seg"]["stages"]), " ".join(qj["stages"]), False)
        if qj and a["mem_gb"] not in ("", None) and qj.get("mem_gb") not in (None,) and float(a["mem_gb"]) != float(qj["mem_gb"]):
            check("queue_plan", f"{a['job']} #{a['attempt']} declared mem_gb", a["mem_gb"], qj["mem_gb"], True,
                  "differs from v7 (queue version at the time)")
    final_state = {}
    for e in events:
        if e["state"] in ("ok", "failed", "blocked"):
            final_state[e["job"]] = e["state"]
    blocked = sorted(j for j, s in final_state.items() if s == "blocked")
    check("queue_plan", "jobs whose final state is blocked (never ran, no GPU time)", len(blocked), "reported", True, ", ".join(blocked))
    # (8) time zone
    diffs = []
    last_state = {}
    for e in events:
        if e["state"] in ("ok", "failed", "blocked"):
            last_state[(e["job"], e["state"])] = e["_t"]
    for p in sorted(glob.glob(os.path.join(STATUS, "*.ok")) + glob.glob(os.path.join(STATUS, "*.failed"))
                    + glob.glob(os.path.join(STATUS, "*.blocked"))):
        job, kind = os.path.basename(p).rsplit(".", 1)
        if (job, kind) in last_state:
            diffs.append(mtime(p) - last_state[(job, kind)])
    check("time zone", "marker file time - event time read as UTC (s)", f"{min(diffs):.2f} .. {max(diffs):.2f}",
          "0 .. 2", diffs and 0 <= min(diffs) and max(diffs) <= 2, f"{len(diffs)} markers in runs/tifs/_status")
    doc = open(ADDENDUM_DOC).read()
    known = [("tifs_rcc_bert_sst2", "ok", "18:02"), ("tifs_rcc_bert_agnews", "ok", "18:34"),
             ("tifs_rcc_bert_sst2__eval", "ok", "18:04"), ("tifs_rcc_bert_agnews__eval", "ok", "18:47")]
    phrase = "SST-2 18:02, AG News 18:34 KST 학습; 평가 18:04, 18:47"
    for job, st, hhmm in known:
        t = last_state.get((job, st))
        got = datetime.fromtimestamp(t, KST).strftime("%H:%M") if t else ""
        check("time zone", f"{job} {st} event in KST vs docs/tifs_addendum.md A10 result record", got, hhmm,
              got == hhmm and phrase in doc, f"event {utc(t)} UTC")
    # (9) reconciliation
    s_all = sum(a["wall_h"] for a in all_attempts)
    s_ok = sum(a["wall_h"] for a in ok)
    s_bad = sum(a["wall_h"] for a in bad)
    check("reconcile", "all attempts = ok + non-ok", r4(s_all), r4(s_ok + s_bad), abs(s_all - s_ok - s_bad) < 1e-9)
    det_sum = sum(r["wall_hours"] for r in summ_rows if r["row_type"] == "detail")
    check("reconcile", "cost_summary detail rows = overall (ok)", r4(det_sum), r4(s_ok), abs(det_sum - s_ok) < 1e-9)
    st_sum = sum(r["wall_hours"] for r in summ_rows if r["row_type"] == "stage")
    check("reconcile", "stage totals = overall (ok)", r4(st_sum), r4(s_ok), abs(st_sum - s_ok) < 1e-9)
    bl_sum = sum(r["wall_hours"] for r in summ_rows if r["row_type"] == "block")
    check("reconcile", "block totals = overall (ok)", r4(bl_sum), r4(s_ok), abs(bl_sum - s_ok) < 1e-9)
    fam_sum = sum(r["wall_hours"] for r in summ_rows if r["row_type"] == "family")
    check("reconcile", "family totals = overall (ok)", r4(fam_sum), r4(s_ok), abs(fam_sum - s_ok) < 1e-9)
    sxf = sum(r["wall_hours"] for r in summ_rows if r["row_type"] == "stage x family")
    check("reconcile", "stage x family totals = overall (ok)", r4(sxf), r4(s_ok), abs(sxf - s_ok) < 1e-9)
    for rt in ("stage x family (all attempts)", "stage (all attempts)", "block (all attempts)"):
        x = sum(r["wall_hours"] for r in summ_rows if r["row_type"] == rt)
        check("reconcile", f"cost_summary '{rt}' rows = all attempts", r4(x), r4(s_all), abs(x - s_all) < 1e-9)
    ap = sum(sum(a["apport"].values()) for a in all_attempts)
    check("reconcile", "apportioned hours (sst2 + agnews + unattributed) = wall hours", r4(ap), r4(s_all), abs(ap - s_all) < 1e-9)
    g_sum = sum(r["sum_hours_all"] for r in gpu_rows[:-2])
    cpu_sum = gpu_rows[-1]["sum_hours_all"]
    check("reconcile", "per-GPU sums + CPU steps = all attempts", r4(g_sum + cpu_sum), r4(s_all), abs(g_sum + cpu_sum - s_all) < 1e-9)
    for r in gpu_rows[:-2]:
        check("reconcile", f"GPU {r['gpu']}: union <= sum; union(ok) <= union(all)",
              f"{r['union_hours_all']:.4f} <= {r['sum_hours_all']:.4f}; {r['union_hours_ok']:.4f} <= {r['union_hours_all']:.4f}",
              "true", r["union_hours_all"] <= r["sum_hours_all"] + 1e-9 and r["union_hours_ok"] <= r["union_hours_all"] + 1e-9)
        c_sum = r["hours_1_job"] + r["hours_2_jobs"] + r["hours_3_jobs"]
        check("reconcile", f"GPU {r['gpu']}: hours at 1, 2, 3 jobs = union", r4(c_sum), r4(r["union_hours_all"]),
              abs(c_sum - r["union_hours_all"]) < 1e-6)
        w_sum = r["hours_1_job"] + 2 * r["hours_2_jobs"] + 3 * r["hours_3_jobs"]
        check("reconcile", f"GPU {r['gpu']}: 1*h1 + 2*h2 + 3*h3 = sum of attempt hours", r4(w_sum), r4(r["sum_hours_all"]),
              abs(w_sum - r["sum_hours_all"]) < 1e-6 and r["max_concurrent"] <= 3)
    g0row = [r for r in gpu_rows if r["gpu"] == "0"][0]
    check("reconcile", "GPU 0 time with 3 jobs = the A9 re-evaluation (s)", round(g0row["hours_3_jobs"] * 3600, 1),
          round(a9_sec, 1), abs(g0row["hours_3_jobs"] * 3600 - a9_sec) < 1.0,
          "the A9 re-evaluation ran by hand beside two pool jobs; no other three-job period")
    att_total = [r for r in att_rows if r["row_type"] == "total"][0]["wall_hours"]
    check("reconcile", "cost_attempts total = non-ok attempts", att_total, r4(s_bad), abs(att_total - s_bad) < 1e-3)
    # (10) frozen files untouched
    for path, want in FROZEN_SHA256.items():
        got = sha256(os.path.join(ROOT, path))
        check("frozen", path, got[:12], want[:12], got == want)

    write_csv(checks, os.path.join(out, "selfcheck.csv"))

    # ------------------------------------------------------------------ provenance.csv
    prov = []
    for p in (EVENTS, os.path.join(ROOT, "scripts", "tifs", "queue_plan.py"), DETERMINISM,
              os.path.join(TEACHERS, "REUSED.json"), TOKENS_META, ADDENDUM_DOC,
              os.path.join(ROOT, "docs", "tifs_plan.md"), os.path.join(ROOT, "scripts", "tifs", "lane.py")):
        prov.append({"kind": "file", "path": rel(p), "n_files": 1, "sha256": sha256(p), "bytes": os.path.getsize(p),
                     "newest_mtime_utc": utc(mtime(p)), "note": ""})

    def group_hash(paths):
        h = hashlib.sha256()
        for p in sorted(paths):
            h.update(rel(p).encode() + b"\0" + sha256(p).encode() + b"\n")
        return h.hexdigest()

    sets = [
        ("kd_meta.json (students)", [s["meta"] for s in students]),
        ("ob_meta.json (teachers)", [t["meta"] for t in teachers]),
        ("Stage 2 prediction files", [c["path"] for c in cells]),
        ("run manifests", [m[1] for m in manifests.values()]),
        ("job logs read", [p for p, segs in logs.values() if os.path.exists(p)]),
        ("configs read", sorted({os.path.join(ROOT, a["cfg_path"]) for a in attempts if a["cfg_path"]})),
        ("status markers", glob.glob(os.path.join(STATUS, "*.ok")) + glob.glob(os.path.join(STATUS, "*.failed"))
         + glob.glob(os.path.join(STATUS, "*.blocked"))),
    ]
    for name, paths in sets:
        prov.append({"kind": "file set", "path": name, "n_files": len(paths), "sha256": group_hash(paths),
                     "bytes": sum(os.path.getsize(p) for p in paths),
                     "newest_mtime_utc": utc(max(mtime(p) for p in paths)),
                     "note": "sha256 over sorted (path, file sha256) pairs"})
    newest_ev = max(events, key=lambda e: e["_t"])
    newest_job = max(attempts, key=lambda a: a["end"])
    prov.append({"kind": "snapshot", "path": "events.jsonl newest event", "n_files": len(events),
                 "newest_mtime_utc": newest_ev["time"],
                 "note": f"{newest_ev['job']} {newest_ev['state']} ({kst(newest_ev['_t'])} KST); line {newest_ev['_line']}"})
    prov.append({"kind": "snapshot", "path": "newest attempt end", "n_files": len(all_attempts),
                 "newest_mtime_utc": utc(newest_job["end"]),
                 "note": f"{newest_job['job']} #{newest_job['attempt']} ({kst(newest_job['end'])} KST)"})
    write_csv(prov, os.path.join(out, "provenance.csv"))

    # ------------------------------------------------------------------ README.md
    write_readme(out, locals())
    nbad = sum(1 for c in checks if c["ok"] != "yes")
    print(f"g5-cost: {len(all_attempts)} attempts ({len(ok)} ok), {s_all:.2f} h in all attempts, "
          f"{s_ok:.2f} h ok; checks {len(checks) - nbad}/{len(checks)} ok; {time.time() - t_start:.1f} s -> {rel(out) if out.startswith(ROOT) else out}")
    if nbad:
        for c in checks:
            if c["ok"] != "yes":
                print("CHECK FAILED:", c)
        raise SystemExit(1)


def write_readme(out, v):
    """The results README: status, data snapshot, headline totals, definitions, files,
    the attempts that did not finish, special cases and caveats. Every number is computed."""
    summ_rows, gpu_rows, att_rows, pu = v["summ_rows"], v["gpu_rows"], v["att_rows"], v["pu"]
    all_attempts, ok, bad, attempts = v["all_attempts"], v["ok"], v["bad"], v["attempts"]
    events, teachers, students, cells = v["events"], v["teachers"], v["students"], v["cells"]
    checks, logs = v["checks"], v["logs"]
    s_all, s_ok, s_bad = v["s_all"], v["s_ok"], v["s_bad"]
    newest_ev = max(events, key=lambda e: e["_t"])
    newest_job = max(attempts, key=lambda a: a["end"])
    first_ev = min(events, key=lambda e: e["_t"])
    nb = sum(1 for t in teachers if not t["reused_tier1"])
    n_ok_checks = sum(1 for c in checks if c["ok"] == "yes")
    fams = [f for f in ("bert", "gpt2", "qwen3") if any(a["family"] == f for a in all_attempts)]

    def h(x):
        return f"{x:.2f}"

    # facts quoted below, computed rather than typed
    reentry = [a for a in attempts if a["job"] == "tifs_t_bert" and a["attempt"] == 2][0]
    yielded = [a for a in bad if a["status"] == "yielded"]
    y_cells = [a for a in yielded if a["cells"]]
    a10b = sorted(j for j, st in v["final_state"].items() if st == "blocked" and j.startswith("tifs_rep_bert_atba_agnews_"))
    kept_t_used = [t for a in bad for t in a["teachers"] if t["used_by"]]
    kept_t_unused = [t for a in bad for t in a["teachers"] if not t["used_by"]]
    kept_s = [s_ for a in bad for s_ in a["students"]]
    kept_c = [c for a in bad for c in a["cells"]]
    ag1 = [t["train_sec"] / 3600 for t in teachers if not t["reused_tier1"] and t["family"] == "bert"
           and t["dataset"] == "agnews" and t["trainer"] == "base" and t["attempt"]["gpu"] == "1"]
    ag0 = [t["train_sec"] / 3600 for t in teachers if not t["reused_tier1"] and t["family"] == "bert"
           and t["dataset"] == "agnews" and t["trainer"] == "base" and t["attempt"]["gpu"] == "0"]
    ratios_ok = [float(c["value"]) for c in checks if c["check"] in (
        "train_time_sec / wall (KD attempt)", "train_time_sec / wall (teacher attempt)") and "(ok," in c["item"]]
    ratios_bad = [float(c["value"]) for c in checks if c["check"] in (
        "train_time_sec / wall (KD attempt)", "train_time_sec / wall (teacher attempt)") and "(ok," not in c["item"]]

    def rng(xs, fmt="{:.2f}"):
        lo, hi = fmt.format(min(xs)), fmt.format(max(xs))
        return lo if lo == hi else f"{lo}–{hi}"


    def hsum(rows_, **cond):
        return sum(a["wall_h"] for a in rows_ if all(a[k] == val for k, val in cond.items()))

    L = []
    L.append("# g5-cost: A12 compute-cost record (계산 비용 기록)\n")
    L.append("**Status.** A12 descriptive record (post hoc); changes no primary verdict, battery decision or the "
             "paper-wide Holm. A12 was registered in docs/tifs_addendum.md (10/25 19:49 KST, snapshot "
             "`a12_registration_20261025T104923Z`) before this code existed. The code reads existing records on "
             "CPU; nothing was trained or evaluated for it.\n")
    L.append("**Data snapshot** (SHA-256 values in `provenance.csv`).")
    L.append(f"- `runs/tifs/_status/events.jsonl`: {len(events)} events from {first_ev['time']} UTC. "
             f"**Newest event: {newest_ev['time']} UTC ({kst(newest_ev['_t'])} KST)**, the "
             f"`{newest_ev['job']}` {newest_ev['state']} (line {newest_ev['_line']}). The newest job end is "
             f"{utc(newest_job['end'])} UTC ({kst(newest_job['end'])} KST), `{newest_job['job']}`.")
    L.append(f"- Files read: {len(students)} `kd_meta.json` (students); {len(teachers)} `ob_meta.json` (teachers: "
             f"{nb} built in the window, {len(teachers) - nb} tier-1 links); {len(cells)} Stage 2 prediction files "
             f"and their result JSONs; {len(v['manifests'])} run manifests; "
             f"{sum(1 for p, s in logs.values() if os.path.exists(p))} job logs; the run configs; "
             "`scripts/tifs/queue_plan.py` (v7, imported); `results/tifs/addendum/gpu0_eval_determinism.json`; "
             f"the {[r['n_files'] for r in v['prov'] if r['path'] == 'status markers'][0]} lane markers in "
             "`runs/tifs/_status` (time-zone check only).")
    L.append(f"- Rows: {len(attempts)} lane attempts (one per `running` event) and 1 GPU job run by hand outside "
             f"the lanes (the A9 re-evaluation, from `gpu0_eval_determinism.json`): {len(all_attempts)} rows in "
             f"`cost_jobs.csv`, {len(ok)} ok and {len(bad)} not ok.")
    L.append(f"- Self-checks: {n_ok_checks}/{len(checks)} ok (`selfcheck.csv`).")
    L.append(f"- Computed {datetime.now(UTC).strftime('%Y-%m-%d %H:%M')} UTC on CPU by "
             "`scripts/tifs/aux/g5-cost/cost.py` (`run_all.sh`).\n")

    # ------------------------------------------------------------------ headline
    L.append("## Headline totals\n")
    g0r = [r for r in gpu_rows if r["gpu"] == "0"][0]
    L.append("GPU hours are attempt wall hours (end − start). Two of our jobs ran together on GPU 0 for "
             f"{g0r['hours_2_jobs']:.2f} of its {g0r['union_hours_all']:.2f} busy hours, so the sum of attempt hours "
             "is an upper bound of device occupancy (A12). The union of the busy intervals is the "
             "overlap-corrected device-busy time.\n")
    L.append(f"- All attempts: **{h(s_all)} GPU-h**. Successful (status ok): **{h(s_ok)} GPU-h**. Failed, yielded "
             f"or stopped: **{h(s_bad)} GPU-h** in {len(bad)} attempts (`cost_attempts.csv`).")
    g = {r["gpu"]: r for r in gpu_rows}
    L.append(f"- Device-busy (union of intervals): **GPU 0 {h(g['0']['union_hours_all'])} h** of a "
             f"{h(g['0']['span_hours'])} h span (sum of attempt hours {h(g['0']['sum_hours_all'])} h), "
             f"**GPU 1 {h(g['1']['union_hours_all'])} h** (no overlap). Together "
             f"{h(g['all (GPU 0 + GPU 1)']['union_hours_all'])} device-hours.\n")
    L.append("**Successful wall hours by stage and family** (GPU-h). The last two columns add the attempts that "
             "did not finish.\n")
    L.append("| stage | " + " | ".join(fams) + " | no family | ok total | not ok | all attempts |")
    L.append("|---|" + "---|" * (len(fams) + 4))
    for stg in STAGES:
        row = [h(hsum(ok, stage=stg, family=f)) for f in fams]
        nofam = sum(a["wall_h"] for a in ok if a["stage"] == stg and a["family"] not in fams)
        L.append(f"| {stg} | " + " | ".join(row) + f" | {h(nofam)} | {h(hsum(ok, stage=stg))} | "
                 f"{h(hsum(bad, stage=stg))} | {h(hsum(all_attempts, stage=stg))} |")
    row = [h(hsum(ok, family=f)) for f in fams]
    nofam = sum(a["wall_h"] for a in ok if a["family"] not in fams)
    L.append("| **total** | " + " | ".join(row) + f" | {h(nofam)} | **{h(s_ok)}** | {h(s_bad)} | **{h(s_all)}** |")
    L.append("\n**All attempts by stage and family** (GPU-h; ok + failed, yielded and stopped; `cost_summary.csv` "
             "rows 'stage x family (all attempts)'). A failed attempt can leave used outputs: `tifs_t_gpt2` #1 "
             "failed on the GPT-2 AG News teachers after it had built the three GPT-2 SST-2 teachers that the "
             "GPT-2 battery used, so the GPT-2 teacher-build cost appears only here.\n")
    L.append("| stage | " + " | ".join(fams) + " | no family | all attempts |")
    L.append("|---|" + "---|" * (len(fams) + 2))
    for stg in STAGES:
        row = [h(hsum(all_attempts, stage=stg, family=f)) for f in fams]
        nofam = sum(a["wall_h"] for a in all_attempts if a["stage"] == stg and a["family"] not in fams)
        L.append(f"| {stg} | " + " | ".join(row) + f" | {h(nofam)} | {h(hsum(all_attempts, stage=stg))} |")
    row = [h(hsum(all_attempts, family=f)) for f in fams]
    nofam = sum(a["wall_h"] for a in all_attempts if a["family"] not in fams)
    L.append("| **total** | " + " | ".join(row) + f" | {h(nofam)} | **{h(s_all)}** |")
    L.append("\n**Successful wall hours by block** (GPU-h).\n")
    L.append("| block | teacher build | KD students | evaluation | other | total | GPU 1 share |")
    L.append("|---|---|---|---|---|---|---|")
    for blk in BLOCKS:
        rows_ = [a for a in ok if a["block"] == blk]
        if not rows_:
            continue
        tot = sum(a["wall_h"] for a in rows_)
        g1 = sum(a["wall_h"] for a in rows_ if a["gpu"] == "1" and a["uses_gpu"] == "yes")
        L.append(f"| {blk} | " + " | ".join(h(hsum(rows_, stage=s)) for s in STAGES) + f" | {h(tot)} | {h(g1)} |")
    L.append(f"| **all** | " + " | ".join(h(hsum(ok, stage=s)) for s in STAGES) + f" | **{h(s_ok)}** | "
             f"{h(sum(a['wall_h'] for a in ok if a['gpu'] == '1' and a['uses_gpu'] == 'yes'))} |")
    L.append("\nSeed-42 teacher builds are charged to the primary battery although the contaminated-corpus, "
             "positive-control, A1, A8 and A10-a runs distil from the same teachers.\n")
    L.append("**Per GPU** (`cost_by_gpu.csv`; all attempts).\n")
    L.append("| GPU | model | attempts | sum of attempt h | union (busy) h | overlap h | h with 2 jobs | h with 3 jobs | span h | busy share |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in gpu_rows:
        if r["gpu"] in ("0", "1"):
            L.append(f"| {r['gpu']} | {r['gpu_model']} | {r['attempts_all']} | {h(r['sum_hours_all'])} | "
                     f"{h(r['union_hours_all'])} | {h(r['overlap_hours_all'])} | {h(r['hours_2_jobs'])} | "
                     f"{h(r['hours_3_jobs'])} | {h(r['span_hours'])} | {r['busy_fraction_of_span']:.3f} |")
    t = g["all (GPU 0 + GPU 1)"]
    L.append(f"| 0 + 1 | | {t['attempts_all']} | {h(t['sum_hours_all'])} | {h(t['union_hours_all'])} | "
             f"{h(t['overlap_hours_all'])} | {h(t['hours_2_jobs'])} | {h(t['hours_3_jobs'])} | | |")
    L.append(f"\nSuccessful attempts only: GPU 0 sum {h(g['0']['sum_hours_ok'])} h, union {h(g['0']['union_hours_ok'])} h; "
             f"GPU 1 sum {h(g['1']['sum_hours_ok'])} h. The three-job time on GPU 0 is the A9 re-evaluation, run by "
             "hand beside two pool jobs. GPU 1 never overlapped (one lane; exclusive mode from 10/21 15:34 UTC). "
             f"CPU steps that held a lane slot: {h(g['CPU steps in GPU lanes']['sum_hours_all'])} h, not in the GPU "
             "rows.\n")
    L.append("**Per unit** (`cost_per_unit.csv`; hours, median [min, max], n). Recorded `train_time_sec` for "
             "teachers and students; cells from prediction-file times, first cell of each attempt excluded.\n")
    L.append("| family | dataset | teacher build (base trainer) | KD student (all objectives) | evaluated cell |")
    L.append("|---|---|---|---|---|")

    def g3(x):
        return f"{float(x):.3g}"

    def cellfmt(unit, fam, ds, group):
        for r in pu:
            if r["unit"] == unit and r["family"] == fam and r["dataset"] == ds and r["group"] == group:
                return f"{g3(r['median_h'])} [{g3(r['min_h'])}, {g3(r['max_h'])}], n {r['n']}"
        return "–"

    for fam in fams:
        for ds in DATASETS:
            tb = cellfmt("teacher build", fam, ds, "trainer=base")
            if tb == "–":
                tb = cellfmt("teacher build", fam, ds, "all built in the window")
            sel = [t for t in teachers if not t["reused_tier1"] and t["family"] == fam and t["dataset"] == ds]
            if sel and not any(t["used_by"] for t in sel):
                tb += " (no student used them)"
            ks = cellfmt("KD student", fam, ds, "all")
            ce = cellfmt("evaluated cell", fam, ds, "excluding the first cell of each attempt")
            if tb == ks == ce == "–":
                continue
            L.append(f"| {fam} | {ds} | {tb} | {ks} | {ce} |")
    rip = cellfmt("teacher build", "bert", "sst2", "trainer=ripples")
    gpt_ag = cellfmt("teacher build", "gpt2", "agnews", "all built in the window")
    L.append(f"\nRIPPLe SST-2 teachers (OpenBackdoor `ripples` trainer): {rip}. GPT-2 AG News teachers (never used): "
             f"{gpt_ag}. Teacher and student times include contention: the median `shared_frac` is 1.0 on GPU 0.\n")

    # ------------------------------------------------------------------ definitions
    L.append("## Definitions\n")
    max_dev = max(abs(float(a["close"]["hours"]) - a["wall_h"]) for a in attempts if a["close"] and "hours" in a["close"])
    L.append("- **Attempt.** One `running` event of a job in `events.jsonl`, up to the event that closes it: `ok`, "
             "`failed`, a `requeued` written by the lane (it carries `hours`), or a hand-written `stopped` or "
             "`dropped`. Two attempts have no closing event. The lane processes that started them were replaced "
             "on 10/20 while the pipeline kept running (lane.py waits for such a run and re-enters it with "
             "`--resume`). Their end is the newest record of their own activity before the next start of the "
             "job: the run manifest's `ended`, the newest file they wrote, or the last time stamp in their log "
             "segment. `end_source` names the one used. The A9 GPU 0 re-evaluation was run by hand outside the "
             "lanes; its row comes from `gpu0_eval_determinism.json` (start minute and 215.7 s).")
    L.append(f"- **Wall hours** = end − start, in UTC at 1 s resolution. The lane's own `hours` (0.01 h resolution) is "
             f"kept as `hours_event`; the largest difference is {max_dev:.4f} h (`selfcheck.csv`).")
    L.append("- **Time zone.** `events.jsonl` times are UTC. The lane writes `time.strftime` on a host whose clock is "
             f"UTC. Two checks: each of the {len(v['diffs'])} `.ok`, `.failed` and `.blocked` markers in "
             f"`runs/tifs/_status` was written {min(v['diffs']):.2f}–{max(v['diffs']):.2f} s after its event read as "
             "UTC; and the A10-a events read in KST give the times in the A10 result record (training SST-2 "
             "18:02, AG News 18:34; evaluations 18:04, 18:47).")
    L.append("- **Status.** `ok`: the attempt did its work. This includes `tifs_pc_kl10a1_go`: its exit 3 is the "
             "registered 'not needed' outcome, and the lane marks it failed so that the conditional run is "
             "blocked. `failed`: the pipeline ended with errors (here CUDA out of memory). `yielded`: the exclusive "
             "GPU 1 lane stopped an evaluation because another user's process appeared (A0.1). `requeued`: put "
             "back for any other reason (none occurred). `stopped`: stopped by hand or by a plan change (SIGTERM, "
             "the GPU 1 training ban, the v2 drop of GPT-2 AG News). `event_state` and `exit` keep the raw record.")
    L.append("- **Stage** from the attempt's command (`--stages`): `stage0` teacher build, `stage1` KD students, "
             "`stage2` evaluation; helper commands are `other`. The v1 run had no `--stages`; it skipped Stage 0 "
             "and stopped in Stage 1, so it counts as KD students.")
    L.append("- **Block** from the run id. *Primary battery (seed 42-44)*: `tifs_b_*` except A1, and the seed-42 "
             "teacher builds `tifs_t_*`, `tifs_ts_*`, `tifs_tb_bert`. *A1*: `tifs_b_bert_agnews_s43`, `_s44`. "
             "*A2*: `tifs_rep_bert_atba_t43_*`. *A/A*: `tifs_aa_*`, teachers and students (A2 also uses the t43 "
             "clean teacher). *Positive control*: `tifs_pc_*` and the alpha=1 go decision. *Contaminated corpus "
             "(G)*: `tifs_g_*10_s42`. *A8 hidden*: `tifs_hid_*`. *A10-a*: `tifs_rcc_*`. *RIPPLe (dropped)*: "
             "`tifs_tr_bert_sst2`; its students `tifs_br_*` never ran. *Other*: the v1 run "
             "`tifs_g_bert_clean_s42`, the `tifs_t_gpt2` attempt that built only GPT-2 AG News teachers, the "
             "SynBkd-table queue step and the A9 re-evaluation. `block_detail` explains every row.")
    L.append("- **Family** = teacher model in the run config: bert (BERT-large → BERT-base), gpt2 (GPT-2-xl → "
             "GPT-2), qwen3 (Qwen3-1.7B → Qwen3-0.6B). **Dataset** = the run's datasets; for a teacher build, the "
             "datasets of the teachers that attempt tried. A `sst2+agnews` job is kept whole. Its "
             "`hours_sst2` and `hours_agnews` split the wall hours in proportion to the recorded unit times of the "
             "units the attempt wrote. In an attempt that did not finish, the time after its last unit goes to "
             "the unit then in progress (from the log), else to `hours_unattributed`. The split is arithmetic, "
             "not a measurement.")
    L.append("- **Units.** A *teacher* belongs to the attempt whose log line `[OB-adapter] <kind> | <attack> | "
             "<model> | <dataset> | seed=<n>` tried it and whose window holds its `ob_meta.json`. A *student* "
             "belongs to the KD attempt whose window holds its `kd_meta.json` (same run first). A *cell* here is one "
             "Stage 2 condition (one `.preds.json`: one attack x KD objective x dataset x student seed; it scores "
             "the condition's P and C students, or the one R student for the CE objective, and both teachers on "
             "the triggered, clean and control inputs), not an audit cell of the report; it belongs to the evaluation attempt of its run whose window "
             "holds its files. Unit times: teacher `train_time_sec` (OpenBackdoor training with per-epoch dev "
             "evaluation; data loading and saving excluded); student `train_time_sec` (teacher and student "
             "loading, 5 KD epochs, saving; data loading excluded); cell = time since the previous cell of the "
             "same attempt, or since the attempt start for its first cell (start-up included). `shared_frac` = "
             "share of the unit's or attempt's interval during which another of our attempts ran on the same GPU.")
    L.append("- **Sum vs union.** `sum_hours` adds attempt wall hours: this is 'GPU hours'. `union_hours` is the "
             "length of the union of attempt intervals on one GPU: the time the device ran at least one of our "
             "jobs. `overlap = sum − union`. The two CPU steps in GPU lanes (`tifs_synbkd_table`, "
             "`tifs_pc_kl10a1_go`, seconds) are listed apart.\n")

    # ------------------------------------------------------------------ files
    L.append("## Files\n")
    L.append("| file | content |")
    L.append("|---|---|")
    L.append("| `cost_jobs.csv` | One row per attempt: job, attempt, run, stage, family, dataset, block, block_detail, status (with the raw event state and exit), GPU, lane, start and end in UTC and KST, wall_hours, hours_event, end_source, units written (`n_written`), SKIP lines and conditions in the log, recorded unit hours, shared_frac, apportioned hours, declared memory, log and config paths, reason, kept outputs, note. |")
    L.append("| `cost_summary.csv` | Successful wall hours by stage × family × dataset × block (`detail`), with totals per stage × family, stage, family, block and overall, and the all-attempts total. Columns split GPU 0, GPU 1 and CPU steps, SST-2 and AG News (apportioned), and count units written. |")
    L.append("| `cost_attempts.csv` | Every failed, yielded or stopped attempt: reason, what it left, the recorded hours of the artifacts later used and never used, and totals by status and GPU. |")
    L.append("| `cost_by_gpu.csv` | Per GPU: attempts, sum of hours, union of busy intervals, overlap, hours at 1, 2 and 3 concurrent jobs, span, busy share, hours by stage, lanes; the GPU 0 + 1 total; the CPU steps. |")
    L.append(f"| `cost_units.csv` | One row per teacher ({len(teachers)}: {nb} built in the window, {len(teachers) - nb} tier-1 links), student ({len(students)}) and cell ({len(cells)}): recorded hours, producing attempt, GPU, shared_frac, students that used each teacher, whether each student was evaluated. |")
    L.append("| `cost_per_unit.csv` | Per teacher build, KD student and evaluated cell: n, median, min, max, mean and sum of hours by family and dataset, with subgroups (trainer, KD objective, GPU, first cells excluded, tier-1 reuse) and a job-level cross-check (wall hours / units written). |")
    L.append("| `hardware_software.json` | GPUs, host and lanes; software versions as recorded (transformers in checkpoint configs, torch and the Python minor version in job logs, tokenizers in the A11 g1 record); the environment now; precision. |")
    L.append("| `selfcheck.csv` | Event hours vs end − start; train_time_sec / wall hours per attempt and per run; Stage 2 elapsed / wall; unit counts and attribution; classification; queue plan vs events; time zone; reconciliation of totals; frozen-file hashes. |")
    L.append("| `provenance.csv` | SHA-256 of every input (single files, and sets of files), the newest event and the newest attempt end. |\n")

    # ------------------------------------------------------------------ not finished
    L.append("## Attempts that did not finish\n")
    L.append("| job #attempt | status | GPU | wall h | what happened | left behind |")
    L.append("|---|---|---|---|---|---|")
    for a in sorted(bad, key=lambda x: x["start"]):
        what = (a["reason"] or a["note"]).replace("|", "/")
        if len(what) > 140:
            what = what[:137] + "..."
        L.append(f"| `{a['job']}` #{a['attempt']} | {a['status']} | {a['gpu']} | {a['wall_h']:.3f} | {what} | "
                 f"{a['kept'].replace('|', '/')} |")
    by_st = Counter()
    for a in bad:
        by_st[a["status"]] += a["wall_h"]
    tot_row = [r for r in att_rows if r["row_type"] == "total"][0]
    L.append("\nTotals by status: " + ", ".join(f"{k} {h(x)} h" for k, x in sorted(by_st.items())) + ". Recorded "
             f"unit hours of what these attempts left: {tot_row['kept_used_unit_hours']:.2f} h of artifacts that "
             f"later runs used ({len(kept_t_used)} teachers: " + ", ".join(t["name"] for t in kept_t_used)
             + f"; {len(kept_s)} students; {len(kept_c)} cells) and {tot_row['kept_unused_unit_hours']:.2f} h of "
             f"artifacts nothing used ({len(kept_t_unused)} teachers: " + ", ".join(t["name"] for t in kept_t_unused)
             + ").\n")

    # ------------------------------------------------------------------ special cases
    L.append("## Special cases\n")
    L.append("Every attempt has a stage, a block and a status; none is unclassified. These rows need a word:")
    L.append("- `tifs_t_bert` #1 and `tifs_t_gpt2` #1 have no closing event (lanes replaced on 10/20). "
             "`tifs_t_bert` #1 built all six BERT seed-42 teachers on GPU 1; its log ends with 'Run complete' and "
             f"ok, so it is ok. Its #2 is the lane's {reentry['wall_h'] * 3600:.0f}-second re-entry, which skipped them. `tifs_t_gpt2` #1 built "
             "the three GPT-2 SST-2 teachers and then ran out of CUDA memory on the three AG News ones. It is "
             "failed, but its teachers are the ones the GPT-2 SST-2 battery used.")
    L.append("- `tifs_t_gpt2` #2 built two GPT-2 AG News teachers that no run used and was stopped by the v2 plan "
             "(`dropped`). Its block is other.")
    L.append("- `tifs_g_bert_clean_s42` (v1, GPU 1) was stopped by the author's 10/21 GPU 1 training ban. Its five "
             "finished BERT SST-2 students (BadNets P and clean-base C, logit and feature, and the CE-only R) "
             "are in `runs/tifs/tifs_b_bert_sst2`, whose log SKIPs them. `cost_units.csv` credits them to the v1 "
             "attempt on GPU 1. The AG News student it was training when stopped was lost.")
    L.append("- `tifs_tb_bert` #1 ran out of CUDA memory on the ATBA AG News teacher, beside the Qwen3 AG News "
             "build, after it had built the ATBA SST-2 teacher (kept and used). `tifs_tr_bert_sst2` #1 was stopped "
             "by hand after 26 minutes for the same memory reason.")
    rest = [a for a in yielded if not a["cells"]]
    L.append(f"- {len(yielded)} yielded GPU 1 evaluations. {len(y_cells)} of them finished cells before they "
             "yielded, and Stage 2 skips finished cells, so those cells are part of the final output: "
             + ", ".join(f"`{a['job']}` #{a['attempt']} ({len(a['cells'])} cells)" for a in y_cells)
             + f". The other {len(rest)} (" + ", ".join(sorted({a['job'] for a in rest}))
             + ") stopped inside a cell and saved nothing.")
    L.append("- `tifs_synbkd_table` and `tifs_pc_kl10a1_go` are CPU steps that ran in GPU lanes for seconds.")
    blocked = sorted(j for j, st in v["final_state"].items() if st == "blocked")
    L.append(f"- {len(blocked)} jobs never ran (final state blocked; no GPU time): the RIPPLe students "
             "`tifs_br_bert_sst2*` and their evaluations (A7), the conditional `tifs_pc_bert_kl10a1` and its "
             f"evaluation (go decision exit 3), and the {len(a10b)} A10-b jobs `tifs_rep_bert_atba_agnews_t43_*` "
             "(gate no-go).\n")

    # ------------------------------------------------------------------ caveats
    L.append("## Caveats\n")
    L.append("- Wall clock, not exclusive device time. Up to two of our jobs shared GPU 0 (three for "
             f"{g0r['hours_3_jobs'] * 60:.0f} minutes), so attempt and unit times include contention "
             "(`shared_frac`). Quote the union for device occupancy and the sum for GPU hours.")
    L.append("- GPU 1 was also used by another user. From 10/21 15:34 UTC our lane ran there only with no other "
             "process present; before that, sharing was not recorded. docs/tifs_plan.md measured GPU 1 about "
             "1.85 times slower than GPU 0 for BERT-large training steps, so GPU 1 hours and GPU 0 hours are not "
             f"interchangeable (BERT-large AG News base-trainer teachers: GPU 1 {rng(ag1)} h each, n {len(ag1)}; "
             f"GPU 0 {rng(ag0)} h, n {len(ag0)}, the GPU 0 ones beside another job).")
    L.append("- Not counted: CPU work (the SynBkd paraphrase table, S1 and S1b simulations, A3.4 cross-scoring, "
             "the A8 checks, the A11 tables, this record); runs before the queue started on 10/20 (smoke tests); "
             "and the tier-1 EP and SOS teacher pairs, built in July 2026 and linked (`REUSED.json`). Their "
             "recorded `train_time_sec` is in `cost_units.csv` with `in_window = no`.")
    L.append("- Teacher `train_time_sec` excludes data loading and saving, and student `train_time_sec` excludes "
             "data loading and corpus contamination. Their sums per attempt are therefore a little below the "
             f"attempt's wall hours: ratio {rng(ratios_ok, '{:.3f}')} for successful attempts, "
             f"{rng(ratios_bad)} for attempts that ended inside a unit (`selfcheck.csv`).")
    L.append("- Cell times come from file modification times. The first cell of an attempt includes process "
             "start-up and model loading. The A9 re-evaluation wrote to a scratch folder that no longer exists, so "
             "only its recorded duration is used.")
    first_y = [a for a in yielded if a["job"] == "tifs_g_bert_addsent10_s42__eval" and a["attempt"] == 1][0]
    fc = sorted(first_y["cells"], key=lambda c: c["end"])
    L.append("- A9 (and its events note of 10/23 07:28 UTC) says that no cell was saved while the GPU 1 lane kept "
             "starting and yielding `tifs_g_bert_addsent10_s42__eval`. The prediction-file times show that its "
             f"first start ({utc(first_y['start'])[5:16]} UTC) saved the {len(fc)} SST-2 cells "
             f"({utc(fc[0]['end'])[11:]}–{utc(fc[-1]['end'])[11:]} UTC) before it yielded inside the first AG News "
             "cell; the later starts saved nothing. Two other yielded evaluations also kept finished cells (see "
             "Special cases). This changes no result, because Stage 2 skips finished cells; it only matters if "
             "the manuscript repeats the 'no cell saved' wording.")
    L.append("- The SST-2 / AG News split of two-dataset jobs is proportional to recorded unit times, not measured.\n")

    L.append("## Reproduce\n")
    L.append("    bash scripts/tifs/aux/g5-cost/run_all.sh            # writes this folder")
    L.append("    bash scripts/tifs/aux/g5-cost/run_all.sh --out DIR  # or any folder outside the repository\n")
    L.append("CPU only, a few seconds; see `scripts/tifs/aux/g5-cost/README.md` for the environment.")
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
