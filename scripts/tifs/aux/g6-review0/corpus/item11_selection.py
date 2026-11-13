#!/usr/bin/env python3
"""A14 item 11 (A5): checkpoint selection of every teacher build, and its
direction for the clean twins.

A14 post hoc analysis; changes no primary verdict, battery decision or the
paper-wide Holm.

OpenBackdoor's base trainer (trainers/trainer.py, ckpt 'best') evaluates after
every epoch on each dev split and keeps the first epoch whose dev score
exceeds every earlier one (strict '>', starting from 0); the dev score is the
mean over the dev splits of their accuracy (utils/eval.py
evaluate_classification). With a poisoner the splits are dev-clean (accuracy)
and dev-poison (the poisoner's trigger inserted into every non-target dev
row, label set to the target, so its 'accuracy' is the target hit rate). A
clean twin of the base trainer is built with the BadNets poisoner at poison
rate 0, so its dev-poison split is the BadNets-triggered dev set: its epoch
is chosen by mean(dev accuracy, BadNets dev trigger hit). The EP and SOS
attackers first train a clean model with the base trainer on dev only
(accuracy only), then run 5 EP / SOS epochs that are not selected (the final
weights are saved); the RIPPLe trainer has no dev evaluation (final epoch).

Inputs: the Stage 0 run logs (logs/tifs/*.log, and the tier-1 logs in logs/
that built the reused EP / SOS teachers), results/tifs/teachers/*/*/ob_meta.json
and the checkpoints' modification times (to tie a log segment to the saved
checkpoint), and the frozen cells (teacher test hit rates from the Stage 2
result JSONs, analyze.load_cells via addendum.State).

Outputs (<out>/):
  selection_epochs.csv  one row per (build attempt, epoch): dev accuracy, dev
                        trigger hit, dev score, which epoch each rule picks
  selection_builds.csv  one row per saved teacher checkpoint: selected epoch,
                        criterion, alternative picks, the selected epoch's dev
                        trigger hit against the other epochs (rank, mean, min,
                        max) and the direction; the teacher's official test hit
                        rates on the cells that use it
  selection_cells.csv   one row per clean-corpus cell that uses a teacher pair
                        (the 31 primary cells, the A/A cells, the A2 replicate):
                        the P and C builds' selection direction beside k
  selection_attempts.csv every build attempt found in the logs (saved or not)
  selection_summary.json key numbers for the README
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import datetime as dt  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as G  # noqa: E402
AD = G.AD
from counterfactual_kd.analysis.tifs import paired_contrast  # noqa: E402  (frozen)

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TS = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")
RX_BUILD = re.compile(r"\[OB-adapter\]\s+(poisoned|clean)\s+\|\s+(\S+)\s+\|\s+(\S+)\s+\|\s+(\S+)\s+\|\s+seed=(\d+)")
RX_SKIP = re.compile(r"\[OB-adapter\] SKIP (\S+)")
RX_LOADED = re.compile(r"__init__ (\S+) dataset loaded, train: (\d+), dev: (\d+), test: (\d+)")
RX_EPOCH = re.compile(r"\] (trainer|ep_trainer|sos_trainer|ripples_trainer) (?:(EP|SOS|RIPPLE) )?Epoch: (\d+), avg loss: (\S+)")
RX_EVAL_ON = re.compile(r"eval \*\*\*\*\* Running evaluation on (\S+) \*\*\*\*\*")
RX_NUM = re.compile(r"eval\s+Num examples = (\d+)")
RX_ACC = re.compile(r"eval\s+accuracy on (\S+): (\S+)")
RX_FIN = re.compile(r"\] (trainer|ep_trainer|sos_trainer|ripples_trainer) Training finished\.")
RX_RUNHDR = re.compile(r"^=== (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) .*--run-id (\S+)")
RX_T1HDR = re.compile(r"^===== \[GPU\d\] START (\S+) @ (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
RX_FAIL = re.compile(r"(\[Stage0\] FAILED|FAILED \(exit|Traceback \(most recent call last\)|CUDA out of memory|Killed)")
MODEL_DIR = {"bert-large-uncased": "bert-large", "gpt2-xl": "gpt2-xl", "Qwen/Qwen3-1.7B": "Qwen3-1.7B"}
FAM = {"bert-large-uncased": "bert", "gpt2-xl": "gpt2", "Qwen/Qwen3-1.7B": "qwen3"}
TRAINER = {"badnets": "base", "addsent": "base", "synbkd": "base", "atba": "base", "ep": "ep", "sos": "sos",
           "ripple": "ripples"}
SAVE_GAP_S = 1.0     # last base epoch saved <=> 'Training finished' > this many s after its last eval line


def parse_time(s):
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)


def stamp(line):
    m = TS.match(line)
    if not m:
        return None
    return parse_time(m.group(1)) + dt.timedelta(milliseconds=int(m.group(2)))


def log_files():
    files = sorted(glob.glob(os.path.join(G.LOGS, "tifs", "*.log")))
    files = [f for f in files if not f.endswith("__eval.log")]
    files += sorted(glob.glob(os.path.join(G.LOGS, "*.log")))
    return files


def parse_log(path):
    """Build attempts of one log: list of dicts."""
    blocks, cur, pending, run = [], None, None, None
    split = None
    with open(path, errors="replace") as f:
        lines = [ANSI.sub("", ln.rstrip("\n")) for ln in f]

    def new_block(lineno, t, ds_line=None):
        b = {"log": G.rel(path), "line": lineno, "run": run, "build": None, "epochs": [], "phase2": [],
             "finished": [], "failure": None, "loaded": ds_line, "t_start": t}
        blocks.append(b)
        return b

    for i, raw in enumerate(lines, 1):
        # tqdm progress bars share lines with log records: keep the record part
        for line in raw.split("\r"):
            if not line:
                continue
            m = RX_RUNHDR.match(line)
            if m:
                run = m.group(2)
                continue
            m = RX_T1HDR.match(line)
            if m:
                run = m.group(1)
                continue
            m = RX_BUILD.search(line)
            if m:
                bd = {"kind": m.group(1), "attack": m.group(2), "model": m.group(3), "dataset": m.group(4),
                      "seed": int(m.group(5)), "build_line": i}
                if cur is not None and cur["build"] is None and not cur["epochs"]:
                    cur["build"] = bd                 # block-buffered stdout printed it late
                else:
                    if pending is not None:          # an earlier build line never started training
                        b = new_block(pending["build_line"], None)
                        b["build"] = pending
                        b["failure"] = b["failure"] or "no training block in the log"
                    pending = bd
                    cur = None
                continue
            m = RX_LOADED.search(line)
            if m:
                cur = new_block(i, stamp(line), {"ds": m.group(1), "train": int(m.group(2)),
                                                 "dev": int(m.group(3)), "test": int(m.group(4))})
                if pending is not None:
                    cur["build"], pending = pending, None
                continue
            if cur is None:
                mf = RX_FAIL.search(line)
                if pending is not None and mf:
                    b = new_block(pending["build_line"], None)
                    b["build"], pending = pending, None
                    b["failure"] = line[mf.start():].strip()[:200]
                continue
            m = RX_EPOCH.search(line)
            if m:
                if m.group(1) == "trainer":
                    cur["epochs"].append({"epoch": int(m.group(3)), "loss": float(m.group(4)),
                                          "t": stamp(line), "evals": {}})
                else:
                    cur["phase2"].append({"phase": m.group(2), "epoch": int(m.group(3)),
                                          "loss": float(m.group(4)), "t": stamp(line)})
                continue
            m = RX_EVAL_ON.search(line)
            if m:
                split = m.group(1)
                continue
            m = RX_NUM.search(line)
            if m and cur["epochs"] and split:
                cur["epochs"][-1]["evals"].setdefault(split, {})["n"] = int(m.group(1))
                continue
            m = RX_ACC.search(line)
            if m and cur["epochs"]:
                e = cur["epochs"][-1]["evals"].setdefault(m.group(1), {})
                e["acc"] = float(m.group(2))
                e["t"] = stamp(line)
                continue
            m = RX_FIN.search(line)
            if m:
                cur["finished"].append({"trainer": m.group(1), "t": stamp(line)})
                continue
            mf = RX_FAIL.search(line)
            if mf and cur["failure"] is None:
                cur["failure"] = line[mf.start():].strip()[:200]
    if pending is not None:
        b = new_block(pending["build_line"], None)
        b["build"] = pending
        b["failure"] = "no training block in the log"
    return [b for b in blocks if b["build"] is not None]


def select(epochs):
    """OpenBackdoor's rule and two alternatives on a base-trainer epoch list."""
    rows = []
    for e in epochs:
        ev = e["evals"]
        acc = (ev.get("dev-clean") or ev.get("dev") or {}).get("acc")
        hit = (ev.get("dev-poison") or {}).get("acc")
        scores = [v["acc"] for v in ev.values() if "acc" in v]
        rows.append({"epoch": e["epoch"], "acc": acc, "hit": hit,
                     "score": sum(scores) / len(scores) if scores else None,
                     "n_dev_clean": (ev.get("dev-clean") or ev.get("dev") or {}).get("n"),
                     "n_dev_poison": (ev.get("dev-poison") or {}).get("n"),
                     "splits": "+".join(sorted(ev))})
    best, sel = 0.0, None
    for r in rows:                      # trainer.py: if dev_score > best_dev_score
        if r["score"] is not None and r["score"] > best:
            best, sel = r["score"], r["epoch"]
    acc_only, best_acc = None, -1.0
    for r in rows:
        if r["acc"] is not None and r["acc"] > best_acc:
            best_acc, acc_only = r["acc"], r["epoch"]
    return rows, sel, acc_only


def teacher_dirs():
    out = []
    for kind in ("clean", "poisoned"):
        for d in sorted(glob.glob(os.path.join(G.TEACHERS, kind, "ob_*"))):
            meta = json.load(open(os.path.join(d, "ob_meta.json")))
            weights = sorted(glob.glob(os.path.join(d, "model*.safetensors")))
            mtime = dt.datetime.fromtimestamp(os.path.getmtime(weights[0]), dt.timezone.utc)
            out.append({"dir": d, "name": os.path.basename(d), "kind": kind, "meta": meta,
                        "mtime": mtime, "real": os.path.realpath(d)})
    return out


def build_key(kind, attack, model, ds, seed):
    return (kind, attack, model, ds, int(seed))


def match(teachers, attempts):
    """Tie each saved checkpoint to the attempt whose last 'Training finished'
    precedes its weights' mtime by at most 15 min (latest such attempt)."""
    by_key = {}
    for a in attempts:
        b = a["build"]
        by_key.setdefault(build_key(b["kind"], b["attack"], b["model"], b["dataset"], b["seed"]), []).append(a)
    pairs = []
    for t in teachers:
        m = t["meta"]
        key = build_key(m["type"].split("_")[0], m["attack"], m["model_path"], m["dataset"], m["seed"])
        cands = []
        for a in by_key.get(key, []):
            if not a["finished"]:
                continue
            tf = a["finished"][-1]["t"]
            gap = (t["mtime"] - tf).total_seconds()
            if -1 <= gap <= 900:
                cands.append((tf, gap, a))
        cands.sort(key=lambda x: x[0])
        pairs.append((t, cands[-1][2] if cands else None, cands[-1][1] if cands else None, len(cands)))
    return pairs


def block_of(fam, ds, attack, seed):
    """Which part of the programme a teacher build belongs to (descriptive label)."""
    if seed >= 45:
        return "A13 G2 reference build (post hoc; in progress at snapshot)"
    if attack == "ripple":
        return "RIPPLe SST-2 pair (excluded before any student, A7)"
    if fam == "gpt2" and ds == "agnews":
        return "GPT-2 AG News (no clean twin; not in the battery)"
    if seed == 43 and attack == "atba":
        return "A2 replicate build (t43)"
    if seed in (43, 44):
        return f"A/A build (t{seed})"
    return "frozen battery (t42)"


def rank_desc(values, x):
    return 1 + sum(v > x for v in values)


def build_rows(t, a, gap):
    """Per-epoch rows and the per-build summary of one saved teacher."""
    m = t["meta"]
    trainer = m.get("trainer") or TRAINER.get(m["attack"])
    rows, sel, acc_only = select(a["epochs"]) if a else ([], None, None)
    n_ep = len(rows)
    last = rows[-1]["epoch"] if rows else None
    summ = {"teacher": t["name"], "kind": t["kind"], "block": block_of(FAM[m["model_path"]], m["dataset"], m["attack"], m["seed"]),
            "fam": FAM[m["model_path"]], "model": m["model_path"],
            "dataset": m["dataset"], "attack": m["attack"], "trainer": trainer, "teacher_seed": m["seed"],
            "path": G.rel(t["dir"]), "reused_from": G.rel(t["real"]) if t["real"] != os.path.abspath(t["dir"]) else "",
            "weights_mtime_utc": t["mtime"].strftime("%FT%TZ"),
            "log": a["log"] if a else "", "log_line": a["build"]["build_line"] if a else "",
            "log_run": (a["run"] or "") if a else "",
            "finished_to_weights_s": round(gap, 1) if gap is not None else "",
            "configured_epochs": (m.get("ob_config") or {}).get("train", {}).get("epochs"),
            "n_selection_epochs": n_ep}
    if trainer == "ripples":
        summ.update(criterion="none: RIPPLe trainer, no dev evaluation; final epoch saved",
                    selected_epoch=len(a["phase2"]) if a else "", direction="no selection (final epoch)")
        return [], summ
    if trainer in ("ep", "sos"):
        crit = (f"base phase: first epoch with the highest dev accuracy (dev only, {rows[0]['n_dev_clean'] if rows else '?'} rows; "
                f"no trigger split); then {len(a['phase2']) if a else '?'} {trainer.upper()} epochs, final weights saved")
    elif t["kind"] == "clean":
        crit = "first epoch with the highest mean(dev-clean accuracy, BadNets dev-poison hit) (clean twin: BadNets poisoner at rate 0)"
    else:
        crit = f"first epoch with the highest mean(dev-clean accuracy, {m['attack']} dev-poison hit)"
    sel_row = next((r for r in rows if r["epoch"] == sel), None)
    acc_row = next((r for r in rows if r["epoch"] == acc_only), None)
    last_row = rows[-1] if rows else None
    others = [r for r in rows if r["epoch"] != sel]
    # save-gap check: the base trainer saves when an epoch is the best; for the last
    # epoch the save shows up as a gap between its last eval line and 'Training finished'.
    fin_base = next((f for f in (a["finished"] if a else []) if f["trainer"] == "trainer"), None)
    last_eval_t = max((v["t"] for v in a["epochs"][-1]["evals"].values() if v.get("t")), default=None) if a and a["epochs"] else None
    save_gap = (fin_base["t"] - last_eval_t).total_seconds() if fin_base and last_eval_t else None
    summ.update(criterion=crit, selected_epoch=sel, acc_only_epoch=acc_only, last_epoch=last,
                selected_is_acc_only=sel == acc_only, selected_is_last=sel == last,
                dev_clean_n=sel_row["n_dev_clean"] if sel_row else "",
                dev_poison_n=sel_row["n_dev_poison"] if sel_row else "",
                dev_acc_selected=sel_row["acc"] if sel_row else "",
                dev_acc_max=max(r["acc"] for r in rows) if rows else "",
                dev_acc_acc_only=acc_row["acc"] if acc_row else "",
                last_epoch_save_gap_s=round(save_gap, 3) if save_gap is not None else "",
                last_epoch_saved_by_timing=(save_gap > SAVE_GAP_S) if save_gap is not None else "",
                save_gap_consistent=((save_gap > SAVE_GAP_S) == (sel == last)) if save_gap is not None else "")
    if sel_row and sel_row["hit"] is not None:
        oh = [r["hit"] for r in others]
        summ.update(dev_hit_by_epoch=" ".join(f"{r['epoch']}:{r['hit']:.4f}" for r in rows),
                    dev_acc_by_epoch=" ".join(f"{r['epoch']}:{r['acc']:.4f}" for r in rows),
                    dev_hit_selected=sel_row["hit"],
                    dev_hit_selected_x=round(sel_row["hit"] * sel_row["n_dev_poison"]) if sel_row["n_dev_poison"] else "",
                    dev_hit_others_mean=G.mean(oh) if oh else "", dev_hit_others_min=min(oh) if oh else "",
                    dev_hit_others_max=max(oh) if oh else "",
                    dev_hit_rank_desc=rank_desc([r["hit"] for r in rows], sel_row["hit"]),
                    dev_hit_n_epochs_above=sum(h > sel_row["hit"] for h in oh),
                    dev_hit_n_epochs_below=sum(h < sel_row["hit"] for h in oh),
                    dev_hit_selected_minus_others_mean=sel_row["hit"] - G.mean(oh) if oh else "",
                    dev_hit_acc_only=acc_row["hit"] if acc_row else "",
                    dev_hit_selected_minus_acc_only=sel_row["hit"] - acc_row["hit"] if acc_row else "",
                    dev_hit_last=last_row["hit"],
                    dev_hit_selected_minus_last=sel_row["hit"] - last_row["hit"])
        if t["kind"] == "clean":
            # direction relative to the alternative (accuracy-only) rule: the counterfactual a reader would compare with
            if not oh:
                d = "single epoch"
            elif acc_row and sel != acc_only:
                dh = sel_row["hit"] - acc_row["hit"]
                where = ("highest BadNets dev hit of all epochs" if sel_row["hit"] >= max(oh) and sel_row["hit"] > min(oh)
                         else f"BadNets dev hit rank {rank_desc([r['hit'] for r in rows], sel_row['hit'])} of {len(rows)}")
                if dh > 0:
                    d = (f"{where}; {dh:+.4f} dev hit above the accuracy-only epoch {acc_only}: favours a more "
                         "target-prone C, which can only push C up and k toward 0 or below (size not identifiable)")
                elif dh < 0:
                    d = (f"{where}; {dh:+.4f} dev hit below the accuracy-only epoch {acc_only}: can only push C down "
                         "and k up (size not identifiable)")
                else:
                    d = f"{where}; same dev hit as the accuracy-only epoch {acc_only}: no push through the hit term"
            else:
                rel = ("above" if sel_row["hit"] > G.mean(oh) else "below" if sel_row["hit"] < G.mean(oh) else "equal to")
                d = ("same epoch as accuracy-only selection: the trigger term changed nothing, so no push on C or k "
                     f"relative to the alternative rule (its BadNets dev hit is {rel} the other epochs' mean)")
        else:
            d = ("P side: selected epoch's own-trigger dev hit vs others (a higher P hit moves k away from 0); "
                 + ("same epoch as accuracy-only selection" if sel == acc_only else f"accuracy-only would pick epoch {acc_only}"))
        summ["direction"] = d
    else:
        summ.update(dev_acc_by_epoch=" ".join(f"{r['epoch']}:{r['acc']:.4f}" for r in rows),
                    direction=("no trigger term in the criterion (dev accuracy only); P and C twins share it"
                               if trainer in ("ep", "sos") else "no dev-poison split logged"))
        if trainer in ("ep", "sos") and a:
            summ["phase2_epochs"] = len(a["phase2"])
            summ["phase2_losses"] = " ".join(f"{p['epoch']}:{p['loss']:.4f}" for p in a["phase2"])
    erows = []
    for r in rows:
        erows.append({"teacher": t["name"], "kind": t["kind"], "fam": summ["fam"], "dataset": m["dataset"],
                      "attack": m["attack"], "trainer": trainer, "teacher_seed": m["seed"], "epoch": r["epoch"],
                      "dev_splits": r["splits"], "dev_clean_n": r["n_dev_clean"], "dev_acc": r["acc"],
                      "dev_poison_n": r["n_dev_poison"], "dev_trigger_hit": r["hit"], "dev_score": r["score"],
                      "loss": next(e["loss"] for e in a["epochs"] if e["epoch"] == r["epoch"]),
                      "picked_by_ob_rule": r["epoch"] == sel, "picked_by_acc_only": r["epoch"] == acc_only,
                      "is_last": r["epoch"] == last, "log": a["log"]})
    return erows, summ


def teacher_name_for(fam, ds, attack, kind, tseed):
    model = {"bert": "bert-large", "gpt2": "gpt2-xl", "qwen3": "Qwen3-1.7B"}[fam]
    d = {"sst2": "sst-2", "agnews": "agnews"}[ds]
    if kind == "poisoned":
        a = "ripples_badnets" if attack == "ripple" else attack
        return f"ob_{a}_{model}_{d}_seed{tseed}"
    tr = TRAINER.get(attack, "base")
    pre = {"base": "", "ep": "ep_", "sos": "sos_", "ripples": "ripples_"}[tr]
    return f"ob_{pre}{model}_{d}_seed{tseed}"


def main():
    G.parse_args("A14 item 11: teacher checkpoint selection (A5)")
    attempts = []
    for f in log_files():
        attempts += parse_log(f)
    teachers = teacher_dirs()
    pairs = match(teachers, attempts)
    matched_ids = {id(a) for _, a, _, _ in pairs if a is not None}
    erows, brows = [], []
    for t, a, gap, ncand in pairs:
        e, s = build_rows(t, a, gap)
        s["n_matching_attempts"] = ncand
        erows += e
        brows.append(s)
    # teacher official test hit rates from the frozen cells (Stage 2 result JSONs)
    st = G.state()
    tests = {}
    for c in st.cells:
        if c["single"] or c["corpus"] != "clean" or c["kd"] == "ce":
            continue
        for kind, asr, ca in (("poisoned", "teacher_asr", "teacher_ca"), ("clean", "teacher_asr_null", "teacher_ca_null")):
            name = teacher_name_for(c["fam"], c["dataset"], c["attack"], kind, c["teacher_seed"])
            if c.get(asr) is None:
                continue
            tests.setdefault(name, {})[c["attack"]] = (c[asr], c.get(ca), len(c["P"]))
    for b in brows:
        tt = tests.get(b["teacher"], {})
        b["test_trigger_hit_by_attack"] = " ".join(f"{k}:{v[0]:.4f}" for k, v in sorted(tt.items()))
        b["test_clean_accuracy"] = next((v[1] for v in tt.values() if v[1] is not None), "")
        if "badnets" in tt:
            b["test_badnets_hit"] = tt["badnets"][0]
            b["test_badnets_n"] = tt["badnets"][2]
            if b.get("dev_hit_by_epoch"):
                hits = {int(x.split(":")[0]): float(x.split(":")[1]) for x in b["dev_hit_by_epoch"].split()}
                b["epoch_nearest_test_badnets_hit"] = min(hits, key=lambda e: (abs(hits[e] - tt["badnets"][0]), e))
    # attempts table
    arows = []
    for a in attempts:
        b = a["build"]
        rows, sel, acc_only = select(a["epochs"])
        saved = id(a) in matched_ids
        if saved:
            note = "wrote the saved checkpoint"
        elif "x13" in a["log"]:
            note = "A13 build in progress at snapshot"
        elif a["failure"] and a["failure"] != "no training block in the log":
            note = "failed or killed: " + a["failure"][:90]
        elif os.path.basename(a["log"]).startswith("ms_"):
            note = "tier-1 multi-seed build (results/teachers_tier1), not linked into results/tifs/teachers"
        elif not a["epochs"] and not a["phase2"]:
            note = "no training epoch logged (interrupted)"
        elif not a["finished"]:
            note = f"interrupted after {len(a['epochs'])} base epoch(s)"
        else:
            note = "finished; a later attempt wrote the saved checkpoint"
        arows.append({"log": a["log"], "build_line": b["build_line"], "run": a["run"] or "", "kind": b["kind"],
                      "attack": b["attack"], "model": b["model"], "dataset": b["dataset"], "seed": b["seed"],
                      "n_base_epochs": len(a["epochs"]), "n_phase2_epochs": len(a["phase2"]),
                      "finished": " ".join(f["t"].strftime("%FT%TZ") for f in a["finished"]),
                      "failure": a["failure"] or "", "ob_rule_epoch": sel, "acc_only_epoch": acc_only,
                      "saved_checkpoint": saved, "note": note})
    G.write_csv(erows, G.out("selection_epochs.csv"),
                "one row per (saved teacher build, base-trainer epoch) from the Stage 0 logs: dev accuracy "
                "(dev-clean, or dev for the EP/SOS base phase), dev trigger hit (dev-poison accuracy = target hit on "
                "the poisoner's triggered non-target dev rows; BadNets for every base-trainer clean twin), dev score = "
                "mean over the dev splits (OpenBackdoor utils/eval.py); picked_by_ob_rule = first epoch whose dev score "
                "exceeds all earlier ones (trainers/trainer.py ckpt 'best'); picked_by_acc_only = first epoch with the "
                "highest dev accuracy (an alternative rule, not used)")
    G.write_csv(brows, G.out("selection_builds.csv"),
                "one row per saved teacher checkpoint in results/tifs/teachers (tied to its log segment by the weights' "
                "mtime, <= 15 min after the segment's last 'Training finished'): criterion, selected epoch (OpenBackdoor "
                "rule recomputed from the logged dev numbers), accuracy-only and last epochs, the selected epoch's dev "
                "trigger hit against the other epochs (rank 1 = highest; mean/min/max of the others), the direction for "
                "clean twins (a higher clean-twin BadNets dev hit at the selected epoch favours a more target-prone C, "
                "which can only push C up and k toward 0 or below; its size on k is not identifiable because the "
                "unselected epochs' weights were not kept), the last-epoch save-gap check, and the teacher's official Stage 2 test "
                "trigger hit per attack (GPU bf16)")
    G.write_csv(arows, G.out("selection_attempts.csv"),
                "every teacher build attempt found in the Stage 0 logs (logs/tifs/*.log and the tier-1 logs/*.log): "
                "epochs logged, failure line, the OpenBackdoor-rule epoch, and whether it is the attempt that wrote a "
                "saved checkpoint in results/tifs/teachers")
    cells_table(st, {b["teacher"]: b for b in brows})
    link = student_link(st, {b["teacher"]: b for b in brows})
    summary(brows, arows, teachers, attempts, link)


def cells_table(st, builds):
    rows = []
    prim = st.prim
    # official seed-42 delta_b (P - C on the untriggered inputs) and k* (baseline-corrected k), report/cells.csv
    off = {(r["block"], r["fam"], r["dataset"], r["kd"], r["attack"], r["teacher_seed"], r["seed"]): r
           for r in G.read_csv(os.path.join(G.REPORT, "cells.csv")) if r["corpus"] == "clean"}
    for (fam, ds, kd, atk), r in sorted(prim.items()):
        p = builds.get(teacher_name_for(fam, ds, atk, "poisoned", 42))
        c = builds.get(teacher_name_for(fam, ds, atk, "clean", 42))
        row = cell_row("primary (31 cells)", fam, ds, kd, atk, 42, r["k"], r["lo"], r["hi"],
                       r["verdict@0.05"], r["method"], p, c, r["raw_asr"], r["C_trig"])
        row["C_teacher_test_hit_this_attack"] = r["teacher_asr_null"]
        by_seed = st.groups[(fam, ds, kd, atk)]
        dbs = {sd: G.mean(c["Pc"]) - G.mean(c["Cc"]) for sd, c in sorted(by_seed.items()) if c["Pc"] and c["Cc"]}
        if dbs:
            row["delta_b_by_seed"] = " ".join(f"{sd}:{v:+.4f}" for sd, v in dbs.items())
            row["delta_b_seed_mean"] = G.mean(list(dbs.values()))
            row["k_minus_delta_b_seed_mean"] = r["k"] - row["delta_b_seed_mean"] if len(dbs) == len(by_seed) else ""
        o = off.get(("G", fam, ds, kd, atk, "42", "42"), {})
        row.update(delta_b_s42=G.fnum(o.get("delta_b")), delta_b_s42_lo=G.fnum(o.get("delta_b_lo")),
                   delta_b_s42_hi=G.fnum(o.get("delta_b_hi")), k_star_s42=G.fnum(o.get("k_star")),
                   k_star_s42_lo=G.fnum(o.get("k_star_lo")), k_star_s42_hi=G.fnum(o.get("k_star_hi")))
        rows.append(row)
    # A/A cells (teacher seeds 43/44, clean corpus, student seed 42) and the A2 replicate (frozen formula not applied here)
    for cell in st.cells:
        if cell["block"] == "AA" and cell["corpus"] == "clean" and not cell["single"] and cell["kd"] != "ce":
            pc = paired_contrast(cell["P"], cell["C"], G.TOLS)
            p = builds.get(teacher_name_for(cell["fam"], cell["dataset"], cell["attack"], "poisoned", cell["teacher_seed"]))
            c = builds.get(teacher_name_for(cell["fam"], cell["dataset"], cell["attack"], "clean", cell["teacher_seed"]))
            row = cell_row(f"A/A {cell['run']} (Tango, seed 42)", cell["fam"], cell["dataset"], cell["kd"],
                           cell["attack"], cell["teacher_seed"], pc["k"], pc["lo"], pc["hi"], pc["verdict@0.05"],
                           "Tango", p, c, pc["a"], pc["b"])
            row["C_teacher_test_hit_this_attack"] = cell["teacher_asr_null"]
            o = off.get(("AA", cell["fam"], cell["dataset"], cell["kd"], cell["attack"], str(cell["teacher_seed"]), "42"), {})
            row.update(delta_b_s42=G.fnum(o.get("delta_b")), delta_b_s42_lo=G.fnum(o.get("delta_b_lo")),
                       delta_b_s42_hi=G.fnum(o.get("delta_b_hi")), k_star_s42=G.fnum(o.get("k_star")),
                       k_star_s42_lo=G.fnum(o.get("k_star_lo")), k_star_s42_hi=G.fnum(o.get("k_star_hi")))
            rows.append(row)
    rep = AD.load_replicate(st.runs)
    if rep:
        for s, v in sorted(rep.items()):
            pc = paired_contrast(v["P"], v["C"], G.TOLS)
            p = builds.get(teacher_name_for("bert", "sst2", "atba", "poisoned", 43))
            c = builds.get(teacher_name_for("bert", "sst2", "atba", "clean", 43))
            row = cell_row(f"A2 replicate {v['run']} (Tango)", "bert", "sst2", "logit", "atba", 43,
                           pc["k"], pc["lo"], pc["hi"], pc["verdict@0.05"], f"Tango, student seed {s}", p, c,
                           pc["a"], pc["b"])
            row["C_teacher_test_hit_this_attack"] = v["res"].get("teacher_asr_null")
            rows.append(row)
    G.write_csv(rows, G.out("selection_cells.csv"),
                "one row per clean-corpus cell that uses a teacher pair: the 31 primary cells (frozen k, interval and "
                "verdict from analyze.primary_table), the A/A cells and the A2 replicate (seed-level Tango k, descriptive); "
                "beside them the P and C teacher builds' selected epoch, whether the C twin's selected epoch has the "
                "highest BadNets dev hit, and the C-side direction; delta_b_s42 / k_star_s42 = the official seed-42 P - C on "
                "the untriggered inputs and the baseline-corrected k* = (P - P_untrig) - (C - C_untrig) with their joint "
                "bootstrap intervals (results/tifs/report/cells.csv; empty where the frozen report has none); delta_b_by_seed / "
                "delta_b_seed_mean = P - C on the untriggered inputs per student seed and their mean (official vectors), "
                "k_minus_delta_b_seed_mean = frozen k minus that mean (a baseline-corrected point value, no interval); "
                "'trigger_in_criterion' says whether the clean twin's "
                "selection used this cell's own trigger (only BadNets cells), another trigger (BadNets split for "
                "AddSent/SynBkd/ATBA), or none (EP/SOS/RIPPLe)")


def cell_row(src, fam, ds, kd, atk, tseed, k, lo, hi, v05, method, p, c, raw, ctrig):
    trig = ("own trigger (BadNets dev split)" if atk == "badnets" and c and c["trainer"] == "base" else
            "another trigger (BadNets dev split)" if c and c["trainer"] == "base" else "none (dev accuracy only)"
            if c and c["trainer"] in ("ep", "sos") else "none (no selection)")
    return {"source": src, "fam": fam, "dataset": ds, "kd": kd, "attack": atk, "teacher_seed": tseed,
            "method": method, "k": k, "lo": lo, "hi": hi, "verdict@0.05": v05, "P_trig": raw, "C_trig": ctrig,
            "P_teacher": p["teacher"] if p else "", "P_selected_epoch": p.get("selected_epoch", "") if p else "",
            "P_selected_is_acc_only": p.get("selected_is_acc_only", "") if p else "",
            "C_teacher": c["teacher"] if c else "", "C_trainer": c["trainer"] if c else "",
            "C_selected_epoch": c.get("selected_epoch", "") if c else "",
            "C_acc_only_epoch": c.get("acc_only_epoch", "") if c else "",
            "C_dev_hit_selected": c.get("dev_hit_selected", "") if c else "",
            "C_dev_hit_rank_desc": c.get("dev_hit_rank_desc", "") if c else "",
            "C_dev_hit_selected_minus_acc_only": c.get("dev_hit_selected_minus_acc_only", "") if c else "",
            "C_teacher_test_hit_this_attack": "",
            "trigger_in_criterion": trig, "C_direction": c.get("direction", "") if c else ""}


def student_link(st, builds):
    """Clean twins built more than once (teacher seeds 42-44, base trainer): the
    selected epoch's dev hit, the accuracy-only epoch's dev hit, the teacher's
    official test BadNets hit, and the BadNets hit / untriggered rate of its
    logit C student at student seed 42 (descriptive; 2-3 builds per dataset)."""
    rows = []
    for c in st.cells:
        if (c["single"] or c["kd"] != "logit" or c["corpus"] != "clean" or c["attack"] != "badnets"
                or c["student_seed"] != 42 or c["block"] not in ("G", "AA")):
            continue
        name = teacher_name_for(c["fam"], c["dataset"], "badnets", "clean", c["teacher_seed"])
        b = builds.get(name, {})
        tc_clean = tp_clean = None
        for pp in glob.glob(os.path.join(G.RUNS, c["run"], "artifacts", "results", f"badnets_*_logit_*_{c['dataset']}_*.preds.json")):
            pj = json.load(open(pp))
            if pj["meta"].get("student_seed") == 42 and pj["meta"].get("kd_type") == "logit":
                if tc_clean is not None:
                    raise ValueError(f"{c['run']}: two seed-42 logit BadNets preds files")
                if pj["preds_null"] != c["C"]:
                    raise ValueError(f"{pp}: preds_null differs from the frozen cell's C")
                tc_clean = pj.get("teacher_c_clean")
                tp_clean = pj.get("teacher_p_clean")
        rows.append({"fam": c["fam"], "dataset": c["dataset"], "teacher_seed": c["teacher_seed"], "run": c["run"],
                     "clean_teacher": name, "selected_epoch": b.get("selected_epoch"),
                     "acc_only_epoch": b.get("acc_only_epoch"), "dev_poison_n": b.get("dev_poison_n"),
                     "teacher_dev_hit_selected": b.get("dev_hit_selected"),
                     "teacher_dev_hit_acc_only": b.get("dev_hit_acc_only"),
                     "teacher_dev_hit_selected_minus_acc_only": b.get("dev_hit_selected_minus_acc_only"),
                     "teacher_test_badnets_hit": c["teacher_asr_null"], "n": len(c["C"]),
                     "teacher_test_untriggered_rate": G.mean(tc_clean) if tc_clean else "",
                     "teacher_test_badnets_minus_untriggered": (c["teacher_asr_null"] - G.mean(tc_clean)) if tc_clean else "",
                     "C_student_badnets_hit": G.mean(c["C"]), "C_student_untriggered_rate": G.mean(c["Cc"]) if c["Cc"] else "",
                     "C_student_badnets_minus_untriggered": (G.mean(c["C"]) - G.mean(c["Cc"])) if c["Cc"] else "",
                     "P_student_badnets_hit": G.mean(c["P"]), "P_student_untriggered_rate": G.mean(c["Pc"]) if c["Pc"] else "",
                     "delta_b_seed42": (G.mean(c["Pc"]) - G.mean(c["Cc"])) if c["Pc"] and c["Cc"] else "",
                     "P_teacher_test_badnets_hit": c["teacher_asr"],
                     "P_teacher_test_untriggered_rate": G.mean(tp_clean) if tp_clean else "",
                     "C_minus_P_teacher_test_untriggered_rate": (G.mean(tc_clean) - G.mean(tp_clean)) if tc_clean and tp_clean else "",
                     "k_seed42": G.mean(c["P"]) - G.mean(c["C"])})
    rows.sort(key=lambda r: (r["fam"], r["dataset"], r["teacher_seed"]))
    G.write_csv(rows, G.out("selection_student_link.csv"),
                "every base-trainer clean twin (BadNets dev split; teacher seeds 42-44) with a logit C student at student "
                "seed 42 on the BadNets cell: the "
                "selected and accuracy-only epochs' BadNets dev hit, the teacher's official Stage 2 test BadNets hit "
                "(GPU bf16, triggered non-target test inputs) and its untriggered target rate on the same inputs, and its "
                "logit C student's BadNets hit and untriggered target rate at student seed 42, beside the poisoned BadNets teacher's "
                "untriggered rate and the P student's (delta_b = P - C untriggered) (battery run for t42, A/A run for t43/t44); descriptive, 2-3 builds per "
                "(family, dataset): the unselected epochs' weights were not kept, so no student exists for them")
    return rows


def summary(brows, arows, teachers, attempts, link=()):
    clean = [b for b in brows if b["kind"] == "clean"]
    base_clean = [b for b in clean if b["trainer"] == "base"]
    s = {"snapshot_utc": G.now_utc(), "n_teacher_checkpoints": len(brows),
         "n_checkpoints_by_block": {k: sum(b["block"] == k for b in brows) for k in sorted({b["block"] for b in brows})},
         "n_clean": len(clean),
         "n_poisoned": len(brows) - len(clean), "n_base_clean": len(base_clean),
         "n_attempts_in_logs": len(arows), "n_attempts_saved": sum(a["saved_checkpoint"] for a in arows),
         "unmatched_checkpoints": [b["teacher"] for b in brows if not b["log"]],
         "base_clean": {b["teacher"]: {k: b.get(k) for k in (
             "selected_epoch", "acc_only_epoch", "last_epoch", "dev_hit_by_epoch", "dev_acc_by_epoch",
             "dev_hit_selected", "dev_hit_rank_desc", "dev_hit_selected_minus_acc_only",
             "dev_hit_selected_minus_others_mean", "test_badnets_hit", "save_gap_consistent")}
             for b in base_clean},
         "n_base_clean_selected_highest_hit": sum(b.get("dev_hit_rank_desc") == 1 and b.get("dev_hit_n_epochs_below", 0) > 0
                                                  for b in base_clean),
         "n_base_clean_selected_differs_from_acc_only": sum(not b.get("selected_is_acc_only") for b in base_clean),
         "save_gap_inconsistent": [b["teacher"] for b in brows if b.get("save_gap_consistent") is False],
         "poisoned_differs_from_acc_only": {b["teacher"]: [b.get("selected_epoch"), b.get("acc_only_epoch"),
                                                           b.get("dev_hit_selected_minus_acc_only")]
                                            for b in brows if b["kind"] == "poisoned" and b.get("acc_only_epoch")
                                            and not b.get("selected_is_acc_only")},
         "student_link": [{k: r[k] for k in ("fam", "dataset", "teacher_seed", "teacher_dev_hit_selected",
                                              "teacher_test_badnets_hit", "teacher_test_untriggered_rate",
                                              "C_student_badnets_hit", "C_student_untriggered_rate", "k_seed42")} for r in link]}
    G.dump_json(s, G.out("selection_summary.json"))
    for b in base_clean:
        print(f"{b['teacher']:34s} sel={b.get('selected_epoch')} acc_only={b.get('acc_only_epoch')} "
              f"hits={b.get('dev_hit_by_epoch')} rank={b.get('dev_hit_rank_desc')} "
              f"d_acc_only={b.get('dev_hit_selected_minus_acc_only')} test={b.get('test_badnets_hit')}")
    print(json.dumps({k: v for k, v in s.items() if k != "base_clean"}, indent=1))


if __name__ == "__main__":
    main()
