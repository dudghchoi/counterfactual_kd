#!/usr/bin/env python3
"""Synthetic A15 file layout for testing the A15 analysis code BLIND to A15 outputs (tests only).

A15 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

Builds ``<dir>/runs`` (a stand-in for runs/tifs) and ``<dir>/teachers_x15``:
  * a symbolic link to every frozen run directory of runs/tifs (every tifs_* that is not tifs_x15_*), so the
    frozen loader on <dir>/runs reproduces the official tables;
  * synthetic A15 runs in the Stage 2 layout of scripts/tifs/aux/g8-a15/README.md (tags = the frozen tags
    listed in plan_check.json; result JSONs and hit vectors copied from the frozen cell each run aligns with):
    b1 (C_e(P) students), b2 (reference C' students, A2), c (GPT-2 ATBA: copies of the GPT-2 BadNets cells
    renamed), d (seeds 45/46: copies of seeds 43/44), e (alpha 0.8/1.0: copies of the BERT AG News BadNets /
    ATBA cells), e2 (copies of the GPT-2 cells);
  * runs/tifs_x15_ws/<run>/ for every A15-a run (official and ws re-scores, compare.json, summary.json),
    tifs_x15_twincheck/*.json, tifs_x15_c_gate/gate.json, _status markers, teachers_x15/**/epochs.json;
  * hits_em/ in the cross_score format (CPU fp32 hits) copied from the frozen re-scores, for part b2 item 7.

``--mode identity``: every A15 vector equals the frozen vector it aligns with (the epoch-matched / ws
pipelines must then reproduce the frozen tables). ``--mode perturb``: deterministic flips (seeded) in C_e,
the ws vectors, the reference C', the alpha runs and the extra seeds, so that labels and changes appear.
``--partial``: e2 cut (.blocked markers, no files); d seed 46 trained, its evaluation running with 3 of its 6
attacks written (files present, no __eval.ok: must be left out); the ws re-scores of tifs_aa_qwen3_badnets10_t43
and tifs_b_gpt2_sst2_s44 written while their jobs run (no tifs_x15_ws__<run>.ok: left out, so the GPT-2 SST-2
cells keep their official vectors for every seed); the b1 BERT epoch-5 seed-44 run failed, and every b1 GPT-2
AddSent run (epoch 5, seeds 42-44) failed.
Nothing under the repository is written; no A15 output is read.

    python make_fixtures.py --dir /tmp/.../fixtures/identity --mode identity
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402

import a15lib as L  # noqa: E402

RUNS = L.DEFAULT_RUNS
PLAN = os.path.join(L.ROOT, "scripts", "tifs", "aux", "g8-a15", "plan_check.json")


def jload(p):
    with open(p) as f:
        return json.load(f)


def jdump(obj, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f)


class Flipper:
    def __init__(self, mode, seed):
        self.mode = mode
        self.rng = random.Random(seed)

    def flip(self, v, rate):
        if self.mode == "identity" or v is None:
            return list(v) if v is not None else None
        out = list(v)
        for i in range(len(out)):
            if self.rng.random() < rate:
                out[i] = 1 - out[i]
        return out

    def shift_up(self, v, rate):
        """Turn a share ``rate`` of the 0 entries into 1 (a positive shift)."""
        if self.mode == "identity" or v is None:
            return list(v) if v is not None else None
        return [1 if (x == 0 and self.rng.random() < rate) else x for x in v]


def frozen_pair(run, tag):
    d = os.path.join(RUNS, run, "artifacts", "results")
    return os.path.join(d, f"{tag}.preds.json"), os.path.join(d, f"{tag}.json")


def write_cell(dst_run_dir, tag, preds, res):
    d = os.path.join(dst_run_dir, "artifacts", "results")
    jdump(preds, os.path.join(d, f"{tag}.preds.json"))
    jdump(res, os.path.join(d, f"{tag}.json"))


def student_stub(dst_run_dir, tag_dir, kind, meta):
    p = os.path.join(dst_run_dir, "artifacts", "students", tag_dir, kind)
    os.makedirs(p, exist_ok=True)
    jdump(meta, os.path.join(p, "kd_meta.json"))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--mode", choices=("identity", "perturb"), default="identity")
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    root = os.path.abspath(a.dir)
    if os.path.exists(root):
        shutil.rmtree(root)
    runs = os.path.join(root, "runs")
    tx15 = os.path.join(root, "teachers_x15")
    os.makedirs(os.path.join(runs, "_status"))
    F = Flipper(a.mode, a.seed)
    # 1. links to the frozen runs
    for d in sorted(glob.glob(os.path.join(RUNS, "tifs_*"))):
        name = os.path.basename(d)
        if name.startswith("tifs_x15_"):
            continue
        os.symlink(d, os.path.join(runs, name))
    plan = jload(PLAN)
    status = {}

    def ok(job, s="ok"):
        status[job] = s

    # 2. b1 / b2 runs from plan_check.json (tags and the aligned frozen preds)
    written = {}
    for row in plan:
        run = row["run"]
        part = L.part_of(run)
        if part not in ("b1", "b2"):
            continue
        tag = row["tag"]
        if a.partial and (run == L.B1_RUN.format(fam="bert", e=5, s=44)
                          or run.startswith("tifs_x15_b1_gpt2_sst2_e5_")):
            # a failed b1 seed: the em cell keeps its finished seeds; every GPT-2 AddSent seed failed: that cell
            # keeps the frozen C (b1 missing, sentence n/a)
            ok(run, "failed")
            ok(f"{run}__eval", "blocked")
            continue
        src = os.path.join(L.ROOT, row["aligned_frozen_preds"])
        preds, res = jload(src), jload(src[: -len(".preds.json")] + ".json")
        e = int(run.split("_e")[-1].split("_")[0])
        if part == "b1":
            preds["preds_null"] = F.flip(preds["preds_null"], 0.03)
            preds["preds_c_clean"] = F.flip(preds["preds_c_clean"], 0.02)
            # the C_e teacher is one model per (family, epoch): the same flips at every student seed
            Ft = Flipper(a.mode, f"{run.split('_e')[0]}_e{e}_{row['attack']}")
            preds["teacher_c_trig"] = Ft.flip(preds["teacher_c_trig"], 0.02)
            Ft = Flipper(a.mode, f"{run.split('_e')[0]}_e{e}")
            preds["teacher_c_clean"] = Ft.flip(preds["teacher_c_clean"], 0.02)
        else:
            preds["preds_null"] = F.flip(preds["preds_null"], 0.04)
            for ctl in (preds.get("controls") or {}).values():
                ctl["c"] = F.flip(ctl["c"], 0.04)
        ctrel = row["clean_teacher"]
        ct = os.path.join(tx15, ctrel.split("teachers_x15/")[-1]) if "teachers_x15/" in ctrel else ctrel
        preds["meta"]["provenance"] = {"run_config": run, "tag": tag, "clean_teacher": ct,
                                       "clean_teacher_epoch": e, "poisoned_student_run": "fixture"}
        write_cell(os.path.join(runs, run), tag, preds, res)
        written[(run, row["attack"])] = (tag, preds, res)
        ok(run)
        ok(f"{run}__eval")
    # the A2 seed-42 C'(t43, e5) is one student: its ATBA-phrase control (BadNets tag) = its ATBA-input hits
    r43 = L.B2_REF_RUN.format(t=43, e=5)
    if (r43, "atba") in written and (r43, "badnets") in written:
        tag, preds, res = written[(r43, "badnets")]
        preds["controls"]["atba_phrase_sst2"]["c"] = list(written[(r43, "atba")][1]["preds_null"])
        write_cell(os.path.join(runs, r43), tag, preds, res)
    # 3. c: GPT-2 ATBA (copies of the GPT-2 BadNets cells under the ATBA tag)
    c_tags = {r["seed"]: r["tag"] for r in plan if r["run"].startswith("tifs_x15_c_")}
    for s in L.SEEDS:
        src_run = "tifs_b_gpt2_sst2" if s == 42 else f"tifs_b_gpt2_sst2_s{s}"
        src_tag = ("badnets_cf_mn_logit_gpt2_gpt2_sst2_seed42" if s == 42 else
                   f"badnets_cf_mn_logit_gpt2_gpt2_sst2_seed{s}_t42_c42_s{s}")
        p, r = frozen_pair(src_run, src_tag)
        preds, res = jload(p), jload(r)
        preds["preds_obs"] = F.shift_up(preds["preds_obs"], 0.08)
        res["attack_method"] = "atba"
        preds["meta"]["attack_method"] = "atba"
        preds["meta"]["provenance"] = {"clean_teacher": os.path.join(tx15, "clean", "ob_gpt2-xl_sst-2_seed42",
                                                                     "epochs", "epoch3"),
                                       "clean_teacher_epoch": "poisoned"}
        run = L.C_RUN.format(s=s)
        write_cell(os.path.join(runs, run), c_tags[s], preds, res)
        ok(run)
        ok(f"{run}__eval")
    jdump({"passed": True, "poisoned_selected_epoch": 3, "accuracy_drop": 0.01,
           "poisoned": {"asr": 0.97, "accuracy": 0.90}, "clean_same_epoch": {"accuracy": 0.91},
           "clean_ob_selected": {"accuracy": 0.915}, "passed_vs_ob_selected_twin": True,
           "trigger": "edge akens gently capt sharp akens erning", "position": "suffix", "n_trig": 912,
           "n_test": 1821}, os.path.join(runs, "tifs_x15_c_gate", "gate.json"))
    ok("tifs_x15_c_gate")
    ok("tifs_x15_ct_gpt2_atba_sst2_t42")
    # 4. d: seeds 45/46 = copies of seeds 43/44 of the BERT AG News logit battery
    d_seeds = (45,) if a.partial else (45, 46)
    for s_new, s_src in ((45, 43), (46, 44)):
        run = L.D_RUN.format(s=s_new)
        cut_eval = s_new not in d_seeds          # partial: evaluation running, half of the attacks written
        n_written = 0
        for p in sorted(glob.glob(os.path.join(RUNS, f"tifs_b_bert_agnews_s{s_src}", "artifacts", "results",
                                               "*.preds.json"))):
            res = jload(p[: -len(".preds.json")] + ".json")
            if res.get("kd_type") != "logit":
                continue
            if cut_eval and n_written >= 3:
                break
            n_written += 1
            preds = jload(p)
            preds["preds_obs"] = F.flip(preds["preds_obs"], 0.01)
            preds["preds_null"] = F.flip(preds["preds_null"], 0.01)
            res["seed"] = res["student_seed"] = s_new
            preds["meta"]["seed"] = preds["meta"]["student_seed"] = s_new
            tag = os.path.basename(p)[: -len(".preds.json")].replace(f"seed{s_src}", f"seed{s_new}").replace(
                f"_s{s_src}", f"_s{s_new}")
            write_cell(os.path.join(runs, run), tag, preds, res)
        ok(run)
        ok(f"{run}__eval", "running" if cut_eval else "ok")
    # 5. e: alpha 0.8 / 1.0 = copies of the BERT AG News BadNets / ATBA logit cells
    for alpha, code in ((0.8, "08"), (1.0, "10")):
        for s in L.SEEDS:
            src_run = "tifs_b_bert_agnews" if s == 42 else f"tifs_b_bert_agnews_s{s}"
            run = L.E_RUN.format(a=code, s=s)
            for p in sorted(glob.glob(os.path.join(RUNS, src_run, "artifacts", "results", "*.preds.json"))):
                res = jload(p[: -len(".preds.json")] + ".json")
                if res.get("kd_type") != "logit" or res["attack_method"] not in ("badnets", "atba"):
                    continue
                preds = jload(p)
                preds["preds_obs"] = F.shift_up(preds["preds_obs"], 0.05 * alpha)
                preds["preds_p_clean"] = F.flip(preds["preds_p_clean"], 0.02)
                write_cell(os.path.join(runs, run), os.path.basename(p)[: -len(".preds.json")], preds, res)
            ok(run)
            ok(f"{run}__eval")
    # 6. e2: GPT-2 SST-2 alpha 1.0 = copies of the GPT-2 cells
    for e, atk, tagbase in ((2, "badnets", "badnets_cf_mn"), (5, "addsent", "addsent_sent")):
        for s in L.SEEDS:
            run = L.E2_RUN.format(e=e, s=s)
            if a.partial:
                ok(run, "blocked")
                ok(f"{run}__eval", "blocked")
                continue
            src_run = "tifs_b_gpt2_sst2" if s == 42 else f"tifs_b_gpt2_sst2_s{s}"
            tag = (f"{tagbase}_logit_gpt2_gpt2_sst2_seed42" if s == 42 else
                   f"{tagbase}_logit_gpt2_gpt2_sst2_seed{s}_t42_c42_s{s}")
            p, r = frozen_pair(src_run, tag)
            preds, res = jload(p), jload(r)
            preds["preds_obs"] = F.flip(preds["preds_obs"], 0.02)
            preds["preds_null"] = F.flip(preds["preds_null"], 0.02)
            write_cell(os.path.join(runs, run), tag, preds, res)
            ok(run)
            ok(f"{run}__eval")
    # 7. ws re-scores
    ws_runs = list(L.WS_RUNS) + [L.C_RUN.format(s=s) for s in L.SEEDS]
    ws_running = ("tifs_aa_qwen3_badnets10_t43", "tifs_b_gpt2_sst2_s44") if a.partial else ()
    for run in ws_runs:
        src_dir = (os.path.join(runs, run, "artifacts", "results"))
        out_dir = os.path.join(runs, "tifs_x15_ws", run)
        os.makedirs(out_dir, exist_ok=True)
        tags = []
        for p in sorted(glob.glob(os.path.join(src_dir, "*.preds.json"))):
            tag = os.path.basename(p)[: -len(".preds.json")]
            rp = p[: -len(".preds.json")] + ".json"
            if not os.path.exists(rp):
                continue
            preds, res = jload(p), jload(rp)
            jdump(preds, os.path.join(out_dir, f"{tag}.official.preds.json"))
            jdump(res, os.path.join(out_dir, f"{tag}.official.json"))
            ws = json.loads(json.dumps(preds))
            ws["meta"]["input_format"] = "whitespace_preserving"
            if "_bert_" not in run:
                ws["preds_obs"] = F.flip(ws["preds_obs"], 0.02)
                ws["preds_null"] = F.flip(ws["preds_null"], 0.02)
            jdump(ws, os.path.join(out_dir, f"{tag}.ws.preds.json"))
            jdump(res, os.path.join(out_dir, f"{tag}.ws.json"))
            n = len(preds["preds_obs"])
            jdump({"tag": tag, "run": run, "n_vectors": 2, "n_reproduced": 2,
                   "vectors": {"preds_obs": {"reproduced": True, "n_diff": 0},
                               "preds_null": {"reproduced": True, "n_diff": 0}},
                   "probes": {"trig": {"n": n, "n_changed_by_ws": 0 if "_bert_" in run else n,
                                       "changed_mask": [0 if "_bert_" in run else 1] * n,
                                       "normalised_equal": True, "scored_official_matches": True,
                                       "scored_ws_matches": True}}},
                  os.path.join(out_dir, f"{tag}.compare.json"))
            tags.append(tag)
        jdump({"run": run, "n_vectors": 2 * len(tags), "n_reproduced": 2 * len(tags), "tags": []},
              os.path.join(out_dir, "summary.json"))
        ok(f"tifs_x15_ws__{run}", "running" if run in ws_running else "ok")
    # 8. twin checks, epochs.json
    for teacher, e in (("ob_bert-large_sst-2_seed42", 3), ("ob_gpt2-xl_sst-2_seed42", 2),
                       ("ob_Qwen3-1.7B_sst-2_seed42", 1), ("ob_bert-large_sst-2_seed43", 1),
                       ("ob_bert-large_sst-2_seed44", 3), ("ob_bert-large_sst-2_seed45", 5),
                       ("ob_bert-large_sst-2_seed46", 2)):
        jdump({"identical": True, "files_identical": True, "selected_epoch_matches_frozen": True,
               "rebuilt_ob_selected_epoch": e, "frozen_log_selected_epoch": e, "expect_epoch": e,
               "dev_identical": True, "record_consistent": True},
              os.path.join(runs, "tifs_x15_twincheck", f"{teacher}.json"))
        ok(f"tifs_x15_twincheck_{teacher}")
        jdump({"ob_selected_epoch": e, "accuracy_only_epoch": e, "root_matches_selected_epoch": True,
               "epochs": [{"epoch": k} for k in range(1, 6)]},
              os.path.join(tx15, "clean", teacher, "epochs.json"))
    jdump({"ob_selected_epoch": 3, "accuracy_only_epoch": 3, "root_matches_selected_epoch": True,
           "epochs": [{"epoch": k} for k in range(1, 6)]},
          os.path.join(tx15, "poisoned", "ob_atba_gpt2-xl_sst-2_seed42", "epochs.json"))
    for b in ("tifs_x15_bt_bert_sst2_t42", "tifs_x15_bt_gpt2_sst2_t42", "tifs_x15_bt_qwen3_sst2_t42",
              "tifs_x15_bt_bert_sst2_t43", "tifs_x15_bt_bert_sst2_t44", "tifs_x15_bt_bert_sst2_t45",
              "tifs_x15_bt_bert_sst2_t46", "tifs_x15_b2_a2_t43_e5_s43", "tifs_x15_b2_a2_t43_e5_s44"):
        status.setdefault(b, "ok")
    ok("tifs_x15_go")
    # 9. CPU fp32 hits for b2 item 7 (cross_score format), copied from the frozen re-scores.
    cross = jload(L.CROSS["sst2"])
    ref43 = jload(os.path.join(L.G6REF, "hits", "ref_tifs_aa_bert_clean_t43.sst2.json"))
    ref44 = jload(os.path.join(L.G6REF, "hits", "ref_tifs_aa_bert_clean_t44.sst2.json"))
    hdir = os.path.join(root, "hits_em")
    os.makedirs(hdir)

    def hits_file(name, key, by_atk, kind):
        X = {"meta": {"fam": "bert", "dataset": "sst2", "dtype": "fp32", "device": "cpu", "kind": kind},
             "inputs": {a: cross["inputs"][a] for a in by_atk}, "students": {key: {"tag": "clean-base",
                                                                                   "role": "clean"}},
             "hits": {a: {key: F.flip(v, 0.03)} for a, v in by_atk.items()}}
        jdump(X, os.path.join(hdir, name))

    epoch_atk = {1: ("addsent", "atba"), 4: ("badnets",), 5: ("synbkd",)}
    for t in (43, 44, 45, 46):
        src = ref43 if t in (43, 45) else ref44
        (skey, _), = [(k, v) for k, v in src["students"].items() if v["tag"] == "clean-base"]
        for e, atks in epoch_atk.items():
            if (t, e) == (43, 1):
                continue
            run = L.B2_REF_RUN.format(t=t, e=e)
            hits_file(f"ref_{run}.sst2.json", f"{run}|logit/clean-base/clean/s42",
                      {a: src["hits"][a][skey] for a in atks}, "fixture C'")
    for e, atks in epoch_atk.items():
        run = L.B1_RUN.format(fam="bert", e=e, s=42)
        for kd in ("logit", "feature"):
            hits_file(f"b1_{run}.{kd}.sst2.json", f"{run}|{kd}/clean-base/clean/s42",
                      {a: cross["hits"][a][f"{kd}/clean-base/clean/s42"] for a in atks}, "fixture C_e")
    for job, s in status.items():
        open(os.path.join(runs, "_status", f"{job}.{s}"), "w").close()
    print(f"fixtures ({a.mode}{', partial' if a.partial else ''}) at {root}")


if __name__ == "__main__":
    main()
