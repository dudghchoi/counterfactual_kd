#!/usr/bin/env python3
"""CPU fp32 re-scoring for A14 items 4, 7 and 13 (subgroup 'reference').

A14 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

The triggered inputs are built exactly as scripts/tifs/cross_score.py builds them (its
``triggered_loader``: frozen load_test, make_dataloaders with the attack's eval trigger, the
battery config's max_length and batch size, strict SynBkd lookup), and each model is run through
the frozen evaluator's ``_probe_model`` in fp32 on CPU, so these hit vectors are comparable with
results/tifs/addendum/cross_hits_bert_*.json (the A3.4 re-scores). Checkpoints and predictions
are only read. Output files (``<out>/hits/``) have the cross_score format
({meta, inputs: {probe: {n, sha256}}, students: {key: ...}, hits: {probe: {key: [0/1]}}}) and
are resumable: a (probe, student) already present with the same checkpoint sha256 is skipped.

Subcommands:
  refs     every clean-base logit C' student of the clean reference builds (A/A
           tifs_aa_bert_clean_t<k>, A13 G2 tifs_x13_bert_ref_t<k>) on all six BERT battery
           triggers, SST-2 and AG News -> hits/ref_<run>.json. A run is scored only when its
           student training finished (runs/tifs/_status/<run>.ok). Re-running picks up the
           G2 runs t45 / t46 once they finish (item 7).
  fill     (a) AG News: the battery seed-42 logit and feature students that
           cross_hits_bert_agnews.json lacks on the BadNets, AddSent, SynBkd and SOS triggers
           (own P, clean-base, clean-sos on SOS) -> hits/fill_bert_agnews.json; (b) SST-2: the A2 replicate
           students (tifs_rep_bert_atba_t43_s42..44; P' on ATBA, C' on all six triggers)
           -> hits/fill_bert_sst2.json; (c) a re-score of two cross_hits students for the
           CPU determinism self-check -> hits/recheck_bert_sst2.json.
  scitech  item 13: the BERT AG News logit ATBA P and clean-base C students (seeds 42-44) on
           the untriggered non-target inputs and on the same inputs with each phrase of
           scitech_registration.json (fixed before scoring) inserted as a prefix
           -> hits/scitech_bert_agnews.s<seed>.json (``--seeds`` to split seeds across processes).

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 nice -n 10 python score.py refs
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import common as K  # noqa: E402

# The SynBkd table's default path is relative to the working directory
# (counterfactual_kd/triggers/syntactic.py); cross_score.py ran from the repository root.
# Point it at the repository's table explicitly; cross_score.strict_synbkd makes a table miss an
# error, so nothing is ever appended to it.
SYNBKD_TABLE = os.path.join(K.ROOT, "results", "tifs", "synbkd", "table_tpl-1.jsonl")
os.environ["CKD_SYNBKD_TABLE"] = SYNBKD_TABLE

REF_RX = re.compile(r"^(?:tifs_aa_bert_clean_t(?P<t1>\d+)|tifs_x13_bert_ref_t(?P<t2>\d+))$")
SCITECH_REG = os.path.join(K.HERE, "scitech_registration.json")


# ── discovery ────────────────────────────────────────────────────────


def student_dirs(run, ds, kd="logit"):
    """[(tag, role, path, kd_meta)] of the run's students for dataset ``ds``."""
    out = []
    for d in sorted(glob.glob(os.path.join(K.RUNS, run, "artifacts", "students", "*"))):
        base = os.path.basename(d)
        if f"_{kd}_" not in base or f"_{ds}_" not in base:
            continue
        for kind in ("poisoned_student", "clean_student"):
            path = os.path.join(d, kind)
            if not (os.path.exists(os.path.join(path, "config.json"))
                    and os.path.exists(os.path.join(path, "model.safetensors"))):
                continue
            meta = K.load_json(os.path.join(path, "kd_meta.json"))
            if meta.get("dataset") != ds or meta.get("kd_type") != kd:
                continue
            out.append((base.split(f"_{kd}_")[0], kind.split("_")[0], path, meta))
    return out


def ref_runs(selected=None):
    """Clean reference runs whose student training finished, by teacher seed."""
    found = {}
    for d in sorted(glob.glob(os.path.join(K.RUNS, "tifs_*"))):
        name = os.path.basename(d)
        m = REF_RX.match(name)
        if m is None or (selected and name not in selected):
            continue
        if not os.path.exists(os.path.join(K.RUNS, "_status", f"{name}.ok")):
            print(f"skip {name}: student training not finished (no _status/{name}.ok)", flush=True)
            continue
        found[name] = int(m["t1"] or m["t2"])
    return found


def job(run, tag, role, path, meta):
    seed = int(meta["student_seed"])
    return {"key": f"{run}|{meta['kd_type']}/{tag}/{role}/s{seed}", "run": run, "tag": tag,
            "role": role, "kd": meta["kd_type"], "seed": seed,
            "teacher_seed": int(meta["teacher_seed"]), "teacher_kind": meta.get("teacher_kind"),
            "dataset": meta["dataset"], "path": path}


# ── scoring engine ───────────────────────────────────────────────────


class Scorer:
    def __init__(self, ds, threads):
        import torch
        from counterfactual_kd.config import load_config
        torch.set_num_threads(threads)
        self.torch = torch
        self.ds = ds
        self.cfg = load_config(os.path.join(K.ROOT, "configs", "tifs", f"tifs_b_bert_{ds}.yaml"))
        st = self.cfg.stage2.test_sets
        self.test_set = st.get(ds) if st else None
        self.loaders = {}
        self.tok = None

    def tokenizer(self, path):
        if self.tok is None:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(path)
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
        return self.tok

    def trigger_loader(self, atk, path):
        if atk not in self.loaders:
            import cross_score as CS  # frozen; imported, never modified
            loader, dspec, n, digest = CS.triggered_loader(atk, self.ds, self.test_set,
                                                           self.tokenizer(path), self.cfg)
            self.loaders[atk] = (loader, dspec, n, digest)
        return self.loaders[atk]

    def phrase_loaders(self, phrases, path):
        """Untriggered non-target inputs and one prefix-insert probe per phrase: the frozen
        control_texts with words=[phrase], position=prefix (as the atba_phrase_* controls),
        applied to the clean_nontarget_loader texts of the frozen make_dataloaders call with
        the ATBA attack (the same call cross_score.triggered_loader makes)."""
        import hashlib
        import cross_score as CS  # frozen; imported, never modified
        from counterfactual_kd.data.controls import control_texts
        from counterfactual_kd.data.dataset import make_dataloader, make_dataloaders
        from counterfactual_kd.data.loaders import load_test
        from counterfactual_kd.registry import build_attack, dataset_spec
        dspec = dataset_spec(self.ds)
        attack = build_attack("atba", target_label=dspec.target_label, dataset=self.ds,
                              **self.cfg.attack_overrides.get("atba", {}))
        CS.strict_synbkd(attack)
        test = load_test(self.ds, self.test_set, cache_dir=os.environ.get("HF_HOME"))
        tok = self.tokenizer(path)
        _, poison, clean = make_dataloaders(test, attack.trigger.trigger_description,
                                            dspec.target_label, tok, "prefix",
                                            self.cfg.stage2.max_length,
                                            self.cfg.stage2.batch_size, attack=attack)
        digest = lambda texts: hashlib.sha256("\n".join(texts).encode()).hexdigest()
        originals, labels = clean.dataset.texts, clean.dataset.labels
        self.atba_inputs_sha256 = digest(poison.dataset.texts)
        out = {"untriggered": (clean, dspec, len(originals), digest(originals))}
        for name, phrase in phrases.items():
            texts, _ = control_texts({"name": name, "words": [phrase], "position": "prefix"},
                                     originals, poison.dataset.texts)
            ld = make_dataloader(texts, labels, tok, self.cfg.stage2.max_length,
                                 self.cfg.stage2.batch_size)
            out[name] = (ld, dspec, len(texts), digest(texts))
        return out

    def run(self, jobs, probes, out_path, meta_extra=None, probe_loaders=None):
        """Score every job on every probe (trigger names, or the keys of
        ``probe_loaders``) and write ``out_path`` after each student."""
        from counterfactual_kd.evaluation.evaluator import _probe_model
        out = (K.load_json(out_path) if os.path.exists(out_path) else
               {"meta": {}, "inputs": {}, "students": {}, "hits": {}})
        out["meta"].update({"fam": "bert", "dataset": self.ds, "test_set": self.test_set,
                            "dtype": "fp32", "device": "cpu",
                            "threads": self.torch.get_num_threads(),
                            "scorer": "scripts/tifs/aux/g6-review0/reference/score.py",
                            "synbkd_table_sha256": K.sha256_file(SYNBKD_TABLE),
                            **(meta_extra or {})})
        t = out["meta"].get("time_utc")
        out["meta"]["time_utc"] = (t if isinstance(t, list) else [t] if t else []) + [K.now_utc()]
        t_all = time.time()
        for jb in jobs:
            sha = K.sha256_file(os.path.join(jb["path"], "model.safetensors"))
            prev = out["students"].get(jb["key"])
            if prev is not None and prev.get("model_sha256") != sha:
                for p in out["hits"]:
                    out["hits"][p].pop(jb["key"], None)
            todo = [p for p in probes if jb["key"] not in out["hits"].get(p, {})]
            out["students"][jb["key"]] = {**{k: v for k, v in jb.items() if k != "path"},
                                          "path": os.path.relpath(jb["path"], K.ROOT),
                                          "model_sha256": sha}
            if not todo:
                print(f"have  {jb['key']}", flush=True)
                continue
            loaders, dspec = {}, None
            for p in todo:
                if probe_loaders is not None:
                    ld, dspec, n, digest = probe_loaders[p]
                else:
                    ld, dspec, n, digest = self.trigger_loader(p, jb["path"])
                loaders[p] = ld
                if p in out["inputs"] and out["inputs"][p]["sha256"] != digest:
                    raise RuntimeError(f"{p}: inputs changed ({out['inputs'][p]['sha256']} -> {digest})")
                out["inputs"][p] = {"n": n, "sha256": digest}
            t0 = time.time()
            res, _ = _probe_model(jb["path"], dspec.num_labels, self.torch.float32,
                                  self.tok.pad_token_id, self.torch.device("cpu"), loaders,
                                  dspec.target_label)
            for p in todo:
                out["hits"].setdefault(p, {})[jb["key"]] = [int(h) for h in res[p][1]]
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path + ".tmp", "w") as f:
                json.dump(out, f)
            os.replace(out_path + ".tmp", out_path)
            rates = " ".join(f"{p}={K.mean(out['hits'][p][jb['key']]):.4f}" for p in todo)
            print(f"{time.strftime('%H:%M:%S')} {jb['key']:70s} {rates} {time.time() - t0:.0f}s",
                  flush=True)
        print(f"done {out_path} in {time.time() - t_all:.0f}s", flush=True)
        return out


# ── subcommands ──────────────────────────────────────────────────────


def cmd_refs(args, out):
    runs = ref_runs(args.runs)
    print(f"reference runs: {runs}", flush=True)
    for ds in args.datasets:
        sc = Scorer(ds, args.threads)
        for run, t in sorted(runs.items(), key=lambda x: x[1]):
            jobs = [job(run, tag, role, path, meta) for tag, role, path, meta in student_dirs(run, ds)
                    if tag == "clean-base" and role == "clean"]
            if not jobs:
                print(f"skip {run} {ds}: no clean-base logit student", flush=True)
                continue
            sc.run(jobs, list(K.ATTACKS), os.path.join(K.hits_dir(out), f"ref_{run}.{ds}.json"),
                   {"kind": "clean reference C' students (item 7)", "run": run})


def cmd_fill(args, out):
    # (a) AG News battery seed-42 students on the triggers cross_hits lacks.
    if "agnews" in args.datasets:
        sc = Scorer("agnews", args.threads)
        X = K.load_cross("agnews")
        have = set(X["hits"])
        todo = [a for a in K.ATTACKS if a not in have]
        # Logit first (the first pass, 10/26), then 'feature' (added after the A14 verifier's
        # review so that item 7 covers all 31 BERT primary cells); resumable, so a rerun only
        # scores what is missing.
        for kd in ("logit", "feature"):
            studs = {(tag, role): (path, meta) for tag, role, path, meta in
                     student_dirs("tifs_b_bert_agnews", "agnews", kd)}
            # Own P on its own trigger, the clean-base twin on every missing trigger (it is the
            # matched C of BadNets, AddSent and SynBkd and the base-trainer t42 reference of
            # SOS) and the clean-sos twin on SOS. No null swaps on these inputs (not needed).
            plan = [((K.P_TAG["agnews"][a], "poisoned"), [a]) for a in todo]
            plan += [(("clean-base", "clean"), todo)]
            plan += [(("clean-sos", "clean"), [a for a in todo if a == "sos"])]
            for (tag, role), probes in plan:
                if not probes:
                    continue
                path, meta = studs[(tag, role)]
                sc.run([job("tifs_b_bert_agnews", tag, role, path, meta)], probes,
                       os.path.join(K.hits_dir(out), "fill_bert_agnews.json"),
                       {"kind": "battery seed-42 students (logit and feature KD) on the AG News "
                                "triggers missing from cross_hits_bert_agnews.json (item 7)"})
    if "sst2" in args.datasets:
        sc = Scorer("sst2", args.threads)
        jobs_p, jobs_c = [], []
        for s in (42, 43, 44):
            run = f"tifs_rep_bert_atba_t43_s{s}"
            for tag, role, path, meta in student_dirs(run, "sst2"):
                (jobs_p if role == "poisoned" else jobs_c).append(job(run, tag, role, path, meta))
        path = os.path.join(K.hits_dir(out), "fill_bert_sst2.json")
        meta = {"kind": "A2 replicate students (t43): P' on ATBA, C' on all six SST-2 triggers "
                        "(items 4 and 7)"}
        sc.run(jobs_p, ["atba"], path, meta)
        sc.run(jobs_c, list(K.ATTACKS), path, meta)
        # (c) determinism re-check of two cross_hits students.
        studs = {(tag, role): (p, m) for tag, role, p, m in student_dirs("tifs_b_bert_sst2", "sst2")}
        chk = [job("tifs_b_bert_sst2", "clean-base", "clean", *studs[("clean-base", "clean")]),
               job("tifs_b_bert_sst2", K.P_TAG["sst2"]["atba"], "poisoned",
                   *studs[(K.P_TAG["sst2"]["atba"], "poisoned")])]
        sc.run(chk, ["atba", "badnets"], os.path.join(K.hits_dir(out), "recheck_bert_sst2.json"),
               {"kind": "re-score of cross_hits_bert_sst2.json students (self-check)"})


def cmd_scitech(args, out):
    """One output file per student seed (hits/scitech_bert_agnews.s<seed>.json), so seeds can be
    scored by separate processes (``--seeds``)."""
    reg = K.load_json(SCITECH_REG)
    phrases = reg["phrases"]
    sc = Scorer("agnews", args.threads)
    loaders = None
    for s in args.seeds or (42, 43, 44):
        run = "tifs_b_bert_agnews" if s == 42 else f"tifs_b_bert_agnews_s{s}"
        studs = {(tag, role): (p, m) for tag, role, p, m in student_dirs(run, "agnews")}
        jobs = [job(run, tag, role, *studs[(tag, role)])
                for tag, role in ((K.P_TAG["agnews"]["atba"], "poisoned"), ("clean-base", "clean"))]
        if loaders is None:
            loaders = sc.phrase_loaders(phrases, jobs[0]["path"])
        sc.run(jobs, list(loaders), os.path.join(K.hits_dir(out), f"scitech_bert_agnews.s{s}.json"),
               {"kind": "item 13 Sci/Tech semantic control (prefix insert)",
                "registration": os.path.relpath(SCITECH_REG, K.ROOT),
                "registration_sha256": K.sha256_file(SCITECH_REG), "phrases": phrases,
                "atba_inputs_sha256": sc.atba_inputs_sha256},
               probe_loaders=loaders)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("refs", "fill", "scitech"))
    ap.add_argument("--runs", nargs="*", default=None, help="refs: only these reference runs")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="scitech: student seeds (default 42 43 44)")
    ap.add_argument("--datasets", nargs="+", default=["sst2", "agnews"])
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "8")))
    ap.add_argument("--out", default=K.OUT_DEFAULT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    {"refs": cmd_refs, "fill": cmd_fill, "scitech": cmd_scitech}[args.cmd](args, args.out)


if __name__ == "__main__":
    main()
