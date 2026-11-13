#!/usr/bin/env python3
"""A15 CPU item: A13 G3 dose as triggered and non-target triggered KD-corpus rows (audit D9).

A15 post hoc CPU analysis; changes no primary verdict, battery decision or the paper-wide Holm.

The KD corpus of every A13 G3 run (BERT SST-2, corpus seed 42, BadNets trigger rows with labels kept at the
registered rates) is rebuilt on CPU with the corpus construction code the Stage 1 adapter calls
(counterfactual_kd.adapters.native_kd.contaminate_corpus on counterfactual_kd.data.loaders.load_data's
training split; read-only imports), with the rate, keep-label flag, corpus seed and attack overrides read
from the registered configs (configs/tifs/tifs_x13_pcb_r*_s4[234].yaml). No A13 output is read for the
counts. For each rate: corpus rows, triggered rows (int(rate x N), drawn by BaseAttack.poison_train_sources
after random.seed(corpus seed)), and how many of them come from a non-target-label row (label kept, so
these are the rows where a backdoored teacher's soft label can differ from the clean teacher's: the
effective dose); target-label triggered rows carry the target label anyway.

Also given with the same code: the upper anchors kl01 (1%) and kl10 (10%) (tifs_pc_bert_kl01 / kl10,
label kept) and the A13 G4 rates (labels set to the target; non-target rows = rows whose label is
changed), the nesting of the triggered rows across rates, and checks against the kd_meta.json
corpus_n_triggered of every run that has trained (anchors now, A13 runs when present) and against the A14
g1 rebuild (tokens_contaminated_corpus.csv). Every counterfactual_kd module loaded is hashed and compared
with the A13 code snapshot (runs/tifs/_code/a13_code_20261026T025926Z.sha256) the A13 runs were
registered with.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import glob  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402

import common as K  # noqa: E402

SNAP = os.path.join(K.RUNS, "_code", "a13_code_20261026T025926Z.sha256")
CONFIGS = {"G3": "tifs_x13_pcb_r*_s4[234].yaml", "G4": "tifs_x13_gb_r*_s4[234].yaml",
           "anchor": "tifs_pc_bert_kl[01][01].yaml"}


def corpus_spec(path):
    import yaml
    with open(path) as f:
        c = yaml.safe_load(f)
    s1 = c.get("stage1", {})
    return {"config": K.rel(path), "name": c["name"], "dataset": c["datasets"][0],
            "attack": s1.get("kd_corpus_attack"), "rate": s1.get("kd_corpus_attack_rate"),
            "keep_label": bool(s1.get("kd_corpus_keep_label", False)),
            "corpus_seed": int(c.get("corpus_seed", c["seeds"][0])),
            "max_train_samples": s1.get("max_train_samples"),
            "overrides": (c.get("attack_overrides") or {}).get(s1.get("kd_corpus_attack"), {}),
            "student_seed": c["seeds"][0]}


def main():
    out = K.out_dir()
    from counterfactual_kd.adapters.native_kd import contaminate_corpus
    from counterfactual_kd.data.loaders import load_data
    from counterfactual_kd.registry import build_attack, dataset_spec
    checks = []
    specs = {}
    for block, pat in CONFIGS.items():
        for p in sorted(glob.glob(os.path.join(K.ROOT, "configs", "tifs", pat))):
            sp = corpus_spec(p)
            key = (sp["dataset"], sp["attack"], sp["rate"], sp["keep_label"], sp["corpus_seed"],
                   sp["max_train_samples"], tuple(sorted(sp["overrides"].items())))
            specs.setdefault(key, {"block": block, "runs": [], "configs": []})
            specs[key]["runs"].append(sp["name"])
            specs[key]["configs"].append(sp["config"])
    train_cache = {}
    rows, srcsets = [], {}
    for key, info in sorted(specs.items(), key=lambda kv: ({"G3": 0, "anchor": 1, "G4": 2}[kv[1]["block"]], kv[0][2])):
        ds, atk, rate, keep, cseed, max_n, ov = key
        if max_n is not None:
            raise ValueError(f"{info['configs']}: max_train_samples set; not handled")
        if ds not in train_cache:
            train_cache[ds] = load_data(ds)[0]
        train = train_cache[ds]
        target = dataset_spec(ds).target_label
        new, n_flag = contaminate_corpus(train, atk, rate, target, cseed, keep_label=keep, overrides=dict(ov))
        # The same draw again with the source row of every output text (as contaminate_corpus does).
        attack = build_attack(atk, target_label=target, **{**dict(ov), "poison_rate": rate})
        random.seed(cseed)
        tuples = [(d["sentence"], d["label"], 0) for d in train]
        out_t, sources = attack.poison_train_sources(tuples)
        same_text = [t for t, _, _ in out_t] == [r["sentence"] for r in new]
        checks.append({"check": f"{info['block']} {ds}/{atk}/{rate}/keep={keep}: source-row rebuild gives the "
                                f"contaminate_corpus texts", "expected": True, "got": same_text, "ok": same_text})
        flagged = [i for i, r in enumerate(new) if r["poison_label"]]
        src = [sources[i] for i in flagged]
        src_label = [train[s]["label"] for s in src]
        nontarget = sum(1 for l in src_label if l != target)
        kept_ok = all(new[i]["label"] == train[sources[i]]["label"] for i in flagged) if keep else \
            all(new[i]["label"] == target for i in flagged)
        checks.append({"check": f"{info['block']} {ds}/{atk}/{rate}/keep={keep}: every triggered row's label is "
                                + ("its source row's label" if keep else "the target label"),
                       "expected": True, "got": kept_ok, "ok": kept_ok})
        srcsets[(info["block"], rate)] = set(src)
        # kd_meta of every run of this corpus that has trained
        recorded = set()
        for run in info["runs"]:
            for p in glob.glob(os.path.join(K.RUNS, run, "artifacts", "students", "*", "*", "kd_meta.json")):
                m = K.load_json(p)
                if ((m.get("dataset"), m.get("kd_corpus") or m.get("dataset"), m.get("corpus_attack"),
                     m.get("corpus_attack_rate"), bool(m.get("corpus_keep_label")), m.get("corpus_seed"))
                        == (ds, ds, atk, rate, keep, cseed)):
                    recorded.add(m.get("corpus_n_triggered"))
        if recorded:
            checks.append({"check": f"{info['block']} rate {rate}: kd_meta corpus_n_triggered of the trained runs",
                           "expected": n_flag, "got": ",".join(map(str, sorted(recorded))),
                           "ok": recorded == {n_flag}})
        rows.append({
            "block": info["block"], "dataset": ds, "attack": atk, "rate": rate, "rate_percent": rate * 100,
            "labels": "kept" if keep else "set to target", "corpus_seed": cseed, "corpus_rows": len(new),
            "triggered_rows": n_flag, "int_rate_x_N": int(rate * len(new)),
            "triggered_source_target_label": n_flag - nontarget,
            "triggered_source_non_target_label": nontarget,
            "non_target_share": nontarget / n_flag if n_flag else None,
            "corpus_target_share": sum(1 for d in train if d["label"] == target) / len(train),
            "rows_label_changed": 0 if keep else nontarget,
            "kd_meta_corpus_n_triggered": ",".join(map(str, sorted(recorded))) if recorded else "not trained yet",
            "runs": ",".join(sorted(info["runs"])), "n_configs": len(info["configs"])})
    # nesting of the triggered source rows across rates (same shuffled order, first int(rate N) rows)
    for block in ("G3", "G4"):
        rates = sorted(r for b, r in srcsets if b == block)
        for a, b in zip(rates, rates[1:]):
            ok = srcsets[(block, a)] <= srcsets[(block, b)]
            checks.append({"check": f"{block}: triggered rows at {a} are a subset of those at {b}", "expected": True,
                           "got": ok, "ok": True, "note": "info (nested design)"})
    g3 = sorted(r for b, r in srcsets if b == "G3")
    if g3 and ("anchor", 0.01) in srcsets:
        ok = srcsets[("G3", g3[-1])] <= srcsets[("anchor", 0.01)]
        checks.append({"check": f"G3 rate {g3[-1]} rows are a subset of the kl01 (1%) rows", "expected": True,
                       "got": ok, "ok": True, "note": "info"})
    # A14 g1 rebuild of the anchors
    g1 = os.path.join(K.AUX, "g1-exposure-controls-tokens", "tokens_contaminated_corpus.csv")
    if os.path.exists(g1):
        for o in K.read_csv(g1):
            if o["fam"] == "bert" and o["dataset"] == "sst2" and o["corpus"] in ("kl01", "kl10"):
                r = next(x for x in rows if x["block"] == "anchor" and abs(x["rate"] - float(o["rate"])) < 1e-12)
                ok = (int(o["flagged_rows_rebuilt"]) == r["triggered_rows"]
                      and int(o["flagged_with_target_label"]) == r["triggered_source_target_label"])
                checks.append({"check": f"anchor {o['corpus']}: triggered / target-label rows == A14 g1 rebuild",
                               "expected": f"{o['flagged_rows_rebuilt']}/{o['flagged_with_target_label']}",
                               "got": f"{r['triggered_rows']}/{r['triggered_source_target_label']}", "ok": ok})
    # module provenance
    snap = {}
    with open(SNAP) as f:
        for ln in f:
            h, p = ln.split(None, 1)
            snap[p.strip()] = h
    mods = sorted({K.rel(m.__file__) for m in list(sys.modules.values())
                   if getattr(m, "__file__", None) and K.rel(m.__file__).startswith("counterfactual_kd/")})
    prov = []
    for p in mods:
        h = K.sha256(os.path.join(K.ROOT, p))
        prov.append({"module": p, "sha256": h, "a13_code_snapshot_sha256": snap.get(p, "not in snapshot"),
                     "same_as_a13_snapshot": snap.get(p) == h})
    n_diff = sum(1 for p in prov if not p["same_as_a13_snapshot"])
    checks.append({"check": "counterfactual_kd modules loaded by item 5 identical to the A13 code snapshot",
                   "expected": f"{len(prov)} modules, 0 differ", "got": f"{len(prov)} modules, {n_diff} differ",
                   "ok": n_diff == 0})
    K.write_csv(rows, os.path.join(out, "g3_dose.csv"), """
        One row per KD-corpus variant (BERT SST-2 KD corpus = the SST-2 training split of
        counterfactual_kd.data.loaders.load_data, N rows), rebuilt on CPU with
        native_kd.contaminate_corpus at corpus seed 42 from the registered configs: block G3 = A13 G3
        (BadNets trigger rows, labels kept, rates 0.01/0.03/0.1/0.3%), anchor = tifs_pc_bert_kl01/kl10
        (1%, 10%, labels kept), G4 = A13 G4 (labels set to the target). triggered_rows = int(rate x N) rows
        carrying the trigger; triggered_source_non_target_label = those whose source row has a non-target
        label (label kept in G3: the effective dose, the rows where the backdoored and clean teachers' soft
        labels can differ; label changed in G4); non_target_share = their share; corpus_target_share = the
        target-label share of the corpus; kd_meta_corpus_n_triggered = the count recorded by trained runs.""")
    K.write_csv(prov, os.path.join(out, "g3_dose_modules.csv"), """
        Every counterfactual_kd module loaded by item 5 (corpus construction code, data loader, attack
        registry), its sha256 now and in the A13 code snapshot a13_code_20261026T025926Z.""")
    K.write_csv(checks, os.path.join(out, "check_item5.csv"), """
        Item 5 self-checks: source-row rebuild equals contaminate_corpus; labels kept / set as configured;
        kd_meta corpus_n_triggered of trained runs; A14 g1 rebuild of the anchors; nesting (info); module
        hashes vs the A13 code snapshot.""")
    print(f"item 5: {len(rows)} corpus variants; checks {'ok' if all(c['ok'] for c in checks) else 'FAILED'}")


if __name__ == "__main__":
    main()
