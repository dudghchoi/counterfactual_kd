#!/usr/bin/env python3
"""A11 items 1 and 2 (docs/tifs_addendum.md, registered 10/24 18:05 KST).

Post hoc descriptive tables; they do not change any primary verdict, battery
decision or the paper-wide Holm. Read-only: preds/result JSONs, kd_meta.json
sidecars and checkpoints under runs/tifs, and the frozen functions
(analyze.load_cells, addendum.State, metrics.paired.audit_verdict,
metrics.paired.wilson_interval). All hit vectors are the official Stage 2
outputs (GPU, eval dtype bf16); nothing is re-scored.

Item 1 (S3/S4 registered): per cell and student seed, the triggered target
rates of P, C and R with Wilson 95% intervals, the frozen audit_verdict flags
(HIGH: Wilson lower bound >= 0.5; saturated: both point rates > 0.97) and
Cohen's h (P vs C); cells with >= 2 seeds also get the seed mean beside the
per-seed values.
  exposure_by_seed.csv, exposure_seed_mean.csv, exposure_primary_cells.csv

Item 2: for every run's P, C, R (R_B where present) and the teachers where
recorded, the target rate on the untriggered inputs and on each control input
(6 controls; the ATBA phrase controls where present), split clean vs
contaminated KD corpus (the registered two-way split) and, as an added
labelled variant, clean / label-poisoned / labels kept.
  controls_long.csv, controls_dedup.csv, controls_by_model.csv,
  controls_by_model_attack.csv, controls_summary.csv, controls_split.csv,
  controls_duplicate_check.csv, controls_replicate_check.csv,
  controls_contaminated_vs_clean.csv

Checkpoints vs configurations: a configuration is (family, dataset, KD type,
role, teacher lineage, teacher seed, corpus, corpus seed, student seed) for a
student and (family, dataset, role, lineage, teacher seed) for a teacher. Two
checkpoints of one configuration (trained in different runs) are listed
apart but counted once in every cross-block pool (controls_split.csv,
controls_contaminated_vs_clean.csv); controls_replicate_check.csv compares
their kd_meta.json, weights and hit vectors.

    python exposure_controls.py [--out DIR]     (default results/tifs/addendum/aux/g1-exposure-controls-tokens)
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True   # before any local import: no __pycache__ in the repository

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1_common as G  # noqa: E402
from counterfactual_kd.metrics.paired import audit_verdict, wilson_interval  # noqa: E402

A, AD, TOLS = G.A, G.AD, G.TOLS
mean = G.mean
EVAL = "official Stage 2 (GPU, bf16)"
SOURCE_ORDER = {"frozen": 0, "A8 hidden": 1, "A2 replicate": 2, "A10-a R_B": 3}


def ids(c):
    return {"source": c["source"], "block": c["block_label"], "run": c["run"], "fam": c["fam"],
            "dataset": c["dataset"], "kd": c["kd"], "attack": c["attack"], "corpus": c["corpus"],
            "corpus_split": G.corpus_split(c["corpus"]), "corpus_split3": G.corpus_split3(c["corpus"]),
            "corpus_class": G.corpus_class(c["corpus"]),
            "teacher_seed": c["teacher_seed"], "corpus_seed": c["corpus_seed"],
            "student_seed": c["student_seed"], "test_set": c["test_set"]}


def counts(av):
    k = av["counts"]
    return k["n11"] + k["n10"], k["n11"] + k["n01"]


def is_primary(st, c):
    return (c["source"] == "frozen" and c["block"] == "G" and c["corpus"] == "clean"
            and not c["single"] and c["kd"] != "ce"
            and (c["fam"], c["dataset"], c["kd"], c["attack"]) in st.prim)


def primary_fields(st, key):
    pr = st.prim.get(key)
    if pr is None:
        return {}
    out = {"primary_method": pr["method"], "primary_seeds": pr["seeds"], "primary_k": pr["k"],
           "primary_lo": pr.get("lo"), "primary_hi": pr.get("hi"),
           "teacher_qualified": pr["teacher_qualified"], "structural_null": pr["structural_null"]}
    out.update({f"primary_verdict@{t:g}": pr.get(f"verdict@{t:g}") for t in TOLS})
    return out


# ── item 1 ───────────────────────────────────────────────────────────

def exposure_rows(st, cells):
    rows = []
    for c in cells:
        if c["single"]:
            continue
        av = audit_verdict(c["P"], c["C"], 0.05)
        xP, xC = counts(av)
        row = ids(c)
        row.update(n=av["n"], hits_from=EVAL,
                   x_P=xP, P=av["p_trig"], P_lo=av["p_trig_ci"][0], P_hi=av["p_trig_ci"][1],
                   P_flag=av["p_exposure"],
                   x_C=xC, C=av["c_trig"], C_lo=av["c_trig_ci"][0], C_hi=av["c_trig_ci"][1],
                   C_flag=av["c_exposure"], saturated_PC=av["saturated"], h_PC=av["h"], k=av["k"])
        for t in TOLS:
            row[f"tango_seed_verdict@{t:g}"] = audit_verdict(c["P"], c["C"], t)["relative"]
        r, rsrc = G.reference_for(st, c)
        if r is not None:
            if len(r["P"]) != av["n"] or r["test_set"] != c["test_set"]:
                raise ValueError(f"R inputs differ from {c['run']} {c['attack']}")
            pr_ = audit_verdict(c["P"], r["P"], 0.05)
            cr_ = audit_verdict(c["C"], r["P"], 0.05)
            row.update(R_source=rsrc, R_run=r["run"], x_R=counts(pr_)[1], R=pr_["c_trig"],
                       R_lo=pr_["c_trig_ci"][0], R_hi=pr_["c_trig_ci"][1], R_flag=pr_["c_exposure"],
                       h_PR=pr_["h"], h_CR=cr_["h"], saturated_PR=pr_["saturated"],
                       saturated_CR=cr_["saturated"])
        rb = G.rcc_for(cells, c)
        if rb is not None and r is not None:
            b_ = audit_verdict(rb["P"], r["P"], 0.05)
            row.update(R_B_run=rb["run"], x_R_B=counts(b_)[0], R_B=b_["p_trig"],
                       R_B_lo=b_["p_trig_ci"][0], R_B_hi=b_["p_trig_ci"][1], R_B_flag=b_["p_exposure"],
                       h_RB_RC=b_["h"])
        if c["TP"] and c["TC"]:
            t_ = audit_verdict(c["TP"], c["TC"], 0.05)
            row.update(TP=t_["p_trig"], TP_lo=t_["p_trig_ci"][0], TP_hi=t_["p_trig_ci"][1],
                       TP_flag=t_["p_exposure"], TC=t_["c_trig"], TC_lo=t_["c_trig_ci"][0],
                       TC_hi=t_["c_trig_ci"][1], TC_flag=t_["c_exposure"],
                       saturated_teachers=t_["saturated"], h_teachers=t_["h"])
        row["primary_cell"] = is_primary(st, c)
        if row["primary_cell"]:
            row.update(primary_fields(st, (c["fam"], c["dataset"], c["kd"], c["attack"])))
        rows.append(row)
    return rows


def h_of(p, q):
    return 2.0 * (math.asin(math.sqrt(p)) - math.asin(math.sqrt(q)))


def fmt_by_seed(rows, key, digits=4):
    return "|".join(f"{r['student_seed']}:{r[key]:.{digits}f}" if isinstance(r[key], float)
                    else f"{r['student_seed']}:{r[key]}" for r in rows)


def seed_mean_rows(st, by_seed_rows):
    groups = defaultdict(list)
    for r in by_seed_rows:
        groups[(r["source"], r["block"], r["fam"], r["dataset"], r["kd"], r["attack"], r["corpus"],
                r["teacher_seed"], r["corpus_seed"])].append(r)
    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        if len(rs) < 2:
            continue
        rs = sorted(rs, key=lambda r: r["student_seed"])
        source, block, fam, ds, kd, atk, corpus, tseed, cseed = key
        pm, cm = mean([r["P"] for r in rs]), mean([r["C"] for r in rs])
        row = {"source": source, "block": block, "fam": fam, "dataset": ds, "kd": kd, "attack": atk,
               "corpus": corpus, "corpus_split": G.corpus_split(corpus), "corpus_split3": G.corpus_split3(corpus),
               "teacher_seed": tseed, "corpus_seed": cseed, "seeds": ",".join(str(r["student_seed"]) for r in rs),
               "r": len(rs), "n": rs[0]["n"],
               "P_mean": pm, "P_by_seed": fmt_by_seed(rs, "P"),
               "P_HIGH_seeds": f"{sum(r['P_flag'] == 'HIGH' for r in rs)}/{len(rs)}",
               "C_mean": cm, "C_by_seed": fmt_by_seed(rs, "C"),
               "C_HIGH_seeds": f"{sum(r['C_flag'] == 'HIGH' for r in rs)}/{len(rs)}",
               "saturated_seeds": f"{sum(bool(r['saturated_PC']) for r in rs)}/{len(rs)}",
               "k_mean": mean([r["k"] for r in rs]), "k_by_seed": fmt_by_seed(rs, "k"),
               "h_PC_mean_of_seeds": mean([r["h_PC"] for r in rs]),
               "h_PC_at_mean_rates": h_of(pm, cm), "h_PC_by_seed": fmt_by_seed(rs, "h_PC")}
        r42 = next((r for r in rs if r["student_seed"] == 42 and r.get("R") is not None), None)
        if r42:
            row.update(R_s42=r42["R"], R_s42_lo=r42["R_lo"], R_s42_hi=r42["R_hi"],
                       R_s42_flag=r42["R_flag"], R_source=r42["R_source"])
        if rs[0].get("TP") is not None:
            row.update(TP=rs[0]["TP"], TC=rs[0]["TC"])
        if rs[0]["primary_cell"]:
            pf = primary_fields(st, (fam, ds, kd, atk))
            row.update(pf)
            pr = st.prim[(fam, ds, kd, atk)]
            row.update(check_P_mean_minus_primary_raw_asr=pm - pr["raw_asr"],
                       check_C_mean_minus_primary_C_trig=cm - pr["C_trig"],
                       check_k_mean_minus_primary_k=(row["k_mean"] - pr["k"])
                       if pr["method"].startswith("seed-pooled") else None)
        out.append(row)
    return out


def exposure_axis(rs):
    if not rs:
        return ""
    p = sum(r["P_flag"] == "HIGH" for r in rs)
    c = sum(r["C_flag"] == "HIGH" for r in rs)
    s = sum(bool(r["saturated_PC"]) for r in rs)
    lab = ("P HIGH in all seeds" if p == len(rs) else "P HIGH in some seeds" if p else "P not HIGH")
    lab += ("; C HIGH in all seeds" if c == len(rs) else "; C HIGH in some seeds" if c else "; C not HIGH")
    if s:
        lab += f"; saturated in {s}/{len(rs)} seeds"
    return lab


def primary_cell_rows(st, by_seed_rows):
    """One row per frozen primary cell (31): exposure axis beside the
    (unchanged) attribution verdict."""
    groups = defaultdict(list)
    for r in by_seed_rows:
        if r["primary_cell"]:
            groups[(r["fam"], r["dataset"], r["kd"], r["attack"])].append(r)
    out = []
    for key, pr in sorted(st.prim.items()):
        seeds_used = [int(s) for s in pr["seeds"].split(",")]
        rs = sorted([r for r in groups.get(key, []) if r["student_seed"] in seeds_used],
                    key=lambda r: r["student_seed"])
        if [r["student_seed"] for r in rs] != seeds_used:
            raise ValueError(f"{key}: seeds {seeds_used} vs rows {[r['student_seed'] for r in rs]}")
        fam, ds, kd, atk = key
        pm, cm = mean([r["P"] for r in rs]), mean([r["C"] for r in rs])
        row = {"fam": fam, "dataset": ds, "kd": kd, "attack": atk, "seeds": pr["seeds"],
               "r": len(rs), "n": pr["n"], "teacher_asr": pr["teacher_asr"],
               "teacher_asr_null": pr["teacher_asr_null"],
               "P_mean": pm, "P_by_seed": fmt_by_seed(rs, "P"),
               "P_wilson_by_seed": "|".join(f"{r['student_seed']}:[{r['P_lo']:.4f},{r['P_hi']:.4f}]" for r in rs),
               "P_HIGH_seeds": f"{sum(r['P_flag'] == 'HIGH' for r in rs)}/{len(rs)}",
               "C_mean": cm, "C_by_seed": fmt_by_seed(rs, "C"),
               "C_wilson_by_seed": "|".join(f"{r['student_seed']}:[{r['C_lo']:.4f},{r['C_hi']:.4f}]" for r in rs),
               "C_HIGH_seeds": f"{sum(r['C_flag'] == 'HIGH' for r in rs)}/{len(rs)}",
               "saturated_seeds": f"{sum(bool(r['saturated_PC']) for r in rs)}/{len(rs)}",
               "h_PC_mean_of_seeds": mean([r["h_PC"] for r in rs]),
               "h_PC_at_mean_rates": h_of(pm, cm),
               "h_PC_by_seed": fmt_by_seed(rs, "h_PC")}
        r42 = next((r for r in rs if r["student_seed"] == 42 and r.get("R") is not None), None)
        if r42:
            row.update(R_s42=r42["R"], R_s42_lo=r42["R_lo"], R_s42_hi=r42["R_hi"], R_s42_flag=r42["R_flag"],
                       h_PR_s42=r42["h_PR"], h_CR_s42=r42["h_CR"])
            if r42.get("R_B") is not None:
                row.update(R_B_s42=r42["R_B"], R_B_s42_lo=r42["R_B_lo"], R_B_s42_hi=r42["R_B_hi"],
                           R_B_s42_flag=r42["R_B_flag"])
        row.update(primary_fields(st, key))
        row["exposure_axis"] = exposure_axis(rs)
        out.append(row)
    return out


# ── item 2 ───────────────────────────────────────────────────────────

ROLE_NAME = {"P": "P (poisoned-teacher student)", "C": "C (clean-teacher student)",
             "R": "R (CE-only, no teacher)", "R_B": "R_B (CE-only, 0.5*CE, A10-a)",
             "teacher_P": "teacher P (poisoned)", "teacher_C": "teacher C (clean)"}
# Probes whose inputs depend on the attack's own triggered inputs.
ATTACK_DEPENDENT = ("triggered", "rare_unplanted_sub", "sent_neutral_sub")
PROBE_ORDER = ["untriggered", "rare_unplanted", "common_neutral", "loaded_positive", "sent_partial",
               "atba_phrase_sst2", "atba_phrase_agnews", "triggered", "rare_unplanted_sub",
               "sent_neutral_sub"]
INDEP_PROBES = [p for p in PROBE_ORDER if p not in ATTACK_DEPENDENT]
# The ATBA attack's own Stage 2 inputs are the dataset's ATBA phrase as a
# prefix, i.e. the same texts as the atba_phrase_<dataset> control
# (tokens_eval_reconstruction_check.csv; hit vectors in controls_replicate_check.csv).
VIA_ATBA = {ds: f"atba_phrase_{ds}_via_atba_trigger" for ds in ("sst2", "agnews")}
VIA_ATBA_LABEL = "ATBA trigger inputs (identical to the atba_phrase_<dataset> prefix inputs)"
TRIGGER_FREE = ("untriggered", "rare_unplanted", "common_neutral", "loaded_positive", "sent_partial")
# Contaminated corpus -> the attack whose trigger its flagged rows carry (checked against the cells).
CORPUS_ATTACK = {"badnets10": "badnets", "addsent10": "addsent", "synbkd10": "synbkd",
                 "kl01": "badnets", "kl10": "badnets"}
SPLIT_COLS = {"clean": "clean", "contaminated": "contaminated", "label-poisoned": "label_poisoned",
              "labels kept": "labels_kept"}


def model_ids(c):
    """One id per trained checkpoint. Students: run + KD type + teacher lineage
    + student seed (the C student and R are shared by the attacks of a run);
    teachers: family, dataset, attack/clean trainer, teacher seed (shared
    across runs)."""
    base = f"{c['run']}/{c['dataset']}"
    s, t = c["student_seed"], c["teacher_seed"]
    ct = AD.clean_tag(c["attack"])
    return {
        "P": f"{base}/{c['kd']}/P:{c['attack']}/s{s}",
        "C": f"{base}/{c['kd']}/C:{ct}/s{s}",
        "R": f"{base}/ce/R/s{s}",
        "R_B": f"{base}/ce(0.5)/R_B/s{s}",
        "teacher_P": f"teacher/{c['fam']}/{c['dataset']}/P:{c['attack']}/t{t}",
        "teacher_C": f"teacher/{c['fam']}/{c['dataset']}/C:{ct}/t{t}",
    }


def config_ids(c):
    """One id per training configuration (run-independent). R has no teacher,
    so the teacher seed does not enter; a teacher's id is its checkpoint id."""
    s, t = c["student_seed"], c["teacher_seed"]
    ct = AD.clean_tag(c["attack"])
    base = f"{c['fam']}/{c['dataset']}"
    tail = f"{c['corpus']}/c{c['corpus_seed']}/s{s}"
    return {
        "P": f"{base}/{c['kd']}/P:{c['attack']}/t{t}/{tail}",
        "C": f"{base}/{c['kd']}/C:{ct}/t{t}/{tail}",
        "R": f"{base}/ce/R/{tail}",
        "R_B": f"{base}/ce(0.5)/R_B/{tail}",
        "teacher_P": f"teacher/{c['fam']}/{c['dataset']}/P:{c['attack']}/t{t}",
        "teacher_C": f"teacher/{c['fam']}/{c['dataset']}/C:{ct}/t{t}",
    }


def wilson(x, n):
    return wilson_interval(x, n, 0.95)


def control_rows(cells, pidx):
    rows, vectors = [], {}
    for c in cells:
        path = pidx.get(G.cell_key(c))
        if path is None:
            raise KeyError(f"no preds file for {G.cell_key(c)}")
        raw = json.load(open(path))
        if raw["preds_obs"] != c["P"] or raw["preds_null"] != c["C"]:
            raise ValueError(f"loader / file mismatch: {path}")
        mid, cid = model_ids(c), config_ids(c)
        if c["single"]:
            roles = {("R_B" if c["source"] == "A10-a R_B" else "R"): ("P", "Pc", "p")}
        else:
            roles = {"P": ("P", "Pc", "p"), "C": ("C", "Cc", "c")}
        probes = {}
        for role, (trig, untrig, side) in roles.items():
            probes[role] = {"triggered": c[trig], "untriggered": c[untrig]}
            for name, ctl in c["controls"].items():
                probes[role][name] = ctl.get(side)
        tp = {"triggered": raw.get("teacher_p_trig"), "untriggered": raw.get("teacher_p_clean")}
        tc = {"triggered": raw.get("teacher_c_trig"), "untriggered": raw.get("teacher_c_clean")}
        for name, ctl in c["controls"].items():
            tp[name], tc[name] = ctl.get("teacher_p"), ctl.get("teacher_c")
        if not c["single"]:
            probes["teacher_P"], probes["teacher_C"] = tp, tc
        else:
            # The teachers a CE-only cell records are the same models (and
            # inputs) as in the run's paired cells of that attack; kept for
            # the duplicate check, marked so the summaries do not count them twice.
            probes["teacher_P(ce-cell)"], probes["teacher_C(ce-cell)"] = tp, tc
        for role, pr in probes.items():
            for probe, hits in pr.items():
                if not hits:
                    continue
                spec = c["controls"][probe]["spec"] if probe in c["controls"] else {}
                mode = ("own trigger" if probe == "triggered" else "none" if probe == "untriggered"
                        else spec.get("mode", "insert") + ("/" + spec["position"] if spec.get("position") else ""))
                att_dep = probe in ATTACK_DEPENDENT
                if att_dep != (probe == "triggered" or spec.get("mode") == "substitute"):
                    raise ValueError(f"probe {probe}: attack dependence does not match its mode")
                base_role = role.replace("(ce-cell)", "")
                m_id = mid[base_role]
                x, n = sum(hits), len(hits)
                lo, hi = wilson(x, n)
                row = ids(c)
                row.update(role=role, role_name=ROLE_NAME[base_role], model_id=m_id, config_id=cid[base_role],
                           probe=probe, probe_mode=mode, probe_depends_on_attack=att_dep,
                           fallbacks=spec.get("fallbacks"), x=x, n=n, rate=x / n,
                           wilson_lo=lo, wilson_hi=hi, hits_from=EVAL)
                row.update(cell_block=row["block"], cell_kd=c["kd"], cell_student_seed=c["student_seed"],
                           cell_corpus=c["corpus"])
                if base_role.startswith("teacher"):
                    # A teacher does not depend on the KD corpus or the student;
                    # the same checkpoint is recorded by every run that uses it.
                    row.update(block="teachers", corpus="n/a (teacher)", corpus_split="n/a (teacher)",
                               corpus_split3="n/a (teacher)", corpus_class="n/a (teacher)", kd="-",
                               student_seed="-", corpus_seed="-")
                rows.append(row)
                dkey = (m_id, probe, c["attack"] if att_dep else "")
                vectors.setdefault(dkey, []).append((c["run"], c["attack"], role, hits))
    return rows, vectors


def dedup(rows):
    """First row per (model_id, probe[, attack for attack-dependent probes]),
    excluding the duplicate teacher rows of CE-only cells."""
    seen, out = set(), []
    for r in rows:
        if r["role"].endswith("(ce-cell)"):
            continue
        key = (r["model_id"], r["probe"], r["attack"] if r["probe_depends_on_attack"] else "")
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(r))           # a copy: annotate() must not touch controls_long rows
    return out


def config_map(dd):
    """{model_id: (config_id, counted model_id, n checkpoints of the config)}.
    The counted checkpoint of a configuration is the first by (source:
    frozen, A8, A2, A10-a; then model_id), so the frozen blocks keep theirs."""
    source, cfg_of = {}, {}
    for r in dd:
        source.setdefault(r["model_id"], r["source"])
        if cfg_of.setdefault(r["model_id"], r["config_id"]) != r["config_id"]:
            raise ValueError(f"{r['model_id']}: two configuration ids")
    members = defaultdict(list)
    for m, cfg in cfg_of.items():
        members[cfg].append(m)
    out = {}
    for cfg, ms in members.items():
        ms = sorted(ms, key=lambda m: (SOURCE_ORDER[source[m]], m))
        for m in ms:
            out[m] = (cfg, ms[0], len(ms))
    return out


def annotate(rows, cmap):
    for r in rows:
        cfg, counted, n_ck = cmap[r["model_id"]]
        r["n_checkpoints_of_config"] = n_ck
        r["replicate_checkpoint_of"] = "" if counted == r["model_id"] else counted
    return rows


MODEL_KEYS = ("fam", "dataset", "block", "corpus", "corpus_split", "corpus_split3", "corpus_class", "kd",
              "role_name", "teacher_seed", "corpus_seed", "student_seed")


def by_model_rows(dd, cmap):
    """Wide tables: attack-independent probes, one row per checkpoint; attack-
    dependent probes, one row per (checkpoint, attack whose trigger is probed).
    Cells hold the target rate; '<probe>_x' the hit count (n inputs). The
    attack-independent table also carries atba_phrase_<ds>_via_atba_trigger:
    the checkpoint's rate on the ATBA attack's own inputs of its dataset (the
    same texts as the atba_phrase_<ds> control; recorded for the models
    evaluated on the ATBA trigger)."""
    heads, rates, attacks = {}, defaultdict(dict), defaultdict(set)
    via = {}
    for r in dd:
        if r["role"] in ("R", "R_B"):
            lineage = "-"
        elif r["role"] in ("P", "teacher_P"):
            lineage = r["attack"]
        else:
            lineage = AD.clean_tag(r["attack"])
        key = (r["model_id"], r["attack"]) if r["probe_depends_on_attack"] else (r["model_id"], "")
        cfg, counted, n_ck = cmap[r["model_id"]]
        heads.setdefault(key, {"model_id": r["model_id"], **{k: r[k] for k in MODEL_KEYS},
                               "lineage": lineage, "config_id": cfg, "n_checkpoints_of_config": n_ck,
                               "replicate_checkpoint_of": "" if counted == r["model_id"] else counted,
                               "n": r["n"], **({"attack_probed": r["attack"]} if key[1] else {})})
        if heads[key]["n"] != r["n"]:
            raise ValueError(f"{key}: probes with different n")
        rates[key][r["probe"]] = (r["rate"], r["x"], r["fallbacks"])
        attacks[key].add(r["attack"])
        if r["probe"] == "triggered" and r["attack"] == "atba":
            via[r["model_id"]] = (r["dataset"], r["rate"], r["x"])
    indep, dep = [], []
    for key, head in heads.items():
        row = dict(head)
        if key[1]:
            for p in ATTACK_DEPENDENT:
                rate, x, fb = rates[key].get(p, (None, None, None))
                row[p], row[f"{p}_x"] = rate, x
                if p != "triggered":
                    row[f"{p}_fallbacks"] = fb
            dep.append(row)
        else:
            row["cells_seen_for_attacks"] = ",".join(sorted(attacks[key]))
            for p in INDEP_PROBES:
                rate, x, _ = rates[key].get(p, (None, None, None))
                row[p], row[f"{p}_x"] = rate, x
            v = via.get(key[0])
            for ds, col in VIA_ATBA.items():
                ok = v is not None and v[0] == ds
                row[col], row[f"{col}_x"] = (v[1], v[2]) if ok else (None, None)
            indep.append(row)
    order = ("fam", "dataset", "block", "corpus", "kd", "role_name", "model_id", "attack_probed")
    srt = lambda rows: sorted(rows, key=lambda r: tuple(str(r.get(k)) for k in order))  # noqa: E731
    return srt(indep), srt(dep)


def summary_rows(dd, cmap):
    """Mean / min / max target rate over the distinct checkpoints (checkpoint x
    attack for attack-dependent probes) of each block-level group. A group
    never holds two checkpoints of one configuration; 'replicate_checkpoints'
    names any checkpoint of the group whose configuration is counted in
    another block (pooling such groups across blocks counts it twice)."""
    groups = defaultdict(list)
    for r in dd:
        teacher = r["role"].startswith("teacher")
        key = (r["fam"], r["dataset"], r["corpus_split"], r["corpus_split3"], r["corpus_class"], r["corpus"],
               "teachers" if teacher else r["block"], r["kd"], r["role_name"],
               f"t{r['teacher_seed']}" if teacher else "", r["probe"],
               r["attack"] if r["probe_depends_on_attack"] else "(any)")
        groups[key].append(r)
    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        fam, ds, split, split3, cc, corpus, block, kd, role, tseed, probe, atk = key
        cfgs = [cmap[r["model_id"]][0] for r in rs]
        if len(set(cfgs)) != len(cfgs):
            raise ValueError(f"group {key} holds two checkpoints of one configuration")
        rates = [r["rate"] for r in rs]
        reps = [f"{r['model_id']} (replicate of {cmap[r['model_id']][1]})" for r in rs
                if cmap[r["model_id"]][1] != r["model_id"]]
        out.append({"fam": fam, "dataset": ds, "corpus_split": split, "corpus_split3": split3,
                    "corpus_class": cc, "corpus": corpus,
                    "block": block, "kd": kd, "role": role, "teacher_seed": tseed, "probe": probe,
                    "attack_probed": atk, "n_models": len(rs), "n_inputs": rs[0]["n"],
                    "mean_rate": mean(rates), "min_rate": min(rates), "max_rate": max(rates),
                    "models": "|".join(r["model_id"] for r in rs), "replicate_checkpoints": "|".join(reps)})
    return out


def config_values(dd, cmap):
    """Per student configuration: head fields (from the counted checkpoint)
    and {probe: (rate, x, n)} over all its checkpoints for the attack-
    independent probes and the ATBA-trigger phrase pseudo-probe. Shared probes
    of two checkpoints must agree exactly."""
    out = {}
    for r in sorted(dd, key=lambda r: r["model_id"] != cmap[r["model_id"]][1]):   # counted checkpoint first
        if r["role"].startswith("teacher"):
            continue
        if r["probe_depends_on_attack"]:
            if not (r["probe"] == "triggered" and r["attack"] == "atba"):
                continue
            probe = VIA_ATBA[r["dataset"]]
        else:
            probe = r["probe"]
        cfg, counted, n_ck = cmap[r["model_id"]]
        d = out.setdefault(cfg, {"head": r, "counted": counted, "n_checkpoints": n_ck, "probes": {},
                                 "probe_checkpoints": defaultdict(set)})
        v = (r["rate"], r["x"], r["n"])
        if probe in d["probes"] and d["probes"][probe] != v:
            raise ValueError(f"{cfg} / {probe}: checkpoints disagree ({d['probes'][probe]} vs {v})")
        d["probes"][probe] = v
        d["probe_checkpoints"][probe].add(r["model_id"])
    return out


def stats(vals):
    return {"n": len(vals), "mean": mean(vals) if vals else None,
            "min": min(vals) if vals else None, "max": max(vals) if vals else None}


def split_rows(cv):
    """Item 2's clean vs contaminated split for the students (the registered
    two-way split; 'contaminated' pools label-poisoned and labels-kept
    corpora), with the added three-way columns label_poisoned / labels_kept:
    per family, dataset, role and probe, the mean / min / max rate over the
    distinct configurations (replicate checkpoints counted once) trained on
    each kind of KD corpus, every block pooled. Attack-independent probes, plus
    the ATBA trigger inputs as the dataset's ATBA phrase for C, R and R_B (the
    P students with that vector carry the ATBA backdoor; their rate is item 1's
    triggered rate)."""
    groups = defaultdict(lambda: defaultdict(list))
    for cfg, d in cv.items():
        h = d["head"]
        for probe, (rate, x, n) in d["probes"].items():
            if probe in VIA_ATBA.values() and h["role"] == "P":
                continue
            g = groups[(h["fam"], h["dataset"], h["role_name"], probe)]
            n_ck = len(d["probe_checkpoints"][probe])        # checkpoints that recorded this probe
            g[h["corpus_split"]].append((rate, h["corpus"], n_ck, h["corpus_split3"]))
            if h["corpus_split"] != "clean":
                g[h["corpus_split3"]].append((rate, h["corpus"], n_ck, h["corpus_split3"]))
    probe_rank = {p: i for i, p in enumerate(INDEP_PROBES + list(VIA_ATBA.values()))}
    out = []
    for (fam, ds, role, probe), by in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2],
                                                                             probe_rank[kv[0][3]])):
        row = {"fam": fam, "dataset": ds, "role": role, "probe": probe,
               "probe_inputs": VIA_ATBA_LABEL if probe in VIA_ATBA.values() else "as named"}
        for split, col in SPLIT_COLS.items():
            rs = by.get(split, [])
            s = stats([x[0] for x in rs])
            row[f"{col}_n_configurations"] = s["n"]
            row[f"{col}_n_checkpoints"] = sum(x[2] for x in rs)
            row[f"{col}_mean"], row[f"{col}_min"], row[f"{col}_max"] = s["mean"], s["min"], s["max"]
            row[f"{col}_corpora"] = ",".join(sorted({x[1] for x in rs}))
        for col in ("contaminated", "label_poisoned", "labels_kept"):
            row[f"{col}_minus_clean"] = (row[f"{col}_mean"] - row["clean_mean"]
                                         if row[f"{col}_mean"] is not None and row["clean_mean"] is not None else None)
        out.append(row)
    return out


def contaminated_vs_clean_rows(cv, wide_att, cmap):
    """Every student configuration trained on a contaminated corpus, on each
    trigger-free probe and the two substitute controls of its corpus attack,
    against the clean-corpus students of the same family and dataset (all
    roles, configurations counted once): range of the rate and of the rate
    minus the model's own untriggered rate."""
    att = {}
    for r in wide_att:
        cfg, counted, _ = cmap[r["model_id"]]
        if counted == r["model_id"]:
            att[(cfg, r["attack_probed"])] = r
    clean_ref = defaultdict(list)          # (fam, ds, probe, attack) -> [(rate, rate - untriggered)]
    for cfg, d in cv.items():
        h = d["head"]
        if h["corpus"] != "clean":
            continue
        un = d["probes"].get("untriggered")
        for p in TRIGGER_FREE:
            if p in d["probes"]:
                clean_ref[(h["fam"], h["dataset"], p, "")].append((d["probes"][p][0], d["probes"][p][0] - un[0]))
    for (cfg, atk), r in att.items():
        d = cv.get(cfg)
        if d is None or d["head"]["corpus"] != "clean":
            continue
        un = d["probes"]["untriggered"][0]
        for p in ("rare_unplanted_sub", "sent_neutral_sub"):
            if r.get(p) is not None:
                clean_ref[(d["head"]["fam"], d["head"]["dataset"], p, atk)].append((r[p], r[p] - un))
    out = []
    for cfg, d in sorted(cv.items()):
        h = d["head"]
        if h["corpus"] == "clean":
            continue
        catk = CORPUS_ATTACK[h["corpus"]]
        a = att.get((cfg, catk))
        if a is None or a.get("triggered") is None:
            raise ValueError(f"{cfg}: not evaluated on its corpus attack {catk}")
        n = d["probes"]["untriggered"][2]
        tlo, thi = wilson(a["triggered_x"], n)
        un = d["probes"]["untriggered"][0]
        items = [(p, "", d["probes"][p][0], d["probes"][p][1]) for p in TRIGGER_FREE]
        items += [(p, catk, a[p], a[f"{p}_x"]) for p in ("rare_unplanted_sub", "sent_neutral_sub") if a.get(p) is not None]
        for p, patk, rate, x in items:
            ref = clean_ref[(h["fam"], h["dataset"], p, patk)]
            lo, hi = wilson(x, n)
            row = {"model_id": d["counted"], "config_id": cfg, "fam": h["fam"], "dataset": h["dataset"],
                   "block": h["block"], "corpus": h["corpus"], "corpus_split3": h["corpus_split3"],
                   "corpus_attack": catk, "role_name": h["role_name"], "kd": h["kd"],
                   "teacher_seed": h["teacher_seed"], "student_seed": h["student_seed"],
                   "probe": p, "attack_probed": patk or "(any)", "x": x, "n": n, "rate": rate,
                   "wilson_lo": lo, "wilson_hi": hi, "untriggered": un, "rate_minus_untriggered": rate - un,
                   "corpus_trigger_rate": a["triggered"], "corpus_trigger_wilson_lo": tlo,
                   "learned_corpus_trigger": "HIGH" if tlo >= 0.5 else "-",
                   "clean_n_configurations": len(ref),
                   "clean_min": min(x_[0] for x_ in ref) if ref else None,
                   "clean_max": max(x_[0] for x_ in ref) if ref else None,
                   "clean_max_rate_minus_untriggered": max(x_[1] for x_ in ref) if ref else None,
                   "clean_min_rate_minus_untriggered": min(x_[1] for x_ in ref) if ref else None}
            row["above_clean_max"] = bool(ref) and rate > row["clean_max"]
            row["below_clean_min"] = bool(ref) and rate < row["clean_min"]
            row["above_clean_max_minus_untriggered"] = bool(ref) and rate - un > row["clean_max_rate_minus_untriggered"]
            out.append(row)
    return out


def duplicate_check(vectors):
    out = []
    for key, occ in sorted(vectors.items()):
        if len(occ) < 2:
            continue
        first = occ[0][3]
        dis = [sum(a != b for a, b in zip(first, o[3])) for o in occ[1:]]
        out.append({"model_id": key[0], "probe": key[1], "attack_key": key[2],
                    "n_evaluations": len(occ), "n": len(first),
                    "where": "|".join(f"{o[0]}:{o[1]}:{o[2]}" for o in occ),
                    "max_disagreeing_inputs": max(dis),
                    "rates": "|".join(f"{sum(o[3]) / len(o[3]):.4f}" for o in occ)})
    return out


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def student_dir(run, ds, kd, role, lineage, student_seed):
    """The checkpoint folder of a student (located through its kd_meta.json)."""
    base = os.path.join(G.RUNS, run, "artifacts", "students")
    sub = {"P": "poisoned_student", "C": "clean_student", "R": "ce_student", "R_B": "ce_student"}[role]
    hits = []
    for d in sorted(os.listdir(base)):
        p = os.path.join(base, d, sub, "kd_meta.json")
        if not os.path.exists(p):
            continue
        m = json.load(open(p))
        if m.get("dataset") != ds or int(m.get("student_seed", -1)) != int(student_seed):
            continue
        if role in ("P", "C") and (m.get("kd_type") != kd or not d.startswith(f"{lineage}_")):
            continue
        hits.append(os.path.join(base, d, sub))
    if len(hits) != 1:
        raise ValueError(f"{run}/{ds}/{kd}/{role}:{lineage}/s{student_seed}: {len(hits)} checkpoint folders")
    return hits[0]


def replicate_rows(wide, vectors, cmap):
    """controls_replicate_check.csv: (1) for every configuration with two or
    more checkpoints, kd_meta.json (without train_time_sec) and the weights'
    sha256; (2) their hit vectors on every probe both record; (3) for every
    configuration (or teacher checkpoint) recorded both on the ATBA trigger
    inputs and on the atba_phrase_<ds> control, the two hit vectors."""
    out = []
    head = {r["model_id"]: r for r in wide}
    members = defaultdict(list)
    for m, (cfg, counted, n_ck) in cmap.items():
        members[cfg].append(m)
    for cfg, ms in sorted(members.items()):
        if len(ms) < 2:
            continue
        ms = sorted(ms, key=lambda m: m != cmap[m][1])
        metas, shas, dirs = [], [], []
        for m in ms:
            h = head[m]
            run = m.split("/")[0]
            role = h["role_name"].split(" ")[0]
            dpath = student_dir(run, h["dataset"], h["kd"], role, h["lineage"], h["student_seed"])
            km = json.load(open(os.path.join(dpath, "kd_meta.json")))
            km.pop("train_time_sec", None)
            metas.append(km)
            shas.append(sha256_file(os.path.join(dpath, "model.safetensors")))
            dirs.append(G.rel(dpath))
        out.append({"kind": "replicate checkpoints: kd_meta.json and weights", "config_id": cfg,
                    "checkpoints": "|".join(ms), "counted_checkpoint": ms[0], "checkpoint_dirs": "|".join(dirs),
                    "kd_meta_identical_except_train_time_sec": all(x == metas[0] for x in metas[1:]),
                    "weights_sha256": "|".join(s[:16] for s in shas),
                    "weights_identical": all(s == shas[0] for s in shas[1:])})
        for (m_id, probe, akey), occ in sorted(vectors.items()):
            if m_id != ms[0]:
                continue
            others = [(m, vectors.get((m, probe, akey))) for m in ms[1:]]
            others = [(m, o) for m, o in others if o]
            if not others:
                continue
            first = occ[0][3]
            dis = [sum(a != b for a, b in zip(first, o[0][3])) for _, o in others]
            out.append({"kind": "replicate checkpoints: hit vectors on a shared probe", "config_id": cfg,
                        "checkpoints": "|".join([ms[0]] + [m for m, _ in others]), "counted_checkpoint": ms[0],
                        "probe": probe, "attack_key": akey, "n": len(first),
                        "max_disagreeing_inputs": max(dis),
                        "rates": "|".join(f"{sum(v) / len(v):.4f}" for v in [first] + [o[0][3] for _, o in others])})
    # (3) ATBA trigger inputs vs the atba_phrase_<ds> control
    phrase, trig = defaultdict(list), defaultdict(list)
    for (m_id, probe, akey), occ in vectors.items():
        if m_id not in cmap:
            continue
        cfg = cmap[m_id][0]
        ds = m_id.split("/")[1] if not m_id.startswith("teacher/") else m_id.split("/")[2]
        if probe == f"atba_phrase_{ds}":
            phrase[(cfg, ds)].append((m_id, occ[0][3]))
        elif probe == "triggered" and akey == "atba":
            trig[(cfg, ds)].append((m_id, occ[0][3]))
    for key in sorted(set(phrase) & set(trig)):
        (mp, vp), (mt, vt) = phrase[key][0], trig[key][0]
        out.append({"kind": "ATBA trigger inputs vs atba_phrase_<ds> control", "config_id": key[0],
                    "checkpoints": mp if mp == mt else f"{mp}|{mt}", "probe": f"atba_phrase_{key[1]} vs triggered",
                    "attack_key": "atba", "n": len(vp), "max_disagreeing_inputs": sum(a != b for a, b in zip(vp, vt)),
                    "rates": f"{sum(vp) / len(vp):.4f}|{sum(vt) / len(vt):.4f}"})
    return out


def main():
    G.parse_args(__doc__.splitlines()[0])
    t0 = time.time()
    st, cells = G.load_everything()
    pidx = G.preds_index()
    print(f"{len(cells)} cells ({sum(c['source'] == 'frozen' for c in cells)} frozen); "
          f"{len(st.prim)} primary cells; snapshot {time.strftime('%FT%TZ', time.gmtime())}; out {G.OUT}")

    by_seed = exposure_rows(st, cells)
    seed_mean = seed_mean_rows(st, by_seed)
    prim_rows = primary_cell_rows(st, by_seed)
    long_rows, vectors = control_rows(cells, pidx)
    dd = dedup(long_rows)
    cmap = config_map(dd)
    annotate(dd, cmap)
    wide, wide_att = by_model_rows(dd, cmap)
    summ = summary_rows(dd, cmap)
    cv = config_values(dd, cmap)
    split = split_rows(cv)
    cvc = contaminated_vs_clean_rows(cv, wide_att, cmap)
    dup = duplicate_check(vectors)
    rep = replicate_rows(wide, vectors, cmap)
    for name, rows in (("exposure_by_seed", by_seed), ("exposure_seed_mean", seed_mean),
                       ("exposure_primary_cells", prim_rows), ("controls_long", long_rows),
                       ("controls_dedup", dd), ("controls_by_model", wide),
                       ("controls_by_model_attack", wide_att), ("controls_summary", summ),
                       ("controls_split", split), ("controls_duplicate_check", dup),
                       ("controls_replicate_check", rep), ("controls_contaminated_vs_clean", cvc)):
        n = G.write_csv(rows, os.path.join(G.OUT, f"{name}.csv"))
        print(f"  {name}: {n} rows")
    newest, n_preds = G.newest_preds_utc()
    n_ck = len(wide)
    n_cfg = len({r["config_id"] for r in wide})
    meta = {"snapshot_utc": time.strftime("%FT%TZ", time.gmtime()),
            "newest_preds_file_utc": newest, "n_preds_files": n_preds,
            "n_cells": len(cells), "cells_by_source": {s: sum(c["source"] == s for c in cells)
                                                       for s in sorted({c["source"] for c in cells})},
            "n_primary_cells": len(st.prim),
            "n_checkpoints": n_ck, "n_student_checkpoints": sum(r["block"] != "teachers" for r in wide),
            "n_teacher_checkpoints": sum(r["block"] == "teachers" for r in wide),
            "n_configurations": n_cfg,
            "replicate_checkpoints": {r["model_id"]: r["replicate_checkpoint_of"] for r in wide if r["replicate_checkpoint_of"]},
            "runs": sorted({c["run"] for c in cells}), "seconds": round(time.time() - t0, 1)}
    json.dump(meta, open(os.path.join(G.OUT, "exposure_controls_meta.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "runs"}))


if __name__ == "__main__":
    main()
