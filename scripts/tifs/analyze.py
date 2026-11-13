#!/usr/bin/env python3
"""Build the TIFS result tables from runs/tifs (works on partial results).

Outputs (``--out``, default results/tifs/report/):

  primary.csv      the primary estimate of every clean-corpus battery cell
                   (docs/tifs_plan.md §0.3): seed-pooled where the cell has
                   student seeds 42-44, transported (seed 42 plus the
                   family's between-seed variance) where it has one,
                   seed-42 Tango for the structural nulls
  battery.csv      the primary decision per (family, dataset, KD type):
                   "no transfer within delta" needs every qualified attack's
                   one-sided upper bound below delta (intersection-union),
                   "transfers" needs one Holm-adjusted positive
  variance.csv     between-seed variance per battery (pooled, Satterthwaite
                   df) and per attack
  gate_sensitivity.csv  the primary decisions under teacher gates ASR
                   {0.80..0.95} x accuracy drop {2, 5, 10} points
  reference_variance.csv  between-teacher-seed variance of k (A/A), added to
                   the primary SE in primary.csv's robust_* columns
  secondary.csv    contaminated-corpus and positive-control cells: Tango and
                   the transported interval
  audit.csv        per clean-corpus cell and student seed, the raw student
                   ASR an attack paper would report next to the
                   teacher-attributable k = P - C (Tango), k*, the CE-only
                   R and the verdicts
  battery_by_seed.csv  the battery decision from each student seed alone
  cells.csv        every evaluated cell: k = P_trig - C_trig (Tango, delta*,
                   verdicts, TOST p), teacher and clean-accuracy columns,
                   Δb and k* with joint intervals
  factorial.csv    E1/E4: P, C and the CE-only reference R on the same
                   triggered inputs; contrasts P-C, C-R, P-R
  interaction.csv  E1: corpus x teacher, (P1-C1) - (P0-C0), suppression
                   C1-R1 and amplification P1-R1 by joint bootstrap
  seeds.csv        S2: per configuration, per-seed k and the Hartung–Knapp
                   random-effects pool across seeds
  matching.csv     S8: spread of matched (same seed) vs unmatched
                   (cross-seed) P-C contrasts
  aa.csv           S7/E2: independent-teacher contrasts C-C', P-C', P'-C', P-P',
                   and C-C' at the ATBA operating point (ATBA phrase controls)
  controls.csv     E8: P-C on every control trigger
  summary.md       verdict counts and the multiplicity statements (S5)

    python scripts/tifs/analyze.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import os
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from counterfactual_kd.analysis.tifs import (  # noqa: E402
    holm, joint_bootstrap, paired_contrast, random_effects_hk, se_from_interval,
    t_contrast,
)
from counterfactual_kd.metrics.paired import _z  # noqa: E402

TOLS = (0.02, 0.05, 0.10)
# Teacher qualification, fixed before any v2 result was seen (2026-10-21):
# an attack enters the audit only if its poisoned teacher reaches this ASR on
# the triggered non-target inputs and loses at most this much clean accuracy
# against its matched clean teacher. Unqualified cells are still tabulated.
# The accuracy allowance is 5 points (author's decision, 2026-10-21, before
# any Stage 2 result existed).
GATE_ASR = 0.90
GATE_CA_DROP = 0.05
# One-sided level of every decision, so a battery decision agrees with its
# cells: the cell verdicts read the two-sided 95% interval, i.e. each side
# at 0.025. "TRANSFERS" needs a Holm-adjusted positive p below it, hence at
# least one POSITIVE cell; "NO TRANSFER" needs every attack's upper bound
# below delta (one-sided p_up below it). The two-sided equivalence of every
# attack is reported next to it.
ALPHA = 0.025
# Primary rule (docs/tifs_plan.md §0.3, frozen before any v2 Stage 2
# output). A structural null is a (attack, dataset) whose teacher pair
# differs only in the input embeddings of tokens that no clean-corpus row
# contains (scripts/tifs/structural_nulls.py, from the weights alone): the
# P and C students then train identically and k = 0 for any student seed.
# Such cells are decided on seed 42's Tango interval and reported apart.
STRUCTURAL_JSON = os.path.join(ROOT, "results", "tifs", "structural_nulls.json")
# The headline battery: logit KD. A cell with one student seed borrows the
# between-seed variance of its family's logit battery on the same dataset,
# else on SST-2 (the noisier dataset; AG News has no seed replicates).
PRIMARY_KD = "logit"
# The logit batteries the headline is planned over (shown if one is missing).
PLANNED_LOGIT = (("bert", "agnews"), ("bert", "sst2"), ("gpt2", "sst2"),
                 ("qwen3", "agnews"), ("qwen3", "sst2"))
RUN_RE = [
    # The battery runs are the clean-corpus cells of the grid; _s<k> marks a
    # student-seed replicate (teachers and corpus stay at seed 42).
    ("B", re.compile(r"^tifs_br?_(?P<fam>[a-z0-9]+)_(?P<bds>[a-z0-9]+)(?:_s(?P<sseed>\d+))?$")),
    ("G", re.compile(r"^tifs_g_(?P<fam>[a-z0-9]+)_(?P<corpus>[a-z0-9]+)_s(?P<seed>\d+)$")),
    ("H", re.compile(r"^tifs_h_(?P<fam>[a-z0-9]+)_(?P<corpus>[a-z0-9]+)_s(?P<seed>\d+)$")),
    ("S", re.compile(r"^tifs_s_(?P<fam>[a-z0-9]+)_(?P<corpus>[a-z0-9]+)_s(?P<seed>\d+)$")),
    ("L", re.compile(r"^tifs_l_(?P<fam>qwen8b)_(?P<corpus>[a-z0-9]+)_s(?P<seed>\d+)$")),
    ("AA", re.compile(r"^tifs_aa_(?P<fam>[a-z0-9]+)_(?P<corpus>[a-z0-9]+)_t(?P<tseed>\d+)$")),
    ("PC", re.compile(r"^tifs_pc_(?P<fam>[a-z0-9]+)_(?P<variant>[a-z0-9]+)$")),
    ("CC", re.compile(r"^tifs_cc_(?P<fam>[a-z0-9]+)_(?P<corpus>[a-z0-9]+)$")),
]


def parse_run(run_id: str) -> dict | None:
    for block, rx in RUN_RE:
        m = rx.match(run_id)
        if m:
            d = {"block": block, **m.groupdict()}
            if block == "PC":
                d["corpus"] = d.pop("variant")
            if block == "B":
                d.pop("bds")
                d.update(block="G", corpus="clean", seed=d.pop("sseed") or "42")
            return d
    return None


def load_cells(runs_dir: str) -> list[dict]:
    cells = []
    for preds_path in sorted(glob.glob(os.path.join(runs_dir, "tifs_*", "artifacts", "results", "*.preds.json"))):
        run_id = preds_path.split(os.sep)[-4]
        info = parse_run(run_id)
        if info is None:
            continue
        res_path = preds_path[: -len(".preds.json")] + ".json"
        if not os.path.exists(res_path):
            continue
        p = json.load(open(preds_path))
        r = json.load(open(res_path))
        if (p.get("meta") or {}).get("schema") != 2:
            continue
        cells.append({
            "run": run_id, **info,
            "dataset": r["dataset"], "attack": r["attack_method"],
            "kd": r["kd_type"], "seed": int(r.get("seed", -1)),
            "teacher_seed": int(r.get("teacher_seed", -1)),
            "corpus_seed": int(r.get("corpus_seed", -1)),
            "student_seed": int(r.get("student_seed", -1)),
            "ca_p": r.get("ca_poisoned"), "ca_c": r.get("ca_clean"),
            "teacher_asr": r.get("teacher_asr"), "teacher_asr_null": r.get("teacher_asr_null"),
            "teacher_ca": r.get("teacher_ca"), "teacher_ca_null": r.get("teacher_ca_null"),
            "single": bool(p["meta"].get("single_student")),
            "P": p["preds_obs"], "C": p["preds_null"],
            "Pc": p.get("preds_p_clean"), "Cc": p.get("preds_c_clean"),
            "TP": p.get("teacher_p_trig"), "TC": p.get("teacher_c_trig"),
            "controls": p.get("controls") or {},
            "test_set": p["meta"].get("test_set", "default"),
        })
    return cells


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


KEY = ("block", "fam", "dataset", "attack", "kd", "corpus", "seed")


def cell_table(cells, R):
    rows = []
    for c in cells:
        if c["single"]:
            continue
        row = {k: c[k] for k in ("run", *KEY, "teacher_seed", "corpus_seed", "student_seed")}
        row.update(paired_contrast(c["P"], c["C"], TOLS))
        row.update(ca_p=c["ca_p"], ca_c=c["ca_c"], teacher_asr=c["teacher_asr"],
                   teacher_asr_null=c["teacher_asr_null"], teacher_ca=c["teacher_ca"],
                   teacher_ca_null=c["teacher_ca_null"])
        if c["Pc"] and c["Cc"]:
            row.update(joint_bootstrap(
                {"P": c["P"], "C": c["C"], "Pc": c["Pc"], "Cc": c["Cc"]},
                {"delta_b": {"Pc": 1, "Cc": -1},
                 "k_star": {"P": 1, "Pc": -1, "C": -1, "Cc": 1}}, R=R))
        row["teacher_qualified"] = qualified(c)
        rows.append(row)
    return rows


def qualified(c, min_asr=GATE_ASR, max_drop=GATE_CA_DROP) -> bool:
    asr, ca, ca0 = c["teacher_asr"], c["teacher_ca"], c["teacher_ca_null"]
    if asr is None or ca is None or ca0 is None:
        return False
    # The tolerance keeps an exact 5-point drop (e.g. 380 of 7,600) inside.
    return asr >= min_asr and ca0 - ca <= max_drop + 1e-9


def audit_table(cells, R):
    """Raw student ASR vs the audited, teacher-attributable effect."""
    ref = reference_index(cells)
    rows = []
    for c in cells:
        if c["single"] or c["block"] != "G" or c["corpus"] != "clean":
            continue
        r = ref.get((c["block"], c["fam"], c["dataset"], c["attack"], c["corpus"], c["seed"]))
        pc = paired_contrast(c["P"], c["C"], TOLS)
        row = {k: c[k] for k in ("fam", "dataset", "attack", "kd", "seed")}
        row.update(teacher_asr=c["teacher_asr"], teacher_asr_null=c["teacher_asr_null"],
                   teacher_qualified=qualified(c), n=pc["n"],
                   raw_asr=pc["a"], C_trig=pc["b"], R_trig=mean(r["P"]) if r else None,
                   k=pc["k"], k_lo=pc["lo"], k_hi=pc["hi"], delta_star=pc["delta_star"],
                   attributable_share=pc["k"] / pc["a"] if pc["a"] else None,
                   **{f"verdict@{t:g}": pc[f"verdict@{t:g}"] for t in TOLS},
                   **{f"p_up@{t:g}": pc[f"p_up@{t:g}"] for t in TOLS},
                   **{f"p_tost@{t:g}": pc[f"p_tost@{t:g}"] for t in TOLS},
                   **{f"p_pos@{t:g}": pc[f"p_pos@{t:g}"] for t in TOLS})
        if c["Pc"] and c["Cc"]:
            jb = joint_bootstrap({"P": c["P"], "C": c["C"], "Pc": c["Pc"], "Cc": c["Cc"]},
                                 {"k_star": {"P": 1, "Pc": -1, "C": -1, "Cc": 1}}, R=R)
            row.update(k_star=jb["k_star"], k_star_lo=jb["k_star_lo"], k_star_hi=jb["k_star_hi"])
        rows.append(row)
    return rows


def decide(rs):
    """Battery decision at each tolerance over qualified rows ``rs``."""
    out = {}
    for t in TOLS:
        p_pos = holm([r[f"p_pos@{t:g}"] for r in rs])
        transfers = [r["attack"] for r, p in zip(rs, p_pos) if p < ALPHA]
        below = max(r[f"p_up@{t:g}"] for r in rs) < ALPHA
        out[f"decision@{t:g}"] = ("TRANSFERS" if transfers else
                                  "NO TRANSFER" if below else "INCONCLUSIVE")
        out[f"transfers@{t:g}"] = ",".join(transfers)
        out[f"all_equivalent@{t:g}"] = max(r[f"p_tost@{t:g}"] for r in rs) < ALPHA
    return out


def battery_table(audit_rows, by_seed=True):
    """One decision per (family, dataset, KD type[, seed]) over its
    qualified attacks.

    Unqualified attacks are listed, not decided on; a group whose attacks
    all fail the teacher gate is reported as NO QUALIFIED ATTACK.
    """
    groups = defaultdict(list)
    for r in audit_rows:
        if r["kd"] != "ce":
            key = (r["fam"], r["dataset"], r["kd"]) + ((r["seed"],) if by_seed else ())
            groups[key].append(r)
    rows = []
    for key, group in sorted(groups.items()):
        fam, ds, kd = key[:3]
        rs = [r for r in group if r["teacher_qualified"]]
        row = {"fam": fam, "dataset": ds, "kd": kd,
               **({"seed": key[3]} if by_seed else {}),
               "n_attacks": len(group), "n_qualified": len(rs),
               "attacks": ",".join(r["attack"] for r in rs),
               "unqualified": ",".join(r["attack"] for r in group if not r["teacher_qualified"])}
        if not rs:
            row.update({f"decision@{t:g}": "NO QUALIFIED ATTACK" for t in TOLS})
            rows.append(row)
            continue
        row.update(max_raw_asr=max(r["raw_asr"] for r in rs),
                   max_k=max(r["k"] for r in rs), max_k_hi=max(r["k_hi"] for r in rs))
        row.update(decide(rs))
        rows.append(row)
    return rows


def pool_seed_variance(ks_by_attack: dict) -> dict | None:
    """Between-seed variance of k pooled over a battery's attacks.

    ``ks_by_attack``: {attack: {seed: k_s}}. Uses the seeds every attack
    has (a balanced attack x seed table). The pooled variance is the mean
    within-attack variance across seeds, (MS_seed + (m - 1) MS_res) / m.
    The attacks of a seed share its C student, so the seed effect carries
    only r - 1 df; Satterthwaite's df for that combination lies between
    r - 1 (all noise shared) and m (r - 1) (none shared).
    """
    if not ks_by_attack:
        return None
    seeds = sorted(set.intersection(*(set(v) for v in ks_by_attack.values())))
    attacks = sorted(ks_by_attack)
    m, r = len(attacks), len(seeds)
    if r < 2:
        return None
    K = [[ks_by_attack[a][s] for s in seeds] for a in attacks]
    row = [mean(x) for x in K]
    col = [mean(c) for c in zip(*K)]
    g = mean(row)
    ms_seed = m * sum((c - g) ** 2 for c in col) / (r - 1)
    ms_res = (sum((K[i][j] - row[i] - col[j] + g) ** 2 for i in range(m) for j in range(r))
              / ((m - 1) * (r - 1))) if m > 1 else 0.0
    comb = ms_seed + (m - 1) * ms_res
    denom = ms_seed ** 2 + (m - 1) * ms_res ** 2
    return {"s2": comb / m, "df": comb ** 2 * (r - 1) / denom if denom > 0 else m * (r - 1),
            "attacks": attacks, "seeds": seeds, "ms_seed": ms_seed, "ms_res": ms_res}


def load_pairs(path=STRUCTURAL_JSON) -> tuple[frozenset, frozenset]:
    """(structural nulls, embedding-only pairs) as {(attack, dataset)} sets,
    from scripts/tifs/structural_nulls.py; both empty if not measured. An
    embedding-only pair differs only in some tokens' input embeddings: its
    students differ only through the corpus rows that contain them, so its
    between-seed variance is not that of a full teacher swap and it stays
    out of the pooled variance."""
    if not os.path.exists(path):
        return frozenset(), frozenset()
    pairs = json.load(open(path))["pairs"]
    return (frozenset((p["attack"], p["dataset"]) for p in pairs if p.get("structural_null")),
            frozenset((p["attack"], p["dataset"]) for p in pairs if p.get("only_input_embeddings")))


def load_structural(path=STRUCTURAL_JSON) -> frozenset:
    """{(attack, dataset)} of the structural nulls; empty if not measured."""
    return load_pairs(path)[0]


def reference_variance(cells) -> dict:
    """Between-teacher-seed variance of k for BadNets, logit KD, clean
    corpus, student seed 42: the battery's teacher-42 cell and the A/A
    runs' teacher-43/44 cells. {(family, dataset): {s2, df, teacher_seeds}}.
    It also contains student-training and input noise, so adding it to a
    primary SE over-covers (a sensitivity bound)."""
    ks = defaultdict(dict)
    for c in cells:
        if (c["single"] or c["kd"] != PRIMARY_KD or c["corpus"] != "clean"
                or c["attack"] != "badnets" or c["student_seed"] != 42):
            continue
        if (c["block"] == "G" and c["teacher_seed"] == 42) or c["block"] == "AA":
            ks[(c["fam"], c["dataset"])][c["teacher_seed"]] = mean(c["P"]) - mean(c["C"])
    out = {}
    for key, by_t in ks.items():
        if len(by_t) >= 2:
            v = list(by_t.values())
            out[key] = {"s2": sum((x - mean(v)) ** 2 for x in v) / (len(v) - 1),
                        "df": len(v) - 1, "teacher_seeds": sorted(by_t)}
    return out


def _tango_fields(pc):
    return {**{k: pc[k] for k in ("k", "lo", "hi", "delta_star")},
            **{f"{p}@{t:g}": pc[f"{p}@{t:g}"]
               for p in ("verdict", "p_up", "p_tost", "p_pos") for t in TOLS}}


def primary_table(cells, structural=frozenset(), near_null=frozenset(),
                  gate=(GATE_ASR, GATE_CA_DROP), ref_var=None):
    """Primary estimate per clean-corpus battery cell (§0.3).

    Returns (rows, variance rows). For attack a of a battery (family,
    dataset, KD type) with r >= 2 student seeds, d_s = P_s - C_s on the
    same n inputs:

        k = mean_s k_s,  SE^2 = s2_pool / r + var_i(mean_s d_s,i) / n

    s2_pool: the between-seed variance of k_s pooled over the battery's
    replicated, qualified attacks (structural nulls and embedding-only pairs
    left out), with Satterthwaite df (``pool_seed_variance``). A one-seed
    cell adds the pool to its Tango SE (``transported``). The pool comes
    from the cell's own battery, else the family's logit battery on the
    same dataset, else on SST-2, else the largest SST-2 logit pool of any
    family. A cell with no pool at all cannot be decided (p = 1); the
    structural nulls use seed 42's Tango interval.

    Sensitivity columns: ``*_df_min`` (df = r - 1, every seed effect
    shared) and ``robust_*`` (the A/A between-teacher-seed variance added,
    ``reference_variance``).
    """
    ref_var = ref_var or {}
    base = [c for c in cells if c["block"] == "G" and c["corpus"] == "clean"
            and not c["single"] and c["kd"] != "ce"]
    odd = sorted({c["run"] for c in base if c["teacher_seed"] != 42})
    if odd:
        raise ValueError(f"clean-corpus battery cells with teacher seed != 42: {odd}")
    groups = defaultdict(dict)
    for c in base:
        key = (c["fam"], c["dataset"], c["kd"], c["attack"])
        if c["student_seed"] in groups[key]:
            raise ValueError(f"{key}: two cells for student seed {c['student_seed']}")
        groups[key][c["student_seed"]] = c
    stats = {}
    for key, by_seed in groups.items():
        seeds = sorted(by_seed)
        n = len(by_seed[seeds[0]]["P"])
        if (len({by_seed[s].get("test_set") for s in seeds}) > 1
                or any(len(by_seed[s]["P"]) != n or len(by_seed[s]["C"]) != n for s in seeds)):
            raise ValueError(f"{key}: seeds evaluated on different inputs")
        d = [[p - q for p, q in zip(by_seed[s]["P"], by_seed[s]["C"])] for s in seeds]
        ks = [mean(v) for v in d]
        dbar = [mean(col) for col in zip(*d)]
        kbar = mean(ks)
        stats[key] = {"seeds": seeds, "n": n, "ks": ks, "k": kbar,
                      "v": sum((x - kbar) ** 2 for x in dbar) / (n - 1) / n,
                      "s2": (sum((x - kbar) ** 2 for x in ks) / (len(ks) - 1)
                             if len(ks) > 1 else None),
                      "qualified": qualified(by_seed[seeds[0]], *gate),
                      "cell": by_seed[seeds[0]]}
    members = defaultdict(dict)
    for (fam, ds, kd, atk), st in stats.items():
        if (st["s2"] is not None and st["qualified"] and (atk, ds) not in structural
                and (atk, ds) not in near_null):
            members[(fam, ds, kd)][atk] = dict(zip(st["seeds"], st["ks"]))
    pools = {}
    for key, ks_by_attack in members.items():
        pool = pool_seed_variance(ks_by_attack)
        if pool is not None:
            pools[key] = pool
    var_rows = []
    for (fam, ds, kd), pool in sorted(pools.items()):
        var_rows.append({"fam": fam, "dataset": ds, "kd": kd, "attack": "(pooled)",
                         "df": pool["df"], "s_between": pool["s2"] ** 0.5,
                         "ms_seed": pool["ms_seed"], "ms_res": pool["ms_res"],
                         "attacks": ",".join(pool["attacks"]),
                         "seeds": ",".join(str(s) for s in pool["seeds"])})
    for (fam, ds, kd, atk), st in sorted(stats.items()):
        if st["s2"] is not None:
            var_rows.append({"fam": fam, "dataset": ds, "kd": kd, "attack": atk,
                             "df": len(st["ks"]) - 1, "s_between": st["s2"] ** 0.5,
                             "input_se": st["v"] ** 0.5,
                             **{f"k_s{s}": k for s, k in zip(st["seeds"], st["ks"])}})

    def source_for(fam, ds, kd):
        for key in ((fam, ds, kd), (fam, ds, PRIMARY_KD), (fam, "sst2", PRIMARY_KD)):
            if key in pools:
                return key
        others = [k for k in pools if k[1] == "sst2" and k[2] == PRIMARY_KD]
        return max(others, key=lambda k: pools[k]["s2"]) if others else None

    rows = []
    for (fam, ds, kd, atk), st in sorted(stats.items()):
        c = st["cell"]
        by_seed = groups[(fam, ds, kd, atk)]
        r = len(st["seeds"])
        row = {"fam": fam, "dataset": ds, "kd": kd, "attack": atk,
               "seeds": ",".join(str(s) for s in st["seeds"]), "n": st["n"],
               "teacher_asr": c["teacher_asr"], "teacher_asr_null": c["teacher_asr_null"],
               "teacher_qualified": st["qualified"], "structural_null": (atk, ds) in structural,
               "embedding_only_pair": (atk, ds) in near_null,
               "raw_asr": mean([mean(by_seed[s]["P"]) for s in st["seeds"]]),
               "C_trig": mean([mean(by_seed[s]["C"]) for s in st["seeds"]])}
        tango = paired_contrast((by_seed.get(42) or c)["P"], (by_seed.get(42) or c)["C"], TOLS)
        src = source_for(fam, ds, kd)
        if (atk, ds) in structural:
            row["method"] = "structural"
            row.update(_tango_fields(tango))
            rows.append(row)
            continue
        if src is None:
            # No between-seed variance anywhere: shown, never decided.
            row["method"] = "no variance source"
            row.update(_tango_fields(tango))
            row.update({f"{p}@{t:g}": 1.0 for p in ("p_up", "p_tost", "p_pos") for t in TOLS})
            row.update({f"verdict@{t:g}": "UNDECIDED" for t in TOLS})
            rows.append(row)
            continue
        pool = pools[src]
        row["variance_from"] = "/".join(src)
        if r >= 2:
            est, se = st["k"], (pool["s2"] / r + st["v"]) ** 0.5
            row["method"] = "seed-pooled" if src == (fam, ds, kd) else "seed-pooled, borrowed variance"
        else:
            est = tango["k"]
            se = (se_from_interval(tango["lo"], tango["hi"]) ** 2 + pool["s2"]) ** 0.5
            row["method"] = "transported"
        if se == 0:
            row["method"] += ", degenerate: Tango"
            row.update(_tango_fields(tango))
            rows.append(row)
            continue
        row.update(t_contrast(est, se, pool["df"], TOLS))
        if r >= 2:
            cons = t_contrast(est, se, r - 1, (0.05,))
            row.update(lo_df_min=cons["lo"], hi_df_min=cons["hi"],
                       **{"verdict_df_min@0.05": cons["verdict@0.05"]})
        ref = ref_var.get((fam, ds))
        ref_from = f"{fam}/{ds}"
        if ref is None:
            same_ds = [k for k in ref_var if k[1] == ds]
            if same_ds:
                k_max = max(same_ds, key=lambda k: ref_var[k]["s2"])
                ref, ref_from = ref_var[k_max], f"{k_max[0]}/{k_max[1]}"
        if ref is not None:
            rob = t_contrast(est, (se ** 2 + ref["s2"]) ** 0.5, pool["df"], (0.05,))
            row.update(robust_lo=rob["lo"], robust_hi=rob["hi"], robust_from=ref_from,
                       **{"robust_verdict@0.05": rob["verdict@0.05"]})
        rows.append(row)
    return rows, var_rows


def primary_battery(primary_rows):
    """The primary decisions: per (family, dataset, KD type) battery, Holm
    within the battery; ``transfers_global`` applies Holm over the qualified
    cells of every battery at once."""
    rows = battery_table([dict(r, k_hi=r["hi"]) for r in primary_rows], by_seed=False)
    qual = [r for r in primary_rows if r["teacher_qualified"]]
    for t in TOLS:
        p_adj = holm([r[f"p_pos@{t:g}"] for r in qual])
        hits = defaultdict(list)
        for r, p in zip(qual, p_adj):
            if p < ALPHA:
                hits[(r["fam"], r["dataset"], r["kd"])].append(r["attack"])
        for b in rows:
            b[f"transfers_global@{t:g}"] = ",".join(hits.get((b["fam"], b["dataset"], b["kd"]), []))
    for b in rows:
        b["methods"] = ",".join(sorted({r["method"] for r in primary_rows
                                        if (r["fam"], r["dataset"], r["kd"]) == (b["fam"], b["dataset"], b["kd"])}))
    return rows


GATE_GRID = ((0.80, 0.85, 0.90, 0.95), (0.02, 0.05, 0.10))


def gate_sensitivity(cells, structural, near_null, ref_var):
    """Primary battery decisions under other teacher-gate thresholds."""
    out = []
    for asr in GATE_GRID[0]:
        for drop in GATE_GRID[1]:
            rows, _ = primary_table(cells, structural, near_null, (asr, drop), ref_var)
            for b in primary_battery(rows):
                out.append({"gate_asr": asr, "gate_ca_drop": drop,
                            **{k: b[k] for k in ("fam", "dataset", "kd", "n_attacks", "n_qualified")},
                            **{f"decision@{t:g}": b[f"decision@{t:g}"] for t in TOLS}})
    return out


def secondary_table(cells, var_rows):
    """Contaminated-corpus and positive-control cells (seed 42): the Tango
    interval these blocks are read with, and the transported interval (the
    family's logit pool on the dataset, else on SST-2) as a sensitivity
    check. The positive-control pass criterion uses Tango (§0.3)."""
    pools = {(v["fam"], v["dataset"], v["kd"]): v for v in var_rows if v["attack"] == "(pooled)"}
    rows = []
    for c in cells:
        if c["single"] or c["kd"] == "ce":
            continue
        if not ((c["block"] == "G" and c["corpus"] != "clean") or c["block"] == "PC"):
            continue
        pc = paired_contrast(c["P"], c["C"], TOLS)
        row = {k: c[k] for k in ("run", *KEY)}
        row.update(_tango_fields(pc))
        src = next((pools[k] for k in ((c["fam"], c["dataset"], PRIMARY_KD),
                                       (c["fam"], "sst2", PRIMARY_KD)) if k in pools), None)
        if src is not None:
            tr = t_contrast(pc["k"], (se_from_interval(pc["lo"], pc["hi"]) ** 2
                                      + src["s_between"] ** 2) ** 0.5, src["df"], TOLS)
            row.update(transported_lo=tr["lo"], transported_hi=tr["hi"],
                       variance_from=f"{src['fam']}/{src['dataset']}/{src['kd']}",
                       **{f"transported_verdict@{t:g}": tr[f"verdict@{t:g}"] for t in TOLS})
        rows.append(row)
    return rows


def reference_index(cells):
    """CE-only reference R by (block, fam, dataset, attack, corpus, seed)."""
    idx = {}
    for c in cells:
        if c["kd"] == "ce":
            idx[(c["block"], c["fam"], c["dataset"], c["attack"], c["corpus"], c["seed"])] = c
    return idx


def factorial_table(cells):
    ref = reference_index(cells)
    rows = []
    for c in cells:
        if c["single"]:
            continue
        r = ref.get((c["block"], c["fam"], c["dataset"], c["attack"], c["corpus"], c["seed"]))
        if r is None:
            continue
        row = {k: c[k] for k in ("run", *KEY)}
        row.update(P=mean(c["P"]), C=mean(c["C"]), R=mean(r["P"]))
        for name, (a, b) in {"P-C": (c["P"], c["C"]), "C-R": (c["C"], r["P"]),
                             "P-R": (c["P"], r["P"])}.items():
            pc = paired_contrast(a, b, (0.05,))
            row.update({f"{name}": pc["k"], f"{name}_lo": pc["lo"], f"{name}_hi": pc["hi"],
                        f"{name}_verdict@0.05": pc["verdict@0.05"]})
        rows.append(row)
    return rows


def interaction_table(cells, R):
    by = {(c["block"], c["fam"], c["dataset"], c["attack"], c["kd"], c["corpus"], c["seed"]): c
          for c in cells}
    ref = reference_index(cells)
    rows = []
    for (block, fam, ds, atk, kd, corpus, seed), c1 in by.items():
        if c1["single"] or corpus not in ("badnets10", "addsent10", "synbkd10"):
            continue
        c0 = by.get((block, fam, ds, atk, kd, "clean", seed))
        r1 = ref.get((block, fam, ds, atk, corpus, seed))
        r0 = ref.get((block, fam, ds, atk, "clean", seed))
        if not (c0 and r1 and r0):
            continue
        vec = {"P1": c1["P"], "C1": c1["C"], "R1": r1["P"],
               "P0": c0["P"], "C0": c0["C"], "R0": r0["P"]}
        row = {"block": block, "fam": fam, "dataset": ds, "attack": atk, "kd": kd,
               "corpus": corpus, "seed": seed, **{k: mean(v) for k, v in vec.items()}}
        row.update(joint_bootstrap(vec, {
            "teacher_x_corpus": {"P1": 1, "C1": -1, "P0": -1, "C0": 1},
            "suppression_C1_R1": {"C1": 1, "R1": -1},
            "amplification_P1_R1": {"P1": 1, "R1": -1},
            "corpus_effect_R1_R0": {"R1": 1, "R0": -1},
        }, R=R))
        rows.append(row)
    return rows


def seeds_table(cells):
    groups = defaultdict(dict)
    for c in cells:
        if c["single"] or c["block"] not in ("G", "H", "S", "L"):
            continue
        groups[(c["block"], c["fam"], c["dataset"], c["attack"], c["kd"], c["corpus"])][c["seed"]] = c
    rows = []
    for key, by_seed in sorted(groups.items()):
        per = {s: paired_contrast(c["P"], c["C"], TOLS) for s, c in sorted(by_seed.items())}
        row = dict(zip(("block", "fam", "dataset", "attack", "kd", "corpus"), key))
        row["seeds"] = ",".join(str(s) for s in per)
        for s, pc in per.items():
            row[f"k_s{s}"] = pc["k"]
            row[f"verdict@0.05_s{s}"] = pc["verdict@0.05"]
        if len(per) >= 2:
            ks = [pc["k"] for pc in per.values()]
            ses = [se_from_interval(pc["lo"], pc["hi"]) for pc in per.values()]
            re_ = random_effects_hk(ks, ses)
            row.update({f"re_{k}": v for k, v in re_.items()})
            mhk = random_effects_hk(ks, ses, floor=True)
            row.update(mhk_lo=mhk["lo"], mhk_hi=mhk["hi"])
            for tol in TOLS:
                inside = -tol <= re_["lo"] and re_["hi"] <= tol
                row[f"re_verdict@{tol:g}"] = ("EQUIVALENT" if inside else
                                             "POSITIVE" if re_["lo"] > tol else
                                             "NEGATIVE" if re_["hi"] < -tol else "INCONCLUSIVE")
        rows.append(row)
    return rows


def matching_table(cells):
    groups = defaultdict(dict)
    for c in cells:
        if c["single"] or c["block"] not in ("G", "H", "S"):
            continue
        groups[(c["block"], c["fam"], c["dataset"], c["attack"], c["kd"], c["corpus"])][c["seed"]] = c
    rows = []
    for key, by_seed in sorted(groups.items()):
        seeds = sorted(by_seed)
        if len(seeds) < 2:
            continue
        matched = [mean(by_seed[s]["P"]) - mean(by_seed[s]["C"]) for s in seeds]
        unmatched = [mean(by_seed[s]["P"]) - mean(by_seed[t]["C"])
                     for s, t in itertools.permutations(seeds, 2)]
        sd = lambda v: (sum((x - mean(v)) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else float("nan")
        rows.append({**dict(zip(("block", "fam", "dataset", "attack", "kd", "corpus"), key)),
                     "n_seeds": len(seeds), "matched_mean": mean(matched), "matched_sd": sd(matched),
                     "unmatched_mean": mean(unmatched), "unmatched_sd": sd(unmatched)})
    return rows


def aa_table(cells):
    """E2: seed-42 grid (teachers 42) vs runs with teacher seed 43/44."""
    base = {(c["fam"], c["dataset"], c["attack"], c["kd"], c["corpus"]): c
            for c in cells if c["block"] == "G" and c["seed"] == 42 and not c["single"]}
    rows = []
    for c in cells:
        if c["block"] != "AA" or c["single"]:
            continue
        b = base.get((c["fam"], c["dataset"], c["attack"], c["kd"], c["corpus"]))
        if b is None:
            continue
        row = {"fam": c["fam"], "dataset": c["dataset"], "attack": c["attack"], "kd": c["kd"],
               "corpus": c["corpus"], "teacher_seed": c["teacher_seed"]}
        pairs = {"C-C'": (b["C"], c["C"]), "P-C'": (b["P"], c["C"]),
                 "P'-C'": (c["P"], c["C"]), "P-P'": (b["P"], c["P"])}
        # ATBA's operating point: the seed-42 C student on the ATBA-triggered
        # inputs vs this run's C' on the same phrase (prefix, deterministic).
        atba = base.get((c["fam"], c["dataset"], "atba", c["kd"], "clean"))
        ctl = c["controls"].get(f"atba_phrase_{c['dataset']}")
        if atba is not None and ctl is not None and c["corpus"] == "clean":
            pairs["C-C'@atba"] = (atba["C"], ctl["c"])
        for name, (x, y) in pairs.items():
            pc = paired_contrast(x, y, TOLS)
            row.update({name: pc["k"], f"{name}_lo": pc["lo"], f"{name}_hi": pc["hi"],
                        **{f"{name}_verdict@{t:g}": pc[f"verdict@{t:g}"] for t in TOLS}})
        rows.append(row)
    return rows


def controls_table(cells):
    rows = []
    for c in cells:
        if c["single"]:
            continue
        planted = paired_contrast(c["P"], c["C"], (0.05,))
        for name, ctl in c["controls"].items():
            pc = paired_contrast(ctl["p"], ctl["c"], (0.05,))
            rows.append({k: c[k] for k in ("run", *KEY)} | {
                "control": name, "mode": ctl["spec"].get("mode", "insert"),
                "fallbacks": ctl["spec"].get("fallbacks"),
                "planted_k": planted["k"], "P_ctrl": pc["a"], "C_ctrl": pc["b"],
                "k_ctrl": pc["k"], "lo": pc["lo"], "hi": pc["hi"],
                "verdict@0.05": pc["verdict@0.05"],
                "teacher_P_ctrl": mean(ctl.get("teacher_p") or []),
                "teacher_C_ctrl": mean(ctl.get("teacher_c") or []),
            })
    return rows


def summary(cells_rows, seeds_rows, out_md, battery_rows=(), primary_rows=()):
    lines = ["# TIFS results summary (auto-generated)", ""]
    if battery_rows:
        lines += ["## Battery decisions (primary rule, qualified attacks only)", "",
                  "One decision per family x dataset x KD type over the attacks whose teacher "
                  "passed the gate (docs/tifs_plan.md §0.3): NO TRANSFER if every attack's "
                  "upper bound is below delta, TRANSFERS if any attack is above delta after "
                  "Holm (one-sided 0.025 for both). Cells with student seeds 42-44 use the "
                  "seed-pooled interval, one-seed cells the transported one.", "",
                  "| family | dataset | KD | qualified / attacks | @0.02 | @0.05 | @0.10 | not qualified |",
                  "|---|---|---|---|---|---|---|---|"]
        for b in battery_rows:
            lines.append(f"| {b['fam']} | {b['dataset']} | {b['kd']} | {b['n_qualified']} / {b['n_attacks']} | "
                         + " | ".join(b[f"decision@{t:g}"] for t in TOLS)
                         + f" | {b['unqualified'] or '-'} |")
        lines.append("")
    head = [r for r in primary_rows if r["kd"] == PRIMARY_KD and r["teacher_qualified"]]
    if head:
        p_max = max(r["p_up@0.05"] for r in head)
        present = sorted({(r["fam"], r["dataset"]) for r in head})
        lines += ["## Headline (intersection-union over every qualified logit-KD cell, δ=0.05)", "",
                  f"- {len(head)} cells; every upper bound below 0.05: "
                  f"**{'yes' if p_max < ALPHA else 'no'}** (max one-sided p = {p_max:.3g}).",
                  f"- batteries contributing: {', '.join(f'{f}/{d}' for f, d in present)}; "
                  f"planned: {', '.join(f'{f}/{d}' for f, d in PLANNED_LOGIT)}"
                  + ("" if set(PLANNED_LOGIT) <= set(present) else
                     f" — **missing: {', '.join(f'{f}/{d}' for f, d in PLANNED_LOGIT if (f, d) not in present)}**"),
                  f"- methods: {dict(Counter(r['method'] for r in head))}"
                  + ("" if any(r["method"].startswith("seed-pooled") for r in head) else
                     " — no student-seed replicate evaluated yet"), ""]
    groups = defaultdict(list)
    for r in cells_rows:
        if r["student_seed"] in (42, -1):
            groups[(r["block"], "clean" if r["corpus"] == "clean" else r["corpus"])].append(r)
    lines.append("## Per-cell verdicts (student seed 42 only, Tango, raw k; descriptive)")
    lines.append("")
    lines.append("| block | corpus | cells | @0.02 | @0.05 | @0.10 |")
    lines.append("|---|---|---|---|---|---|")
    for (block, corpus), rows in sorted(groups.items()):
        cnt = {t: Counter(r[f"verdict@{t:g}"] for r in rows) for t in TOLS}
        fmt = lambda c: ", ".join(f"{k[:5]} {v}" for k, v in sorted(c.items()))
        lines.append(f"| {block} | {corpus} | {len(rows)} | {fmt(cnt[0.02])} | {fmt(cnt[0.05])} | {fmt(cnt[0.10])} |")
    lines.append("")
    lines.append("## Multiplicity across blocks (student seed 42 only, Tango; descriptive)")
    lines.append("")
    for (block, corpus), rows in sorted(groups.items()):
        for tol in TOLS:
            p_tost = [r[f"p_tost@{tol:g}"] for r in rows]
            iut = max(p_tost) < ALPHA
            p_pos = holm([r[f"p_pos@{tol:g}"] for r in rows])
            lines.append(f"- {block}/{corpus} δ={tol:g}: all {len(rows)} cells equivalent "
                         f"(intersection-union, α={ALPHA:g} per side): **{'yes' if iut else 'no'}** "
                         f"(max TOST p = {max(p_tost):.3g}); cells with Holm-adjusted "
                         f"positive p < {ALPHA:g}: {sum(p < ALPHA for p in p_pos)}")
    if seeds_rows:
        lines += ["", "## Seed-pooled verdicts (S2, Hartung–Knapp)", ""]
        cnt = defaultdict(Counter)
        for r in seeds_rows:
            if "re_verdict@0.05" in r:
                cnt[(r["block"], "clean" if r["corpus"] == "clean" else "contaminated")][r["re_verdict@0.05"]] += 1
        for k, c in sorted(cnt.items()):
            lines.append(f"- {k[0]}/{k[1]} @0.05: {dict(c)}")
    open(out_md, "w").write("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=os.path.join(ROOT, "runs", "tifs"))
    p.add_argument("--out", default=os.path.join(ROOT, "results", "tifs", "report"))
    p.add_argument("--R", type=int, default=5000)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cells = load_cells(args.runs)
    print(f"{len(cells)} cells from {len({c['run'] for c in cells})} runs")
    audit = audit_table(cells, args.R)
    structural, near_null = load_pairs()
    print(f"structural nulls: {sorted(structural) or 'none measured'}; "
          f"embedding-only pairs (kept out of the pooled variance): {sorted(near_null) or 'none'}")
    ref_var = reference_variance(cells)
    primary, variance = primary_table(cells, structural, near_null, ref_var=ref_var)
    tables = {
        "primary": primary,
        "battery": primary_battery(primary),
        "variance": variance,
        "secondary": secondary_table(cells, variance),
        "gate_sensitivity": gate_sensitivity(cells, structural, near_null, ref_var),
        "reference_variance": [{"fam": f, "dataset": d, **v, "teacher_seeds": ",".join(map(str, v["teacher_seeds"]))}
                               for (f, d), v in sorted(ref_var.items())],
        "audit": audit,
        "battery_by_seed": battery_table(audit),
        "cells": cell_table(cells, args.R),
        "factorial": factorial_table(cells),
        "interaction": interaction_table(cells, args.R),
        "seeds": seeds_table(cells),
        "matching": matching_table(cells),
        "aa": aa_table(cells),
        "controls": controls_table(cells),
    }
    for name, rows in tables.items():
        write_csv(rows, os.path.join(args.out, f"{name}.csv"))
        print(f"  {name}: {len(rows)} rows")
    summary(tables["cells"], tables["seeds"], os.path.join(args.out, "summary.md"),
            tables["battery"], tables["primary"])


if __name__ == "__main__":
    main()
