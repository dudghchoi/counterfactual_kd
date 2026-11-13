#!/usr/bin/env python3
"""Tests of the A15 analysis code on synthetic fixtures (make_fixtures.py), BLIND to A15 outputs.

A15 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

Builds four fixture trees under --dir (identity, perturb with seeds 7 and 11, partial), runs run_all.sh on each (--no-score: the
CPU scorer is tested separately with score_em.py --only-recheck on a frozen checkpoint) and asserts:
  identity  every A15 vector = the frozen one, so: the epoch-matched frame = the frozen frame (all 31 rows, every
            column), the ws frame = the frozen frame, primary.csv reproduced through the b1 tables, alpha runs =
            the alpha 0.5 two-attack frame, the A15-c ATBA copies = the BadNets cell, A3.3 with the epoch-1
            references = reference_atba.csv (3 references) and g7-a13 g2_reference_atba5.csv (5), the item 7 rows
            of t42/t43/t44 = A14 item 7, the A2 frozen-pool row = replicate.csv, no label anywhere;
  perturb   the pipeline runs, every self-check and independent comparison passes, labels appear, cells outside
            the substituted batteries are unchanged, and the output README's label lists agree with the CSVs'
            sentence columns (registered and claim rule only; own / Welch changes under the sensitivity heading);
            run twice (fixture seeds 7 and 11);
  partial   run_all.sh stops with exit 4 before writing anything while a job is running before the deadline
            (--now one hour before it); the readiness check passes at the deadline;
            with --force: e2 reported missing (cut), d at r = 4 although d seed 46 has 3 of 6 attacks written by
            a running evaluation (files present, job not terminal-ok: excluded, listed in inventory.csv and every
            check file), the ws re-scores of running jobs left out (GPT-2 SST-2 cells keep the official vectors
            for every seed; no c ws reading), the missing ws run listed, a failed b1 seed (BERT SynBkd judged on
            seeds 42/43) and a b1 cell with every seed failed (GPT-2 AddSent: frozen C kept, sentence n/a), the
            README lists that cell, self-checks pass.
  every run the sentence columns (a_cells, b1_cells, e_cells) follow the registered and claim-rule labels only.

    python test_analysis.py --dir /tmp/.../a15prep/fixtures [--keep]
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402

import a15lib as L  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
TOLS = L.TOLS
RESULTS = []


def ok(name, cond, note=""):
    RESULTS.append((name, bool(cond), note))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({note})" if note and not cond else ""))


def rd(d, name):
    return L.read_csv(os.path.join(d, name))


def paired(row, a, b, cell_only=False):
    """Columns present under both prefixes whose values differ (exact text); ``cell_only`` leaves out the family
    (F31 / F19 Holm) columns, which move whenever another member of the family changes."""
    keys = [k[len(a):] for k in row if k.startswith(a)]
    if cell_only:
        keys = [k for k in keys if not k.startswith(("F31_", "F19_"))]
    return [k for k in keys if (b + k) in row and row[a + k] != row[b + k]]


def readme_list(out, head):
    """Cell ids listed under the README line starting with ``head`` (the '  - <cell> [rule]: ...' lines)."""
    lines = open(os.path.join(out, "README.md")).read().splitlines()
    got, on = set(), False
    for ln in lines:
        if ln.startswith(head):
            on = True
            continue
        if on:
            if ln.startswith("  - "):
                got.add(ln[4:].split(" [")[0])
            else:
                on = False
    return got


def readme_consistency(out):
    """The README's format- / epoch-dependent lists = the cells whose CSV sentence says so (registered and claim
    rule only); the sensitivity heading is present in the a and b1 sections."""
    for csvname, head, dep in (("a_cells.csv", "- format-dependent cells", "format-dependent"),
                               ("b1_cells.csv", "- epoch-dependent cells", "epoch-dependent")):
        rows = rd(out, csvname)
        want = {L.cid(r) for r in rows if r["sentence"].startswith(dep)}
        got = readme_list(out, head)
        ok(f"README {head[2:]} = {csvname} sentence '{dep}' cells", got == want,
           f"README only {sorted(got - want)}; CSV only {sorted(want - got)}")
        robust = {L.cid(r) for r in rows if r["sentence"] in ("삽입 형식에 강건", "teacher epoch에 강건")}
        ok(f"README {head[2:]}: no cell whose {csvname} sentence is the robust one", not (got & robust),
           str(sorted(got & robust)))
    text = open(os.path.join(out, "README.md")).read()
    ok("README has the sensitivity heading (own variance / Welch df; not a label) in a, b1, d and e",
       text.count("sensitivity changes (own variance / Welch df; not a label)") >= 4)
    # the sentence columns follow the registered and claim-rule labels only (never the own / Welch columns)
    for csvname, lab, robust, dep in (("a_cells.csv", "format_dependent", "삽입 형식에 강건", "format-dependent"),
                                      ("b1_cells.csv", "epoch_dependent", "teacher epoch에 강건", "epoch-dependent"),
                                      ("e_cells.csv", "recipe_dependent", "α 0.5–1.0에서 같다", "recipe-dependent")):
        bad = []
        for r in rd(out, csvname):
            s = r.get("sentence", "")
            if s.startswith("n/a") or f"{lab}_reg" not in r:
                continue
            want = dep if (r[f"{lab}_reg"] or r[f"{lab}_claim"]) else robust
            if not s.startswith(want):
                bad.append(f"{L.cid(r)}: {s} (reg {r[f'{lab}_reg']!r}, claim {r[f'{lab}_claim']!r})")
        ok(f"{csvname}: sentence = '{robust}' unless {lab}_reg or {lab}_claim is set (own / Welch ignored)", not bad,
           "; ".join(bad[:3]))


def sh(args, env=None):
    p = subprocess.run(args, cwd=HERE, capture_output=True, text=True, env=env)
    return p.returncode, p.stdout + p.stderr


def build(fx, mode, partial=False, seed=7, name=None):
    d = os.path.join(fx, name or ("partial" if partial else mode))
    if os.path.exists(d):
        shutil.rmtree(d)
    rc, log = sh([PY, "make_fixtures.py", "--dir", d, "--mode", mode, "--seed", str(seed)]
                 + (["--partial"] if partial else []))
    assert rc == 0, log
    return d


BEFORE_DEADLINE = "2026-10-29 04:00:00"   # UTC, one hour before the registered deadline (05:00 UTC)
AFTER_DEADLINE = "2026-10-29 05:00:01"


def run(d, force=False, now=None):
    args = ["bash", "run_all.sh", "--runs", f"{d}/runs", "--out", f"{d}/out", "--teachers-x15", f"{d}/teachers_x15",
            "--hits-dir", f"{d}/hits_em", "--no-score"] + (["--force"] if force else []) + (["--now", now] if now else [])
    return sh(args)


def test_identity(d):
    print("identity fixture")
    rc, log = run(d)
    ok("run_all.sh exit 0", rc == 0, log[-500:])
    out = os.path.join(d, "out")
    prim = {(r["fam"], r["dataset"], r["kd"], r["attack"]): r for r in L.read_csv(L.OFFICIAL["primary"])}
    b1 = rd(out, "b1_cells.csv")
    ok("b1: 31 rows", len(b1) == 31)
    diffs = [(r["attack"], paired(r, "em_", "frozen_")) for r in b1 if paired(r, "em_", "frozen_")]
    ok("b1: epoch-matched frame = frozen frame in every column of all 31 rows", not diffs, str(diffs[:3]))
    ok("b1: no epoch-dependent label", all(not r[f"epoch_dependent_{x}"] for r in b1 for x in ("reg", "own", "claim", "welch")))
    ok("b1: sentence 'teacher epoch에 강건' for every in-scope cell, n/a outside (no epoch-dependent)",
       all(r["sentence"] == "teacher epoch에 강건" or r["sentence"].startswith("n/a") for r in b1)
       and sum(r["sentence"] == "teacher epoch에 강건" for r in b1) >= 15
       and all(r["sentence"].startswith("n/a") for r in b1 if (r["fam"], r["dataset"]) == ("bert", "agnews")))
    nb = 0
    for r in b1:
        p = prim[(r["fam"], r["dataset"], r["kd"], r["attack"])]
        for f in ("k", "lo", "hi", "se", "df"):
            nb += r[f"frozen_{f}"] != p[f]
        for t in TOLS:
            nb += r[f"frozen_reg_verdict@{t:g}"] != p[f"verdict@{t:g}"]
            nb += r[f"frozen_reg_p_up@{t:g}"] != p[f"p_up@{t:g}"]
            nb += r[f"frozen_reg_p_pos@{t:g}"] != p[f"p_pos@{t:g}"]
    ok("b1: frozen_* columns reproduce primary.csv (k, lo, hi, se, df, verdicts, p_up, p_pos; exact text)", nb == 0,
       f"{nb} mismatches")
    nb = sum(r[f"em_{f}"] != prim[(r["fam"], r["dataset"], r["kd"], r["attack"])][f]
             for r in b1 for f in ("k", "lo", "hi"))
    ok("b1: the epoch-matched pipeline on frozen P and frozen C reproduces primary.csv (k, lo, hi)", nb == 0)
    bb = rd(out, "b1_battery.csv")
    ok("b1: battery decisions identical", all(not paired(r, "em_", "frozen_") for r in bb))
    bat = {(r["fam"], r["dataset"], r["kd"]): r for r in L.read_csv(L.OFFICIAL["battery"])}
    ok("b1: frozen battery decisions = battery.csv",
       all(r[f"frozen_reg_decision@{t:g}"] == bat[(r["fam"], r["dataset"], r["kd"])][f"decision@{t:g}"]
           for r in bb for t in TOLS))
    ok("b1: no Holm change", all(not r[f"epoch_dependent_{x}"] for r in rd(out, "b1_holm.csv")
                                 for x in L.FAMILY_READINGS))
    gh = {(r["fam"], r["dataset"], r["kd"], r["attack"]): r for r in L.read_csv(L.OFFICIAL["global_holm"])}
    readme_consistency(out)
    ok("b1: frozen holm_global = global_holm.csv",
       all(r[f"frozen_holm_global@{t:g}"] == gh[(r["fam"], r["dataset"], r["kd"], r["attack"])][f"holm_global@{t:g}"]
           for r in rd(out, "b1_holm.csv") for t in TOLS))
    sel = rd(out, "b1_selection.csv")
    ok("b1: Delta_sel = 0 for every registered cell", all(float(r["dsel"]) == 0.0 for r in sel))
    b3 = [r for r in rd(out, "b3_condition.csv") if r["kd"] == "logit"][0]
    ok("b3: identical C_e4 and C_e3 -> own bound 0, not triggered",
       float(b3["own_abs_upper95"]) == 0.0 and b3["b3_would_be_triggered"] == "False")
    a = rd(out, "a_cells.csv")
    ok("a: ws frame = frozen frame for the 7 decoder cells (every column)",
       len(a) == 7 and all(not paired(r, "ws_", "frozen_") for r in a))
    ok("a: sentence '삽입 형식에 강건' everywhere", all(r["sentence"] == "삽입 형식에 강건" for r in a))
    ok("a: no Holm change", all(not r[f"format_dependent_{x}"] for r in rd(out, "a_holm.csv") for x in L.FAMILY_READINGS))
    g1 = [r for r in rd(out, "a_g1.csv") if r.get("attack")]
    ok("a: G1 ws = official", all(not paired(r, "ws_", "official_") for r in g1))
    e = rd(out, "e_cells.csv")
    ee = [r for r in e if r["part"] == "e"]
    ok("e: alpha runs = alpha 0.5 two-attack frame (identity copies)",
       len(ee) == 4 and all(not paired(r, f"a{float(r['alpha']):g}_", "a0.5_twoattack_") for r in ee))
    e2 = [r for r in e if r["part"] == "e2"]
    ok("e: sentence from the like-for-like comparison (identity copies -> 'α 0.5–1.0에서 같다'), the registered-row "
       "comparison beside it", all(r["sentence"] == "α 0.5–1.0에서 같다" and not r["recipe_dependent_reg"]
                                    and not r["recipe_dependent_claim"] and "sentence_vs_registered" in r
                                    and "recipe_dependent_vs_registered_reg" in r for r in ee))
    ok("e2: alpha 1.0 run = epoch-matched alpha 0.5 frame (identity copies)",
       len(e2) == 2 and all(not paired(r, "a1_", "a0.5_em_") for r in e2) and all(not r["recipe_dependent_reg"] for r in e2))
    c = rd(out, "c_cells.csv")
    for reading in ("official", "em", "ws"):
        at = [r for r in c if r["reading"] == reading and r["attack"] == "atba"]
        bn = [r for r in c if r["reading"] == reading and r["attack"] == "badnets"]
        ok(f"c ({reading}): the ATBA copy has the BadNets k", at and bn and at[0]["k"] == bn[0]["k"])
    r33 = rd(out, "b2_reference_atba.csv")
    fz3 = {r["kd"]: r for r in L.read_csv(L.OFFICIAL["reference_atba"]) if r["dataset"] == "sst2"}
    fz5 = {r["kd"]: r for r in L.read_csv(os.path.join(L.G7, "g2_reference_atba5.csv")) if r["dataset"] == "sst2"}
    nb = 0
    for r in r33:
        if r["reference_set"].startswith("3"):
            f = fz3[r["kd"]]
            nb += sum(r[x] != f[x] for x in ("s_ref", "df_ref", "ref_lo", "ref_hi", "ref_df", "C_t42", "C_t43", "C_t44"))
            nb += sum(r[f"ref_verdict@{t:g}"] != f[f"ref_verdict@{t:g}"] for t in TOLS)
        else:
            f = fz5[r["kd"]]
            nb += sum(r[x] != f[y] for x, y in (("s_ref", "s_ref"), ("ref_lo", "ref5_lo"), ("ref_hi", "ref5_hi"),
                                                ("ref_df", "ref5_df"), ("C_t45", "C_t45"), ("C_t46", "C_t46")))
            nb += sum(r[f"ref_verdict@{t:g}"] != f[f"ref5_verdict@{t:g}"] for t in TOLS)
    ok("b2: A3.3 with the epoch-1 references = reference_atba.csv (3 refs) and g2_reference_atba5.csv (5 refs)",
       nb == 0 and len(r33) == 4, f"{nb} mismatches")
    g6 = {(r["kd"], r["trigger"], r["reference"]): r for r in L.read_csv(os.path.join(L.G6REF, "reference_k_all_triggers.csv"))
          if r["dataset"] == "sst2"}
    lab = {"t42": "t42 twin", "t43": "t43 C'", "t44": "t44 C'"}
    nb = nn = 0
    for r in rd(out, "b2_reference_k.csv"):
        if r["precision"] != "CPU fp32" or r["reference"] not in lab:
            continue
        o = g6[(r["kd"], r["trigger"], lab[r["reference"]])]
        nn += 1
        nb += sum(r[x] != o[y] for x, y in (("tango_k", "tango_k"), ("tango_lo", "tango_lo"), ("tango_hi", "tango_hi"),
                                           ("frozen_pool_lo", "transported_lo"), ("frozen_pool_hi", "transported_hi")))
        nb += sum(r[f"frozen_pool_verdict@{t:g}"] != o[f"transported_verdict@{t:g}"] for t in TOLS)
    ok("b2: item 7 rows of t42/t43/t44 (identity copies of the A14 hits) = A14 item 7", nb == 0 and nn == 24,
       f"{nn} rows, {nb} mismatches")
    a2 = [r for r in rd(out, "b2_a2.csv") if r["pool"].startswith("frozen")][0]
    rep = L.read_csv(L.OFFICIAL["replicate"])[0]
    nb = sum(a2[x] != rep[y] for x, y in (("k_em", "k"), ("se", "se"), ("lo", "lo"), ("hi", "hi"), ("own_lo", "own_lo"),
                                         ("own_hi", "own_hi"), ("build_diff_em", "build_diff"), ("build_diff_lo", "build_diff_lo"),
                                         ("build_diff_hi", "build_diff_hi")))
    nb += sum(a2[f"reg_verdict@{t:g}"] != rep[f"verdict@{t:g}"] or a2[f"reg_reading@{t:g}"] != rep[f"reading@{t:g}"]
              for t in TOLS)
    ok("b2: A2 frozen-pool row with C'(e5) := C'(e1) = replicate.csv", nb == 0, f"{nb} mismatches")
    c2 = {r["check"]: r for r in rd(out, "check_b2.csv")}
    t43 = [r for k, r in c2.items() if k.startswith("rebuilt t43 identical to frozen t43")]
    ok("b2: twin check t43 row (identical, selected epoch 1) True; twin rows t43-t46 in b2_references.csv",
       t43 and t43[0]["ok"] == "True" and
       sorted(r["teacher_seed"] for r in rd(out, "b2_references.csv") if r.get("run") == "(twin check)")
       == ["43", "44", "45", "46"])
    ok("b2: A3.3 rows carry the Welch-df sensitivity columns",
       all(r.get("ref_welch_df") and r.get("ref_welch_verdict@0.05") for r in r33))
    for f in ("a_g1.csv", "a_holm.csv", "b1_holm.csv", "d_holm.csv", "b2_reference_atba.csv"):
        rows = rd(out, f)
        ok(f"{f}: a15_jobs column", rows and all(r.get("a15_jobs", "").startswith("part ") for r in rows))
    sc = {r["check"]: r for r in rd(out, "check_b1.csv")}
    ok("b1: F31 family size of the em frame = 31", any(k.startswith("F31 family size of the em frame") and r["ok"] == "True"
                                                       for k, r in sc.items()))
    ok("b1: provenance check requires a recorded epoch", any(k.startswith("every b1 cell's C teacher epoch (provenance)")
                                                             and r["ok"] == "True" for k, r in sc.items()))
    st = {r["check"]: r for r in rd(out, "selfcheck.csv")}
    ok("selfcheck: RUN_RE identity and content after load_x15 / load_ws, restored after an exception, full digests",
       all(any(k.startswith(h) and r["ok"] == "True" for k, r in st.items())
           for h in ("analyze.RUN_RE is the object saved at import", "analyze.RUN_RE content (block, pattern)",
                     "analyze.RUN_RE restored after an exception"))
       and all(len(r["expected"]) == 64 for k, r in st.items() if k.endswith("== registered constant (full digest)"))
       and sum(k.endswith("== registered constant (full digest)") for k in st) == 7)
    ok("selfcheck: F31 family size rows of the ws, em and r5 frames; frozen pool_welch identity",
       all(any(k.startswith(f"F31 family size of the {x} frame = 31") and r["ok"] == "True" for k, r in st.items())
           for x in ("ws", "em", "r5"))
       and any(k.startswith("pool_welch identity") and r["ok"] == "True" for k, r in st.items()))
    cl = rd(out, "b1_claims.csv")
    fr = [r for r in cl if r["frame"] == "frozen"]
    em = [r for r in cl if r["frame"] == "em"]
    ok("claims: em counts = frozen counts", all({k: v for k, v in x.items() if k != "frame"}
                                                == {k: v for k, v in y.items() if k != "frame"} for x, y in zip(fr, em)))
    common(out)


def common(out):
    sc = rd(out, "selfcheck.csv")
    ok("selfcheck: every check ok", sc and all(r["ok"] == "True" for r in sc),
       "; ".join(r["check"] for r in sc if r["ok"] != "True"))
    ind = rd(out, "independent_check.csv")
    ok("independent: every comparison equal", ind and all(r["ok"] in ("True", "") for r in ind) and
       sum(r["ok"] == "True" for r in ind) > 50, f"{sum(r['ok'] == 'True' for r in ind)} equal")
    ok("README.md written with the status line", os.path.exists(os.path.join(out, "README.md")) and
       L.STATUS in open(os.path.join(out, "README.md")).read())
    text = open(os.path.join(out, "README.md")).read()
    ok("README data snapshot has the newest A15 input mtime", "newest A15 input read (mtime)" in text)
    for f in sorted(os.listdir(out)):
        if f.startswith("check_") and f.endswith(".csv"):
            ok(f"{f}: the '{L.EXCLUDED_TEXT}' row", any(L.EXCLUDED_TEXT in r["check"] for r in rd(out, f)))
    for f in ("b1_cells.csv", "a_cells.csv"):
        for r in rd(out, f):
            for pre in ("frozen_", "em_", "ws_"):
                for t in TOLS:
                    if r.get(f"{pre}method") == "structural" and r.get(f"{pre}claim@{t:g}") not in (None, ""):
                        if r[f"{pre}claim@{t:g}"] != "structural (counted apart)":
                            ok(f"{f}: structural claim label", False, L.cid(r))
    for f in os.listdir(out):
        if f.endswith(".csv"):
            with open(os.path.join(out, f)) as fh:
                if not fh.readline().startswith("# definition:"):
                    ok(f"{f} starts with '# definition:'", False)


def test_perturb(d):
    print("perturb fixture")
    rc, log = run(d)
    ok("run_all.sh exit 0", rc == 0, log[-500:])
    out = os.path.join(d, "out")
    b1 = rd(out, "b1_cells.csv")
    ok("b1: some epoch-dependent label", any(r["epoch_dependent_reg"] for r in b1))
    ok("b1: BERT AG News rows unchanged (cell-level columns)", all(not paired(r, "em_", "frozen_", True) for r in b1
                                                                   if (r["fam"], r["dataset"]) == ("bert", "agnews")))
    ok("b1: Qwen3 AG News marked pool-only", all(r["pool_only_change"] == "True" for r in b1
                                                 if (r["fam"], r["dataset"]) == ("qwen3", "agnews")))
    a = rd(out, "a_cells.csv")
    ok("a: k changes in the ws frame", any(r["ws_k"] != r["frozen_k"] for r in a))
    ok("a: BERT control unchanged", all(r["unchanged"] == "True" for r in rd(out, "a_bert_control.csv")))
    d_ = rd(out, "d_cells.csv")
    ok("d: logit rows at r = 5 with own df 4", all(r["r_r5"] == "5" and r["r5_own_df"] == "4" for r in d_ if r["kd"] == "logit"))
    ok("d: r = 5 labelled post hoc", all("post hoc" in r["label_r5"] for r in d_))
    readme_consistency(out)
    common(out)


def test_partial(d):
    print("partial fixture")
    rc, log = run(d, now=BEFORE_DEADLINE)
    ok("run_all.sh stops with exit 4 while a job runs before the deadline", rc == 4, log[-300:])
    ok("nothing written before the readiness check", not os.path.exists(os.path.join(d, "out")))
    rc, log = sh([PY, "inventory.py", "--check-ready", "--runs", f"{d}/runs", "--out", f"{d}/out", "--teachers-x15",
                  f"{d}/teachers_x15", "--hits-dir", f"{d}/hits_em", "--now", AFTER_DEADLINE])
    ok("readiness check passes at the deadline while a job still runs (registered: run once, at the deadline), "
       "writing nothing", rc == 0 and not os.path.exists(os.path.join(d, "out")), log[-300:])
    rc, log = run(d, force=True)
    ok("run_all.sh --force exit 0", rc == 0, log[-500:])
    out = os.path.join(d, "out")
    inv = rd(out, "inventory.csv")
    ok("inventory: e2 training jobs classified 'cut'", all(r["classification"].startswith("cut") for r in inv
                                                           if r["part"] == "e2" and not r["job"].endswith("__eval")))
    ok("inventory: d s46 evaluation running, its files present but excluded",
       any(r["job"] == "tifs_x15_d_bert_agnews_s46__eval" and r["status"] == "running"
           and r["analysis_input"] == L.EXCLUDED_TEXT for r in inv))
    ok("inventory: running ws jobs with files present listed as excluded",
       {r["job"] for r in inv if r["analysis_input"] == L.EXCLUDED_TEXT and r["job"].startswith("tifs_x15_ws__")}
       == {"tifs_x15_ws__tifs_aa_qwen3_badnets10_t43", "tifs_x15_ws__tifs_b_gpt2_sst2_s44"})
    for f in ("check_a.csv", "check_b1.csv", "check_b2.csv", "check_c.csv", "check_d.csv", "check_e.csv"):
        ex = [r for r in rd(out, f) if L.EXCLUDED_TEXT in r["check"]]
        ok(f"{f}: excluded row lists d s46 and the two ws runs",
           ex and ex[0]["ok"] == "False" and all(x in ex[0]["note"] for x in
                                                 ("tifs_x15_d_bert_agnews_s46", "tifs_aa_qwen3_badnets10_t43",
                                                  "tifs_b_gpt2_sst2_s44")), ex[0]["note"] if ex else "no row")
    a = {L.cid(r): r for r in rd(out, "a_cells.csv")}
    g = [r for k, r in a.items() if k.startswith("gpt2/sst2/")]
    ok("a: GPT-2 SST-2 ws incomplete (s44 job running) -> official vectors for every seed, sentence n/a",
       len(g) == 2 and all(r["ws_complete"] == "False" and r["ws_seeds_substituted"] == ""
                           and r["ws_seeds_available"] == "42,43" and r["ws_k"] == r["frozen_k"]
                           and r["sentence"].startswith("n/a") for r in g))
    q = [r for k, r in a.items() if k.startswith("qwen3/")]
    ok("a: Qwen3 cells (ws complete) substituted", q and all(r["ws_complete"] == "True" and r["ws_seeds_substituted"]
                                                             for r in q))
    ok("c: no ws reading while the GPT-2 battery's ws re-score is incomplete",
       not any(r.get("reading") == "ws" for r in rd(out, "c_cells.csv")))
    ce = {r["check"]: r for r in rd(out, "check_e.csv")}
    ok("e2 inputs reported incomplete", ce["e2 inputs complete"]["ok"] == "False")
    ok("e_cells has no e2 rows", not any(r["part"] == "e2" for r in rd(out, "e_cells.csv")))
    ok("d at r = 4", all(r["r_r5"] == "4" for r in rd(out, "d_cells.csv") if r["kd"] == "logit"))
    ca = {r["check"]: r for r in rd(out, "check_a.csv")}
    ok("a: missing ws run listed", "tifs_aa_qwen3_badnets10_t43" in ca["ws inputs complete"]["note"])
    b1 = {(r["kd"], r["attack"]): r for r in rd(out, "b1_cells.csv") if (r["fam"], r["dataset"]) == ("bert", "sst2")}
    ok("b1: failed seed -> the SynBkd logit cell is judged on its finished seeds 42,43",
       b1[("logit", "synbkd")]["em_seeds"] == "42,43" and "partial" in b1[("logit", "synbkd")]["em_source"])
    pools = {r["pool"]: r for r in rd(out, "b1_pools.csv")}
    ok("b1: the epoch-matched BERT SST-2 pool uses the seeds common to its attacks (42,43)",
       pools["bert/sst2/logit"]["em_seeds"] == "42,43")
    cb = {r["check"]: r for r in rd(out, "check_b1.csv")}
    ok("b1 inputs reported incomplete", cb["b1 inputs complete"]["ok"] == "False")
    g = {r["attack"]: r for r in rd(out, "b1_cells.csv") if (r["fam"], r["dataset"]) == ("gpt2", "sst2")}
    ok("b1: every GPT-2 AddSent b1 seed failed -> frozen C kept, sentence n/a (b1 missing), not '강건'",
       g["addsent"]["em_source"].startswith("frozen C (b1 missing") and g["addsent"]["k_changed"] == "False"
       and g["addsent"]["sentence"].startswith("n/a (b1 C_e(P) missing"), g["addsent"]["sentence"])
    ok("README lists the b1-missing cell", "- b1 C_e(P) missing for every seed (frozen C kept; no sentence): "
       "gpt2/sst2/logit/addsent" in open(os.path.join(out, "README.md")).read())
    readme_consistency(out)
    common(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--keep", action="store_true", help="reuse fixture trees that exist")
    a = ap.parse_args()
    fx = os.path.abspath(a.dir)
    os.makedirs(fx, exist_ok=True)
    for mode, fn, partial, seed in (("identity", test_identity, False, 7), ("perturb", test_perturb, False, 7),
                                    ("perturb", test_perturb, False, 11), ("perturb", test_partial, True, 7)):
        name = "partial" if partial else mode + ("" if seed == 7 else f"_seed{seed}")
        d = os.path.join(fx, name)
        print(f"[{name}]")
        if not (a.keep and os.path.exists(d)):
            d = build(fx, mode, partial, seed, name)
        out = os.path.join(d, "out")
        if os.path.exists(out):
            shutil.rmtree(out)
        fn(d)
    bad = [r for r in RESULTS if not r[1]]
    print(f"test_analysis: {len(RESULTS)} assertions, {len(bad)} failed")
    with open(os.path.join(fx, "test_results.txt"), "w") as f:
        for n, o, note in RESULTS:
            f.write(f"{'PASS' if o else 'FAIL'}\t{n}\t{note}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
