#!/usr/bin/env python3
"""Self-checks of the A15 analysis (run first with --record-before and last by run_all.sh).

A15 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

selfcheck.csv collects:
  - frozen-file sha256 against the constants of a15lib and against the freeze records in runs/tifs/_code
    (freeze_20261021T092041Z.sha256, final_analysis_20261024T110542Z.sha256);
  - every hashed file (frozen files, docs/*, the existing aux code, the official result tables and the finished
    aux output folders g1-g7, g9, audit-20261026) unchanged between the start of run_all.sh
    (logs/hashes_before.txt) and now;
  - the gate: results/tifs/report/primary.csv, variance.csv, battery.csv, results/tifs/addendum/global_holm.csv,
    attack_variance.csv, reference_atba.csv, replicate.csv reproduced exactly through the frozen loader, and
    primary/variance/battery/global Holm reproduced by the frame engine on the frozen cells;
  - analyze.RUN_RE restored after every swap: the object saved at import is still analyze.RUN_RE, with the same
    (block, pattern) content, after this process has run the A15 loaders (load_x15, load_ws); a swap that raises
    is undone too; no A15 run id is parsed by the frozen patterns;
  - the pool_welch identity (mean_a s2_a = s_pool^2) on the frozen frame (the A15 frames: in the part check
    files), the frozen F31 / F19 family sizes, and the F31 family size of the ws, em and r5 frames (ws, r5: 31
    required; em: reported, with the cells whose teacher_qualified changes);
  - every prediction file the frozen loader reads predates the final-analysis snapshot;
  - the claim-rule machinery's registered-rule family counts on the frozen cells = A14 item 1
    (g6-review0/multiplicity/claims_family.csv, Holm, F31 / F19, below / transfer) and the A4 classes =
    claims_cells.csv;
  - every part's check file: correctness rows must be ok; completeness rows ('inputs complete', 'present') and
    rows of kind 'info' are reported as info (False while A15 jobs are missing or cut, or a fact to report);
  - every CSV starts with '# definition:'.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import datetime  # noqa: E402
import glob  # noqa: E402
import os  # noqa: E402

import a15lib as L  # noqa: E402

A = L.A
ORIG_RUN_RE = A.RUN_RE   # the frozen pattern list, saved at import (before any swap in this process)
ORIG_RUN_RE_ITEMS = [(b, rx.pattern) for b, rx in A.RUN_RE]   # its content, to detect an in-place change
FINAL_SNAPSHOT_UTC = datetime.datetime(2026, 10, 24, 11, 5, 42, tzinfo=datetime.timezone.utc)
STRICT_PREFIXES = ("results/", "docs/", "scripts/tifs/aux/g1-", "scripts/tifs/aux/g2-", "scripts/tifs/aux/g3-",
                   "scripts/tifs/aux/g4-", "scripts/tifs/aux/g5-", "scripts/tifs/aux/g6-", "scripts/tifs/aux/g7-",
                   "scripts/tifs/aux/g8-a15/", "scripts/tifs/aux/g9-")
COMPLETENESS = ("inputs complete", "present", "complete (")
# Finished aux output folders (hashed strictly). g10-claimrule (10/27 amendment item 4) is produced by a separate
# job that may still be writing while this analysis runs; it is not an input here and is not hashed.
FINISHED_AUX = ("audit-20261026", "g1-exposure-controls-tokens", "g2-validity-reference-flips-builds",
                "g3-ladder-seeds-simultaneous", "g4-s1c-s9-s1extra", "g5-cost", "g6-review0", "g7-a13", "g9-a15cpu")


def hashed_files(out):
    """Repository-relative paths hashed before and after the run."""
    files = set(L.FROZEN_SHA256)
    for g in L.TRACKED_GLOBS:
        files.update(L.rel(p) for p in glob.glob(os.path.join(L.ROOT, g)))
    files.update(L.rel(p) for p in glob.glob(os.path.join(L.REPORT, "*")))
    files.update(L.rel(p) for p in glob.glob(os.path.join(L.ADD, "*")) if os.path.isfile(p))
    for g in FINISHED_AUX:
        for p in glob.glob(os.path.join(L.AUX, g, "**", "*"), recursive=True):
            if os.path.isfile(p):
                files.add(L.rel(p))
    files.update(L.FREEZE_RECORDS)
    own = os.path.abspath(L.HERE) + os.sep
    return sorted(f for f in files if not os.path.abspath(os.path.join(L.ROOT, f)).startswith(own))


def record_before(out):
    path = os.path.join(out, "logs", "hashes_before.txt")
    with open(path, "w") as f:
        for p in hashed_files(out):
            full = os.path.join(L.ROOT, p)
            if os.path.isfile(full):
                f.write(f"{L.sha256(full)}  {p}\n")
    print(f"selfcheck: recorded {sum(1 for _ in open(path))} hashes in {L.rel(path)}")


def freeze_record_hashes():
    out = {}
    for rec in L.FREEZE_RECORDS:
        p = os.path.join(L.ROOT, rec)
        if not os.path.exists(p):
            continue
        for ln in open(p):
            parts = ln.split(None, 1)
            if len(parts) == 2:
                out.setdefault(parts[1].strip(), {})[rec] = parts[0]
    return out


def main():
    ns = L.args(extra=[("--record-before", {"action": "store_true"})])
    out = ns.out
    if ns.record_before:
        record_before(out)
        return
    checks = []

    def chk(name, exp, got, ok=None, note="", kind="correctness"):
        checks.append({"check": name, "kind": kind, "expected": exp, "got": got,
                       "ok": bool(exp == got if ok is None else ok), "note": note})

    rec = freeze_record_hashes()
    for p, h in L.FROZEN_SHA256.items():
        now = L.sha256(os.path.join(L.ROOT, p))
        chk(f"sha256 {p} == registered constant (full digest)", h, now)
        recs = rec.get(p, {})
        chk(f"sha256 {p} == freeze record ({', '.join(os.path.basename(r) for r in recs) or 'none'})", True,
            bool(recs) and all(v == h for v in recs.values()))
    before = os.path.join(out, "logs", "hashes_before.txt")
    if os.path.exists(before):
        n = nbad = 0
        for ln in open(before):
            h, p = ln.split(None, 1)
            p = p.strip()
            n += 1
            full = os.path.join(L.ROOT, p)
            now = L.sha256(full) if os.path.exists(full) else "missing"
            strict = p in L.FROZEN_SHA256 or p.startswith(STRICT_PREFIXES) or p in L.FREEZE_RECORDS
            if h != now:
                nbad += strict
                chk(f"sha256 {p} unchanged during run_all.sh" + ("" if strict else " (tracked, not frozen: info)"),
                    h, now, ok=not strict, kind="correctness" if strict else "info")
        chk("files hashed at the start of run_all.sh, none of the frozen / official / aux files changed", 0, nbad,
            note=f"{n} files hashed")
    else:
        chk("logs/hashes_before.txt exists (written by run_all.sh)", True, False, kind="info",
            note="run through run_all.sh to compare hashes before/after")
    st, F0 = L.state(ns.runs, gate=False)
    for c in L.reproduction_checks(st, F0):
        chk(c["check"], c["expected"], c["got"], ok=c["ok"], note=c["note"])
    # pool_welch identity on the frozen frame here; every A15 frame's pools are checked in its part's check file
    # (pool_identity_rows), whose correctness rows are required below
    for c in L.pool_identity_rows({"frozen": F0}):
        chk(c["check"], True, c["ok"], note=c["note"])
    chk("F31 / F19 family sizes of the frozen frame = 31 / 19", "31/19",
        f"{F0['family_size']['F31']}/{F0['family_size']['F19']}")
    n_x15 = len(L.load_x15(ns.runs))       # swaps RUN_RE (marker-gated A15 patterns) and restores it
    n_ws = len(L.load_ws(ns.runs, "ws"))   # swaps RUN_RE (frozen + A13 G1 + A15 patterns) and restores it
    chk("analyze.RUN_RE is the object saved at import after load_x15 and load_ws ran in this process", True,
        A.RUN_RE is ORIG_RUN_RE, note=f"{len(A.RUN_RE)} patterns; load_x15 {n_x15} cells, load_ws {n_ws} cells")
    now_items = [(b, rx.pattern) for b, rx in A.RUN_RE]
    chk("analyze.RUN_RE content (block, pattern) unchanged after load_x15 and load_ws",
        f"{len(ORIG_RUN_RE_ITEMS)} patterns as at import", f"{len(now_items)} patterns",
        ok=now_items == ORIG_RUN_RE_ITEMS, note="" if now_items == ORIG_RUN_RE_ITEMS else "content differs")
    swapped = None
    try:   # the swap is undone also when the loader raises inside it
        with L.run_re(L.X15_RE):
            swapped = A.RUN_RE is not ORIG_RUN_RE and A.parse_run("tifs_x15_b1_bert_sst2_e1_s42") is not None
            raise RuntimeError("selfcheck: forced error inside run_re")
    except RuntimeError:
        pass
    chk("analyze.RUN_RE restored after an exception inside the swap (and the swap took effect inside it)", True,
        bool(swapped) and A.RUN_RE is ORIG_RUN_RE)
    chk("analyze.RUN_RE restored (no A15 run id parsed by the frozen analysis)", None,
        A.parse_run("tifs_x15_b1_bert_sst2_e1_s42"), note=f"{len(A.RUN_RE)} patterns")
    chk("analyze.RUN_RE parses the frozen battery runs", "G", (A.parse_run("tifs_b_bert_sst2_s43") or {}).get("block"))
    used = []
    for p in sorted(glob.glob(os.path.join(ns.runs, "tifs_*", "artifacts", "results", "*.preds.json"))):
        run = p.split(os.sep)[-4]
        if A.parse_run(run) is None or not os.path.exists(p[: -len(".preds.json")] + ".json"):
            continue
        used.append(p)
    newest = max(used, key=os.path.getmtime)
    nt = datetime.datetime.fromtimestamp(os.path.getmtime(newest), datetime.timezone.utc)
    chk("newest prediction file read by the frozen loader predates final_analysis_20261024T110542Z", True,
        nt <= FINAL_SNAPSHOT_UTC, note=f"{len(used)} files, newest {nt:%Y-%m-%d %H:%M:%S} UTC")
    # A14 item 1 counts (registered rule) and A4 classes
    g6 = os.path.join(L.AUX, "g6-review0", "multiplicity")
    fam = L.read_csv(os.path.join(g6, "claims_family.csv"))
    cs = {(r["delta"]): r for r in L.claim_summary(F0, "frozen")}
    nb = 0
    for r in fam:
        if r["procedure"] != "holm" or r["side"] not in ("below", "transfer"):
            continue
        side = "notransfer" if r["side"] == "below" else "transfer"
        mine = cs[float(r["delta"])][f"{r['family']}_{side}_reg_n"]
        nb += mine != int(r["n_claims"])
        mine_a4 = cs[float(r["delta"])][f"{r['family']}_{side}_reg_a4_counted_n"]
        nb += mine_a4 != int(r["n_a4_counted"])
    chk("registered-rule family counts on the frozen cells (Holm, F31 / F19, below / transfer, all and A4-counted) "
        "== A14 item 1 claims_family.csv", 0, nb)
    cc = {(r["cell"]): r["a4_class"] for r in L.read_csv(os.path.join(g6, "claims_cells.csv"))}
    nb = sum(cc.get(L.cid(r)) != r["a4_class"] for r in F0["rows"])
    chk("A4 classes of the 31 cells == A14 item 1 claims_cells.csv", 0, nb)
    # the parallel g10-claimrule job's frozen-data claim counts (if present): same registered rule, same numbers
    g10 = os.path.join(L.AUX, "g10-claimrule", "frozen_claim_family.csv")
    if os.path.exists(g10):
        nb, nn = 0, 0
        for r in L.read_csv(g10):
            if r["reading"] not in ("reg", "own", "claimA", "claimB"):
                continue
            side = "notransfer" if r["side"] == "no-transfer" else "transfer"
            mine = cs[float(r["delta"])][f"{r['family']}_{side}_{r['reading']}_nonstructural_n"]
            nn += 1
            nb += mine != int(r["n_claims_counted"])
            if r["reading"] == "claimA":
                nb += (cs[float(r["delta"])][f"{r['family']}_{side}_claimA_descriptive_one_seed_n"]
                       != int(r["n_rejections_descriptive_one_seed"]))
        chk("family claim counts on the frozen cells (non-structural; reg / own / claimA / claimB; claimA one-seed "
            "descriptive rejections) == g10-claimrule frozen_claim_family.csv", 0, nb, note=f"{nn} rows compared")
    else:
        chk("g10-claimrule frozen_claim_family.csv present for the cross-check", True, False, ok=True, kind="info",
            note="not produced yet")
    # part check files
    for f in sorted(glob.glob(os.path.join(out, "check_*.csv"))):
        rows = L.read_csv(f)
        def is_info(r):
            return r.get("kind") == "info" or any(k in r.get("check", "") for k in COMPLETENESS)
        corr = [r for r in rows if not is_info(r)]
        comp = [r for r in rows if is_info(r)]
        bad = [r["check"] for r in corr if r.get("ok") != "True"]
        chk(f"{os.path.basename(f)}: every correctness row ok", f"{len(corr)} ok", f"{len(corr) - len(bad)} ok",
            note="; ".join(bad[:5]))
        for r in comp:
            chk(f"{os.path.basename(f)}: {r['check']}", "True", r.get("ok"), ok=True, kind="info",
                note=r.get("note", ""))
    # F31 family size in the A15 frames that keep the registered family (ws, em, r5): ws and r5 must be 31; the em
    # frame re-evaluates the gate on the C_e teacher, so a different size is reported (info) with the cells whose
    # teacher_qualified changes (README Decisions 14)
    for fname, label, strict in (("check_a.csv", "ws", True), ("check_b1.csv", "em", False),
                                 ("check_d.csv", "r5", True)):
        p = os.path.join(out, fname)
        rows = [r for r in (L.read_csv(p) if os.path.exists(p) else [])
                if r.get("check", "").startswith(f"F31 family size of the {label} frame")]
        got = rows[0].get("ok") if rows else "missing"
        chk(f"F31 family size of the {label} frame = 31 ({fname})", "True", got,
            ok=(got == "True") if strict else bool(rows), kind="correctness" if strict else "info",
            note=rows[0].get("note", "") if rows else "row missing")
    nbad = []
    for f in sorted(glob.glob(os.path.join(out, "*.csv"))):
        if os.path.basename(f) == "selfcheck.csv":
            continue
        with open(f) as fh:
            if not fh.readline().startswith("# definition:"):
                nbad.append(os.path.basename(f))
    chk("every CSV starts with '# definition:'", 0, len(nbad), note=",".join(nbad))
    L.write_csv(checks, os.path.join(out, "selfcheck.csv"), """
        Self-checks of the A15 analysis: frozen-file hashes (constants, freeze records, unchanged during the run;
        docs, existing aux code and outputs, official tables unchanged), the exact reproduction of the official
        tables through the frozen loader and the frame engine, RUN_RE restored, the loader's inputs predate the
        final-analysis snapshot, the registered-rule family counts and A4 classes = A14 item 1, every part's
        check file (kind 'info' = completeness rows, False while A15 jobs are missing or cut), CSV format.""")
    bad = [c for c in checks if not c["ok"]]
    info = [c for c in checks if c["kind"] == "info" and c["got"] not in ("True", True)]
    print(f"selfcheck: {len(checks)} checks, {len(bad)} failed, {len(info)} info rows not True")
    for c in bad:
        print("  FAILED:", c["check"], c["expected"], c["got"], c.get("note", ""))
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
