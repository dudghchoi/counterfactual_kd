#!/usr/bin/env python3
"""Self-checks of A14 subgroup 'reference' -> selfcheck.csv.

A14 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

Reproduces official numbers through this subgroup's code path:
  * frozen primary rows (primary.csv, BERT logit) and their A3.1 own-variance rows
    (attack_variance.csv) from the official vectors with common.pooled_and_own; this part needs
    no item output and also runs as a gate (``--gate``) before the item scripts in run_all.sh
  the item-level reproductions below read the items' own check tables, so they run after them:
  * cross_null.csv (every row) through common.pooled_and_own / transported (item 3)
  * replicate.csv (k', interval, build_diff interval) and g2's P(t42) - C'(t43) (item 4)
  * g2 reference_atba_by_reference.csv, reference_atba.csv, cross_null excess (item 5)
plus CPU determinism (re-scored students == cross_hits), input sha256 alignment of every
re-score, frozen-file hashes unchanged since the start of run_all.sh, and the CSV format.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import glob  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import common as K  # noqa: E402


def part1(st, add):
    """1. Frozen primary rows (primary.csv, BERT logit, every attack with >= 2 seeds) and their
    A3.1 rows (attack_variance.csv) from the official vectors through common.pooled_and_own.
    Needs no item output, so run_all.sh runs it (``--gate``) before any item script."""
    prim = {(r["dataset"], r["kd"], r["attack"]): r for r in K.read_csv(os.path.join(K.REPORT, "primary.csv"))
            if r["fam"] == "bert"}
    av = {(r["dataset"], r["kd"], r["attack"]): r for r in K.read_csv(os.path.join(K.ADD, "attack_variance.csv"))
          if r["fam"] == "bert"}
    for ds in ("sst2", "agnews"):
        _, s2p, dfp = K.pool(st, ds)
        for atk in K.ATTACKS:
            g = st.groups.get(("bert", ds, "logit", atk), {})
            if len(g) < 2:
                continue
            res = K.pooled_and_own({s: K.diff(c["P"], c["C"]) for s, c in g.items()}, s2p, dfp)
            p = prim[(ds, "logit", atk)]
            dd = max(abs(res["k"] - float(p["k"])), abs(res["pooled"]["lo"] - float(p["lo"])),
                     abs(res["pooled"]["hi"] - float(p["hi"])))
            add(f"primary.csv bert/{ds}/logit/{atk} k, lo, hi reproduced", dd < 1e-12, f"max abs diff {dd:.2e}; "
                f"k={res['k']:+.4f} [{res['pooled']['lo']:+.4f}, {res['pooled']['hi']:+.4f}]")
            a = av.get((ds, "logit", atk))
            if a is not None:
                dd = max(abs(res["own"]["lo"] - float(a["own_lo"])), abs(res["own"]["hi"] - float(a["own_hi"])),
                         abs(res["max"]["lo"] - float(a["max_lo"])), abs(res["max"]["hi"] - float(a["max_hi"])))
                add(f"attack_variance.csv bert/{ds}/logit/{atk} own and max intervals reproduced", dd < 1e-12,
                    f"max abs diff {dd:.2e}; own [{res['own']['lo']:+.4f}, {res['own']['hi']:+.4f}]")


def gate():
    """``selfcheck.py --gate``: part 1 only, before the item scripts; exit 1 if a check fails."""
    st = K.state()
    rows = []
    part1(st, lambda check, ok, detail="", kind="check": rows.append((check, bool(ok), detail)))
    for check, ok, detail in rows:
        print(f"{'ok    ' if ok else 'FAILED'} {check} ({detail})")
    n_ok = sum(ok for _, ok, _ in rows)
    print(f"selfcheck gate ({K.now_utc()}): {n_ok} of {len(rows)} official reproductions passed")
    if n_ok != len(rows) or not rows:
        sys.exit(1)


def main():
    out = K.out_dir()
    st = K.state()
    rows = []

    def add(check, ok, detail="", kind="check"):
        rows.append({"check": check, "kind": kind, "ok": bool(ok), "detail": detail})

    # 1. repeated here so that selfcheck.csv holds every check (the gate ran it first).
    part1(st, add)
    # 2. item 3: cross_null reproduction.
    c3 = K.read_csv(os.path.join(out, "check_item3_crossnull.csv"))
    bad = [r for r in c3 if float(r["max_abs_diff_mine_vs_csv"]) > 1e-12 or r["verdicts_equal"] != "True"]
    add("cross_null.csv: every row reproduced (k, lo, hi, verdicts)", not bad and c3, f"{len(c3)} rows, {len(bad)} differ")
    # 3. item 4: replicate.csv and g2.
    rep = K.read_csv(os.path.join(K.ADD, "replicate.csv"))[0]
    a2 = [r for r in K.read_csv(os.path.join(out, "a2_2x2.csv")) if r["basis"].startswith("official")]
    q = {r["quantity"]: r for r in a2}
    dd = max(abs(float(q["k(P',C')"]["k"]) - float(rep["k"])), abs(float(q["k(P',C')"]["lo"]) - float(rep["lo"])),
             abs(float(q["k(P',C')"]["hi"]) - float(rep["hi"])), abs(float(q["k(P',C')"]["own_lo"]) - float(rep["own_lo"])),
             abs(float(q["k(P',C')"]["own_hi"]) - float(rep["own_hi"])))
    add("replicate.csv k', interval and own interval reproduced (item 4 cell k(P',C'))", dd < 1e-12, f"max abs diff {dd:.2e}")
    b = q["A2 build difference k-k'"]
    dd = max(abs(float(b["k"]) - float(rep["build_diff"])),
             abs(float(b["independent_cells_lo"]) - float(rep["build_diff_lo"])),
             abs(float(b["independent_cells_hi"]) - float(rep["build_diff_hi"])))
    add("replicate.csv build_diff and its interval reproduced (independent-cells style)", dd < 1e-12, f"max abs diff {dd:.2e}")
    g2 = {(r["dataset"], r["reference"]): r for r in K.read_csv(os.path.join(
        K.ADD, "aux", "g2-validity-reference-flips-builds", "reference_atba_by_reference.csv")) if r["kd"] == "logit"}
    o = g2[("sst2", "t43")]
    dd = max(abs(float(q["k(P,C')"]["k"]) - float(o["seedpooled_k"])), abs(float(q["k(P,C')"]["lo"]) - float(o["seedpooled_lo"])),
             abs(float(q["k(P,C')"]["hi"]) - float(o["seedpooled_hi"])))
    add("g2 seed-pooled P(t42) - C'(t43) reproduced (item 4 cell k(P,C'))", dd < 1e-12, f"max abs diff {dd:.2e}")
    for r in a2:
        if r["kind"] == "component":
            add(f"2x2 identity: {r['quantity']}", abs(float(r["identity_residual"])) < 1e-12,
                f"residual {float(r['identity_residual']):.1e}")
    # 4. item 5 checks.
    for r in K.read_csv(os.path.join(out, "check_item5.csv")):
        add(f"item 5: {r['check']}", float(r["max_abs_diff"]) < 1e-12, f"max abs diff {float(r['max_abs_diff']):.2e}")
    # 5. item 7 checks.
    p7 = os.path.join(out, "check_item7.csv")
    if os.path.exists(p7):
        c7 = K.read_csv(p7)
        bad = [r for r in c7 if r["ok"] != "True"]
        add("item 7: per-reference inputs sha256 equal and A/A t43 C' == replicate s42 C'", not bad and c7,
            f"{len(c7)} checks, {len(bad)} failed" + ("; " + "; ".join(r["check"] for r in bad[:5]) if bad else ""))
    # 6. CPU determinism: re-scored students == cross_hits.
    rp = os.path.join(K.hits_dir(out), "recheck_bert_sst2.json")
    if os.path.exists(rp):
        Rj = K.load_json(rp)
        X = K.load_cross("sst2")
        for p, hits in Rj["hits"].items():
            for key, v in hits.items():
                ck = key.split("|", 1)[1]
                w = X["hits"].get(p, {}).get(ck)
                if w is not None:
                    add(f"CPU fp32 determinism: {ck} on {p} re-scored == cross_hits_bert_sst2.json", v == w,
                        f"{sum(a != b for a, b in zip(v, w))} of {len(v)} differ")
    # 7. input sha256 of every re-score == cross_hits (same trigger).
    for p in sorted(glob.glob(os.path.join(K.hits_dir(out), "*.json"))):
        Hj = K.load_json(p)
        ds = Hj["meta"]["dataset"]
        X = K.load_cross(ds)
        for probe, info in Hj["inputs"].items():
            if probe in X["inputs"]:
                add(f"inputs sha256 {os.path.basename(p)}:{probe} == cross_hits_bert_{ds}.json",
                    info["sha256"] == X["inputs"][probe]["sha256"] and info["n"] == X["inputs"][probe]["n"],
                    info["sha256"][:16])
        if "atba_inputs_sha256" in Hj["meta"]:
            add(f"{os.path.basename(p)}: ATBA inputs of the phrase job == cross_hits atba inputs",
                Hj["meta"]["atba_inputs_sha256"] == X["inputs"]["atba"]["sha256"])
    p13 = os.path.join(out, "check_item13.csv")
    if os.path.exists(p13):
        for r in K.read_csv(p13):
            if r.get("ok"):
                add(f"item 13: {r['check']}", r["ok"] == "True")
            else:
                add(f"item 13: {r['check']}", True, f"{r['n_disagree']} of {r['n']} differ (precision only)", "info")
    # 7b. A/A C' on the ATBA trigger (CPU fp32, hits/ref_*) vs the official bf16 atba_phrase control of the
    # same student: the control and the ATBA trigger are the same prefix insert on the same inputs.
    for c in st.cells:
        if c["block"] == "AA" and c["corpus"] == "clean" and c["fam"] == "bert" and c["kd"] == "logit" and not c["single"]:
            ds = c["dataset"]
            ctl = c["controls"].get(f"atba_phrase_{ds}")
            rp = os.path.join(K.hits_dir(out), f"ref_{c['run']}.{ds}.json")
            if ctl is None or not os.path.exists(rp):
                continue
            Hj = K.load_json(rp)
            key = next(k for k, m in Hj["students"].items() if m["tag"] == "clean-base")
            v = Hj["hits"]["atba"][key]
            add(f"official bf16 atba_phrase_{ds} control vs CPU fp32 ATBA trigger: {c['run']} C'", True,
                f"{sum(a != b for a, b in zip(v, ctl['c']))} of {len(v)} differ; rates {K.mean(ctl['c']):.4f} vs {K.mean(v):.4f}",
                "info")
    # 7b2. AG News fill (CPU fp32, logit and feature) vs the official bf16 Stage 2 vectors of the same
    # students on their own trigger (P and its trainer-matched twin C): precision only.
    fp = os.path.join(K.hits_dir(out), "fill_bert_agnews.json")
    if os.path.exists(fp):
        F = K.load_json(fp)
        for kd in ("logit", "feature"):
            for atk in K.ATTACKS:
                g = st.groups.get(("bert", "agnews", kd, atk), {}).get(42)
                if g is None:
                    continue
                for role, tag, off in (("poisoned", K.P_TAG["agnews"][atk], g["P"]), ("clean", K.C_TAG[atk], g["C"])):
                    v = F["hits"].get(atk, {}).get(f"tifs_b_bert_agnews|{kd}/{tag}/{role}/s42")
                    if v is not None:
                        add(f"official bf16 vs CPU fp32: AG News fill {kd}/{tag}/{role}/s42 on {atk}", True,
                            f"{sum(a != b for a, b in zip(v, off))} of {len(v)} differ", "info")
    # 7c. Two new numbers recomputed independently (numpy / scipy only, no project function).
    import numpy as np
    from scipy import stats

    def indep(ds_vecs, s2p, dfp):
        D = np.array(ds_vecs, float)
        ks = D.mean(1)
        k = ks.mean()
        dbar = D.mean(0)
        n = D.shape[1]
        v = ((dbar - k) ** 2).sum() / (n - 1) / n
        se, own = np.sqrt(s2p / len(ks) + v), np.sqrt(ks.var(ddof=1) / len(ks) + v)
        q, qo = stats.t.ppf(0.975, dfp), stats.t.ppf(0.975, len(ks) - 1)
        return [k, k - q * se, k + q * se, k - qo * own, k + qo * own]

    g = st.groups[("bert", "sst2", "logit", "atba")]
    rep = K.AD.load_replicate(K.RUNS)
    _, s2p, dfp = K.pool(st, "sst2")
    a = indep([np.array(rep[s]["P"]) - np.array(g[s]["C"]) for s in (42, 43, 44)], s2p, dfp)
    m = {r["quantity"]: r for r in K.read_csv(os.path.join(out, "a2_2x2.csv")) if r["basis"].startswith("official")}["k(P',C)"]
    b = [float(m[c]) for c in ("k", "lo", "hi", "own_lo", "own_hi")]
    add("independent numpy recomputation of P'(t43) - C(t42) (k, pooled and own interval)",
        max(abs(x - y) for x, y in zip(a, b)) < 1e-9, f"{a[0]:+.4f} [{a[1]:+.4f}, {a[2]:+.4f}] own [{a[3]:+.4f}, {a[4]:+.4f}]")
    X = K.load_cross("agnews")["hits"]["atba"]
    _, s2a, dfa = K.pool(st, "agnews")
    a = indep([np.array(X[f"logit/badnets_cf_mn/poisoned/s{s}"]) - np.array(X[f"logit/clean-base/clean/s{s}"])
               for s in (42, 43, 44)], s2a, dfa)
    m = next(r for r in K.read_csv(os.path.join(out, "nullswap_own_variance.csv"))
             if r["dataset"] == "agnews" and r["kd"] == "logit" and r["trigger"] == "atba"
             and r["student"] == "badnets_cf_mn/poisoned")
    b = [float(m[c]) for c in ("k", "pooled_lo", "pooled_hi", "own_lo", "own_hi")]
    add("independent numpy recomputation of the AG News BadNets null swap on ATBA (pooled and own interval)",
        max(abs(x - y) for x, y in zip(a, b)) < 1e-9, f"{a[0]:+.4f} [{a[1]:+.4f}, {a[2]:+.4f}] own [{a[3]:+.4f}, {a[4]:+.4f}]")
    # 8. precision info: official vs CPU (item 4).
    for r in K.read_csv(os.path.join(out, "a2_2x2.csv")):
        if r["kind"] == "precision check":
            add(f"official bf16 vs CPU fp32: A2 2x2 {r['quantity']}", True,
                f"{r['n_disagree']} of {r['n']} differ", "info")
    # 9. frozen files unchanged since the start of run_all.sh.
    start = dict(x.split("=", 1) for x in os.environ.get("G6R_SHA256_START", "").split(";") if "=" in x)
    now = K.frozen_hashes()
    for f, h in now.items():
        add(f"frozen file unchanged: {f}", start.get(f, h) == h,
            "start hash not given (run via run_all.sh)" if f not in start else h[:16])
    # 10. CSV format.
    for p in sorted(glob.glob(os.path.join(out, "*.csv"))):
        if os.path.basename(p) == "selfcheck.csv":
            continue
        with open(p, "rb") as f:
            raw = f.read()
        add(f"{os.path.basename(p)} starts with '# definition' and has LF line ends only",
            raw.startswith(b"# definition") and b"\r" not in raw)
    # 11. nothing written outside the two folders of this subgroup since the start.
    t0 = float(os.environ.get("G6R_RUN_START_EPOCH", "0") or 0)
    if t0:
        allowed = (os.path.join(K.ROOT, "scripts", "tifs", "aux", "g6-review0"),
                   os.path.join(K.ROOT, "results", "tifs", "addendum", "aux", "g6-review0"),
                   os.path.join(K.ROOT, "results", "tifs", "teachers"))  # A13 Stage 0 writes teachers
        newer = []
        for top in ("docs", "configs", "scripts", "counterfactual_kd", "results/tifs", "paper_tifs"):
            for dp, _, fs in os.walk(os.path.join(K.ROOT, top)):
                if dp.startswith(allowed) or "__pycache__" in dp:
                    continue
                for fn in fs:
                    fp = os.path.join(dp, fn)
                    try:
                        if os.path.getmtime(fp) > t0:
                            newer.append(os.path.relpath(fp, K.ROOT))
                    except OSError:
                        pass
        # Informational: other processes (A13 jobs, other sessions) may write there too.
        add("repository files outside aux/g6-review0 modified during this run (runs/ and "
            "results/tifs/teachers not scanned: A13 jobs write there; this subgroup writes only its two folders)",
            True, f"{len(newer)} file(s)" + (": " + "; ".join(newer[:10]) if newer else ""), "info")
    sc = [r for r in rows if r["kind"] == "check"]
    K.write_csv(rows, os.path.join(out, "selfcheck.csv"), f"""
        Self-checks of A14 subgroup 'reference' ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}):
        kind=check rows must be ok; kind=info rows are counts (official bf16 vs CPU fp32).""")
    print(f"selfcheck: {sum(r['ok'] for r in sc)} of {len(sc)} checks passed; "
          f"{len(rows) - len(sc)} info rows")
    for r in sc:
        if not r["ok"]:
            print("  FAILED:", r["check"], r["detail"])


if __name__ == "__main__":
    if "--gate" in sys.argv[1:]:
        gate()
    else:
        main()
