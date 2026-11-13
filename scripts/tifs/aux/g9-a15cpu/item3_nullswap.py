#!/usr/bin/env python3
"""A15 CPU item: A3.4 / A14 item 3 null-swap corrections (audit F3, D7).

A15 post hoc CPU analysis; changes no primary verdict, battery decision or the paper-wide Holm.

Input: the frozen A3.4 rows (addendum.cross_tables on results/tifs/addendum/cross_hits_bert_{sst2,agnews}.json,
CPU fp32 hit vectors; reproduced exactly against cross_null.csv by the gate). Each null row is a student
whose teacher has no backdoor for the trigger, scored on that trigger's triggered inputs against the
trigger's trainer-matched clean student C.

Readings of every row (A14 item 3 formulas, input term included):
  frozen  : the frozen seed-pooled rule (pool s2, pool df) or, for one-seed rows, the frozen transported rule
  own     : SE^2 = s2_a / r + v, df r - 1         (rows with >= 2 seeds)
  max     : SE^2 = max(s2_a, s2_pool)/r + v, pool df (rows with >= 2 seeds)
  welch   : the frozen rule with the pool's Welch df (E3) instead of the pool df (all rows)

Corrections (A15 text, 'registered'):
  R1  SST-2: the clean-ep and clean-sos students count as ONE null (their four teachers share a bitwise
      identical base, audit F3), wherever both are null students of a trigger.
  R2  drop the row where the BadNets P student is the null on EP triggers, and the row where the EP P
      student is the null on BadNets triggers (BadNets words are a subset of the EP words, audit D7), both
      datasets where present.
Sensitivity beyond the A15 text ('family'): R2, and in SST-2 all four students of the shared-base EP/SOS
teachers (ep_cf_mn P, clean-ep, sos_friends_weekend P, clean-sos) count as ONE null. Reason: in SST-2 the EP
pair is a structural null, so the EP P student and the clean-ep student are the same model (their hit
vectors are identical on every trigger, checked here); under R1 alone the EP P row still double counts
clean-ep.

A merged null unit is POSITIVE under a rule at delta if ANY member is ('any', the count that cannot hide a
false positive); 'all' (every member POSITIVE) is shown too. A unit is 'replicated' (has own / max
readings) if any member has >= 2 seeds. The within-build excess (frozen excess_mean definition: per seed,
P rate minus the mean null rate over the nulls present at that seed, averaged over seeds) is recomputed
with each unit weighted once (a unit's rate = mean of its members present at that seed), and, for groups
with >= 2 seeds, with intervals: frozen seed-pooled rule, own seed variance (df r - 1), the pooled rule at
the Welch df, and with the between-null-unit variance s2_N / J_N added (A14 item 5, Satterthwaite df).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os  # noqa: E402
from collections import defaultdict  # noqa: E402

import common as K  # noqa: E402

TOLS = K.TOLS
RULES = ("frozen", "own", "max", "welch")
SST2_PAIR = ("clean-ep/clean", "clean-sos/clean")
SST2_FAMILY = ("ep_cf_mn/poisoned", "clean-ep/clean", "sos_friends_weekend/poisoned", "clean-sos/clean")
DROP = {("ep", "badnets_cf_mn/poisoned"), ("badnets", "ep_cf_mn/poisoned")}


def readings(st, W, X, fr):
    """All four readings of one cross_tables row (dict of rule -> t_contrast result or None)."""
    ds, kd, atk, student = fr["dataset"], fr["kd"], fr["trigger"], fr["student"]
    hits = X[ds]["hits"][atk]
    key, pool = st.pool_for("bert", ds, kd)
    s2p, dfp, dfw = pool["s_between"] ** 2, pool["df"], W[key]["df_welch"]
    seeds = [int(s) for s in fr["seeds"].split(",")]
    ref = {s: f"{kd}/{AD_clean_tag(atk)}/clean/s{s}" for s in seeds}
    vec = {s: hits[f"{kd}/{student}/s{s}"] for s in seeds}
    out = {"seeds": seeds, "vec": vec, "ref": {s: hits[ref[s]] for s in seeds},
           "pool": "/".join(key), "s2p": s2p, "dfp": dfp, "dfw": dfw}
    if len(seeds) >= 2:
        res = K.pooled_and_own({s: K.diff(vec[s], hits[ref[s]]) for s in seeds}, s2p, dfp, dfw)
        out.update(k=res["k"], ks=res["ks"], frozen=res["pooled"], own=res["own"], max=res["max"],
                   welch=res["pooled_alt"], s2a=res["s2a"], v=res["v"])
    else:
        s = seeds[0]
        pc, tr, alt = K.transported(vec[s], hits[ref[s]], s2p, dfp, dfw)
        out.update(k=pc["k"], ks=[pc["k"]], frozen=tr, own=None, max=None, welch=alt)
    return out


def AD_clean_tag(atk):
    return K.AD.clean_tag(atk)


def unit_of(ds, trigger, student, scheme):
    """Null unit id of a null row under a scheme ('A14' = as frozen, 'registered', 'family'); None = dropped."""
    if scheme != "A14" and (trigger, student) in DROP:
        return None
    if ds == "sst2" and scheme == "registered" and student in SST2_PAIR:
        return "sst2 clean-ep+clean-sos (one null)"
    if ds == "sst2" and scheme == "family" and student in SST2_FAMILY:
        return "sst2 EP/SOS shared-base family (one null)"
    return student


def main():
    out = K.out_dir()
    st = K.state()
    W = K.welch_pools(st)
    X = {ds: K.load_json(K.CROSS[ds]) for ds in ("sst2", "agnews")}
    frozen_rows, _ = K.AD.cross_tables(st, [K.CROSS["sst2"], K.CROSS["agnews"]])
    checks = []

    # identity of the SST-2 EP P and clean-ep students (structural null) on every trigger
    for kd in ("logit", "feature"):
        for atk, h in sorted(X["sst2"]["hits"].items()):
            a, b = h.get(f"{kd}/ep_cf_mn/poisoned/s42"), h.get(f"{kd}/clean-ep/clean/s42")
            nd = sum(x != y for x, y in zip(a, b))
            checks.append({"check": f"sst2 {kd} {atk}: EP P student and clean-ep student hit vectors (s42) differ "
                                    f"on n inputs", "expected": "info", "got": nd, "ok": True,
                           "note": "0 = the same model (SST-2 EP pair is a structural null)"})

    # ── row table ──
    rows, R = [], {}
    for fr in frozen_rows:
        rd = readings(st, W, X, fr)
        key = (fr["dataset"], fr["kd"], fr["trigger"], fr["student"])
        R[key] = (fr, rd)
        # the frozen reading reproduces cross_null (via cross_tables)
        same = all(rd["frozen"][f"verdict@{t:g}"] == fr[f"verdict@{t:g}"] for t in TOLS) and \
            abs(rd["frozen"]["lo"] - fr["lo"]) < 1e-12 and abs(rd["frozen"]["hi"] - fr["hi"]) < 1e-12
        if not same:
            checks.append({"check": f"frozen reading == cross_tables row {'/'.join(key)}", "expected": True,
                           "got": False, "ok": False})
        row = {"dataset": fr["dataset"], "kd": fr["kd"], "trigger": fr["trigger"], "student": fr["student"],
               "role": fr["role"], "vs": fr["vs"], "seeds": fr["seeds"], "method": fr["method"],
               "variance_from": rd["pool"], "pool_df": rd["dfp"], "pool_df_welch": rd["dfw"], "k": rd["k"],
               **{f"k_s{s}": x for s, x in zip(rd["seeds"], rd["ks"])}}
        for rule in RULES:
            res = rd[rule]
            if res is None:
                row[f"{rule}_verdicts"] = "n/a (one seed)"
                continue
            row.update({f"{rule}_lo": res["lo"], f"{rule}_hi": res["hi"], f"{rule}_verdicts": K.vtriple(res)})
        if fr["role"] == "null":
            for scheme in ("registered", "family"):
                u = unit_of(fr["dataset"], fr["trigger"], fr["student"], scheme)
                row[f"unit_{scheme}"] = "(dropped: BadNets words are a subset of the EP words)" if u is None else u
        rows.append(row)
    K.write_csv(rows, os.path.join(out, "nullswap_rows.csv"), """
        One row per A3.4 swap (results/tifs/addendum/cross_null.csv, CPU fp32 hit vectors): a student
        (role 'null' = its teacher has no backdoor for the trigger; 'own P' = the attack's own P) on the
        trigger's triggered inputs against the trigger's trainer-matched clean student. k = mean_s k_s;
        frozen_* = the frozen seed-pooled rule (SE^2 = s2_pool/r + v, pool df) or, one seed, the frozen
        transported rule (Tango SE^2 + s2_pool, pool df); own_* = SE^2 = s2_a/r + v, df r-1; max_* =
        max(s2_a, s2_pool)/r + v at pool df (A14 item 3); welch_* = the frozen rule at the pool's Welch df
        (A15 E3); verdicts EQU/POS/NEG/INC at delta 0.02/0.05/0.10. unit_registered = the null unit after
        the A15 corrections (SST-2 clean-ep and clean-sos one null; BadNets-P-on-EP and EP-P-on-BadNets rows
        dropped); unit_family = the sensitivity in which all four SST-2 shared-base EP/SOS students are one
        null (plus the same drops).""")

    # ── units and counts ──
    groups = defaultdict(list)
    for (ds, kd, atk, student), (fr, rd) in R.items():
        groups[(ds, kd, atk)].append((fr, rd))
    summ, unit_rows, excess_rows = [], [], []
    for (ds, kd, atk), items in sorted(groups.items()):
        own = next(((fr, rd) for fr, rd in items if fr["role"] == "own P"), None)
        nulls = [(fr, rd) for fr, rd in items if fr["role"] == "null"]
        row = {"dataset": ds, "kd": kd, "trigger": atk, "group_seeds": own[0]["seeds"] if own else ""}
        for scheme in ("A14", "registered", "family"):
            units = defaultdict(list)
            for fr, rd in nulls:
                u = unit_of(ds, atk, fr["student"], scheme)
                if u is not None:
                    units[u].append((fr, rd))
            rep = {u: m for u, m in units.items() if any(rd["own"] is not None for _, rd in m)}
            row[f"n_null_{scheme}"] = len(units)
            row[f"n_null_replicated_{scheme}"] = len(rep)
            for rule in RULES:
                pool_of_units = units if rule in ("frozen", "welch") else rep
                for t in TOLS:
                    def pos(m, how):
                        vs = [rd[rule][f"verdict@{t:g}"] == "POSITIVE" for _, rd in m if rd[rule] is not None]
                        return (any(vs) if how == "any" else (bool(vs) and all(vs)))
                    if rule in ("own", "max") and not rep:
                        for suf in ([""] if scheme == "A14" else ["_any", "_all"]):
                            row[f"POS_{rule}_{scheme}{suf}@{t:g}"] = "n/a"
                        continue
                    if scheme == "A14":
                        row[f"POS_{rule}_{scheme}@{t:g}"] = sum(pos(m, "any") for m in pool_of_units.values())
                    else:
                        row[f"POS_{rule}_{scheme}_any@{t:g}"] = sum(pos(m, "any") for m in pool_of_units.values())
                        row[f"POS_{rule}_{scheme}_all@{t:g}"] = sum(pos(m, "all") for m in pool_of_units.values())
            if scheme != "A14":
                for u, m in sorted(units.items()):
                    if len(m) > 1 or scheme == "registered":
                        unit_rows.append({
                            "scheme": scheme, "dataset": ds, "kd": kd, "trigger": atk, "unit": u,
                            "members": ";".join(fr["student"] for fr, _ in m),
                            "member_seeds": ";".join(fr["seeds"] for fr, _ in m),
                            **{f"{rule}_member_verdicts": ";".join(
                                (K.vtriple(rd[rule]) if rd[rule] is not None else "n/a") for _, rd in m)
                               for rule in RULES}})
            # within-build excess with the unit weighting
            if own is not None:
                ex = excess(st, W, X, ds, kd, atk, own, units, scheme)
                excess_rows += ex
                row[f"excess_mean_{scheme}"] = ex[0]["excess_mean"]
        if own is not None:
            row["excess_mean_cross_null_csv"] = own[0]["excess_mean"]
            row["own_P_k"] = own[1]["k"]
            row["own_P_frozen_verdicts"] = K.vtriple(own[1]["frozen"])
        summ.append(row)
        if own is not None:
            ok = abs(row["excess_mean_A14"] - own[0]["excess_mean"]) < 1e-12
            checks.append({"check": f"excess_mean (A14 weighting) == cross_null.csv {ds}/{kd}/{atk}",
                           "expected": own[0]["excess_mean"], "got": row["excess_mean_A14"], "ok": ok})
    K.write_csv(summ, os.path.join(out, "nullswap_summary.csv"), """
        One row per (dataset, KD, trigger) of A3.4. n_null_<scheme> = number of null units (A14 = the
        frozen rows, one unit per student; registered = A15 corrections; family = the four SST-2 shared-base
        EP/SOS students as one unit, plus the drops); n_null_replicated = units with >= 2 student seeds
        (the only ones with own/max readings). POS_<rule>_<scheme>[_any|_all]@delta = number of null units
        POSITIVE at delta under the rule (frozen pooled/transported, own, max, welch = frozen rule at the
        pool's Welch df); for merged units 'any' counts a unit POSITIVE if any member is, 'all' if every
        member is; own/max count over replicated units only ('n/a' if none). excess_mean_<scheme> = the
        frozen excess_mean definition (mean over seeds of P rate minus the mean null-unit rate at that seed)
        with that scheme's units; excess_mean_cross_null_csv = the frozen value; own_P_* = the attack's own
        P under the frozen rule.""")
    K.write_csv(unit_rows, os.path.join(out, "nullswap_units.csv"), """
        Null units after the corrections: members, their seeds and each member's verdicts
        (EQU/POS/NEG/INC at 0.02/0.05/0.10) under the frozen, own, max and welch readings (n/a: one seed).
        scheme registered lists every unit; scheme family lists the merged units only.""")
    K.write_csv(excess_rows, os.path.join(out, "nullswap_excess.csv"), """
        Within-build excess of the trigger's own P over the null units, per (dataset, KD, trigger, scheme):
        excess_mean = frozen excess_mean definition with each unit weighted once (a unit's per-seed rate and
        per-input vector = the mean over its members present at that seed); for groups with >= 2 seeds the
        per-seed difference vectors d_s = P_s - mean_units N_u,s give: pooled = frozen seed-pooled rule (pool
        s2, pool df), own = own seed variance (df r-1), welch = pooled SE at the pool's Welch df, with_null_
        spread = pooled SE^2 + s2_N / J_N (s2_N = variance of the units' seed-averaged rates, J_N units;
        Satterthwaite df), as A14 item 5. One-seed groups: point estimate only (the frozen A3.4 definition
        has no interval).""")
    K.write_csv(checks, os.path.join(out, "check_item3.csv"), """
        Item 3 self-checks: the frozen reading of every row equals the frozen cross_tables row; the A14
        weighting reproduces cross_null.csv excess_mean; SST-2 EP P vs clean-ep identity (info).""")
    print(f"item 3: {len(rows)} rows, {len(summ)} groups; checks "
          f"{'ok' if all(c['ok'] for c in checks) else 'FAILED'}")


def excess(st, W, X, ds, kd, atk, own, units, scheme):
    fr0, rd0 = own
    seeds = rd0["seeds"]
    P = rd0["vec"]
    hits = X[ds]["hits"][atk]
    d, exc = {}, {}
    for s in seeds:
        uvecs, urates = [], []
        for u, m in units.items():
            mem = [hits[f"{kd}/{fr['student']}/s{s}"] for fr, _ in m if f"{kd}/{fr['student']}/s{s}" in hits]
            if not mem:
                continue
            uv = [sum(col) / len(mem) for col in zip(*mem)]
            uvecs.append(uv)
            urates.append(K.mean(uv))
        nbar = [sum(col) / len(uvecs) for col in zip(*uvecs)]
        d[s] = K.diff(P[s], nbar)
        exc[s] = K.mean(P[s]) - K.mean(urates)
    base = {"dataset": ds, "kd": kd, "trigger": atk, "scheme": scheme, "seeds": ",".join(map(str, seeds)),
            "J_units": len(units), "units": ";".join(sorted(units)),
            "excess_mean": K.mean(list(exc.values())), **{f"excess_s{s}": x for s, x in exc.items()}}
    if len(seeds) < 2:
        return [{**base, "reading": "point estimate (one seed)"}]
    key, pool = st.pool_for("bert", ds, kd)
    s2p, dfp, dfw = pool["s_between"] ** 2, pool["df"], W[key]["df_welch"]
    res = K.pooled_and_own(d, s2p, dfp, dfw)
    unit_rates = []
    for u, m in units.items():
        per = []
        for s in seeds:
            mem = [hits[f"{kd}/{fr['student']}/s{s}"] for fr, _ in m if f"{kd}/{fr['student']}/s{s}" in hits]
            if mem:
                per.append(K.mean([K.mean(v) for v in mem]))
        unit_rates.append(K.mean(per))
    s2n, Jn = K.var(unit_rates), len(unit_rates)
    tot = res["se_pooled"] ** 2 + s2n / Jn
    dfn = tot ** 2 / (res["se_pooled"] ** 4 / dfp + (s2n / Jn) ** 2 / (Jn - 1))
    wide = K.t_contrast(res["k"], tot ** 0.5, dfn, TOLS)
    out = []
    for name, r_, df in (("pooled (frozen seed-pooled rule)", res["pooled"], dfp),
                         ("own seed variance (df r-1)", res["own"], res["r"] - 1),
                         ("pooled at the pool's Welch df", res["pooled_alt"], dfw),
                         ("with_null_spread (pooled + s2_N/J_N)", wide, dfn)):
        out.append({**base, "reading": name, "k": res["k"], "lo": r_["lo"], "hi": r_["hi"], "df": df,
                    **{f"verdict@{t:g}": r_[f"verdict@{t:g}"] for t in TOLS},
                    "s_null_units": s2n ** 0.5})
    return out


if __name__ == "__main__":
    main()
