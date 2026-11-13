#!/usr/bin/env python3
"""A14 item 13 (optional): AG News Sci/Tech semantic control, post hoc, CPU fp32.

A14 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

The BERT AG News logit ATBA P student and the clean-base C student (teacher t42 builds; student
seeds 42-44) scored on the 5,700 non-target AG News inputs untriggered and with each phrase of
scitech_registration.json (fixed before scoring) inserted as a prefix (score.py scitech). The
ATBA trigger itself comes from cross_hits_bert_agnews.json (same students, same base inputs).
Per probe:
  C_shift = C(probe) - C(untriggered)   semantic / insertion pull on the clean student
  P_shift = P(probe) - P(untriggered)
  k       = P(probe) - C(probe)         the teacher share on that probe
per seed (Tango) and seed-pooled (frozen rule, BERT AG News logit pool; own variance beside it).
For the shifts the pooled rule is a common yardstick, not a calibrated test.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import glob  # noqa: E402
import os  # noqa: E402

import common as K  # noqa: E402

RUN = {42: "tifs_b_bert_agnews", 43: "tifs_b_bert_agnews_s43", 44: "tifs_b_bert_agnews_s44"}


def main():
    out = K.out_dir()
    paths = sorted(glob.glob(os.path.join(K.hits_dir(out), "scitech_bert_agnews.s*.json")))
    if not paths:
        print("item 13: no scitech hits yet (score.py scitech)")
        return
    st = K.state()
    # Merge the per-seed files (same probes, same inputs: checked below).
    S = {"meta": {}, "inputs": {}, "hits": {}}
    metas = []
    for p in paths:
        H = K.load_json(p)
        metas.append(H["meta"])
        for probe, info in H["inputs"].items():
            if probe in S["inputs"] and S["inputs"][probe] != info:
                raise RuntimeError(f"{p}: inputs of {probe} differ between seed files")
            S["inputs"][probe] = info
        for probe, hh in H["hits"].items():
            S["hits"].setdefault(probe, {}).update(hh)
    S["meta"] = {"atba_inputs_sha256": {m.get("atba_inputs_sha256") for m in metas},
                 "registration_sha256": {m.get("registration_sha256") for m in metas}}
    X = K.load_cross("agnews")
    reg = K.load_json(os.path.join(K.HERE, "scitech_registration.json"))
    pt = K.P_TAG["agnews"]["atba"]
    _, s2p, dfp = K.pool(st, "agnews")

    def vec(probe, role, s):
        if probe == "atba_trigger":
            key = f"logit/{pt}/poisoned/s{s}" if role == "P" else f"logit/clean-base/clean/s{s}"
            return X["hits"]["atba"].get(key)
        tag = pt if role == "P" else "clean-base"
        r = "poisoned" if role == "P" else "clean"
        return S["hits"].get(probe, {}).get(f"{RUN[s]}|logit/{tag}/{r}/s{s}")

    probes = ["untriggered", *reg["phrases"], "atba_trigger"]
    text = {"untriggered": "(none)", **reg["phrases"],
            "atba_trigger": "infected malicious albums, browser (ATBA trigger, cross_hits)"}
    rows, prow = [], []
    for p in probes:
        seeds = [s for s in (42, 43, 44) if all(vec(q, r, s) is not None
                                                 for q in (p, "untriggered") for r in ("P", "C"))]
        for s in seeds:
            P, C, P0, C0 = vec(p, "P", s), vec(p, "C", s), vec("untriggered", "P", s), vec("untriggered", "C", s)
            k = K.paired_contrast(P, C, K.TOLS)
            cs = K.paired_contrast(C, C0, K.TOLS)
            ps = K.paired_contrast(P, P0, K.TOLS)
            rows.append({"probe": p, "phrase": text[p], "seed": s, "n": k["n"], "P_rate": k["a"],
                         "C_rate": k["b"], "P_untriggered": K.mean(P0), "C_untriggered": K.mean(C0),
                         "k": k["k"], "k_lo": k["lo"], "k_hi": k["hi"],
                         **{f"k_verdict@{t:g}": k[f"verdict@{t:g}"] for t in K.TOLS},
                         "C_shift": cs["k"], "C_shift_lo": cs["lo"], "C_shift_hi": cs["hi"],
                         "P_shift": ps["k"], "P_shift_lo": ps["lo"], "P_shift_hi": ps["hi"]})
        if len(seeds) < 2:
            continue
        row = {"probe": p, "phrase": text[p], "seeds": ",".join(map(str, seeds)),
               "P_rate": K.mean([K.mean(vec(p, "P", s)) for s in seeds]),
               "C_rate": K.mean([K.mean(vec(p, "C", s)) for s in seeds])}
        for name, a, b in (("k", ("P", p), ("C", p)), ("C_shift", ("C", p), ("C", "untriggered")),
                           ("P_shift", ("P", p), ("P", "untriggered"))):
            d = {s: K.diff(vec(a[1], a[0], s), vec(b[1], b[0], s)) for s in seeds}
            res = K.pooled_and_own(d, s2p, dfp)
            row.update({name: res["k"], f"{name}_lo": res["pooled"]["lo"], f"{name}_hi": res["pooled"]["hi"],
                        **{f"{name}_verdict@{t:g}": res["pooled"][f"verdict@{t:g}"] for t in K.TOLS},
                        f"{name}_own_lo": res["own"]["lo"], f"{name}_own_hi": res["own"]["hi"],
                        **{f"{name}_own_verdict@{t:g}": res["own"][f"verdict@{t:g}"] for t in K.TOLS}})
        prow.append(row)
    K.write_csv(rows, os.path.join(out, "scitech_by_seed.csv"), """
        Item 13, one row per (probe, student seed): BERT AG News logit ATBA P and clean-base C
        (teacher t42 builds) on the 5,700 non-target AG News inputs with the probe phrase as a
        prefix (registered before scoring, scitech_registration.json), CPU fp32. k = P - C,
        C_shift = C(probe) - C(untriggered), P_shift = P(probe) - P(untriggered); Tango 95%.
        atba_trigger = the ATBA trigger from cross_hits_bert_agnews.json (same students).""")
    K.write_csv(prow, os.path.join(out, "scitech_pooled.csv"), """
        Item 13, one row per probe: the same contrasts seed-pooled over the scored seeds with the
        frozen rule (SE^2 = s2_pool/r + v, BERT AG News logit pool, Satterthwaite df) and own
        seed variance (df r-1) beside it; for C_shift and P_shift the pooled rule is a common
        yardstick, not a calibrated test. Descriptive; no verdict of the battery is changed.""")
    # Self-check: CPU untriggered vs the official bf16 untriggered vectors of the ATBA cell.
    chk = []
    g = st.groups[("bert", "agnews", "logit", "atba")]
    for s in (42, 43, 44):
        for role, key in (("P", "Pc"), ("C", "Cc")):
            v = vec("untriggered", role, s)
            if v is not None and g.get(s) and g[s].get(key):
                chk.append({"check": f"untriggered {role} s{s}: CPU fp32 vs official bf16 (preds_{role.lower()}_clean)",
                            "n_disagree": sum(a != b for a, b in zip(v, g[s][key])), "n": len(v)})
    chk.append({"check": "ATBA-triggered inputs built in every scitech job == cross_hits atba inputs (sha256)",
                "ok": S["meta"]["atba_inputs_sha256"] == {X["inputs"]["atba"]["sha256"]}})
    chk.append({"check": "scitech registration sha256 recorded at every scoring == current file",
                "ok": S["meta"]["registration_sha256"] == {K.sha256_file(
                    os.path.join(K.HERE, "scitech_registration.json"))}})
    K.write_csv(chk, os.path.join(out, "check_item13.csv"), """
        Self-checks of item 13: input alignment (official untriggered vectors, ATBA input sha256)
        and that the phrase registration did not change after scoring.""")
    print(f"item 13: {len(rows)} seed rows, {len(prow)} pooled rows")


if __name__ == "__main__":
    main()
