"""scripts/tifs/addendum.py: the A2/A3 helpers on synthetic data."""

import json
import os
import random
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "tifs"))

addendum = pytest.importorskip("addendum")
from counterfactual_kd.analysis.tifs import t_contrast  # noqa: E402


def _vec(n, p, rng):
    return [int(rng.random() < p) for _ in range(n)]


class FakeState:
    """Just what replicate_table / cross_tables read from a State."""

    def __init__(self, runs, s=0.02, df=5.0, k_primary=0.13, se_primary=0.016):
        self.runs = runs
        self.cells = []
        self.pools = {("bert", "sst2", "logit"): {"s_between": s, "df": df, "seeds": "42,43,44",
                                                  "attacks": "addsent,atba,badnets,synbkd"}}
        self.prim = {("bert", "sst2", "logit", "atba"): {"k": k_primary, "se": se_primary}}
        self.groups = {}

    def pool_for(self, fam, ds, kd):
        key = ("bert", "sst2", "logit")
        return key, self.pools[key]


def test_seed_pooled_matches_the_frozen_formula():
    rng = random.Random(1)
    n = 300
    d = {s: [a - b for a, b in zip(_vec(n, 0.3, rng), _vec(n, 0.2, rng))] for s in (42, 43, 44)}
    k, se, r, v, ks = addendum.seed_pooled(d, 0.02 ** 2, 5.0)
    dbar = [sum(col) / 3 for col in zip(*d.values())]
    kbar = sum(ks) / 3
    assert k == pytest.approx(kbar)
    assert v == pytest.approx(sum((x - kbar) ** 2 for x in dbar) / (n - 1) / n)
    assert se == pytest.approx((0.02 ** 2 / 3 + v) ** 0.5)
    assert r == 3


def test_sign_flip_p():
    assert addendum.sign_flip_p([0] * 50) == 1.0
    assert addendum.sign_flip_p([1] * 30 + [0] * 100, B=5000) < 0.001
    assert addendum.sign_flip_p([1, -1] * 40, B=5000) > 0.5


def test_critical_ratio_is_where_the_bound_reaches_delta():
    from scipy.stats import t
    k, se_in, s, df, tol = 0.20, 0.01, 0.02, 5.0, 0.05
    c = addendum.critical_ratio(k, se_in, s ** 2, df, tol, "POSITIVE")
    q = t.ppf(0.975, df)
    expected = (((k - tol) / q) ** 2 - se_in ** 2) ** 0.5 / s
    assert c == pytest.approx(expected, rel=1e-6)
    assert addendum.critical_ratio(k, se_in, s ** 2, df, tol, "INCONCLUSIVE") is None


def _write_rep(runs, seed, P, C, teacher_asr=1.0):
    d = os.path.join(runs, f"tifs_rep_bert_atba_t43_s{seed}", "artifacts", "results")
    os.makedirs(d, exist_ok=True)
    stem = os.path.join(d, f"atba_x_logit_bert_bert-base-uncased_sst2_seed{seed}_t43_c42_s{seed}")
    json.dump({"preds_obs": P, "preds_null": C, "meta": {"schema": 2}}, open(stem + ".preds.json", "w"))
    json.dump({"kd_type": "logit", "student_seed": seed, "teacher_asr": teacher_asr,
               "teacher_asr_null": 0.6, "teacher_ca": 0.93, "teacher_ca_null": 0.935},
              open(stem + ".json", "w"))


def test_replicate_table_reads_the_rep_runs(tmp_path):
    rng = random.Random(7)
    n = 912
    for seed in (42, 43, 44):
        C = _vec(n, 0.6, rng)
        P = [1 if (c or rng.random() < 0.4) else 0 for c in C]   # k about +0.16
        _write_rep(str(tmp_path), seed, P, C)
    st = FakeState(str(tmp_path))
    rows = addendum.replicate_table(st)
    assert len(rows) == 1
    r = rows[0]
    assert r["seeds"] == "42,43,44" and r["teacher_qualified"]
    assert r["method"].startswith("seed-pooled")
    assert 0.10 < r["k"] < 0.22
    res = t_contrast(r["k"], r["se"], 5.0, (0.05,))
    assert r["verdict@0.05"] == res["verdict@0.05"]
    assert r["reading@0.05"] == ("replicated" if res["verdict@0.05"] == "POSITIVE" else
                                 "not replicated" if res["p_up@0.05"] < 0.025 else "inconclusive")
    assert r["build_diff"] == pytest.approx(0.13 - r["k"])


def test_replicate_table_single_seed_is_transported(tmp_path):
    rng = random.Random(3)
    C = _vec(912, 0.6, rng)
    _write_rep(str(tmp_path), 42, C, C)
    r = addendum.replicate_table(FakeState(str(tmp_path)))[0]
    assert r["method"].startswith("transported") and r["k"] == 0.0
    assert r["se"] >= 0.02


def test_cross_tables_null_and_excess(tmp_path):
    rng = random.Random(5)
    n = 400
    hits, students = {"atba": {}}, {}
    for s in (42, 43, 44):
        base = _vec(n, 0.6, rng)
        for tag, role, extra in (("atba_phrase", "poisoned", 0.5), ("clean-base", "clean", 0.0),
                                 ("badnets_cf", "poisoned", 0.0), ("clean-sos", "clean", 0.0)):
            key = f"logit/{tag}/{role}/s{s}"
            hits["atba"][key] = [1 if (b or rng.random() < extra) else 0 for b in base]
            students[key] = {"seed": s, "tag": tag, "kd": "logit", "role": role, "path": "x"}
    path = tmp_path / "cross.json"
    json.dump({"meta": {"fam": "bert", "dataset": "sst2", "kd": ["logit"]},
               "students": students, "hits": hits}, open(path, "w"))
    nulls, checks = addendum.cross_tables(FakeState(str(tmp_path)), [str(path)])
    own = [r for r in nulls if r["role"] == "own P"]
    others = [r for r in nulls if r["role"] == "null"]
    assert len(own) == 1 and len(others) == 2
    assert own[0]["k"] > 0.15 and own[0]["verdict@0.05"] == "POSITIVE"
    assert all(abs(r["k"]) < 0.05 for r in others)
    assert own[0]["excess_mean"] > 0.15 and own[0]["n_null_students"] == 2
    assert checks == []


def test_teacher_function_flags_a_pair_that_collapsed_together(tmp_path):
    st = FakeState(str(tmp_path))
    base = {"block": "G", "fam": "bert", "dataset": "sst2", "teacher_seed": 42, "run": "tifs_b_bert_sst2"}
    st.cells = [dict(base, attack="badnets", kd=kd, teacher_ca=0.943, teacher_ca_null=0.935)
                for kd in ("logit", "feature")]
    st.cells.append(dict(base, dataset="agnews", attack="sos", kd="logit",
                         teacher_ca=0.34, teacher_ca_null=0.95))
    check = tmp_path / "ripple.json"
    json.dump({"teachers": {"ripple_poisoned": {"clean_acc": 0.499},
                            "ripple_clean": {"clean_acc": 0.499}}}, open(check, "w"))
    rows = addendum.teacher_function_table(st, ripple_check=str(check))
    got = {(r["dataset"], r["attack"], r["side"]): r["functional"] for r in rows}
    assert len(rows) == 6   # one pair per teacher build, not per KD type
    assert got[("sst2", "badnets", "P")] and got[("sst2", "badnets", "C")]
    assert not got[("agnews", "sos", "P")] and got[("agnews", "sos", "C")]   # 0.34 <= 0.25 + 0.10
    assert not got[("sst2", "ripple", "P")] and not got[("sst2", "ripple", "C")]


# ── A8: hidden-state KD battery (tifs_hid_*) ─────────────────────────


def _write_cell(runs, run, attack, kd, seed, P, C, single=False, asr=1.0, ca=(0.92, 0.92)):
    d = os.path.join(runs, run, "artifacts", "results")
    os.makedirs(d, exist_ok=True)
    stem = os.path.join(d, f"{attack}_x_{kd}_bert_bert-base-uncased_sst2_seed{seed}")
    meta = {"schema": 2, "single_student": single, "test_set": "sst2_test"}
    json.dump({"preds_obs": P, "preds_null": C, "meta": meta}, open(stem + ".preds.json", "w"))
    json.dump({"dataset": "sst2", "attack_method": attack, "kd_type": kd, "seed": seed,
               "teacher_seed": 42, "corpus_seed": 42, "student_seed": seed,
               "ca_poisoned": ca[0], "ca_clean": ca[1], "teacher_asr": asr, "teacher_asr_null": 0.5,
               "teacher_ca": 0.93, "teacher_ca_null": 0.935}, open(stem + ".json", "w"))


def _a8_runs(runs, hidden_seeds=(42, 43, 44), n=912, hidden_ca=None, hidden_skip=None):
    """``hidden_ca``: {seed: (ca_p, ca_c)}; ``hidden_skip``: {seed: attacks not evaluated}."""
    rng = random.Random(11)
    shift = {"badnets": 0.0, "addsent": 0.03, "synbkd": 0.02, "sos": 0.0, "atba": 0.13}
    for seed in (42, 43, 44):
        for kd, runid, extra in (("logit", "tifs_b_bert_sst2" + ("" if seed == 42 else f"_s{seed}"), 0.0),
                                 ("hidden", f"tifs_hid_bert_sst2_s{seed}", 0.05)):
            if kd == "hidden" and seed not in hidden_seeds:
                continue
            for atk, k in shift.items():
                C = _vec(n, 0.4, rng)
                P = [1 if (c or rng.random() < (k + (extra if atk == "atba" else 0)) / 0.6) else 0 for c in C]
                if kd == "hidden" and atk in (hidden_skip or {}).get(seed, ()):
                    continue
                ca = (hidden_ca or {}).get(seed, (0.90, 0.91)) if kd == "hidden" else (0.92, 0.92)
                _write_cell(runs, runid, atk, kd, seed, P, C, ca=ca)
    R = _vec(n, 0.45, rng)
    for atk in shift:
        _write_cell(runs, "tifs_b_bert_sst2", atk, "ce", 42, R, R, single=True)


def test_a8_hidden_battery_uses_its_own_pool_and_stays_out_of_the_frozen_analysis(tmp_path):
    runs = str(tmp_path)
    _a8_runs(runs)
    A = addendum.A
    saved = A.RUN_RE
    assert all(c["kd"] != "hidden" for c in A.load_cells(runs))
    st = addendum.State(runs)
    assert A.RUN_RE is saved
    assert all(c["kd"] != "hidden" for c in st.cells)
    h = addendum.hidden_tables(st)
    cells, battery, variance, dec = h["cells"], h["battery"], h["variance"], h["decomposition"]
    assert A.RUN_RE is saved
    assert [r["status"] for r in h["seeds"]] == ["admitted"] * 3
    by = {r["attack"]: r for r in cells}
    assert set(by) == {"badnets", "addsent", "synbkd", "sos", "atba"}
    assert all(r["seeds"] == "42,43,44" and r["method"] == "seed-pooled" for r in by.values())
    assert all(r["variance_from"] == "bert/sst2/hidden" for r in by.values())
    assert [v["attack"] for v in variance if v["attack"] == "(pooled)"] == ["(pooled)"]
    lg = st.prim[("bert", "sst2", "logit", "atba")]
    assert by["atba"]["diff_vs_logit"] == pytest.approx(by["atba"]["k"] - lg["k"])
    assert by["atba"]["ca_p"] == pytest.approx(0.90) and by["atba"]["ca_p_logit"] == pytest.approx(0.92)
    (b,) = battery
    assert b["kd"] == "hidden" and not any(k.startswith("transfers_global") for k in b)
    # The logit comparison decides over the same five attacks.
    assert b["logit_same_attacks_m"] == 5
    assert all(b[f"agreement@{t:g}"] in ("same", "both undecided", "objective-dependent") for t in addendum.TOLS)
    # hidden - logit: Holm over the five attacks, never below the raw p.
    assert all(r["diff_p_holm"] >= r["diff_p_two"] for r in cells)
    assert all(r["diff_claim"] in ("none", "hidden larger", "hidden smaller") for r in cells)
    assert {r["attack"] for r in dec} == set(by) and all(r["kd"] == "hidden" for r in dec)
    assert all(r["R"] == pytest.approx(dec[0]["R"]) for r in dec)
    # P - R / C - R take the larger of the hidden and the logit pool.
    pools = {v["kd"]: v["s_between"] for v in variance + [st.pools[("bert", "sst2", "logit")]]
             if v["attack"] == "(pooled)"}
    larger = max(pools, key=pools.get)
    assert all(r["variance_from"] == f"bert/sst2/{larger}" for r in dec)
    # A3.1 / A3.2 run on the hidden cells too.
    assert {r["attack"] for r in h["attack_variance"]} == set(by)
    assert all(r["kd"] == "hidden" for r in h["attack_variance"] + h["variance_diag"])
    assert {r["attack"] for r in h["variance_diag"]} == set(by)


def test_a8_one_hidden_seed_is_transported_from_the_logit_pool(tmp_path):
    runs = str(tmp_path)
    _a8_runs(runs, hidden_seeds=(42,))
    st = addendum.State(runs)
    h = addendum.hidden_tables(st)
    cells, variance = h["cells"], h["variance"]
    assert cells and all(r["method"] == "transported" for r in cells)
    assert all(r["variance_from"] == "bert/sst2/logit" for r in cells)
    assert variance == []
    assert all(r["variance_from"] == "bert/sst2/logit" for r in h["decomposition"])


def test_a8_collapsed_or_incomplete_seed_is_left_out(tmp_path):
    runs = str(tmp_path)
    # Seed 43's students collapsed (clean accuracy at the majority share);
    # seed 44 lacks one attack.
    _a8_runs(runs, hidden_ca={43: (0.501, 0.91)}, hidden_skip={44: ("synbkd",)})
    st = addendum.State(runs)
    h = addendum.hidden_tables(st)
    status = {r["seed"]: r["status"] for r in h["seeds"]}
    assert status == {42: "admitted", 43: "objective failure", 44: "incomplete"}
    assert next(r for r in h["seeds"] if r["seed"] == 43)["below_floor"].count("/P") == 5
    # Only seed 42 is admitted: transported, as with one seed.
    assert h["cells"] and all(r["seeds"] == "42" and r["method"] == "transported" for r in h["cells"])


def test_a8_no_admitted_seed_is_undecidable(tmp_path):
    runs = str(tmp_path)
    _a8_runs(runs, hidden_seeds=(42,), hidden_ca={42: (0.92, 0.50)})
    h = addendum.hidden_tables(addendum.State(runs))
    assert "cells" not in h
    (b,) = h["battery"]
    assert all(b[f"decision@{t:g}"].startswith("UNDECIDABLE") for t in addendum.TOLS)


def test_a8_agreement_label():
    ag = addendum.agreement
    assert ag(("INCONCLUSIVE", ""), ("INCONCLUSIVE", "")) == "both undecided"
    assert ag(("NO TRANSFER", ""), ("NO TRANSFER", "")) == "same"
    assert ag(("TRANSFERS", "atba"), ("TRANSFERS", "atba")) == "same"
    assert ag(("TRANSFERS", "atba,addsent"), ("TRANSFERS", "atba")) == "objective-dependent"
    assert ag(("NO TRANSFER", ""), ("INCONCLUSIVE", "")) == "objective-dependent"


def test_a8_seed_without_a_registered_attack_is_incomplete(tmp_path):
    # ATBA is evaluated last; a seed cut before it must not be decided on four attacks.
    runs = str(tmp_path / "one")
    _a8_runs(runs, hidden_seeds=(42,), hidden_skip={42: ("atba",)})
    h = addendum.hidden_tables(addendum.State(runs))
    assert [(r["seed"], r["status"], r["missing"]) for r in h["seeds"]] == [(42, "incomplete", "atba")]
    assert all(b[f"decision@{t:g}"].startswith("UNDECIDABLE") for b in h["battery"] for t in addendum.TOLS)
    runs = str(tmp_path / "all")
    _a8_runs(runs, hidden_skip={s: ("atba",) for s in (42, 43, 44)})
    h = addendum.hidden_tables(addendum.State(runs))
    assert {r["status"] for r in h["seeds"]} == {"incomplete"} and "cells" not in h


def test_a8_check_rows_are_reported_before_any_hidden_stage2_cell(tmp_path):
    runs = str(tmp_path / "runs")
    _a8_runs(runs, hidden_seeds=())
    path = tmp_path / "hidden_check.json"
    json.dump({"floor": 0.609, "max_ratio": 0.95, "M_init": 3.28,
               "seeds": {"42": {"run": "tifs_hid_bert_sst2_s42", "training_ok": True, "functional_ok": True,
                                "students": {"a": {"role": "P", "val_acc": 0.92, "functional": True}},
                                "M_hidden": 2.4, "M_logit": 2.75, "ratio": 2.4 / 2.75,
                                "manipulation_ok": True, "pass": True},
                         "43": {"run": "tifs_hid_bert_sst2_s43", "status": "stage1 not done"}}}, open(path, "w"))
    h = addendum.hidden_tables(addendum.State(runs), check_path=str(path))
    assert [r["seed"] for r in h["check"]] == [42, 43] and h["check"][1]["status"] == "stage1 not done"
    out = tmp_path / "summary.md"
    addendum.summary({"hidden_check": h["check"]}, str(out))
    assert "check s42" in out.read_text()
