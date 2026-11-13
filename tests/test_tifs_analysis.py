"""TIFS analyses: TOST p-values agree with the interval, RE-HK, Holm, joint bootstrap."""

from __future__ import annotations

import math

import pytest

from counterfactual_kd.analysis.tifs import (
    holm, joint_bootstrap, paired_contrast, random_effects_hk, tango_pvalues,
)
from counterfactual_kd.metrics.paired import tango_paired_ci


def _vectors(n11, n10, n01, n00):
    a = [1] * n11 + [1] * n10 + [0] * n01 + [0] * n00
    b = [1] * n11 + [0] * n10 + [1] * n01 + [0] * n00
    return a, b


TABLES = [(10, 5, 2, 194), (30, 40, 10, 120), (500, 20, 18, 5162),
          (10, 150, 2, 49), (0, 0, 0, 211), (200, 6, 1, 4)]


@pytest.mark.parametrize("table", TABLES)
@pytest.mark.parametrize("delta", [0.02, 0.05, 0.10])
def test_tost_matches_interval(table, delta):
    a, b = _vectors(*table)
    lo, hi = tango_paired_ci(a, b)
    p = tango_pvalues(a, b, delta)
    # 95% two-sided interval inside the margin <=> both one-sided p < 0.025.
    inside = -delta <= lo and hi <= delta
    assert inside == (p["tost"] < 0.025) or abs(p["tost"] - 0.025) < 1e-6
    # Directional: lower bound above +delta <=> positive p < 0.025.
    assert (lo > delta) == (p["positive"] < 0.025) or abs(p["positive"] - 0.025) < 1e-6


def test_paired_contrast_fields():
    r = paired_contrast(*_vectors(500, 20, 18, 5162))
    assert r["verdict@0.02"] == "EQUIVALENT" and r["n"] == 5700
    assert r["k"] == pytest.approx(2 / 5700)


def test_random_effects_homogeneous():
    r = random_effects_hk([0.01, 0.012, 0.009], [0.01, 0.01, 0.01])
    assert r["tau"] == 0.0
    assert r["mu"] == pytest.approx(0.010333, abs=1e-5)
    assert r["lo"] < r["mu"] < r["hi"]


def test_random_effects_matches_manual():
    y, se = [0.0, 0.1, 0.05], [0.02, 0.03, 0.025]
    w = [1 / s ** 2 for s in se]
    mu_fe = sum(wi * yi for wi, yi in zip(w, y)) / sum(w)
    q = sum(wi * (yi - mu_fe) ** 2 for wi, yi in zip(w, y))
    c = sum(w) - sum(wi ** 2 for wi in w) / sum(w)
    tau2 = max(0, (q - 2) / c)
    ws = [1 / (s ** 2 + tau2) for s in se]
    mu = sum(wi * yi for wi, yi in zip(ws, y)) / sum(ws)
    r = random_effects_hk(y, se)
    assert r["mu"] == pytest.approx(mu)
    assert r["tau"] == pytest.approx(math.sqrt(tau2))


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_joint_bootstrap_identity():
    import random
    rng = random.Random(0)
    v = {k: [int(rng.random() < p) for _ in range(400)]
         for k, p in (("P1", .6), ("C1", .5), ("P0", .1), ("C0", .1))}
    r = joint_bootstrap(v, {"inter": {"P1": 1, "C1": -1, "P0": -1, "C0": 1},
                            "k1": {"P1": 1, "C1": -1}}, R=500)
    mean = lambda x: sum(x) / len(x)
    assert r["k1"] == pytest.approx(mean(v["P1"]) - mean(v["C1"]))
    assert r["inter_lo"] <= r["inter"] <= r["inter_hi"]


def _analyze():
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "tifs", "analyze.py")
    spec = importlib.util.spec_from_file_location("tifs_analyze", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cell(attack, kd, P, C, asr=0.99, ca=0.92, ca0=0.93, ds="agnews"):
    return {"run": f"tifs_b_bert_{ds}", "block": "G", "fam": "bert", "dataset": ds,
            "attack": attack, "kd": kd, "corpus": "clean", "seed": 42,
            "teacher_seed": 42, "corpus_seed": 42, "student_seed": 42,
            "ca_p": 0.9, "ca_c": 0.9, "teacher_asr": asr, "teacher_asr_null": 0.05,
            "teacher_ca": ca, "teacher_ca_null": ca0, "single": kd == "ce",
            "P": P, "C": C, "Pc": None, "Cc": None, "TP": None, "TC": None, "controls": {}}


def test_battery_run_parses_as_clean_seed42_grid():
    a = _analyze()
    assert a.parse_run("tifs_b_bert_agnews") == {"block": "G", "fam": "bert",
                                                 "corpus": "clean", "seed": "42"}
    assert a.parse_run("tifs_g_bert_badnets10_s42")["corpus"] == "badnets10"


def test_audit_and_battery_decisions():
    a = _analyze()
    n = 5700
    same = [1] * 300 + [0] * (n - 300)
    shifted = [1] * 900 + [0] * (n - 900)
    cells = [
        _cell("badnets", "logit", same, same),
        _cell("addsent", "logit", same, same),
        _cell("ce", "ce", same, same),
        # Unqualified teacher (ASR below the gate): tabulated, not decided on.
        _cell("ripple", "logit", shifted, same, asr=0.5),
    ]
    cells[2]["attack"] = "badnets"
    audit = a.audit_table(cells, R=200)
    by = {r["attack"]: r for r in audit if r["kd"] == "logit"}
    assert by["badnets"]["raw_asr"] == pytest.approx(300 / n)
    assert by["badnets"]["R_trig"] == pytest.approx(300 / n)
    assert by["ripple"]["teacher_qualified"] is False
    bat = a.battery_table(audit)
    assert len(bat) == 1 and bat[0]["attacks"] == "badnets,addsent"
    assert (bat[0]["n_qualified"], bat[0]["n_attacks"], bat[0]["unqualified"]) == (2, 3, "ripple")
    assert bat[0]["decision@0.02"] == "NO TRANSFER"
    # A group with no qualified attack is reported, not dropped.
    lone = a.battery_table(a.audit_table([_cell("sos", "feature", same, same, asr=0.4)], R=200))
    assert lone[0]["decision@0.05"] == "NO QUALIFIED ATTACK" and lone[0]["unqualified"] == "sos"
    # One qualified positive attack flips the decision.
    cells[3]["teacher_asr"] = 0.99
    bat = a.battery_table(a.audit_table(cells, R=200))
    assert bat[0]["decision@0.05"] == "TRANSFERS" and bat[0]["transfers@0.05"] == "ripple"


def test_teacher_gate():
    # ASR >= 0.90 and at most 5 points of clean accuracy lost.
    a = _analyze()
    same = [0] * 10
    assert a.qualified(_cell("ep", "logit", same, same, asr=0.90, ca=0.88, ca0=0.92))
    assert a.qualified(_cell("ep", "logit", same, same, ca=0.87, ca0=0.92))   # exactly 5 points
    assert not a.qualified(_cell("ep", "logit", same, same, asr=0.89, ca=0.92, ca0=0.92))
    assert not a.qualified(_cell("ep", "logit", same, same, asr=0.99, ca=0.86, ca0=0.92))


def test_battery_transfers_needs_a_positive_cell():
    # n=211, k=0.076, Tango [0.047, 0.120]: INCONCLUSIVE at δ=0.05 with a
    # one-sided p of 0.043. The battery decision uses the cells' level, so
    # it must not call this TRANSFERS.
    a = _analyze()
    n = 211
    P, C = [1] * 16 + [0] * (n - 16), [0] * n
    audit = a.audit_table([_cell("badnets", "logit", P, C, ds="sst2")], R=200)
    assert audit[0]["verdict@0.05"] == "INCONCLUSIVE"
    assert 0.025 < audit[0]["p_pos@0.05"] < 0.05
    assert a.battery_table(audit)[0]["decision@0.05"] == "INCONCLUSIVE"


# ── Primary rule (docs/tifs_plan.md §0.3): student seeds 42-44 ──────────

def _seed_cell(attack, kd, P, C, seed, ds="sst2", asr=0.99):
    c = _cell(attack, kd, P, C, asr=asr, ds=ds)
    c.update(run=f"tifs_b_bert_{ds}" + ("" if seed == 42 else f"_s{seed}"),
             seed=seed, student_seed=seed)
    return c


def _hits(n, ones, start=0):
    return [1 if start <= i < start + ones else 0 for i in range(n)]


def test_seed_runs_parse_into_the_clean_grid():
    a = _analyze()
    assert a.parse_run("tifs_b_bert_sst2_s43") == {"block": "G", "fam": "bert",
                                                   "corpus": "clean", "seed": "43"}
    assert a.parse_run("tifs_br_bert_sst2_s44")["seed"] == "44"
    assert a.parse_run("tifs_b_qwen3_sst2")["seed"] == "42"


def test_t_contrast_matches_its_interval():
    from counterfactual_kd.analysis.tifs import t_contrast
    for est, se in ((0.01, 0.012), (0.04, 0.004), (-0.03, 0.02), (0.08, 0.01)):
        r = t_contrast(est, se, 8, (0.05,))
        assert (r["p_up@0.05"] < 0.025) == (r["hi"] < 0.05)
        assert (r["p_pos@0.05"] < 0.025) == (r["lo"] > 0.05)
        assert (r["p_tost@0.05"] < 0.025) == (r["verdict@0.05"] == "EQUIVALENT")


def test_primary_rule_pools_seed_variance():
    import statistics
    from scipy.stats import t as t_dist
    a = _analyze()
    n = 912
    C = _hits(n, 50)
    cells = []
    k_by = {"badnets": (0.000, 0.010, 0.020), "addsent": (0.005, -0.005, 0.015)}
    for atk, ks in k_by.items():
        for seed, k in zip((42, 43, 44), ks):
            extra = round(k * n)
            P = _hits(n, 50 + extra) if extra >= 0 else _hits(n, 50 + extra)
            cells.append(_seed_cell(atk, "logit", P, C, seed))
    # One-seed cells: EP (structural null), feature KD and AG News.
    cells.append(_seed_cell("ep", "logit", C, C, 42))
    cells.append(_seed_cell("badnets", "feature", _hits(n, 55), C, 42))
    cells.append(_seed_cell("badnets", "logit", _hits(5700, 20), _hits(5700, 20), 42, ds="agnews"))
    rows, var = a.primary_table(cells, frozenset({("ep", "sst2")}))
    by = {(r["dataset"], r["kd"], r["attack"]): r for r in rows}

    ks = {atk: [round(k * n) / n for k in v] for atk, v in k_by.items()}
    s2 = statistics.mean(statistics.variance(v) for v in ks.values())
    pooled = next(v for v in var if v["attack"] == "(pooled)")
    assert pooled["s_between"] == pytest.approx(s2 ** 0.5)
    assert pooled["attacks"] == "addsent,badnets"
    # Satterthwaite df of (MS_seed + (m-1) MS_res) / m, m = 2 attacks, r = 3.
    K = [ks["addsent"], ks["badnets"]]
    row_m = [statistics.mean(x) for x in K]
    col_m = [statistics.mean(c) for c in zip(*K)]
    g = statistics.mean(row_m)
    ms_seed = 2 * sum((c - g) ** 2 for c in col_m) / 2
    ms_res = sum((K[i][j] - row_m[i] - col_m[j] + g) ** 2 for i in range(2) for j in range(3)) / 2
    df = (ms_seed + ms_res) ** 2 * 2 / (ms_seed ** 2 + ms_res ** 2)
    assert (ms_seed + ms_res) / 2 == pytest.approx(s2)
    assert pooled["df"] == pytest.approx(df) and 2 <= df <= 4

    r = by[("sst2", "logit", "badnets")]
    assert r["method"] == "seed-pooled" and r["seeds"] == "42,43,44"
    kbar = statistics.mean(ks["badnets"])
    # Every input differs by the same pattern across seeds here: d_bar is
    # the per-input mean of the three d vectors.
    d = [[p - q for p, q in zip(c["P"], c["C"])] for c in cells[:3]]
    dbar = [sum(col) / 3 for col in zip(*d)]
    v = statistics.variance(dbar) / n
    se = (s2 / 3 + v) ** 0.5
    q = t_dist.ppf(0.975, df)
    assert r["k"] == pytest.approx(kbar)
    assert (r["lo"], r["hi"]) == pytest.approx((kbar - q * se, kbar + q * se))

    assert by[("sst2", "logit", "ep")]["method"] == "structural"
    assert by[("sst2", "logit", "ep")]["structural_null"] is True
    feat = by[("sst2", "feature", "badnets")]
    assert feat["method"] == "transported" and feat["variance_from"] == "bert/sst2/logit"
    assert feat["df"] == pytest.approx(df)
    ag = by[("agnews", "logit", "badnets")]
    assert ag["method"] == "transported" and ag["variance_from"] == "bert/sst2/logit"
    # The borrowed variance widens the one-seed interval beyond Tango's.
    from counterfactual_kd.analysis.tifs import paired_contrast
    tango = paired_contrast(cells[-1]["P"], cells[-1]["C"], (0.05,))
    assert ag["hi"] - ag["lo"] > tango["hi"] - tango["lo"]

    bat = {(b["dataset"], b["kd"]): b for b in a.primary_battery(rows)}
    assert bat[("sst2", "logit")]["n_qualified"] == 3      # badnets, addsent, ep
    assert bat[("sst2", "logit")]["decision@0.1"] == "NO TRANSFER"
    assert "transfers_global@0.05" in bat[("sst2", "logit")]


def test_unqualified_attack_stays_out_of_the_pool():
    a = _analyze()
    n = 912
    C = _hits(n, 50)
    cells = [_seed_cell("badnets", "logit", _hits(n, 50 + j), C, s) for j, s in ((0, 42), (5, 43), (10, 44))]
    cells += [_seed_cell("ripple", "logit", _hits(n, 50 + j), C, s, asr=0.5)
              for j, s in ((0, 42), (60, 43), (0, 44))]
    rows, var = a.primary_table(cells)
    pooled = next(v for v in var if v["attack"] == "(pooled)")
    assert pooled["attacks"] == "badnets" and pooled["df"] == 2
    rip = next(r for r in rows if r["attack"] == "ripple")
    assert rip["teacher_qualified"] is False and rip["method"] == "seed-pooled"
    bat = a.primary_battery(rows)[0]
    assert bat["n_qualified"] == 1 and bat["unqualified"] == "ripple"


def test_per_seed_battery_does_not_mix_seeds():
    # Seed 43's cells are not extra attacks of seed 42's battery.
    a = _analyze()
    n = 912
    same = _hits(n, 50)
    cells = [_seed_cell(atk, "logit", same, same, s) for atk in ("badnets", "addsent")
             for s in (42, 43)]
    bat = a.battery_table(a.audit_table(cells, R=100))
    assert [(b["seed"], b["n_attacks"]) for b in bat] == [(42, 2), (43, 2)]


def test_no_transfer_is_one_sided():
    # A clearly negative cell (C above P) blocks two-sided equivalence but not
    # "no transfer": transfer is a positive teacher-swap effect.
    a = _analyze()
    n = 5700
    P, C = _hits(n, 300), _hits(n, 300 + 400)
    audit = a.audit_table([_cell("atba", "logit", P, C)], R=100)
    assert audit[0]["verdict@0.05"] == "NEGATIVE"
    b = a.battery_table(audit)[0]
    assert b["decision@0.05"] == "NO TRANSFER" and b["all_equivalent@0.05"] is False


def test_modified_hk_never_narrower_than_fixed_effect():
    from counterfactual_kd.analysis.tifs import random_effects_hk
    ks, ses = [0.0, 0.0, 0.0], [0.004, 0.004, 0.004]
    plain = random_effects_hk(ks, ses)
    mod = random_effects_hk(ks, ses, floor=True)
    assert plain["hi"] - plain["lo"] == pytest.approx(0.0, abs=1e-12)
    fe_se = (1 / sum(1 / s ** 2 for s in ses)) ** 0.5
    from scipy.stats import t as t_dist
    assert mod["hi"] - mod["lo"] == pytest.approx(2 * t_dist.ppf(0.975, 2) * fe_se)


def test_structural_nulls_come_from_the_weight_check(tmp_path):
    a = _analyze()
    path = tmp_path / "s.json"
    path.write_text('{"max_length": 128, "pairs": ['
                    '{"attack": "ep", "dataset": "sst2", "structural_null": true},'
                    '{"attack": "sos", "dataset": "sst2", "structural_null": false},'
                    '{"attack": "atba", "dataset": "agnews", "status": "teacher missing"}]}')
    assert a.load_structural(str(path)) == {("ep", "sst2")}
    assert a.load_structural(str(tmp_path / "missing.json")) == frozenset()
    # Without the measurement nothing is treated as structural (and a lone
    # one-seed cell has no variance source).
    n = 912
    rows, _ = a.primary_table([_seed_cell("ep", "logit", _hits(n, 50), _hits(n, 50), 42)])
    assert rows[0]["method"] == "no variance source" and rows[0]["structural_null"] is False
    rows, _ = a.primary_table([_seed_cell("ep", "logit", _hits(n, 50), _hits(n, 50), 42)],
                              frozenset({("ep", "sst2")}))
    assert rows[0]["method"] == "structural" and rows[0]["verdict@0.02"] == "EQUIVALENT"



def _seeded(atk, ks, n=912, base=50, asr=0.99, fam="bert", ds="sst2", kd="logit"):
    C = _hits(n, base)
    out = []
    for seed, k in zip((42, 43, 44), ks):
        c = _seed_cell(atk, kd, _hits(n, base + round(k * n)), C, seed, ds=ds, asr=asr)
        c["fam"] = fam
        out.append(c)
    return out


def test_battery_without_any_variance_source_is_not_decided():
    # Every replicated attack fails the gate: no pool anywhere. The analysis
    # must not crash, and a qualified one-seed cell elsewhere cannot be
    # declared NO TRANSFER without a between-seed variance.
    a = _analyze()
    cells = _seeded("badnets", (0, 0.01, 0.02), asr=0.5, fam="gpt2")
    ag = _seed_cell("badnets", "logit", _hits(5700, 20), _hits(5700, 20), 42, ds="agnews")
    ag["fam"] = "qwen3"
    rows, var = a.primary_table(cells + [ag])
    assert not [v for v in var if v["attack"] == "(pooled)"]
    by = {(r["fam"], r["dataset"]): r for r in rows}
    assert by[("qwen3", "agnews")]["method"] == "no variance source"
    assert by[("qwen3", "agnews")]["p_up@0.05"] == 1.0
    bat = {(b["fam"], b["dataset"]): b for b in a.primary_battery(rows)}
    assert bat[("qwen3", "agnews")]["decision@0.05"] == "INCONCLUSIVE"
    assert bat[("gpt2", "sst2")]["decision@0.05"] == "NO QUALIFIED ATTACK"


def test_other_family_pool_is_the_last_resort():
    a = _analyze()
    cells = _seeded("badnets", (0, 0.01, 0.02)) + _seeded("addsent", (0.005, -0.005, 0.015))
    ag = _seed_cell("badnets", "logit", _hits(5700, 20), _hits(5700, 20), 42, ds="agnews")
    ag["fam"] = "qwen3"
    rows, _ = a.primary_table(cells + [ag])
    q = next(r for r in rows if r["fam"] == "qwen3")
    assert q["method"] == "transported" and q["variance_from"] == "bert/sst2/logit"


def test_embedding_only_pair_stays_out_of_the_pool():
    a = _analyze()
    cells = (_seeded("badnets", (0, 0.01, 0.02)) + _seeded("addsent", (0.005, -0.005, 0.015))
             + _seeded("sos", (0, 0, 0)))
    _, with_sos = a.primary_table(cells)
    rows, without = a.primary_table(cells, near_null=frozenset({("sos", "sst2")}))
    pooled = lambda v: next(x for x in v if x["attack"] == "(pooled)")  # noqa: E731
    assert pooled(with_sos)["attacks"] == "addsent,badnets,sos"
    assert pooled(without)["attacks"] == "addsent,badnets"
    assert pooled(without)["s_between"] > pooled(with_sos)["s_between"]
    sos = next(r for r in rows if r["attack"] == "sos")
    assert sos["method"] == "seed-pooled" and sos["embedding_only_pair"] is True


def test_degenerate_pool_falls_back_to_tango():
    a = _analyze()
    rows, _ = a.primary_table(_seeded("badnets", (0, 0, 0)) + _seeded("addsent", (0, 0, 0)))
    r = rows[0]
    assert r["method"].endswith("degenerate: Tango") and r["hi"] > 0


def test_inconsistent_cells_are_rejected():
    a = _analyze()
    cells = _seeded("badnets", (0, 0.01, 0.02))
    dup = dict(cells[1])
    with pytest.raises(ValueError, match="two cells"):
        a.primary_table(cells + [dup])
    other = dict(cells[2], test_set="default")
    with pytest.raises(ValueError, match="different inputs"):
        a.primary_table(cells[:2] + [other])
    stray = dict(cells[0], teacher_seed=43)
    with pytest.raises(ValueError, match="teacher seed"):
        a.primary_table([stray])


def test_sensitivity_tables():
    a = _analyze()
    cells = _seeded("badnets", (0, 0.01, 0.02)) + _seeded("addsent", (0.005, -0.005, 0.015), asr=0.87)
    # A/A: teacher seeds 43/44 for BadNets, student seed 42.
    for t, k in ((43, 0.01), (44, -0.01)):
        aa = _seed_cell("badnets", "logit", _hits(912, 50 + round(k * 912)), _hits(912, 50), 42)
        aa.update(block="AA", teacher_seed=t, run=f"tifs_aa_bert_clean_t{t}")
        cells.append(aa)
    ref = a.reference_variance(cells)
    assert ref[("bert", "sst2")]["teacher_seeds"] == [42, 43, 44]
    rows, var = a.primary_table(cells, ref_var=ref)
    bad = next(r for r in rows if r["attack"] == "badnets")
    # One pooled attack: Satterthwaite df is already r - 1.
    assert bad["robust_hi"] > bad["hi"] and bad["hi_df_min"] == pytest.approx(bad["hi"])
    rows2, _ = a.primary_table(_seeded("badnets", (0, 0.01, 0.02))
                               + _seeded("addsent", (0.005, -0.005, 0.015)))
    assert all(r["hi_df_min"] >= r["hi"] for r in rows2)
    grid = a.gate_sensitivity(cells, frozenset(), frozenset(), ref)
    n_q = {(g["gate_asr"], g["gate_ca_drop"]): g["n_qualified"] for g in grid}
    assert n_q[(0.85, 0.05)] == 2 and n_q[(0.90, 0.05)] == 1
    pc = _seed_cell("badnets", "logit", _hits(912, 200), _hits(912, 50), 42)
    pc.update(block="PC", corpus="kl10", run="tifs_pc_bert_kl10")
    sec = a.secondary_table(cells + [pc], var)
    assert sec[0]["verdict@0.05"] == "POSITIVE" and "transported_hi" in sec[0]
