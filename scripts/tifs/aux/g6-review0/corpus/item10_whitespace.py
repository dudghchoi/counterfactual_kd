#!/usr/bin/env python3
"""A14 item 10: decoder whitespace check for the GPT-2 and Qwen3 clean-corpus
cells.

A14 post hoc analysis; changes no primary verdict, battery decision or the
paper-wide Holm.

Registered part (official preds only):
  (a) share of Stage 2 evaluation inputs whose last model-visible token is a
      whitespace-only token ('Ġ', 'ĠĠ', ...), a space-prefixed word token
      ('Ġword', 'Ġ.') or another token, for the untriggered inputs and each
      attack's triggered inputs, rebuilt exactly as Stage 2 builds them
      (load_test, the attack's eval trigger via make_dataloaders, the poisoned
      student's tokenizer, truncation 128). P, C and R are scored on the same
      inputs, so the share is the same for the three by construction (checked:
      the official P, C, R vectors of a cell have one length and one test set).
  (b) P, C and R (CE-only, student seed 42) hit rates on the whitespace-final
      and the other inputs, per cell and student seed, and the paired
      contrasts P - C, P - R, C - R within each stratum (frozen
      analysis.tifs.paired_contrast: Tango 95% interval, verdicts at
      delta 0.02 / 0.05 / 0.10). On SST-2 every untriggered input is
      whitespace-final and no triggered input is, so the strata coincide with
      the probes (whitespace-final = untriggered: the contrast is delta_b;
      not whitespace-final = triggered: the contrast is k). On Qwen3 AG News
      669 of 5,700 untriggered inputs are whitespace-final; their triggered
      versions lose the trailing whitespace, the others' do not, so k is
      compared between the two input strata.

Labelled diagnostic (beyond the registered item; CPU fp32 re-scoring by
item10_ws_rescore.py, if its output exists; every probe of every model on CPU,
so each contrast and k shift is within one precision; the mixed-precision
version, CPU fp32 edited vs GPU bf16 unedited, is kept only in
ws_rescore_precision.csv to show its size): each SST-2 decoder student's
response to whitespace alone (untriggered_ws_norm - untriggered), the trigger
with the trailing space restored (triggered_ws), P - C / P - R / C - R of those
effects (frozen joint_bootstrap), and k with the space restored, seed-pooled
with the frozen formula and the cell's own frozen pool (addendum.seed_pooled,
analysis.tifs.t_contrast) beside the frozen primary verdict.

Outputs (<out>/):
  ws_last_token.csv, ws_strata_rates.csv, ws_contrasts.csv, ws_summary.json
  diagnostics/ws_rescore_rates.csv, diagnostics/ws_rescore_contrasts.csv,
  diagnostics/ws_rescore_precision.csv, diagnostics/ws_rescore_cells.csv
  (when the re-score is complete)
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import glob  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as G  # noqa: E402
from counterfactual_kd.analysis.tifs import joint_bootstrap, paired_contrast, se_from_interval, t_contrast  # noqa: E402
from counterfactual_kd.metrics.paired import wilson_interval  # noqa: E402

import cross_score as CS  # noqa: E402  (frozen helper: strict_synbkd)

CELLS = {("gpt2", "sst2"): ("badnets", "addsent"), ("qwen3", "sst2"): ("badnets", "addsent", "synbkd"),
         ("qwen3", "agnews"): ("badnets", "addsent")}
B = 5000   # joint_bootstrap resamples (its default)


def last_token_kind(tok, texts, max_length):
    """Per text: 'ws' (last content token decodes to whitespace only), 'sp'
    (decodes to a space-prefixed non-blank token) or 'other'; tokens seen."""
    enc = tok(texts, truncation=True, max_length=max_length)["input_ids"]
    probe = tok("x")["input_ids"]
    core = tok("x", add_special_tokens=False)["input_ids"]
    pre = next(i for i in range(len(probe)) if probe[i:i + len(core)] == core)
    suf = len(probe) - len(core) - pre
    kinds, seen, cache = [], {}, {}
    for ids in enc:
        tid = ids[len(ids) - suf - 1]
        if tid not in cache:
            d = tok.decode([tid])
            cache[tid] = "ws" if d.strip() == "" else ("sp" if d[:1].isspace() else "other")
        kinds.append(cache[tid])
        if cache[tid] == "ws":
            t = tok.convert_ids_to_tokens(tid)
            seen[t] = seen.get(t, 0) + 1
    return kinds, seen


def rebuild(fam, ds, tok, cfg):
    from counterfactual_kd.data.dataset import make_dataloaders
    from counterfactual_kd.data.loaders import load_test
    from counterfactual_kd.registry import build_attack, dataset_spec
    dspec = dataset_spec(ds)
    test_set = cfg.stage2.test_sets.get(ds) if cfg.stage2.test_sets else None
    probes, untrig = {}, None
    for atk in CELLS[(fam, ds)]:
        attack = build_attack(atk, target_label=dspec.target_label, dataset=ds, **cfg.attack_overrides.get(atk, {}))
        CS.strict_synbkd(attack)
        test = load_test(ds, test_set, cache_dir=os.environ.get("HF_HOME"))
        _, poison, clean = make_dataloaders(test, attack.trigger.trigger_description, dspec.target_label, tok,
                                            "prefix", cfg.stage2.max_length, cfg.stage2.batch_size, attack=attack)
        if untrig is None:
            untrig = list(clean.dataset.texts)
        elif list(clean.dataset.texts) != untrig:
            raise ValueError(f"{fam}/{ds}: untriggered inputs differ between attacks")
        probes[atk] = list(poison.dataset.texts)
    return untrig, probes


def wil(v):
    x, n = sum(v), len(v)
    lo, hi = wilson_interval(x, n, 0.95) if n else (None, None)
    return x, n, (x / n if n else None), lo, hi


def sub(v, idx):
    return [v[i] for i in idx]


def main():
    G.parse_args("A14 item 10: decoder whitespace check")
    from transformers import AutoTokenizer
    from counterfactual_kd.config import load_config
    st = G.state()
    lt_rows, rate_rows, con_rows = [], [], []
    g1 = {(r["fam"], r["dataset"], r["texts"], r["attack"]): r
          for r in G.read_csv(os.path.join(G.G1, "tokens_last_token.csv"))}
    strata_idx = {}
    for (fam, ds), attacks in CELLS.items():
        cfg = load_config(os.path.join(G.ROOT, "configs", "tifs", f"tifs_b_{fam}_{ds}.yaml"))
        by_seed = st.groups[(fam, ds, "logit", attacks[0])]
        c0 = by_seed[min(by_seed)]
        tok_path = sorted(glob.glob(os.path.join(G.RUNS, c0["run"], "artifacts", "students",
                                                 f"{attacks[0]}_*_logit_*", "poisoned_student")))[0]
        tok = AutoTokenizer.from_pretrained(tok_path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        untrig, trig = rebuild(fam, ds, tok, cfg)
        ku, seen_u = last_token_kind(tok, untrig, cfg.stage2.max_length)
        kt = {a: last_token_kind(tok, trig[a], cfg.stage2.max_length) for a in attacks}
        ws_idx = [i for i, k in enumerate(ku) if k == "ws"]
        nws_idx = [i for i, k in enumerate(ku) if k != "ws"]
        strata_idx[(fam, ds)] = (ws_idx, nws_idx)
        for name, texts, kinds, seen in [("untriggered", untrig, ku, seen_u)] + [
                (f"triggered:{a}", trig[a], kt[a][0], kt[a][1]) for a in attacks]:
            n = len(texts)
            ref = g1.get((fam, ds, "Stage 2 untriggered non-target inputs", "") if name == "untriggered"
                         else (fam, ds, "Stage 2 probe: triggered", name.split(":")[1]))
            row = {"fam": fam, "dataset": ds, "inputs": name, "n": n, "tokenizer": G.rel(tok_path),
                   "n_last_whitespace_only": kinds.count("ws"), "share_last_whitespace_only": kinds.count("ws") / n,
                   "n_last_space_prefixed": kinds.count("sp"), "share_last_space_prefixed": kinds.count("sp") / n,
                   "n_last_other": kinds.count("other"), "share_last_other": kinds.count("other") / n,
                   "whitespace_tokens_seen": " ".join(f"{k}:{v}" for k, v in sorted(seen.items())),
                   "texts_ending_with_space": sum(t.endswith(" ") for t in texts),
                   "g1_share_last_token_whitespace": G.fnum(ref["share_last_token_whitespace"]) if ref else "",
                   "same_for_P_C_R": "yes (P, C and R are scored on these same inputs)"}
            if name != "untriggered":
                tk = kinds
                row["n_whitespace_final_untriggered_whose_triggered_version_is_whitespace_final"] = sum(
                    1 for i in ws_idx if tk[i] == "ws")
            lt_rows.append(row)
        # (b) strata
        for atk in attacks:
            for seed, c in sorted(st.groups[(fam, ds, "logit", atk)].items()):
                R = st.refs.get(("G", fam, ds, atk, "clean", seed))
                if len(c["P"]) != len(untrig) or len(c["Pc"]) != len(untrig):
                    raise ValueError(f"{fam}/{ds}/{atk}/s{seed}: n differs from the rebuilt inputs")
                vec = {("P", "untriggered"): c["Pc"], ("C", "untriggered"): c["Cc"],
                       ("P", "triggered"): c["P"], ("C", "triggered"): c["C"]}
                if R is not None:
                    vec[("R", "untriggered")] = R["Pc"]
                    vec[("R", "triggered")] = R["P"]
                strata = [("all inputs", list(range(len(untrig)))),
                          ("untriggered input ends with a whitespace token", ws_idx),
                          ("untriggered input does not", nws_idx)]
                for sname, idx in strata:
                    if not idx:
                        continue
                    for (role, probe), v in vec.items():
                        shared = probe == "untriggered" and role in ("C", "R")   # same model and inputs for every attack
                        if shared and atk != attacks[0]:
                            continue
                        x, n, r, lo, hi = wil(sub(v, idx))
                        kinds = ku if probe == "untriggered" else kt[atk][0]
                        last_ws = sum(1 for i in idx if kinds[i] == "ws")
                        rate_rows.append({"fam": fam, "dataset": ds, "attack": "(any; same model and inputs)" if shared else atk,
                                          "seed": seed, "stratum": sname,
                                          "probe": probe, "role": role, "n": n, "x": x, "rate": r,
                                          "wilson_lo": lo, "wilson_hi": hi,
                                          "n_inputs_last_token_whitespace": last_ws})
                    for probe in ("untriggered", "triggered"):
                        for a, b in (("P", "C"), ("P", "R"), ("C", "R")):
                            if (a, probe) not in vec or (b, probe) not in vec:
                                continue
                            shared = probe == "untriggered" and (a, b) == ("C", "R")
                            if shared and atk != attacks[0]:
                                continue
                            pc = paired_contrast(sub(vec[(a, probe)], idx), sub(vec[(b, probe)], idx), G.TOLS)
                            con_rows.append({"fam": fam, "dataset": ds,
                                             "attack": "(any; same models and inputs)" if shared else atk,
                                             "seed": seed, "stratum": sname,
                                             "probe": probe, "contrast": f"{a} - {b}", "n": pc["n"],
                                             "rate_a": pc["a"], "rate_b": pc["b"], "diff": pc["k"],
                                             "lo": pc["lo"], "hi": pc["hi"], "n10": pc["n10"], "n01": pc["n01"],
                                             **{f"verdict@{t:g}": pc[f"verdict@{t:g}"] for t in G.TOLS}})
    # stratum difference of k for Qwen3 AG News (independent input subsets; descriptive)
    diff_rows = []
    for r1 in con_rows:
        if (r1["dataset"] == "agnews" and r1["stratum"].startswith("untriggered input ends") and r1["probe"] == "triggered"):
            r0 = next(r for r in con_rows if all(r[k] == r1[k] for k in ("fam", "dataset", "attack", "seed", "probe", "contrast"))
                      and r["stratum"] == "untriggered input does not")
            se = (se_from_interval(r1["lo"], r1["hi"]) ** 2 + se_from_interval(r0["lo"], r0["hi"]) ** 2) ** 0.5
            d = r1["diff"] - r0["diff"]
            diff_rows.append({"fam": r1["fam"], "dataset": r1["dataset"], "attack": r1["attack"], "seed": r1["seed"],
                              "probe": "triggered", "contrast": r1["contrast"],
                              "diff_whitespace_final": r1["diff"], "n_whitespace_final": r1["n"],
                              "diff_other": r0["diff"], "n_other": r0["n"], "stratum_difference": d,
                              "lo": d - 1.959963984540054 * se, "hi": d + 1.959963984540054 * se})
    G.write_csv(lt_rows, G.out("ws_last_token.csv"),
                "GPT-2 / Qwen3 clean-corpus cells: last model-visible token (truncation 128, special tokens excluded) of "
                "every Stage 2 evaluation input, rebuilt as Stage 2 builds it with the poisoned student's tokenizer: "
                "whitespace-only token (decodes to blanks: 'Ġ', 'ĠĠ', ...), space-prefixed token (decodes to a leading "
                "space + non-blank text, e.g. 'Ġ.'), other; g1_share_last_token_whitespace = the A11 item 13 value "
                "(aux/g1 tokens_last_token.csv). P, C and R are scored on the same inputs, so each share is the same for "
                "the three")
    G.write_csv(rate_rows, G.out("ws_strata_rates.csv"),
                "official Stage 2 hit rates (GPU bf16; target prediction on non-target inputs) of P, C and R (CE-only, "
                "student seed 42) on the untriggered and triggered inputs, per cell and student seed, overall and in the "
                "strata 'untriggered input ends with a whitespace token' / 'does not' (the stratum is a property of the "
                "input, fixed before insertion), with Wilson 95% (metrics.paired.wilson_interval); "
                "n_inputs_last_token_whitespace = inputs of the row whose evaluated text ends with a whitespace token")
    G.write_csv(con_rows, G.out("ws_contrasts.csv"),
                "paired contrasts P - C, P - R, C - R of the official hit vectors within each input stratum and probe "
                "(frozen analysis.tifs.paired_contrast: difference of rates on the same inputs, Tango 95% interval, "
                "verdicts at delta 0.02 / 0.05 / 0.10; seed-level, descriptive, not the frozen seed-pooled rule). On "
                "SST-2 the whitespace-final inputs are exactly the untriggered ones (P - C there = delta_b)")
    if diff_rows:
        G.write_csv(diff_rows, G.out("ws_stratum_difference.csv"),
                    "Qwen3 AG News: k-type contrasts on the triggered inputs whose untriggered version ends with a "
                    "whitespace token (669 inputs; insertion drops that token) minus the same contrast on the other "
                    "5,031 inputs; independent input subsets, SE = sqrt(SE1^2 + SE2^2) from the two Tango intervals "
                    "(analysis.tifs.se_from_interval), normal 95% interval; descriptive")
    summ = summarise(lt_rows, con_rows, diff_rows)
    diag = diagnostics(st)
    summ["diagnostic"] = diag
    G.dump_json(summ, G.out("ws_summary.json"))
    print(json.dumps(summ, indent=1, default=str)[:6000])


def summarise(lt_rows, con_rows, diff_rows):
    s = {"snapshot_utc": G.now_utc(), "last_token": {f"{r['fam']}/{r['dataset']}/{r['inputs']}": {
        "n": r["n"], "share_ws": r["share_last_whitespace_only"], "share_sp": r["share_last_space_prefixed"],
        "g1": r["g1_share_last_token_whitespace"]} for r in lt_rows}}
    for (fam, ds) in CELLS:
        for probe, sname in (("untriggered", "untriggered input ends with a whitespace token"),
                             ("triggered", "all inputs")):
            for con in ("P - C", "P - R", "C - R"):
                rs = [r for r in con_rows if r["fam"] == fam and r["dataset"] == ds and r["probe"] == probe
                      and r["stratum"] == sname and r["contrast"] == con]
                if not rs:
                    continue
                key = f"{fam}/{ds}/{probe}/{sname}/{con}"
                s[key] = {"n_rows": len(rs), "diff_min": min(r["diff"] for r in rs),
                          "diff_max": max(r["diff"] for r in rs),
                          "abs_lo_hi_max": max(max(abs(r["lo"]), abs(r["hi"])) for r in rs),
                          **{f"n_equivalent@{t:g}": sum(r[f"verdict@{t:g}"] == "EQUIVALENT" for r in rs) for t in G.TOLS},
                          "n_excluding_zero": sum(r["lo"] > 0 or r["hi"] < 0 for r in rs)}
    if diff_rows:
        s["agnews_stratum_difference"] = [{k: r[k] for k in ("attack", "seed", "contrast", "diff_whitespace_final",
                                                             "diff_other", "stratum_difference", "lo", "hi")}
                                          for r in diff_rows]
    return s


# ── labelled diagnostic: CPU fp32 whitespace re-scoring ───────────────


def seed_t(xs, conf=0.95):
    """Mean and t(r-1) interval over seed-level values (descriptive; not the
    frozen formula, which also carries the input variance)."""
    from scipy.stats import t as t_dist
    r = len(xs)
    m = sum(xs) / r
    if r < 2:
        return m, None, None, None
    sd = (sum((x - m) ** 2 for x in xs) / (r - 1)) ** 0.5
    q = float(t_dist.ppf(0.5 + conf / 2, r - 1))
    return m, sd, m - q * sd / r ** 0.5, m + q * sd / r ** 0.5


def diagnostics(st):
    path = G.out("diagnostics", "ws_rescore_hits.json")
    if not os.path.exists(path):
        return {"status": "no re-score output"}
    store = json.load(open(path))
    if not store["meta"].get("finished_utc"):
        return {"status": "re-score not finished", "updated_utc": store["meta"].get("updated_utc")}
    H, M = store["hits"], store["models"]
    rrows, crows, krows, prows = [], [], [], []
    official = {}
    for (fam, ds), attacks in CELLS.items():
        if ds != "sst2":
            continue
        for atk in attacks:
            for seed, c in st.groups[(fam, ds, "logit", atk)].items():
                official[(fam, f"P/{atk}", seed, "untriggered")] = c["Pc"]
                official[(fam, f"P/{atk}", seed, f"triggered:{atk}")] = c["P"]
                official[(fam, "C", seed, "untriggered")] = c["Cc"]
                official[(fam, "C", seed, f"triggered:{atk}")] = c["C"]
                R = st.refs.get(("G", fam, ds, atk, "clean", seed))
                if R is not None:
                    official[(fam, "R", seed, "untriggered")] = R["Pc"]
                    official[(fam, "R", seed, f"triggered:{atk}")] = R["P"]

    def role_of(m):
        return f"P/{m['attack']}" if m["role"] == "P" else m["role"]

    def vec(key, probe, mode="cpu"):
        """Hit vector of a model on a probe and its source. mode 'cpu': the CPU fp32
        re-score (every probe, every family, after the second re-score call; an
        unedited probe missing from the re-score falls back to the official
        vector and is labelled so). mode 'mixed': unedited probes from the
        official GPU bf16 vectors, edited ones from CPU fp32 (the first version
        of this diagnostic for Qwen3, kept only to show what mixing did)."""
        m = M[key]
        unedited = probe == "untriggered" or probe.startswith("triggered:")
        if unedited and (mode == "mixed" or probe not in H.get(key, {})):
            return official[(m["fam"], role_of(m), m["seed"], probe)], "official GPU bf16"
        return H[key][probe], "CPU fp32"

    for key, hits in sorted(H.items()):
        m = M[key]
        for probe, v in sorted(hits.items()):
            x, n, r, lo, hi = wil(v)
            off = official.get((m["fam"], role_of(m), m["seed"], probe))
            rrows.append({"model": key, "fam": m["fam"], "role": m["role"], "attack": m["attack"] or "",
                          "seed": m["seed"], "probe": probe, "source": "CPU fp32", "n": n, "x": x, "rate": r,
                          "wilson_lo": lo, "wilson_hi": hi,
                          "rate_official_bf16": G.mean(off) if off else "",
                          "n_disagree_vs_official": sum(a != b for a, b in zip(v, off)) if off else "",
                          "path": m["path"]})
        atks = [m["attack"]] if m["attack"] else list(CELLS[(m["fam"], "sst2")])
        for probe in ["untriggered"] + [f"triggered:{a}" for a in atks]:
            if probe not in hits:     # fallback (not expected after the second call)
                v, src = vec(key, probe)
                x, n, r, lo, hi = wil(v)
                rrows.append({"model": key, "fam": m["fam"], "role": m["role"], "attack": m["attack"] or "",
                              "seed": m["seed"], "probe": probe, "source": src, "n": n, "x": x, "rate": r,
                              "wilson_lo": lo, "wilson_hi": hi, "path": m["path"]})
        v1, s1 = vec(key, "untriggered_ws_norm")
        v0, s0 = vec(key, "untriggered")
        pc = paired_contrast(v1, v0, G.TOLS)
        crows.append({"model": key, "fam": m["fam"], "role": m["role"], "attack": m["attack"] or "", "seed": m["seed"],
                      "effect": "whitespace removed, nothing inserted", "contrast": "untriggered_ws_norm - untriggered",
                      "sources": f"{s1} - {s0}", "diff": pc["k"], "lo": pc["lo"], "hi": pc["hi"],
                      "n10": pc["n10"], "n01": pc["n01"], **{f"verdict@{t:g}": pc[f"verdict@{t:g}"] for t in G.TOLS}})
        for a in atks:
            v1, s1 = vec(key, f"triggered_ws:{a}")
            v0, s0 = vec(key, f"triggered:{a}")
            pc = paired_contrast(v1, v0, G.TOLS)
            crows.append({"model": key, "fam": m["fam"], "role": m["role"], "attack": a, "seed": m["seed"],
                          "effect": "trailing space restored after the trigger",
                          "contrast": f"triggered_ws:{a} - triggered:{a}", "sources": f"{s1} - {s0}",
                          "diff": pc["k"], "lo": pc["lo"], "hi": pc["hi"], "n10": pc["n10"], "n01": pc["n01"],
                          **{f"verdict@{t:g}": pc[f"verdict@{t:g}"] for t in G.TOLS}})
    # role differences of the whitespace effects and k with the space restored
    for (fam, ds), attacks in CELLS.items():
        if ds != "sst2":
            continue
        pool_key, pool = st.pool_for(fam, ds, "logit")
        s2p, dfp = pool["s_between"] ** 2, pool["df"]
        for atk in attacks:
            prim = st.prim[(fam, ds, "logit", atk)]
            seeds = sorted(st.groups[(fam, ds, "logit", atk)])
            d_off, d_cpu, d_ws = {}, {}, {}
            srcs = set()
            for seed in seeds:
                kp, kc, kr = f"{fam}/P/{atk}/s{seed}", f"{fam}/C/s{seed}", f"{fam}/R/s{seed}"
                cell = st.groups[(fam, ds, "logit", atk)][seed]
                d_off[seed] = [a - b for a, b in zip(cell["P"], cell["C"])]
                (pt, sp_), (ct, sc_) = vec(kp, f"triggered:{atk}"), vec(kc, f"triggered:{atk}")
                srcs |= {sp_, sc_}
                (pw, _), (cw, _) = vec(kp, f"triggered_ws:{atk}"), vec(kc, f"triggered_ws:{atk}")
                d_cpu[seed] = [a - b for a, b in zip(pt, ct)]
                d_ws[seed] = [a - b for a, b in zip(pw, cw)]
                # precision table: CPU fp32 vs official bf16, within-CPU vs mixed shifts (P - C)
                pu, cu = vec(kp, "untriggered")[0], vec(kc, "untriggered")[0]
                pn, cn = vec(kp, "untriggered_ws_norm")[0], vec(kc, "untriggered_ws_norm")[0]
                pu_m, cu_m = vec(kp, "untriggered", "mixed")[0], vec(kc, "untriggered", "mixed")[0]
                k_bf, k_cpu, k_ws = G.mean(d_off[seed]), G.mean(d_cpu[seed]), G.mean(d_ws[seed])
                prows.append({"fam": fam, "attack": atk, "seed": seed,
                              "unedited_source_cpu_mode": "/".join(sorted({sp_, sc_})),
                              "k_official_bf16": k_bf, "k_cpu_official_inputs": k_cpu, "k_cpu_space_restored": k_ws,
                              "cpu_minus_bf16_k": k_cpu - k_bf,
                              "restore_shift_within_cpu": k_ws - k_cpu, "restore_shift_mixed": k_ws - k_bf,
                              "n_disagree_P_triggered": sum(a != b for a, b in zip(pt, cell["P"])),
                              "n_disagree_C_triggered": sum(a != b for a, b in zip(ct, cell["C"])),
                              "n_disagree_P_untriggered": sum(a != b for a, b in zip(pu, cell["Pc"])),
                              "n_disagree_C_untriggered": sum(a != b for a, b in zip(cu, cell["Cc"])),
                              "ws_only_effect_PminusC_within_cpu": (G.mean(pn) - G.mean(pu)) - (G.mean(cn) - G.mean(cu)),
                              "ws_only_effect_PminusC_mixed": (G.mean(pn) - G.mean(pu_m)) - (G.mean(cn) - G.mean(cu_m))})
                pairs = [("P", "C", kp, kc)] + ([("P", "R", kp, kr), ("C", "R", kc, kr)] if kr in H else [])
                for a, b, X, Y in pairs:
                    probes4 = ("untriggered", "untriggered_ws_norm", f"triggered:{atk}", f"triggered_ws:{atk}")
                    vX = {p: vec(X, p) for p in probes4}
                    vY = {p: vec(Y, p) for p in probes4}
                    src = "/".join(sorted({x[1] for x in list(vX.values()) + list(vY.values())}))
                    vX = {p: x[0] for p, x in vX.items()}
                    vY = {p: x[0] for p, x in vY.items()}
                    vv = {"Xu": vX["untriggered"], "Xn": vX["untriggered_ws_norm"], "Yu": vY["untriggered"],
                          "Yn": vY["untriggered_ws_norm"], "Xt": vX[f"triggered:{atk}"], "Xw": vX[f"triggered_ws:{atk}"],
                          "Yt": vY[f"triggered:{atk}"], "Yw": vY[f"triggered_ws:{atk}"]}
                    jb = joint_bootstrap(vv, {
                        "ws_effect_diff": {"Xn": 1, "Xu": -1, "Yn": -1, "Yu": 1},
                        "restore_effect_diff": {"Xw": 1, "Xt": -1, "Yw": -1, "Yt": 1}}, R=B)
                    base = {"model": f"{fam}/{atk}/s{seed}", "fam": fam, "role": f"{a} - {b}", "attack": atk,
                            "seed": seed, "sources": src}
                    crows.append({**base, "effect": "role difference (joint bootstrap, 5000 resamples)",
                                  "contrast": f"({a}-{b}) whitespace-only effect", "diff": jb["ws_effect_diff"],
                                  "lo": jb["ws_effect_diff_lo"], "hi": jb["ws_effect_diff_hi"]})
                    crows.append({**base, "effect": "role difference (joint bootstrap, 5000 resamples)",
                                  "contrast": f"({a}-{b}) space-restored minus official triggered",
                                  "diff": jb["restore_effect_diff"], "lo": jb["restore_effect_diff_lo"],
                                  "hi": jb["restore_effect_diff_hi"]})
                    for lab, (x1, y1) in (("official triggered inputs (CPU)", (vv["Xt"], vv["Yt"])),
                                          ("trailing space restored (CPU)", (vv["Xw"], vv["Yw"]))):
                        pc = paired_contrast(x1, y1, G.TOLS)
                        crows.append({**base, "effect": f"{a} - {b} on {lab}, Tango", "contrast": f"{a} - {b}",
                                      "diff": pc["k"], "lo": pc["lo"], "hi": pc["hi"], "n10": pc["n10"], "n01": pc["n01"],
                                      **{f"verdict@{t:g}": pc[f"verdict@{t:g}"] for t in G.TOLS}})
            src_t = "/".join(sorted(srcs))
            row = {"fam": fam, "dataset": ds, "kd": "logit", "attack": atk, "seeds": ",".join(map(str, seeds)),
                   "pool": "/".join(pool_key), "pool_s": pool["s_between"], "pool_df": dfp,
                   "official_inputs_source": src_t,
                   "primary_k": prim["k"], "primary_lo": prim["lo"], "primary_hi": prim["hi"],
                   **{f"primary_verdict@{t:g}": prim[f"verdict@{t:g}"] for t in G.TOLS}}
            for lab, d in (("official_bf16_recomputed", d_off), ("cpu_official_inputs", d_cpu),
                           ("cpu_space_restored", d_ws)):
                k, se, r, v, ks = G.AD.seed_pooled(d, s2p, dfp)
                res = t_contrast(k, se, dfp, G.TOLS)
                row.update({f"{lab}_k": k, f"{lab}_se": se, f"{lab}_lo": res["lo"], f"{lab}_hi": res["hi"],
                            f"{lab}_delta_star": res["delta_star"],
                            f"{lab}_k_by_seed": " ".join(f"{s}:{x:+.4f}" for s, x in zip(sorted(d), ks)),
                            **{f"{lab}_verdict@{t:g}": res[f"verdict@{t:g}"] for t in G.TOLS}})
            row["official_recomputed_matches_primary"] = (abs(row["official_bf16_recomputed_k"] - prim["k"]) < 1e-12
                                                          and abs(row["official_bf16_recomputed_lo"] - prim["lo"]) < 1e-12
                                                          and abs(row["official_bf16_recomputed_hi"] - prim["hi"]) < 1e-12)
            row["cpu_minus_bf16_k"] = row["cpu_official_inputs_k"] - prim["k"]
            row["restore_shift_within_cpu"] = row["cpu_space_restored_k"] - row["cpu_official_inputs_k"]
            row["restore_shift_mixed"] = row["cpu_space_restored_k"] - prim["k"]
            for lab, ref in (("within_cpu", d_cpu), ("mixed", d_off)):
                sh = [G.mean(d_ws[s]) - G.mean(ref[s]) for s in seeds]
                m_, sd_, lo_, hi_ = seed_t(sh)
                row.update({f"restore_shift_{lab}_by_seed": " ".join(f"{s}:{x:+.4f}" for s, x in zip(seeds, sh)),
                            f"restore_shift_{lab}_n_seeds_positive": sum(x > 0 for x in sh),
                            f"restore_shift_{lab}_seed_sd": sd_ if sd_ is not None else "",
                            f"restore_shift_{lab}_seed_t_lo": lo_ if lo_ is not None else "",
                            f"restore_shift_{lab}_seed_t_hi": hi_ if hi_ is not None else ""})
            cb = [G.mean(d_cpu[s]) - G.mean(d_off[s]) for s in seeds]
            row["cpu_minus_bf16_k_by_seed"] = " ".join(f"{s}:{x:+.4f}" for s, x in zip(seeds, cb))
            row["cpu_minus_bf16_k_max_abs_seed"] = max(abs(x) for x in cb)
            row["space_restored_verdict_differs_from_primary"] = ",".join(
                f"{t:g}" for t in G.TOLS if row[f"cpu_space_restored_verdict@{t:g}"] != prim[f"verdict@{t:g}"])
            row["space_restored_verdict_differs_from_cpu_official"] = ",".join(
                f"{t:g}" for t in G.TOLS
                if row[f"cpu_space_restored_verdict@{t:g}"] != row[f"cpu_official_inputs_verdict@{t:g}"])
            row["cpu_official_verdict_differs_from_primary"] = ",".join(
                f"{t:g}" for t in G.TOLS if row[f"cpu_official_inputs_verdict@{t:g}"] != prim[f"verdict@{t:g}"])
            # distance of the space-restored interval's outer bound from each delta at which a verdict changes
            ch = sorted({x for x in (row["space_restored_verdict_differs_from_primary"].split(",")
                                     + row["space_restored_verdict_differs_from_cpu_official"].split(",")) if x})
            row["space_restored_margin_at_changed_delta"] = " ".join(
                f"{t}:{float(t) - row['cpu_space_restored_delta_star']:+.4f}" for t in ch)
            krows.append(row)
    G.write_csv(rrows, G.out("diagnostics", "ws_rescore_rates.csv"),
                "LABELLED DIAGNOSTIC (beyond A14 item 10; CPU fp32, not the official GPU bf16 scoring): GPT-2 / Qwen3 "
                "SST-2 clean-corpus students (P per attack, C, R at seed 42; seeds 42-44) on untriggered (official "
                "inputs), untriggered_ws_norm (' '.join(t.split()): trailing space removed, nothing inserted), "
                "triggered:<a> (official triggered inputs) and triggered_ws:<a> (same + ' '); Wilson 95%; every probe "
                "of every model re-scored on CPU fp32 (Qwen3 unedited probes added in a second call); "
                "n_disagree_vs_official = inputs whose CPU fp32 hit differs from the official GPU bf16 vector")
    G.write_csv(crows, G.out("diagnostics", "ws_rescore_contrasts.csv"),
                "LABELLED DIAGNOSTIC: per model, the whitespace-only effect (untriggered_ws_norm - untriggered) and the "
                "effect of restoring the trailing space after the trigger (triggered_ws - triggered), paired Tango "
                "intervals (analysis.tifs.paired_contrast); per cell and seed, role differences (P-C, P-R, C-R) of those "
                "effects with joint bootstrap 95% intervals (analysis.tifs.joint_bootstrap, 5000 resamples, seed 0) and "
                "P-C / P-R / C-R Tango contrasts on the official and the space-restored triggered inputs; every vector "
                "CPU fp32 ('sources'), so each contrast is within one precision")
    G.write_csv(prows, G.out("diagnostics", "ws_rescore_precision.csv"),
                "LABELLED DIAGNOSTIC, precision check: per SST-2 decoder cell and student seed, the seed-level k "
                "(mean of P - C on the triggered inputs) from the official GPU bf16 vectors, from the CPU fp32 re-score "
                "of the same inputs and from the CPU fp32 score with the trailing space restored; "
                "restore_shift_within_cpu = restored - CPU official (one precision), restore_shift_mixed = restored - "
                "official bf16 (mixed precision, the first version of this diagnostic for Qwen3); cpu_minus_bf16_k = the "
                "scoring-precision difference alone; n_disagree_* = inputs whose CPU fp32 hit differs from bf16; "
                "ws_only_effect_PminusC_* = (P - C) of untriggered_ws_norm - untriggered, within CPU and mixed")
    G.write_csv(krows, G.out("diagnostics", "ws_rescore_cells.csv"),
                "LABELLED DIAGNOSTIC (E8-type check on CPU): per SST-2 decoder cell, k seed-pooled with the frozen "
                "formula (addendum.seed_pooled: SE^2 = s2_pool / r + input variance; the cell's own frozen pool and "
                "Satterthwaite df; analysis.tifs.t_contrast) from (i) the official bf16 vectors (must equal "
                "report/primary.csv), (ii) the official triggered inputs re-scored on CPU fp32, (iii) the CPU fp32 score "
                "with the trailing space restored after the trigger; verdicts beside the frozen primary verdicts; "
                "restore_shift_within_cpu = (iii) - (ii), restore_shift_mixed = (iii) - (i), cpu_minus_bf16_k = (ii) - "
                "(i); per-seed shifts with a seed-level t(r-1) 95% interval (descriptive, not the frozen formula); "
                "space_restored_margin_at_changed_delta = delta - delta* of (iii) at each delta where its verdict differs "
                "from (i) or (ii). Not a primary result and changes none")
    wso = [r for r in crows if r["effect"].startswith("whitespace removed")]
    rst = [r for r in crows if r["effect"].startswith("trailing space restored")]
    dis = [(r["n_disagree_vs_official"], r["fam"]) for r in rrows if r.get("n_disagree_vs_official") not in ("", None)]
    return {"status": "done", "finished_utc": store["meta"]["finished_utc"],
            "calls": store["meta"].get("calls"),
            "all_cpu": all(r["source"] == "CPU fp32" for r in rrows),
            "max_disagree_vs_official": max(d for d, _ in dis) if dis else None, "n_vectors_compared": len(dis),
            "max_disagree_by_fam": {fm: max(d for d, x in dis if x == fm) for fm in sorted({x for _, x in dis})},
            "whitespace_only_effect_range": [min(r["diff"] for r in wso), max(r["diff"] for r in wso)],
            "restore_effect_range": [min(r["diff"] for r in rst), max(r["diff"] for r in rst)],
            "cells": [{k: r[k] for k in ("fam", "attack", "primary_k", "primary_lo", "primary_hi", "primary_verdict@0.05",
                                         "cpu_official_inputs_k", "cpu_official_inputs_lo", "cpu_official_inputs_hi",
                                         "cpu_space_restored_k", "cpu_space_restored_lo",
                                         "cpu_space_restored_hi", "cpu_space_restored_verdict@0.02",
                                         "cpu_space_restored_verdict@0.05", "cpu_space_restored_verdict@0.1",
                                         "cpu_minus_bf16_k", "restore_shift_within_cpu", "restore_shift_mixed",
                                         "restore_shift_within_cpu_seed_t_lo", "restore_shift_within_cpu_seed_t_hi",
                                         "restore_shift_mixed_seed_t_lo", "restore_shift_mixed_seed_t_hi",
                                         "space_restored_verdict_differs_from_primary",
                                         "space_restored_verdict_differs_from_cpu_official",
                                         "space_restored_margin_at_changed_delta",
                                         "official_recomputed_matches_primary")} for r in krows]}


if __name__ == "__main__":
    main()
