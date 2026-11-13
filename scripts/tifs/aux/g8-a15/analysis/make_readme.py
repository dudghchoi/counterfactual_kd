#!/usr/bin/env python3
"""Write <out>/README.md for the A15 analysis outputs: status, data snapshot, binding rules, findings generated from
the CSVs of this run (no number is typed by hand), the definition of every table, self-checks.

A15 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import glob  # noqa: E402
import os  # noqa: E402

import a15lib as L  # noqa: E402

TOLS = L.TOLS
# Labels (format- / epoch- / recipe-dependent, r = 5 differs) and the A15 sentences come from the registered and the
# claim-rule verdicts only (10/27 20:58 amendment: Welch df is a sensitivity column; own variance enters through the
# claim rule). Own-variance and Welch changes are printed apart, under a sensitivity heading, and set no label.
LABEL_RULES = (("reg", "registered"), ("claim", "claim rule"))
SENS_RULES = (("own", "own variance (A3.1)"), ("welch", "Welch df"))
RULES = LABEL_RULES + SENS_RULES
LABEL_READINGS = ("reg", "claimA", "claimB")   # family readings of the registered rule and the claim rule
SENS_READINGS = ("own", "welch")
SENS_HEAD = "- sensitivity changes (own variance / Welch df; not a label):"


def rd(out, name):
    p = os.path.join(out, name)
    return L.read_csv(p) if os.path.exists(p) else []


def f4(x):
    try:
        return f"{float(x):+.4f}"
    except (TypeError, ValueError):
        return "n/a"


def iv(r, pre, lo="lo", hi="hi"):
    return f"[{f4(r.get(pre + lo))}, {f4(r.get(pre + hi))}]"


def vkey(pre, rule, t):
    return f"{pre}claim@{t:g}" if rule == "claim" else f"{pre}{rule}_verdict@{t:g}"


def changes(rows, label, pre_a, pre_b, cell=lambda r: f"{r['fam']}/{r['dataset']}/{r['kd']}/{r['attack']}",
            rules=LABEL_RULES):
    """Lines 'cell [rule] delta: A -> B' for every rule of ``rules`` (default: registered and claim rule, the rules
    that set a label) and delta where the label column is set."""
    out = []
    for r in rows:
        for rule, name in rules:
            ds = r.get(f"{label}_{rule}") or ""
            if ds in ("", "n/a"):
                continue
            parts = []
            for d in ds.split(","):
                t = float(d)
                parts.append(f"@{t:g} {r.get(vkey(pre_a, rule, t))} -> {r.get(vkey(pre_b, rule, t))}")
            out.append(f"- {cell(r)} [{name}]: " + "; ".join(parts)
                       + f" (k {f4(r.get(pre_a + 'k'))} -> {f4(r.get(pre_b + 'k'))})")
    return out


def bat_changes(rows, label, pre_a, pre_b, rules=LABEL_RULES):
    out = []
    for r in rows:
        for rule, name in rules:
            ds = r.get(f"{label}_{rule}") or ""
            if ds in ("", "n/a"):
                continue
            parts = [f"@{float(d):g} {r.get(f'{pre_a}{rule}_decision@{float(d):g}')} -> "
                     f"{r.get(f'{pre_b}{rule}_decision@{float(d):g}')}" for d in ds.split(",")]
            out.append(f"- battery {r['fam']}/{r['dataset']}/{r['kd']} [{name}]: " + "; ".join(parts))
    return out


def holm_changes(rows, label, readings=LABEL_READINGS):
    out = []
    for r in rows:
        for rule in readings:
            v = r.get(f"{label}_{rule}") or ""
            if v:
                out.append(f"- {r['fam']}/{r['dataset']}/{r['kd']}/{r['attack']} [{rule}]: {v}")
    return out


def sensitivity(cell_rows, bat_rows, holm_rows_, label, pre_a, pre_b):
    """The own-variance and Welch-df changes (cells, batteries, family readings) under their own heading."""
    ch = (changes(cell_rows, label, pre_a, pre_b, rules=SENS_RULES)
          + bat_changes(bat_rows, label, pre_a, pre_b, rules=SENS_RULES)
          + holm_changes(holm_rows_, label, SENS_READINGS))
    return [SENS_HEAD] + (["  " + c for c in ch] if ch else ["  none"])


def sentence_lines(rows, what):
    """Counts of the CSV's sentence column and the cells whose sentence is not the robust one."""
    cnt = {}
    for r in rows:
        cnt[r.get("sentence", "")] = cnt.get(r.get("sentence", ""), 0) + 1
    out = [f"- sentence column of {what}: " + "; ".join(f"{n} x \"{k}\"" for k, n in sorted(cnt.items()))]
    return out


def checks_line(out, name):
    rows = rd(out, name)
    if not rows:
        return f"{name}: not written"
    bad = [r["check"] for r in rows if r.get("ok") not in ("True",)]
    return f"{name}: {len(rows) - len(bad)}/{len(rows)} True" + (f" (not True: {'; '.join(bad)})" if bad else "")


def findings(out):
    lines = []
    inv = rd(out, "inventory.csv")
    if inv:
        lines += ["### Jobs (inventory.csv)", ""]
        for part in ("a", "b1", "b2", "c", "d", "e", "e2"):
            js = [j for j in inv if j["part"] == part]
            cnt = {}
            for j in js:
                c = j["classification"].split(":")[0].split(" (")[0]
                cnt[c] = cnt.get(c, 0) + 1
            lines.append(f"- part {part}: {len(js)} jobs, " + ", ".join(f"{v} {k}" for k, v in sorted(cnt.items())))
        cut = [j["job"] for j in inv if j["classification"].startswith("cut")]
        lines.append(f"- cut by hand-written .blocked markers: {', '.join(cut) if cut else 'none'}")
        exc = [f"{j['job']} ({j['status']})" for j in inv if j.get("analysis_input") == L.EXCLUDED_TEXT]
        lines.append(f"- inputs {L.EXCLUDED_TEXT}: {', '.join(exc) if exc else 'none'}")
        lines.append("")
    # a
    a = rd(out, "a_cells.csv")
    if a and "fam" in a[0]:
        ch = changes(a, "format_dependent", "frozen_", "ws_")
        inc = [f"{r['fam']}/{r['dataset']}/{r['kd']}/{r['attack']} (ws seeds {r.get('ws_seeds_available') or 'none'})"
               for r in a if r.get("ws_complete") != "True"]
        lines += ["### A15-a: whitespace-preserving insertion", ""]
        lines += ([f"- every {'ws-complete ' if inc else ''}decoder cell and delta: same registered and claim-rule "
                   "verdict -> \"삽입 형식에 강건\""]
                  if not ch else ["- format-dependent cells (frozen -> ws; registered / claim rule):"]
                  + ["  " + c for c in ch])
        lines += sentence_lines(a, "a_cells.csv")
        if inc:
            lines.append("- ws re-score incomplete (official vectors kept for every seed; no sentence): " + ", ".join(inc))
        ab = rd(out, "a_battery.csv")
        ah = rd(out, "a_holm.csv")
        lines += bat_changes(ab, "format_dependent", "frozen_", "ws_")
        hc = holm_changes(ah, "format_dependent")
        lines += ["- 31-cell Holm / family claims (reg, claimA, claimB) with the ws vectors: "
                  + ("no change" if not hc else "changes:")]
        lines += ["  " + h for h in hc]
        lines += sensitivity(a, ab, ah, "format_dependent", "frozen_", "ws_")
        lines.append("- " + checks_line(out, "check_a.csv"))
        bc = rd(out, "a_bert_control.csv")
        if bc and "unchanged" in bc[0]:
            lines.append(f"- BERT control: {sum(r['unchanged'] == 'True' for r in bc)}/{len(bc)} tags unchanged")
        lines.append("")
    # b1
    b = rd(out, "b1_cells.csv")
    if b and "fam" in b[0]:
        ch = changes(b, "epoch_dependent", "frozen_", "em_")
        n_in = sum(not (r.get("sentence") or "").startswith("n/a") for r in b)
        lines += ["### A15-b1: teacher-epoch-matched C (H2: these verdicts set the interpretation)", ""]
        lines += ([f"- every A15-b1 cell ({n_in} cells: C epoch-matched or pool recomputed) and delta: same "
                   "registered and claim-rule verdict -> \"teacher epoch에 강건\""]
                  if not ch else ["- epoch-dependent cells (frozen -> epoch-matched; registered / claim rule):"]
                  + ["  " + c for c in ch])
        lines += sentence_lines(b, "b1_cells.csv")
        bm = [f"{r['fam']}/{r['dataset']}/{r['kd']}/{r['attack']}" for r in b
              if (r.get("sentence") or "").startswith("n/a (b1 C_e(P) missing")]
        if bm:
            lines.append("- b1 C_e(P) missing for every seed (frozen C kept; no sentence): " + ", ".join(bm))
        po = [f"{r['fam']}/{r['dataset']}/{r['kd']}/{r['attack']}" for r in b if r.get("pool_only_change") == "True"
              and any(r.get(f"epoch_dependent_{x}") for x, _ in LABEL_RULES)]
        if po:
            lines.append("- of these, changed only through the recomputed pool (k unchanged): " + ", ".join(po))
        bb = rd(out, "b1_battery.csv")
        bh = rd(out, "b1_holm.csv")
        lines += bat_changes(bb, "epoch_dependent", "frozen_", "em_")
        hc = holm_changes(bh, "epoch_dependent")
        lines += ["- 31-cell Holm / family claims (reg, claimA, claimB) with the 13 cells replaced: "
                  + ("no change" if not hc else "changes:")]
        lines += ["  " + h for h in hc]
        lines += sensitivity(b, bb, bh, "epoch_dependent", "frozen_", "em_")
        for r in rd(out, "check_b1.csv"):
            if r.get("check", "").startswith("F31 family size of the em frame"):
                lines.append(f"- F31 family of the em frame (gate re-evaluated on the C_e teacher): {r.get('note')}"
                             + ("" if r.get("ok") == "True" else " -- NOT 31: the family counts below use a smaller family"))
        for r in rd(out, "b3_condition.csv"):
            if r.get("kd") == "logit":
                lines.append(f"- b3 condition (BERT BadNets, |C_e4 - C_e3| 95% bound, own variance): "
                             f"{f4(r.get('own_abs_upper95'))} vs 0.01 -> "
                             f"{'TRIGGERED' if r.get('b3_would_be_triggered') == 'True' else 'not triggered'} "
                             f"(pool-formula bounds: em {f4(r.get('em_pool_abs_upper95'))}, frozen "
                             f"{f4(r.get('frozen_pool_abs_upper95'))}; {r.get('rule_agreement')}); nothing was enabled")
        tw = [r for r in rd(out, "b1_repro.csv") if r.get("run") == "(twin check)"]
        if tw:
            lines.append("- twin checks: " + "; ".join(f"{r['twin']} identical={r.get('identical')}" for r in tw))
        lines.append("- " + checks_line(out, "check_b1.csv"))
        lines.append("")
    # b2
    r33 = rd(out, "b2_reference_atba.csv")
    if r33 and "kd" in r33[0]:
        lines += ["### A15-b2: epoch-matched references and A2", ""]
        for r in r33:
            lines.append(f"- A3.3 {r['reference_set']}, {r['kd']}: k {f4(r.get('k'))}, ref interval "
                         f"{iv(r, 'ref_')} (df {f4(r.get('ref_df'))}), verdicts "
                         + "/".join(r.get(f'ref_verdict@{t:g}', '') for t in TOLS)
                         + "; claim " + "/".join(r.get(f'ref_claim@{t:g}', '') for t in TOLS)
                         + f"; sentence@0.05: {r.get('sentence@0.05', '')}")
        for r in rd(out, "b2_reference_k_summary.csv"):
            if r.get("reference_dependent"):
                lines.append(f"- item 7 ({r['precision']}) {r['trigger']}/{r['kd']} reference-dependent at "
                             f"{r['reference_dependent']} (k {f4(r.get('k_min'))} to {f4(r.get('k_max'))})")
        for r in rd(out, "b2_a2.csv"):
            if "k_em" in r:
                lines.append(f"- A2 ({r['pool']}): k'_em {f4(r['k_em'])} {iv(r, '')}; readings "
                             + "/".join(r.get(f'reg_reading@{t:g}', '') for t in TOLS)
                             + "; claim " + "/".join(r.get(f'claim_reading@{t:g}', '') for t in TOLS)
                             + f" (frozen k' {f4(r.get('k_prime_frozen'))})")
        lines.append("- " + checks_line(out, "check_b2.csv"))
        lines.append("")
    # c
    g = rd(out, "c_gate.csv")
    if g:
        lines += ["### A15-c: GPT-2 SST-2 ATBA", "", f"- gate: {g[0].get('reading')} (ASR {f4(g[0].get('asr'))}, "
                                                    f"accuracy drop {f4(g[0].get('accuracy_drop'))})"]
        for r in rd(out, "c_cells.csv"):
            if r.get("attack") == "atba":
                lines.append(f"- ATBA cell ({r['reading']}): k {f4(r.get('k'))} {iv(r, '')}, registered "
                             + "/".join(r.get(f'reg_verdict@{t:g}', '') for t in TOLS) + ", claim "
                             + "/".join(r.get(f'claim@{t:g}', '') for t in TOLS))
        for r in rd(out, "c_battery.csv"):
            lines.append(f"- auxiliary 3-attack battery ({r['reading']}): registered "
                         + "/".join(r.get(f'aux3_reg_decision@{t:g}', '') for t in TOLS) + "; claim "
                         + "/".join(r.get(f'aux3_claim_decision@{t:g}', '') for t in TOLS)
                         + " (registered two-attack: " + "/".join(r.get(f'registered2_reg_decision@{t:g}', '')
                                                             for t in TOLS) + ")")
        lines.append("- " + checks_line(out, "check_c.csv"))
        lines.append("")
    # d
    d = rd(out, "d_cells.csv")
    if d and "fam" in d[0]:
        ch = changes(d, "r5_differs", "r3_", "r5_")
        lines += ["### A15-d: BERT AG News r = 5 (post hoc)", ""]
        lines += (["- no registered or claim-rule verdict differs between r = 3 and r = 5"] if not ch else
                  ["- r = 3 -> r = 5 (registered / claim rule):"] + ["  " + c for c in ch])
        for r in d:
            marks = {t: r.get(f"pooling_mark@{t:g}") for t in TOLS}
            if any(m and m.startswith(("drop", "keep")) for m in marks.values()):
                lines.append(f"- pooling-dependent mark {r['kd']}/{r['attack']}: " +
                             "; ".join(f"@{t:g} {m}" for t, m in marks.items() if m and m.startswith(("drop", "keep"))))
        db = rd(out, "d_battery.csv")
        dh = rd(out, "d_holm.csv")
        lines += bat_changes(db, "r5_differs", "r3_", "r5_")
        hc = holm_changes(dh, "r5_differs")
        lines += ["- 31-cell Holm / family claims (reg, claimA, claimB) with BERT AG News logit at r = 5: "
                  + ("no change" if not hc else "changes:")]
        lines += ["  " + h for h in hc]
        lines += sensitivity(d, db, dh, "r5_differs", "r3_", "r5_")
        lines.append("- " + checks_line(out, "check_d.csv"))
        lines.append("")
    # e
    e = rd(out, "e_cells.csv")
    if e and "part" in e[0]:
        lines += ["### A15-e / e2: KD recipe alpha", ""]
        for r in e:
            pre = f"a{float(r['alpha']):g}_" if r["part"] == "e" else "a1_"
            lines.append(f"- {r['part']} alpha {r['alpha']} {r['fam']}/{r['dataset']}/{r['attack']}: k {f4(r.get(pre + 'k'))}, "
                         "registered " + "/".join(r.get(f'{pre}reg_verdict@{t:g}', '') for t in TOLS) + ", claim "
                         + "/".join(r.get(f'{pre}claim@{t:g}', '') for t in TOLS) + f" -> {r.get('sentence')}"
                         + (f" (recipe-dependent vs the like-for-like alpha 0.5 reading: reg "
                            f"{r.get('recipe_dependent_reg') or '-'}, claim {r.get('recipe_dependent_claim') or '-'})"
                            if r.get("sentence") != "α 0.5–1.0에서 같다" else "")
                         + f"; against the registered alpha 0.5 row: {r.get('sentence_vs_registered')}"
                         + (f" (reg {r.get('recipe_dependent_vs_registered_reg') or '-'}, claim "
                            f"{r.get('recipe_dependent_vs_registered_claim') or '-'})"
                            if r.get("sentence_vs_registered") != "α 0.5–1.0에서 같다" else ""))
        ec = []
        for r in e:   # per row: like-for-like comparator prefix -> alpha prefix
            pa, pb = (("a0.5_twoattack_", f"a{float(r['alpha']):g}_") if r["part"] == "e" else ("a0.5_em_", "a1_"))
            ec += changes([r], "recipe_dependent", pa, pb, rules=SENS_RULES,
                          cell=lambda x: f"{x['part']} alpha {x['alpha']} {x['fam']}/{x['dataset']}/{x['attack']}")
        for r in rd(out, "e_battery.csv"):
            if "part" not in r:
                continue
            pa, pb = (("a0.5_twoattack_", f"a{float(r['alpha']):g}_") if r["part"] == "e" else ("a0.5_em_", "a1_"))
            lines += [f"  {x} ({r['part']} alpha {r['alpha']}; like-for-like alpha 0.5 -> alpha)"
                      for x in bat_changes([r], "recipe_dependent", pa, pb)]
            ec += [f"{x} ({r['part']} alpha {r['alpha']})"
                   for x in bat_changes([r], "recipe_dependent", pa, pb, rules=SENS_RULES)]
        lines += [SENS_HEAD] + (["  " + c for c in ec] if ec else ["  none"])
        for r in rd(out, "e_transmission.csv"):
            lines.append(f"- transmission {r['part']} alpha {r['alpha']} ({r['reading']}) {r['fam']}/{r['dataset']}/"
                         f"{r['attack']}: follow_P {f4(r.get('follow_P_untrig'))}, match_C {f4(r.get('match_C_untrig'))}, "
                         f"gap {f4(r.get('gap_untrig'))} [{f4(r.get('gap_lo'))}, {f4(r.get('gap_hi'))}], match_R "
                         f"{f4(r.get('match_R_untrig_s42'))} (|D| {r.get('n_D_untrig')})")
        lines.append("- " + checks_line(out, "check_e.csv"))
        lines.append("")
    # claim counts
    cl = rd(out, "b1_claims.csv") + rd(out, "a_claims.csv") + rd(out, "d_claims.csv")
    if cl and "frame" in cl[0]:
        lines += ["### Claim-rule counts (qualified cells; claimable = seed-pooled)", "",
                  "| frame | delta | claimable | claim EQU | claim POS | rule-dependent | one-seed | structural | "
                  "F31 no-transfer claimB | F31 transfer claimB | F19 no-transfer claimB |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        seen = set()
        for r in cl:
            k = (r["frame"], r["delta"])
            if k in seen:
                continue
            seen.add(k)
            lines.append(f"| {r['frame']} | {r['delta']} | {r['n_claimable']} | {r['claim_EQUIVALENT']} | "
                         f"{r['claim_POSITIVE']} | {r['claim_rule_dependent']} | {r['n_one_seed']} | {r['n_structural']} | "
                         f"{r.get('F31_notransfer_claimB_nonstructural_n')} | {r.get('F31_transfer_claimB_nonstructural_n')} | "
                         f"{r.get('F19_notransfer_claimB_nonstructural_n')} |")
        lines.append("")
    return lines


def main():
    ns = L.args()
    out = ns.out
    sc = rd(out, "selfcheck.csv")
    ind = rd(out, "independent_check.csv")
    inv_in = rd(out, "inventory_inputs.csv")
    n_in = sum(r.get("present") == "True" for r in inv_in)
    # A15 job outputs the analysis reads: present and (Stage 2 / ws files) written by a terminal-ok job; the CPU
    # fp32 hits written by score_em.py during this run are not A15 job outputs and are left out of the snapshot
    read = [os.path.join(L.ROOT, r["file"]) for r in inv_in if r.get("present") == "True" and r.get("job_ok") != "False"
            and "score_em.py" not in r.get("kind", "")]
    read = [p for p in read if os.path.exists(p)]
    newest = max(read, key=os.path.getmtime) if read else None
    code = sorted(glob.glob(os.path.join(L.HERE, "*.py")) + glob.glob(os.path.join(L.HERE, "*.sh")))
    lines = [
        "# A15 analysis outputs (scripts/tifs/aux/g8-a15/analysis)", "",
        f"**Status.** {L.STATUS}. H1 (10/27 20:58 KST amendment): the frozen 31-cell primary verdicts, battery "
        "decisions and the paper-wide Holm stay as registered (results/tifs/report, results/tifs/addendum). Every "
        "table here is post hoc and auxiliary.", "",
        "## Data snapshot", "",
        f"- written {L.now_utc()}; runs directory `{L.rel(ns.runs)}`; registered deadline {L.DEADLINE_UTC} "
        "(10/29 14:00 KST; cut order e2 -> e -> d, by .blocked markers).",
        f"- inputs present: {n_in}/{len(inv_in)} files listed in inventory_inputs.csv ({len(read)} read)"
        + (f"; newest A15 input read (mtime) {L.mtime_utc(newest)} (`{L.rel(newest)}`)." if newest else "."),
        f"- inputs {L.EXCLUDED_TEXT}: "
        + (", ".join(sorted({r['run'] for r in inv_in if r.get('present') == 'True' and r.get('job_ok') == 'False'}))
           or "none") + " (never read; inventory.csv analysis_input).",
        "- code: " + ", ".join(f"`{os.path.basename(p)}` {L.sha256(p)[:12]}" for p in code),
        "- frozen files: sha256 checked against the freeze records before and after the run (selfcheck.csv).", "",
        "## Binding rules (docs/tifs_addendum.md, 10/27 20:58 KST amendment)", "",
        "- H1: the registered verdicts stay in the tables. H2: the epoch-matched verdicts (b1/b2) set the "
        "interpretation; ws (a) differences are marked format-dependent; r = 5 (d) is written beside r = 3 and is "
        "post hoc; alpha (e, e2) differences are marked recipe-dependent. H3: the claim rule decides every "
        "cell-level or family-level claim.",
        "- Claim rule: a cell's verdict is claimed only when the registered verdict (shared pool, Satterthwaite df) "
        "and the A3.1 own-variance verdict (s2_a / r + v, df r - 1) agree; otherwise 'rule-dependent' (counted as "
        "INCONCLUSIVE). One-seed (transported) cells: 'one-seed, descriptive', not counted. Structural nulls apart. "
        "Family: Holm over F31 and F19 of p_max = max(p_registered, p_own), one-sided 0.025; p_up for no transfer, "
        "p_pos for transfer (claimA: one-seed cells at p_registered, their rejections descriptive; claimB: one-seed "
        "cells at p = 1). Welch df: sensitivity column only.", "",
        "## Findings (generated from the CSVs of this run)", ""] + findings(out) + [
        "## Tables", ""]
    for p in sorted(glob.glob(os.path.join(out, "*.csv"))):
        with open(p) as f:
            first = f.readline().strip()
        d = first[len("# definition: "):] if first.startswith("# definition: ") else first
        d = d.split(" Status: ")[0]
        lines.append(f"- `{os.path.basename(p)}`: {d}")
    lines += ["", "Column conventions (every comparison table): `<frame>_` prefix (frozen_, ws_, em_, r3_, r5_, "
              "a<alpha>_ ...); `reg_` registered rule, `own_` A3.1 own variance, `claim@d` claim rule, `pmax_` "
              "max(p_registered, p_own), `welch_` Welch df, `F31_/F19_<side>_<reading>` family claims; "
              "`<label>_<rule>` = the deltas where the A15 verdict differs from the reference one under that rule; "
              "`a15_jobs` = the part's queue v9 job status at analysis time (status markers; cut / blocked / missing "
              "jobs listed).",
              "", "## Self-checks", ""]
    if sc:
        bad = [r for r in sc if r.get("ok") != "True"]
        info = [r for r in sc if r.get("kind") == "info" and r.get("got") not in ("True",)]
        lines.append(f"- selfcheck.csv: {len(sc)} checks, {len(bad)} failed, {len(info)} completeness rows not True")
        for r in info:
            lines.append(f"  - info: {r['check']} ({r.get('note', '')})")
        for r in bad:
            lines.append(f"  - FAILED: {r['check']} ({r.get('note', '')})")
    if ind:
        lines.append(f"- independent_check.csv: {len(ind)} comparisons, {sum(r['ok'] == 'True' for r in ind)} equal, "
                     f"{sum(r['ok'] == 'False' for r in ind)} differ, {sum(r['ok'] == '' for r in ind)} not applicable")
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {L.rel(os.path.join(out, 'README.md'))}")


if __name__ == "__main__":
    main()
