# g8-a15/analysis: the A15 analysis code

**Status.** A15 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.
Code for docs/tifs_addendum.md "## A15" (registration 10/26 19:15 KST, parts a, b1, b2, the b3 condition, c, d,
e), its "보정 (10/26 19:55 KST)" (e2), and item 5 of "보정 (10/27 20:58 KST, A15 분석 전): 마감 연장, 판정 위계,
claim rule" (binding: hierarchy H1-H3 and the claim rule). Outputs go to `results/tifs/addendum/aux/g8-a15/` when
it is run for real; **it has not been run on A15 data**.

**Blinding.** Written and tested on 2026-10-27 without opening, loading, printing or summarising any A15 student or
teacher hit vector, prediction file, result JSON, kd_meta accuracy, any file under `runs/tifs/tifs_x15_*/artifacts/
results`, or any `runs/tifs/tifs_x15_ws/*` prediction / official / ws / compare JSON. Read for the design: the
implementer's description (`../README.md`), its code (`ws_score.py`, `check_twin.py`, `gate_c.py`,
`plan_check.py` and the pre-data `plan_check.json`), `scripts/tifs/queue_plan.py` (v9 `a15_parts`), the A15
configs, and A15 file *names* and status markers (`runs/tifs/_status/*`). The `--dry-run` reads names and markers
only. Tests used the frozen runs and synthetic fixtures (`make_fixtures.py`) in the session scratchpad. The
10/27 revision and the 10/30 revision (nine review issues each, "Revision log" below) kept the same rule: no A15
output was read. On 10/30 every A15 GPU job had finished (121 jobs ok, 2026-10-28 11:10 UTC, none cut); the 10/30
revision was still done blind (A15 file names and status markers only).

## Data snapshot (inputs of the tests; the real run records its own in the output README)

- Frozen code: sha256 of `scripts/tifs/analyze.py` 1f70bbf4..., `scripts/tifs/addendum.py` d49a912a...,
  `counterfactual_kd/analysis/tifs.py` cc0e0545..., `counterfactual_kd/metrics/paired.py` aed57ad5...,
  `scripts/tifs/cross_score.py` 4bd94f6d..., `docs/tifs_plan.md` bc9dfd8f..., `results/tifs/structural_nulls.json`
  a6dc5091... (full digests in `a15lib.FROZEN_SHA256`), equal to the freeze records
  `runs/tifs/_code/freeze_20261021T092041Z.sha256` (c8ba47f3...) and `final_analysis_20261024T110542Z.sha256`
  (d54317c5...), checked before and after every test run.
- Official tables reproduced by the gate (sha256 prefix, mtime): `results/tifs/report/primary.csv` a5a562b5
  (10/24 11:04:46 UTC), `variance.csv` 76c863b5, `battery.csv` a9eddb32 (11:04:46); `results/tifs/addendum/
  global_holm.csv` f0ec8a81, `attack_variance.csv` 9e8d710e, `reference_atba.csv` cd7f2b0e, `replicate.csv`
  667605ca (11:05:09 UTC).
- Frozen predictions: every `runs/tifs/tifs_*` run the frozen loader parses (10/30: 116 prediction files of 24
  runs, newest 2026-10-24 07:17:09 UTC, before the final-analysis snapshot 2026-10-24 11:05:42 UTC; selfcheck.csv
  prints the count and the newest mtime); the A13 G1 runs
  `tifs_x13_qwen3_agnews_s43/_s44`; `scripts/tifs/aux/g8-a15/plan_check.json` 0c509e57 (10/26 10:52:48 UTC,
  pre-data); `results/tifs/addendum/aux/g10-claimrule/frozen_claim_family.csv` d793c99b (10/27 13:32:25 UTC) for
  the claim-count cross-check.
- A15: file names and status markers only (fixtures are built from the frozen files). 10/30 dry-run on the real
  runs directory: 121 queue v9 jobs, all done (a 12, b1 36, b2 34, c 11, d 4, e 12, e2 12); every Stage 2, ws,
  twin-check and gate input present; none excluded; the 17 CPU fp32 hit files are written by score_em.py at run
  time. The real run's output README records the newest A15 input it reads (mtime).

## Run

```
bash run_all.sh --dry-run     # inputs, cut / blocked / missing jobs, CPU scoring plan; writes nothing
bash run_all.sh               # the registered single run (CPU only; nice -n 10; no bytecode)
```

Options: `--runs DIR --out DIR --teachers-x15 DIR --hits-dir DIR --no-score --force` (and `--now`, tests only: the
UTC time the readiness check compares with the deadline). `run_all.sh` first runs
`inventory.py --check-ready`: while an A15 job is running or pending and the deadline (10/29 14:00 KST = 05:00 UTC)
has not passed, it exits 4 **before writing or reading anything** (the analysis is registered to run once, when
A15 is done or at the deadline; `--force` overrides). Only inputs of terminal-ok jobs are read: an A15 run's
Stage 2 files when `runs/tifs/_status/<run>__eval.ok` exists, a run's re-scores when
`_status/tifs_x15_ws__<run>.ok` exists; files of a running, failed or cut job are never opened ("present but job
not terminal-ok: excluded", listed in `inventory.csv` column `analysis_input` and in every `check_*.csv`). Then:
frozen/official/aux file hashes recorded
(`selfcheck.py --record-before`), `inventory.py`, `score_em.py --recheck` (CPU fp32 scoring, about 20 BERT-base
students x 1-2 triggers, a few minutes), `part_a.py`, `part_b1.py`, `part_b2.py`, `part_c.py`, `part_d.py`,
`part_e.py`, `independent.py`, `selfcheck.py`, `make_readme.py` (writes the output README with the findings of the
run, generated from the CSVs). Logs: `<out>/logs/`.

## Files

| file | what |
|---|---|
| `a15lib.py` | shared library: frozen imports, the frozen loader for A15 runs (RUN_RE swapped for the call only) and for the ws re-scores (symbolic-link view in the Stage 2 layout), the gate, the **frame engine** (frozen `primary_table` -> `primary_battery` -> `global_holm_table` on a cell list) and `annotate` (own variance, claim rule, Welch df, family Holm readings, battery rules, A4 class), comparison / battery / Holm table builders, claim counts, A8 floor, job markers |
| `inventory.py` | job inventory (queue v9) and input listing; `--dry-run`, `--check-ready`; `inventory.csv`, `inventory_inputs.csv` |
| `score_em.py` | CPU fp32 scoring of the epoch-matched C_e (b1, seed 42) and C' (b2) students on the triggers of their epoch with the registered A14 engine (`g6-review0/reference/score.py` `Scorer`, unchanged); `--recheck` re-scores the frozen C'(t43) and requires its A14 hits |
| `part_a.py` | A15-a ws re-scores: `a_cells`, `a_battery`, `a_holm`, `a_g1`, `a_contaminated`, `a_bert_control`, `a_vectors`, `a_shift`, `a_claims`, `check_a` |
| `part_b1.py` | A15-b1 epoch-matched batteries, selection effect, b3 condition: `b1_cells`, `b1_battery`, `b1_holm`, `b1_pools`, `b1_selection`, `b3_condition`, `b1_repro`, `b1_claims`, `b1_floor`, `check_b1` |
| `part_b2.py` | A15-b2 references at e(P), A3.3, A14 item 7, A2: `b2_references`, `b2_reference_atba`, `b2_reference_k`, `b2_reference_k_summary`, `b2_a2`, `b2_floor`, `check_b2` |
| `part_c.py` | A15-c GPT-2 ATBA: `c_gate`, `c_cells`, `c_battery`, `c_claims`, `c_floor`, `check_c` |
| `part_d.py` | A15-d r = 5: `d_cells`, `d_battery`, `d_holm`, `d_pools`, `d_claims`, `d_floor`, `check_d` |
| `part_e.py` | A15-e / e2 alpha and transmission: `e_cells`, `e_battery`, `e_transmission`, `e_claims`, `e_floor`, `check_e` |
| `independent.py` | independent re-computation (numpy / scipy only, no frozen module, no a15lib) of the key numbers from the raw files: `independent_check.csv` |
| `selfcheck.py` | frozen hashes (full sha256 digests) vs the constants and the freeze records, files unchanged during the run, the gate, RUN_RE restored (same object and content after the A15 loaders ran, also after an exception inside a swap), the frozen pool_welch identity and family sizes, the F31 size of the ws / em / r5 frames, loader inputs predate the final-analysis snapshot, A14 item 1 and g10 counts reproduced, every check file, CSV format: `selfcheck.csv` |
| `make_readme.py` | the output README (status, snapshot, binding rules, findings from the CSVs, table definitions, self-checks) |
| `make_fixtures.py`, `test_analysis.py` | synthetic A15 layouts (identity / perturb / partial) and the assertions run on them (tests only; they write to a scratch directory) |

Every CSV starts with a `# definition:` line (with the status sentence); `selfcheck.py` checks it.

## Definitions

**Frames.** A frame is the frozen pipeline run on a list of cells in the frozen cell format:
`analyze.reference_variance` -> `analyze.primary_table` (pools, seed-pooled / transported / structural rows) ->
`analyze.primary_battery` (within-battery decisions, `transfers_global`) -> `addendum.global_holm_table`. On the
frozen cells it reproduces primary.csv, variance.csv, battery.csv and global_holm.csv exactly (gate, every script).
- `frozen` / `r3_`: the frozen cells.
- `ws_` (a): decoder battery cells (GPT-2 SST-2 and Qwen3 SST-2 seeds 42-44, Qwen3 AG News seed 42) with the ws
  hit vectors; the teacher gate stays the frozen one (g8-a15 README: ws teacher numbers are descriptive). A cell
  gets its ws vectors only when every seed of it has a ws re-score; otherwise all its seeds keep the official
  vectors (`ws_complete` False, `ws_seeds_substituted` empty, sentence n/a). The same rule in the A13 G1 ws frame;
  the A15-c ws reading exists only when every A15-c cell and every GPT-2 SST-2 battery cell is ws.
- `em_` (b1): the 13 SST-2 base-trainer cells with C := C_e(P) (P vectors and P teacher frozen; the C teacher and
  its gate numbers from the b1 run). GPT-2 BadNets (e(P) = 2 = frozen C epoch) keeps the frozen C. Every pool is
  recomputed, so a cell can change through its pool only (`pool_only_change`: Qwen3 AG News borrows the Qwen3
  SST-2 pool; BERT SST-2 SOS uses the BERT SST-2 pool).
- `r5_` (d): the frozen cells plus seeds 45/46 of the BERT AG News logit battery (post hoc).
- c: frozen cells (`official`), em cells (`em`) or ws cells (`ws`) plus the A15-c ATBA cells: the GPT-2 battery
  then has three attacks and its pool is recomputed over them (auxiliary battery, outside the 31-cell family).
- e / e2: the alpha cells alone (two attacks, their own pool); comparators (like for like first): e at alpha
  0.5 = the same two attacks with a two-attack pool (`recipe_dependent_*`, `sentence`) and, beside it, the frozen
  primary rows (six-attack pool, df 9.83; `recipe_dependent_vs_registered_*`, `sentence_vs_registered`;
  `a05_pool_design_differs_*` = where the two readings differ at alpha 0.5 itself); e2 at alpha 0.5 = the
  epoch-matched b1 GPT-2 rows (`recipe_dependent_*`, `sentence`) and the frozen rows (`_vs_registered_`).

**Rules per cell and delta in {0.02, 0.05, 0.10}.**
- `reg_`: the registered rule (seed-pooled: k = mean_s k_s, SE^2 = s2_pool/r + v, Satterthwaite pool df;
  transported: SE^2 = SE_Tango^2 + s2_pool; structural: Tango). Verdict from the two-sided 95% interval
  (EQUIVALENT inside [-d, d], POSITIVE above d, NEGATIVE below -d, else INCONCLUSIVE); `p_up` = one-sided p of
  H0 k >= d, `p_pos` of H0 k <= d.
- `own_`: A3.1 own variance, SE^2 = s2_a/r + v, t at df r - 1 (equal to addendum.attack_variance_table, checked).
- `claim@d`: the claim rule (10/27 amendment item 3): the registered verdict when the own verdict agrees, else
  `rule-dependent` (counted as INCONCLUSIVE); one-seed (transported) cells `one-seed, descriptive`; structural
  nulls `structural (counted apart)` (their Tango verdict stays in the `reg_` columns). `pmax_` =
  max(p_registered, p_own).
- `welch_`: the registered k and SE at the Welch-Satterthwaite df of the pool, (sum_a s2_a)^2 (r-1) / sum_a s2_a^2
  (A15 E3; sensitivity only, never part of the claim rule).
- Family claims `F31_/F19_<side>_<reading>(_holm)@d`: frozen `holm` over the qualified cells (31) / their logit
  cells (19), one-sided 0.025; side `notransfer` (p_up) or `transfer` (p_pos); readings (named as in
  `scripts/tifs/aux/g10-claimrule/frozen_claims.py`): `reg`; `own` (one-seed cells p_registered); `claimA`
  (p_max; one-seed cells at p_registered, their rejections reported as descriptive and not counted); `claimB`
  (p_max; one-seed cells at p = 1); `welch`. Structural nulls enter with their Tango p.
- Battery decisions: registered = `primary_battery`; `own_` / `claim_` / `welch_decision` = the frozen
  `analyze.decide` on those p-values (one-seed cells p = 1 for own and claim, as claimB); a battery whose
  non-structural cells are all one-seed is `one-seed, descriptive` under own and claim.
- Labels: `format_dependent_<rule>` (a), `epoch_dependent_<rule>` (b1, b2), `r5_differs_<rule>` (d),
  `recipe_dependent_<rule>` (e, e2) = the deltas where the verdict differs from the reference frame's under that
  rule (empty = same at every delta). Only `<label>_reg` and `<label>_claim` set a label or an A15 sentence
  (`sentence` in a_cells, b1_cells, e_cells; 10/27 amendment: Welch df is a sensitivity column, own variance enters
  through the claim rule); `<label>_own` and `<label>_welch` are sensitivity columns, printed in the output README
  under "sensitivity changes (own variance / Welch df; not a label)". b1 `sentence` mirrors a's: "teacher epoch에
  강건" unless `epoch_dependent_reg` or `epoch_dependent_claim` is set, else "epoch-dependent (두 판정을 나란히
  쓴다)"; n/a outside A15-b1 (C not replaced, not a registered b1 cell, variance not from a pool that holds
  replaced cells) and for a registered b1 cell whose C_e(P) is missing for every seed (frozen C kept). `a15_jobs`
  = the part's job status (markers), on every verdict table (a_cells, a_battery, a_holm, a_g1, b1_cells,
  b1_battery, b1_holm, b2_reference_atba, b2_reference_k, b2_reference_k_summary, b2_a2, c_cells, c_battery, d_cells, d_battery,
  d_holm, e_cells, e_battery).
- A4 class (`a4_class`): structural / dagger (trigger words in no KD-corpus row) / boundary (SOS SST-2) /
  counted, the rule of A14 item 1 (checked against `claims_cells.csv`).

**Part-specific quantities.**
- `dsel` (b1_selection): Delta_sel = C_e(P) - C_sel (C_sel = the frozen, OB-selected clean twin's student) on the
  cell's triggered inputs = k_frozen - k_em; descriptive, with the frozen seed-pooled formula (epoch-matched and
  frozen pools) and own variance.
- b3 condition (b3_condition): 95% bound of |C_e4 - C_e3| (BERT BadNets logit, seeds 42-44) = max(|lo|, |hi|) of
  the two-sided 95% interval; decision column = own between-seed variance (see Decisions); printed; nothing enabled.
- A3.3 epoch-matched (b2_reference_atba): the frozen `addendum.reference_atba_table` on a proxy state (em groups and
  rows; epoch-1 reference cells t43 (frozen, epoch 1), t44-t46 (b2 e1)); sets t42-t44 (df 2) and t42-t46 (df 4);
  `ref_welch_*` = the same interval at the Satterthwaite df of SE^2 (on the pool's Welch df) and s2_ref
  (sensitivity). b2_references.csv also holds the twin checks of the rebuilt t43-t46 (`run` = '(twin check)');
  check_b2 reports (info) whether the rebuilt t43 is identical to the frozen one and selects epoch 1, the
  condition under which the frozen C'(t43) is a valid epoch-1 reference beside rebuilt-t43 epoch-4/5 references.
- Item 7 epoch-matched (b2_reference_k): k = P(t42, s42) - C_ref(e(P)) per trigger / KD / reference; CPU fp32
  (score_em.py; t43 epoch 1 = the A14 hits) and GPU bf16 Stage 2 (BadNets, ATBA phrase); transported interval with
  the em pool; reference-dependent across functional references.
- A2 epoch-matched (b2_a2): k'_em = P'(t43, e5) - C'(t43, e5), registered A2 formula with the em pool (frozen pool
  as sensitivity), own variance, readings replicated / not replicated / inconclusive, build difference.
- Transmission (e_transmission): the definitions of g9-a15cpu item 4 (D = untriggered inputs where the teachers
  disagree; follow_P, match_C, gap with own-variance interval, match_R of the frozen CE student).
- A8 floor (`*_floor.csv`): Stage 2 clean accuracy > majority share + 0.10; a run with any student at or below it is
  left out of every verdict (A13 practice) and listed.

## Decisions fixed here, before any A15 output was read

The registration leaves these open or ambiguous; each was fixed in code on 10/27 without data:
1. **b3 decision column**: "seed-pooled" bound with the variance of the contrast itself (s2_d/r + v, df r - 1).
   With the battery pool the bound is at least t s_pool/sqrt(r) (t_{0.975,4.67} 0.02166/sqrt(3) = 0.0328 for BERT
   SST-2) whatever C_e4 and C_e3 are, so the registered threshold 0.01 would be exceeded by construction. Both
   pool-formula bounds are shown; `rule_agreement` says whether the readings agree. Note that the own reading has
   df 2 (t = 4.30), so it also exceeds 0.01 unless C_e4 and C_e3 are nearly identical. Needs the author's
   confirmation (open issue). b3_condition.csv's definition line states this decision column.
2. One-seed cells in the family Holm: both g10 readings (claimA, claimB) are reported; battery decisions under
   the claim rule use claimB (p = 1) and the label `one-seed, descriptive`.
3. ws frame: frozen teacher gate; the comparison is against the frozen (official Stage 2) verdict, with the
   official re-score of the same job beside it (`format_dependent_vs_rescore_*`).
4. em frame: P and the P teacher frozen, C and the C teacher (gate numbers) from the b1 run; pools recomputed;
   pool-only changes labelled.
5. c: three readings (official, em, ws); the GPT-2 BadNets / AddSent verdicts of the auxiliary battery are reported
   beside the registered two-attack ones, never instead of them.
6. e comparator for `recipe_dependent_*` and the sentence = the like-for-like alpha 0.5 reading (the same two
   attacks with their own two-attack pool, the design of the alpha frames), so a verdict cannot change through the
   pool and df alone; the registered alpha 0.5 rows (six-attack pool) are compared beside it
   (`recipe_dependent_vs_registered_*`, `sentence_vs_registered`). e2 comparator = the epoch-matched b1 GPT-2 rows
   (same C epoch), the frozen rows beside it. (Revised 10/27, before data; was the registered rows.)
7. A3.3: both reference sets (3, 5); a reference is used only if its teacher at that epoch and its C' pass the
   functional floor (A13 G2 rule).
8. A2: the em BERT SST-2 logit pool as the main pool, the frozen pool as sensitivity.
9. Format-, epoch-, recipe-dependent labels and sentences (and the output README's label lists): set when the
   registered or the claim-rule verdict differs; own-variance-only and Welch-only changes never set a label
   (printed under a separate sensitivity heading).
10. d: pooling-dependent mark dropped when the r = 5 own-variance verdict equals the frozen verdict, else kept.
11. Inputs: only terminal-ok jobs (`<run>__eval.ok`, `tifs_x15_ws__<run>.ok`); a partial file set of a running or
    cut job is never mixed in (it would otherwise give r = 5 for some attacks of a battery and r = 4 for others).
12. ws substitution per cell over all its seeds (never ws and official seeds in one cell; see Frames).
13. Structural nulls under the claim rule: `structural (counted apart)` (g10-claimrule label), not their Tango verdict.
14. F31 family size: checked = 31 in the ws and r5 frames (check_a, check_d, selfcheck: correctness); in the em
    frame (gate re-evaluated on the C_e teacher) it is checked and reported as info with the cells whose
    `teacher_qualified` changes (check_b1, selfcheck; `em_teacher_qualified` in b1_cells.csv), and printed in the b1
    findings; a smaller em family is reported, not fixed.
15. b1 sentence (10/30, before data): a registered b1 cell with no epoch-matched seed at all keeps the frozen C and
    gets no sentence (n/a; listed in the output README), instead of "teacher epoch에 강건" from an unchanged C; a
    cell with some seeds missing is judged on its finished seeds (`em_source` says partial).
16. Output README snapshot (10/30): the newest mtime among the A15 job outputs the analysis reads (present and
    terminal-ok; the hits score_em.py writes during the run are left out), with the number of files read.

## Tests run (10/30 revision, CPU, frozen data and fixtures only)

- Gate on the real frozen runs (116 frozen cells, no tifs_x15_* run among them): the seven official tables
  reproduced exactly by the frozen state and by the frame engine; own-variance verdicts = attack_variance.csv; the
  frozen pool_welch identity (4 pools), family sizes 31 / 19, structural nulls 'structural (counted apart)';
  analyze.RUN_RE untouched.
- `score_em.py --only-recheck` on the frozen C'(t43) checkpoint: hits and inputs identical to the A14 file.
- `test_analysis.py` on four fixture trees (identity, perturb seeds 7 and 11, partial): 151 assertions, all pass;
  selfcheck 69 checks, 0 failed in every tree (12 info rows not True in partial, as designed); independent
  125/125 (identity, perturb x2) and 100/100 (partial). New on 10/30: the sentence columns of a_cells, b1_cells
  and e_cells follow only the reg / claim labels; a b1 cell with every seed failed (partial: GPT-2 AddSent) keeps
  the frozen C with sentence n/a and is listed in the output README; RUN_RE content and exception-path restore;
  F31 size rows of the ws / em / r5 frames; the readiness check is tested with `--now` before and at the deadline
  (the wall clock is past the deadline since 10/29).
- `bash run_all.sh --dry-run` on the real runs directory (names and markers only; writes nothing).
- Hashes of the frozen code, docs/*, the other aux folders and the official / aux result tables (870 files) equal
  before and after the tests; nothing written under runs/tifs or results/.

## Tests run (10/27, CPU, frozen data and fixtures only)

- Gate on the real frozen runs: primary.csv, variance.csv, battery.csv, global_holm.csv, attack_variance.csv,
  reference_atba.csv, replicate.csv reproduced exactly by the frozen state and by the frame engine; annotate's
  own-variance verdicts = attack_variance.csv.
- `test_analysis.py` on three fixtures (54 assertions, all pass): identity (every A15 vector = the frozen one: the
  epoch-matched and ws frames equal the frozen frame in every column of every row; primary.csv, battery.csv and
  global_holm.csv reproduced through the A15 tables; A3.3 with the epoch-1 references = reference_atba.csv and
  g2_reference_atba5.csv; item 7 = A14 item 7 for t42/t43/t44; A2 = replicate.csv; A13 G1 official frame =
  g1_cells.csv; transmission = g9 transmission_cells.csv), perturb (labels appear; cells outside the substituted
  batteries unchanged; self-checks and 125 independent comparisons pass), partial (exit 4 and nothing written
  before the deadline while a job runs; with --force e2 reported cut, d at r = 4, the missing ws run listed, a
  failed b1 seed leaves the SynBkd cell on its finished seeds 42/43 and the pool on the common seeds).
- `score_em.py --only-recheck` on the frozen C'(t43) checkpoint: hits identical to the A14 file (BadNets, ATBA).
- `selfcheck.py`: the registered-rule family counts on the frozen cells equal A14 item 1, and the claimA / claimB /
  own / reg counts equal the parallel g10-claimrule output (48 rows).
- `inventory.py --dry-run` on the real runs directory (names and markers only).

## Revision log

- 10/27 (before data): nine review issues (terminal-ok inputs only, ws substitution per cell, b3 decision column,
  e like-for-like comparator, label rules reg / claim only, structural claim label, Welch columns, a15_jobs,
  selfcheck hardening).
- 10/30 (blind; the A15 jobs had finished on 10/28 but no A15 output was opened): the nine fixes of the second
  blind review checked one by one and completed: (1) labels and sentences from reg / claim only, sensitivity
  heading, b1_cells `sentence` derived from `epoch_dependent_reg` / `_claim` like a's (plus n/a for a b1 cell with
  no epoch-matched seed, Decisions 15), "every ws-complete" / "every A15-b1 cell" wording in the output README,
  d Holm label changes listed; (2) b3_condition definition line as specified; (3) marker-gated loaders and the
  excluded rows; (4) ws substitution only when every seed has a re-score; (5) e sentence from the like-for-like
  comparison, registered-row comparison beside it (Decisions 6); (6) b2 twin rows t43-t46 and the t43 info row;
  (7) ref_welch_* in b2_reference_atba, a15_jobs on a_g1 / a_holm / b1_holm / d_holm (and b2_reference_k);
  (8) selfcheck: RUN_RE identity + content + exception path, full digests, provenance epoch must be recorded,
  pool_welch identity, F31 size of the ws / em / r5 frames; (9) b1_floor definition, structural claim label also in
  the b1_cells definition, newest A15 input read in the output README (Decisions 16), this snapshot,
  g8-a15/*.json tracked. Bug fixed on the way: a b1 cell's missing seeds were matched without the dataset, so an
  AG News cell with the same family / KD / attack (BERT logit SynBkd) was labelled "b1 missing" when a BERT SST-2
  SynBkd seed failed (em_source text only; no number was affected). Tests: `run_all.sh --now` for the deadline
  check.
