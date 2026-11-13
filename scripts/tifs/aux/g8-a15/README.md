# g8-a15: A15 re-experiments (implementation, queue v9)

**Status.** Implementation of docs/tifs_addendum.md **A15** (registered 10/26 19:15 KST,
snapshot `a15_registration_20261026T100903Z`) and its amendment **"보정 (10/26 19:55 KST,
A15 데이터 전): e2 추가"**. Post hoc, outside the primary family: no frozen verdict, battery
decision or paper-wide Holm changes. Nothing here has run on a GPU yet. The queue block is
**disabled** (`A15_ENABLED = False` in scripts/tifs/queue_plan.py) until it is verified.
The analysis (scripts/tifs/aux/g8-a15/ analysis code, outputs in
results/tifs/addendum/aux/g8-a15/) is written later; this file says what it will read.

## Files here

| file | what |
|---|---|
| `make_configs.py` | writes the 65 `configs/tifs/tifs_x15_*.yaml` from frozen templates; `--check` verifies the files on disk |
| `configs_manifest.json` | per A15 run: part, template config, and every key changed from the template |
| `plan_check.py` / `plan_check.json` | CPU dry run: per condition the tag, the P student (reused or trained), the C student, the teacher pair, and the frozen preds file it aligns with |
| `ws_score.py` | A15-a GPU job: official + whitespace-preserving re-scoring of a run's students |
| `check_twin.py` | A15-b CPU job: rebuilt clean twin vs frozen twin (SHA-256, max abs diff, dev metrics) |
| `gate_c.py` | A15-c GPU job: frozen teacher gate for the GPT-2 ATBA teacher; exit 3 blocks the students |
| `estimate.py` / `a15_jobs_estimate.csv` | GPU-h per job (id, part, kind, mem_gb, deps) from g5-cost medians and a 2-slot `lane.pick` simulation; the CSV includes b3 |

## Default-off options (shared pipeline code)

Every behaviour change is behind an option whose default is the historical behaviour.
A15 options are not written into a run's `config.yaml` while at their defaults, so the
snapshot of every existing config re-serialises byte-identically (113 configs and run
snapshots checked).

| option | default | effect when set |
|---|---|---|
| trigger `preserve_whitespace` (WordTrigger, DistributedWordTrigger, SentenceTrigger, SyntacticTrigger; `triggers/whitespace.py`) | False | same random draws, same word index; the trigger is spliced into the original string, all original whitespace kept (SynBkd paraphrase and suffix: original trailing whitespace appended) |
| `control_texts(..., preserve_whitespace)` | False | the same probes, original whitespace kept (substitute mode uses the same spans and draws) |
| `evaluate_model_pair(..., preserve_whitespace, provenance)` | False / None | ws inputs; `meta.provenance` (code fingerprint, SHA-256 of every probe's texts) |
| `stage0.save_every_epoch` | False | every epoch's weights under `<teacher>/epochs/epoch<e>/` (+ `epoch_meta.json`) and `<teacher>/epochs.json`; base trainer only; OB-selected checkpoint still at `<teacher>/` |
| `stage0.ob_ckpt_pid` | False | OB intermediate checkpoint dir `models/checkpoints/pid<pid>/...` |
| `stage0.build_kinds` | None (both) | train only `[clean]` or `[poisoned]` teachers |
| `stage1.clean_teacher_dir` | None (= output.teacher_dir) | root of the clean teachers |
| `stage1.clean_teacher_epoch` | None (OB-selected) | int e: `<clean teacher>/epochs/epoch<e>`; `"poisoned"`: the poisoned teacher's OB-selected epoch, read at run time from its `epochs.json`. Stage 2 probes the same teacher |
| `stage1.poisoned_student_run` | None | run id or `{attack: run id}`: reuse that run's poisoned students (same tags); Stage 1 never trains or writes them, Stage 2 re-evaluates them |
| `stage1.record_provenance` | False | `kd_meta.json` gets `provenance`: teacher path, weight-file SHA-256, teacher epoch, code fingerprint |
| `stage2.record_provenance` | False | preds `meta.provenance`: students, teachers, teacher epoch, code fingerprint, input SHA-256 per probe |

"Code fingerprint" = SHA-256 over the sorted `<sha256>  <path>` lines of every
`counterfactual_kd/**/*.py` (same line format as scripts/tifs/snapshot_code.py, package only).

## Jobs (queue v9, GPU 0 pool, 2 slots, 78 GB)

Order: all A15 evaluations first (so a free slot evaluates ready runs), then a, b1, b2, c,
d, e, e2 (b3 last when enabled). Every evaluation waits for its training job and
`tifs_rule_frozen`. GPU-h from `estimate.py` (g5-cost medians; ws re-score charged one
Stage 2 cell per condition, an upper bound).

### a: whitespace-preserving re-scoring (A15-a)

`tifs_x15_ws__<run>` (cmd, `ws_score.py --run <run>`, after `<run>__eval`) for
tifs_b_gpt2_sst2{,_s43,_s44}, tifs_b_qwen3_sst2{,_s43,_s44}, tifs_b_qwen3_agnews,
tifs_x13_qwen3_agnews_s43/_s44 (wait for the A13 evaluations), tifs_g_qwen3_badnets10_s42 and
tifs_aa_qwen3_badnets10_t43 (contaminated corpus, descriptive), tifs_b_bert_sst2 (BERT control:
must not change), and in part c tifs_x15_c_gpt2_atba_s42..44. 2.8 GPU-h (+0.1 in c).

Output `runs/tifs/tifs_x15_ws/<run>/`, per condition tag (the frozen tag):
- `<tag>.official.preds.json`, `<tag>.official.json`: Stage 2 schema re-score of the official inputs.
- `<tag>.ws.preds.json`, `<tag>.ws.json`: the whitespace-preserving inputs (`meta.input_format`).
- `<tag>.compare.json`: per vector (`preds_obs`, `preds_null`, `preds_p_clean`, `preds_c_clean`,
  every control `p`/`c`) whether the official re-score reproduces the frozen preds and `n_diff`;
  per probe `n_changed_by_ws`, `changed_mask` (0/1 per input), SHA-256 of the official and ws
  texts and whether they equal the texts actually scored.
- `summary.json`: reproduced vectors (`n_reproduced`/`n_vectors`) per tag and in total.

The queue runs every ws job with `--teachers` (verifier advice 10/26), so teachers are probed
on both input forms; without it the output's teacher_asr/teacher_ca are 0.0 placeholders, and
the teacher gate for any ws judgement must come from the frozen result JSONs. On real data the ws mode changes all 912 SST-2 triggered
inputs (they keep the trailing space), 1,623 of 5,700 AG News inputs (669 for SynBkd), and
no BERT/GPT-2 ATBA prefix input on SST-2 (the prefix already kept the original string). On AG
News the ws prefix differs from the official prefix on the 14 inputs that start with whitespace
(ws keeps the leading whitespace first); no part-a run uses a prefix trigger.

### b1: epoch-matched C, seed-42 twins (A15-b1)

| job | kind | after | GPU-h |
|---|---|---|---|
| `tifs_x15_bt_bert_sst2_t42` | stage0 build, 34 GB | — | 0.64 |
| `tifs_x15_bt_gpt2_sst2_t42` | stage0 build, 72 GB (not measured) | — | 1.28 |
| `tifs_x15_bt_qwen3_sst2_t42` | stage0 build, 66 GB | — | 1.87 |
| `tifs_x15_twincheck_ob_<teacher>` ×3 | cmd, CPU | its build | 0.05–0.15 |
| `tifs_x15_b1_bert_sst2_e{1,4,5}_s{42,43,44}` (9) | stage1 + eval | BERT build | 0.41 (s42 logit+feature 0.82) |
| `tifs_x15_b1_gpt2_sst2_e5_s{42,43,44}` | stage1 + eval | GPT-2 build | 0.61 |
| `tifs_x15_b1_qwen3_sst2_e3_s{42,43,44}` | stage1 + eval | Qwen3 build | 1.29 |

Part total 15.5 GPU-h. Teacher root `results/tifs/teachers_x15/clean/ob_<model>_sst-2_seed42/`:
the OB-selected checkpoint (as before), `epochs/epoch<e>/` (HF weights + tokenizer +
`epoch_meta.json`: dev metrics, file and tensor SHA-256), `epochs.json` (OB-selected epoch,
accuracy-only epoch, per-epoch dev metrics, weight hashes, `root_matches_selected_epoch`, pid).
The builds use the frozen templates' settings and seeds (tifs_t_bert / tifs_t_gpt2 / tifs_t_qwen3,
BadNets poisoner at rate 0 for the dev split, as the frozen twins; ob_meta.json agrees except ob_config.train.save_path, which now carries the
pid directory). For epoch teachers, kd_meta's teacher_meta_ref points to a non-existent
epochs/epoch<e>/ob_meta.json; the provenance block holds the correct epoch and hashes.

Epochs (verified 10/26 from the raw build logs, recomputing OB's rule, and from
`g6-review0/corpus/selection_builds.csv`): BERT t42 P: ATBA 1, AddSent 1, BadNets 4, SynBkd 5;
C 3. GPT-2 P: BadNets 2, AddSent 5; C 2. Qwen3 P: 3 (all); C 1. Hence C at e1 for ATBA and
AddSent, e4 for BadNets, e5 for SynBkd (BERT), e5 for AddSent (GPT-2), e3 (Qwen3). GPT-2 BadNets
(e2 = frozen C epoch) keeps the frozen C. The P students are the frozen ones
(`stage1.poisoned_student_run`): Stage 2 of a b1 run re-evaluates them, so its `preds_obs`
must equal the frozen file's (an evaluation-reproducibility check).

Twin check output `runs/tifs/tifs_x15_twincheck/<teacher>.json`: `identical` (canonical tensor
SHA-256 of the rebuilt OB-selected epoch = frozen twin), `files_identical`,
`selected_epoch_matches_frozen` (frozen epoch recomputed from its build log; `expect_epoch`),
`dev_identical` (per-epoch dev accuracies vs the frozen log), and when different
`diff_vs_selected.max_abs_diff` and `max_abs_diff_by_epoch`. It exits 0 either way (A15-b1:
report and proceed).

### b2: references and A2 (A15-b2)

Builds `tifs_x15_bt_bert_sst2_t{43,44,45,46}` (34 GB, 0.64 h each), **chained** t42 → t43 →
t44 → t45 → t46 so two BERT twin builds never run together; checks against the A/A (t43, t44)
and A13 G2 (t45, t46; t46 waits for `tifs_x13_bert_ref_t46_teachers`) twins.
C′ students (logit, corpus and student seed 42) `tifs_x15_b2_bert_ref_t<t>_e<e>` for
t ∈ {43,44,45,46}, e ∈ {1,4,5} except t43 e1 (the frozen C′(t43) is epoch 1): 11 runs,
0.41 h each. The BadNets P of the same reference (auxiliary) is reused from
tifs_aa_bert_clean_t43/_t44 and tifs_x13_bert_ref_t45/_t46 (t45/t46 wait for those A13 jobs).
**A2:** C′(t43, e5) seed 42 is the clean student of `tifs_x15_b2_bert_ref_t43_e5`, which also
holds the A2 seed-42 cell (P′ reused from tifs_rep_bert_atba_t43_s42; tag
`atba_soothing sharing how_logit_bert_bert-base-uncased_sst2_seed42_t43_c42_s42`); seeds 43/44
are `tifs_x15_b2_a2_t43_e5_s{43,44}` (P′ from tifs_rep_bert_atba_t43_s{43,44}). Part total 8.5 GPU-h.

### b3 (conditional, disabled)

`tifs_x15_b3_pcb_r{0001,0003,0010,0030}_s{42,43,44}`: A13 G3 C students at epoch 4 (P reused
from the A13 G3 runs). In the pool only with `A15_B3_ENABLED = True`, and then each job also
waits for the hand-written marker `tifs_x15_b3_go` (.ok = run, .failed = block), written from
the rule fixed in A15 (95% upper bound of |C_e4 − C_e3| on BERT BadNets inputs > 0.01). 5.3 GPU-h.

### c: GPT-2 SST-2 ATBA (A15-c)

| job | kind | after | GPU-h |
|---|---|---|---|
| `tifs_x15_ct_gpt2_atba_sst2_t42` | stage0 build (poisoned only; suffix; every epoch), 72 GB | GPT-2 twin build | 1.28 |
| `tifs_x15_c_gate` | cmd GPU (`gate_c.py`), 10 GB | ATBA build, GPT-2 twin build | 0.05 |
| `tifs_x15_c_gpt2_atba_s{42,43,44}` | stage1 + eval | build, gate, twin build | 1.21 |
| `tifs_x15_ws__tifs_x15_c_gpt2_atba_s*` | cmd | its eval | 0.04 |

Trigger: the registry's GPT-2 SST-2 ATBA phrase (`edge akens gently capt sharp akens erning`),
`stage0.trigger_position: suffix` and `attack_overrides.atba.position: suffix`. Gate (frozen
analyze.py rule, ASR ≥ 0.90 and accuracy drop ≤ 0.05): P = OB-selected ATBA teacher, C = the
b1 GPT-2 twin at the same epoch; `runs/tifs/tifs_x15_c_gate/gate.json` (also the comparison
with the twin's OB-selected checkpoint). Exit 3 = failed → lane marks it failed → every c
student is **blocked** (no student trains). The C students read the epoch at run time
(`clean_teacher_epoch: poisoned`). Part total 5.2 GPU-h.

### d, e, e2

- d `tifs_x15_d_bert_agnews_s{45,46}`: the frozen BERT AG News logit battery (6 attacks, shared
  C, C_ep, C_sos; 9 students) at student seeds 45/46, teachers and corpus seed 42; template
  tifs_b_bert_agnews_s43. 6.1 h each, 12.8 GPU-h with evaluations.
- e `tifs_x15_e_bert_agnews_a{08,10}_s{42,43,44}`: alpha 0.8 / 1.0 (T 4), P_BadNets, P_ATBA, shared
  C. 2.05 h each, 13.0 GPU-h.
- e2 `tifs_x15_e2_gpt2_sst2_a10_e{2,5}_s{42,43,44}`: GPT-2 SST-2 alpha 1.0; P_BadNets (e2 runs)
  and P_AddSent (e5 runs) distilled anew from the frozen P teachers, C from the b1 GPT-2 twin at
  epoch 2 / 5. 1.21 h each, 7.4 GPU-h.

## Totals

65.3 GPU-h over 121 jobs without b3 (70.5 with b3). Simulated wall clock with the pool's own
`pick` from the moment A13 is done: **33.2 h** (a done 1.4 h, b1 12.8 h, b2 17.9 h, c 21.8 h,
d 23.1 h, e 29.6 h, e2 33.2 h); 35.5 h with b3. Every A15 job waits for the hand-written marker
`runs/tifs/_status/tifs_x15_go.ok`, written once every A13 job is terminal, so A15 does not
start in the A13 tail and the lanes do not exit between the bundles. Registered deadline 10/29 06:00 KST; cut order e2 → e → d (by hand: write
`.blocked` markers for that part's job ids; their evaluations then block automatically). Never
cut by setting the flag off: that drops all of A15, and running A15 jobs would count as 78 GB.

## How the analysis reads the outputs

All new runs keep the frozen tags, so hit vectors align input by input with the frozen ones
(`plan_check.json` lists every condition with its `aligned_frozen_preds`).

- b1: `k_em = P − C_e(P)`. P from the frozen preds `runs/tifs/<poisoned_student_run>/artifacts/results/<tag>.preds.json`
  (`preds_obs`; the b1 run's re-evaluated `preds_obs` should be identical), C_e from the b1 run's
  `<tag>.preds.json` `preds_null`. GPT-2 BadNets: frozen file unchanged. Teacher-level vectors of
  the epoch-matched pair are in the b1 preds (`teacher_c_trig` = C_e teacher).
- b2: C′ vectors from `tifs_x15_b2_bert_ref_t<t>_e<e>` (`preds_null`, and the A/A controls incl.
  `atba_phrase_sst2`); new C′ checkpoints for CPU cross-scoring (A14 item 7 style) at
  `runs/tifs/tifs_x15_b2_bert_ref_t<t>_e<e>/artifacts/students/clean-base_logit_bert_bert-base-uncased_sst2_seed42_t<t>_c42_s42/clean_student`.
  A2 `k′_em`: tag above in tifs_x15_b2_bert_ref_t43_e5 (s42) and tifs_x15_b2_a2_t43_e5_s{43,44}.
- a: `tifs_x15_ws/<run>/<tag>.ws.preds.json` vs `.official.preds.json` (same schema as Stage 2).
- c, d, e, e2: ordinary Stage 2 outputs of the new runs; e/e2 tags equal the alpha 0.5 tags.
- Every new preds file has `meta.provenance`; every new student `kd_meta.json` has `provenance`
  (teacher weight SHA-256 and epoch). Teacher epochs and hashes: `epochs.json`.

## Enabling (operator)

1. Verify (tests: `tests/test_tifs_a15.py`; `make_configs.py --check`; `plan_check.py`).
2. Set `A15_ENABLED = True` in scripts/tifs/queue_plan.py by an atomic replace, import-test first.
3. Take a code snapshot (`scripts/tifs/snapshot_code.py --label a15_queue_v9`) and an events note.
4. The pool lanes exit once every pool job is terminal; if A13 has finished by then, start
   `lane.py --lane gpu0p1` and `--lane gpu0p2` again.
