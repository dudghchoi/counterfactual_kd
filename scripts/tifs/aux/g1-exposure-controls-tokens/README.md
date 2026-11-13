# A11 items 1, 2, 13 — exposure axis, control-input responses, trigger tokens (group g1)

These scripts produce the A11 auxiliary tables registered in docs/tifs_addendum.md A11 (10/24 18:05 KST) for items 1, 2
and 13. The tables are post hoc and descriptive. They change no primary verdict, no battery decision and not the
paper-wide Holm. The scripts are read-only: they read runs/tifs (preds and result JSONs, kd_meta.json, checkpoints),
results/tifs/report (frozen analyze.py output) and results/tifs/addendum (addendum.py output), and call the frozen
functions without editing them. They write only to the output folder.

- Output folder: `results/tifs/addendum/aux/g1-exposure-controls-tokens/`. Pass `--out DIR` to a script, or set
  `OUT=DIR` for run_all.sh, to write elsewhere. The output README.md lists every file and gives the definitions.
- Paths: `g1_common.py` computes the repository root from its own location (`scripts/tifs/aux/<group>/`, four levels
  down), so the folder needs no edits when the repository moves.
- Environment: `/data/Youngho/envs/blade/bin/python`, `PYTHONDONTWRITEBYTECODE=1` and `CUDA_VISIBLE_DEVICES=""` (CPU
  only). The tokenizer and model steps also need `HF_HOME=/data/Youngho/cache/huggingface HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1`. run_all.sh sets all of these. Every script also sets
  `sys.dont_write_bytecode` before its first local import, so no `__pycache__` is written into the repository.

Run order (`./run_all.sh` runs every step; `./run_all.sh <step> ...` runs only the steps named):

| step | script | what | time (CPU) |
|---|---|---|---|
| exposure | exposure_controls.py | items 1 and 2: exposure axis, h, control-input responses, corpus splits, replicate and identity checks | ~10 s |
| tokens | trigger_tokens.py | item 13: token counts, rebuilt Stage 2 inputs, KD-corpus occurrences, whitespace | ~8 min |
| ws_qwen3 | ws_diag.py qwen3 | diagnostic, not a registered table: CPU fp32 whitespace probes on the Qwen3 SST-2 students | ~20 min |
| ws_gpt2 | ws_diag.py gpt2 | diagnostic, not a registered table: the same probes on the GPT-2 SST-2 battery students | ~4 min |
| twins | teacher_twins_check.py | diagnostic, not a registered table: BERT EP / SOS clean-twin teachers | ~1 min |
| checks | self_checks.py | cross-checks against the official report / addendum CSVs and internal checks (self_checks.csv) | ~10 s |
| readme | make_readme.py | README.md, diagnostics/README.md, logs/README.md (every number read from the CSVs) | ~1 s |

`checks` and `readme` read the outputs of the steps before them, so run them last. Run logs go to `<out>/logs/`.
`G1_SESSION_START` (epoch seconds; run_all.sh defaults it to its own start time) opens the window in which
self_checks.py looks for new `__pycache__` / `.pyc` anywhere in the repository; the official run of `checks` used
1792832400 (2026-10-24 09:00 UTC, the start of the g1 work).

Provenance: promoted on 2026-10-25 from the session scratchpad version (first run 2026-10-24, 11:23–11:46 UTC, on the
same 146 preds files). Changes against that version: repository-relative paths and `--out`; the cross-checks read
results/tifs/report and results/tifs/addendum instead of scratchpad snapshots; and the fixes asked for by the A11
review (configuration ids, so the one pair of replicate checkpoints is counted once: 222 configurations, 223
checkpoints; the added three-way corpus split; controls_contaminated_vs_clean.csv; the ATBA-trigger phrase columns;
the words that carry each token match; diagnostics/README.md). The registered definitions of items 1, 2 and 13 are
unchanged; added variants are extra, labelled columns.
