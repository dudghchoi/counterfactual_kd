# g6-review0 / corpus: A14 items 8, 10, 11 (code)

A14 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

Code for docs/tifs_addendum.md A14 items 8 (trigger exposure in the KD corpus vs k), 10 (decoder
whitespace check) and 11 (clean-twin checkpoint selection, A5), answering the internal round-0 review
(paper_tifs/reviews/experiment_options.md §2 C8, C11, C12). Outputs, definitions, data snapshot and
findings: results/tifs/addendum/aux/g6-review0/corpus/README.md.

| script | item | reads | writes |
|---|---|---|---|
| common.py | all | paths, frozen imports (analyze.py, addendum.py), CSV writer with the '# definition' line | – |
| item8_recount.py | 8 | clean KD corpora (load_data), teacher tokenizers | exposure_recount.csv (incl. token occurrences inside longer words by the aux/g1 and the alphanumeric definitions) |
| item8_exposure.py | 8 | frozen cells (addendum.State), aux/g1 token tables, exposure_recount.csv | exposure_vs_k.csv, exposure_classes.csv, exposure_summary.json |
| item11_selection.py | 11 | logs/tifs/*.log, tier-1 logs/*.log, results/tifs/teachers/*/*/ob_meta.json, frozen cells, report/cells.csv | selection_*.csv, selection_summary.json |
| item11_devcheck.py | 11 (check) | saved clean twins, OpenBackdoor dev-clean split (datasets/) | selection_devcheck.csv |
| item10_ws_rescore.py | 10 (diagnostic) | SST-2 decoder students, Stage 2 inputs rebuilt as in cross_score.py; every probe on CPU fp32 (Qwen3 unedited probes added in a second call after verification round 1) | diagnostics/ws_rescore_hits.json |
| item10_whitespace.py | 10 | frozen cells, rebuilt inputs, aux/g1 tokens_last_token.csv, the re-score | ws_*.csv, ws_summary.json, diagnostics/ws_rescore_{rates,contrasts,precision,cells}.csv |
| self_checks.py | all | the outputs above, report/primary.csv, aux/g1, structural_nulls.json, frozen file hashes | self_checks.csv |
| make_readme.py | all | the outputs above | README.md (results folder) |

Run: `scripts/tifs/aux/g6-review0/corpus/run_all.sh [step ...]` (steps recount, exposure, selection,
devcheck, ws_rescore, whitespace, checks, readme). CPU only, nice 10, 8 threads, no bytecode, Hugging
Face offline. `OUT=... run_all.sh` writes elsewhere. Nothing outside the output folder is written; the
frozen code is imported, never edited.
