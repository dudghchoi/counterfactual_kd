# A14 subgroup 'multiplicity': code (docs/tifs_addendum.md A14 items 1, 2, 6, 9, 12)

Status: A14 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

Answers the internal round-0 review (paper_tifs/reviews/round0_meta.md section 2; experiment_options.md section 2,
items on family-wise 'no transfer' claims, headline power, the Holm family, the delta rationale and the S1c config H
numbers). Outputs, full definitions, the data snapshot and the key findings with exact numbers are in
`results/tifs/addendum/aux/g6-review0/multiplicity/README.md` (written by make_readme.py from the CSVs).

## Definitions (short)

- F31: the 31 qualified primary cells (the frozen transfers_global family). F19: its 19 logit-KD cells (the headline
  cells).
- p_up@delta: one-sided p of H0 k >= delta ('97.5% one-sided upper bound below delta'); p_tost@delta: two-sided
  equivalence; p_pos@delta: H0 k <= delta (transfer). All from the frozen `analysis.tifs.t_contrast` with the primary
  row's SE and df; the two EP SST-2 structural nulls use the frozen seed-42 Tango p of their primary row.
- A claim holds when the adjusted p < 0.025. Holm (frozen `analysis.tifs.holm`) controls the family-wise error of the
  rejections and gives no simultaneous bounds; Bonferroni (min(1, m p)) corresponds to simultaneous one-sided
  (1 - 0.025/m) bounds.
- A4 classes: structural, dagger (trigger words in no clean KD-corpus row), boundary (SOS SST-2), counted
  (paper_tifs/tools/make_numbers.py rule). A4-counted = counted + boundary.
- Headline (item 2): max p_up@0.05 over the 19 logit cells < 0.025 (analyze.summary rule), simulated with every true
  k = 0 by the registered aux/g4 S1c model (configs H and E), r = 3 (measured input covariance) and r = 3, 5, 6
  (exchangeable covariance), Qwen3 AG News transported from the same simulated Qwen3 SST-2 pool.
- Item 12 S1c rows: c_sd is the SD ratio of the target cell's true training SD to the model's (variance ratio
  c_var = c_sd^2).

## Scripts

| script | item | what |
|---|---|---|
| g6m_common.py | - | paths, frozen imports (analyze through addendum.State, analysis/tifs, aux/g4 common.py and s1c.py), CSV writer with the `# definition` line, A4 classes, data snapshot, freeze hashes |
| claims.py | 1 | claim type x error control: p_up / p_tost / p_pos per cell; unadjusted, Holm and Bonferroni over F31 and F19; counts split by A4 class |
| holm_family.py | 6 | transfers_global under the families 31 / logit 19 / no 'feature' 19 / no structural 29 / non-structural logit 18 |
| delta_rationale.py | 9 | delta*, Cohen's h and log OR, k = 0 floors (aux/g3), per-cell half-widths, A/A C - C', cells with lo > 0 not POSITIVE |
| s1c_main.py | 12 | copies the S1c config H (and E) numbers from aux/g4 into one table (nothing re-simulated) |
| headline_power.py | 2 | joint Monte Carlo of the 19-cell headline (`--blocks 8 --reps 10000 --workers 6` in run_all.sh) |
| selfchecks.py | all | reproduces official numbers through this code path (primary.csv every column, battery.csv, global_holm.csv, summary.md headline, NUMBERS.md Finding 1, aux/g1, aux/g3, aux/g4), frozen hashes, review-preview comparison |
| make_readme.py | all | output README.md and provenance.csv, every number read from the CSVs |
| run_all.sh | all | everything above in order, CPU only |

## Run

```
bash scripts/tifs/aux/g6-review0/multiplicity/run_all.sh            # -> results/tifs/addendum/aux/g6-review0/multiplicity/
bash scripts/tifs/aux/g6-review0/multiplicity/run_all.sh /some/dir  # other output folder
```

`results/tifs/addendum/aux/g6-review0/multiplicity/run_all.sh` is a thin wrapper around this script. Environment (set
by run_all.sh): `CUDA_VISIBLE_DEVICES=""`, `PYTHONDONTWRITEBYTECODE=1` (every script also sets
`sys.dont_write_bytecode`), `OMP_NUM_THREADS=MKL_NUM_THREADS=8`, `nice -n 10`, interpreter
`/data/Youngho/envs/blade/bin/python` (override with `PY=...`), `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1`, although no tokenizer or model is loaded. About 3 min wall and 17 CPU-min, almost all in
headline_power.py. Nothing outside this folder and the output folder is written; the frozen files are imported, never
edited.
