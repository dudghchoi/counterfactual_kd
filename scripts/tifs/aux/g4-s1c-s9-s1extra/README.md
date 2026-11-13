# A11 items 10–12 (group g4): S1c, S9, S1 extra rows — code

Post hoc / registered auxiliary analysis (docs/tifs_addendum.md A11, items 10, 11 and 12). These scripts change no
primary verdict, battery decision or the paper-wide Holm. They read existing files only and call the frozen
functions unchanged (analyze.py, addendum.py functions, counterfactual_kd/analysis/tifs.py, metrics/paired.py,
analysis/calibration.py, s1_calibration.py, s1b_seed_noise.py). The tables, definitions and key findings are in
`results/tifs/addendum/aux/g4-s1c-s9-s1extra/README.md`.

## Run

```
bash scripts/tifs/aux/g4-s1c-s9-s1extra/run_all.sh            # -> results/tifs/addendum/aux/g4-s1c-s9-s1extra/
bash scripts/tifs/aux/g4-s1c-s9-s1extra/run_all.sh /some/dir  # other output folder
```

Environment (set by run_all.sh): CPU only, `CUDA_VISIBLE_DEVICES=""`, `PYTHONDONTWRITEBYTECODE=1` (no `__pycache__`
in the repository; every script also sets `sys.dont_write_bytecode`), `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`,
interpreter `/data/Youngho/envs/blade/bin/python` (override with `PY=...`), `nice -n 10`. `HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1` are exported, although this group loads no tokenizer or model.

Order (each script takes `--out DIR`; the default is `results/tifs/addendum/aux/g4-s1c-s9-s1extra/`):

| step | script | item | what it does |
|---|---|---|---|
| 1 | `s1_extra.py --workers 4` | 12 | frozen `run_scenario` on the boundary (n = 912, k = δ = 0.02), sparse and real plug-in scenarios |
| 2 | `s1c.py --workers 6` | 10 | S1b's normal model fitted to the measured batteries; Monte Carlo of the frozen pooled and transported rules |
| 3 | `s9.py` | 11 | stage-wise rules R1, R1h, R2–R5 (R5b, R5t) scored on S1-generator scenarios and on the empirical known-truth items |
| 4 | `recheck_s1b.py` | 10 | reruns the S1b-mode setting with the largest \|z\| with two fresh seeds (reads `s1c_check_s1b.csv`) |
| 5 | `checks.py` | all | self-checks against results/tifs/report, results/tifs/addendum and results/tifs/s1; input SHA-256 |
| 6 | `make_readme.py` | all | output README.md; every number read from the CSVs |

`common.py` holds the paths (repository root from `__file__`), the CSV writer (every CSV starts with a `# definition
+ status` line), the data snapshot, the cost log (`logs/cpu_seconds.csv`) and the plug-in model builders.

## Inputs (read only)

- `runs/tifs/tifs_*/artifacts/results/*.preds.json` through the frozen `analyze.load_cells` (via `addendum.State`).
- `results/tifs/report/{primary,variance,aa,secondary,factorial,cells}.csv` (frozen analyze.py output).
- `results/tifs/addendum/{attack_variance,variance_diag}.csv` (addendum.py output).
- `results/tifs/s1/{s1b,grid,cells}.csv` (archived S1 and S1b) and, through the frozen
  `s1_calibration.cell_scenarios`, the v1 seed-42 predictions in `server_handoff_20261002/repair/zenodo_v1.1/cells/`.

## Seeds and cost

All Monte Carlo seeds are fixed: s1c 20261025–20261123 in job-creation order (blocks 2–4 of the config-H
transported jobs: seed + 1,000,000·b, block 1 keeps the job seed); s9 110000 + scenario index (the S1-cell sets are
appended last); s1_extra 120000 + scenario index; recheck_s1b 777001, 777002. The promoted run (2026-10-25,
6 + 4 workers, `nice -n 10`) took 404 s wall and 1,202 CPU-seconds (20.0 CPU-minutes): s1_extra.py 508, s1c.py 497,
s9.py 122, checks.py 55, recheck_s1b.py 20 and make_readme.py 1 CPU-second (`logs/cpu_seconds.csv` in the output
folder; rerunning make_readme.py alone, e.g. after a wording fix, replaces its own row there).

In the S1c loops, `scipy.stats.t` is routed to `common.ExactFastT`, which calls the same `scipy.special.stdtr` /
`stdtrit` that scipy 1.13's `t.cdf`, `t.sf` and `t.ppf` call, without the `rv_continuous` argument handling (about
95% of the time spent in `t_contrast`: 4,000 calls take 0.16 s instead of 3.39 s). The single-block S1c jobs took
248 s in the promoted run against 2,516 s in the pre-promotion run with stock scipy (about 10 times faster; the
machine load differed). The frozen `t_contrast` is unchanged and returns bit-identical values; `checks.py` compares
the two on random inputs ('exact fast t'), the unchanged single-block jobs reproduce the pre-promotion (stock
scipy) rows exactly, and `s1c.py --stock-scipy` runs without it.
