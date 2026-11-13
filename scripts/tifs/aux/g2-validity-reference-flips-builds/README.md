# A11 items 3, 4, 6, 7 (group g2-validity-reference-flips-builds)

Code for the A11 auxiliary / post hoc tables of docs/tifs_addendum.md: item 3 (판별 타당성 표),
item 4 (기준별 ATBA 표), item 6 (전환 빈도) and item 7 (빌드별 요인 설계). The tables change no
primary verdict, battery decision or the paper-wide Holm. Definitions, numbers and self-checks are
in the output folder's README.md (`results/tifs/addendum/aux/g2-validity-reference-flips-builds/`).

## Run

```bash
scripts/tifs/aux/g2-validity-reference-flips-builds/run_all.sh            # default output folder
scripts/tifs/aux/g2-validity-reference-flips-builds/run_all.sh --out DIR  # somewhere else
```

Order (run_all.sh): `item3_validity.py`, `item4_reference_atba.py`, `item6_flips.py`,
`item7_builds.py`, `selfcheck.py` (stops the run if a check fails), `make_readme.py`.
`g2common.py` holds the shared paths and helpers. Each script also runs alone from any working
directory (`python <script> [--out DIR]`); selfcheck.py and make_readme.py read the CSVs of the
same output folder, so run the item scripts first.

Environment (set by run_all.sh): CPU only, `CUDA_VISIBLE_DEVICES=""`,
`PYTHONDONTWRITEBYTECODE=1` (no `__pycache__` in the repository; every script also sets
`sys.dont_write_bytecode`), `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`, Python
`/data/Youngho/envs/blade/bin/python` (override with `PY=...`). No tokenizer or model is loaded;
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1` are set only as a guard.
A full run takes about 30 seconds on CPU.

## Inputs (read only)

- `runs/tifs/*/artifacts/results/*.preds.json` and `*.json`, through the frozen loaders
  (`analyze.load_cells`, `addendum.load_replicate`, `load_hidden_cells`, `load_rcc`).
- `results/tifs/addendum/cross_hits_bert_sst2.json`, `cross_hits_bert_agnews.json` (A3.4 CPU fp32
  re-scores; both required, no fallback copy), `gpu0_eval_determinism.json` (A9),
  `cprime_determinism.json` (A2), and the addendum CSVs used by the self-checks.
- `results/tifs/report/*.csv` (frozen analyze.py output), used by the self-checks.
- `runs/tifs/_code/freeze_20261021T092041Z.sha256` and `final_analysis_20261024T110542Z.sha256`
  (integrity checks of the frozen files and of every imported module).

Optional: `--gpu0-runs DIR` points item 3 at a runs folder holding the GPU0 re-evaluation of
`tifs_b_bert_sst2_s43` (A9). That folder is not in the repository; without it the N2 rows use the
sha256 record in `gpu0_eval_determinism.json` (the GPU0 files are byte-identical to the official ones).

The frozen functions (`counterfactual_kd/analysis/tifs.py`, `scripts/tifs/analyze.py`,
`counterfactual_kd/metrics/paired.py`) and `scripts/tifs/addendum.py` are imported, never edited.
Every file write goes into the output folder.
