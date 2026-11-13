# g3: A11 items 5, 8, 9 (verdict ladder, δ and seed count, simultaneous intervals)

Auxiliary, post hoc tables registered in docs/tifs_addendum.md A11. They change no primary
verdict, battery decision or paper-wide Holm. Definitions, findings and the data snapshot are
in the output folder's README:
`results/tifs/addendum/aux/g3-ladder-seeds-simultaneous/README.md`.

## Run

    bash scripts/tifs/aux/g3-ladder-seeds-simultaneous/run_all.sh [--out DIR]

`run_all.sh` runs the four scripts in this order, each with `--out` passed through:

| order | script | A11 item | writes |
|---|---|---|---|
| 1 | `g3_ladder.py` | 5, verdict ladder | ladder.csv, ladder_intervals.csv, ladder_labels.csv, ladder_counts.csv, ladder_battery.csv, ladder_selfchecks.csv |
| 2 | `g3_seeds.py` | 8, δ and seed count | seeds_floor.csv, seeds_rmin.csv, seeds_cells.csv, seeds_pbelow_compare.csv, seeds_selfchecks.csv |
| 3 | `g3_simultaneous.py` | 9, Bonferroni intervals | simultaneous.csv, simultaneous_summary.csv, simultaneous_selfchecks.csv |
| 4 | `g3_provenance.py` | data snapshot | provenance.csv |

The scripts do not depend on each other's outputs, so any one of them can be rerun alone.
`g3_common.py` holds the shared paths and loaders.

## Environment

- Python: `/data/Youngho/envs/blade/bin/python` (override with `PY=...`). It needs scipy.
- `CUDA_VISIBLE_DEVICES=""`: CPU only. `g3_common.py` also sets it to "" before it imports the
  frozen modules.
- `PYTHONDONTWRITEBYTECODE=1`: the scripts refuse to run without it, and set
  `sys.dont_write_bytecode` before importing anything local, so no `__pycache__` is written
  into the repository.
- g3 loads no tokenizer or model. `run_all.sh` still sets `HF_HUB_OFFLINE=1`,
  `TRANSFORMERS_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`, to match the other A11 groups.
- A full run takes about 10 seconds.

## Inputs and outputs

- Paths are relative to the repository root, found from this folder's location four levels
  below it.
- Read only:
  - `runs/tifs/` (Stage 2 prediction files, through the frozen `analyze.load_cells` and
    `addendum.load_hidden_cells`)
  - `runs/tifs/_status/tifs_a10b_go.failed`
  - `results/tifs/report/` (official analyze.py output)
  - `results/tifs/addendum/` (official addendum.py output)
  - `results/tifs/structural_nulls.json`
  - `provenance.csv` lists every input with its SHA-256.
- Imported, never modified: `scripts/tifs/analyze.py`, `counterfactual_kd/analysis/tifs.py`
  and `scripts/tifs/addendum.py`. `main()` of these modules is never called.
- Written: `results/tifs/addendum/aux/g3-ladder-seeds-simultaneous/` by default. Inside the
  repository, `--out` may point only below `results/tifs/addendum/aux/`.
- Every script recomputes the official tables it builds on and checks them against the
  official CSVs (`*_selfchecks.csv`). A script exits with status 1 if a check fails.
