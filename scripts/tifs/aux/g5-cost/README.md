# g5-cost: A12 compute-cost record (code)

Descriptive, post hoc record registered in docs/tifs_addendum.md A12 (10/25 19:49 KST, snapshot
`a12_registration_20261025T104923Z`). It changes no primary verdict, battery decision or the paper-wide
Holm. It reads existing records only; nothing is trained or evaluated. Definitions, headline totals,
caveats and the data snapshot are in the output folder's README:
`results/tifs/addendum/aux/g5-cost/README.md`.

## Run

    bash scripts/tifs/aux/g5-cost/run_all.sh            # -> results/tifs/addendum/aux/g5-cost/
    bash scripts/tifs/aux/g5-cost/run_all.sh --out DIR  # any folder outside the repository

Run order: there is one step. `run_all.sh` sets the environment and runs `cost.py`, which writes every
output in this order: `cost_jobs.csv`, `cost_summary.csv`, `cost_attempts.csv`, `cost_by_gpu.csv`,
`cost_units.csv`, `cost_per_unit.csv`, `hardware_software.json`, `selfcheck.csv`, `provenance.csv`,
`README.md`. It exits with status 1 if any self-check fails (the outputs are still written, so the
failing row can be read in `selfcheck.csv`).

## Environment

- Python: `/data/Youngho/envs/blade/bin/python` (override with `PY=...`), Python 3.10. It needs PyYAML
  (configs) and nothing else outside the standard library. The only repository module it imports is
  `scripts/tifs/queue_plan.py`; it does not import torch, transformers or `counterfactual_kd`.
- `CUDA_VISIBLE_DEVICES=""`: CPU only. `cost.py` also sets it before anything else.
- `PYTHONDONTWRITEBYTECODE=1`: `cost.py` refuses to run without it and sets `sys.dont_write_bytecode`
  before its first import, so importing `scripts/tifs/queue_plan.py` writes no `__pycache__`.
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1` are exported although nothing is
  downloaded. No network access.
- A full run takes about 6 seconds and reads about 105 MB (mostly job logs and prediction files).

## Inputs (read only)

| input | used for |
|---|---|
| `runs/tifs/_status/events.jsonl` | attempts: `running` events and the events that close them (time in UTC, lane, GPU, pid, `hours`, exit) |
| `runs/tifs/_status/*.ok`, `*.failed`, `*.blocked` | time-zone check only (file times against event times) |
| `scripts/tifs/queue_plan.py` | job definitions of queue v7 (stages, declared memory, pools, lanes); imported, its module code only reads `.running` markers |
| `configs/tifs/<run>.yaml`, `configs/tifs/_dropped/<run>.yaml` | family (teacher model) and datasets of each run |
| `logs/tifs/<job>.log` | one command header per attempt (`--stages`, GPU), teachers tried, SKIP lines, stage summaries; torch version and the Python minor version (site-packages paths in library warnings) |
| `runs/tifs/<run>/manifest.json` | the record of each run's last invocation (end of one attempt without a closing event; GPU 0 memory from an out-of-memory message) |
| `runs/tifs/<run>/artifacts/students/*/*/kd_meta.json`, `config.json` | student `train_time_sec`, KD objective, dataset, teacher; transformers version |
| `runs/tifs/<run>/artifacts/results/*.json`, `*.preds.json` | Stage 2 cells and their file times |
| `results/tifs/teachers/{poisoned,clean}/*/ob_meta.json`, `config.json`, `REUSED.json` | teacher `train_time_sec`, trainer, tier-1 links; transformers version |
| `results/tifs/addendum/gpu0_eval_determinism.json` | GPU models; the A9 GPU 0 re-evaluation, which ran outside the lanes |
| `results/tifs/addendum/aux/g1-exposure-controls-tokens/trigger_tokens_meta.json` | tokenizers version |
| `docs/tifs_addendum.md` | A10-a result times in KST (time-zone check); read only |
| `docs/tifs_plan.md`, `scripts/tifs/analyze.py`, `counterfactual_kd/analysis/tifs.py`, `results/tifs/structural_nulls.json` | SHA-256 against the freeze record only |

`provenance.csv` in the output folder lists the SHA-256 of every input.

## Writes

Only the output folder (default `results/tifs/addendum/aux/g5-cost/`). Inside the repository, `--out`
may point only below `results/tifs/addendum/aux/` and never into an A11 folder (`g1-` to `g4-`).
