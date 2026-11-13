#!/usr/bin/env bash
# A12 compute-cost record (group g5-cost; docs/tifs_addendum.md A12). Descriptive and post hoc:
# it changes no primary verdict, battery decision or the paper-wide Holm. CPU only and
# read-only on the repository except the output folder. It reads runs/tifs/_status/events.jsonl,
# the run metadata under runs/tifs, results/tifs/teachers, logs/tifs, configs/tifs and
# scripts/tifs/queue_plan.py, and writes results/tifs/addendum/aux/g5-cost/ (or --out DIR,
# which inside the repository must lie below results/tifs/addendum/aux/). About 6 seconds.
#
#   bash scripts/tifs/aux/g5-cost/run_all.sh [--out DIR]
#
# PY overrides the interpreter.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/data/Youngho/envs/blade/bin/python}"

export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
# No tokenizer or model is loaded; the offline switches only make sure nothing reaches the hub.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

"$PY" "$HERE/cost.py" "$@"
