#!/usr/bin/env bash
# A11 items 5, 8, 9 (g3: verdict ladder, delta and seed count, simultaneous intervals).
# CPU only and read-only: reads runs/tifs, results/tifs/report and results/tifs/addendum, and
# writes results/tifs/addendum/aux/g3-ladder-seeds-simultaneous/ (or the folder given with
# --out DIR, passed through to every script). About 10 seconds.
#
#   bash scripts/tifs/aux/g3-ladder-seeds-simultaneous/run_all.sh [--out DIR]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/data/Youngho/envs/blade/bin/python}"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
# g3 loads no tokenizer or model; the offline variables only keep the environment the same as
# the other A11 groups.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
for script in g3_ladder.py g3_seeds.py g3_simultaneous.py g3_provenance.py; do
  "$PY" "$HERE/$script" "$@"
done
