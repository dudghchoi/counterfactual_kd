#!/usr/bin/env bash
# A11 items 3, 4, 6, 7 (group g2-validity-reference-flips-builds); docs/tifs_addendum.md A11.
# Post hoc / registered auxiliary tables: they change no primary verdict, battery decision or
# the paper-wide Holm.
#
# CPU only. Reads runs/tifs (predictions), results/tifs/report and results/tifs/addendum;
# writes only into the output folder (default results/tifs/addendum/aux/g2-validity-reference-flips-builds).
#
# usage: scripts/tifs/aux/g2-validity-reference-flips-builds/run_all.sh [--out DIR] [--gpu0-runs DIR]
#   --out DIR        write the CSVs and README.md into DIR instead of the default folder
#   --gpu0-runs DIR  optional: a runs dir holding the GPU0 re-evaluation of tifs_b_bert_sst2_s43 (A9);
#                    without it the N2 rows are built from results/tifs/addendum/gpu0_eval_determinism.json
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
PY=${PY:-/data/Youngho/envs/blade/bin/python}

export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
# No tokenizer or model is loaded by these scripts; the offline variables only guard against downloads.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

# Start-of-run state for selfcheck.py: run start time and the sha256 of the frozen files and imported code.
export G2_RUN_START_EPOCH="$(date +%s)"
export G2_SHA256_START="$(cd "$ROOT" && sha256sum docs/tifs_plan.md scripts/tifs/analyze.py \
  counterfactual_kd/analysis/tifs.py results/tifs/structural_nulls.json counterfactual_kd/metrics/paired.py \
  counterfactual_kd/analysis/calibration.py scripts/tifs/addendum.py | awk '{printf "%s=%s;", $2, $1}')"

cd "$HERE"
for s in item3_validity.py item4_reference_atba.py item6_flips.py item7_builds.py selfcheck.py make_readme.py; do
  echo "== $s"
  "$PY" "$s" "$@"
done
