#!/usr/bin/env bash
# g10 claim-rule calibration (docs/tifs_addendum.md, amendment 10/27 20:58 KST, item 4) and the claim rule applied
# to the frozen 31 primary cells (items 2-3).
# A15 post hoc auxiliary analysis; changes no primary verdict, battery decision or the paper-wide Holm.
#
# CPU only (CUDA_VISIBLE_DEVICES=''), nice 10, one BLAS thread per process, HF offline. Read-only on runs/,
# results/tifs/report, results/tifs/addendum and the code; writes only results/tifs/addendum/aux/g10-claimrule
# (or --out DIR). About 13-16 min wall with 10 workers (about 2 CPU-hours, almost all in sim.py).
# Every analysis script (frozen_claims.py, sim.py, independent.py) first reproduces the official tables through the
# frozen loader and stops on a mismatch; selfcheck.py records the same checks; set -e stops the pipeline at the first
# failure.
#
#   scripts/tifs/aux/g10-claimrule/run_all.sh [--out DIR] [--workers N] [--quick]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
PY=${PY:-/data/Youngho/envs/blade/bin/python}
OUT="$ROOT/results/tifs/addendum/aux/g10-claimrule"
WORKERS=10
QUICK=""
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift ;;
    --workers) WORKERS="$2"; shift ;;
    --quick) QUICK="--quick" ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
mkdir -p "$OUT/logs"

export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

date -u +"%Y-%m-%d %H:%M:%S UTC" > "$OUT/logs/run_start.txt"
# Frozen files, the imported aux code, and every official input.
(cd "$ROOT" && sha256sum \
  docs/tifs_plan.md docs/tifs_addendum.md scripts/tifs/analyze.py scripts/tifs/addendum.py \
  counterfactual_kd/analysis/tifs.py counterfactual_kd/metrics/paired.py scripts/tifs/cross_score.py \
  results/tifs/structural_nulls.json \
  scripts/tifs/aux/g4-s1c-s9-s1extra/common.py scripts/tifs/aux/g4-s1c-s9-s1extra/s1c.py \
  scripts/tifs/aux/g6-review0/multiplicity/g6m_common.py scripts/tifs/aux/g6-review0/multiplicity/headline_power.py \
  results/tifs/report/primary.csv results/tifs/report/variance.csv results/tifs/report/battery.csv \
  results/tifs/addendum/global_holm.csv results/tifs/addendum/attack_variance.csv \
  results/tifs/addendum/aux/g4-s1c-s9-s1extra/s1c_cells.csv results/tifs/addendum/aux/g4-s1c-s9-s1extra/s1c_battery.csv \
  results/tifs/addendum/aux/g4-s1c-s9-s1extra/s1c_fit.csv \
  results/tifs/addendum/aux/g6-review0/multiplicity/claims_family.csv \
  results/tifs/addendum/aux/g6-review0/multiplicity/claims_cells.csv \
  results/tifs/addendum/aux/g9-a15cpu/welch_cells.csv \
  results/tifs/addendum/aux/g1-exposure-controls-tokens/tokens_summary.csv) > "$OUT/logs/hashes_before.txt"

cd "$HERE"
run() {
  local name="$1"; shift
  echo "== $name"
  local t0=$SECONDS
  nice -n 10 "$PY" "$@" --out "$OUT" 2>&1 | tee "$OUT/logs/$name.log"
  echo "$name,$((SECONDS - t0))" >> "$OUT/logs/wall_seconds.csv"
}
echo "script,wall_s" > "$OUT/logs/wall_seconds.csv"
run frozen_claims frozen_claims.py
run sim           sim.py --workers "$WORKERS" $QUICK
if [ -n "$QUICK" ]; then
  run independent independent.py --reps 2000
else
  run independent independent.py
fi
run selfcheck     selfcheck.py
run make_readme   make_readme.py
date -u +"%Y-%m-%d %H:%M:%S UTC" > "$OUT/logs/run_end.txt"
echo "done: $OUT"
