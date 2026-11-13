#!/usr/bin/env bash
# A14 subgroup 'reference' (docs/tifs_addendum.md A14 items 3, 4, 5, 7, 13).
# A14 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.
#
# CPU only. Reads runs/tifs (checkpoints and predictions, read only), results/tifs/report and
# results/tifs/addendum; writes only results/tifs/addendum/aux/g6-review0/reference (or --out DIR).
#
# usage: scripts/tifs/aux/g6-review0/reference/run_all.sh [--no-score] [--skip-scitech] [--out DIR]
#   --no-score      skip the CPU fp32 re-scoring (score.py) and only rebuild the tables
#   --skip-scitech  skip the item 13 scoring (about 22 min per student at 8 threads on the shared CPU,
#                   6 students; resumable, one file per seed: score.py scitech --seeds 44 splits it)
#
# Adding the A13 G2 clean references t45 / t46 (runs tifs_x13_bert_ref_t45 / _t46) later is
# this same command: score.py refs scores every reference run whose student training finished
# (runs/tifs/_status/<run>.ok) and that is not scored yet (resumable, checkpoint sha256 keyed),
# and item 5 / item 7 read every hits/ref_*.json. A build whose clean teacher or C' student is
# at or below majority share + 0.10 (Stage 2 clean accuracy) is listed but left out; a build
# without a Stage 2 result yet is listed as 'function unchecked' and left out until its
# __eval job has written one.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
PY=${PY:-/data/Youngho/envs/blade/bin/python}
OUT="$ROOT/results/tifs/addendum/aux/g6-review0/reference"
SCORE=1
SCITECH=1
while [ $# -gt 0 ]; do
  case "$1" in
    --no-score) SCORE=0 ;;
    --skip-scitech) SCITECH=0 ;;
    --out) OUT="$2"; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
mkdir -p "$OUT/logs"

export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=/data/Youngho/cache/huggingface
export G6R_RUN_START_EPOCH="$(date +%s)"
export G6R_SHA256_START="$(cd "$ROOT" && sha256sum docs/tifs_plan.md scripts/tifs/analyze.py \
  counterfactual_kd/analysis/tifs.py results/tifs/structural_nulls.json counterfactual_kd/metrics/paired.py \
  counterfactual_kd/analysis/calibration.py scripts/tifs/addendum.py scripts/tifs/cross_score.py \
  | awk '{printf "%s=%s;", $2, $1}')"

cd "$HERE"
if [ "$SCORE" = 1 ]; then
  echo "== score.py fill (SST-2 replicate students, determinism re-check)"
  nice -n 10 "$PY" score.py fill --datasets sst2 --out "$OUT" > "$OUT/logs/score_fill_sst2.log" 2>&1
  echo "== score.py refs (every finished clean reference build, both datasets)"
  nice -n 10 "$PY" score.py refs --out "$OUT" > "$OUT/logs/score_refs.log" 2>&1
  echo "== score.py fill (AG News battery students, logit and feature, on the triggers cross_hits lacks)"
  nice -n 10 "$PY" score.py fill --datasets agnews --out "$OUT" > "$OUT/logs/score_fill_agnews.log" 2>&1
  if [ "$SCITECH" = 1 ]; then
    echo "== score.py scitech (item 13; phrases fixed in scitech_registration.json)"
    nice -n 10 "$PY" score.py scitech --out "$OUT" > "$OUT/logs/score_scitech.log" 2>&1
  fi
fi
# Gate: reproduce the official primary.csv and attack_variance.csv rows through this code path
# before any new number is computed (stops here if one differs).
echo "== selfcheck.py --gate (official numbers reproduced first)"
nice -n 10 "$PY" selfcheck.py --gate --out "$OUT" | tee "$OUT/logs/selfcheck_gate.log"
for s in item3_nullswap_own.py item4_a2_2x2.py item7_reference_k.py item5_kappa_marg.py item13_scitech.py \
         selfcheck.py make_readme.py; do
  echo "== $s"
  nice -n 10 "$PY" "$s" --out "$OUT"
done
