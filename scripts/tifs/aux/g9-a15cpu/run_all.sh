#!/usr/bin/env bash
# A15 CPU items (docs/tifs_addendum.md A15, block 'CPU 항목 (GPU 없음)' and E3):
#   item 1 Welch df (E3), item 2 k_T, item 3 null-swap corrections, item 4 transmission, item 5 G3 dose,
#   then the independent review, the self-checks and the READMEs.
# A15 post hoc CPU analysis; changes no primary verdict, battery decision or the paper-wide Holm.
#
# CPU only (CUDA_VISIBLE_DEVICES=''), nice 10, HF offline. Read-only on runs/, results/tifs/report,
# results/tifs/addendum and the code; writes only results/tifs/addendum/aux/g9-a15cpu (or --out DIR).
# About 1 min wall. Every item script first reproduces the official tables through the frozen loader and
# stops on a mismatch.
#
#   scripts/tifs/aux/g9-a15cpu/run_all.sh [--out DIR]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
PY=${PY:-/data/Youngho/envs/blade/bin/python}
OUT="$ROOT/results/tifs/addendum/aux/g9-a15cpu"
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
mkdir -p "$OUT/logs"

export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=/data/Youngho/cache/huggingface

date -u +"%Y-%m-%d %H:%M:%S UTC" > "$OUT/logs/run_start.txt"
# Frozen files, tracked code, the corpus construction code item 5 imports, and every official input.
(cd "$ROOT" && sha256sum \
  docs/tifs_plan.md scripts/tifs/analyze.py counterfactual_kd/analysis/tifs.py results/tifs/structural_nulls.json \
  scripts/tifs/addendum.py counterfactual_kd/metrics/paired.py counterfactual_kd/analysis/calibration.py \
  docs/tifs_addendum.md \
  counterfactual_kd/adapters/native_kd.py counterfactual_kd/attacks/base.py counterfactual_kd/attacks/badnets.py \
  counterfactual_kd/registry.py counterfactual_kd/data/loaders.py \
  results/tifs/report/primary.csv results/tifs/report/variance.csv results/tifs/report/battery.csv \
  results/tifs/addendum/global_holm.csv results/tifs/addendum/cross_null.csv \
  results/tifs/addendum/cross_hits_bert_sst2.json results/tifs/addendum/cross_hits_bert_agnews.json \
  results/tifs/addendum/aux/g6-review0/reference/nullswap_own_summary.csv \
  results/tifs/addendum/aux/g6-review0/reference/kappa_marg.csv \
  results/tifs/addendum/aux/g1-exposure-controls-tokens/tokens_contaminated_corpus.csv) > "$OUT/logs/hashes_before.txt"

cd "$HERE"
run() {
  local name="$1"; shift
  echo "== $name"
  nice -n 10 "$PY" "$@" --out "$OUT" 2>&1 | tee "$OUT/logs/$name.log"
}
run item1_welch_df      item1_welch_df.py
run item2_teacher_effect item2_teacher_effect.py
run item3_nullswap      item3_nullswap.py
run item4_transmission  item4_transmission.py
run item5_g3_dose       item5_g3_dose.py
run independent         -I independent.py
run selfcheck           selfcheck.py
run make_readme         make_readme.py
date -u +"%Y-%m-%d %H:%M:%S UTC" > "$OUT/logs/run_end.txt"
echo "done: $OUT"
