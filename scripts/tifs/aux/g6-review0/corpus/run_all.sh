#!/usr/bin/env bash
# A14 items 8, 10, 11 (group g6-review0, subgroup 'corpus'): run order and environment.
# A14 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.
# CPU only. Reads runs/tifs, logs/, results/tifs/{report,addendum,teachers}; writes only to
# $OUT (default results/tifs/addendum/aux/g6-review0/corpus/), run logs in $OUT/logs/.
#
#   scripts/tifs/aux/g6-review0/corpus/run_all.sh                 # every step, in order
#   scripts/tifs/aux/g6-review0/corpus/run_all.sh selection checks # named steps
#
# Steps: recount (item 8 recount, ~1.5 min), exposure (item 8), selection (item 11),
# devcheck (item 11 check, CPU fp32 scoring of 7 teachers, ~15-30 min), ws_rescore
# (item 10 diagnostic, CPU fp32 scoring of 23 students on every probe, ~2 h from scratch; resumable), whitespace
# (item 10), checks (self_checks.csv), readme (README.md). exposure needs recount;
# devcheck needs selection; whitespace reads the ws_rescore output if it is complete.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
OUT="${OUT:-$ROOT/results/tifs/addendum/aux/g6-review0/corpus}"
PY="${PY:-/data/Youngho/envs/blade/bin/python}"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" \
       HF_HOME="${HF_HOME:-/data/Youngho/cache/huggingface}" \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 RAYON_NUM_THREADS=8 TOKENIZERS_PARALLELISM=true
mkdir -p "$OUT/logs"
cd "$ROOT"

run() {  # run <log name> <script> [args...]
  local log="$1"; shift
  echo "$(date -u +%FT%TZ) start $*" >&2
  nice -n 10 "$PY" -B "$HERE/$1" "${@:2}" --out "$OUT" > "$OUT/logs/$log" 2>&1
  echo "$(date -u +%FT%TZ) done  $*" >&2
}

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(recount exposure selection devcheck ws_rescore whitespace checks readme)
for s in "${steps[@]}"; do
  case "$s" in
    recount)    run item8_recount.log item8_recount.py ;;
    exposure)   run item8_exposure.log item8_exposure.py ;;
    selection)  run item11_selection.log item11_selection.py ;;
    devcheck)   run item11_devcheck.log item11_devcheck.py --all-differing ;;
    ws_rescore) run item10_ws_rescore.log item10_ws_rescore.py --fams gpt2 qwen3 --seeds 42 43 44 --threads 8 ;;
    whitespace) run item10_whitespace.log item10_whitespace.py ;;
    checks)     run self_checks.log self_checks.py ;;
    readme)     run make_readme.log make_readme.py ;;
    *) echo "unknown step: $s" >&2; exit 2 ;;
  esac
done
