#!/usr/bin/env bash
# A11 items 1, 2, 13 (group g1-exposure-controls-tokens): run order and environment.
# Post hoc auxiliary tables (docs/tifs_addendum.md A11); they change no primary verdict.
# CPU only (no GPU). Reads runs/tifs, results/tifs/report and results/tifs/addendum;
# writes only to $OUT (default results/tifs/addendum/aux/g1-exposure-controls-tokens/),
# run logs in $OUT/logs/.
#
#   scripts/tifs/aux/g1-exposure-controls-tokens/run_all.sh            # every step, in order
#   scripts/tifs/aux/g1-exposure-controls-tokens/run_all.sh tokens     # one or more named steps
#
# Steps (in this order): exposure (items 1, 2; ~10 s), tokens (item 13; ~8 min),
# ws_qwen3 (diagnostic, CPU fp32; ~20 min), ws_gpt2 (diagnostic; ~5 min),
# twins (diagnostic; ~1 min), checks (self_checks.csv), readme (README.md files).
# checks and readme read the outputs of the earlier steps, so run them last.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
OUT="${OUT:-$ROOT/results/tifs/addendum/aux/g1-exposure-controls-tokens}"
PY="${PY:-/data/Youngho/envs/blade/bin/python}"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" \
       HF_HOME="${HF_HOME:-/data/Youngho/cache/huggingface}" \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       RAYON_NUM_THREADS=8 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=true
# self_checks.py reports any __pycache__ / .pyc in the repository newer than this (epoch s).
export G1_SESSION_START="${G1_SESSION_START:-$(date +%s)}"
mkdir -p "$OUT/logs"

run() {  # run <log name> <script> [args...]
  local log="$1"; shift
  echo "$(date -u +%FT%TZ) start $*" >&2
  "$PY" -B "$HERE/$1" "${@:2}" --out "$OUT" > "$OUT/logs/$log" 2>&1
  echo "$(date -u +%FT%TZ) done  $*" >&2
}

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(exposure tokens ws_qwen3 ws_gpt2 twins checks readme)
for s in "${steps[@]}"; do
  case "$s" in
    exposure) run exposure_controls.log exposure_controls.py ;;
    tokens)   run trigger_tokens.log trigger_tokens.py ;;
    ws_qwen3) run ws_diag.log ws_diag.py qwen3 ;;
    ws_gpt2)  run ws_diag_gpt2.log ws_diag.py gpt2 ;;
    twins)    run teacher_twins_check.log teacher_twins_check.py ;;
    checks)   run self_checks.log self_checks.py ;;
    readme)   run make_readme.log make_readme.py ;;
    *) echo "unknown step: $s" >&2; exit 2 ;;
  esac
done
