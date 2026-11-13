#!/usr/bin/env bash
# A11 items 10-12 (group g4-s1c-s9-s1extra): S1c, S9 and the S1 extra rows.
# Post hoc / registered auxiliary tables (docs/tifs_addendum.md A11); they change no
# primary verdict, battery decision or the paper-wide Holm. CPU only, read-only on the
# repository except the output folder.
#
#   bash scripts/tifs/aux/g4-s1c-s9-s1extra/run_all.sh [OUT]
#
# OUT defaults to results/tifs/addendum/aux/g4-s1c-s9-s1extra. PY overrides the interpreter.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
OUT="${1:-$ROOT/results/tifs/addendum/aux/g4-s1c-s9-s1extra}"
PY="${PY:-/data/Youngho/envs/blade/bin/python}"

export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# This group loads no tokenizer or model; the offline switches only make sure nothing reaches the hub.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

mkdir -p "$OUT/logs"
rm -f "$OUT/logs/cpu_seconds.csv"
date -u +"%Y-%m-%d %H:%M:%S UTC" > "$OUT/logs/run_start.txt"

run() {
    local name="$1"; shift
    echo "== $name"
    nice -n 10 "$PY" "$@" --out "$OUT" 2>&1 | tee "$OUT/logs/$name.log"
}

run s1_extra    "$HERE/s1_extra.py" --workers 4   # item 12: frozen run_scenario rows (about 8 CPU-min)
run s1c         "$HERE/s1c.py" --workers 6        # item 10: S1c Monte Carlo (99 jobs as 162 seed blocks)
run s9          "$HERE/s9.py"                     # item 11: S9 (imports s1_extra.scenarios)
run recheck_s1b "$HERE/recheck_s1b.py"            # S1b-mode recheck; reads s1c_check_s1b.csv
run checks      "$HERE/checks.py"                 # self-checks -> checks.csv
run make_readme "$HERE/make_readme.py"            # README.md from the CSVs
echo "done: $OUT"
