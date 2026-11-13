#!/usr/bin/env bash
# A14 items 1, 2, 6, 9, 12 (g6-review0/multiplicity): claim type x error control, headline power,
# Holm family sensitivity, delta rationale, S1c config H main-text numbers.
# A14 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.
# CPU only and read-only on the repository except the output folder. About 3-4 min wall, about
# 17-20 CPU-min (almost all in headline_power.py).
#
#   bash scripts/tifs/aux/g6-review0/multiplicity/run_all.sh [OUT]
#
# OUT defaults to results/tifs/addendum/aux/g6-review0/multiplicity. PY overrides the interpreter.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
OUT="${1:-$ROOT/results/tifs/addendum/aux/g6-review0/multiplicity}"
PY="${PY:-/data/Youngho/envs/blade/bin/python}"

export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
# No tokenizer or model is loaded here; the offline switches only make sure nothing reaches the hub.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=/data/Youngho/cache/huggingface

mkdir -p "$OUT/logs"
date -u +"%Y-%m-%d %H:%M:%S UTC" > "$OUT/logs/run_start.txt"
(cd "$ROOT" && sha256sum docs/tifs_plan.md scripts/tifs/analyze.py counterfactual_kd/analysis/tifs.py \
    results/tifs/structural_nulls.json scripts/tifs/addendum.py counterfactual_kd/metrics/paired.py \
    counterfactual_kd/analysis/calibration.py scripts/tifs/aux/g4-s1c-s9-s1extra/common.py \
    scripts/tifs/aux/g4-s1c-s9-s1extra/s1c.py) > "$OUT/logs/hashes_before.txt"

run() {
    local name="$1"; shift
    echo "== $name"
    nice -n 10 "$PY" "$@" --out "$OUT" 2>&1 | tee "$OUT/logs/$name.log"
}

run claims          "$HERE/claims.py"            # item 1
run holm_family     "$HERE/holm_family.py"       # item 6
run delta_rationale "$HERE/delta_rationale.py"   # item 9
run s1c_main        "$HERE/s1c_main.py"          # item 12
run headline_power  "$HERE/headline_power.py" --workers 6 --blocks 8 --reps 10000   # item 2
run selfchecks      "$HERE/selfchecks.py"        # reproductions + review0 comparison
run make_readme     "$HERE/make_readme.py"       # README.md + provenance.csv from the CSVs
date -u +"%Y-%m-%d %H:%M:%S UTC" > "$OUT/logs/run_end.txt"
echo "done: $OUT"
