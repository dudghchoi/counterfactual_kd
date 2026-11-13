#!/usr/bin/env bash
# A13 post hoc analysis (docs/tifs_addendum.md A13: G1-G4 and 'common').
# A13 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.
#
# CPU only (CUDA_VISIBLE_DEVICES=""), nice -n 10, HF offline. Reads runs/tifs (predictions, checkpoints and
# events.jsonl, read only), results/tifs/report, results/tifs/addendum (incl. aux/g6-review0, aux/g9-a15cpu);
# writes only results/tifs/addendum/aux/g7-a13 (item 7 extension in its item7/ subfolder) and this folder's
# README.md. The frozen files are imported and called, never edited.
#
# usage: scripts/tifs/aux/g7-a13/run_all.sh [--no-score]
#   --no-score   skip the CPU fp32 re-scoring of the G2 references for item 7 (score.py refs is resumable:
#                with the hits already present it only re-checks checkpoint sha256 and skips)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
PY=${PY:-/data/Youngho/envs/blade/bin/python}
OUT="$ROOT/results/tifs/addendum/aux/g7-a13"
ITEM7="$OUT/item7"
G6="$ROOT/scripts/tifs/aux/g6-review0/reference"
G6OUT="$ROOT/results/tifs/addendum/aux/g6-review0/reference"
SCORE=1
while [ $# -gt 0 ]; do
  case "$1" in
    --no-score) SCORE=0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
mkdir -p "$OUT/logs" "$ITEM7/hits" "$ITEM7/logs"

export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=/data/Youngho/cache/huggingface
export G7_RUN_START_UTC="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
export G7_SHA256_START="$(cd "$ROOT" && sha256sum docs/tifs_plan.md scripts/tifs/analyze.py \
  counterfactual_kd/analysis/tifs.py results/tifs/structural_nulls.json scripts/tifs/addendum.py \
  counterfactual_kd/metrics/paired.py counterfactual_kd/analysis/calibration.py scripts/tifs/cross_score.py \
  docs/tifs_addendum.md scripts/tifs/aux/g6-review0/reference/common.py \
  scripts/tifs/aux/g6-review0/reference/score.py scripts/tifs/aux/g6-review0/reference/item7_reference_k.py \
  | awk '{printf "%s=%s;", $2, $1}')"
cd "$HERE"
export G7_G6_DIGEST_START="$(nice -n 10 "$PY" -c 'import selfcheck; print(selfcheck.g6_digest())')"

echo "== selfcheck.py --gate (frozen hashes; official tables reproduced before anything is computed)"
nice -n 10 "$PY" selfcheck.py --gate --out "$OUT" | tee "$OUT/logs/selfcheck_gate.log"

# Item 7 extension (A14 item 7 + A13 G2 t45/t46): the A14 subgroup's scripts, unchanged, with --out ITEM7.
# Its hits folder starts from copies of the A14 hit files (the A14 folder is not written).
for f in ref_tifs_aa_bert_clean_t43.sst2.json ref_tifs_aa_bert_clean_t43.agnews.json \
         ref_tifs_aa_bert_clean_t44.sst2.json ref_tifs_aa_bert_clean_t44.agnews.json \
         fill_bert_sst2.json fill_bert_agnews.json recheck_bert_sst2.json; do
  [ -e "$ITEM7/hits/$f" ] || cp -p "$G6OUT/hits/$f" "$ITEM7/hits/$f"
done
if [ "$SCORE" = 1 ]; then
  echo "== g6-review0 score.py refs (t45, t46; CPU fp32; resumable)"
  nice -n 10 "$PY" "$G6/score.py" refs --runs tifs_x13_bert_ref_t45 tifs_x13_bert_ref_t46 --out "$ITEM7" \
    > "$ITEM7/logs/score_refs_run_all.log" 2>&1
fi
echo "== g6-review0 item7_reference_k.py --out item7"
nice -n 10 "$PY" "$G6/item7_reference_k.py" --out "$ITEM7" | tee "$ITEM7/logs/item7.log"

for s in g1_qwen3_agnews.py "g2_reference.py --with-item7" g3_near_delta.py g4_low_dose.py common_floor_jobs.py; do
  echo "== $s"
  # shellcheck disable=SC2086
  nice -n 10 "$PY" $s --out "$OUT" | tee "$OUT/logs/${s%% *}.log"
done
echo "== independent.py (no frozen import)"
nice -n 10 "$PY" -I independent.py --out "$OUT" | tee "$OUT/logs/independent.log"
echo "== selfcheck.py (full)"
nice -n 10 "$PY" selfcheck.py --out "$OUT" | tee "$OUT/logs/selfcheck.log"
echo "== make_readme.py"
nice -n 10 "$PY" make_readme.py --out "$OUT"
