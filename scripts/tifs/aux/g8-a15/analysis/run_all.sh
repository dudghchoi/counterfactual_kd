#!/usr/bin/env bash
# A15 analysis (docs/tifs_addendum.md "## A15", the 10/26 19:55 e2 amendment and the 10/27 20:58 KST amendment
# "마감 연장, 판정 위계, claim rule", item 5). CPU only. A15 post hoc analysis; changes no primary verdict,
# battery decision or the paper-wide Holm.
#
# Registered to run ONCE, when every A15 job is terminal or at the deadline (10/29 14:00 KST). inventory.py
# stops the run (exit 4) before any A15 output is read if a job is still running/pending and the deadline has
# not passed (override: --force).
#
#   bash run_all.sh --dry-run        list the inputs and the cut / blocked / missing jobs; writes nothing
#   bash run_all.sh                  the analysis -> results/tifs/addendum/aux/g8-a15/
#   options: --runs DIR --out DIR --teachers-x15 DIR --hits-dir DIR --no-score --force
#            --now 'YYYY-mm-dd HH:MM:SS' (UTC; tests only: the time the readiness check compares with the deadline)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
PY="${PY:-/data/Youngho/envs/blade/bin/python}"
RUNS="$ROOT/runs/tifs"
OUT="$ROOT/results/tifs/addendum/aux/g8-a15"
TX15="$ROOT/results/tifs/teachers_x15"
HITS=""
DRY=0; FORCE=""; SCORE=1; NOWARG=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --force) FORCE="--force" ;;
    --no-score) SCORE=0 ;;
    --runs) RUNS="$2"; shift ;;
    --out) OUT="$2"; shift ;;
    --teachers-x15) TX15="$2"; shift ;;
    --hits-dir) HITS="$2"; shift ;;
    --now) NOWARG=(--now "$2"); shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
[ -n "$HITS" ] || HITS="$OUT/hits_em"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
cd "$HERE"
COMMON=(--runs "$RUNS" --out "$OUT" --teachers-x15 "$TX15")
if [ "$DRY" = 1 ]; then
  nice -n 10 "$PY" inventory.py --dry-run "${COMMON[@]}" --hits-dir "$HITS"
  echo "CPU fp32 scoring plan (score_em.py):"
  nice -n 10 "$PY" score_em.py --dry-run --runs "$RUNS" --out "$OUT"
  exit 0
fi
# Readiness first, before anything is written or any A15 output is read.
nice -n 10 "$PY" inventory.py --check-ready "${COMMON[@]}" --hits-dir "$HITS" "${NOWARG[@]}" $FORCE
rc=$?
if [ "$rc" = 4 ]; then echo "A15 not finished and the deadline has not passed: nothing written or read (use --force)"; exit 4; fi
[ "$rc" = 0 ] || exit "$rc"
mkdir -p "$OUT/logs"
FAIL=0
step() { local name="$1"; shift; echo "== $name"; nice -n 10 "$@" 2>&1 | tee "$OUT/logs/$name.log"; local rc=${PIPESTATUS[0]}
         if [ "$rc" != 0 ]; then echo "!! $name exited $rc" | tee -a "$OUT/logs/failures.log"; FAIL=1; fi; return "$rc"; }
step record_hashes "$PY" selfcheck.py --record-before "${COMMON[@]}" || exit 1
nice -n 10 "$PY" inventory.py "${COMMON[@]}" --hits-dir "$HITS" "${NOWARG[@]}" $FORCE 2>&1 | tee "$OUT/logs/inventory.log"
rc=${PIPESTATUS[0]}
if [ "$rc" = 4 ]; then echo "A15 not finished and the deadline has not passed: nothing read (use --force)"; exit 4; fi
[ "$rc" = 0 ] || exit "$rc"
if [ "$SCORE" = 1 ]; then
  step score_em "$PY" score_em.py --runs "$RUNS" --out "$OUT" --recheck
fi
step part_a "$PY" part_a.py "${COMMON[@]}"
step part_b1 "$PY" part_b1.py "${COMMON[@]}"
step part_b2 "$PY" part_b2.py "${COMMON[@]}" --hits-dir "$HITS"
step part_c "$PY" part_c.py "${COMMON[@]}"
step part_d "$PY" part_d.py "${COMMON[@]}"
step part_e "$PY" part_e.py "${COMMON[@]}"
step independent "$PY" independent.py --runs "$RUNS" --out "$OUT"
step selfcheck "$PY" selfcheck.py "${COMMON[@]}"
step readme "$PY" make_readme.py "${COMMON[@]}"
echo "run_all.sh: done (failures: $FAIL); outputs in $OUT"
exit "$FAIL"
