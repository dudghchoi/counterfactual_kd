#!/usr/bin/env bash
# A18 analysis (docs/tifs_addendum.md "## A18", registered 10/31 16:05 KST). CPU only. Post hoc, outside the
# 31-cell family.
#
# Registered to run ONCE, when every A18 job (queue v11) is terminal (ok / failed / blocked) or at the deadline
# (11/01 18:00 KST = 09:00 UTC). Before that it stops (exit 4) without reading any A18 output (override: --force).
# The tracked files (frozen code and records, the A15 analysis library, this directory, docs, the queue plan) are
# hashed before and after the run; any difference fails the run.
#
#   bash run_all.sh [--runs DIR] [--out DIR] [--force]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
PY="${PY:-/data/Youngho/envs/blade/bin/python}"
RUNS="$ROOT/runs/tifs"
OUT="$ROOT/results/tifs/addendum/aux/g12-a18"
FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --force) FORCE=1 ;;
    --runs) RUNS="$2"; shift ;;
    --out) OUT="$2"; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
cd "$HERE"
# Readiness (names and status markers only).
"$PY" - "$RUNS" "$FORCE" <<'EOF'
import os, sys, time, calendar
sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "..")))
import queue_plan as Q
runs, force = sys.argv[1], sys.argv[2] == "1"
def status(j):
    for s in ("ok", "failed", "blocked", "running"):
        if os.path.exists(os.path.join(runs, "_status", f"{j}.{s}")):
            return s
    return "pending"
open_ = [j["id"] for j in Q.a18_pool_jobs() if status(j["id"]) in ("running", "pending")]
deadline = calendar.timegm(time.strptime("2026-11-01 09:00:00", "%Y-%m-%d %H:%M:%S"))
if open_ and time.time() < deadline and not force:
    print(f"A18 not finished ({len(open_)} jobs open: {','.join(open_[:6])}...) and the deadline has not passed")
    sys.exit(4)
print(f"A18 ready: {len(open_)} jobs open" + (" (deadline passed or --force: open jobs are cut)" if open_ else ""))
EOF
rc=$?
if [ "$rc" = 4 ]; then echo "nothing written or read (use --force)"; exit 4; fi
[ "$rc" = 0 ] || exit "$rc"
mkdir -p "$OUT/logs"
hash_tracked() {
  (cd "$ROOT" && ls docs/tifs_plan.md docs/tifs_addendum.md scripts/tifs/analyze.py scripts/tifs/addendum.py \
      scripts/tifs/queue_plan.py counterfactual_kd/analysis/tifs.py counterfactual_kd/metrics/paired.py \
      results/tifs/structural_nulls.json scripts/tifs/aux/g8-a15/analysis/*.py scripts/tifs/aux/g11-a17/*.py scripts/tifs/aux/g11-a17/analysis/*.py scripts/tifs/aux/g12-a18/*.py \
      scripts/tifs/aux/g12-a18/analysis/*.py scripts/tifs/aux/g12-a18/analysis/run_all.sh \
      results/tifs/report/*.csv results/tifs/addendum/*.csv 2>/dev/null | sort | xargs sha256sum)
}
hash_tracked > "$OUT/logs/hashes_before.txt"
date -u '+%F %T UTC' > "$OUT/logs/started.txt"
nice -n 10 "$PY" a18.py --runs "$RUNS" --out "$OUT" 2>&1 | tee "$OUT/logs/a18.log"
rc1=${PIPESTATUS[0]}
nice -n 10 "$PY" independent.py --runs "$RUNS" --out "$OUT" 2>&1 | tee "$OUT/logs/independent.log"
rc2=${PIPESTATUS[0]}
hash_tracked > "$OUT/logs/hashes_after.txt"
date -u '+%F %T UTC' > "$OUT/logs/finished.txt"
if ! diff -q "$OUT/logs/hashes_before.txt" "$OUT/logs/hashes_after.txt" >/dev/null; then
  echo "tracked files changed during the run"; exit 5
fi
echo "a18.py exit $rc1, independent.py exit $rc2; tracked files unchanged ($(wc -l < "$OUT/logs/hashes_before.txt") files)"
[ "$rc1" = 0 ] && [ "$rc2" = 0 ]
