#!/usr/bin/env bash
# Phase 2E.0 resume after the C4 safety stop (Option A approved): C4 from scratch, S4,
# reproducibility subset, hash check. A and C1 (complete, clean) are not rerun.
#   bash ml/ipl_rl/crossplay/run_rest.sh   (from the repository root)
set -u
cd "$(dirname "$0")/../../.."
OUT=ml/runs/_2e0/full
REP=ml/runs/_2e0/repro
HASHES=ml/reports/phase2e0/frozen-hashes.json
PY=ml/.venv/Scripts/python
export PYTHONIOENCODING=utf-8
mkdir -p "$OUT" "$REP"
stop() { echo "PIPELINE STOP: $*"; exit 2; }
run() { $PY ml/ipl_rl/crossplay/run_node.py ml/ipl_rl/crossplay/run.mjs "$@" || stop "run.mjs $*"; }

echo "STEP 5 frozen hashes (resume after the approved Option A; A and C1 already complete and not rerun)"
node ml/ipl_rl/crossplay/hashes.mjs --check "$HASHES" || stop "hash mismatch before the run"
echo "STEP 9 C4 saturated"
run --cond C4 --workers 14 --hashes "$HASHES" --out "$OUT/C4.jsonl"
echo "STEP 10 S4 mixed"
run --cond S4 --workers 14 --hashes "$HASHES" --out "$OUT/S4.jsonl"

echo "STEP 11 reproducibility subset (separate processes, other worker counts)"
run --cond A  --limit 50  --workers 7  --hashes "$HASHES" --out "$REP/A.jsonl"
run --cond C1 --limit 30  --workers 9  --hashes "$HASHES" --out "$REP/C1.jsonl"
run --cond C4 --limit 30  --workers 11 --hashes "$HASHES" --out "$REP/C4.jsonl"
run --cond S4 --limit 100 --workers 5  --hashes "$HASHES" --out "$REP/S4.jsonl"
$PY ml/ipl_rl/crossplay/compare_runs.py --subset \
  "$OUT/A.jsonl" "$REP/A.jsonl" "$OUT/C1.jsonl" "$REP/C1.jsonl" "$OUT/C4.jsonl" "$REP/C4.jsonl" "$OUT/S4.jsonl" "$REP/S4.jsonl" \
  "$OUT/A.jsonl" ml/runs/_2e0/smoke/A_a.jsonl "$OUT/C1.jsonl" ml/runs/_2e0/smoke/C1_a.jsonl "$OUT/C4.jsonl" ml/runs/_2e0/smoke/C4_a.jsonl "$OUT/S4.jsonl" ml/runs/_2e0/smoke/S4_a.jsonl \
  > "$REP/compare.txt" || { grep -v '^\[' "$REP/compare.txt"; stop "NONDETERMINISTIC REPLAY"; }
grep -v '^\[' "$REP/compare.txt"

node ml/ipl_rl/crossplay/hashes.mjs --check "$HASHES" || stop "hash mismatch after the run"
echo ALLDONE-2E0
