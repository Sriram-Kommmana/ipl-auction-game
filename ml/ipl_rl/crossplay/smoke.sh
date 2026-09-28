#!/usr/bin/env bash
# Phase 2E.0 deterministic smoke test: every composition on a small block,
# twice (14 workers, then 5 workers — scheduling must not matter), then an
# exact episode-by-episode comparison and the Stage-A check.
#   bash ml/ipl_rl/crossplay/smoke.sh <out_dir>
set -u
OUT=${1:-ml/runs/_2e0/smoke}
cd "$(dirname "$0")/../../.."
mkdir -p "$OUT"
PY=ml/.venv/Scripts/python
run() { $PY ml/ipl_rl/crossplay/run_node.py ml/ipl_rl/crossplay/run.mjs "$@" || { echo "SMOKE STOP ($*)"; exit 2; }; }
for pass in a b; do
  W=14; [ "$pass" = b ] && W=5
  run --cond A  --limit 20 --workers $W --out "$OUT/A_$pass.jsonl"
  run --cond C1 --limit 15 --seeds 1 --workers $W --out "$OUT/C1_$pass.jsonl"
  run --cond C4 --limit 15 --seeds 1 --workers $W --out "$OUT/C4_$pass.jsonl"
  run --cond S4 --limit 20 --seeds 1 --workers $W --out "$OUT/S4_$pass.jsonl"
done
PYTHONIOENCODING=utf-8 $PY ml/ipl_rl/crossplay/compare_runs.py \
  "$OUT/A_a.jsonl" "$OUT/A_b.jsonl" "$OUT/C1_a.jsonl" "$OUT/C1_b.jsonl" \
  "$OUT/C4_a.jsonl" "$OUT/C4_b.jsonl" "$OUT/S4_a.jsonl" "$OUT/S4_b.jsonl" > "$OUT/compare.txt"; echo "compare exit $?" >> "$OUT/compare.txt"
PYTHONIOENCODING=utf-8 $PY ml/ipl_rl/crossplay/check_stage_a.py "$OUT/A_a.jsonl" --limit 20 > "$OUT/stage_a.txt"; echo "stage-a exit $?" >> "$OUT/stage_a.txt"
echo SMOKEDONE
