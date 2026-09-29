#!/usr/bin/env bash
# Phase 2E.1 read-only trajectory replays (digest-verified against Phase 2E.0 records).
set -u
cd "$(dirname "$0")/../../.."
PY=ml/.venv/Scripts/python
R=ml/runs/_2e0/full
O=ml/runs/_2e1
run() { $PY ml/ipl_rl/diagnosis/run_node.py ml/ipl_rl/diagnosis/replay.mjs "$@" || { echo "REPLAY STOP ($*)"; exit 2; }; }
run traj $O/traj_A.jsonl A $R/A.jsonl
run traj $O/traj_S4.jsonl S4 $R/S4.jsonl
run traj $O/traj_C1.jsonl C1 $R/C1.jsonl --same-s1
run traj $O/traj_C4.jsonl C4 $R/C4.jsonl --same-s1
echo REPLAYALLDONE
