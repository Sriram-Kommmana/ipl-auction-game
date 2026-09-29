#!/usr/bin/env bash
# Phase 2F — evaluation of all Stage-B pilot checkpoints (after training).
#   c100 / c250: A, C1, C4, S4 on validation entries 0–99 (decision: reduced earlier)
#   c500:        the exact Phase 2E.0 protocol on all 500 validation entries
#   c500 per-decision replay on the 40 Phase 2E.2 entries (digest-verified)
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python
NODE="$PY ipl_rl/diagnosis/run_node.py"
E=runs/stage_b/eval
(cd .. && node ml/ipl_rl/stage_b/hashes_b.mjs --check ml/reports/phase2f/raw/hashes-pre.json) || { echo "HASH STOP (before)"; exit 2; }
$PY -m ipl_rl.stage_b.make_learners || { echo "LEARNER STOP"; exit 2; }
for lv in c100 c250 c500; do
  lim=100; [ $lv = c500 ] && lim=500
  mkdir -p $E/$lv
  for c in A S4 C1 C4; do
    echo "=== $lv $c $(date -Iseconds)"
    $NODE ipl_rl/stage_b/eval_b.mjs --learners $E/learners_$lv.json --cond $c --limit $lim --out $E/$lv/$c.jsonl --workers 14 | tail -1
    [ -f $E/$lv/$c.jsonl.STOP.json ] && { echo "EVAL SAFETY STOP $lv $c"; exit 2; }
  done
done
mkdir -p $E/replay_c500
echo "=== replay c500 $(date -Iseconds)"
$NODE ipl_rl/stage_b/replay_b.mjs $E/replay_c500 $E/learners_c500.json $E/c500 $(cat runs/_2e2/K.txt) | tail -1
(cd .. && node ml/ipl_rl/stage_b/hashes_b.mjs --check ml/reports/phase2f/raw/hashes-pre.json) || { echo "HASH STOP (after)"; exit 2; }
echo RUNEVALDONE
