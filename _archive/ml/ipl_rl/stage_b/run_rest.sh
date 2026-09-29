#!/usr/bin/env bash
# Phase 2F — remaining 14 Stage-B pilot runs (sequential), each verified after training.
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
for spec in ppo:102 ppo:103 a2c:101 a2c:102 a2c:103 d3qn:101 d3qn:102 d3qn:103 qrdqn:101 qrdqn:102 qrdqn:103 es:101 es:102 es:103; do
  a=${spec%%:*}; s=${spec##*:}
  echo "=== $a s$s $(date -Iseconds)"
  .venv/Scripts/python -u -m ipl_rl.stage_b.train_b --algo $a --seed $s > runs/stage_b/logs/${a}_s${s}.log 2>&1
  echo "train exit $? ($a s$s)"
  .venv/Scripts/python -m ipl_rl.stage_b.verify_run runs/stage_b/$a/s$s > runs/stage_b/logs/${a}_s${s}.verify.log 2>&1
  echo "verify exit $? ($a s$s) $(tail -1 runs/stage_b/logs/${a}_s${s}.verify.log)"
done
echo RUNRESTDONE
