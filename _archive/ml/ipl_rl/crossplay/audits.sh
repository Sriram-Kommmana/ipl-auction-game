#!/usr/bin/env bash
# Phase 2E.0 step 12 — safety / parity / performance audits (run on an idle machine,
# after the cross-play runs). Nothing here changes a model or the runtime.
#   bash ml/ipl_rl/crossplay/audits.sh
set -u
cd "$(dirname "$0")/../../.."
AUD=ml/runs/_2e0/audits
PY=ml/.venv/Scripts/python
export PYTHONIOENCODING=utf-8
mkdir -p "$AUD"
NODE="$PY ml/ipl_rl/crossplay/run_node.py"
R=ml/runs
exp() { echo "$R/$1-s1/checkpoints/$2/policy.json $R/$1-s2/checkpoints/$2/policy.json $R/$1-s3/checkpoints/$2/policy.json"; }

echo "AUDIT fallback suites (existing production-contract checks, all 15 exports)"
{
  $NODE ml/ipl_rl/crossplay/ppo_fallback.mjs $(exp ppo-2c3 update_0325); echo "exit $?"
  $NODE ml/reports/phase2d1/a2c_fallback.mjs $(exp a2c-2d1 update_0325); echo "exit $?"
  $NODE ml/reports/phase2d2/d3qn_fallback.mjs $(exp d3qn-2d2 update_0325); echo "exit $?"
  $NODE ml/reports/phase2d3/qrdqn_fallback.mjs $(exp qr-dqn-2d3 update_0325); echo "exit $?"
  $NODE ml/reports/phase2d4/es_fallback.mjs $(exp openai-es-2d4 gen_2000); echo "exit $?"
} > "$AUD/fallback.txt" 2>&1
grep -E "production contract|exit|Error|assert" "$AUD/fallback.txt"

echo "AUDIT parity (real Stage-B states)"
$NODE ml/ipl_rl/crossplay/dump_states.mjs "$AUD/parity_states.json" 12 600 > "$AUD/parity_states.txt" 2>&1; echo "dump exit $?"
$PY ml/ipl_rl/crossplay/parity.py "$AUD/parity_states.json" "$AUD/parity.json" > "$AUD/parity.txt" 2>&1; echo "parity exit $?"
tail -1 "$AUD/parity.txt"

echo "AUDIT performance (real clock, production guard unmodified)"
for i in 1 2 3; do
  $NODE ml/ipl_rl/crossplay/latency.mjs --entries 40 --workers 1 --offset $((i * 100)) --out "$AUD/latency_single_$i.json"
done
for i in 1 2; do
  $NODE ml/ipl_rl/crossplay/latency.mjs --entries 12 --workers 14 --offset $((i * 150)) --out "$AUD/latency_concurrent14_$i.json"
done
$NODE ml/ipl_rl/crossplay/latency.mjs --entries 20 --workers 4 --offset 400 --out "$AUD/latency_concurrent4.json"

node ml/ipl_rl/crossplay/hashes.mjs --check ml/reports/phase2e0/frozen-hashes.json
echo AUDITSDONE
