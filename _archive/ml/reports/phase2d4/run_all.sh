#!/bin/sh
# Phase 2D.4: OpenAI-ES seeds 1, 2, 3 sequentially (same as PPO, A2C, D3QN and QR-DQN, for comparable throughput).
cd "/c/web dev projects/ipl-auction-game/ml"
for s in 1 2 3; do
  .venv/Scripts/python -m ipl_rl.algos.openai_es --config ipl_rl/configs/openai_es_2d4.json --set seed=$s "run_name=\"openai-es-2d4-s$s\"" > runs/openai-es-2d4-s$s.log 2>&1
  rc=$?; echo "seed $s exit $rc" >> runs/_2d4/run_all.status
  if [ $rc -ne 0 ]; then echo STOPPED >> runs/_2d4/run_all.status; exit $rc; fi
done
echo ALLDONE >> runs/_2d4/run_all.status
