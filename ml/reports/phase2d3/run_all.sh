#!/bin/sh
# Phase 2D.3: QR-DQN seeds 1, 2, 3 sequentially (same as PPO 2C.3, A2C 2D.1 and D3QN 2D.2, for comparable throughput).
cd "/c/web dev projects/ipl-auction-game/ml"
for s in 1 2 3; do
  .venv/Scripts/python -m ipl_rl.algos.qr_dqn --config ipl_rl/configs/qr_dqn_2d3.json --set seed=$s "run_name=\"qr-dqn-2d3-s$s\"" > runs/qr-dqn-2d3-s$s.log 2>&1
  rc=$?; echo "seed $s exit $rc" >> runs/_2d3/run_all.status
  if [ $rc -ne 0 ]; then echo STOPPED >> runs/_2d3/run_all.status; exit $rc; fi
done
echo ALLDONE >> runs/_2d3/run_all.status
