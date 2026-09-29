#!/bin/sh
# Phase 2D.2: D3QN seeds 1, 2, 3 sequentially (same as PPO 2C.3 and A2C 2D.1, for comparable throughput).
cd "/c/web dev projects/ipl-auction-game/ml"
for s in 1 2 3; do
  .venv/Scripts/python -m ipl_rl.algos.d3qn --config ipl_rl/configs/d3qn_2d2.json --set seed=$s "run_name=\"d3qn-2d2-s$s\"" > runs/d3qn-2d2-s$s.log 2>&1
  rc=$?; echo "seed $s exit $rc" >> runs/_2d2/run_all.status
  if [ $rc -ne 0 ]; then echo STOPPED >> runs/_2d2/run_all.status; exit $rc; fi
done
echo ALLDONE >> runs/_2d2/run_all.status
