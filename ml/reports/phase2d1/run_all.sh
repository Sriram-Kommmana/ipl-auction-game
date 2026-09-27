#!/bin/sh
# Phase 2D.1: A2C seeds 1, 2, 3 sequentially (same as PPO 2C.3, for comparable throughput).
cd "/c/web dev projects/ipl-auction-game/ml"
for s in 1 2 3; do
  .venv/Scripts/python -m ipl_rl.algos.a2c --config ipl_rl/configs/a2c_2d1.json --set seed=$s "run_name=\"a2c-2d1-s$s\"" > runs/a2c-2d1-s$s.log 2>&1
  rc=$?; echo "seed $s exit $rc" >> runs/_2d1/run_all.status
  if [ $rc -ne 0 ]; then echo STOPPED >> runs/_2d1/run_all.status; exit $rc; fi
done
echo ALLDONE >> runs/_2d1/run_all.status
