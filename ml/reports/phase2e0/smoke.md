# Phase 2E.0 smoke test (run before the full evaluation)

**Result: PASS.**

## What ran

The script is `ml/ipl_rl/crossplay/smoke.sh`, and raw outputs are in `ml/runs/_2e0/smoke/`. Every run used the Windows EcoQoS opt-out; this changes speed only.

| Configuration | Controllers | Episodes |
|---|---|---|
| A: Stage A control | all 15 exports | 15 × 20 validation entries = 300 |
| C1: head-to-head | all 20 ordered algorithm pairs, s1 × s1 | 20 × 15 entries = 300 |
| C4: saturated | all 20 ordered algorithm pairs, s1 × s1 | 20 × 15 entries = 300 |
| S4: mixed room | all 5 learners (s1) with the other 4 algorithms (s1) | 5 × 20 entries = 100 |

Each configuration ran twice: pass a on 14 worker threads, pass b on 5.

## Checks

| Check | Result |
|---|---|
| Policies load | 15/15 accepted by the production loader. Algorithm id, head, selection mode, obs hash `629b25783f833af7` and act hash `5f72f510c48b1f46` are all correct, and every sha256 matches the gap-report record. |
| Seat assignments | C1 has 1 RL opponent seat in 300/300 rooms; C4 has 4 in 300/300; S4 has 4 distinct algorithms (not the learner's) in 100/100. Every opponent seat holds the assigned export (weights digest verified). |
| No training | The harness has no optimiser, gradient, replay or perturbation path. Frozen files and exports hash-identical before and after; frozen sources clean vs git HEAD. |
| Opponent inference | Production `createRlSeat` path. 222,705 (C4) and 62,791 (S4) opponent RL decisions; **0 fallbacks**. |
| Illegal or masked actions | 0. A masked learner action or any opponent fallback would have stopped the run. |
| NaN / Infinity | 0 |
| Crashes / deadlocks | 0 |
| Invariant violations (all teams) | 0 |
| Learner legal XI | 1,000/1,000 |
| RL-opponent incomplete XI | 0 |
| Deterministic replay | Pass a and pass b are **IDENTICAL** episode by episode in all four configurations: auction history, learner actions, opponent actions, metrics and summaries. |
| Stage A reproduction | 300/300 control episodes are field-for-field identical to the stored Stage A evaluator episodes. |

The only incomplete XIs belong to the passive human proxy, which never bids by design (120 in C4, 50 in S4). These are recorded but are not a safety event.
