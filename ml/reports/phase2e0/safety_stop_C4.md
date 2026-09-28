# Phase 2E.0 safety stop in C4: incomplete XI for an RL opponent seat

**Status: the experiment is stopped at step 9 (C4).** Nothing was patched and no frozen file changed; `hashes.mjs --check` passes after the stop.

## Where the pipeline stands

| Step | Status |
|---|---|
| 1–4 Infrastructure, tests, smoke test | done, PASS |
| 5 Frozen hashes | PASS |
| 6–7 Stage A control + regression | 7,500/7,500 episodes identical to the stored Stage A episodes; means and standard deviations exact. **PASS** |
| 8 C1 head-to-head | **complete**: 90,000 episodes. 0 safety events: learner legal XI 90,000/90,000, 0 opponent fallbacks in 11.3M opponent RL decisions, 0 invariant violations, 0 incomplete XIs for any RL team. |
| 9 C4 saturated | **STOPPED** after 6,280 clean episodes |
| 10–13 S4, reproducibility, audits, report | not run |

## The event

| | |
|---|---|
| Stop | `INCOMPLETE XI (opponent d3qn:s1, seat 4) seed 100490 cond C4` |
| Room | Validation entry 490 (seed 100,490), purse ₹33,800L (high stratum). Learner `ppo:s2`; all four other RL seats `d3qn:s1`. |
| Team | Seat 4, `d3qn:s1` in the overseasSpecialist RL seat. It ended with 22 players, ₹50L purse and **no wicketkeeper**, so its Best XI has 1 empty slot (XI 78.55). |
| Everyone else | Every other team, including the learner (XI 90.18) and the other three D3QN seats, has a complete XI. 0 invariant violations. |

## Reproduction (deterministic, not the harness)

- The harness diagnostic replay reproduces it exactly.
- It also reproduces **twice through the unmodified frozen path**: `evaluate.runEpisode`, `policyController`, and `RlEpisode`'s own production opponent runtimes with the **real clock**. No runtime was disabled.
- The injected clock and the harness wrapper therefore play no part.

Full trace: `safety_stop_C4_seed100490_trace.txt`.

## What happened (from the per-decision shield trace of seat 4)

1. **Early spending.** The D3QN bid on the first three keeper lots and lost all three: it bid up to ₹4,635L and lost at ₹5,250L, ₹1,300L and ₹3,900L. It spent the rest of its ₹33,800L on others. By 75% of the main round it had **22 players and ₹50L left**, and still needed exactly one player: a keeper.
2. **Main round, from 75% on.** Seat 4 had a legal bid on 13 keeper lots, each with a base price of ₹20–50L. The act-v3 shield rated the requirement **SAFE**, because many keepers were still to come, so bids were legal but not forced. **The D3QN policy chose PASS on every one.** Nine of these keepers went unsold, so a base-price bid would have won them.
3. **Re-auction.** The shield moved to WARNING, then CRITICAL, and **forced** bids (including one final-path bid) on keepers 265 and 245. Seat 4 bid its whole purse (₹50L) and lost both to the **PPO learner**. The learner already had 3 keepers from the main round (4 by the time of lot 265) and ended with 6; these were depth purchases.
4. **No keeper left in reach.** The shield then reports IMPOSSIBLE (`alreadyInfeasible`).

## Assessment

**Not an implementation defect in the harness.** It reproduces through the frozen environment unchanged, every decision came from the assigned model, and all invariants hold.

The frozen system behaved exactly as specified, and the outcome follows from two frozen properties interacting with an RL rival:

- **The D3QN policy.** In Stage A the purse-fragility and keeper rules of the shield bought its keeper (4–6 forced keeper purchases per 500 episodes). Here it passed voluntarily on cheap keepers while the shield said SAFE.
- **A limit of completion shield v2 (act-v3).** Its forcing rule counts *rivals that need* the requirement. A rival that does **not** need keepers, here the PPO learner buying depth, can take the last reachable candidates. And a ₹50L purse that exactly equals the base price cannot win a contested lot.

In Stage A the rule-fallback rivals never produced this pattern, and neither did any C1 room (90,000 episodes). It appeared after 6,281 of 90,000 C4 rooms.

## Why I stopped instead of continuing

Two rules conflict here:

- The spec's stop condition is an incomplete XI "caused by an implementation defect".
- It also says safety counts "MUST remain zero", and your instructions say to STOP on any safety violation rather than patch around it.

Treating this as a recorded finding and carrying on would change a pre-agreed experimental rule, so I haven't done that.

## Options (your decision)

- **A (recommended).** Treat an incomplete XI of any RL-controlled team (learner or opponent) in Stage B as a **recorded behavioural finding**, not a stop, provided it passes a defect screen: it reproduces through the unmodified `runEpisode` path, there are 0 invariant violations, 0 fallbacks, and every decision is legal under act-v3.
  - Every such episode gets a per-seat shield trace in the report.
  - Every other stop condition stays as it is.
  - C4 is re-run **from scratch**; its first 6,280 episodes are deterministic and will reproduce. S4 and the rest of the pipeline follow unchanged.
- **B.** Keep the stop rule as it is. C4 (and possibly S4) cannot complete, and the report covers A, C1 and the partial C4.
- **C.** Anything involving the shield, mask or policies is outside this phase; the freeze forbids it.
