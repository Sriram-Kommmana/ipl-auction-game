# Phase 2E.0 — Stage B infrastructure gap report

This covers inspection only. Nothing has been implemented, run or changed: no files were modified, no model was loaded for play, and no smoke test was run.

## 1. What the frozen specification says about Stage B

| Source | Stage B rule |
|---|---|
| Phase 2A §12 | "Stage B, league: the 4 RL seats are drawn from a frozen pool: past snapshots of *all five algorithms* plus rule fallbacks. Rule bots keep at least 40% of seats." |
| Phase 2A §12 | Trembling: 1% random legal action, "only on RL snapshot opponents". |
| Phase 2A §13 | League in rounds (Round 1 = train against rule bots + Round-0 finals). This is **training**, and it is out of scope here. |
| Phase 2A §17 | Evaluation suites: **S4** is "the product lineup (4 rule bots, the 5 RL agents and a human proxy)". **S6** is "5×5 cross-play between the algorithms", with no seat composition given. |
| Phase 2B.3 | Samplers must keep rule bots ≥ 40% "during league configuration". |
| `samplers.js` (frozen) | Each episode has 1 human proxy, the 4 rule bots, the learner's RL seat and 4 other RL seats. Each of those 4 seats independently becomes `rlSnapshot` with probability `snapshotShare`, drawn uniformly from a pool of size `poolSize`; otherwise it is `rlFallback`. Rule bots are therefore always at least 4/9 = 44% of opponents. |

**`snapshotShare` is not frozen anywhere.** It is a runtime parameter: the bridge default is 0, the legacy v1 bridge used 0.4, and every Stage A config sets 0.0. Beyond "at most 4 snapshots, at least 4 rule bots", no document fixes how many RL seats hold learned policies, which seat an opponent takes, or which algorithm sits where.

## 2. Already implemented (frozen, reusable without modification)

| Capability | File |
|---|---|
| RL opponent seats (`rlSnapshot`) running a frozen rl-policy-v2 through the production runtime, with rule-persona fallback | `packages/shared/src/rl/env.js` (`#opponent`, `#opponentCap`), `runtime.js` |
| Mixed rooms: RL seats, rule bots and human proxy in one auction | `env.js` + `samplers.js` seat types `rule / rlFallback / rlSnapshot / human / learner` |
| Loader for all five heads (ppo, a2c, d3qn, qrdqn, es) with spec-hash checks | `packages/shared/src/rl/policy.js` |
| Selection contract: PPO and A2C sample at temperature 0.3; D3QN, QR-DQN and ES use argmax (legal actions only) | `policy.js` `ALGORITHMS` |
| Deterministic learner harness: the learner's random stream comes from the seed, and `tremble` defaults to 0 in evaluation | `evaluate.js` `runEpisode`, `policyController` |
| Paired reporting: bootstrap CI, W/T/L, purse strata, seed-pairing check | `evaluate.js` `buildReport`, `bootstrapCI` |
| All-team safety audit (purse, squad ≤ 25, overseas ≤ 8, duplicates, sale validity, XI legality, deadlock) | `invariants.js` `auditAuction` (checks every team, not only the learner) |
| Per-decision shield trace (forced, forcedBy, finalPath, state) | `RlEpisode({ shieldTrace: true })` |
| Validation manifest: 500 entries with fixed purse, seating, learner seat and human proxy | `packages/shared/data/rl-manifests/validation.json` |
| Parity, fallback, latency, requirement-audit and replay-probe scripts from Stage A | `ml/runs/_2d2…_2d4/*`, `ml/ipl_rl/tests/replay_probe.mjs` |
| 15 frozen final exports, all with obs `629b25783f833af7` and act `5f72f510c48b1f46` | see §5 |

## 3. Partially implemented

| Gap | Detail |
|---|---|
| **Stage B episode composition** | `sampleEpisode(seed, {league})` draws snapshot seats from the same random stream *before* the seating shuffle. The same seed therefore gets **a different seating order** from its Stage A manifest entry, which would break pairing with Stage A. It also assigns snapshots at random rather than as a controlled matchup. A controlled cross-play needs a new, deterministic transformation of the validation manifest entry: replace named `rlFallback` seats with `rlSnapshot`, and change nothing else. This is new evaluation code, not a sampler change. |
| **Opponent-seat metrics** | `RlEpisode.summary()` reports only the learner. The opponent's XI, rank, purse, squad, overseas and stars can be read after the episode from `ep.sim.teams[i]`, and runtime failures from `ep.seats[i].runtime.state`. Opponent decision-level metrics (bid rate, shield use) are not recorded. They come from the reverse ordered pairing, where that algorithm is the learner. |
| **Opponent-seat safety** | If an opponent's output is masked, NaN or invalid, the runtime silently falls back to the rule persona and only logs it in `state.log`. The harness must read `failures` and `log` for every opponent seat after every episode and treat any entry as a safety stop. |
| **Requirement and shield audit** | `audit_requirements.mjs` exists for one algorithm and Stage A. It needs a Stage B version that uses the composed entries. |
| **Performance** | `latency_bench.mjs` (2D.4) benchmarks one export in 9 seats. It needs a mixed five-algorithm room. |

## 4. Missing

1. A cross-play runner: ordered pairs × compositions × seed pairings × 500 validation entries, on worker threads, writing `episode-results.json`.
2. A frozen-file and export hash guard, run at start and end: the 15 export digests plus the source hashes of `obsSpec.js`, `actionSpec.js`, `mask.js`, `reward.js`, `planning.js`, `sim.js`, `personas.js`, `samplers.js`, `env.js`, `runtime.js`, `policy.js`, the players CSV and the manifests.
3. A zero-training proof. This is structural, since the Node harness has no optimiser, gradient, replay or perturbation code path. It will also be asserted: export digests identical before and after, and no Python training module imported.
4. An analysis script: transfer Δ vs Stage A, head-to-head Δ, W/T/L, strata, seed stability, behavioural shifts, and plots.
5. A reproducibility rerun comparing episode digests and action-stream digests.

**Files involved:** all new and none frozen. The proposal is `ml/ipl_rl/crossplay/{compose.mjs, run.mjs, audit.mjs, latency.mjs, analyse.py}`, a test file `ml/ipl_rl/tests/test_crossplay.mjs`, and outputs in `ml/reports/phase2e0/`. **No frozen file needs modifying**, subject to decision D4.

## 5. Frozen exports found

| Algorithm | Export | sha256 (first 12 hex) |
|---|---|---|
| PPO | `runs/ppo-2c3-s{1,2,3}/checkpoints/update_0325/policy.json` | 1484f74db7ab · ffa4a6bbdb18 · 9cdd49ed48c9 |
| A2C | `runs/a2c-2d1-s{1,2,3}/…/update_0325/policy.json` | 9482565a9322 · 8d54a091f8e4 · 338ed33cb8a4 |
| D3QN | `runs/d3qn-2d2-s{1,2,3}/…/update_0325/policy.json` | 92d77f58a27a · e4b54c2a8b27 · 7d938abb7cb5 |
| QR-DQN | `runs/qr-dqn-2d3-s{1,2,3}/…/update_0325/policy.json` | 957d8847988e · d4089e713f6d · a5bdbe2fd1dd |
| ES | `runs/openai-es-2d4-s{1,2,3}/checkpoints/gen_2000/policy.json` (identical to `reports/phase2d4/`) | cccc7996e636 · 37bb9702b090 · 76bcd8ff0638 |

**Note:** the PPO, A2C, D3QN and QR-DQN exports exist only in git-ignored `ml/runs/`; only the ES exports are committed.

## 6. Ambiguities that block the experiment (decisions needed)

- **D1. RL-vs-RL seat composition.** For ordered pair (A learner, B opponent), the specification says B occupies "another RL seat", but not how many of the 4 other RL seats B fills, which seat, or what the rest hold.
- **D2. Mixed RL/rule room.** S4 names the lineup (human proxy + 4 rule bots + 5 RL agents) but not which algorithm sits in which RL seat. There is no production algorithm-to-seat mapping yet, because `apps/server` still loads the legacy single `rl-policy.json`, and choosing one would be production selection. There is also no frozen `snapshotShare` for a random-mix alternative.
- **D3. Which exports.** Each algorithm has 3 seeds. "The five frozen RL exports" does not say which seed, or how seeds pair across algorithms.
- **D4. The 20 ms wall-clock guard makes opponent seats non-deterministic.** The learner is played by `policyController`, which has no guard. Opponent RL seats run through `createRlSeat` with the real clock. Phase 2D.4 measured rare > 20 ms pauses (up to 10 per 17,820 decisions). Each one silently turns that opponent into its rule persona for the rest of the room, so replays differ from run to run and "wrong model playing" goes undetected.
- **D5. Opponent trembling.** Phase 2A §12 gives 1% trembling to snapshot opponents during league **training**. The frozen evaluator (`runEpisode`) defaults to `tremble = 0`.
- **D6. Reference for "paired ΔXI".** Two readings are possible: transfer (Stage B vs the same export in Stage A on the same auction) or head-to-head (learner XI − opponent XI in the same auction).
- **D7. Seed block.** Stage A used the 500-entry validation manifest. The 1,000-entry test split is reserved "only for the final report".

### Recommended answers

| # | Recommendation |
|---|---|
| D1 | Run two compositions, reported separately. **C1 head-to-head:** B takes exactly one other RL seat, the same seed-derived seat for every pairing on a given entry; the other 3 RL seats stay on rule fallback, exactly as in Stage A. **C4 saturated:** B fills all 4 other RL seats, the most the ≥ 40% rule allows (4 of 9 opponents are rule bots, 44%). |
| D2 | **S4:** learner A in its manifest seat, and the other four algorithms one each in the other four RL seats, assigned by a seed-derived permutation. Every algorithm therefore sits in every RL seat about equally often and no production mapping is implied. |
| D3 | Final exports only (update 325; ES gen 2,000). **All 3×3 seed pairings per ordered pair** (9 per cell) for seed stability, with per-pairing results kept. S4 uses matched seed index k for every seat (3 rooms per learner). |
| D4 | Opponent seats in the harness get a deterministic clock (`now: () => 0`, the injection point `createRlSeat` already has), installed from the new evaluation file. The runtime and its 20 ms guard are unchanged; the guard is measured with the real clock in §15, and would-be trips are reported separately. All other fallback paths (NaN, masked, invalid, exception) stay active and are **safety stops**. |
| D5 | `tremble = 0`, matching the evaluator default and the contract's deterministic evaluation. |
| D6 | Report both. **Transfer Δ** is primary; the Stage A control is re-run in the same harness and must reproduce the reported Stage A numbers exactly, which also serves as the Stage A regression check. **Head-to-head Δ** is secondary. |
| D7 | Validation manifest (500), with the test split untouched. |

**Estimated cost with these answers:**

| Run | Episodes |
|---|---|
| C1 | 20 ordered pairs × 9 seed pairings × 500 = 90,000 |
| C4 | 90,000 |
| S4 | 5 × 3 × 500 = 7,500 |
| Stage A control | 7,500 |
| Reproducibility rerun subset | about 10,000 |
| **Total** | **about 205,000** |

At the Stage A evaluator's speed of about 20–25 episodes per second on 12–14 workers, that is roughly 2.5–3 hours.
