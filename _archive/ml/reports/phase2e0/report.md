# Phase 2E.0: Stage B frozen cross-play evaluation

This phase is evaluation only. No model was trained, retuned, re-exported or changed, and no frozen specification, rule or runtime was modified. **It does not produce an overall winner or ranking**: every table keeps the fixed algorithm order PPO, A2C, D3QN, QR-DQN, OpenAI-ES, and every comparison stays within its ordered pairing.

---

## A. Executive summary

1. **Stage A performance does not fully transfer to rooms with learned opponents.**
   - Adding even **one** learned opponent (C1) lowers every export's validation XI, relative to the same export on the same auctions in Stage A, in all 20 ordered pairings.
   - All 45 learner/composition cells (20 in C1, 20 in C4, 5 in S4) have 95% CIs that exclude zero.
   - In C1, 98.4% of paired episodes decline and 1.6% gain or tie.
   - Transfer Δ, mean over the four opponents:

   | Composition | PPO | A2C | D3QN | QR-DQN | OpenAI-ES |
   |---|---|---|---|---|---|
   | C1 (one RL opponent) | −1.95 | −2.79 | −2.60 | −2.46 | −3.37 |
   | C4 (opponent in all four RL seats) | −2.32 | −3.72 | −3.29 | −3.23 | −5.06 |
   | S4 (the other four algorithms) | −2.43 | −5.14 | −4.82 | −4.34 | −6.83 |

2. **Matchup dependence is strong, and it depends mostly on the *opponent*.**
   - **A PPO opponent is the mildest for every other learner:** −1.52 to −2.44 in C1 and −1.86 to −2.58 in C4. PPO-as-opponent bids selectively: bid rate 0.36–0.38, cap/fair 0.91–0.97, and it leaves 10–12% of its purse in C4.
   - **D3QN and QR-DQN opponents are the harshest:** averaged over learners, −3.21 and −3.31 in C1, and −4.82 and −4.60 in C4. They bid on most lots at about fair value and win them at 1.15–1.29× fair.
   - The same learner can lose 1.9 XI against one opponent and 6.7 against another; the ES learner's C4 range is −2.58 to −6.68.

3. **The largest degradations are ES and A2C against the value-based learners in saturated rooms, and both in the mixed S4 room.**
   - This is not a legality collapse: legal XI stays at 100% except in 10 of 90,000 C4 learner episodes (section K).
   - It is a large quality loss. ES in S4 averages XI 84.24 (Stage A 91.07), a strong XI in 33.7% of auctions, average rank 9.0 of 10, a squad of 24.9 at the 8-overseas cap, and 17.8% of its purse unspent because the squad is full.
   - ES's near-fixed "bid about 1.25× fair value" rule (Phase 2D.4) keeps winning marginal players and loses the stars. It averages 1.9 stars against 10.1 in Stage A.

4. **PPO is the most transfer-robust export on every axis measured, but it still degrades.** Its losses are about −2 XI, the smallest and narrowest across opponents (C4 range −2.16 to −2.47). It keeps a 100% strong XI and an average rank of about 1.2. Its shield use rises from 0 in Stage A to 34 forced decisions in 18,000 C1 episodes and 87 in C4, and its final-path forced bids stay at 0.

5. **Behavioural shifts from Stage A to Stage B are consistent across algorithms:**
   - higher bid rates and prices (price/fair up 5–15%);
   - fewer stars and more marginal purchases (2–7× more);
   - later completion of keeper, bowling and Indian requirements;
   - more reliance on the completion shield;
   - final-path forced bids, never seen for any export in Stage A: 31 in Stage B.

6. **Safety.**
   - No hard safety violation occurred in 195,000 cross-play episodes: 0 illegal or masked actions, 0 purse, squad, overseas or duplicate violations, 0 invariant violations, 0 NaN/∞, 0 crashes or deadlocks, 0 wrong-model events.
   - There were 0 opponent fallbacks in 80.6M opponent RL decisions.
   - One safety stop fired in C4 and was investigated. It was legal policy behaviour (an incomplete XI: no wicketkeeper), and you approved Option A.
   - C4 found **20 incomplete-XI episodes**, all C4, all a missing keeper, 19 of them in the high-purse stratum. They cover 10 learners and 10 opponent seats. All 20 passed the defect screen and are reported as findings.

7. **Reproducibility.** 21,330 re-run episodes, run in separate processes with 5, 7, 9, 11 and 14 workers, are identical to the originals. That includes the 6,280 C4 episodes played before the stop by the earlier harness version. The Stage A control reproduces all 7,500 stored Stage A episodes field for field.

8. **Parity and performance.**
   - Production parity passes for all 15 exports on 35,069 real states, including Stage B learner and opponent states: maximum score difference 2.5e-14, 100% argmax agreement, and 100% agreement with recorded production choices.
   - All 138 fallback-suite checks pass.
   - With the real clock, 0 of 358,000 decisions exceeded 20 ms at 1–4 concurrent rooms.
   - At 14 concurrent rooms on a saturated 16-thread CPU, the unmodified 20 ms guard tripped 15 times in 336 rooms (20.1–34.6 ms, OS or GC scheduling), switching 1.1% of RL seat-rooms to their rule persona. This is reported separately and the guard is unchanged.

---

## B. Frozen configuration

Nothing listed here was modified. The frozen-hash record `frozen-hashes.json` was taken before the full run. It holds sha256 digests of 25 frozen source, data and manifest files and of all 15 exports with their training checkpoints. It was verified before and after every run segment and after the audits. All 25 frozen sources are also clean against git HEAD.

| Item | Value |
|---|---|
| Environment | `IplAuctionEnv-v2` (JavaScript `RlEpisode`, unchanged `AuctionSim`), Full Pool |
| Observation | obs-v2, 80 features, hash `629b25783f833af7` |
| Action / mask | act-v3, 20 actions, completion shield v2, hash `5f72f510c48b1f46` |
| Reward, γ | ΔBestXI/11/10, −2 per empty XI slot, γ = 1. Not used by evaluation beyond the reported `return`. |
| Planner, rule bots, pool, order, increments, purse, squad, overseas, scoring, re-auction | Unchanged (covered by the source hashes) |
| Seed block | Frozen 500-entry validation manifest (seeds 100,000–100,499). **The test split was not touched.** |
| Selection | Production contract. PPO and A2C sample at temperature 0.3: the learner from its own seed-derived stream, opponents from the simulator stream exactly as `RlEpisode` does. D3QN, QR-DQN and ES use masked argmax. |
| Trembling | 0 (D5) |
| 20 ms guard | Unmodified. The harness injects `now: () => 0` into opponent seats only (D4); section N measures the real clock. |
| Production completion guard (opt-in) | Off, as in all evaluation |

**Exports.** These are the final Stage A checkpoints; nothing was regenerated.

| Algorithm | Checkpoint | Seeds | sha256 (first 12 hex) |
|---|---|---|---|
| PPO (2C.3) | `update_0325` | s1, s2, s3 | 1484f74db7ab · ffa4a6bbdb18 · 9cdd49ed48c9 |
| A2C (2D.1) | `update_0325` | s1, s2, s3 | 9482565a9322 · 8d54a091f8e4 · 338ed33cb8a4 |
| D3QN (2D.2) | `update_0325` | s1, s2, s3 | 92d77f58a27a · e4b54c2a8b27 · 7d938abb7cb5 |
| QR-DQN (2D.3) | `update_0325` | s1, s2, s3 | 957d8847988e · d4089e713f6d · a5bdbe2fd1dd |
| OpenAI-ES (2D.4) | `gen_2000` | s1, s2, s3 | cccc7996e636 · 37bb9702b090 · 76bcd8ff0638 |

**Zero training.**
- The harness has no optimiser, gradient, replay, perturbation or checkpoint-writing code path. It is Node-only and imports no Python training module.
- Export and checkpoint digests are identical before and after every run.

---

## C. Infrastructure gap report (summary)

The full report is in `gap_report.md`; decisions D1–D7 were approved before implementation.

**Already present and reused unchanged:**
- `rlSnapshot` opponent seats through the production runtime;
- mixed rooms;
- the loader for all five heads;
- the selection contract;
- the deterministic learner evaluator;
- the all-team safety audit;
- the shield trace;
- the validation manifest.

**Gaps closed with new files only (no frozen file modified):**
- **Stage B room composition.** The league sampler reorders seating, so each room is built instead by editing a copy of the manifest entry.
- **Opponent-fallback detection.** A fallback is now a hard stop rather than a silent rule-bot decision.
- **Opponent outcome metrics.**
- **Frozen-hash guard.**
- **Runner, analysis and reproducibility tooling.**

New code is in `ml/ipl_rl/crossplay/`, with tests in `ml/ipl_rl/tests/test_crossplay.mjs`:

| File | Role |
|---|---|
| `common.mjs` | Export registry and digest check, frozen hashes, room composition |
| `harness.mjs` | One episode |
| `run.mjs` | Parallel deterministic runner |
| `hashes.mjs` | Frozen-hash guard |
| `analyse.py` | Analysis |
| `check_stage_a.py` | Stage A regression check |
| `compare_runs.py` | Reproducibility comparison |
| `dump_states.mjs`, `parity.py` | Parity audit |
| `latency.mjs` | Performance measurement |
| `ppo_fallback.mjs` | PPO fallback suite: A2C's suite with only the algorithm id changed |
| `findings.py`, `diagnose_incomplete.mjs` | Option A findings and traces |
| `plots.py` | Plots |
| `package_report.py` | Report packaging |
| `smoke.sh`, `run_all.sh`, `run_rest.sh`, `audits.sh` | Pipeline scripts |
| `run_node.py` | Windows EcoQoS opt-out; affects speed only |

---

## D. Smoke test (before the full run)

Details are in `smoke.md`. **Result: PASS.**

- **Coverage:** 1,000 episodes across A (15 exports × 20 entries), C1 and C4 (all 20 ordered algorithm pairs, s1 × s1, × 15 entries each) and S4 (5 learners × 20 entries). Each ran twice, on 14 and on 5 workers.
- **Checks:**
  - policies loaded correctly;
  - seat assignments correct (C1 = 1, C4 = 4, S4 = 4 distinct other algorithms);
  - every opponent decision came from its assigned model (0 fallbacks);
  - 0 illegal or masked actions, crashes or invariant violations;
  - production JavaScript inference used throughout.
- **Reproducibility:** both passes were identical episode by episode, and the 300 Stage A control episodes were identical to the stored Stage A episodes.

**Tests.**

| Suite | Result |
|---|---|
| New cross-play tests | 17/17 (14 before the Option A change, plus 3 for it) |
| JavaScript suite | 209 pass + 1 known legacy skip (the 210 tests committed since act-v3) |
| Python suite | 102/102 |

---

## E. Cross-play methodology

**Rooms.** Every room is a copy of a validation-manifest entry.
- **Unchanged:** purse, stratum, seating, the learner's seat and RL-seat name, the four frozen rule bots, and the human proxy (passive, noisy or star-chasing).
- **Changed:** only the four `rlFallback` seats (the RL seats that play their rule persona in Stage A), which become `rlSnapshot` seats. This is the same transformation the frozen league sampler makes.
- Rule bots are 4 of 9 opponents (44%) in every Stage B room, so the ≥ 40% rule holds.

| Condition | The 4 other RL seats | Episodes |
|---|---|---|
| **A** (Stage A control) | Rule fallback (manifest unchanged) | 15 × 500 = 7,500 |
| **C1** head-to-head | 1 seat = opponent export B. The seat is seed-derived with the existing mechanism (`createRng(deriveSeed(seed, 'phase2e0/c1-opponent-seat'))`) and is the same for every pairing on that auction. The other 3 seats stay on rule fallback. | 20 ordered pairs × 3 × 3 seed pairings × 500 = 90,000 |
| **C4** saturated | All 4 seats = opponent export B | 90,000 |
| **S4** mixed | The other four algorithms (same seed index k), one each, placed by a seed-derived permutation (`phase2e0/s4-algorithm-permutation`). Every algorithm occupies every RL seat name about equally often (test-checked). **This is not a production seat mapping.** | 5 × 3 × 500 = 7,500 |

**Players.**
- **Learner seat:** the export exactly as the Stage A evaluator plays it (`evaluate.policyController`).
- **Opponent RL seats:** the export through the production runtime (`createRlSeat`, unmodified) with the deterministic clock. Any non-RL decision stops the run. That covers non-finite output, masked or invalid action, exception, a disabled model, and a runtime whose weight digest does not match the assigned export (checked in every room).

**Pairing.**
- For each auction entry, every condition plays the same purse, seating and lot order.
- Once decisions differ, the simulator's random draws differ, which is inherent to cross-play.
- Transfer Δ therefore pairs each Stage B episode with the **same export's** Stage A episode on the **same entry**.

**Statistics.**
- **Transfer Δ (primary).** Learner Stage B XI minus the same export's Stage A XI on the same entry.
- **Head-to-head Δ (secondary).** Learner XI minus opponent XI in the same auction. In C4 the opponent value is the mean of the four opponent seats (the best seat is also in `matchup-results.json`); in S4 it is per algorithm.
- **95% CIs.** Percentile bootstrap (2,000 resamples, fixed seed) over the 500 auction entries. Each entry's value is its mean over the cell's seed pairings, so resampling keeps both the auction pairing and the shared Stage A control.
- **W/T/L.** Counted per episode.
- **Seed stability.** Measured over the 9 seed pairings of each cell.
- **No overall ranking** is computed.

---

## F. RL-vs-RL matchup matrix

Rows are the learner and columns the opponent. Each cell covers 9 seed pairings × 500 auctions (4,500 episodes). Values are XI points, with 95% CIs in brackets.

### C1 head-to-head (opponent in one of the four RL seats)

**Transfer Δ (primary)**

| Learner \ opponent | PPO | A2C | D3QN | QR-DQN | OpenAI-ES |
|---|---|---|---|---|---|
| **PPO** | — | −1.648 [−1.68, −1.61] | −2.143 [−2.18, −2.11] | −2.254 [−2.29, −2.22] | −1.739 [−1.77, −1.71] |
| **A2C** | −1.857 [−1.93, −1.79] | — | −3.472 [−3.54, −3.40] | −3.423 [−3.49, −3.35] | −2.401 [−2.47, −2.34] |
| **D3QN** | −1.641 [−1.69, −1.59] | −2.951 [−3.01, −2.90] | — | −3.562 [−3.62, −3.51] | −2.232 [−2.28, −2.18] |
| **QR-DQN** | −1.521 [−1.57, −1.47] | −2.769 [−2.82, −2.72] | −3.312 [−3.36, −3.26] | — | −2.224 [−2.27, −2.18] |
| **OpenAI-ES** | −2.438 [−2.53, −2.35] | −3.101 [−3.18, −3.02] | −3.931 [−4.03, −3.84] | −4.001 [−4.09, −3.92] | — |

**Head-to-head Δ (secondary): learner XI − opponent XI**

| Learner \ opponent | PPO | A2C | D3QN | QR-DQN | OpenAI-ES |
|---|---|---|---|---|---|
| **PPO** | — | +2.347 [+2.28, +2.41] | +0.576 [+0.53, +0.62] | −0.008 [−0.05, +0.03] | +2.183 [+2.11, +2.26] |
| **A2C** | −2.376 [−2.44, −2.31] | — | −1.580 [−1.64, −1.52] | −2.079 [−2.13, −2.03] | +0.076 [+0.01, +0.14] |
| **D3QN** | −0.605 [−0.65, −0.56] | +1.585 [+1.52, +1.64] | — | −0.597 [−0.64, −0.55] | +2.110 [+2.04, +2.18] |
| **QR-DQN** | −0.015 [−0.06, +0.03] | +2.071 [+2.02, +2.13] | +0.614 [+0.57, +0.66] | — | +2.561 [+2.50, +2.62] |
| **OpenAI-ES** | −2.234 [−2.31, −2.16] | −0.056 [−0.12, +0.01] | −2.102 [−2.17, −2.03] | −2.556 [−2.62, −2.50] | — |

Head-to-head W/T/L per episode:

| Learner | vs PPO | vs A2C | vs D3QN | vs QR-DQN | vs OpenAI-ES |
|---|---|---|---|---|---|
| PPO | — | 4331/24/145 | 2980/129/1391 | 2028/157/2315 | 4145/57/298 |
| A2C | 151/27/4322 | — | 864/56/3580 | 534/53/3913 | 2499/76/1925 |
| D3QN | 1350/136/3014 | 3577/58/865 | — | 1720/106/2674 | 4278/19/203 |
| QR-DQN | 2299/159/2042 | 3889/58/553 | 2704/122/1674 | — | 4379/22/99 |
| OpenAI-ES | 284/38/4178 | 1977/60/2463 | 212/33/4255 | 87/23/4390 | — |

The C1 head-to-head matrix is antisymmetric to within about 0.05 XI; this is a check, since A>B and B>A share the same auctions. The only pairing whose CI includes 0 is PPO vs QR-DQN, in both directions: +0.008 / −0.015, W/T/L 2028/157/2315.

### C4 saturated (opponent in all four RL seats)

**Transfer Δ (primary)**

| Learner \ opponent | PPO | A2C | D3QN | QR-DQN | OpenAI-ES |
|---|---|---|---|---|---|
| **PPO** | — | −2.266 [−2.31, −2.23] | −2.470 [−2.51, −2.43] | −2.384 [−2.43, −2.34] | −2.161 [−2.20, −2.12] |
| **A2C** | −1.985 [−2.06, −1.91] | — | −5.162 [−5.24, −5.08] | −4.794 [−4.87, −4.72] | −2.929 [−3.00, −2.86] |
| **D3QN** | −1.908 [−1.96, −1.85] | −3.779 [−3.83, −3.73] | — | −5.067 [−5.13, −5.01] | −2.401 [−2.46, −2.34] |
| **QR-DQN** | −1.859 [−1.91, −1.81] | −3.632 [−3.68, −3.58] | −4.959 [−5.01, −4.91] | — | −2.470 [−2.52, −2.42] |
| **OpenAI-ES** | −2.579 [−2.67, −2.49] | −4.815 [−4.90, −4.73] | −6.685 [−6.80, −6.58] | −6.140 [−6.25, −6.03] | — |

**Head-to-head Δ: learner XI − mean XI of the four opponent seats**

| Learner \ opponent | PPO | A2C | D3QN | QR-DQN | OpenAI-ES |
|---|---|---|---|---|---|
| **PPO** | — | +4.652 [+4.62, +4.69] | +4.845 [+4.81, +4.88] | +4.195 [+4.16, +4.23] | +4.366 [+4.33, +4.40] |
| **A2C** | +0.201 [+0.14, +0.26] | — | +0.468 [+0.42, +0.52] | +0.169 [+0.11, +0.22] | +1.351 [+1.31, +1.40] |
| **D3QN** | +1.736 [+1.68, +1.79] | +2.422 [+2.38, +2.47] | — | +1.143 [+1.10, +1.19] | +3.242 [+3.20, +3.29] |
| **QR-DQN** | +2.207 [+2.16, +2.25] | +2.949 [+2.90, +3.00] | +2.059 [+2.02, +2.10] | — | +3.531 [+3.49, +3.57] |
| **OpenAI-ES** | +0.259 [+0.19, +0.33] | +0.374 [+0.32, +0.43] | −0.752 [−0.83, −0.67] | −0.911 [−1.01, −0.82] | — |

In C4 the four identical opponents compete with each other for the same players, so their average XI falls: 84.8–88.2 against 85.3–90.9 in C1. That is why most C4 head-to-head values are positive even though every learner's own XI falls. **C4 head-to-head must not be read as "the learner beats the opponent algorithm"**; transfer Δ is the primary measure.

**Per-cell detail** is in `matchup-results.json`:
- learner XI mean, sd and CI; opponent XI;
- W/T/L; all 9 seed pairings;
- purse strata;
- learner and opponent behaviour profiles.

---

## G. Mixed RL/rule room (S4)

In S4 the learner shares the room with the other four algorithms (one each, seed-matched), four frozen rule bots and the human proxy. Each learner covers 3 seeds × 500 auctions.

| Learner | XI mean ± sd [95% CI] | Transfer Δ [95% CI] | Transfer W/T/L | Average rank | Strong XI | Legal XI |
|---|---|---|---|---|---|---|
| PPO | 90.167 ± 0.758 [90.13, 90.21] | −2.427 [−2.48, −2.38] | 4/0/1496 | 1.19 | 100% | 1500/1500 |
| A2C | 85.283 ± 1.014 [85.24, 85.33] | −5.144 [−5.22, −5.07] | 1/0/1499 | 8.45 | 61.1% | 1500/1500 |
| D3QN | 86.670 ± 1.259 [86.61, 86.73] | −4.820 [−4.89, −4.75] | 0/0/1500 | 6.45 | 89.1% | 1500/1500 |
| QR-DQN | 87.512 ± 0.945 [87.46, 87.56] | −4.341 [−4.41, −4.28] | 0/0/1500 | 5.15 | 98.9% | 1500/1500 |
| OpenAI-ES | 84.239 ± 1.418 [84.16, 84.32] | −6.829 [−6.94, −6.72] | 0/0/1500 | 9.02 | 33.7% | 1500/1500 |

**Head-to-head in S4: learner XI − that algorithm's seat XI in the same auction**

| Learner \ seat | PPO | A2C | D3QN | QR-DQN | OpenAI-ES |
|---|---|---|---|---|---|
| **PPO** | — | +4.87 [+4.82, +4.93] | +3.48 [+3.42, +3.55] | +2.64 [+2.59, +2.69] | +5.87 [+5.78, +5.96] |
| **A2C** | −4.85 [−4.91, −4.80] | — | −1.39 [−1.46, −1.33] | −2.24 [−2.31, −2.17] | +1.05 [+0.97, +1.13] |
| **D3QN** | −3.51 [−3.57, −3.45] | +1.38 [+1.31, +1.44] | — | −0.87 [−0.94, −0.81] | +2.41 [+2.32, +2.52] |
| **QR-DQN** | −2.67 [−2.72, −2.61] | +2.22 [+2.16, +2.29] | +0.85 [+0.78, +0.91] | — | +3.22 [+3.12, +3.33] |
| **OpenAI-ES** | −5.94 [−6.03, −5.85] | −1.02 [−1.11, −0.93] | −2.45 [−2.55, −2.35] | −3.28 [−3.39, −3.18] | — |

The S4 head-to-head matrix is antisymmetric to within about 0.1 XI. Per seed, S4 XI and transfer Δ are:

| Learner | s1 | s2 | s3 |
|---|---|---|---|
| PPO | 90.32 / −2.22 | 90.49 / −2.20 | 89.69 / −2.86 |
| A2C | 84.88 / −5.52 | 86.24 / −4.24 | 84.73 / −5.67 |
| D3QN | 87.51 / −3.94 | 86.75 / −4.95 | 85.75 / −5.57 |
| QR-DQN | 87.55 / −4.26 | 87.19 / −4.61 | 87.81 / −4.15 |
| OpenAI-ES | 84.73 / −6.34 | 83.39 / −7.72 | 84.60 / −6.42 |

---

## H. Stage A vs Stage B behavioural comparison

Means are per learner episode. A = the Stage A control (1,500 episodes per algorithm); C1 and C4 = the mean over all four opponents (18,000 episodes per algorithm); S4 = 1,500 episodes per algorithm.

| Metric | Condition | PPO | A2C | D3QN | QR-DQN | ES |
|---|---|---|---|---|---|---|
| **XI** | A | 92.59 | 90.43 | 91.49 | 91.85 | 91.07 |
| | C1 | 90.65 | 87.64 | 88.89 | 89.40 | 87.70 |
| | C4 | 90.27 | 86.71 | 88.20 | 88.62 | 86.01 |
| | S4 | 90.17 | 85.28 | 86.67 | 87.51 | 84.24 |
| **Strong XI** | A | 100% | 100% | 100% | 100% | 100% |
| | C1 | 100% | 98.5% | 99.4% | 99.97% | 99.0% |
| | C4 | 99.97% | 83.3% | 95.6% | 98.5% | 69.8% |
| | S4 | 100% | 61.1% | 89.1% | 98.9% | 33.7% |
| **Rank** | A | 1.00 | 1.19 | 1.00 | 1.00 | 1.09 |
| | C1 | 1.24 | 5.62 | 2.88 | 2.10 | 5.69 |
| | C4 | 1.17 | 5.80 | 3.65 | 3.11 | 6.60 |
| | S4 | 1.19 | 8.45 | 6.45 | 5.15 | 9.02 |
| **Bid rate** | A | 0.436 | 0.839 | 0.790 | 0.652 | 0.857 |
| | C1 | 0.429 | 0.853 | 0.814 | 0.726 | 0.882 |
| | C4 | 0.462 | 0.886 | 0.833 | 0.759 | 0.946 |
| | S4 | 0.447 | 0.917 | 0.801 | 0.790 | 0.995 |
| **Price/fair** | A | 0.790 | 1.112 | 1.095 | 0.979 | 1.047 |
| | C1 | 0.859 | 1.191 | 1.182 | 1.036 | 1.093 |
| | C4 | 0.849 | 1.192 | 1.178 | 1.078 | 1.107 |
| | S4 | 0.859 | 1.259 | 1.249 | 1.136 | 1.127 |
| **Cap/fair** | A | 0.747 | 1.052 | 1.069 | 0.968 | 1.090 |
| | C1 | 0.791 | 1.113 | 0.959 | 0.887 | 1.163 |
| | C4 | 0.814 | 1.170 | 0.965 | 0.916 | 1.257 |
| | S4 | 0.809 | 1.208 | 0.957 | 0.906 | 1.281 |
| **Squad size** | A | 16.6 | 15.7 | 16.5 | 16.2 | 15.8 |
| | C1 | 16.7 | 18.5 | 15.3 | 15.9 | 20.3 |
| | C4 | 18.3 | 21.2 | 16.6 | 16.7 | 23.1 |
| | S4 | 18.2 | 23.1 | 15.9 | 16.0 | 24.9 |
| **Overseas** | A | 6.25 | 5.17 | 5.06 | 5.54 | 5.76 |
| | S4 | 6.34 | 7.75 | 5.43 | 6.16 | 8.00 |
| **Purse left (share)** | A | 0.1% | 0.03% | 0.1% | 0.1% | 0.04% |
| | C4 | 1.3% | 3.0% | 0.5% | 0.3% | 11.9% |
| | S4 | 0.7% | 2.2% | 0.1% | 0.1% | 17.8% |
| **Stars bought** | A | 10.73 | 9.11 | 9.85 | 10.19 | 10.08 |
| | C1 | 8.59 | 6.58 | 7.82 | 7.85 | 6.65 |
| | C4 | 8.44 | 5.39 | 7.27 | 6.93 | 4.47 |
| | S4 | 8.80 | 3.35 | 6.21 | 5.98 | 1.91 |
| **Marginal buys** | A | 0.18 | 0.51 | 0.29 | 0.30 | 0.62 |
| | C1 | 0.77 | 1.19 | 0.71 | 0.69 | 1.72 |
| | C4 | 1.34 | 2.15 | 0.97 | 1.00 | 3.08 |
| | S4 | 1.29 | 2.69 | 0.90 | 0.94 | 4.03 |
| **Re-auction buys** | A | 0.05 | 0.00 | 0.16 | 0.24 | 0.04 |
| | S4 | 0.57 | 0.00 | 0.15 | 0.10 | 0.00 |
| **XI gain per ₹1000L** | A | 8.49 | 8.29 | 8.38 | 8.42 | 8.35 |
| | S4 | 8.30 | 7.97 | 7.94 | 8.01 | 9.49 |

**Reading:**
- RL opponents contest the players the Stage A rule fallbacks let through. Every learner pays more (price/fair up), gets fewer stars and buys more marginal players.
- PPO changes least; it is the only algorithm with bid rate and squad size close to Stage A.
- A2C and ES answer competition by bidding on almost everything (bid rate 0.92–1.00 in S4). They fill their squads (ES: 24.9 of 25, overseas at the cap of 8) and end with unspent purse because the squad is full.
- ES's high XI gain per ₹1000L in S4 (9.49) is spending efficiency on cheap players, not team quality: its XI is the lowest.

**Requirement behaviour**

Figures are the learner's progress through the auction (0 = start, 1 = end of the main round) when each requirement was completed, and shield-forced keeper purchases per 500 episodes.

| Metric | Condition | PPO | A2C | D3QN | QR-DQN | ES |
|---|---|---|---|---|---|---|
| **Keeper completed at** | A | 0.09 | 0.22 | 0.07 | 0.07 | 0.21 |
| | C1 | 0.24 | 0.53 | 0.17 | 0.11 | 0.60 |
| | C4 | 0.38 | 0.60 | 0.25 | 0.15 | 0.70 |
| | S4 | 0.57 | 0.74 | 0.40 | 0.20 | 0.77 |
| **Bowling completed at** | A | 0.25 | 0.07 | 0.08 | 0.12 | 0.07 |
| | S4 | 0.43 | 0.14 | 0.16 | 0.22 | 0.18 |
| **Indian players completed at** | A | 0.23 | 0.06 | 0.08 | 0.11 | 0.11 |
| | S4 | 0.56 | 0.11 | 0.29 | 0.32 | 0.32 |
| **Forced keeper purchases / 500** | A | 0.0 | 57.3 | 5.0 | 9.7 | 53.3 |
| | C1 | 0.2 | 156.3 | 22.5 | 22.4 | 117.0 |
| | C4 | 0.5 | 100.7 | 32.8 | 32.5 | 50.4 |
| | S4 | 0.0 | 114.0 | 80.7 | 43.3 | 13.7 |

- The Stage A forced-keeper figures reproduce the Phase 2D reports: A2C 81/38/53 and ES 61/39/60 per seed average to 57.3 and 53.3.
- **Indian-player completion.** Forced bids for Indian players, never seen in Stage A, appear in Stage B (up to 1.9% of D3QN S4 episodes).
- **Keepers in the re-auction.** Keeper requirements closed in the re-auction rise for D3QN (0.3% → 2.5%) and QR-DQN (0.9% → 2.6%).
- **Minimum purse while any requirement was unmet** falls for every algorithm, for example D3QN from ₹2,685L (A) to ₹626L (S4).

---

## I. Robustness analysis

**A. Stage A transfer.**
- No export keeps its Stage A XI once learned opponents are present.
- All 45 learner/composition cells decline, with CIs excluding 0. In all 40 C1/C4 cells, all 9 seed pairings decline.
- PPO keeps its Stage A behaviour best: bid rate, cap/fair and squad size are close to Stage A, strong XI stays at 100%, and rank stays about 1.2.
- The value-based learners keep their legality and most of their quality in C1 (strong XI ≥ 99.4%) but lose more in C4 and S4.
- A2C and ES lose the most.

**B. Matchup dependence.** It is substantial, and it is driven by the opponent's bidding style more than by the learner.

| Opponent | Mean transfer Δ it causes, C1 | C4 |
|---|---|---|
| PPO | −1.87 | −2.08 |
| ES | −2.15 | −2.49 |
| A2C | −2.62 | −3.62 |
| D3QN | −3.21 | −4.82 |
| QR-DQN | −3.31 | −4.60 |

- PPO opponents bid on few lots (bid rate about 0.37 in C4), cap below fair value (0.91–0.97) and leave 10–12% of their purse, so they leave the most for others.
- D3QN and QR-DQN opponents bid on 80–87% of lots and win at 1.15–1.29× fair.
- Learner-side ranges are wide. ES in C4 runs from −2.58 (vs PPO) to −6.68 (vs D3QN); A2C in C4 from −1.99 to −5.16. PPO's range is narrowest (C4 −2.16 to −2.47).
- Direction matters. In C1, PPO vs D3QN is +0.58 head-to-head while D3QN vs PPO is −0.61, which is consistent. Transfer is not symmetric: PPO loses 2.14 against D3QN, D3QN loses 1.64 against PPO.

**C. Exploitability (the worst opponent for each learner, by transfer Δ).**

| Learner | Worst opponent in C1 | Worst opponent in C4 |
|---|---|---|
| PPO | QR-DQN −2.25 | D3QN −2.47 |
| A2C | D3QN −3.47 | D3QN −5.16 |
| D3QN | QR-DQN −3.56 | QR-DQN −5.07 |
| QR-DQN | D3QN −3.31 | D3QN −4.96 |
| ES | QR-DQN −4.00 | D3QN −6.68 |

- No export collapses to illegal play. Learner legal XI is 100% everywhere except 10 C4 episodes (A2C 3, QR-DQN 4, ES 3; section K).
- There are **quality collapses**:

  | Setting | Strong XI | Average rank |
  |---|---|---|
  | ES, C4 vs D3QN | 44% (1,994/4,500) | — |
  | ES, C4 vs QR-DQN | 52% | — |
  | A2C, C4 vs D3QN | 64% | — |
  | ES, S4 | 33.7% | 9.0 |
  | A2C, S4 | 61.1% | 8.45 |

- The mechanism for ES and A2C is the same: high-cap bidding on most lots wins marginal players, loses contested stars to opponents with more purse left, and saturates the squad.
- PPO shows no exploitable weakness against any of the four learned opponents measured here. This is not a best-response exploitability test (Phase 2A suite S7), which would require training and is out of scope.

**D. Behavioural shifts.** See section H: higher bid rates and prices, fewer stars, more marginal and re-auction buys, later requirement completion, and more shield dependence.

**E. Seed stability.**
- Across the 9 seed pairings of each cell, the sd of transfer Δ has median 0.45 in C1 (maximum 0.85, A2C vs ES) and median 0.62 in C4 (maximum 1.51, ES vs D3QN).
- The sign is stable: **all 9 seed pairings are negative in all 40 C1/C4 cells**, and each pairing's own CI excludes 0 (360/360).
- Variation comes more from the opponent's seed than the learner's. For example, opponent seed s2 of PPO is the harshest for every learner (C1 A2C vs PPO by opponent seed: −1.42 / −2.38 / −1.77).
- PPO's cells are the most seed-stable (C1 sd 0.14–0.20).
- Per-seed values for every cell are in `matchup-results.json` (`seedPairs`, `seedStability`) and in `plots/c1_seed_stability.svg` and `plots/c4_seed_stability.svg`.

**F. Safety preservation.** All frozen safety invariants held under RL-vs-RL interaction (section K). The one new phenomenon is the 20 incomplete XIs in C4: legal behaviour, reported as findings.

**Purse strata.**
- Transfer losses are similar across strata, and slightly smaller in the high stratum for PPO, D3QN and QR-DQN (S4: PPO low −2.63, normal −2.40, high −2.22).
- ES loses most in the normal and high strata (S4 −7.11 and −6.98, against −6.45 in low).
- The incomplete-XI findings concentrate in the high stratum (19 of 20), where large purses are spent early and cheap keepers are passed over.
- Per-cell strata are in `matchup-results.json`.

---

## J. Shield analysis

Per learner, across 18,000 episodes each in C1 and C4 and 1,500 each in A and S4.

| Metric | Condition | PPO | A2C | D3QN | QR-DQN | ES |
|---|---|---|---|---|---|---|
| **Shield activations** (all forced) | A | 0 | 375 | 22 | 53 | 339 |
| | C1 | 34 | 12,730 | 2,090 | 1,955 | 9,613 |
| | C4 | 87 | 9,159 | 3,347 | 2,920 | 4,144 |
| | S4 | 2 | 842 | 644 | 430 | 102 |
| **Forced bids per episode** | A | 0 | 0.250 | 0.015 | 0.035 | 0.226 |
| | C1 | 0.002 | 0.707 | 0.116 | 0.109 | 0.534 |
| | C4 | 0.005 | 0.509 | 0.186 | 0.162 | 0.230 |
| | S4 | 0.001 | 0.561 | 0.429 | 0.287 | 0.068 |
| **Won after forced activation** | C4 | 56% | 40% | 41% | 44% | 45% |
| **Forced keeper bids per episode** | C4 | 0.003 | 0.495 | 0.155 | 0.140 | 0.224 |
| **Final-path forced bids** (total) | A | 0 | 0 | 0 | 0 | 0 |
| | C1 | 0 | 0 | 0 | 3 | 0 |
| | C4 | 0 | 8 | 3 | 12 | 4 |
| | S4 | 0 | 0 | 0 | 1 | 0 |
| **Re-auction forced bids** (per episode) | S4 | 0 | 0 | 0.016 | 0.011 | 0 |

- **Shield dependence rises everywhere.** Forced bids per episode rise 8–30× for D3QN and 3–8× for QR-DQN relative to Stage A.
- **PPO's first shield activations** appear: 34 forced decisions in 18,000 C1 episodes and 87 in C4, where Stage A had 0. Its final-path forced bids remain 0.
- **Final-path forced bids**, which no trained policy made in Stage A validation, appear 31 times in Stage B (C1 3, C4 27, S4 1).
- **Forced bids are often lost** in Stage B: 44–60% of forced bids are lost in C4 (PPO 44%, the others 55–60%). A forced bid only guarantees that the learner bids, not that it wins against an RL rival with more purse.
- **The 20 incomplete XIs (section K)** are exactly the cases where the last forced, final-path keeper bid was lost and no keeper was left within reach.

---

## K. Safety audit

The full data is in `safety.json`, `safety-episodes.json` and `findings.json`.

| Counter (hard stop) | A | C1 | C4 | S4 |
|---|---|---|---|---|
| Illegal or masked action reaching the simulator | 0 | 0 | 0 | 0 |
| Purse / squad > 25 / overseas > 8 / duplicate purchase (all teams, `auditAuction`) | 0 | 0 | 0 | 0 |
| Invariant violations (all teams) | 0 | 0 | 0 | 0 |
| NaN / Infinity | 0 | 0 | 0 | 0 |
| Crashes / deadlocks | 0 | 0 | 0 | 0 |
| Wrong model (weight digest, algorithm id, spec hashes) | 0 | 0 | 0 | 0 |
| Opponent runtime fallbacks | 0 | 0 | 0 | 0 |
| Opponent RL decisions checked | — | 11,335,475 | 64,906,717 | 4,393,964 |
| Learner decisions checked | 790,640 | 11,317,433 | 12,307,579 | 1,100,917 |
| Learner legal XI | 7,500/7,500 | 90,000/90,000 | 89,990/90,000 | 7,500/7,500 |
| Incomplete XI, RL opponent seats | 0 | 0 | 10 | 0 |
| Incomplete XI, frozen rule bots and rule fallbacks | 0 | 0 | 0 | 0 |

The passive human proxy never bids by design, so its incomplete XIs (3,165 / 37,980 / 37,980 / 3,165) are recorded but are not safety events.

**The C4 safety stop and Option A.**
- **The stop.** The first C4 pass stopped at validation seed 100,490 (learner `ppo:s2` against four `d3qn:s1` seats), when a D3QN opponent seat ended without a keeper.
- **The investigation** is in `safety_stop_C4.md` and the trace in `safety_stop_C4_seed100490_trace.txt`. It showed:
  - The room reproduces through the unmodified frozen path with the real clock.
  - Every decision was legal and there were 0 invariant violations.
  - The D3QN had ₹50L left and passed on 13 keepers with base prices of ₹20–50L while the shield rated the requirement SAFE; 9 of them went unsold.
  - Its forced final-path bids in the re-auction were then lost to the PPO learner buying depth keepers.
- **Option A (approved).** An incomplete XI of an RL-controlled team is a recorded finding, but only if it passes a defect screen. The screen replays the room through the unmodified `RlEpisode` runtimes with the real clock plus the `runEpisode` learner loop, and requires an identical summary, auction history and team outcomes, with 0 invariant violations. Fallbacks and illegal actions remain hard stops.
- **C4 was rerun from scratch.** Its first 6,280 episodes are identical to the pre-stop run.

**Findings (20, all in C4, all passed the defect screen on the first attempt).**

| Who | Opponent | Count |
|---|---|---|
| Learner A2C (a2c:s1 ×2, a2c:s2 ×1) | PPO s3 | 3 |
| Learner QR-DQN (qrdqn:s2) | PPO s1 / s2 | 4 |
| Learner ES (s1, s2, s3) | PPO s3 | 3 |
| Opponent seat QR-DQN (qrdqn:s2) | vs learners PPO s2 (2), D3QN s1 (4), ES s1/s2 (2) | 8 |
| Opponent seat D3QN (d3qn:s1, d3qn:s3) | vs learner PPO s2 | 2 |

Every finding shares four features:
- one empty slot, and it is the **wicketkeeper** (0 keepers in the squad);
- 19 of 20 in the high-purse stratum;
- mostly at the 8-overseas cap;
- ≤ ₹500L of purse left at the end, after a lost forced final-path keeper bid.

In the 10 learner cases the four opponents are PPO. Four PPO seats that buy keepers early (keeper completed at 9% of the main round in Stage A) empty the keeper pool, and a learner that deferred its keeper, as A2C, ES and QR-DQN s2 do, is left without one. Each finding has a per-seat decision and shield trace in `findings/`.

These are behavioural findings about the frozen policies and the frozen completion shield v2, not implementation defects. The shield's forcing rule counts rivals that *need* a requirement, so rivals buying spare keepers can take the last reachable candidates. Nothing was changed in response.

---

## L. Reproducibility audit

The full data is in `reproducibility.json`.

| Check | Episodes | Result |
|---|---|---|
| Smoke: 14 workers vs 5 workers (A, C1, C4, S4) | 1,000 | **Identical** |
| Subset rerun vs full run: A (50 entries, 7 workers), C1 (30 entries × 180 pairings, 9 workers), C4 (30 × 180, 11 workers), S4 (100 entries, 5 workers) | 13,050 | **Identical** |
| Smoke run (earlier process) vs full run | 1,000 | **Identical** |
| C4 before the stop (earlier harness version) vs C4 rerun | 6,280 | **Identical** |
| Stage A control vs stored Stage A validation episodes | 7,500 | **Identical** field for field; the three-seed means and sds reproduce the reported values exactly (PPO 92.595 ± 0.082, A2C 90.427 ± 0.046, D3QN 91.490 ± 0.191, QR-DQN 91.853 ± 0.089, ES 91.068 ± 0.044) |

"Identical" covers:
- the auction history digest;
- the learner action-sequence digest;
- every opponent seat's action-sequence digest;
- every metric and the full episode summary.

Fixed throughout:
- the manifest entries;
- the export files (sha256);
- the environment and spec hashes;
- the seat composition (seed-derived, depending on the auction seed only);
- the selection rule and temperature;
- tremble 0;
- the deterministic opponent clock.

---

## M. Production parity

The full data is in `parity.json` and `raw/parity.txt`. **Result: PASS for all 15 exports.**

**The chain checked, per export:**

```
training checkpoint (Python, float64)
  → exported rl-policy-v2 JSON
  → production loader + JavaScript inference (bin/rl-policy-scores.js)
  → same observation, same act-v3 mask
  → same chosen action
```

**States.** 35,069 real states in total: 23,759 opponent-seat and 5,160 learner states from Stage B rooms (S4 and C4, both roles), plus the Stage A final-path fixture (410 states per export, 6,150 in total).

**Category coverage.** Every category is covered for every export. These are the minimums across exports:

| Category | Minimum states per export |
|---|---|
| main | 820 |
| re-auction | 123 |
| low purse (≤ ₹200L) | 160 |
| high purse (≥ ₹9,000L) | 150 |
| keeper needed | 153 |
| Indian needed | 527 |
| bowling needed | 524 |
| shield-forced | 6 |
| final-path | 3 |

**Results.**

| Check | Result |
|---|---|
| Export fidelity: JSON layers vs the checkpoint's export layers | 0.0 for all 15. PPO, A2C, QR-DQN and ES are exact float32; D3QN is the float64 dueling fold. |
| Scores: Python float64 forward of the training network vs production JavaScript | Maximum \|Δ\| 2.5e-14 (logits, Q, mean of 32 quantiles). QR-DQN raw quantiles match to 7.1e-15. |
| Masked argmax agreement under the real mask, PASS-only, one-legal and all-legal masks | 100% |
| Recorded production choices (D3QN, QR-DQN, ES) vs Python masked argmax | 100% |
| PPO/A2C temperature-0.3 distributions | Maximum \|Δp\| 3.0e-15 |
| Recorded choices legal under act-v3 | 100% |
| Identifiers | Correct algorithm, head, selection, obs hash `629b25783f833af7` and act hash `5f72f510c48b1f46` |

**Fallback suites** (existing production-contract checks, unmodified; PPO uses A2C's suite with the algorithm id changed):

| Algorithm | Checks passed |
|---|---|
| PPO | 24/24 |
| A2C | 24/24 |
| D3QN | 30/30 |
| QR-DQN | 30/30 |
| ES | 30/30 |

The suites cover: missing, invalid or mismatched model; malformed architecture; NaN/∞; masked or out-of-range action; timeout; healthy decision = masked argmax; cap ≤ maxSafeBid.

---

## N. Performance

The full data is in `performance.json` and `raw/latency_*.json`.
- **Clock and guard:** the real clock and the production `createRlSeat` with its unmodified 20 ms guard.
- **What is timed:** the whole `decide()` call (plan, mask, observation, inference, checks) in mixed S4 rooms, with all five algorithms in every room.

| Load | Episodes | RL decisions | Decisions/s | p50 | p95 | p99 | Maximum (model decisions) | > 20 ms | Guard trips |
|---|---|---|---|---|---|---|---|---|---|
| 1 room at a time (3 runs) | 120 | 214,848 | about 3,000 | 0.05–0.20 ms | 0.31–0.47 ms | 0.43–0.60 ms | 8.16 ms | 0 | 0 |
| 4 concurrent rooms | 80 | 143,176 | 10,754 | 0.05–0.19 ms | 0.34–0.51 ms | 0.48–0.67 ms | 4.00 ms | 0 | 0 |
| 14 concurrent rooms (CPU saturated, 16 threads) | 336 | 595,257 | about 19,000 | 0.09–0.38 ms | 0.64–0.90 ms | 0.95–1.34 ms | 19.5 ms | see trips | **15** |

Per-algorithm p50 values for a single room: PPO 0.19 ms, A2C 0.06 ms, D3QN 0.04 ms, QR-DQN 0.04–0.05 ms, ES 0.05 ms.

**Guard trips (reported separately; the guard is unchanged).**
- All 15 trips happened at 14 concurrent rooms: 20.1–34.6 ms, spread across PPO, A2C, D3QN, QR-DQN and ES.
- Each tripped seat played its rule persona for the rest of the room, as production specifies: 6,452 rule decisions in total, and about 1.1% of the 1,344 RL seat-rooms.
- These are OS or GC scheduling outliers under full CPU saturation, consistent with the Phase 2D.4 finding. At ≤ 4 concurrent rooms there were 0 trips in about 358,000 real-clock decisions.

**Cross-play throughput.** On 14 workers the runs managed 10.0–11.0 episodes/s. Combined learner and opponent decision rates were A 1,049/s, C1 2,718/s, C4 8,888/s and S4 8,082/s.

---

## O. Limitations

1. **Validation seeds only (500 auctions).** The test split is untouched, as instructed. Results describe the validation distribution.
2. **One frozen checkpoint per training seed** (the final export). Earlier checkpoints were not evaluated.
3. **Opponent-side decision metrics** (bid rate, cap/fair) come from the opponent runtime. Opponent-side shield statistics are not recorded; they come from the reverse ordered pairing.
4. **C4 head-to-head** compares the learner with the *mean* of four identical opponents that also compete with each other. It is not a symmetric duel; transfer Δ is the primary measure.
5. **Transfer Δ measures the combined effect** of an RL opponent's style and of losing a predictable rule fallback. The two cannot be separated here.
6. **Exploitability is measured only against the four other frozen learners.** No best-response training was done (S7 would require training).
7. **S4 uses seed-matched opponents** (seed k for all four), a design choice. Other seed mixes were not evaluated.
8. **Latency was measured on this development machine.** The concurrent figures depend on its 16-thread CPU and on Windows scheduling.
9. **Harness change during the phase.** After the C4 stop, `harness.mjs` gained the Option A screen. Rooms without an incomplete XI are unaffected, shown by 6,280 identical C4 episodes and the smoke-vs-full comparisons. A and C1 were not rerun.
10. **Human proxy.** The passive human proxy's incomplete XIs are by design.

---

## P. Raw result references

All paths are relative to `ml/reports/phase2e0/` unless stated otherwise.

| File | Contents |
|---|---|
| `report.md` | This report |
| `gap_report.md` | Stage B infrastructure gap report (decisions D1–D7) |
| `smoke.md` | Smoke test |
| `matchup-matrix.json` | Transfer and head-to-head matrices (C1, C4, S4) with CIs; Stage A reference |
| `matchup-results.json` | Every ordered pairing: learner and opponent XI, transfer, head-to-head (and vs best seat in C4), W/T/L, 9 seed pairings, seed stability, strata, learner and opponent behaviour profiles; S4 detail; Stage A vs B behaviour; worst opponents |
| `episode-results.json` | One row per episode (195,000 cross-play + 7,500 control; 27 columns) |
| `safety.json`, `safety-episodes.json` | Safety counters per condition; fallback suites; the stop record |
| `findings.json`, `findings/` | The 20 Option A findings, each with its per-seat decision and shield trace |
| `safety_stop_C4.md`, `safety_stop_C4.STOP.json`, `safety_stop_C4_seed100490_trace.txt` | The C4 stop and its investigation |
| `reproducibility.json` | Every rerun comparison and the Stage A regression |
| `parity.json` | Per-export parity, per-category results and identifiers |
| `performance.json` | Latency (single, 4 and 14 concurrent rooms), guard trips, throughput |
| `frozen-hashes.json` | sha256 of the frozen files, exports and checkpoints |
| `plots/` | `index.html` plus the SVGs: C1/C4 transfer and head-to-head heatmaps, Stage A → B XI, C1/C4 seed stability, S4 transfer and head-to-head |
| `raw/` | Analysis printout, Stage A regression, comparison logs, fallback/parity/latency outputs, pipeline logs, test logs, run metadata |
| `ml/runs/_2e0/` (git-ignored) | Full per-episode records, about 870 MB of JSONL (`full/A,C1,C4,S4.jsonl`), plus the smoke, reproducibility and audit runs |
