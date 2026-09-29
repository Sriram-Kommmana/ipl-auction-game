# Phase 2F: Stage-B RL training pilot

This is a pilot: the first Stage-B training run, trained under the frozen contract.

- **Frozen and unchanged:** observation `629b25783f833af7`, action space `5f72f510c48b1f46`, act-v3 shield, reward (γ = 1), simulator, planner and rule bots.
- **Not touched:** no Stage-A artifact was overwritten, no production file changed, no model was promoted, and nothing is committed.
- **Exports are experimental:** Stage-B exports carry the tag `stage_b_pilot` in their config.

## Decision gate: RESULT B, MIXED / PROMISING

> **Implication.** Analyse the failure mechanisms before deciding whether to continue, tune or redesign. **Do not launch the full Stage-B budget automatically.**

Training against a mixed frozen population of RL policies and rule bots moved four of the five algorithms (PPO, A2C, QR-DQN, ES) in the intended direction. It came with a small but statistically clear Stage-A cost, it hurt D3QN, and several of the targeted mechanisms did not move.

**What improved:**
- **Absolute cross-play XI.** Compared with the frozen Stage-A export on the same entries, cells and opponents, XI is higher in **every** C1/C4/S4 cell for PPO, A2C, QR-DQN and ES: +0.13 to +0.80 XI, every bootstrap 95% CI above 0. Transfer Δ shrank for all five algorithms.
- **Keeper failures fell.** Learner incomplete-XI findings dropped from 10 to 5. M1-type (a needed keeper passed while affordable) fell from 6 to 3, including QR-DQN's four to 0; M2-type (purse and overseas slots exhausted early) fell from 4 to 2. QR-DQN almost stopped passing keepers that were affordable and needed (0.61 → 0.03 per C4 episode). Final-path forced bids fell from 28 to 12.
- **Early stars rose for PPO, A2C and ES** in aggressive rooms. For example, stars won in the first 30% of a PPO C4 episode went from 1.9 to 3.4.
- **Sensitivity to the opponent algorithm fell** for A2C, D3QN, QR-DQN and ES. The C1→C4 escalation shrank for A2C, D3QN and ES.

**What did not improve, or got worse:**
- **Stage-A control regressed for every algorithm.** Paired with the frozen export on the same 500 entries: PPO −0.22, A2C −0.41, QR-DQN −0.49, ES −0.31. D3QN fell by **−2.69**.
- **D3QN regressed outright.**
  - Its cross-play XI is *lower* than Stage A in every condition: C1 −1.66, C4 −1.34, S4 −0.49.
  - Its large transfer "improvement" comes entirely from the collapse of its own Stage-A control. One seed ends at 86.53 on validation, below the Moneyball rule bot.
  - It learned to under-bid for stars (bid cap 0.79 × fair value, against 1.26 in Stage A), to leave 14–19% of its purse unspent, and to fill 21–24-player squads, many bought in the re-auction.
- **Shield dependence rose for A2C and ES.** Forced keeper bids went up: A2C +30% in C4 and +86% in S4; ES +27% in C4 and ×4.9 in S4. The share of ES S4 episodes whose requirements were completed by a forced bid rose from 3% to 14%.
- **Responsiveness to opponent signals did not change.** Correlations between the cap and the recent price ratio or rival-purse features, and same-lot responses to aggressive rooms, are at Stage-A levels.
- **The C1→C4 escalation did not shrink for PPO or QR-DQN.** QR-DQN and D3QN won *fewer* early stars.

**Why RESULT B:**
- **Not A (clear improvement):** Stage-A performance is not preserved, and the mechanisms are mixed.
- **Not C (no meaningful improvement):** four algorithms show real, CI-separated gains in absolute cross-play XI.
- **Not D (regression):** the RESULT D criteria (significant Stage-A degradation and new behavioural problems) **are met for D3QN alone**. The other four algorithms show small (−0.2 to −0.5 XI) but significant Stage-A costs and no new safety problems.

This classification is a judgement over mixed evidence. A reviewer who treats any significant Stage-A regression as disqualifying would classify the pilot as D.

## 1. What ran

### Design decisions (approved in chat)

| Item | Value |
|---|---|
| Opponent curriculum | the frozen sampler's league draw. Each of the four non-learner RL seats becomes a frozen snapshot with probability **0.5**, drawn uniformly from all **15 Stage-A exports** (5 algorithms × 3 seeds, including the learner's own algorithm). The 4 rule bots and the human proxy always sit. Trembling (a random legal action 1% of the time) applies to snapshots only. |
| Initialisation | **warm start** from each seed's frozen Stage-A final checkpoint, with its sha256 checked against the Phase 2E.0 record, and a fresh optimizer. D3QN and QR-DQN keep exploration ε constant at its Stage-A final value of 0.05, and start with an empty replay buffer. |
| Seeds | run seeds **101 / 102 / 103**, warm-started from Stage-A seeds 1 / 2 / 3. Train split only. |
| Budget | **497,664** learner decisions per seed: 81 updates of 12 × 512 for PPO and A2C, 41,472 cycles of 12 for the DQNs. ES stops at the first generation boundary at or after the budget (499,984–502,843 decisions). |
| Checkpoints | 98,304 / 245,760 / 497,664 decisions. ES checkpoints fall at the first generation at or after each. |
| Hyperparameters | exactly the Phase 2C/2D configs: architecture, learning rate, batch, entropy, γ, GAE, replay, ES σ and population, optimizer. The only override is the approved ε rule. |
| Evaluation | c100 and c250 on validation entries 0–99. c500 on all 500 entries. Both use the exact Phase 2E.0 protocol: Stage-A control; C1 and C4 against the other four algorithms × 3 seeds; S4 with the seed-derived rotation at the learner's Stage-A seed index. |

The ES budget is asymmetric. ES used 10.2 M decisions in Stage A, so 500k decisions is about 5% of its Stage-A budget, against 25% for the other four algorithms.

### Code changes

Three new pieces, all under `ml/ipl_rl/stage_b/`:
- **`bridge_b.mjs`, the Stage-B training bridge.** It uses the frozen sampler, `RlEpisode`, mask, shield and reward unchanged, and loads the 15 exports with digest checks. Snapshot seats run the production runtime with the fixed clock, as in Phase 2E.0 decision D4. Any fallback is a hard stop, and the pool's weights are re-hashed every 250 episodes.
- **A launcher**, `train_b.py`.
- **Default-off hooks** in the five trainers, `bridge.py` and `vec_env.py`: 42 lines added and 26 removed, all tracked edits (`raw/trainer-hooks.diff`).
  - With `stage = "A"`, every hook is a no-op that keeps the original checks and messages. All **102 existing Python tests pass** (`raw/unittest-after-hooks.log`).
  - In Stage B only, a learner training episode that ends without a legal XI is recorded as a finding (§26), not a hard stop. There were none: 0 such training episodes out of 64,678.

### §21 pre-flight on one run (PPO s101), before the other 14

All checks passed:
- **Opponent composition:**
  - snapshots per room matched Binomial(4, 0.5) (0: 6.4%, 1: 26.9%, 2: 37.7%, 3: 23.0%, 4: 6.0%);
  - the rule-bot share was at least 4/9 in every room;
  - each of the 15 exports appeared at about 1/15 (own algorithm about 20%).
- **Hashes:**
  - frozen hashes and obs/act/γ unchanged;
  - warm-start sha256 matches the frozen record;
  - 2,439 distinct train seeds, with 0 validation or test seeds.
- **Safety:** no fallbacks and no invariant violations.
- **Evaluation pipeline:** the harness's Stage-A control equals the trainer's own validation run exactly (max |ΔXI| 0.0 on 100 entries).

### Training summary (`pilot-summary.json`, `raw/runs.json`)

- 15 of 15 runs completed.
- 7.48 M learner decisions over 64,678 training episodes, in 3.8 h of training wall time.
- Mean RL snapshots per room: 1.95–2.03.
- **0 hard stops, 0 fallbacks, 0 illegal actions, 0 invariant violations, 0 learner incomplete-XI training episodes.**

## 2. Safety (§25, §26)

Hard stops were checked in training, evaluation (273,000 episodes) and replay (15,600 episodes):
- **Hard stops: none.**
  - 0 illegal or masked actions;
  - 0 purse, squad or overseas violations;
  - 0 simulator inconsistencies;
  - export parity passed at every checkpoint (maximum score difference ≤ 6.7e-6; argmax agreement 1.000);
  - frozen hashes unchanged at 4 checks;
  - opponent-policy digests unchanged;
  - reward and γ unchanged;
  - no test-split seed.
- **Behavioural findings (not stops), 500k evaluation, 500 entries:**

| | Stage B pilot | Stage A (same cells) |
|---|---|---|
| Learner incomplete XI: A2C C4 | 2 | 3 |
| Learner incomplete XI: ES C4 | 3 | 3 |
| Learner incomplete XI: QR-DQN C4 | **0** | 4 |
| M2-type | 2 | 4 |
| M1-type | 3 | 6 |
| Opponent incomplete XI (frozen RL opponents in the room) | 12 | 10 |

Every finding passed the Phase 2E.0 defect screen: the room reproduces through the unmodified frozen path with 0 violations.

## 3. Primary result: XI and transfer at 500k (`algorithm-results.json`)

**Transfer Δ** is XI in the condition minus the same policy's Stage-A-control XI on the same entry.
- **Improvement** is the Stage-B transfer minus the frozen Stage-A export's transfer, paired on the same (entry, cell, opponent).
- **ΔXI (B − A)** is the paired absolute XI difference in the same cell. Because a lower control mechanically shrinks transfer Δ, this is the decisive column.
- Values are the mean of three seeds; CIs are bootstrap 95% over entries.

| Algorithm | Condition | Stage-B XI | Stage-A XI | **ΔXI (B − A)** [95% CI] | Transfer B | Transfer A (§30 ref.) | Improvement [95% CI] |
|---|---|---|---|---|---|---|---|
| PPO | A | 92.38 | 92.60 | **−0.22** [−0.24, −0.19] | — | — | — |
| | C1 | 91.00 | 90.65 | **+0.35** [0.34, 0.37] | −1.38 | −1.95 (−1.95) | +0.57 [0.55, 0.60] |
| | C4 | 90.64 | 90.28 | **+0.36** [0.34, 0.38] | −1.74 | −2.32 (−2.32) | +0.58 [0.55, 0.61] |
| | S4 | 90.48 | 90.17 | **+0.32** [0.28, 0.36] | −1.90 | −2.43 (−2.43) | +0.53 [0.48, 0.59] |
| A2C | A | 90.02 | 90.43 | **−0.41** [−0.46, −0.35] | — | — | — |
| | C1 | 87.95 | 87.64 | **+0.31** [0.28, 0.34] | −2.07 | −2.79 (−2.79) | +0.72 [0.67, 0.77] |
| | C4 | 87.29 | 86.71 | **+0.58** [0.55, 0.60] | −2.73 | −3.72 (−3.72) | +0.98 [0.93, 1.04] |
| | S4 | 86.03 | 85.28 | **+0.75** [0.70, 0.79] | −3.99 | −5.14 (−5.14) | +1.15 [1.09, 1.23] |
| D3QN | A | 88.81 | 91.49 | **−2.69** [−2.77, −2.60] | — | — | — |
| | C1 | 87.24 | 88.89 | **−1.66** [−1.70, −1.62] | −1.57 | −2.60 (−2.60) | +1.03 [0.95, 1.10] |
| | C4 | 86.87 | 88.20 | **−1.34** [−1.38, −1.29] | −1.94 | −3.29 (−3.29) | +1.35 [1.27, 1.43] |
| | S4 | 86.18 | 86.67 | **−0.49** [−0.57, −0.41] | −2.62 | −4.82 (−4.82) | +2.20 [2.09, 2.31] |
| QR-DQN | A | 91.37 | 91.85 | **−0.49** [−0.52, −0.45] | — | — | — |
| | C1 | 89.56 | 89.40 | **+0.16** [0.14, 0.19] | −1.81 | −2.46 (−2.46) | +0.65 [0.60, 0.69] |
| | C4 | 88.75 | 88.62 | **+0.13** [0.10, 0.16] | −2.62 | −3.23 (−3.23) | +0.61 [0.57, 0.66] |
| | S4 | 88.31 | 87.51 | **+0.80** [0.74, 0.85] | −3.06 | −4.34 (−4.34) | +1.28 [1.21, 1.35] |
| ES | A | 90.76 | 91.07 | **−0.31** [−0.38, −0.24] | — | — | — |
| | C1 | 87.92 | 87.70 | **+0.22** [0.19, 0.25] | −2.84 | −3.37 (−3.37) | +0.53 [0.46, 0.59] |
| | C4 | 86.49 | 86.01 | **+0.48** [0.45, 0.50] | −4.27 | −5.05 (−5.06) | +0.78 [0.71, 0.85] |
| | S4 | 85.03 | 84.24 | **+0.79** [0.72, 0.86] | −5.73 | −6.83 (−6.83) | +1.10 [1.00, 1.19] |

**The Stage-A column reproduces the §30 references to rounding.** ES C4 recomputes as −5.05 against −5.06 in the spec; this was already reported in the pre-flight.

### Across seeds

The cells above show the entry-level CI. Seed spread of the Stage-B XI at 500k:
- PPO: 0.14–0.67;
- A2C: 0.30–0.47;
- QR-DQN: 0.22–0.33;
- ES: 0.02–0.80;
- **D3QN: 0.55–2.23.** Its Stage-A control by seed is 86.5 / 88.9 / 91.0.

### Across checkpoints (plots 05a–c; entries 0–99 at every checkpoint)

| Algorithm | Stage-A control ΔXI | C4 ΔXI | Trend |
|---|---|---|---|
| PPO | −0.07 → −0.12 → −0.16 | +0.08 → +0.15 → +0.32 | improves monotonically |
| A2C | −0.16 → −0.34 → −0.38 | −0.02 → +0.30 → +0.55 | improves monotonically |
| ES | −0.25 → −0.32 → −0.35 | +0.31 → +0.48 → +0.53 | improves monotonically |
| QR-DQN | −2.05 → −0.66 → −0.52 | −1.05 → −0.17 → +0.03 | dips at 100k, recovers |
| D3QN | −1.41 → −2.64 → −2.71 | −0.93 → −0.99 → −1.31 | worsens |

- PPO, A2C and ES improve steadily through 500k. The pilot has not shown them saturating.
- QR-DQN dips at 100k and then recovers. The empty replay buffer is a likely contributor.
- D3QN keeps getting worse.

## 4. Stage-A control (`stage-a-control.json`, §12)

The trainer's own 500-episode validation, the exact Stage-A evaluation, at 500k:

| Algorithm | Seed 101 | Seed 102 | Seed 103 | Stage-A anchor (§31) |
|---|---|---|---|---|
| PPO | 92.41 | 92.50 | 92.22 | 92.595 ± 0.082 |
| A2C | 90.08 | 90.29 | 89.69 | 90.427 ± 0.046 |
| D3QN | **86.53** | 88.90 | 90.98 | 91.490 ± 0.191 |
| QR-DQN | 91.57 | 91.40 | 91.14 | 91.853 ± 0.089 |
| ES | 90.78 | 90.76 | 90.75 | 91.068 ± 0.044 |

- **Every algorithm lost Stage-A performance.**
  - The loss is 0.2–0.5 XI for PPO, A2C, QR-DQN and ES.
  - It is 0.5–5.0 XI for D3QN.
- **D3QN s101 falls below Moneyball** (−0.93 against Moneyball, versus about +4 for Stage-A D3QN).
- **That is the trade-off §33 asks to report:** better transfer came at the cost of the original environment.

## 5. C1, C4 and S4 detail (`c1-results.json`, `c4-results.json`, `s4-results.json`)

**Robustness to the opponent algorithm.** This is the sd of C4 transfer across the four opponent algorithms:

| Algorithm | Stage A | Stage B | Change |
|---|---|---|---|
| A2C | 1.31 | 1.14 | lower |
| D3QN | 1.23 | 0.23 | lower |
| QR-DQN | 1.18 | 0.73 | lower |
| ES | 1.58 | 1.43 | lower |
| PPO | 0.12 | 0.15 | unchanged |

**The worst opponent is still D3QN copies** for every learner except D3QN, whose worst opponent is QR-DQN copies.

**C1→C4 escalation (§14C),** C4 transfer minus C1 transfer for the same learner:

| Algorithm | Stage A | Stage B | Escalation shrank by |
|---|---|---|---|
| A2C | −0.93 | −0.66 | +0.27 |
| D3QN | −0.69 | −0.37 | +0.32 |
| ES | −1.69 | −1.43 | +0.26 |
| PPO | −0.37 | −0.37 | 0.01 |
| QR-DQN | −0.77 | −0.80 | −0.03 |

- It shrank for A2C, D3QN and ES.
- It did not shrink for PPO or QR-DQN. Multiple aggressive opponents remain costlier than one.

**Learner × opponent (C4, plot 16).**
- The improvement is positive in all 20 cells, with every CI above 0.
- It is largest against D3QN and QR-DQN copies: A2C +1.29 and +0.94; ES +0.81 and +1.30; QR-DQN vs D3QN +1.27.
- For D3QN, this "improvement" reflects its lower control, not higher XI.

## 6. Mechanisms (§14): the full behaviour profile

**Metric sources:**
- **Record level:** 500 entries (`behavioral-metrics.json`).
- **Per decision:** digest-verified replays on the 40 Phase 2E.2 entries, paired with the Stage-A replays of the same cells (`star-analysis.json`, `keeper-analysis.json`, `shield-analysis.json`).

### A. Early stars

Stars won in the first 30% of the main round, per episode:

| Algorithm | C4: A → B | S4: A → B | Verdict |
|---|---|---|---|
| PPO | 1.9 → **3.4** | 1.5 → **2.2** | improved |
| A2C | 5.2 → **5.8** | 3.4 → **4.5** | improved |
| ES | 3.8 → **4.7** | 1.6 → **2.4** | improved |
| QR-DQN | 6.7 → **5.0** | 5.9 → **3.9** | worse |
| D3QN | 7.0 → **3.1** | 6.1 → **2.3** | worse |

- For PPO, A2C and ES, stars won in the first 10% of the main round rose similarly. PPO also raised its star bids (cap/fair 1.03 → 1.17 in C4).
- D3QN's star bids fell from 1.24 to 0.82 × fair.
- **Across the whole auction** (500 entries), total stars rose for PPO, A2C and ES in C4/S4 and fell for QR-DQN and D3QN.
- **The rise is partial.** Early stars stay well below the Stage-A-control level (about 10 per episode). *Early stars improved for three algorithms, not for all five.*

### B. Responsiveness to opponents

Within-bin Spearman correlation of the learner's cap/fair with each feature, main round:

| Algorithm | Recent price ratio: B (A) | Own purse: B (A) |
|---|---|---|
| PPO | 0.07 (0.07) | 0.08 (0.12) |
| A2C | 0.22 (0.28) | 0.40 (0.44) |
| D3QN | 0.13 (0.08) | −0.31 (0.26) |
| QR-DQN | 0.10 (0.08) | 0.18 (0.24) |
| ES | 0.22 (0.18) | 0.41 (0.31) |

- **Same-lot comparison** (C4 D3QN/QR-DQN room vs Stage-A room, similar own state): the mean change in cap/fair is −0.01 to +0.08 for Stage B, against −0.01 to +0.10 for Stage A.
- **No meaningful change in how bids respond to opponent pressure.** The XI gains come with the same weak responsiveness Phase 2E.2 found. They appear to be a **better prior** (bidding differently from the start), not *adaptation* within an auction. This is associational: no policy-sensitivity test was run.

### C. Multiple aggressive opponents

- The C1→C4 escalation shrank for A2C, D3QN and ES, and did not for PPO or QR-DQN (§5).
- Absolute C4 XI rose for PPO, A2C, QR-DQN and ES.

### D. Keepers

| Metric (per episode unless stated) | Stage A → Stage B |
|---|---|
| Keeper passes while affordable and needed, C4 | QR-DQN 0.61 → **0.03**; D3QN 0.85 → 0.31; PPO 0.73 → 0.63; A2C 0.52 → 0.49; ES 0.21 → **0.58** |
| Keeper acquired (progress at which the keeper requirement closed), C4 | earlier for every algorithm: PPO 0.38 → 0.30, A2C 0.60 → 0.51, D3QN 0.25 → 0.09, QR-DQN 0.15 → 0.08, ES 0.70 → 0.65 |
| Needed keepers bid on and lost, C4 | lower for every algorithm (e.g. D3QN 2.30 → 0.87, QR-DQN 1.54 → 1.03) |
| Forced keeper bids (500 entries, C4 total) | QR-DQN 2,512 → **716**; D3QN 2,796 → 86; A2C 8,911 → **11,614**; ES 4,032 → **5,131**; PPO 48 → 58 |
| M1-type learner failures | 6 → **3** (QR-DQN 4 → 0; A2C 1 → 1; ES 1 → 2) |
| M2-type learner failures | 4 → **2** (A2C 2 → 1, ES 2 → 1) |

- **Keeper behaviour improved for QR-DQN and D3QN,** and keepers are secured earlier across the board.
- **A2C and ES lean more on the shield.** ES also passes more affordable keepers it needs.

### E. Final path and shield dependence

- **Final-path forced bids (C4 + S4, 500 entries):** 28 → 12.
  - QR-DQN 13 → 0;
  - A2C 8 → 5;
  - D3QN 3 → 1;
  - ES 4 → 6.
- **Shield intervention rate** stays below 1% of decisions for every algorithm.
  - A2C S4: 0.51% → 0.94%.
  - ES S4: 0.06% → 0.28%.
- **Episodes whose requirements were completed by a shield-forced bid:**
  - A2C S4: 23% → 41%.
  - ES S4: 3% → 14%.
  - QR-DQN C4: 7% → 2%.
  - D3QN S4: 16% → 0.4%.
- **So shield dependence fell for QR-DQN and D3QN and rose for A2C and ES** (§33: an XI gain alongside higher shield dependence, reported as a trade-off).

### F. Purse and squad balance

| Algorithm | Stage A → Stage B (C4 / S4) | Reading |
|---|---|---|
| ES | stranded purse 12% / 18% → **7% / 9%**; squad 23.1 → 21.7 (C4) | healthier |
| A2C | stranded purse 3% / 2% → 0.5% / 0.1%; squad 21.2 / 23.1 → 19.7 / 21.0 | healthier |
| QR-DQN | squad 16.7 → 19.7; bid rate 0.76 → 0.88 | buys more and wider |
| **D3QN** | unspent purse **0.1–0.5% → 14–19%**; squad 16 → 21–24; re-auction buys 0.1 → 1.8–4.6; bid rate 0.83 → 0.63 | **pathological** |

Expensive late purchases (≥ 75% progress at ≥ 1.5 × fair) rose slightly for PPO (0.12 → 0.22 per C4 episode) and D3QN (0.01 → 0.12).

## 7. Did Stage-B training materially reduce the Phase 2E transfer failures?

**Partly.**

| Criterion (§15) | Evidence | Assessment |
|---|---|---|
| Transfer improvement | transfer Δ smaller in all 15 algorithm × condition cells; absolute cross-play XI higher in 12 of 15 (not D3QN) | **yes for 4 of 5 algorithms** |
| Robustness improvement | opponent sensitivity lower for 4 of 5 algorithms; the worst opponent is unchanged | **partial** |
| Keeper improvement | M1 6 → 3 (QR-DQN 4 → 0), M2 4 → 2, final path 28 → 12, QR-DQN passes almost eliminated; ES passes more affordable keepers and A2C/ES lean more on the shield | **mostly yes** |
| Early-star improvement | PPO, A2C and ES win more early stars; QR-DQN and D3QN fewer; all remain far below the Stage-A-control level | **partial** |
| Adaptation improvement | responsiveness to opponent signals unchanged | **no** |
| Safety preservation | 0 hard stops, 0 fallbacks, 0 illegal actions; incomplete-XI findings fell | **yes** |
| Stage-A preservation | −0.2 to −0.5 XI (4 algorithms), −2.7 (D3QN) | **no** (small for 4, major for D3QN) |

## 8. Reproducibility (`reproducibility.json`)

- Seeds, config hashes, warm-start hashes and the sha256 of all 45 checkpoint policies are recorded.
- **Evaluation determinism.** 720 C4 episodes were re-run with 7 and 13 workers. Every digest matched the recorded run.
- **Training determinism.** The PPO Stage-B configuration was trained twice for one update, giving identical parameters and exported layers.
- **Replay.** All 15,600 c500 episodes replay to their recorded digests.
- **Frozen hashes.** The pre-training record (`raw/hashes-pre.json`) passes at 4 checks: before training, before evaluation, after evaluation, and in `integrity-final.txt`.

## 9. Limitations

- **Short run.** This is one pilot budget: 500k decisions, which is 25% of Stage A for four algorithms and about 5% for ES. PPO, A2C and ES were still improving at 500k.
- **Opponents are the frozen Stage-A exports.** Stage-B exports were not added to the evaluation opponent pool.
- **Earlier checkpoints are on 100 entries.** The trend compares c100 and c250 with c500 on the same entries 0–99.
- **Per-decision metrics use 40 entries,** with the high stratum over-represented (20 of 40). Record-level metrics use all 500.
- **The D3QN behaviour may come from the approved warm-start rule** (constant ε = 0.05 and an empty replay buffer) rather than from the opponent distribution. That cannot be separated here, because no ablation was run (§35).
- **Statistics:**
  - the intervals are bootstrap CIs over entries, not multiplicity-corrected tests;
  - "significant" in this report means the CI excludes 0;
  - the three seeds give only a rough idea of seed variance.

## 10. Files

### `ml/reports/phase2f/`

- `report.md`
- `pilot-summary.json`
- `algorithm-results.json`
- `stage-a-control.json`
- `c1-results.json`
- `c4-results.json`
- `s4-results.json`
- `behavioral-metrics.json`
- `keeper-analysis.json`
- `star-analysis.json`
- `shield-analysis.json`
- `transfer-analysis.json`
- `reproducibility.json`
- `integrity-final.txt`
- `plots/`: 18 SVGs plus `index.html` (the 16 required, with plot 5 split into one per condition).
- `checkpoints/`: 45 per-checkpoint reports.
- `raw/`:
  - `hashes-pre.json`
  - `analysis-core.json`
  - `decisions.json`
  - `runs.json`
  - `trainer-hooks.diff`
  - logs

### Per-checkpoint reports (§27)

`stage_b_checkpoint_report.{json,md}` sits next to every checkpoint in `ml/runs/stage_b/<algo>/s<seed>/checkpoints/`.

### Code (uncommitted)

- **New:** `ml/ipl_rl/stage_b/`
  - `bridge_b.mjs`
  - `hooks.py`
  - `train_b.py`
  - `verify_run.py`
  - `hashes_b.mjs`
  - `eval_b.mjs`
  - `replay_b.mjs`
  - `make_learners.py`
  - `analyse_b.py`
  - `decisions_b.py`
  - `package_b.py`
  - `plots_b.py`
  - `run_rest.sh`
  - `run_eval.sh`
- **Edited with default-off hooks:** the five trainers, `bridge.py` and `vec_env.py`.

### Artifacts

Runs, evaluation records and replays are in `ml/runs/stage_b/`. That directory is git-ignored and contains no production artifacts.

**Stopped as §37 requires.** The full Stage-B budget was not launched; there was no tuning, no model was promoted and nothing is committed.
