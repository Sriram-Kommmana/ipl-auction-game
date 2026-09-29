# Phase 2E.3: Minimum observation-gap design

**This is a design analysis only.**

- `obsSpec.js`, the 80-feature vector, the obs hash `629b25783f833af7` and the action hash `5f72f510c48b1f46` are unchanged.
- act-v3, the shield, reward, planner, simulator, rule bots, algorithms, networks, training scripts, production inference and exports are unchanged.
- Nothing was implemented, trained or tuned.
- No feature index, observation version or hash is assigned. Nothing is committed.

## Summary

**Result: no observation change is justified by the evidence. The minimum delta is empty (ADD 0, TRANSFORM 0).**

Every candidate signal was built only from information available at the exact decision point: the current public state plus lots already resolved. Each was then tested against the failure states from Phase 2E.1/2E.2. The tests are simple statistics: rank separation |2·AUC−1|, stratification cells and near-identical-observation pairs. No model was fitted. The data come from the same 15,600 digest-verified replayed episodes and 2,659,482 decision rows as Phase 2E.2.

| Gap | What the evidence shows | Verdict |
|---|---|---|
| **1: Early opponent aggression** | **Lot 1:** every observation feature and every candidate is bit-identical across A, C1, C4 and S4. The distinguishing state does not exist yet (Case B).<br>**Lots 2–20:** the 20-lot window *is* the whole history, and the existing window already separates C4 D3QN/QR-DQN rooms from Stage A at 0.67 in lots 2–5.<br>**Later:** the information is present and the frozen policies don't use it (Case D). | DO NOT ADD |
| **2: How many aggressive opponents** | Where the current rival block aliases one copy vs four, no public statistic differs (separation ≤ 0.02; the sorted purse vector differs in 8% of those pairs). The difference lies in hidden policy identity and future caps. Elsewhere, the best counts add no more than existing features left out of the same cells. | DO NOT ADD |
| **3: Keeper competition from rivals who don't need one** | `riv_able_share` already separates "few rivals can compete" from "many can compete" at 0.997. The non-needing contestant count is 93% determined by existing features, and separates outcomes no better than existing `mkt_scarcity_keeper` (0.455 vs 0.443). No candidate consistently separates the 20 dangerous states. | KEEP |
| **4: Saturated features** | The clip removes only values above 5. The price needed exceeded 5 × fair in 0.03% of decisions, and no sale did. Bowling/Indian scarcity with an own need never fell below 3.7. | KEEP |

**Correction to Phase 2E.2.** Phase 2E.2 §6 said the observation counts only rivals that *need* a keeper. That is true of:
- the demand features: `riv_fills_share`, `mkt_scarcity_keeper` and `self_status_keeper`;
- the planner and the shield.

It is not true of `riv_able_share` and `riv_free_slots_share`. These count every rival able to pay or holding a free slot, whether or not it needs a keeper. So the **breadth** of potential keeper competition is observable. What stays unobservable is whether those rivals **will** bid later, which is future behaviour.

**Implication.** The "targeted observation-spec discussion" that Phase 2E.2 called for can close with **no change to obs-v2** on this evidence. The evidence suggests the remaining Stage-B problems are about training/generalisation and policy behaviour:
- signals that are present but unused;
- affordable keepers that fill a need being passed.

This should be investigated during Stage-B design.

## 1. Freeze and method

**Frozen state was checked.**
- `hashes.mjs --check` passed at the end of Phase 2E.2 and again at the end of this phase (`raw/integrity-final.txt`).
- There are no tracked changes; only new untracked analysis files exist.

**Candidate replay** (`replay_cand.mjs`):
- Same recorded episodes, same row selection and same order as Phase 2E.2 `replay_obs.mjs`. All 2,659,482 rows are verified identical on `(ep, who, seat, dec, slNo, phase)`.
- Every episode reproduced its recorded digests.
- It computes 45 candidate columns per row:
  - **current state:** rival squads, purses, slots, completion costs and rule blocking;
  - **history of resolved lots:** results, per-rival purchases, and the per-lot caps (read from a pass-through wrapper on the simulator instance);
  - **one evaluation label**, the winner type of each keeper lot. It is used only to test candidates, never as a candidate.

**Candidates derived offline.** Some candidates were computed from public state already in the Phase 2E.2 rows. One example is the count of rivals spending ahead of pace, from the nine rival purses and `mkt_premium_passed`.

**Leakage rule (§14).** No candidate uses future player order, results, winners, squads, prices or opponent actions.
- The remaining supply is known at the decision point: it is the catalogue minus auctioned players, and its order is never used.
- Future quantities appear only as **evaluation labels**:
  - realised future star prices;
  - keeper-lot winners;
  - the price needed to win the current lot;
  - RL bidders on the current lot.

## 2. Gap 1: Early opponent aggression (§4, §15)

### What the environment knows before behavioural evidence appears

| Information | Status | Where |
|---|---|---|
| Opponent bid behaviour on past lots (who bid, how far) | **internal, not exposed** | `resolveLot` sees every cap and runs the ladder. History keeps only the winner, price and number of raises. |
| Opponent willingness to pay for the *current* lot | **genuinely unavailable** | Caps are private and chosen simultaneously. Using them would be future leakage. |
| Current demand (which rivals need which role) | derivable, partly exposed | `riv_fills_share` (pooled), `mkt_scarcity_*` |
| Opponent remaining purse | observable (4 statistics); per rival derivable | `riv_purse_*`; `ctx.rivals[].purseLeft` |
| Player-level competition | derivable, partly exposed | `riv_gain_mean`, `riv_able_share`, `riv_capacity_*` (saturated) |
| Historical bidding behaviour | pooled 20-lot window exposed; per rival derivable; ladder participation internal | `mkt_recent_*`, `riv_recent_spend`; `extras.recent` has `winnerTeamId` |
| Opponent policy identity | **internal (training rooms), hidden identity** | `entry.seats[].type`. It is constant in production, which always uses the same lineup of 5 RL and 4 rule seats. |

### Can anything separate the rooms early?

Separation is |2·AUC−1| for learner observations. The comparison is C4 D3QN/QR-DQN rooms vs Stage-A rooms (plot 01).

| Lots | Best of the 80 features | Best candidate | Best candidate inside deciles of that feature | Hidden identity (excluded) |
|---|---|---|---|---|
| 1 | 0.00 (all identical) | 0.00 | 0.00 | 1.00 |
| 2–5 | 0.67 `mkt_recent_price_ratio` | 0.64 `cumStarPF` | 0.29 `wMaxRivSpend` | 1.00 |
| 6–16 | 0.34 `self_need_indians` | 0.24 `overpay15` | 0.35 `cumStarPF` | 1.00 |
| 17–32 | 0.96 `riv_recent_spend` | 0.98 `cumStarPF` | 0.92 `cumStarPF` | 1.00 |

A second comparison labels rooms by what they later became: top vs bottom tercile of realised star price/fair in lots 33–162 (plot 02).

| Lots | Best existing | Best candidate |
|---|---|---|
| 2–5 | 0.40 | 0.40 (`cumStarPF`) |
| 6–16 | 0.29 | 0.18 |
| 17–32 | 0.69 | 0.64 |

- The lots 33–65 bin is not used for this label, because it overlaps the lots that define the label.
- Hidden identity reaches only 0.89 here, and 0.00 between D3QN/QR-DQN rooms and PPO rooms: a count of RL seats says nothing about *which* policy occupies them.

**Classification (§15), by lot range:**
- **Lot 1: Case B.** "This cannot be solved by observation alone at this point, because the required state is not yet determined by the environment." Every feature and every candidate is identical: max |Δ| = 0.000000 across conditions for the same entry and learner.
- **Lots 2–20: Case C was tested and is not supported.**
  - Before lot 21, a cumulative price/fair is identical to `mkt_recent_price_ratio` in 100% of decisions.
  - The compact history candidates are bidding breadth (`partic*`, `bids*`), per-buyer breakdowns, and role-specific (star) prices. They add at most 0.29–0.35 inside deciles of the best existing feature. They are not ahead of the existing window.
- **After ~lot 16: Case D.** The rooms are separable by existing features (0.96–0.99), and the frozen policies do not respond to them (Phase 2E.2 §7).

**Verdict: DO NOT ADD.** Hidden policy identity is excluded for two reasons:
- it is not state (§9);
- in production it would be a constant.

## 3. Gap 2: Number and distribution of aggressive opponents (§5)

The test uses C1 vs C4 pairs: same learner, opponent export, entry and lot, and a similar own state (195,678 pairs). In 11,882 of them, the 17 rival and window features are within 0.05 of each other.

| Candidate (all public) | Tracks RL bidders (within progress) | C4 vs C1 where the rival block is aliased | Share of variance explained by rival-block + price cells | Separation added inside those cells |
|---|---|---|---|---|
| **A. counts:** `aheadPace15` (rivals > 15 pp ahead of pace) | 0.38 | 0.006 | 0.82 | 0.29 |
| `overpay13` (rivals averaging ≥ 1.3 × fair) | 0.09 | 0.008 | 0.80 | 0.32 |
| `starBuyers2` (rivals with ≥ 2 stars) | 0.28 | 0.005 | 0.91 | 0.27 |
| `partic20` (bidders per lot, last 20) | 0.24 | 0.003 | 0.93 | 0.23 |
| `capGe2` (rivals able to pay ≥ 2 × fair) | 0.14 | 0.000 | 0.97 | 0.02 |
| **B. distribution:** sorted 9-purse vector | n/a | differs > 0.05 in 8% of aliased pairs | n/a | n/a |
| **C. identity:** RL seat count | n/a | 1.00 by construction | n/a | n/a |
| existing features left *out* of those cells | 0.25 (best) | n/a | n/a | `riv_xi_max` 0.28, `riv_xi_mean` 0.24, `self_pace_gap` 0.22 |

**Answer:**
- **Where the observation aliases one copy vs four,** no public statistic (count, distribution or history) differs. The difference lives only in the opponents' hidden policies and future caps.
- **Where it does not alias,** the best counts add about as much separation as existing features that were left out of the same cells.
- **So no minimum statistic is justified.**
  - The candidate with the most direct meaning is `aheadPace15`: a count of rivals spending ahead of pace. It is a function of public purses.
  - It is retained as *partial novelty* for the reviewer's information. It does not resolve any aliasing (it differs in 0.08% of near-identical pairs).

**Verdict: DO NOT ADD.**

## 4. Gap 3: Keeper competition from rivals who don't need one (§6, §16)

**Population.** 298,803 keeper-lot decisions where the deciding seat needs a keeper (learner and RL-opponent rows).

**Outcome used for the test.** Whether the lot was taken by a rival that already held a keeper. This happened in 9.6% of these lots.

### State A vs State B (§6)

Both states have few rivals needing a keeper (≤ 1).

| State | Non-needing rivals able to buy | Share of lots taken by a non-needing rival |
|---|---|---|
| A | ≤ 2 | 6.8% |
| B | ≥ 5 | 24.3% |

The rate rises steadily with the number of non-needing contestants: 0% at 0, 9% at 2, 19% at 5 and 36% at 9 (plot 03).

**Existing features already separate A from B:**
- `riv_able_share`: 0.997;
- `riv_capacity_3`: 0.82;
- `riv_free_slots_share`: 0.64.

The information is an **existing capacity statistic combined with existing need statistics**. It needs no new scalar and no transformation.

### Redundancy inside fine cells (plot 04)

The cells hold these fixed: progress, own purse, keeper fair value, keeper supply, rivals needing, `riv_able_share` and `riv_free_slots_share`. There are 6,716 cells.

| Signal | Kind | Share of variance between cells | Separation inside cells |
|---|---|---|---|
| `kContestNoNeed` (non-needing able rivals) | candidate | 0.93 | 0.455 |
| `kW20Buys` (rival keeper buys, last 20) | candidate | 0.89 | 0.446 |
| `kContestAll` | candidate | 0.91 | 0.039 |
| `kHistExtraBuyers` (rivals that bought a keeper while holding one) | candidate | 0.92 | 0.038 |
| `mkt_scarcity_keeper` | existing, not in the cells | n/a | 0.443 |
| `mkt_equivalent_left` | existing, not in the cells | n/a | 0.431 |

The non-needing count adds no more than existing scarcity does. Part of its separation is also true by construction: the outcome is defined as "a non-needing rival won", so more non-needing rivals means more chances.

### The 20 findings (§16; `keeper-gap-analysis.json`; plot 05)

Each dangerous decision was placed among matched safe decisions. These are decisions where the seat needed a keeper, the phase and progress match within ±0.05, and the seat **finished with** a keeper. A percentile of 0.5 means the dangerous decision looked like a typical safe one.

**M1 (14 cases):**
- Non-needing able rivals (`kContestNoNeed`): at or above the 90th percentile in **0** cases. The median case is at 0.63.
- Propensity (`kHistExtraBuyers`): at or above 0.9 in **3** cases (k490, k27, k253), and at or below 0.5 in **7**.
- Stockpile (`kStockExcess`): at or above 0.9 in 1 case, at or below 0.5 in 6.

**M2 (6 cases):**
- Stockpile is at or above 0.9 in 4 cases.
- But existing `self_purse` is at or below the 10th percentile in 5 of 6, and `mkt_scarcity_keeper` at or below the 20th in all 6. M2 is already extreme in the existing features.

The per-finding answers to the seven §16 questions are in `keeper-gap-analysis.json`. By mechanism:

| Question | M1 (14) | M2 (6) |
|---|---|---|
| 1. What made it dangerous? | An affordable, needed keeper was deferred. The remaining keepers were later taken by rivals that already had one. The seat's own purse was nearly empty. | Own purse and overseas slots ran out early. Later keeper bids were lost to rivals with large purses. |
| 2. Already available? | Need, affordability, supply and competition breadth: yes. Future bidding by non-needing rivals: no (future behaviour). | Yes |
| 3. Encoded? | Yes, except propensity | Yes (`self_purse`, `self_overseas_slots_left`, `self_max_safe_*`) |
| 4. Encoded too coarsely? | No (candidate counts 93% determined by existing features) | No |
| 5. Could one scalar separate dangerous from safe? | No. At most 3 of 14 cases reach the 90th percentile on any candidate. | No additional scalar is needed; existing features already flag these cases. |
| 6. Transform an existing feature instead? | No | No |
| 7. Caused by the observation? | **No.** Every one of the 299 passes was on an affordable keeper that filled a visible need (Phase 2E.2). The shield rates danger only by rivals that need a keeper. | **No.** The cause is observable early overspending. |

**Verdict: KEEP.**
- The breadth of competition is already represented.
- The one missing component, willingness to bid in the future, is prediction, not state (§9).

## 5. Gap 4: Saturated features (§7)

| Feature | Raw state | Why it saturates | Strategic range kept? | Information destroyed | Verdict |
|---|---|---|---|---|---|
| `riv_capacity_1–3` | (rival purse − its cheapest legal completion) / lot fair value, for the top 3 rivals | Fair values are small next to purses. Median raw values are 31.7 / 28.3 / 26.6 × fair. At clip: 99.3% / 98.4% / 97.4%. | Yes: values below 5 are exact. The 3rd-best rival is below 2 × fair in 1.5% of decisions. | Only values above 5 × fair. The price needed exceeded that in 0.03% of decisions (99th percentile 3.0), and no sale did (99th percentile 2.5). A log transform would track the price needed at only 0.08–0.10. | **KEEP** |
| `mkt_scarcity_bowling` / `mkt_scarcity_indians` | suitable players left / (own need + rivals needing + 1) | The pool is large: medians 21.9 / 15.3 | Yes: values below 5 are exact | Only degrees of abundance. With an own need, the raw value never fell below 3.73 / 5.09. No bowling or Indians incomplete-XI finding exists. | **KEEP** |

Both are largely uninformative in these rooms, but nothing strategic is lost, so keeping them unchanged costs nothing. Changing them only because they saturate is explicitly out of scope (§7).

## 6. Counterfactual feature ablation (§8, §17; `counterfactual-tests.json`; plot 07)

**Test design.** Pairs of decisions on the same entry, phase and lot, from different episodes or seats, whose **current 80 features are within L∞ ≤ 0.05**. There are 5,538,516 such pairs, 889,318 of them on keeper lots.
- For each candidate: how often it differs when the 80 features don't.
- Then: whether outcome differences are more frequent when it does differ.

| Candidate | Differs among near-identical pairs | Outcome-difference lift when it differs |
|---|---|---|
| keeper candidates (`kContestNoNeed`, `kContestAll`, `kStockExcess`, `kHistExtraBuyers`) | 0–47 pairs out of 5.5 M (0 among the 889,318 keeper pairs, except 1 for `kHistExtraBuyers`) | not measurable |
| `partic20` | 1.9% | ×1.27 (price needed), ×7.4 (keeper absorption) |
| `partic5` | 1.0% | ×0.89 |
| `overpay13` | 0.29% | ×5.5 (price needed) |
| `aheadPace15` | 0.08% | ×4.8 (price needed) |
| `starBuyers2`, `bids20`, `wMaxRivSpend`, `capGe2` | ≤ 0.04% | n/a |

**Reading.**
- Near-identical current observations occur mostly early in the auction.
- There, the public state and public history coincide too, so no candidate resolves the aliasing.
- The candidates that occasionally differ with an outcome lift do so in under 2% of pairs.

**No candidate meets the "genuinely new information" rule:** it must differ in ≥ 5% of pairs with a lift of ≥ 2, *and* beat existing features inside cells.

## 7. Candidates, redundancy, collapse and cost (§10–§14)

The full table is in `candidate-features.json`: source state, current-feature overlap, missing information, availability, leakage risk, computation and production feasibility. The per-candidate evidence and class are in `redundancy-analysis.json`.

**By class:**
- **REDUNDANT:**
  - `cumPF` (identical to the window before lot 21);
  - `capGe*`;
  - `kContestAll` (≈ `riv_able_share`);
  - `kStockExcess` / `kStock2`;
  - `kHistExtraBuyers` / `kW20Extra`;
  - unclipped capacity and scarcity, in the strategic range.
- **PARTIAL NOVELTY:**
  - `partic*` / `bids*`;
  - `cumStarPF`;
  - the per-buyer window breakdown;
  - `overpay*` / `starBuyers2`;
  - `aheadPace*`;
  - the sorted purse vector;
  - `kContestNoNeed*`;
  - `kW20Buys`.
- **GENUINELY NEW INFORMATION:** none.
- **DO NOT ADD (identity):** `rlRivals`.

**Collapsed concepts (§12).**
- The partial-novelty candidates collapse into four ideas:
  1. **A count of high-spending rivals**: `aheadPace15` stands in for `overpay*` and `starBuyers2`.
  2. **Bidding breadth**: `partic20` stands in for `partic5` and `bids*`.
  3. **Non-needing keeper contestants**: `kContestNoNeed`.
  4. **Role-specific price history**: `cumStarPF`.
- If a reviewer still wanted a rival-distribution statistic, each concept has one representative.
- None passes the test, so none is proposed.

**Cost (§13), for those four, in case they are reconsidered:**
- `aheadPace15` and `kContestNoNeed`:
  - cheap and deterministic;
  - no leakage;
  - available immediately from `ctx.rivals`, so computable in the production JS runtime;
  - can be bounded as a count / 9.
- `cumStarPF`: needs the full results history in the runtime `extras`, which currently carries only the last 20 lots.
- `partic20`:
  - needs bid-level events, which exist on the server as `bidPlaced` but are not in `extras`;
  - possible leakage, because a cap at or above the base price can reveal interest that the public ladder would not show.

## 8. Minimum delta (§18)

| Gap | Classification | Information gap closed |
|---|---|---|
| 1: Early opponent aggression | **DO NOT ADD** | None can be closed by observation at lot 1. Afterwards the history is already exposed and unused. |
| 2: Number / distribution of aggressive opponents | **DO NOT ADD** | The aliased difference is hidden identity or future caps. No public statistic carries it. |
| 3: Non-needing keeper competition | **KEEP** | Breadth is already carried by `riv_able_share` / `riv_free_slots_share` plus the need features. |
| 4: Saturated features | **KEEP** | The clip removes nothing strategically relevant. |

**Proposed observation delta: 0 ADD, 0 TRANSFORM** (`minimum-delta.json`, `decision-gate.json`, plot 09). No final design is chosen. No index, hash, network dimension, training code or export is touched.

## 9. Limitations

- **Frozen-policy states only.** The information tests are measured on the states the **frozen Stage-A policies** visited. A policy trained for Stage B would visit other states, where a partial-novelty candidate might matter more. This analysis cannot see that, and it is the main residual uncertainty.
- **Small keeper sample.** The keeper conclusions rest on 20 findings (467 decisions).
- **Coverage.** The replays cover 40 of the 500 validation entries, and the high stratum is over-represented (20 of 40).
- **Arbitrary settings.** Thresholds and cell resolutions are conventions: L∞ 0.05, candidate difference thresholds, deciles and ninths. The raw values are in `raw/gap-stats.json` and `raw/gap-extra.json`.
- **Participation is approximate.** "Participation" counts seats whose cap reached the base price. A live auction reveals only bids actually placed.
- **Associational, not causal.** Separation and between-cell shares are associations. They do not show that a policy would use a signal.

## 10. Files

### `ml/reports/phase2e3/`

- `report.md`: this report.
- `candidate-features.json`: §10, §13, §14.
- `gap-analysis.json`: the four gaps, including the state inventory.
- `keeper-gap-analysis.json`: §6, §16, including all 20 findings.
- `multi-opponent-gap-analysis.json`: §5.
- `early-aggression-analysis.json`: §4, §15.
- `saturation-analysis.json`: §7.
- `redundancy-analysis.json`: §11, §12.
- `counterfactual-tests.json`: §8, §17.
- `minimum-delta.json`: §18.
- `decision-gate.json`.
- `plots/`: 9 SVGs plus `index.html`.
- `raw/`:
  - `gap-stats.json` and `gap-extra.json`;
  - `replay.log`;
  - `integrity-final.txt`;
  - `scripts.txt`.

### New scripts in `ml/ipl_rl/diagnosis/`

- `replay_cand.mjs`
- `gapdesign.py`
- `gap_extra.py`
- `synth2e3.py`
- `plots2e3.py`

The raw candidate arrays are in `ml/runs/_2e3/full/`: 0.5 GB, git-ignored.

Nothing is committed.
