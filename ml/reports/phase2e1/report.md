# Phase 2E.1: Stage B robustness diagnosis

This phase is **analysis only**. Nothing was trained, retuned or modified, no frozen file changed, and no intervention is prescribed.

All numbers come from the frozen Phase 2E.0 records or from read-only replays of them. Each of the 35,020 replayed episodes reproduced its recorded learner-action, auction and summary digests exactly.

Everything below is an **association** unless it is stated as established. No algorithm is ranked or called inferior.

---

## A. Executive summary

**The question.** Why do policies that score about 90.4–92.6 XI in Stage A lose 2–7 XI against learned opponents?

The evidence points to **one dominant mechanism, two amplifiers, and one safety-relevant edge case**.

1. **Losing the early star contests (dominant).**
   - Of all measured behaviours, the loss of star players is the strongest correlate of the XI loss:
     - episode-level Spearman ρ = +0.73 between Δstars and ΔXI, paired against the same auction in Stage A;
     - across the 45 learner × composition cells, Pearson r = +0.96 between mean Δstars and transfer Δ.
   - Stage A rule opponents let every policy buy about 9–10 stars in the first 30% of the auction. RL opponents contest those lots.
   - In S4, stars bought by 30% progress fall to 1.5 (PPO), 1.7 (ES), 3.3 (A2C), 5.9 (QR-DQN) and 6.2 (D3QN).
   - 77% of Stage B episodes lose at least two stars relative to their Stage A pair. Those episodes average ΔXI −3.60, against −1.61 for the rest.
2. **Amplifier: fixed bidding rules meet a different price level.**
   - Paired on the *same player in the same auction*, the policies' offers barely change when their own purse is similar:
     - PPO: mean Δcap/fair −0.02 to 0.00, and the same cap in 78–86% of such lots;
     - ES: −0.02 to +0.05;
     - D3QN, QR-DQN and A2C: +0.01 to +0.11.
   - Offers change mainly with the learner's own purse (Spearman 0.24–0.44 between Δcap/fair and Δpurse share).
   - The data therefore suggests the policies are **approximately fixed functions meeting a new state and price distribution**, not adapting to opponents. How much of this comes from training distribution and how much from the observation cannot be separated here.
3. **Amplifier: population pressure.**
   - Moving from one opponent seat (C1) to four (C4) adds degradation, but sublinearly: the C4/C1 ratio is 1.06–1.70, never 4×.
   - The extra loss is disproportionate for aggressive opponents: −1.60 (D3QN), −1.29 (QR-DQN) and −1.01 (A2C) XI, against −0.34 (ES) and −0.22 (PPO).
   - The escalation is associated with room-level price inflation (Spearman −0.49) and with RL opponents' early spending (−0.48).
   - PPO copies instead back off: they leave 10–12% of their purse unspent and buy late depth, including keepers.
4. **The response to losing early decides how much is lost.**
   - **PPO** holds its purse (92% left at 10% progress in S4, against 39% in Stage A) and buys its stars and keeper late and cheaply. It loses about 2 XI and keeps 100% strong XIs.
   - **ES and A2C** keep bidding about 1.25× fair value on nearly every lot. They fill their squads with marginal and overseas players (ES in S4: squad 24.9, 8 overseas in 100% of rooms, 18% of purse stranded) and lose the most.
   - **D3QN and QR-DQN** bid at base on many lots and pay up on what they win. They end in between.
5. **Keeper failures (the safety-relevant edge case) come from two distinct mechanisms, not one.** Rival "depth buying" of keepers is present in every case.
   - **M1 (14 cases, all D3QN/QR-DQN; 12 involve `qrdqn:s2`).** This fits the proposed hypothesis in all 14. The seat passed on cheap keepers while the shield said SAFE (all 14 saw at least one passed keeper go unsold). It spent its purse by mid-auction; the requirement turned CRITICAL only in the re-auction. Its forced bids, with ≤ ₹500L, were lost, mostly to *another copy of the same export* (in 8 of the 10 QR-DQN opponent-seat cases) or to PPO seats. No viable keeper remained.
   - **M2 (6 cases, A2C and ES learners against four PPO seats).** This does **not** fit the hypothesis's first step. The purse and the 8 overseas slots were exhausted within the first quarter, so the requirement was never comfortably SAFE with purse to spend. The learner then made 2–16 keeper bids (mostly forced, already in the main round) with ≤ ₹80L and lost every one to rivals holding ₹7,000–47,000L.

**The minimum set of problems Stage B design must address** (section O):
1. early star competition and the policies' lack of response to it;
2. price-level and population-pressure robustness;
3. squad and overseas saturation with stranded purse;
4. keeper completion under rival depth-buying (a completion-shield blind spot);
5. seed-specific fragility.

This report diagnoses only; no change is prescribed.

---

## B. Data integrity

The full check is in `data-integrity.json`. **Result: PASS.**

| Check | Result |
|---|---|
| Phase 2E.0 result files | All present: 79 files hashed (report JSON/MD, plots, 20 finding traces, 4 raw JSONL + metadata). Phase 2E.0 recorded no hashes of its result files, so these sha256 values are the Phase 2E.1 baseline; they were re-verified at the end of this phase. |
| Episode counts | A 7,500 · C1 90,000 · C4 90,000 · S4 7,500 = 195,000, exactly as expected |
| Cells | A 15, C1 180, C4 180, S4 15; every one of the 390 cells has exactly 500 episodes. 0 duplicates. |
| Fields | 0 missing required fields. The only nulls are price/fair and cap/fair when a learner bought nothing or never bid; these are excluded from means, as in Phase 2E.0. |
| The 20 findings | Raw records hold exactly the 20 published findings: all C4, all screen-passed, identical to `findings.json` |
| Stage A control | Reproduces every stored Stage A episode (7,500/7,500) and the reported means ± sd exactly |
| Raw vs published | All 45 transfer cells recomputed from the raw records match `matchup-matrix.json` (maximum difference 8.9e-16). All 195,000 rows of `episode-results.json` match the raw XI. |
| Frozen system | `hashes.mjs --check`: obs `629b25783f833af7`, act `5f72f510c48b1f46`, 25 frozen files and 15 exports and checkpoints unchanged, frozen sources clean against git HEAD |

**What the raw records do not contain**, and how it was obtained:
- per-lot trajectories (purse, squad, stars and requirements over progress);
- role-level purchases by every team;
- per-decision shield and planner state.

These were obtained by **read-only reproduction** (`ml/ipl_rl/diagnosis/replay.mjs`). The replays built each room exactly as the Phase 2E.0 harness did and aborted on any digest mismatch; there were none.

| Replay | Episodes |
|---|---|
| Stage A | 7,500 |
| S4 | 7,500 |
| C1 (s1 × s1: all 20 ordered pairs × 500 auctions) | 10,000 |
| C4 (s1 × s1) | 10,000 |
| The 20 finding rooms, per-decision | 20 |

**Not obtainable from existing data:**
- opponent-seat shield state (the production runtime doesn't expose it);
- counterfactual policy behaviour under modified observations (would need a new experiment).

---

## C. Methods

- **Pairing.** Every Stage B episode is paired with the same export's Stage A episode on the same auction entry. Δ = Stage B − Stage A.
- **Confidence intervals.** 95%, percentile bootstrap (2,000 resamples, fixed seed) over the 500 auction entries.
- **Correlations.** Pearson and Spearman, computed on paired episode-level changes or on cell means. These are associations only.
- **Same-lot adaptation test.** For each learner and auction, the learner's decision on the *same player* in the main round is compared between Stage A and Stage B.
- **Trajectories.** Measured at 10% progress bins of the main round, plus the end of the re-auction.
- **Tools.** The scripts are in `ml/ipl_rl/diagnosis/`:

| Script | Role |
|---|---|
| `integrity.py` | Data-integrity check |
| `extract.py` | Flattens the raw records |
| `replay.mjs` | Read-only, digest-verified replays |
| `analyse.py` | Analyses A–L |
| `synthesis.py` | Analyses M–N, built from the analysis JSON |
| `plots.py` | Plots |

---

## D. Analysis A: transfer decomposition

The full data is in `transfer-analysis.json`. Values are the Stage A mean → C1 / C4 / S4 means, with the S4 relative change in brackets.

| Metric | PPO | A2C | D3QN | QR-DQN | ES |
|---|---|---|---|---|---|
| Best XI | 92.6 → 90.6 / 90.3 / 90.2 (−3%) | 90.4 → 87.6 / 86.7 / 85.3 (−6%) | 91.5 → 88.9 / 88.2 / 86.7 (−5%) | 91.9 → 89.4 / 88.6 / 87.5 (−5%) | 91.1 → 87.7 / 86.0 / 84.2 (−8%) |
| Strong XI rate | 1.00 → 1.00 / 1.00 / 1.00 | 1.00 → 0.985 / 0.833 / 0.611 | 1.00 → 0.994 / 0.956 / 0.891 | 1.00 → 1.00 / 0.985 / 0.989 | 1.00 → 0.990 / 0.698 / 0.337 |
| Legal XI rate | 1 / 1 / 1 / 1 | 1 / 1 / 0.9998 / 1 | 1 / 1 / 1 / 1 | 1 / 1 / 0.9998 / 1 | 1 / 1 / 0.9998 / 1 |
| Rank (of 10) | 1.00 → 1.24 / 1.17 / 1.19 | 1.19 → 5.62 / 5.80 / 8.45 | 1.00 → 2.88 / 3.65 / 6.45 | 1.00 → 2.10 / 3.11 / 5.15 | 1.09 → 5.69 / 6.60 / 9.02 |
| Stars | 10.7 → 8.6 / 8.4 / 8.8 (−18%) | 9.1 → 6.6 / 5.4 / 3.3 (−63%) | 9.9 → 7.8 / 7.3 / 6.2 (−37%) | 10.2 → 7.9 / 6.9 / 6.0 (−41%) | 10.1 → 6.7 / 4.5 / 1.9 (−81%) |
| Price/fair | 0.79 → 0.86 / 0.85 / 0.86 (+9%) | 1.11 → 1.19 / 1.19 / 1.26 (+13%) | 1.09 → 1.18 / 1.18 / 1.25 (+14%) | 0.98 → 1.04 / 1.08 / 1.14 (+16%) | 1.05 → 1.09 / 1.11 / 1.13 (+8%) |
| Cap/fair | 0.75 → 0.79 / 0.81 / 0.81 (+8%) | 1.05 → 1.11 / 1.17 / 1.21 (+15%) | 1.07 → 0.96 / 0.97 / 0.96 (−11%) | 0.97 → 0.89 / 0.92 / 0.91 (−6%) | 1.09 → 1.16 / 1.26 / 1.28 (+17%) |
| Bid rate | 0.44 → 0.43 / 0.46 / 0.45 | 0.84 → 0.85 / 0.89 / 0.92 | 0.79 → 0.81 / 0.83 / 0.80 | 0.65 → 0.73 / 0.76 / 0.79 | 0.86 → 0.88 / 0.95 / 1.00 |
| Squad size | 16.6 → 16.7 / 18.3 / 18.2 | 15.7 → 18.5 / 21.2 / 23.1 | 16.5 → 15.3 / 16.6 / 15.9 | 16.2 → 15.9 / 16.7 / 16.0 | 15.8 → 20.3 / 23.1 / 24.9 |
| Overseas | 6.3 → 5.9 / 6.2 / 6.3 | 5.2 → 6.3 / 7.0 / 7.7 | 5.1 → 4.8 / 5.2 / 5.4 | 5.5 → 5.8 / 6.0 / 6.2 | 5.8 → 7.4 / 7.7 / 8.0 |
| Purse left (share, absolute) | 0.001 → 0.002 / 0.013 / 0.007 | 0.000 → 0.001 / 0.030 / 0.022 | 0.001 → 0.001 / 0.005 / 0.001 | 0.001 → 0.001 / 0.003 / 0.001 | 0.000 → 0.005 / 0.119 / 0.178 |
| Total spend (share) | 0.999 → 0.998 / 0.987 / 0.994 | 1.000 → 0.999 / 0.970 / 0.978 | 0.999 → 0.999 / 0.995 / 0.999 | 0.999 → 0.999 / 0.997 / 0.999 | 1.000 → 0.995 / 0.881 / 0.822 |
| XI gain / ₹1000L | 8.49 → 8.32 / 8.37 / 8.30 | 8.29 → 8.04 / 8.21 / 7.97 | 8.38 → 8.14 / 8.12 / 7.94 | 8.42 → 8.19 / 8.13 / 8.01 | 8.35 → 8.10 / 9.14 / 9.49 |
| Marginal purchases | 0.18 → 0.77 / 1.34 / 1.29 | 0.51 → 1.19 / 2.15 / 2.69 | 0.29 → 0.71 / 0.97 / 0.90 | 0.30 → 0.69 / 1.00 / 0.94 | 0.62 → 1.72 / 3.08 / 4.03 |
| Re-auction purchases | 0.05 → 0.25 / 0.53 / 0.57 | 0.00 → 0.00 / 0.00 / 0.00 | 0.16 → 0.15 / 0.10 / 0.15 | 0.24 → 0.14 / 0.09 / 0.10 | 0.04 → 0.04 / 0.03 / 0.00 |
| Keeper completion (mean progress) | 0.09 → 0.24 / 0.38 / 0.57 | 0.22 → 0.53 / 0.60 / 0.74 | 0.07 → 0.17 / 0.25 / 0.40 | 0.07 → 0.11 / 0.15 / 0.20 | 0.21 → 0.60 / 0.70 / 0.77 |
| Indian completion (mean progress) | 0.23 → 0.45 / 0.51 / 0.56 | 0.06 → 0.11 / 0.10 / 0.11 | 0.08 → 0.15 / 0.19 / 0.29 | 0.11 → 0.22 / 0.24 / 0.32 | 0.11 → 0.23 / 0.26 / 0.32 |
| Shield activations / episode | 0 → 0.002 / 0.005 / 0.001 | 0.25 → 0.71 / 0.51 / 0.56 | 0.015 → 0.12 / 0.19 / 0.43 | 0.035 → 0.11 / 0.16 / 0.29 | 0.23 → 0.53 / 0.23 / 0.07 |
| Forced keeper bids / episode | 0 → 0.001 / 0.003 / 0 | 0.25 → 0.69 / 0.50 / 0.56 | 0.013 → 0.09 / 0.16 / 0.34 | 0.034 → 0.08 / 0.14 / 0.19 | 0.23 → 0.52 / 0.22 / 0.07 |
| Final-path forced bids (total) | 0 → 0 / 0 / 0 | 0 → 0 / 8 / 0 | 0 → 0 / 3 / 0 | 0 → 3 / 12 / 1 | 0 → 0 / 4 / 0 |

**Which changes correlate most with the XI change?** Spearman correlation between the paired change in each metric and ΔXI across all 187,500 Stage B episodes. Rank is omitted because it is an outcome.

| Δ metric | All | PPO | A2C | D3QN | QR-DQN | ES |
|---|---|---|---|---|---|---|
| Stars | **+0.73** | +0.44 | +0.79 | +0.74 | +0.76 | +0.75 |
| Squad size | −0.42 | −0.16 | −0.69 | −0.16 | −0.17 | −0.60 |
| Marginal purchases | −0.41 | −0.24 | −0.55 | −0.23 | −0.23 | −0.59 |
| Overseas | −0.40 | −0.11 | −0.55 | −0.19 | −0.20 | −0.42 |
| Contested lots | −0.34 | −0.25 | −0.51 | −0.44 | −0.38 | −0.43 |
| XI gain / ₹1000L | +0.32 | +0.49 | +0.30 | +0.63 | +0.63 | −0.29 |
| Total spend | +0.30 | +0.11 | +0.42 | +0.11 | +0.05 | +0.62 |
| Keeper completion (later) | −0.30 | −0.19 | −0.29 | −0.24 | −0.28 | −0.20 |
| Indian completion (later) | −0.28 | −0.33 | −0.49 | −0.56 | −0.47 | −0.49 |

Across the 45 cells, the strongest correlates of transfer Δ are:

| Cell-mean change | Pearson r |
|---|---|
| Δstars | +0.96 |
| Δmarginal buys | −0.83 |
| Δoverseas | −0.76 |
| Δpurse left | −0.73 |
| Δsquad size | −0.70 |
| Δbid rate | −0.68 |
| Opponent price/fair | −0.51 |
| Shield activations and forced bids | +0.06 |

Shield activations are nearly uncorrelated at cell level: shield dependence co-occurs with the loss but does not track its size.

**Reading.**
- The XI loss is mostly a **quality** loss: fewer stars, substituted with marginal and overseas players.
- It is not a spending loss: total spend is essentially unchanged except for ES and A2C in C4/S4.
- It is not a legality loss: 10 in 187,500 learner episodes.

---

## E. Analysis B: opponent effect (C1 learner × opponent matrix)

The full data is in `opponent-effects.json`, with every requested variable per ordered pair. Columns below are: ΔXI; learner XI; opponent XI; learner and opponent price/fair; learner and opponent bid rate; learner shield activations per episode; learner keeper requirements closed by a forced bid per 500 episodes; incomplete XIs (learner/opponent).

| Learner vs opponent | ΔXI | L XI | O XI | L p/f | O p/f | L bid | O bid | L shield | L forced keeper /500 | Incomplete L/O |
|---|---|---|---|---|---|---|---|---|---|---|
| PPO vs A2C | −1.65 | 90.95 | 88.60 | 0.86 | 1.14 | 0.44 | 0.83 | 0.001 | 0.0 | 0/0 |
| PPO vs D3QN | −2.14 | 90.45 | 89.88 | 0.86 | 1.12 | 0.42 | 0.80 | 0.003 | 0.4 | 0/0 |
| PPO vs QR-DQN | −2.25 | 90.34 | 90.35 | 0.86 | 1.00 | 0.41 | 0.69 | 0.002 | 0.2 | 0/0 |
| PPO vs ES | −1.74 | 90.86 | 88.67 | 0.86 | 1.08 | 0.45 | 0.86 | 0.002 | 0.0 | 0/0 |
| A2C vs PPO | −1.86 | 88.57 | 90.95 | 1.14 | 0.86 | 0.83 | 0.44 | 0.766 | 171.6 | 0/0 |
| A2C vs D3QN | −3.47 | 86.95 | 88.54 | 1.21 | 1.22 | 0.88 | 0.81 | 0.708 | 159.2 | 0/0 |
| A2C vs QR-DQN | −3.42 | 87.00 | 89.08 | 1.20 | 1.05 | 0.87 | 0.74 | 0.816 | 179.4 | 0/0 |
| A2C vs ES | −2.40 | 88.03 | 87.95 | 1.22 | 1.10 | 0.84 | 0.88 | 0.539 | 114.8 | 0/0 |
| D3QN vs PPO | −1.64 | 89.85 | 90.45 | 1.13 | 0.85 | 0.80 | 0.41 | 0.105 | 25.9 | 0/0 |
| D3QN vs A2C | −2.95 | 88.54 | 86.95 | 1.22 | 1.21 | 0.82 | 0.88 | 0.115 | 15.7 | 0/0 |
| D3QN vs QR-DQN | −3.56 | 87.93 | 88.52 | 1.18 | 1.05 | 0.81 | 0.73 | 0.207 | 41.6 | 0/0 |
| D3QN vs ES | −2.23 | 89.26 | 87.15 | 1.20 | 1.08 | 0.82 | 0.90 | 0.038 | 7.0 | 0/0 |
| QR-DQN vs PPO | −1.52 | 90.33 | 90.35 | 1.00 | 0.85 | 0.69 | 0.41 | 0.085 | 21.2 | 0/0 |
| QR-DQN vs A2C | −2.77 | 89.08 | 87.01 | 1.05 | 1.20 | 0.75 | 0.87 | 0.089 | 13.3 | 0/0 |
| QR-DQN vs D3QN | −3.31 | 88.54 | 87.93 | 1.05 | 1.19 | 0.73 | 0.81 | 0.191 | 42.9 | 0/0 |
| QR-DQN vs ES | −2.22 | 89.63 | 87.07 | 1.04 | 1.10 | 0.74 | 0.89 | 0.070 | 12.0 | 0/0 |
| ES vs PPO | −2.44 | 88.63 | 90.86 | 1.08 | 0.86 | 0.86 | 0.45 | 0.672 | 150.9 | 0/0 |
| ES vs A2C | −3.10 | 87.97 | 88.02 | 1.11 | 1.22 | 0.88 | 0.83 | 0.431 | 90.3 | 0/0 |
| ES vs D3QN | −3.93 | 87.14 | 89.24 | 1.08 | 1.20 | 0.90 | 0.82 | 0.516 | 115.1 | 0/0 |
| ES vs QR-DQN | −4.00 | 87.07 | 89.62 | 1.10 | 1.04 | 0.90 | 0.73 | 0.517 | 111.8 | 0/0 |

- **Purse.** Learner and opponent purse left are ≤ 0.9% in every C1 cell (see the JSON).
- **Opponent shield usage** is not observable. The production runtime doesn't expose it; the reverse pairing gives each algorithm's own shield profile.

**Decomposition.**
- A purely additive learner + opponent model explains **96.6%** of the variance of the 20 C1 transfer cells: the opponent alone 54%, the learner alone 36%, interaction 3.4%.
- The opponent is therefore the larger single factor, and matchup-specific interactions are small.

**Mean transfer associated with each opponent, and its style:**

| Opponent | Mean transfer | Bid rate | Cap/fair | Price/fair |
|---|---|---|---|---|
| PPO | −1.87 | 0.43 | 0.79 | 0.86 |
| ES | −2.15 | 0.88 | 1.16 | 1.09 |
| A2C | −2.62 | 0.85 | 1.11 | 1.19 |
| D3QN | −3.21 | 0.81 | 0.96 | 1.18 |
| QR-DQN | −3.31 | 0.72 | 0.89 | 1.04 |

- Across cells, the damage is associated with the opponent's **price paid** (Pearson −0.51) more than its bid rate (−0.36). Opponent cap/fair shows no association (+0.06).
- The value-based opponents are **not** the most frequent bidders, but they are associated with the largest losses.

  **Reading.** This is consistent with them winning the *contested high-value* lots (the stars the learner also wants) rather than bidding on everything. That fits the star-loss mechanism; a direct lot-level contest analysis would be needed to establish it.

---

## F. Analysis C: C1 → C4 escalation

The full data is in `c1-c4-analysis.json`.

| Learner \ opponent | PPO | A2C | D3QN | QR-DQN | ES |
|---|---|---|---|---|---|
| **PPO** | — | −0.62 | −0.33 | −0.13 | −0.42 |
| **A2C** | −0.13 | — | −1.69 | −1.37 | −0.53 |
| **D3QN** | −0.27 | −0.83 | — | −1.51 | −0.17 |
| **QR-DQN** | −0.34 | −0.86 | −1.65 | — | −0.25 |
| **ES** | −0.14 | −1.71 | **−2.75** | −2.14 | — |

Values are escalation = C4 transfer − C1 transfer, in XI. All 20 CIs exclude 0; the heatmap is `plots/04_escalation_C4_minus_C1.svg`.

- **Sublinear.** The C4/C1 ratio is 1.06–1.70 (a linear 4× effect would be ≈ 4.0).
- **By opponent:** D3QN −1.60, QR-DQN −1.29, A2C −1.01, ES −0.34, PPO −0.22.
- **By learner:** ES −1.69, A2C −0.93, QR-DQN −0.77, D3QN −0.69, PPO −0.37.

**Mechanism variables.** These come from the digest-verified s1 × s1 replays; each value is the mean over the four learners facing that opponent (A → C1 → C4).

| Opponent (as 4 copies) | Room price/fair | Keepers bought by RL opponents | Stars by 0.3 (RL opponents) | RL-opponent spend by 0.3 (sum of shares) | Learner stars by 0.3 | Learner first-keeper progress |
|---|---|---|---|---|---|---|
| PPO | 0.995 → 0.993 → **0.902** | 0 → 2.1 → **19.2** | 0 → 6.0 → 8.9 | 0 → 0.58 → 1.05 | 9.7 → 8.6 → 8.5 | 0.17 → 0.42 → 0.53 |
| A2C | 0.988 → 1.023 → 1.123 | 0 → 1.6 → 4.7 | 0 → 5.8 → 13.4 | 0 → 0.99 → 3.28 | 10.1 → 7.9 → 5.6 | 0.11 → 0.12 → 0.15 |
| D3QN | 0.986 → 1.025 → 1.068 | 0 → 1.4 → 5.4 | 0 → 7.8 → 16.4 | 0 → 0.99 → 3.51 | 9.9 → 6.1 → 2.7 | 0.17 → 0.45 → 0.62 |
| QR-DQN | 0.988 → 1.007 → 1.061 | 0 → 2.1 → 5.7 | 0 → 8.3 → 16.7 | 0 → 0.97 → 3.66 | 9.8 → 5.7 → 2.6 | 0.17 → 0.55 → 0.70 |
| ES | 0.989 → 1.016 → 1.076 | 0 → 1.5 → 4.6 | 0 → 7.2 → 13.5 | 0 → 0.99 → 2.91 | 9.8 → 6.7 → 5.0 | 0.13 → 0.23 → 0.21 |

**Reading: this is population pressure, not simply "one strong opponent".**
- Four copies of an aggressive, early-spending style spend almost their whole purses on stars by 30% progress (3.3–3.7 of 4 purses). That is associated with room-wide price inflation and with the learner's early stars collapsing (about 2.6–2.7 against D3QN/QR-DQN copies).
- Four PPO copies do the opposite: room price/fair *falls* to 0.90, they spend only about 1.05 of 4 purses by 30%, and they absorb **19 keepers**.
- Keeper, Indian and overseas competition rise most with PPO copies. That is associated with *less* star-driven escalation (Spearman +0.70 for the rise in RL-opponent Indian purchases) but with the keeper failures in section H.
- Overseas and bowling purchases by RL opponents rise 4–5× in C4 for every opponent. Their association with escalation is mixed (Spearman +0.45 and +0.36, i.e. rooms with more such purchases escalate *less*).

---

## G. Analysis D: S4 mixed-room decomposition

The full data is in `s4-analysis.json`.

**Per-seat behaviour.** These are the same across learners to within about 0.1, because every S4 room contains all four other algorithms. Values are means over all 7,500 S4 rooms.

| Opponent seat | XI | Rank | Price/fair | Bid rate | Stars (by 0.3 progress) | Keepers (by 0.3) | Keeper price/fair | Spend share by 0.3 | Purse left at end |
|---|---|---|---|---|---|---|---|---|---|
| PPO | 90.2 | 1.2 | 0.86 | 0.45 | 8.8 (1.5) | 3.4 (0.29) | 0.61 | 0.18 | 0.6% |
| A2C | 85.3 | 8.4 | **1.26** | 0.92 | 3.4 (3.3) | 1.2 (0.07) | 0.83 | 0.94 | 2.2% |
| D3QN | 86.7 | 6.4 | 1.25 | 0.81 | 6.2 (**6.2**) | 1.3 (0.57) | 1.21 | **0.98** | 0.1% |
| QR-DQN | 87.5 | 5.1 | 1.13 | 0.79 | 6.0 (5.9) | 1.4 (**0.89**) | 1.31 | 0.95 | 0.1% |
| ES | 84.3 | 9.0 | 1.13 | 1.00 | 1.9 (1.7) | 1.0 (0.00) | 0.94 | 0.61 | **17%** |

For each learner, which opponent seat recorded the extreme on each measure:

| Learner | Highest price/fair | Most keepers by 0.3 | Most spend by 0.3 | Most stars by 0.3 |
|---|---|---|---|---|
| PPO | A2C | QR-DQN | D3QN | D3QN |
| A2C | D3QN | QR-DQN | D3QN | D3QN |
| D3QN | A2C | QR-DQN | QR-DQN | QR-DQN |
| QR-DQN | A2C | D3QN | D3QN | D3QN |
| ES | A2C | QR-DQN | D3QN | D3QN |

- **Early pressure (stars and spend by 0.3)** is concentrated in the D3QN and QR-DQN seats.
- **The highest prices** are paid by A2C (and D3QN).
- **Early keeper purchases** are concentrated in QR-DQN and D3QN.
- **Total keeper depth** is concentrated in PPO (3.4 keepers per room, cheaply and late).
- **Purse depletion by 0.3** occurs in the D3QN, QR-DQN and A2C seats. ES strands purse at the end because its squad is full.
- **Late requirement failures:** none occurred in S4.

**Attribution limits.** All four opponents are present in every room, so the learner's loss cannot be attributed to one opponent. Within-learner associations between an opponent seat's stars and the learner's ΔXI are weak (|ρ| ≤ 0.33; the largest is the A2C seat's stars vs the ES learner, ρ −0.30). This is consistent with a pooled market effect rather than one dominant rival.

---

## H. Analysis F: keeper-failure forensics (all 20 findings)

The full per-finding reconstruction is in `keeper-failures.json`: purse, squad and overseas trajectories; every decision with the shield's keeper analysis (viable candidates, rival demand, contestants), maxSafeBid, max rival purse, the action, the result; and every keeper lot. The chart is `plots/15_keeper_failures.svg`.

| Seed | Stratum | Room | Affected (type) | Purse at 0.25 → 0.5 → end | Squad end | Overseas cap reached at | Keeper lots / legal bid / passed while SAFE (unsold) | Forced keeper bids (phase, final path) | Mechanism |
|---|---|---|---|---|---|---|---|---|---|
| 100490 | high | ppo:s2 vs 4× d3qn:s1 | d3qn:s1 (opp) | 10,180 → 130 → 50 | 22 | 0.33 | 65 / 25 / 17 (12) | 2 (RA, final) lost at ₹50 | M1 |
| 100027 | high | ppo:s2 vs 4× d3qn:s3 | d3qn:s3 (opp) | 920 → 100 → 50 | 17 | — (7) | 64 / 26 / 17 (12) | 2 (RA) lost at ₹50 | M1 |
| 100244 | high | ppo:s2 vs 4× qrdqn:s2 | qrdqn:s2 (opp) | 6,050 → 900 → 500 | 24 | 0.38 | 66 / 36 / 22 (16) | 3 (RA) lost at ₹90–500 | M1 |
| 100285 | high | ppo:s2 vs 4× qrdqn:s2 | qrdqn:s2 (opp) | 6,650 → 150 → 50 | 20 | 0.38 | 68 / 30 / 21 (13) | 2 (RA) lost at ₹50–60 | M1 |
| 100025 | high | d3qn:s1 vs 4× qrdqn:s2 | qrdqn:s2 (opp) | 8,350 → 90 → 50 | 21 | 0.43 | 70 / 29 / 19 (17) | 3 (RA) lost | M1 |
| 100034 | high | d3qn:s1 vs 4× qrdqn:s2 | qrdqn:s2 (opp) | 13,150 → 170 → 60 | 20 | 0.42 | 65 / 24 / 14 (10) | 3 (RA) lost | M1 |
| 100142 | high | d3qn:s1 vs 4× qrdqn:s2 | qrdqn:s2 (opp) | 8,900 → 250 → 50 | 16 | 0.42 | 72 / 28 / 19 (13) | 3 (RA) lost | M1 |
| 100262 | high | d3qn:s1 vs 4× qrdqn:s2 | qrdqn:s2 (opp) | 11,450 → 290 → 80 | 24 | 0.35 | 66 / 26 / 16 (13) | 4 (RA) lost | M1 |
| 100132 | high | es:s1 vs 4× qrdqn:s2 | qrdqn:s2 (opp) | 12,650 → 170 → 50 | 21 | 0.48 | 70 / 28 / 18 (15) | 4 (RA) lost | M1 |
| 100278 | high | es:s2 vs 4× qrdqn:s2 | qrdqn:s2 (opp) | 12,250 → 490 → 40 | 24 | 0.46 | 69 / 27 / 20 (12) | 1 (RA, final) lost at ₹40 | M1 |
| 100253 | high | qrdqn:s2 vs 4× ppo:s1 | qrdqn:s2 (learner) | 1,800 → 440 → 280 | 25 | 0.39 | 61 / 31 / 23 (8) | 2 (RA) lost at ₹200–300 | M1 |
| 100275 | high | qrdqn:s2 vs 4× ppo:s1 | qrdqn:s2 (learner) | 2,150 → 100 → 0 | 23 | 0.40 | 58 / 21 / 13 (5) | 2 (RA) lost | M1 |
| 100424 | high | qrdqn:s2 vs 4× ppo:s2 | qrdqn:s2 (learner) | 1,600 → 400 → 180 | 24 | 0.26 | 62 / 28 / 20 (9) | 2 (RA) lost at ₹180 | M1 |
| 100433 | high | qrdqn:s2 vs 4× ppo:s2 | qrdqn:s2 (learner) | 780 → 430 → 50 | 24 | 0.27 | 71 / 28 / 20 (17) | 2 (RA) lost | M1 |
| 100308 | high | es:s1 vs 4× ppo:s3 | es:s1 (learner) | **120** → 90 → 0 | 22 | **0.13** | 58 / 12 / 0 | 9 (main) lost at ₹30–40 | M2 |
| 100077 | normal | es:s2 vs 4× ppo:s3 | es:s2 (learner) | **150** → 80 → 10 | 18 | 0.39 | 55 / 16 / 1 | 2 (main) lost at ₹80 | M2 |
| 100180 | high | es:s3 vs 4× ppo:s3 | es:s3 (learner) | **100** → 60 → 0 | 20 | 0.24 | 58 / 12 / 0 | 9 (main) lost | M2 |
| 100342 | high | a2c:s1 vs 4× ppo:s3 | a2c:s1 (learner) | **120** → 90 → 0 | 22 | 0.17 | 58 / 12 / 0 | 9 (main) lost | M2 |
| 100445 | high | a2c:s1 vs 4× ppo:s3 | a2c:s1 (learner) | **500** → 220 → 10 | 25 | 0.17 | 55 / 16 / 0 | 2 (main) lost | M2 |
| 100308 | high | a2c:s2 vs 4× ppo:s3 | a2c:s2 (learner) | **210** → 60 → 0 | 23 | 0.27 | 57 / 12 / 0 | 9 (main) lost | M2 |

Purse values are in ₹L; RA = re-auction.

**Testing the proposed hypothesis step by step:**

| Step | Findings where it holds |
|---|---|
| H1 Requirement SAFE while the seat passed on keeper lots | 15 / 20 (all 14 M1 cases, plus 1 M2 case with a single pass; all 14 M1 cases saw ≥ 1 passed keeper go unsold) |
| H2 Keeper requirement later reached WARNING or CRITICAL | 20 / 20 |
| H3 Shield forced ≥ 1 keeper bid | 20 / 20 |
| H4 Every forced keeper bid lost | 20 / 20 |
| H5 The winning side had more purchasing power (maximum rival purse > the seat's cap) | 20 / 20 |
| H6 No viable keeper left after the final forced bid (the shield's own viable-candidate count = 0) | 20 / 20 |
| All six steps | **15 / 20** |

**Conclusion: two mechanisms, and the hypothesis fits only M1.**

- **M1 (14 cases: D3QN ×2 opponent seats, QR-DQN ×8 opponent seats, QR-DQN ×4 learners; all involve `qrdqn:s2` except the two D3QN cases).** The proposed hypothesis holds in every one.
  - The seat passed on 13–23 keeper lots while SAFE (5–17 of them went unsold), and spent most of a large purse by 50% progress.
  - The requirement stayed SAFE through the main round because many keepers were still to come.
  - In the re-auction it became CRITICAL. The forced bid (the seat's whole remaining purse, ₹40–500L) lost at the same or a higher price, and the shield then reported IMPOSSIBLE. The winner was another copy of the *same* export in 8 of the 10 QR-DQN opponent-seat cases: four identical copies that had all deferred their keepers then competed for the last few. Otherwise it was a PPO seat (the learner, or the four PPO opponents of the QR-DQN learners).
- **M2 (6 cases: A2C ×3 and ES ×3 learners, all against four `ppo:s3`/PPO seats).** The hypothesis's first step fails.
  - The learner spent **> 99%** of a large purse and reached the **8-overseas cap within 13–39% progress**. It never passed on keepers while SAFE: it had 0–1 such passes, and a legal bid on only 12–16 keeper lots.
  - The shield then **forced 2–9 keeper bids already in the main round**, several while its state was still SAFE or WARNING (purse-fragility forcing, `keeper(purse)`). Each was made with ₹30–80L, and every one was lost, mostly to the PPO seats, to rivals holding ₹7,000–47,000L.
- **Common to both.**
  - Rivals ending with 3 or more keepers held **12–24 keepers** in every finding room, on the depth-buying side. In M2 it is the four PPO seats (3–8 keepers each).
  - The shield's forcing rule counts rivals that *need* the requirement, and these depth-buyers did not need keepers.

---

## I. Analysis E: policy behaviour shift

The full data is in `behavior-shifts.json`, with the per-algorithm table: bid rate, price/fair, cap/fair, stars, marginal, squad, purse left, overseas, re-auction purchases, requirement timings, shield activations, forced bids, decisions (A / C1 / C4 / S4 and the change). The chart is `plots/14_behaviour_comparison.svg`.

**Largest shifts, S4 vs A:**

| Algorithm | Largest shifts |
|---|---|
| ES | Stars −81%; squad +57%; purse left +17.8 points; pass share 0.20 → 0.005; FV_1.25 share of bids 62% → 86% |
| A2C | Stars −63%; squad +47%; overseas +50%; forced keeper bids 0.25 → 0.56 per episode |
| D3QN | Forced bids ×29; keeper completion 0.07 → 0.40; stars −37% |
| QR-DQN | Forced bids ×8; bid rate +21%; stars −41%; top action FV_0.70 → BASE |
| PPO | Keeper completion 0.09 → 0.57; re-auction purchases 0.05 → 0.57; stars −18%; bid rate and cap nearly unchanged |

**Does the policy become more aggressive, or stay fixed while the distribution changes?** This is the same-lot test: the same player in the same auction, main round, decided in both conditions.

| Learner | Similar own purse (\|Δpurse share\| < 0.05): mean Δcap/fair; identical cap | Different purse: mean Δcap/fair (C1 / C4 / S4) | Spearman(Δcap, Δpurse) (C1 / C4 / S4) |
|---|---|---|---|
| PPO | C1 −0.004 (78%) · C4 +0.002 (86%) · S4 −0.023 (80%) | +0.14 / +0.24 / +0.14 | 0.30 / 0.42 / 0.26 |
| A2C | +0.049 (57%) · +0.006 (67%) · +0.114 (62%) | +0.27 / +0.33 / +0.38 | 0.43 / 0.44 / 0.33 |
| D3QN | +0.046 (70%) · +0.084 (66%) · +0.052 (71%) | +0.06 / +0.10 / +0.08 | 0.25 / 0.24 / 0.12 |
| QR-DQN | +0.059 (68%) · +0.063 (67%) · +0.084 (50%) | +0.13 / +0.17 / +0.21 | 0.28 / 0.30 / 0.21 |
| ES | +0.048 (64%) · −0.021 (69%) · +0.012 (95%) | +0.16 / +0.22 / +0.29 | 0.29 / 0.31 / 0.04 |

**Reading.**
- For PPO and ES the offer on the same player at the same purse is essentially unchanged.
- Most of the average cap increase in Stage B comes from **having more purse later**, because the early lots were lost.
- A2C, D3QN and QR-DQN show a modest response to other state features (+0.05 to +0.11).
- The data therefore suggests the degradation is primarily **a frozen policy meeting a different state and price distribution**, not an adversarial adaptation.
- Whether the missing adaptation stems from the training distribution (Stage A never showed contested stars) or from the observation (rival features not informative enough) **cannot be determined from these data**.

---

## J. Analysis G: shield dependence

The full data is in `shield-analysis.json`.

| Algorithm | Forced bids / episode (A → C1 / C4 / S4) | Episodes with a forced bid | Forced-bid win rate (A → S4) | Episodes with any non-SAFE state (S4) |
|---|---|---|---|---|
| PPO | 0 → 0.002 / 0.005 / 0.001 | 0% → 0.1–0.3% | — → 100% (2 bids) | 0.07% |
| A2C | 0.25 → 0.71 / 0.51 / 0.56 | 11% → 31% / 20% / 23% | 46% → 41% | 13% |
| D3QN | 0.015 → 0.12 / 0.19 / 0.43 | 1% → 5% / 7% / 17% | 68% → 42% | 11% |
| QR-DQN | 0.035 → 0.11 / 0.16 / 0.29 | 2% → 5% / 7% / 10% | 55% → 37% | 6% |
| ES | 0.23 → 0.53 / 0.23 / 0.07 | 11% → 23% / 10% / 3% | 47% → 40% | 1.4% |

**Final-path forced bids.** There were **0 in Stage A** (7,500 control episodes) and **31 in Stage B** (in 31 distinct episodes, 21 of them won).

| Cell | Final-path bids | Won |
|---|---|---|
| C4 QR-DQN vs PPO | 9 | 5 |
| C4 A2C vs PPO | 8 | 5 |
| C4 ES vs PPO | 4 | 1 |
| C4 QR-DQN vs D3QN | 3 | 3 |
| C1 QR-DQN vs ES | 2 | 2 |
| C4 D3QN vs PPO | 2 | 2 |
| C1 QR-DQN vs PPO | 1 | 1 |
| C4 D3QN vs QR-DQN | 1 | 1 |
| S4 QR-DQN | 1 | 1 |

- 23 of the 31 are against four PPO seats, and 24 of the 31 are in the high-purse stratum.
- **Increasing reliance:** D3QN (×29 in S4) and QR-DQN (×8) rely on the shield increasingly under learned opponents. A2C relies on it at a high level throughout. ES's reliance *falls* in S4 because it keeps its purse and meets its keeper need in time or not at all. PPO stays near zero.
- **Association with loss:** at cell level, shield activations are barely associated with the size of the XI loss (Pearson +0.06). The shield is working as a completion backstop, not as the source of the loss.

---

## K. Analysis H: purse dynamics

The full data is in `purse-analysis.json`; the charts are `plots/07_purse_over_progress.svg` and `plots/08_squad_over_progress.svg`.

**Purse left (share) at 10% / 30% / 50% progress, Stage A → S4:**

| Algorithm | Stage A | S4 |
|---|---|---|
| PPO | 0.39 / 0.12 / 0.02 | **0.92 / 0.82 / 0.24** |
| A2C | 0.04 / 0.00 / 0.00 | 0.49 / 0.06 / 0.03 |
| D3QN | 0.08 / 0.01 / 0.00 | 0.31 / 0.02 / 0.01 |
| QR-DQN | 0.15 / 0.01 / 0.00 | 0.37 / 0.04 / 0.01 |
| ES | 0.09 / 0.00 / 0.00 | 0.94 / 0.40 / 0.19 |

**Squad size at 50% progress, Stage A → S4:** PPO 13.6 → 9.2 · A2C 14.9 → 20.9 · D3QN 15.8 → 13.8 · QR-DQN 15.1 → 13.9 · ES 15.2 → 23.7.

**Spending vs final XI.** Spearman correlations within each algorithm and condition; the rank-based quintile tables are in the JSON.

| Algorithm | Early spend (by 0.3) ↔ XI, A / S4 | Stars by 0.3 ↔ XI, A / S4 | Price/fair ↔ XI, A / S4 | Marginal buys ↔ XI, A / S4 |
|---|---|---|---|---|
| PPO | +0.34 / **−0.17** | +0.41 / −0.23 | +0.02 / −0.26 | −0.25 / −0.17 |
| A2C | +0.28 / +0.50 | +0.52 / +0.71 | −0.22 / −0.54 | −0.34 / −0.46 |
| D3QN | −0.03 / +0.07 | +0.70 / +0.39 | −0.39 / −0.21 | −0.29 / −0.17 |
| QR-DQN | +0.22 / −0.17 | +0.58 / +0.39 | −0.11 / −0.41 | −0.27 / +0.02 |
| ES | +0.39 / **+0.67** | +0.45 / +0.64 | −0.17 / +0.10 | −0.51 / −0.17 |

**Reading.** "Spending less is better" is **not** supported.
- For ES and A2C, *more* early spending is associated with a *higher* XI in S4. When they win early lots, they win stars.
- For PPO in S4 the association reverses: its best S4 episodes spend least early (quintile XI 90.43 at 0% early spend vs 89.92 at 44%).
- High price/fair is associated with a lower XI for A2C and QR-DQN.
- Marginal purchases are associated with a lower XI for every algorithm in Stage A and for all but QR-DQN in S4.
- **What matters, in these data, is what the early spending buys**, not the spending level.

---

## L. Analysis I: requirement timing

The full data is in `requirement-analysis.json`; the charts are `plots/09_keeper_completion.svg` and `plots/09_indians_completion.svg`. Values are the mean progress at completion, with the median in brackets, A → S4.

| Algorithm | Keeper | Bowling | Indian | XI slots (11th player, 0.1 bins) | 4th overseas | Overseas at the cap (8) |
|---|---|---|---|---|---|---|
| PPO | 0.09 (0.04) → 0.57 (0.77) | 0.25 → 0.43 | 0.23 → 0.56 | 0.4 → 0.6 | 0.4 → 0.4 | 38% → 42% |
| A2C | 0.22 (0.04) → 0.74 (0.78) | 0.07 → 0.14 | 0.06 → 0.11 | 0.1 → 0.2 | 0.1 → 0.2 | 7% → 87% |
| D3QN | 0.07 (0.03) → 0.40 (0.06) | 0.08 → 0.16 | 0.08 → 0.29 | 0.1 → 0.3 | 0.1 → 0.3 | 13% → 20% |
| QR-DQN | 0.07 (0.03) → 0.20 (0.05) | 0.12 → 0.22 | 0.11 → 0.32 | 0.2 → 0.4 | 0.2 → 0.3 | 19% → 34% |
| ES | 0.21 (0.05) → 0.77 (0.77) | 0.07 → 0.18 | 0.11 → 0.32 | 0.1 → 0.3 | 0.1 → 0.2 | 15% → 100% |

**Learned opponents systematically delay requirement completion for every algorithm and requirement.** The size of the delay depends on the learner:
- **PPO's delay** is deliberate-looking. It defers its keeper to late and cheap buys and is never forced.
- **A2C's and ES's delays** come with purse exhaustion and forced closures: 23% of A2C S4 keepers were closed by a forced bid.
- **D3QN and QR-DQN** keep an early median but grow a late tail: 16% and 9% forced closures in S4, and 2.5–2.6% closed in the re-auction.

---

## M. Analysis J: robustness by purse stratum

The full data is in `stratum-analysis.json`; the chart is `plots/12_degradation_by_stratum.svg`.

| Stratum | Stage B episodes | Mean transfer | Forced bids | Final-path | Findings | Stage A XI |
|---|---|---|---|---|---|---|
| Low | 68,250 | −3.18 | 18,621 | 2 | 0 | 91.42 |
| Normal | 61,875 | −3.18 | 16,169 | 5 | 1 | 91.51 |
| High | 57,375 | −3.05 | 13,309 | **24** | **19** | 91.54 |

- **Degradation is not concentrated in a stratum.** The mean loss is almost identical across strata. For PPO, D3QN and QR-DQN it is slightly *smaller* in the high stratum (e.g. S4 PPO −2.63 / −2.40 / −2.22). ES is the exception, losing more in normal and high (S4 −6.45 / −7.11 / −6.98).
- **The rare legality-relevant failures are concentrated in the high stratum:** 19 of 20 findings and 24 of 31 final-path forced bids.
- **Why high purse is not "easier":** the traces suggest large purses let the policies (and their copies) buy many players early. They reach the overseas cap early (M1 cases by 0.26–0.48 progress) and exhaust even a very large purse by mid-auction. With a large purse the *absolute* stakes of the early spree are larger, and the planner's reserve is small relative to what rivals hold late.

---

## N. Analysis K: seed stability

The full data is in `seed-stability.json`; the chart is `plots/13_seed_variability.svg`. C1 and C4 figures are the standard deviation of transfer Δ over the 9 seed pairings of each cell, as a mean with the maximum in brackets.

| Algorithm | Stage A seed sd (XI) | C1 | C4 | S4 seed sd (transfer) |
|---|---|---|---|---|
| PPO | 0.08 | 0.18 (0.20) | 0.36 (0.45) | 0.37 |
| A2C | 0.05 | 0.65 (0.85) | 0.85 (1.23) | 0.79 |
| D3QN | 0.19 | 0.54 (0.77) | 0.69 (1.23) | 0.82 |
| QR-DQN | 0.09 | 0.44 (0.71) | 0.56 (0.87) | 0.24 |
| ES | 0.04 | 0.37 (0.59) | 0.90 (1.51) | 0.78 |

- **Stage B seed variability is high for every algorithm:** the seed-pair sd of transfer Δ is 2–20× the Stage A seed sd of XI. Stage A seed stability does not predict Stage B stability: ES is the most seed-stable in Stage A and among the least in C4.
- **Most stable degradation:** PPO in every composition, and QR-DQN in S4.
- **High variance:** ES vs D3QN in C4 (1.51), A2C vs ES in C4 (1.23), D3QN vs A2C in C4 (1.23).
- **Source of variance.** Across C1 and C4 cells, 56% of seed-pair variance comes from the **opponent's** seed and 37% from the learner's. PPO s2 as an opponent is associated with the most damage (C1 mean −2.41 vs −1.47 / −1.71).
- **Seed-specific failures:** `qrdqn:s2` is involved in 12 of the 20 findings. Every learner/opponent cell's 9 seed pairings are negative (360/360).

---

## O. Analyses L–N: failure modes, algorithm diagnosis, Stage B requirements

### L. Dominant failure modes

The full data is in `failure-modes.json`. Every Stage B episode is compared with its paired Stage A episode. The categories overlap: an episode averages 2.9 categories, and the loss grows with the number of categories (0 → −1.01, 3 → −3.17, 5 → −5.02, 9 → −6.01). Only 96 episodes with ΔXI ≤ −3 fall into no category.

| Category (paired criterion) | Episodes (share of Stage B) | Mean ΔXI affected vs not | Most affected | Top matchups |
|---|---|---|---|---|
| Star loss (≥ 2 fewer stars) | 144,702 (77%) | **−3.60** vs −1.61 | All; S4 A2C/D3QN/QR/ES ≥ 98% | C4 ES vs D3QN, ES vs QR-DQN, QR vs D3QN |
| Price inflation (price/fair +0.10 or more) | 77,558 (41%) | −3.42 vs −2.95 | S4 A2C/D3QN/QR (65–69%) | C4 and C1 D3QN vs A2C, C4 D3QN vs ES |
| Purse exhausted with a requirement unmet (min purse while unmet < 5%) | 73,970 (39%) | −3.47 vs −2.93 | D3QN (47–65%) | C4 and C1 D3QN vs QR-DQN |
| Requirement delay (keeper ≥ 0.2 later, or in the re-auction) | 66,508 (35%) | −4.03 vs −2.65 | ES and A2C (46–76%), PPO in S4 (66%) | C4 PPO vs QR-DQN, ES vs D3QN/QR |
| Overseas saturation (8 overseas, fewer in the pair) | 63,058 (34%) | −4.29 vs −2.56 | ES (62–85%), A2C S4 (81%) | C4 A2C vs D3QN/QR, ES vs QR |
| Marginal purchasing (≥ 2 more marginal buys) | 55,207 (29%) | −4.26 vs −2.68 | ES S4 (84%), A2C S4 (62%) | C4 ES vs QR/D3QN, A2C vs D3QN |
| Keeper competition (forced, final-path or not completed) | 20,033 (11%) | −3.35 vs −3.12 | A2C (20–31%), ES C1 (23%) | C1 A2C vs QR, C4 A2C vs PPO |
| Shield dependence (forced, none in the pair) | 18,167 (10%) | −3.52 vs −3.10 | A2C (18–27%), ES C1 (20%) | C4 A2C vs PPO, C1 A2C vs QR/PPO |
| Squad saturation (squad ≥ 24 with ≥ 5% purse unspent) | 15,374 (8%) | **−6.30** vs −2.86 | ES S4 (79%), ES C4 (49%), A2C (15–18%) | C4 ES vs QR/D3QN, A2C vs D3QN |
| Re-auction effects (more re-auction buys, or forced in the re-auction) | 11,091 (6%) | −2.68 vs −3.17 (*smaller* loss) | PPO (9–20%) | C4 PPO vs QR/D3QN |
| Other (ΔXI ≤ −3, none of the above) | 96 (0.05%) | — | — | — |

**Representative example.** C4, `es:s3` vs four `d3qn:s2`, seed 100,260: ΔXI −13.7. This single episode falls into star loss, price inflation, requirement delay, marginal purchasing, overseas saturation and squad saturation at once.

**Confidence.** The categories are paired and counted exactly. Their association with ΔXI is correlational; section Q covers causality.

### M. Algorithm-specific diagnosis

The full data is in `algorithm-diagnosis.json`, with an evidence block per algorithm.

- **PPO: why it transfers best.** It bids least and lowest (bid rate about 0.44, cap/fair about 0.75–0.81 everywhere), so it rarely wins early lots that learned opponents push above fair value.
  - It keeps 92% of its purse through the first 10% of S4 lots (Stage A: 39%) and buys stars and keepers later and cheaply (keeper price/fair 0.61).
  - It meets requirements without the shield (≤ 0.005 forced per episode, 0 final-path).
  - Its offers are unchanged for the same player at the same purse: its Stage A style already avoids price wars.
  
  **What still fails:**
  - about 2 XI of star losses in every room;
  - four PPO copies leave 10–12% of their purse unspent and hoard keepers, which is associated with keeper scarcity for other learners (23 of 31 final-path forced bids and all 10 learner findings are in four-PPO rooms);
  - PPO s2 is the harshest opponent seed.
- **A2C: why it degrades.** Its Stage A high-bid style persists and intensifies: bid rate 0.84 → 0.92, cap/fair 1.05 → 1.21, FV_1.25 bids 21% → 31%.
  - Against opponents also bidding above fair value it wins fewer stars (9.1 → 3.3) and fills its squad with marginal and overseas players (squad 23.1, 87% at the overseas cap).
  - **Shield dependence** (0.25 → 0.56–0.71 forced per episode; win rate 46% → 40%) co-occurs with the loss but is not associated with its size across cells (r = +0.06). It looks like a symptom of early purse exhaustion, not the cause.
- **D3QN: why it degrades.** Its greedy argmax most often bids BASE (36–44% of bids): it loses contested lots and pays 1.25× for what it wins in S4. Stars fall 9.9 → 6.2, and forced bids rise ×29.
  - **Greedy selection weakness.** It is consistent with, but not proof of, a greedy-selection weakness that D3QN copies produce the largest escalation (−1.60).
  - **Why D3QN itself fails keeper completion.** Its two findings (both as an opponent seat) are mechanism M1: it passed on 17 cheap keepers while SAFE and lost re-auction forced bids with ₹50L.
- **QR-DQN: does distributional modelling help?** Relative to D3QN it degrades slightly less in every composition (C1 −2.46 vs −2.60, C4 −3.23 vs −3.29, S4 −4.34 vs −4.82). It keeps a much higher strong-XI rate in S4 (98.9% vs 89.1%) and lower S4 seed variance (0.24 vs 0.82). Its keeper completes earlier than D3QN's in S4 (0.20 vs 0.40).
  - **Cannot be attributed to the quantile head:** the two exports differ in more than the head.
  - **`qrdqn:s2`** alone accounts for 12 of the 20 findings (M1). This is consistent with its Stage A profile (the most shield-forced keeper purchases of the three QR-DQN seeds), so it is a **seed-specific**, not algorithm-wide, fragility.
- **OpenAI-ES: why it degrades most.** It is the near-fixed 1.25× fair-value rule: FV_1.25 makes up 62% of its bids in Stage A and 86% in S4, and its pass share falls from 20% to 0.5%.
  - Against opponents paying more than 1.25× for stars it loses the early lots: stars by 0.3 progress fall 10.1 → 1.7. It then wins nearly everything else at 1.25× until the squad is full (24.9 players, 8 overseas in 100% of S4 rooms, 18% of purse stranded).
  - Its early spend correlates *positively* with XI (ρ +0.67): the problem is losing the early contests, not spending early.
  - **Vulnerability.** In this setup ES is the most vulnerable learner, with the largest loss everywhere and the largest escalation (−1.69). This describes this frozen export and its training setup, **not** evolution strategies in general.

### N. What Stage B training needs to address

The full data is in `stage-b-requirements.json`. This is diagnosis only; nothing is prescribed.

| # | Problem | Evidence (summary) | Algorithms | Matchups | Severity | Apparent type |
|---|---|---|---|---|---|---|
| 1 | Early star contests are lost to learned opponents | Δstars is the strongest correlate of ΔXI (ρ 0.73; cell r 0.96); 77% of episodes lose ≥ 2 stars (−3.60 vs −1.61) | All | All; worst vs D3QN/QR-DQN, C4 and S4 | High | Training distribution; opponent modelling |
| 2 | Bidding rules barely respond to opponent pressure | Same lot, same purse: Δcap/fair about 0 for PPO and ES; offers track own purse only | ES, A2C (harmless for PPO) | All | High | Training distribution; observation (**unknown**) |
| 3 | Population pressure from identical aggressive opponents | C4 − C1: D3QN −1.60, QR −1.29, A2C −1.01; associated with room price inflation (ρ −0.49) and early RL spend (−0.48) | All | C4 vs D3QN/QR/A2C; S4 | Medium–high | Training distribution; opponent modelling |
| 4 | Squad and overseas saturation with stranded purse | 15,374 episodes (−6.30 XI); ES S4 squad 24.9 with 18% purse left | ES, A2C | C4 vs value-based; S4 | High for ES/A2C | Purse management; reward (**unknown**); training distribution |
| 5 | Keeper completion under rival depth-buying | 20 findings, M1 (14) and M2 (6); rivals held 12–24 keepers; final-path forced bids 0 → 31; forced win rates fall | QR-DQN, D3QN, A2C, ES | C4, high purse, vs 4× PPO, `qrdqn:s2` | Rare (20 / 195,000) but a legality failure | Completion/shield; purse management; training distribution |
| 6 | Later requirement completion and rising shield use | Keeper completion 0.07–0.22 → 0.20–0.77; D3QN forced ×29, QR ×8 | D3QN, QR-DQN, A2C, ES | C4, S4 | Medium | Purse management; completion/shield; training distribution |
| 7 | Seed-specific fragility | `qrdqn:s2` in 12 of 20 findings; opponent seed explains 56% of seed-pair variance; PPO s2 the harshest | QR-DQN, PPO | Specific exports | Medium | Training distribution; unknown |

**Questions for Stage B design** (open, not decisions):
1. Should a learner contest early stars, or learn PPO's later-value allocation? Both appear viable in these data.
2. Is the insensitivity in #2 a training-distribution effect or an observation limitation? This needs a counterfactual probe.
3. Does the XI-delta reward give any signal about squad slots and overseas slots as resources (#4)?
4. Should completion safety account for rivals' depth buying and for the seat's ability to contest (#5)? This concerns the frozen act-v3 shield and needs an explicit decision.
5. Which opponent mix reproduces population pressure (#3) without collapsing to one style?
6. How many seeds per opponent are needed for reliable evaluation (#7)?

---

## P. Required visualisations

All plots are in `plots/`, with `index.html` showing them together, and all are generated from the Phase 2E.0 data or its verified replays.

| # | Chart | File |
|---|---|---|
| 1 | Stage A → C1 → C4 → S4 XI | `01_xi_by_condition.svg` |
| 2 | Transfer Δ heatmaps | `02_transfer_C1.svg`, `02_transfer_C4.svg` |
| 3 | Opponent-induced degradation (C1), with column means and variance decomposition | `03_opponent_degradation_C1.svg` |
| 4 | C1 → C4 escalation | `04_escalation_C4_minus_C1.svg` |
| 5 | Bid-rate shift | `05_bid_rate_shift.svg` |
| 6 | Price/fair shift | `06_price_fair_shift.svg` |
| 7 | Purse over progress (A vs S4) | `07_purse_over_progress.svg` |
| 8 | Squad over progress | `08_squad_over_progress.svg` |
| 9 | Requirement timing | `09_keeper_completion.svg`, `09_indians_completion.svg` |
| 10 | Shield-forced bids by condition | `10_shield_forced_per_episode.svg` |
| 11 | Forced keeper purchases | `11_forced_keeper_purchases.svg` |
| 12 | Degradation by stratum | `12_degradation_by_stratum.svg` |
| 13 | Seed variability | `13_seed_variability.svg` |
| 14 | Behaviour comparison | `14_behaviour_comparison.svg` |
| 15 | The 20 keeper-failure timelines | `15_keeper_failures.svg` |

---

## Q. What the data establishes, suggests, and cannot determine

**Established** (direct measurement on the frozen experiment):
- Every export loses XI against learned opponents in every composition.
- The loss is larger with four copies of an aggressive opponent than with one, but far less than 4×.
- The opponent identity explains more of the C1 loss than the learner identity, and the matrix is about 97% additive.
- Learners buy far fewer stars early, complete requirements later, and pay more per player.
- The 20 keeper failures follow two distinguishable sequences (M1, M2), reconstructed decision by decision.
- For the same player at a similar own purse, PPO's and ES's offers are nearly unchanged.
- Final-path forced bids appear only in Stage B, 24 of 31 in the high stratum.

**Suggested** (consistent associations; not causal):
- Star loss is the main pathway from opponent pressure to XI loss.
- The value-based and A2C opponents cause loss by winning contested high-value lots and inflating prices.
- PPO's robustness comes from its low-cap, purse-preserving style rather than from adapting.
- ES's and A2C's saturation stems from a fixed high-cap rule applied after losing early lots.
- Rival depth-buying of keepers is associated with the completion failures.

**Cannot be determined without new experiments:**
- Whether any policy *could* have done better against these opponents: needs best-response or counterfactual training.
- Whether the lack of adaptation is caused by the training distribution or by the observation: needs perturbation probes of the frozen policies or controlled training.
- The individual causal contribution of each opponent in S4: all four are always present.
- Whether reward, shield or observation changes would help: that is intervention, which is out of scope.
- Distributional (QR-DQN) vs dueling (D3QN) effects: confounded by other differences between the exports.

**Composition confound.** C1 vs Stage A changes the opponent *and* removes one rule fallback at the same time, so the transfer Δ bundles "new RL opponent" with "one fewer predictable rule bot".

---

## R. Limitations

1. **The C1/C4 trajectory replays cover the s1 × s1 subset** (20,000 of 180,000 episodes). All other C1/C4 statistics use all 180,000 raw records.
2. **Opponent-seat shield state and per-decision opponent observations** were not recorded in Phase 2E.0 and are unavailable, except in the 20 finding rooms, where the affected opponent seats were traced by replay.
3. **Progress bins are 0.1 wide.** Requirement-completion times for keeper, bowling and Indian players are exact (planner events); XI-slot and overseas times are at bin resolution.
4. **Validation manifest only (500 auctions).** The test split is untouched.
5. **All correlations are associations.** Several variables are mechanically linked (squad size, marginal buys, overseas).
6. **Findings are few (20).** The mechanism classification is exact for these 20 but may not cover rarer paths at larger scale.

---

## S. Files and reproducibility

- **Report data:**
  - `data-integrity.json`
  - `transfer-analysis.json`
  - `opponent-effects.json`
  - `c1-c4-analysis.json`
  - `s4-analysis.json`
  - `behavior-shifts.json`
  - `keeper-failures.json`
  - `shield-analysis.json`
  - `purse-analysis.json`
  - `requirement-analysis.json`
  - `stratum-analysis.json`
  - `seed-stability.json`
  - `failure-modes.json`
  - `algorithm-diagnosis.json`
  - `stage-b-requirements.json`
  - `plots/`
- **`raw/`:** the replay log, keeper traces (per decision), the final integrity re-check, and the script list.
- **Scripts:** `ml/ipl_rl/diagnosis/` (new files only).
- **Large intermediates, git-ignored:** `ml/runs/_2e1/` (flattened records `episodes.npz`; trajectory replays `traj_{A,S4,C1,C4}.jsonl`, about 240 MB).
- **To reproduce:**
  1. `extract.py`
  2. `replay_all.sh`
  3. `replay.mjs keeper …`
  4. `analyse.py ml/reports/phase2e1`
  5. `synthesis.py ml/reports/phase2e1`
  6. `plots.py ml/reports/phase2e1`
  7. `integrity.py`
- **Nothing was committed, trained or modified.**
