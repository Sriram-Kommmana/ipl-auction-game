# Phase 2D.4 — OpenAI-ES × 3 seeds × Stage A × act-v3

Final algorithm of the Phase-2D matrix. Frozen environment contract (IplAuctionEnv-v2, obs-v2 `629b25783f833af7`, act-v3 `5f72f510c48b1f46`, reward ΔBestXI/110 with −2 for an empty XI slot, γ = 1, Stage-A opponents, locked baselines) — nothing frozen was modified. All numbers below come from `analysis.txt`, `audit_report.txt`, `final_parity.txt`, `fallback.txt`, `latency_*.json` and `smoke_reproducibility.txt` in this folder.

**Headline.** All three seeds completed the frozen budget (2,000 generations, ≈ 10.1–10.9 M decisions each) with no safety stop, no instability stop and no implementation defect. Final (generation 2,000) validation XI **91.068 ± 0.044**; at equal decisions to the other algorithms (≈ 2.0 M) **90.728 ± 0.037**. Legal XI 24,000/24,000 over all 48 validation checkpoints. ES learns fast and is remarkably seed-stable, but plateaus early and finishes below PPO, QR-DQN and D3QN (even with ~5× their decisions) and above A2C.

## A. Pre-flight
- `ml/ipl_rl/tests/test_openai_es.py`: **15/15** — ES mathematics (deterministic Gaussian noise; antithetic ±σε symmetry; CRN `seed_plus == seed_minus` and identical action streams; centered ranks against hand examples incl. ties; the estimator against a hand calculation; 1/σ scaling; population averaging; zero-difference → zero gradient; tied pair → zero contribution; recovery of a known linear gradient; the first Adam ascent step; zero gradient → no change), the policy (80→64→64→20, tanh, no value head, 10,644 parameters, deterministic seeded init, seeds differ, numpy forward = network, masked categorical: masked probability exactly 0, renormalised, empirical frequencies, never a masked action; masked argmax with PASS-only / single / all masks), real environment (training seeds only — no validation/test seeds; scheduling independence: 1 simulator in order = 3 simulators in reverse order; CRN with σ = 0 gives identical episodes; a sampled ES episode equals an independent JavaScript replay — obs, masks, rewards, termination), the production contract (loads as `rl-policy-v2` `es`; JS/Python parity under all mask kinds incl. re-auction states).
- Full suites: Python **102/102** (all earlier tests incl. PPO/A2C/D3QN/QR-DQN, JS/Python observation/mask/reward parity), JavaScript **211/212** (known legacy skip; act-v3 hash, future-order leakage B3, reward parity K4, production O-tests).
- Production fallback on a real export: 10/10 before training.
- Throughput benchmark before training: one seed 795 decisions/s; three seeds concurrently 799 decisions/s in aggregate (no gain) → seeds run sequentially.
- Six-run reproducibility smoke test (§I): all pass.

**Decisions made before training (none changes a frozen hyperparameter):**
1. **Export identifier.** The production loader's identifier for this network is `es` (head `logits`, hidden `[64, 64]`, selection `argmax` — exactly the specified 80→64→64→20 masked-argmax policy). A file stamped `openai_es` is refused as an unknown algorithm and would silently fall back, so exports carry `algorithm: "es"` and `meta.algorithmName: "openai_es"`; the initialisation seed is derived from `("openai_es", seed)`.
2. **Training seed range.** The frozen split puts training seeds at ≥ 1,000,000 (not < 100,000 as the brief states); validation 100,000–100,499 and test 200,000–200,999 match. ES draws only training seeds (`episode_seed(seed, pair, generation)`) and consumes no validation or test seed (tested).
3. **Budget.** Measured cost ≈ 5,000 decisions per generation → 2,000 generations ≈ 10 M decisions per seed ≈ 3.5 h per seed. Feasible, so the frozen 2,000 generations were kept (not reduced). This is ~5× the other algorithms' 2 M decisions — see §H/§X.
4. **One episode per perturbation**, both members of a pair on the same auction with the same action stream.
5. **Centered ranks with average ties**, so that f+ = f− ⇒ zero contribution (the classic argsort ranks would give tied pairs a non-zero contribution).
6. **Checkpoint mapping (frozen before training):** validation at the first generation boundary at or after each of the eight decision checkpoints (245,760 … 1,996,800), plus every 250 generations; actual decision counts are reported.

## B. ES algorithm specification
Per generation g (frozen, implemented in `ml/ipl_rl/algos/openai_es.py`):
1. For pair i = 0…31: ε_i ~ N(0, I₁₀₆₄₄) from the stream ("openai_es/noise", seed, g, i).
2. Policies θ + σε_i and θ − σε_i (σ = 0.05) each play **one complete episode** on training auction `episode_seed(seed, i, g)` with action stream ("openai_es/act", seed, g, i) — common random numbers.
3. Fitness = episode return (frozen reward, γ = 1); every episode checked against the JavaScript summary and the safety invariants.
4. Centered ranks over the 64 fitnesses; estimator and Adam step (§E–G).
Training actions: categorical over legal actions (masked softmax). Evaluation/export: masked argmax. No value head, critic, replay, targets, quantiles, entropy bonus, elites or crossover.

## C. Network
PolicyNet("es"): 80 → 64 (tanh) → 64 (tanh) → 20 linear logits; 10,644 parameters; orthogonal init (√2 hidden, 0.01 head), zero biases, from init seed ("openai_es", seed). Initial digests: s1 `39aebfbffba6d801`, s2 `085641f777d3b132`, s3 `01352a387e871f8d` (all different).

## D. Antithetic sampling
32 pairs → 64 perturbed policies per generation; +ε and −ε of a pair differ only in parameters (same auction seed, same action stream — asserted every generation and in tests; the first noise vector is regenerated every generation and checked for determinism). Across training, 8.5 / 9.0 / 12.6 of the 32 pairs per generation (seed means) had exactly equal returns — the perturbation did not change a single sampled decision of that episode.

## E. Fitness shaping
Centered ranks over all 64 fitnesses: s = rank / 63 − 0.5, rank 0-based, ties receive the average rank (tested: [3,1,2] → [0.5, −0.5, 0]; [1,1,2,0] → [0, 0, 0.5, −0.5]; sums to 0; range [−0.5, 0.5]; invariant to monotone transforms). Validation XI is never used by the optimizer.

## F. Gradient estimator (exact)
ĝ = 1 / (2N·σ) · Σ_{i=1..N} (s_i⁺ − s_i⁻) ε_i, with N = 32 pairs (2N = 64 = population size in perturbations), s = the **centered-rank shaped** fitness (not the raw paired difference). Hand-checked: s⁺ = [0.5, −0.1], s⁻ = [−0.5, 0.1], ε = [[1,0,2],[0,1,−1]], σ = 0.5 → ĝ = [0.5, −0.1, 1.1].

## G. Optimizer
Adam ascent (θ.grad = −ĝ): lr 0.01, β 0.9/0.999, eps 1e-8, weight decay 0; one update per generation. Tested: first step = lr·ĝ/(|ĝ|+eps). Optimizer state checked finite every generation (0 failures).

## H. Training budget

| | seed 1 | seed 2 | seed 3 |
|---|---:|---:|---:|
| Generations | 2,000 | 2,000 | 2,000 |
| Environment decisions | 10,175,309 | 10,878,423 | 10,105,571 |
| Episodes (= perturbation-policy evaluations) | 128,000 | 128,000 | 128,000 |
| Optimizer updates | 2,000 | 2,000 | 2,000 |
| Decisions per generation (first → last) | 5,350 → 4,614 | 5,241 → 6,072 | 5,020 → 4,839 |

## I. Reproducibility
Six smoke runs (each seed twice, all six concurrently under CPU load; real population of 32 pairs, 12 simulators, 4 generations, decision- and generation-triggered checkpoints): **identical** initial parameters, perturbation vectors, perturbation/episode seeds, policy actions, fitness values, centered ranks, gradient estimates, optimizer states, parameter digests every generation, final parameters, training/episode statistics, checkpoint statistics + parity, validation episodes, exported weights — 60/60 checks; seeds differ (`smoke_reproducibility.txt`). Final digests of the long runs: s1 `ebb24136c7e621ce`, s2 `91d89e037311d02a`, s3 `10286b6e0b2904f6`.

## J. Checkpoint results (500 validation seeds, masked argmax; XI)

| checkpoint | seed 1 (gen, decisions) | seed 2 | seed 3 | mean ± std | PPO | QR-DQN | D3QN | A2C |
|---|---|---|---|---|---:|---:|---:|---:|
| ≥ 245,760 | 90.774 (49, 249,592) | 90.595 (51, 245,948) | 90.715 (51, 247,215) | 90.695 ± 0.091 | 91.428 | 89.825 | 89.756 | 86.232 |
| ≥ 497,664 | 90.773 (100) | 90.605 (111) | 90.875 (108) | 90.751 ± 0.137 | 91.859 | 90.472 | 90.556 | 86.971 |
| ≥ 749,568 | 90.775 (151) | 90.601 (169) | 90.772 (160) | 90.716 ± 0.100 | 92.045 | 90.970 | 90.473 | 88.221 |
| ≥ 1,001,472 | 90.775 (204) | 90.591 (228) | 90.754 (210) | 90.707 ± 0.101 | 92.209 | 91.245 | 90.441 | 89.323 |
| ≥ 1,247,232 | 90.926 (253) | 90.593 (285) | 90.761 (258) | 90.760 ± 0.166 | 92.385 | 91.659 | 91.261 | 89.569 |
| ≥ 1,499,136 | 90.778 (306) | 90.622 (344) | 90.762 (308) | 90.721 ± 0.086 | 92.452 | 91.830 | 89.847 | 90.208 |
| ≥ 1,751,040 | 90.865 (357) | 90.612 (404) | 90.777 (355) | 90.751 ± 0.128 | 92.513 | 91.903 | 91.392 | 90.219 |
| **≥ 1,996,800** | 90.748 (406, 2,000,117) | 90.685 (459, 1,997,105) | 90.751 (401, 2,000,822) | **90.728 ± 0.037** | 92.595 | 91.853 | 91.490 | 90.427 |
| gen 250 | 90.826 | 90.640 | 90.745 | 90.737 ± 0.093 | | | | |
| gen 500 | 90.808 | 90.762 | 90.745 | 90.772 ± 0.032 | | | | |
| gen 750 | 91.127 | 90.979 | 90.760 | 90.955 ± 0.184 | | | | |
| gen 1,000 | 91.117 | 91.093 | 91.044 | 91.085 ± 0.037 | | | | |
| gen 1,250 | 91.127 | 91.067 | 90.927 | 91.040 ± 0.103 | | | | |
| gen 1,500 | 91.161 | 91.006 | 90.963 | 91.044 ± 0.104 | | | | |
| gen 1,750 | 91.130 | 91.047 | 91.009 | 91.062 ± 0.062 | | | | |
| **gen 2,000** | 91.072 (10,175,309) | 91.110 (10,878,423) | 91.022 (10,105,571) | **91.068 ± 0.044** | | | | |

Legal XI 500/500 and strong XI 500/500 at **every** one of the 48 checkpoints (24,000/24,000 each).

## K. Final per-seed results (generation 2,000)

| | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| XI | 91.072 | 91.110 | 91.022 |
| legal / strong | 500/500 · 500/500 | 500/500 · 500/500 | 500/500 · 500/500 |
| purse left (₹L) | 3.70 | 7.16 | 3.76 |
| squad / overseas / stars | 15.54 / 5.44 / 10.00 | 15.69 / 5.55 / 10.11 | 16.23 / 6.29 / 10.14 |
| price/fair · cap/fair | 1.052 · 1.047 | 1.054 · 1.170 | 1.036 · 1.054 |
| bid rate | 0.957 | 0.674 | 0.941 |
| re-auction / marginal buys | 0.000 / 0.562 | 0.114 / 0.590 | 0.000 / 0.718 |
| XI gain per ₹1,000L · rank | 8.347 · 1.072 | 8.351 · 1.048 | 8.343 · 1.152 |
| shield-forced | 130/37,363 | 80/52,058 | 129/38,894 |

Mean ± std at generation 2,000: XI 91.068 ± 0.044 (CV 0.05 %), purse ₹4.9L ± 2.0, squad 15.82 ± 0.37, overseas 5.76 ± 0.46, stars 10.08 ± 0.07, price/fair 1.047 ± 0.010, bid rate 0.857 ± 0.159, re-auction buys 0.038 ± 0.066. Seed pairs: s1−s2 −0.038 [−0.104, +0.029], s1−s3 +0.050 [−0.031, +0.131], s2−s3 +0.088 [+0.021, +0.158].
At ≈ 2 M decisions: XI 90.728 ± 0.037 (CV 0.04 %); stars 10.02, price/fair 1.054, bid rate 0.875, marginal buys 0.714, rank 1.19.

## L–O. ES vs PPO / A2C / D3QN / QR-DQN (paired, same 500 validation seeds; seed-mean ΔXI [95 % CI])

| checkpoint (decisions) | vs PPO | vs QR-DQN | vs D3QN | vs A2C |
|---|---:|---:|---:|---:|
| 245,760 | −0.734 [−0.801, −0.665] | **+0.869** [+0.778, +0.959] | **+0.938** [+0.837, +1.037] | +4.462 |
| 497,664 | −1.108 | +0.279 [+0.212, +0.346] | +0.195 | +3.780 |
| 749,568 | −1.329 | −0.254 | +0.243 | +2.495 |
| 1,001,472 | −1.502 | −0.538 | +0.266 | +1.384 |
| 1,247,232 | −1.624 | −0.899 | −0.501 | +1.191 |
| 1,499,136 | −1.731 | −1.109 | +0.873 (D3QN dip) | +0.512 |
| 1,751,040 | −1.762 | −1.152 | −0.641 | +0.532 |
| **1,996,800 (equal decisions)** | **−1.867 [−1.943, −1.787]** | **−1.126 [−1.199, −1.047]** | **−0.762 [−0.838, −0.684]** | **+0.301 [+0.229, +0.375]** |
| **ES gen 2,000 (~10.4 M) vs their 2 M finals** | **−1.527 [−1.594, −1.458]** | **−0.785 [−0.853, −0.719]** | **−0.422 [−0.489, −0.357]** | **+0.641 [+0.578, +0.707]** |

ES gen 2,000 vs final, all nine seed pairs: vs PPO −1.44 … −1.67; vs QR-DQN −0.68 … −0.93; vs D3QN −0.22 … −0.68; vs A2C +0.54 … +0.71 — every CI excludes 0. W/L per matched seed: vs PPO 13/484, 4/493, 6/490; vs QR-DQN 104/363, 108/363, 92/390; vs D3QN 180/290, 126/332, 204/263; vs A2C 389/94, 353/129, 353/128. Strata (gen 2,000): vs PPO low −1.67 / normal −1.46 / high −1.44; ES is weakest relative to every algorithm in the low-purse stratum except vs A2C.
Secondary (gen 2,000, ES vs PPO / QR-DQN / D3QN / A2C finals): stars 10.08 vs 10.73 / 10.19 / 9.85 / 9.12; price/fair 1.047 vs 0.789 / 0.979 / 1.094 / 1.112; bid rate 0.857 vs 0.436 / 0.652 / 0.790 / 0.839; shield-forced 339/128,315 (0.26 %) vs 0 / 0.030 % / 0.016 % / 0.335 %; legal and strong XI identical (all 1,500/1,500).

## P. ES vs the eight locked baselines (unchanged file; seed-mean ΔXI [95 % CI], W/L per seed)

| baseline | ES gen 2,000 | ES at ≈ 2 M |
|---|---|---|
| Moneyball (87.464) | +3.604 [+3.507, +3.699] · 499/0, 498/2, 491/7 | +3.264 [+3.165, +3.367] |
| Star Chaser (88.632) | +2.436 [+2.337, +2.533] | +2.096 |
| Balanced Builder (88.147) | +2.921 [+2.828, +3.014] | +2.581 |
| Opportunist (87.848) | +3.220 [+3.128, +3.313] | +2.880 |
| Product fallback (88.158) | +2.910 [+2.814, +3.008] | +2.570 |
| Random legal (87.452) | +3.616 [+3.485, +3.750] | +3.275 |
| Fair-value (89.177) | +1.891 [+1.780, +1.997] · 458/31, 456/35, 451/40 | +1.550 [+1.439, +1.667] |
| Planner-greedy (87.708) | +3.360 [+3.166, +3.547] | +3.020 |

Every stratum positive for every baseline (per-stratum values in `analysis.txt`).

## Q. Purse / behaviour audit
Requirement audit: 24,000 replays of all 48 ES checkpoints + the 2 M finals of PPO, A2C, D3QN and QR-DQN — **30,000/30,000 identical to the evaluator**.
- Purse at generation 2,000: median ₹0 / ₹10L / ₹0; exact ₹0 in 315 / 209 / 315 of 500; ≤ ₹20L in 500 / 497 / 499. ES spends everything (like A2C).
- Style: the greedy policy is a near-fixed rule early on — at generation 49–100 (seed 1) ~70 % of validation decisions are **FV_1.25** ("bid up to 1.25 × fair value"), ~30 % BASE / low fair-value levels; price/fair ≈ 1.03–1.07 throughout, high bid rate (0.67–0.99), many marginal buys (0.56–0.76 vs PPO 0.18), stars ≈ 10.0–10.1.
- Keeper: bought while needed 1,500/1,500, all in the main round; median at ~4–5 % of the main round, but a late tail (90th percentile at **78–80 %** of the main round — A2C-like; PPO/D3QN/QR-DQN 5–7 %). Indian and bowling requirements met 1,500/1,500 in the main round; forced Indian purchases 0 at generation 2,000 (≤ 5 per 500 at earlier checkpoints), forced bowling purchases 0–1.

## R. Shield / feasibility audit
Shield-forced keeper purchases per 500 episodes (the keeper purse-fragility rule):

| | seed 1 | seed 2 | seed 3 |
|---|---:|---:|---:|
| PPO (2 M) | 0 | 0 | 0 |
| QR-DQN (2 M) | 5 | 19 | 5 |
| D3QN (2 M) | 5 | 6 | 4 |
| A2C (2 M) | 81 | 38 | 53 |
| **ES ≈ 2 M** | **79** | **51** | **85** |
| **ES gen 2,000** | **61** | **39** | **60** |

Episodes with purse ≤ ₹40L while a keeper was still needed follow the same pattern (ES gen 2,000: 61 / 39 / 60). Validation shield-forced decisions at generation 2,000: 130/37,363, 80/52,058, 129/38,894 (≈ 0.26 %). **Final-path forced bids: 0 in all 48 ES checkpoints.** Re-auction forced 0; CRITICAL 0–3 per checkpoint; IMPOSSIBLE 0 everywhere.
**Verdict:** ES does **not** learn keeper feasibility on its own — it leans on the act-v3 purse-fragility forcing at A2C's level (8–17 % of episodes), declining only slightly with 5× more decisions. It never depends on a final-chance bid, and the legal XI always held.

## S. Stability
- Seed-to-seed: the most seed-stable algorithm of the five at the final checkpoint (XI std 0.044; PPO 0.082, QR-DQN 0.089, A2C 0.046, D3QN 0.191).
- Checkpoint-to-checkpoint (paired, generation order): significant decreases 2/15, 2/15, 4/15 — all small (largest −0.148 [−0.20, −0.09], seed 1 gen 253 → 306; largest over the second half −0.117, seed 3 gen 1,000 → 1,250). Best checkpoints: s1 gen 1,500 (91.161; gen 2,000 is −0.089 [−0.159, −0.020] below it), s2 gen 2,000 (91.110), s3 gen 1,000 (91.044; gen 2,000 −0.022 [−0.087, +0.040]).
- Learning shape (all seeds): a fast rise to ≈ 90.6–90.9 within ~50 generations (~250 k decisions), a long plateau to ~500 generations, a small second step to ≈ 91.0–91.2 between generations 500 and 1,000, then flat to 2,000. Training fitness (sampled policies) rose 8.73–8.78 → 9.08–9.14 and flattened similarly; |θ| grew 16.0 → 48.9–51.4 with step/|θ| falling from 6.5e-2 to ~4.7e-3.

## T. Safety (exact counts)
- Training: **0/384,000** incomplete XI (128,000 per seed), 0 invariant violations, 384,000/384,000 episode consistency checks (return, decisions, action counts vs the JavaScript summary; played seed = scheduled seed), 0 illegal / masked actions reaching the simulator, 0 purse / squad / overseas / duplicate violations (part of the invariants), 0 NaN/Infinity in fitness, gradient, parameters or optimizer state, 0 pair-seed mismatches, 0 perturbation non-determinism, 0 crashes, 0 deadlocks.
- Validation: 48 checkpoints × 500 — legal 24,000/24,000, strong 24,000/24,000, 0 invariant violations, safety OK everywhere; obs-v2 / act-v3 hashes unchanged in every run and report; export parity passed at every checkpoint (max |Δlogit| ≤ 8.9e-15, agreement 100 %).
- `completionGuard` never enabled; no fallback used during evaluation; test split untouched.

## U. Export / parity
- Exports (`export_metadata.txt`): `rl-policy-v2`, `algorithm: "es"`, `meta.algorithmName: "openai_es"`, architecture `{input 80, hidden [64, 64], tanh, head logits, actions 20}`, selection argmax, obs-v2 `629b25783f833af7`, act-v3 `5f72f510c48b1f46`, seed, ES hyperparameters, training budget, parameter digest (s1 `ebb24136c7e621ce`, s2 `91d89e037311d02a`, s3 `10286b6e0b2904f6`); 225 kB each; no population, optimizer, noise or exploration state.
- Final-export parity (`final_parity.txt`): 1,005 real states from four JavaScript episodes, each classified by an independent JavaScript replay — normal 746, low purse 24, high purse 477, main 746, re-auction 259, keeper needed 570, Indian needed 914, bowling needed 911, shield-forced 17, final-path 3 (the frozen fixture) — under real / PASS-only / single-action / all-20-legal masks: **max |Δlogit| 1.1e-14, masked-argmax agreement 100 % in every category and mask kind**, production decisions legal 1,005/1,005 per seed.

## V. Production fallback
`fallback.txt`: **30/30** checks over the three final exports — healthy decisions are the masked argmax of the exported logits and legal act-v3 caps ≤ maxSafeBid; fallback to the frozen rule persona for missing model, malformed / truncated / non-JSON input, wrong architecture (hidden 128-128, head, selection, activation, action count), wrong output count (19 / 21), NaN parameter, Infinity parameter (in memory and as JSON → null), observation-hash mismatch, act-hash mismatch (incl. the real act-v2 hash), action-count mismatch, NaN output and Infinity output (weights corrupted after validation), masked action, out-of-range action, and a decision slower than 20 ms. Fallback behaviour unchanged.

## W. Throughput and latency
Training (per seed): wall 12,834 / 12,555 / 12,892 s (3.5 h; 10.6 h for three seeds) = episodes 11,985 / 11,738 / 12,044 s (of which waiting on the simulators 10,572 / 10,271 / 10,642 s, policy forward + sampling 867 / 906 / 853 s) + noise 9 s + ES updates 4.3 / 4.2 / 4.3 s (**2.1 ms per update**) + validation 802 / 771 / 802 s (16 checkpoints). Overall **793 / 866 / 784 decisions/s**, 9.9–10.2 episodes/s, 558–573 generations/h (PPO 894–898, A2C 588–696, D3QN 536–556, QR-DQN 437–523 decisions/s). ES is simulator-bound (≈ 82 % of wall time waiting on the auctions); lock-step is not applicable — every episode runs asynchronously and deterministically (1 vs 3 simulators identical, tested); three concurrent seeds gave no aggregate gain (799 vs 795 decisions/s).
Inference (`latency_*.json`, real exports, `seat.decide` = observation + planner + mask + forward + selection): single seat p50 0.30 ms, p95 0.66, p99 1.15, max 4.9 ms; nine seats deciding the same lot in turn: per decision p50 0.51, p99 3.3 ms; per lot (all nine) p50 5.1, p95 10.8, p99 22.2 ms. **Finding (not ES-specific):** rare single decisions exceed 20 ms — 0–10 per 17,820 timed decisions for ES, and likewise 0–3 per 17,820 for the PPO and QR-DQN exports (maxima 16–158 ms; GC / OS scheduling). Under the frozen runtime rule one such decision disables that RL seat for the rest of the room (fallback to the rule persona). Worth reviewing for production; not changed here.

## X. Optimization-budget disclosure
| | PPO | A2C | D3QN | QR-DQN | **OpenAI-ES** |
|---|---:|---:|---:|---:|---:|
| environment decisions / seed | 1,996,800 | 1,996,800 | 1,996,800 | 1,996,800 | **10.1–10.9 M** |
| episodes / seed | ~17–18 k | ~25–29 k | ~26–27 k | ~23–25 k | **128,000** |
| optimizer updates / seed | 31,200 | 325 | ~165.6 k | ~165.6 k | **2,000** |
| what each update uses | 256 fresh transitions | 6,144 fresh transitions | 256 replayed transitions | 256 replayed transitions | **64 episode returns (scalars)** |
| gradient source | backprop, clipped surrogate | backprop, A2C | backprop, TD (Huber) | backprop, quantile Huber | **none — black-box (finite differences via antithetic returns)** |
| parameters | 80-128-128-20 (+critic) | same | 80-128-128-(128,128) | 80-128-128-640 | **80-64-64-20** |

ES's 2,000 updates each consume 64 scalar returns from 64 complete episodes: 128,000 episodes and ≈ 10 M decisions per seed, about 5× the others' decisions, for 2,000 parameter updates. Equal decisions (the ≈ 2 M column in §L–O) is the fair sample-efficiency comparison; generation 2,000 is ES's own frozen budget.

## Y. Final interpretation
- **Algorithm performance.** ES learns a competitive policy very quickly — at 246 k decisions it is second only to PPO and ahead of QR-DQN (+0.87), D3QN (+0.94) and A2C (+4.46) — but then plateaus: at equal decisions (≈ 2 M) it is 4th of 5 (90.73; above A2C, below D3QN, QR-DQN, PPO), and with ~5× the decisions (generation 2,000) it reaches 91.07, still below PPO (−1.53), QR-DQN (−0.79) and D3QN (−0.42) at 2 M, above A2C (+0.64). It beats every locked baseline (fair-value +1.89).
- **Convergence / variance.** Exceptionally consistent across seeds (std 0.04; the same two-phase curve on every seed) and small checkpoint noise; the limitation is the ceiling, not instability.
- **Behavioural strategy.** The black-box search found — and largely stayed with — a near-fixed pricing rule (bid up to ~1.25× fair value on most lots, spend the purse), with a high bid rate and many marginal buys; state-dependent refinements (the kind PPO learns: patience, keeping purse, cheaper keepers) emerge only slowly. This differs clearly from the gradient-based and value-based methods.
- **Feasibility.** ES relies on the shield's keeper purse-fragility forcing at A2C's level (8–17 % of episodes), unlike PPO (0) and D3QN/QR-DQN (≈ 1–4 %); no final-path dependence; the legal XI always held.
- **Safety.** Clean: 0 incomplete XI in 384,000 training and 24,000 validation episodes, 0 illegal actions, 0 non-finite values.
- **Compute cost.** 10.6 h for three seeds; ~5× the decisions of the other algorithms for a lower result; cheap updates (2 ms), simulator-bound.
- **Production feasibility.** The export is the smallest network (10,644 parameters), with the fastest inference (p50 0.30 ms) and exact JS parity; production-valid. The rare > 20 ms tail-latency events are a runtime-wide property to review, not an ES defect.

No production selection is made; the test split is untouched. Final Phase-2D matrix (Stage A, validation XI at the final checkpoint): PPO 92.595 ± 0.082 · QR-DQN 91.853 ± 0.089 · D3QN 91.490 ± 0.191 · **OpenAI-ES 91.068 ± 0.044 (gen 2,000; 90.728 ± 0.037 at ≈ 2 M decisions)** · A2C 90.427 ± 0.046.
