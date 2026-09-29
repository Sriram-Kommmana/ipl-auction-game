"""Phase 2E.1 synthesis: algorithm-diagnosis.json and stage-b-requirements.json.
Every number is read from the analysis JSON files (nothing typed by hand);
the interpretation text is diagnosis only — no intervention is prescribed.

    python synthesis.py <report_dir>
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ALGOS = ["ppo", "a2c", "d3qn", "qrdqn", "es"]


def main(rep):
    rep = Path(rep)
    L = lambda n: json.loads((rep / f"{n}.json").read_text(encoding="utf-8"))
    TA, OE, CC, S4, BS, KF, SH, PA, RA, SA, SS, FM = (L(x) for x in ("transfer-analysis", "opponent-effects", "c1-c4-analysis", "s4-analysis", "behavior-shifts",
                                                                     "keeper-failures", "shield-analysis", "purse-analysis", "requirement-analysis", "stratum-analysis",
                                                                     "seed-stability", "failure-modes"))
    by = TA["byAlgorithm"]
    cur = PA["curves"]
    adapt = BS["sameLotAdaptation"]
    acts = BS["actionDistributions"]
    mech = KF["mechanisms"]

    def ev(a):
        """Common evidence block per algorithm."""
        return {
            "xi": {c: by[a]["xi"][c] for c in ("A", "C1", "C4", "S4")},
            "transfer": {c: by[a]["xi"][f"{c}_abs"] for c in ("C1", "C4", "S4")},
            "strongXI": {c: by[a]["strongXI"][c] for c in ("A", "C1", "C4", "S4")},
            "stars": {c: by[a]["stars"][c] for c in ("A", "C1", "C4", "S4")},
            "priceToFair": {c: by[a]["priceToFair"][c] for c in ("A", "C1", "C4", "S4")},
            "capToFair": {c: by[a]["capToFair"][c] for c in ("A", "C1", "C4", "S4")},
            "bidRate": {c: by[a]["bidRate"][c] for c in ("A", "C1", "C4", "S4")},
            "squadSize": {c: by[a]["squadSize"][c] for c in ("A", "C1", "C4", "S4")},
            "purseLeftAt0.1_A_vs_S4": [round(1 - cur["A"][a]["spend"][0], 3), round(1 - cur["S4"][a]["spend"][0], 3)],
            "purseLeftAt0.3_A_vs_S4": [round(1 - cur["A"][a]["spend"][2], 3), round(1 - cur["S4"][a]["spend"][2], 3)],
            "squadAt0.5_A_vs_S4": [round(cur["A"][a]["squad"][4], 2), round(cur["S4"][a]["squad"][4], 2)],
            "starsBy0.3_A_vs_S4": [round(cur["A"][a]["stars"][2], 2), round(cur["S4"][a]["stars"][2], 2)],
            "sameLotSimilarPurseDeltaCapFair": {c: adapt[c][a]["similarPurse(|Δpurse share|<0.05)"]["meanDeltaCapFair"] for c in adapt},
            "sameLotSimilarPurseIdenticalCap": {c: adapt[c][a]["similarPurse(|Δpurse share|<0.05)"]["identicalCapShare"] for c in adapt},
            "passShare": {c: acts[a][c]["passShare"] for c in acts[a]},
            "topActionShareOfBids": {c: [acts[a][c]["topAction"], acts[a][c]["topActionShareOfBids"]] for c in acts[a]},
            "forcedPerEpisode": {c: SH["byAlgorithm"][a][c]["forcedPerEpisode"] for c in ("A", "C1", "C4", "S4")},
            "forcedWinRate": {c: SH["byAlgorithm"][a][c]["forcedWinRate"] for c in ("A", "C1", "C4", "S4")},
            "keeperClosedByForcedPer500": {c: SH["byAlgorithm"][a][c]["keeperClosedByForcedBidPer500"] for c in ("A", "C1", "C4", "S4")},
            "finalPathForced": {c: SH["byAlgorithm"][a][c]["finalPathForcedTotal"] for c in ("A", "C1", "C4", "S4")},
            "keeperCompletionMean": {c: RA["fromRawRecords"][a][c]["keeper"]["mean"] for c in ("A", "C1", "C4", "S4")},
            "C1_learnerMainEffect": round(OE["learnerMainEffect"][a], 3), "C1_opponentMainEffect": round(OE["opponentMainEffect"][a], 3),
            "escalationAsLearner": CC["byLearner"][a], "escalationAsOpponent": CC["byOpponent"][a],
            "seedSdC1C4": [SS["C1"][a]["meanSd9"], SS["C4"][a]["meanSd9"]], "seedSdS4": SS["S4"][a]["sd"], "seedSdStageA": SS["stageA"][a]["sd"],
            "findings": {"asLearner": sum(1 for x in KF["findings"] if x["affected"].startswith(a + ":") and x["affectedType"] == "learner"),
                         "asOpponentSeat": sum(1 for x in KF["findings"] if x["affected"].startswith(a + ":") and x["affectedType"] == "rlSnapshot")},
        }

    diag = {
        "ppo": {
            "evidence": ev("ppo"),
            "whyItTransfersBest": [
                "It bids least and lowest: bid rate ≈ 0.43–0.46 and cap/fair ≈ 0.75–0.81 in every condition, so it rarely wins the early, contested star lots that RL opponents push above fair value.",
                "In S4 it keeps ~92% of its purse through the first 10% of lots (Stage A: ~39%) and buys its stars and keeper later and cheaply (keeper price/fair ≈ 0.61 as an S4 seat); its early-spend correlation with XI turns slightly negative in S4 (ρ ≈ −0.17).",
                "Its offer for the same player at a similar own purse is essentially unchanged between Stage A and Stage B (mean Δcap/fair ≈ 0, 78–86% identical caps) — the policy did not have to change; its Stage-A style already avoids price wars.",
                "It completes requirements without the shield in almost every episode (forced bids ≤ 0.005 per episode; 0 final-path forced bids).",
            ],
            "whatStillFails": [
                "It still loses ≈ 2 XI in every room: fewer stars (10.7 → 8.4–8.8), more marginal buys, a higher rank number (1.00 → 1.17–1.24).",
                "Four PPO copies (C4) compete with each other: they leave 10–12% of their purse unspent and hoard keepers (~4.8 each), and are associated with keeper scarcity for other learners — 23 of 31 final-path forced bids and all 10 learner incomplete-XI findings occurred in rooms with four PPO seats.",
                "Seed sensitivity as an opponent: PPO s2 is associated with more damage to others than s1 / s3 (C1 mean transfer caused −2.41 vs −1.47 / −1.71).",
            ],
        },
        "a2c": {
            "evidence": ev("a2c"),
            "whyItDegrades": [
                "Its Stage-A high-bid style persists and intensifies: bid rate 0.84 → 0.92 (S4), cap/fair 1.05 → 1.21, price/fair 1.11 → 1.26; the most frequent bid is FV_1.25 (21% → 31% of bids).",
                "Against RL opponents that also bid above fair value it wins fewer stars (9.1 → 3.3 in S4) and fills the squad with marginal and overseas players (squad 15.7 → 23.1; 8 overseas in 87% of S4 rooms).",
                "Shield dependence rises (forced bids 0.25 → 0.56–0.71 per episode; keeper closed by a forced bid 57 → 114–156 per 500) and forced bids are won less often (46% → 40%).",
                "Its keeper is completed late (mean progress 0.22 → 0.74 in S4). All 3 A2C incomplete-XI findings follow mechanism M2 (early purse and overseas exhaustion against four PPO seats).",
            ],
            "isShieldDependenceResponsible": "Shield dependence is a symptom that co-occurs with the loss (both follow early purse exhaustion); the data cannot show that it causes the XI loss — forced bids are ~0.5 per episode, while the loss is dominated by fewer stars.",
        },
        "d3qn": {
            "evidence": ev("d3qn"),
            "whyItDegrades": [
                "Greedy (argmax) selection at a low-to-fair cap: its most frequent action is BASE (≈ 36–44% of bids), and its cap/fair falls (1.07 → 0.96) while prices paid rise (1.09 → 1.25 in S4): it bids on many lots at base, loses the contested ones, and pays up for what it does win.",
                "Its star count falls (9.9 → 6.2 in S4) and its keeper completion shifts later in the tail (mean 0.07 → 0.40 in S4); shield dependence rises (keeper closed by a forced bid 5 → 81 per 500 in S4; forced-bid win rate 68% → 42%).",
                "It changes its offer for the same player somewhat even at a similar purse (+0.05 to +0.08 cap/fair) — slightly more state-responsive than PPO / ES.",
            ],
            "greedyActionSpecificWeakness": "Argmax makes the same cap decision in every similar state; against copies of itself (and QR-DQN) the resulting price war is the largest C1→C4 escalation measured (as an opponent: −1.60 mean). This is consistent with, not proof of, a greedy-policy weakness.",
            "whyItFailsKeeperCompletion": "2 D3QN opponent-seat findings, both mechanism M1: it passed on 17 cheap keepers each while the shield rated the requirement SAFE, spent its purse by mid-auction, and lost the forced re-auction keeper bids with ≤ ₹50L.",
        },
        "qrdqn": {
            "evidence": ev("qrdqn"),
            "doesDistributionalModellingHelp": "Relative to D3QN, QR-DQN degrades slightly less in every composition (C1 −2.46 vs −2.60, C4 −3.23 vs −3.29, S4 −4.34 vs −4.82) and keeps a higher strong-XI rate in S4 (98.9% vs 89.1%), with lower S4 seed variance (0.24 vs 0.82). The experiment cannot attribute this to the distributional head — the two differ in more than the head.",
            "whereItDiffersFromD3QN": [
                "Its top action shifts from FV_0.70 (Stage A) to BASE in Stage B; pass share falls 0.42 → 0.22.",
                "Its keeper is completed earlier than D3QN's in S4 (mean 0.20 vs 0.40).",
                "One export (qrdqn:s2) accounts for 12 of the 20 incomplete-XI findings (8 as an opponent seat, 4 as learner), all mechanism M1 — consistent with its Stage-A profile (the highest shield-forced keeper count of the three QR-DQN seeds). This is seed-specific, not algorithm-wide.",
            ],
        },
        "es": {
            "evidence": ev("es"),
            "whyItDegradesMost": [
                "It is the near-fixed 1.25× fair-value rule: FV_1.25 is 62% of its bids in Stage A and 86% in S4; its pass share collapses (0.20 → 0.005 in S4).",
                "Against opponents that pay more than 1.25× for stars, ES loses the early star lots (stars by 0.3 progress: 10.1 → 1.7 in S4), then wins almost every later lot it bids on at 1.25×: squad 15.8 → 24.9, 8 overseas in 100% of S4 rooms, with 18% of its purse stranded because the squad is full.",
                "Its early spend is strongly and positively associated with its XI in S4 (ρ ≈ 0.67): the ES episodes that do win early lots do better — the problem is losing the early contests, not spending too early.",
                "All 3 ES incomplete-XI findings follow M2 (purse and overseas exhausted in the first quarter, against four PPO seats).",
            ],
            "isItParticularlyVulnerable": "Yes in this setup: it has the largest loss in every composition and the largest C1→C4 escalation as a learner (−1.69), and ~0 same-lot adaptation. Its behaviour is consistent with a state-insensitive bidding rule meeting opponents that bid above that rule. This is a statement about this frozen export and its training setup, not about evolution strategies in general.",
        },
    }
    (rep / "algorithm-diagnosis.json").write_text(json.dumps(diag, indent=1, ensure_ascii=False), encoding="utf-8")

    star = FM["star loss (scarcity at the top)"]
    sat = FM["squad saturation (squad ≥ 24 with ≥ 5% purse unspent)"]
    reqs = [
        {"problem": "Early star contests are lost to learned opponents, and the policies do not compensate",
         "evidence": [f"Δstars is the strongest correlate of ΔXI (episode Spearman {TA['correlationWithDeltaXI']['all']['stars']['spearman']:.2f}; cell-level Pearson {TA['cellLevel']['correlationWithTransfer']['stars']['pearson']:.2f}).",
                      f"Star-loss episodes (≥ 2 fewer stars than the Stage-A pair): {star['episodes']:,} ({star['shareOfStageB']:.0%} of Stage-B episodes), mean ΔXI {star['meanDeltaXI_affected']:.2f} vs {star['meanDeltaXI_unaffected']:.2f}.",
                      "Stars bought by 0.3 progress fall for every algorithm in S4 (e.g. ES 10.1 → 1.7, A2C 9.1 → 3.3)."],
         "affectedAlgorithms": ALGOS, "affectedMatchups": "all; largest against D3QN / QR-DQN opponents and in C4 / S4", "severity": "high",
         "classification": ["training-distribution issue (Stage-A rule opponents rarely contest stars above fair value)", "opponent-modelling issue"],
         "openQuestions": "Is the loss recoverable by re-allocating purse to later value (PPO's pattern), or must stars be contested? Stage-B design should measure both."},
        {"problem": "Frozen bidding rules respond little to opponent pressure (policy approximately fixed, state distribution changes)",
         "evidence": ["Same player, similar own purse (|Δpurse share| < 0.05): mean Δcap/fair −0.02 to +0.00 for PPO (identical cap in 78–86% of lots) and −0.02 to +0.05 for ES (64–95%); +0.01 to +0.11 for A2C (57–67%); +0.05 to +0.08 for D3QN / QR-DQN (50–71%).",
                      "Cap changes that do occur track the purse difference (Spearman 0.24–0.44 between Δcap/fair and Δpurse share).",
                      "ES bids FV_1.25 on 62% → 86% of bids regardless of opponents."],
         "affectedAlgorithms": ["es", "a2c"], "affectedMatchups": "all (for PPO the same insensitivity is harmless because its Stage-A style already avoids price wars)", "severity": "high",
         "classification": ["training-distribution issue", "observation issue (unknown whether the rival features carry enough signal — cannot be determined from this data)", "unknown"],
         "openQuestions": "Is the insensitivity a training-distribution effect (no variation in Stage A) or an observation limitation? Needs a counterfactual probe or training experiment."},
        {"problem": "Population pressure: identical aggressive opponents escalate prices",
         "evidence": [f"C4 − C1 escalation by opponent: D3QN {CC['byOpponent']['d3qn']['meanEscalation']}, QR-DQN {CC['byOpponent']['qrdqn']['meanEscalation']}, A2C {CC['byOpponent']['a2c']['meanEscalation']}, ES {CC['byOpponent']['es']['meanEscalation']}, PPO {CC['byOpponent']['ppo']['meanEscalation']} XI.",
                      "Escalation is sublinear (C4/C1 ratio 1.06–1.70, never 4×) and is associated with the C4−C1 rise in room price/fair (Spearman −0.49) and in RL-opponent spend by 0.3 progress (−0.48)."],
         "affectedAlgorithms": ALGOS, "affectedMatchups": "C4 vs D3QN / QR-DQN / A2C; S4", "severity": "medium–high",
         "classification": ["training-distribution issue", "opponent-modelling issue"],
         "openQuestions": "What opponent mix should Stage-B training sample so that population pressure is represented without collapsing to one style?"},
        {"problem": "Squad and overseas saturation with stranded purse",
         "evidence": [f"Squad ≥ 24 with ≥ 5% purse unspent: {sat['episodes']:,} episodes, mean ΔXI {sat['meanDeltaXI_affected']:.2f} (vs {sat['meanDeltaXI_unaffected']:.2f}).",
                      "ES in S4: squad 24.9, 8 overseas in 100% of rooms, 18% purse left; A2C in S4: squad 23.1, 8 overseas in 87%."],
         "affectedAlgorithms": ["es", "a2c"], "affectedMatchups": "C4 vs D3QN / QR-DQN; S4", "severity": "high for ES / A2C",
         "classification": ["purse-management issue", "reward issue (unknown — marginal buys are low-value but not penalised; cannot be determined from this data)", "training-distribution issue"],
         "openQuestions": "Does the XI-delta reward give the policy any signal about slot / overseas capacity as a resource? To be investigated in Stage-B design."},
        {"problem": "Keeper completion under scarcity (completion-shield blind spot)",
         "evidence": [f"20 incomplete-XI findings, all missing a keeper; mechanism M1 ({len(mech.get('M1', []))}): passes on cheap keepers while SAFE, CRITICAL only in the re-auction, forced bids lost with ≤ ₹500L; mechanism M2 ({len(mech.get('M2', []))}): purse and overseas exhausted in the first quarter, forced keeper bids lost to rivals with large purses.",
                      "Rivals ending with ≥ 3 keepers (depth buying, often PPO) held 12–24 keepers in every finding room; the shield's forcing rule counts rivals that NEED keepers.",
                      f"Final-path forced bids: 0 in Stage A, {SH['finalPathForcedBids']['StageB']} in Stage B (23 of them against four PPO seats); forced-bid win rates fall (e.g. D3QN 68% → 42%)."],
         "affectedAlgorithms": ["qrdqn", "d3qn", "a2c", "es"], "affectedMatchups": "C4 (all findings), mostly high-purse auctions; learners vs 4× PPO; qrdqn:s2 in any room", "severity": "low frequency (20 / 195,000) but a legality failure",
         "classification": ["completion/shield issue", "purse-management issue", "training-distribution issue"],
         "openQuestions": "Should completion safety account for rivals' depth buying and for the seat's ability to contest (purse vs rivals)? This concerns the frozen act-v3 shield and needs an explicit decision."},
        {"problem": "Late requirement completion and rising shield dependence",
         "evidence": ["Keeper completion moves later for every algorithm (e.g. PPO 0.09 → 0.57, A2C 0.22 → 0.74, ES 0.21 → 0.77 mean progress in S4); Indian completion later too.",
                      "Forced bids per episode rise for D3QN (0.015 → 0.43 in S4) and QR-DQN (0.035 → 0.29); PPO stays ≤ 0.005."],
         "affectedAlgorithms": ["d3qn", "qrdqn", "a2c", "es"], "affectedMatchups": "C4, S4", "severity": "medium",
         "classification": ["purse-management issue", "completion/shield issue", "training-distribution issue"],
         "openQuestions": "Late completion is not harmful for PPO (it completes cheaply late); is it harmful for the others only because they lack purse late?"},
        {"problem": "Seed-specific fragility",
         "evidence": ["qrdqn:s2 is involved in 12 of 20 findings; PPO s2 as an opponent is associated with more damage than s1 / s3; C1/C4 seed-pair variance is mostly opponent-seed (56%) vs learner-seed (37%)."],
         "affectedAlgorithms": ["qrdqn", "ppo"], "affectedMatchups": "specific exports", "severity": "medium",
         "classification": ["training-distribution issue", "unknown"],
         "openQuestions": "Stage-B evaluation should keep multi-seed opponents; single-seed conclusions would be misleading."},
    ]
    (rep / "stage-b-requirements.json").write_text(json.dumps({"note": "Diagnosis only. Each item is an evidence-backed problem for Stage-B design to investigate; no change is prescribed or approved.",
                                                                "problems": reqs}, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote algorithm-diagnosis.json, stage-b-requirements.json")


if __name__ == "__main__":
    main(sys.argv[1])
