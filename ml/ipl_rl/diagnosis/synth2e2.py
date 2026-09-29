"""Phase 2E.2 — synthesis: mechanism → feature matrix (A), observability
categories (B), information-sufficiency scorecard (K) and the decision gate.
Every number is read from the Phase 2E.2 analysis JSON files; the category /
label for each row is a documented judgement over that evidence (rules below).

    python synth2e2.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REP = ROOT / "ml/reports/phase2e2"
J = lambda n: json.loads((REP / n).read_text(encoding="utf-8"))
INV = J("observation-inventory.json")
FI = {f["name"]: f["index"] for f in INV["features"]}
FO, CF, AL, KO, OO, MO, PS, TA = (J(x) for x in ("failure-observability.json", "counterfactual-pairs.json", "aliasing-analysis.json", "keeper-observability.json",
                                                "opponent-observability.json", "multi-opponent-analysis.json", "purse-star-analysis.json", "temporal-analysis.json"))
SUP = J("raw/supplement.json")

pct = lambda x: f"{100 * x:.0f}%" if x is not None else "n/a"
feat = lambda *names: [f"{FI[n]} {n}" for n in names]
mech = FO["mechanisms"]
best = lambda m: mech[m].get("bestCarrier") or {}
sep = {x["feature"]: {b["progress"]: b for b in x["byProgress"]} for x in OO["separation"]}
S = lambda f, b, k="sep_C4high_vs_A": sep[f][b][k]
cfC4 = CF["conditions"]["C4"]
cfC1 = CF["conditions"]["C1"]
dv = CF["divergence"]["conditions"]
tr = OO["hiddenTracking"]
sat = {x["feature"]: x for x in SUP["saturation"]["features"]}
stars = SUP["starTiming"]
kc = SUP["keeperHiddenVsObservedCells"]["betweenCellShare"]
kbm = SUP["keeperByMechanism"]
fk = SUP["forcedPredictability"]["forcedKeeper_withinKeeperNeedDecisions"]["top"]
su = OO["stageASupport"]
pr = OO["policyResponse"]["C4"]["byLearnerAlgorithm"]
am = AL["mechanisms"]
esp = SUP["earlyStarPriceByCondition"]["byCondition"]
M1, M2 = kbm["M1"], kbm["M2"]

# ── A — mechanism → information → features (with category B) ────────────
MATRIX = [
    {"id": "early_star", "mechanism": "Early star competition",
     "relevantInformation": "how strongly the other seats will bid for THIS star now (their caps), i.e. the room's bidding propensity before and during the first star lots",
     "features": feat("lot_is_star", "lot_rating", "lot_fair_value", "mkt_recent_price_ratio", "riv_recent_spend", "mkt_recent_sold_share", "riv_purse_mean", "riv_purse_min", "riv_purse_std", "riv_capacity_1", "riv_capacity_2", "riv_capacity_3", "riv_gain_mean"),
     "present": "partially", "directness": "indirect (only through what earlier lots revealed: the 20-lot window and rival purses)",
     "quality": f"weak — best within-progress |ρ| with the price needed to win on early star lots {abs(best('early_star')['withinProgressSpearman']):.2f} ({best('early_star')['feature']}); riv_capacity_1–3 sit at the clip value 5 in 100% of decisions before 30% progress",
     "timeAvailable": f"none at the first lot (the first learner observation is bit-identical to Stage A in every Stage-B episode); room-level separation stays < 0.5 until ~5% of the main round and reaches ≥ 0.9 only in the 5–10% bin (riv_purse_min {S('riv_purse_min', '5–10%'):.2f}); {pct(stars['cumulative'][2])} of star lots are auctioned in lots 1–16 and {pct(stars['cumulative'][3])} in lots 1–32",
     "category": "C", "categoryReason": "the lot and the rivals' purses are visible, but the component that decides the contest — how aggressively these opponents bid — is invisible before any sale and only weakly reflected lot by lot afterwards"},
    {"id": "aggressiveness", "mechanism": "Opponent aggressiveness",
     "relevantInformation": "whether the room's bidders pay more than a Stage-A room would (room-level), and how much more on this lot (lot-level)",
     "features": feat("mkt_recent_price_ratio", "riv_recent_spend", "mkt_recent_sold_share", "riv_purse_mean", "riv_purse_min", "riv_purse_std", "riv_xi_mean", "riv_xi_max", "mkt_premium_passed", "self_pace_gap"),
     "present": "yes at room level after the first ~5–10% of lots; weakly at lot level", "directness": "indirect (history window + cumulative rival purse spread)",
     "quality": f"room level strong: C4 D3QN/QR-DQN rooms vs Stage A separation {S('riv_purse_min', '10–20%'):.2f} (riv_purse_min) / {S('riv_purse_std', '10–20%'):.2f} (riv_purse_std) at 10–20% progress; lot level moderate at best (within-progress |ρ| {abs(best('aggressiveness')['withinProgressSpearman']):.2f}, paired Δ-tracking ≤ {max(abs(x['spearman_dFeature_dNeed'] or 0) for x in cfC4['deltaTracking']):.2f})",
     "timeAvailable": "from ~5–10% of the main round (≈ 16–32 lots)",
     "category": "B", "categoryReason": "not a single feature, but the combination of rival-purse statistics and the window separates aggressive rooms from Stage-A rooms almost perfectly after the early lots"},
    {"id": "multi_aggressive", "mechanism": "Multiple aggressive opponents",
     "relevantInformation": "HOW MANY rivals are aggressive / RL-controlled and able to bid on this lot",
     "features": feat("riv_purse_std", "riv_purse_min", "riv_purse_mean", "riv_purse_max", "riv_capacity_1", "riv_capacity_2", "riv_capacity_3", "riv_able_share", "riv_recent_spend", "mkt_recent_price_ratio"),
     "present": "partially", "directness": "indirect, through permutation-symmetric aggregates only",
     "quality": f"weak — number of RL rivals bidding tracked at best |ρ| {tr['RL rivals bidding on this lot']['bestAbs']:.2f}; C1 vs C4 separation ≤ {max(b['sep'] or 0 for b in MO['separationC4vsC1']['riv_purse_std'][:4]):.2f} (riv_purse_std) in the first 20%; {pct(MO['rlBiddersDiffer2plus_rivalBlockLeTight'])} of C1/C4 pairs with ≥ 2 more RL bidders have all 17 rival/window features within 0.02",
     "timeAvailable": "mid-auction at best (C1 vs C4 separation peaks at 30–75% progress)",
     "category": "C", "categoryReason": "the observation carries max/mean/min/std of purses and the top-3 capacities (saturated), never a count or identity of aggressive rivals"},
    {"id": "rival_purse", "mechanism": "Rival purse pressure",
     "relevantInformation": "how much money rivals hold", "features": feat("riv_purse_max", "riv_purse_mean", "riv_purse_min", "riv_purse_std", "riv_capacity_1", "riv_capacity_2", "riv_capacity_3", "riv_able_share"),
     "present": "yes (four summary statistics of the nine purses)", "directness": "direct",
     "quality": f"strong for the summaries (mean rival purse ρ = {best('rival_purse')['withinProgressSpearman']:.2f}; 0 of {am['rival_purse']['materialPairs']:,} pairs with a ≥ 0.10 purse gap are aliased); the purse-minus-completion-cost capacities are clipped at 5 in {pct(sat['riv_capacity_1']['atUpperClip5'])} / {pct(sat['riv_capacity_3']['atUpperClip5'])} of decisions (riv_capacity_1 / _3)",
     "timeAvailable": "always", "category": "A", "categoryReason": "directly encoded; only the per-rival distribution beyond four statistics and the (saturated) capacities is lost"},
    {"id": "rival_role_demand", "mechanism": "Rival role demand",
     "relevantInformation": "how many rivals still need the lot's role / requirement",
     "features": feat("riv_fills_share", "mkt_scarcity_keeper", "mkt_scarcity_bowling", "mkt_scarcity_indians", "self_status_keeper", "self_status_bowling", "self_status_indians"),
     "present": "partially", "directness": "indirect (pooled share; rivals-needing enters only the scarcity denominators)",
     "quality": f"keeper demand strong (riv_fills_share ρ {best('rival_role_demand')['withinProgressSpearman']:.2f} on keeper lots); bowling / Indian demand effectively invisible: mkt_scarcity_bowling / _indians sit at the clip value 5 in {pct(sat['mkt_scarcity_bowling']['atUpperClip5'])} / {pct(sat['mkt_scarcity_indians']['atUpperClip5'])} of decisions; riv_fills_share pools keeper, bowling, Indians and squad-count needs",
     "timeAvailable": "always (current state)", "category": "C", "categoryReason": "visible for keepers, saturated / pooled for the other requirements"},
    {"id": "rival_keeper_demand", "mechanism": "Rival keeper demand",
     "relevantInformation": "rivals that still NEED a keeper", "features": feat("mkt_scarcity_keeper", "riv_fills_share", "self_status_keeper", "mkt_supply_Wi", "mkt_supply_Wo"),
     "present": "yes", "directness": "indirect (scarcity ratio and fills share)",
     "quality": f"strong (mkt_scarcity_keeper ρ {best('rival_keeper_demand')['withinProgressSpearman']:.2f}; between-cell share of 'rivals needing a keeper' {kc['rivals NEEDING a keeper (control: in the observation)']:.3f})",
     "timeAvailable": "always", "category": "B", "categoryReason": "recoverable from scarcity × supply × own need"},
    {"id": "keeper_stockpiling", "mechanism": "Rival keeper stockpiling",
     "relevantInformation": "keepers held by rivals who no longer need one, and their willingness / ability to buy MORE keepers",
     "features": feat("mkt_same_role_left", "mkt_supply_Wi", "mkt_supply_Wo", "mkt_scarcity_keeper", "riv_fills_share", "riv_gain_mean", "riv_purse_mean", "riv_free_slots_share"),
     "present": "partially", "directness": "indirect (the holdings level is implied by keeper supply depletion; no per-rival keeper count, no buyer identity in the history)",
     "quality": f"holdings level largely inferable: {pct(kc['max keepers held by one rival'])} (max held by one rival), {pct(kc['rivals holding ≥ 2 keepers'])} (rivals holding ≥ 2) and {pct(kc['rivals NOT needing a keeper but able to buy one'])} (non-needing rivals able to buy) of the variance lies between cells of the observed keeper features; the WILLINGNESS of non-needing rivals to buy is not encoded anywhere (the planner, the shield and the observation count only rivals that NEED a keeper) — at the M1 forced bids that were lost, {M1['atLostForcedBids']['meanRivalsNeedingKeeper']} rivals needed a keeper on average vs {M1['atLostForcedBids']['meanRivalsNotNeedingButAbleToBuy']} non-needing rivals able to buy",
     "timeAvailable": "current state only; the purchases that caused the failures happened after the pass decisions",
     "category": "C", "categoryReason": "how many keepers rivals hold is largely inferable; whether non-needing rivals will contest the remaining keepers is not observable"},
    {"id": "overseas", "mechanism": "Overseas competition",
     "relevantInformation": "own overseas slots (8-cap) and rivals' remaining overseas slots / demand",
     "features": feat("lot_is_overseas", "self_overseas_slots_left", "self_xi_overseas", "mkt_overseas_contested", "mkt_supply_Wo", "mkt_supply_Bo", "mkt_supply_Oo", "riv_capacity_1", "riv_able_share", "riv_gain_mean"),
     "present": "own: yes; rivals: partially", "directness": "own direct; rivals only through their exclusion from riv_capacity / riv_able_share / riv_gain_mean on overseas lots when blocked",
     "quality": f"own strong (direct); rival overseas slot counts absent; overseas lots are identified (lot_is_overseas / mkt_overseas_contested within-progress ρ with the price needed {tr['needToWin (price the other nine seats would pay / fair)']['top'][2]['withinProgressSpearman']:.2f} / {tr['needToWin (price the other nine seats would pay / fair)']['top'][3]['withinProgressSpearman']:.2f})",
     "timeAvailable": "always", "category": "C", "categoryReason": "own overseas state is direct; rivals' overseas capacity is not encoded"},
    {"id": "squad_slots", "mechanism": "Squad-slot competition",
     "relevantInformation": "own squad slots; rivals' remaining squad slots",
     "features": feat("self_slots_left", "self_bench", "self_xi_empty", "riv_free_slots_share"),
     "present": "own: yes; rivals: coarse", "directness": "own direct; rivals as the share with ≥ 1 free slot",
     "quality": "own strong; rival slot counts reduced to one binary-per-rival share", "timeAvailable": "always",
     "category": "C", "categoryReason": "own slot state is direct (A); rivals' slot state is coarsened to 'any slot left'"},
    {"id": "price_inflation", "mechanism": "Price inflation",
     "relevantInformation": "recent prices relative to fair value (and what future prices will be)",
     "features": feat("mkt_recent_price_ratio", "riv_recent_spend", "mkt_recent_sold_share"),
     "present": "yes (recent); future prices partially", "directness": "direct (20-lot window, pooled over all buyers including self)",
     "quality": f"recent inflation is literally encoded; it tracks the price needed on the current lot moderately (ρ {best('price_inflation')['withinProgressSpearman']:.2f}); realised FUTURE star prices differ by room type (between-room share {PS['futureStarPrice_betweenRoomTypeShare']:.2f}) but the best single feature tracks them at |ρ| {abs(PS['futureStarPrice_tracking_mainBefore50pct'][0]['withinProgressSpearman']):.2f}",
     "timeAvailable": "window of 20 lots; empty at the first lot", "category": "A", "categoryReason": "recent price/fair is an explicit feature; its horizon (20 lots, pooled) limits it as a forecast"},
    {"id": "requirement_timing", "mechanism": "Requirement timing",
     "relevantInformation": "own unmet requirements, how many suitable players remain, whether this is a final opportunity",
     "features": feat("self_need_keeper", "self_need_bowling", "self_need_indians", "self_need_players", "self_status_keeper", "self_status_bowling", "self_status_indians", "self_status_players", "self_final_opportunity", "self_fills_keeper", "mkt_supply_Wi", "mkt_supply_Wo", "mkt_supply_Bi", "mkt_supply_Bo", "main_progress", "lots_left_in_phase"),
     "present": "yes", "directness": "direct", "quality": "strong (own needs, planner statuses and supply are exact)", "timeAvailable": "always",
     "category": "A", "categoryReason": "own requirement state and remaining supply are encoded directly"},
    {"id": "purse_depletion", "mechanism": "Purse depletion", "relevantInformation": "own purse, completion reserve, max safe bid, pace",
     "features": feat("self_purse", "self_max_safe_purse", "self_max_safe_fv", "self_reserve_if_passed", "self_pace_gap"),
     "present": "yes", "directness": "direct", "quality": f"strong (self_purse ρ {best('purse_depletion')['withinProgressSpearman']:.2f}); self_max_safe_fv clipped at 5 in {pct(sat['self_max_safe_fv']['atUpperClip5'])} of decisions (only while the purse is large)",
     "timeAvailable": "always", "category": "A", "categoryReason": "encoded directly"},
    {"id": "reauction_pressure", "mechanism": "Re-auction pressure",
     "relevantInformation": "whether the re-auction has started, how many unsold players return, who can still buy them",
     "features": feat("phase_reauction", "returning_count", "lots_left_in_phase", "mkt_supply_Wi", "mkt_supply_Wo", "riv_purse_max", "riv_free_slots_share", "self_reserve_if_passed"),
     "present": "yes (supply); partially (competitors)", "directness": "supply direct; competitors via purse aggregates and free-slot share",
     "quality": "returning supply exact; competitor capacity coarse", "timeAvailable": "throughout the main round (returning pool grows) and in the re-auction",
     "category": "B", "categoryReason": "the returning pool is explicit; who will compete for it is inferable only from aggregates"},
    {"id": "final_path", "mechanism": "Final-path risk",
     "relevantInformation": "approaching the point where only forced bids can still complete the XI, and whether those bids will win",
     "features": feat("self_final_opportunity", "self_status_keeper", "self_status_players", "mkt_scarcity_keeper", "self_reserve_if_passed", "self_max_safe_fv", "self_purse", "riv_fills_share"),
     "present": "partially", "directness": "the approach is direct; the outcome of forced bids depends on non-needing rivals (absent)",
     "quality": f"forced keeper bids are well separated from other keeper-need decisions by single features ({fk[0]['feature']} separation {fk[0]['separation']:.2f}, mkt_scarcity_keeper {next(x['separation'] for x in fk if x['feature'] == 'mkt_scarcity_keeper'):.2f}); the rivals who won the lost forced bids mostly did not need a keeper (see stockpiling)",
     "timeAvailable": "approach visible as it happens", "category": "C", "categoryReason": "the approach to the final path is observable; the risk that forced bids fail is not"},
]
LABEL_OF = {"A": "YES", "B": "YES", "C": "PARTIAL", "D": "NO"}
FO["matrix"] = MATRIX
FO["categoryDefinitions"] = {"A": "directly observable — the observation explicitly contains the required state",
                             "B": "indirectly observable — inferable from several existing features",
                             "C": "partially observable — some components visible, a critical component missing / coarsened",
                             "D": "not observable — the policy cannot distinguish the relevant situations"}
FO["categorySummary"] = {c: [m["mechanism"] for m in MATRIX if m["category"] == c] for c in "ABCD"}
(REP / "failure-observability.json").write_text(json.dumps(FO, indent=1, ensure_ascii=False), encoding="utf-8")

# ── K — scorecard (the seven Phase 2E.1 Stage-B problems) ────────────────
SC = [
    {"problem": "Loss of early star contests", "observable": "PARTIAL", "confidence": "high",
     "evidence": f"First learner decision bit-identical to Stage A in every Stage-B episode ({dv['C4']['episodes']:,} C4 episodes; divergence at lot {dv['C4']['firstDifferentLot']['median']:.0f}–{dv['C4']['firstDifferentLot']['p90']:.0f} of 323). Room-level separation < 0.5 before ~5% of the main round, ≥ 0.9 from 5–10% (riv_purse_min). {pct(stars['cumulative'][2])} of stars are sold in lots 1–16. Early star price/fair: Stage A {esp['A']['meanPriceToFair']}, C4 {esp['C4']['meanPriceToFair']}, S4 {esp['S4']['meanPriceToFair']}. Lot-level price needed tracked at best |ρ| {abs(best('early_star')['withinProgressSpearman']):.2f}.",
     "missingInformation": "any signal of the opponents' bidding propensity before sales reveal it; per-lot competing-bid strength beyond pooled aggregates (the three capacity features are saturated)"},
    {"problem": "Bids change with purse rather than opponent behaviour", "observable": "YES", "confidence": "medium",
     "evidence": f"Room-level opponent pressure is encoded after the first ~5–10% of lots (separation {S('riv_purse_min', '10–20%'):.2f} / {S('riv_purse_std', '10–20%'):.2f} / {S('riv_purse_mean', '10–20%'):.2f} for riv_purse_min / _std / _mean at 10–20%). The frozen caps move with the own purse (C4 same-lot Spearman ΔCap~ΔownPurse {min(v['dCap_vs_dOwnPurse'] for v in pr.values()):.2f}–{max(v['dCap_vs_dOwnPurse'] for v in pr.values()):.2f}) and barely with the recent price ratio ({min(v['dCap_vs_dRecentPriceRatio'] for v in pr.values()):+.2f} to {max(v['dCap_vs_dRecentPriceRatio'] for v in pr.values()):+.2f}). {pct(su['C4 high-pressure']['anyDirectFeatureOutside'])} of learner observations in C4 D3QN/QR-DQN rooms have an opponent-direct feature outside the Stage-A 0.5–99.5% range.",
     "missingInformation": "nothing structural at room level; lot-level competition is only weakly encoded (see early stars)"},
    {"problem": "Pressure from multiple aggressive RL opponents", "observable": "PARTIAL", "confidence": "medium",
     "evidence": f"RL rivals bidding on the lot tracked at best |ρ| {tr['RL rivals bidding on this lot']['bestAbs']:.2f}. {pct(MO['rlBiddersDiffer2plus_rivalBlockLeTight'])} of same-lot C1/C4 pairs with ≥ 2 more RL bidders have all rival + window features within 0.02. Rival features are symmetric aggregates (max/mean/min/std, top-3 capacities clipped at 5 in {pct(sat['riv_capacity_3']['atUpperClip5'])}+ of decisions).",
     "missingInformation": "count / identity of aggressive rivals; per-rival purse and recent spend; buyer identity in the 20-lot window"},
    {"problem": "Squad / overseas slots filling while purse remains", "observable": "YES", "confidence": "high",
     "evidence": "Own squad slots, overseas slots, purse, bench, XI state and remaining upgrades are direct features (self_slots_left, self_overseas_slots_left, self_purse, self_bench, self_xi_*, self_upgrades_left).",
     "missingInformation": "rivals' overseas / squad-slot counts (not needed to see the own-state problem)"},
    {"problem": "Wicketkeeper completion failures when rivals stockpile keepers", "observable": "PARTIAL", "confidence": "medium",
     "evidence": f"M1 (14 cases): all {M1['passes']} passes were on affordable keepers that filled the own need ({M1['ofWhichWentUnsold']} of them went unsold); planner status NEED visible; shield keeper state SAFE in {pct(M1['shieldKeeperStateSafeShare'])} of decisions. Observed keeper features at the dangerous decisions were typical of matched safe decisions (median percentiles {min(KO['summary']['medianPercentileObserved'].values()):.2f}–{max(KO['summary']['medianPercentileObserved'].values()):.2f}); the nearest safe decision is within 0.05 in the keeper sub-space for only {pct(KO['summary']['nearestSafeLinfKeeperSubspace']['shareLe0.05'])}. Holdings are {pct(kc['max keepers held by one rival'])} inferable; non-needing demand is not encoded (lost forced bids: {M1['atLostForcedBids']['meanRivalsNeedingKeeper']} needing vs {M1['atLostForcedBids']['meanRivalsNotNeedingButAbleToBuy']} non-needing able rivals). M2 (6 cases): purse / overseas exhaustion is directly visible.",
     "missingInformation": "willingness / capacity of rivals that do NOT need a keeper to buy one; per-rival keeper holdings; which rival bought recent keepers"},
    {"problem": "Later requirement completion and greater shield dependence", "observable": "PARTIAL", "confidence": "medium",
     "evidence": f"Own needs, statuses, supply and final-opportunity flags are direct; forced keeper bids are separable from other keeper-need decisions by self_purse ({fk[0]['separation']:.2f}) and mkt_scarcity_keeper. Rival demand for bowling / Indians is effectively invisible (scarcity features clipped at 5 in {pct(sat['mkt_scarcity_bowling']['atUpperClip5'])} of decisions); the shield's contestant count and non-needing demand are not encoded.",
     "missingInformation": "competition for requirement players from rivals that do not need them; unsaturated bowling / Indian scarcity"},
    {"problem": "Seed-specific weaknesses", "observable": "UNKNOWN", "confidence": "low",
     "evidence": "All three training seeds of an algorithm receive the identical 80-feature observation; seed differences are differences in learned behaviour, which this data cannot attribute to information content.",
     "missingInformation": "not an observation question on the available data"},
]
MECH_SC = [{"mechanism": m["mechanism"], "category": m["category"], "observable": LABEL_OF[m["category"]], "quality": m["quality"], "missing": m["categoryReason"]} for m in MATRIX]
(REP / "information-scorecard.json").write_text(json.dumps({
    "labels": {"YES": "sufficient evidence of observability", "PARTIAL": "some relevant information exists", "NO": "relevant information is absent", "UNKNOWN": "data insufficient"},
    "note": "No numeric scores, no ranking. Mechanism rows map category A/B → YES, C → PARTIAL, D → NO.",
    "scorecard": SC, "mechanisms": MECH_SC}, indent=1, ensure_ascii=False), encoding="utf-8")

# ── decision gate ─────────────────────────────────────────────────────────
GATE = {
    "result": "B",
    "name": "PARTIALLY SUFFICIENT",
    "implication": "Stage-B design requires a targeted observation-spec discussion before training.",
    "notChosen": {
        "A": f"not sufficient for every problem: opponent propensity is invisible at the start of the auction ({pct(stars['cumulative'][2])} of stars are sold in lots 1–16), the number of aggressive rivals and non-needing keeper demand are not encoded, and five features are clipped at 5 in ≥ 97% of decisions (riv_capacity_1–3, mkt_scarcity_bowling, mkt_scarcity_indians)",
        "C": "no major mechanism depends ENTIRELY on absent information: room-level aggressiveness is encoded after ~5–10% of the main round and the frozen policies still do not respond to it; every M1 keeper failure passed on an affordable keeper that filled a visible need, so the failure was avoidable with observable information",
        "D": "the existing replays were sufficient to establish, feature by feature, what is present, absent, aggregated or saturated; what they cannot establish (whether a retrained policy WOULD use the information) is a learning question, not an information question",
    },
    "evidence": {
        "present": ["own purse / squad / overseas / requirement state and remaining supply (direct)", "rival purse summaries (direct)", "rival keeper demand (indirect, strong)",
                    f"room-level opponent pressure after ~5–10% of lots (separation ≥ 0.9)", f"rival keeper holdings (≈ {pct(kc['max keepers held by one rival'])} inferable from supply depletion)"],
        "missingOrCoarsened": ["opponent bidding propensity before sales reveal it (first lot identical to Stage A by construction)", "count / identity of aggressive rivals (aggregates only)",
                               "willingness of rivals that do not need a keeper to buy one", "buyer identity in the 20-lot history",
                               f"saturated features: riv_capacity_1–3 ({pct(sat['riv_capacity_1']['atUpperClip5'])} / {pct(sat['riv_capacity_2']['atUpperClip5'])} / {pct(sat['riv_capacity_3']['atUpperClip5'])} at 5), mkt_scarcity_bowling ({pct(sat['mkt_scarcity_bowling']['atUpperClip5'])}), mkt_scarcity_indians ({pct(sat['mkt_scarcity_indians']['atUpperClip5'])})"],
        "presentButUnused": f"the frozen caps respond to own purse, not to opponent-direct features that do separate the rooms; {pct(su['S4']['anyDirectFeatureOutside'])} (S4) and {pct(su['C4 high-pressure']['anyDirectFeatureOutside'])} (C4 D3QN/QR-DQN) of Stage-B learner observations lie outside the Stage-A range in at least one opponent-direct feature — consistent with a policy that never saw these values during Stage-A training",
    },
    "residualUncertainty": ["Whether a policy trained on Stage-B data would exploit the present signals cannot be established without training (out of scope).",
                            "Keeper findings: 20 cases; conclusions about non-needing demand rest on 35 + 40 lost forced bids.",
                            "Replays cover 40 of 500 validation entries (19 with a finding; high stratum over-represented 20/40).",
                            "Correlation thresholds (strong ≥ 0.5, weak ≥ 0.2) and the aliasing tolerance (L∞ ≤ 0.02) are conventions; the JSON files carry the raw values."],
    "doesNotPrescribe": "This gate classifies the frozen observation contract only. It does not specify which features to add or change; that is the Stage-B design discussion.",
}
(REP / "decision-gate.json").write_text(json.dumps(GATE, indent=1, ensure_ascii=False), encoding="utf-8")
print("synthesis written:", {c: len(v) for c, v in FO["categorySummary"].items()}, [s["observable"] for s in SC])
