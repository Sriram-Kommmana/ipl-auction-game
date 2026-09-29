"""Phase 2E.3 — synthesis of the minimum observation-gap design.
Reads raw/gap-stats.json + raw/gap-extra.json (this phase) and the Phase 2E.1 /
2E.2 outputs; writes the required JSON deliverables. Every number is pulled from
those files; every classification applies the rules written in RULES below.

    python synth2e3.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REP = ROOT / "ml/reports/phase2e3"
P2 = ROOT / "ml/reports/phase2e2"
J = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
G = J(REP / "raw/gap-stats.json")
X = J(REP / "raw/gap-extra.json")
INV = J(P2 / "observation-inventory.json")
KO2 = J(P2 / "keeper-observability.json")
SUP2 = J(P2 / "raw/supplement.json")
FI = {f["name"]: f["index"] for f in INV["features"]}
dump = lambda n, o: (REP / n).write_text(json.dumps(o, indent=1, ensure_ascii=False), encoding="utf-8")
META = {"phase": "2E.3", "status": "DESIGN PROPOSAL ONLY — obs-v2 (629b25783f833af7) unchanged; nothing implemented, trained or committed",
        "source": "digest-verified read-only replays of recorded Phase 2E.0 episodes (15,600 episodes, 40 entries, 2,659,482 decision rows; same rows as Phase 2E.2)"}
RULES = {
    "REDUNDANT": "a deterministic transformation of existing features, OR ≥ 0.95 of its variance lies between cells of the existing features, OR it adds no separation (< 0.10) in any test",
    "PARTIAL NOVELTY": "not redundant, and adds within-cell separation ≥ 0.10 somewhere — but does not resolve observational aliasing (differs in < 5% of near-identical-observation pairs) and does not beat existing features left out of the same cells",
    "GENUINELY NEW INFORMATION": "resolves aliasing (differs in ≥ 5% of near-identical-observation pairs with an outcome-difference lift ≥ 2) AND adds within-cell separation beyond every existing feature tested",
    "delta": {"KEEP": "existing feature sufficient", "TRANSFORM": "useful information lost by the current representation", "ADD": "required information not represented", "DO NOT ADD": "unnecessary, redundant, speculative, impossible or leaking"},
}
cf = G["counterfactual"]["tests"]
cfP = cf["price needed to win differs ≥ 0.5 × fair"]["candidates"]
cfK = cf["keeper lot absorbed by a non-needing rival (one yes, one no)"]["candidates"]
cfR = cf["RL rivals bidding differ ≥ 2"]["candidates"]
em = G["early"]["labels"]
E_HP = em["C4 D3QN/QR-DQN room vs Stage-A room"]
E_FU = em["future-aggressive vs future-passive room (realised star price/fair, lots 33–162, top vs bottom tercile)"]
bin_ = lambda L, b: next(x for x in L if x["bin"] == b)
MX = X["multi"]["candidates"]
KX = X["keeper"]["candidates"]
MG = G["multi"]["candidates"]
KG = G["keeper"]["candidates"]
SAT = G["saturation"]

# ── candidate table (§10) ─────────────────────────────────────────────────
CANDS = [
    # id, concept, gap, source state, overlap, missing information, availability, leakage, computation, production
    ("rlRivals", "hidden policy identity", 1, "entry.seats[].type (rlSnapshot vs rule) — environment-internal", "none", "which seats are RL-controlled / which algorithm", "internal only", "high (policy identity)", "cheap", "not meaningful: the production lineup is fixed (5 RL + 4 rule personas), so it would be a constant"),
    ("partic5 / partic20", "bidding breadth", 1, "caps each seat brought to each resolved lot (resolveLot) — ladder participants", "none (obs has only price/fair and sold share)", "how many rivals entered the bidding on recent lots", "internal; public only as live bid events", "possible (a cap ≥ base reveals interest that the public ladder may not show)", "cheap", "needs the bid-event history in the runtime extras (server has bidPlaced events; extras carries only {poolSize, recent})"),
    ("bids5 / bids20", "bidding breadth", 1, "sim history h.bids (ladder raises per lot)", "partly mkt_recent_price_ratio (more raises ↔ higher price)", "number of raises per lot", "internal; public as live bid events", "none", "cheap", "needs bid events in extras"),
    ("cumPF", "pooled price history beyond 20 lots", 1, "public results so far (price, fair value)", "identical to mkt_recent_price_ratio before lot 21", "prices older than 20 lots", "after history", "none", "cheap", "needs full results history in extras"),
    ("cumStarPF", "role-specific price history", 1, "public results so far, star lots only", "mkt_recent_price_ratio pools all roles", "star prices separately", "after history", "none", "cheap", "needs full results history in extras"),
    ("wMaxRivSpend / wBuyers / buyersSoFar", "per-buyer history breakdown", 2, "public results with winner ids", "riv_recent_spend pools buyers", "concentration of recent spend across rivals", "after history", "none", "cheap", "extras.recent already carries winnerTeamId"),
    ("overpay13 / overpay15 / starBuyers2", "count of high-paying / star-buying rivals", 2, "public results per rival so far", "riv_xi_max, riv_purse_* aggregate", "how many distinct rivals pay above fair / buy stars", "after history", "none", "cheap", "needs per-rival results history in extras"),
    ("aheadPace15 / aheadPace30", "count of rivals spending ahead of pace", 2, "each rival's public purse + mkt_premium_passed", "riv_purse_min/std/mean summarise the same nine purses", "the COUNT of rivals ahead of pace (not recoverable from 4 statistics)", "immediate (current state)", "none", "cheap", "yes — ctx.rivals carries every purse"),
    ("rival purse vector (sorted, 9 values)", "rival purse distribution", 2, "public purses", "4 summary statistics", "the full 9-value distribution", "immediate", "none", "cheap", "yes"),
    ("capGe1_5 / capGe2 / capGe3 / capGe5", "count of rivals able to pay k × fair", 2, "rival purse − completion cost (as riv_capacity_*)", "riv_able_share (k = 1), riv_capacity_1–3 (clipped)", "counts at k > 1", "immediate", "none", "moderate (completion cost per rival — already computed by obs-v2)", "yes"),
    ("kContestAll", "keeper competition breadth", 3, "rivals with a free slot and purse ≥ base, not rule-blocked", "riv_able_share × 9 (Spearman 0.95), riv_free_slots_share", "—", "immediate", "none", "cheap", "yes"),
    ("kContestNoNeed / kContestNoNeedFair", "non-needing keeper competition", 3, "same, restricted to rivals already holding a keeper", "riv_able_share − rivals needing (riv_fills_share, mkt_scarcity_keeper)", "the non-needing part of the breadth", "immediate", "none", "cheap", "yes"),
    ("kStockExcess / kStock2", "rival keeper stockpile", 3, "public squads", "keeper supply depletion (mkt_supply_Wi/Wo)", "holdings per rival", "immediate", "none", "cheap", "yes"),
    ("kHistExtraBuyers / kW20Extra", "keeper-buying propensity without need", 3, "public results: keeper bought by a team already holding one", "none directly", "history of buying keepers without need", "after history", "none", "cheap", "needs per-rival results history in extras"),
    ("kW20Buys", "recent rival keeper purchases", 3, "public results, last 20 lots", "riv_recent_spend (pooled), supply counts", "role of recent purchases", "after history", "none", "cheap", "extras.recent carries winnerTeamId (role from the player catalogue)"),
    ("capacity unclipped (log)", "rival capacity above 5 × fair", 4, "same quantity as riv_capacity_1–3", "riv_capacity_1–3 (identical below 5)", "values above 5 × fair", "immediate", "none", "cheap", "yes"),
    ("scarcity unclipped (bowling / Indians)", "abundance above 5", 4, "same quantity as mkt_scarcity_bowling / _indians", "identical below 5", "values above 5", "immediate", "none", "cheap", "yes"),
]
cand_json = [{"candidate": c[0], "concept": c[1], "gap": c[2], "sourceState": c[3], "currentFeatureOverlap": c[4], "missingInformation": c[5],
              "temporalAvailability": c[6], "leakageRisk": c[7], "computation": c[8], "productionFeasibility": c[9]} for c in CANDS]

# ── redundancy (§11) with evidence ─────────────────────────────────────────
def within_best(d, *keys):
    vals = [d.get(k) for k in keys if d.get(k) is not None]
    return max(vals) if vals else None


RED = [
    {"candidate": "rlRivals", "class": "DO NOT ADD (identity)", "evidence": "separates C4 from Stage A perfectly at lot 1 (1.00) only because it IS the room's composition; 0.00 between C4 D3QN/QR-DQN and C4 PPO rooms (same count, different policies); constant in production"},
    {"candidate": "partic5 / partic20", "class": "PARTIAL NOVELTY",
     "evidence": f"between-cell share given rival block + price ratio {MX['partic20']['betweenCellShare']}; within-cell separation of ≥ 3 RL bidders {MX['partic20']['withinCellSep_rlBidders≥3']}; differs in {cfP['partic20']['shareCandidateDiffers']:.1%} of near-identical-observation pairs (lift {cfP['partic20']['lift']} for price needed); early separation ≤ {max(bin_(E_HP, b)['candidates']['partic20'] for b in ('lots 2–5', 'lots 6–16')):.2f} (lots 2–16)"},
    {"candidate": "bids5 / bids20", "class": "PARTIAL NOVELTY",
     "evidence": f"differs in {cfP['bids20']['shareCandidateDiffers']:.2%} of near-identical pairs; early separation ≤ {max(bin_(E_HP, b)['candidates']['bids5'] for b in ('lots 2–5', 'lots 6–16')):.2f}; tracks ≥ 3 RL bidders at {MG['bids20']['trackRlBiddersWithinProgress']}"},
    {"candidate": "cumPF", "class": "REDUNDANT", "evidence": f"identical to mkt_recent_price_ratio in {X['early']['cumPF_equals_windowPriceRatio_share']:.0%} of decisions before lot 21 (the window is the whole history there)"},
    {"candidate": "cumStarPF", "class": "PARTIAL NOVELTY",
     "evidence": f"lots 2–5: {bin_(E_HP, 'lots 2–5')['candidates']['cumStarPF']} vs best existing {bin_(E_HP, 'lots 2–5')['bestExisting']['separation']} ({bin_(E_HP, 'lots 2–5')['bestExisting']['feature']}); adds inside deciles of that feature {bin_(E_HP, 'lots 2–5')['candidatesWithinBestExistingDeciles']['cumStarPF']} (lots 2–5) / {bin_(E_HP, 'lots 17–32')['candidatesWithinBestExistingDeciles']['cumStarPF']} (lots 17–32) — after the existing features already separate at {bin_(E_HP, 'lots 17–32')['bestExisting']['separation']}"},
    {"candidate": "wMaxRivSpend / wBuyers / buyersSoFar", "class": "PARTIAL NOVELTY",
     "evidence": f"between-cell {MX['wMaxRivSpend']['betweenCellShare']}; within-cell C4 vs C1 {MX['wMaxRivSpend']['withinCellSep_C4vsC1']}; differs in {cfP['wMaxRivSpend']['shareCandidateDiffers']:.2%} of near-identical pairs (lift {cfP['wMaxRivSpend']['lift']})"},
    {"candidate": "overpay13 / overpay15 / starBuyers2", "class": "PARTIAL NOVELTY",
     "evidence": f"within-cell C4 vs C1 {MX['overpay13']['withinCellSep_C4vsC1']} / {MX['overpay15']['withinCellSep_C4vsC1']} / {MX['starBuyers2']['withinCellSep_C4vsC1']} vs existing features left out of the cells: riv_xi_max {X['multi']['existingNotInCells']['riv_xi_max']['withinCellSep_C4vsC1']}, self_pace_gap {X['multi']['existingNotInCells']['self_pace_gap']['withinCellSep_C4vsC1']}; in pairs whose rival block is aliased: {MG['overpay13']['sepC4vsC1_aliasedRivalBlock']} / {MG['overpay15']['sepC4vsC1_aliasedRivalBlock']} / {MG['starBuyers2']['sepC4vsC1_aliasedRivalBlock']}; differs in {cfP['overpay13']['shareCandidateDiffers']:.2%} of near-identical pairs"},
    {"candidate": "aheadPace15 / aheadPace30", "class": "PARTIAL NOVELTY",
     "evidence": f"tracks RL bidders {MG['aheadPace15']['trackRlBiddersWithinProgress']} (best existing 0.25); between-cell {MX['aheadPace15']['betweenCellShare']}; within-cell C4 vs C1 {MX['aheadPace15']['withinCellSep_C4vsC1']} (riv_xi_max, outside the cells, {X['multi']['existingNotInCells']['riv_xi_max']['withinCellSep_C4vsC1']}); aliased rival block {MG['aheadPace15']['sepC4vsC1_aliasedRivalBlock']}; differs in {cfP['aheadPace15']['shareCandidateDiffers']:.2%} of near-identical pairs"},
    {"candidate": "rival purse vector (sorted, 9 values)", "class": "PARTIAL NOVELTY",
     "evidence": f"differs by > 0.05 in {G['multi']['distributionCandidate_sortedPurses']['aliasedPairs_vectorDiffers>0.05']:.0%} of C1/C4 pairs whose rival block is aliased — the distribution does not carry the one-vs-four difference there"},
    {"candidate": "capGe1_5 / capGe2 / capGe3 / capGe5", "class": "REDUNDANT", "evidence": f"between-cell {MX['capGe2']['betweenCellShare']}; within-cell C4 vs C1 {MX['capGe2']['withinCellSep_C4vsC1']}; near-identical pairs where it differs: {cfP['capGe2']['pairsCandidateDiffers']}"},
    {"candidate": "kContestAll", "class": "REDUNDANT", "evidence": f"Spearman {X['keeper']['rivAbleCount_vs_kContestAll_spearman']} with riv_able_share × 9 (equal in {X['keeper']['shareRowsRivAbleCountEqualsContestAll']:.0%} of keeper decisions); within fine cells {KX['kContestAll']['withinCellSep']}"},
    {"candidate": "kContestNoNeed / kContestNoNeedFair", "class": "PARTIAL NOVELTY",
     "evidence": f"between-cell share {KX['kContestNoNeed']['betweenCellShare']} given controls + riv_able_share + riv_free_slots_share + riv_fills_share; within those cells it separates 'absorbed by a non-needing rival' at {KX['kContestNoNeed']['withinCellSep']} — the same as existing mkt_scarcity_keeper ({X['keeper']['existingOutsideCells']['mkt_scarcity_keeper']}) and mkt_equivalent_left ({X['keeper']['existingOutsideCells']['mkt_equivalent_left']}); differs in {cfK['kContestNoNeed']['pairsCandidateDiffers']} of {cf['keeper lot absorbed by a non-needing rival (one yes, one no)']['pairs']:,} near-identical keeper pairs; median percentile at the M1 dangerous decisions {G['keeper']['findingsMedianPercentileByMechanism']['M1:kContestNoNeed']} (0.5 = typical of safe)"},
    {"candidate": "kStockExcess / kStock2", "class": "REDUNDANT", "evidence": f"between-cell {KG['kStockExcess']['betweenCellShareGivenControls+rivalsNeeding']}; within-cell separation {KG['kStockExcess']['withinCellSep_absorbedByNonNeeding']}"},
    {"candidate": "kHistExtraBuyers / kW20Extra", "class": "REDUNDANT", "evidence": f"within fine cells {KX['kHistExtraBuyers']['withinCellSep']}; within control cells {KG['kHistExtraBuyers']['withinCellSep_absorbedByNonNeeding']} / {KG['kW20Extra']['withinCellSep_absorbedByNonNeeding']}; median percentile at M1 decisions {G['keeper']['findingsMedianPercentileByMechanism']['M1:kHistExtraBuyers']}"},
    {"candidate": "kW20Buys", "class": "PARTIAL NOVELTY", "evidence": f"within fine cells {KX['kW20Buys']['withinCellSep']} (= mkt_scarcity_keeper {X['keeper']['existingOutsideCells']['mkt_scarcity_keeper']}); between-cell {KX['kW20Buys']['betweenCellShare']}"},
    {"candidate": "capacity unclipped (log)", "class": "REDUNDANT (in the strategic range)",
     "evidence": f"identical to riv_capacity_* below 5; price needed > 5 × fair in {SAT['priceNeededToWin']['share>5xFair']:.2%} of decisions and no sale above 5 × fair ({SAT['salePriceToFair']['share>5']:.1%}); log raw capacity tracks the price needed at {SAT['riv_capacity_1']['trackPriceNeeded_raw(log)']}–{SAT['riv_capacity_3']['trackPriceNeeded_raw(log)']}"},
    {"candidate": "scarcity unclipped (bowling / Indians)", "class": "REDUNDANT (in the strategic range)",
     "evidence": f"identical below 5; with an own need the raw ratio was never below {SAT['mkt_scarcity_bowling']['minRawWhenOwnNeed']} (bowling) / {SAT['mkt_scarcity_indians']['minRawWhenOwnNeed']} (Indians); no bowling / Indians incomplete-XI finding exists"},
]
for x in RED:
    x["classRule"] = RULES.get(x["class"].split(" (")[0], RULES["delta"].get(x["class"].split(" (")[0], ""))

# ── gap-level JSONs ─────────────────────────────────────────────────────────
STATE_INVENTORY = [
    {"information": "opponent bid behaviour on past lots (who bid, how far)", "status": "available internally but not exposed", "where": "resolveLot sees every cap and runs the ladder; history keeps only winner, price, number of raises; obs has pooled price/fair, sold share and rival spend over 20 lots"},
    {"information": "opponent willingness to pay for the CURRENT lot", "status": "genuinely unavailable at the decision point", "where": "each seat's cap is private and chosen simultaneously; using it would be future leakage"},
    {"information": "current demand (which rivals need which role)", "status": "derivable from existing state; partly exposed", "where": "public squads → riv_fills_share (pooled), mkt_scarcity_* (rivals needing in the denominator)"},
    {"information": "opponent remaining purse", "status": "observable (4 statistics); per-rival values derivable", "where": "riv_purse_max/mean/min/std; ctx.rivals[].purseLeft"},
    {"information": "player-level competition (who gains, who can afford)", "status": "derivable; partly exposed", "where": "riv_gain_mean, riv_able_share, riv_capacity_1–3 (saturated)"},
    {"information": "historical bidding behaviour", "status": "pooled over 20 lots exposed; per-rival history derivable from public results; ladder participation internal", "where": "mkt_recent_price_ratio, mkt_recent_sold_share, riv_recent_spend; extras.recent carries winnerTeamId"},
    {"information": "opponent policy identity / type", "status": "available internally (training rooms) — hidden policy identity", "where": "entry.seats[].type; constant in production (fixed lineup)"},
]
b1 = bin_(E_HP, "lots 2–5")
b2 = bin_(E_HP, "lots 6–16")
b3 = bin_(E_HP, "lots 17–32")
early = {"meta": META, "stateInventory": STATE_INVENTORY,
         "lot1": {**G["early"]["lot1"], "finding": "every observation feature AND every candidate (history, breadth, per-rival, capacity) is bit-identical across A / C1 / C4 / S4 at lot 1 for the same entry and learner — nothing in the environment's public or history state differs yet"},
         "separationByLotBin": em,
         "case": {"lot 1": "B — the environment does not yet contain opponent behavioural history; no observation can distinguish the rooms (only hidden policy identity could, and it is excluded)",
                  "lots 2–20": f"C is tested and not supported: before lot 21 the 20-lot window already IS the whole history (cumulative price/fair identical to mkt_recent_price_ratio in {X['early']['cumPF_equals_windowPriceRatio_share']:.0%}); the existing window separates C4 D3QN/QR-DQN rooms from Stage A at {b1['bestExisting']['separation']} already in lots 2–5; the best compact history candidates add ≤ {max(v for v in b1['candidatesWithinBestExistingDeciles'].values() if v is not None):.2f} inside deciles of that feature",
                  "after ~lot 16": f"the existing features separate the rooms at {b3['bestExisting']['separation']} ({b3['bestExisting']['feature']}) in lots 17–32; the frozen policies do not respond to them (Phase 2E.2) — D: a training / generalisation question, not an observation question"},
         "conclusion": "This cannot be solved by observation alone at lot 1, because the required state is not yet determined by the environment. From the second lot the relevant history is already exposed by the existing window; no compact history candidate adds material separation in lots 2–16."}
multi = {"meta": META, **G["multi"], "fineCellCheck": X["multi"],
         "question": "count (A) vs distribution (B) vs identity (C) vs other sufficient statistic (D)",
         "answer": {"A_count": f"public-state counts (aheadPace15, overpay13, starBuyers2) add within-cell separation {MX['aheadPace15']['withinCellSep_C4vsC1']}–{MX['overpay13']['withinCellSep_C4vsC1']}, no more than existing features left out of the same cells (riv_xi_max {X['multi']['existingNotInCells']['riv_xi_max']['withinCellSep_C4vsC1']}), and ≈ 0 where the rival block is aliased",
                    "B_distribution": f"the full sorted purse vector differs in only {G['multi']['distributionCandidate_sortedPurses']['aliasedPairs_vectorDiffers>0.05']:.0%} of aliased C1/C4 pairs",
                    "C_identity": "separates perfectly by construction; hidden policy identity, constant in production — excluded",
                    "D_other": "no lower-dimensional public statistic separates the aliased states: there the one-vs-four difference exists only in the opponents' (hidden) policies and future caps",
                    "minimumStatistic": "none justified — where the current observation aliases one vs four aggressive copies, no public state differs; where it does not alias, existing features already carry comparable separation"}}
kf2 = {(c["k"], c["learner"], c["seat"]): c for c in KO2["cases"]}
kfind = []
for c in G["keeper"]["findings"]:
    m = c["mechanism"]
    mp = c["medianPercentile"]
    kfind.append({
        "k": c["k"], "room": f"{c['learner']} vs 4× {c['opponent']}", "seat": c["seat"], "mechanism": m,
        "1_whatMadeItDangerous": ("deferring an affordable, needed keeper while the remaining keepers were later absorbed by rivals that already held one; own purse near empty"
                                  if m == "M1" else "own purse and overseas slots exhausted early; later keeper bids lost to rivals with large purses"),
        "2_alreadyAvailable": ("own need, affordability and supply: yes; how many rivals COULD compete: yes (public); whether they WILL compete later: no (future behaviour)"
                               if m == "M1" else "yes — own purse / overseas exhaustion is current own state"),
        "3_encoded": ("own need / supply / affordability: yes; competition breadth: yes (riv_able_share, riv_free_slots_share); non-needing propensity: no"
                      if m == "M1" else "yes (self_purse, self_overseas_slots_left, self_max_safe_*, self_reserve_if_passed)"),
        "4_tooCoarse": f"no evidence: candidate counts are {KX['kContestNoNeed']['betweenCellShare']:.0%} determined by existing features (fine cells)",
        "5_oneScalarSeparates": (lambda elev: f"{'in this case elevated (≥ 0.9): ' + ', '.join(elev) if elev else 'no candidate reaches the 0.9 percentile in this case'} — medianPercentile among matched safe decisions: kContestNoNeed {mp.get('kContestNoNeed')}, kHistExtraBuyers {mp.get('kHistExtraBuyers')}, kStockExcess {mp.get('kStockExcess')} (0.5 = typical of safe). Across cases the elevation is not consistent (see consistency).")([n for n in ('kContestNoNeed', 'kHistExtraBuyers', 'kStockExcess') if (mp.get(n) or 0) >= 0.9]),
        "6_transformInstead": "no transformation of an existing feature separates these decisions better",
        "7_cause": ("policy (passed on affordable keepers that filled a visible need) together with the shield's need-based criticality — not a missing observation"
                    if m == "M1" else "policy (early purse / overseas exhaustion, observable) — not a missing observation"),
        "passes": kf2[(c["k"], c["learner"], c["seat"])]["passes"], "passesOnKeepersThatWentUnsold": kf2[(c["k"], c["learner"], c["seat"])]["passesOnKeepersThatWentUnsold"],
        "candidatePercentiles": mp, "candidateMedianValues": c["medianValue"]})
CONS = {}
for mech_ in ("M1", "M2"):
    rows_ = [f for f in G["keeper"]["findings"] if f["mechanism"] == mech_]
    CONS[mech_] = {"cases": len(rows_), **{n: {"atOrAbove0.9": sum(1 for f in rows_ if (f["medianPercentile"].get(n) or 0) >= 0.9), "atOrBelow0.5": sum(1 for f in rows_ if (f["medianPercentile"].get(n) or 1) <= 0.5)}
                                           for n in ("kContestNoNeed", "kHistExtraBuyers", "kStockExcess", "self_purse", "mkt_scarcity_keeper")}}
    CONS[mech_]["self_purse"]["atOrBelow0.1"] = sum(1 for f in rows_ if (f["medianPercentile"].get("self_purse") or 1) <= 0.1)
    CONS[mech_]["mkt_scarcity_keeper"]["atOrBelow0.2"] = sum(1 for f in rows_ if (f["medianPercentile"].get("mkt_scarcity_keeper") or 1) <= 0.2)
keeper = {"meta": META, "consistency": CONS, "population": {k: G["keeper"][k] for k in ("rows", "labelRate_absorbedByNonNeedingRival", "labelRate_wonByAnyRival", "cells", "cellsWithRivalsNeeding")},
          "stateAvsB": G["keeper"]["stateAvsB"], "absorptionByNonNeedingContestants": G["keeper"]["absorptionByNonNeedingContestants"],
          "controlCells": {"bestExisting": G["keeper"]["bestExisting"], "candidates": G["keeper"]["candidates"]},
          "fineCells_plus_rivAble_rivFreeSlots": X["keeper"],
          "findingsMedianPercentileByMechanism": G["keeper"]["findingsMedianPercentileByMechanism"],
          "findings": kfind,
          "stateAvsB_answer": f"State A (≤ 2 non-needing rivals able to buy) vs State B (≥ 5), few rivals needing: absorption by a non-needing rival {G['keeper']['stateAvsB']['absorbedByNonNeeding_A']:.1%} vs {G['keeper']['stateAvsB']['absorbedByNonNeeding_B']:.1%}. The existing riv_able_share already separates A from B at {G['keeper']['stateAvsB']['existingFeatureSepAvsB'][0]['sep']} (riv_capacity_3 {G['keeper']['stateAvsB']['existingFeatureSepAvsB'][1]['sep']}, riv_free_slots_share {G['keeper']['stateAvsB']['existingFeatureSepAvsB'][2]['sep']}). The information is an existing capacity statistic combined with the existing need statistic; no new scalar is required to represent it.",
          "correctionToPhase2E2": "Phase 2E.2 §6 said the observation counts only rivals that NEED a keeper. That holds for the demand features (riv_fills_share, mkt_scarcity_keeper, self_status_keeper) and for the planner and shield, but riv_able_share and riv_free_slots_share count every rival able to pay / with a free slot regardless of need, so the BREADTH of potential keeper competition is observable. What remains unobservable is whether those rivals will actually bid later (future behaviour)."}
sat = {"meta": META, **SAT, "perFeature": {
    "riv_capacity_1–3": {"rawState": "(rival purse − rival's cheapest legal completion) / fair value of the lot, top-3 rivals able to bid",
                        "whySaturating": f"fair values are small relative to purses: median raw capacity {SAT['riv_capacity_1']['rawPercentiles']['50']} / {SAT['riv_capacity_3']['rawPercentiles']['50']} × fair (1st / 3rd); clip at 5",
                        "rawQuantityUseful": f"only where it binds (< ~3 × fair: {SAT['riv_capacity_3']['shareRawBelow2']:.1%} of decisions below 2 for the 3rd rival) — that range is preserved below the clip",
                        "informationDestroyed": f"only values above 5 × fair; the price needed exceeded 5 × fair in {SAT['priceNeededToWin']['share>5xFair']:.2%} of decisions and no sale exceeded 5 × fair",
                        "betterTransform": f"a log transform would track the price needed at {SAT['riv_capacity_1']['trackPriceNeeded_raw(log)']}–{SAT['riv_capacity_3']['trackPriceNeeded_raw(log)']} (clipped: ~0) — no strategic gain",
                        "unnecessary": "largely uninformative in these rooms (constant 5 for 97–99%), but nothing strategic is lost; leaving it unchanged costs nothing", "verdict": "KEEP"},
    "mkt_scarcity_bowling / mkt_scarcity_indians": {"rawState": "suitable players left after this lot / (own need + rivals needing + 1)",
                                                    "whySaturating": f"the pool holds many bowling options / Indians: median raw {SAT['mkt_scarcity_bowling']['rawPercentiles']['50']} / {SAT['mkt_scarcity_indians']['rawPercentiles']['50']}",
                                                    "rawQuantityUseful": "only when it approaches 1 (scarcity); that range is fully preserved below the clip",
                                                    "informationDestroyed": f"only degrees of abundance; with an own need the raw ratio never fell below {SAT['mkt_scarcity_bowling']['minRawWhenOwnNeed']} / {SAT['mkt_scarcity_indians']['minRawWhenOwnNeed']}, and no bowling / Indians incomplete-XI finding exists",
                                                    "betterTransform": "not needed", "unnecessary": "uninformative in these rooms, harmless", "verdict": "KEEP"}}}
dump("candidate-features.json", {"meta": META, "rule": "every candidate is computed only from information available at the exact decision point (current public state + already-resolved lots); evaluation labels (future prices, winners) are used only to TEST candidates, never as candidates", "candidates": cand_json})
dump("redundancy-analysis.json", {"meta": META, "rules": RULES, "candidates": RED})
dump("early-aggression-analysis.json", early)
dump("multi-opponent-gap-analysis.json", multi)
dump("keeper-gap-analysis.json", keeper)
dump("saturation-analysis.json", sat)
dump("counterfactual-tests.json", {"meta": META, **G["counterfactual"],
                                   "reading": "Near-identical current observations (L∞ ≤ 0.05) occur mostly early in the auction. In those pairs every candidate differs in < 2% of pairs (keeper candidates in 0), so no candidate resolves the observed aliasing: where the current 80 features coincide, the public state and public history coincide too."})

# ── minimum delta (§18) and gate ─────────────────────────────────────────────
DELTA = [
    {"item": "Gap 1 — early opponent aggression", "classification": "DO NOT ADD", "candidates": ["rlRivals", "partic5 / partic20", "bids5 / bids20", "cumPF", "cumStarPF"],
     "reason": "lot 1: nothing in the environment differs yet (Case B); lots 2–20: the window already is the full history and the history candidates add little; later: the information is present and unused (Case D). Identity is hidden policy information and constant in production."},
    {"item": "Gap 2 — number / distribution of aggressive opponents", "classification": "DO NOT ADD", "candidates": ["aheadPace15 / aheadPace30", "overpay13 / overpay15 / starBuyers2", "wMaxRivSpend / wBuyers / buyersSoFar", "rival purse vector (sorted, 9 values)", "capGe*"],
     "reason": "no public statistic separates one vs four aggressive copies where the current rival block aliases them; elsewhere the best counts add no more than existing features left out of the same cells"},
    {"item": "Gap 3 — non-needing keeper competition", "classification": "KEEP", "candidates": ["kContestNoNeed / kContestNoNeedFair", "kContestAll", "kStockExcess / kStock2", "kHistExtraBuyers / kW20Extra", "kW20Buys"],
     "reason": "the breadth of potential competition is already carried by riv_able_share / riv_free_slots_share with the need features; the non-needing count is 93% determined by them and separates no better than existing mkt_scarcity_keeper; no candidate separates the dangerous M1 decisions (median percentile ~0.6); the failures trace to policy passes on affordable needed keepers and to the shield's need-based criticality"},
    {"item": "Gap 4 — saturated features (riv_capacity_1–3, mkt_scarcity_bowling, mkt_scarcity_indians)", "classification": "KEEP", "candidates": ["capacity unclipped (log)", "scarcity unclipped (bowling / Indians)"],
     "reason": "the clip removes only values above 5, a range that never mattered in these rooms (price needed > 5 × fair in 0.03% of decisions; no scarcity below 3.7 with an own need)"},
]
dump("minimum-delta.json", {"meta": META, "classificationRules": RULES["delta"], "delta": DELTA,
                            "proposedObservationDelta": {"ADD": [], "TRANSFORM": [], "size": 0},
                            "retainedForReviewerAwareness": [x for x in RED if x["class"] == "PARTIAL NOVELTY"],
                            "note": "No feature index, observation version or hash is assigned; nothing is implemented."})
dump("gap-analysis.json", {"meta": META, "gaps": [
    {"gap": 1, "name": "Early opponent aggression", "case": "B at lot 1; D afterwards (information present in the existing window from lot 2, unused)", "stateInventory": STATE_INVENTORY, "verdict": "DO NOT ADD"},
    {"gap": 2, "name": "Number / distribution of aggressive opponents", "needed": "none of A/B/C/D is justified: the aliased difference lives in hidden policy identity and future caps", "verdict": "DO NOT ADD"},
    {"gap": 3, "name": "Non-needing keeper competition", "stateAvsB": keeper["stateAvsB_answer"], "representation": "an existing capacity statistic (riv_able_share) combined with existing need statistics", "verdict": "KEEP"},
    {"gap": 4, "name": "Saturated features", "verdict": "KEEP (no strategically relevant information is lost)"}]})
dump("decision-gate.json", {"meta": META,
    "result": "NO OBSERVATION CHANGE JUSTIFIED",
    "minimumDelta": {"ADD": 0, "TRANSFORM": 0},
    "summary": "None of the four Phase 2E.2 gaps is closed by a new or transformed observation feature on this evidence. Gap 1 at lot 1 and the aliased part of Gap 2 cannot be solved by observation, because the distinguishing state does not exist in the environment at that point (only hidden policy identity or future actions would separate them). The rest of Gap 1, the breadth part of Gap 3 and the room-level part of Gap 2 are already represented by existing features. The saturation in Gap 4 removes no strategically relevant values.",
    "implication": "The Phase 2E.2 'targeted observation-spec discussion' can conclude with no change to obs-v2 on the current evidence. Evidence suggests the remaining Stage-B problems are training / generalisation and policy-behaviour questions (present signals unused, affordable needed keepers passed); this should be investigated during Stage-B design.",
    "alternativesConsidered": {"MINIMAL DELTA PROPOSED": "rejected — no candidate met the GENUINELY NEW INFORMATION rule; the best partial-novelty candidates (aheadPace15, overpay13, kContestNoNeed, partic20, cumStarPF) add no more separation than existing features and resolve no aliasing",
                               "INCONCLUSIVE": "rejected — the replays cover every gap with large samples, except the 20 keeper findings (467 decisions), where no candidate came close to separating (percentiles ~0.6)"},
    "residualUncertainty": ["The tests measure information on trajectories of the FROZEN Stage-A policies; a Stage-B-trained policy would visit other states, where the partial-novelty candidates might matter more.",
                            "Keeper conclusions rest on 20 findings.",
                            "40 of 500 validation entries; high stratum over-represented.",
                            "Thresholds (L∞ 0.05, candidate difference thresholds, cell resolutions) are conventions; raw values are in raw/gap-stats.json and raw/gap-extra.json."],
    "doesNotDo": "No feature is added, no index or hash assigned, no training or code change made; the shield is not assessed for change."})
print("synthesis written")
