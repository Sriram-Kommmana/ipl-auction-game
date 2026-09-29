"""Phase 2E.3 — conditional redundancy checks for the surviving candidates
(descriptive only; no fitting). Adds the existing features that could already
carry each candidate to the stratification cells and asks whether the candidate
still separates the outcome inside those finer cells.

    python gap_extra.py     → ml/reports/phase2e3/raw/gap-extra.json
"""
import json

import numpy as np

import gapdesign as G
import observability as O

FI, FN, r = O.FI, O.FN, O.r
D = G.load()
M, C, obs = D["M"], D["C"], D["obs"]
out = {}

# ── keeper: does the non-needing contestant count add to riv_able_share / riv_free_slots_share / riv_fills_share?
K = np.flatnonzero((M["lotRole"] == 3) & (M["keeperNeed"] > 0))
y1 = C["kWinnerType"][K] == 2
ninth = lambda f: np.round(obs[K, FI[f]] * 9).astype(int)
base = [D["pbin"][K], G.dec(obs[K, FI["self_purse"]], 5), G.dec(obs[K, FI["lot_fair_value"]], 4),
        np.round((obs[K, FI["mkt_supply_Wi"]] + obs[K, FI["mkt_supply_Wo"]]) * 40).astype(int) // 3, ninth("riv_fills_share")]
fine = G.cells_of(base + [ninth("riv_able_share"), ninth("riv_free_slots_share")])
res = {"rows": int(len(K)), "cells": int(len(np.unique(fine))), "candidates": {}, "existingOutsideCells": {}}
for c in ["kContestNoNeed", "kContestNoNeedFair", "kContestAll", "kHistExtraBuyers", "kW20Buys"]:
    s, sg, nc, nw = G.within_sep(C[c][K], y1, fine)
    res["candidates"][c] = {"betweenCellShare": r(G.between_share(C[c][K], fine), 3), "withinCellSep": r(s, 3), "signed": r(sg, 3), "cellsUsed": nc}
for f in ["mkt_scarcity_keeper", "mkt_equivalent_left", "riv_gain_mean", "riv_purse_mean", "riv_capacity_3", "mkt_recent_price_ratio", "self_xi_gain"]:
    res["existingOutsideCells"][f] = r(G.within_sep(obs[K, FI[f]], y1, fine)[0], 3)
# structural overlap: riv_able_share × 9 vs contestant counts
able9 = np.round(obs[K, FI["riv_able_share"]] * 9)
res["rivAbleCount_vs_kContestAll_spearman"] = r(O.spearman(able9, C["kContestAll"][K]), 3)
res["rivAbleCount_vs_kContestNoNeedFair_spearman"] = r(O.spearman(able9, C["kContestNoNeedFair"][K]), 3)
res["shareRowsRivAbleCountEqualsContestAll"] = r(np.mean(able9 == C["kContestAll"][K]), 3)
# dangerous seats (finished without a keeper) vs safe, inside the fine cells
yD = D["seatKeepers"][K] == 0
res["dangerousSeatRows"] = int(yD.sum())
res["dangerousVsSafe_withinFineCells"] = {c: r(G.within_sep(C[c][K], yD, fine, min_class=3)[0], 3) for c in ["kContestNoNeed", "kHistExtraBuyers", "kStockExcess"]}
res["dangerousVsSafe_withinFineCells_existing"] = {f: r(G.within_sep(obs[K, FI[f]], yD, fine, min_class=3)[0], 3) for f in ["self_purse", "mkt_scarcity_keeper", "self_overseas_slots_left", "self_slots_left"]}
out["keeper"] = res

# ── multi: add mkt_recent_price_ratio and riv_able_share to the rival-block cells
L = np.flatnonzero(D["learner"])
R = L[np.isin(D["cond"][L], ["C1", "C4"])]
cid = G.cells_of([D["pbin"][R], G.dec(obs[R, FI["riv_purse_mean"]], 10), G.dec(obs[R, FI["riv_purse_min"]], 10), G.dec(obs[R, FI["riv_purse_std"]], 10),
                  G.dec(obs[R, FI["riv_recent_spend"]], 5), G.dec(obs[R, FI["mkt_recent_price_ratio"]], 5), np.round(obs[R, FI["riv_able_share"]] * 9).astype(int)])
yC4 = D["cond"][R] == "C4"
yB = M["rlBidders"][R] >= 3
mres = {"rows": int(len(R)), "cells": int(len(np.unique(cid))), "candidates": {}}
for c in ["aheadPace15", "aheadPace30", "starBuyers2", "overpay13", "overpay15", "wMaxRivSpend", "partic20", "capGe2"]:
    mres["candidates"][c] = {"betweenCellShare": r(G.between_share(C[c][R], cid), 3),
                             "withinCellSep_C4vsC1": r(G.within_sep(C[c][R], yC4, cid)[0], 3),
                             "withinCellSep_rlBidders≥3": r(G.within_sep(C[c][R], yB, cid)[0], 3)}
mres["existingNotInCells"] = {f: {"withinCellSep_C4vsC1": r(G.within_sep(obs[R, FI[f]], yC4, cid)[0], 3), "withinCellSep_rlBidders≥3": r(G.within_sep(obs[R, FI[f]], yB, cid)[0], 3)}
                              for f in ["riv_xi_max", "riv_xi_mean", "self_pace_gap", "mkt_recent_sold_share", "riv_purse_max", "riv_gain_mean"]}
mres["hiddenIdentity_withinCellSep_C4vsC1_note"] = "rlRivals (count of RL-controlled seats) separates C1 vs C4 perfectly by construction — hidden policy identity, constant in production"
out["multi"] = mres

# ── early: is anything beyond the 20-lot window available before lot 20? (window == full history there)
Lm = np.flatnonzero(D["learner"] & (M["phase"] == 0) & (M["lotIdx"] < 20) & (M["lotIdx"] >= 1))
out["early"] = {"rows": int(len(Lm)),
                "cumPF_equals_windowPriceRatio_share": r(np.mean(np.abs(C["cumPF"][Lm] - obs[Lm, FI["mkt_recent_price_ratio"]]) < 1e-4), 4),
                "note": "before lot 21 the 20-lot window IS the whole public history; any cumulative-history statistic of price/fair is identical to mkt_recent_price_ratio there"}
(G.OUT / "raw" / "gap-extra.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=1, ensure_ascii=False))
