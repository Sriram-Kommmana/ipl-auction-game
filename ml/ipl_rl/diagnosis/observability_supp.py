"""Phase 2E.2 — supplementary descriptive statistics (analysis only).

    python observability_supp.py      → ml/reports/phase2e2/raw/supplement.json
Star timing, feature saturation at the clip bounds, forced-bid predictability
(rank statistic, no fitting), keeper-failure summaries by mechanism, and the
share of hidden rival keeper state that lies BETWEEN cells of the observed
keeper features (a descriptive variance decomposition — no model is fitted).
"""
import json
from collections import defaultdict

import numpy as np

import observability as O

D = O.load()
M, obs, FI, FN = D["M"], D["obs"], O.FI, O.FN
L = np.flatnonzero(D["learner"])
out = {}

# 1 · star timing (main-round order of each replayed entry)
stars = defaultdict(set)
allr = np.flatnonzero((M["lotStar"] == 1) & (M["phase"] == 0))
for k, li in zip(D["k"][allr], M["lotIdx"][allr]):
    stars[int(k)].add(int(li))
edges = [0, 1, 5, 16, 32, 65, 97, 162, 243, 323]
share = np.zeros(len(edges) - 1)
for v in stars.values():
    v = np.array(sorted(v))
    share += np.histogram(v, bins=edges)[0] / len(v)
out["starTiming"] = {"definition": "share of each entry's star lots (rating ≥ 90) by main-round lot index, averaged over the 40 replayed entries",
                     "bins": [f"lots {a + 1}–{b}" for a, b in zip(edges[:-1], edges[1:])], "share": [O.r(x / len(stars), 3) for x in share],
                     "cumulative": [O.r(x, 3) for x in np.cumsum(share / len(stars))], "starsPerEntry": O.r(np.mean([len(v) for v in stars.values()]), 1)}

# 2 · saturation at the global clip bounds and the internal caps
X = obs[L[:: max(1, len(L) // 400_000)]]
sat = []
for i, n in enumerate(FN):
    x = X[:, i]
    hi = (x >= 5 - 1e-6).mean()
    lo = (x <= -1 + 1e-6).mean()
    cap = (x >= 4 - 1e-6).mean() if n == "lot_fair_value" else None
    if hi > 0.01 or lo > 0.01 or (cap or 0) > 0.01:
        sat.append({"feature": n, "atUpperClip5": O.r(hi, 3), "atLowerClipMinus1": O.r(lo, 3), **({"atInternalCap4": O.r(cap, 3)} if cap is not None else {})})
bins = []
for b, lab in enumerate(O.PLABEL):
    m = D["pbin"][L] == b
    x = obs[L[m]]
    bins.append({"progress": lab, "riv_capacity_1_at5": O.r((x[:, FI["riv_capacity_1"]] >= 5 - 1e-6).mean(), 3),
                 "riv_capacity_3_at5": O.r((x[:, FI["riv_capacity_3"]] >= 5 - 1e-6).mean(), 3),
                 "riv_able_share_eq1": O.r((x[:, FI["riv_able_share"]] >= 1 - 1e-6).mean(), 3)})
out["saturation"] = {"definition": "share of learner observations at the obs-v2 clip bounds [-1, 5] (all conditions)", "features": sat, "rivalCapacityByProgress": bins}

# 3 · forced bids: how well single features separate forced from free decisions (|2·AUC − 1|)
fo = {}
for hid in ["forced", "forcedKeeper", "finalPath"]:
    y = M[hid][L] == 1
    rows = []
    for i, n in enumerate(FN):
        a = O.auc(obs[L[y], i], obs[L[~y], i])
        if a is not None:
            rows.append((n, abs(2 * a - 1), a))
    rows.sort(key=lambda z: -z[1])
    fo[hid] = {"positives": int(y.sum()), "decisions": int(len(y)), "top": [{"feature": n, "separation": O.r(s, 3), "auc": O.r(a, 3)} for n, s, a in rows[:6]]}
kl = L[(M["lotRole"][L] == 3) & (M["keeperNeed"][L] > 0)]
y = M["forcedKeeper"][kl] == 1
rows = []
for i, n in enumerate(FN):
    a = O.auc(obs[kl[y], i], obs[kl[~y], i])
    if a is not None:
        rows.append((n, abs(2 * a - 1), a))
rows.sort(key=lambda z: -z[1])
fo["forcedKeeper_withinKeeperNeedDecisions"] = {"definition": "learner decisions on keeper lots while needing a keeper: forced keeper bid vs not", "positives": int(y.sum()), "decisions": int(len(y)),
                                                "top": [{"feature": n, "separation": O.r(s_, 3), "auc": O.r(a, 3)} for n, s_, a in rows[:8]]}
out["forcedPredictability"] = fo

# 4 · keeper failures by mechanism (from keeper-observability.json)
KO = json.loads((O.OUT / "keeper-observability.json").read_text(encoding="utf-8"))
mech = defaultdict(lambda: defaultdict(list))
for c in KO["cases"]:
    g = c["mechanism"]
    for d in c["decisions"]:
        affordable = d["maxSafeBid"] >= d["base"]
        fills = d["obs"]["self_fills_keeper"] == 1
        mech[g]["decisions"].append(1)
        mech[g]["pass"].append(d["passed"])
        mech[g]["passAffordableFills"].append(d["passed"] and affordable and fills)
        mech[g]["passAffordableFillsUnsold"].append(d["passed"] and affordable and fills and d["outcome"] == "unsold")
        mech[g]["forcedLost"].append(d["forced"] and d["outcome"] != "won")
        mech[g]["plannerStatusNeed"].append(d["obs"]["self_status_keeper"] == round(1 / 3, 4))
        mech[g]["scarcity"].append(d["obs"]["mkt_scarcity_keeper"])
        mech[g]["ownPurse"].append(d["obs"]["self_purse"])
        mech[g]["slotsLeft"].append(d["obs"]["self_slots_left"])
        mech[g]["overseasSlotsLeft"].append(d["obs"]["self_overseas_slots_left"])
        mech[g]["rivFillsShare"].append(d["obs"]["riv_fills_share"])
        mech[g]["rivNeedKeeper"].append(d["hidden"]["rivNeedKeeper"])
        mech[g]["rivNoNeedCanBuy"].append(d["hidden"]["rivNoNeedCanBuy"])
        mech[g]["rivKeepersMax"].append(d["hidden"]["rivKeepersMax"])
        mech[g]["shieldSafe"].append(d["shieldKeeperState"] == "SAFE")
        if d["forced"] and d["outcome"] != "won":
            mech[g]["forcedLostRivalNeedKeeper"].append(d["hidden"]["rivNeedKeeper"])
            mech[g]["forcedLostRivalNoNeedCanBuy"].append(d["hidden"]["rivNoNeedCanBuy"])
km = {}
for g, v in mech.items():
    km[g] = {"cases": sum(1 for c in KO["cases"] if c["mechanism"] == g), "keeperDecisionsWhileNeeding": len(v["decisions"]),
             "passes": int(np.sum(v["pass"])), "passesOnAffordableKeeperThatFillsNeed": int(np.sum(v["passAffordableFills"])),
             "ofWhichWentUnsold": int(np.sum(v["passAffordableFillsUnsold"])), "forcedBidsLost": int(np.sum(v["forcedLost"])),
             "shieldKeeperStateSafeShare": O.r(np.mean(v["shieldSafe"]), 3), "plannerStatusNEEDShare": O.r(np.mean(v["plannerStatusNeed"]), 3),
             "median": {k: O.r(np.nanmedian(np.array(v[k], dtype=float)), 3) for k in ["scarcity", "ownPurse", "slotsLeft", "overseasSlotsLeft", "rivFillsShare", "rivNeedKeeper", "rivNoNeedCanBuy", "rivKeepersMax"]},
             "atLostForcedBids": {"meanRivalsNeedingKeeper": O.r(np.mean(v["forcedLostRivalNeedKeeper"]), 2) if v["forcedLostRivalNeedKeeper"] else None,
                                  "meanRivalsNotNeedingButAbleToBuy": O.r(np.mean(v["forcedLostRivalNoNeedCanBuy"]), 2) if v["forcedLostRivalNoNeedCanBuy"] else None}}
out["keeperByMechanism"] = km

# 5 · hidden rival keeper state: share of variance BETWEEN cells of the observed keeper features
kr = np.flatnonzero((M["lotRole"] == 3) & (M["keeperNeed"] > 0))
cellkey = np.stack([
    D["pbin"][kr].astype(int),
    np.round((obs[kr, FI["mkt_supply_Wi"]] + obs[kr, FI["mkt_supply_Wo"]]) * 40).astype(int) // 2,  # keepers still to come, in pairs
    np.round(obs[kr, FI["riv_fills_share"]] * 9).astype(int),
    np.round(obs[kr, FI["self_status_keeper"]] * 3).astype(int),
    np.clip(np.floor(obs[kr, FI["mkt_scarcity_keeper"]] * 2), -2, 10).astype(int),
    np.round(obs[kr, FI["riv_purse_mean"]] * 10).astype(int),
], 1)
_, cid = np.unique(cellkey, axis=0, return_inverse=True)
cid = cid.ravel()
counts = np.bincount(cid)
keep = counts[cid] >= 20


def between_share(y):
    y = y[keep].astype(float)
    c = cid[keep]
    g = np.bincount(c, weights=y) / np.maximum(1, np.bincount(c))
    tot = ((y - y.mean()) ** 2).sum()
    return O.r(((g[c] - y.mean()) ** 2).sum() / tot, 3) if tot > 0 else None


out["keeperHiddenVsObservedCells"] = {
    "definition": "keeper-lot decisions of seats still needing a keeper (learner + RL-opponent rows). Cells = progress bin × keepers still to come (pairs) × rival fills share (ninths) × own planner keeper status × keeper scarcity (halves) × mean rival purse (tenths). "
                  "Share of each hidden variable's variance that lies between cells (1 = fully determined by the cell, 0 = invisible within a cell). Cells with < 20 rows dropped.",
    "rows": int(keep.sum()), "rowsTotal": int(len(kr)), "cells": int((counts >= 20).sum()),
    "betweenCellShare": {"rivals NEEDING a keeper (control: in the observation)": between_share(M["rivNeedKeeper"][kr]),
                         "max keepers held by one rival": between_share(M["rivKeepersMax"][kr]),
                         "rivals holding ≥ 2 keepers": between_share(M["rivKeep2"][kr]),
                         "rivals holding ≥ 3 keepers": between_share(M["rivKeep3"][kr]),
                         "rivals NOT needing a keeper but able to buy one": between_share(M["rivNoNeedCanBuy"][kr]),
                         "keepers bought by rivals in the last 20 lots": between_share(M["wRivKeepers"][kr])}}

# 6 · realised star prices early in the auction, by condition
sp = {}
for c in ["A", "C1", "C4", "S4"]:
    X_ = L[(D["cond"][L] == c) & (M["phase"][L] == 0) & (M["lotStar"][L] == 1) & (M["winner"][L] != -1) & (M["progress"][L] < 0.3)]
    sp[c] = {"meanPriceToFair": O.r(np.nanmean(M["priceFair"][X_]), 3), "starLotsObserved": int(len(X_))}
out["earlyStarPriceByCondition"] = {"definition": "price / fair value of star lots sold in the first 30% of the main round (learner-observed lots)", "byCondition": sp}
out["note"] = "descriptive only; no model trained or fitted"
(O.OUT / "raw" / "supplement.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=1, ensure_ascii=False)[:6000])
