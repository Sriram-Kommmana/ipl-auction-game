"""Phase 2E.3 — minimum observation-gap design: counterfactual statistics.

Joins the Phase 2E.2 replays (exact obs-v2 vectors + hidden state) with the
Phase 2E.3 candidate-signal replay (same rows, same order — verified) and asks,
for every candidate signal, whether it separates the failure states that the
current 80 features cannot. Descriptive statistics only: rank statistics
(|2·AUC − 1|), stratification cells, aliased-pair comparisons. No model is
trained or fitted; obs-v2 is untouched.

    python gapdesign.py     → ml/reports/phase2e3/raw/gap-stats.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import observability as O

ROOT = Path(__file__).resolve().parents[3]
CAND = ROOT / "ml/runs/_2e3/full"
OUT = ROOT / "ml/reports/phase2e3"
r = O.r
FI, FN = O.FI, O.FN


def sep(x1, x0):
    a = O.auc(x1, x0)
    return None if a is None else abs(2 * a - 1)


def load():
    D = O.load()
    C = json.loads((CAND / "cand-episodes.json").read_text(encoding="utf-8"))
    cm = C["meta"]
    c = np.fromfile(CAND / "cand.f32", dtype=np.float32).reshape(-1, len(cm))
    assert len(c) == D["N"], (len(c), D["N"])
    Cc = {k: c[:, i] for i, k in enumerate(cm)}
    for k in ["ep", "who", "seat", "dec", "slNo", "phase"]:
        if not np.array_equal(Cc[k], D["M"][k]):
            raise SystemExit(f"ROW MISALIGNMENT on {k}")
    D["C"] = Cc
    M, obs = D["M"], D["obs"]
    # candidates derived offline from public state already in the Phase 2E.2 rows
    prem = obs[:, FI["mkt_premium_passed"]]
    rp = np.stack([M[f"rp{i}"] for i in range(9)], 1)
    ahead = (1 - rp) - prem[:, None]
    Cc["aheadPace15"] = np.nansum(ahead > 0.15, 1).astype(np.float32)
    Cc["aheadPace30"] = np.nansum(ahead > 0.30, 1).astype(np.float32)
    Cc["wMaxRivSpend"] = M["wMaxRivSpend"]
    Cc["wBuyers"] = M["wBuyers"]
    Cc["rivPurseSorted"] = rp  # 9-d distribution (not a scalar)
    Cc["rivNeedKeeper"] = M["rivNeedKeeper"]
    return D


def cells_of(cols):
    key = np.stack(cols, 1)
    _, cid = np.unique(key, axis=0, return_inverse=True)
    return cid.ravel()


def within_sep(x, y, cid, min_class=5):
    """size-weighted mean of |2·AUC − 1| of x for binary y inside each cell (signed mean also returned)."""
    x = np.asarray(x, float)
    order = np.argsort(cid, kind="mergesort")
    cs = cid[order]
    edges = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1], True])
    tot = w = sgn = 0.0
    ncell = 0
    for g in range(len(edges) - 1):
        idx = order[edges[g]:edges[g + 1]]
        yy = y[idx]
        p, n = yy.sum(), (~yy).sum()
        if p < min_class or n < min_class:
            continue
        a = O.auc(x[idx][yy], x[idx][~yy], max_n=10_000)
        if a is None:
            continue
        ww = len(idx)
        tot += abs(2 * a - 1) * ww
        sgn += (2 * a - 1) * ww
        w += ww
        ncell += 1
    return (tot / w if w else None), (sgn / w if w else None), ncell, int(w)


def between_share(y, cid, min_n=20):
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    cnt = np.bincount(cid[ok])
    keep = ok & (np.bincount(cid[ok], minlength=cid.max() + 1)[cid] >= min_n)
    yk, ck = y[keep], cid[keep]
    if len(yk) < 50 or yk.std() == 0:
        return None
    g = np.bincount(ck, weights=yk) / np.maximum(1, np.bincount(ck))
    return float(((g[ck] - yk.mean()) ** 2).sum() / ((yk - yk.mean()) ** 2).sum())


def dec(x, q=5):
    x = np.asarray(x, float)
    edges = np.nanpercentile(x, np.linspace(0, 100, q + 1)[1:-1])
    return np.searchsorted(edges, x)


# ── GAP 1 — early opponent aggression ────────────────────────────────────
EARLY_CAND = ["partic5", "partic20", "bids5", "bids20", "overpay13", "overpay15", "starBuyers2", "buyersSoFar", "cumStarPF", "wMaxRivSpend", "wBuyers", "aheadPace15", "capGe2"]
LOT_BINS = [(0, 1, "lot 1"), (1, 5, "lots 2–5"), (5, 16, "lots 6–16"), (16, 32, "lots 17–32"), (32, 65, "lots 33–65")]


def early(D):
    M, C, obs = D["M"], D["C"], D["obs"]
    L = np.flatnonzero(D["learner"] & (M["phase"] == 0))
    # future room aggression (evaluation label only): realised star price/fair in lots 33–162 of the same episode
    s = np.flatnonzero(D["learner"] & (M["phase"] == 0) & (M["lotStar"] == 1) & (M["winner"] != -1) & (M["lotIdx"] >= 32) & (M["lotIdx"] < 162))
    fut = np.full(len(D["eps"]), np.nan)
    sums = np.bincount(D["ep"][s], weights=M["priceFair"][s], minlength=len(D["eps"]))
    cnts = np.bincount(D["ep"][s], minlength=len(D["eps"]))
    fut[cnts >= 3] = sums[cnts >= 3] / cnts[cnts >= 3]
    lo, hi = np.nanpercentile(fut, [33.3, 66.7])
    epLab = np.where(fut >= hi, 1, np.where(fut <= lo, 0, -1))
    cond, oal = D["epCond"], D["epOalgo"]
    labels = {
        "future-aggressive vs future-passive room (realised star price/fair, lots 33–162, top vs bottom tercile)": epLab,
        "C4 D3QN/QR-DQN room vs Stage-A room": np.where((cond == "C4") & np.isin(oal, ["d3qn", "qrdqn"]), 1, np.where(cond == "A", 0, -1)),
        "C4 D3QN/QR-DQN room vs C4 PPO room": np.where((cond == "C4") & np.isin(oal, ["d3qn", "qrdqn"]), 1, np.where((cond == "C4") & (oal == "ppo"), 0, -1)),
    }
    out = {"futureStarPriceTerciles": [r(lo, 3), r(hi, 3)], "labels": {}}
    for ln, lab in labels.items():
        rowlab = lab[D["ep"][L]]
        res = []
        for a, b, name in LOT_BINS:
            m = (M["lotIdx"][L] >= a) & (M["lotIdx"][L] < b) & (rowlab >= 0)
            rows = L[m]
            y = rowlab[m] == 1
            if y.sum() < 20 or (~y).sum() < 20:
                continue
            ex = sorted(((f, sep(obs[rows[y], i], obs[rows[~y], i])) for i, f in enumerate(FN)), key=lambda z: -(z[1] or 0))
            cands = {c: r(sep(C[c][rows[y]], C[c][rows[~y]]), 3) for c in EARLY_CAND}
            # novelty: candidate separation inside deciles of the best existing feature
            bi = FI[ex[0][0]]
            cid = dec(obs[rows, bi], 10)
            within = {c: r(within_sep(C[c][rows], y, cid)[0], 3) for c in EARLY_CAND}
            res.append({"bin": name, "rows": int(len(rows)), "bestExisting": {"feature": ex[0][0], "separation": r(ex[0][1], 3)},
                        "top3Existing": [{"feature": f, "separation": r(v, 3)} for f, v in ex[:3]],
                        "candidates": cands, "candidatesWithinBestExistingDeciles": within,
                        "identityOracle_rlRivals": r(sep(M["rlRivals"][rows[y]], M["rlRivals"][rows[~y]]), 3)})
        out["labels"][ln] = res
    # lot 1: is anything at all different between rooms?
    first = L[M["lotIdx"][L] == 0]
    out["lot1"] = {"rows": int(len(first)),
                   "maxAbsDiffOfAnyCandidateAcrossConditionsSameEntryLearner": None}
    key = D["lid"][first] * 1000 + D["k"][first]
    worst = 0.0
    for c in EARLY_CAND + ["kContestAll", "capRaw1", "cumPF"]:
        v = C[c][first]
        for kk in np.unique(key):
            q = v[key == kk]
            q = q[np.isfinite(q)]
            if len(q):
                worst = max(worst, float(q.max() - q.min()))
    obsw = 0.0
    for kk in np.unique(key):
        X = obs[first[key == kk]]
        obsw = max(obsw, float(np.abs(X - X[0]).max()))
    out["lot1"]["maxAbsDiffOfAnyCandidateAcrossConditionsSameEntryLearner"] = r(worst, 6)
    out["lot1"]["maxAbsDiffObsAcrossConditionsSameEntryLearner"] = r(obsw, 6)
    return out


# ── GAP 2 — number / distribution of aggressive opponents ────────────────
MULTI_CAND = ["overpay13", "overpay15", "starBuyers2", "buyersSoFar", "aheadPace15", "aheadPace30", "wMaxRivSpend", "wBuyers", "partic5", "partic20", "bids5", "bids20", "capGe1_5", "capGe2", "capGe3", "capGe5"]


def multi(D):
    M, C, obs = D["M"], D["C"], D["obs"]
    L = np.flatnonzero(D["learner"])
    C1 = L[D["cond"][L] == "C1"]
    C4 = L[D["cond"][L] == "C4"]
    ia, ix = O.match(D, C1, C4, with_opp=True)
    sim = O.similar_state(D, ia, ix)
    ia, ix = ia[sim], ix[sim]
    riv = [FI[f] for f in O.RIVAL_F + O.WINDOW_F]
    drv = np.abs(obs[ix][:, riv] - obs[ia][:, riv]).max(1)
    dfull = np.abs(obs[ix] - obs[ia]).max(1)
    al = drv <= O.DEF["aliasLoose"]
    dbid = M["rlBidders"][ix] - M["rlBidders"][ia]
    out = {"pairs": int(len(ia)), "aliasedRivalBlockLe0.05": int(al.sum()), "aliasedFull80Le0.05": int((dfull <= 0.05).sum()), "candidates": {}}
    # cells over the existing rival block (for redundancy) — all C1 ∪ C4 learner rows
    R = np.r_[C1, C4]
    cid = cells_of([D["pbin"][R], dec(obs[R, FI["riv_purse_mean"]], 10), dec(obs[R, FI["riv_purse_min"]], 10), dec(obs[R, FI["riv_purse_std"]], 10), dec(obs[R, FI["riv_recent_spend"]], 5)])
    ylab = D["cond"][R] == "C4"
    base_within = {}
    for f in ["riv_purse_std", "riv_purse_min", "riv_recent_spend", "mkt_recent_price_ratio", "riv_able_share"]:
        base_within[f] = r(within_sep(obs[R, FI[f]], ylab, cid)[0], 3)
    for c in MULTI_CAND:
        va, vx = C[c][ia], C[c][ix]
        rho = []
        for b in range(len(O.PLABEL)):
            m = D["pbin"][R] == b
            if m.sum() > 500:
                s_ = O.spearman(C[c][R][m], M["rlBidders"][R][m], max_n=80_000)
                if s_ is not None:
                    rho.append((s_, m.sum()))
        out["candidates"][c] = {
            "pairedShareC4Higher": r(np.mean(vx > va), 3), "pairedShareC4Lower": r(np.mean(vx < va), 3),
            "sepC4vsC1_allPairs": r(sep(vx, va), 3), "sepC4vsC1_aliasedRivalBlock": r(sep(vx[al], va[al]), 3),
            "trackRlBiddersWithinProgress": r(np.average([a for a, _ in rho], weights=[w for _, w in rho]), 3) if rho else None,
            "betweenCellShareGivenRivalBlock": r(between_share(C[c][R], cid), 3),
            "withinCellSepC4vsC1": r(within_sep(C[c][R], ylab, cid)[0], 3),
        }
    # distribution candidate: sorted 9-purse vector
    rp = C["rivPurseSorted"]
    dvec = np.nanmax(np.abs(rp[ix] - rp[ia]), 1)
    out["distributionCandidate_sortedPurses"] = {"aliasedPairs_vectorDiffers>0.05": r(np.mean(dvec[al] > 0.05), 3), "allPairs_vectorDiffers>0.05": r(np.mean(dvec > 0.05), 3)}
    out["existingWithinCellSepC4vsC1"] = base_within
    out["existingBestTrackRlBidders"] = 0.251
    out["identityOracle"] = "C1 vs C4 differ by construction in the number of RL-controlled seats (1 vs 4): hidden policy identity, constant in production (5 RL + 4 rule seats)"
    out["rlBiddersDiffer2plusShare"] = r(np.mean(np.abs(dbid) >= 2), 3)
    return out


# ── GAP 3 — non-needing keeper competition ───────────────────────────────
KEEPER_CAND = ["kContestAll", "kContestNoNeed", "kContestNoNeedFair", "kContestNeed", "kStockExcess", "kStock2", "kHistExtraBuyers", "kW20Extra", "kW20Buys", "capGe2"]


def keeper(D):
    M, C, obs = D["M"], D["C"], D["obs"]
    K = np.flatnonzero((M["lotRole"] == 3) & (M["keeperNeed"] > 0))
    y1 = C["kWinnerType"][K] == 2  # absorbed by a rival that already held a keeper
    yRival = np.isin(C["kWinnerType"][K], [1, 2])
    # controls named by the spec: own purse, own keeper need (=1 here), keeper price/fair, progress, keeper supply
    cid = cells_of([D["pbin"][K], dec(obs[K, FI["self_purse"]], 5), dec(obs[K, FI["lot_fair_value"]], 4),
                    np.round((obs[K, FI["mkt_supply_Wi"]] + obs[K, FI["mkt_supply_Wo"]]) * 40).astype(int) // 3])
    cid2 = cells_of([cid, np.round(obs[K, FI["riv_fills_share"]] * 9).astype(int)])  # + rivals needing (in obs)
    out = {"rows": int(len(K)), "labelRate_absorbedByNonNeedingRival": r(y1.mean(), 3), "labelRate_wonByAnyRival": r(yRival.mean(), 3),
           "cells": int(len(np.unique(cid))), "cellsWithRivalsNeeding": int(len(np.unique(cid2))), "candidates": {}, "existing": {}}
    for f in FN:
        s1 = within_sep(obs[K, FI[f]], y1, cid2)[0]
        out["existing"][f] = r(s1, 3)
    ex_sorted = sorted(((f, v) for f, v in out["existing"].items() if v is not None), key=lambda z: -z[1])
    out["bestExisting"] = [{"feature": f, "withinCellSep": v} for f, v in ex_sorted[:6]]
    for c in KEEPER_CAND:
        s1, sg, nc, nw = within_sep(C[c][K], y1, cid2)
        out["candidates"][c] = {"withinCellSep_absorbedByNonNeeding": r(s1, 3), "signed": r(sg, 3), "cells": nc,
                                "betweenCellShareGivenControls+rivalsNeeding": r(between_share(C[c][K], cid2), 3),
                                "withinCellSep_wonByAnyRival": r(within_sep(C[c][K], yRival, cid2)[0], 3)}
    # State A vs State B (few rivals NEED a keeper; few vs many can compete)
    few_need = M["rivNeedKeeper"][K] <= 1
    nn = C["kContestNoNeed"][K]
    A = few_need & (nn <= 2)
    B = few_need & (nn >= 5)
    out["stateAvsB"] = {"definition": "few rivals need a keeper (≤ 1); State A: ≤ 2 non-needing rivals able to buy the lot at base; State B: ≥ 5",
                        "rowsA": int(A.sum()), "rowsB": int(B.sum()),
                        "absorbedByNonNeeding_A": r(y1[A].mean(), 3), "absorbedByNonNeeding_B": r(y1[B].mean(), 3),
                        "existingFeatureSepAvsB": sorted([{"feature": f, "sep": r(sep(obs[K[B], i], obs[K[A], i]), 3)} for i, f in enumerate(FN)], key=lambda z: -(z["sep"] or 0))[:6]}
    # rate curve: absorption by number of non-needing contestants, within few-need rows
    out["absorptionByNonNeedingContestants"] = [{"kContestNoNeed": int(v), "rows": int(((nn == v) & few_need).sum()), "rate": r(y1[(nn == v) & few_need].mean(), 3) if ((nn == v) & few_need).sum() >= 50 else None} for v in range(10)]
    # the 20 findings: candidate percentiles among matched safe references (as Phase 2E.2)
    KO = json.loads((ROOT / "ml/reports/phase2e2/keeper-observability.json").read_text(encoding="utf-8"))
    epmap = {(e["cond"], e["learner"], e["opp"], e["k"]): e["ep"] for e in D["eps"]}
    safe = np.flatnonzero((M["lotRole"] == 3) & (M["keeperNeed"] > 0) & (D["seatKeepers"] > 0))
    cases = []
    for cse in KO["cases"]:
        e = epmap[("C4", cse["learner"], cse["opponent"], cse["k"])]
        rows = np.flatnonzero((D["ep"] == e) & (M["seat"] == cse["seat"]) & (M["lotRole"] == 3) & (M["keeperNeed"] > 0))
        pc = defaultdict(list)
        vals = defaultdict(list)
        for i in rows:
            ref = safe[(M["phase"][safe] == M["phase"][i]) & (np.abs(M["progress"][safe] - M["progress"][i]) <= 0.05) & (M["who"][safe] == M["who"][i]) & (D["ep"][safe] != e)]
            if len(ref) < 30:
                continue
            for c in ["kContestNoNeed", "kContestNoNeedFair", "kHistExtraBuyers", "kStockExcess", "kContestAll"]:
                v = C[c][ref]
                pc[c].append(float(((v < C[c][i]).sum() + 0.5 * (v == C[c][i]).sum()) / len(v)))
                vals[c].append(float(C[c][i]))
            for f in ["mkt_scarcity_keeper", "riv_fills_share", "riv_free_slots_share", "self_purse"]:
                v = obs[ref, FI[f]]
                pc[f].append(float(((v < obs[i, FI[f]]).sum() + 0.5 * (v == obs[i, FI[f]]).sum()) / len(v)))
        cases.append({"k": cse["k"], "learner": cse["learner"], "opponent": cse["opponent"], "seat": cse["seat"], "mechanism": cse["mechanism"],
                      "medianPercentile": {c: r(np.median(v), 3) for c, v in pc.items()}, "medianValue": {c: r(np.median(v), 2) for c, v in vals.items()}})
    out["findings"] = cases
    agg = defaultdict(list)
    for cse in cases:
        for c, v in cse["medianPercentile"].items():
            agg[(cse["mechanism"], c)].append(v)
    out["findingsMedianPercentileByMechanism"] = {f"{m}:{c}": r(np.median(v), 3) for (m, c), v in agg.items()}
    return out


# ── GAP 4 — saturated features ───────────────────────────────────────────
def saturation(D):
    M, C, obs = D["M"], D["C"], D["obs"]
    L = np.flatnonzero(D["learner"])
    out = {}
    need_raw = M["othCap1"][L]
    out["priceNeededToWin"] = {"share>5xFair": r(np.nanmean(need_raw > 5), 4), "share>3xFair": r(np.nanmean(need_raw > 3), 4), "p99": r(np.nanpercentile(need_raw, 99), 2)}
    sold = L[np.isfinite(M["priceFair"][L])]
    out["salePriceToFair"] = {"share>5": r(np.mean(M["priceFair"][sold] > 5), 4), "share>3": r(np.mean(M["priceFair"][sold] > 3), 4), "p99": r(np.percentile(M["priceFair"][sold], 99), 2)}
    for i, (c, f) in enumerate([("capRaw1", "riv_capacity_1"), ("capRaw2", "riv_capacity_2"), ("capRaw3", "riv_capacity_3")]):
        raw = C[c][L]
        rho_raw, rho_clip, w = [], [], []
        for b in range(len(O.PLABEL)):
            m = D["pbin"][L] == b
            if m.sum() < 500:
                continue
            a1 = O.spearman(np.log1p(raw[m]), np.minimum(need_raw[m], 50), max_n=80_000)
            a2 = O.spearman(obs[L[m], FI[f]], np.minimum(need_raw[m], 50), max_n=80_000)
            if a1 is not None:
                rho_raw.append(a1); w.append(m.sum()); rho_clip.append(a2 if a2 is not None else 0.0)
        # decisions where the price that mattered could exceed what the clipped value can express
        out[f] = {"rawPercentiles": {p: r(np.percentile(raw, p), 2) for p in (1, 5, 25, 50, 75, 95)}, "shareAtClip": r(np.mean(raw >= 5), 3),
                  "shareRawBelow2": r(np.mean(raw < 2), 3),
                  "trackPriceNeeded_raw(log)": r(np.average(rho_raw, weights=w), 3) if w else None,
                  "trackPriceNeeded_clipped": r(np.average(rho_clip, weights=w), 3) if w else None,
                  "shareClippedWhilePriceNeeded>5": r(np.mean((raw >= 5) & (need_raw > 5)), 4)}
    for c, f, nk in [("scarBowlRaw", "mkt_scarcity_bowling", "needBowl"), ("scarIndRaw", "mkt_scarcity_indians", "needInd"), ("scarKeeperRaw", "mkt_scarcity_keeper", None)]:
        raw = C[c][L]
        need = C[nk][L] if nk else M["keeperNeed"][L]
        m = need > 0
        out[f] = {"rawPercentiles": {p: r(np.percentile(raw, p), 2) for p in (1, 5, 25, 50, 75, 95)}, "shareAtClip": r(np.mean(raw >= 5), 3),
                  "decisionsWithOwnNeed": int(m.sum()), "shareAtClipWhenOwnNeed": r(np.mean(raw[m] >= 5), 3) if m.any() else None,
                  "minRawWhenOwnNeed": r(raw[m].min(), 2) if m.any() else None,
                  "shareRawBelow2WhenOwnNeed": r(np.mean(raw[m] < 2), 4) if m.any() else None}
    # did any Phase 2E.0 finding involve bowling or Indians? (incomplete-XI findings were all keeper)
    F = json.loads((ROOT / "ml/reports/phase2e0/findings.json").read_text(encoding="utf-8"))["findings"]
    out["phase2e0FindingsByMissingRequirement"] = "all 20 findings are missing-keeper XIs (Phase 2E.1 keeper-failures.json)"
    return out


# ── §17 — counterfactual aliasing test: current 80 vs 80 + candidate ─────
CF_CAND = {"kContestNoNeed": 2, "kContestNoNeedFair": 2, "kHistExtraBuyers": 1, "kStockExcess": 2, "kContestAll": 2,
           "overpay13": 2, "starBuyers2": 2, "aheadPace15": 2, "wMaxRivSpend": 0.1, "partic5": 1.0, "partic20": 0.5, "bids20": 5.0, "capGe2": 2}


def cf_tests(D, max_group=420):
    M, C, obs = D["M"], D["C"], D["obs"]
    rows = np.arange(D["N"])
    key = (D["k"].astype(np.int64) * 2 + M["phase"].astype(np.int64)) * 1024 + M["slNo"].astype(np.int64)
    order = np.argsort(key, kind="mergesort")
    ks = key[order]
    edges = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
    rng = np.random.RandomState(7)
    A, B, Dd = [], [], []
    for g in range(len(edges) - 1):
        idx = order[edges[g]:edges[g + 1]]
        if len(idx) < 2:
            continue
        if len(idx) > max_group:
            idx = np.sort(rng.choice(idx, max_group, replace=False))
        X = obs[idx]
        d = np.abs(X[:, None, :] - X[None, :, :]).max(2)
        iu = np.triu_indices(len(idx), 1)
        m = d[iu] <= 0.05
        a, b = idx[iu[0][m]], idx[iu[1][m]]
        ok = ~((D["ep"][a] == D["ep"][b]) & (M["seat"][a] == M["seat"][b]))
        A.append(a[ok]); B.append(b[ok]); Dd.append(d[iu][m][ok])
    a, b, dd = np.concatenate(A), np.concatenate(B), np.concatenate(Dd)
    isK = (M["lotRole"][a] == 3) & (M["keeperNeed"][a] > 0) & (M["keeperNeed"][b] > 0)
    outcomes = {
        "keeper lot absorbed by a non-needing rival (one yes, one no)": (isK, (C["kWinnerType"][a] == 2) != (C["kWinnerType"][b] == 2)),
        "price needed to win differs ≥ 0.5 × fair": (np.ones(len(a), bool), np.abs(D["need"][a] - D["need"][b]) >= 0.5),
        "RL rivals bidding differ ≥ 2": (np.ones(len(a), bool), np.abs(M["rlBidders"][a] - M["rlBidders"][b]) >= 2),
        "lot won by this seat in one, not the other": (np.ones(len(a), bool), (M["winner"][a] == 0) != (M["winner"][b] == 0)),
    }
    res = {"definition": {"similarPairs": "pairs of captured decisions on the same entry / phase / lot, from different (episode, seat), with the current 80 features within L∞ ≤ 0.05",
                          "test": "for each candidate: share of similar pairs where the candidate differs by ≥ its threshold; the outcome-difference rate when it differs vs when it does not (lift = ratio)"},
           "similarPairs": int(len(a)), "similarKeeperPairs": int(isK.sum()), "tests": {}}
    for oname, (pop, ydiff) in outcomes.items():
        t = {"pairs": int(pop.sum()), "outcomeDiffRate": r(ydiff[pop].mean(), 4) if pop.any() else None, "candidates": {}}
        for c, thr in CF_CAND.items():
            dc = np.abs(C[c][a] - C[c][b])
            dcd = np.nan_to_num(dc, nan=0.0) >= thr
            p1, p0 = pop & dcd, pop & ~dcd
            t["candidates"][c] = {"threshold": thr, "shareCandidateDiffers": r(dcd[pop].mean(), 4) if pop.any() else None,
                                  "outcomeDiffRate_candidateDiffers": r(ydiff[p1].mean(), 4) if p1.sum() >= 30 else None,
                                  "outcomeDiffRate_candidateSame": r(ydiff[p0].mean(), 4) if p0.sum() >= 30 else None,
                                  "lift": r(ydiff[p1].mean() / max(1e-9, ydiff[p0].mean()), 2) if p1.sum() >= 30 and p0.sum() >= 30 and ydiff[p0].mean() > 0 else None,
                                  "pairsCandidateDiffers": int(p1.sum())}
        res["tests"][oname] = t
    return res


def main():
    D = load()
    print("rows", D["N"], "aligned")
    out = {"meta": {"phase": "2E.3", "rows": int(D["N"]), "note": "descriptive statistics only; no model trained or fitted; obs-v2 untouched"}}
    out["early"] = early(D); print("early done")
    out["multi"] = multi(D); print("multi done")
    out["keeper"] = keeper(D); print("keeper done")
    out["saturation"] = saturation(D); print("saturation done")
    out["counterfactual"] = cf_tests(D); print("cf done")
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    (OUT / "raw" / "gap-stats.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("GAPSTATSDONE")


if __name__ == "__main__":
    main()
