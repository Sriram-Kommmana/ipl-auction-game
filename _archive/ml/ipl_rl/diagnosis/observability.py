"""Phase 2E.2 — observation sufficiency / counterfactual diagnosis (analysis only).

Reads the digest-verified read-only replays written by replay_obs.mjs
(ml/runs/_2e2/full: obs.f32 = the exact 80-feature observations, meta.f32 =
decision context + hidden state + lot outcome) and the frozen observation
inventory. No model is trained, fitted or evaluated on new inputs; every
statistic is a descriptive comparison (differences, ranks, distances).

    python observability.py            → ml/reports/phase2e2/*.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "ml/runs/_2e2/full"
OUT = ROOT / "ml/reports/phase2e2"
INV = json.loads((OUT / "observation-inventory.json").read_text(encoding="utf-8"))
FN = [f["name"] for f in INV["features"]]
FI = {n: i for i, n in enumerate(FN)}
CHANNEL = np.array([f["opponentChannel"] for f in INV["features"]])
BLOCK = np.array([f["block"] for f in INV["features"]])
SOURCE = np.array([f["source"] for f in INV["features"]])
ALGOS = ["ppo", "a2c", "d3qn", "qrdqn", "es"]
# Phase 2E.1 measured opponent effect (C1 mean transfer ΔXI): QR-DQN −3.31, D3QN −3.21 (largest), PPO −1.87 (smallest).
HIGH_PRESSURE = {"d3qn", "qrdqn"}
LOW_PRESSURE = {"ppo"}

# ── transparent thresholds (all reported in every JSON) ─────────────────
DEF = {
    "aliasTight": 0.02,   # L∞ over all 80 normalised features: every feature within 0.02
    "aliasLoose": 0.05,
    "featureDiffers": 0.01,  # a feature "differs" between two observations when |Δ| > 0.01
    "materialPrice": 0.5,  # price needed to win (max other cap / fair value, capped at 5) differs by ≥ 0.5 fair value
    "materialKeeperStock": 2,  # max keepers held by one rival differs by ≥ 2
    "materialNoNeedBuyers": 3,  # rivals with no keeper need but a slot and base price differ by ≥ 3
    "similarState": "|Δ own purse share| ≤ 0.05, |Δ squad| ≤ 1, |Δ overseas| ≤ 1, same keeper need",
    "capHidden": 5.0,
    "spearmanStrong": 0.5, "spearmanWeak": 0.2,
}
PBINS = [(0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 0.75), (0.75, 1.01)]
PLABEL = ["0–2%", "2–5%", "5–10%", "10–20%", "20–30%", "30–50%", "50–75%", "75–100%", "re-auction"]
DIRECT = np.flatnonzero(CHANNEL == "direct")
WINDOW_F = ["mkt_recent_price_ratio", "mkt_recent_sold_share", "riv_recent_spend"]
RIVAL_F = [n for n in FN if n.startswith("riv_")]
KEEPER_F = ["lot_is_keeper", "self_class_Wi", "self_class_Wo", "self_status_keeper", "self_need_keeper", "self_fills_keeper", "self_unlocks",
            "self_final_opportunity", "mkt_supply_Wi", "mkt_supply_Wo", "mkt_scarcity_keeper", "mkt_same_role_left", "mkt_equivalent_left",
            "mkt_better_left", "riv_fills_share", "riv_gain_mean", "riv_capacity_1", "riv_capacity_2", "riv_capacity_3", "riv_able_share",
            "riv_free_slots_share", "riv_purse_max", "riv_purse_mean", "riv_purse_min", "self_purse", "self_max_safe_purse", "self_max_safe_fv",
            "self_reserve_if_passed", "self_slots_left", "self_overseas_slots_left", "self_status_players", "self_need_players",
            "lot_base_price", "lot_fair_value", "returning_count", "lots_left_in_phase", "phase_reauction", "main_progress", "mkt_recent_price_ratio"]


def r(x, n=4):
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, n)


def rankdata(x):
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    edges = np.flatnonzero(np.r_[True, xs[1:] != xs[:-1], True])
    avg = (edges[:-1] + edges[1:] - 1) / 2.0 + 1.0
    out = np.empty(len(x))
    out[order] = np.repeat(avg, np.diff(edges))
    return out


def spearman(a, b, max_n=300_000):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) > max_n:
        s = np.linspace(0, len(a) - 1, max_n).astype(int)
        a, b = a[s], b[s]
    if len(a) < 30 or a.std() == 0 or b.std() == 0:
        return None
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def auc(x1, x0, max_n=200_000):
    """P(value in group 1 > value in group 0) + ½ P(tie) — rank statistic, no fitting."""
    x1 = np.asarray(x1, dtype=np.float64)
    x0 = np.asarray(x0, dtype=np.float64)
    x1, x0 = x1[np.isfinite(x1)], x0[np.isfinite(x0)]
    if len(x1) > max_n:
        x1 = x1[np.linspace(0, len(x1) - 1, max_n).astype(int)]
    if len(x0) > max_n:
        x0 = x0[np.linspace(0, len(x0) - 1, max_n).astype(int)]
    if len(x1) < 20 or len(x0) < 20:
        return None
    rk = rankdata(np.r_[x1, x0])
    return float((rk[: len(x1)].sum() - len(x1) * (len(x1) + 1) / 2) / (len(x1) * len(x0)))


def pbin(progress, phase):
    b = np.full(len(progress), len(PBINS), dtype=np.int8)
    for i, (lo, hi) in enumerate(PBINS):
        b[(phase == 0) & (progress >= lo) & (progress < hi)] = i
    return b


# ── load ────────────────────────────────────────────────────────────────
def load():
    E = json.loads((RUN / "episodes.json").read_text(encoding="utf-8"))
    META = E["meta"]
    meta = np.fromfile(RUN / "meta.f32", dtype=np.float32).reshape(-1, len(META))
    obs = np.fromfile(RUN / "obs.f32", dtype=np.float32).reshape(-1, 80)
    assert len(meta) == len(obs)
    M = {m: meta[:, i] for i, m in enumerate(META)}
    eps = E["episodes"]
    assert [e["ep"] for e in eps] == list(range(len(eps)))
    D = {"E": E, "eps": eps, "M": M, "obs": obs, "N": len(meta)}
    ep = M["ep"].astype(np.int64)
    D["ep"] = ep
    cond = np.array([e["cond"] for e in eps])
    lkeys = sorted({e["learner"] for e in eps})
    D["lkeys"] = lkeys
    lid = np.array([lkeys.index(e["learner"]) for e in eps])
    oalgo = np.array([e["opp"].split(":")[0] if e["opp"] else "" for e in eps])
    okey = np.array([e["opp"] for e in eps])
    D["epCond"], D["epLid"], D["epOalgo"], D["epOkey"] = cond, lid, oalgo, okey
    D["epK"] = np.array([e["k"] for e in eps])
    D["epStratum"] = np.array([e["stratum"] for e in eps])
    D["cond"] = cond[ep]
    D["lid"] = lid[ep]
    D["oalgo"] = oalgo[ep]
    D["k"] = D["epK"][ep]
    D["stratum"] = D["epStratum"][ep]
    D["learner"] = M["who"] == 0
    D["pbin"] = pbin(M["progress"], M["phase"])
    D["need"] = np.clip(np.nan_to_num(M["othCap1"], nan=0.0), 0, DEF["capHidden"])  # price needed to win / fair
    # seat-level final outcome of the deciding seat
    seat = M["seat"].astype(np.int64)
    legal = np.zeros(len(eps) * 10, dtype=bool)
    keepers = np.zeros(len(eps) * 10, dtype=np.int16)
    for e in eps:
        for t in e["teams"]:
            legal[e["ep"] * 10 + t["seat"]] = t["legal"]
            keepers[e["ep"] * 10 + t["seat"]] = t["keepers"]
    D["seatLegal"] = legal[ep * 10 + seat]
    D["seatKeepers"] = keepers[ep * 10 + seat]
    return D


def learner_key(D, rows, with_opp=False):
    M = D["M"]
    k = (D["lid"][rows].astype(np.int64) * 500 + D["k"][rows]) * 2 + M["phase"][rows].astype(np.int64)
    k = k * 1024 + M["slNo"][rows].astype(np.int64)
    if with_opp:
        oid = np.array([(ALGOS.index(a) if a else 9) for a in D["oalgo"][rows]]) * 4 + np.array(
            [int(x.split(":s")[1]) if x else 0 for x in D["epOkey"][D["ep"][rows]]])
        k = k * 64 + oid
    return k


def match(D, rowsA, rowsX, with_opp=False):
    ka = learner_key(D, rowsA, with_opp)
    kx = learner_key(D, rowsX, with_opp)
    o = np.argsort(ka)
    ka_s = ka[o]
    if len(np.unique(ka_s)) != len(ka_s):
        raise SystemExit("pair key not unique")
    pos = np.clip(np.searchsorted(ka_s, kx), 0, len(ka_s) - 1)
    ok = ka_s[pos] == kx
    return rowsA[o[pos[ok]]], rowsX[ok]


def similar_state(D, ia, ix):
    M = D["M"]
    return ((np.abs(M["purseShare"][ia] - M["purseShare"][ix]) <= 0.05) & (np.abs(M["squad"][ia] - M["squad"][ix]) <= 1)
            & (np.abs(M["overseas"][ia] - M["overseas"][ix]) <= 1) & (M["keeperNeed"][ia] == M["keeperNeed"][ix]))


def pair_stats(D, ia, ix, chunk=200_000):
    """Per-pair distances (overall, per block, per opponent channel) and per-feature difference stats."""
    obs = D["obs"]
    n = len(ia)
    linf = np.empty(n, np.float32)
    l2 = np.empty(n, np.float32)
    nd = np.empty(n, np.int16)
    blocks = {b: np.empty(n, np.float32) for b in ["global", "player", "self", "market", "rivals"]}
    chans = {c: np.empty(n, np.float32) for c in ["none", "own", "supply", "direct"]}
    win = np.empty(n, np.float32)
    fdiff = np.zeros(80)
    fabs = np.zeros(80)
    wi = [FI[f] for f in WINDOW_F]
    for s in range(0, n, chunk):
        a = np.asarray(obs[ia[s:s + chunk]])
        x = np.asarray(obs[ix[s:s + chunk]])
        d = np.abs(x - a)
        linf[s:s + chunk] = d.max(1)
        l2[s:s + chunk] = np.sqrt((d ** 2).sum(1))
        nd[s:s + chunk] = (d > DEF["featureDiffers"]).sum(1)
        for b in blocks:
            blocks[b][s:s + chunk] = d[:, BLOCK == b].max(1)
        for c in chans:
            chans[c][s:s + chunk] = d[:, CHANNEL == c].max(1)
        win[s:s + chunk] = d[:, wi].max(1)
        fdiff += (d > DEF["featureDiffers"]).sum(0)
        fabs += d.sum(0)
    return {"linf": linf, "l2": l2, "nd": nd, "blocks": blocks, "chans": chans, "window": win, "fdiffShare": fdiff / max(1, n), "fabsMean": fabs / max(1, n)}


def top_features(share, mean_abs, k=12):
    o = np.argsort(-share)[:k]
    return [{"feature": FN[i], "index": int(i), "channel": str(CHANNEL[i]), "shareDiffering": r(share[i], 3), "meanAbsDiff": r(mean_abs[i], 4)} for i in o]


# ── Analysis A/B — mechanism → feature mapping, with empirical carriers ─
MECHANISMS = [
    # id, name, hidden variable(s) (meta column), candidate features, structural category, row filter
    ("early_star", "Early star competition", "need", ["mkt_recent_price_ratio", "riv_recent_spend", "mkt_recent_sold_share", "riv_capacity_1", "riv_capacity_2",
                                                         "riv_capacity_3", "riv_able_share", "riv_gain_mean", "riv_purse_max", "riv_purse_mean", "riv_purse_min",
                                                         "riv_purse_std", "riv_xi_max", "lot_is_star", "lot_fair_value", "mkt_premium_passed"], "star_early"),
    ("aggressiveness", "Opponent aggressiveness", "need", ["mkt_recent_price_ratio", "riv_recent_spend", "mkt_recent_sold_share", "riv_purse_mean", "riv_purse_min",
                                                           "riv_purse_std", "riv_xi_mean", "riv_xi_max", "mkt_premium_passed", "self_pace_gap"], "main"),
    ("multi_aggressive", "Multiple aggressive opponents", "rlBidders", ["riv_purse_std", "riv_purse_min", "riv_purse_mean", "riv_capacity_1", "riv_capacity_2",
                                                                        "riv_capacity_3", "riv_able_share", "riv_recent_spend", "mkt_recent_price_ratio"], "rl_rooms"),
    ("rival_purse", "Rival purse pressure", "rpMean", ["riv_purse_max", "riv_purse_mean", "riv_purse_min", "riv_purse_std", "riv_capacity_1", "riv_able_share"], "all"),
    ("rival_role_demand", "Rival role demand (keeper demand measured)", "rivNeedKeeper", ["riv_fills_share", "mkt_scarcity_keeper", "self_status_keeper"], "keeper_lots"),
    ("rival_keeper_demand", "Rival keeper demand", "rivNeedKeeper", ["mkt_scarcity_keeper", "riv_fills_share", "self_status_keeper"], "keeper_lots_need"),
    ("keeper_stockpiling", "Rival keeper stockpiling", "rivKeepersMax", KEEPER_F, "keeper_lots"),
    ("overseas", "Overseas competition", None, ["lot_is_overseas", "self_overseas_slots_left", "mkt_overseas_contested", "mkt_supply_Wo", "mkt_supply_Bo",
                                                "mkt_supply_Oo", "riv_capacity_1", "riv_able_share", "riv_gain_mean"], None),
    ("squad_slots", "Squad-slot competition", None, ["self_slots_left", "self_bench", "riv_free_slots_share"], None),
    ("price_inflation", "Price inflation", "need", ["mkt_recent_price_ratio", "riv_recent_spend"], "main"),
    ("requirement_timing", "Requirement timing", None, ["self_need_keeper", "self_need_bowling", "self_need_indians", "self_need_players", "self_status_keeper",
                                                        "self_status_bowling", "self_status_indians", "self_final_opportunity", "mkt_scarcity_keeper",
                                                        "mkt_scarcity_bowling", "mkt_scarcity_indians", "main_progress", "lots_left_in_phase"], None),
    ("purse_depletion", "Purse depletion", "purseShare", ["self_purse", "self_max_safe_purse", "self_max_safe_fv", "self_reserve_if_passed", "self_pace_gap"], "all"),
    ("reauction_pressure", "Re-auction pressure", None, ["phase_reauction", "returning_count", "lots_left_in_phase", "mkt_supply_Wi", "mkt_supply_Wo",
                                                         "riv_purse_max", "riv_free_slots_share", "self_reserve_if_passed"], None),
    ("final_path", "Final-path risk", "forced", ["self_final_opportunity", "self_status_keeper", "self_status_bowling", "self_status_indians", "self_status_players",
                                                 "mkt_scarcity_keeper", "mkt_scarcity_bowling", "mkt_scarcity_indians", "self_reserve_if_passed", "self_max_safe_fv"], "all"),
]


def carriers(D, hidden, feats, filt, rows_all):
    M = D["M"]
    rows = rows_all
    if filt == "star_early":
        rows = rows[(M["lotStar"][rows] == 1) & (M["phase"][rows] == 0) & (M["progress"][rows] < 0.3)]
    elif filt == "main":
        rows = rows[M["phase"][rows] == 0]
    elif filt == "rl_rooms":
        rows = rows[np.isin(D["cond"][rows], ["C1", "C4"])]
    elif filt == "keeper_lots":
        rows = rows[M["lotRole"][rows] == 3]
    elif filt == "keeper_lots_need":
        rows = rows[(M["lotRole"][rows] == 3) & (M["keeperNeed"][rows] > 0)]
    if hidden == "need":
        y = D["need"][rows]
    elif hidden == "rpMean":
        y = np.nanmean(np.stack([M[f"rp{i}"][rows] for i in range(9)]), 0)
    else:
        y = M[hidden][rows]
    out = []
    for f in feats:
        x = D["obs"][rows, FI[f]]
        # within-progress-bin Spearman (removes the shared auction clock), averaged by bin size
        rhos, ws = [], []
        for b in range(len(PLABEL)):
            m = D["pbin"][rows] == b
            if m.sum() < 200:
                continue
            s = spearman(x[m], y[m])
            if s is not None:
                rhos.append(s)
                ws.append(m.sum())
        out.append({"feature": f, "index": FI[f], "withinProgressSpearman": r(np.average(rhos, weights=ws), 3) if rhos else None,
                    "pooledSpearman": r(spearman(x, y), 3)})
    out.sort(key=lambda z: -abs(z["withinProgressSpearman"] or 0))
    return {"rows": int(len(rows)), "hidden": hidden, "features": out}


def quality(rho):
    if rho is None:
        return "not measurable"
    a = abs(rho)
    return "strong" if a >= DEF["spearmanStrong"] else "moderate" if a >= 0.35 else "weak" if a >= DEF["spearmanWeak"] else "none"


# ── F — same-lot counterfactual pairs (A vs C1 / C4 / S4) ───────────────
def counterfactual(D):
    M = D["M"]
    L = np.flatnonzero(D["learner"])
    A = L[D["cond"][L] == "A"]
    res = {"definition": {"pair": "same learner export, same auction entry (identical player order, purse, seats), same lot and phase; Stage-A control (A) vs Stage-B condition",
                          "similarState": DEF["similarState"], "material": f"price needed to win (max cap of the other nine seats / fair value, capped at {DEF['capHidden']}) differs by ≥ {DEF['materialPrice']}"},
           "conditions": {}}
    keep = {}
    for c in ["C1", "C4", "S4"]:
        X = L[D["cond"][L] == c]
        ia, ix = match(D, A, X)
        sim = similar_state(D, ia, ix)
        ps = pair_stats(D, ia, ix)
        dneed = D["need"][ix] - D["need"][ia]
        material = np.abs(dneed) >= DEF["materialPrice"]
        won_a = M["winner"][ia] == 0
        won_x = M["winner"][ix] == 0
        star = M["lotStar"][ix] == 1
        keeper = M["lotRole"][ix] == 3
        cats = {"all": np.ones(len(ix), bool), "star": star, "keeper": keeper, "early (main < 10%)": (M["phase"][ix] == 0) & (M["progress"][ix] < 0.1)}
        cres = {"pairs": int(len(ix)), "similarStatePairs": int(sim.sum())}
        for cn, cm in cats.items():
            m = sim & cm
            mm = m & material
            cres[cn] = {
                "pairs": int(m.sum()), "materialPairs": int(mm.sum()),
                "materialShare": r(mm.sum() / max(1, m.sum()), 3),
                "identicalObs": r((ps["linf"][m] == 0).mean(), 3) if m.any() else None,
                "linfLeTight": r((ps["linf"][m] <= DEF["aliasTight"]).mean(), 3) if m.any() else None,
                "material_identicalObs": r((ps["linf"][mm] == 0).mean(), 3) if mm.any() else None,
                "material_linfLeTight": r((ps["linf"][mm] <= DEF["aliasTight"]).mean(), 3) if mm.any() else None,
                "material_linfLeLoose": r((ps["linf"][mm] <= DEF["aliasLoose"]).mean(), 3) if mm.any() else None,
                "material_directChannelLeTight": r((ps["chans"]["direct"][mm] <= DEF["aliasTight"]).mean(), 3) if mm.any() else None,
                "material_rivalsBlockLeTight": r((ps["blocks"]["rivals"][mm] <= DEF["aliasTight"]).mean(), 3) if mm.any() else None,
                "material_windowLeTight": r((ps["window"][mm] <= DEF["aliasTight"]).mean(), 3) if mm.any() else None,
                "medianLinf": r(np.median(ps["linf"][m]), 4) if m.any() else None,
                "medianFeaturesDiffering": r(np.median(ps["nd"][m]), 1) if m.any() else None,
                "material_wonInA_lostInX": r((won_a[mm] & ~won_x[mm]).mean(), 3) if mm.any() else None,
                "material_identicalObs_wonInA_lostInX": int((won_a & ~won_x & mm & (ps["linf"] == 0)).sum()),
                "meanDeltaNeed": r(dneed[m].mean(), 3) if m.any() else None,
            }
        # which blocks / channels carry the difference (similar-state pairs)
        cres["blockMedianLinf"] = {b: r(np.median(v[sim]), 4) for b, v in ps["blocks"].items()}
        cres["channelMedianLinf"] = {b: r(np.median(v[sim]), 4) for b, v in ps["chans"].items()}
        cres["topDifferingFeatures"] = top_features(ps["fdiffShare"], ps["fabsMean"])
        # do opponent-direct feature differences track the hidden difference in price needed?
        di = []
        for f in WINDOW_F + ["riv_purse_mean", "riv_purse_min", "riv_purse_std", "riv_capacity_1", "riv_capacity_3", "riv_able_share", "riv_xi_max", "riv_gain_mean"]:
            dx = D["obs"][ix[sim], FI[f]] - D["obs"][ia[sim], FI[f]]
            di.append({"feature": f, "spearman_dFeature_dNeed": r(spearman(dx, dneed[sim]), 3)})
        cres["deltaTracking"] = sorted(di, key=lambda z: -abs(z["spearman_dFeature_dNeed"] or 0))
        # by progress
        cres["byProgress"] = []
        for b, lab in enumerate(PLABEL):
            m = sim & (D["pbin"][ix] == b)
            mm = m & material
            cres["byProgress"].append({"progress": lab, "pairs": int(m.sum()), "materialShare": r(mm.sum() / max(1, m.sum()), 3),
                                       "identicalObs": r((ps["linf"][m] == 0).mean(), 3) if m.any() else None,
                                       "material_linfLeTight": r((ps["linf"][mm] <= DEF["aliasTight"]).mean(), 3) if mm.any() else None,
                                       "medianDirectLinf": r(np.median(ps["chans"]["direct"][m]), 4) if m.any() else None,
                                       "medianWindowLinf": r(np.median(ps["window"][m]), 4) if m.any() else None})
        res["conditions"][c] = cres
        keep[c] = (ia, ix, ps, sim, dneed)
    return res, keep


# ── divergence: how long the observation stays identical to Stage A ─────
def divergence(D, keep):
    M = D["M"]
    out = {"definition": "For each Stage-B learner episode, the first main-round lot at which its observation differs from the Stage-A control on the same entry (any feature |Δ| > 0). Star lots before that point were contested with an observation bit-identical to Stage A.", "conditions": {}}
    for c, (ia, ix, ps, sim, dneed) in keep.items():
        main = M["phase"][ix] == 0
        epx = D["ep"][ix]
        lot = M["lotIdx"][ix]
        sel = main & (ps["linf"] > 0)
        e_, l_ = epx[sel], lot[sel]
        o = np.lexsort((l_, e_))
        e_s = e_[o]
        firsts = l_[o][np.r_[True, e_s[1:] != e_s[:-1]]].astype(float)
        stars_identical = (M["lotStar"][ix] == 1) & main & (ps["linf"] == 0)
        m_id = stars_identical & (np.abs(dneed) >= DEF["materialPrice"])
        out["conditions"][c] = {
            "episodes": int(len(np.unique(epx))),
            "firstDifferentLot": {"median": r(np.median(firsts), 1), "p10": r(np.percentile(firsts, 10), 1), "p90": r(np.percentile(firsts, 90), 1)},
            "firstDifferentProgress": r(np.median(firsts) / 323, 4),
            "starDecisionsWithIdenticalObs": int(stars_identical.sum()),
            "starDecisionsTotal": int(((M["lotStar"][ix] == 1) & main).sum()),
            "starIdentical_materialPriceDiff": int(m_id.sum()),
            "starIdentical_wonInA_lostInX": int((stars_identical & (M["winner"][ia] == 0) & (M["winner"][ix] != 0)).sum()),
            "starIdentical_meanNeedA": r(D["need"][ia][stars_identical].mean(), 3) if stars_identical.any() else None,
            "starIdentical_meanNeedX": r(D["need"][ix][stars_identical].mean(), 3) if stars_identical.any() else None,
            "decisionsIdenticalObsShare": r((ps["linf"] == 0).mean(), 4),
        }
    return out


# ── G — aliasing within (entry, phase, lot) across all episodes ─────────
def aliasing(D, max_group=420):
    M = D["M"]
    obs = D["obs"]
    rows = np.arange(D["N"])
    key = (D["k"].astype(np.int64) * 2 + M["phase"].astype(np.int64)) * 1024 + M["slNo"].astype(np.int64)
    order = np.argsort(key, kind="mergesort")
    ks = key[order]
    edges = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
    cond = D["cond"]
    oalgo = D["oalgo"]
    ep = D["ep"]
    seat = M["seat"].astype(np.int64)
    rpm = np.nanmean(np.stack([M[f"rp{i}"] for i in range(9)]), 0)
    mech = {
        "price_pressure": {"desc": f"price needed to win differs by ≥ {DEF['materialPrice']} fair value", "tot": 0, "tight": 0, "loose": 0, "outcome": 0, "groups": set(), "ex": []},
        "weak_vs_aggressive": {"desc": "Stage-A room (rule-bot rivals) vs a C4 room with four high-pressure copies (D3QN / QR-DQN)", "tot": 0, "tight": 0, "loose": 0, "outcome": 0, "groups": set(), "ex": []},
        "one_vs_multiple": {"desc": "C1 vs C4 with the same opponent export (one copy vs four copies)", "tot": 0, "tight": 0, "loose": 0, "outcome": 0, "groups": set(), "ex": []},
        "keeper_competition": {"desc": f"keeper lot, both seats need a keeper; max keepers held by one rival differ by ≥ {DEF['materialKeeperStock']} or rivals able to buy without needing differ by ≥ {DEF['materialNoNeedBuyers']}", "tot": 0, "tight": 0, "loose": 0, "outcome": 0, "groups": set(), "ex": []},
        "safe_vs_dangerous_keeper": {"desc": "keeper lot, both seats need a keeper; one seat finished WITHOUT a keeper, the other with one", "tot": 0, "tight": 0, "loose": 0, "outcome": 0, "groups": set(), "ex": []},
        "rival_purse": {"desc": "mean rival purse differs by ≥ 0.10 of the purse (control: this is IN the observation, so aliasing must be ~0)", "tot": 0, "tight": 0, "loose": 0, "outcome": 0, "groups": set(), "ex": []},
        "rl_rivals_bidding": {"desc": "number of RL-controlled rivals bidding on the lot differs by ≥ 2", "tot": 0, "tight": 0, "loose": 0, "outcome": 0, "groups": set(), "ex": []},
    }
    fdiff_alias = np.zeros(80)
    n_alias_mat = 0
    total_pairs = 0
    tight_pairs = 0
    rng = np.random.RandomState(7)
    seatKeep = D["seatKeepers"]
    for g in range(len(edges) - 1):
        idx = order[edges[g]:edges[g + 1]]
        if len(idx) < 2:
            continue
        if len(idx) > max_group:
            idx = np.sort(rng.choice(idx, max_group, replace=False))
        X = np.asarray(obs[idx])
        d = np.abs(X[:, None, :] - X[None, :, :]).max(2)
        iu = np.triu_indices(len(idx), 1)
        dd = d[iu]
        a, b = idx[iu[0]], idx[iu[1]]
        valid = ~((ep[a] == ep[b]) & (seat[a] == seat[b]))
        a, b, dd = a[valid], b[valid], dd[valid]
        total_pairs += len(dd)
        tight_pairs += int((dd <= DEF["aliasTight"]).sum())
        need = D["need"]
        isK = M["lotRole"][a] == 3
        bothNeed = (M["keeperNeed"][a] > 0) & (M["keeperNeed"][b] > 0)
        conds = {
            "price_pressure": np.abs(need[a] - need[b]) >= DEF["materialPrice"],
            "weak_vs_aggressive": ((cond[a] == "A") & (cond[b] == "C4") & np.isin(oalgo[b], list(HIGH_PRESSURE))) | ((cond[b] == "A") & (cond[a] == "C4") & np.isin(oalgo[a], list(HIGH_PRESSURE))),
            "one_vs_multiple": (((cond[a] == "C1") & (cond[b] == "C4")) | ((cond[a] == "C4") & (cond[b] == "C1"))) & (D["epOkey"][ep[a]] == D["epOkey"][ep[b]]),
            "keeper_competition": isK & bothNeed & ((np.abs(M["rivKeepersMax"][a] - M["rivKeepersMax"][b]) >= DEF["materialKeeperStock"]) | (np.abs(M["rivNoNeedCanBuy"][a] - M["rivNoNeedCanBuy"][b]) >= DEF["materialNoNeedBuyers"])),
            "safe_vs_dangerous_keeper": isK & bothNeed & ((seatKeep[a] == 0) != (seatKeep[b] == 0)),
            "rival_purse": np.abs(rpm[a] - rpm[b]) >= 0.10,
            "rl_rivals_bidding": np.abs(M["rlBidders"][a] - M["rlBidders"][b]) >= 2,
        }
        outcome = (M["winner"][a] == 0) != (M["winner"][b] == 0)
        for name, cm in conds.items():
            if not cm.any():
                continue
            s = mech[name]
            s["tot"] += int(cm.sum())
            t = cm & (dd <= DEF["aliasTight"])
            s["tight"] += int(t.sum())
            s["loose"] += int((cm & (dd <= DEF["aliasLoose"])).sum())
            s["outcome"] += int((t & outcome).sum())
            if t.any():
                s["groups"].add(int(g))
                if len(s["ex"]) < 4:
                    j = np.flatnonzero(t)[0]
                    s["ex"].append({"k": int(D["k"][a[j]]), "slNo": int(M["slNo"][a[j]]), "phase": "main" if M["phase"][a[j]] == 0 else "reauction",
                                    "progress": r(M["progress"][a[j]], 3), "linf": r(dd[j], 4),
                                    "a": {"cond": str(cond[a[j]]), "opp": str(D["epOkey"][ep[a[j]]]), "who": "learner" if M["who"][a[j]] == 0 else "rl opponent", "needToWin": r(need[a[j]], 2), "won": bool(M["winner"][a[j]] == 0), "rivKeepersMax": int(M["rivKeepersMax"][a[j]]), "rlBidders": int(M["rlBidders"][a[j]]), "seatFinalKeepers": int(seatKeep[a[j]])},
                                    "b": {"cond": str(cond[b[j]]), "opp": str(D["epOkey"][ep[b[j]]]), "who": "learner" if M["who"][b[j]] == 0 else "rl opponent", "needToWin": r(need[b[j]], 2), "won": bool(M["winner"][b[j]] == 0), "rivKeepersMax": int(M["rivKeepersMax"][b[j]]), "rlBidders": int(M["rlBidders"][b[j]]), "seatFinalKeepers": int(seatKeep[b[j]])}})
            if name == "price_pressure" and t.any():
                jj = np.flatnonzero(t)
                if len(jj) > 2000:
                    jj = jj[:: len(jj) // 2000]
                fdiff_alias += (np.abs(np.asarray(obs[a[jj]]) - np.asarray(obs[b[jj]])) > 1e-6).sum(0)
                n_alias_mat += len(jj)
    out = {"definition": {"groups": "all captured decisions (learner + RL-opponent rows) on the same auction entry, phase and lot, i.e. the same player card at the same point of the same player order",
                          "aliased": f"L∞ over the 80 features ≤ {DEF['aliasTight']} (tight) / ≤ {DEF['aliasLoose']} (loose)",
                          "pairs": "unordered pairs from different (episode, seat)", "groupCap": f"groups larger than {max_group} rows are randomly subsampled (RandomState(7))"},
           "pairsExamined": int(total_pairs), "tightAliasedPairs": int(tight_pairs), "mechanisms": {}}
    for name, s in mech.items():
        out["mechanisms"][name] = {"description": s["desc"], "materialPairs": s["tot"], "aliasedTight": s["tight"], "aliasedLoose": s["loose"],
                                   "aliasRateTight": r(s["tight"] / max(1, s["tot"]), 4), "aliasRateLoose": r(s["loose"] / max(1, s["tot"]), 4),
                                   "aliasedTight_outcomeDiffers": s["outcome"], "lotsAffected": len(s["groups"]), "examples": s["ex"]}
    out["pricePressureAliased_featuresThatDifferAtAll"] = [{"feature": FN[i], "share": r(fdiff_alias[i] / max(1, n_alias_mat), 3)} for i in np.argsort(-fdiff_alias)[:12]]
    return out


# ── C — keeper-failure observability (the 20 Phase 2E.0 findings) ───────
def keeper_obs(D):
    M = D["M"]
    obs = D["obs"]
    eps = D["eps"]
    F = json.loads((ROOT / "ml/reports/phase2e0/findings.json").read_text(encoding="utf-8"))["findings"]
    kf = json.loads((ROOT / "ml/reports/phase2e1/keeper-failures.json").read_text(encoding="utf-8"))
    mech_by = {(c["seed"], c["seat"], c["affected"]): c["mechanism"][:2] for c in kf["findings"]}
    epmap = {(e["cond"], e["learner"], e["opp"], e["k"]): e["ep"] for e in eps}
    kf_feats = [FI[f] for f in KEEPER_F]
    # reference: keeper-lot decisions where the seat needed a keeper and finished WITH one (safe outcome)
    kl = (M["lotRole"] == 3) & (M["keeperNeed"] > 0)
    safe = np.flatnonzero(kl & (D["seatKeepers"] > 0))
    hidden = ["rivKeepersMax", "rivKeep2", "rivKeep3", "rivNoNeedCanBuy", "rivNeedKeeper", "rivKeepersTot", "kSpare", "kContestants"]
    cases = []
    pct_obs = defaultdict(list)
    pct_hid = defaultdict(list)
    nn_all = []
    for f in F:
        e = epmap.get(("C4", f["learner"], f["opponents"][0], f["k"]))
        rows = np.flatnonzero((D["ep"] == e) & (M["seat"] == f["seat"]) & (M["lotRole"] == 3) & (M["keeperNeed"] > 0))
        rows = rows[np.argsort(M["lotIdx"][rows])]
        mech = mech_by.get((eps[e]["seed"], f["seat"], f["key"]))
        crit = []
        for i in rows:
            passed = M["action"][i] == 0 or M["cap"][i] < M["base"][i]
            # matched safe references: same phase, progress within ±0.05, same keeper need, same seat role (learner / opponent)
            ref = safe[(M["phase"][safe] == M["phase"][i]) & (np.abs(M["progress"][safe] - M["progress"][i]) <= 0.05) & (M["who"][safe] == M["who"][i])]
            ref = ref[D["ep"][ref] != e]
            row = {"lot": int(M["slNo"][i]), "phase": "main" if M["phase"][i] == 0 else "reauction", "progress": r(M["progress"][i], 3),
                   "action": int(M["action"][i]), "passed": bool(passed), "forced": bool(M["forced"][i]), "forcedKeeper": bool(M["forcedKeeper"][i]),
                   "finalPath": bool(M["finalPath"][i]), "shieldKeeperState": ["SAFE", "WARNING", "CRITICAL", "IMPOSSIBLE"][int(M["kState"][i])] if np.isfinite(M["kState"][i]) and M["kState"][i] >= 0 else None,
                   "outcome": {-1: "unsold", 0: "won", 1: "lost to RL seat", 2: "lost to rule bot", 3: "lost to human proxy"}[int(M["winner"][i])],
                   "priceFair": r(M["priceFair"][i], 3), "base": int(M["base"][i]), "purse": int(M["purse"][i]), "maxSafeBid": int(M["maxSafe"][i]),
                   "squad": int(M["squad"][i]), "overseas": int(M["overseas"][i]),
                   "obs": {n: r(obs[i, FI[n]], 4) for n in KEEPER_F},
                   "hidden": {h: (r(M[h][i], 3) if np.isfinite(M[h][i]) else None) for h in hidden},
                   "referenceStates": int(len(ref))}
            if len(ref) >= 30:
                R = np.asarray(obs[ref])
                x = np.asarray(obs[i])
                pctl = {}
                for n in ["self_need_keeper", "self_status_keeper", "mkt_supply_Wi", "mkt_supply_Wo", "mkt_scarcity_keeper", "riv_fills_share", "riv_gain_mean",
                          "riv_capacity_1", "riv_able_share", "self_purse", "self_max_safe_fv", "self_reserve_if_passed", "self_slots_left", "self_overseas_slots_left", "mkt_recent_price_ratio"]:
                    v = R[:, FI[n]]
                    p = float(((v < x[FI[n]]).sum() + 0.5 * (v == x[FI[n]]).sum()) / len(v))
                    pctl[n] = r(p, 3)
                    pct_obs[n].append(p)
                hp = {}
                for h in ["rivKeepersMax", "rivKeep2", "rivKeep3", "rivNoNeedCanBuy", "rivNeedKeeper"]:
                    v = M[h][ref]
                    p = float(((v < M[h][i]).sum() + 0.5 * (v == M[h][i]).sum()) / len(v))
                    hp[h] = r(p, 3)
                    pct_hid[h].append(p)
                row["percentileAmongSafeReferences"] = {"observed": pctl, "hidden": hp}
                # nearest safe reference in the keeper-relevant sub-space and in the full observation
                dk = np.abs(R[:, kf_feats] - x[kf_feats]).max(1)
                dfull = np.abs(R - x).max(1)
                j = int(np.argmin(dk))
                row["nearestSafe"] = {"linfKeeperSubspace": r(dk[j], 4), "linfFull": r(dfull[j], 4), "minLinfFull": r(dfull.min(), 4),
                                      "within_0.05_keeperSubspace": int((dk <= DEF["aliasLoose"]).sum()), "within_0.10_keeperSubspace": int((dk <= 0.10).sum()),
                                      "hiddenAtNearest": {h: r(M[h][ref[j]], 3) for h in ["rivKeepersMax", "rivKeep2", "rivNoNeedCanBuy", "rivNeedKeeper"]},
                                      "largestFeatureGaps": [{"feature": KEEPER_F[q], "dangerous": r(x[kf_feats][q], 3), "safe": r(R[j, kf_feats][q], 3)} for q in np.argsort(-np.abs(R[j, kf_feats] - x[kf_feats]))[:4]]}
                nn_all.append(dk[j])
            crit.append(row)
        passes = [c for c in crit if c["passed"]]
        cases.append({"k": f["k"], "learner": f["learner"], "opponent": f["opponents"][0], "seat": f["seat"], "type": f["type"], "key": f["key"], "mechanism": mech,
                      "keeperLotDecisionsWhileNeeding": len(crit), "passes": len(passes), "passesOnKeepersThatWentUnsold": sum(1 for c in passes if c["outcome"] == "unsold"),
                      "decisions": crit})
    summ = {"medianPercentileObserved": {n: r(np.median(v), 3) for n, v in pct_obs.items()},
            "medianPercentileHidden": {n: r(np.median(v), 3) for n, v in pct_hid.items()},
            "shareDecisionsHiddenAbove90thPct": {n: r(np.mean(np.array(v) >= 0.9), 3) for n, v in pct_hid.items()},
            "shareDecisionsObservedOutside10_90": {n: r(np.mean((np.array(v) < 0.1) | (np.array(v) > 0.9)), 3) for n, v in pct_obs.items()},
            "nearestSafeLinfKeeperSubspace": {"median": r(np.median(nn_all), 4) if nn_all else None, "shareLe0.05": r(np.mean(np.array(nn_all) <= 0.05), 3) if nn_all else None,
                                              "shareLe0.10": r(np.mean(np.array(nn_all) <= 0.10), 3) if nn_all else None},
            "referencePool": int(len(safe))}
    return {"definition": {"criticalDecisions": "every keeper-lot decision of the affected seat while it still needed a keeper (main round and re-auction), exact observation from the digest-verified replay",
                           "safeReferences": "keeper-lot decisions (any captured seat/episode among the 15,600 replays, excluding the finding episode) where the seat needed a keeper and FINISHED WITH ONE; matched on phase, progress ±0.05 and seat role",
                           "percentile": "mid-rank percentile of the dangerous decision's value within its matched safe references (0.5 = typical)"},
            "summary": summ, "cases": cases}


# ── E / H — opponent signals: separation over time, tracking, support ───
def signals(D, keep):
    M = D["M"]
    L = np.flatnonzero(D["learner"])
    cond = D["cond"][L]
    oal = D["oalgo"][L]
    obs = D["obs"]
    feats = ["mkt_recent_price_ratio", "mkt_recent_sold_share", "riv_recent_spend", "riv_purse_mean", "riv_purse_min", "riv_purse_std", "riv_purse_max",
             "riv_capacity_1", "riv_capacity_3", "riv_able_share", "riv_fills_share", "riv_gain_mean", "riv_xi_mean", "riv_xi_max", "mkt_premium_passed",
             "returning_count", "mkt_scarcity_keeper", "mkt_scarcity_bowling", "self_purse", "self_pace_gap"]
    A = L[cond == "A"]
    HP = L[(cond == "C4") & np.isin(oal, list(HIGH_PRESSURE))]
    LP = L[(cond == "C4") & np.isin(oal, list(LOW_PRESSURE))]
    S4 = L[cond == "S4"]
    sep = []
    for f in feats:
        x = obs[:, FI[f]]
        row = {"feature": f, "channel": str(CHANNEL[FI[f]]), "byProgress": []}
        for b, lab in enumerate(PLABEL):
            a_ = A[D["pbin"][A] == b]
            h_ = HP[D["pbin"][HP] == b]
            l_ = LP[D["pbin"][LP] == b]
            s_ = S4[D["pbin"][S4] == b]
            ah = auc(np.asarray(x[h_]), np.asarray(x[a_]))
            al = auc(np.asarray(x[l_]), np.asarray(x[a_]))
            as_ = auc(np.asarray(x[s_]), np.asarray(x[a_]))
            row["byProgress"].append({"progress": lab, "sep_C4high_vs_A": r(abs(2 * ah - 1), 3) if ah is not None else None, "auc_C4high_vs_A": r(ah, 3),
                                      "sep_C4low_vs_A": r(abs(2 * al - 1), 3) if al is not None else None, "sep_S4_vs_A": r(abs(2 * as_ - 1), 3) if as_ is not None else None,
                                      "medianA": r(np.median(np.asarray(x[a_])), 3) if len(a_) else None, "medianC4high": r(np.median(np.asarray(x[h_])), 3) if len(h_) else None})
        sep.append(row)
    # hidden-state tracking: which features follow what opponents actually did (Spearman within progress bins)
    main = L[M["phase"][L] == 0]
    hid = {"needToWin (price the other nine seats would pay / fair)": D["need"], "RL-rival spend, last 20 lots (hidden split of the pooled window)": M["wRlSpend"],
           "max single-rival spend, last 20 lots": M["wMaxRivSpend"], "keepers bought by rivals, last 20 lots": M["wRivKeepers"],
           "stars bought by rivals, last 20 lots": M["wRivStars"], "cumulative RL-rival spend": M["cumRlSpend"], "cumulative stars bought by RL rivals": M["cumRlStars"],
           "RL rivals bidding on this lot": M["rlBidders"]}
    track = {}
    for hn, h in hid.items():
        rows = main
        best = []
        for f in FN:
            rhos, ws = [], []
            for b in range(len(PBINS)):
                m = rows[D["pbin"][rows] == b]
                if len(m) < 500:
                    continue
                s = spearman(obs[m, FI[f]], h[m], max_n=60_000)
                if s is not None:
                    rhos.append(s)
                    ws.append(len(m))
            if rhos:
                best.append((f, float(np.average(rhos, weights=ws))))
        best.sort(key=lambda z: -abs(z[1]))
        track[hn] = {"top": [{"feature": f, "withinProgressSpearman": r(v, 3)} for f, v in best[:6]], "bestAbs": r(abs(best[0][1]), 3) if best else None, "quality": quality(best[0][1] if best else None)}
    # Stage-A support: share of Stage-B learner observations outside the Stage-A 0.5–99.5% range, per feature
    Arows = A[:: max(1, len(A) // 300_000)]
    XA = np.asarray(obs[Arows])
    lo = np.percentile(XA, 0.5, axis=0)
    hi = np.percentile(XA, 99.5, axis=0)
    support = {}
    for name, rows in {"C1": L[cond == "C1"], "C4": L[cond == "C4"], "C4 high-pressure": HP, "C4 PPO": LP, "S4": S4}.items():
        rows = rows[:: max(1, len(rows) // 300_000)]
        X = np.asarray(obs[rows])
        outside = (X < lo - 1e-6) | (X > hi + 1e-6)
        per = outside.mean(0)
        support[name] = {"rows": int(len(rows)), "anyFeatureOutside": r(outside.any(1).mean(), 3),
                         "anyDirectFeatureOutside": r(outside[:, DIRECT].any(1).mean(), 3),
                         "anyOwnFeatureOutside": r(outside[:, CHANNEL == "own"].any(1).mean(), 3),
                         "topFeatures": [{"feature": FN[i], "shareOutside": r(per[i], 3), "channel": str(CHANNEL[i])} for i in np.argsort(-per)[:8]]}
    # does the frozen policy's cap move with opponent-direct feature differences? (same-lot pairs, similar state)
    resp = {}
    for c, (ia, ix, ps, sim, dneed) in keep.items():
        m = sim & (M["phase"][ix] == 0)
        dcap = np.clip(M["capFair"][ix][m], 0, 5) - np.clip(M["capFair"][ia][m], 0, 5)
        out = {}
        for f in ["mkt_recent_price_ratio", "riv_recent_spend", "riv_purse_mean", "riv_capacity_1", "self_purse", "self_max_safe_fv"]:
            dx = obs[ix[m], FI[f]] - obs[ia[m], FI[f]]
            out[f] = r(spearman(dx, dcap), 3)
        alg = np.array([D["lkeys"][x].split(":")[0] for x in D["lid"][ix[m]]])
        per_algo = {}
        for a in ALGOS:
            q = alg == a
            dx = np.asarray(obs[ix[m][q]][:, FI["mkt_recent_price_ratio"]]) - np.asarray(obs[ia[m][q]][:, FI["mkt_recent_price_ratio"]])
            dp = np.asarray(obs[ix[m][q]][:, FI["self_purse"]]) - np.asarray(obs[ia[m][q]][:, FI["self_purse"]])
            per_algo[a] = {"dCap_vs_dRecentPriceRatio": r(spearman(dx, dcap[q]), 3), "dCap_vs_dOwnPurse": r(spearman(dp, dcap[q]), 3), "dCap_vs_dNeedToWin": r(spearman(dneed[m][q], dcap[q]), 3)}
        resp[c] = {"pairs": int(m.sum()), "spearman_dCap_vs_dFeature": out, "byLearnerAlgorithm": per_algo}
    return {"separation": sep, "hiddenTracking": track, "stageASupport": support, "policyResponse": resp,
            "definition": {"separation": "|2·AUC − 1| between Stage-B and Stage-A learner observations in the same progress bin (0 = identical distributions, 1 = no overlap); AUC is the Mann–Whitney rank statistic",
                           "highPressure": "C4 rooms whose four RL opponents are D3QN or QR-DQN copies (largest Phase 2E.1 transfer loss)", "lowPressure": "C4 rooms with four PPO copies",
                           "support": "Stage-A range = 0.5–99.5th percentile of each feature over Stage-A (A) learner observations on the same 40 entries"}}


# ── I — purse / star trade-off ───────────────────────────────────────────
def purse_star(D):
    M = D["M"]
    obs = D["obs"]
    L = np.flatnonzero(D["learner"] & (M["phase"] == 0))
    # star positions per entry (union over every captured decision of that entry)
    stars = defaultdict(set)
    allr = np.flatnonzero((M["lotStar"] == 1) & (M["phase"] == 0))
    for k, li in zip(D["k"][allr], M["lotIdx"][allr]):
        stars[int(k)].add(int(li))
    starpos = {k: np.array(sorted(v)) for k, v in stars.items()}
    kL = D["k"][L]
    liL = M["lotIdx"][L]
    remStars = np.zeros(len(L))
    for k, pos in starpos.items():
        q = kL == k
        remStars[q] = len(pos) - np.searchsorted(pos, liL[q], side="right")
    # realised future star price/fair in the same episode (stars sold after this decision, learner-observed lots)
    fut = np.full(len(L), np.nan)
    epL = D["ep"][L]
    srows = np.flatnonzero(D["learner"] & (M["lotStar"] == 1) & (M["phase"] == 0) & (M["winner"] != -1))
    so = np.lexsort((M["lotIdx"][srows], D["ep"][srows]))
    srows = srows[so]
    s_ep = D["ep"][srows]
    bounds = np.flatnonzero(np.r_[True, s_ep[1:] != s_ep[:-1], True])
    ordL = np.argsort(epL, kind="mergesort")
    epL_s = epL[ordL]
    for g in range(len(bounds) - 1):
        grp = srows[bounds[g]:bounds[g + 1]]
        e = s_ep[bounds[g]]
        lo_, hi_ = np.searchsorted(epL_s, e), np.searchsorted(epL_s, e, side="right")
        jj = ordL[lo_:hi_]
        li = M["lotIdx"][grp]
        pf = M["priceFair"][grp].astype(float)
        suf = np.r_[np.cumsum(pf[::-1])[::-1], 0.0]
        cnt = np.r_[np.arange(len(pf), 0, -1), 0]
        at = np.searchsorted(li, liL[jj], side="right")
        with np.errstate(invalid="ignore", divide="ignore"):
            fut[jj] = np.where(cnt[at] > 0, suf[at] / np.maximum(cnt[at], 1), np.nan)
    early = M["progress"][L] < 0.5
    feats = ["lot_is_star", "lot_rating", "self_xi_gain", "self_upgrades_left", "self_typical_upgrade", "mkt_better_left", "mkt_equivalent_left",
             "mkt_premium_passed", "self_pace_gap", "self_reserve_if_passed", "self_max_safe_fv", "mkt_recent_price_ratio", "riv_recent_spend",
             "riv_purse_mean", "riv_capacity_1", "lots_left_in_phase"]
    def within(y, rows_mask):
        out = []
        for f in feats:
            rhos, ws = [], []
            for b in range(len(PBINS)):
                m = rows_mask & (D["pbin"][L] == b)
                if m.sum() < 500:
                    continue
                s = spearman(obs[L[m], FI[f]], y[m], max_n=80_000)
                if s is not None:
                    rhos.append(s)
                    ws.append(m.sum())
            out.append({"feature": f, "withinProgressSpearman": r(np.average(rhos, weights=ws), 3) if rhos else None})
        return sorted(out, key=lambda z: -abs(z["withinProgressSpearman"] or 0))
    # between-room share of future star price variance (room type = condition × opponent algorithm)
    lab = np.array([c if c in ("A", "S4") else f"{c}:{o}" for c, o in zip(D["cond"][L], D["oalgo"][L])])
    y = fut
    ok = np.isfinite(y) & early
    grand = y[ok].mean()
    ss_tot = ((y[ok] - grand) ** 2).sum()
    ss_b = sum(((y[ok & (lab == g)].mean() - grand) ** 2) * (ok & (lab == g)).sum() for g in np.unique(lab[ok]))
    return {"definition": {"remainingStars": "star lots (rating ≥ 90) later in the same entry's main-round order", "futureStarPrice": "mean price / fair value of the star lots sold after this decision in the same episode (hidden: realised future prices)",
                           "roomType": "condition × opponent algorithm"},
            "remainingStars_tracking": within(remStars, np.ones(len(L), bool))[:6],
            "futureStarPrice_tracking_mainBefore50pct": within(fut, early)[:8],
            "futureStarPrice_betweenRoomTypeShare": r(ss_b / ss_tot, 3),
            "futureStarPrice_byRoomType": {g: r(np.nanmean(y[ok & (lab == g)]), 3) for g in ["A", "C1:ppo", "C1:d3qn", "C1:qrdqn", "C4:ppo", "C4:es", "C4:a2c", "C4:d3qn", "C4:qrdqn", "S4"]},
            "starsInFirst10pct": r(np.mean([(v < 0.1 * 323).sum() / max(1, len(v)) for v in starpos.values()]), 3),
            "starsInFirst30pct": r(np.mean([(v < 0.3 * 323).sum() / max(1, len(v)) for v in starpos.values()]), 3),
            "meanStarsPerEntry": r(np.mean([len(v) for v in starpos.values()]), 1)}


# ── J — one vs several aggressive opponents (C1 vs C4) ──────────────────
def multi(D):
    M = D["M"]
    obs = D["obs"]
    L = np.flatnonzero(D["learner"])
    C1 = L[D["cond"][L] == "C1"]
    C4 = L[D["cond"][L] == "C4"]
    ia, ix = match(D, C1, C4, with_opp=True)
    sim = similar_state(D, ia, ix)
    ps = pair_stats(D, ia, ix)
    riv = [FI[f] for f in RIVAL_F + WINDOW_F]
    drv = np.zeros(len(ia), np.float32)
    for s in range(0, len(ia), 200_000):
        drv[s:s + 200_000] = np.abs(np.asarray(obs[ix[s:s + 200_000]])[:, riv] - np.asarray(obs[ia[s:s + 200_000]])[:, riv]).max(1)
    dbid = M["rlBidders"][ix] - M["rlBidders"][ia]
    dneed = D["need"][ix] - D["need"][ia]
    mat = sim & (np.abs(dbid) >= 2)
    out = {"definition": "Same learner, same opponent export, same entry, same lot: one copy (C1) vs four copies (C4). Rival block = the 14 riv_* features + the 3 window features.",
           "pairs": int(len(ia)), "similarStatePairs": int(sim.sum()),
           "similarState_rivalBlockLeTight": r((drv[sim] <= DEF["aliasTight"]).mean(), 3),
           "rlBiddersDiffer2plus": int(mat.sum()),
           "rlBiddersDiffer2plus_rivalBlockLeTight": r((drv[mat] <= DEF["aliasTight"]).mean(), 3) if mat.any() else None,
           "rlBiddersDiffer2plus_rivalBlockLeLoose": r((drv[mat] <= DEF["aliasLoose"]).mean(), 3) if mat.any() else None,
           "rlBiddersDiffer2plus_fullLeTight": r((ps["linf"][mat] <= DEF["aliasTight"]).mean(), 3) if mat.any() else None,
           "meanRlBidders": {"C1": r(M["rlBidders"][ia][sim].mean(), 3), "C4": r(M["rlBidders"][ix][sim].mean(), 3)},
           "meanNeedToWin": {"C1": r(D["need"][ia][sim].mean(), 3), "C4": r(D["need"][ix][sim].mean(), 3)},
           "topDifferingFeatures": top_features(ps["fdiffShare"], ps["fabsMean"], 10), "byOpponent": {}}
    oal = D["oalgo"][ix]
    for a in ALGOS:
        q = sim & (oal == a)
        qm = q & (np.abs(dbid) >= 2)
        out["byOpponent"][a] = {"pairs": int(q.sum()), "meanDeltaRlBidders": r(dbid[q].mean(), 3), "meanDeltaNeedToWin": r(dneed[q].mean(), 3),
                                "rlBiddersDiffer2plus_rivalBlockLeTight": r((drv[qm] <= DEF["aliasTight"]).mean(), 3) if qm.any() else None,
                                "medianRivalBlockLinf": r(np.median(drv[q]), 4) if q.any() else None}
    # features that could count aggressive rivals: separation C4 vs C1 by progress
    sepf = {}
    for f in ["riv_purse_std", "riv_purse_min", "riv_purse_mean", "riv_capacity_1", "riv_capacity_3", "riv_able_share", "riv_recent_spend", "mkt_recent_price_ratio"]:
        rowsep = []
        for b, lab in enumerate(PLABEL):
            m = sim & (D["pbin"][ix] == b)
            a_ = auc(obs[ix[m], FI[f]], obs[ia[m], FI[f]])
            rowsep.append({"progress": lab, "sep": r(abs(2 * a_ - 1), 3) if a_ is not None else None})
        sepf[f] = rowsep
    out["separationC4vsC1"] = sepf
    # hidden count of RL rivals able to pay the lot vs its best observable proxy (within C4 ∪ C1)
    return out


def main():
    D = load()
    print("rows", D["N"], "episodes", len(D["eps"]))
    L = np.flatnonzero(D["learner"])
    M = D["M"]
    OUT.mkdir(parents=True, exist_ok=True)
    # A/B carriers
    fo = {"definition": {"withinProgressSpearman": "Spearman rank correlation between the hidden variable and the feature inside each main-round progress bin (plus re-auction), averaged by bin size — how well the feature tracks the hidden quantity at a fixed point of the auction",
                         "quality": f"strong |ρ| ≥ {DEF['spearmanStrong']}, moderate ≥ 0.35, weak ≥ {DEF['spearmanWeak']}, none below",
                         "rows": "learner decisions for most mechanisms; keeper mechanisms also use the RL-opponent keeper-lot rows"}, "thresholds": DEF, "mechanisms": {}}
    allrows = np.arange(D["N"])
    for mid, name, hidden, feats, filt in MECHANISMS:
        entry = {"name": name, "candidateFeatures": [{"feature": f, "index": FI[f], "channel": str(CHANNEL[FI[f]]), "scope": INV["features"][FI[f]]["temporalScope"]} for f in feats]}
        if hidden is not None:
            rows = allrows if mid in ("rival_role_demand", "rival_keeper_demand", "keeper_stockpiling") else L
            c = carriers(D, hidden, feats, filt, rows)
            entry["empirical"] = c
            best = c["features"][0]
            entry["bestCarrier"] = {"feature": best["feature"], "withinProgressSpearman": best["withinProgressSpearman"], "quality": quality(best["withinProgressSpearman"])}
        fo["mechanisms"][mid] = entry
        print("A", mid, entry.get("bestCarrier"))
    cf, keep = counterfactual(D)
    print("F done")
    dv = divergence(D, keep)
    cf["divergence"] = dv
    print("divergence", json.dumps(dv["conditions"]["C4"]))
    sg = signals(D, keep)
    print("signals done")
    ko = keeper_obs(D)
    print("keeper done", json.dumps(ko["summary"]))
    ps_ = purse_star(D)
    print("purse-star done")
    mo = multi(D)
    print("multi done")
    al = aliasing(D)
    print("aliasing done", json.dumps({k: (v["materialPairs"], v["aliasRateTight"]) for k, v in al["mechanisms"].items()}))
    meta = {"phase": "2E.2", "source": "digest-verified read-only replays of recorded Phase 2E.0 episodes (replay_obs.mjs); no model trained, fitted or modified",
            "rows": int(D["N"]), "learnerRows": int(len(L)), "opponentRows": int(D["N"] - len(L)), "episodes": len(D["eps"]), "entries": D["E"]["entries"], "thresholds": DEF}
    dump = lambda name, obj: (OUT / name).write_text(json.dumps({"meta": meta, **obj}, indent=1, ensure_ascii=False), encoding="utf-8")
    dump("failure-observability.json", fo)
    dump("counterfactual-pairs.json", cf)
    dump("keeper-observability.json", ko)
    dump("opponent-observability.json", {k: sg[k] for k in ("definition", "separation", "hiddenTracking", "stageASupport", "policyResponse")})
    dump("aliasing-analysis.json", al)
    dump("temporal-analysis.json", {"definition": {"window": "the observation's only explicit history is the pooled 20-lot window (mkt_recent_price_ratio, mkt_recent_sold_share, riv_recent_spend); rival purses / XI totals and the supply counts carry cumulative history implicitly",
                                                   "tracking": sg["definition"]["separation"]},
                                    "divergenceFromStageA": dv,
                                    "separationOverTime": [x for x in sg["separation"] if x["feature"] in WINDOW_F + ["riv_purse_mean", "riv_purse_min", "riv_purse_std", "riv_xi_max", "riv_capacity_1", "self_purse"]],
                                    "recentAndCumulativeEventTracking": sg["hiddenTracking"]})
    dump("purse-star-analysis.json", ps_)
    dump("multi-opponent-analysis.json", mo)
    print("ANALYSISDONE")


if __name__ == "__main__":
    main()
