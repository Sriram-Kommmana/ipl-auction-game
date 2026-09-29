"""Phase 2F — per-decision behaviour of the Stage-B pilot (500k checkpoints) vs the
frozen Stage-A exports, on the same 40 validation entries and the same cells.

Stage B: digest-verified replays of the c500 evaluation episodes (replay_b.mjs).
Stage A: the Phase 2E.2 digest-verified replays (runs/_2e2/full), identical format.
Episode metrics are paired by (algorithm, Stage-A seed index, condition,
opponent export, entry). Responsiveness uses simple within-progress-bin Spearman
correlations and same-lot paired differences — no model is fitted.

    python -m ipl_rl.stage_b.decisions_b   → ml/reports/phase2f/raw/decisions.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ML_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ML_ROOT / "ipl_rl" / "diagnosis"))
import observability as O  # noqa: E402  (rank statistics, feature index)

FI = O.FI
RB = ML_ROOT / "runs/stage_b/eval/replay_c500"
RA = ML_ROOT / "runs/_2e2/full"
OUT = ML_ROOT / "reports/phase2f/raw/decisions.json"
ALGOS = ["ppo", "a2c", "d3qn", "qrdqn", "es"]
R = O.r


def load(d):
    E = json.loads((d / "episodes.json").read_text(encoding="utf-8"))
    M = E["meta"]
    meta = np.fromfile(d / "meta.f32", dtype=np.float32).reshape(-1, len(M))
    obs = np.fromfile(d / "obs.f32", dtype=np.float32).reshape(-1, 80)
    cols = {m: meta[:, i] for i, m in enumerate(M)}
    eps = E["episodes"]
    return cols, obs, eps


def episode_metrics(cols, obs, eps, stage):
    ep = cols["ep"].astype(int)
    L = cols["who"] == 0
    out = {}
    order = np.lexsort((cols["dec"], ep))
    first_crit = {}
    for i in order:
        if not L[i] or cols["shieldState"][i] < 2:
            continue
        e = ep[i]
        if e not in first_crit:
            first_crit[e] = (float(cols["purseShare"][i]), float(cols["progress"][i]) if cols["phase"][i] == 0 else 1.0)
    groups = defaultdict(list)
    for i in np.flatnonzero(L):
        groups[ep[i]].append(i)
    for e, idx in groups.items():
        idx = np.array(idx)
        info = eps[e]
        learner = info["learner"]
        algo = learner.split(":")[0]
        seed = int(learner.split(":s")[1].split("@")[0])
        prog = np.where(cols["phase"][idx] == 0, cols["progress"][idx], 1.0)
        won = cols["winner"][idx] == 0
        star = cols["lotStar"][idx] == 1
        keeper = cols["lotRole"][idx] == 3
        bid = cols["cap"][idx] >= cols["base"][idx]
        need_k = cols["keeperNeed"][idx] > 0
        afford = cols["maxSafe"][idx] >= cols["base"][idx]
        sold_other = (cols["winner"][idx] != 0) & (cols["winner"][idx] != -1)
        star_early = star & (prog < 0.3) & (cols["phase"][idx] == 0)
        kw = prog[keeper & won]
        fc = first_crit.get(e)
        out[(algo, seed - 100 if seed > 100 else seed, info["cond"], info["opp"] if info["cond"] in ("C1", "C4") else "", info["k"])] = {
            "starsBy10": int((star & won & (prog < 0.1)).sum()), "starsBy20": int((star & won & (prog < 0.2)).sum()), "starsBy30": int((star & won & (prog < 0.3)).sum()),
            "earlyStarDecisions": int(star_early.sum()), "earlyStarBidShare": float(bid[star_early].mean()) if star_early.any() else np.nan,
            "earlyStarLostAfterBid": int((star_early & bid & sold_other).sum()),
            "keeperPassAffordableNeeded": int((keeper & need_k & afford & (cols["action"][idx] == 0)).sum()),
            "keeperBidLost": int((keeper & bid & sold_other & need_k).sum()), "keeperWins": int((keeper & won).sum()),
            "firstKeeperWinProgress": float(kw.min()) if len(kw) else np.nan,
            "expensiveLateBuys": int((won & (prog >= 0.75) & (cols["priceFair"][idx] >= 1.5)).sum()),
            "purseAtFirstCritical": fc[0] if fc else np.nan, "progressAtFirstCritical": fc[1] if fc else np.nan, "reachedCritical": 1.0 if fc else 0.0,
            "forcedDecisionShare": float(cols["forced"][idx].mean()), "finalPathDecisions": int(cols["finalPath"][idx].sum()),
            "lateShieldForced": int(((cols["forced"][idx] == 1) & (prog >= 0.9)).sum()),
            "starCapFair": float(np.nanmean(np.clip(cols["capFair"][idx][star & (prog < 0.3)], 0, 5))) if (star & (prog < 0.3)).any() else np.nan,
        }
    return out


def responsiveness(cols, obs, eps, stage):
    """within-progress-bin Spearman of the learner's cap/fair with opponent-pressure features, main round, per algorithm"""
    L = np.flatnonzero((cols["who"] == 0) & (cols["phase"] == 0) & (cols["cap"] >= 0))
    algo = np.array([eps[int(e)]["learner"].split(":")[0] for e in cols["ep"][L]])
    pb = O.pbin(cols["progress"][L], cols["phase"][L])
    y = np.clip(cols["capFair"][L], 0, 5)
    out = {}
    for a in ALGOS:
        m = algo == a
        res = {}
        for f in ["mkt_recent_price_ratio", "riv_recent_spend", "riv_purse_mean", "riv_purse_min", "riv_purse_std", "self_purse"]:
            rhos, ws = [], []
            for b in range(len(O.PBINS)):
                q = m & (pb == b)
                if q.sum() < 300:
                    continue
                s = O.spearman(obs[L[q], FI[f]], y[q], max_n=60_000)
                if s is not None:
                    rhos.append(s); ws.append(q.sum())
            res[f] = R(np.average(rhos, weights=ws), 3) if rhos else None
        out[a] = res
    return out


def same_lot_response(cols, obs, eps):
    """paired same-lot change of the learner's cap/fair, C4 D3QN/QR-DQN room vs Stage-A room (A), similar own state"""
    L = np.flatnonzero((cols["who"] == 0))
    info = [eps[int(e)] for e in cols["ep"][L]]
    key = {}
    for j, i in enumerate(L):
        e = info[j]
        a = e["learner"].split(":")[0]
        s = int(e["learner"].split(":s")[1].split("@")[0])
        if e["cond"] == "A":
            key[(a, s, e["k"], int(cols["phase"][i]), int(cols["slNo"][i]))] = i
    out = defaultdict(lambda: {"dCap": [], "dPrice": [], "dPurse": []})
    for j, i in enumerate(L):
        e = info[j]
        if e["cond"] != "C4" or e["opp"].split(":")[0] not in ("d3qn", "qrdqn"):
            continue
        a = e["learner"].split(":")[0]
        s = int(e["learner"].split(":s")[1].split("@")[0])
        ia = key.get((a, s, e["k"], int(cols["phase"][i]), int(cols["slNo"][i])))
        if ia is None or abs(cols["purseShare"][i] - cols["purseShare"][ia]) > 0.05 or abs(cols["squad"][i] - cols["squad"][ia]) > 1:
            continue
        o = out[a]
        o["dCap"].append(min(cols["capFair"][i], 5) - min(cols["capFair"][ia], 5))
        o["dPrice"].append(obs[i, FI["mkt_recent_price_ratio"]] - obs[ia, FI["mkt_recent_price_ratio"]])
        o["dPurse"].append(obs[i, FI["self_purse"]] - obs[ia, FI["self_purse"]])
    return {a: {"pairs": len(v["dCap"]), "meanDeltaCapFair": R(np.mean(v["dCap"]), 4), "spearman_dCap_dRecentPrice": R(O.spearman(v["dPrice"], v["dCap"]), 3),
                "spearman_dCap_dOwnPurse": R(O.spearman(v["dPurse"], v["dCap"]), 3)} for a, v in out.items()}


def summarise(mb, ma):
    keys = sorted(set(mb) & set(ma))
    names = list(next(iter(mb.values())).keys())
    out = {}
    for c in ("A", "C1", "C4", "S4"):
        out[c] = {}
        for a in ALGOS:
            ks = [k for k in keys if k[0] == a and k[2] == c]
            if not ks:
                continue
            d = {"pairedEpisodes": len(ks)}
            for n in names:
                b = np.array([mb[k][n] for k in ks], float)
                s = np.array([ma[k][n] for k in ks], float)
                seeds = sorted({k[1] for k in ks})
                sm = [np.nanmean([mb[k][n] for k in ks if k[1] == sd]) for sd in seeds]
                d[n] = {"stageB": R(np.nanmean(b)), "stageBsd": R(np.nanstd(sm, ddof=1)) if len(sm) > 1 else None, "stageA": R(np.nanmean(s)),
                        "pairedDiff": R(np.nanmean(b - s)), "totalB": R(np.nansum(b), 1), "totalA": R(np.nansum(s), 1)}
            out[c][a] = d
    return out


def main():
    cb, ob, eb = load(RB)
    ca, oa, ea = load(RA)
    mb = episode_metrics(cb, ob, eb, "B")
    ma = episode_metrics(ca, oa, ea, "A")
    res = {"definition": {"pairing": "Stage-B c500 episode vs the frozen Stage-A export of the same algorithm and seed index, same condition, opponent export and entry (40 Phase 2E.2 entries)",
                          "earlyStarBidShare": "share of the learner's star decisions before 30% of the main round where it bid (cap ≥ base)",
                          "keeperPassAffordableNeeded": "keeper-lot decisions with an unmet keeper need, maxSafeBid ≥ base, and action PASS",
                          "expensiveLateBuys": "purchases at ≥ 75% main-round progress (or re-auction) at ≥ 1.5 × fair value",
                          "purseAtFirstCritical": "own purse share at the first decision whose shield state is CRITICAL or IMPOSSIBLE"},
           "episodes": {"stageB": len(mb), "stageA": len(ma), "paired": len(set(mb) & set(ma))},
           "byCondition": summarise(mb, ma),
           "responsiveness": {"stageB": responsiveness(cb, ob, eb, "B"), "stageA": responsiveness(ca, oa, ea, "A")},
           "sameLotAggressiveRoom": {"stageB": same_lot_response(cb, ob, eb), "stageA": same_lot_response(ca, oa, ea)}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("DECISIONSDONE", res["episodes"])


if __name__ == "__main__":
    main()
