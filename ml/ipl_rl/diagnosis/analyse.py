"""Phase 2E.1 — Stage-B robustness diagnosis (analysis only).

Inputs (all frozen Phase 2E.0 data or digest-verified read-only replays of it):
  ml/runs/_2e1/episodes.npz          raw Phase 2E.0 records, flattened (extract.py)
  ml/runs/_2e1/traj_{A,S4,C1,C4}.jsonl  per-team purchase trajectories (replay.mjs traj)
  ml/runs/_2e1/keeper_traces.json    the 20 findings, per-decision (replay.mjs keeper)
  ml/reports/phase2e0/matchup-results.json

    python analyse.py <report_dir>

Statistics: paired by (export, auction entry) against the Stage-A control;
95% CIs = percentile bootstrap (2,000, fixed seed) over auction entries.
Correlations are associations only.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
R1 = ROOT / "ml/runs/_2e1"
ALGOS = ["ppo", "a2c", "d3qn", "qrdqn", "es"]
NAMES = {"ppo": "PPO", "a2c": "A2C", "d3qn": "D3QN", "qrdqn": "QR-DQN", "es": "OpenAI-ES"}
CONDS = ["A", "C1", "C4", "S4"]
BCONDS = ["C1", "C4", "S4"]
STRATA = ["low", "normal", "high"]
BIN_LABELS = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0", "re-auction end"]
rng_seed = 7


def boot(per_entry, B=2000):
    x = np.asarray(per_entry, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return [None, None]
    idx = np.random.default_rng(rng_seed).integers(0, len(x), size=(B, len(x)))
    m = x[idx].mean(1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def entry_means(v, k):
    v, k = np.asarray(v, float), np.asarray(k)
    ok = ~np.isnan(v)
    out = {}
    for kk, vv in zip(k[ok], v[ok]):
        out.setdefault(kk, []).append(vv)
    return {kk: np.mean(vs) for kk, vs in out.items()}


def mci(v, k):
    v = np.asarray(v, float)
    ok = ~np.isnan(v)
    if not ok.any():
        return {"mean": None, "ci95": [None, None], "n": 0}
    em = entry_means(v, k)
    return {"mean": float(v[ok].mean()), "ci95": boot(list(em.values())), "n": int(ok.sum())}


def rankdata(x):
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    # average ties
    xs = x[o]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[ok], b[ok]
    if len(a) < 10 or a.std() == 0 or b.std() == 0:
        return {"pearson": None, "spearman": None, "n": int(len(a))}
    return {"pearson": float(np.corrcoef(a, b)[0, 1]), "spearman": float(np.corrcoef(rankdata(a), rankdata(b))[0, 1]), "n": int(len(a))}


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def load():
    D = dict(np.load(R1 / "episodes.npz", allow_pickle=False))
    D["spendShare"] = D["purseSpent"] / D["purse"]
    n = len(D["xi"])
    a_rows = {(l, k): i for i, (c, l, k) in enumerate(zip(D["cond"], D["learner"], D["k"])) if c == "A"}
    D["pairA"] = np.array([a_rows[(l, k)] for l, k in zip(D["learner"], D["k"])])
    D["dXI"] = D["xi"] - D["xi"][D["pairA"]]
    return D


def sel(D, cond=None, algo=None, opp=None, stratum=None):
    m = np.ones(len(D["xi"]), bool)
    if cond:
        m &= D["cond"] == cond
    if algo:
        m &= D["algo"] == algo
    if opp:
        m &= D["oppAlgo"] == opp
    if stratum:
        m &= D["stratum"] == stratum
    return m


def load_traj():
    T = {}
    for c in CONDS:
        p = R1 / f"traj_{c}.jsonl"
        if p.exists():
            T[c] = [json.loads(l) for l in open(p, encoding="utf-8")]
    return T


# ── A. transfer decomposition ───────────────────────────────────────────
DECOMP = [("xi", "Best XI"), ("strongXI", "strong XI rate"), ("legalXI", "legal XI rate"), ("rank", "rank"), ("purseLeftShare", "purse remaining (share)"),
          ("squadSize", "squad size"), ("overseas", "overseas count"), ("stars", "stars"), ("priceToFair", "price/fair"), ("capToFair", "cap/fair"),
          ("bidRate", "bid rate"), ("spendShare", "total spend (share of purse)"), ("xiGainPer1000", "XI gain per ₹1000L"), ("marginalBuys", "marginal purchases"),
          ("reauctionBuys", "re-auction purchases"), ("keeperProg", "keeper completion (progress)"), ("bowlingProg", "bowling completion (progress)"),
          ("indiansProg", "Indian-requirement completion (progress)"), ("shieldActivations", "shield activations"), ("forced", "forced bids"),
          ("forcedKeeperBids", "forced keeper bids"), ("finalPath", "final-path forced bids"), ("decisions", "decisions"), ("contests", "contested lots")]


def analysis_A(D):
    out = {"metrics": [m for m, _ in DECOMP], "byAlgorithm": {}, "correlationWithDeltaXI": {}, "cellLevel": {}}
    for a in ALGOS:
        rows = {}
        for m, label in DECOMP:
            base = np.nanmean(D[m][sel(D, "A", a)])
            r = {"label": label, "A": r4(base)}
            for c in BCONDS:
                v = np.nanmean(D[m][sel(D, c, a)])
                r[c] = r4(v)
                r[f"{c}_abs"] = r4(v - base)
                r[f"{c}_rel"] = r4((v - base) / abs(base)) if base not in (0, None) and not np.isnan(base) and abs(base) > 1e-12 else None
            rows[m] = r
        out["byAlgorithm"][a] = rows
    # episode-level paired deltas vs ΔXI (Stage-B episodes only)
    B = D["cond"] != "A"
    pa = D["pairA"]
    for scope in ["all"] + ALGOS:
        m = B & (D["algo"] == scope if scope != "all" else True)
        res = {}
        for met, label in DECOMP:
            if met == "xi":
                continue
            d = D[met][m] - D[met][pa[m]]
            res[met] = {"label": label, **corr(d, D["dXI"][m])}
        out["correlationWithDeltaXI"][scope] = dict(sorted(res.items(), key=lambda kv: -abs(kv[1]["spearman"] or 0)))
    # cell level (45 learner×composition×opponent cells)
    cells = []
    for c in BCONDS:
        for a in ALGOS:
            opps = [b for b in ALGOS if b != a] if c != "S4" else [""]
            for b in opps:
                m = sel(D, c, a, b or None)
                row = {"cond": c, "learner": a, "opp": b, "transfer": float(D["dXI"][m].mean())}
                for met, _ in DECOMP:
                    if met != "xi":
                        row[met] = float(np.nanmean(D[met][m] - D[met][pa[m]]))
                for met in ("opp_bidRate", "opp_priceToFair", "opp_capToFair", "opp_purseLeftShare", "opp_stars", "opp_squadSize"):
                    row[met] = float(np.nanmean(D[met][m]))
                cells.append(row)
    out["cellLevel"]["cells"] = cells
    t = np.array([c["transfer"] for c in cells])
    out["cellLevel"]["correlationWithTransfer"] = dict(sorted({k: corr([c[k] for c in cells], t) for k in cells[0] if k not in ("cond", "learner", "opp", "transfer")}.items(),
                                                                  key=lambda kv: -abs(kv[1]["pearson"] or 0)))
    out["note"] = "Correlations are associations between paired changes (Stage-B episode minus the same export's Stage-A episode on the same auction); not causal."
    return out


# ── B. opponent effect (C1) ─────────────────────────────────────────────
def analysis_B(D):
    m1 = D["cond"] == "C1"
    mat = {}
    for a in ALGOS:
        for b in ALGOS:
            if a == b:
                continue
            m = m1 & (D["algo"] == a) & (D["oppAlgo"] == b)
            mat[f"{a}>{b}"] = {
                "deltaXI": mci(D["dXI"][m], D["k"][m]), "learnerXI": r4(D["xi"][m].mean()), "opponentXI": r4(D["opp_xi"][m].mean()),
                "learnerPriceFair": r4(np.nanmean(D["priceToFair"][m])), "opponentPriceFair": r4(np.nanmean(D["opp_priceToFair"][m])),
                "learnerCapFair": r4(np.nanmean(D["capToFair"][m])), "opponentCapFair": r4(np.nanmean(D["opp_capToFair"][m])),
                "learnerBidRate": r4(D["bidRate"][m].mean()), "opponentBidRate": r4(D["opp_bidRate"][m].mean()),
                "learnerPurseLeft": r4(D["purseLeftShare"][m].mean()), "opponentPurseLeft": r4(D["opp_purseLeftShare"][m].mean()),
                "learnerStars": r4(D["stars"][m].mean()), "opponentStars": r4(D["opp_stars"][m].mean()),
                "learnerShieldActivationsPerEpisode": r4(D["shieldActivations"][m].mean()), "learnerForcedKeeperPurchasesPer500": r4(D["keeperForced"][m].mean() * 500),
                "learnerFinalPathForcedBids": int(D["finalPath"][m].sum()),
                "requirementFailures": {"learnerIncompleteXI": int((D["legalXI"][m] == 0).sum()), "opponentIncompleteXI": int(D["findingOpp"][m].sum())},
                "learnerKeeperCompletionProgress": r4(np.nanmean(D["keeperProg"][m])),
                "opponentShieldUsage": "not recorded for opponent seats in Phase 2E.0 (runtime does not expose the shield state); see the reverse pairing",
            }
    # two-way additive decomposition of the C1 transfer matrix
    y = np.array([mat[f"{a}>{b}"]["deltaXI"]["mean"] for a in ALGOS for b in ALGOS if a != b])
    Xl = np.array([[1.0 if a == x else 0.0 for x in ALGOS] for a in ALGOS for b in ALGOS if a != b])
    Xo = np.array([[1.0 if b == x else 0.0 for x in ALGOS] for a in ALGOS for b in ALGOS if a != b])

    def fit(X):
        X1 = np.hstack([np.ones((len(y), 1)), X])
        beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
        return 1 - ((y - X1 @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum(), beta
    r2_l, _ = fit(Xl)
    r2_o, _ = fit(Xo)
    r2_lo, beta = fit(np.hstack([Xl, Xo]))
    learner_eff = {a: float(np.mean([mat[f"{a}>{b}"]["deltaXI"]["mean"] for b in ALGOS if b != a])) for a in ALGOS}
    opp_eff = {b: float(np.mean([mat[f"{a}>{b}"]["deltaXI"]["mean"] for a in ALGOS if a != b])) for b in ALGOS}
    # opponent style (as observed when that algorithm is the opponent), and its association with the damage it is associated with
    style = {}
    for b in ALGOS:
        m = m1 & (D["oppAlgo"] == b)
        style[b] = {"bidRate": r4(D["opp_bidRate"][m].mean()), "capFair": r4(np.nanmean(D["opp_capToFair"][m])), "priceFair": r4(np.nanmean(D["opp_priceToFair"][m])),
                    "purseLeft": r4(D["opp_purseLeftShare"][m].mean()), "stars": r4(D["opp_stars"][m].mean()), "squad": r4(D["opp_squadSize"][m].mean()),
                    "xi": r4(D["opp_xi"][m].mean()), "meanTransferCaused": r4(opp_eff[b])}
    cells = [(mat[k]["deltaXI"]["mean"], mat[k]) for k in mat]
    style_corr = {v: corr([c[1][v] for c in cells], [c[0] for c in cells]) for v in ("opponentBidRate", "opponentCapFair", "opponentPriceFair", "opponentPurseLeft", "opponentStars", "opponentXI")}
    return {"C1matrix": mat, "learnerMainEffect": learner_eff, "opponentMainEffect": opp_eff,
            "varianceExplained": {"learnerOnly": float(r2_l), "opponentOnly": float(r2_o), "additiveLearnerPlusOpponent": float(r2_lo),
                                  "note": "R² of least-squares fits to the 20 C1 transfer cells; the remainder is learner×opponent interaction"},
            "opponentStyle": style, "styleVsTransferAcrossCells": style_corr,
            "C4matrixSummary": {f"{a}>{b}": r4(D["dXI"][sel(D, "C4", a, b)].mean()) for a in ALGOS for b in ALGOS if a != b}}


# ── C. C1 → C4 escalation ───────────────────────────────────────────────
def room_stats(rec):
    """Room-level quantities from a trajectory record."""
    T = rec["teams"]
    rl_opp = [t for t in T if t["type"] == "rlSnapshot"]
    rule = [t for t in T if t["type"] in ("rule", "rlFallback")]
    learner = next(t for t in T if t["type"] == "learner")
    buys = sum(t["squad"][-1] for t in T)
    pf_w = sum((t["pf"] or 0) * t["squad"][-1] for t in T) / max(1, buys)
    early = lambda t, i: t["keepers"][i]
    return {
        "roomPriceFair": pf_w,
        "rlOppKeepers": sum(t["keepers"][-1] for t in rl_opp), "rlOppKeepersBy03": sum(early(t, 2) for t in rl_opp),
        "rlOppBowl": sum(t["bowl"][-1] for t in rl_opp), "rlOppIndians": sum(t["indians"][-1] for t in rl_opp), "rlOppOverseas": sum(t["overseas"][-1] for t in rl_opp),
        "rlOppStars": sum(t["stars"][-1] for t in rl_opp), "rlOppStarsBy03": sum(t["stars"][2] for t in rl_opp), "rlOppSpendBy03": sum(t["spendShare"][2] for t in rl_opp),
        "rlOppSquad": sum(t["squad"][-1] for t in rl_opp),
        "ruleKeepers": sum(t["keepers"][-1] for t in rule), "ruleStars": sum(t["stars"][-1] for t in rule), "ruleXI": float(np.mean([t["xi"] for t in rule])),
        "allKeepersBy03": sum(early(t, 2) for t in T if t["type"] != "learner"),
        "learnerKeeperPriceFair": float(np.mean([k[2] for k in learner["keeperBuys"]])) if learner["keeperBuys"] else np.nan,
        "learnerFirstKeeperProg": learner["keeperBuys"][0][0] if learner["keeperBuys"] else np.nan,
        "learnerStarsBy03": learner["stars"][2], "learnerSpendBy03": learner["spendShare"][2],
    }


def analysis_C(D, T, MR):
    pairs = {}
    for a in ALGOS:
        for b in ALGOS:
            if a == b:
                continue
            m1, m4 = sel(D, "C1", a, b), sel(D, "C4", a, b)
            e1, e4 = entry_means(D["dXI"][m1], D["k"][m1]), entry_means(D["dXI"][m4], D["k"][m4])
            diff = [e4[k] - e1[k] for k in e1 if k in e4]
            t1, t4 = float(D["dXI"][m1].mean()), float(D["dXI"][m4].mean())
            row = {"C1": r4(t1), "C4": r4(t4), "escalation": r4(t4 - t1), "escalationCI95": boot(diff), "ratioC4overC1": r4(t4 / t1),
                   "linearIf4xC1": r4(4 * t1)}
            for met in ("priceToFair", "capToFair", "bidRate", "stars", "purseLeftShare", "squadSize", "overseas", "keeperProg", "keeperForced", "forced", "marginalBuys", "minPurseAnyUnmet"):
                row[f"learner_{met}"] = {"C1": r4(np.nanmean(D[met][m1])), "C4": r4(np.nanmean(D[met][m4]))}
            for met in ("priceToFair", "bidRate", "purseLeftShare", "stars", "xi", "squadSize", "overseas"):
                row[f"opponentPerCopy_{met}"] = {"C1": r4(np.nanmean(D[f"opp_{met}"][m1])), "C4": r4(np.nanmean(D[f"opp_{met}"][m4]))}
            pairs[f"{a}>{b}"] = row
    by_opp = {b: {"meanEscalation": r4(np.mean([pairs[f"{a}>{b}"]["escalation"] for a in ALGOS if a != b])),
                  "meanRatio": r4(np.mean([pairs[f"{a}>{b}"]["ratioC4overC1"] for a in ALGOS if a != b]))} for b in ALGOS}
    by_learner = {a: {"meanEscalation": r4(np.mean([pairs[f"{a}>{b}"]["escalation"] for b in ALGOS if b != a])),
                      "meanRatio": r4(np.mean([pairs[f"{a}>{b}"]["ratioC4overC1"] for b in ALGOS if b != a]))} for a in ALGOS}
    # role competition from the digest-verified s1 × s1 replays
    roles = {}
    if "C1" in T and "C4" in T and "A" in T:
        rs = {c: defaultdict(list) for c in ("A", "C1", "C4")}
        for c in ("A", "C1", "C4"):
            for rec in T[c]:
                if not rec["learner"].endswith(":s1"):
                    continue
                key = (rec["learner"].split(":")[0], rec["opponents"][0].split(":")[0] if rec["opponents"] else "")
                rs[c][key].append(room_stats(rec))
        for a in ALGOS:
            base = rs["A"][(a, "")]
            for b in ALGOS:
                if a == b:
                    continue
                r = {}
                for q in base[0]:
                    r[q] = {c: r4(np.nanmean([x[q] for x in (base if c == "A" else rs[c][(a, b)])])) for c in ("A", "C1", "C4")}
                roles[f"{a}>{b}"] = r
        # opponent-level averages
        roles_by_opp = {}
        for b in ALGOS:
            ks = [f"{a}>{b}" for a in ALGOS if a != b]
            roles_by_opp[b] = {q: {c: r4(np.nanmean([roles[k][q][c] for k in ks])) for c in ("A", "C1", "C4")} for q in roles[ks[0]]}
    else:
        roles_by_opp = {}
    esc = np.array([pairs[k]["escalation"] for k in pairs])
    # association of escalation with the change in opponent-copy style and room pressure
    assoc = {}
    for v in ("opponentPerCopy_priceToFair", "opponentPerCopy_bidRate", "opponentPerCopy_purseLeftShare", "opponentPerCopy_stars"):
        assoc[f"C1 level of {v}"] = corr([pairs[k][v]["C1"] for k in pairs], esc)
    if roles:
        for q in ("roomPriceFair", "rlOppKeepers", "rlOppKeepersBy03", "rlOppStars", "rlOppStarsBy03", "rlOppOverseas", "rlOppBowl", "rlOppIndians", "rlOppSpendBy03"):
            assoc[f"C4−C1 change in {q}"] = corr([(roles[k][q]["C4"] or 0) - (roles[k][q]["C1"] or 0) for k in pairs], esc)
    return {"pairs": pairs, "byOpponent": by_opp, "byLearner": by_learner, "roleCompetition_s1xs1": {"byPair": roles, "byOpponent": roles_by_opp,
            "note": "digest-verified read-only replays of the recorded s1×s1 episodes (20 ordered pairs × 500 auctions per composition) and the s1 Stage-A controls"},
            "escalationAssociations": assoc,
            "note": "escalation = C4 transfer − C1 transfer (more negative = extra degradation when the opponent is multiplied to four seats); CI = bootstrap over auction entries of the per-entry difference"}


# ── D. S4 decomposition ─────────────────────────────────────────────────
def analysis_D(D, T):
    out = {}
    trajS4 = {}
    for rec in T.get("S4", []):
        trajS4[(rec["learner"], rec["k"])] = rec
    for a in ALGOS:
        m = sel(D, "S4", a)
        per = {}
        for b in ALGOS:
            if b == a:
                continue
            per[b] = {x: r4(np.nanmean(D[f"s4_{b}_{x}"][m])) for x in ("xi", "rank", "purseLeftShare", "priceToFair", "bidRate", "capToFair", "stars", "squadSize", "overseas")}
            per[b]["assocWithLearnerDeltaXI"] = {x: corr(D[f"s4_{b}_{x}"][m], D["dXI"][m])["spearman"] for x in ("xi", "stars", "priceToFair", "purseLeftShare")}
        # trajectory-based: keepers / early spend / stars by opponent algorithm seat
        kt = defaultdict(lambda: defaultdict(list))
        learner_keeper_prog, learner_dxi = [], []
        idx = {(l, k): i for i, (l, k) in enumerate(zip(D["learner"][m], D["k"][m]))}
        rows_m = np.where(m)[0]
        for (l, k), rec in trajS4.items():
            if l.split(":")[0] != a:
                continue
            for t in rec["teams"]:
                if t["type"] != "rlSnapshot":
                    continue
                b = t["key"].split(":")[0]
                kt[b]["keepers"].append(t["keepers"][-1])
                kt[b]["keepersBy03"].append(t["keepers"][2])
                kt[b]["spendBy03"].append(t["spendShare"][2])
                kt[b]["starsBy03"].append(t["stars"][2])
                kt[b]["keeperPriceFair"].extend([x[2] for x in t["keeperBuys"]])
        for b in kt:
            per[b].update({q: r4(np.mean(v)) for q, v in kt[b].items()})
        pick = lambda q, hi=True: max(per, key=lambda b: (per[b][q] if per[b][q] is not None else -1e9) * (1 if hi else -1))
        out[a] = {"perOpponentSeat": per,
                  "highestPriceFair": pick("priceToFair"), "mostKeepersBy03": pick("keepersBy03") if kt else None, "mostSpendBy03": pick("spendBy03") if kt else None,
                  "mostStarsBy03": pick("starsBy03") if kt else None, "lowestPurseLeft": pick("purseLeftShare", hi=False), "highestXI": pick("xi"),
                  "learner": {"xi": r4(D["xi"][m].mean()), "transfer": mci(D["dXI"][m], D["k"][m]), "keeperProg": r4(np.nanmean(D["keeperProg"][m])),
                              "keeperForcedPer500": r4(D["keeperForced"][m].mean() * 500), "forcedPerEpisode": r4(D["forced"][m].mean())}}
    out["note"] = ("All four opponent algorithms are present in every S4 room, so the per-seat figures describe what each algorithm DID in the room; "
                   "the experiment cannot attribute the learner's loss to one opponent. Associations are Spearman correlations across the 1,500 rooms of a learner.")
    return out


# ── E. policy behaviour shift ───────────────────────────────────────────
SHIFT = [("bidRate", "bid rate"), ("priceToFair", "price/fair"), ("capToFair", "cap/fair"), ("stars", "stars"), ("marginalBuys", "marginal purchases"),
         ("squadSize", "squad size"), ("purseLeftShare", "purse remaining (share)"), ("overseas", "overseas count"), ("reauctionBuys", "re-auction purchases"),
         ("keeperProg", "keeper completion progress"), ("bowlingProg", "bowling completion progress"), ("indiansProg", "Indian completion progress"),
         ("shieldActivations", "shield activations / episode"), ("forced", "forced bids / episode"), ("decisions", "decisions / episode")]


def analysis_E(D, T):
    tables = {}
    for a in ALGOS:
        rows = []
        for m, label in SHIFT:
            v = {c: float(np.nanmean(D[m][sel(D, c, a)])) for c in CONDS}
            rows.append({"metric": label, **{c: r4(v[c]) for c in CONDS}, "changeS4minusA": r4(v["S4"] - v["A"]),
                         "relChangeS4": r4((v["S4"] - v["A"]) / abs(v["A"])) if abs(v["A"]) > 1e-12 else None})
        tables[a] = rows
    # action distributions
    acts = {}
    for a in ALGOS:
        acts[a] = {}
        for c in CONDS:
            ac = np.stack(D["actions"][sel(D, c, a)]).sum(0).astype(float)
            share = ac / ac.sum()
            bids = ac[1:].sum()
            acts[a][c] = {"passShare": r4(share[0]), "topAction": int(np.argmax(ac[1:]) + 1), "topActionShareOfBids": r4(ac[1:].max() / bids),
                          "FV_1.25ShareOfBids": r4(ac[12] / bids), "MAX_SAFEShareOfBids": r4(ac[19] / bids), "shares": [r4(x) for x in share]}
    # same-lot paired decisions (adaptation test)
    adapt = {}
    if "A" in T:
        A_lots = {}
        for rec in T["A"]:
            A_lots[(rec["learner"], rec["k"])] = {s: (cf, ps) for s, cf, ps in rec["lots"]}
        for c in ("C1", "C4", "S4"):
            if c not in T:
                continue
            per = defaultdict(lambda: {"n": 0, "d": [], "dp": [], "same": 0})
            for rec in T[c]:
                base = A_lots.get((rec["learner"], rec["k"]))
                if not base:
                    continue
                a = rec["learner"].split(":")[0]
                for s, cf, ps in rec["lots"]:
                    if s in base:
                        cfa, psa = base[s]
                        per[a]["n"] += 1
                        per[a]["d"].append(cf - cfa)
                        per[a]["dp"].append(ps - psa)
                        per[a]["same"] += int(cf == cfa)
            adapt[c] = {}
            for a, v in per.items():
                d, dp = np.array(v["d"]), np.array(v["dp"])
                similar = np.abs(dp) < 0.05
                adapt[c][a] = {"pairedLots": v["n"], "meanDeltaCapFair": r4(d.mean()), "identicalCapShare": r4(v["same"] / v["n"]),
                               "similarPurse(|Δpurse share|<0.05)": {"lots": int(similar.sum()), "meanDeltaCapFair": r4(d[similar].mean()) if similar.any() else None,
                                                                    "identicalCapShare": r4((d[similar] == 0).mean()) if similar.any() else None},
                               "differentPurse": {"lots": int((~similar).sum()), "meanDeltaCapFair": r4(d[~similar].mean()) if (~similar).any() else None},
                               "corrDeltaCapWithDeltaPurse": corr(dp, d)["spearman"]}
    return {"behaviourTables": tables, "actionDistributions": acts, "sameLotAdaptation": adapt,
            "note": ("sameLotAdaptation pairs the learner's decision on the SAME player in the SAME auction entry between the Stage-A control and the Stage-B room "
                     "(main round, lots decided in both). With a similar own purse share, an unchanged cap indicates the policy's offer is insensitive to the new opponents; "
                     "a changed cap indicates a response to other state features (opponent-related observation features, squad/needs).")}


# ── F. keeper forensics ─────────────────────────────────────────────────
def analysis_F():
    K = json.loads((R1 / "keeper_traces.json").read_text(encoding="utf-8"))
    table = []
    for t in K:
        f, rows, kl = t["finding"], t["decisions"], t["keeperLots"]
        who = f["key"]
        keeper_rows = [r for r in rows if r["role"] == "WICKET KEEPER"]
        passes_cheap_safe = [r for r in keeper_rows if r["hasBid"] and r["action"] == 0 and r["shieldState"] == "SAFE" and r["need"]["keeper"] > 0]
        unsold_after_pass = [r for r in passes_cheap_safe if r["result"] == "unsold"]
        forced = [r for r in rows if r["forced"]]
        forced_keeper = [r for r in forced if any(x == "keeper" or x.startswith("keeper") or x.startswith("class:W") for x in r["forcedBy"])]
        first_warn = next((r for r in rows if r["need"]["keeper"] > 0 and r["shieldState"] in ("WARNING", "CRITICAL", "IMPOSSIBLE")), None)
        first_bad = next((r for r in rows if r["alreadyInfeasible"] or r["shieldState"] == "IMPOSSIBLE"), None)
        kl_after_last_forced = [x for x in kl if forced_keeper and (x["phase"], x["progress"]) > (forced_keeper[-1]["phase"], forced_keeper[-1]["progress"])]
        # purse / squad / overseas trajectory (sampled at the affected seat's decisions)
        traj = [(r["progress"] if r["phase"] == "main" else 1.0, r["purse"], r["squad"], r["overseas"]) for r in rows]
        def at(p):
            prev = [x for x in traj if x[0] <= p]
            return prev[-1] if prev else traj[0]
        need_zero_other = rows[-1]["need"] if rows else {}
        lost_forced = [r for r in forced_keeper if r["result"] != "won"]
        winners = [r["result"] for r in lost_forced]
        # reason
        fin = t["final"]
        last_forced = (forced_keeper[-1]["phase"], forced_keeper[-1]["progress"]) if forced_keeper else ("main", -1)
        # keeper lots after the last forced keeper bid that the seat could still LEGALLY buy
        # (base ≤ purse at the end, squad < 25, overseas cap respected) — upper bound, uses end-of-auction state
        legal_after = [x for x in kl if (x["phase"], x["progress"]) > last_forced and x["base"] <= fin["purseLeft"] and fin["squad"] < 25 and (not x["os"] or fin["overseas"] < 8)]
        remaining_unsold = [x for x in kl if x["winner"] is None and x["phase"] == "reauction"]
        reason = []
        if lost_forced:
            reason.append(f"lost {len(lost_forced)} forced keeper bid(s) ({', '.join(winners)})")
        if first_bad:
            reason.append(f"shield IMPOSSIBLE from progress {first_bad['progress']} ({first_bad['phase']})")
        final_purse = t["final"]["purseLeft"]
        affordable_left = [x for x in remaining_unsold if x["base"] <= final_purse and (not x["os"] or fin["overseas"] < 8) and fin["squad"] < 25]
        reason.append(f"final purse ₹{final_purse}L, squad {t['final']['squad']}, overseas {t['final']['overseas']}")
        if t["final"]["squad"] >= 25:
            reason.append("squad full (25)")
        rival_depth = [x for x in t["teamsKeepers"] if x["keepers"] >= 3 and x["seat"] != f["seat"]]
        # chart points: purse share and keeper events over progress (re-auction mapped to 1.0–1.1 by order)
        ra = [r for r in rows if r["phase"] == "reauction"]
        pos = lambda r: r["progress"] if r["phase"] == "main" else 1.0 + 0.1 * (ra.index(r) + 1) / (len(ra) + 1)
        trace_pts = [[round(pos(r), 4), round(r["purse"] / t["room"]["purse"], 4)] for r in rows]
        events = []
        for r in keeper_rows:
            if r["forced"]:
                events.append([round(pos(r), 4), "forced"])
            elif r["hasBid"] and r["action"] != 0 and r["result"] != "won":
                events.append([round(pos(r), 4), "lost"])
            elif r in passes_cheap_safe:
                events.append([round(pos(r), 4), "pass"])
        table.append({
            "_trace_points": trace_pts, "_event_points": events,
            "seed": t["room"]["seed"], "stratum": t["room"]["stratum"], "purse": t["room"]["purse"], "room": f"learner {t['room']['learner']} vs 4× {t['room']['opponents'][0]}",
            "affected": who, "affectedType": f["type"], "seat": f["seat"],
            "purseTrajectory": {"p0.25": at(0.25)[1], "p0.5": at(0.5)[1], "p0.75": at(0.75)[1], "end": final_purse},
            "squadTrajectory": {"p0.25": at(0.25)[2], "p0.5": at(0.5)[2], "p0.75": at(0.75)[2], "end": t["final"]["squad"]},
            "overseasTrajectory": {"p0.25": at(0.25)[3], "p0.5": at(0.5)[3], "p0.75": at(0.75)[3], "end": t["final"]["overseas"]},
            "keepersOwnedEnd": t["final"]["keepers"],
            "keeperLotsInAuction": len(kl), "keeperLotsWithLegalBid": sum(1 for r in keeper_rows if r["hasBid"]),
            "keeperPassesWhileSafe": len(passes_cheap_safe), "ofWhichWentUnsold": len(unsold_after_pass),
            "keeperBidsLost": sum(1 for r in keeper_rows if r["action"] != 0 and r["result"] not in ("won",)),
            "firstNonSafeKeeperState": {"progress": first_warn["progress"], "phase": first_warn["phase"], "state": first_warn["shieldState"], "purse": first_warn["purse"],
                                        "maxSafeBid": first_warn["maxSafeBid"], "keeperAnalysis": first_warn["keeperAnalysis"]} if first_warn else None,
            "forcedKeeperBids": [{"lot": r["lot"], "phase": r["phase"], "progress": r["progress"], "state": r["shieldState"], "finalPath": r["finalPath"], "cap": r["cap"],
                                  "purse": r["purse"], "maxSafeBid": r["maxSafeBid"], "result": r["result"], "soldPrice": r["soldPrice"],
                                  "maxRivalPurse": r["maxRivalPurse"], "rivalsAbleToPayBase": r["rivalsAbleToPayBase"],
                                  "keeperCandidatesRemaining": [x["viableAfterLot"] for x in r["keeperAnalysis"]], "rivalKeeperDemand": [x["rivalsNeeding"] for x in r["keeperAnalysis"]]} for r in forced_keeper],
            "keeperLotsAfterLastForced": len(kl_after_last_forced), "keeperLotsAfterLastForcedLegallyBuyable": len(legal_after),
            "unsoldReauctionKeepersAffordableAtEnd": len(affordable_left),
            "viableKeepersAfterFinalForcedBid": forced_keeper[-1]["keeperAnalysis"][-1]["viableAfterLot"] if forced_keeper and forced_keeper[-1]["keeperAnalysis"] else None,
            "purseShareAt025": round(at(0.25)[1] / t["room"]["purse"], 4), "overseasCapReachedBy": next((x[0] for x in traj if x[3] >= 8), None),
            "firstForcedPhase": forced_keeper[0]["phase"] if forced_keeper else None,
            "keepersHeldByRivalsWith3Plus": sum(x["keepers"] for x in rival_depth),
            "rivalsWith3PlusKeepers": [f"{x['who']} ({x['keepers']})" for x in rival_depth],
            "finalReason": "; ".join(reason),
        })
    # mechanism classification (from the traces)
    for x in table:
        m1 = x["keeperPassesWhileSafe"] >= 5 and x["firstNonSafeKeeperState"] and x["firstNonSafeKeeperState"]["phase"] == "reauction"
        m2 = x["purseShareAt025"] < 0.05 and x["firstForcedPhase"] == "main"
        x["mechanism"] = ("M1: passed on cheap keepers while SAFE; requirement became CRITICAL only in the re-auction; forced bids with an almost-empty purse lost" if m1
                          else "M2: purse and overseas slots exhausted in the first quarter; repeated keeper bids (mostly forced) at ≤ ₹50L lost to rivals with large purses" if m2
                          else "other")
    # hypothesis test
    H = {
        "H1 requirement SAFE while the seat passed on keeper lots": sum(1 for x in table if x["keeperPassesWhileSafe"] > 0),
        "H1b of those, at least one passed keeper went unsold (cheap keeper available)": sum(1 for x in table if x["ofWhichWentUnsold"] > 0),
        "H2 keeper requirement later reached WARNING/CRITICAL": sum(1 for x in table if x["firstNonSafeKeeperState"]),
        "H3 shield forced ≥1 keeper bid": sum(1 for x in table if x["forcedKeeperBids"]),
        "H4 every forced keeper bid lost": sum(1 for x in table if x["forcedKeeperBids"] and all(b["result"] != "won" for b in x["forcedKeeperBids"])),
        "H5 the winning rival could pay more (max rival purse > the seat's cap at the lost forced bid)": sum(1 for x in table if x["forcedKeeperBids"] and any(b["maxRivalPurse"] > b["cap"] for b in x["forcedKeeperBids"])),
        "H6 no viable keeper left after the final forced bid (the shield's own viable-candidate count = 0)": sum(1 for x in table if x["viableKeepersAfterFinalForcedBid"] == 0),
    }
    H["all six steps hold"] = sum(1 for x in table if x["keeperPassesWhileSafe"] > 0 and x["firstNonSafeKeeperState"] and x["forcedKeeperBids"]
                                  and all(b["result"] != "won" for b in x["forcedKeeperBids"]) and any(b["maxRivalPurse"] > b["cap"] for b in x["forcedKeeperBids"])
                                  and x["viableKeepersAfterFinalForcedBid"] == 0)
    mech = defaultdict(list)
    for x in table:
        mech[x["mechanism"].split(":")[0]].append(f"{x['seed']} {x['affected']} ({x['affectedType']})")
    return {"findings": table, "hypothesisTest": H, "mechanisms": dict(mech), "n": len(table)}


# ── G. shield ───────────────────────────────────────────────────────────
def analysis_G(D):
    out = {}
    for a in ALGOS:
        out[a] = {}
        for c in CONDS:
            m = sel(D, c, a)
            dec = D["decisions"][m].sum()
            forced = D["forced"][m].sum()
            out[a][c] = {
                "episodes": int(m.sum()), "decisions": int(dec),
                "activationRatePerDecision": r4(D["shieldActivations"][m].sum() / dec), "forcedPerEpisode": r4(D["forced"][m].mean()),
                "episodesWithForcedBid": r4((D["forced"][m] > 0).mean()), "keeperForcedPerEpisode": r4(D["forcedKeeperBids"][m].mean()),
                "finalPathForcedTotal": int(D["finalPath"][m].sum()), "finalPathWon": int(D["finalPathWon"][m].sum()),
                "forcedWon": int(D["forcedWon"][m].sum()), "forcedLost": int(D["forcedLost"][m].sum()),
                "forcedWinRate": r4(D["forcedWon"][m].sum() / forced) if forced else None,
                "stateShare": {s: r4(D[f"state{s}"][m].sum() / dec) for s in ("SAFE", "WARNING", "CRITICAL", "IMPOSSIBLE")},
                "reauctionForced": int(D["reauctionForced"][m].sum()),
                "keeperClosedByForcedBidPer500": r4(D["keeperForced"][m].mean() * 500),
                "episodesWithAnyNonSafeState": r4(((D["stateWARNING"][m] + D["stateCRITICAL"][m] + D["stateIMPOSSIBLE"][m]) > 0).mean()),
            }
    fp = []
    for c in BCONDS:
        for a in ALGOS:
            for b in (ALGOS if c != "S4" else [""]):
                m = sel(D, c, a, b or None)
                n = int(D["finalPath"][m].sum())
                if n:
                    fp.append({"cond": c, "learner": a, "opponent": b or "S4 room", "finalPathForcedBids": n, "won": int(D["finalPathWon"][m].sum()), "episodes": int((D["finalPath"][m] > 0).sum())})
    return {"byAlgorithm": out, "finalPathForcedBids": {"StageA": int(D["finalPath"][D["cond"] == "A"].sum()), "StageB": int(D["finalPath"][D["cond"] != "A"].sum()), "byCell": fp}}


# ── H. purse dynamics ───────────────────────────────────────────────────
def analysis_H(D, T):
    out = {"bins": BIN_LABELS, "curves": {}, "xiRelationships": {}}
    ridx = {(c, l, k): i for i, (c, l, k) in enumerate(zip(D["cond"], D["learner"], D["k"])) if c in ("A", "S4")}
    for c in ("A", "S4"):
        if c not in T:
            continue
        per = defaultdict(lambda: defaultdict(list))
        feats = defaultdict(lambda: defaultdict(list))
        for rec in T[c]:
            a = rec["learner"].split(":")[0]
            L = next(t for t in rec["teams"] if t["type"] == "learner")
            per[a]["spend"].append(L["spendShare"])
            per[a]["squad"].append(L["squad"])
            per[a]["keeperLeft"].append([max(0, 1 - x) for x in L["keepers"]])
            per[a]["bowlLeft"].append([max(0, 5 - x) for x in L["bowl"]])
            per[a]["indiansLeft"].append([max(0, 7 - x) for x in L["indians"]])
            per[a]["slotsLeft"].append([max(0, 11 - x) for x in L["squad"]])
            per[a]["stars"].append(L["stars"])
            i = ridx[(c, rec["learner"], rec["k"])]
            feats[a]["xi"].append(D["xi"][i])
            feats[a]["spendBy03"].append(L["spendShare"][2])
            feats[a]["spendBy05"].append(L["spendShare"][4])
            feats[a]["starsBy03"].append(L["stars"][2])
            feats[a]["priceToFair"].append(D["priceToFair"][i])
            feats[a]["marginalBuys"].append(D["marginalBuys"][i])
            feats[a]["reserveAt05"].append(1 - L["spendShare"][4])
            feats[a]["keeperProg"].append(D["keeperProg"][i])
            feats[a]["squadEnd"].append(L["squad"][-1])
        out["curves"][c] = {a: {q: [r4(x) for x in np.mean(np.array(v, float), 0)] for q, v in per[a].items()} for a in per}
        rel = {}
        for a in feats:
            xi = np.array(feats[a]["xi"])
            rel[a] = {q: corr(feats[a][q], xi) for q in feats[a] if q != "xi"}
            s = np.array(feats[a]["spendBy03"])
            bins = np.empty(len(s), int)
            bins[np.argsort(s, kind="mergesort")] = np.arange(len(s)) * 5 // len(s)  # rank-based quintiles (ties split by order)
            rel[a]["xiBySpendBy03Quintile"] = [{"quintile": int(q + 1), "spendBy03": r4(s[bins == q].mean()), "xi": r4(xi[bins == q].mean()), "n": int((bins == q).sum())} for q in range(5)]
        out["xiRelationships"][c] = rel
    return out


# ── I. requirement timing ───────────────────────────────────────────────
def analysis_I(D, T):
    out = {"fromRawRecords": {}, "fromTrajectories": {}}
    for a in ALGOS:
        out["fromRawRecords"][a] = {}
        for c in CONDS:
            m = sel(D, c, a)
            row = {}
            for q in ("keeper", "bowling", "indians"):
                p = D[f"{q}Prog"][m]
                need = D[f"{q}Need"][m] > 0
                p = p[need]
                closed = ~np.isnan(p)
                pc = p[closed]
                row[q] = {"episodesNeeding": int(need.sum()), "completed": r4(closed.mean()), "median": r4(np.median(pc)) if len(pc) else None,
                          "p25": r4(np.quantile(pc, 0.25)) if len(pc) else None, "p75": r4(np.quantile(pc, 0.75)) if len(pc) else None,
                          "mean": r4(pc.mean()) if len(pc) else None, "afterHalf": r4((pc > 0.5).mean()) if len(pc) else None,
                          "inReauction": r4(D[f"{q}Reauction"][m][need].mean()), "byForcedBid": r4(D[f"{q}Forced"][m][need].mean())}
            out["fromRawRecords"][a][c] = row
    edges = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    for c, recs in T.items():
        per = defaultdict(lambda: defaultdict(list))
        for rec in recs:
            a = rec["learner"].split(":")[0]
            L = next(t for t in rec["teams"] if t["type"] == "learner")
            first = lambda arr, n: next((edges[i] for i, x in enumerate(arr) if x >= n), None)
            per[a]["xiSlots11"].append(first(L["squad"], 11))
            per[a]["overseas4"].append(first(L["overseas"], 4))
            per[a]["overseasEnd"].append(L["overseas"][-1])
            per[a]["overseasCapped"].append(1.0 if L["overseas"][-1] >= 8 else 0.0)
        out["fromTrajectories"][c] = {}
        for a, v in per.items():
            r = {}
            for q in ("xiSlots11", "overseas4"):
                x = np.array([np.nan if y is None else y for y in v[q]], float)
                ok = ~np.isnan(x)
                r[q] = {"reached": r4(ok.mean()), "medianBin": r4(np.median(x[ok])) if ok.any() else None, "meanBin": r4(x[ok].mean()) if ok.any() else None}
            r["overseasEnd"] = r4(np.mean(v["overseasEnd"]))
            r["overseasCappedShare"] = r4(np.mean(v["overseasCapped"]))
            out["fromTrajectories"][c][a] = r
    out["note"] = ("Keeper / bowling / Indian timing = the learner's main-round progress (0–1; re-auction = 1) when the planner marked the requirement satisfied (raw records). "
                   "XI-slot and overseas timing come from the replays at 0.1 progress resolution (1.1 = in the re-auction); C1/C4 trajectories cover the s1×s1 subset.")
    return out


# ── J. strata ───────────────────────────────────────────────────────────
def analysis_J(D, MR):
    out = {"byAlgorithmAndCondition": {}, "matchups": {}, "concentration": {}}
    for a in ALGOS:
        out["byAlgorithmAndCondition"][a] = {}
        for c in CONDS:
            out["byAlgorithmAndCondition"][a][c] = {}
            for s in STRATA:
                m = sel(D, c, a, stratum=s)
                out["byAlgorithmAndCondition"][a][c][s] = {"xi": r4(D["xi"][m].mean()), "transfer": mci(D["dXI"][m], D["k"][m]) if c != "A" else None,
                                                           "strongXI": r4(D["strongXI"][m].mean()), "purseLeft": r4(D["purseLeftShare"][m].mean()),
                                                           "priceToFair": r4(np.nanmean(D["priceToFair"][m])), "forcedPerEpisode": r4(D["forced"][m].mean()),
                                                           "keeperProg": r4(np.nanmean(D["keeperProg"][m])), "squad": r4(D["squadSize"][m].mean()),
                                                           "stars": r4(D["stars"][m].mean()), "n": int(m.sum())}
    for c in ("C1", "C4"):
        out["matchups"][c] = {k: {s: r4(v["strata"][s]["transfer"]["mean"]) for s in STRATA} for k, v in MR[c].items()}
    B = D["cond"] != "A"
    for s in STRATA:
        m = B & (D["stratum"] == s)
        out["concentration"][s] = {"episodes": int(m.sum()), "forcedBids": int(D["forced"][m].sum()), "finalPath": int(D["finalPath"][m].sum()),
                                   "findings": int((D["findingLearner"][m] + D["findingOpp"][m]).sum()), "meanTransfer": r4(D["dXI"][m].mean()),
                                   "stageAXI": r4(D["xi"][(D["cond"] == "A") & (D["stratum"] == s)].mean())}
    return out


# ── K. seed stability ───────────────────────────────────────────────────
def analysis_K(D, MR):
    out = {"stageA": {}, "C1": {}, "C4": {}, "S4": {}}
    for a in ALGOS:
        seeds = [D["xi"][sel(D, "A", a) & (D["lseed"] == s)].mean() for s in (1, 2, 3)]
        out["stageA"][a] = {"bySeed": [r4(x) for x in seeds], "sd": r4(np.std(seeds, ddof=1))}
    for c in ("C1", "C4"):
        for a in ALGOS:
            cells = {}
            for b in ALGOS:
                if a == b:
                    continue
                sp = MR[c][f"{a}>{b}"]["seedPairs"]
                M = np.array([[sp[f"s{i}>s{j}"]["transfer"] for j in (1, 2, 3)] for i in (1, 2, 3)])
                grand = M.mean()
                ls = ((M.mean(1) - grand) ** 2).sum() * 3
                os_ = ((M.mean(0) - grand) ** 2).sum() * 3
                tot = ((M - grand) ** 2).sum()
                cells[b] = {"matrix": [[r4(x) for x in row] for row in M], "sd9": r4(M.std(ddof=1)), "range": [r4(M.min()), r4(M.max())],
                            "shareLearnerSeed": r4(ls / tot) if tot else None, "shareOpponentSeed": r4(os_ / tot) if tot else None,
                            "shareInteraction": r4(1 - (ls + os_) / tot) if tot else None, "allNegative": bool((M < 0).all())}
            out[c][a] = {"cells": cells, "meanSd9": r4(np.mean([v["sd9"] for v in cells.values()])), "maxSd9": r4(max(v["sd9"] for v in cells.values()))}
    for a in ALGOS:
        seeds = [D["dXI"][sel(D, "S4", a) & (D["lseed"] == s)].mean() for s in (1, 2, 3)]
        out["S4"][a] = {"transferBySeed": [r4(x) for x in seeds], "sd": r4(np.std(seeds, ddof=1)),
                        "xiBySeed": [r4(D["xi"][sel(D, "S4", a) & (D["lseed"] == s)].mean()) for s in (1, 2, 3)]}
    # opponent-seed effect: mean transfer caused by each opponent export (C1)
    opp_seed = {}
    for b in ALGOS:
        opp_seed[b] = {f"s{s}": r4(D["dXI"][(D["cond"] == "C1") & (D["oppKey"] == f"{b}:s{s}")].mean()) for s in (1, 2, 3)}
    out["C1_meanTransferCausedByOpponentExport"] = opp_seed
    return out


# ── L. failure modes ────────────────────────────────────────────────────
def analysis_L(D):
    B = np.where(D["cond"] != "A")[0]
    pa = D["pairA"][B]
    g = lambda m: D[m][B]
    ga = lambda m: D[m][pa]
    P = D["purse"][B]
    cats = {
        "aggressive opponent bidding / price inflation": (g("priceToFair") - ga("priceToFair")) >= 0.10,
        "star loss (scarcity at the top)": (g("stars") - ga("stars")) <= -2,
        "purse exhaustion with a requirement unmet": (g("minPurseAnyUnmet") / P < 0.05) & (ga("minPurseAnyUnmet") / P >= 0.05),
        "requirement delay (keeper ≥ 0.2 later or closed in the re-auction)": ((g("keeperProg") - ga("keeperProg")) >= 0.2) | ((g("keeperReauction") > 0) & (ga("keeperReauction") == 0)),
        "shield dependence (forced bid, none in the Stage-A pair)": (g("forced") > 0) & (ga("forced") == 0),
        "marginal purchasing (≥ 2 more marginal buys)": (g("marginalBuys") - ga("marginalBuys")) >= 2,
        "squad saturation (squad ≥ 24 with ≥ 5% purse unspent)": (g("squadSize") >= 24) & (g("purseLeftShare") >= 0.05),
        "overseas saturation (8 overseas, fewer in the Stage-A pair)": (g("overseas") >= 8) & (ga("overseas") < 8),
        "keeper competition (keeper by forced bid / final path / not completed)": (g("keeperForced") > 0) | (g("keeperFinalPath") > 0) | (g("legalXI") == 0),
        "re-auction effects (more re-auction buys or re-auction forced)": ((g("reauctionBuys") - ga("reauctionBuys")) >= 1) | (g("reauctionForced") > 0),
    }
    for k in cats:
        cats[k] = np.nan_to_num(cats[k].astype(float)).astype(bool)
    dx = D["dXI"][B]
    any_cat = np.zeros(len(B), bool)
    for v in cats.values():
        any_cat |= v
    cats["other (ΔXI ≤ −3 with none of the above)"] = (~any_cat) & (dx <= -3)
    out = {}
    cond, algo, opp, seed = D["cond"][B], D["algo"][B], D["oppAlgo"][B], D["seed"][B]
    for name, m in cats.items():
        by = {}
        for c in BCONDS:
            by[c] = {a: r4(m[(cond == c) & (algo == a)].mean()) for a in ALGOS}
        top = defaultdict(int)
        for c_, a_, o_ in zip(cond[m], algo[m], opp[m]):
            top[f"{c_} {a_}>{o_ or 'S4'}"] += 1
        worst = np.argsort(dx[m])[:3]
        ex = [{"cond": str(cond[m][i]), "learner": str(D["learner"][B][m][i]), "opponent": str(D["oppKey"][B][m][i]) or "S4", "seed": int(seed[m][i]), "deltaXI": r4(dx[m][i])} for i in worst]
        out[name] = {"episodes": int(m.sum()), "shareOfStageB": r4(m.mean()), "shareByConditionAndAlgorithm": by,
                     "meanDeltaXI_affected": r4(dx[m].mean()) if m.any() else None, "meanDeltaXI_unaffected": r4(dx[~m].mean()),
                     "topMatchups": dict(sorted(top.items(), key=lambda kv: -kv[1])[:6]), "examples": ex}
    # overlap: how many categories per episode
    ncat = sum(v.astype(int) for k, v in cats.items() if not k.startswith("other"))
    out["_overlap"] = {"episodesWithNoCategory": int((ncat == 0).sum()), "meanCategoriesPerEpisode": r4(ncat.mean()),
                       "deltaXIByNumberOfCategories": {int(n): r4(dx[ncat == n].mean()) for n in range(0, int(ncat.max()) + 1) if (ncat == n).any()},
                       "episodesByNumberOfCategories": {int(n): int((ncat == n).sum()) for n in range(0, int(ncat.max()) + 1)}}
    out["_definitions"] = "All categories compare the Stage-B episode with the same export's Stage-A episode on the same auction (paired). Categories overlap; associations only."
    return out


def main(rep):
    rep = Path(rep)
    D = load()
    for q in ("keeper", "bowling", "indians"):
        D[f"{q}Prog"] = D[f"{q}Prog"]
    T = load_traj()
    MR = json.loads((ROOT / "ml/reports/phase2e0/matchup-results.json").read_text(encoding="utf-8"))
    res = {}
    res["transfer-analysis"] = analysis_A(D)
    res["opponent-effects"] = analysis_B(D)
    res["c1-c4-analysis"] = analysis_C(D, T, MR)
    res["s4-analysis"] = analysis_D(D, T)
    res["behavior-shifts"] = analysis_E(D, T)
    res["keeper-failures"] = analysis_F()
    res["shield-analysis"] = analysis_G(D)
    res["purse-analysis"] = analysis_H(D, T)
    res["requirement-analysis"] = analysis_I(D, T)
    res["stratum-analysis"] = analysis_J(D, MR)
    res["seed-stability"] = analysis_K(D, MR)
    res["failure-modes"] = analysis_L(D)
    for name, obj in res.items():
        (rep / f"{name}.json").write_text(json.dumps(obj, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, np.integer) else str(o)), encoding="utf-8")
    print("wrote", ", ".join(res))


if __name__ == "__main__":
    main(sys.argv[1])
