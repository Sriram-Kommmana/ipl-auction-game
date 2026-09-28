"""Phase 2E.0 analysis: Stage-B frozen cross-play (evaluation only).

Reads the harness outputs (one JSON line per episode) and writes the report
data. No ranking and no winner are computed anywhere.

    python analyse.py <run_dir> <report_dir>
      run_dir: A.jsonl, C1.jsonl, C4.jsonl, S4.jsonl (+ *.meta.json)

Statistics
  transfer Δ   (primary)   learner XI in the Stage-B room − the SAME export's XI in
                           the Stage-A control on the SAME manifest entry.
  head-to-head (secondary) learner XI − opponent XI in the same auction
                           (C4: mean of the four opponent seats; S4: each algorithm's seat).
  95% CI       percentile bootstrap (2,000 resamples, fixed seed) over the 500
               auction entries: an entry's value is the mean over the seed
               pairings in the cell, so every resample keeps the pairing by auction.
  W/T/L        per episode (Δ > 0 / = 0 / < 0).
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ALGOS = ["ppo", "a2c", "d3qn", "qrdqn", "es"]
NAMES = {"ppo": "PPO", "a2c": "A2C", "d3qn": "D3QN", "qrdqn": "QR-DQN", "es": "OpenAI-ES"}
STRATA = ["low", "normal", "high"]
B = 2000

LEARNER = [
    "xi", "legalXI", "strongXI", "rank", "purseLeft", "purseLeftShare", "purseSpent", "squadSize", "overseas", "stars", "buys",
    "marginalBuys", "priceToFair", "capToFair", "bidRate", "xiGainPer1000", "reauctionBuys", "decisions", "shieldActivations", "shieldWins",
    "contests",
]
AUDIT = ["forced", "forcedWon", "forcedLost", "finalPath", "finalPathWon", "reauctionForced", "forcedKeeperBids", "forcedKeeperWon"]
REQS = ["keeper", "bowling", "indians"]


def f(x):
    return float("nan") if x is None else float(x)


def extract(path):
    """One flat row of numbers per episode (streamed; the raw files are large)."""
    cols = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            s, a = r["summary"], r["audit"]
            cols["job"].append(r["job"])
            cols["k"].append(r["k"])
            cols["seed"].append(r["seed"])
            cols["stratum"].append(r["stratum"])
            cols["learner"].append(r["learner"])
            cols["oppKey"].append(r["opponents"][0] if r["opponents"] and len(set(r["opponents"])) == 1 else "|".join(r["opponents"]))
            for m in LEARNER:
                cols[m].append(f(s.get(m)))
            for m in AUDIT:
                cols[m].append(f(a.get(m)))
            cols["shieldForcedTotal"].append(f((s.get("shield") or {}).get("forced")))
            cols["invariantViolations"].append(s["invariantViolations"])
            for q in REQS:
                c = r["req"][q]["closed"]
                need = r["req"][q]["initialNeed"]
                cols[f"{q}InitialNeed"].append(f(need))
                cols[f"{q}Closed"].append(1.0 if c else 0.0)
                cols[f"{q}ClosedForced"].append(1.0 if c and c["forced"] else 0.0)
                cols[f"{q}ClosedFinalPath"].append(1.0 if c and c["finalPath"] else 0.0)
                cols[f"{q}ClosedProgress"].append(f(c["progress"]) if c else float("nan"))
                cols[f"{q}ClosedReauction"].append(1.0 if c and c["phase"] == "reauction" else 0.0)
                cols[f"{q}ClosedPrice"].append(f(c["price"]) if c else float("nan"))
            cols["minPurseAnyUnmet"].append(f(a.get("minPurseAnyUnmet")))
            opp = r["opp"]
            cols["nOpp"].append(len(opp))
            cols["oppXiMean"].append(float(np.mean([o["xi"] for o in opp])) if opp else float("nan"))
            cols["oppXiBest"].append(max(o["xi"] for o in opp) if opp else float("nan"))
            cols["oppLegalAll"].append(float(all(o["legalXI"] for o in opp)) if opp else float("nan"))
            cols["oppFallbacks"].append(sum(len(o["fallbacks"]) for o in opp))
            cols["oppDecisions"].append(sum(o["decisions"] for o in opp))
            for m in ("rank", "bidRate", "capToFair", "priceToFair", "purseLeftShare", "squadSize", "stars", "overseas", "buys", "reauctionBuys", "strongXI"):
                vals = [o[m] for o in opp if o.get(m) is not None]
                cols[f"opp_{m}"].append(float(np.mean(vals)) if vals else float("nan"))
            for algo in ALGOS:  # S4: per-algorithm opponent seat
                o = [x for x in opp if x["algo"] == algo]
                cols[f"s4xi_{algo}"].append(o[0]["xi"] if len(o) == 1 else float("nan"))
                cols[f"s4rank_{algo}"].append(o[0]["rank"] if len(o) == 1 else float("nan"))
            inc = r["incompleteByType"]
            for t in ("learner", "rlSnapshot", "rule", "rlFallback", "human"):
                cols[f"incomplete_{t}"].append(inc.get(t, 0))
            cols["rlSnapshotSeats"].append(sum(1 for o in r["opponentSeats"]))
            fs = r.get("findings") or []
            cols["findingsLearner"].append(sum(1 for x in fs if x["type"] == "learner"))
            cols["findingsOpponent"].append(sum(1 for x in fs if x["type"] == "rlSnapshot"))
    out = {k: np.asarray(v) for k, v in cols.items()}
    order = np.argsort(out["job"])
    return {k: v[order] for k, v in out.items()}


def boot_ci(per_entry, seed=7):
    x = np.asarray(per_entry, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    m = x[idx].mean(1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def paired(diff, k):
    """diff per episode, k = manifest entry index; CI clustered by entry."""
    diff, k = np.asarray(diff, float), np.asarray(k)
    ok = ~np.isnan(diff)
    diff, k = diff[ok], k[ok]
    ent = np.array([diff[k == e].mean() for e in np.unique(k)])
    return {
        "mean": float(diff.mean()), "sd": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0, "ci95": boot_ci(ent), "n": int(len(diff)), "entries": int(len(ent)),
        "wtl": [int((diff > 0).sum()), int((diff == 0).sum()), int((diff < 0).sum())],
    }


def mean_ci(x, k):
    x, k = np.asarray(x, float), np.asarray(k)
    ok = ~np.isnan(x)
    if not ok.any():
        return {"mean": None, "ci95": [None, None], "n": 0}
    ent = np.array([x[ok & (k == e)].mean() for e in np.unique(k[ok])])
    return {"mean": float(x[ok].mean()), "sd": float(x[ok].std(ddof=1)) if ok.sum() > 1 else 0.0, "ci95": boot_ci(ent), "n": int(ok.sum())}


BEHAVIOUR = ["xi", "legalXI", "strongXI", "rank", "bidRate", "priceToFair", "capToFair", "purseLeftShare", "purseSpent", "squadSize", "overseas", "stars",
             "buys", "marginalBuys", "reauctionBuys", "xiGainPer1000", "decisions", "contests"]
SHIELD = ["shieldActivations", "shieldWins", "forced", "forcedWon", "forcedLost", "finalPath", "finalPathWon", "reauctionForced", "forcedKeeperBids", "forcedKeeperWon"]
REQ_COLS = [f"{q}{c}" for q in REQS for c in ("Closed", "ClosedForced", "ClosedFinalPath", "ClosedReauction", "ClosedProgress", "ClosedPrice")] + ["minPurseAnyUnmet"]


def profile(d, sel):
    out = {}
    for m in BEHAVIOUR + SHIELD + REQ_COLS:
        v = d[m][sel].astype(float)
        v = v[~np.isnan(v)]
        out[m] = float(v.mean()) if len(v) else None
    out["episodes"] = int(sel.sum())
    # totals that the Stage-A reports quoted per 500 episodes
    out["forcedKeeperPurchasesPer500"] = float(d["keeperClosedForced"][sel].sum() * 500 / max(1, sel.sum()))
    out["finalPathForcedBidsTotal"] = int(np.nansum(d["finalPath"][sel]))
    out["shieldForcedTotal"] = int(np.nansum(d["shieldForcedTotal"][sel]))
    return out


def main(run_dir, report_dir):
    run_dir, report_dir = Path(run_dir), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    D = {c: extract(run_dir / f"{c}.jsonl") for c in ("A", "C1", "C4", "S4") if (run_dir / f"{c}.jsonl").exists()}
    meta = {c: json.loads((run_dir / f"{c}.jsonl.meta.json").read_text()) for c in D if (run_dir / f"{c}.jsonl.meta.json").exists()}
    A = D["A"]
    a_xi = {(l, k): x for l, k, x in zip(A["learner"], A["k"], A["xi"])}
    a_row = {(l, k): i for i, (l, k) in enumerate(zip(A["learner"], A["k"]))}

    def transfer(d, sel):
        return np.array([x - a_xi[(l, k)] for l, k, x in zip(d["learner"][sel], d["k"][sel], d["xi"][sel])])

    # ── Stage-A control ─────────────────────────────────────────────────
    stage_a = {}
    for algo in ALGOS:
        seeds = [A["xi"][A["learner"] == f"{algo}:s{s}"].mean() for s in (1, 2, 3)]
        sel = np.char.startswith(A["learner"].astype(str), f"{algo}:")
        stage_a[algo] = {"xiBySeed": seeds, "xiMean": float(np.mean(seeds)), "xiSeedStd": float(np.std(seeds, ddof=1)), "profile": profile(A, sel),
                         "bySeed": {f"s{s}": profile(A, A["learner"] == f"{algo}:s{s}") for s in (1, 2, 3)}}

    # ── C1 / C4 matchup matrices ────────────────────────────────────────
    matrix, results = {}, {}
    for comp in ("C1", "C4"):
        if comp not in D:
            continue
        d = D[comp]
        la = np.array([x.split(":")[0] for x in d["learner"]])
        ob = np.array([x.split(":")[0] for x in d["oppKey"]])
        cells, res = {}, {}
        for a in ALGOS:
            for b in ALGOS:
                if a == b:
                    continue
                sel = (la == a) & (ob == b)
                tr = transfer(d, sel)
                h2h = d["xi"][sel] - d["oppXiMean"][sel]
                h2h_best = d["xi"][sel] - d["oppXiBest"][sel]
                seedpairs = {}
                for sa in (1, 2, 3):
                    for sb in (1, 2, 3):
                        s2 = sel & (d["learner"] == f"{a}:s{sa}") & (d["oppKey"] == f"{b}:s{sb}")
                        t2 = transfer(d, s2)
                        seedpairs[f"s{sa}>s{sb}"] = {"xi": float(d["xi"][s2].mean()), "transfer": float(t2.mean()), "h2h": float((d["xi"][s2] - d["oppXiMean"][s2]).mean()),
                                                     "transferCI": paired(t2, d["k"][s2])["ci95"], "legal": int(d["legalXI"][s2].sum()), "n": int(s2.sum())}
                sp_t = np.array([v["transfer"] for v in seedpairs.values()])
                by_ls = [np.mean([seedpairs[f"s{sa}>s{sb}"]["transfer"] for sb in (1, 2, 3)]) for sa in (1, 2, 3)]
                by_os = [np.mean([seedpairs[f"s{sa}>s{sb}"]["transfer"] for sa in (1, 2, 3)]) for sb in (1, 2, 3)]
                strata = {}
                for st in STRATA:
                    s3 = sel & (d["stratum"] == st)
                    strata[st] = {"transfer": paired(transfer(d, s3), d["k"][s3]), "h2h": paired(d["xi"][s3] - d["oppXiMean"][s3], d["k"][s3]), "xi": float(d["xi"][s3].mean())}
                t = paired(tr, d["k"][sel])
                h = paired(h2h, d["k"][sel])
                cells[f"{a}>{b}"] = {"transfer": {"mean": t["mean"], "ci95": t["ci95"]}, "h2h": {"mean": h["mean"], "ci95": h["ci95"]}}
                res[f"{a}>{b}"] = {
                    "learner": a, "opponent": b, "composition": comp, "episodes": int(sel.sum()),
                    "learnerXI": mean_ci(d["xi"][sel], d["k"][sel]), "opponentXI": mean_ci(d["oppXiMean"][sel], d["k"][sel]),
                    "transfer": t, "h2h": h, "h2hVsBestOpponentSeat": paired(h2h_best, d["k"][sel]) if comp == "C4" else None,
                    "seedPairs": seedpairs,
                    "seedStability": {"transferRange": [float(sp_t.min()), float(sp_t.max())], "transferSdAcrossSeedPairs": float(sp_t.std(ddof=1)),
                                      "signAgreement": f"{int((sp_t < 0).sum())} negative / {int((sp_t > 0).sum())} positive of 9",
                                      "byLearnerSeed": [float(x) for x in by_ls], "byOpponentSeed": [float(x) for x in by_os],
                                      "seedPairsWithCIExcludingZero": int(sum(1 for v in seedpairs.values() if v["transferCI"][0] > 0 or v["transferCI"][1] < 0))},
                    "strata": strata,
                    "legalXI": int(d["legalXI"][sel].sum()), "strongXI": int(d["strongXI"][sel].sum()),
                    "opponentLegalAll": int(np.nansum(d["oppLegalAll"][sel])),
                    "learnerProfile": profile(d, sel),
                    "opponentProfile": {m: float(np.nanmean(d[f"opp_{m}"][sel])) for m in ("rank", "bidRate", "capToFair", "priceToFair", "purseLeftShare", "squadSize", "stars", "overseas", "buys", "reauctionBuys", "strongXI")},
                }
        matrix[comp] = cells
        results[comp] = res

    # ── S4 mixed rooms ──────────────────────────────────────────────────
    s4 = {}
    if "S4" in D:
        d = D["S4"]
        la = np.array([x.split(":")[0] for x in d["learner"]])
        for a in ALGOS:
            sel = la == a
            tr = transfer(d, sel)
            per = {}
            for b in ALGOS:
                if b == a:
                    continue
                per[b] = {"h2h": paired(d["xi"][sel] - d[f"s4xi_{b}"][sel], d["k"][sel]), "opponentXI": float(np.nanmean(d[f"s4xi_{b}"][sel])), "opponentRank": float(np.nanmean(d[f"s4rank_{b}"][sel]))}
            s4[a] = {
                "episodes": int(sel.sum()), "learnerXI": mean_ci(d["xi"][sel], d["k"][sel]), "transfer": paired(tr, d["k"][sel]),
                "bySeed": {f"s{s}": {"xi": float(d["xi"][sel & (d["learner"] == f"{a}:s{s}")].mean()), "transfer": float(transfer(d, sel & (d["learner"] == f"{a}:s{s}")).mean())} for s in (1, 2, 3)},
                "strata": {st: paired(transfer(d, sel & (d["stratum"] == st)), d["k"][sel & (d["stratum"] == st)]) for st in STRATA},
                "vsEachAlgorithm": per, "rank": float(d["rank"][sel].mean()), "legalXI": int(d["legalXI"][sel].sum()), "profile": profile(d, sel),
            }

    # ── behavioural comparison Stage A vs B ─────────────────────────────
    behaviour = {}
    for algo in ALGOS:
        row = {"A": stage_a[algo]["profile"]}
        for comp in ("C1", "C4", "S4"):
            if comp in D:
                sel = np.char.startswith(D[comp]["learner"].astype(str), f"{algo}:")
                row[comp] = profile(D[comp], sel)
        behaviour[algo] = row

    # ── safety ──────────────────────────────────────────────────────────
    safety = {"episodes": {c: int(len(d["job"])) for c, d in D.items()}}
    for c, d in D.items():
        safety[c] = {
            "invariantViolations": int(d["invariantViolations"].sum()),
            "learnerIncompleteXI": int(d["incomplete_learner"].sum()),
            "rlOpponentIncompleteXI": int(d["incomplete_rlSnapshot"].sum()),
            "ruleBotIncompleteXI": int(d["incomplete_rule"].sum()), "ruleFallbackIncompleteXI": int(d["incomplete_rlFallback"].sum()), "humanProxyIncompleteXI": int(d["incomplete_human"].sum()),
            "opponentFallbacks": int(d["oppFallbacks"].sum()),
            "opponentRlDecisions": int(d["oppDecisions"].sum()), "learnerDecisions": int(d["decisions"].sum()),
            "learnerLegalXI": f"{int(d['legalXI'].sum())}/{len(d['xi'])}",
            "optionAFindings": {"learnerIncompleteXI": int(d["findingsLearner"].sum()), "rlOpponentIncompleteXI": int(d["findingsOpponent"].sum()),
                                "note": "incomplete XI of an RL-controlled team from legal policy behaviour, defect screen passed (approved Option A); see findings.json"},
        }
    # ── seed stability summary, exploitability ──────────────────────────
    worst = {}
    for comp, res in results.items():
        worst[comp] = {a: min(((b, res[f"{a}>{b}"]["transfer"]["mean"]) for b in ALGOS if b != a), key=lambda x: x[1]) for a in ALGOS}

    json.dump({"note": "Ordered learner>opponent cells. transfer = learner Stage-B XI − same export's Stage-A XI on the same entry; h2h = learner XI − opponent XI (C4: mean of the four opponent seats). No ranking.",
               "stageA": {a: {"xiMean": v["xiMean"], "xiSeedStd": v["xiSeedStd"], "xiBySeed": v["xiBySeed"]} for a, v in stage_a.items()},
               **matrix, "S4": {a: {"transfer": {"mean": v["transfer"]["mean"], "ci95": v["transfer"]["ci95"]}, "vsEachAlgorithm": {b: {"mean": w["h2h"]["mean"], "ci95": w["h2h"]["ci95"]} for b, w in v["vsEachAlgorithm"].items()}} for a, v in s4.items()}},
              open(report_dir / "matchup-matrix.json", "w"), indent=1)
    json.dump({"stageA": stage_a, **results, "S4": s4, "behaviour": behaviour, "worstOpponentByTransfer": worst, "meta": meta}, open(report_dir / "matchup-results.json", "w"), indent=1)
    json.dump(safety, open(report_dir / "safety-episodes.json", "w"), indent=1)
    # compact per-episode table (full raw records stay in the run directory)
    cols = ["job", "k", "seed", "stratum", "learner", "oppKey", "xi", "legalXI", "strongXI", "rank", "purseLeft", "squadSize", "overseas", "stars", "buys", "bidRate",
            "priceToFair", "capToFair", "reauctionBuys", "marginalBuys", "forced", "finalPath", "forcedKeeperBids", "keeperClosedForced", "oppXiMean", "oppXiBest", "oppLegalAll"]
    ep = {"columns": cols, "note": "one row per episode; raw per-episode records (all fields, digests, opponent seats) in ml/runs/_2e0/full/*.jsonl", "conditions": {}}
    for c, d in D.items():
        rows = []
        for i in range(len(d["job"])):
            row = []
            for col in cols:
                v = d[col][i]
                if isinstance(v, (np.floating, float)):
                    row.append(None if math.isnan(v) else round(float(v), 6))
                elif isinstance(v, np.integer):
                    row.append(int(v))
                else:
                    row.append(str(v))
            rows.append(row)
        ep["conditions"][c] = rows
    json.dump(ep, open(report_dir / "episode-results.json", "w"), separators=(",", ":"))
    return stage_a, results, s4, behaviour, safety, worst


if __name__ == "__main__":
    stage_a, results, s4, behaviour, safety, worst = main(sys.argv[1], sys.argv[2])
    print(json.dumps(safety, indent=1))
    for a, v in stage_a.items():
        print(f"Stage A {a:6s} {v['xiMean']:.3f} ± {v['xiSeedStd']:.3f}")
    for comp, res in results.items():
        print(f"\n{comp} transfer Δ (rows learner, cols opponent)")
        print("        " + "".join(f"{b:>18s}" for b in ALGOS))
        for a in ALGOS:
            cells = []
            for b in ALGOS:
                if a == b:
                    cells.append(f"{'—':>18s}")
                else:
                    t = res[f"{a}>{b}"]["transfer"]
                    cells.append(f"{t['mean']:+7.3f} [{t['ci95'][0]:+.2f},{t['ci95'][1]:+.2f}]".rjust(18))
            print(f"{a:8s}" + "".join(cells))
        print(f"{comp} head-to-head Δ")
        for a in ALGOS:
            print(f"{a:8s}" + "".join((f"{'—':>18s}" if a == b else f"{res[f'{a}>{b}']['h2h']['mean']:+7.3f} [{res[f'{a}>{b}']['h2h']['ci95'][0]:+.2f},{res[f'{a}>{b}']['h2h']['ci95'][1]:+.2f}]".rjust(18)) for b in ALGOS))
    for a, v in s4.items():
        print(f"S4 {a:6s} XI {v['learnerXI']['mean']:.3f} transfer {v['transfer']['mean']:+.3f} {v['transfer']['ci95']} rank {v['rank']:.2f} " + " ".join(f"vs {b} {w['h2h']['mean']:+.2f}" for b, w in v["vsEachAlgorithm"].items()))
