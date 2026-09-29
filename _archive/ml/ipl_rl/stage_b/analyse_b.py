"""Phase 2F — Stage-B pilot analysis (evaluation data only; nothing trained here).

Pairs every Stage-B pilot checkpoint episode with the frozen Stage-A export of
the same algorithm and Stage-A seed index on the SAME validation entry, the same
condition and the same opponent export (Phase 2E.0 records), and computes the
Phase 2F §13/§14 metrics. Statistics: per-algorithm mean ± sd across the three
seeds, n, and a bootstrap 95% CI over entries (2,000 resamples, seed 7; the
per-entry value is the mean over seeds and opponents) — the Phase 2E.0 method.

    python -m ipl_rl.stage_b.analyse_b   → ml/reports/phase2f/*.json (+ checkpoint reports)
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ML_ROOT = Path(__file__).resolve().parents[2]
EV = ML_ROOT / "runs/stage_b/eval"
OLD = ML_ROOT / "runs/_2e0/full"
REP = ML_ROOT / "reports/phase2f"
ALGOS = ["ppo", "a2c", "d3qn", "qrdqn", "es"]
NAMES = {"ppo": "PPO", "a2c": "A2C", "d3qn": "D3QN", "qrdqn": "QR-DQN", "es": "OpenAI-ES"}
SEEDS_B = [101, 102, 103]
LEVELS = {"c100": (98_304, 100), "c250": (245_760, 100), "c500": (497_664, 500), "c500_first100": (497_664, 100)}  # last = c500 on entries 0–99 (trend comparability)
CONDS = ["A", "C1", "C4", "S4"]
REF_TRANSFER = {"C1": {"ppo": -1.95, "a2c": -2.79, "d3qn": -2.60, "qrdqn": -2.46, "es": -3.37},
                "C4": {"ppo": -2.32, "a2c": -3.72, "d3qn": -3.29, "qrdqn": -3.23, "es": -5.06},
                "S4": {"ppo": -2.43, "a2c": -5.14, "d3qn": -4.82, "qrdqn": -4.34, "es": -6.83}}
ANCHORS = {"ppo": (92.595, 0.082), "qrdqn": (91.853, 0.089), "d3qn": (91.490, 0.191), "es": (91.068, 0.044), "a2c": (90.427, 0.046)}
R = lambda x, n=4: None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), n)


def boot_ci(per_entry, reps=2000, seed=7):
    x = np.asarray(per_entry, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return None
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(reps, len(x)))].mean(1)
    return [R(np.percentile(m, 2.5)), R(np.percentile(m, 97.5))]


def rec_row(r, stage):
    s, a, q = r["summary"], r["audit"], r["req"]
    learner = r["learner"]
    algo = learner.split(":")[0]
    seed = int(learner.split(":s")[1].split("@")[0])
    finds = [f for f in (r.get("findings") or []) if f["type"] == "learner"]
    fb = a.get("forcedBy") or {}
    return {
        "stage": stage, "algo": algo, "seedIdx": seed - 100 if seed > 100 else seed, "learner": learner, "cond": r["cond"], "k": r["k"],
        "opp": r["opponents"][0] if r["cond"] in ("C1", "C4") else "", "stratum": r.get("stratum"),
        "xi": s["xi"], "legalXI": 1.0 if s["legalXI"] else 0.0, "purseLeftShare": s["purseLeftShare"], "squadSize": s["squadSize"], "overseas": s["overseas"],
        "stars": s["stars"], "marginalBuys": s["marginalBuys"], "reauctionBuys": s["reauctionBuys"], "buys": s["buys"],
        "priceToFair": s["priceToFair"] if s["priceToFair"] is not None else np.nan, "bidRate": s["bidRate"],
        "capToFair": s["capToFair"] if s["capToFair"] is not None else np.nan, "decisions": s["decisions"],
        "shieldActivations": s["shieldActivations"], "shieldRate": s["shieldActivations"] / max(1, s["decisions"]),
        "forced": a["forced"], "forcedLost": a["forcedLost"], "finalPath": a["finalPath"], "finalPathWon": a["finalPathWon"],
        "reauctionForced": a["reauctionForced"], "forcedKeeperBids": a["forcedKeeperBids"], "forcedKeeperWon": a["forcedKeeperWon"],
        "forcedPurseKeeper": fb.get("keeper(purse)", 0),
        "keeperClosedProgress": (q["keeper"]["closed"] or {}).get("progress", np.nan) if q["keeper"]["initialNeed"] else np.nan,
        "keeperClosedForced": 1.0 if (q["keeper"]["closed"] or {}).get("forced") else 0.0,
        "anyReqClosedForced": 1.0 if any((q[x]["closed"] or {}).get("forced") for x in q) else 0.0,
        "anyReqReauction": 1.0 if any((q[x]["closed"] or {}).get("phase") == "reauction" for x in q) else 0.0,
        "learnerFinding": len(finds), "learnerFindingM2": sum(1 for f in finds if fb.get("keeper(purse)", 0) > 0),
        "opponentFinding": sum(1 for f in (r.get("findings") or []) if f["type"] == "rlSnapshot"),
        "invariantViolations": s.get("invariantViolations", 0),
    }


def load_level(lv):
    limit = LEVELS[lv][1]
    src = "c500" if lv.startswith("c500") else lv
    rows = []
    for c in CONDS:
        p = EV / src / f"{c}.jsonl"
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            if r["k"] < limit:
                rows.append(rec_row(r, "B"))
    return rows, limit


_OLD = {}


def load_old(limit):
    if limit in _OLD:
        return _OLD[limit]
    rows = []
    for c in CONDS:
        for line in (OLD / f"{c}.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            if r["k"] < limit:
                rows.append(rec_row(r, "A"))
    _OLD[limit] = rows
    return rows


def keyed(rows):
    return {(r["algo"], r["seedIdx"], r["cond"], r["opp"], r["k"]): r for r in rows}


def transfer_table(rows_b, rows_a):
    """transfer Δ = cond XI − the same learner's A XI on the same entry (Stage B and Stage A), and the paired difference."""
    B, A = keyed(rows_b), keyed(rows_a)
    ctrlB = {(r["algo"], r["seedIdx"], r["k"]): r["xi"] for r in rows_b if r["cond"] == "A"}
    ctrlA = {(r["algo"], r["seedIdx"], r["k"]): r["xi"] for r in rows_a if r["cond"] == "A"}
    out = {}
    for c in CONDS:
        per_algo = {}
        for a in ALGOS:
            cells = [(key, rb) for key, rb in B.items() if key[0] == a and key[2] == c]
            if not cells:
                continue
            tb, ta, xb, xa, dxi = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
            ent_imp = defaultdict(list)
            ent_tb = defaultdict(list)
            ent_d = defaultdict(list)
            for key, rb in cells:
                ra = A.get(key)
                if ra is None:
                    raise SystemExit(f"no Stage-A pair for {key}")
                s, k = key[1], key[4]
                t_b = rb["xi"] - ctrlB[(a, s, k)]
                t_a = ra["xi"] - ctrlA[(a, s, k)]
                tb[s].append(t_b); ta[s].append(t_a); xb[s].append(rb["xi"]); xa[s].append(ra["xi"]); dxi[s].append(rb["xi"] - ra["xi"])
                ent_imp[k].append(t_b - t_a)
                ent_d[k].append(rb["xi"] - ra["xi"])
                ent_tb[k].append(t_b)
            seed_tb = [np.mean(tb[s]) for s in sorted(tb)]
            seed_ta = [np.mean(ta[s]) for s in sorted(ta)]
            seed_xb = [np.mean(xb[s]) for s in sorted(xb)]
            seed_xa = [np.mean(xa[s]) for s in sorted(xa)]
            seed_d = [np.mean(dxi[s]) for s in sorted(dxi)]
            per_algo[a] = {
                "n": len(cells), "seeds": len(tb),
                "xiStageB": {"mean": R(np.mean(seed_xb)), "sd": R(np.std(seed_xb, ddof=1)) if len(seed_xb) > 1 else None},
                "xiStageA": {"mean": R(np.mean(seed_xa)), "sd": R(np.std(seed_xa, ddof=1)) if len(seed_xa) > 1 else None},
                "xiDiffBminusA": {"mean": R(np.mean(seed_d)), "sd": R(np.std(seed_d, ddof=1)) if len(seed_d) > 1 else None,
                                  "ci95": boot_ci([np.mean(v) for v in ent_d.values()])},
                **({"transferStageB": {"mean": R(np.mean(seed_tb)), "sd": R(np.std(seed_tb, ddof=1)) if len(seed_tb) > 1 else None, "ci95": boot_ci([np.mean(v) for v in ent_tb.values()])},
                    "transferStageA": {"mean": R(np.mean(seed_ta)), "sd": R(np.std(seed_ta, ddof=1)) if len(seed_ta) > 1 else None},
                    "transferImprovement": {"mean": R(np.mean(seed_tb) - np.mean(seed_ta)), "sd": R(np.std(np.array(seed_tb) - np.array(seed_ta), ddof=1)) if len(seed_tb) > 1 else None,
                                            "ci95": boot_ci([np.mean(v) for v in ent_imp.values()])},
                    "historicalReference": REF_TRANSFER[c][a]} if c != "A" else {}),
            }
        out[c] = per_algo
    return out


def by_opponent(rows_b, rows_a, cond):
    B, A = keyed(rows_b), keyed(rows_a)
    ctrlB = {(r["algo"], r["seedIdx"], r["k"]): r["xi"] for r in rows_b if r["cond"] == "A"}
    ctrlA = {(r["algo"], r["seedIdx"], r["k"]): r["xi"] for r in rows_a if r["cond"] == "A"}
    cell = defaultdict(lambda: {"b": [], "a": [], "ent": defaultdict(list)})
    for key, rb in B.items():
        if key[2] != cond:
            continue
        a, s, _, opp, k = key
        ob = opp.split(":")[0]
        tb = rb["xi"] - ctrlB[(a, s, k)]
        ta = A[key]["xi"] - ctrlA[(a, s, k)]
        cell[(a, ob)]["b"].append(tb)
        cell[(a, ob)]["a"].append(ta)
        cell[(a, ob)]["ent"][k].append(tb - ta)
    return {f"{a}|{o}": {"transferStageB": R(np.mean(v["b"])), "transferStageA": R(np.mean(v["a"])), "improvement": R(np.mean(v["b"]) - np.mean(v["a"])),
                         "improvementCi95": boot_ci([np.mean(x) for x in v["ent"].values()]), "n": len(v["b"])}
            for (a, o), v in sorted(cell.items())}


METRICS = ["xi", "legalXI", "purseLeftShare", "squadSize", "overseas", "stars", "marginalBuys", "reauctionBuys", "buys", "priceToFair", "bidRate", "capToFair",
           "shieldRate", "shieldActivations", "forced", "forcedLost", "finalPath", "reauctionForced", "forcedKeeperBids", "forcedKeeperWon", "forcedPurseKeeper",
           "keeperClosedProgress", "keeperClosedForced", "anyReqClosedForced", "anyReqReauction", "learnerFinding", "learnerFindingM2", "opponentFinding", "invariantViolations"]


def behaviour(rows_b, rows_a):
    out = {}
    for c in CONDS:
        out[c] = {}
        for a in ALGOS:
            rb = [r for r in rows_b if r["algo"] == a and r["cond"] == c]
            ra = [r for r in rows_a if r["algo"] == a and r["cond"] == c]
            if not rb:
                continue
            d = {}
            for m in METRICS:
                vb = np.array([r[m] for r in rb], float)
                va = np.array([r[m] for r in ra], float)
                seeds = sorted({r["seedIdx"] for r in rb})
                sm = [np.nanmean([r[m] for r in rb if r["seedIdx"] == s]) for s in seeds]
                d[m] = {"stageB": R(np.nanmean(vb)), "stageBsd": R(np.std(sm, ddof=1)) if len(sm) > 1 else None, "stageA": R(np.nanmean(va)),
                        "total_stageB": R(np.nansum(vb), 1) if m in ("finalPath", "forcedKeeperBids", "learnerFinding", "learnerFindingM2", "opponentFinding", "invariantViolations", "forced") else None,
                        "total_stageA": R(np.nansum(va), 1) if m in ("finalPath", "forcedKeeperBids", "learnerFinding", "learnerFindingM2", "opponentFinding", "invariantViolations", "forced") else None}
            d["episodes"] = {"stageB": len(rb), "stageA": len(ra)}
            out[c][a] = d
    return out


def robustness(transfer, byopp):
    """Sensitivity to the opponent algorithm: sd over opponent algorithms of the per-opponent transfer (C1, C4)."""
    out = {}
    for c in ("C1", "C4"):
        out[c] = {}
        for a in ALGOS:
            cells = {k.split("|")[1]: v for k, v in byopp[c].items() if k.split("|")[0] == a}
            if not cells:
                continue
            b = [v["transferStageB"] for v in cells.values()]
            s = [v["transferStageA"] for v in cells.values()]
            out[c][a] = {"sdAcrossOpponentsStageB": R(np.std(b)), "sdAcrossOpponentsStageA": R(np.std(s)), "rangeStageB": R(max(b) - min(b)), "rangeStageA": R(max(s) - min(s)),
                         "worstOpponentStageB": min(cells, key=lambda o: cells[o]["transferStageB"]), "worstOpponentStageA": min(cells, key=lambda o: cells[o]["transferStageA"])}
    esc = {}
    for a in ALGOS:
        if a in transfer["C1"] and a in transfer["C4"]:
            b = transfer["C4"][a]["transferStageB"]["mean"] - transfer["C1"][a]["transferStageB"]["mean"]
            s = transfer["C4"][a]["transferStageA"]["mean"] - transfer["C1"][a]["transferStageA"]["mean"]
            esc[a] = {"c1ToC4StageB": R(b), "c1ToC4StageA": R(s), "reduction": R(b - s)}
    out["c1ToC4Escalation"] = esc
    return out


def per_run_rows(rows_b, lv):
    """one checkpoint report per (algorithm, seed) at this level"""
    out = {}
    ctrl = {(r["algo"], r["seedIdx"], r["k"]): r["xi"] for r in rows_b if r["cond"] == "A"}
    for a in ALGOS:
        for s in (1, 2, 3):
            rr = [r for r in rows_b if r["algo"] == a and r["seedIdx"] == s]
            if not rr:
                continue
            d = {}
            for c in CONDS:
                x = [r for r in rr if r["cond"] == c]
                d[c] = {"n": len(x), "xi": R(np.mean([r["xi"] for r in x])),
                        **({"transfer": R(np.mean([r["xi"] - ctrl[(a, s, r["k"])] for r in x]))} if c != "A" else {}),
                        **{m: R(np.nanmean([r[m] for r in x])) for m in ("legalXI", "purseLeftShare", "squadSize", "overseas", "stars", "priceToFair", "bidRate", "shieldRate",
                                                                          "keeperClosedProgress", "keeperClosedForced")},
                        **{f"{m}Total": int(np.nansum([r[m] for r in x])) for m in ("forced", "finalPath", "forcedKeeperBids", "learnerFinding", "learnerFindingM2", "opponentFinding", "invariantViolations")}}
            out[f"{a}:s{s + 100}"] = d
    return out


def main():
    REP.mkdir(parents=True, exist_ok=True)
    res = {}
    for lv, (dec, limit) in LEVELS.items():
        rows_b, _ = load_level(lv)
        rows_a = load_old(limit)
        t = transfer_table(rows_b, rows_a)
        bo = {c: by_opponent(rows_b, rows_a, c) for c in ("C1", "C4")}
        res[lv] = {"decisions": dec, "entries": limit, "transfer": t, "byOpponent": bo, "robustness": robustness(t, bo),
                   "behaviour": behaviour(rows_b, rows_a), "perRun": per_run_rows(rows_b, lv), "episodes": len(rows_b)}
        print(lv, "episodes", len(rows_b), {c: {a: (t[c][a].get("transferStageB", {}).get("mean"), t[c][a].get("transferStageA", {}).get("mean")) for a in t[c]} for c in ("C1", "C4", "S4")}, flush=True)
    (REP / "raw").mkdir(exist_ok=True)
    (REP / "raw" / "analysis-core.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("ANALYSEDONE")


if __name__ == "__main__":
    main()
