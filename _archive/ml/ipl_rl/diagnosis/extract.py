"""Phase 2E.1 — flatten the frozen Phase 2E.0 raw records into numpy columns
(read-only; cached as ml/runs/_2e1/episodes.npz).

    python extract.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "ml/runs/_2e0/full"
OUT = ROOT / "ml/runs/_2e1/episodes.npz"
ALGOS = ["ppo", "a2c", "d3qn", "qrdqn", "es"]
NUM = ["xi", "legalXI", "strongXI", "rank", "purseLeftShare", "squadSize", "overseas", "stars", "buys", "marginalBuys", "reauctionBuys", "priceToFair",
       "capToFair", "bidRate", "xiGainPer1000", "decisions", "contests", "shieldActivations", "shieldWins", "purseSpent", "purse"]
AUD = ["forced", "forcedWon", "forcedLost", "finalPath", "finalPathWon", "reauctionForced", "forcedKeeperBids", "forcedKeeperWon", "minPurseAnyUnmet"]
REQS = ["keeper", "bowling", "indians"]
OPPM = ["xi", "rank", "purseLeftShare", "priceToFair", "bidRate", "capToFair", "stars", "squadSize", "overseas", "buys", "reauctionBuys", "strongXI", "decisions"]


def f(x):
    return np.nan if x is None else float(x)


def main():
    cols = {}

    def put(k, v):
        cols.setdefault(k, []).append(v)

    for cond in ("A", "C1", "C4", "S4"):
        with open(RUN / f"{cond}.jsonl", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                s, a = r["summary"], r["audit"]
                put("cond", cond)
                put("learner", r["learner"])
                put("algo", r["learner"].split(":")[0])
                put("lseed", int(r["learner"].split(":s")[1]))
                opp = r["opponents"][0] if cond in ("C1", "C4") else ""
                put("oppKey", opp)
                put("oppAlgo", opp.split(":")[0] if opp else "")
                put("oseed", int(opp.split(":s")[1]) if opp else 0)
                put("k", r["k"])
                put("seed", r["seed"])
                put("stratum", r["stratum"])
                for m in NUM:
                    put(m, f(s.get(m)))
                for m in AUD:
                    put(m, f(a.get(m)))
                st = a["shieldStates"]
                for x in ("SAFE", "WARNING", "CRITICAL", "IMPOSSIBLE"):
                    put(f"state{x}", st.get(x, 0))
                for q in REQS:
                    c = r["req"][q]["closed"]
                    put(f"{q}Need", f(r["req"][q]["initialNeed"]))
                    put(f"{q}Closed", 1.0 if c else 0.0)
                    put(f"{q}Prog", f(c["progress"]) if c else np.nan)
                    put(f"{q}Forced", 1.0 if c and c["forced"] else 0.0)
                    put(f"{q}FinalPath", 1.0 if c and c["finalPath"] else 0.0)
                    put(f"{q}Reauction", 1.0 if c and c["phase"] == "reauction" else 0.0)
                    put(f"{q}Price", f(c["price"]) if c else np.nan)
                    put(f"{q}MinPurseUnmet", f(r["req"][q]["minPurseUnmet"]))
                put("actions", s["actionCounts"])
                o = r["opp"]
                for m in OPPM:
                    vals = [x[m] for x in o if x.get(m) is not None]
                    put(f"opp_{m}", float(np.mean(vals)) if vals else np.nan)
                put("opp_xiBest", max((x["xi"] for x in o), default=np.nan))
                put("opp_xiMin", min((x["xi"] for x in o), default=np.nan))
                for algo in ALGOS:  # S4 per-algorithm seat
                    x = [y for y in o if y["algo"] == algo]
                    for m in ("xi", "rank", "purseLeftShare", "priceToFair", "bidRate", "capToFair", "stars", "squadSize", "overseas"):
                        put(f"s4_{algo}_{m}", f(x[0].get(m)) if len(x) == 1 else np.nan)
                others = [t for t in r["others"] if t["type"] in ("rule", "rlFallback")]
                put("rule_xi", float(np.mean([t["xi"] for t in others])) if others else np.nan)
                put("findingLearner", sum(1 for x in r.get("findings", []) if x["type"] == "learner"))
                put("findingOpp", sum(1 for x in r.get("findings", []) if x["type"] == "rlSnapshot"))
    arr = {k: np.asarray(v) for k, v in cols.items()}
    np.savez_compressed(OUT, **arr)
    print({k: v.shape for k, v in list(arr.items())[:5]}, len(arr["xi"]), "episodes →", OUT)


if __name__ == "__main__":
    main()
