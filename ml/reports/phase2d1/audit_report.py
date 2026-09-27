"""Summarise audit_requirements.json per tag and cross-check it against the evaluator's episodes."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent
RUNS = D.parent
REQS = ["keeper", "bowling", "indians"]
rows = json.loads((D / "audit_requirements.json").read_text())
by = defaultdict(list)
for r in rows:
    by[r["tag"]].append(r)

RUN = {"a2c": ("a2c-2d1-s{}", "policy:a2c"), "ppo": ("ppo-2c3-s{}", "policy:ppo")}
mismatch = 0
for tag, rs in by.items():
    algo, s, u = tag.split("-")
    run, name = RUN[algo]
    ev = json.loads((RUNS / run.format(s[1:]) / "checkpoints" / f"update_{int(u[1:]):04d}" / "validation" / "episodes.json").read_text())["episodes"][name]
    ev = {e["seed"]: e for e in ev}
    for r in rs:
        e = ev[r["seed"]]
        if (r["legalXI"], r["xi"], r["purseLeft"], r["decisions"], r["forced"]) != (e["legalXI"], e["xi"], e["purseLeft"], e["decisions"], e["shield"]["forced"]):
            mismatch += 1
print(f"replay vs evaluator episodes: {len(rows) - mismatch}/{len(rows)} identical (legal XI, XI, purse, decisions, forced)")


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def order(t):
    a, s, u = t.split("-")
    return (a, s, int(u[1:]))


summary = {}
for tag in sorted(by, key=order):
    rs = by[tag]
    n = len(rs)
    print(f"== {tag}  (n={n})  legal {sum(r['legalXI'] for r in rs)}/{n}")
    row = {}
    for req in REQS:
        needing = [r for r in rs if r["req"][req]["initialNeed"]]
        closed = [r["req"][req]["closed"] for r in needing if r["req"][req]["closed"]]
        mp = np.array([r["req"][req]["minPurseUnmet"] for r in needing if r["req"][req]["minPurseUnmet"] is not None])
        prog = [c["progress"] for c in closed]
        price = [c["price"] for c in closed]
        stats = {
            "needing": len(needing), "closed": len(closed), "closedMain": sum(c["phase"] == "main" for c in closed),
            "closedForced": sum(c["forced"] for c in closed), "closedFinalPath": sum(c["finalPath"] for c in closed),
            "closedAtFinalOpportunity": sum(c["finalOpportunity"] for c in closed),
            "progressP50": pct(prog, 50), "progressP90": pct(prog, 90), "priceP50": pct(price, 50),
            "minPurseUnmetP10": pct(mp, 10), "minPurseUnmetP50": pct(mp, 50), "minPurseUnmetLe40": int((mp <= 40).sum()),
        }
        row[req] = stats
        print(f"  {req:8s} closed {stats['closed']}/{stats['needing']} (main {stats['closedMain']}) | forced {stats['closedForced']} final-path {stats['closedFinalPath']} "
              f"at final opportunity {stats['closedAtFinalOpportunity']} | progress at close p50 {stats['progressP50']:.3f} p90 {stats['progressP90']:.3f} | "
              f"closing price p50 ₹{stats['priceP50']:.0f}L | min purse while unmet p10 ₹{stats['minPurseUnmetP10']:.0f}L p50 ₹{stats['minPurseUnmetP50']:.0f}L, ≤₹40L in {stats['minPurseUnmetLe40']}/{len(mp)}")
    anyp = np.array([r["minPurseAnyUnmet"] for r in rs if r["minPurseAnyUnmet"] is not None])
    states, comp = defaultdict(int), defaultdict(int)
    forced_by = defaultdict(int)
    for r in rs:
        for k, v in r["shieldStates"].items():
            states[k] += v
        for k, v in r["completion"].items():
            comp[k] += v
        for k, v in r["forcedBy"].items():
            forced_by[k] += v
    tot = {k: sum(r[k] for r in rs) for k in ("forced", "forcedWon", "finalPath", "finalPathWon", "reauctionForced", "alreadyInfeasible", "decisions")}
    row.update({"anyUnmetP10": pct(anyp, 10), "anyUnmetP50": pct(anyp, 50), "anyUnmetLe40": int((anyp <= 40).sum()), **tot,
                "shieldStates": dict(states), "completion": dict(comp), "forcedBy": dict(forced_by)})
    summary[tag] = row
    print(f"  any requirement unmet: min purse p10 ₹{row['anyUnmetP10']:.0f}L p50 ₹{row['anyUnmetP50']:.0f}L, ≤₹40L in {row['anyUnmetLe40']}/{len(anyp)}")
    print(f"  forced {tot['forced']}/{tot['decisions']} (won {tot['forcedWon']}) | final-path forced {tot['finalPath']} (won {tot['finalPathWon']}) | re-auction forced {tot['reauctionForced']} | "
          f"already infeasible {tot['alreadyInfeasible']} | shield states {dict(states)} | planner completion {dict(comp)} | forcedBy {dict(forced_by)}")
(D / "audit_summary.json").write_text(json.dumps(summary, indent=1))
