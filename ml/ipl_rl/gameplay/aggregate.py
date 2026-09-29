"""Aggregate gameplay-audit JSON lines (audit.mjs) into per-model product metrics.

    python aggregate.py <glob> [--by cand|label] [--json out.json]
Means per episode; p50/p95/p99/max latency over sampled RL decisions.
"""
import glob
import json
import sys
from collections import defaultdict

import numpy as np

pattern = sys.argv[1]
by = sys.argv[sys.argv.index("--by") + 1] if "--by" in sys.argv else "cand"
out_json = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None

seats, rooms = defaultdict(list), defaultdict(list)
for f in sorted(glob.glob(pattern)):
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        if r["type"] == "seat":
            seats[r.get(by) or r["label"]].append(r)
        else:
            rooms[r.get("cand") or "roster"].append(r)


def mean(rows, k):
    v = [r[k] for r in rows if r.get(k) is not None]
    return float(np.mean(v)) if v else None


def summarise(rows):
    s = {"n": len(rows)}
    for k in ["xi", "legalXI", "squad", "overseas", "keepers", "bowlers", "allRounders", "batsmen", "purseLeftShare", "spent25", "spent50", "spent75", "spentMain",
              "stars", "starsBy30", "buys", "reBuys", "contestedWins", "warWins", "winPriceFair", "overbid2", "overbid3", "lostAtCap", "bidShare", "topAction",
              "actionEntropy", "capFairStar", "capFairAll", "msMean"]:
        s[k] = mean(rows, k)
    rl = [r for r in rows if r["kind"] == "rl"]
    if rl:
        tot = lambda k: int(sum(r.get(k) or 0 for r in rl))
        s.update({k + "Total": tot(k) for k in ["decisions", "fallback", "guard", "forced", "capViol", "critical", "criticalBid", "criticalPassAfford",
                                                  "starEarly", "starEarlyPassRich", "lowPurse", "lowPurseBid", "reDec", "reBid", "violations"]})
        s["incompleteXI"] = int(sum(1 for r in rl if not r["legalXI"]))
        s["disabledSeats"] = int(sum(1 for r in rl if r.get("disabled")))
        ms = np.array([x for r in rl for x in r.get("msSample", [])])
        mx = max(r.get("msMax") or 0 for r in rl)
        if len(ms):
            s["latency"] = {"mean": float(ms.mean()), "p50": float(np.percentile(ms, 50)), "p95": float(np.percentile(ms, 95)), "p99": float(np.percentile(ms, 99)), "max": float(mx), "n": int(len(ms))}
        acts = np.sum([r["actions"] for r in rl], axis=0)
        s["actionShares"] = (acts / max(1, acts.sum())).round(3).tolist()
        s["purseStrata"] = {st: mean([r for r in rl if r["stratum"] == st], "xi") for st in ("low", "normal", "high")}
        s["legalByStratum"] = {st: mean([r for r in rl if r["stratum"] == st], "legalXI") for st in ("low", "normal", "high")}
    return s


res = {k: summarise(v) for k, v in seats.items()}
room_res = {}
for k, v in rooms.items():
    room_res[k] = {m: float(np.mean([r[m] for r in v])) for m in ["lots", "unsold", "reLots", "contested", "wars", "starContested", "stars", "meanBidsSold", "soldPriceFair"]}
    room_res[k]["violationRooms"] = sum(1 for r in v if r["violations"])
    room_res[k]["n"] = len(v)

cols = [("xi", "XI", "{:.2f}"), ("legalXI", "legal", "{:.3f}"), ("squad", "squad", "{:.1f}"), ("overseas", "os", "{:.1f}"), ("keepers", "WK", "{:.1f}"),
        ("purseLeftShare", "left", "{:.3f}"), ("spent25", "sp25", "{:.2f}"), ("spent50", "sp50", "{:.2f}"), ("spent75", "sp75", "{:.2f}"),
        ("stars", "stars", "{:.1f}"), ("starsBy30", "st30", "{:.1f}"), ("reBuys", "reBuy", "{:.1f}"), ("winPriceFair", "p/f", "{:.2f}"), ("overbid2", "ob2", "{:.2f}"),
        ("contestedWins", "cWins", "{:.1f}"), ("bidShare", "bid", "{:.2f}"), ("topAction", "topA", "{:.2f}"), ("actionEntropy", "H", "{:.2f}"), ("capFairStar", "cfStar", "{:.2f}")]
print("key".ljust(14) + " n   " + " ".join(h.rjust(6) for _, h, _ in cols) + "  inc guard forced fb capV critPassAff  p99ms  maxms")
for k in sorted(res):
    s = res[k]
    vals = " ".join((f.format(s[c]) if s.get(c) is not None else "-").rjust(6) for c, _, f in cols)
    extra = ""
    if "incompleteXI" in s:
        lat = s.get("latency", {})
        extra = f"  {s['incompleteXI']:>3} {s['guardTotal']:>5} {s['forcedTotal']:>6} {s['fallbackTotal']:>3} {s['capViolTotal']:>4} {s['criticalPassAffordTotal']:>4}/{s['criticalTotal']:<5} {lat.get('p99', 0):6.3f} {lat.get('max', 0):6.2f}"
    print(k.ljust(14) + f" {s['n']:<4}" + vals + extra)
print()
for k in sorted(room_res):
    r = room_res[k]
    print(f"rooms[{k}] n {r['n']} lots {r['lots']:.0f} unsold {r['unsold']:.0f} re {r['reLots']:.0f} contested {r['contested']:.0f} wars(≥6 bids) {r['wars']:.0f} "
          f"stars {r['stars']:.0f} starContested {r['starContested']:.0f} bids/sold {r['meanBidsSold']:.1f} price/fair {r['soldPriceFair']:.2f} violationRooms {r['violationRooms']}")
if out_json:
    json.dump({"seats": res, "rooms": room_res}, open(out_json, "w"), indent=1)
