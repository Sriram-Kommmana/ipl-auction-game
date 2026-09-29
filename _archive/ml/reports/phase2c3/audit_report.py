"""Summarise audit_keepers.json per checkpoint, and cross-check it against the evaluator's episodes."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent
RUNS = D.parent
rows = json.loads((D / "audit_keepers.json").read_text())
by = defaultdict(list)
for r in rows:
    by[r["tag"]].append(r)

mismatch = 0
for tag, rs in sorted(by.items()):
    s, u = tag.split("-")
    ev = json.loads((RUNS / f"ppo-2c3-{s}" / "checkpoints" / f"update_{u[1:]}" / "validation" / "episodes.json").read_text())["episodes"]["policy:ppo"]
    ev = {e["seed"]: e for e in ev}
    for r in rs:
        e = ev[r["seed"]]
        if (r["legalXI"], r["xi"], r["purseLeft"], r["decisions"]) != (e["legalXI"], e["xi"], e["purseLeft"], e["decisions"]) or r["forced"] != e["shield"]["forced"]:
            mismatch += 1
print(f"replay vs evaluator episodes: {len(rows) - mismatch}/{len(rows)} identical (legal, XI, purse, decisions, forced)")

print(f"{'ckpt':9s} {'keeper bought':>14s} {'in main':>8s} {'forced':>7s} {'final-path':>10s} {'price p50/p90':>14s} {'progress p50':>12s} "
      f"{'min purse while keeper needed p10/p50':>38s} {'≤₹40L':>7s} {'forced won/total':>17s} {'final-path won/total':>21s} keeperForcedLots")
for tag in sorted(by, key=lambda t: (t.split("-")[0], t.split("-")[1])):
    rs = by[tag]
    n = len(rs)
    k = [r["keeper"] for r in rs if r["keeper"]]
    mp = np.array([r["minPurseKeeperNeeded"] for r in rs if r["minPurseKeeperNeeded"] is not None])
    price = np.array([x["price"] for x in k]) if k else np.array([0])
    prog = np.array([x["progress"] for x in k]) if k else np.array([0])
    f = sum(r["forced"] for r in rs); fw = sum(r["forcedWon"] for r in rs)
    fp = sum(r["finalPath"] for r in rs); fpw = sum(r["finalPathWon"] for r in rs)
    print(f"{tag:9s} {len(k):>6d}/{n:<6d} {sum(x['phase'] == 'main' for x in k):>4d}/{len(k):<3d} {sum(x['forced'] for x in k):>7d} {sum(x['finalPath'] for x in k):>10d} "
          f"{np.percentile(price, 50):>6.0f}/{np.percentile(price, 90):<6.0f} {np.percentile(prog, 50):>12.3f} "
          f"{np.percentile(mp, 10):>18.0f}/{np.percentile(mp, 50):<18.0f} {int((mp <= 40).sum()):>4d}/{len(mp):<3d} {fw:>8d}/{f:<8d} {fpw:>10d}/{fp:<10d} {sum(r['keeperForcedLots'] for r in rs)}")
