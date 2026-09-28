"""Phase 2D.3 reproducibility smoke: each seed run twice (a/b) must be identical; seeds must differ."""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parents[1] / "_smoke2d3"
IGNORE = {"perf", "seconds", "wallSeconds", "timers", "decisionsPerSecOverall", "runName", "createdAt"}


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in IGNORE}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def load(seed, r):
    d = D / f"s{seed}-{r}"
    sm = json.loads((d / "summary.json").read_text())
    rows = [json.loads(l) for l in open(d / "metrics.jsonl")]
    ups = [x for x in rows if x["type"] == "update"]
    evs = [x for x in rows if x["type"] == "evaluation"]
    te = json.loads((d / "training_episodes.json").read_text())
    ve = [json.loads(p.read_text())["episodes"]["policy:qrdqn"] for p in sorted(d.glob("checkpoints/*/validation/episodes.json"))]
    pol = [json.loads(p.read_text())["layers"] for p in sorted(d.glob("checkpoints/*/policy.json"))]
    return sm, ups, evs, te, ve, pol


ok = True
inits = {}
for s in (1, 2, 3):
    a, b = load(s, "a"), load(s, "b")
    col = lambda run, k: [u[k] for u in run[1]]
    checks = {
        "status completed": a[0]["status"] == b[0]["status"] == "completed",
        "initial online + target digests": (a[0]["initialParamsDigest"], a[0]["initialTargetDigest"]) == (b[0]["initialParamsDigest"], b[0]["initialTargetDigest"]),
        "initial target = initial online": a[0]["initialParamsDigest"] == a[0]["initialTargetDigest"],
        "final online + target digests": (a[0]["finalParamsDigest"], a[0]["finalTargetDigest"]) == (b[0]["finalParamsDigest"], b[0]["finalTargetDigest"]),
        "action sequence (per window)": col(a, "actionsDigest") == col(b, "actionsDigest"),
        "epsilon / exploration sequence": col(a, "exploreDigest") == col(b, "exploreDigest") and [u["train"]["explored"] for u in a[1]] == [u["train"]["explored"] for u in b[1]],
        "replay sampling (sampled indices)": col(a, "samplesDigest") == col(b, "samplesDigest"),
        "replay priorities": col(a, "priorityDigest") == col(b, "priorityDigest"),
        "online / target digests per window": col(a, "onlineDigest") == col(b, "onlineDigest") and col(a, "targetDigest") == col(b, "targetDigest"),
        "losses and training statistics": strip([u["train"] for u in a[1]]) == strip([u["train"] for u in b[1]]),
        "target-network sync digests": a[0]["targetSyncs"] == b[0]["targetSyncs"],
        "training episode summaries": strip(a[3]) == strip(b[3]),
        "validation episodes": strip(a[4]) == strip(b[4]),
        "exports (weights)": a[5] == b[5],
        "evaluation metrics + parity": strip(a[2]) == strip(b[2]),
        "optimisation budget": a[0]["optimisationBudget"] == b[0]["optimisationBudget"],
        "act-v3 5f72f510c48b1f46": a[0]["actSpec"]["hash"] == b[0]["actSpec"]["hash"] == "5f72f510c48b1f46",
        "consistency checks = episodes": a[0]["episodeConsistencyChecks"] == len(a[3]) and b[0]["episodeConsistencyChecks"] == len(b[3]),
    }
    inits[s] = a[0]["initialParamsDigest"]
    ob = a[0]["optimisationBudget"]
    print(f"seed {s}: initial {a[0]['initialParamsDigest']} final {a[0]['finalParamsDigest']} target {a[0]['finalTargetDigest']} | decisions {a[0]['totalDecisions']:,} "
          f"updates {ob['gradientUpdates']:,} target syncs {ob['targetUpdates']} replay added {ob['replayTransitionsGenerated']:,} episodes {len(a[3])} | "
          f"window action digests {col(a, 'actionsDigest')}")
    for k, v in checks.items():
        print(f"   {'PASS' if v else 'FAIL'}  {k}")
        ok &= v
distinct = len(set(inits.values())) == 3
print(f"{'PASS' if distinct else 'FAIL'}  initial digests differ across seeds: {inits}")
ok &= distinct
print("REPRODUCIBILITY:", "ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
