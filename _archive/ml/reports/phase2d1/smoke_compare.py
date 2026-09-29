"""Phase 2D.1 reproducibility smoke: each seed run twice (a/b) must be identical."""
import json
import sys
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "_smoke2d1"
PPO_INITIAL = {1: "5d3196dbd3f13883", 2: "0f7d17d3a2595f05", 3: "f6c2f9465df72b60"}
IGNORE = {"perf", "seconds", "wallSeconds", "envSeconds", "decisionsPerSecOverall", "simulatorEpisodesPerSec", "runName", "createdAt"}


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in IGNORE}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def load(seed, r):
    d = D / f"s{seed}-{r}"
    sm = json.loads((d / "summary.json").read_text())
    ups = [json.loads(l) for l in open(d / "metrics.jsonl") if '"type": "update"' in l]
    te = json.loads((d / "training_episodes.json").read_text())
    ve = json.loads(next(d.glob("checkpoints/*/validation/episodes.json")).read_text())["episodes"]["policy:a2c"]
    pol = json.loads(next(d.glob("checkpoints/*/policy.json")).read_text())
    return sm, ups, te, ve, pol


ok = True
inits = {}
for s in (1, 2, 3):
    a, b = load(s, "a"), load(s, "b")
    checks = {
        "status completed": a[0]["status"] == b[0]["status"] == "completed",
        "initial params digest": a[0]["initialParamsDigest"] == b[0]["initialParamsDigest"],
        "final params digest": a[0]["finalParamsDigest"] == b[0]["finalParamsDigest"],
        "per-update action digests": [u["actionsDigest"] for u in a[1]] == [u["actionsDigest"] for u in b[1]],
        "training statistics": strip([u["train"] for u in a[1]]) == strip([u["train"] for u in b[1]]),
        "rollout episode stats": strip([u["episode"] for u in a[1]]) == strip([u["episode"] for u in b[1]]),
        "training episode summaries": strip(a[2]) == strip(b[2]),
        "validation episodes": strip(a[3]) == strip(b[3]),
        "exported weights": a[4]["layers"] == b[4]["layers"],
        "act-v3 5f72f510c48b1f46": a[0]["actSpec"]["hash"] == b[0]["actSpec"]["hash"] == "5f72f510c48b1f46",
        "consistency checks = episodes": a[0]["episodeConsistencyChecks"] == len(a[2]) and b[0]["episodeConsistencyChecks"] == len(b[2]),
        "initial digest ≠ PPO 2C.3 initial": a[0]["initialParamsDigest"] != PPO_INITIAL[s],
    }
    inits[s] = a[0]["initialParamsDigest"]
    print(f"seed {s}: initial {a[0]['initialParamsDigest']} final {a[0]['finalParamsDigest']} | updates {len(a[1])} decisions {a[0]['totalDecisions']} "
          f"training episodes {len(a[2])} first seeds {[e['seed'] for e in a[2][:3]]} | action digests {[u['actionsDigest'] for u in a[1]]}")
    for k, v in checks.items():
        print(f"   {'PASS' if v else 'FAIL'}  {k}")
        ok &= v
distinct = len(set(inits.values())) == 3
print(f"{'PASS' if distinct else 'FAIL'}  initial digests differ across seeds: {inits}")
ok &= distinct
print("REPRODUCIBILITY:", "ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
