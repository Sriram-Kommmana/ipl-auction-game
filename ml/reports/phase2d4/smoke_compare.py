"""Phase 2D.4 reproducibility smoke: each ES seed run twice (a/b) must be identical; seeds must differ."""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parents[1] / "_smoke2d4"
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
    ve = [json.loads(p.read_text())["episodes"]["policy:es"] for p in sorted(d.glob("checkpoints/*/validation/episodes.json"))]
    pol = [json.loads(p.read_text())["layers"] for p in sorted(d.glob("checkpoints/*/policy.json"))]
    return sm, ups, evs, te, ve, pol


ok = True
inits = {}
for s in (1, 2, 3):
    a, b = load(s, "a"), load(s, "b")
    col = lambda run, k: [u[k] for u in run[1]]
    checks = {
        "status completed": a[0]["status"] == b[0]["status"] == "completed",
        "initial parameters": a[0]["initialParamsDigest"] == b[0]["initialParamsDigest"],
        "perturbation vectors (noise digests)": col(a, "noiseDigest") == col(b, "noiseDigest"),
        "perturbation / episode seeds": col(a, "episodeSeeds") == col(b, "episodeSeeds"),
        "policy actions (per-episode action counts)": col(a, "actionsDigest") == col(b, "actionsDigest"),
        "fitness values": col(a, "fitnessDigest") == col(b, "fitnessDigest"),
        "centered ranks": col(a, "shapedDigest") == col(b, "shapedDigest"),
        "gradient estimates": col(a, "gradDigest") == col(b, "gradDigest"),
        "optimizer states": col(a, "optimizerDigest") == col(b, "optimizerDigest"),
        "parameter digests (every generation)": col(a, "paramsDigest") == col(b, "paramsDigest"),
        "final parameters": a[0]["finalParamsDigest"] == b[0]["finalParamsDigest"],
        "training statistics": strip([u["train"] for u in a[1]]) == strip([u["train"] for u in b[1]]),
        "episode statistics": strip([u["episode"] for u in a[1]]) == strip([u["episode"] for u in b[1]]),
        "training episode summaries": strip(a[3]) == strip(b[3]),
        "checkpoint statistics + parity": strip(a[2]) == strip(b[2]),
        "validation episodes": strip(a[4]) == strip(b[4]),
        "exports (weights)": a[5] == b[5],
        "act-v3 / obs-v2": a[0]["actSpec"]["hash"] == "5f72f510c48b1f46" and a[0]["obsSpec"]["hash"] == "629b25783f833af7",
        "consistency checks = episodes": a[0]["episodeConsistencyChecks"] == len(a[3]) == a[0]["episodes"],
        "checkpoints written (decision + generation triggered)": len(a[2]) == len(b[2]) >= 2,
    }
    inits[s] = a[0]["initialParamsDigest"]
    print(f"seed {s}: initial {a[0]['initialParamsDigest']} final {a[0]['finalParamsDigest']} | generations {a[0]['generations']} decisions {a[0]['totalDecisions']:,} "
          f"episodes {a[0]['episodes']} evaluations {[(e['generation'], e['reasons']) for e in a[2]]} | fitness digests {col(a, 'fitnessDigest')}")
    for k, v in checks.items():
        print(f"   {'PASS' if v else 'FAIL'}  {k}")
        ok &= v
distinct = len(set(inits.values())) == 3
print(f"{'PASS' if distinct else 'FAIL'}  initial parameters differ across seeds: {inits}")
ok &= distinct
print("REPRODUCIBILITY:", "ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
