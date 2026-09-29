"""Phase 2E.0 Stage-A regression: the Stage-A control rerun through the
cross-play harness must reproduce every stored Stage-A validation episode of
every frozen export exactly (the evaluator episodes written when each export
was validated in Phases 2C.3 / 2D.1–2D.4), and the reported three-seed means.

    python check_stage_a.py <A.jsonl> [--limit N]
Exit 1 on any difference.
"""
import json
import statistics
import sys
from pathlib import Path

ML = Path(__file__).resolve().parents[2]
DIRS = {"ppo": "ppo-2c3", "a2c": "a2c-2d1", "d3qn": "d3qn-2d2", "qrdqn": "qr-dqn-2d3", "es": "openai-es-2d4"}
CKPT = {"ppo": "update_0325", "a2c": "update_0325", "d3qn": "update_0325", "qrdqn": "update_0325", "es": "gen_2000"}
# Reported Stage-A final validation XI (three-seed mean ± sample std), from the phase reports.
REPORTED = {"ppo": (92.595, 0.082), "a2c": (90.427, 0.046), "d3qn": (91.490, 0.191), "qrdqn": (91.853, 0.089), "es": (91.068, 0.044)}


def main(path, limit):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["cond"] == "A":
                rows[(r["learner"], r["k"])] = r
    problems, per_export, out = [], {}, {}
    for algo, d in DIRS.items():
        means = []
        for s in (1, 2, 3):
            key = f"{algo}:s{s}"
            stored = json.loads((ML / "runs" / f"{d}-s{s}" / "checkpoints" / CKPT[algo] / "validation" / "episodes.json").read_text())["episodes"][f"policy:{algo}"]
            n = min(limit, len(stored))
            same = 0
            for k in range(n):
                want = {x: v for x, v in stored[k].items() if x != "violations"}
                got = rows.get((key, k))
                if got is None:
                    problems.append(f"{key} entry {k}: missing")
                elif got["summary"] != want:
                    diff = sorted(x for x in set(want) | set(got["summary"]) if want.get(x) != got["summary"].get(x))
                    problems.append(f"{key} entry {k}: {diff}")
                else:
                    same += 1
            xi = statistics.mean(rows[(key, k)]["summary"]["xi"] for k in range(n) if (key, k) in rows)
            stored_xi = statistics.mean(e["xi"] for e in stored[:n])
            per_export[key] = {"episodes": n, "identical": same, "xi": xi, "storedXi": stored_xi}
            means.append(xi)
        m, sd = statistics.mean(means), statistics.stdev(means)
        out[algo] = {"mean": m, "std": sd, "reported": REPORTED[algo], "roundsToReported": (round(m, 3), round(sd, 3)) == REPORTED[algo] if limit >= 500 else None}
    return problems, per_export, out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    path = sys.argv[1]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 500
    problems, per_export, algos = main(path, limit)
    for key, v in per_export.items():
        print(f"{key:9s} {v['identical']}/{v['episodes']} episodes identical to the stored Stage-A evaluator episodes  XI {v['xi']:.4f} (stored {v['storedXi']:.4f})")
    for algo, v in algos.items():
        print(f"{algo:6s} three-seed XI {v['mean']:.3f} ± {v['std']:.3f}  (reported {v['reported'][0]:.3f} ± {v['reported'][1]:.3f})  match: {v['roundsToReported']}")
    for p in problems[:20]:
        print("  PROBLEM", p)
    ok = not problems and all(v["roundsToReported"] in (True, None) for v in algos.values())
    print(f"STAGE-A REGRESSION: {'PASS' if ok else 'FAIL'}")
    print(json.dumps({"perExport": per_export, "algorithms": algos, "problems": problems[:50], "pass": ok}))
    sys.exit(0 if ok else 1)
