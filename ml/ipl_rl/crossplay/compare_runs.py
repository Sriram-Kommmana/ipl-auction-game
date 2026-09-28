"""Phase 2E.0 reproducibility: two runs of the same jobs must be identical,
episode by episode (every field: auction results, learner actions and
opponent actions via their digests, metrics, summaries).

    python compare_runs.py [--subset] <run_a.jsonl> <run_b.jsonl> [...pairs]
Episodes are matched by (condition, learner export, opponent exports, manifest
entry), not by job number, so a rerun of a SUBSET (fewer entries, other worker
count) can be checked against the full run: with --subset every episode of b
must exist in a and be identical. Exit 1 on any difference or missing episode.
"""
import json
import sys


def load(path, keys=None):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (r["cond"], r["learner"], tuple(r["opponents"]), r["k"])
            if keys is not None and key not in keys:
                continue
            r.pop("job")
            out[key] = r
    return out


def compare(a_path, b_path, subset=False):
    b = load(b_path)
    a = load(a_path, set(b) if subset else None)
    problems = []
    if set(a) != set(b):
        problems.append(f"episode sets differ: {len(set(a) ^ set(b))} episodes")
    fields_checked = 0
    for job in sorted(set(a) & set(b)):
        if a[job] != b[job]:
            diff = sorted(k for k in set(a[job]) | set(b[job]) if a[job].get(k) != b[job].get(k))
            problems.append(f"{job}: {diff}")
        fields_checked += len(json.dumps(a[job]))
    digests = {k: sorted({r["digests"][k] for r in a.values()}) for k in ("learnerActions", "auction", "summary")}
    return {
        "a": a_path, "b": b_path, "episodes": len(a), "identical": not problems, "problems": problems[:20],
        "distinctDigests": {k: len(v) for k, v in digests.items()},
        "opponentActionDigests": len({o["actionsDigest"] for r in a.values() for o in r["opp"]}),
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    subset = "--subset" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--subset"]
    results = [compare(args[i], args[i + 1], subset) for i in range(0, len(args), 2)]
    for r in results:
        print(f"{'IDENTICAL' if r['identical'] else 'DIFFERENT'}  {r['episodes']} episodes  {r['a']}  vs  {r['b']}  distinct digests {r['distinctDigests']} opponent-action digests {r['opponentActionDigests']}")
        for p in r["problems"]:
            print("   ", p)
    print(json.dumps(results))
    sys.exit(0 if all(r["identical"] for r in results) else 1)
