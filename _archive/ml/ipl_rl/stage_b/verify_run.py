"""Phase 2F — verify one Stage-B pilot run (composition, seeds, safety, checkpoints).

    python -m ipl_rl.stage_b.verify_run runs/stage_b/ppo/s101   → <run>/verification.json
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[2]
TRAIN_START, VAL, TEST = 1_000_000, (100_000, 100_500), (200_000, 201_000)
EXPECTED_POOL = [f"{a}:s{s}" for a in ("ppo", "a2c", "d3qn", "qrdqn", "es") for s in (1, 2, 3)]


def verify(run):
    run = Path(run)
    rows = [json.loads(l) for f in sorted((run / "bridge_log").glob("*.jsonl")) for l in f.read_text(encoding="utf-8").splitlines() if l]
    summ = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    meta = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    warm = json.loads((run / "warm_start.json").read_text(encoding="utf-8"))
    n = len(rows)
    ks = Counter(len(r["snapshots"]) for r in rows)
    binom = {k: math.comb(4, k) * 0.5 ** 4 for k in range(5)}
    per = Counter(k for r in rows for k in r["snapshots"])
    seeds = [r["seed"] for r in rows]
    problems = []
    if min(r["ruleShare"] for r in rows) < 4 / 9 - 1e-9:
        problems.append("rule-bot share below 4/9")
    if any(s < TRAIN_START for s in seeds):
        problems.append("non-train seed played")
    if set(per) - set(EXPECTED_POOL):
        problems.append(f"unexpected snapshot keys {set(per) - set(EXPECTED_POOL)}")
    if summ.get("status") != "completed":
        problems.append(f"status {summ.get('status')}: {summ.get('failure')}")
    for ev in summ.get("evaluations", []):
        if ev.get("safetyProblems"):
            problems.append(f"validation safety problems at {ev.get('decisions')}: {ev['safetyProblems'][:3]}")
    tot = sum(per.values())
    out = {
        "run": str(run.relative_to(ML_ROOT)), "status": summ.get("status"), "configHash": summ.get("configHash") or meta.get("configHash"),
        "obsSpec": meta.get("obsSpec"), "actSpec": meta.get("actSpec"), "gamma": meta.get("gamma"),
        "warmStart": {k: warm[k] for k in ("export", "checkpoint", "sha256")},
        "trainingEpisodes": n, "bridgeLoggedDecisions": sum(r["decisions"] for r in rows), "trainerDecisions": summ.get("totalDecisions"),
        "snapshotsPerRoom": {str(k): {"share": round(ks[k] / n, 4), "expectedBinomial": round(binom[k], 4)} for k in range(5)},
        "meanSnapshotsPerRoom": round(sum(len(r["snapshots"]) for r in rows) / n, 4),
        "ruleShare": {"min": round(min(r["ruleShare"] for r in rows), 4), "mean": round(sum(r["ruleShare"] for r in rows) / n, 4)},
        "rlOpponentSeatShare": round(sum(len(r["snapshots"]) for r in rows) / (9 * n), 4),
        "snapshotSeatsByExport": {k: {"seats": per[k], "share": round(per[k] / tot, 4)} for k in EXPECTED_POOL},
        "ownAlgorithmSeatShare": round(sum(per[k] for k in per if k.split(":")[0] == run.parent.name.replace("qrdqn", "qrdqn")) / tot, 4) if tot else None,
        "seeds": {"min": min(seeds), "max": max(seeds), "distinct": len(set(seeds)), "allTrain": all(s >= TRAIN_START for s in seeds),
                  "validationOrTestPlayed": sum(1 for s in seeds if VAL[0] <= s < VAL[1] or TEST[0] <= s < TEST[1])},
        "learnerIncompleteXiTrainingEpisodes": sum(1 for r in rows if not r["legalXI"]),
        "incompleteSeeds": [r["seed"] for r in rows if not r["legalXI"]][:50],
        "checkpoints": [{"decisions": ev.get("decisions"), "validationEpisodes": ev.get("validationEpisodes"), "xi": ev["metrics"].get("xi"),
                         "legalXI": ev["metrics"].get("legalXI"), "parity": ev.get("parity"), "safetyProblems": ev.get("safetyProblems")} for ev in summ.get("evaluations", [])],
        "problems": problems,
    }
    (run / "verification.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    o = verify(ML_ROOT / sys.argv[1])
    print(json.dumps({k: o[k] for k in ("status", "trainingEpisodes", "trainerDecisions", "meanSnapshotsPerRoom", "ruleShare", "snapshotsPerRoom",
                                        "learnerIncompleteXiTrainingEpisodes", "seeds", "problems")}, indent=1))
    print("CHECKPOINTS", [(c["decisions"], round(c["xi"], 3)) for c in o["checkpoints"]])
    sys.exit(1 if o["problems"] else 0)
