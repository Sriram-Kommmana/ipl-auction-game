"""Phase 2F — list the Stage-B pilot checkpoints per evaluation level.

    python -m ipl_rl.stage_b.make_learners  → runs/stage_b/eval/learners_{c100,c250,c500}.json
A level's checkpoint is the run's first checkpoint at or after the target
decision count (PPO/A2C/DQN hit the targets exactly; ES stops at the first
generation boundary at or after them).
"""
import json
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[2]
LEVELS = {"c100": 98_304, "c250": 245_760, "c500": 497_664}
ALGOS = ("ppo", "a2c", "d3qn", "qrdqn", "es")
SEEDS = (101, 102, 103)


def main():
    out = {lv: [] for lv in LEVELS}
    missing = []
    for a in ALGOS:
        for s in SEEDS:
            run = ML_ROOT / "runs/stage_b" / a / f"s{s}"
            cks = []
            for p in sorted((run / "checkpoints").glob("*/policy.json")):
                meta = json.loads(p.read_text(encoding="utf-8"))["meta"]
                if meta.get("config", {}).get("stage_b", {}).get("tag") != "stage_b_pilot":
                    raise SystemExit(f"{p} is not a stage_b_pilot export")
                cks.append((int(meta["trainedSteps"]), p))
            for lv, target in LEVELS.items():
                hit = sorted((d, p) for d, p in cks if d >= target)
                if not hit:
                    missing.append(f"{a}:s{s}@{lv}")
                    continue
                d, p = hit[0]
                out[lv].append({"key": f"{a}:s{s}@{lv}", "algo": a, "seed": s, "stageASeed": s - 100, "decisions": d,
                                "policy": str(p.relative_to(ML_ROOT)).replace("\\", "/")})
    (ML_ROOT / "runs/stage_b/eval").mkdir(parents=True, exist_ok=True)
    for lv, rows in out.items():
        (ML_ROOT / f"runs/stage_b/eval/learners_{lv}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(lv, len(rows), "learners")
    if missing:
        print("MISSING", missing)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
