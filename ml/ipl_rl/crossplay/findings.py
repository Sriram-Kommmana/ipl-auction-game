"""Phase 2E.0 Option A findings: every RL-controlled team (learner or opponent)
that ended with an incomplete XI after passing the defect screen. Collects
them from the run outputs and writes a per-seat shield trace for each
(diagnose_incomplete.mjs replays the room exactly; read-only).

    python findings.py <run_dir> <report_dir> [--traces]
"""
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]


def collect(run_dir):
    out = []
    for cond in ("A", "C1", "C4", "S4"):
        path = Path(run_dir) / f"{cond}.jsonl"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if '"findings"' not in line:
                    continue
                r = json.loads(line)
                for f in r["findings"]:
                    out.append({"cond": cond, "k": r["k"], "seed": r["seed"], "stratum": r["stratum"], "purse": r["purse"], "learner": r["learner"],
                                "opponents": r["opponents"], "seat": f["seat"], "type": f["type"], "key": f["key"], "rlSeat": f["rlSeat"],
                                "emptySlots": f["emptySlots"], "squadSize": f["squadSize"], "overseas": f["overseas"], "purseLeft": f["purseLeft"],
                                "xi": f["xi"], "screenPass": f["screen"]["pass"], "screenAttempts": f["screen"]["attempts"],
                                "plainReplayIdentical": f["screen"]["plainReplayIdentical"], "learnerXI": r["summary"]["xi"]})
    return out


def trace(f, out_dir):
    if f["cond"] == "S4":
        args = [str(f["k"]), f["learner"], "S4", f["learner"].split(":s")[1], str(f["seat"])]
    else:
        args = [str(f["k"]), f["learner"], f["cond"], f["opponents"][0], str(f["seat"])]
    name = f"{f['cond']}_seed{f['seed']}_{f['learner'].replace(':', '-')}_vs_{(f['opponents'][0] if f['cond'] != 'S4' else 'S4').replace(':', '-')}_seat{f['seat']}.txt"
    res = subprocess.run(["node", str(ROOT / "ml/ipl_rl/crossplay/diagnose_incomplete.mjs"), *args], capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    (out_dir / name).write_text(res.stdout + res.stderr, encoding="utf-8")
    return name, res.returncode


if __name__ == "__main__":
    run_dir, report_dir = sys.argv[1], Path(sys.argv[2])
    F = collect(run_dir)
    summary = {
        "total": len(F),
        "byCondition": Counter(f["cond"] for f in F),
        "byType": Counter(f["type"] for f in F),
        "byExport": Counter(f["key"] for f in F),
        "byAlgorithm": Counter(f["key"].split(":")[0] for f in F),
        "byConditionAndAlgorithm": Counter(f"{f['cond']} {f['key'].split(':')[0]} ({f['type']})" for f in F),
        "byStratum": Counter(f["stratum"] for f in F),
        "screenPassed": sum(f["screenPass"] for f in F),
        "plainReplayIdentical": sum(f["plainReplayIdentical"] for f in F),
    }
    if "--traces" in sys.argv and F:
        tdir = report_dir / "findings"
        tdir.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(12) as pool:
            done = list(pool.map(lambda f: trace(f, tdir), F))
        for f, (name, rc) in zip(F, done):
            f["trace"] = f"findings/{name}"
            f["traceExit"] = rc
    (report_dir / "findings.json").write_text(json.dumps({"summary": summary, "findings": F}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
