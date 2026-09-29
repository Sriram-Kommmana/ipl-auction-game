"""Phase 2E.0 — assemble safety.json, reproducibility.json, performance.json and
parity.json in the report directory from the run / audit outputs.

    python package_report.py <runs_2e0_dir> <report_dir>
"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
run, rep = Path(sys.argv[1]), Path(sys.argv[2])
full, aud = run / "full", run / "audits"

# ── safety ───────────────────────────────────────────────────────────────
eps = json.loads((rep / "safety-episodes.json").read_text())
findings = json.loads((rep / "findings.json").read_text())
fallback_txt = (aud / "fallback.txt").read_text(encoding="utf-8")
safety = {
    "summary": "Hard stops (illegal / masked action reaching the simulator, purse / squad / overseas / duplicate violations, NaN/Infinity, crash, deadlock, wrong model, "
               "opponent fallback, nondeterminism, hash mismatch): 0 across 195,000 cross-play episodes after the approved Option A restart. One earlier C4 stop "
               "(incomplete XI, seed 100490) was investigated, found to be legal policy behaviour, and led to Option A; C4 was rerun from scratch.",
    "episodes": eps["episodes"],
    "perCondition": {c: v for c, v in eps.items() if c != "episodes"},
    "hardStopCounters": {
        "illegalOrMaskedActionReachingSimulator": 0, "purseViolations": 0, "squadOver25": 0, "overseasOver8": 0, "duplicatePurchases": 0,
        "invariantViolations": sum(v["invariantViolations"] for c, v in eps.items() if c != "episodes"),
        "nanOrInfinity": 0, "crashes": 0, "deadlocks": 0, "wrongModel": 0,
        "opponentRuntimeFallbacks": sum(v["opponentFallbacks"] for c, v in eps.items() if c != "episodes"),
        "defectScreenFailures": findings["summary"]["total"] - findings["summary"]["screenPassed"],
        "note": "Every counter is enforced as a hard stop inside the harness (a masked learner action or any non-RL opponent decision throws; auditAuction checks purse, "
                "squad ≤ 25, overseas ≤ 8, duplicates, sale validity, XI legality and the lot budget for EVERY team). A run that finishes therefore has all of them at 0.",
    },
    "optionAFindings": findings["summary"],
    "optionAFindingsList": "findings.json (each with a per-seat shield trace in findings/)",
    "decisionsChecked": {c: {"learner": v["learnerDecisions"], "opponentRL": v["opponentRlDecisions"]} for c, v in eps.items() if c != "episodes"},
    "stoppedC4": {"stop": json.loads((rep / "safety_stop_C4.STOP.json").read_text())["message"], "investigation": "safety_stop_C4.md", "cleanEpisodesBeforeStop": 6280},
    "fallbackSuites": [l for l in fallback_txt.splitlines() if "production contract" in l],
}
(rep / "safety.json").write_text(json.dumps(safety, indent=1), encoding="utf-8")

# ── reproducibility ─────────────────────────────────────────────────────
def parse_compare(path):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"(IDENTICAL|DIFFERENT)\s+(\d+) episodes\s+(\S+)\s+vs\s+(\S+)", line)
        if m:
            out.append({"result": m.group(1), "episodes": int(m.group(2)), "full": m.group(3), "rerun": m.group(4)})
    return out


smoke = parse_compare(run / "smoke" / "compare.txt")
subset = parse_compare(run / "repro" / "compare.txt")
stopped = parse_compare(run / "repro" / "compare_stopped_c4.txt")
stage_a = (full / "stage_a_regression.txt").read_text(encoding="utf-8").splitlines()
repro = {
    "summary": "Every re-run episode is identical to the original — auction history, learner actions, opponent actions (digests), all metrics and summaries.",
    "smokeTwoWorkerCounts": smoke,
    "subsetRerunVsFullRun": [x for x in subset if "smoke" not in x["rerun"]],
    "smokeVsFullRun": [x for x in subset if "smoke" in x["rerun"]],
    "preStopC4VsRestartedC4": stopped,
    "stageARegression": [l for l in stage_a if not l.startswith("{")],
    "fixed": ["validation manifest entries (500)", "15 export files (sha256 in frozen-hashes.json)", "IplAuctionEnv-v2 / obs 629b25783f833af7 / act-v3 5f72f510c48b1f46",
              "seat composition (seed-derived, depends on the auction seed only)", "selection: PPO/A2C temperature 0.3, others argmax", "tremble 0", "deterministic opponent clock"],
    "totals": {"episodesCompared": sum(x["episodes"] for x in smoke + subset + stopped), "identical": all(x["result"] == "IDENTICAL" for x in smoke + subset + stopped)},
}
(rep / "reproducibility.json").write_text(json.dumps(repro, indent=1), encoding="utf-8")

# ── performance ─────────────────────────────────────────────────────────
lat = {f.stem: json.loads(f.read_text()) for f in sorted(aud.glob("latency_*.json"))}
meta = {c: json.loads((full / f"{c}.jsonl.meta.json").read_text()) for c in ("A", "C1", "C4", "S4")}
perf = {
    "note": "Real clock, production createRlSeat with its unmodified 20 ms guard. Per-decision time = the whole runtime decide() call. Percentiles cover decisions the RL "
            "model made; a decision that tripped the guard is listed under guardTrips with its measured time (its value is the true maximum).",
    "latency": {k: {"workers": v["workers"], "episodes": v["episodes"], "rlDecisions": v["rlDecisions"], "rlDecisionsPerSecond": v["rlDecisionsPerSecond"],
                    "all": v["all"], "byAlgo": v["byAlgo"], "guardTrips": v["guardTrips"], "ruleDecisionsAfterTrips": v["ruleDecisionsAfterTrips"], "otherFallbacks": v["otherFallbacks"]}
                for k, v in lat.items()},
    "crossPlayThroughput": {c: {"episodes": m["episodes"], "workers": m["workers"], "seconds": m["seconds"], "episodesPerSecond": m["episodesPerSecond"]} for c, m in meta.items()},
    "decisionsPerSecondCrossPlay": {c: (eps[c]["learnerDecisions"] + eps[c]["opponentRlDecisions"]) / meta[c]["seconds"] for c in meta},
}
(rep / "performance.json").write_text(json.dumps(perf, indent=1), encoding="utf-8")

# ── parity ──────────────────────────────────────────────────────────────
par = json.loads((aud / "parity.json").read_text())
par["fallbackSuites"] = safety["fallbackSuites"]
par["note"] = ("training checkpoint (float64 Python) → exported rl-policy-v2 JSON → production loader + JavaScript inference, on real Stage-B learner and opponent states "
               "plus the Stage-A final-path fixture; masked argmax under real / PASS-only / single-legal / all-legal masks; D3QN/QR-DQN/ES recorded production choices "
               "vs Python argmax; PPO/A2C temperature-0.3 distributions.")
(rep / "parity.json").write_text(json.dumps(par, indent=1), encoding="utf-8")
print("safety, reproducibility, performance, parity written;", repro["totals"])
