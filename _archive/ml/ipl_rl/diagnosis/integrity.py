"""Phase 2E.1 §3 — data integrity of the frozen Phase 2E.0 results (read-only).

    python integrity.py <out.json>
Checks: files present + sha256 (recorded as the Phase 2E.1 baseline), episode
counts per condition and per cell, duplicate/missing episodes, required fields,
the 20 findings, the Stage-A control, raw records vs published summaries
(matchup-matrix.json, episode-results.json, safety-episodes.json), frozen hashes.
"""
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
REP = ROOT / "ml/reports/phase2e0"
RUN = ROOT / "ml/runs/_2e0/full"
ALGOS = ["ppo", "a2c", "d3qn", "qrdqn", "es"]
REQUIRED = ["job", "k", "cond", "learner", "opponents", "seed", "stratum", "purse", "summary", "req", "audit", "opp", "others", "incompleteByType", "digests"]
SUMMARY_FIELDS = ["xi", "legalXI", "strongXI", "rank", "purseLeft", "purseSpent", "squadSize", "overseas", "stars", "marginalBuys", "reauctionBuys", "priceToFair",
                  "capToFair", "bidRate", "xiGainPer1000", "shieldActivations", "shield", "actionCounts", "invariantViolations"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(out):
    res = {"files": {}, "problems": []}
    report_files = sorted(p for p in REP.rglob("*") if p.is_file())
    raw_files = sorted(p for p in RUN.glob("*") if p.is_file())
    for p in report_files + raw_files:
        res["files"][str(p.relative_to(ROOT)).replace("\\", "/")] = {"bytes": p.stat().st_size, "sha256": sha(p)}
    res["filesNote"] = "Phase 2E.0 did not record hashes of its result files; these sha256 values are recorded now as the Phase 2E.1 baseline (verified again at the end of the analysis)."
    expected = {"report.md", "matchup-matrix.json", "matchup-results.json", "episode-results.json", "safety.json", "safety-episodes.json", "findings.json", "reproducibility.json",
                "parity.json", "performance.json", "frozen-hashes.json"}
    present = {p.name for p in REP.glob("*")}
    res["expectedReportFilesMissing"] = sorted(expected - present)
    res["plots"] = sorted(p.name for p in (REP / "plots").glob("*"))
    res["findingTraces"] = len(list((REP / "findings").glob("*.txt")))

    # ── raw records ───────────────────────────────────────────────────
    counts, cells, keys_seen, dup, missing_fields, nulls = Counter(), Counter(), set(), 0, Counter(), Counter()
    findings_raw, xi_by = [], {}
    incomplete = Counter()
    a_xi = {}
    transfer = defaultdict(list)
    raw_ep = {}
    for cond in ("A", "C1", "C4", "S4"):
        with open(RUN / f"{cond}.jsonl", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                counts[cond] += 1
                for f in REQUIRED:
                    if f not in r:
                        missing_fields[f"{cond}:{f}"] += 1
                for f in SUMMARY_FIELDS:
                    if f not in r["summary"]:
                        missing_fields[f"{cond}:summary.{f}"] += 1
                    elif r["summary"][f] is None:
                        nulls[f"{cond}:summary.{f}"] += 1
                key = (cond, r["learner"], tuple(r["opponents"]), r["k"])
                if key in keys_seen:
                    dup += 1
                keys_seen.add(key)
                opp_algo = r["opponents"][0].split(":")[0] if cond in ("C1", "C4") else None
                cells[(cond, r["learner"], r["opponents"][0] if cond in ("C1", "C4") else ("S4" if cond == "S4" else "A"))] += 1
                if "findings" in r:
                    for f in r["findings"]:
                        findings_raw.append((cond, r["seed"], r["learner"], r["opponents"][0], f["seat"], f["type"], f["key"], f["screen"]["pass"]))
                for t, n in r["incompleteByType"].items():
                    incomplete[f"{cond}:{t}"] += n
                raw_ep[(cond, r["job"])] = r["summary"]["xi"]
                if cond == "A":
                    a_xi[(r["learner"], r["k"])] = r["summary"]["xi"]
                else:
                    transfer[(cond, r["learner"].split(":")[0], opp_algo)].append((r["learner"], r["k"], r["summary"]["xi"]))
    res["episodeCounts"] = dict(counts)
    res["expectedCounts"] = {"A": 7500, "C1": 90000, "C4": 90000, "S4": 7500}
    res["countsMatch"] = dict(counts) == res["expectedCounts"]
    per_cell = Counter(cells.values())
    res["cells"] = {"A": sum(1 for c in cells if c[0] == "A"), "C1": sum(1 for c in cells if c[0] == "C1"), "C4": sum(1 for c in cells if c[0] == "C4"), "S4": sum(1 for c in cells if c[0] == "S4"),
                    "episodesPerCellDistribution": dict(per_cell)}
    res["duplicateEpisodes"] = dup
    res["missingFields"] = dict(missing_fields)
    res["nullValues"] = {**dict(nulls), "note": "priceToFair / capToFair are null only when the learner bought nothing / never bid (defined as undefined, excluded from means)"}
    res["incompleteXIByType"] = dict(incomplete)
    # ── findings ──────────────────────────────────────────────────────
    fj = json.loads((REP / "findings.json").read_text(encoding="utf-8"))
    pub = sorted((f["cond"], f["seed"], f["learner"], f["opponents"][0], f["seat"], f["type"], f["key"], f["screenPass"]) for f in fj["findings"])
    res["findings"] = {"raw": len(findings_raw), "published": len(pub), "identical": sorted(findings_raw) == pub, "allScreenPassed": all(x[-1] for x in findings_raw),
                       "allC4": all(x[0] == "C4" for x in findings_raw)}
    # ── Stage-A control ──────────────────────────────────────────────
    sa = subprocess.run([str(ROOT / "ml/.venv/Scripts/python"), str(ROOT / "ml/ipl_rl/crossplay/check_stage_a.py"), str(RUN / "A.jsonl")], capture_output=True, text=True, encoding="utf-8")
    res["stageAControl"] = {"exit": sa.returncode, "lines": [l for l in sa.stdout.splitlines() if l and not l.startswith("{")][-7:]}
    # ── raw vs published matrix ──────────────────────────────────────
    mm = json.loads((REP / "matchup-matrix.json").read_text(encoding="utf-8"))
    worst = 0.0
    for (cond, a, b), rows in transfer.items():
        d = np.mean([x - a_xi[(l, k)] for l, k, x in rows])
        pub_v = mm[cond][f"{a}>{b}"]["transfer"]["mean"] if cond in ("C1", "C4") else mm["S4"][a]["transfer"]["mean"]
        worst = max(worst, float(abs(d - pub_v)))
    res["rawVsMatchupMatrix"] = {"cellsChecked": len(transfer), "maxAbsDifference": worst, "match": worst < 1e-9}
    ep = json.loads((REP / "episode-results.json").read_text(encoding="utf-8"))
    ci = ep["columns"].index("xi")
    jobi = ep["columns"].index("job")
    mism, n = 0, 0
    for cond, rows in ep["conditions"].items():
        for row in rows:
            n += 1
            if abs(raw_ep[(cond, row[jobi])] - row[ci]) > 1e-6:
                mism += 1
    res["rawVsEpisodeResults"] = {"rows": n, "xiMismatches": mism, "match": mism == 0 and n == sum(counts.values())}
    se = json.loads((REP / "safety-episodes.json").read_text(encoding="utf-8"))
    res["rawVsSafetyEpisodes"] = {c: {"published": se[c]["optionAFindings"], "rawLearnerIncomplete": incomplete.get(f"{c}:learner", 0), "rawOpponentIncomplete": incomplete.get(f"{c}:rlSnapshot", 0)} for c in ("A", "C1", "C4", "S4")}
    # ── frozen system ────────────────────────────────────────────────
    h = subprocess.run(["node", str(ROOT / "ml/ipl_rl/crossplay/hashes.mjs"), "--check", str(REP / "frozen-hashes.json")], capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    res["frozenHashes"] = {"exit": h.returncode, "output": h.stdout.strip()}
    g = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT)
    res["gitStatus"] = g.stdout.strip().splitlines()
    ok = (res["countsMatch"] and not dup and not missing_fields and res["findings"]["identical"] and res["findings"]["raw"] == 20 and sa.returncode == 0
          and res["rawVsMatchupMatrix"]["match"] and res["rawVsEpisodeResults"]["match"] and h.returncode == 0 and not res["expectedReportFilesMissing"])
    res["pass"] = ok
    Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "files"}, indent=1)[:6000])
    print(f"DATA INTEGRITY: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main(sys.argv[1])
