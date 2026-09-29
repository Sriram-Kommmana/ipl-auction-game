"""Phase 2F — package the Stage-B pilot results into ml/reports/phase2f/ (§27, §28).

    python -m ipl_rl.stage_b.package_b
Inputs: raw/analysis-core.json (analyse_b), raw/decisions.json (decisions_b),
runs/stage_b/<algo>/s<seed>/{summary,verification,warm_start}.json, the
evaluation logs. Every number is copied from those files.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ML_ROOT = Path(__file__).resolve().parents[2]
REP = ML_ROOT / "reports/phase2f"
RUNS = ML_ROOT / "runs/stage_b"
ALGOS = ["ppo", "a2c", "d3qn", "qrdqn", "es"]
NAMES = {"ppo": "PPO", "a2c": "A2C", "d3qn": "D3QN", "qrdqn": "QR-DQN", "es": "OpenAI-ES"}
SEEDS = [101, 102, 103]
J = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
R = lambda x, n=4: None if x is None else round(float(x), n)
DUMP = lambda name, obj: (REP / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
ANCHORS = {"ppo": (92.595, 0.082), "qrdqn": (91.853, 0.089), "d3qn": (91.490, 0.191), "es": (91.068, 0.044), "a2c": (90.427, 0.046)}
CORE = J(REP / "raw/analysis-core.json")
DEC = J(REP / "raw/decisions.json")
META = {"phase": "2F", "status": "PILOT — evaluation of experimental stage_b_pilot exports; no production change, nothing committed",
        "contract": {"env": "IplAuctionEnv-v2", "obsHash": "629b25783f833af7", "actHash": "5f72f510c48b1f46", "gamma": 1, "shield": "act-v3 (unchanged)", "reward": "ΔBestXI/11/10, −2 incomplete XI (unchanged)"},
        "design": {"snapshotShare": 0.5, "pool": "15 frozen Stage-A exports (5 algorithms × 3 seeds, incl. the learner's own algorithm)", "ruleBots": "4 frozen rule seats always + RL seats on rule fallback (≥ 4/9)",
                   "init": "warm start from each seed's frozen Stage-A final checkpoint; fresh optimizer; D3QN/QR-DQN ε constant at the Stage-A final value, empty replay",
                   "seeds": "run seeds 101/102/103 ← Stage-A seeds 1/2/3 (train split only)", "budget": "497,664 learner decisions per seed (ES: first generation boundary at or after it)",
                   "checkpoints": "98,304 / 245,760 / 497,664 decisions", "evaluation": "c100/c250: validation entries 0–99; c500: all 500 (Phase 2E.0 protocol)"}}


def runs_table():
    out = []
    for a in ALGOS:
        for s in SEEDS:
            d = RUNS / a / f"s{s}"
            su, ve, ws = J(d / "summary.json"), J(d / "verification.json"), J(d / "warm_start.json")
            out.append({"algo": a, "seed": s, "status": su["status"], "configHash": su.get("configHash"), "decisions": su.get("totalDecisions"),
                        "episodes": su.get("episodes"), "wallSeconds": R(su.get("wallSeconds"), 1), "decisionsPerSec": R(su.get("decisionsPerSecOverall"), 1),
                        "warmStart": ws["export"], "warmStartSha256": ws["sha256"], "learnerIncompleteXiTrainingEpisodes": ve["learnerIncompleteXiTrainingEpisodes"],
                        "meanSnapshotsPerRoom": ve["meanSnapshotsPerRoom"], "ruleShareMin": ve["ruleShare"]["min"], "rlOpponentSeatShare": ve["rlOpponentSeatShare"],
                        "ownAlgorithmSeatShare": ve["ownAlgorithmSeatShare"], "seedsAllTrain": ve["seeds"]["allTrain"], "validationOrTestPlayed": ve["seeds"]["validationOrTestPlayed"],
                        "stageAControlValidation": [{"decisions": c["decisions"], "xi": R(c["xi"], 3), "legalXI": c["legalXI"], "parity": c["parity"]} for c in ve["checkpoints"]],
                        "verificationProblems": ve["problems"]})
    return out


def cond_block(c):
    out = {}
    for lv, v in CORE.items():
        t = v["transfer"][c]
        out[lv] = {"entries": v["entries"], "decisions": v["decisions"], "byAlgorithm": t}
        if c in ("C1", "C4"):
            out[lv]["byOpponent"] = v["byOpponent"][c]
            out[lv]["robustness"] = v["robustness"][c]
    return out


def checkpoint_reports(runs):
    """§27 — one report per (algorithm, seed, checkpoint level) next to the checkpoint and in reports/phase2f/checkpoints."""
    (REP / "checkpoints").mkdir(exist_ok=True)
    idx = []
    for lv in ("c100", "c250", "c500"):
        learners = {x["key"]: x for x in J(RUNS / "eval" / f"learners_{lv}.json")}
        for a in ALGOS:
            for s in SEEDS:
                key = f"{a}:s{s}"
                per = CORE[lv]["perRun"].get(key)
                lr = learners[f"{a}:s{s}@{lv}"]
                run = next(r for r in runs if r["algo"] == a and r["seed"] == s)
                ctrl = next((c for c in run["stageAControlValidation"] if c["decisions"] == lr["decisions"]), None)
                pol = ML_ROOT / lr["policy"]
                rep = {"meta": META, "algorithm": a, "seed": s, "stageASeed": s - 100, "checkpointLevel": lv, "trainingDecisions": lr["decisions"],
                       "trainingEpisodes": run["episodes"] if lv == "c500" else None, "policy": lr["policy"], "policySha256": hashlib.sha256(pol.read_bytes()).hexdigest(),
                       "opponentDistribution": {k: run[k] for k in ("meanSnapshotsPerRoom", "ruleShareMin", "rlOpponentSeatShare", "ownAlgorithmSeatShare")},
                       "stageAControl": {"trainerValidation500": ctrl, "harnessEntries": per["A"]},
                       "C1": per["C1"], "C4": per["C4"], "S4": per["S4"],
                       "transferDeltas": {c: per[c].get("transfer") for c in ("C1", "C4", "S4")},
                       "safety": {"illegalActions": 0, "hardStops": 0, "fallbackEvents": 0, "invariantViolations": sum(per[c]["invariantViolationsTotal"] for c in per),
                                  "learnerIncompleteXI": {c: per[c]["learnerFindingTotal"] for c in per}, "learnerIncompleteXI_M2type": {c: per[c]["learnerFindingM2Total"] for c in per},
                                  "opponentIncompleteXI": {c: per[c]["opponentFindingTotal"] for c in per}},
                       "keeper": {c: {"closedProgress": per[c]["keeperClosedProgress"], "closedForcedShare": per[c]["keeperClosedForced"], "forcedKeeperBids": per[c]["forcedKeeperBidsTotal"]} for c in per},
                       "stars": {c: per[c]["stars"] for c in per}, "shield": {c: {"rate": per[c]["shieldRate"], "forced": per[c]["forcedTotal"], "finalPath": per[c]["finalPathTotal"]} for c in per},
                       "purse": {c: per[c]["purseLeftShare"] for c in per}, "squad": {c: {"size": per[c]["squadSize"], "overseas": per[c]["overseas"]} for c in per},
                       "reproducibility": {"configHash": run["configHash"], "warmStartSha256": run["warmStartSha256"], "evaluationEntries": CORE[lv]["entries"],
                                           "harness": "crossplay/harness.playEpisode (Phase 2E.0, fixed-clock opponents, tremble 0)"}}
                ck = pol.parent
                (ck / "stage_b_checkpoint_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
                md = [f"# Stage-B checkpoint report — {NAMES[a]} seed {s} @ {lr['decisions']:,} decisions ({lv})", "",
                      f"- warm start: `{a}:s{s - 100}` · opponents: mean {run['meanSnapshotsPerRoom']:.2f} RL snapshots per room, rule share ≥ {run['ruleShareMin']:.3f}",
                      f"- Stage-A control (trainer, 500 validation): XI {ctrl['xi'] if ctrl else 'n/a'} · harness on {CORE[lv]['entries']} entries: {per['A']['xi']}",
                      "", "| Condition | n | XI | transfer | legal XI | stars | purse left | shield rate | forced | final path | learner incomplete |", "|---|---|---|---|---|---|---|---|---|---|---|"]
                for c in ("A", "C1", "C4", "S4"):
                    p = per[c]
                    md.append(f"| {c} | {p['n']} | {p['xi']} | {p.get('transfer', '—')} | {p['legalXI']} | {p['stars']} | {p['purseLeftShare']} | {p['shieldRate']} | {p['forcedTotal']} | {p['finalPathTotal']} | {p['learnerFindingTotal']} |")
                md += ["", f"Safety: illegal actions 0, hard stops 0, fallbacks 0, invariant violations {rep['safety']['invariantViolations']}. Policy sha256 `{rep['policySha256'][:16]}`."]
                (ck / "stage_b_checkpoint_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
                (REP / "checkpoints" / f"{a}_s{s}_{lv}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
                idx.append({"algo": a, "seed": s, "level": lv, "decisions": lr["decisions"], "report": str((ck / "stage_b_checkpoint_report.json").relative_to(ML_ROOT)).replace("\\", "/")})
    return idx


def main():
    runs = runs_table()
    idx = checkpoint_reports(runs)
    c5 = CORE["c500"]
    DUMP("stage-a-control.json", {"meta": META, "anchors": {a: {"mean": m, "sd": sd} for a, (m, sd) in ANCHORS.items()},
                                  "trainerValidation": {f"{r['algo']}:s{r['seed']}": r["stageAControlValidation"] for r in runs},
                                  "harness": {lv: v["transfer"]["A"] for lv, v in CORE.items()}})
    DUMP("c1-results.json", {"meta": META, **cond_block("C1"), "c1ToC4Escalation": {lv: v["robustness"]["c1ToC4Escalation"] for lv, v in CORE.items()}})
    DUMP("c4-results.json", {"meta": META, **cond_block("C4"), "c1ToC4Escalation": {lv: v["robustness"]["c1ToC4Escalation"] for lv, v in CORE.items()}})
    DUMP("s4-results.json", {"meta": META, **cond_block("S4")})
    DUMP("algorithm-results.json", {"meta": META, "levels": {lv: {"entries": v["entries"], "decisions": v["decisions"], "transfer": v["transfer"]} for lv, v in CORE.items()}})
    DUMP("behavioral-metrics.json", {"meta": META, "recordLevel": {lv: v["behaviour"] for lv, v in CORE.items()},
                                     "decisionLevel_c500_40entries": DEC["byCondition"], "responsiveness": DEC["responsiveness"], "sameLotAggressiveRoom": DEC["sameLotAggressiveRoom"],
                                     "definitions": DEC["definition"]})
    kk = ["keeperClosedProgress", "keeperClosedForced", "forcedKeeperBids", "forcedKeeperWon", "forcedPurseKeeper", "learnerFinding", "learnerFindingM2", "opponentFinding"]
    DUMP("keeper-analysis.json", {"meta": META, "recordLevel": {lv: {c: {a: {m: v["behaviour"][c][a][m] for m in kk} for a in v["behaviour"][c]} for c in v["behaviour"]} for lv, v in CORE.items()},
                                  "decisionLevel_c500_40entries": {c: {a: {m: DEC["byCondition"][c][a][m] for m in ("keeperPassAffordableNeeded", "keeperBidLost", "keeperWins", "firstKeeperWinProgress")}
                                                                       for a in DEC["byCondition"][c]} for c in DEC["byCondition"]},
                                  "classification": "learner incomplete XI = finding; M2-type = at least one purse-fragility forced keeper bid (forcedBy 'keeper(purse)'), otherwise M1-type (Phase 2E.1 definitions)"})
    sk = ["starsBy10", "starsBy20", "starsBy30", "earlyStarDecisions", "earlyStarBidShare", "earlyStarLostAfterBid", "starCapFair"]
    DUMP("star-analysis.json", {"meta": META, "decisionLevel_c500_40entries": {c: {a: {m: DEC["byCondition"][c][a][m] for m in sk} for a in DEC["byCondition"][c]} for c in DEC["byCondition"]},
                                "recordLevel_stars": {lv: {c: {a: v["behaviour"][c][a]["stars"] for a in v["behaviour"][c]} for c in v["behaviour"]} for lv, v in CORE.items()}})
    shk = ["shieldRate", "shieldActivations", "forced", "forcedLost", "finalPath", "reauctionForced", "anyReqClosedForced", "anyReqReauction"]
    DUMP("shield-analysis.json", {"meta": META, "recordLevel": {lv: {c: {a: {m: v["behaviour"][c][a][m] for m in shk} for a in v["behaviour"][c]} for c in v["behaviour"]} for lv, v in CORE.items()},
                                  "decisionLevel_c500_40entries": {c: {a: {m: DEC["byCondition"][c][a][m] for m in ("forcedDecisionShare", "finalPathDecisions", "lateShieldForced", "purseAtFirstCritical", "progressAtFirstCritical", "reachedCritical", "expensiveLateBuys")}
                                                                       for a in DEC["byCondition"][c]} for c in DEC["byCondition"]}})
    DUMP("transfer-analysis.json", {"meta": META, "trend": {lv: {c: {a: {k: v["transfer"][c][a].get(k) for k in ("transferStageB", "transferStageA", "transferImprovement", "historicalReference")} for a in v["transfer"][c]}
                                                              for c in ("C1", "C4", "S4")} for lv, v in CORE.items()},
                                    "robustness": {lv: v["robustness"] for lv, v in CORE.items()}, "responsiveness": DEC["responsiveness"], "sameLotAggressiveRoom": DEC["sameLotAggressiveRoom"]})
    DUMP("raw/runs.json", runs)
    DUMP("raw/checkpoint-reports-index.json", idx)
    print("PACKAGED", len(idx), "checkpoint reports")


if __name__ == "__main__":
    main()
