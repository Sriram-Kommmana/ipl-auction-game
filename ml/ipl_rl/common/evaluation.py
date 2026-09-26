"""Evaluation hooks — everything is judged by the Node evaluator.

node_evaluate()  runs packages/shared/bin/rl-evaluate.js on an exported
                 rl-policy-v2: the validation manifest, paired against the
                 LOCKED baselines (data/rl-baselines/<split>.episodes.json),
                 with the per-episode safety invariants.
safety_problems() every Step-7 invariant over an evaluation's episodes.
export_parity()  PyTorch actor vs production JavaScript inference on the
                 same observations (scores and masked-argmax actions).
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from ..bridge import NODE_BIN, NODE_FLAGS, REPO_ROOT, run_node_script
from .win_qos import disable_throttling

BASELINE_DIR = REPO_ROOT / "packages" / "shared" / "data" / "rl-baselines"


def locked_baselines(split="validation"):
    path = BASELINE_DIR / f"{split}.episodes.json"
    return path if path.exists() else None


def node_evaluate(policy_path, out_dir, split="validation", limit=None, workers=12, compare=True, node="node"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path, episodes_path = out_dir / "report.json", out_dir / "episodes.json"
    cmd = [node, *NODE_FLAGS, str(NODE_BIN / "rl-evaluate.js"), "--split", split, "--policy", str(policy_path), "--baselines", "none",
           "--workers", str(workers), "--out", str(report_path), "--episodes-out", str(episodes_path), "--quiet"]
    if limit:
        cmd += ["--limit", str(int(limit))]
    baseline_file = locked_baselines(split) if compare else None
    if baseline_file:
        cmd += ["--compare", str(baseline_file), "--reference", "moneyball"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    disable_throttling(proc._handle)  # speed only (Windows EcoQoS)
    _, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"Node evaluation failed (exit {proc.returncode}):\n{stderr[-4000:]}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    episodes = json.loads(episodes_path.read_text(encoding="utf-8"))
    return report, episodes


def safety_problems(report, episodes):
    """Every Step-7 invariant. Returns a list of problems (empty = safe).

    Covered: illegal / masked actions (the JavaScript step() throws, so the
    run would have failed), purse, squad size, overseas, duplicate purchases,
    re-auction integrity, the learner's cap vs its plan, Best XI legality,
    lot budget (no deadlock) — all from auditAuction in invariants.js — and
    completion of every scheduled episode (no crashes).
    """
    problems = []
    if report.get("invariantViolations", 0):
        problems.append(f"{report['invariantViolations']} invariant violations")
    for name, rows in episodes["episodes"].items():
        if len(rows) != report["n"]:
            problems.append(f"{name}: {len(rows)} of {report['n']} episodes completed")
        for r in rows:
            if r["invariantViolations"]:
                problems.append(f"{name} seed {r['seed']}: {r['violations']}")
            if not r["legalXI"]:
                problems.append(f"{name} seed {r['seed']}: final XI not legal ({r['emptySlots']} empty slots)")
            if sum(r["actionCounts"]) != r["decisions"]:
                problems.append(f"{name} seed {r['seed']}: {sum(r['actionCounts'])} actions for {r['decisions']} decisions")
    return problems


def export_parity(net, policy, observations, masks):
    """Max |PyTorch − JavaScript| action score and masked-argmax agreement."""
    obs = np.asarray(observations, dtype=np.float32)
    js = run_node_script("rl-policy-scores.js", {"policy": policy, "observations": obs.tolist()})
    if not js["ok"]:
        raise RuntimeError(f"production loader rejected the export: {js['error']}")
    with torch.no_grad():
        py = net.action_scores(torch.from_numpy(obs)).double().numpy()
    jsn = np.asarray(js["scores"], dtype=np.float64)
    m = np.asarray(masks, dtype=bool)
    py_arg = np.where(m, py, -np.inf).argmax(axis=1)
    js_arg = np.where(m, jsn, -np.inf).argmax(axis=1)
    return {"states": int(len(obs)), "maxAbsScoreDiff": float(np.abs(py - jsn).max()), "argmaxAgreement": float((py_arg == js_arg).mean())}


def headline(report, name):
    """The metrics printed / logged at every evaluation."""
    r = report["report"][name]
    keys = ("xi", "legalXI", "strongXI", "rank", "purseLeftShare", "buys", "priceToFair", "capToFair", "bidRate", "reauctionBuys", "return", "xiGainPer1000")
    out = {k: r[k]["mean"] for k in keys}
    if r.get("paired"):
        out["xiDiffVsMoneyball"] = r["paired"]["xiDiffMean"]
    for s, sub in r["strata"].items():
        out[f"xi_{s}"] = sub["xi"]["mean"]
    return out
