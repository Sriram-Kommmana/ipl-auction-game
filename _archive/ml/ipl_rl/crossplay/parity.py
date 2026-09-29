"""Phase 2E.0 §14 — production parity of all 15 frozen exports on real Stage-B
states.

Chain checked, per export:
  training checkpoint (Python, float64)  →  the exported rl-policy-v2 JSON
  →  production loader + JavaScript inference (packages/shared/bin/rl-policy-scores.js)
  →  same observation, same act-v3 mask  →  same chosen action.

  1. export fidelity: the JSON layers equal the checkpoint's export layers
     (PPO/A2C/QR-DQN/ES: exact float32 values; D3QN: the float64 fold).
  2. scores: Python float64 forward of the TRAINING network vs production JS
     (logits / Q / mean of 32 quantiles; QR-DQN raw quantiles too).
  3. decisions: masked argmax agreement under the real mask, PASS-only, a
     single legal action and all 20 legal; for D3QN/QR-DQN/ES the production
     runtime's recorded choice must equal the Python masked argmax on every
     state; for PPO/A2C (temperature 0.3 sampling) the masked argmax and the
     sampling distribution must agree.
  4. identifiers: algorithm, head, selection, obs/act spec hashes.
States: the Stage-B dump (dump_states.mjs) plus the Stage-A final-path fixture
(replayed by the JavaScript probe), tagged by kind.

    python parity.py <states.json> <out.json>
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ML = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ML))
from ipl_rl.algos.d3qn import DuelingQNet  # noqa: E402
from ipl_rl.bridge import run_node_script  # noqa: E402
from ipl_rl.nets import PolicyNet  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
PASS = 0
DIRS = {"ppo": "ppo-2c3", "a2c": "a2c-2d1", "d3qn": "d3qn-2d2", "qrdqn": "qr-dqn-2d3", "es": "openai-es-2d4"}
CKPT = {"ppo": "update_0325", "a2c": "update_0325", "d3qn": "update_0325", "qrdqn": "update_0325", "es": "gen_2000"}
SELECTION = {"ppo": {"mode": "sample", "temperature": 0.3}, "a2c": {"mode": "sample", "temperature": 0.3}, "d3qn": {"mode": "argmax"}, "qrdqn": {"mode": "argmax"}, "es": {"mode": "argmax"}}
HEAD = {"ppo": "logits", "a2c": "logits", "d3qn": "q", "qrdqn": "quantiles", "es": "logits"}
PROBE = ML / "ipl_rl" / "tests" / "replay_probe.mjs"
FIXTURE = ML / "ipl_rl" / "tests" / "fixtures" / "final_path_episode.json"


def training_net(algo, ckpt):
    mods = torch.load(ckpt, map_location="cpu", weights_only=False)["modules"]
    if algo in ("ppo", "a2c"):
        net = PolicyNet(algo)
        net.load_state_dict({k[len("actor."):]: v for k, v in mods["agent"].items() if k.startswith("actor.")})
    elif algo == "d3qn":
        net = DuelingQNet()
        net.load_state_dict(mods["online"])
    elif algo == "qrdqn":
        net = PolicyNet("qrdqn")
        net.load_state_dict(mods["online"])
    else:
        net = PolicyNet("es")
        net.load_state_dict(mods["policy"])
    return net.double().eval()


def masked_argmax(scores, mask):
    return np.where(mask, scores, -np.inf).argmax(1)


def probs(scores, mask, t=0.3):
    z = np.where(mask, scores / t, -np.inf)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fixture_states():
    fx = json.loads(FIXTURE.read_text())
    out = subprocess.run(["node", str(PROBE)], input=json.dumps({"entry": fx["entry"], "actions": fx["actions"]}), capture_output=True, text=True, encoding="utf-8", check=True)
    js = json.loads(out.stdout)
    return [{"obs": o, "mask": [int(x) for x in m], "tags": {"phase": st["phase"], "purse": st["purse"], "need": st["need"], "forced": st["forced"], "finalPath": st["finalPath"]}, "who": "fixture"} for o, m, st in zip(js["obs"], js["masks"], js["states"])]


CATS = {
    "normal": lambda t: t["phase"] == "main" and not t["forced"],
    "main": lambda t: t["phase"] == "main",
    "reauction": lambda t: t["phase"] == "reauction",
    "lowPurse≤200": lambda t: t["purse"] <= 200,
    "highPurse≥9000": lambda t: t["purse"] >= 9000,
    "keeperNeeded": lambda t: t["need"]["keeper"] > 0,
    "indianNeeded": lambda t: t["need"]["indians"] > 0,
    "bowlingNeeded": lambda t: t["need"]["bowling"] > 0,
    "forced": lambda t: t["forced"],
    "finalPath": lambda t: t["finalPath"],
}


def main(states_path, out_path):
    dump = json.loads(Path(states_path).read_text())
    fixture = fixture_states()
    results, ok = {}, True
    for key, states in dump.items():
        algo, s = key.split(":")
        d = ML / "runs" / f"{DIRS[algo]}-{s}" / "checkpoints" / CKPT[algo]
        pol = json.loads((d / "policy.json").read_text())
        net = training_net(algo, d / "checkpoint.pt")
        # 1. export fidelity
        layers = net.export_layers()
        fid = max(max(np.abs(np.asarray(L["weight"]) - w.double().numpy()).max(), np.abs(np.asarray(L["bias"]) - b.double().numpy()).max()) for L, (w, b) in zip(pol["layers"], layers))
        fid_ok = len(layers) == len(pol["layers"]) and (fid == 0.0 if algo != "d3qn" else fid <= 1e-12)
        # 4. identifiers
        ident = {"algorithm": pol["algorithm"] == algo, "head": pol["architecture"]["head"] == HEAD[algo], "selection": pol["selection"] == SELECTION[algo],
                 "obsHash": pol["obsSpec"]["hash"] == "629b25783f833af7", "actHash": pol["actSpec"]["hash"] == "5f72f510c48b1f46"}
        allst = states + fixture
        obs = np.asarray([x["obs"] for x in allst], np.float64)
        masks = np.asarray([x["mask"] for x in allst], bool)
        js = run_node_script("rl-policy-scores.js", {"policy": pol, "observations": obs.tolist()})
        if not js["ok"]:
            raise RuntimeError(f"{key}: production loader rejected the export: {js['error']}")
        jsn = np.asarray(js["scores"], np.float64)
        # the dumped scores are what the production path computed during play
        played = np.asarray([x["scores"] for x in states], np.float64)
        with torch.no_grad():
            t = torch.from_numpy(obs)
            if algo == "qrdqn":
                raw = net(t).numpy()
                py = raw.mean(-1)
                raw_diff = float(np.abs(raw.reshape(len(obs), -1) - np.asarray(js["raw"], np.float64)).max())
            else:
                py = net(t).numpy()
                raw_diff = None
        score_diff = float(np.abs(py - jsn).max())
        played_diff = float(np.abs(played - jsn[: len(states)]).max()) if states else 0.0
        single = np.zeros_like(masks)
        single[np.arange(len(masks)), [np.flatnonzero(m)[-1] for m in masks]] = True
        pass_only = np.zeros_like(masks)
        pass_only[:, PASS] = True
        variants = {"real": masks, "passOnly": pass_only, "single": single, "allLegal": np.ones_like(masks)}
        agree = {k: float((masked_argmax(py, m) == masked_argmax(jsn, m)).mean()) for k, m in variants.items()}
        # 3. the production runtime's recorded choices
        chosen = np.asarray([x["action"] for x in states])
        legal = bool(masks[np.arange(len(states)), chosen].all()) if states else True
        if SELECTION[algo]["mode"] == "argmax":
            decision_agree = float((masked_argmax(py[: len(states)], masks[: len(states)]) == chosen).mean())
            prob_diff = None
        else:
            decision_agree = None
            prob_diff = float(np.abs(probs(py, masks) - probs(jsn, masks)).max())
        cats = {name: int(sum(fn(x["tags"]) for x in allst)) for name, fn in CATS.items()}
        per_cat = {}
        for name, fn in CATS.items():
            sel = np.asarray([fn(x["tags"]) for x in allst])
            if sel.any():
                per_cat[name] = {"n": int(sel.sum()), "maxAbsScoreDiff": float(np.abs(py[sel] - jsn[sel]).max()), "argmaxAgreement": float((masked_argmax(py[sel], masks[sel]) == masked_argmax(jsn[sel], masks[sel])).mean())}
        passed = (fid_ok and all(ident.values()) and score_diff <= 1e-9 and played_diff == 0.0 and min(agree.values()) == 1.0 and legal
                  and (decision_agree in (None, 1.0)) and (prob_diff is None or prob_diff <= 1e-9) and (raw_diff is None or raw_diff <= 1e-9))
        ok &= passed
        results[key] = {
            "states": len(allst), "stageBStates": len(states), "fixtureStates": len(fixture),
            "learnerStates": sum(x["who"] == "learner" for x in states), "opponentStates": sum(x["who"] == "opponent" for x in states),
            "exportFidelityMaxAbs": fid, "identifiers": ident, "maxAbsScoreDiff": score_diff, "maxAbsRawQuantileDiff": raw_diff,
            "playedScoresVsLoaderMaxAbs": played_diff, "argmaxAgreementByMask": agree, "recordedChoiceAgreement": decision_agree,
            "maxAbsProbDiffT03": prob_diff, "recordedChoicesLegal": legal, "categories": cats, "byCategory": per_cat, "pass": passed,
        }
        print(f"{key:9s} {'PASS' if passed else 'FAIL'} states {len(allst)} (opp {results[key]['opponentStates']}, learner {results[key]['learnerStates']}, fixture {len(fixture)}) "
              f"fidelity {fid:.1e} |Δscore| {score_diff:.1e} raw {raw_diff if raw_diff is None else f'{raw_diff:.1e}'} argmax {min(agree.values()):.3f} "
              f"choice {decision_agree} |Δp| {prob_diff if prob_diff is None else f'{prob_diff:.1e}'} cats {cats}")
    Path(out_path).write_text(json.dumps({"pass": ok, "exports": results}, indent=1))
    print(f"PRODUCTION PARITY: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1], sys.argv[2]) else 1)
