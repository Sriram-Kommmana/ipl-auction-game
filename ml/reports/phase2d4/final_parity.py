"""Phase 2D.4 §22 — JS/Python parity of the final OpenAI-ES exports on real
states of every required kind, each state's kind taken from an independent
JavaScript replay (ml/ipl_rl/tests/replay_probe.mjs):
normal · low purse · high purse · main auction · re-auction · keeper needed ·
Indian needed · bowling needed · shield-forced · final-path; each under the
real mask, PASS only, one single legal action and all 20 legal. Also checks
that the production decision (masked argmax of the JS logits under the real
mask) is always a legal act-v3 action.

    .venv/Scripts/python runs/_2d4/final_parity.py <policy.json> ...
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ML = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ML))
from ipl_rl.algos.openai_es import parity  # noqa: E402
from ipl_rl.bridge import BridgeV2  # noqa: E402
from ipl_rl.nets import PolicyNet  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
PROBE = ML / "ipl_rl" / "tests" / "replay_probe.mjs"
FIXTURE = ML / "ipl_rl" / "tests" / "fixtures" / "final_path_episode.json"


def probe(entry, actions):
    out = subprocess.run(["node", str(PROBE)], input=json.dumps({"entry": entry, "actions": actions}), capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(out.stdout)


def scripted(seed, style):
    with BridgeV2() as b:
        r = b.call("reset", seed=seed)
        entry, actions, t = r["info"]["entry"], [], 0
        while not r.get("done"):
            legal = [a for a, ok in enumerate(r["mask"]) if ok]
            a = legal[0] if style == "passive" else legal[-1] if style == "aggressive" else legal[(5 * t) % len(legal)]
            actions.append(a)
            r = b.call("step", action=a)
            t += 1
    return entry, actions


episodes = [scripted(1_000_401, "passive"), scripted(1_000_402, "aggressive"), scripted(1_000_403, "mixed")]
fx = json.loads(FIXTURE.read_text())
episodes.append((fx["entry"], fx["actions"]))
obs, masks, kinds = [], [], []
for entry, actions in episodes:
    js = probe(entry, actions)
    for o, m, st in zip(js["obs"], js["masks"], js["states"]):
        obs.append(o)
        masks.append(m)
        kinds.append(st)
obs, masks = np.asarray(obs, np.float32), np.asarray(masks, bool)
cat = {
    "normal (main, no forced)": [k["phase"] == "main" and not k["forced"] for k in kinds],
    "low purse ≤ ₹200L": [k["purse"] <= 200 for k in kinds],
    "high purse ≥ ₹9,000L": [k["purse"] >= 9000 for k in kinds],
    "main auction": [k["phase"] == "main" for k in kinds],
    "re-auction": [k["phase"] == "reauction" for k in kinds],
    "keeper needed": [k["need"]["keeper"] > 0 for k in kinds],
    "Indian needed": [k["need"]["indians"] > 0 for k in kinds],
    "bowling needed": [k["need"]["bowling"] > 0 for k in kinds],
    "shield-forced": [k["forced"] for k in kinds],
    "final-path": [k["finalPath"] for k in kinds],
}
print(f"{len(obs)} real states from {len(episodes)} JavaScript episodes (incl. the final-path fixture): " + ", ".join(f"{k} {sum(v)}" for k, v in cat.items()))
worst, ok = 0.0, True
for path in sys.argv[1:]:
    pol = json.loads(Path(path).read_text())
    net = PolicyNet("es")
    with torch.no_grad():
        k = 0
        for layer, lin in zip(pol["layers"], [net.body[0], net.body[2], net.head]):
            lin.weight.copy_(torch.tensor(layer["weight"], dtype=torch.float32))
            lin.bias.copy_(torch.tensor(layer["bias"], dtype=torch.float32))
    line = f"{path}:"
    for name, sel in cat.items():
        sel = np.asarray(sel)
        if not sel.any():
            line += f"\n   {name}: 0 states"
            continue
        p = parity(net, pol, obs[sel], masks[sel])
        worst = max(worst, p["maxAbsScoreDiff"])
        ok &= p["argmaxAgreement"] == 1.0 and p["maxAbsScoreDiff"] <= 1e-9
        line += f"\n   {name:24s} n={p['states']:4d} max |Δlogit| {p['maxAbsScoreDiff']:.1e} agreement {p['byMask']}"
    with torch.no_grad():
        logits = net.double()(torch.from_numpy(obs).double()).numpy()
    choice = np.where(masks, logits, -np.inf).argmax(1)
    legal = bool(masks[np.arange(len(masks)), choice].all())
    ok &= legal
    line += f"\n   production decisions legal under act-v3: {int(masks[np.arange(len(masks)), choice].sum())}/{len(masks)}"
    print(line)
print(f"FINAL EXPORT PARITY: {'PASS' if ok else 'FAIL'} (max |Δlogit| {worst:.1e})")
sys.exit(0 if ok else 1)
