"""Phase 2D.2 §25 — export parity of the three FINAL D3QN exports on real
re-auction and extreme-purse states (plus all four mask kinds)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ipl_rl.algos.d3qn import DuelingQNet, parity  # noqa: E402
from ipl_rl.bridge import BridgeV2  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
RUNS = Path(__file__).resolve().parents[1]


def episode_states(seed, passive):
    """One whole episode; passive = PASS whenever allowed (long episodes reaching the re-auction)."""
    with BridgeV2() as b:
        r = b.call("reset", seed=seed)
        obs, masks, t = [], [], 0
        while not r.get("done"):
            obs.append(r["obs"])
            masks.append(r["mask"])
            legal = [a for a, ok in enumerate(r["mask"]) if ok]
            r = b.call("step", action=legal[0] if passive else legal[(7 * t) % len(legal)])
            t += 1
    return np.asarray(obs, np.float32), np.asarray(masks, bool)


parts = [episode_states(s, True) for s in (1_000_401, 1_000_402)] + [episode_states(s, False) for s in (1_000_403, 1_000_404)]
obs = np.concatenate([p[0] for p in parts])
masks = np.concatenate([p[1] for p in parts])
reauc = obs[:, 0] > 0.5
extreme = obs[:40].copy()
extreme[:20, 15] = 0.0     # self_purse at the floor
extreme[20:, 15] = 1.5     # beyond any real purse
obs_all = np.concatenate([obs, extreme])
masks_all = np.concatenate([masks, masks[:40]])
print(f"states: {len(obs_all)} (real {len(obs)}, of which re-auction {int(reauc.sum())}; lowest real purse {obs[:, 15].min():.4f}; 40 synthetic extreme-purse)")
worst = 0.0
for s in (1, 2, 3):
    ck = RUNS / f"d3qn-2d2-s{s}" / "checkpoints" / "update_0325"
    net = DuelingQNet()
    net.load_state_dict(torch.load(ck / "checkpoint.pt", map_location="cpu", weights_only=False)["modules"]["online"])
    pol = json.loads((ck / "policy.json").read_text())
    p_all = parity(net, pol, obs_all, masks_all)
    p_re = parity(net, pol, obs[reauc], masks[reauc])
    worst = max(worst, p_all["maxAbsScoreDiff"], p_re["maxAbsScoreDiff"])
    print(f"seed {s} u325: all states max |Δq| {p_all['maxAbsScoreDiff']:.1e} agreement {p_all['byMask']} | re-auction only ({p_re['states']}) max |Δq| {p_re['maxAbsScoreDiff']:.1e} agreement {p_re['byMask']}")
    assert p_all["argmaxAgreement"] == 1.0 and p_re["argmaxAgreement"] == 1.0
print(f"FINAL EXPORT PARITY: PASS (max |Δq| {worst:.1e})")
