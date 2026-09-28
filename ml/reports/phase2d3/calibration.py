"""Phase 2D.3 §20 — are the learned return distributions meaningful? (evaluation only)

For a checkpoint, play the first K validation-manifest episodes greedily
(ε = 0, masked mean-quantile argmax — the evaluation policy) in the real
JavaScript environment and record, at every learner decision, the predicted
32 quantiles of the chosen action Z(s,a) and the realised return-to-go
R_t = Σ_{k≥t} r_k (γ = 1). A calibrated distribution has P(R ≤ θ_i) ≈ τ_i:
report that coverage per τ, the mean absolute calibration error, the bias
mean(Q − R), and the spread of R vs the predicted spread. No training.

    .venv/Scripts/python runs/_2d3/calibration.py <out.json> <checkpoint.pt> [...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ipl_rl.algos.qr_dqn import TAUS, make_net, masked_argmax  # noqa: E402
from ipl_rl.bridge import REPO_ROOT  # noqa: E402
from ipl_rl.env import IplAuctionEnv  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
K = 100
entries = json.loads((REPO_ROOT / "packages" / "shared" / "data" / "rl-manifests" / "validation.json").read_text())["entries"][:K]
taus = TAUS.numpy()
out = {}
env = IplAuctionEnv(split="validation")
try:
    for ck in sys.argv[2:]:
        net = make_net()
        net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["modules"]["online"])
        net.eval()
        Z, R = [], []
        for e in entries:
            o, _ = env.reset(options={"entry": e})
            zs, rs = [], []
            while True:
                m = env.action_masks()
                with torch.no_grad():
                    z = net(torch.from_numpy(o).unsqueeze(0))[0]
                a = int(masked_argmax(z.mean(-1), torch.from_numpy(m)))
                zs.append(z[a].numpy().astype(np.float64))
                o, r, done, _, _ = env.step(a)
                rs.append(r)
                if done:
                    break
            rtg = np.cumsum(rs[::-1])[::-1]
            Z.extend(zs)
            R.extend(rtg.tolist())
        Z, R = np.asarray(Z), np.asarray(R)
        cover = (R[:, None] <= Z).mean(0)                     # P(R ≤ θ_i) per τ_i
        q = Z.mean(1)
        lvl = lambda ix: float(np.mean([cover[i] for i in ix]))
        res = {
            "decisions": int(len(R)), "episodes": K,
            "coverage": cover.tolist(),
            "calibrationMAE": float(np.abs(cover - taus).mean()),
            "coverageAt": {"0.016": float(cover[0]), "0.25": lvl((7, 8)), "0.5": lvl((15, 16)), "0.75": lvl((23, 24)), "0.984": float(cover[31])},
            "bias_Q_minus_R": float((q - R).mean()), "mae_Q_vs_R": float(np.abs(q - R).mean()),
            "corr_Q_R": float(np.corrcoef(q, R)[0, 1]),
            "predictedIQR": float((Z[:, 23:25].mean(1) - Z[:, 7:9].mean(1)).mean()),
            "predictedSpread": float((Z[:, -1] - Z[:, 0]).mean()),
            "residualIQR": float(np.subtract(*np.percentile(R - q, [75, 25]))),
            "crossingRate": float((np.diff(Z, axis=1) < 0).mean()),
            "baselineCoveragePointMass": "a point estimate (no distribution) would give coverage 0 or 1 per state",
        }
        out[ck] = res
        print(f"{ck}: {res['decisions']} decisions | calibration MAE {res['calibrationMAE']:.3f} | coverage at τ .016/.25/.5/.75/.984 = "
              + "/".join(f"{v:.3f}" for v in res["coverageAt"].values())
              + f" | bias Q−R {res['bias_Q_minus_R']:+.4f} | corr(Q,R) {res['corr_Q_R']:.3f} | predicted IQR {res['predictedIQR']:.4f} vs residual IQR {res['residualIQR']:.4f} "
              f"| crossing {res['crossingRate']:.4f}", flush=True)
finally:
    env.close()
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
