"""Phase 2F — Stage-B pilot launcher (one algorithm, one seed).

    cd ml
    .venv/Scripts/python -m ipl_rl.stage_b.train_b --algo ppo --seed 101

Builds the run configuration from the algorithm's UNCHANGED Phase 2C/2D
config file and overrides only the opponent distribution, the seed, the
budget / checkpoint schedule and the output location (plus, for D3QN / QR-DQN,
exploration continuing at its final ε — the approved warm-start rule):

  stage "B", snapshot_share 0.5, pool = the 15 frozen Stage-A exports
  (decision: all five algorithms incl. the learner's own), warm start from the
  seed's own Stage-A final checkpoint (run seed 101/102/103 ← Stage-A seed 1/2/3),
  fresh optimizer, ~500k decisions with checkpoints at ~100k / 250k / 500k,
  Stage-A control validation (500 manifest episodes) at every checkpoint.

Architecture, learning rate, batch, entropy, γ, GAE, replay, ES σ / population,
optimizer and reward are exactly the Phase 2C/2D values. Output:
ml/runs/stage_b/<algo>/s<seed>/ (never a Stage-A directory). Exports carry
the tag stage_b_pilot in their config and are not production policies.
"""

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

from ..common.config import load_config
from ..common.win_qos import disable_throttling  # noqa: F401  (imported for parity with the trainers)

ML_ROOT = Path(__file__).resolve().parents[2]
FROZEN = json.loads((ML_ROOT / "reports/phase2e0/frozen-hashes.json").read_text(encoding="utf-8"))
ALGOS = {
    # key: (module, Phase 2C/2D config, Stage-A run prefix, final checkpoint)
    "ppo": ("ipl_rl.algos.ppo", "ipl_rl/configs/ppo_2c3.json", "ppo-2c3", "update_0325"),
    "a2c": ("ipl_rl.algos.a2c", "ipl_rl/configs/a2c_2d1.json", "a2c-2d1", "update_0325"),
    "d3qn": ("ipl_rl.algos.d3qn", "ipl_rl/configs/d3qn_2d2.json", "d3qn-2d2", "update_0325"),
    "qrdqn": ("ipl_rl.algos.qr_dqn", "ipl_rl/configs/qr_dqn_2d3.json", "qr-dqn-2d3", "update_0325"),
    "es": ("ipl_rl.algos.openai_es", "ipl_rl/configs/openai_es_2d4.json", "openai-es-2d4", "gen_2000"),
}
SNAPSHOT_SHARE = 0.5
SEEDS = (101, 102, 103)
BUDGET = 497_664                       # 81 PPO/A2C updates × 12 envs × 512 steps; = 41,472 DQN cycles × 12
CHECKPOINTS = (98_304, 245_760, 497_664)  # ≈ 100k / 250k / 500k decisions


def build(algo, seed, budget=BUDGET, checkpoints=CHECKPOINTS, run_dir=None, run_name=None, eval_limit=500):
    module, cfg_path, prefix, final = ALGOS[algo]
    if seed not in SEEDS and run_dir is None:
        raise ValueError(f"approved Stage-B pilot run seeds are {SEEDS}")
    stage_a_seed = seed - 100 if seed in SEEDS else 1
    key = f"{algo}:s{stage_a_seed}"
    ckpt = f"runs/{prefix}-s{stage_a_seed}/checkpoints/{final}/checkpoint.pt"
    if FROZEN["exports"][key]["path"] != f"ml/runs/{prefix}-s{stage_a_seed}/checkpoints/{final}/policy.json":
        raise ValueError(f"frozen record does not name {ckpt} for {key}")
    mod = importlib.import_module(module)
    base = load_config(ML_ROOT / cfg_path, mod.DEFAULTS)
    ov = {"seed": seed, "stage": "B", "snapshot_share": SNAPSHOT_SHARE,
          "run_dir": run_dir or f"runs/stage_b/{algo}", "run_name": run_name or f"s{seed}"}
    sb = {"bridge": "stage_b", "tag": "stage_b_pilot", "pool": "15 frozen Stage-A exports (all five algorithms × 3 seeds)",
          "warm_start": ckpt, "warm_start_sha256": FROZEN["exports"][key]["checkpointSha256"], "warm_start_export": key,
          "budgetDecisions": budget, "checkpointDecisions": list(checkpoints), "phase2dConfig": cfg_path}
    if algo in ("ppo", "a2c"):
        batch = base["num_envs"] * base["num_steps"]
        if budget % batch or any(c % batch for c in checkpoints):
            raise ValueError("budget/checkpoints must be whole updates")
        ov.update({"total_decisions": budget, "eval_updates": [c // batch for c in checkpoints], "eval_limit": eval_limit, "final_eval_limit": eval_limit})
    elif algo in ("d3qn", "qrdqn"):
        n = base["num_envs"]
        if budget % n or any(c % n for c in checkpoints):
            raise ValueError("budget/checkpoints must be whole cycles")
        ov.update({"total_decisions": budget, "eval_decisions": list(checkpoints), "eval_limit": eval_limit, "final_eval_limit": eval_limit,
                   "eps_start": base["eps_final"]})  # approved: continue at the final ε of the Stage-A schedule
        sb["exploration"] = f"ε constant at the Stage-A final value {base['eps_final']} (eps_start := eps_final); replay starts empty"
    else:  # es: generation-based; stop at the first generation boundary at or after the budget
        ov.update({"generations": 10 * (budget // 5000 + 1), "decision_checkpoints": list(checkpoints), "eval_limit": eval_limit})
        sb["stop_decisions"] = budget
    for k in ov:
        if k not in mod.DEFAULTS:
            raise KeyError(f"{algo}: {k} is not a config key")
    cfg = {**base, **ov, "stage_b": sb}
    return mod, cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--algo", required=True, choices=sorted(ALGOS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny budget into runs/stage_b/_smoke (infrastructure check only)")
    args = ap.parse_args(argv)
    if args.smoke:
        n = 12 * 512 if args.algo in ("ppo", "a2c") else 12 * 512
        mod, cfg = build(args.algo, args.seed, budget=n, checkpoints=(n,), run_dir=f"runs/stage_b/_smoke/{args.algo}", run_name=f"s{args.seed}", eval_limit=20)
    else:
        mod, cfg = build(args.algo, args.seed)
    run_dir = (ML_ROOT / cfg["run_dir"] / cfg["run_name"]).resolve()
    if run_dir.exists():
        raise SystemExit(f"{run_dir} exists — refusing to overwrite")
    os.environ["STAGE_B_LOG_DIR"] = str(run_dir / "bridge_log")
    print(f"[stage-b] {args.algo} seed {args.seed}: share {cfg['snapshot_share']}  warm start {cfg['stage_b']['warm_start_export']}  "
          f"budget {cfg['stage_b']['budgetDecisions']:,}  checkpoints {cfg['stage_b']['checkpointDecisions']}", flush=True)
    out_dir, summary = mod.train(cfg)
    print(f"[stage-b] done: {summary.get('status')}  {out_dir}", flush=True)
    return 0 if summary.get("status") == "completed" else 2


if __name__ == "__main__":
    sys.exit(main())
