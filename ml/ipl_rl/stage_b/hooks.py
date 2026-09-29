"""Phase 2F — the only Stage-B entry points the five trainers call.

With cfg["stage"] == "A" (every Phase 2C/2D config) each hook is a no-op that
keeps the original behaviour and error messages exactly, so Stage-A training
is unchanged. With cfg["stage"] == "B" the run must carry a `stage_b` block
(written by ipl_rl.stage_b.train_b) and then:

  · env_kwargs  — the opponent SOURCE: the Stage-B training bridge
                  (bridge_b.mjs: frozen sampler league draw over the 15 frozen
                  Stage-A exports, fixed-clock opponent runtime, fallbacks are
                  hard stops); mechanics, observation, mask, reward unchanged;
  · warm_start  — load the seed's frozen Stage-A final checkpoint (sha256 must
                  match the Phase 2E.0 frozen record) into the fresh model;
  · incomplete_is_stop — a learner training episode that ends without a legal
                  XI is a recorded behavioural finding in Stage B (Phase 2F §26),
                  not a hard stop; invariant violations, masked actions,
                  fallbacks and parity failures remain hard stops.
"""

import hashlib
import json
from pathlib import Path

import torch

ML_ROOT = Path(__file__).resolve().parents[2]
REPO = ML_ROOT.parent
BRIDGE_SCRIPT = ML_ROOT / "ipl_rl" / "stage_b" / "bridge_b.mjs"
BRIDGE_PROTOCOL = "rl-bridge-v2-stage-b"
REQUIRED = ("bridge", "warm_start", "warm_start_sha256", "warm_start_export", "tag")


def check_stage(cfg, stage_a_message):
    if cfg["stage"] == "A":
        if cfg["snapshot_share"] != 0.0 or cfg.get("stage_b") is not None:
            raise ValueError(stage_a_message)
        return
    if cfg["stage"] != "B":
        raise ValueError(stage_a_message)
    sb = cfg.get("stage_b")
    if not isinstance(sb, dict) or any(k not in sb for k in REQUIRED):
        raise ValueError(f"Stage B needs a stage_b block with {REQUIRED}")
    if sb["bridge"] != "stage_b" or not (0.0 < cfg["snapshot_share"] <= 1.0) or sb["tag"] != "stage_b_pilot":
        raise ValueError("Stage B: bridge must be 'stage_b', 0 < snapshot_share ≤ 1, tag 'stage_b_pilot'")


def env_kwargs(cfg):
    if cfg["stage"] != "B":
        return {}
    return {"bridge_script": str(BRIDGE_SCRIPT), "bridge_protocol": BRIDGE_PROTOCOL}


def warm_start(cfg, modules, run_dir=None):
    """modules: {checkpoint module name: nn.Module}. Returns a record (None in Stage A)."""
    if cfg["stage"] != "B":
        return None
    sb = cfg["stage_b"]
    path = (ML_ROOT / sb["warm_start"]).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sb["warm_start_sha256"]:
        raise ValueError(f"HARD STOP warm-start checkpoint {path} sha256 {digest} ≠ frozen record {sb['warm_start_sha256']}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for name, module in modules.items():
        module.load_state_dict(ckpt["modules"][name])
    rec = {"export": sb["warm_start_export"], "checkpoint": sb["warm_start"], "sha256": digest,
           "stageAState": ckpt.get("state"), "stageAConfigHash": ckpt.get("configHash"), "modules": sorted(modules)}
    if run_dir is not None:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / "warm_start.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    print(f"[stage-b] warm start from {sb['warm_start_export']} ({sb['warm_start']}, sha256 {digest[:12]})", flush=True)
    return rec


def incomplete_is_stop(cfg):
    return cfg["stage"] != "B"
