"""IPL auction RL infrastructure (Phase 2B).

  bridge   BridgeV2 — the Node process that runs the real game (stdlib only)
  env      IplAuctionEnv — Gymnasium environment (needs gymnasium, numpy)
  nets     deployable networks of the five algorithms (needs torch)
  export   rl-policy-v2 exporter (needs torch)

Submodules are imported explicitly so the bridge works without the ML stack.
"""
