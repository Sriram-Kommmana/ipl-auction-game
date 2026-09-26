"""Shared training infrastructure (Phase 2C) — used by every algorithm.

  config      JSON run configs, defaults, config hash
  seeding     global seeding and the deterministic training-seed schedule
  vec_env     VecIplAuctionEnv: N Node simulators stepped in parallel
  masking     masked sampling / argmax / entropy over act-v2
  stats       episode and rollout statistics
  logger      TensorBoard + JSON-lines logs + run summary
  checkpoint  checkpoints and rl-policy-v2 exports
  metadata    reproducibility metadata (code, versions, specs)
  evaluation  Node evaluation hook (validation manifest, locked baselines,
              safety invariants) and export/inference parity

Python never implements auction rules: every environment step goes to the
JavaScript simulator through the bridge.
"""
