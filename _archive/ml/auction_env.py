"""Gymnasium environment: one RL franchise in a 10-team IPL auction.

Learning notes (Gymnasium "create a custom environment" guide):
  - reset() starts a new auction and returns the first observation.
  - step(action) plays one decision and returns
        (observation, reward, terminated, truncated, info).
  - action_masks() is NOT part of Gymnasium — it's the hook sb3-contrib's
    MaskablePPO calls to learn which actions are legal right now.

What one step is
  One step = one lot where this franchise has a real choice. The action picks
  a price CAP: action i means "pay up to CAP_MULTIPLIERS[i] × fair value"
  (action 0 = pass). The simulator then runs the bidding war between all ten
  franchises and moves on to the next lot where we have a choice.

Reward
  Almost all of it arrives at the very end (terminated=True): how our best XI
  compares with the other nine teams, plus a small persona-style bonus. A tiny
  negative nudge is given for buying players who don't make the XI. See
  packages/shared/src/rewards.js — that's the source of truth.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bridge import Bridge


class AuctionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, tremble=0.01, snapshot_share=0.0, persona=None):
        super().__init__()
        self.bridge = Bridge()
        info = self.bridge.call("info")
        self.feature_names = info["features"]
        self.fixed_persona = persona  # None → a jittered preset every episode

        self.observation_space = spaces.Box(
            low=0.0, high=5.0, shape=(info["observationSize"],), dtype=np.float32
        )
        self.action_space = spaces.Discrete(info["actionCount"])
        self._mask = np.ones(info["actionCount"], dtype=bool)
        self.configure(tremble=tremble, snapshot_share=snapshot_share)

    # ── Opponent pool controls (called by train.py via env_method) ─────────
    def configure(self, tremble=None, snapshot_share=None):
        fields = {}
        if tremble is not None:
            fields["tremble"] = tremble
        if snapshot_share is not None:
            fields["snapshotShare"] = snapshot_share
        self.bridge.call("configure", **fields)

    def add_snapshot(self, path):
        """Add a frozen copy of an earlier policy to the opponent pool."""
        return self.bridge.call("addSnapshot", path=str(path))["snapshots"]

    # ── Gymnasium API ──────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        persona = (options or {}).get("persona", self.fixed_persona)
        fields = {"seed": episode_seed}
        if persona is not None:
            fields["persona"] = list(map(float, persona))
        reply = self.bridge.call("reset", **fields)
        self._mask = np.array(reply["mask"], dtype=bool)
        return np.array(reply["obs"], dtype=np.float32), reply["info"]

    def step(self, action):
        reply = self.bridge.call("step", action=int(action))
        self._mask = np.array(reply["mask"], dtype=bool)
        obs = np.array(reply["obs"], dtype=np.float32)
        return obs, float(reply["reward"]), bool(reply["done"]), False, reply["info"]

    def action_masks(self):
        return self._mask.copy()

    def close(self):
        self.bridge.close()
