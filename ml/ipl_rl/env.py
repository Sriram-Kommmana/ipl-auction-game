"""IplAuctionEnv-v2 — Gymnasium interface to the JavaScript auction.

One episode = one complete Full Pool auction (main round + re-auction).
One step = one lot where the learner has at least one legal bid; the Node
side resolves every other lot with the learner passing.

  observation  Box(-1, 5, (80,), float32)   obs-v2, built in JavaScript
  action       Discrete(20)                 act-v2 willingness-to-pay levels
  action_masks()                            the canonical JavaScript mask
  reward                                    ΔBestXI / 110, −2 terminal if an XI slot is empty
  gamma                                     1.0 (exposed as env.gamma)

Masked actions raise — legality is never learned from penalties. Algorithms
must sample from action_masks() (masked logits / masked argmax).
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .bridge import BridgeV2

OBS_SIZE = 80
ACTION_COUNT = 20
PASS = 0

# Seed ranges (Phase 2A, frozen). Training draws fresh seeds from its range;
# validation/test episodes come from the committed manifests.
SPLIT_RANGES = {
    "train": (1_000_000, 2**31 - 1),
    "validation": (100_000, 100_499),
    "test": (200_000, 200_999),
}


class MaskedActionError(ValueError):
    pass


class IplAuctionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, split="train", tremble=0.01, snapshot_share=0.0, node="node"):
        super().__init__()
        if split not in SPLIT_RANGES:
            raise ValueError(f"unknown split {split}")
        self.split = split
        self.bridge = BridgeV2(node)
        info = self.bridge.info
        self.obs_spec = {k: info["obsSpec"][k] for k in ("version", "hash", "size")}
        self.act_spec = {k: info["actSpec"][k] for k in ("version", "hash", "count")}
        self.feature_names = list(info["obsSpec"]["features"])
        self.action_names = list(info["actSpec"]["actions"])
        self.gamma = float(info["gamma"])
        if self.obs_spec["size"] != OBS_SIZE or self.act_spec["count"] != ACTION_COUNT:
            raise RuntimeError(f"bridge specs {self.obs_spec} / {self.act_spec} do not match obs-v2 / act-v2")

        self.observation_space = spaces.Box(low=-1.0, high=5.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(ACTION_COUNT)
        self._mask = np.zeros(ACTION_COUNT, dtype=bool)
        self._done = True
        self.configure(tremble=tremble, snapshot_share=snapshot_share)

    # ── opponent pool (Stage B league) ─────────────────────────────────────
    def configure(self, tremble=None, snapshot_share=None):
        fields = {}
        if tremble is not None:
            fields["tremble"] = float(tremble)
        if snapshot_share is not None:
            fields["snapshotShare"] = float(snapshot_share)
        return self.bridge.call("configure", **fields)["config"]

    def add_snapshot(self, policy):
        """Add a frozen rl-policy-v2 (dict) to the opponent pool."""
        return self.bridge.call("addSnapshot", policy=policy)["snapshots"]

    # ── Gymnasium API ──────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "entry" in options:
            reply = self.bridge.call("reset", entry=options["entry"])
        else:
            episode_seed = options.get("episode_seed")
            if episode_seed is None:
                lo, hi = SPLIT_RANGES[self.split]
                episode_seed = int(self.np_random.integers(lo, hi + 1))
            reply = self.bridge.call("reset", seed=int(episode_seed), split=self.split)
        self._mask = np.array(reply["mask"], dtype=bool)
        self._done = False
        return np.asarray(reply["obs"], dtype=np.float32), reply["info"]

    def step(self, action):
        if self._done:
            raise RuntimeError("episode is over — call reset()")
        action = int(action)
        if not (0 <= action < ACTION_COUNT) or not self._mask[action]:
            raise MaskedActionError(f"action {action} is masked (legal: {np.flatnonzero(self._mask).tolist()})")
        reply = self.bridge.call("step", action=action)
        done = bool(reply["done"])
        if done:
            self._done = True
            self._mask = np.zeros(ACTION_COUNT, dtype=bool)
            self._mask[PASS] = True  # placeholder; no further steps are allowed
            obs = np.zeros(OBS_SIZE, dtype=np.float32)
        else:
            self._mask = np.array(reply["mask"], dtype=bool)
            obs = np.asarray(reply["obs"], dtype=np.float32)
        return obs, float(reply["reward"]), done, False, reply["info"]

    def action_masks(self):
        return self._mask.copy()

    def close(self):
        self.bridge.close()
