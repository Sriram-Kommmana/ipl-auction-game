"""Deterministic seeding.

Two independent things are seeded:

1. The learner's own randomness (network init, action sampling, minibatch
   order): seed_everything(seed).
2. Which auctions are played: episode k of environment i in run `seed` plays
   train seed episode_seed(seed, i, k). The schedule depends only on
   (seed, i, k) — not on timing, the number of other environments, or what
   the learner does — so two algorithms given the same run seed train on
   exactly the same auctions (used by the SB3 cross-check).

Only train seeds (1,000,000 upward) are ever produced here; validation and
test episodes come from the committed manifests.
"""

import os
import random

import numpy as np
import torch

from ..env import SPLIT_RANGES


def seed_everything(seed, torch_threads=None):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch_threads:
        torch.set_num_threads(int(torch_threads))
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def episode_seed(run_seed, env_index, episode_index, split="train"):
    lo, hi = SPLIT_RANGES[split]
    if split != "train":
        raise ValueError("the schedule only draws training seeds; evaluate on the manifests")
    state = np.random.SeedSequence([int(run_seed), int(env_index), int(episode_index)]).generate_state(2, np.uint32)
    x = (int(state[0]) << 32) | int(state[1])
    return lo + x % (hi - lo + 1)
