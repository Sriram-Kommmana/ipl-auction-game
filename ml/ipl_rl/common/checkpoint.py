"""Checkpoints (full training state, torch) and deployable exports (rl-policy-v2)."""

import random
from pathlib import Path

import numpy as np
import torch

from ..export import to_policy_json, write_policy


def save_checkpoint(path, *, modules, optimizer, step_state, cfg, cfg_hash):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "modules": {name: m.state_dict() for name, m in modules.items()},
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "state": step_state,
        "config": cfg,
        "configHash": cfg_hash,
        "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()},
    }, path)
    return path


def load_checkpoint(path, modules, optimizer=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for name, m in modules.items():
        m.load_state_dict(ckpt["modules"][name])
    if optimizer is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def export_policy(path, net, obs_spec, act_spec, *, trained_steps, seed, cfg, cfg_hash, extra_meta=None):
    policy = to_policy_json(net, obs_spec, act_spec, meta={
        "trainedSteps": int(trained_steps), "seed": seed, "config": cfg, "configHash": cfg_hash, **(extra_meta or {}),
    })
    write_policy(path, policy)
    return policy
