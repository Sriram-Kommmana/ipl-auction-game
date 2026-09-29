"""Run configuration: a JSON file layered over an algorithm's defaults.

    cfg = load_config("ml/ipl_rl/configs/ppo_pilot.json", PPO_DEFAULTS, overrides={"seed": 2})

Unknown keys are rejected (a typo must not silently fall back to a default).
The config hash is SHA-256 over canonical JSON (sorted keys), first 16 hex.
"""

import hashlib
import json
from pathlib import Path


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# Keys that name or place a run, or only parallelise evaluation, without
# changing what is trained: left out of the config hash.
NON_SEMANTIC = frozenset({"run_name", "run_dir", "eval_workers"})


def config_hash(cfg):
    semantic = {k: v for k, v in cfg.items() if k not in NON_SEMANTIC}
    return hashlib.sha256(canonical_json(semantic).encode("utf-8")).hexdigest()[:16]


def load_config(path, defaults, overrides=None):
    cfg = dict(defaults)
    layers = []
    if path:
        layers.append((str(path), json.loads(Path(path).read_text(encoding="utf-8"))))
    if overrides:
        layers.append(("overrides", overrides))
    for source, layer in layers:
        unknown = sorted(set(layer) - set(defaults) - {"_comment"})
        if unknown:
            raise KeyError(f"{source}: unknown config keys {unknown}")
        cfg.update({k: v for k, v in layer.items() if k != "_comment"})
    return cfg
