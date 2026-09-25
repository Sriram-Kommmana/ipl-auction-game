"""Export a network to rl-policy-v2 — the JSON the Node server runs.

    policy = to_policy_json(net, obs_spec, act_spec, meta={...})
    write_policy(path, policy)

The exporter stamps the obs-v2 / act-v2 versions and hashes it was built
against (take them from the environment: env.obs_spec / env.act_spec), so
the JavaScript loader refuses the model if the specs ever change. Selection
is fixed per algorithm and must match packages/shared/src/rl/policy.js.
"""

import json
import time
from pathlib import Path

from .nets import ACTION_COUNT, ARCHITECTURES, OBS_SIZE, QUANTILES

FORMAT = "rl-policy-v2"
SELECTION = {
    "ppo": {"mode": "sample", "temperature": 0.3},
    "a2c": {"mode": "sample", "temperature": 0.3},
    "d3qn": {"mode": "argmax"},
    "qrdqn": {"mode": "argmax"},
    "es": {"mode": "argmax"},
}


def to_policy_json(net, obs_spec, act_spec, meta=None):
    algo = net.algorithm
    arch = ARCHITECTURES[algo]
    architecture = {
        "input": OBS_SIZE,
        "hidden": list(arch["hidden"]),
        "activation": "tanh",
        "head": arch["head"],
        "actions": ACTION_COUNT,
    }
    if arch["head"] == "quantiles":
        architecture["quantiles"] = QUANTILES
    return {
        "format": FORMAT,
        "algorithm": algo,
        "modelType": "mlp",
        "architecture": architecture,
        "obsSpec": {"version": obs_spec["version"], "hash": obs_spec["hash"], "size": obs_spec["size"]},
        "actSpec": {"version": act_spec["version"], "hash": act_spec["hash"], "count": act_spec["count"]},
        "selection": SELECTION[algo],
        # float32 weights → Python floats (exact), JSON numbers.
        "layers": [{"weight": w.double().tolist(), "bias": b.double().tolist()} for w, b in net.export_layers()],
        "meta": {
            "trainedSteps": 0,
            "seed": None,
            "config": {},
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "exporter": "ml/ipl_rl/export.py",
            **(meta or {}),
        },
    }


def write_policy(path, policy):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(policy))
    return path
