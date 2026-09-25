"""Turn a trained MaskablePPO model into the JSON file the server runs.

    python ml/export.py runs/v1/model.zip

Writes:
  apps/server/src/bots/models/rl-policy.json          ← the server loads this at start-up
  packages/shared/test/fixtures/policy-parity.json    ← proves JS and PyTorch agree

Only the ACTOR (the part that picks actions) is exported — the value network
is a training aid and isn't needed to play. The server runs it with ~20 lines
of plain JavaScript (packages/shared/src/mlp.js), no PyTorch or ONNX needed.

Restart the server after exporting so it picks up the new policy.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from bridge import REPO_ROOT, Bridge

SERVER_MODEL = REPO_ROOT / "apps" / "server" / "src" / "bots" / "models" / "rl-policy.json"
PARITY_FIXTURE = REPO_ROOT / "packages" / "shared" / "test" / "fixtures" / "policy-parity.json"


def actor_layers(model):
    """The actor is Linear→tanh→Linear→tanh (mlp_extractor.policy_net) then
    Linear (action_net). Each nn.Linear stores weight as [out, in] — the same
    orientation mlp.js expects."""
    linears = [m for m in model.policy.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
    linears.append(model.policy.action_net)
    return [
        {"weight": layer.weight.detach().cpu().numpy().tolist(), "bias": layer.bias.detach().cpu().numpy().tolist()}
        for layer in linears
    ]


def actor_to_json(model, features, trained_steps):
    return {
        "format": "mlp-v1",
        "observationSize": len(features),
        "actionCount": int(model.action_space.n),
        "hiddenActivation": "tanh",
        "features": features,
        "layers": actor_layers(model),
        "meta": {
            "algorithm": "MaskablePPO",
            "trainedSteps": int(trained_steps),
            "exportedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }


def torch_logits(model, observations):
    with torch.no_grad():
        obs = torch.as_tensor(np.asarray(observations, dtype=np.float32))
        latent = model.policy.mlp_extractor.forward_actor(obs)
        return model.policy.action_net(latent).cpu().numpy().tolist()


def sample_observations(count=64):
    """Real observations from real auctions — a better parity test than noise."""
    bridge = Bridge()
    try:
        features = bridge.call("info")["features"]
        observations, step = [], bridge.call("reset", seed=12345)
        while len(observations) < count:
            observations.append(step["obs"])
            legal = [a for a, ok in enumerate(step["mask"]) if ok]
            step = bridge.call("step", action=legal[len(observations) % len(legal)])
            if step["done"]:
                step = bridge.call("reset", seed=len(observations))
        return features, observations
    finally:
        bridge.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path, help="path to a MaskablePPO model.zip")
    parser.add_argument("--out", type=Path, default=SERVER_MODEL)
    args = parser.parse_args()

    model = MaskablePPO.load(args.model, device="cpu")
    features, observations = sample_observations()
    payload = actor_to_json(model, features, model.num_timesteps)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload))
    print(f"policy → {args.out}")

    PARITY_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    PARITY_FIXTURE.write_text(json.dumps({
        "model": str(args.out.relative_to(REPO_ROOT)).replace("\\", "/"),
        "observations": observations,
        "logits": torch_logits(model, observations),
    }))
    print(f"parity fixture → {PARITY_FIXTURE}")
    print("check it with:  node --test packages/shared/test/policyParity.test.js")


if __name__ == "__main__":
    main()
