"""The deployable network of each algorithm (Phase 2A, frozen).

Only what production runs is defined here — the part exported to
rl-policy-v2. Training-only parts (PPO/A2C critics, DQN target networks, ES
optimiser state) belong to Phase 2C.

  ppo, a2c  80 → 128 → 128 → 20 logits            (tanh)
  d3qn      80 → 128 → 128 → V(1) + A(20) → Q(20) (tanh; dueling folded at export)
  qrdqn     80 → 128 → 128 → 20 × 32 quantiles   (tanh)
  es        80 → 64 → 64 → 20 logits              (tanh)
"""

import torch
from torch import nn

OBS_SIZE = 80
ACTION_COUNT = 20
QUANTILES = 32

ARCHITECTURES = {
    "ppo": {"hidden": (128, 128), "head": "logits"},
    "a2c": {"hidden": (128, 128), "head": "logits"},
    "d3qn": {"hidden": (128, 128), "head": "q"},
    "qrdqn": {"hidden": (128, 128), "head": "quantiles", "quantiles": QUANTILES},
    "es": {"hidden": (64, 64), "head": "logits"},
}


class PolicyNet(nn.Module):
    def __init__(self, algorithm):
        super().__init__()
        if algorithm not in ARCHITECTURES:
            raise ValueError(f"unknown algorithm {algorithm}")
        self.algorithm = algorithm
        arch = ARCHITECTURES[algorithm]
        self.head_type = arch["head"]
        h1, h2 = arch["hidden"]
        self.body = nn.Sequential(nn.Linear(OBS_SIZE, h1), nn.Tanh(), nn.Linear(h1, h2), nn.Tanh())
        if self.head_type == "q":
            self.value = nn.Linear(h2, 1)
            self.advantage = nn.Linear(h2, ACTION_COUNT)
        elif self.head_type == "quantiles":
            self.head = nn.Linear(h2, ACTION_COUNT * QUANTILES)
        else:
            self.head = nn.Linear(h2, ACTION_COUNT)

    def forward(self, obs):
        """logits (B, 20) | Q (B, 20) | quantiles (B, 20, 32)."""
        z = self.body(obs)
        if self.head_type == "q":
            a = self.advantage(z)
            return self.value(z) + a - a.mean(dim=-1, keepdim=True)
        out = self.head(z)
        if self.head_type == "quantiles":
            return out.view(*out.shape[:-1], ACTION_COUNT, QUANTILES)
        return out

    def action_scores(self, obs):
        """One score per action — what production ranks legal actions by."""
        out = self.forward(obs)
        return out.mean(dim=-1) if self.head_type == "quantiles" else out

    def export_layers(self):
        """[(W, b)] for the exported plain MLP; the dueling head folds into one layer:
        Q = V + A − mean(A) = (W_a − mean_rows(W_a) + W_v) z + (b_a − mean(b_a) + b_v)."""
        linears = [m for m in self.body if isinstance(m, nn.Linear)]
        layers = [(m.weight.detach(), m.bias.detach()) for m in linears]
        if self.head_type == "q":
            wa, ba = self.advantage.weight.detach(), self.advantage.bias.detach()
            wv, bv = self.value.weight.detach(), self.value.bias.detach()
            layers.append((wa - wa.mean(dim=0, keepdim=True) + wv, ba - ba.mean() + bv))
        else:
            layers.append((self.head.weight.detach(), self.head.bias.detach()))
        return layers


def build(algorithm, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return PolicyNet(algorithm)
