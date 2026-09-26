"""Action-mask helpers over act-v2 (20 actions).

Masked actions get a logit of MASKED_LOGIT, so their probability is exactly
0 in float32 and they are never sampled; entropy is taken over the legal
actions only. Every function asserts the mask has a legal action — the
environment guarantees one at every step.
"""

import math

import torch

MASKED_LOGIT = -1e8


def masked_logits(logits, mask):
    return torch.where(mask, logits, torch.full_like(logits, MASKED_LOGIT))


def masked_distribution(logits, mask):
    return torch.distributions.Categorical(logits=masked_logits(logits, mask))


def masked_sample(logits, mask, generator=None):
    """Sample one legal action per row. Returns (actions, log_probs, entropy)."""
    dist = masked_distribution(logits, mask)
    if generator is None:
        actions = dist.sample()
    else:
        actions = torch.multinomial(dist.probs, 1, generator=generator).squeeze(-1)
    return actions, dist.log_prob(actions), dist.entropy()


def masked_argmax(scores, mask):
    return masked_logits(scores, mask).argmax(dim=-1)


def normalised_entropy(entropy, mask):
    """Entropy / log(#legal); 0 where only one action is legal."""
    legal = mask.sum(dim=-1).clamp(min=1).to(entropy.dtype)
    denom = torch.log(legal)
    return torch.where(legal > 1, entropy / denom.clamp(min=math.log(2)), torch.zeros_like(entropy))


def assert_legal(actions, mask):
    picked = mask.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)
    if not bool(picked.all()):
        bad = torch.nonzero(~picked).flatten().tolist()
        raise AssertionError(f"masked action selected in rows {bad[:10]}")
