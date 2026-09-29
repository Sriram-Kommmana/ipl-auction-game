"""Episode and rollout statistics.

Episode summaries come from the JavaScript environment (RlEpisode.summary):
the same fields the Node evaluator reports, so training curves and
validation reports use one vocabulary.
"""

import numpy as np

# Episode fields logged as means (booleans become rates).
EPISODE_FIELDS = (
    "return", "xi", "strength", "legalXI", "strongXI", "rank", "purseLeft", "purseLeftShare", "purseSpent",
    "squadSize", "overseas", "buys", "stars", "marginalBuys", "priceToFair", "xiGainPer1000",
    "reauctionBuys", "reauctionCritical", "reauctionUseful", "reauctionMarginal", "reauctionDepth", "reauctionNone",
    "bidRate", "capToFair", "contests", "decisions", "shieldActivations", "shieldWins", "invariantViolations",
)


class EpisodeLog:
    """Collects finished-episode summaries; drain() returns and clears them."""

    def __init__(self):
        self.pending = []
        self.total = 0
        self.violations = []

    def add(self, summary):
        self.pending.append(summary)
        self.total += 1
        if summary.get("invariantViolations", 0):
            self.violations.append({"seed": summary["seed"], "violations": summary.get("violations", [])})

    def drain(self):
        out, self.pending = self.pending, []
        return out


def summarise_episodes(episodes):
    """{field: mean} over episode summaries (None-valued entries skipped)."""
    out = {"episodes": len(episodes)}
    if not episodes:
        return out
    for f in EPISODE_FIELDS:
        xs = [float(e[f]) for e in episodes if e.get(f) is not None]
        if xs:
            out[f] = float(np.mean(xs))
    for s in ("low", "normal", "high"):
        sub = [e for e in episodes if e.get("stratum") == s]
        if sub:
            out[f"xi_{s}"] = float(np.mean([e["xi"] for e in sub]))
            out[f"return_{s}"] = float(np.mean([e["return"] for e in sub]))
            out[f"episodes_{s}"] = len(sub)
    return out


def action_stats(actions, masks, action_count=20):
    """Distribution of chosen actions over a rollout, plus how much choice there was."""
    actions = np.asarray(actions).reshape(-1)
    masks = np.asarray(masks).reshape(-1, action_count)
    counts = np.bincount(actions, minlength=action_count)
    return {
        "action_shares": (counts / max(1, counts.sum())).tolist(),
        "mean_action": float(actions.mean()),
        "bid_share": float((actions != 0).mean()),
        "legal_actions_mean": float(masks.sum(axis=1).mean()),
        "pass_only_share": float((masks.sum(axis=1) == 1).mean()),
    }
