"""How good is the trained policy, and are its five personalities different?

    python ml/evaluate.py runs/v1/model.zip --episodes 40

For each persona preset, plays full auctions against the rule-bot field (no
self-play opponents, no trembling) and reports:
  - average XI strength and finishing rank (1 = best of 10)
  - win rate
  - style: share of purse spent, overseas in the XI, 90+ rated stars signed

What "good" looks like (checkpoints in ml/README.md):
  (c) beats the rule-bot field — average rank clearly better than 5.5
  (e) personas differ — e.g. Global Scout fields more overseas players,
      Aggressor spends more, Pace Factory's XI leans on bowling
If every persona looks the same, the persona terms in the reward are too weak
relative to winning, or the policy is ignoring the persona inputs.

Exploitability check (from d'Eon et al.): train a NEW policy only against
frozen copies of this one (put its export in the opponent pool with
snapshot_share=1.0). If the newcomer wins easily, this policy is exploitable
— a human will eventually find the same hole.
"""

import argparse
from statistics import mean

import numpy as np
from sb3_contrib import MaskablePPO

from auction_env import AuctionEnv

PRESETS = {
    "Aggressor": [1, 0, 0, 0, 0.5],
    "Pace Factory": [0.3, 1, 0, 0, 0],
    "Run Machine": [0.3, 0, 1, 0, 0],
    "Global Scout": [0.3, 0, 0, 1, 0],
    "Adaptive": [0, 0, 0, 0, 0],
}


def play(model, env, persona, episodes, deterministic):
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=10_000 + ep, options={"persona": persona})
        done = False
        while not done:
            action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=deterministic)
            obs, _, done, _, info = env.step(action)
        rows.append(info)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--sample", action="store_true", help="sample actions instead of taking the best one")
    args = parser.parse_args()

    model = MaskablePPO.load(args.model, device="cpu")
    env = AuctionEnv(tremble=0.0, snapshot_share=0.0)
    try:
        print(f"{'persona':14s} {'XI':>6s} {'rank':>5s} {'win%':>5s}  {'spent':>6s} {'OS XI':>6s} {'stars':>6s}")
        for name, persona in PRESETS.items():
            rows = play(model, env, persona, args.episodes, deterministic=not args.sample)
            ranks = [r["rank"] + 1 for r in rows]
            print(
                f"{name:14s} {mean(r['strength'] for r in rows):6.1f} {mean(ranks):5.2f} "
                f"{100 * np.mean([rk == 1 for rk in ranks]):4.0f}%  "
                f"{mean(r['terms']['aggression'] + 0.5 for r in rows) * 100:5.0f}% "
                f"{mean((r['terms']['overseasFocus'] + 0.5) * 4 for r in rows):6.1f} "
                f"{mean((r['terms']['starFocus'] + 0.5) * 4 for r in rows):6.1f}"
            )
        print("\nrank is out of 10 teams; 5.5 = average. OS XI = overseas players in the best XI (max 4).")
    finally:
        env.close()


if __name__ == "__main__":
    main()
