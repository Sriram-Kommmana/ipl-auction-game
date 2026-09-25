"""Starter PPO training script for the five RL franchises.

    python ml/train.py --steps 2000000 --name v1

This is a working baseline, not a finished recipe — tuning it is the point of
the exercise. Read the comments, run it, look at the evaluation, change one
thing at a time.

────────────────────────────────────────────────────────────────────────────
HOW TRAINING IS ORGANISED

One network, five personalities
  The observation ends with a 5-number persona vector
  [aggression, bowling focus, batting focus, overseas focus, star focus], and
  the same vector weights part of the reward. Each episode the learner gets a
  jittered persona, so ONE policy learns to play all five styles. At run time
  the server feeds each RL seat its own preset (packages/shared/src/personas.js).

Phase A — learn the game against the rule bots
  Opponents are the four hand-written bots plus a noisy "human proxy".

Phase B — self-play against frozen copies of itself
  Every --snapshot-every steps the current policy is exported and added to
  every environment's opponent pool; afterwards ~40% of opponent seats are
  earlier versions of the learner. This is a light version of the league /
  fictitious self-play idea (NFSP, Heinrich & Silver 2016; AlphaStar's league):
  playing only the newest self tends to chase its own tail, playing the whole
  history keeps it honest.

Throughout — "trembling" opponents
  Opponents play a random legal action 1% of the time. d'Eon et al. (2024,
  MARL for auctions) found self-play policies can look perfect against their
  training partners and collapse against anyone else; small noise helps.

Invalid actions are MASKED, not punished
  MaskablePPO sets illegal actions' logits to -inf (Huang & Ontañón 2022,
  "A Closer Look at Invalid Action Masking"). The environment exposes the mask
  via action_masks(). With SubprocVecEnv the mask method has to live on the
  environment itself (not an ActionMasker wrapper) — which it does.
────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from auction_env import AuctionEnv
from export import actor_to_json

ML_DIR = Path(__file__).resolve().parent


def make_env(rank, seed, pool_mode):
    def _init():
        env = AuctionEnv(pool_mode=pool_mode, snapshot_share=0.0)
        env.reset(seed=seed + rank)
        return env
    return _init


class SelfPlaySnapshots(BaseCallback):
    """Phase B: periodically freeze the policy and add it to the opponent pool."""

    def __init__(self, every, start_after, out_dir, features, share=0.4):
        super().__init__()
        self.every, self.start_after, self.share = every, start_after, share
        self.out_dir, self.features = out_dir, features
        self.last = 0
        self.enabled = False

    def _on_step(self):
        if self.num_timesteps < self.start_after or self.num_timesteps - self.last < self.every:
            return True
        self.last = self.num_timesteps
        path = self.out_dir / f"snapshot_{self.num_timesteps}.json"
        path.write_text(json.dumps(actor_to_json(self.model, self.features, self.num_timesteps)))
        count = self.training_env.env_method("add_snapshot", str(path))[0]
        if not self.enabled:
            self.training_env.env_method("configure", snapshot_share=self.share)
            self.enabled = True
            print(f"[self-play] phase B started — {int(self.share * 100)}% of opponent seats are past selves")
        print(f"[self-play] snapshot at {self.num_timesteps:,} steps ({count} in pool)")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="v1")
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool", default="mixed", choices=["quick", "full", "mixed"])
    parser.add_argument("--phase-a-steps", type=int, default=500_000,
                        help="train against rule bots only for this long before self-play")
    parser.add_argument("--snapshot-every", type=int, default=100_000)
    args = parser.parse_args()

    run_dir = ML_DIR / "runs" / args.name
    (run_dir / "snapshots").mkdir(parents=True, exist_ok=True)

    envs = VecMonitor(SubprocVecEnv([make_env(i, args.seed, args.pool) for i in range(args.n_envs)]))
    features = envs.get_attr("feature_names")[0]

    # Hyper-parameters worth experimenting with first:
    #   gamma     — nearly all reward comes at the end of a ~80-300 step
    #               episode, so it needs to be close to 1 or early decisions
    #               barely "see" the result.
    #   ent_coef  — exploration. Too low and it settles on "always pass";
    #               too high and it never commits.
    #   n_steps   — rollout length per env before each update.
    # See "The 37 Implementation Details of PPO" (ICLR blog track 2022).
    model = MaskablePPO(
        "MlpPolicy",
        envs,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=2048,
        n_epochs=10,
        gamma=0.997,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),  # must stay 2×64 tanh — mlp.js mirrors it
        tensorboard_log=str(ML_DIR / "runs" / "tensorboard"),
        seed=args.seed,
        verbose=1,
        device="cpu",
    )

    callbacks = [
        SelfPlaySnapshots(args.snapshot_every, args.phase_a_steps, run_dir / "snapshots", features),
        CheckpointCallback(save_freq=max(1, 250_000 // args.n_envs), save_path=str(run_dir / "checkpoints")),
    ]
    model.learn(total_timesteps=args.steps, callback=callbacks, tb_log_name=args.name)
    model.save(run_dir / "model")
    envs.close()

    print(f"\nsaved {run_dir / 'model.zip'}")
    print(f"next:  python ml/evaluate.py {run_dir / 'model.zip'}")
    print(f"then:  python ml/export.py {run_dir / 'model.zip'}")


if __name__ == "__main__":  # required on Windows: SubprocVecEnv re-imports this file in each worker
    main()
