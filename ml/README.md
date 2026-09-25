> **Phase 2 (v2) supersedes this starter.** The frozen RL design is implemented in
> [`ml/ipl_rl/`](ipl_rl/README.md) (obs-v2, act-v2, bridge v2, `IplAuctionEnv`).
> The v1 files described below are legacy. Don't train against them.

# Training the RL franchises

Solo mode seats 9 AI franchises: 4 **rule-based** bots (hand-written, already
done — `packages/shared/src/ruleBots.js`) and 5 **reinforcement-learning** bots.
All five RL bots are one trained network. Each gets a different *persona
vector*, so they play five different styles.

Until you export a trained model, the RL seats play a fallback rule
personality. Solo mode works today; training only makes those five seats
smarter.

```
ml/                          ← you work here (Python)
  smoke_bridge.py            checkpoint 0 — no installs needed
  auction_env.py             the Gymnasium environment
  train.py                   MaskablePPO starter (tune me)
  evaluate.py                how good is it? do the personas differ?
  export.py                  model.zip → JSON the server runs
packages/shared/             ← the game itself (JavaScript, shared with the server)
  bin/rl-bridge.js           the simulator Python talks to
  src/observation.js         the 38 numbers a bot sees + the 8 actions
  src/rewards.js             the reward
apps/server/src/bots/models/rl-policy.json   ← export lands here
```

Python never re-implements the game. `auction_env.py` sends actions to
`rl-bridge.js`, which runs the same rules, scoring and rule bots the live
server uses. What you train against is exactly what players face.

## Setup

```bash
cd ml
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

A CPU is enough: the network is tiny, and the simulator does about 480
decisions/s per environment (8 in parallel ≈ 1M steps in ~5 minutes).

## Checkpoints

Work through these in order. Each one checks the previous one before you add
complexity.

| # | Command | Pass when |
|---|---|---|
| 0 | `python smoke_bridge.py` | Episodes finish; illegal action refused. **Runs today, no installs.** |
| a | `python -c "from gymnasium.utils.env_checker import check_env; from auction_env import AuctionEnv; check_env(AuctionEnv())"` | No errors (warnings about bounds are fine) |
| b | A random masked agent (see `smoke_bridge.py`) | No illegal bids, ranks around 6–9 |
| c | `python train.py --steps 1000000 --name v1` then `python evaluate.py runs/v1/model.zip` | Average rank clearly better than 5.5 against the rule bots |
| d | Cross-play: evaluate v1 against a pool containing another seed's snapshots | No collapse against opponents it never trained with |
| e | `evaluate.py` persona table | The five personas play visibly differently (spend, overseas, stars) |
| ship | `python export.py runs/v1/model.zip`, then `node --test packages/shared/test/policyParity.test.js`, then restart the server | Parity test passes; server logs "Loaded RL policy" |

## What to tune first

1. **`ent_coef`**: if the policy learns to always pass, raise it. If it never
   settles, lower it.
2. **`gamma`**: reward arrives at the end of an 80–300 step episode, so keep
   it close to 1.
3. **Reward weights** in `packages/shared/src/rewards.js`: `PERSONA_WEIGHT`
   trades "win" against "play in character". If the personas look identical
   (checkpoint e), raise it. If they stop winning, lower it.
4. **`--phase-a-steps` / `snapshot_share`**: when self-play starts, and how
   much of it is used.

Change **one thing per run** and keep the `evaluate.py` table from each run.

> ⚠ Changing `observation.js` (features) or `CAP_MULTIPLIERS` changes what
> the network sees. Old models will be rejected (the server checks the feature
> list), and you'll need to retrain.

## Learning path

Read in this order. Each stage maps to a part of this codebase.

| Stage | Learn | Resource | Where it shows up here |
|---|---|---|---|
| 1 | How real franchises think | Hindustan Times, *"IPL auction: the calm calculations hidden in the frenzied buying"* (2022) | `ruleBots.js`: money goes "to the position, not the player"; walking away when a substitute is coming |
| 1 | Valuing players | Malhotra, *Journal of Sports Analytics* 8(3), 2022 (abstract) | `valuation.js`: the fair-value curve |
| 1 | Strategy archetypes | Joglekar, Pifer & Narayanan, *Journal of Sports Analytics*, 2025 (abstract) | `personas.js`: distinct spending styles |
| 2 | RL basics: state → action → reward → policy | Gymnasium, [Create a custom environment](https://gymnasium.farama.org/introduction/create_custom_env/) | `auction_env.py` |
| 3 | PPO and illegal actions | [sb3-contrib MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html) · Huang & Ontañón, [arXiv:2006.14171](https://arxiv.org/abs/2006.14171) · [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) | `train.py`, `action_masks()` |
| 4 | Budget-aware observations and rewards | [cocoa-huang/rl-ad-bidding](https://github.com/cocoa-huang/rl-ad-bidding), `environment/gym_wrapper.py` | `observation.js`: money as a share of purse, pacing ratio |
| 5 | Multi-agent pitfalls | d'Eon, Newman & Leyton-Brown, [arXiv:2402.19420](https://arxiv.org/abs/2402.19420): the sections on self-play collapse and cross-play | trembling opponents, checkpoint d |
| 6 | Self-play with opponent pools | NFSP, [arXiv:1603.01121](https://arxiv.org/abs/1603.01121) · AlphaStar league training | `SelfPlaySnapshots` in `train.py` |
| 7 | Shipping the model | [SB3 export guide](https://stable-baselines3.readthedocs.io/en/master/guide/export.html) | `export.py` + `packages/shared/src/mlp.js` (plain JS, no ONNX) |

**Skip these**, even though they come up in searches:
- arXiv:2407.08022: its RL agent is the *auctioneer*, not a bidder.
- The Kavya-Upadhyay "AI Bidding Agent" repo: the DQN/Flask backend in its
  README doesn't exist in the code.
- The algorithm in arXiv:2212.02723: it assumes no budgets and a single item.
  The idea (model your rivals) is already covered by putting rivals' state in
  the observation.

## Honest expectations

A well-tuned fixed rule can come surprisingly close to PPO. That's the main
lesson of the rl-ad-bidding project. Checkpoint (c) exists so you'll know
whether RL is actually beating the rule bots, not just assume it is. If it
isn't yet, that's normal. The usual culprits are reward shaping and
exploration, not the network size.
