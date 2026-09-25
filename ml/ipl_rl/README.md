# RL infrastructure v2 (Phase 2B)

This implements the **frozen Phase 2A specification**. Nothing here trains a
model: training is Phase 2C.

## Where things live

| Piece | File | Notes |
|---|---|---|
| obs-v2: the 80 features, normalisation, hash | `packages/shared/src/rl/obsSpec.js` | Canonical; the only definition |
| act-v2: the 20 actions, cap rule, ladder price | `packages/shared/src/rl/actionSpec.js` | Canonical |
| Action mask and completion shield | `packages/shared/src/rl/mask.js` | Canonical; used by training, evaluation and production |
| Reward (ΔXI/110, −2 terminal, γ = 1, λ_rel = 0) | `packages/shared/src/rl/reward.js` | |
| Purse / lineup / human-proxy samplers | `packages/shared/src/rl/samplers.js` | Seed-derived and reproducible |
| Environment core (one learner, nine frozen seats) | `packages/shared/src/rl/env.js` | Runs the unchanged `AuctionSim` |
| rl-policy-v2 format, validation, inference | `packages/shared/src/rl/policy.js` | |
| Production seat runtime with fallback | `packages/shared/src/rl/runtime.js` | |
| Baselines and evaluation harness | `packages/shared/src/rl/evaluate.js`, `bin/rl-evaluate.js` | Node, deterministic |
| Python ↔ JS bridge | `packages/shared/bin/rl-bridge-v2.js`, `ml/ipl_rl/bridge.py` | JSON lines |
| Gymnasium environment | `ml/ipl_rl/env.py` | `IplAuctionEnv` |
| Deployable networks (5 algorithms) | `ml/ipl_rl/nets.py` | Definitions only |
| Exporter → rl-policy-v2 | `ml/ipl_rl/export.py` | |
| Seed manifests | `packages/shared/data/rl-manifests/{train,validation,test}.json` | Committed |

**Python never implements game rules.** `IplAuctionEnv` sends actions over the
bridge. The Node process runs the real auction, the frozen rule bots, the
observation, the mask and the reward.

## Setup

```bash
py -3.12 -m venv ml/.venv
ml/.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
ml/.venv/Scripts/python -m pip install -r ml/requirements-v2.txt
```

## Tests

```bash
npm test --prefix packages/shared
cd ml && .venv/Scripts/python -m unittest discover -s ipl_rl/tests -t .
node packages/shared/bin/rl-manifests.js --check
```

- The first runs every JavaScript test, Phase 1 and RL.
- The second runs the bridge, Gymnasium, parity and export tests.
- The third checks that the committed manifests still match the sampler.

## Using the environment

```python
from ipl_rl.env import IplAuctionEnv
env = IplAuctionEnv(split="train")        # seeds from 1,000,000 upward
obs, info = env.reset(seed=0)             # info["entry"] = purse, seating, human proxy
mask = env.action_masks()                 # never step a masked action (it raises)
obs, reward, terminated, truncated, info = env.step(action)
# at the end: info["episode"] holds final XI, legality, purse, buys, shield activations, ...
```

- **Validation and test episodes** come from the committed manifests:
  `env.reset(options={"entry": manifest_entry})`.
- **Stage B (league):**
  - `env.add_snapshot(policy_dict)` loads a frozen rl-policy-v2 opponent;
  - `env.configure(snapshot_share=...)` sets how often RL seats use snapshots.

  Rule bots keep at least 4 of the 9 opponent seats.

## Evaluation

```bash
node packages/shared/bin/rl-evaluate.js --split validation --limit 50
node packages/shared/bin/rl-evaluate.js --split test --policy path/to/policy.json
```

- **Baselines:**
  - `moneyball`, `starChaser`, `balancedBuilder`, `opportunist`: a frozen bot in the learner's seat;
  - `productFallback`: the seat's own fallback persona;
  - `randomLegal`;
  - `fairValue`;
  - `plannerGreedy`.
- **Reporting:** every controller plays the same auctions, paired by seed. Results come per purse stratum, with bootstrap 95% confidence intervals and paired XI differences.

## Export

```python
from ipl_rl.export import to_policy_json, write_policy
policy = to_policy_json(net, env.obs_spec, env.act_spec, meta={"trainedSteps": n, "seed": s, "config": cfg})
write_policy("ml/runs/<name>/policy.json", policy)
```

- **Loading:** the JavaScript loader (`loadPolicy`) rejects any mismatch in spec hashes, dimensions or weights.
- **Production wiring** (one model per RL seat in `botManager`) is deliberately **not** done yet, because no trained model exists.

## Legacy v1 files
`ml/{auction_env,train,evaluate,export,bridge,smoke_bridge}.py`, `packages/shared/bin/rl-bridge.js` and `packages/shared/src/{observation,mlp,rewards}.js` are the pre-Phase-2 RL starter. They are left untouched because the frozen rule bots and today's server import parts of `observation.js`. v2 replaces the rest; don't train against v1.
