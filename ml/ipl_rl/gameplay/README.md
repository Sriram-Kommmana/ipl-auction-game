# Gameplay audit and production RL roster

These tools evaluate the already-trained RL exports **as game opponents**. They play complete Full Pool auctions (main round + re-auction) using:
- the unchanged `AuctionSim`;
- the frozen rule bots;
- the production seat runtime (`createRlSeat`: act-v3 mask and shield, cap ≤ maxSafeBid, the 20 ms room guard, the completion guard and the rule fallback).

Nothing is trained or written back.

| Script | What it does |
|---|---|
| `audit.mjs --mode list` | Lists every candidate export (15 Stage-A + 15 Stage-B pilot), with sha256 checks. |
| `audit.mjs --mode export` | Puts each export in one RL seat of product rooms (1 human proxy, 4 rule bots, 5 RL seats), with Stage-A exports of the other algorithms in the remaining RL seats. The rooms depend only on the seed, so the Stage-A and Stage-B exports of an algorithm meet identical rooms. |
| `audit.mjs --mode roster --roster r.json` | Plays a given roster (RL persona → model key); `R:fallback` means the seat plays its rule fallback. |
| `aggregate.py "<glob>" [--by label]` | Per-model product metrics, safety totals and latency. |
| `stress.mjs --roster r.json --rooms K` | Runs K concurrent auctions in one Node process with the real clock: latency percentiles and 20 ms guard trips. |

All seeds come from the regression range (90000+). They are never train, validation or test seeds.

## Findings (200 rooms per export; 400 rooms per roster)

- **Safety.** Legal XI was 100% for every export except Stage-A QR-DQN s2 (0.995 in the roster check). There were 0 cap > maxSafeBid decisions and 0 auction-invariant violations.
- **Play strength in product rooms.** The rule fallback (today's production) scores 87.9 XI.
  - PPO: 89.7–91.0.
  - Stage-B QR-DQN: 88.1–88.4.
  - Stage-A QR-DQN: 87.5.
  - Stage-A D3QN: 85.9–87.5.
  - A2C: 84.6–86.6.
  - ES: 84.2–85.5.
- **Styles:**
  - **PPO:** patient value buyer (buys at 0.78–0.89 × fair value). Some seeds hoard wicket-keepers (4–7 per squad).
  - **QR-DQN:** early competitor that contests stars.
  - **A2C and ES:** bid on almost every early lot and are spent out by about 25% of the auction. ES strands 6–17% of its purse and repeats one action on 50–88% of its decisions.
  - **Stage-B D3QN s101:** pathological. It passes 89% of lots and leaves 30% of its purse unspent.
- **Stage A vs Stage B.** Stage-B QR-DQN removes Stage-A QR-DQN's passes on critical, affordable keepers (64 → 0–2 in the roster check) and its incomplete XIs (2 → 0). Stage-B PPO s103 is about as strong as Stage-A PPO, with fewer keepers hoarded.
- **Latency.**
  - One process, 100 concurrent rooms (96k decisions): p50 0.35 ms, p99 0.73 ms, max 8.1 ms, 0 guard trips.
  - First (cold) decision: ≤ 3.9 ms. The server warms each model at startup.
  - Guard trips appeared only with 10–15 audit processes competing for the CPU. That is the intended degradation: the seat plays its rule fallback.

## Production roster (`apps/server/src/bots/models/registry.json`)

| RL persona | Model | Why |
|---|---|---|
| Aggressor | QR-DQN Stage-B s103 | spends ~80% by 25% of the auction; most early stars |
| Pace Factory | PPO Stage-A s1 | patient value buyer; most bowling-heavy RL seat |
| Run Machine | QR-DQN Stage-B s101 | early competitor, batting-leaning |
| Global Scout | QR-DQN Stage-B s102 | busy bidder that fills its overseas slots |
| Adaptive | PPO Stage-B s103 | balanced; strongest XI |

The four rule bots keep their seats. Any seat whose model is missing, corrupted, hash-mismatched or rejected plays its rule fallback.
