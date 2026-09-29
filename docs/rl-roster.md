# Gameplay audit and production RL roster

These tools evaluate the already-trained RL exports **as game opponents**. They play complete Full Pool auctions (main round + re-auction) using:
- the unchanged `AuctionSim`;
- the frozen rule bots;
- the production seat runtime (`createRlSeat`: act-v3 mask and shield, cap ≤ maxSafeBid, the 20 ms room guard, the completion guard and the rule fallback).

Nothing is trained or written back.

The scripts below, and every `ml/…` path in this document (training runs, reports), now live under `_archive/ml/` (`_archive/ml/ipl_rl/gameplay/` for the scripts). They are archived research tooling and are not guaranteed to run from there. The production game needs none of them.

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
  - **Live server, 3 concurrent solo games on a Windows laptop, same load both times:**

    | Server process | RL guard trips | Latency p99 / max | Slowest GC pause |
    |---|---|---|---|
    | Throttled by Windows (EcoQoS, the default for a background process) | 2 of 15 seats (24.2 ms, 43.5 ms) | 2.3–7.7 ms / 21.5 ms | 25.8 ms |
    | Throttling lifted from the server process only | 0 of 15 seats | 1.0–3.0 ms / 4.4 ms | 3.2 ms |

    Neither trip overlapped a GC pause: the process was waiting for the CPU. The guard is working as designed, and the models are not slow. On Linux there is no EcoQoS. On a shared-CPU VPS, CPU steal would be the analogous risk.
  - **Checking a deployment.** The server logs each finished solo auction as `[bots] room … auction complete`, with every RL seat's decision sources and its latency p50/p99/max, plus the slowest GC pause. A guard trip is logged when it happens, followed by a line saying whether a GC pause overlapped the decision. Play a game or two on the VPS, then `grep '\[bots\]'` the server log: `sources {"rl":N}` with no `fallback` count means the seat played its model all game.

## Production roster (`apps/server/src/bots/models/registry.json`)

| RL persona (id) | Model | How it plays (roster check, 400 rooms) |
|---|---|---|
| Aggressor (`aggressor`) | QR-DQN Stage-B s103 | spends ~80% by 25% of the auction; pays 1.12× fair value |
| Bargain Hunter (`paceFirst`) | PPO Stage-A s1 | spends 6% by 25% of the auction, then buys stars at 0.74× fair value; hoards keepers (5.6) |
| Fast Starter (`battingFirst`) | QR-DQN Stage-B s101 | early competitor (71% spent by 25%) at about fair value; 39% of its squad are all-rounders |
| Price Pusher (`overseasSpecialist`) | QR-DQN Stage-B s102 | sets a bid cap on 99% of its decisions (other RL seats 42–84%), so it pushes up prices on almost every lot |
| Adaptive (`adaptive`) | PPO Stage-B s103 | balanced; strongest XI |

The four rule bots keep their seats. Any seat whose model is missing, corrupted, hash-mismatched or rejected plays its rule fallback.

The display names and blurbs in `packages/shared/src/personas.js` were changed after Phase 2E.0 to describe this roster. The old names (Pace Factory, Run Machine, Global Scout) promised bowling, batting and overseas focus, which none of the models show: bowling, batting and overseas shares are within the rule bots' range. Persona ids, vectors and fallbacks are unchanged. As a result, `hashes.mjs --check` against the Phase 2E.0 record reports `personas.js` as modified, which is expected.
