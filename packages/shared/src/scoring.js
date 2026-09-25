// Team strength = the best playing XI a squad can field.
//
// Replaces the old "average rating of the whole squad", which rewarded
// buying one superstar and stopping. A playing XI must respect:
//   - at most 4 overseas players
//   - at least 1 wicket keeper
//   - at least 5 bowling options (bowlers or all-rounders)
// A role requirement the squad can't meet becomes an EMPTY slot worth 0,
// so an incomplete squad is penalised instead of flattered.
//
// Strength = (sum of the XI's ratings) / 11, rounded to 1 decimal — the same
// 0-100 scale the leaderboard already displays.
//
// The XI is found exactly, not greedily: a small knapsack-style dynamic
// program over (players picked, keeper met, bowling options met, overseas
// used). That's 12 × 2 × 6 × 5 = 720 states per player — microseconds for a
// 25-man squad, which matters because bots evaluate "how much would this
// player improve my XI?" for every lot.

import { XI_SIZE } from './rules.js'

export const XI_RULES = Object.freeze({
    size: XI_SIZE,
    minKeepers: 1,
    minBowlingOptions: 5,
    maxOverseas: 4
})

const W_DIM = 2 // keeper requirement met? (capped at 1)
const B_DIM = 6 // bowling options 0..5 (capped at 5)
const O_DIM = 5 // overseas 0..4
const C_DIM = XI_SIZE + 1
const STATE_COUNT = C_DIM * W_DIM * B_DIM * O_DIM

const stateIndex = (c, w, b, o) => ((c * W_DIM + w) * B_DIM + b) * O_DIM + o

const isKeeper = (p) => p.role === 'WICKET KEEPER'
const isBowlingOption = (p) => p.role === 'BOWLER' || p.role === 'ALL ROUNDER'
const isOverseas = (p) => p.nationality === 'Overseas'

// Deterministic order: best rating first, then lowest slNo — ties always
// resolve the same way on the server, in the browser and in training.
const byRatingThenSlNo = (a, b) => (b.rating - a.rating) || ((a.slNo ?? 0) - (b.slNo ?? 0))

// How many players an XI can actually field once unmet role requirements
// are left as empty slots.
const fillableSlots = (w, b) => XI_SIZE - (XI_RULES.minKeepers - w) - (XI_RULES.minBowlingOptions - b)

const solve = (squad, track) => {
    const players = (squad || []).filter(Boolean).slice().sort(byRatingThenSlNo)

    let dp = new Float64Array(STATE_COUNT).fill(-Infinity)
    dp[stateIndex(0, 0, 0, 0)] = 0
    const layers = track ? [] : null

    for (const p of players) {
        const wk = isKeeper(p) ? 1 : 0
        const bw = isBowlingOption(p) ? 1 : 0
        const os = isOverseas(p) ? 1 : 0
        const taken = track ? new Uint8Array(STATE_COUNT) : null
        const parent = track ? new Int16Array(STATE_COUNT) : null

        // Iterate the pick count downwards so each player is used at most
        // once per layer (classic 0/1 knapsack, updated in place).
        for (let c = XI_SIZE - 1; c >= 0; c--) {
            for (let w = 0; w < W_DIM; w++) {
                for (let b = 0; b < B_DIM; b++) {
                    for (let o = 0; o < O_DIM; o++) {
                        const s = stateIndex(c, w, b, o)
                        const v = dp[s]
                        if (v === -Infinity) continue
                        const o2 = o + os
                        if (o2 > XI_RULES.maxOverseas) continue
                        const t = stateIndex(c + 1, Math.min(1, w + wk), Math.min(5, b + bw), o2)
                        const nv = v + p.rating
                        if (nv > dp[t]) {
                            dp[t] = nv
                            if (track) {
                                taken[t] = 1
                                parent[t] = s
                            }
                        }
                    }
                }
            }
        }
        if (track) layers.push({ taken, parent })
    }

    let best = -Infinity
    let bestState = stateIndex(0, 0, 0, 0)
    for (let c = 0; c < C_DIM; c++) {
        for (let w = 0; w < W_DIM; w++) {
            for (let b = 0; b < B_DIM; b++) {
                if (c > fillableSlots(w, b)) continue
                for (let o = 0; o < O_DIM; o++) {
                    const s = stateIndex(c, w, b, o)
                    if (dp[s] > best) {
                        best = dp[s]
                        bestState = s
                    }
                }
            }
        }
    }

    let xi = null
    if (track) {
        xi = []
        let s = bestState
        for (let i = players.length - 1; i >= 0; i--) {
            if (layers[i].taken[s]) {
                xi.push(players[i])
                s = layers[i].parent[s]
            }
        }
    }

    return { total: best === -Infinity ? 0 : best, xi }
}

const DISPLAY_ORDER = { BATSMAN: 0, 'WICKET KEEPER': 1, 'ALL ROUNDER': 2, BOWLER: 3 }

// Full answer for display: the chosen XI in batting-card order, how many
// slots had to be left empty, and the strength score.
export const selectBestXI = (squad) => {
    const { total, xi } = solve(squad, true)
    const players = xi.sort(
        (a, b) => (DISPLAY_ORDER[a.role] ?? 9) - (DISPLAY_ORDER[b.role] ?? 9) || byRatingThenSlNo(a, b)
    )
    return {
        players,
        emptySlots: XI_SIZE - players.length,
        total,
        strength: Math.round((total / XI_SIZE) * 10) / 10
    }
}

export const pickBestXI = (squad) => selectBestXI(squad).players

// Unrounded rating sum of the best XI — cheaper (no path tracking), used on
// the hot path by bots and the simulator.
//
// Memoised per squad array: in the simulator a team's squad only changes
// when it buys someone, but every bot asks about every team on every lot.
// Keyed on the array itself plus its length, so a push invalidates it.
const totals = new WeakMap()

export const xiTotal = (squad) => {
    if (!squad) return 0
    const hit = totals.get(squad)
    if (hit && hit.length === squad.length) return hit.total
    const total = solve(squad, false).total
    totals.set(squad, { length: squad.length, total })
    return total
}

export const teamStrength = (squad) => Math.round((xiTotal(squad) / XI_SIZE) * 10) / 10

// How much adding `player` would lift the XI, in strength points (0-100
// scale, unrounded). 0 means the player wouldn't make the XI.
export const xiGain = (squad, player) => {
    const before = xiTotal(squad)
    const after = solve([...(squad || []), player], false).total
    return (after - before) / XI_SIZE
}
