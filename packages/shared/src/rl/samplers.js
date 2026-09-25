// Deterministic episode samplers (Phase 2A, frozen). Everything an episode
// varies — purse, seating, which RL seat the learner takes, the human
// proxy — is drawn from a random stream derived from the episode seed, so
// the same seed always gives the same episode and the simulator's own
// random stream (createRng(seed)) is left untouched.
//
// Product lineup (10 seats, seating order shuffled):
//   1 human proxy · the 4 frozen rule bots (one each) · the 5 RL seats.
// The learner takes one RL seat. The other four RL seats play their frozen
// rule fallback (Stage A) or, in league configuration (Stage B), a frozen
// RL snapshot with probability `snapshotShare`. The four rule-bot seats are
// fixed, so frozen rule bots are always ≥ 4/9 ≈ 44% of the opponents.
//
// Never randomised: ratings, prices, roles, pool, set order, limits,
// increments, scoring.

import { createRng } from '../sim.js'
import { RL_PERSONAS, RULE_PERSONAS } from '../personas.js'
import { deriveSeed } from './hash.js'

export const PURSE_MIN = 5000
export const PURSE_MAX = 50000
export const PURSE_NORMAL = 12500
export const PURSE_STRATA = Object.freeze({
    low: [5000, 9000],
    normal: [9000, 18000],
    high: [18000, 50000]
})
export const HUMAN_PROXIES = Object.freeze({ passive: 0.4, noisy: 0.4, starChaser: 0.2 })
export const HUMAN_NOISE = Object.freeze([0.1, 0.5])
export const MIN_RULE_OPPONENT_SHARE = 0.4

export const SPLITS = Object.freeze({
    regression: Object.freeze({ start: 0, count: 100000 }), // reserved for tests
    validation: Object.freeze({ start: 100000, count: 500 }),
    test: Object.freeze({ start: 200000, count: 1000 }),
    train: Object.freeze({ start: 1000000, count: Infinity })
})

export const splitOfSeed = (seed) => {
    for (const [name, { start, count }] of Object.entries(SPLITS)) if (seed >= start && seed < start + count) return name
    return null
}

const RULE_IDS = Object.keys(RULE_PERSONAS)
const RL_IDS = Object.keys(RL_PERSONAS)
const logUniform = (rng, lo, hi) => Math.exp(Math.log(lo) + rng() * (Math.log(hi) - Math.log(lo)))
const round100 = (x) => Math.round(x / 100) * 100
const pick = (rng, list) => list[Math.floor(rng() * list.length)]
const shuffleWith = (rng, list) => {
    const a = [...list]
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1))
        ;[a[i], a[j]] = [a[j], a[i]]
    }
    return a
}

// ⅓ low (log-uniform 5,000–9,000) · ⅓ normal (½ exactly 12,500, ½ log-uniform
// 9,000–18,000) · ⅓ high (log-uniform 18,000–50,000); rounded to ₹100L.
export const samplePurse = (rng) => {
    const u = rng()
    if (u < 1 / 3) return { stratum: 'low', purse: round100(logUniform(rng, ...PURSE_STRATA.low)) }
    if (u < 2 / 3) {
        return rng() < 0.5
            ? { stratum: 'normal', purse: PURSE_NORMAL }
            : { stratum: 'normal', purse: round100(logUniform(rng, ...PURSE_STRATA.normal)) }
    }
    return { stratum: 'high', purse: round100(logUniform(rng, ...PURSE_STRATA.high)) }
}

export const sampleHumanProxy = (rng) => {
    const u = rng()
    const noise = () => Math.round((HUMAN_NOISE[0] + rng() * (HUMAN_NOISE[1] - HUMAN_NOISE[0])) * 1000) / 1000
    if (u < HUMAN_PROXIES.passive) return { type: 'human', proxy: 'passive' }
    if (u < HUMAN_PROXIES.passive + HUMAN_PROXIES.noisy) return { type: 'human', proxy: 'noisy', persona: pick(rng, RULE_IDS), noise: noise() }
    return { type: 'human', proxy: 'starChaser', persona: 'starChaser', noise: noise() }
}

// One episode's configuration. `league` = { snapshotShare, poolSize } for
// Stage B; omitted → Stage A (every other RL seat on its rule fallback).
export const sampleEpisode = (seed, { league = null } = {}) => {
    if (!Number.isInteger(seed) || seed < 0) throw new Error(`episode seed must be a non-negative integer (${seed})`)
    const rng = createRng(deriveSeed(seed, 'episode-v2'))
    const { purse, stratum } = samplePurse(rng)
    const learnerRlSeat = pick(rng, RL_IDS)
    const human = sampleHumanProxy(rng)
    const seats = [
        human,
        ...RULE_IDS.map((persona) => ({ type: 'rule', persona })),
        ...RL_IDS.map((rlSeat) => (rlSeat === learnerRlSeat
            ? { type: 'learner', rlSeat, fallback: RL_PERSONAS[rlSeat].fallback }
            : { type: 'rlFallback', rlSeat, persona: RL_PERSONAS[rlSeat].fallback }))
    ]
    if (league) {
        validateLeague(league)
        for (const seat of seats) {
            if (seat.type === 'rlFallback' && league.poolSize > 0 && rng() < league.snapshotShare) {
                seat.type = 'rlSnapshot'
                seat.snapshot = Math.floor(rng() * league.poolSize)
            }
        }
    }
    const seating = shuffleWith(rng, seats)
    return {
        seed,
        split: splitOfSeed(seed),
        stage: league ? 'B' : 'A',
        purse,
        stratum,
        learnerSeat: seating.findIndex((s) => s.type === 'learner'),
        learnerRlSeat,
        seats: seating
    }
}

// Frozen rule bots (rule seats + RL seats on their rule fallback) as a share
// of the nine opponents.
export const ruleOpponentShare = (entry) =>
    entry.seats.filter((s) => s.type === 'rule' || s.type === 'rlFallback').length / (entry.seats.length - 1)

export const validateLeague = ({ snapshotShare, poolSize }) => {
    if (!(snapshotShare >= 0 && snapshotShare <= 1)) throw new Error(`snapshotShare must be in [0, 1] (${snapshotShare})`)
    if (!(Number.isInteger(poolSize) && poolSize >= 0)) throw new Error(`poolSize must be a non-negative integer (${poolSize})`)
    // Worst case every other RL seat is a snapshot: 4 rule seats of 9 opponents.
    const worst = RULE_IDS.length / (RULE_IDS.length + RL_IDS.length)
    if (worst < MIN_RULE_OPPONENT_SHARE) throw new Error(`rule bots would fall to ${worst} of opponents`)
    return true
}

export const manifestSeeds = (split, count = SPLITS[split].count) => {
    const { start, count: size } = SPLITS[split]
    if (!Number.isFinite(count) || count > size) throw new Error(`${split}: count ${count} exceeds split size ${size}`)
    return Array.from({ length: count }, (_, i) => start + i)
}

export const generateManifest = (split, count) => ({
    format: 'rl-manifest-v2',
    split,
    sampler: 'sampleEpisode(seed) — packages/shared/src/rl/samplers.js',
    entries: manifestSeeds(split, count).map((seed) => sampleEpisode(seed))
})
