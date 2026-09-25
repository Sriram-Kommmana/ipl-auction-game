// Phase 2B — obs-v2 / act-v2 / reward specification integrity (A, B, C, D, K).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { DEFAULT_RULES, nextBidAmount } from '../src/rules.js'
import { fairValue } from '../src/valuation.js'
import { selectBestXI, xiTotal } from '../src/scoring.js'
import { planBid } from '../src/planning.js'
import {
    ACTIONS, ACTION_COUNT, ACT_SPEC, ACT_SPEC_HASH, BASE, CLIP, EMPTY_SLOT_PENALTY, FV_MULTIPLIERS, GAMMA, LAMBDA_REL, MAX_SAFE,
    OBS_BLOCKS, OBS_FEATURES, OBS_SIZE, OBS_SPEC, OBS_SPEC_HASH, PASS, XI_REWARD_SCALE,
    buildRlObservation, capForAction, episodeReturn, ladderFloor, specHash, stepReward, terminalReward, xiPotential
} from '../src/rl/index.js'
import { BAT, BWL, WK, ctxOf, legalXI, many, rivalsWith } from './safety.js'
import { realStates } from './rlHelpers.js'

const states = realStates([7001, 7002, 7003, 7004, 7005, 7006], 1)

// ── A. spec integrity ─────────────────────────────────────────────────────
test('A1. obs-v2: version, 80 features, block sizes 6/9/35/16/14 in order, unique names', () => {
    assert.equal(OBS_SPEC.version, 'obs-v2')
    assert.equal(OBS_SIZE, 80)
    assert.deepEqual(OBS_BLOCKS, { global: 6, player: 9, self: 35, market: 16, rivals: 14 })
    const blocks = OBS_SPEC.features.map((f) => f.block)
    const expected = Object.entries(OBS_BLOCKS).flatMap(([b, n]) => Array(n).fill(b))
    assert.deepEqual(blocks, expected, 'features grouped by block in the frozen order')
    assert.equal(new Set(OBS_FEATURES).size, 80)
    assert.deepEqual(CLIP, [-1, 5])
    assert.ok(!OBS_FEATURES.some((n) => /slno|name|id$|country|stat_|persona/i.test(n)), 'no identity, stats or persona features')
})

test('A2. spec hashes are deterministic and change when the spec changes', () => {
    assert.equal(specHash(OBS_SPEC), OBS_SPEC_HASH)
    assert.equal(specHash(JSON.parse(JSON.stringify(OBS_SPEC))), OBS_SPEC_HASH, 'hash independent of object identity')
    assert.equal(specHash(ACT_SPEC), ACT_SPEC_HASH)
    const renamed = { ...OBS_SPEC, features: OBS_SPEC.features.map((f, i) => (i === 3 ? { ...f, name: 'x' } : f)) }
    const reordered = { ...OBS_SPEC, features: [OBS_SPEC.features[1], OBS_SPEC.features[0], ...OBS_SPEC.features.slice(2)] }
    const renorm = { ...OBS_SPEC, features: OBS_SPEC.features.map((f, i) => (i === 10 ? { ...f, norm: f.norm + ' ' } : f)) }
    for (const changed of [renamed, reordered, renorm, { ...OBS_SPEC, clip: [-1, 6] }]) assert.notEqual(specHash(changed), OBS_SPEC_HASH)
    assert.notEqual(specHash({ ...ACT_SPEC, actions: ACT_SPEC.actions.slice(0, -1) }), ACT_SPEC_HASH)
    assert.match(OBS_SPEC_HASH, /^[0-9a-f]{16}$/)
})

// ── B. observation shape / order ──────────────────────────────────────────
test('B1. every real observation has exactly 80 finite values in [-1, 5]', () => {
    assert.ok(states.length > 100)
    for (const s of states) {
        assert.equal(s.obs.length, OBS_SIZE)
        for (const [i, v] of s.obs.entries()) assert.ok(Number.isFinite(v) && v >= -1 && v <= 5, `${OBS_FEATURES[i]} = ${v}`)
    }
})

test('B2. observation is a pure function of (ctx, extras): recomputing gives identical values', () => {
    for (const s of states.slice(0, 40)) assert.deepEqual(buildRlObservation(s.ctx, s.extras), s.obs)
})

test('B3. no future-order leakage: permuting the remaining players within each set, and the unsold list, changes nothing', () => {
    let checked = 0
    for (const s of states) {
        const bySet = new Map()
        for (const p of s.ctx.upcoming) {
            if (!bySet.has(p.setNo)) bySet.set(p.setNo, [])
            bySet.get(p.setNo).push(p)
        }
        // Re-auction: the whole list is one shuffled block.
        const permuted = s.ctx.phase === 'reauction'
            ? [...s.ctx.upcoming].reverse()
            : [...bySet.keys()].flatMap((k) => [...bySet.get(k)].reverse())
        const ctx2 = { ...s.ctx, upcoming: permuted, returning: [...s.ctx.returning].reverse() }
        assert.deepEqual(buildRlObservation(ctx2, s.extras), s.obs, `state ${checked}`)
        checked++
    }
    assert.ok(checked > 100)
})

// ── C. normalisation ──────────────────────────────────────────────────────
test('C1. hand-checked features on a crafted state', () => {
    const lot = BAT({ os: true, rating: 92, base: 200 })
    const ctx = ctxOf({ squad: legalXI(80), purse: 6250, lot, upcoming: many(10, BWL, { rating: 78 }), progress: 0.25, rivals: rivalsWith(3000, () => legalXI(82)) })
    const obs = buildRlObservation(ctx, { poolSize: 323, recent: [] })
    const f = Object.fromEntries(OBS_FEATURES.map((n, i) => [n, obs[i]]))
    const near = (a, b, msg) => assert.ok(Math.abs(a - b) < 1e-12, `${msg ?? ''} ${a} vs ${b}`)
    near(f.phase_reauction, 0)
    near(f.main_progress, 0.25)
    near(f.lots_left_in_phase, 10 / 323)
    near(f.purse_scale, 0)
    assert.deepEqual([f.lot_is_batsman, f.lot_is_bowler, f.lot_is_all_rounder, f.lot_is_keeper, f.lot_is_overseas], [1, 0, 0, 0, 1])
    near(f.lot_rating, 0.8)
    near(f.lot_is_star, 1)
    near(f.lot_base_price, 200 / 500)
    assert.equal(f.lot_fair_value, Math.min(4, fairValue(lot) / 500))
    near(f.self_purse, 0.5)
    near(f.self_slots_left, 14 / 25)
    near(f.self_xi_strength, 0.8)
    near(f.self_xi_empty, 0)
    near(f.self_xi_floor, 0.5)
    near(f.self_status_keeper, 0)
    near(f.mkt_recent_price_ratio, 1, 'no history → neutral')
    near(f.mkt_recent_sold_share, 1)
    near(f.riv_purse_mean, 0.24)
    near(f.riv_purse_std, 0)
    near(f.riv_xi_mean, 0.82)
    assert.ok(Math.abs(f.self_xi_gain - (12 / 11) / 10) < 1e-12)
})

test('C2. purse invariance: money features are ratios — doubling purse and prices-in-purse leaves ratio features intact', () => {
    const at = (purse) => {
        const lot = BAT({ rating: 88, base: 100 })
        const ctx = { ...ctxOf({ squad: legalXI(80), purse, lot, upcoming: many(20, BWL, { rating: 80 }), rivals: rivalsWith(purse / 2) }), rules: { ...DEFAULT_RULES, pursePerTeam: purse } }
        return Object.fromEntries(OBS_FEATURES.map((n, i) => [n, buildRlObservation(ctx, { poolSize: 323, recent: [] })[i]]))
    }
    const a = at(12500)
    const b = at(50000)
    assert.equal(a.purse_scale, 0)
    assert.equal(b.purse_scale, 1)
    assert.equal(a.self_purse, b.self_purse)
    assert.equal(a.riv_purse_mean, b.riv_purse_mean)
    assert.ok(b.lot_base_price < a.lot_base_price, 'base prices do not scale with purse, so they shrink relative to it')
})

// ── D. action mapping ─────────────────────────────────────────────────────
test('D1. act-v2: 20 actions in the frozen order', () => {
    assert.equal(ACTION_COUNT, 20)
    assert.deepEqual(FV_MULTIPLIERS, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0])
    assert.equal(ACTIONS[PASS].name, 'PASS')
    assert.equal(ACTIONS[BASE].name, 'BASE')
    assert.equal(ACTIONS[MAX_SAFE].name, 'MAX_SAFE')
    assert.equal(MAX_SAFE, 19)
    ACTIONS.slice(2, 19).forEach((a, i) => assert.equal(a.multiplier, FV_MULTIPLIERS[i]))
    assert.equal(ACT_SPEC.version, 'act-v2')
})

test('D2. effective cap = floor(min(level, maxSafeBid)); PASS = 0', () => {
    const f = { basePrice: 50, fairValue: 333, maxSafeBid: 700 }
    assert.equal(capForAction(PASS, f), 0)
    assert.equal(capForAction(BASE, f), 50)
    assert.equal(capForAction(2, f), Math.floor(0.2 * 333))
    assert.equal(capForAction(10, f), 333)
    assert.equal(capForAction(12, f), Math.floor(1.25 * 333))
    assert.equal(capForAction(18, f), 700, '3 × 333 = 999 clamped to maxSafeBid')
    assert.equal(capForAction(MAX_SAFE, f), 700)
})

test('D3. ladderFloor matches walking the real increment ladder (nextBidAmount)', () => {
    for (const base of [20, 50, 100, 150, 200, 35, 290, 310]) {
        const ladder = [base]
        while (ladder.at(-1) < 60000) ladder.push(nextBidAmount(ladder.at(-1), true))
        for (let cap = base - 5; cap < 6000; cap += 7) {
            const expect = cap < base ? null : ladder.filter((x) => x <= cap).at(-1)
            assert.equal(ladderFloor(base, cap), expect, `base ${base} cap ${cap}`)
        }
        for (const cap of [49999, 50000, 50001]) assert.equal(ladderFloor(base, cap), ladder.filter((x) => x <= cap).at(-1))
    }
})

// ── K. reward ─────────────────────────────────────────────────────────────
test('K1. reward constants: γ = 1, λ_rel = 0, scale 110, empty-slot penalty −2', () => {
    assert.equal(GAMMA, 1)
    assert.equal(LAMBDA_REL, 0)
    assert.equal(XI_REWARD_SCALE, 110)
    assert.equal(EMPTY_SLOT_PENALTY, -2)
})

test('K2. step reward = Δ unrounded XI total / 110; bench signings earn 0; terminal −2 only with an empty slot', () => {
    const squad = legalXI(84).slice(0, 10) // 10 players: one empty slot
    const before = xiPotential(squad)
    assert.equal(before, xiTotal(squad) / 110)
    const filler = BAT({ rating: 77 })
    squad.push(filler)
    assert.ok(Math.abs(stepReward(before, squad) - 77 / 110) < 1e-12, 'filling an empty slot')
    const full = xiPotential(squad)
    squad.push(BAT({ rating: 70 }))
    assert.equal(stepReward(full, squad), 0, 'bench player: no XI change, no reward')
    const up = xiPotential(squad)
    squad.push(BAT({ rating: 90 }))
    assert.ok(Math.abs(stepReward(up, squad) - (90 - 77) / 110) < 1e-12, 'upgrade pays only the difference')
    assert.equal(terminalReward(squad), 0)
    assert.equal(terminalReward([...many(5, BWL), ...many(6, BAT)]), -2, 'no keeper → empty slot → −2')
    const xi = selectBestXI(squad)
    assert.equal(xi.total, xiTotal(squad), 'unrounded total, not the 0.1-rounded strength')
})

test('K3. with γ = 1 the rewards of a real episode sum exactly to the final XI / 110 (+ terminal)', () => {
    // Covered end to end in rl-env.test.js; here the identity on a planner-built squad.
    const squad = []
    let sum = 0
    for (const p of [WK({ rating: 81 }), ...many(5, BWL, { rating: 83 }), ...many(6, BAT, { rating: 85 }), BAT({ rating: 90 })]) {
        const before = xiPotential(squad)
        squad.push(p)
        sum += stepReward(before, squad)
    }
    sum += terminalReward(squad)
    assert.ok(Math.abs(sum - episodeReturn(squad)) < 1e-12)
    void planBid
})
