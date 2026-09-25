// Phase 2B — canonical action mask + completion shield (E, F).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { DEFAULT_RULES, bidBlocker } from '../src/rules.js'
import { fairValue } from '../src/valuation.js'
import { planBid } from '../src/planning.js'
import { createRng } from '../src/sim.js'
import { ACTION_COUNT, BASE, MAX_SAFE, PASS, ladderFloor, rlActionMask, shieldActiveFor, validateRlDecision } from '../src/rl/index.js'
import { BAT, BWL, WK, ctxOf, legalXI, many, rivalsWith, team } from './safety.js'
import { players, realStates } from './rlHelpers.js'

const legal = (m) => m.mask.flatMap((ok, a) => (ok ? [a] : []))
const bids = (m) => legal(m).filter((a) => a !== PASS)
const onlyPass = (m) => legal(m).length === 1 && m.mask[PASS] === 1

// Every structural invariant of the canonical mask for one state.
const assertMaskInvariants = (ctx, label) => {
    const plan = planBid(ctx)
    const m = rlActionMask(ctx, plan)
    assert.equal(m.mask.length, ACTION_COUNT, label)
    const blocked = bidBlocker({ team: ctx.self, lot: { ...ctx.lot, currentBidderId: '' }, rules: ctx.rules, amount: ctx.lot.basePrice })
    if (blocked || !plan.allowed) {
        assert.ok(onlyPass(m), `${label}: blocked/disallowed → PASS only`)
        assert.equal(m.shieldActive, false, `${label}: shield never overrides legality`)
        return m
    }
    assert.equal(m.mask[BASE], 1, `${label}: BASE legal whenever the planner allows`)
    assert.equal(m.mask[PASS], m.shieldActive ? 0 : 1, `${label}: PASS masked exactly when shielded`)
    assert.equal(m.shieldActive, shieldActiveFor(plan))
    let last = -Infinity
    for (let a = 1; a < ACTION_COUNT; a++) {
        const cap = m.caps[a]
        assert.ok(cap <= plan.budget.maxSafeBid && cap <= ctx.self.purseLeft, `${label}: action ${a} cap ${cap} above maxSafeBid/purse`)
        if (m.mask[a]) {
            assert.ok(cap >= ctx.lot.basePrice, `${label}: legal action ${a} below base`)
            assert.equal(m.prices[a], ladderFloor(ctx.lot.basePrice, cap))
            assert.ok(m.prices[a] > last, `${label}: legal action ${a} duplicates a lower action's ladder price`)
            last = m.prices[a]
            assert.equal(validateRlDecision(ctx, a, cap, m), null)
        } else {
            // Masked only because below base or a duplicate ladder price.
            assert.ok(cap < ctx.lot.basePrice || m.prices[a] <= last, `${label}: action ${a} masked without reason`)
            assert.notEqual(validateRlDecision(ctx, a, cap, m), null)
        }
    }
    return m
}

// ── E. mask correctness ───────────────────────────────────────────────────
test('E1. ordinary lot: PASS, BASE and increasing fair-value levels are legal; caps follow the spec', () => {
    const lot = BAT({ rating: 90, base: 100 })
    const ctx = ctxOf({ squad: legalXI(80), purse: 8000, lot, upcoming: many(40, BWL, { rating: 80 }) })
    const m = assertMaskInvariants(ctx, 'E1')
    const fv = fairValue(lot)
    assert.equal(m.mask[PASS], 1)
    assert.equal(m.caps[BASE], 100)
    assert.equal(m.caps[10], Math.floor(fv))
    assert.equal(m.caps[18], Math.floor(3 * fv))
    assert.equal(m.caps[MAX_SAFE], planBid(ctx).budget.maxSafeBid)
    assert.ok(bids(m).length >= 12)
})

test('E2. below-base levels are masked (fair value close to base at a small purse)', () => {
    const rules = { ...DEFAULT_RULES, pursePerTeam: 5000 }
    const lot = BAT({ rating: 78, base: 100 }) // fair value at ₹5,000L floors at base
    const ctx = { ...ctxOf({ squad: legalXI(80), purse: 3000, lot, upcoming: many(40, BWL, { rating: 80 }), rivals: rivalsWith(3000) }), rules }
    const m = assertMaskInvariants(ctx, 'E2')
    assert.equal(fairValue(lot, 5000), 100)
    for (const a of [2, 3, 4, 5, 6, 7, 8, 9]) assert.equal(m.mask[a], 0, `FV level ${a} is below base`)
    assert.equal(m.mask[10], 0, '1.0 × fair value = base: same price as BASE → masked')
})

test('E3. duplicate ladder prices are masked (two caps that can only win at the same bid)', () => {
    const lot = BAT({ rating: 76, base: 20 })
    const ctx = ctxOf({ squad: legalXI(80), purse: 5000, lot, upcoming: many(40, BWL, { rating: 80 }) })
    const m = assertMaskInvariants(ctx, 'E3')
    const prices = legal(m).filter((a) => a !== PASS).map((a) => m.prices[a])
    assert.equal(new Set(prices).size, prices.length)
    assert.ok(m.caps.some((c, a) => a > BASE && !m.mask[a] && c >= 20), 'some level collapsed onto a lower price')
})

test('E4. maxSafeBid clamps every level; only the first action reaching it stays legal', () => {
    const squad = [...many(5, BWL, { rating: 84 }), ...many(5, BAT, { rating: 84 })] // needs a keeper
    const lot = BAT({ rating: 93, base: 200 })
    const ctx = ctxOf({ squad, purse: 1000, lot, upcoming: [WK({ base: 300 }), ...many(10, BAT, { base: 20, rating: 70 })] })
    const plan = planBid(ctx)
    assert.ok(plan.allowed && plan.budget.maxSafeBid < fairValue(lot))
    const m = assertMaskInvariants(ctx, 'E4')
    const clamped = bids(m).filter((a) => m.caps[a] === plan.budget.maxSafeBid)
    assert.equal(clamped.length, 1, 'one action at the safe maximum')
    assert.equal(m.mask[MAX_SAFE], clamped[0] === MAX_SAFE ? 1 : 0)
})

test('E5. rules and planner block → PASS only: full squad, overseas limit, purse exhaustion, reserve', () => {
    const cases = {
        'full squad': ctxOf({ squad: [...legalXI(80), ...many(14, BAT, { rating: 70 })], purse: 5000, lot: BAT({ rating: 95 }) }),
        'overseas limit': ctxOf({ squad: [...legalXI(80), ...many(8, BAT, { os: true, rating: 70 })], purse: 5000, lot: BAT({ os: true, rating: 95 }) }),
        'purse below base': ctxOf({ squad: legalXI(80), purse: 150, lot: BAT({ rating: 95, base: 200 }) }),
        'completion reserve': ctxOf({ squad: [...many(5, BWL, { rating: 84 }), ...many(5, BAT, { rating: 84 })], purse: 100, lot: BAT({ rating: 92, base: 20 }), upcoming: [WK({ base: 100 })] })
    }
    for (const [name, ctx] of Object.entries(cases)) assert.ok(onlyPass(assertMaskInvariants(ctx, name)), name)
    // The same Indian player is fine at 8 overseas.
    const indian = ctxOf({ squad: [...legalXI(80), ...many(8, BAT, { os: true, rating: 70 })], purse: 5000, lot: BAT({ rating: 95 }) })
    assert.ok(bids(assertMaskInvariants(indian, 'overseas limit, Indian lot')).length > 0)
})

// ── F. completion shield ──────────────────────────────────────────────────
const noKeeper = () => [...many(5, BWL, { rating: 84 }), ...many(6, BAT, { rating: 84 })]

test('F1. final opportunity (last keeper) → PASS masked; BASE still legal; unlock also flagged', () => {
    const ctx = ctxOf({ squad: noKeeper(), purse: 3000, lot: WK({ rating: 80, base: 50 }), upcoming: many(30, BAT, { rating: 72 }) })
    const plan = planBid(ctx)
    assert.equal(plan.lotContext.finalOpportunity, true)
    assert.equal(plan.playerImpact.unlocksRequirement, true, 'without him the keeper slot stays empty')
    const m = assertMaskInvariants(ctx, 'F1')
    assert.equal(m.shieldActive, true)
    assert.equal(m.mask[PASS], 0)
    assert.equal(m.mask[BASE], 1)
})

test('F2. plenty of keepers left → no shield', () => {
    const ctx = ctxOf({ squad: noKeeper(), purse: 3000, lot: WK({ rating: 80, base: 50 }), upcoming: [...many(12, WK, { rating: 80 }), ...many(20, BAT)] })
    const m = assertMaskInvariants(ctx, 'F2')
    assert.equal(m.shieldActive, false)
    assert.equal(m.mask[PASS], 1)
})

test('F3. shield rule on synthetic plans: unlock, final opportunity needs XI gain > 0.05, never when disallowed', () => {
    const base = { allowed: true, playerImpact: { unlocksRequirement: false, xiGain: 0.3, fillsRequirement: ['bowling'] }, requirements: { bowling: { finalOpportunity: true } } }
    assert.equal(shieldActiveFor(base), true)
    assert.equal(shieldActiveFor({ ...base, playerImpact: { ...base.playerImpact, xiGain: 0.05 } }), false, 'an overseas bowler the XI cannot field (gain ≤ 0.05)')
    assert.equal(shieldActiveFor({ ...base, requirements: { bowling: { finalOpportunity: false } } }), false)
    assert.equal(shieldActiveFor({ ...base, playerImpact: { unlocksRequirement: true, xiGain: 0, fillsRequirement: [] } }), true)
    assert.equal(shieldActiveFor({ ...base, allowed: false }), false)
})

test('F4. re-auction: the last returning keeper is shielded', () => {
    const ctx = ctxOf({ squad: noKeeper(), purse: 2000, lot: WK({ rating: 76, base: 50 }), upcoming: many(8, BAT, { rating: 70 }), phase: 'reauction', progress: 1 })
    const m = assertMaskInvariants(ctx, 'F4')
    assert.equal(m.shieldActive, true)
    assert.equal(m.mask[PASS], 0)
})

test('F5. shield never overrides legality: last keeper but squad full / broke / keeper is overseas at the cap', () => {
    const full = ctxOf({ squad: [...noKeeper(), ...many(14, BAT, { rating: 70 })], purse: 3000, lot: WK({ rating: 80 }), upcoming: many(5, BAT) })
    const broke = ctxOf({ squad: noKeeper(), purse: 30, lot: WK({ rating: 80, base: 50 }), upcoming: many(5, BAT) })
    const osFull = ctxOf({ squad: [...many(5, BWL, { rating: 84 }), ...many(6, BAT, { rating: 84 }), ...many(8, BAT, { os: true, rating: 70 })], purse: 3000, lot: WK({ os: true, rating: 80 }), upcoming: many(5, BAT) })
    for (const [name, ctx] of Object.entries({ full, broke, osFull })) {
        const m = assertMaskInvariants(ctx, `F5 ${name}`)
        assert.ok(onlyPass(m), name)
        assert.equal(m.shieldActive, false, name)
    }
})

// ── E/F property: real and random states ─────────────────────────────────
test('E6. mask invariants on 400+ real auction states and 1,500 random states', () => {
    let shielded = 0
    for (const s of realStates([7101, 7102, 7103], 2)) {
        const m = assertMaskInvariants(s.ctx, `real ${s.entry.seed}`)
        if (m.shieldActive) shielded++
    }
    const rng = createRng(99)
    const shuffled = (list) => {
        const a = [...list]
        for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [a[i], a[j]] = [a[j], a[i]] }
        return a
    }
    for (let i = 0; i < 1500; i++) {
        const all = shuffled(players)
        const size = rng() < 0.3 ? 18 + Math.floor(rng() * 8) : Math.floor(rng() * 26)
        const squad = []
        let os = 0
        for (const p of all) {
            if (squad.length >= size) break
            if (rng() < 0.5 || (p.nationality === 'Overseas' && os >= 8)) continue
            squad.push(p)
            if (p.nationality === 'Overseas') os++
        }
        const taken = new Set(squad.map((p) => p.slNo))
        const rest = all.filter((p) => !taken.has(p.slNo))
        const purse = [5000, 12500, 50000][i % 3]
        const rules = { ...DEFAULT_RULES, pursePerTeam: purse }
        const self = team(squad, Math.floor(rng() * purse))
        const supply = rest.slice(1, 1 + (rng() < 0.3 ? Math.floor(rng() * 5) : Math.floor(rng() * 150)))
        const phase = rng() < 0.3 ? 'reauction' : 'main'
        const cut = phase === 'main' ? Math.floor(rng() * supply.length) : supply.length
        const ctx = { rules, lot: rest[0], self, rivals: [], upcoming: supply.slice(0, cut), returning: phase === 'main' ? supply.slice(cut) : [], phase, progress: phase === 'main' ? rng() : 1 }
        if (assertMaskInvariants(ctx, `random ${i}`).shieldActive) shielded++
    }
    assert.ok(shielded > 0, 'the property run exercises the shield')
})
