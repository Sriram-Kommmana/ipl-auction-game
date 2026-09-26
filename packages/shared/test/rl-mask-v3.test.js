// Phase 2C.2 — CANDIDATE completion shield v2 (not adopted): deterministic
// adversarial cases 1–24, the replayed failure, and property tests.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { AuctionSim, createRng } from '../src/sim.js'
import { DEFAULT_RULES } from '../src/rules.js'
import { planBid } from '../src/planning.js'
import { PASS, rlActionMask, rlActionMaskV2 } from '../src/rl/index.js'
import { AR, BAT, BWL, WK, ctxOf, legalXI, many, rivalsWith } from './safety.js'
import { realStates } from './rlHelpers.js'

// ── fixtures ──────────────────────────────────────────────────────────────
const noKeeper10 = () => [...many(5, BWL, { rating: 85 }), ...many(5, BAT, { rating: 85 })]
const settled = (purse = 6000) => rivalsWith(purse, () => legalXI(80))
const keeper = (o = {}) => WK({ base: 20, rating: 75, ...o })
const fillers = (n = 20, o = {}) => many(n, BAT, { base: 20, rating: 60, ...o })
const scenario = ({ purse, lot, upcoming = fillers(), returning = [], rivals = settled(), phase = 'main', squad = noKeeper10() }) =>
    ctxOf({ squad, purse, lot, upcoming, returning, rivals, phase, progress: phase === 'main' ? 0.9 : 1 })
const legal = (m) => m.mask.flatMap((ok, a) => (ok ? [a] : []))
const bidCaps = (m) => legal(m).filter((a) => a !== PASS).map((a) => m.caps[a])
const both = (ctx) => ({ v1: rlActionMask(ctx), v2: rlActionMaskV2(ctx) })

// Resolve one lot on the real ladder: team 0 = learner, team 1 = rival.
const ladder = (lot, learner, rival, rngValue) => {
    const sim = new AuctionSim({ players: [lot], teamCount: 2, rng: () => rngValue })
    sim.pool = [lot.slNo]
    sim.teams[0].purseLeft = learner.purse
    sim.teams[1].purseLeft = rival.purse
    return sim.resolveLot([learner.cap, rival.cap])
}

// Structural checks every v2 result must pass.
const assertSound = (ctx, label) => {
    const plan = planBid(ctx)
    const v1 = rlActionMask(ctx, plan)
    const v2 = rlActionMaskV2(ctx, plan)
    for (let a = 0; a < v1.mask.length; a++) if (v2.mask[a]) assert.equal(v1.mask[a], 1, `${label}: v2 legalised action ${a}`)
    assert.ok(legal(v2).length >= 1, `${label}: no legal action`)
    if (v2.mask[PASS] === 0) assert.ok(plan.allowed && bidCaps(v2).length > 0, `${label}: forced without an allowed bid`)
    for (const c of bidCaps(v2)) assert.ok(c >= ctx.lot.basePrice && c <= plan.budget.maxSafeBid && c <= ctx.self.purseLeft, `${label}: cap ${c}`)
    assert.ok(['SAFE', 'WARNING', 'CRITICAL', 'IMPOSSIBLE'].includes(v2.shield.state))
    return { v1, v2, plan }
}

// ── keeper (1–7) ──────────────────────────────────────────────────────────
test('1. final keeper at ₹20L: forced but the tie is unwinnable — and v2 prevents reaching it', () => {
    const lot = keeper()
    const { v1, v2 } = assertSound(scenario({ purse: 20, lot }), '1')
    assert.equal(v1.mask[PASS], 0, 'frozen shield forces the final opportunity too')
    assert.equal(v2.shield.state, 'CRITICAL')
    assert.deepEqual(bidCaps(v2), [20])
    assert.equal(v2.shield.finalPath, true)
    // The discovered failure: a rival at base who leads first cannot be beaten.
    assert.equal(ladder(lot, { purse: 20, cap: 20 }, { purse: 6000, cap: 24 }, 0.99).winner, 1)
    // Prevention: with ₹40L and a keeper still needed, v1 lets the learner
    // spend ₹20L on a non-keeper (leaving exactly the base-price floor); v2 does not.
    const earlier = scenario({ purse: 40, lot: BAT({ base: 20, rating: 60 }), upcoming: [keeper(), ...fillers()] })
    const e = assertSound(earlier, '1-prevent')
    assert.ok(bidCaps(e.v1).includes(20))
    assert.deepEqual(legal(e.v2), [PASS])
})

test('2. final keeper tie with ₹40L: forced at the highest safe bid (₹40L), beats a base-price rival whoever leads first', () => {
    const lot = keeper()
    const { v2 } = assertSound(scenario({ purse: 40, lot }), '2')
    assert.equal(v2.mask[PASS], 0)
    assert.equal(v2.shield.finalPath, true)
    assert.deepEqual(bidCaps(v2), [40], 'final path: the highest safe bid only')
    for (const r of [0, 0.99]) assert.equal(ladder(lot, { purse: 40, cap: 40 }, { purse: 6000, cap: 24 }, r).winner, 0)
    for (const r of [0, 0.99]) assert.equal(ladder(lot, { purse: 40, cap: 40 }, { purse: 6000, cap: 30 }, r).winner, 0)
})

test('3. final keeper tie with ₹60L: forced at ₹60L, beats a rival capped at ₹50L', () => {
    const lot = keeper()
    const { v2 } = assertSound(scenario({ purse: 60, lot }), '3')
    assert.equal(v2.mask[PASS], 0)
    assert.deepEqual(bidCaps(v2), [60])
    for (const r of [0, 0.99]) assert.equal(ladder(lot, { purse: 60, cap: 60 }, { purse: 6000, cap: 50 }, r).winner, 0)
})

test('4. two keeper candidates left (this + one): CRITICAL and forced (not final: floor one increment over base); the frozen shield still allows PASS', () => {
    const { v1, v2 } = assertSound(scenario({ purse: 200, lot: keeper(), upcoming: [keeper(), ...fillers()] }), '4')
    assert.equal(v1.mask[PASS], 1)
    assert.equal(v2.shield.requirements.keeper.state, 'CRITICAL')
    assert.equal(v2.mask[PASS], 0)
    assert.equal(v2.shield.finalPath, false)
    assert.equal(v2.shield.floor, 30)
    assert.ok(bidCaps(v2).length > 1 && bidCaps(v2).every((c) => c >= 30), 'the policy still chooses how much, above the floor')
})

test('5. three keeper candidates left: WARNING — not forced', () => {
    const { v2 } = assertSound(scenario({ purse: 200, lot: keeper(), upcoming: [keeper(), keeper(), ...fillers()] }), '5')
    assert.equal(v2.shield.requirements.keeper.state, 'WARNING')
    assert.equal(v2.mask[PASS], 1)
})

test('6. several cheap keepers left: SAFE — no forced purchase', () => {
    const { v2 } = assertSound(scenario({ purse: 200, lot: keeper(), upcoming: [...many(8, WK, { base: 20, rating: 70 }), ...fillers()] }), '6')
    assert.equal(v2.shield.requirements.keeper.state, 'SAFE')
    assert.equal(v2.mask[PASS], 1)
    assert.equal(v2.shield.forced, false)
})

test('7. expensive keeper (base ₹200L): not final → floor ₹220L; final → the highest safe bid', () => {
    const notFinal = assertSound(scenario({ purse: 1000, lot: keeper({ base: 200, rating: 88 }), upcoming: [keeper(), ...fillers()] }), '7a')
    assert.equal(notFinal.v2.shield.floor, 220)
    assert.ok(bidCaps(notFinal.v2).every((c) => c >= 220 && c <= notFinal.plan.budget.maxSafeBid))
    const { v2, plan } = assertSound(scenario({ purse: 1000, lot: keeper({ base: 200, rating: 88 }) }), '7b')
    assert.equal(v2.mask[PASS], 0)
    assert.deepEqual(bidCaps(v2), [plan.budget.maxSafeBid])
})

// ── other requirements (8–12) ─────────────────────────────────────────────
test('8. final Indian: overseas XI places full, last Indian candidate is forced', () => {
    const squad = [WK({ rating: 80 }), ...many(5, BWL, { rating: 85 }), ...many(4, BAT, { os: true, rating: 88 })]
    const { v2 } = assertSound(scenario({ purse: 200, squad, lot: BAT({ base: 20, rating: 70 }), upcoming: fillers(20, { os: true }) }), '8')
    assert.ok(v2.shield.forcedBy.includes('indians'), JSON.stringify(v2.shield.forcedBy))
    assert.equal(v2.mask[PASS], 0)
})

test('9. final bowling option: forced', () => {
    const squad = [WK({ rating: 80 }), ...many(4, BWL, { rating: 85 }), ...many(5, BAT, { rating: 85 })]
    const { v2 } = assertSound(scenario({ purse: 200, squad, lot: BWL({ base: 20, rating: 70 }) }), '9')
    assert.ok(v2.shield.forcedBy.includes('bowling'))
    assert.equal(v2.mask[PASS], 0)
})

test('10. final batting / XI place (the "players" requirement — the XI has no separate batting rule): forced', () => {
    const squad = [WK({ rating: 80 }), ...many(5, BWL, { rating: 85 }), ...many(4, BAT, { rating: 85 })]
    const { v2 } = assertSound(scenario({ purse: 200, squad, lot: BAT({ base: 20, rating: 70 }), upcoming: [] }), '10')
    assert.ok(v2.shield.forcedBy.includes('players'))
    assert.equal(v2.mask[PASS], 0)
})

test('11. two simultaneous critical requirements (keeper + bowling): forced; the other need keeps its margin unless this is the final path', () => {
    const squad = [...many(4, BWL, { rating: 85 }), ...many(5, BAT, { rating: 85 })]
    const upcoming = [BWL({ base: 20, rating: 70 }), ...fillers()]
    // Not final (a second keeper is still to come): forced, and the bowling margin is kept.
    const nf = assertSound(scenario({ purse: 100, squad, lot: keeper(), upcoming: [keeper(), ...upcoming] }), '11a')
    assert.equal(nf.v2.shield.requirements.keeper.state, 'CRITICAL')
    assert.equal(nf.v2.shield.requirements.bowling.state, 'CRITICAL')
    assert.equal(nf.v2.mask[PASS], 0)
    assert.ok(Math.max(...bidCaps(nf.v2)) <= nf.plan.budget.maxSafeBid - 10, 'bowling margin kept')
    // Final keeper: the highest safe bid — maxSafeBid still reserves the base-price bowler.
    const { v2, plan } = assertSound(scenario({ purse: 100, squad, lot: keeper(), upcoming }), '11b')
    assert.equal(v2.shield.finalPath, true)
    assert.deepEqual(bidCaps(v2), [plan.budget.maxSafeBid])
    assert.ok(plan.budget.maxSafeBid <= 100 - 20, 'the base-price bowler stays reserved')
})

test('12. three simultaneous critical requirements (keeper, bowling, Indians): forced, sound', () => {
    const squad = [...many(4, BAT, { os: true, rating: 88 }), ...many(3, BWL, { rating: 85 }), ...many(2, BAT, { rating: 85 })]
    const { v2 } = assertSound(scenario({ purse: 300, squad, lot: keeper(), upcoming: [...many(2, BWL, { base: 20, rating: 70 }), ...fillers(20, { os: true })] }), '12')
    for (const r of ['keeper', 'bowling', 'indians']) assert.equal(v2.shield.requirements[r].state, 'CRITICAL', r)
    assert.equal(v2.mask[PASS], 0)
})

// ── phases (13–15) ────────────────────────────────────────────────────────
test('13–15. main round, re-auction and the main → re-auction transition (returning players count as supply)', () => {
    const two = [keeper(), ...fillers()]
    assert.equal(rlActionMaskV2(scenario({ purse: 200, lot: keeper(), upcoming: two, phase: 'main' })).mask[PASS], 0, '13 main')
    assert.equal(rlActionMaskV2(scenario({ purse: 200, lot: keeper(), upcoming: two, phase: 'reauction' })).mask[PASS], 0, '14 re-auction')
    const oneReturning = rlActionMaskV2(scenario({ purse: 200, lot: keeper(), upcoming: fillers(), returning: [keeper()] }))
    assert.equal(oneReturning.shield.requirements.keeper.viableAfterLot, 1)
    assert.equal(oneReturning.mask[PASS], 0, '15 one keeper left, only in the re-auction')
    const manyReturning = rlActionMaskV2(scenario({ purse: 200, lot: keeper(), upcoming: fillers(), returning: many(8, WK, { base: 20, rating: 70 }) }))
    assert.equal(manyReturning.shield.requirements.keeper.state, 'SAFE', '15 plenty returning')
    assert.equal(manyReturning.mask[PASS], 1)
})

// ── rival pressure (16–19) ────────────────────────────────────────────────
test('16. rivals who also need a keeper make the boundary earlier', () => {
    const upcoming = [...many(4, WK, { base: 20, rating: 70 }), ...fillers()]
    const calm = rlActionMaskV2(scenario({ purse: 200, lot: keeper(), upcoming }))
    const needy = rlActionMaskV2(scenario({ purse: 200, lot: keeper(), upcoming, rivals: rivalsWith(6000, (i) => (i < 3 ? noKeeper10() : legalXI(80))) }))
    assert.equal(calm.shield.requirements.keeper.state, 'WARNING')
    assert.equal(needy.shield.requirements.keeper.rivalsNeeding, 3)
    assert.equal(needy.shield.requirements.keeper.state, 'CRITICAL')
    assert.equal(needy.mask[PASS], 0)
})

test('17 & 19. rivals able to contest (high purse): floor above base on a forced lot', () => {
    const { v2 } = assertSound(scenario({ purse: 100, lot: keeper(), upcoming: [keeper(), ...fillers()], rivals: settled(20000) }), '17')
    assert.equal(v2.shield.floor, 30)
    assert.ok(bidCaps(v2).every((c) => c >= 30))
})

test('18. rivals too poor to contest: no floor, no margin, only the true final opportunity is forced', () => {
    const poor = settled(10)
    const two = assertSound(scenario({ purse: 100, lot: keeper(), upcoming: [keeper(), ...fillers()], rivals: poor }), '18a')
    assert.equal(two.v2.mask[PASS], 1, 'no contest risk: the second-to-last keeper is not forced')
    const last = assertSound(scenario({ purse: 100, lot: keeper(), rivals: poor }), '18b')
    assert.equal(last.v2.mask[PASS], 0)
    assert.equal(last.v2.shield.floor, null)
    assert.equal(last.v2.shield.margin, 0)
    assert.ok(bidCaps(last.v2).includes(20), 'base bid allowed when nobody can contest')
})

// ── purse edge cases (20–24): a non-keeper lot while a keeper is still needed
test('20–24. purse ₹20L / ₹40L / ₹100L / ₹200L / comfortable: the margin never lets spending reach the base-price floor', () => {
    const lot = BAT({ base: 20, rating: 60 })
    const upcoming = [...many(8, WK, { base: 20, rating: 70 }), ...fillers()]
    const run = (purse) => assertSound(scenario({ purse, lot, upcoming }), `purse ${purse}`)
    const p20 = run(20)
    assert.deepEqual(legal(p20.v1), [PASS], '20: planner already blocks')
    assert.deepEqual(legal(p20.v2), [PASS])
    const p40 = run(40)
    assert.ok(bidCaps(p40.v1).includes(20), '40: frozen mask allows spending to ₹20L')
    assert.deepEqual(legal(p40.v2), [PASS], '40: v2 keeps the contest margin')
    for (const purse of [100, 200, 5000]) {
        const r = run(purse)
        assert.equal(r.v2.mask[PASS], 1)
        assert.ok(Math.max(...bidCaps(r.v2)) <= r.plan.budget.maxSafeBid - r.v2.shield.margin, `${purse}: capped by the margin`)
        assert.ok(purse - Math.max(...bidCaps(r.v2)) >= 30, `${purse}: ≥ ₹30L left for a contested keeper`)
    }
})

test('purse fragility: plenty of keepers left but no slack above the reserve beyond the contest margin → forced', () => {
    const upcoming = [...many(8, WK, { base: 20, rating: 70 }), ...fillers()]
    const fragile = assertSound(scenario({ purse: 30, lot: keeper(), upcoming }), 'fragile')
    assert.equal(fragile.v2.shield.requirements.keeper.state, 'SAFE', 'by count it is safe')
    assert.equal(fragile.v2.shield.fragile, true)
    assert.equal(fragile.v1.mask[PASS], 1)
    assert.equal(fragile.v2.mask[PASS], 0)
    assert.ok(fragile.v2.shield.forcedBy.includes('keeper(purse)'))
    const comfortable = assertSound(scenario({ purse: 200, lot: keeper(), upcoming }), 'comfortable')
    assert.equal(comfortable.v2.shield.fragile, false)
    assert.equal(comfortable.v2.mask[PASS], 1)
})

test('joint completion class: plenty of bowlers and plenty of Indians, but the XI needs an INDIAN bowler and few are left → forced', () => {
    // Squad: keeper, 4 Indian bowlers, 4 overseas batsmen (XI overseas places full), 2 Indian batsmen = 11 → XI needs a 5th bowling option who is Indian.
    const squad = [WK({ rating: 80 }), ...many(4, BWL, { rating: 85 }), ...many(4, BAT, { os: true, rating: 88 }), ...many(1, BAT, { rating: 85 })]
    const upcoming = [BWL({ base: 20, rating: 70 }), ...many(6, BWL, { os: true, base: 20, rating: 70 }), ...fillers(10)]
    const { v1, v2 } = assertSound(scenario({ purse: 300, squad, lot: BWL({ base: 20, rating: 72 }), upcoming }), 'joint')
    assert.notEqual(v2.shield.requirements.bowling?.state, 'CRITICAL', 'by the bowling count alone it looks fine')
    assert.equal(v2.shield.requirements['class:Bi'].state, 'CRITICAL')
    assert.equal(v1.mask[PASS], 1)
    assert.equal(v2.mask[PASS], 0)
    assert.ok(v2.shield.forcedBy.includes('class:Bi'))
    // With plenty of Indian bowlers still to come, nothing is forced.
    const plenty = rlActionMaskV2(scenario({ purse: 300, squad, lot: BWL({ base: 20, rating: 72 }), upcoming: [...many(8, BWL, { base: 20, rating: 70 }), ...fillers(10)] }))
    assert.equal(plenty.mask[PASS], 1)
})

test('IMPOSSIBLE is recorded as already infeasible — the shield does not pretend to repair it', () => {
    const { v2 } = assertSound(scenario({ purse: 200, lot: BAT({ base: 20, rating: 60 }), upcoming: fillers() }), 'impossible')
    assert.equal(v2.shield.requirements.keeper.state, 'IMPOSSIBLE')
    assert.equal(v2.shield.alreadyInfeasible, true)
})

// ── properties ────────────────────────────────────────────────────────────
test('properties on real auction states: v2 ⊆ v1, always a legal action, forced ⇒ allowed, caps ≤ maxSafeBid, no dependence on lot ORDER', () => {
    let n = 0
    let forced = 0
    const rng = createRng(99)
    for (const s of realStates([7201, 7202, 7203, 7204, 7205, 7206, 7207, 7208, 7209, 7210], 1)) {
        const { v2 } = assertSound(s.ctx, `real ${n}`)
        if (v2.mask[PASS] === 0) forced++
        // Future-order leakage: shuffling the upcoming lots must not change anything.
        const shuffled = [...s.ctx.upcoming].sort(() => rng() - 0.5)
        const again = rlActionMaskV2({ ...s.ctx, upcoming: shuffled })
        assert.deepEqual(again.mask, v2.mask, `real ${n}: order changed the mask`)
        assert.equal(again.shield.state, v2.shield.state)
        n++
    }
    assert.ok(n > 300, `${n} states`)
    assert.ok(forced >= 0)
})

test('properties on 1,500 random states (random squads, purses, supplies, rivals, phases)', () => {
    const rng = createRng(2027)
    const roles = [WK, BAT, BWL, AR]
    const pick = (xs) => xs[Math.floor(rng() * xs.length)]
    const player = () => pick(roles)({ os: rng() < 0.3, base: pick([20, 20, 30, 50, 100, 200]), rating: 60 + Math.floor(rng() * 36) })
    for (let i = 0; i < 1500; i++) {
        const squad = []
        for (let k = Math.floor(rng() * 16); k > 0; k--) {
            const p = player()
            if (p.nationality === 'Overseas' && squad.filter((q) => q.nationality === 'Overseas').length >= 8) p.nationality = 'Indian'
            squad.push(p)
        }
        const ctx = ctxOf({
            squad, purse: pick([20, 30, 40, 60, 100, 200, 500, 2000, 8000]), lot: player(),
            upcoming: Array.from({ length: Math.floor(rng() * 40) }, player), returning: Array.from({ length: Math.floor(rng() * 10) }, player),
            rivals: rivalsWith(() => pick([10, 40, 300, 5000]), () => Array.from({ length: Math.floor(rng() * 14) }, () => ({ ...player(), nationality: 'Indian' }))),
            phase: rng() < 0.3 ? 'reauction' : 'main', rules: DEFAULT_RULES
        })
        assertSound(ctx, `random ${i}`)
    }
})

// ── production defence in depth (option C) ────────────────────────────────
test('completionGuard (production, opt-in): raises the cap on a completion boundary, never above maxSafeBid; off = unchanged', async () => {
    const { createRlSeat } = await import('../src/rl/index.js')
    const { randomPolicy } = await import('./rlHelpers.js')
    const policy = randomPolicy('ppo', 5)
    const alwaysPass = (p, scores, mask) => (mask[PASS] ? PASS : mask.findIndex((m, a) => m && a !== PASS))
    const extras = { poolSize: 300, recent: [] }
    const boundary = scenario({ purse: 100, lot: keeper(), upcoming: [keeper(), ...fillers()] }) // 2 keepers left → CRITICAL
    const calm = scenario({ purse: 100, lot: keeper(), upcoming: [...many(8, WK, { base: 20, rating: 70 }), ...fillers()] })
    const off = createRlSeat({ policy, fallbackPersona: 'moneyball', select: alwaysPass })
    const on = createRlSeat({ policy, fallbackPersona: 'moneyball', select: alwaysPass, completionGuard: true })
    assert.equal(off.decide(boundary, extras, () => 0.5).cap, 0, 'raw policy passes')
    const guarded = on.decide(boundary, extras, () => 0.5)
    assert.equal(guarded.source, 'guard')
    assert.ok(guarded.cap >= 30 && guarded.cap <= planBid(boundary).budget.maxSafeBid, `cap ${guarded.cap}`)
    assert.equal(on.state.guardInterventions, 1)
    const quiet = on.decide(calm, extras, () => 0.5)
    assert.equal(quiet.source, 'rl')
    assert.equal(quiet.cap, 0, 'no boundary: the policy decides')
})
