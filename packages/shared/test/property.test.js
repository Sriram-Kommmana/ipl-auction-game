// Rule-bot freeze validation — randomized property tests. Random legal team
// states (real players, random purses, squads, overseas counts, supplies,
// rivals, phases); every state must satisfy the safety invariants in
// safety.js, and the planner's feasibility answers must match brute force.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { parsePlayersCsv } from '../src/playersCsv.js'
import { DEFAULT_RULES } from '../src/rules.js'
import { selectBestXI } from '../src/scoring.js'
import { minimumCostToCompleteXI, planBid } from '../src/planning.js'
import { createRng } from '../src/sim.js'
import { assertDeterministic, assertSafeDecision, team } from './safety.js'

const players = parsePlayersCsv(readFileSync(fileURLToPath(new URL('../../../apps/server/src/db/players.csv', import.meta.url)), 'utf8'))
const isOs = (p) => p.nationality === 'Overseas'

// A random legal squad: ≤ maxPlayers, ≤ maxOverseas, no player twice.
const randomSquad = (rng, pool, size, rules) => {
    const squad = []
    let os = 0
    for (const p of pool) {
        if (squad.length >= size) break
        if (rng() > 0.5) continue
        if (isOs(p) && os >= rules.maxOverseas) continue
        squad.push(p)
        if (isOs(p)) os++
    }
    return squad
}
const shuffled = (rng, list) => {
    const a = [...list]
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1))
        ;[a[i], a[j]] = [a[j], a[i]]
    }
    return a
}
const randomPurse = (rng) => {
    const r = rng()
    if (r < 0.2) return Math.floor(rng() * 300) // nearly broke
    if (r < 0.4) return Math.floor(rng() * 1500)
    return Math.floor(rng() * DEFAULT_RULES.pursePerTeam)
}
const sizeBiased = (rng) => (rng() < 0.3 ? 20 + Math.floor(rng() * 6) : Math.floor(rng() * 26))

const randomState = (rng) => {
    const rules = DEFAULT_RULES
    const all = shuffled(rng, players)
    const selfSquad = randomSquad(rng, all, sizeBiased(rng), rules)
    const taken = new Set(selfSquad.map((p) => p.slNo))
    const rest = all.filter((p) => !taken.has(p.slNo))
    const lot = rest[0]
    const phase = rng() < 0.25 ? 'reauction' : 'main'
    // Supply: anything from the last lot of the auction to most of the pool.
    const supplySize = rng() < 0.3 ? Math.floor(rng() * 6) : Math.floor(rng() * 200)
    const supply = rest.slice(1, 1 + supplySize)
    const cut = phase === 'main' ? Math.floor(rng() * supply.length) : supply.length
    const rivals = Array.from({ length: 9 }, (_, i) => {
        const squad = randomSquad(rng, shuffled(rng, players), sizeBiased(rng), rules)
        return team(squad, randomPurse(rng), `R${i}`)
    })
    return {
        rules,
        lot,
        self: team(selfSquad, randomPurse(rng)),
        rivals,
        upcoming: supply.slice(0, cut),
        returning: phase === 'main' ? supply.slice(cut) : [],
        phase,
        progress: phase === 'main' ? rng() : 1
    }
}

test('property: 4000 random states — every safety invariant for every personality and noise extreme', () => {
    const rng = createRng(20260925)
    let bids = 0
    for (let i = 0; i < 4000; i++) {
        const ctx = randomState(rng)
        const { caps } = assertSafeDecision(ctx, `state ${i}`)
        assertDeterministic(ctx, `state ${i}`)
        bids += Object.values(caps).filter((c) => c > 0).length
    }
    assert.ok(bids > 2000, `the generator must produce plenty of real bids (${bids})`)
})

// Brute force: cheapest set of supply players (≤ slots, overseas within the
// cap, base prices) giving a squad whose Best XI has no empty slot.
const bruteCompletion = (squad, supply, slots, osSlots) => {
    let best = Infinity
    for (let mask = 0; mask < 1 << supply.length; mask++) {
        const pick = supply.filter((_, j) => mask & (1 << j))
        if (pick.length > slots || pick.filter(isOs).length > osSlots) continue
        const cost = pick.reduce((s, p) => s + p.basePrice, 0)
        if (cost >= best) continue
        if (selectBestXI([...squad, ...pick]).emptySlots === 0) best = cost
    }
    return best
}

test('property: 250 random small markets — planner feasibility equals brute force (never claims an impossible XI possible, or a feasible one impossible)', () => {
    const rng = createRng(4242)
    const seen = { feasible: 0, infeasible: 0, impossible: 0, checkedAfter: 0 }
    for (let i = 0; i < 250; i++) {
        const all = shuffled(rng, players)
        const squad = randomSquad(rng, all, Math.floor(rng() * 25), DEFAULT_RULES)
        const taken = new Set(squad.map((p) => p.slNo))
        const rest = all.filter((p) => !taken.has(p.slNo))
        const supply = rest.slice(1, 1 + Math.floor(rng() * 11))
        const slots = DEFAULT_RULES.maxPlayers - squad.length
        const osSlots = DEFAULT_RULES.maxOverseas - squad.filter(isOs).length
        const brute = bruteCompletion(squad, supply, slots, osSlots)
        // Purses around the exact completion cost are the interesting edge.
        const near = Number.isFinite(brute) && rng() < 0.5
        const self = team(squad, near ? Math.max(0, brute + Math.floor(rng() * 21) - 10) : randomPurse(rng))
        const ctx = { rules: DEFAULT_RULES, lot: rest[0], self, rivals: [], upcoming: supply, returning: [], phase: 'main', progress: 0.9 }
        const planner = minimumCostToCompleteXI(ctx)
        assert.equal(planner.minimumCost ?? Infinity, brute, `case ${i}: cost`)
        assert.equal(planner.feasible, brute <= self.purseLeft, `case ${i}: feasible`)
        seen[brute === Infinity ? 'impossible' : brute <= self.purseLeft ? 'feasible' : 'infeasible']++
        // The plan's "after purchase" answer against brute force with the lot added.
        const plan = planBid(ctx)
        if (plan.allowed && plan.completion.reachableEmptySlots === 0) {
            const bruteAfter = bruteCompletion([...squad, ctx.lot], supply, slots - 1, osSlots - (isOs(ctx.lot) ? 1 : 0))
            assert.equal(plan.budget.requiredReserve, bruteAfter, `case ${i}: reserve after purchase`)
            assert.equal(plan.budget.maxSafeBid, self.purseLeft - bruteAfter, `case ${i}: maxSafeBid`)
            seen.checkedAfter++
        }
    }
    // The generator must exercise every outcome, not just the easy one.
    for (const [k, n] of Object.entries(seen)) assert.ok(n >= 10, `${k}: ${n} cases`)
})
