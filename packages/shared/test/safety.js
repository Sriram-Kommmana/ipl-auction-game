// Shared safety checks for the rule-bot validation suites (adversarial.test.js,
// property.test.js). Not a test file itself — the runner only picks up *.test.js.
import assert from 'node:assert/strict'
import { DEFAULT_RULES, bidBlocker } from '../src/rules.js'
import { ruleBotCap, RULE_BOT_IDS } from '../src/ruleBots.js'
import { completionCosts, legalCompletionCost, planBid, supplyOf, teamComposition } from '../src/planning.js'

// ── fixtures ──────────────────────────────────────────────────────────────
let nextSl = 50000
export const P = (role, { os = false, base = 100, rating = 80 } = {}) => ({
    slNo: nextSl++, playerName: `P${nextSl}`, role, nationality: os ? 'Overseas' : 'Indian', basePrice: base, rating
})
export const WK = (o) => P('WICKET KEEPER', o)
export const BAT = (o) => P('BATSMAN', o)
export const BWL = (o) => P('BOWLER', o)
export const AR = (o) => P('ALL ROUNDER', o)
export const many = (n, make, o) => Array.from({ length: n }, () => make(o))
export const legalXI = (rating = 85) => [WK({ rating }), ...many(5, BWL, { rating }), ...many(5, BAT, { rating })]
export const team = (squad, purse, teamId = 'ME') => ({
    teamId, purseLeft: purse, purseSpent: DEFAULT_RULES.pursePerTeam - purse, playerCount: squad.length,
    overseasCount: squad.filter((p) => p.nationality === 'Overseas').length, squad
})
export const rivalsWith = (purse, squadFn = () => [], n = 9) =>
    Array.from({ length: n }, (_, i) => team(squadFn(i), typeof purse === 'function' ? purse(i) : purse, `R${i}`))
export const ctxOf = ({ squad, purse = 6000, lot, upcoming = [], returning = [], rivals = rivalsWith(6000), phase = 'main', progress = 0.5, rules = DEFAULT_RULES }) => ({
    rules, lot, self: team(squad, purse), rivals, upcoming, returning, phase, progress
})

// Noise extremes: −8% and +8% (rng feeds `value × (1 + (2r − 1) × 0.08)`).
const LOW = () => 0
const HIGH = () => 0.999999
const MID = () => 0.5

// Fewest empty XI slots the team can reach with `purse`, from the supply.
const reachable = (counts, opts, purse) => {
    if (legalCompletionCost(counts, opts).cost <= purse) return 0
    const minCost = completionCosts(counts, { ...opts, budget: purse }).minCost
    for (let e = 0; e <= 11; e++) if (minCost[e] <= purse) return e
    return 11
}

// The team after buying the lot at `price`, with the same market still to come.
export const afterPurchase = (ctx, price) => ({
    ...ctx,
    self: {
        ...ctx.self,
        squad: [...ctx.self.squad, ctx.lot],
        purseLeft: ctx.self.purseLeft - price,
        purseSpent: (ctx.self.purseSpent ?? 0) + price,
        playerCount: ctx.self.playerCount + 1,
        overseasCount: ctx.self.overseasCount + (ctx.lot.nationality === 'Overseas' ? 1 : 0)
    }
})

// Every safety invariant for one decision, for every personality, at every
// noise extreme. Returns { caps, plan } for scenario-specific assertions.
export const assertSafeDecision = (ctx, label = '') => {
    const plan = planBid(ctx)
    const rules = ctx.rules
    const self = ctx.self
    const blocked = bidBlocker({ team: self, lot: { ...ctx.lot, currentBidderId: '' }, rules, amount: ctx.lot.basePrice })
    const comp = teamComposition(self, rules)
    const opts = { slots: comp.slotsLeft, overseasSlots: comp.overseasSlots, supply: supplyOf(ctx) }
    const emptyIfPass = reachable(comp.counts, opts, self.purseLeft)

    assert.ok(plan.budget.maxSafeBid >= 0 && plan.budget.maxSafeBid <= self.purseLeft, `${label} maxSafeBid ${plan.budget.maxSafeBid} within [0, purse ${self.purseLeft}]`)
    if (plan.budget.requiredReserve !== null) assert.ok(plan.budget.requiredReserve >= 0, `${label} reserve ≥ 0`)
    if (plan.budget.reserveIfPassed !== null) assert.ok(plan.budget.reserveIfPassed >= 0, `${label} reserve if passed ≥ 0`)
    assert.equal(plan.completion.reachableEmptySlots <= emptyIfPass, true, `${label} plan targets no worse than passing`)

    const caps = {}
    for (const id of RULE_BOT_IDS) {
        for (const [noiseName, rng] of [['low', LOW], ['mid', MID], ['high', HIGH]]) {
            const cap = ruleBotCap(id, ctx, rng, { plan })
            const tag = `${label} ${id}/${noiseName} cap ${cap}`
            assert.ok(Number.isInteger(cap) && cap >= 0, `${tag}: integer ≥ 0`)
            if (cap === 0) continue
            assert.equal(blocked, null, `${tag}: bids although the rules block the lot (${blocked})`)
            assert.ok(plan.allowed, `${tag}: bids although the plan disallows (${plan.reason})`)
            assert.ok(cap >= ctx.lot.basePrice, `${tag}: below base price`)
            assert.ok(cap <= self.purseLeft, `${tag}: above purse ${self.purseLeft}`)
            assert.ok(cap <= plan.budget.maxSafeBid, `${tag}: above maxSafeBid ${plan.budget.maxSafeBid}`)
            assert.ok(self.playerCount + 1 <= rules.maxPlayers, `${tag}: squad overflow`)
            if (ctx.lot.nationality === 'Overseas') assert.ok(self.overseasCount + 1 <= rules.maxOverseas, `${tag}: overseas overflow`)
            // Paying the full cap must not make the team's best reachable XI worse.
            const after = afterPurchase(ctx, cap)
            const compAfter = teamComposition(after.self, rules)
            const emptyAfter = reachable(compAfter.counts, { slots: compAfter.slotsLeft, overseasSlots: compAfter.overseasSlots, supply: supplyOf(ctx) }, after.self.purseLeft)
            assert.ok(emptyAfter <= emptyIfPass, `${tag}: buying at the cap leaves ${emptyAfter} empty XI slots vs ${emptyIfPass} when passing`)
            if (noiseName === 'mid') caps[id] = cap
        }
        caps[id] ??= 0
    }
    return { caps, plan }
}

// Deterministic when noise is off.
export const assertDeterministic = (ctx, label = '') => {
    for (const id of RULE_BOT_IDS) {
        const a = ruleBotCap(id, ctx, Math.random, { noise: 0 })
        const b = ruleBotCap(id, ctx, Math.random, { noise: 0 })
        assert.equal(a, b, `${label} ${id}: noise-free decision not deterministic`)
    }
}
