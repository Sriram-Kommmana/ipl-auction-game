import { test } from 'node:test'
import assert from 'node:assert/strict'
import { DEFAULT_RULES, nextBidAmount } from '../src/rules.js'
import { fairValue } from '../src/valuation.js'
import { ruleBotCap, RULE_BOT_IDS } from '../src/ruleBots.js'
import { classifyOpportunity, planBid, XI_GAIN } from '../src/planning.js'

// ── fixtures ──────────────────────────────────────────────────────────────
let nextSl = 1
const P = (role, { os = false, base = 20, rating = 80 } = {}) => ({
    slNo: nextSl++, playerName: `P${nextSl}`, role, nationality: os ? 'Overseas' : 'Indian', basePrice: base, rating
})
const WK = (o) => P('WICKET KEEPER', o)
const BAT = (o) => P('BATSMAN', o)
const BWL = (o) => P('BOWLER', o)
const many = (n, make, o) => Array.from({ length: n }, () => make(o))
const legalXI = (rating = 85) => [WK({ rating }), ...many(5, BWL, { rating }), ...many(5, BAT, { rating })]

const team = (squad, purse, teamId = 'ME') => ({
    teamId, purseLeft: purse, playerCount: squad.length,
    overseasCount: squad.filter((p) => p.nationality === 'Overseas').length, squad
})
// Re-auction context: progress 1, `upcoming` = rest of the re-auction list, nothing returns after it.
const reCtx = ({ squad, purse = 3000, lot, upcoming = many(8, BAT, { rating: 70 }), rivals = [], teamId }) => ({
    rules: DEFAULT_RULES, lot, self: team(squad, purse, teamId), rivals, upcoming, returning: [], phase: 'reauction', progress: 1
})
const mainCtx = (args) => ({ ...reCtx(args), phase: 'main', progress: 0.6, upcoming: args.upcoming ?? many(40, BAT, { rating: 70 }) })
const capsFor = (ctx) => RULE_BOT_IDS.map((id) => ruleBotCap(id, ctx, Math.random, { noise: 0 }))
const everyPositive = (caps) => caps.every((c) => c > 0)
const everyZero = (caps) => caps.every((c) => c === 0)

// Resolve one lot the way the engine does: one increment at a time while
// someone other than the leader can still pay.
const ladder = (lot, caps) => {
    let price = lot.basePrice
    let leader = -1
    for (;;) {
        const amount = nextBidAmount(price, leader !== -1)
        const next = caps.findIndex((c, i) => i !== leader && c >= amount)
        if (next === -1) break
        leader = next
        price = amount
    }
    return { leader, price: leader === -1 ? null : price }
}

// ── critical requirement ──────────────────────────────────────────────────
test('1. returning keeper, no keeper on the roster → critical, may spend the safe maximum', () => {
    const squad = [...many(5, BWL, { rating: 85 }), ...many(6, BAT, { rating: 85 })]
    const ctx = reCtx({ squad, lot: WK({ rating: 76, base: 50 }) })
    const plan = planBid(ctx)
    const op = classifyOpportunity(ctx, plan)
    assert.equal(op.category, 'critical')
    assert.equal(op.finalOpportunity, true) // no other keeper left anywhere
    assert.deepEqual(capsFor(ctx), RULE_BOT_IDS.map(() => plan.budget.maxSafeBid))
})
test('1b. …with more keepers still to come → critical, but capped at 3× fair value', () => {
    const squad = [...many(5, BWL, { rating: 85 }), ...many(6, BAT, { rating: 85 })]
    const lot = WK({ rating: 76, base: 50 })
    const ctx = reCtx({ squad, lot, upcoming: [...many(4, WK, { rating: 76 }), ...many(5, BAT, { rating: 70 })] })
    const caps = capsFor(ctx)
    assert.equal(classifyOpportunity(ctx).finalOpportunity, false)
    caps.forEach((c) => assert.ok(c >= Math.floor(fairValue(lot) * 3) && c < planBid(ctx).budget.maxSafeBid))
})
test('2. returning bowler when bowling options are short → critical', () => {
    const squad = [WK({ rating: 85 }), ...many(4, BWL, { rating: 85 }), ...many(6, BAT, { rating: 85 })]
    const ctx = reCtx({ squad, lot: BWL({ rating: 75 }) })
    assert.equal(classifyOpportunity(ctx).category, 'critical')
    assert.ok(everyPositive(capsFor(ctx)))
})
test('3. returning Indian when the Indian requirement is critical → critical', () => {
    // 8 overseas (4 can play) + 6 Indians: the XI needs one more Indian.
    const squad = [WK({ rating: 85 }), ...many(3, BWL, { rating: 85 }), ...many(2, BAT, { rating: 85 }),
        ...many(2, BWL, { os: true, rating: 88 }), ...many(6, BAT, { os: true, rating: 88 })]
    const ctx = reCtx({ squad, lot: BAT({ rating: 72 }) })
    assert.equal(planBid(ctx).teamNeeds.indians !== 'SAFE', true)
    assert.equal(classifyOpportunity(ctx).category, 'critical')
    assert.ok(everyPositive(capsFor(ctx)))
})

// ── XI improvement ────────────────────────────────────────────────────────
test('4. returning player who significantly improves the XI → useful, normal valuation', () => {
    const ctx = reCtx({ squad: legalXI(80), lot: BAT({ rating: 92 }) })
    const op = classifyOpportunity(ctx)
    assert.equal(op.category, 'useful')
    assert.ok(op.gain >= XI_GAIN.useful)
    assert.ok(everyPositive(capsFor(ctx)))
})
test('5. marginal improvement → price scaled down by the small gain', () => {
    const lot = BAT({ rating: 87, base: 20 })
    const ctx = reCtx({ squad: legalXI(85), lot })
    const op = classifyOpportunity(ctx)
    assert.equal(op.category, 'marginal')
    const ceiling = fairValue(lot) * 2.2 * (op.gain / XI_GAIN.useful) // no personality value exceeds 2.2× fair value
    for (const cap of capsFor(ctx)) assert.ok(cap > 0 && cap <= Math.ceil(ceiling), `cap ${cap} vs ${ceiling}`)
})
test('6. no XI improvement and no useful cover → pass', () => {
    const squad = [...legalXI(85), WK({ rating: 84 }), ...many(2, BAT, { rating: 84 }), BWL({ rating: 84 })]
    const ctx = reCtx({ squad, lot: BAT({ rating: 78 }) })
    assert.equal(classifyOpportunity(ctx).category, 'none')
    assert.ok(everyZero(capsFor(ctx)))
})

// ── depth ─────────────────────────────────────────────────────────────────
test('7. 12-player squad, healthy purse, backup keeper → depth buy', () => {
    const ctx = reCtx({ squad: [...legalXI(85), BAT({ rating: 80 })], lot: WK({ rating: 78, base: 50 }) })
    const op = classifyOpportunity(ctx)
    assert.equal(op.category, 'depth')
    assert.ok(op.cover.includes('keeper'))
    const depthPrice = Math.floor(Math.min(50 * 1.2, fairValue(ctx.lot) * 0.5))
    assert.deepEqual(capsFor(ctx), RULE_BOT_IDS.map(() => depthPrice))
})
test('8. 20-player squad, healthy purse, depth player → bought in the re-auction (the old ≥15 cut-off is gone there)', () => {
    const squad = [...legalXI(85), WK({ rating: 80 }), ...many(8, BAT, { rating: 80 })]
    const lot = BWL({ rating: 80, base: 50 })
    assert.equal(classifyOpportunity(reCtx({ squad, lot })).category, 'depth')
    assert.ok(everyPositive(capsFor(reCtx({ squad, lot }))))
    assert.ok(everyZero(capsFor(mainCtx({ squad, lot })))) // main auction unchanged
})
test('9. 24 players, missing keeper, random depth player → refused', () => {
    const squad = [...many(12, BWL, { rating: 85 }), ...many(12, BAT, { rating: 85 })]
    const ctx = reCtx({ squad, lot: BAT({ rating: 80 }), upcoming: [WK(), ...many(4, BAT)] })
    assert.equal(planBid(ctx).allowed, false)
    assert.ok(everyZero(capsFor(ctx)))
})

// ── purse ─────────────────────────────────────────────────────────────────
test('10. huge purse does not turn a marginal player into a big spend', () => {
    const lot = BWL({ rating: 86, base: 20 })
    const ctx = reCtx({ squad: legalXI(85), purse: 9000, lot })
    const op = classifyOpportunity(ctx)
    assert.equal(op.category, 'marginal')
    for (const cap of capsFor(ctx)) assert.ok(cap < 0.1 * planBid(ctx).budget.maxSafeBid, `cap ${cap}`)
})
test('11. very low purse → cannot bid even on a critical player', () => {
    const squad = [...many(5, BWL, { rating: 85 }), ...many(6, BAT, { rating: 85 })]
    const ctx = reCtx({ squad, purse: 30, lot: WK({ base: 50 }) })
    assert.ok(everyZero(capsFor(ctx)))
})
test('12. exact completion budget → bids exactly what keeps the XI completable', () => {
    // 9 players, no keeper: needs the keeper (₹50L) + one ₹20L batsman; purse ₹70L.
    const squad = [...many(5, BWL, { rating: 85 }), ...many(4, BAT, { rating: 85 })]
    const ctx = reCtx({ squad, purse: 70, lot: WK({ base: 50 }), upcoming: [BAT({ base: 20 })] })
    assert.equal(planBid(ctx).budget.maxSafeBid, 50)
    assert.deepEqual(capsFor(ctx), RULE_BOT_IDS.map(() => 50))
})

// ── overseas ──────────────────────────────────────────────────────────────
const eightOverseas = () => [WK({ rating: 85 }), ...many(3, BWL, { rating: 85 }), ...many(3, BAT, { rating: 85 }),
    ...many(2, BWL, { os: true, rating: 85 }), ...many(6, BAT, { os: true, rating: 85 })]
test('13. 8 overseas already → no overseas returning player', () => {
    const ctx = reCtx({ squad: eightOverseas(), lot: BWL({ os: true, rating: 94 }) })
    assert.equal(planBid(ctx).allowed, false)
    assert.ok(everyZero(capsFor(ctx)))
})
test('14. 7 overseas and a returning overseas player who improves the XI → allowed', () => {
    const squad = [...legalXI(80), ...many(4, BAT, { os: true, rating: 80 })].slice(0, 15) // 4 overseas… top it up to 7
    squad.push(...many(3, BWL, { os: true, rating: 80 }))
    const ctx = reCtx({ squad, lot: BWL({ os: true, rating: 93 }) })
    assert.equal(ctx.self.overseasCount, 7)
    assert.equal(classifyOpportunity(ctx).category, 'useful')
    assert.ok(everyPositive(capsFor(ctx)))
})
test('15. overseas-heavy squad: the Indian is critical, another overseas player is not', () => {
    const squad = eightOverseas().filter((p) => !(p.role === 'BATSMAN' && p.nationality === 'Indian')).concat(BAT({ rating: 85 }), BAT({ rating: 85 }))
    const indian = reCtx({ squad, lot: BAT({ rating: 74 }) })
    assert.equal(classifyOpportunity(indian).category, 'critical')
    assert.ok(everyPositive(capsFor(indian)))
})

// ── previously rejected players are re-evaluated, not blacklisted ─────────
test('16. rejected in the main auction, fills a requirement now → bought', () => {
    const lot = BWL({ rating: 76, base: 50 })
    const before = [...legalXI(85), ...many(7, BAT, { rating: 84 })] // 18 players, secure XI
    assert.ok(everyZero(capsFor(mainCtx({ squad: before, lot })))) // main auction: passes
    // Later its squad has only 4 bowling options (e.g. it started that way and the others went elsewhere).
    const after = [WK({ rating: 85 }), ...many(4, BWL, { rating: 85 }), ...many(18, BAT, { rating: 84 })]
    const ctx = reCtx({ squad: after, lot })
    assert.equal(classifyOpportunity(ctx).category, 'critical')
    assert.ok(everyPositive(capsFor(ctx)))
})
test('17. rejected in the main auction and still of no value → still passes', () => {
    const lot = BAT({ rating: 76 })
    const squad = [...legalXI(85), ...many(4, BAT, { rating: 84 }), WK({ rating: 84 }), BWL({ rating: 84 })]
    assert.ok(everyZero(capsFor(mainCtx({ squad, lot }))))
    assert.ok(everyZero(capsFor(reCtx({ squad, lot }))))
})

// ── competition ───────────────────────────────────────────────────────────
test('18. several teams need the returning keeper → it is contested and sells above base', () => {
    const lot = WK({ rating: 76, base: 50 })
    const needy = (id) => reCtx({ squad: [...many(5, BWL, { rating: 85 }), ...many(6, BAT, { rating: 85 })], lot, teamId: id })
    const caps = ['A', 'B', 'C'].map((id) => ruleBotCap('moneyball', needy(id), Math.random, { noise: 0 }))
    assert.ok(caps.every((c) => c > lot.basePrice))
    const sale = ladder(lot, caps)
    assert.ok(sale.leader !== -1 && sale.price > lot.basePrice)
})
test('19. only one team needs the returning keeper → it sells at base price', () => {
    const lot = WK({ rating: 76, base: 50 })
    const needy = reCtx({ squad: [...many(5, BWL, { rating: 85 }), ...many(6, BAT, { rating: 85 })], lot })
    const settled = (id) => reCtx({ squad: [...legalXI(85), WK({ rating: 84 })], lot, teamId: id }) // two keepers already
    const caps = [needy, settled('B'), settled('C')].map((ctx) => ruleBotCap('moneyball', ctx, Math.random, { noise: 0 }))
    assert.deepEqual([caps[0] > 0, caps[1], caps[2]], [true, 0, 0])
    assert.deepEqual(ladder(lot, caps), { leader: 0, price: lot.basePrice })
})
