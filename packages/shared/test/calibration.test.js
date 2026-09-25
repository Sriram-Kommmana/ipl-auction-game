// Phase 1B.3 — calibration & hardening. Overseas-slot conservation in the
// main auction, and the re-auction's valuation of marginal upgrades.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { parsePlayersCsv } from '../src/playersCsv.js'
import { DEFAULT_RULES } from '../src/rules.js'
import { ruleBotCap, RULE_BOT_IDS } from '../src/ruleBots.js'
import { classifyOpportunity, planBid, XI_GAIN } from '../src/planning.js'
import { overseasSlotContested } from '../src/botSignals.js'
import { AuctionSim, agentCap, createRng } from '../src/sim.js'

// ── fixtures ──────────────────────────────────────────────────────────────
let nextSl = 20000
const P = (role, { os = false, base = 100, rating = 80 } = {}) => ({
    slNo: nextSl++, playerName: `P${nextSl}`, role, nationality: os ? 'Overseas' : 'Indian', basePrice: base, rating
})
const WK = (o) => P('WICKET KEEPER', o)
const BAT = (o) => P('BATSMAN', o)
const BWL = (o) => P('BOWLER', o)
const many = (n, make, o) => Array.from({ length: n }, () => make(o))
const legalXI = (rating) => [WK({ rating }), ...many(5, BWL, { rating }), ...many(5, BAT, { rating })]
const team = (squad, purse, teamId = 'ME') => ({
    teamId, purseLeft: purse, playerCount: squad.length,
    overseasCount: squad.filter((p) => p.nationality === 'Overseas').length, squad
})
const rivalsWith = (purse, n = 9) => Array.from({ length: n }, (_, i) => team([], purse, `R${i}`))
const ctxOf = ({ squad, purse = 6000, lot, upcoming, returning = [], phase = 'main', progress = 0.5 }) => ({
    rules: DEFAULT_RULES, lot, self: team(squad, purse), rivals: rivalsWith(6000), upcoming, returning, phase, progress
})
const cap = (id, ctx) => ruleBotCap(id, ctx, Math.random, { noise: 0 })
const caps = (ctx) => RULE_BOT_IDS.map((id) => cap(id, ctx))

const weakIndians = () => many(40, BAT, { rating: 76 })
const betterOverseas = (n, rating) => many(n, BAT, { os: true, rating })

// ── overseas-slot conservation (main auction) ─────────────────────────────
test('1. an overseas DEPTH player is refused while better overseas players still to come outnumber the slots left', () => {
    const squad = [...legalXI(85), BAT({ rating: 80 })] // 12 players, 0 overseas
    const lot = BAT({ os: true, rating: 78, base: 50 })
    const ctx = ctxOf({ squad, lot, upcoming: [...betterOverseas(10, 84), ...weakIndians()] })
    assert.equal(overseasSlotContested(ctx), true) // 10 better overseas to come > 7 slots left after him
    assert.deepEqual(caps(ctx), RULE_BOT_IDS.map(() => 0))
    // The same player as an Indian is still a cheap depth signing.
    const indian = ctxOf({ squad, lot: BAT({ rating: 78, base: 50 }), upcoming: [...betterOverseas(10, 84), ...weakIndians()] })
    for (const c of caps(indian)) assert.ok(c >= 50, `depth cap ${c}`)
})

test('2. a MARGINAL overseas upgrade is refused while the slot is contested, and considered once it is not', () => {
    const squad = legalXI(85)
    const lot = BAT({ os: true, rating: 88, base: 100 })
    const contested = ctxOf({ squad, lot, upcoming: [...betterOverseas(10, 90), ...weakIndians()] })
    const free = ctxOf({ squad, lot, upcoming: weakIndians() })
    const gain = planBid(contested).playerImpact.xiGain
    assert.ok(gain > XI_GAIN.none && gain < XI_GAIN.useful, `gain ${gain}`)
    assert.equal(overseasSlotContested(contested), true)
    assert.equal(overseasSlotContested(free), false)
    assert.deepEqual(caps(contested), RULE_BOT_IDS.map(() => 0))
    assert.ok(caps(free).some((c) => c > 0), `free caps ${caps(free)}`)
})

test('3. a USEFUL overseas player is still bought even when better overseas players are coming', () => {
    const squad = legalXI(80)
    const lot = BAT({ os: true, rating: 90, base: 100 })
    const ctx = ctxOf({ squad, lot, upcoming: [...betterOverseas(10, 92), ...weakIndians()] })
    assert.ok(planBid(ctx).playerImpact.xiGain >= XI_GAIN.useful)
    for (const c of caps(ctx)) assert.ok(c > 0, `cap ${c}`)
})

test('4. the slot is not contested when few enough better overseas players remain', () => {
    const squad = [...legalXI(85), ...many(5, BAT, { os: true, rating: 84 })] // 5 overseas → 2 slots left after him
    const lot = BAT({ os: true, rating: 88 })
    assert.equal(overseasSlotContested(ctxOf({ squad, lot, upcoming: [...betterOverseas(2, 90), ...weakIndians()] })), false)
    assert.equal(overseasSlotContested(ctxOf({ squad, lot, upcoming: [...betterOverseas(3, 90), ...weakIndians()] })), true)
    // Returning (unsold) overseas players count too: they come back in the re-auction.
    assert.equal(overseasSlotContested(ctxOf({ squad, lot, upcoming: weakIndians(), returning: betterOverseas(3, 90) })), true)
})

test('5. an Indian player is never affected by the overseas rule', () => {
    const lot = BAT({ rating: 88 })
    assert.equal(overseasSlotContested(ctxOf({ squad: legalXI(85), lot, upcoming: [...betterOverseas(20, 95), ...weakIndians()] })), false)
})

// ── re-auction: marginal upgrades at base price (Issue #2) ────────────────
// A 20-player squad with a spare slot, plenty of purse, and no better
// replacement left in the re-auction list.
const reCtx = (gainRating, base = 100) => ctxOf({
    squad: [...legalXI(85), ...many(9, BAT, { rating: 80 })],
    purse: 2500,
    lot: BAT({ rating: gainRating, base }),
    upcoming: many(8, BAT, { rating: 70 }),
    phase: 'reauction',
    progress: 1
})

test('6. re-auction: a marginal upgrade of ≈ +0.3 is worth base price to the personalities', () => {
    const ctx = reCtx(88) // replaces an 85 → +0.27
    const op = classifyOpportunity(ctx)
    assert.equal(op.category, 'marginal')
    const c = Object.fromEntries(RULE_BOT_IDS.map((id) => [id, cap(id, ctx)]))
    for (const id of ['starChaser', 'balancedBuilder', 'opportunist']) assert.ok(c[id] >= 100, `${id} ${c[id]}`)
})

test('7. re-auction: a +0.45 upgrade is bought by every personality at base price', () => {
    const ctx = reCtx(90) // replaces an 85 → +0.45
    assert.equal(classifyOpportunity(ctx).category, 'marginal')
    for (const c of caps(ctx)) assert.ok(c >= 100, `cap ${c}`)
})

test('8. re-auction: a +0.09 upgrade is NOT made buyable by every personality (Moneyball walks away)', () => {
    const ctx = reCtx(86) // +0.09
    assert.equal(classifyOpportunity(ctx).category, 'marginal')
    assert.equal(cap('moneyball', ctx), 0)
    for (const c of caps(ctx)) assert.ok(c < 0.1 * planBid(ctx).budget.maxSafeBid, `cap ${c}`) // never a big spend
})

test('9. re-auction: the overseas rule does not apply there (the re-auction keeps its own logic)', () => {
    const ctx = { ...reCtx(90), lot: BAT({ os: true, rating: 90 }), upcoming: [...betterOverseas(10, 92)] }
    assert.equal(overseasSlotContested(ctx), true)
    assert.ok(caps(ctx).some((c) => c > 0))
})

// ── full auctions ─────────────────────────────────────────────────────────
const players = parsePlayersCsv(readFileSync(fileURLToPath(new URL('../../../apps/server/src/db/players.csv', import.meta.url)), 'utf8'))
const LINEUP = ['moneyball', 'starChaser', 'balancedBuilder', 'opportunist', 'starChaser', 'balancedBuilder', 'balancedBuilder', 'opportunist', 'moneyball', 'balancedBuilder']

test('10. full auctions: overseas slots are kept for better players — the 8th overseas signing comes after half the main round', () => {
    const fractions = []
    for (const seed of [311, 312]) {
        const rng = createRng(seed)
        const sim = new AuctionSim({ players, teamCount: LINEUP.length, rng })
        const filled = LINEUP.map(() => null)
        while (!sim.done) {
            const progress = sim.progress()
            sim.resolveLot(LINEUP.map((persona, i) => agentCap({ kind: 'rule', persona }, sim.contextFor(i), rng)))
            sim.teams.forEach((t, i) => { if (filled[i] === null && t.overseasCount >= DEFAULT_RULES.maxOverseas) filled[i] = progress })
        }
        filled.forEach((f) => fractions.push(f ?? 1))
    }
    const avg = fractions.reduce((s, x) => s + x, 0) / fractions.length
    assert.ok(avg > 0.5, `average progress at the 8th overseas signing ${avg.toFixed(2)}`)
})
