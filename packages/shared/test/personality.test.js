// Phase 1B.2 — personality-aware main-auction behaviour. Each test builds
// matched states that differ in ONE factor and checks how the caps respond,
// or plays real full-pool auctions.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { parsePlayersCsv } from '../src/playersCsv.js'
import { DEFAULT_RULES } from '../src/rules.js'
import { selectBestXI } from '../src/scoring.js'
import { fairValue } from '../src/valuation.js'
import { ruleBotCap, RULE_BOT_IDS } from '../src/ruleBots.js'
import { planBid } from '../src/planning.js'
import { AuctionSim, agentCap, createRng } from '../src/sim.js'

// ── fixtures ──────────────────────────────────────────────────────────────
let nextSl = 10000
const P = (role, { os = false, base = 100, rating = 80 } = {}) => ({
    slNo: nextSl++, playerName: `P${nextSl}`, role, nationality: os ? 'Overseas' : 'Indian', basePrice: base, rating
})
const WK = (o) => P('WICKET KEEPER', o)
const BAT = (o) => P('BATSMAN', o)
const BWL = (o) => P('BOWLER', o)
const AR = (o) => P('ALL ROUNDER', o)
const many = (n, make, o) => Array.from({ length: n }, () => make(o))
const legalXI = (rating) => [WK({ rating }), ...many(5, BWL, { rating }), ...many(5, BAT, { rating })]
const team = (squad, purse, teamId = 'ME') => ({
    teamId, purseLeft: purse, playerCount: squad.length,
    overseasCount: squad.filter((p) => p.nationality === 'Overseas').length, squad
})
const rivalsWith = (purse, squadFn = () => [], n = 9) => Array.from({ length: n }, (_, i) => team(squadFn(i), purse, `R${i}`))
const ctxOf = ({ squad, purse = 6000, lot, upcoming = many(60, BAT, { rating: 78 }), returning = [], rivals = rivalsWith(6000), progress = 0.5 }) => ({
    rules: DEFAULT_RULES, lot, self: team(squad, purse), rivals, upcoming, returning, phase: 'main', progress
})
const cap = (id, ctx) => ruleBotCap(id, ctx, Math.random, { noise: 0 })
const caps = (ctx) => Object.fromEntries(RULE_BOT_IDS.map((id) => [id, cap(id, ctx)]))

const players = parsePlayersCsv(readFileSync(fileURLToPath(new URL('../../../apps/server/src/db/players.csv', import.meta.url)), 'utf8'))
const LINEUP = ['moneyball', 'starChaser', 'balancedBuilder', 'opportunist', 'starChaser', 'balancedBuilder', 'balancedBuilder', 'opportunist', 'moneyball', 'balancedBuilder']
// Plays one full-pool auction with rule bots; reports each team at 25% of the
// main round and at the end.
const playFull = (seed) => {
    const rng = createRng(seed)
    const sim = new AuctionSim({ players, teamCount: LINEUP.length, rng })
    let at25 = null
    while (!sim.done) {
        if (!at25 && sim.phase === 'main' && sim.index >= sim.mainLength * 0.25) at25 = sim.teams.map((t) => 1 - t.purseLeft / 12500)
        const caps = LINEUP.map((persona, i) => agentCap({ kind: 'rule', persona }, sim.contextFor(i), rng))
        sim.resolveLot(caps)
    }
    return { teams: sim.teams, at25 }
}
const games = [301, 302, 303, 304].map(playFull)
const byPersona = (persona, f) => games.flatMap((g) => LINEUP.map((p, i) => (p === persona ? f(g, i) : null)).filter((x) => x !== null))
const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length

// ── A. critical requirement valuation ─────────────────────────────────────
test('A. a CRITICAL keeper is valued well above the same keeper when keepers are plentiful — within the safe maximum', () => {
    const squad = [...many(5, BWL, { rating: 84 }), ...many(6, BAT, { rating: 84 })]
    const lot = WK({ rating: 82 })
    const plentiful = ctxOf({ squad, lot, upcoming: [...many(30, WK, { rating: 82 }), ...many(30, BAT, { rating: 78 })] })
    const critical = ctxOf({ squad, lot, upcoming: [WK({ rating: 80 }), ...many(30, BAT, { rating: 78 })], rivals: rivalsWith(6000, () => many(11, BAT)) })
    assert.equal(planBid(critical).teamNeeds.keeper, 'CRITICAL')
    for (const id of RULE_BOT_IDS) {
        assert.ok(cap(id, critical) > cap(id, plentiful) * 1.2, `${id}: ${cap(id, critical)} vs ${cap(id, plentiful)}`)
        assert.ok(cap(id, critical) <= planBid(critical).budget.maxSafeBid)
    }
})

// ── B. XI gain drives value ───────────────────────────────────────────────
test('B. the same player is worth far more as a big XI upgrade than as a marginal one', () => {
    const lot = BWL({ rating: 90, base: 150 })
    const market = players.slice(0, 200) // a real market with upgrades still to come
    const big = ctxOf({ squad: legalXI(78), lot, upcoming: market }) // replaces a 78 → large gain
    const tiny = ctxOf({ squad: [...legalXI(89), BWL({ rating: 89 })], lot, upcoming: market }) // replaces an 89 → +0.09
    for (const id of RULE_BOT_IDS) assert.ok(cap(id, big) >= 3 * Math.max(1, cap(id, tiny)), `${id}: ${cap(id, big)} vs ${cap(id, tiny)}`)
})

// ── C. scarcity ───────────────────────────────────────────────────────────
test('C. a bowling option is worth more when bowling options are scarce than when plentiful', () => {
    const squad = [WK({ rating: 84 }), ...many(4, BWL, { rating: 84 }), ...many(6, BAT, { rating: 84 })]
    const lot = BWL({ rating: 83 })
    const plentiful = ctxOf({ squad, lot, upcoming: [...many(40, BWL, { rating: 83 }), ...many(20, BAT, { rating: 78 })] })
    const scarce = ctxOf({ squad, lot, upcoming: [BWL({ rating: 80 }), ...many(40, BAT, { rating: 78 })], rivals: rivalsWith(6000, () => [WK(), ...many(10, BAT)]) })
    for (const id of RULE_BOT_IDS) assert.ok(cap(id, scarce) > cap(id, plentiful), `${id}: ${cap(id, scarce)} vs ${cap(id, plentiful)}`)
})

// ── D. auction position ───────────────────────────────────────────────────
test('D. late, with money idle and few upgrades left, a real upgrade is worth more than early', () => {
    const squad = legalXI(80)
    const lot = BAT({ rating: 88, base: 150 })
    const early = ctxOf({ squad, lot, purse: 7000, progress: 0.1, upcoming: players.slice(0, 250), rivals: rivalsWith(12500) })
    // Late: rivals have bought most of the good players; only weak ones remain.
    const late = ctxOf({ squad, lot, purse: 7000, progress: 0.9, upcoming: many(6, BAT, { rating: 72, base: 20 }),
        rivals: rivalsWith(2000, (i) => players.slice(i * 20, i * 20 + 20)) })
    for (const id of RULE_BOT_IDS) assert.ok(cap(id, late) > cap(id, early), `${id}: late ${cap(id, late)} vs early ${cap(id, early)}`)
})

// ── E. purse ──────────────────────────────────────────────────────────────
test('E. a big purse bids at least as much as a small one, and never beyond the safe maximum', () => {
    const squad = [...legalXI(80).slice(0, 9)]
    const lot = AR({ rating: 90, base: 150 })
    for (const id of RULE_BOT_IDS) {
        const rich = ctxOf({ squad, lot, purse: 9000 })
        const poor = ctxOf({ squad, lot, purse: 900 })
        assert.ok(cap(id, rich) >= cap(id, poor))
        assert.ok(cap(id, poor) <= planBid(poor).budget.maxSafeBid)
    }
})

// ── F. Star Chaser pacing ─────────────────────────────────────────────────
test('F. Star Chaser no longer burns its purse in the first quarter — but still spends faster than Moneyball', () => {
    const sc = mean(byPersona('starChaser', (g, i) => g.at25[i]))
    const mb = mean(byPersona('moneyball', (g, i) => g.at25[i]))
    assert.ok(sc < 0.7, `Star Chaser spent ${(100 * sc).toFixed(0)}% by 25% (was ~95%)`)
    assert.ok(sc > mb, `Star Chaser ${(100 * sc).toFixed(0)}% vs Moneyball ${(100 * mb).toFixed(0)}%`)
})

// ── G. Star Chaser completion safety ──────────────────────────────────────
test('G. Star Chaser always completes a legal XI and never overspends', () => {
    for (const t of byPersona('starChaser', (g, i) => g.teams[i])) {
        assert.equal(selectBestXI(t.squad).emptySlots, 0)
        assert.ok(t.purseLeft >= 0)
    }
})

// ── H. Moneyball ──────────────────────────────────────────────────────────
test('H. Moneyball backs off while comparable players are still to come; Star Chaser does not', () => {
    const squad = legalXI(80).slice(0, 8)
    const lot = BAT({ rating: 86, base: 100 })
    const plenty = ctxOf({ squad, lot, upcoming: [...many(8, BAT, { rating: 86 }), ...many(40, BWL, { rating: 76 })] })
    const none = ctxOf({ squad, lot, upcoming: many(48, BWL, { rating: 76 }) })
    const mbDrop = cap('moneyball', plenty) / cap('moneyball', none)
    const scDrop = cap('starChaser', plenty) / cap('starChaser', none)
    assert.ok(mbDrop < 0.95, `Moneyball ratio ${mbDrop.toFixed(2)}`)
    assert.ok(mbDrop < scDrop, `Moneyball ${mbDrop.toFixed(2)} vs Star Chaser ${scDrop.toFixed(2)}`)
})

test('H2. Moneyball pays relatively more for a big XI gain than for a small one (loves XI gain per rupee)', () => {
    const lot = BWL({ rating: 90, base: 150 })
    const ratio = (id) => cap(id, ctxOf({ squad: legalXI(78), lot })) / Math.max(1, cap(id, ctxOf({ squad: [...legalXI(86), BWL({ rating: 87 })], lot })))
    assert.ok(ratio('moneyball') > ratio('starChaser'), `Moneyball ${ratio('moneyball').toFixed(2)} vs Star Chaser ${ratio('starChaser').toFixed(2)}`)
})

// ── I. Balanced Builder ───────────────────────────────────────────────────
test('I. Balanced Builder reacts most to a missing role versus a role it already has plenty of', () => {
    // No all-rounders yet vs six already; same all-rounder on the block.
    const lot = AR({ rating: 85 })
    const base = [WK({ rating: 84 }), ...many(5, BWL, { rating: 84 }), ...many(5, BAT, { rating: 84 })]
    // Rivals with real squads, so both states sit at the same spending pace.
    const rivals = rivalsWith(6000, (i) => players.slice(i * 18, i * 18 + 18))
    const lacking = ctxOf({ squad: base, lot, rivals, upcoming: players.slice(180, 300) })
    const stacked = ctxOf({ squad: [...base, ...many(6, AR, { rating: 84 })], lot, rivals, upcoming: players.slice(180, 300) })
    const response = (id) => cap(id, lacking) / Math.max(1, cap(id, stacked))
    for (const id of ['moneyball', 'opportunist']) assert.ok(response('balancedBuilder') > response(id), `BB ${response('balancedBuilder').toFixed(2)} vs ${id} ${response(id).toFixed(2)}`)
})

// ── J. Opportunist ────────────────────────────────────────────────────────
test('J. the Opportunist pounces when rivals cannot afford the player; Moneyball barely reacts', () => {
    const squad = legalXI(80).slice(0, 9)
    const lot = BAT({ rating: 88, base: 150 })
    const richRivals = ctxOf({ squad, lot, progress: 0.6, rivals: rivalsWith(9000) })
    const brokeRivals = ctxOf({ squad, lot, progress: 0.6, rivals: rivalsWith(150) })
    const response = (id) => cap(id, brokeRivals) / cap(id, richRivals)
    assert.ok(response('opportunist') > 1.2, `Opportunist ${response('opportunist').toFixed(2)}`)
    assert.ok(response('opportunist') > response('moneyball'), `Opportunist ${response('opportunist').toFixed(2)} vs Moneyball ${response('moneyball').toFixed(2)}`)
})

// ── K. differentiation ────────────────────────────────────────────────────
test('K. the four personalities give different caps in most situations', () => {
    const rng = createRng(12)
    let distinct = 0
    let total = 0
    for (let i = 0; i < 60; i++) {
        const squad = players.filter(() => rng() < 0.04).slice(0, Math.floor(rng() * 20))
        const lot = players[Math.floor(rng() * players.length)]
        if (squad.includes(lot)) continue
        const ctx = ctxOf({ squad, lot, purse: 2000 + Math.floor(rng() * 10000), progress: rng(), upcoming: players.filter((p) => p !== lot && !squad.includes(p)).slice(0, 150) })
        ctx.self.overseasCount = Math.min(8, ctx.self.overseasCount)
        const c = Object.values(caps(ctx))
        if (c.every((x) => x === 0)) continue
        total++
        if (new Set(c).size >= 3) distinct++
    }
    assert.ok(distinct / total > 0.7, `${distinct}/${total} states with ≥3 different caps`)
})

test('K2. full auctions: spending curves and squads differ by personality', () => {
    const at25 = (p) => mean(byPersona(p, (g, i) => g.at25[i]))
    const stars = (p) => mean(byPersona(p, (g, i) => g.teams[i].squad.filter((x) => x.rating >= 90).length))
    const allRounders = (p) => mean(byPersona(p, (g, i) => g.teams[i].squad.filter((x) => x.role === 'ALL ROUNDER').length))
    assert.ok(at25('starChaser') > at25('opportunist'))
    assert.ok(stars('starChaser') > stars('moneyball'))
    assert.ok(new Set(RULE_BOT_IDS.map((p) => allRounders(p).toFixed(1))).size >= 3)
})

// ── L–O. hard safety ──────────────────────────────────────────────────────
test('L. every rule-bot team completes a legal XI in full-pool auctions', () => {
    for (const g of games) for (const t of g.teams) assert.equal(selectBestXI(t.squad).emptySlots, 0)
})

test('M. caps never exceed the safe maximum or the purse, including critical and final-opportunity paths', () => {
    const rng = createRng(33)
    for (let i = 0; i < 300; i++) {
        const squad = players.filter(() => rng() < 0.05).slice(0, Math.floor(rng() * 25))
        const lot = players[Math.floor(rng() * players.length)]
        if (squad.includes(lot)) continue
        const upcoming = players.filter((p) => p !== lot && !squad.includes(p) && rng() < 0.3)
        const ctx = ctxOf({ squad, lot, purse: Math.floor(rng() * 12500), progress: rng(), upcoming })
        ctx.self.overseasCount = Math.min(8, ctx.self.overseasCount)
        const plan = planBid(ctx)
        for (const id of RULE_BOT_IDS) {
            const c = ruleBotCap(id, ctx, rng, { plan })
            assert.ok(c <= plan.budget.maxSafeBid && c <= ctx.self.purseLeft, `${id}: ${c}`)
            if (!plan.allowed) assert.equal(c, 0)
        }
    }
})

test('N. no personality bids for an overseas player with 8 overseas already', () => {
    const squad = [...legalXI(80).slice(0, 3), ...many(8, BAT, { os: true, rating: 85 })]
    const ctx = ctxOf({ squad, lot: BWL({ os: true, rating: 95, base: 200 }) })
    for (const c of Object.values(caps(ctx))) assert.equal(c, 0)
})

test('O. no personality bids with a full squad', () => {
    const ctx = ctxOf({ squad: [...legalXI(85), ...many(14, BAT, { rating: 80 })], lot: BWL({ rating: 96, base: 200 }) })
    for (const c of Object.values(caps(ctx))) assert.equal(c, 0)
})

// ── P. re-auction regression ──────────────────────────────────────────────
test('P. re-auction: a final-chance keeper still gets the safe maximum; a useless player still gets nothing', () => {
    const noKeeper = [...many(5, BWL, { rating: 85 }), ...many(6, BAT, { rating: 85 })]
    const re = (squad, lot) => ({ ...ctxOf({ squad, lot, upcoming: many(5, BAT, { rating: 70 }) }), phase: 'reauction', progress: 1 })
    const keeper = re(noKeeper, WK({ rating: 76, base: 50 }))
    for (const c of Object.values(caps(keeper))) assert.equal(c, planBid(keeper).budget.maxSafeBid)
    const useless = re([...legalXI(85), ...many(3, BAT, { rating: 84 }), WK({ rating: 84 }), BWL({ rating: 84 })], BAT({ rating: 76 }))
    for (const c of Object.values(caps(useless))) assert.equal(c, 0)
})

test('marginal discipline: a +0.1 upgrade never earns a big price, even with plenty of money', () => {
    const lot = BWL({ rating: 90, base: 150 })
    const rich = ctxOf({ squad: [...legalXI(89), BWL({ rating: 89 })], lot, purse: 11000, upcoming: players.slice(0, 200) })
    for (const [id, c] of Object.entries(caps(rich))) assert.ok(c <= 0.6 * fairValue(lot), `${id}: ${c} vs fair value ${fairValue(lot)}`)
})
