import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { parsePlayersCsv } from '../src/playersCsv.js'
import { DEFAULT_RULES } from '../src/rules.js'
import { selectBestXI } from '../src/scoring.js'
import { buildObservation } from '../src/observation.js'
import { ruleBotCap, RULE_BOT_IDS } from '../src/ruleBots.js'
import { createRng } from '../src/sim.js'
import {
    canCompleteXIAfterPurchase, classCounts, completionCosts, legalCompletionCost, maxXiCount, minimumCostToCompleteXI, planBid, supplyOf
} from '../src/planning.js'

// ── fixtures ──────────────────────────────────────────────────────────────
let nextSl = 1
const P = (role, { os = false, base = 20, rating = 80 } = {}) => ({
    slNo: nextSl++, playerName: `P${nextSl}`, role, nationality: os ? 'Overseas' : 'Indian', basePrice: base, rating
})
const WK = (o) => P('WICKET KEEPER', o)
const BAT = (o) => P('BATSMAN', o)
const BWL = (o) => P('BOWLER', o)
const AR = (o) => P('ALL ROUNDER', o)
const many = (n, make, o) => Array.from({ length: n }, () => make(o))

// A legal XI: 1 WK, 5 bowling options, 5 batsmen, all Indian.
const legalXI = () => [WK(), ...many(5, BWL), ...many(5, BAT)]
const team = (squad, purse = 5000, extra = {}) => ({
    teamId: 'ME', purseLeft: purse, playerCount: squad.length,
    overseasCount: squad.filter((p) => p.nationality === 'Overseas').length, squad, ...extra
})
const ctxOf = ({ squad, purse, lot, upcoming = [], returning = [], rivals = [], phase = 'main', progress = 0.5, overseasCount }) => ({
    rules: DEFAULT_RULES, lot,
    self: team(squad, purse, overseasCount !== undefined ? { overseasCount } : {}),
    rivals, upcoming, returning, phase, progress
})
const lacking = (n, what) => Array.from({ length: n }, (_, i) => team(what === 'keeper' ? many(11, BAT) : what === 'bowling' ? [WK(), ...many(10, BAT)] : legalXI(), 5000, { teamId: `R${i}` }))

// ── exactness ─────────────────────────────────────────────────────────────
const players = parsePlayersCsv(readFileSync(fileURLToPath(new URL('../../../apps/server/src/db/players.csv', import.meta.url)), 'utf8'))

test('count-based XI size matches selectBestXI on 2000 random real squads', () => {
    const rng = createRng(21)
    for (let i = 0; i < 2000; i++) {
        const size = Math.floor(rng() * 26)
        const squad = players.filter(() => rng() < size / players.length)
        assert.equal(11 - maxXiCount(classCounts(squad)), selectBestXI(squad).emptySlots, `squad ${i}`)
    }
})

test('minimum completion cost matches brute force over every subset of a small supply', () => {
    const rng = createRng(8)
    const roles = ['BATSMAN', 'BOWLER', 'ALL ROUNDER', 'WICKET KEEPER']
    for (let t = 0; t < 120; t++) {
        const squad = Array.from({ length: Math.floor(rng() * 10) }, () => P(roles[Math.floor(rng() * 4)], { os: rng() < 0.45 }))
        const supply = Array.from({ length: 10 }, () => P(roles[Math.floor(rng() * 4)], { os: rng() < 0.45, base: [20, 50, 100, 150, 200][Math.floor(rng() * 5)] }))
        const slots = 25 - squad.length
        const osSlots = 8 - squad.filter((p) => p.nationality === 'Overseas').length
        let brute = Infinity
        for (let mask = 0; mask < 1 << supply.length; mask++) {
            const pick = supply.filter((_, i) => mask & (1 << i))
            if (pick.length > slots || pick.filter((p) => p.nationality === 'Overseas').length > osSlots) continue
            if (selectBestXI([...squad, ...pick]).emptySlots === 0) brute = Math.min(brute, pick.reduce((s, p) => s + p.basePrice, 0))
        }
        const opts = { slots, overseasSlots: osSlots, supply: supplyOf({ upcoming: supply, returning: [] }) }
        assert.equal(legalCompletionCost(classCounts(squad), opts).cost, brute, `fast search, case ${t}`)
        assert.equal(completionCosts(classCounts(squad), opts).minCost[0], brute, `exhaustive search, case ${t}`)
    }
})

// ── purse ─────────────────────────────────────────────────────────────────
test('1. enough money to complete the XI', () => {
    const squad = legalXI().slice(0, 10) // missing one batsman
    const r = minimumCostToCompleteXI(ctxOf({ squad, purse: 1000, lot: BAT(), upcoming: many(5, BAT, { base: 50 }) }))
    assert.deepEqual([r.feasible, r.minimumCost, r.playersNeeded], [true, 50, 1])
})

test('2. not enough money: completion infeasible, best reachable XI still planned', () => {
    const squad = legalXI().slice(0, 8) // 3 batsmen short
    const plan = planBid(ctxOf({ squad, purse: 30, lot: BAT({ base: 20 }), upcoming: many(6, BAT, { base: 20 }) }))
    assert.equal(minimumCostToCompleteXI(ctxOf({ squad, purse: 30, lot: BAT(), upcoming: many(6, BAT, { base: 20 }) })).feasible, false)
    assert.equal(plan.completion.reachableEmptySlots, 2) // ₹30L buys one more player
    assert.equal(plan.allowed, true) // …and this lot IS that player: never refuse the only affordable improvement
    assert.equal(plan.budget.maxSafeBid, 30)
})

test('3. exactly enough money', () => {
    const squad = legalXI().slice(0, 9) // needs 2 batsmen
    const plan = planBid(ctxOf({ squad, purse: 40, lot: BAT({ base: 20 }), upcoming: many(4, BAT, { base: 20 }) }))
    assert.equal(plan.completion.minimumCompletionCost, 40)
    assert.equal(plan.budget.maxSafeBid, 20) // keep ₹20L for the 11th
    assert.equal(plan.allowed, true)
})

test('4. a critical player may consume most of the purse', () => {
    const squad = [...many(5, BWL), ...many(5, BAT)] // no keeper; everything else done
    const lastKeeper = WK({ base: 100 })
    const plan = planBid(ctxOf({ squad, purse: 3000, lot: lastKeeper, upcoming: many(10, BAT, { base: 20 }) }))
    assert.equal(plan.teamNeeds.keeper, 'CRITICAL')
    assert.equal(plan.lotContext.finalOpportunity, true)
    assert.equal(plan.budget.maxSafeBid, 3000)
})

// ── slots ─────────────────────────────────────────────────────────────────
test('5. a 25-player squad cannot bid', () => {
    const squad = [...legalXI(), ...many(14, BAT)]
    const plan = planBid(ctxOf({ squad, purse: 5000, lot: BWL(), upcoming: many(5, BWL) }))
    assert.equal(plan.allowed, false)
    assert.equal(plan.reason, 'Your squad is full.')
})

test('6. 24 players, no keeper: the last slot is reserved for a keeper', () => {
    const squad = [...many(12, BWL), ...many(12, BAT)]
    const upcoming = [WK(), ...many(5, BAT)]
    const bat = planBid(ctxOf({ squad, purse: 5000, lot: BAT({ rating: 96 }), upcoming }))
    assert.equal(bat.allowed, false)
    assert.match(bat.reason, /legal XI impossible/)
    assert.equal(bat.teamNeeds.keeper, 'CRITICAL') // no slot slack, even with keepers still to come
    assert.equal(planBid(ctxOf({ squad, purse: 5000, lot: WK(), upcoming })).allowed, true)
})

test('7. 24 players, 4 bowling options: only a bowling option may take the last slot', () => {
    const squad = [WK(), ...many(4, BWL), ...many(19, BAT)]
    const upcoming = [BWL(), ...many(5, BAT)]
    assert.equal(planBid(ctxOf({ squad, purse: 5000, lot: BAT(), upcoming })).allowed, false)
    assert.equal(planBid(ctxOf({ squad, purse: 5000, lot: AR(), upcoming })).allowed, true)
})

test('8. two requirements, one slot: legal XI impossible; the plan still prefers filling one', () => {
    const squad = [...many(4, BWL), ...many(20, BAT)] // no keeper, 4 bowling options, 24 players
    const upcoming = [WK(), BWL(), BAT()]
    const keeper = planBid(ctxOf({ squad, purse: 5000, lot: WK(), upcoming }))
    assert.equal(keeper.teamNeeds.completion, 'IMPOSSIBLE')
    assert.equal(keeper.completion.reachableEmptySlots, 1)
    assert.equal(keeper.allowed, true) // fills one of the two gaps
    assert.equal(planBid(ctxOf({ squad, purse: 5000, lot: BAT(), upcoming })).allowed, false) // fills neither
})

// ── keeper ────────────────────────────────────────────────────────────────
const noKeeper = () => [...many(5, BWL), ...many(6, BAT)]
test('9. many keepers remain → NEED', () => {
    const plan = planBid(ctxOf({ squad: noKeeper(), purse: 5000, lot: BAT(), upcoming: many(30, WK) }))
    assert.equal(plan.teamNeeds.keeper, 'NEED')
    assert.equal(plan.scarcity.keepersRemaining, 30)
})
test('10. few keepers remain and rivals need them → CRITICAL', () => {
    const plan = planBid(ctxOf({ squad: noKeeper(), purse: 5000, lot: BAT(), upcoming: many(3, WK), rivals: lacking(4, 'keeper') }))
    assert.equal(plan.teamNeeds.keeper, 'CRITICAL')
    assert.equal(plan.requirements.keeper.realisticRemaining, 0)
})
test('11. the last realistic keeper on the block', () => {
    const plan = planBid(ctxOf({ squad: noKeeper(), purse: 5000, lot: WK(), upcoming: many(10, BAT) }))
    assert.equal(plan.teamNeeds.keeper, 'CRITICAL')
    assert.equal(plan.requirements.keeper.finalOpportunity, true)
    assert.equal(plan.playerImpact.unlocksRequirement, true)
})
test('12. no keeper remaining at all → IMPOSSIBLE', () => {
    const plan = planBid(ctxOf({ squad: noKeeper(), purse: 5000, lot: BAT(), upcoming: many(10, BAT) }))
    assert.equal(plan.teamNeeds.keeper, 'IMPOSSIBLE')
    assert.equal(plan.teamNeeds.completion, 'IMPOSSIBLE')
})

// ── bowling ───────────────────────────────────────────────────────────────
const fourBowlers = () => [WK(), ...many(4, BWL), ...many(6, BAT)]
test('13. enough bowling options', () => {
    const plan = planBid(ctxOf({ squad: fourBowlers(), purse: 5000, lot: BAT(), upcoming: many(20, BWL) }))
    assert.equal(plan.teamNeeds.bowling, 'NEED')
})
test('14. only one viable bowling option remains', () => {
    const plan = planBid(ctxOf({ squad: fourBowlers(), purse: 5000, lot: BAT(), upcoming: [AR(), ...many(5, BAT)] }))
    assert.equal(plan.teamNeeds.bowling, 'CRITICAL')
    assert.equal(plan.requirements.bowling.remainingAfterLot, 1)
})
test('15. no viable bowling option remains', () => {
    const plan = planBid(ctxOf({ squad: fourBowlers(), purse: 5000, lot: BAT(), upcoming: many(5, BAT) }))
    assert.equal(plan.teamNeeds.bowling, 'IMPOSSIBLE')
})

// ── Indians ───────────────────────────────────────────────────────────────
// 8 overseas (only 4 can play) + 5 Indians → the XI needs 2 more Indians.
const overseasHeavy = () => [WK(), ...many(4, BWL), ...many(5, BWL, { os: true }), ...many(3, BAT, { os: true })].slice(0, 13)
const osHeavy = () => { const s = [WK(), BWL(), BWL(), BWL(), BAT(), ...many(4, BWL, { os: true }), ...many(4, BAT, { os: true })]; return s }
test('16. enough Indians remain', () => {
    const plan = planBid(ctxOf({ squad: osHeavy(), purse: 5000, lot: BAT({ os: true }), upcoming: many(20, BAT), overseasCount: 8 }))
    assert.equal(plan.requirements.indians.need, 2)
    assert.equal(plan.teamNeeds.indians, 'NEED')
})
test('17. Indian requirement becomes critical', () => {
    const plan = planBid(ctxOf({ squad: osHeavy(), purse: 5000, lot: BAT({ os: true }), upcoming: [BAT(), BAT(), ...many(10, BAT, { os: true })], rivals: lacking(1, 'indians').map((r) => ({ ...r, squad: overseasHeavy(), overseasCount: 8 })), overseasCount: 8 }))
    assert.equal(plan.teamNeeds.indians, 'CRITICAL')
})
test('18. Indian requirement becomes impossible', () => {
    const plan = planBid(ctxOf({ squad: osHeavy(), purse: 5000, lot: BAT({ os: true }), upcoming: [BAT(), ...many(10, BAT, { os: true })], overseasCount: 8 }))
    assert.equal(plan.teamNeeds.indians, 'IMPOSSIBLE')
})

// ── overlap ───────────────────────────────────────────────────────────────
test('19. one player satisfies several requirements', () => {
    // 10 players, 4 bowling options, only 6 Indians + 4 overseas: needs 1 bowler, 1 Indian, 1 player.
    const squad = [WK(), BWL(), BWL(), BWL(), BAT(), BAT(), BWL({ os: true }), ...many(3, BAT, { os: true })]
    const r = minimumCostToCompleteXI(ctxOf({ squad, purse: 5000, lot: BAT(), upcoming: [BWL({ base: 50 }), ...many(5, BAT, { base: 50 }), ...many(5, BWL, { os: true, base: 50 })] }))
    assert.deepEqual([r.feasible, r.minimumCost, r.playersNeeded], [true, 50, 1]) // one ₹50L Indian bowler, not a bowler + an Indian (₹100L)
})
test('20. the same player is never counted twice', () => {
    const squad = [WK(), ...many(3, BWL), ...many(5, BAT)] // needs 2 bowling options
    const r = minimumCostToCompleteXI(ctxOf({ squad, purse: 5000, lot: BAT(), upcoming: [BWL(), ...many(10, BAT)] }))
    assert.equal(r.feasible, false) // one bowler can't fill two bowling places
})

// ── auction position ──────────────────────────────────────────────────────
test('21. early auction with many alternatives', () => {
    const upcoming = [...many(20, BAT), ...many(25, WK, { rating: 80 })]
    const plan = planBid(ctxOf({ squad: noKeeper(), purse: 8000, lot: WK({ rating: 80 }), upcoming, progress: 0.05 }))
    assert.equal(plan.teamNeeds.keeper, 'NEED')
    assert.equal(plan.lotContext.finalOpportunity, false)
    assert.equal(plan.lotContext.equivalentRemaining, 25)
    assert.equal(plan.requirements.keeper.nextInLots, 21)
})
test('22. late auction with few alternatives', () => {
    const plan = planBid(ctxOf({ squad: noKeeper(), purse: 8000, lot: WK(), upcoming: [BAT(), WK()], rivals: lacking(2, 'keeper'), progress: 0.97 }))
    assert.equal(plan.teamNeeds.keeper, 'CRITICAL')
    assert.equal(plan.lotContext.scarce, true)
})
test('23. re-auction: plans from the remaining re-auction list only', () => {
    const plan = planBid(ctxOf({ squad: noKeeper(), purse: 800, lot: WK({ base: 50 }), upcoming: many(5, BAT), phase: 'reauction', progress: 1 }))
    assert.equal(plan.lotContext.phase, 'reauction')
    assert.equal(plan.allowed, true)
    assert.equal(plan.lotContext.finalOpportunity, true)
    assert.equal(plan.budget.maxSafeBid, 800)
})
test('returning (unsold) players count as supply during the main round', () => {
    const plan = planBid(ctxOf({ squad: noKeeper(), purse: 5000, lot: BAT(), upcoming: many(5, BAT), returning: [WK()] }))
    assert.equal(plan.teamNeeds.keeper, 'CRITICAL')
    assert.equal(plan.requirements.keeper.onlyInReauction, true)
})

// ── candidate purchase ────────────────────────────────────────────────────
test('24. purchase keeps the XI feasible', () => {
    const ctx = ctxOf({ squad: legalXI().slice(0, 9), purse: 500, lot: BAT(), upcoming: many(5, BAT, { base: 20 }) })
    assert.equal(canCompleteXIAfterPurchase(ctx, 300), true)
})
test('25. purchase makes the XI impossible (price too high)', () => {
    const ctx = ctxOf({ squad: legalXI().slice(0, 9), purse: 500, lot: BAT(), upcoming: many(5, BAT, { base: 100 }) })
    assert.equal(canCompleteXIAfterPurchase(ctx, 450), false) // ₹100L must stay for the 11th
    assert.equal(planBid(ctx).budget.maxSafeBid, 400)
})
test('26. purchase fills a critical requirement → budget beyond the discretionary money', () => {
    // 9 players, no keeper: needs a keeper + one more player. One ₹200L keeper
    // left after this lot, and two rivals also want one → keeper is CRITICAL.
    const squad = [...many(5, BWL), ...many(4, BAT)]
    const upcoming = [WK({ base: 200 }), ...many(10, BAT, { base: 20 })]
    const rivals = lacking(2, 'keeper')
    const keeper = planBid(ctxOf({ squad, purse: 1000, lot: WK({ base: 50 }), upcoming, rivals }))
    const batsman = planBid(ctxOf({ squad, purse: 1000, lot: BAT({ base: 50 }), upcoming, rivals }))
    assert.equal(keeper.playerImpact.criticalRequirement, true)
    assert.equal(keeper.budget.discretionaryBudget, 780) // ₹1000L − (₹200L keeper + ₹20L batsman)
    assert.equal(keeper.budget.maxSafeBid, 980) // only the ₹20L batsman must stay in reserve
    assert.equal(batsman.budget.maxSafeBid, 800) // the ₹200L keeper must stay in reserve
})
test('an unlocking purchase reserves for the rest of the XI it unlocks', () => {
    // 9 players, no keeper; the only keeper is on the block; the 11th place costs ₹200L.
    const squad = [...many(5, BWL), ...many(4, BAT)]
    const plan = planBid(ctxOf({ squad, purse: 1000, lot: WK({ base: 50 }), upcoming: many(5, BAT, { base: 200 }) }))
    assert.equal(plan.playerImpact.unlocksRequirement, true)
    assert.equal(plan.completion.reachableEmptySlots, 0)
    assert.equal(plan.budget.maxSafeBid, 800) // not the whole ₹1000L: the 11th player still has to be bought
})
test('27. purchase would consume the final useful squad slot', () => {
    const squad = [...many(12, BWL), ...many(12, BAT)] // 24 players, no keeper
    const plan = planBid(ctxOf({ squad, purse: 5000, lot: BWL({ rating: 96 }), upcoming: [WK()] }))
    assert.equal(plan.allowed, false)
    assert.equal(plan.completion.feasibleAfterPurchase, false)
    assert.ok(plan.playerImpact.xiGain > 0) // he WOULD improve the XI — and is still refused
})

// ── wiring ────────────────────────────────────────────────────────────────
test('rule bots never bid above the planning layer\'s safe maximum', () => {
    const rng = createRng(4)
    const pool = players.slice()
    for (let i = 0; i < 400; i++) {
        const squad = pool.filter(() => rng() < 0.05).slice(0, Math.floor(rng() * 25))
        const lot = pool.find((p) => !squad.includes(p) && rng() < 0.05) ?? pool[0]
        const ctx = ctxOf({ squad, purse: Math.floor(rng() * 12500), lot, upcoming: pool.filter((p) => p !== lot && !squad.includes(p)).slice(0, 120), progress: rng() })
        ctx.self.overseasCount = Math.min(8, ctx.self.overseasCount)
        const plan = planBid(ctx)
        for (const id of RULE_BOT_IDS) {
            const cap = ruleBotCap(id, ctx, rng, { plan })
            assert.ok(cap <= plan.budget.maxSafeBid, `${id} bid ${cap} > safe ${plan.budget.maxSafeBid}`)
            if (!plan.allowed) assert.equal(cap, 0)
        }
    }
})

test('the RL observation ignores the new context fields', () => {
    const base = { rules: DEFAULT_RULES, lot: players[0], self: team([]), rivals: [team([])], upcoming: players.slice(1, 40), progress: 0.3 }
    assert.deepEqual(buildObservation(base), buildObservation({ ...base, returning: players.slice(40, 60), phase: 'main' }))
})
