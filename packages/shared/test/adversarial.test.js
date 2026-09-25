// Rule-bot freeze validation — adversarial states (A–M). Every scenario runs
// the full safety check (safety.js) for all four personalities at the noise
// extremes, then checks what that scenario is specifically about.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { parsePlayersCsv } from '../src/playersCsv.js'
import { DEFAULT_RULES, bidBlocker, nextBidAmount } from '../src/rules.js'
import { selectBestXI } from '../src/scoring.js'
import { fairValue } from '../src/valuation.js'
import { ruleBotCap, RULE_BOT_IDS } from '../src/ruleBots.js'
import { classifyOpportunity, planBid, XI_GAIN } from '../src/planning.js'
import { overseasSlotContested } from '../src/botSignals.js'
import { AuctionSim, agentCap, createRng } from '../src/sim.js'
import { AR, BAT, BWL, WK, P, assertDeterministic, assertSafeDecision, ctxOf, legalXI, many, rivalsWith, team } from './safety.js'

const players = parsePlayersCsv(readFileSync(fileURLToPath(new URL('../../../apps/server/src/db/players.csv', import.meta.url)), 'utf8'))
const market = (n = 120) => players.slice(0, n)
const weak = (n = 40) => many(n, BAT, { rating: 72, base: 20 })
const capOf = (id, ctx) => ruleBotCap(id, ctx, Math.random, { noise: 0 })

// ── A. nearly empty purse ─────────────────────────────────────────────────
test('A. nearly empty purse: never bids beyond purse or reserve, never strands the XI', () => {
    const squadsShort = [
        [...many(5, BWL, { rating: 84 }), ...many(4, BAT, { rating: 84 })], // needs keeper + 1
        legalXI(84).slice(0, 10), // needs 1 batter
        legalXI(84) // complete
    ]
    for (const [si, squad] of squadsShort.entries()) {
        for (const purse of [0, 10, 19, 20, 21, 40, 60, 99, 100, 150, 250]) {
            for (const lot of [BAT({ rating: 86, base: 20 }), BAT({ rating: 90, base: 200 }), WK({ rating: 80, base: 20 }), BWL({ rating: 82, base: 50 })]) {
                const ctx = ctxOf({ squad, purse, lot, upcoming: [...many(3, WK, { base: 20 }), ...weak(10)] })
                const { caps, plan } = assertSafeDecision(ctx, `A squad${si} purse ${purse} ${lot.role}`)
                if (purse < lot.basePrice) assert.ok(Object.values(caps).every((c) => c === 0), 'cannot afford base → no bid')
                if (plan.budget.maxSafeBid < lot.basePrice) assert.ok(Object.values(caps).every((c) => c === 0), 'reserve → no bid')
            }
        }
    }
})

test('A2. purse exactly equal to the completion reserve: a non-requirement player is refused', () => {
    const squad = [...many(5, BWL, { rating: 84 }), ...many(5, BAT, { rating: 84 })] // needs a keeper
    const ctx = ctxOf({ squad, purse: 100, lot: BAT({ rating: 92, base: 20 }), upcoming: [WK({ base: 100 }), ...weak(5)] })
    const { caps, plan } = assertSafeDecision(ctx, 'A2')
    assert.equal(plan.budget.maxSafeBid, 0)
    assert.ok(Object.values(caps).every((c) => c === 0))
})

// ── B. nearly full squad ──────────────────────────────────────────────────
test('B. 23/24/25-player squads: no overflow, correct slot and completion maths', () => {
    for (const size of [23, 24, 25]) {
        const complete = [...legalXI(85), ...many(size - 11, BAT, { rating: 80 })]
        const noKeeper = [...many(5, BWL, { rating: 85 }), ...many(size - 5, BAT, { rating: 80 })]
        for (const [name, squad] of [['complete', complete], ['noKeeper', noKeeper]]) {
            for (const lot of [BAT({ rating: 92 }), WK({ rating: 78, base: 50 }), BWL({ rating: 90, os: true })]) {
                const ctx = ctxOf({ squad, purse: 3000, lot, upcoming: [WK({ base: 50 }), ...weak(10)] })
                const { caps, plan } = assertSafeDecision(ctx, `B ${size} ${name} ${lot.role}`)
                assert.equal(plan.completion.slotsLeft, 25 - size)
                if (size === 25) {
                    assert.equal(plan.allowed, false)
                    assert.ok(Object.values(caps).every((c) => c === 0))
                }
                // One slot left and no keeper: only a keeper may take it.
                if (size === 24 && name === 'noKeeper' && lot.role !== 'WICKET KEEPER') {
                    assert.ok(Object.values(caps).every((c) => c === 0), `last slot kept for the keeper (${lot.role})`)
                }
                if (size === 24 && name === 'noKeeper' && lot.role === 'WICKET KEEPER') {
                    assert.equal(plan.teamNeeds.keeper, 'CRITICAL')
                    assert.ok(Object.values(caps).every((c) => c > 0), 'the keeper for the last slot is bought')
                }
            }
        }
    }
})

// ── C. overseas saturation ────────────────────────────────────────────────
test('C. 6/7/8 overseas: cap respected, contested-slot logic, Indians unaffected', () => {
    for (const os of [6, 7, 8]) {
        const squad = [...legalXI(84), ...many(os, BAT, { os: true, rating: 83 })]
        for (const phase of ['main', 'reauction']) {
            const osLot = BAT({ os: true, rating: 90 })
            const inLot = BAT({ rating: 90 })
            const upcoming = [...many(4, BAT, { os: true, rating: 92 }), ...weak(20)]
            const osCtx = ctxOf({ squad, purse: 5000, lot: osLot, upcoming, phase, progress: phase === 'main' ? 0.6 : 1 })
            const inCtx = ctxOf({ squad, purse: 5000, lot: inLot, upcoming, phase, progress: phase === 'main' ? 0.6 : 1 })
            const o = assertSafeDecision(osCtx, `C os${os} ${phase} overseas`)
            const i = assertSafeDecision(inCtx, `C os${os} ${phase} indian`)
            if (os === 8) assert.ok(Object.values(o.caps).every((c) => c === 0), '8 overseas → no overseas bid')
            assert.equal(overseasSlotContested(inCtx), false)
            if (os < 8) {
                // 4 better overseas to come vs (8 − os − 1) slots left after him.
                assert.equal(overseasSlotContested(osCtx), 4 > 8 - os - 1)
            }
            assert.ok(Object.values(i.caps).some((c) => c > 0), 'an Indian upgrade is still bought')
        }
    }
})

// ── D. keeper scarcity ────────────────────────────────────────────────────
const noKeeperSquad = () => [...many(5, BWL, { rating: 84 }), ...many(6, BAT, { rating: 84 })]
test('D. keeper scarcity: urgency rises as keepers run out; never trapped', () => {
    const lot = WK({ rating: 80, base: 50 })
    const cases = {
        'no keeper left after him': [],
        'one keeper left': [WK({ rating: 79 })],
        'many keepers left': many(12, WK, { rating: 80 }),
        'only expensive keepers left': many(3, WK, { rating: 88, base: 200 }),
        'keeper only at the very end': [...weak(60), WK({ rating: 78 })]
    }
    const rivals = rivalsWith(6000, () => many(11, BAT)) // every rival also needs a keeper
    const out = {}
    for (const [name, keepers] of Object.entries(cases)) {
        const ctx = ctxOf({ squad: noKeeperSquad(), purse: 3000, lot, upcoming: [...keepers, ...weak(30)], rivals })
        const { caps, plan } = assertSafeDecision(ctx, `D ${name}`)
        out[name] = { caps, plan }
        assert.ok(Object.values(caps).every((c) => c > 0), `${name}: the keeper is bid on`)
    }
    assert.equal(out['no keeper left after him'].plan.teamNeeds.keeper, 'CRITICAL')
    assert.equal(out['no keeper left after him'].plan.lotContext.finalOpportunity, true)
    assert.equal(out['one keeper left'].plan.teamNeeds.keeper, 'CRITICAL')
    assert.equal(out['many keepers left'].plan.teamNeeds.keeper, 'NEED') // 12 left > 1 + 9 rivals needing one
    // Final opportunity → at least the personality's share of the safe maximum.
    for (const id of RULE_BOT_IDS) {
        assert.ok(out['no keeper left after him'].caps[id] >= out['many keepers left'].caps[id], `${id}: final keeper ≥ plentiful keeper`)
    }
    // A non-keeper with the last keeper still to come must leave room for him.
    const ctx = ctxOf({ squad: [...noKeeperSquad(), ...many(12, BAT, { rating: 80 })], purse: 400, lot: BAT({ rating: 93, base: 200 }), upcoming: [WK({ base: 150 }), ...weak(5)] })
    const { plan } = assertSafeDecision(ctx, 'D reserve for the late keeper')
    assert.ok(plan.budget.maxSafeBid <= 400 - 150)
})

test('D2. last squad slot, no keeper in the market at all → XI impossible, bots only improve the partial XI', () => {
    const ctx = ctxOf({ squad: [...noKeeperSquad(), ...many(13, BAT, { rating: 80 })], purse: 3000, lot: BAT({ rating: 90 }), upcoming: weak(10) })
    const { plan } = assertSafeDecision(ctx, 'D2')
    assert.equal(plan.teamNeeds.keeper, 'IMPOSSIBLE')
    assert.equal(plan.completion.reachableEmptySlots, 1)
})

// ── E. bowling scarcity ───────────────────────────────────────────────────
test('E. bowling scarcity: 0–4 bowling options owned, supply from none to plenty', () => {
    const rivals = rivalsWith(6000, () => [WK(), ...many(10, BAT)])
    for (const owned of [0, 2, 4]) {
        const squad = [WK({ rating: 84 }), ...many(owned, BWL, { rating: 84 }), ...many(10 - owned, BAT, { rating: 84 })]
        const lot = BWL({ rating: 82, base: 50 })
        for (const [name, supply] of [['none', []], ['exactly enough', many(4 - owned, BWL)], ['plenty', many(30, BWL, { rating: 80 })], ['expensive', many(6, AR, { rating: 88, base: 200 })]]) {
            const ctx = ctxOf({ squad, purse: 2500, lot, upcoming: [...supply, ...weak(20)], rivals })
            const { caps, plan } = assertSafeDecision(ctx, `E owned ${owned} supply ${name}`)
            if (name === 'none' && owned === 4) assert.equal(plan.lotContext.finalOpportunity, true)
            if (plan.completion.reachableEmptySlots === 0 || plan.playerImpact.unlocksRequirement) assert.ok(Object.values(caps).every((c) => c > 0), `E owned ${owned} ${name}: bowler bought`)
        }
    }
})

// ── F. late critical opportunity ──────────────────────────────────────────
test('F. requirement becomes critical in the last lots: safe maximum honoured, reaction strong', () => {
    const squad = [...many(4, BWL, { rating: 85 }), WK({ rating: 85 }), ...many(12, BAT, { rating: 84 })] // 4 bowling options
    const lot = BWL({ rating: 80, base: 30 })
    const rivals = rivalsWith(2000, () => [WK(), ...many(4, BWL), ...many(6, BAT)])
    for (const purse of [200, 800, 3000]) {
        const late = ctxOf({ squad, purse, lot, upcoming: [BWL({ base: 30 })], progress: 0.99, rivals })
        const early = ctxOf({ squad, purse, lot, upcoming: [...many(40, BWL, { rating: 80 }), ...weak(20)], progress: 0.3, rivals: rivalsWith(6000) })
        const l = assertSafeDecision(late, `F late purse ${purse}`)
        const e = assertSafeDecision(early, `F early purse ${purse}`)
        assert.equal(l.plan.teamNeeds.bowling, 'CRITICAL')
        for (const id of RULE_BOT_IDS) assert.ok(l.caps[id] >= e.caps[id], `${id} purse ${purse}: late critical ${l.caps[id]} ≥ early ${e.caps[id]}`)
    }
})

// ── G. high-value stars ───────────────────────────────────────────────────
test('G. stars: Star Chaser is the most aggressive on a big-gain star but never beyond safety; small-gain star stays cheap', () => {
    const star = () => BAT({ rating: 96, base: 200 })
    const scarce = market(200).filter((p) => p.rating < 88)
    const abundant = market(200)
    const results = {}
    for (const [name, squad, upcoming, purse] of [
        ['big gain, scarce', legalXI(80), scarce, 9000],
        ['big gain, abundant', legalXI(80), abundant, 9000],
        ['low gain (strong XI)', [...legalXI(95), ...many(5, BAT, { rating: 95 })], abundant, 9000],
        ['big gain, small purse', legalXI(80), abundant, 1500],
        ['big gain, early empty squad', [], abundant, 12500]
    ]) {
        const ctx = ctxOf({ squad, purse, lot: star(), upcoming, progress: squad.length ? 0.4 : 0.02 })
        results[name] = assertSafeDecision(ctx, `G ${name}`)
    }
    const big = results['big gain, abundant'].caps
    assert.ok(big.starChaser >= Math.max(big.moneyball, big.opportunist), `Star Chaser leads on a star (${JSON.stringify(big)})`)
    assert.ok(results['big gain, scarce'].caps.starChaser >= big.starChaser * 0.95)
    // +0.09 star: the non-star personalities stay well under fair value; Star
    // Chaser's star premium may take it higher, but never past fair value and
    // far below what it pays for a star who transforms the XI.
    const low = results['low gain (strong XI)'].caps
    for (const id of ['moneyball', 'balancedBuilder', 'opportunist']) assert.ok(low[id] <= fairValue(star()) * 0.6, `${id}: tiny-gain star ${low[id]}`)
    assert.ok(low.starChaser <= fairValue(star()), `starChaser: tiny-gain star ${low.starChaser}`)
    assert.ok(low.starChaser < big.starChaser, `starChaser: tiny-gain ${low.starChaser} < big-gain ${big.starChaser}`)
    // Disciplined: even an empty squad never puts more than half the purse into one star.
    for (const [id, c] of Object.entries(results['big gain, early empty squad'].caps)) assert.ok(c <= 12500 * 0.5, `${id}: ${c}`)
})

// ── H. low-value marginal players ─────────────────────────────────────────
test('H. marginal discipline at XI gain ≈ 0 / 0.1 / 0.3 / 0.5', () => {
    const squad = [...legalXI(85), ...many(6, BAT, { rating: 80 })]
    const byGain = {}
    for (const [rating, label] of [[85, '0'], [86, '0.1'], [88, '0.3'], [90, '0.5']]) {
        const lot = BAT({ rating, base: 50 })
        const ctx = ctxOf({ squad, purse: 5000, lot, upcoming: market(200), progress: 0.6 })
        const { caps, plan } = assertSafeDecision(ctx, `H gain≈${label}`)
        byGain[label] = { caps, gain: plan.playerImpact.xiGain, fv: fairValue(lot) }
    }
    assert.ok(Object.values(byGain['0'].caps).every((c) => c === 0), 'no XI gain, 17 players → no bid')
    for (const id of RULE_BOT_IDS) {
        assert.ok(byGain['0.1'].caps[id] <= 0.6 * byGain['0.1'].fv, `${id} +0.1 ≤ 0.6 × fv`)
        assert.ok(byGain['0.3'].caps[id] <= byGain['0.3'].fv, `${id} +0.3 ≤ fv`)
        assert.ok(byGain['0.5'].caps[id] >= byGain['0.1'].caps[id], `${id} value rises with gain`)
    }
})

// ── I. re-auction edge cases ──────────────────────────────────────────────
test('I. re-auction: every category, overseas/Indian, 8 or 0 overseas, nearly full, nearly broke', () => {
    const re = (o) => ctxOf({ ...o, phase: 'reauction', progress: 1, upcoming: o.upcoming ?? weak(8) })
    const cases = [
        ['critical keeper', re({ squad: noKeeperSquad(), purse: 2000, lot: WK({ rating: 76, base: 50 }) }), 'critical'],
        ['useful', re({ squad: legalXI(80), purse: 2000, lot: BAT({ rating: 92 }) }), 'useful'],
        ['marginal', re({ squad: legalXI(85), purse: 2000, lot: BAT({ rating: 88 }) }), 'marginal'],
        ['depth', re({ squad: [...legalXI(85), BAT({ rating: 80 })], purse: 2000, lot: WK({ rating: 78, base: 50 }) }), 'depth'],
        ['none', re({ squad: [...legalXI(85), WK({ rating: 84 }), ...many(3, BAT, { rating: 84 }), BWL({ rating: 84 })], purse: 2000, lot: BAT({ rating: 70 }) }), 'none'],
        ['overseas, 8 overseas', re({ squad: [...legalXI(80), ...many(8, BAT, { os: true, rating: 70 })], purse: 2000, lot: BAT({ os: true, rating: 95 }) }), 'none'],
        ['overseas, 0 overseas', re({ squad: legalXI(80), purse: 2000, lot: BAT({ os: true, rating: 92 }) }), 'useful'],
        ['nearly full (24)', re({ squad: [...legalXI(80), ...many(13, BAT, { rating: 78 })], purse: 2000, lot: BAT({ rating: 92 }) }), 'useful'],
        ['full (25)', re({ squad: [...legalXI(80), ...many(14, BAT, { rating: 78 })], purse: 2000, lot: BAT({ rating: 92 }) }), 'none'],
        ['nearly broke', re({ squad: legalXI(80), purse: 30, lot: BAT({ rating: 92, base: 50 }) }), 'none']
    ]
    for (const [name, ctx, category] of cases) {
        const { caps } = assertSafeDecision(ctx, `I ${name}`)
        assert.equal(classifyOpportunity(ctx).category, category, `I ${name}`)
        if (category === 'none') assert.ok(Object.values(caps).every((c) => c === 0), `I ${name}: no bid`)
        if (['critical', 'useful', 'depth'].includes(category)) assert.ok(Object.values(caps).every((c) => c > 0), `I ${name}: bought`)
    }
})

// ── J. rival extremes ─────────────────────────────────────────────────────
test('J. rivals broke / enormous / uneven / identical / uneven squads: safe, Opportunist reacts', () => {
    const lot = BAT({ rating: 90, base: 150 })
    const squad = legalXI(80)
    const rivalSets = {
        broke: rivalsWith(40, () => legalXI(80)),
        enormous: rivalsWith(12500),
        uneven: rivalsWith((i) => [10, 12500, 300, 9000, 50, 6000, 20, 11000, 700][i]),
        identical: rivalsWith(5000, () => legalXI(82)),
        'uneven squads': rivalsWith(5000, (i) => (i % 2 ? [] : [...legalXI(82), ...many(13, BAT)])),
        none: []
    }
    const caps = {}
    for (const [name, rivals] of Object.entries(rivalSets)) {
        const ctx = ctxOf({ squad, purse: 5000, lot, upcoming: market(200), rivals, progress: 0.6 })
        caps[name] = assertSafeDecision(ctx, `J ${name}`).caps
    }
    assert.ok(caps.broke.opportunist > caps.enormous.opportunist, `Opportunist pounces when rivals are broke (${caps.broke.opportunist} vs ${caps.enormous.opportunist})`)
})

// ── K. simultaneous competition ───────────────────────────────────────────
test('K. several bots want the same player: ladder never lets a bot outbid itself or exceed its cap', () => {
    const rng = createRng(77)
    for (let trial = 0; trial < 300; trial++) {
        const lot = BAT({ rating: 85 + Math.floor(rng() * 11), base: [20, 50, 100, 150, 200][Math.floor(rng() * 5)] })
        const teams = RULE_BOT_IDS.flatMap((id, k) => [0, 1].map((j) => ({ id, t: team(legalXI(78 + ((k + j) % 5)), 1000 + Math.floor(rng() * 9000), `T${k}${j}`) })))
        const caps = teams.map(({ id, t }) => ruleBotCap(id, { rules: DEFAULT_RULES, lot, self: t, rivals: teams.filter((x) => x.t !== t).map((x) => x.t), upcoming: market(150), returning: [], phase: 'main', progress: rng() }, rng))
        let price = lot.basePrice
        let leader = -1
        let steps = 0
        for (;;) {
            const amount = nextBidAmount(price, leader !== -1)
            if (leader !== -1) assert.ok(amount > price, 'increment is positive')
            const eligible = caps.map((c, i) => i).filter((i) => i !== leader && caps[i] >= amount &&
                !bidBlocker({ team: teams[i].t, lot: { ...lot, currentBidderId: leader === -1 ? '' : teams[leader].t.teamId }, amount }))
            if (!eligible.length) break
            const who = eligible[Math.floor(rng() * eligible.length)]
            assert.notEqual(who, leader, 'bot outbid itself')
            assert.ok(amount <= caps[who] && amount <= teams[who].t.purseLeft, 'bid above cap / purse')
            leader = who
            price = amount
            assert.ok(++steps < 500, 'ladder terminates')
        }
        if (leader !== -1) {
            const sorted = [...caps].sort((a, b) => b - a)
            assert.ok(price <= caps[leader], 'winner pays at most its cap')
            // Nobody else could have paid one more increment.
            const next = nextBidAmount(price, true)
            assert.ok(caps.every((c, i) => i === leader || c < next), `another bot could still raise (${sorted})`)
        }
    }
})

// ── L. final lots & M. re-auction transition (real full-pool auctions) ────
const LINEUP = ['moneyball', 'starChaser', 'balancedBuilder', 'opportunist', 'starChaser', 'balancedBuilder', 'moneyball', 'opportunist', 'balancedBuilder', 'starChaser']

test('L/M. last 1/2/5/10 lots, the main→re-auction transition and termination on real auctions', () => {
    for (const seed of [401, 402]) {
        const rng = createRng(seed)
        const sim = new AuctionSim({ players, teamCount: LINEUP.length, rng })
        const mainPool = [...sim.pool]
        const soldTo = new Map()
        let lots = 0
        let transitioned = false
        while (!sim.done) {
            assert.ok(++lots < 1000, 'terminates')
            const remainingMain = sim.phase === 'main' ? sim.mainLength - sim.index : 0
            const ctxs = LINEUP.map((_, i) => sim.contextFor(i))
            // Context integrity.
            const ctx = ctxs[0]
            assert.equal(ctx.phase, sim.phase)
            assert.equal(ctx.lot.slNo, sim.pool[sim.index])
            assert.deepEqual(ctx.upcoming.map((p) => p.slNo), sim.pool.slice(sim.index + 1))
            assert.deepEqual(ctx.returning.map((p) => p.slNo), sim.phase === 'main' ? sim.unsold : [])
            assert.ok(!ctx.upcoming.some((p) => p.slNo === ctx.lot.slNo) && !ctx.returning.some((p) => p.slNo === ctx.lot.slNo), 'lot is not its own supply')
            if ([1, 2, 5, 10].includes(remainingMain) || (sim.phase === 'reauction' && sim.pool.length - sim.index <= 10)) {
                ctxs.forEach((c, i) => assertSafeDecision(c, `seed ${seed} ${sim.phase} ${remainingMain || sim.pool.length - sim.index} left, team ${i}`))
            }
            const wasMain = sim.phase === 'main'
            const unsoldBefore = [...sim.unsold]
            const out = sim.resolveLot(LINEUP.map((persona, i) => agentCap({ kind: 'rule', persona }, ctxs[i], rng)))
            if (out.winner !== null) {
                assert.ok(!soldTo.has(out.slNo), `player ${out.slNo} sold twice`)
                soldTo.set(out.slNo, out.winner)
            }
            if (wasMain && sim.phase === 'reauction') {
                transitioned = true
                const expected = [...unsoldBefore, ...(out.winner === null ? [out.slNo] : [])].sort((a, b) => a - b)
                assert.deepEqual([...sim.pool].sort((a, b) => a - b), expected, 're-auction pool = exactly the main-round unsold')
                assert.equal(sim.index, 0)
                assert.deepEqual(sim.unsold, [])
            }
        }
        assert.ok(transitioned, 'a re-auction happened')
        const main = sim.history.filter((h) => h.phase === 'main').map((h) => h.slNo)
        assert.deepEqual(main, mainPool, 'main-round history = the main pool, in order, each player once')
        const reHistory = sim.history.filter((h) => h.phase === 'reauction').map((h) => h.slNo)
        assert.equal(new Set(reHistory).size, reHistory.length, 're-auction players once each')
        for (const t of sim.teams) {
            assert.ok(t.purseLeft >= 0 && t.playerCount <= 25 && t.overseasCount <= 8)
            assert.equal(t.squad.length, t.playerCount)
            assert.equal(selectBestXI(t.squad).emptySlots, 0, 'legal XI at the end')
        }
    }
})

// ── determinism ───────────────────────────────────────────────────────────
test('noise-free decisions are deterministic; same seed → identical auction', () => {
    assertDeterministic(ctxOf({ squad: legalXI(80), lot: BAT({ rating: 90 }), upcoming: market() }), 'fixture')
    const play = (seed) => {
        const rng = createRng(seed)
        const sim = new AuctionSim({ players, teamCount: 10, rng })
        while (!sim.done) sim.resolveLot(LINEUP.map((persona, i) => agentCap({ kind: 'rule', persona }, sim.contextFor(i), rng)))
        return JSON.stringify(sim.history)
    }
    assert.equal(play(509), play(509))
})

// ── rules other than the defaults (no hidden 10-team / 25 / 8 assumptions) ─
test('non-default rules and team counts: 6 teams, smaller purse, 20-player squads, 6 overseas', () => {
    const rules = { pursePerTeam: 9000, maxPlayers: 20, maxOverseas: 6 }
    for (const teams of [6, 8]) {
        const rng = createRng(600 + teams)
        const sim = new AuctionSim({ players, teamCount: teams, rules, rng })
        const lineup = Array.from({ length: teams }, (_, i) => RULE_BOT_IDS[i % 4])
        let lots = 0
        while (!sim.done) {
            const ctxs = lineup.map((_, i) => sim.contextFor(i))
            if (lots++ % 25 === 0) ctxs.forEach((c, i) => assertSafeDecision(c, `rules ${teams} teams lot ${lots} team ${i}`))
            sim.resolveLot(lineup.map((persona, i) => agentCap({ kind: 'rule', persona }, ctxs[i], rng)))
        }
        for (const t of sim.teams) {
            assert.ok(t.purseLeft >= 0 && t.playerCount <= 20 && t.overseasCount <= 6, JSON.stringify({ p: t.purseLeft, n: t.playerCount, o: t.overseasCount }))
            assert.equal(selectBestXI(t.squad).emptySlots, 0)
        }
    }
})

// P is used by scenario builders in safety.js; keep the import referenced.
void P
