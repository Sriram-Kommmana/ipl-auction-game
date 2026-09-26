// Phase 2B — baseline evaluator: determinism, pairing, baseline semantics.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { AuctionSim, agentCap, createRng } from '../src/sim.js'
import { DEFAULT_RULES } from '../src/rules.js'
import { selectBestXI } from '../src/scoring.js'
import { BASELINES, PASS, actionForTarget, deriveSeed, evaluate, policyController, runEpisode, validatePolicy } from '../src/rl/index.js'
import { players, randomPolicy } from './rlHelpers.js'

const validation = JSON.parse(readFileSync(fileURLToPath(new URL('../data/rl-manifests/validation.json', import.meta.url)), 'utf8')).entries

test('evaluator: deterministic, paired by seed, per-stratum summaries, all eight baselines complete', () => {
    const entries = validation.slice(0, 1)
    const a = evaluate({ players, entries, controllers: BASELINES })
    const b = evaluate({ players, entries, controllers: { moneyball: BASELINES.moneyball, randomLegal: BASELINES.randomLegal } })
    assert.deepEqual(Object.keys(a.report).sort(), ['balancedBuilder', 'fairValue', 'moneyball', 'opportunist', 'plannerGreedy', 'productFallback', 'randomLegal', 'starChaser'])
    assert.deepEqual(a.episodes.randomLegal, b.episodes.randomLegal, 'same episodes whichever other controllers run')
    assert.deepEqual(a.episodes.moneyball, b.episodes.moneyball)
    for (const [name, r] of Object.entries(a.report)) {
        assert.equal(r.n, 1)
        assert.ok(Number.isFinite(r.xi.mean) && r.xi.ci95.every(Number.isFinite), name)
        if (name !== 'moneyball') assert.equal(r.paired.reference, 'moneyball')
    }
    assert.deepEqual(a.episodes.moneyball.map((e) => e.seed), entries.map((e) => e.seed))
})

test('a frozen rule bot in the learner seat plays exactly as the frozen bot (independent replay, same random streams)', () => {
    const entry = validation[3]
    const summary = runEpisode({ players, entry, controller: BASELINES.opportunist })
    const rng = createRng(entry.seed)
    const learnerRng = createRng(deriveSeed(entry.seed, 'learner'))
    const sim = new AuctionSim({ players, teamCount: 10, rules: { ...DEFAULT_RULES, pursePerTeam: entry.purse }, rng })
    while (!sim.done) {
        const ctxs = entry.seats.map((_, i) => sim.contextFor(i))
        const caps = entry.seats.map((seat, i) => {
            if (i === entry.learnerSeat) return agentCap({ kind: 'rule', persona: 'opportunist' }, ctxs[i], learnerRng)
            if (seat.type === 'human' && seat.proxy === 'passive') return 0
            return agentCap(seat.type === 'human' ? { kind: 'rule', persona: seat.persona, noise: seat.noise } : { kind: 'rule', persona: seat.persona }, ctxs[i], rng)
        })
        sim.resolveLot(caps)
    }
    const team = sim.teams[entry.learnerSeat]
    assert.equal(summary.purseLeft, team.purseLeft)
    assert.equal(summary.squadSize, team.playerCount)
    assert.equal(summary.xiTotal, selectBestXI(team.squad).total)
})

test('productFallback plays the learner RL seat’s own fallback persona', () => {
    const entry = validation.find((e) => e.seats[e.learnerSeat].fallback === 'balancedBuilder')
    assert.deepEqual(runEpisode({ players, entry, controller: BASELINES.productFallback }), runEpisode({ players, entry, controller: BASELINES.balancedBuilder }))
})

test('actionForTarget: largest legal cap ≤ target; PASS when nothing fits; cheapest bid when shielded', () => {
    const m = { mask: [1, 1, 0, 1, 1, 0], caps: [0, 50, 60, 80, 120, 200] }
    assert.equal(actionForTarget(m, 100), 3)
    assert.equal(actionForTarget(m, 500), 4)
    assert.equal(actionForTarget(m, 10), PASS)
    assert.equal(actionForTarget({ ...m, mask: [0, 1, 0, 1, 1, 0] }, 10), 1)
})

test('a random-init rl-policy-v2 runs as a controller (inference only)', () => {
    const summary = runEpisode({ players, entry: validation[5], controller: policyController(validatePolicy(randomPolicy('d3qn', 51))) })
    assert.ok(summary.decisions > 0 && Number.isFinite(summary.xi))
})

test('Phase 2C: every evaluated episode passes the safety invariants; the auditor catches tampering', async () => {
    const { RlEpisode, auditAuction } = await import('../src/rl/index.js')
    const entry = validation[7]
    const ep = new RlEpisode({ players, entry })
    const rng = ep.learnerRng
    ep.reset()
    let step
    do step = ep.step(BASELINES.randomLegal.act(ep, rng))
    while (!step.done)
    const summary = step.info.episode
    assert.equal(summary.invariantViolations, 0, summary.violations.join('; '))
    assert.equal(summary.actionCounts.reduce((s, c) => s + c, 0), summary.decisions)
    assert.equal(summary.reauctionBuys, summary.reauctionCritical + summary.reauctionUseful + summary.reauctionMarginal + summary.reauctionDepth + summary.reauctionNone)
    assert.deepEqual(auditAuction(ep.sim), [])

    const t = ep.sim.teams[0]
    const sold = ep.sim.history.find((h) => h.winner !== null)
    const cases = [
        ['negative purse', () => { t.purseLeft = -10 }, /purse -10/],
        ['duplicate sale', () => { ep.sim.history.push({ ...sold, phase: 'reauction' }) }, /sold twice|re-auctioned without/],
        ['overseas overflow', () => { t.overseasCount = 99 }, /overseas/],
        ['squad mismatch', () => { t.playerCount += 1 }, /squad/],
        ['below base', () => { ep.sim.history.find((h) => h.winner !== null).price = 1 }, /sold for 1/]
    ]
    for (const [name, tamper, pattern] of cases) {
        const saved = structuredClone({ t: { ...t }, history: ep.sim.history.map((h) => ({ ...h })) })
        tamper()
        assert.ok(auditAuction(ep.sim).some((v) => pattern.test(v)), name)
        Object.assign(t, saved.t)
        ep.sim.history = saved.history
    }
    assert.deepEqual(auditAuction(ep.sim), [])
})

test('Phase 2C: the locked validation baselines still reproduce exactly (sampled seeds, every controller)', () => {
    const lockedFile = fileURLToPath(new URL('../data/rl-baselines/validation.episodes.json', import.meta.url))
    const locked = JSON.parse(readFileSync(lockedFile, 'utf8'))
    assert.equal(locked.format, 'rl-baselines-v1')
    assert.deepEqual(locked.controllers, Object.keys(BASELINES))
    for (const k of [0, 250]) {
        for (const name of locked.controllers) {
            const again = runEpisode({ players, entry: validation[k], controller: BASELINES[name] })
            assert.deepEqual(again, locked.episodes[name][k], `${name} seed ${validation[k].seed} no longer matches the locked baseline`)
        }
    }
})
