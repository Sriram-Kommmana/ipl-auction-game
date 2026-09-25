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
