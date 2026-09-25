// Baselines and the Node evaluation harness (Phase 2A §17/§20).
//
// A controller plays the learner seat:
//   { kind: 'cap',    cap(episode, rng) }      raw cap (a frozen rule bot in the chair)
//   { kind: 'action', act(episode, rng) }      an act-v2 action (RL policies, grid baselines)
// Every run is deterministic: manifests fix the auction, and the learner
// seat's random stream is derived from the episode seed.

import { agentCap, createRng } from '../sim.js'
import { XI_SIZE } from '../rules.js'
import { fairValue } from '../valuation.js'
import { RULE_PERSONAS } from '../personas.js'
import { PASS } from './actionSpec.js'
import { RlEpisode } from './env.js'
import { actionScores, selectAction } from './policy.js'

const legalActions = (mask) => mask.flatMap((ok, a) => (ok ? [a] : []))

// Largest legal bid whose cap is ≤ target; PASS if none (or the cheapest
// legal bid when the completion shield forbids passing).
export const actionForTarget = (maskResult, target) => {
    const bids = legalActions(maskResult.mask).filter((a) => a !== PASS)
    const fitting = bids.filter((a) => maskResult.caps[a] <= target)
    if (fitting.length) return fitting.reduce((best, a) => (maskResult.caps[a] > maskResult.caps[best] ? a : best))
    return maskResult.mask[PASS] ? PASS : bids[0]
}

const ruleBaseline = (persona) => ({ kind: 'cap', cap: (ep, rng) => agentCap({ kind: 'rule', persona }, ep.pending.ctx, rng) })

export const BASELINES = Object.freeze({
    moneyball: ruleBaseline('moneyball'),
    starChaser: ruleBaseline('starChaser'),
    balancedBuilder: ruleBaseline('balancedBuilder'),
    opportunist: ruleBaseline('opportunist'),
    // The product today: the learner's RL seat plays its frozen rule fallback.
    productFallback: { kind: 'cap', cap: (ep, rng) => agentCap({ kind: 'rule', persona: ep.entry.seats[ep.learner].fallback }, ep.pending.ctx, rng) },
    randomLegal: { kind: 'action', act: (ep, rng) => { const l = legalActions(ep.pending.mask.mask); return l[Math.floor(rng() * l.length)] } },
    // cap = 1.0 × fair value, clamped by the mask.
    fairValue: { kind: 'action', act: (ep) => actionForTarget(ep.pending.mask, fairValue(ep.pending.ctx.lot, ep.rules.pursePerTeam)) },
    // cap = fair value × XI gain as a share of a full XI slot; no bid below a 0.05 gain.
    plannerGreedy: {
        kind: 'action',
        act: (ep) => {
            const { lot } = ep.pending.ctx
            const gain = ep.pending.plan.playerImpact.xiGain
            const share = Math.min(1, Math.max(0, gain / (lot.rating / XI_SIZE)))
            return actionForTarget(ep.pending.mask, gain > 0.05 ? fairValue(lot, ep.rules.pursePerTeam) * share : 0)
        }
    }
})

// A validated rl-policy-v2 as a controller (argmax / temperature per algorithm).
export const policyController = (policy) => ({
    kind: 'action',
    act: (ep, rng) => selectAction(policy, actionScores(policy, ep.pending.obs), ep.pending.mask.mask, rng)
})

export const runEpisode = ({ players, entry, controller, snapshots = [], tremble = 0 }) => {
    const ep = new RlEpisode({ players, entry, snapshots, tremble })
    const rng = ep.learnerRng
    ep.reset()
    let step
    do {
        step = controller.kind === 'cap' ? ep.stepCap(controller.cap(ep, rng)) : ep.step(controller.act(ep, rng))
    } while (!step.done)
    return step.info.episode
}

export const METRICS = ['xi', 'strength', 'legalXI', 'strongXI', 'rank', 'purseLeft', 'purseLeftShare', 'squadSize', 'overseas', 'buys', 'stars', 'marginalBuys', 'reauctionBuys', 'xiGainPer1000', 'contests', 'decisions', 'shieldActivations', 'shieldWins', 'return']

const meanOf = (xs) => xs.reduce((s, x) => s + x, 0) / xs.length
// Deterministic percentile bootstrap of the mean.
export const bootstrapCI = (xs, { iterations = 2000, seed = 7 } = {}) => {
    if (!xs.length) return [NaN, NaN]
    const rng = createRng(seed)
    const means = Array.from({ length: iterations }, () => {
        let s = 0
        for (let i = 0; i < xs.length; i++) s += xs[Math.floor(rng() * xs.length)]
        return s / xs.length
    }).sort((a, b) => a - b)
    return [means[Math.floor(0.025 * iterations)], means[Math.floor(0.975 * iterations)]]
}

export const summarise = (rows) => Object.fromEntries(METRICS.map((m) => {
    const xs = rows.map((r) => Number(r[m]))
    return [m, { mean: meanOf(xs), ci95: bootstrapCI(xs) }]
}))

// Evaluate controllers on the same manifest entries (paired by seed).
// Returns per-controller episodes, overall and per-purse-stratum summaries,
// and paired XI differences against `reference`.
export const evaluate = ({ players, entries, controllers, reference = 'moneyball', snapshots = [] }) => {
    const episodes = {}
    for (const [name, controller] of Object.entries(controllers)) {
        episodes[name] = entries.map((entry) => runEpisode({ players, entry, controller, snapshots }))
    }
    const report = {}
    for (const [name, rows] of Object.entries(episodes)) {
        const strata = {}
        for (const s of ['low', 'normal', 'high']) {
            const sub = rows.filter((r) => r.stratum === s)
            if (sub.length) strata[s] = { n: sub.length, ...summarise(sub) }
        }
        const paired = episodes[reference] && name !== reference
            ? (() => {
                const diffs = rows.map((r, i) => r.xi - episodes[reference][i].xi)
                return { reference, xiDiffMean: meanOf(diffs), ci95: bootstrapCI(diffs) }
            })()
            : null
        report[name] = { n: rows.length, ...summarise(rows), strata, paired }
    }
    return { episodes, report }
}

export const RULE_BASELINE_IDS = Object.freeze(Object.keys(RULE_PERSONAS))
