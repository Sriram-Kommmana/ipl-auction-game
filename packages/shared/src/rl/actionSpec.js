// act-v3 — THE canonical RL action space (Phase 2A actions, frozen; act-v3
// adopted in Phase 2C.2 with the completion shield v2 — mask.js). 20 discrete
// willingness-to-pay levels, decided once when a lot opens:
//
//   0      PASS
//   1      BASE       cap = base price (buy only if uncontested)
//   2..18  FV_m       cap = m × fair value, m ∈ FV_MULTIPLIERS
//   19     MAX_SAFE   cap = the planner's maxSafeBid
//
// Effective cap = floor(min(level, maxSafeBid)). The auction runtime then
// bids one increment at a time up to the cap, exactly as for the rule bots.
// The actions and caps are unchanged from act-v2; act-v3 changes only which
// of them are legal (the completion shield in ACT_SPEC.shield). Policies
// stamped act-v2 are rejected by the loader on purpose: they were trained
// against a different legal-action set.

import { bidIncrement } from '../rules.js'
import { specHash } from './hash.js'

export const ACT_VERSION = 'act-v3'
export const FV_MULTIPLIERS = Object.freeze([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0])

export const ACTIONS = Object.freeze([
    Object.freeze({ index: 0, name: 'PASS', kind: 'pass' }),
    Object.freeze({ index: 1, name: 'BASE', kind: 'base' }),
    ...FV_MULTIPLIERS.map((m, i) => Object.freeze({ index: i + 2, name: `FV_${m.toFixed(2)}`, kind: 'fairValue', multiplier: m })),
    Object.freeze({ index: FV_MULTIPLIERS.length + 2, name: 'MAX_SAFE', kind: 'maxSafe' })
])
export const ACTION_COUNT = ACTIONS.length
export const PASS = 0
export const BASE = 1
export const MAX_SAFE = ACTION_COUNT - 1
export const ACTION_NAMES = Object.freeze(ACTIONS.map((a) => a.name))

export const ACT_SPEC = Object.freeze({
    version: ACT_VERSION,
    count: ACTION_COUNT,
    actions: ACTIONS.map(({ index, name, kind, multiplier }) => ({ index, name, kind, multiplier: multiplier ?? null })),
    cap: 'floor(min(level, maxSafeBid))',
    legal: 'act-v2 legal set (rules allow a bid at base AND planner allows AND cap >= base AND effective ladder price above every lower legal action; PASS masked when buying unlocks a better XI, or he is a final opportunity for a requirement he fills with XI gain > 0.05) minus the actions the completion shield removes; the shield never adds an action',
    shield: {
        version: 'completion-shield-v2',
        params: { criticalBuffer: 1, warningBuffer: 3, marginIncrements: 2 },
        tracked: 'unmet planner requirements keeper, bowling, indians, players (need, remainingAfterLot, rivalsNeeding, cheapestBase, status); plus, while a legal XI is reachable, completion classes Wi Wo Bi Bo with additions > 0 in the planner cheapest legal completion (legalCompletionCost), candidates = that class supply with base <= purse, rivalsNeeding from keeper (W) or bowling (B)',
        contestants: 'rivals with playerCount < maxPlayers AND purseLeft >= cheapest base (overseas classes: AND overseasCount < maxOverseas)',
        state: 'spare = candidates after this lot - need; T = contestants > 0 ? rivalsNeeding + criticalBuffer * need : 0; IMPOSSIBLE if planner status IMPOSSIBLE (requirements only); CRITICAL if spare < T; WARNING if spare < T + warningBuffer; else SAFE',
        margin: 'marginIncrements * bidIncrement(cheapest base over non-IMPOSSIBLE tracked needs) * min(11, largest need, each need reduced by 1 when buying this lot fills it); 0 if no tracked need has a contestant',
        fragile: 'purse - reserveIfPassed <= margin computed without buying',
        force: 'PASS masked if the act-v2 PASS condition holds, or planner allows AND (unlocks OR XI gain > 0.05) AND (the lot fills a CRITICAL tracked need, OR fragile AND it fills a non-IMPOSSIBLE tracked need; players counts for fragility only when no other tracked need exists)',
        floor: 'on a forced lot some rival can bid on at base: final path (act-v2 PASS condition, or a forcing need with spare < 0) keeps only the highest legal cap; otherwise caps < base + bidIncrement(base) are masked',
        bound: 'bid caps > maxSafeBid - margin (buying) are masked; on a forced lot with no bid inside floor and bound the bound is dropped, and if no bid reaches the floor only the highest legal bid stays legal',
        inputs: 'planner plan (set-based supply, never lot order), current lot, phase, rivals public purse, squad size, overseas count and composition'
    }
})
export const ACT_SPEC_HASH = specHash(ACT_SPEC)

// The action's price level before clamping.
export const actionLevel = (action, { basePrice, fairValue, maxSafeBid }) => {
    const a = ACTIONS[action]
    if (!a) throw new Error(`Unknown action ${action}`)
    if (a.kind === 'pass') return 0
    if (a.kind === 'base') return basePrice
    if (a.kind === 'maxSafe') return maxSafeBid
    return a.multiplier * fairValue
}

// Effective cap: floor(min(level, maxSafeBid)). PASS → 0.
export const capForAction = (action, lotFacts) =>
    action === PASS ? 0 : Math.floor(Math.min(actionLevel(action, lotFacts), lotFacts.maxSafeBid))

// Highest price on the lot's real increment ladder (base, then +1 increment
// at a time — rules.nextBidAmount) that is ≤ cap; null if cap < base.
// Closed form per increment band, so a ₹50,000L purse costs no more than ₹5,000L.
export const ladderFloor = (basePrice, cap) => {
    if (!(cap >= basePrice)) return null
    let p = basePrice
    for (;;) {
        const inc = bidIncrement(p)
        const bound = p < 200 ? 200 : p < 300 ? 300 : Infinity
        if (cap < bound) return p + inc * Math.floor((cap - p) / inc)
        const k = Math.ceil((bound - p) / inc) // steps until the band changes
        const next = p + inc * k
        if (next > cap) return p + inc * (k - 1)
        p = next
    }
}
