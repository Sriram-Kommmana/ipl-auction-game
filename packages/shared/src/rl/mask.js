// THE canonical RL action mask + completion shield — act-v3 (adopted in
// Phase 2C.2; replaces the Phase 2A act-v2 shield). Used by the training
// environment, the evaluator and production inference — there is no other
// implementation. The exact semantics are also written into ACT_SPEC
// (actionSpec.js) and therefore into ACT_SPEC_HASH.
//
// ── act-v2 legal set (unchanged, rlActionMaskActV2) ───────────────────────
//   rules block the lot (squad full, overseas cap, purse < base)  → PASS only
//   planner disallows (plan.allowed false: base > maxSafeBid, or
//                      buying would break the reachable XI)       → PASS only
//   otherwise action a ≥ 1 is legal iff
//     cap_a = floor(min(level_a, maxSafeBid)) ≥ base, and
//     its effective price on the real increment ladder is above that of
//     every lower legal action (no two legal actions ever buy at the same price)
//   act-v2 shield — PASS masked when buying unlocks a better reachable XI than
//     passing, or he is a final opportunity for a requirement he fills AND
//     XI gain > 0.05
// The act-v2 mask stays available verbatim for the frozen baseline
// controllers (their decision timing is part of the locked baseline
// reference) and nothing else.
//
// ── act-v3 completion shield (rlActionMask) ───────────────────────────────
// Why: a bid is not a win. The act-v2 shield only forces the FINAL candidate,
// when the learner may hold exactly the base price (maxSafeBid reserves
// base-price completion only) — any rival bidding base can take it on the
// tie-break. Starting from the act-v2 mask, act-v3 only ever REMOVES actions:
//
// Per unmet XI requirement r (planner: need, remainingAfterLot, rivalsNeeding,
// cheapestBase, status) and per completion class (Wi Wo Bi Bo with additions
// > 0 in the planner's cheapest legal completion, while a legal XI is
// reachable; candidates = that class's supply affordable at base):
//   contestants = rivals with playerCount < maxPlayers and purse ≥ cheapest
//                 base (overseas classes: also overseasCount < maxOverseas)
//   spare = candidates after this lot − need
//   T     = contestants > 0 ? rivalsNeeding + CRITICAL_BUFFER × need : 0
//   state = IMPOSSIBLE (planner says so; requirements only — recorded as
//           already infeasible, never "repaired") | CRITICAL if spare < T |
//           WARNING if spare < T + WARNING_BUFFER | SAFE
//   margin(buying) = MARGIN_INCREMENTS × bidIncrement(cheapest base) ×
//           min(11, largest remaining need, each reduced by 1 when buying
//           this lot fills it); 0 if no requirement has a contestant
//   fragile = purse − reserveIfPassed ≤ margin(passing)
//
//   1. FORCE  PASS masked if the act-v2 shield fires, or the planner allows
//      the lot, it helps the XI (unlocks or gain > 0.05) and it fills a
//      CRITICAL requirement/class — or the purse is fragile and it fills any
//      unmet, not-IMPOSSIBLE requirement/class ('players' only when no
//      keeper / bowling / Indian / class need exists).
//   2. FLOOR  on a forced lot some rival can bid on: FINAL path (act-v2
//      shield fires, or a forcing requirement has spare < 0) → only the
//      highest legal cap (maxSafeBid); otherwise caps ≥ base + 1 increment.
//   3. MARGIN bid caps > maxSafeBid − margin(buying) are masked; PASS stays
//      legal. On a forced lot the margin yields to the forced bid; if no bid
//      reaches the floor, the highest legal bid alone stays legal.
//
// Inputs: the planner's plan (set-based supply counts — never the order of
// upcoming lots), the lot, the phase and rivals' public purse / squad size /
// overseas count / composition.

import { XI_SIZE, bidBlocker, bidIncrement } from '../rules.js'
import { REQUIREMENTS, REQUIREMENT_STATUS, XI_GAIN, legalCompletionCost, planBid, playerClass, supplyOf, teamComposition } from '../planning.js'
import { fairValue } from '../valuation.js'
import { ACTION_COUNT, PASS, capForAction, ladderFloor } from './actionSpec.js'

export const SHIELD_VERSION = 'completion-shield-v2'
export const SHIELD_PARAMS = Object.freeze({ criticalBuffer: 1, warningBuffer: 3, marginIncrements: 2 })
export const SHIELD_STATE = Object.freeze({ SAFE: 'SAFE', WARNING: 'WARNING', CRITICAL: 'CRITICAL', IMPOSSIBLE: 'IMPOSSIBLE' })
const RANK = { SAFE: 0, WARNING: 1, CRITICAL: 2, IMPOSSIBLE: 3 }

// ── act-v2 (frozen, verbatim) ─────────────────────────────────────────────
export const shieldActiveFor = (plan) => {
    if (!plan.allowed) return false
    if (plan.playerImpact.unlocksRequirement) return true
    return plan.playerImpact.xiGain > XI_GAIN.none &&
        plan.playerImpact.fillsRequirement.some((r) => plan.requirements[r].finalOpportunity)
}

export const rlActionMaskActV2 = (ctx, plan = planBid(ctx)) => {
    const { lot, self, rules } = ctx
    const mask = new Array(ACTION_COUNT).fill(0)
    const caps = new Array(ACTION_COUNT).fill(0)
    const prices = new Array(ACTION_COUNT).fill(null)
    mask[PASS] = 1

    const blocked = bidBlocker({ team: self, lot: { ...lot, currentBidderId: '' }, rules, amount: lot.basePrice })
    if (blocked) return { mask, caps, prices, shieldActive: false, reason: blocked, plan }
    if (!plan.allowed) return { mask, caps, prices, shieldActive: false, reason: plan.reason, plan }

    const lotFacts = { basePrice: lot.basePrice, fairValue: fairValue(lot, rules.pursePerTeam), maxSafeBid: plan.budget.maxSafeBid }
    let highest = -Infinity
    for (let a = 1; a < ACTION_COUNT; a++) {
        const cap = capForAction(a, lotFacts)
        caps[a] = cap
        const price = ladderFloor(lot.basePrice, cap)
        prices[a] = price
        if (price === null || price <= highest) continue
        mask[a] = 1
        highest = price
    }
    const shieldActive = shieldActiveFor(plan)
    if (shieldActive) mask[PASS] = 0
    return { mask, caps, prices, shieldActive, reason: null, plan }
}

// ── act-v3 completion shield ──────────────────────────────────────────────
const contestantsAt = (ctx, base) => {
    if (base === null || base === undefined) return 0
    return (ctx.rivals || []).filter((t) => t.playerCount < ctx.rules.maxPlayers && t.purseLeft >= base).length
}

// Requirement-by-requirement (and completion-class) feasibility analysis.
export const analyseCompletion = (ctx, plan = planBid(ctx), params = SHIELD_PARAMS) => {
    const out = {}
    let state = SHIELD_STATE.SAFE
    for (const r of REQUIREMENTS) {
        const q = plan.requirements[r]
        if (q.need === 0) continue
        const contestants = contestantsAt(ctx, q.cheapestBase)
        const spare = q.remainingAfterLot - q.need
        const critical = contestants > 0 ? q.rivalsNeeding + params.criticalBuffer * q.need : 0
        const s = q.status === REQUIREMENT_STATUS.IMPOSSIBLE ? SHIELD_STATE.IMPOSSIBLE
            : spare < critical ? SHIELD_STATE.CRITICAL
                : spare < critical + params.warningBuffer ? SHIELD_STATE.WARNING
                    : SHIELD_STATE.SAFE
        out[r] = {
            state: s, need: q.need, viableAfterLot: q.remainingAfterLot, spare, rivalsNeeding: q.rivalsNeeding,
            contestants, cheapestBase: q.cheapestBase, plannerStatus: q.status,
            fillsNow: plan.playerImpact.fillsRequirement.includes(r)
        }
        if (RANK[s] > RANK[state]) state = s
    }
    // Completion classes. Requirements overlap — one Indian bowler can be the
    // bowling option, the Indian AND the XI place a squad still lacks — so the
    // per-requirement counts can look comfortable while the one class that
    // satisfies all of them runs out. The planner's own exact cheapest legal
    // completion names the keeper / bowling classes a legal XI needs (batsmen
    // are plain fillers, covered above). Never IMPOSSIBLE: another class may
    // substitute — the planner decides true infeasibility.
    if (plan.completion.reachableEmptySlots === 0) {
        const comp = teamComposition(ctx.self, ctx.rules)
        const supply = supplyOf(ctx)
        const { additions } = legalCompletionCost(comp.counts, { slots: comp.slotsLeft, overseasSlots: comp.overseasSlots, supply })
        const purse = ctx.self.purseLeft
        for (const [k, n] of Object.entries(additions || {})) {
            if (!n || k.startsWith('O')) continue
            const prices = supply.prices[k].filter((p) => p <= purse)
            const overseas = k.endsWith('o')
            const contestants = prices.length === 0 ? 0 : (ctx.rivals || []).filter((t) =>
                t.playerCount < ctx.rules.maxPlayers && t.purseLeft >= prices[0] && (!overseas || t.overseasCount < ctx.rules.maxOverseas)).length
            const rivalsNeeding = plan.requirements[k.startsWith('W') ? 'keeper' : 'bowling'].rivalsNeeding
            const spare = prices.length - n
            const critical = contestants > 0 ? rivalsNeeding + params.criticalBuffer * n : 0
            const s = spare < critical ? SHIELD_STATE.CRITICAL : spare < critical + params.warningBuffer ? SHIELD_STATE.WARNING : SHIELD_STATE.SAFE
            out[`class:${k}`] = {
                state: s, need: n, viableAfterLot: prices.length, spare, rivalsNeeding, contestants,
                cheapestBase: prices[0] ?? null, plannerStatus: 'COMPLETION_CLASS', fillsNow: playerClass(ctx.lot) === k
            }
            if (RANK[s] > RANK[state]) state = s
        }
    }
    return { state, requirements: out }
}

// Contest reserve: MARGIN_INCREMENTS bid increments (on the cheapest relevant
// base) per XI player still needed (after this lot, if `buying`).
const contestMargin = (analysis, params, buying) => {
    let needed = 0
    let cheapest = Infinity
    let contested = false
    for (const a of Object.values(analysis.requirements)) {
        if (a.state === SHIELD_STATE.IMPOSSIBLE) continue
        const n = Math.max(0, a.need - (buying && a.fillsNow ? 1 : 0))
        if (n > needed) needed = n
        if (a.cheapestBase !== null && a.cheapestBase < cheapest) cheapest = a.cheapestBase
        if (a.contestants > 0) contested = true
    }
    if (!contested || needed === 0 || !Number.isFinite(cheapest)) return 0
    return Math.min(XI_SIZE, needed) * bidIncrement(cheapest) * params.marginIncrements
}

// The canonical act-v3 mask: { mask, caps, prices, shieldActive, reason, plan,
// shield } — `shield` holds the decision's diagnostics (audit/logging only;
// never an observation). `params` exists for research runs only; the
// specification, training and production always use SHIELD_PARAMS.
export const rlActionMask = (ctx, plan = planBid(ctx), params = SHIELD_PARAMS) => {
    const v2 = rlActionMaskActV2(ctx, plan)
    const analysis = analyseCompletion(ctx, plan, params)
    const mask = [...v2.mask]
    const { lot } = ctx
    const reasons = []
    const diag = {
        version: SHIELD_VERSION,
        state: analysis.state,
        requirements: analysis.requirements,
        phase: ctx.phase,
        lot: lot.slNo,
        purse: ctx.self.purseLeft,
        basePrice: lot.basePrice,
        maxSafeBid: plan.budget.maxSafeBid,
        feasibleBefore: plan.completion.reachableEmptySlots === 0,
        alreadyInfeasible: plan.completion.reachableEmptySlots > 0 || analysis.state === SHIELD_STATE.IMPOSSIBLE,
        forced: false, forcedBy: [], floor: null, bound: null, margin: 0,
        actV2Mask: v2.mask, changed: false, removed: [], reasons
    }
    const bids = () => mask.flatMap((m, a) => (m && a !== PASS ? [a] : []))
    if (v2.reason || !bids().length) return { ...v2, shield: diag }

    // 1. FORCE
    const helps = plan.playerImpact.unlocksRequirement || plan.playerImpact.xiGain > XI_GAIN.none
    const forcedBy = Object.entries(analysis.requirements)
        .filter(([, a]) => a.fillsNow && a.state === SHIELD_STATE.CRITICAL).map(([r]) => r)
    // Purse fragility: passing would leave no more money above the base-price
    // completion reserve than the contest margin — the learner could not
    // out-bid anyone later, and the cheap candidates can disappear meanwhile.
    const marginIfPass = contestMargin(analysis, params, false)
    const reserveIfPassed = plan.budget.reserveIfPassed
    const slackIfPass = reserveIfPassed === null ? -Infinity : ctx.self.purseLeft - reserveIfPassed
    const fragile = marginIfPass > 0 && slackIfPass <= marginIfPass
    const specificMissing = Object.keys(analysis.requirements).some((r) => r !== 'players')
    const fragileBy = !fragile ? [] : Object.entries(analysis.requirements)
        .filter(([r, a]) => a.fillsNow && a.state !== SHIELD_STATE.IMPOSSIBLE && !forcedBy.includes(r) && (r !== 'players' || !specificMissing))
        .map(([r]) => r)
    Object.assign(diag, { fragile, slackIfPass: Number.isFinite(slackIfPass) ? slackIfPass : null, marginIfPass })
    const forced = v2.shieldActive || (plan.allowed && helps && (forcedBy.length > 0 || fragileBy.length > 0))
    if (forced) {
        mask[PASS] = 0
        diag.forced = true
        diag.forcedBy = [...(v2.shieldActive ? ['act-v2-shield'] : []), ...forcedBy, ...fragileBy.map((r) => `${r}(purse)`)]
        reasons.push(`forced: ${diag.forcedBy.join(', ')}`)
    }

    // 3. MARGIN (upper bound on the cap)
    const margin = contestMargin(analysis, params, true)
    const bound = plan.budget.maxSafeBid - margin
    Object.assign(diag, { margin, bound })
    // 2. FLOOR (forced lots a rival can bid on)
    const contestable = (ctx.rivals || []).some((t) => !bidBlocker({ team: t, lot: { ...lot, currentBidderId: '' }, rules: ctx.rules, amount: lot.basePrice }))
    const legalBids = bids()
    const finalPath = forced && (v2.shieldActive || forcedBy.some((r) => analysis.requirements[r].spare < 0))
    const topCap = Math.max(...legalBids.map((a) => v2.caps[a]))
    const floor = !forced || !contestable ? null : finalPath ? topCap : lot.basePrice + bidIncrement(lot.basePrice)
    Object.assign(diag, { finalPath, floor })
    if (finalPath && contestable) reasons.push('final path: highest safe bid only')

    const within = legalBids.filter((a) => v2.caps[a] <= bound && (floor === null || v2.caps[a] >= floor))
    let keep
    if (within.length) keep = within
    else if (forced) {
        const aboveFloor = legalBids.filter((a) => floor === null || v2.caps[a] >= floor)
        keep = aboveFloor.length ? aboveFloor : [legalBids.reduce((b, a) => (v2.caps[a] > v2.caps[b] ? a : b))]
        reasons.push(aboveFloor.length ? 'margin yields to forced bid' : 'floor unaffordable: highest legal bid only')
    } else keep = []
    for (const a of legalBids) {
        if (!keep.includes(a)) {
            mask[a] = 0
            diag.removed.push(a)
        }
    }
    if (diag.removed.length) {
        const byMargin = diag.removed.filter((a) => v2.caps[a] > bound).length
        const byFloor = diag.removed.length - byMargin
        if (byMargin) reasons.push(`margin ₹${margin}L: ${byMargin} bid(s) above ₹${bound}L masked`)
        if (byFloor) reasons.push(`floor ₹${floor}L: ${byFloor} low bid(s) masked`)
    }
    diag.changed = mask.some((m, a) => m !== v2.mask[a])
    return { ...v2, mask, shieldActive: mask[PASS] === 0, shield: diag }
}

export const hasBidAction = (mask) => mask.some((m, a) => a !== PASS && m === 1)

// Server-side re-validation of a final decision (action + cap) against the
// canonical act-v3 mask. Returns null when valid, else the reason.
export const validateRlDecision = (ctx, action, cap, maskResult = rlActionMask(ctx)) => {
    if (!Number.isInteger(action) || action < 0 || action >= ACTION_COUNT) return `action ${action} out of range`
    if (!maskResult.mask[action]) return `action ${action} is masked`
    if (cap !== maskResult.caps[action]) return `cap ${cap} does not match action ${action} (${maskResult.caps[action]})`
    if (action === PASS) return null
    const { plan } = maskResult
    if (cap > plan.budget.maxSafeBid) return `cap ${cap} above maxSafeBid ${plan.budget.maxSafeBid}`
    if (cap > ctx.self.purseLeft) return `cap ${cap} above purse ${ctx.self.purseLeft}`
    if (cap < ctx.lot.basePrice) return `cap ${cap} below base ${ctx.lot.basePrice}`
    return null
}
