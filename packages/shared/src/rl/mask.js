// THE canonical RL action mask + completion shield (Phase 2A, frozen). Used
// by the training environment, the evaluator and production inference —
// there is no other implementation.
//
//   rules block the lot (squad full, overseas cap, purse < base)  → PASS only
//   planner disallows (plan.allowed false: base > maxSafeBid, or
//                      buying would break the reachable XI)       → PASS only
//   otherwise action a ≥ 1 is legal iff
//     cap_a = floor(min(level_a, maxSafeBid)) ≥ base, and
//     its effective price on the real increment ladder is above that of
//     every lower legal action (no two legal actions ever buy at the same price)
//   completion shield — PASS masked when
//     buying unlocks a better reachable XI than passing, or
//     he is a final opportunity for a requirement he fills AND XI gain > 0.05
//
// BASE is always legal when the planner allows the lot (maxSafeBid ≥ base),
// so a shielded step always has at least one legal action.

import { bidBlocker } from '../rules.js'
import { XI_GAIN, planBid } from '../planning.js'
import { fairValue } from '../valuation.js'
import { ACTION_COUNT, PASS, capForAction, ladderFloor } from './actionSpec.js'

export const shieldActiveFor = (plan) => {
    if (!plan.allowed) return false
    if (plan.playerImpact.unlocksRequirement) return true
    return plan.playerImpact.xiGain > XI_GAIN.none &&
        plan.playerImpact.fillsRequirement.some((r) => plan.requirements[r].finalOpportunity)
}

export const rlActionMask = (ctx, plan = planBid(ctx)) => {
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

export const hasBidAction = (mask) => mask.some((m, a) => a !== PASS && m === 1)

// Server-side re-validation of a final decision (action + cap) against the
// canonical mask. Returns null when valid, else the reason.
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
