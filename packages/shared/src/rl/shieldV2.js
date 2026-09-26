// CANDIDATE completion shield v2 (Phase 2C.2) — NOT ADOPTED. Opt-in only.
//
// The frozen Phase 2A shield (mask.js, described in ACT_SPEC and therefore
// inside ACT_SPEC_HASH) stays the default everywhere. This module exists so
// the proposed redesign can be replayed, tested and measured before anyone
// decides to change the frozen specification.
//
// ── Why the frozen shield is insufficient ─────────────────────────────────
// It only masks PASS on a requirement's FINAL opportunity. A bid is not a
// win: at that moment the learner may hold exactly the base price (the
// planner's maxSafeBid reserves base-price completion only), so any rival
// bidding the base price can take the lot on the tie-break and the learner
// cannot raise. Every earlier candidate could be passed freely.
//
// ── What v2 adds (only ever REMOVES actions from the frozen mask) ─────────
// Per unmet XI requirement r (planner terms: need, remainingAfterLot,
// rivalsNeeding, cheapestBase), with contestants_r = rivals who could bid
// on such a player (public: not squad-full, purse ≥ cheapest base):
//
//   spare_r   = remainingAfterLot_r − need_r   (chances left if this lot is passed)
//   critical  = contestants_r > 0 ? rivalsNeeding_r + CRITICAL_BUFFER × need_r : 0
//               (needing 3 of the last 4 is a boundary; needing 1 of 4 is not)
//   state_r   = IMPOSSIBLE if the planner says so (already infeasible — the
//               shield cannot repair it and does not pretend to)
//               CRITICAL   if spare_r < critical
//               WARNING    if spare_r < critical + WARNING_BUFFER
//               SAFE       otherwise
//
//   1. FORCE  (PASS masked) — the lot fills a CRITICAL requirement, the
//      planner allows it and it helps the XI (gain > 0.05 or unlocks); the
//      frozen shield's conditions are kept. So a requirement is no longer
//      left to a single last attempt: the last CRITICAL_BUFFER + rival-need
//      + 1 candidates must be contested. Also forced when the purse is
//      FRAGILE: passing would leave no more than the contest margin above
//      the base-price completion reserve, so the learner could not out-bid
//      anyone later — then any lot filling an unmet requirement is forced.
//   2. FLOOR  — on a forced lot that someone else can contest, bids below
//      one increment over base are masked when a higher legal bid exists,
//      so a forced bid is not a coin-flip at the base price. On a FINAL path
//      (the frozen shield's conditions, or no spare candidate left) only the
//      highest safe bid (maxSafeBid) stays legal.
//   3. MARGIN — every bid keeps a contest reserve: one bid increment (on the
//      cheapest relevant base price) per XI player still needed after this
//      lot, on top of the planner's base-price reserve. Bids whose cap would
//      eat into it are masked (PASS stays legal). This is what stops a
//      policy from spending down to exactly the base-price floor. On a
//      forced lot the margin gives way before the forced bid does.
//
// Information used: the planner's plan (set-based supply counts — never the
// order of upcoming lots), the current lot, the phase, and rivals' public
// purse / squad size / overseas count / composition. The same inputs the
// frozen mask and obs-v2 already use.

import { XI_SIZE, bidBlocker, bidIncrement } from '../rules.js'
import { REQUIREMENTS, REQUIREMENT_STATUS, XI_GAIN, legalCompletionCost, planBid, playerClass, supplyOf, teamComposition } from '../planning.js'
import { PASS } from './actionSpec.js'
import { rlActionMask } from './mask.js'

export const SHIELD_VERSION = 'shield-v2-candidate'
export const SHIELD_STATE = Object.freeze({ SAFE: 'SAFE', WARNING: 'WARNING', CRITICAL: 'CRITICAL', IMPOSSIBLE: 'IMPOSSIBLE' })
const RANK = { SAFE: 0, WARNING: 1, CRITICAL: 2, IMPOSSIBLE: 3 }
export const SHIELD_V2_PARAMS = Object.freeze({ criticalBuffer: 1, warningBuffer: 3, marginIncrements: 1 })

// Rivals who could bid on a player costing `base` (public state only).
const contestantsAt = (ctx, base) => {
    if (base === null || base === undefined) return 0
    const { rules } = ctx
    return (ctx.rivals || []).filter((t) => t.playerCount < rules.maxPlayers && t.purseLeft >= base).length
}

// Requirement-by-requirement feasibility analysis (diagnostic + decision).
export const analyseCompletion = (ctx, plan = planBid(ctx), params = SHIELD_V2_PARAMS) => {
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
    // Joint completion classes. Requirements overlap — one Indian bowler can
    // be the bowling option, the Indian AND the XI place a squad still lacks —
    // so per-requirement counts can look comfortable while the one class
    // that satisfies all of them runs out. The planner's own exact
    // cheapest-completion search says which keeper / bowling classes a legal
    // XI needs (Wi Wo Bi Bo; batsmen are plain fillers, covered above); each
    // is tracked like a requirement: candidates of that class still to come
    // (set counts, affordable at base), same thresholds. Never IMPOSSIBLE —
    // another class may substitute; the planner decides true infeasibility.
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

// Contest reserve: one increment per XI player still needed (after this lot
// if it is bought), priced on the cheapest relevant base.
const contestMargin = (ctx, plan, analysis, params, buying) => {
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

// Same shape as rlActionMask plus `shield` diagnostics.
export const rlActionMaskV2 = (ctx, plan = planBid(ctx), params = SHIELD_V2_PARAMS) => {
    const v1 = rlActionMask(ctx, plan)
    const analysis = analyseCompletion(ctx, plan, params)
    const mask = [...v1.mask]
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
        v1Mask: v1.mask, changed: false, removed: [], reasons
    }
    const bids = () => mask.flatMap((m, a) => (m && a !== PASS ? [a] : []))
    if (v1.reason || !bids().length) return { ...v1, shield: diag }

    // 1. FORCE
    const helps = plan.playerImpact.unlocksRequirement || plan.playerImpact.xiGain > XI_GAIN.none
    const forcedBy = Object.entries(analysis.requirements)
        .filter(([, a]) => a.fillsNow && a.state === SHIELD_STATE.CRITICAL).map(([r]) => r)
    // Purse fragility: if passing would leave no more money above the
    // base-price completion reserve than the contest margin, the learner can
    // no longer out-bid anyone later — every lot that fills an unmet
    // requirement becomes a boundary, however many candidates remain (their
    // base prices can rise as the cheap ones go). 'players' counts only when
    // no specific requirement (keeper / bowling / Indians) is missing.
    const marginIfPass = contestMargin(ctx, plan, analysis, params, false)
    const reserveIfPassed = plan.budget.reserveIfPassed
    const slackIfPass = reserveIfPassed === null ? -Infinity : ctx.self.purseLeft - reserveIfPassed
    const fragile = marginIfPass > 0 && slackIfPass <= marginIfPass
    const specificMissing = Object.keys(analysis.requirements).some((r) => r !== 'players')
    const fragileBy = !fragile ? [] : Object.entries(analysis.requirements)
        .filter(([r, a]) => a.fillsNow && a.state !== SHIELD_STATE.IMPOSSIBLE && !forcedBy.includes(r) && (r !== 'players' || !specificMissing))
        .map(([r]) => r)
    Object.assign(diag, { fragile, slackIfPass: Number.isFinite(slackIfPass) ? slackIfPass : null, marginIfPass })
    const forced = v1.shieldActive || (plan.allowed && helps && (forcedBy.length > 0 || fragileBy.length > 0))
    if (forced) {
        mask[PASS] = 0
        diag.forced = true
        diag.forcedBy = [...(v1.shieldActive ? ['frozen-shield'] : []), ...forcedBy, ...fragileBy.map((r) => `${r}(purse)`)]
        reasons.push(`forced: ${diag.forcedBy.join(', ')}`)
    }

    // 3. MARGIN (bound on the cap)
    const margin = contestMargin(ctx, plan, analysis, params, true)
    diag.margin = margin
    const bound = plan.budget.maxSafeBid - margin
    diag.bound = bound
    // 2. FLOOR (forced lots someone can contest). On a FINAL path — the frozen
    // shield's own conditions (last candidate / the purchase that unlocks the
    // only affordable completion) or a critical requirement with no spare
    // candidate — only the highest safe bid is legal: losing it makes the XI
    // impossible, so the learner must contest it with everything it can
    // safely spend (maxSafeBid), exactly as the rule bots do.
    const contestable = (ctx.rivals || []).some((t) => !bidBlocker({ team: t, lot: { ...lot, currentBidderId: '' }, rules: ctx.rules, amount: lot.basePrice }))
    const legalBids = bids()
    const finalPath = forced && (v1.shieldActive || forcedBy.some((r) => analysis.requirements[r].spare < 0))
    diag.finalPath = finalPath
    const topCap = Math.max(...legalBids.map((a) => v1.caps[a]))
    const floor = !forced || !contestable ? null : finalPath ? topCap : lot.basePrice + bidIncrement(lot.basePrice)
    diag.floor = floor
    if (finalPath && contestable) reasons.push('final path: highest safe bid only')

    const within = legalBids.filter((a) => v1.caps[a] <= bound && (floor === null || v1.caps[a] >= floor))
    let keep
    if (within.length) keep = within
    else if (forced) {
        // The forced bid wins over the margin; the floor where affordable.
        const aboveFloor = legalBids.filter((a) => floor === null || v1.caps[a] >= floor)
        keep = aboveFloor.length ? aboveFloor : [legalBids.reduce((b, a) => (v1.caps[a] > v1.caps[b] ? a : b))]
        reasons.push(aboveFloor.length ? 'margin yields to forced bid' : 'floor unaffordable: highest legal bid only')
    } else keep = []
    for (const a of legalBids) {
        if (!keep.includes(a)) {
            mask[a] = 0
            diag.removed.push(a)
        }
    }
    if (diag.removed.length) {
        const byMargin = diag.removed.filter((a) => v1.caps[a] > bound).length
        const byFloor = diag.removed.length - byMargin
        if (byMargin) reasons.push(`margin ₹${margin}L: ${byMargin} bid(s) above ₹${bound}L masked`)
        if (byFloor) reasons.push(`floor ₹${floor}L: ${byFloor} low bid(s) masked`)
    }
    diag.changed = mask.some((m, a) => m !== v1.mask[a])
    return { ...v1, mask, shieldActive: mask[PASS] === 0, shield: diag }
}
