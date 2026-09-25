// obs-v2 — THE canonical RL observation (Phase 2A, frozen). 80 features in
// five blocks: global 6 · player 9 · own team 35 · market 16 · rivals 14.
//
// Every feature is built only from information a real franchise has when
// the lot opens: the player card, public results so far, every team's
// public squad and purse, and WHICH players are still to come (the
// catalogue minus those auctioned). The ORDER of the players still to come
// is never used — it is shuffled live — so the observation is invariant to
// any permutation of the remaining players within a set and of the
// re-auction list (tested).
//
// Inputs
//   ctx    — the standard bot context (see observation.js): rules, lot, self,
//            rivals, upcoming, returning, phase, progress
//   extras — { poolSize,  // players in the main pool (323 for the Full Pool)
//              recent }   // public results of the last RECENT_WINDOW lots:
//                         // [{ sold, price, fairValue, winnerTeamId }]
//   plan   — planBid(ctx), optional (computed if omitted)
//
// Normalisation is fixed and analytic (no running statistics), identical in
// Node and Python, and every value is clipped to CLIP.

import { XI_SIZE, bidBlocker } from '../rules.js'
import { XI_RULES, selectBestXI, xiGain, xiTotal } from '../scoring.js'
import { fairValue, slotBudget } from '../valuation.js'
import { classCounts, legalCompletionCost, planBid, playerClass, supplyOf, teamComposition } from '../planning.js'
import { overseasSlotContested } from '../botSignals.js'
import { specHash } from './hash.js'

export const OBS_VERSION = 'obs-v2'
export const CLIP = Object.freeze([-1, 5])
export const RECENT_WINDOW = 20
const STAR_RATING = 90
const REFERENCE_PURSE = 12500
const SET_COUNT = 10 // normalisation constant for the set number

const CLASSES = ['Wi', 'Wo', 'Bi', 'Bo', 'Oi', 'Oo']
const REQUIREMENTS = ['keeper', 'bowling', 'indians', 'players']
const STATUS_ORDINAL = { SAFE: 0, NEED: 1 / 3, CRITICAL: 2 / 3, IMPOSSIBLE: 1 }
const NEED_SCALE = { keeper: XI_RULES.minKeepers, bowling: XI_RULES.minBowlingOptions, indians: XI_SIZE, players: XI_SIZE }
// Which player classes can fill each XI requirement (as in planning.js).
const REQUIREMENT_CLASSES = { keeper: ['Wi', 'Wo'], bowling: ['Bi', 'Bo'], indians: ['Wi', 'Bi', 'Oi'], players: CLASSES }

const clip = (x, lo, hi) => Math.min(hi, Math.max(lo, x))
const mean = (xs) => (xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : 0)
const isOverseas = (p) => p.nationality === 'Overseas'

// ── Facts shared by several features (computed once per observation) ─────
const factsFor = (ctx, extras, plan) => {
    const { rules, lot, self } = ctx
    const rivals = ctx.rivals || []
    const upcoming = ctx.upcoming || []
    const returning = ctx.returning || []
    const P = rules.pursePerTeam
    const fv = fairValue(lot, P)
    const xi = selectBestXI(self.squad)
    const supplyPlayers = [...upcoming, ...returning]
    const supply = supplyOf(ctx)
    const premium = (p) => Math.max(0, fairValue(p, P) - p.basePrice)

    // Premium value already auctioned (bought by anyone, or unsold) vs still to come.
    const passed = [self, ...rivals].reduce((s, t) => s + t.squad.reduce((a, p) => a + premium(p), 0), 0) +
        returning.reduce((s, p) => s + premium(p), 0)
    const left = upcoming.reduce((s, p) => s + premium(p), 0) + premium(lot)
    const premiumPassed = passed + left > 0 ? passed / (passed + left) : 0

    const floor = xi.emptySlots > 0 || xi.players.length === 0 ? 0 : Math.min(...xi.players.map((p) => p.rating))
    const above = supplyPlayers.filter((p) => p.rating > floor).map((p) => p.rating)
    const typicalUpgrade = xi.emptySlots === 0 && above.length ? (mean(above) - floor) / XI_SIZE : 0

    const lotClass = playerClass(lot)
    const rivalView = rivals.map((r) => {
        const blocked = bidBlocker({ team: r, lot: { ...lot, currentBidderId: '' }, rules, amount: lot.basePrice }) !== null
        const comp = teamComposition(r, rules)
        const cost = legalCompletionCost(comp.counts, { slots: comp.slotsLeft, overseasSlots: comp.overseasSlots, supply }).cost
        const reserve = Number.isFinite(cost) ? cost : 0
        return {
            team: r,
            blocked,
            capacity: blocked ? 0 : Math.max(0, r.purseLeft - reserve) / fv,
            fills: REQUIREMENTS.some((req) => comp.missing[req] > 0 && REQUIREMENT_CLASSES[req].includes(lotClass)),
            gainShare: blocked ? 0 : clip(xiGain(r.squad, lot) / (lot.rating / XI_SIZE), 0, 1)
        }
    })

    const recent = (extras.recent || []).slice(-RECENT_WINDOW)
    const sold = recent.filter((h) => h.sold)
    const rivalIds = new Set(rivals.map((r) => r.teamId))
    const rivalSpend = sold.filter((h) => rivalIds.has(h.winnerTeamId)).reduce((s, h) => s + h.price, 0)

    return {
        rules, lot, self, rivals, upcoming, returning, supplyPlayers, plan, P, fv, xi, floor, typicalUpgrade,
        upgradesLeft: above.length, premiumPassed, rivalView, recent, sold, rivalSpend,
        slot: slotBudget(rules), poolSize: extras.poolSize,
        counts: classCounts(self.squad), supplyCounts: CLASSES.map((k) => supply.prices[k].length)
    }
}

// ── The 80 features, in order: [name, block, normalisation, fn(facts)] ─────
const FEATURES = [
    // GLOBAL (6)
    ['phase_reauction', 'global', '1 if re-auction else 0', (f) => (f.plan.lotContext.phase === 'reauction' ? 1 : 0)],
    ['main_progress', 'global', 'lots auctioned in main round / main pool size; 1 in re-auction', (f) => clip(f.ctx.progress ?? 0, 0, 1)],
    ['lots_left_in_phase', 'global', 'players still to come this phase / pool size', (f) => f.upcoming.length / f.poolSize],
    ['returning_count', 'global', 'unsold so far (back in re-auction) / pool size', (f) => f.returning.length / f.poolSize],
    ['set_index', 'global', 'lot set number / 10', (f) => (f.lot.setNo ?? 0) / SET_COUNT],
    ['purse_scale', 'global', 'log2(purse / 12500) / 2', (f) => Math.log2(f.P / REFERENCE_PURSE) / 2],

    // PLAYER (9)
    ['lot_is_batsman', 'player', 'one-hot', (f) => (f.lot.role === 'BATSMAN' ? 1 : 0)],
    ['lot_is_bowler', 'player', 'one-hot', (f) => (f.lot.role === 'BOWLER' ? 1 : 0)],
    ['lot_is_all_rounder', 'player', 'one-hot', (f) => (f.lot.role === 'ALL ROUNDER' ? 1 : 0)],
    ['lot_is_keeper', 'player', 'one-hot', (f) => (f.lot.role === 'WICKET KEEPER' ? 1 : 0)],
    ['lot_is_overseas', 'player', '1 if overseas', (f) => (isOverseas(f.lot) ? 1 : 0)],
    ['lot_rating', 'player', '(rating - 60) / 40', (f) => (f.lot.rating - 60) / 40],
    ['lot_is_star', 'player', '1 if rating >= 90', (f) => (f.lot.rating >= STAR_RATING ? 1 : 0)],
    ['lot_base_price', 'player', 'base price / slot budget (purse / maxPlayers)', (f) => f.lot.basePrice / f.slot],
    ['lot_fair_value', 'player', 'fair value / slot budget, max 4', (f) => Math.min(4, f.fv / f.slot)],

    // OWN TEAM (35)
    ['self_purse', 'self', 'purse left / purse', (f) => f.self.purseLeft / f.P],
    ['self_slots_left', 'self', 'squad slots left / maxPlayers', (f) => (f.rules.maxPlayers - f.self.playerCount) / f.rules.maxPlayers],
    ['self_overseas_slots_left', 'self', 'overseas slots left / maxOverseas', (f) => (f.rules.maxOverseas - f.self.overseasCount) / f.rules.maxOverseas],
    ...CLASSES.map((k) => [`self_class_${k}`, 'self', `count of class ${k} / 11`, (f) => f.counts[k] / XI_SIZE]),
    ['self_xi_strength', 'self', 'unrounded best-XI total / 11 / 100', (f) => f.xi.total / XI_SIZE / 100],
    ['self_xi_empty', 'self', 'empty XI slots / 11', (f) => f.xi.emptySlots / XI_SIZE],
    ['self_xi_floor', 'self', '(weakest XI rating - 60) / 40, 0 while the XI has empty slots', (f) => (f.floor ? (f.floor - 60) / 40 : 0)],
    ['self_xi_overseas', 'self', 'overseas in best XI / 4', (f) => f.xi.players.filter(isOverseas).length / XI_RULES.maxOverseas],
    ...REQUIREMENTS.map((r) => [`self_status_${r}`, 'self', 'SAFE 0, NEED 1/3, CRITICAL 2/3, IMPOSSIBLE 1', (f) => STATUS_ORDINAL[f.plan.teamNeeds[r]]]),
    ...REQUIREMENTS.map((r) => [`self_need_${r}`, 'self', `players still needed / ${NEED_SCALE[r]}`, (f) => f.plan.requirements[r].need / NEED_SCALE[r]]),
    ['self_reserve_if_passed', 'self', 'completion reserve if passed / purse; 1 if no legal XI is reachable', (f) => (f.plan.budget.reserveIfPassed ?? f.P) / f.P],
    ['self_max_safe_purse', 'self', 'maxSafeBid / purse', (f) => f.plan.budget.maxSafeBid / f.P],
    ['self_max_safe_fv', 'self', 'maxSafeBid / fair value', (f) => f.plan.budget.maxSafeBid / f.fv],
    ['self_xi_gain_share', 'self', 'XI gain / (rating / 11), in [0, 1]', (f) => clip(f.plan.playerImpact.xiGain / (f.lot.rating / XI_SIZE), 0, 1)],
    ['self_xi_gain', 'self', 'XI gain / 10', (f) => f.plan.playerImpact.xiGain / 10],
    ...['keeper', 'bowling', 'indians'].map((r) => [`self_fills_${r}`, 'self', '1 if he fills this unmet requirement', (f) => (f.plan.playerImpact.fillsRequirement.includes(r) ? 1 : 0)]),
    ['self_unlocks', 'self', '1 if buying him reaches a better XI than passing', (f) => (f.plan.playerImpact.unlocksRequirement ? 1 : 0)],
    ['self_final_opportunity', 'self', '1 if the planner flags a final opportunity', (f) => (f.plan.lotContext.finalOpportunity ? 1 : 0)],
    ['self_pace_gap', 'self', 'share of purse spent - share of premium value already auctioned', (f) => (1 - f.self.purseLeft / f.P) - f.premiumPassed],
    ['self_upgrades_left', 'self', 'players still to come rated above the XI floor / 50', (f) => f.upgradesLeft / 50],
    ['self_typical_upgrade', 'self', 'mean XI gain of those upgrades (XI points) / 2; 0 while the XI has empty slots', (f) => f.typicalUpgrade / 2],
    ['self_bench', 'self', '(squad size - XI players) / (maxPlayers - 11)', (f) => (f.self.playerCount - f.xi.players.length) / Math.max(1, f.rules.maxPlayers - XI_SIZE)],

    // MARKET (16)
    ['mkt_equivalent_left', 'market', 'same role within 3 rating points still to come / 10', (f) => f.plan.lotContext.equivalentRemaining / 10],
    ['mkt_better_left', 'market', 'same role, higher rating still to come / 10', (f) => f.plan.lotContext.betterRemaining / 10],
    ['mkt_same_role_left', 'market', 'same role still to come / 40', (f) => f.supplyPlayers.filter((p) => p.role === f.lot.role).length / 40],
    ...CLASSES.map((k, i) => [`mkt_supply_${k}`, 'market', `class ${k} still to come / 40`, (f) => f.supplyCounts[i] / 40]),
    ...['keeper', 'bowling', 'indians'].map((r) => [`mkt_scarcity_${r}`, 'market', 'suitable players left after this lot / (own need + rivals needing + 1)', (f) => {
        const q = f.plan.requirements[r]
        return q.remainingAfterLot / (q.need + q.rivalsNeeding + 1)
    }]),
    ['mkt_overseas_contested', 'market', '1 if an overseas slot is contested (botSignals.overseasSlotContested)', (f) => (overseasSlotContested(f.ctx) ? 1 : 0)],
    ['mkt_premium_passed', 'market', 'share of premium value (fair value above base) already auctioned', (f) => f.premiumPassed],
    ['mkt_recent_price_ratio', 'market', 'mean price / fair value of sales in the last 20 lots; 1 if none', (f) => (f.sold.length ? mean(f.sold.map((h) => h.price / h.fairValue)) : 1)],
    ['mkt_recent_sold_share', 'market', 'share of the last 20 lots that sold; 1 if none', (f) => (f.recent.length ? f.sold.length / f.recent.length : 1)],

    // RIVALS (14)
    ['riv_purse_max', 'rivals', 'max rival purse / purse', (f) => (f.rivals.length ? Math.max(...f.rivals.map((r) => r.purseLeft)) / f.P : 0)],
    ['riv_purse_mean', 'rivals', 'mean rival purse / purse', (f) => mean(f.rivals.map((r) => r.purseLeft / f.P))],
    ['riv_purse_min', 'rivals', 'min rival purse / purse', (f) => (f.rivals.length ? Math.min(...f.rivals.map((r) => r.purseLeft)) / f.P : 0)],
    ['riv_purse_std', 'rivals', 'std of rival purse / purse', (f) => {
        const xs = f.rivals.map((r) => r.purseLeft / f.P)
        const m = mean(xs)
        return Math.sqrt(mean(xs.map((x) => (x - m) ** 2)))
    }],
    ...[0, 1, 2].map((i) => [`riv_capacity_${i + 1}`, 'rivals', `${i + 1}${['st', 'nd', 'rd'][i]}-highest (purse - completion cost) / fair value among rivals able to bid`, (f) => {
        const caps = f.rivalView.filter((v) => !v.blocked).map((v) => v.capacity).sort((a, b) => b - a)
        return caps[i] ?? 0
    }]),
    ['riv_able_share', 'rivals', 'share of rivals able to pay fair value after their completion cost', (f) => (f.rivals.length ? f.rivalView.filter((v) => !v.blocked && v.capacity >= 1).length / f.rivals.length : 0)],
    ['riv_fills_share', 'rivals', 'share of rivals for whom he fills an unmet requirement', (f) => (f.rivals.length ? f.rivalView.filter((v) => v.fills).length / f.rivals.length : 0)],
    ['riv_gain_mean', 'rivals', 'mean rival XI gain / (rating / 11); 0 for rivals who cannot bid', (f) => mean(f.rivalView.map((v) => v.gainShare))],
    ['riv_free_slots_share', 'rivals', 'share of rivals with squad slots left', (f) => (f.rivals.length ? f.rivals.filter((r) => r.playerCount < f.rules.maxPlayers).length / f.rivals.length : 0)],
    ['riv_recent_spend', 'rivals', 'rival spend over the last lots / (rivals x purse) x (pool size / lots in window)', (f) => (f.rivals.length && f.recent.length ? (f.rivalSpend / (f.rivals.length * f.P)) * (f.poolSize / f.recent.length) : 0)],
    ['riv_xi_mean', 'rivals', 'mean rival best-XI total / 11 / 100', (f) => mean(f.rivals.map((r) => xiTotal(r.squad) / XI_SIZE / 100))],
    ['riv_xi_max', 'rivals', 'max rival best-XI total / 11 / 100', (f) => (f.rivals.length ? Math.max(...f.rivals.map((r) => xiTotal(r.squad))) / XI_SIZE / 100 : 0)]
]

export const OBS_FEATURES = Object.freeze(FEATURES.map(([name]) => name))
export const OBS_SIZE = OBS_FEATURES.length
export const OBS_BLOCKS = Object.freeze({ global: 6, player: 9, self: 35, market: 16, rivals: 14 })

// The canonical, hashed description of obs-v2.
export const OBS_SPEC = Object.freeze({
    version: OBS_VERSION,
    size: OBS_SIZE,
    clip: CLIP,
    recentWindow: RECENT_WINDOW,
    features: FEATURES.map(([name, block, norm]) => ({ name, block, norm }))
})
export const OBS_SPEC_HASH = specHash(OBS_SPEC)

export const buildRlObservation = (ctx, extras, plan = planBid(ctx)) => {
    if (!extras || !(extras.poolSize > 0)) throw new Error('buildRlObservation: extras.poolSize is required')
    const f = factsFor(ctx, extras, plan)
    f.ctx = ctx
    const obs = new Array(OBS_SIZE)
    for (let i = 0; i < OBS_SIZE; i++) {
        const v = FEATURES[i][3](f)
        if (!Number.isFinite(v)) throw new Error(`obs-v2 feature ${OBS_FEATURES[i]} is not finite (${v})`)
        obs[i] = clip(v, CLIP[0], CLIP[1])
    }
    return obs
}
