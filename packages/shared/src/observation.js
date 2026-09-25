// What a bot "sees" when a lot opens, and the actions it can take.
//
// ── Decision model ──────────────────────────────────────────────────────
// A bot decides ONCE per lot: the most it is willing to pay (its cap).
// The runtime then bids one increment at a time until the price passes the
// cap or someone else stops. Deciding a cap rather than bid/pass on every
// increment keeps an episode to ~one decision per player (easier credit
// assignment for RL), and lets the server know in advance when no bot will
// bid again — which is what makes solo auto-close possible.
//
// ── The context object (plain data, built by the server or simulator) ────
//   {
//     rules:    { pursePerTeam, maxPlayers, maxOverseas },
//     lot:      { slNo, role, nationality, rating, basePrice, stats },
//     self:     { teamId, purseLeft, playerCount, overseasCount, squad: [player] },
//     rivals:   [ same shape as self ],
//     upcoming: [ player ],   // still to be auctioned after this lot
//     progress: 0..1          // share of the main pool already auctioned
//   }
//   player = { slNo, role, nationality, rating, basePrice, stats }
//
// Observation design follows the budget-aware bidding work in
// cocoa-huang/rl-ad-bidding (money as a share of purse, pacing ratio) and
// the Hindustan Times piece on franchise planning (role needs, how many
// substitutes are still to come, what rivals still need).

import { DEFAULT_RULES, bidBlocker, botSpendLimit, cheapSlotPrice } from './rules.js'
import { fairValue, slotBudget } from './valuation.js'
import { teamStrength, xiGain } from './scoring.js'

// Action i caps the bid at CAP_MULTIPLIERS[i] × fair value. Action 0 = pass.
export const CAP_MULTIPLIERS = Object.freeze([0, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0])
export const ACTION_COUNT = CAP_MULTIPLIERS.length

// How many of each role a sensible squad wants (17 of 25 slots) — used for
// "role need" features and the Balanced Builder rule bot.
export const SQUAD_TARGETS = Object.freeze({
    BATSMAN: 5,
    BOWLER: 6,
    'ALL ROUNDER': 4,
    'WICKET KEEPER': 2
})

const ROLE_ORDER = ['BATSMAN', 'BOWLER', 'ALL ROUNDER', 'WICKET KEEPER']

// Persona vector — appended to the observation so ONE policy can play five
// different personalities. Also weights the persona part of the RL reward.
export const PERSONA_DIMS = Object.freeze(['aggression', 'bowlingFocus', 'battingFocus', 'overseasFocus', 'starFocus'])
export const ZERO_PERSONA = Object.freeze(PERSONA_DIMS.map(() => 0))

export const OBSERVATION_FEATURES = Object.freeze([
    // the player on the block
    'lot_is_batsman', 'lot_is_bowler', 'lot_is_all_rounder', 'lot_is_keeper',
    'lot_is_overseas',
    'lot_rating',               // (rating - 60) / 40
    'lot_base_price',           // ÷ slot budget
    'lot_fair_value',           // ÷ slot budget
    'lot_bat', 'lot_bowl', 'lot_power', 'lot_technique', 'lot_clutch', // stats ÷ 100
    // my franchise
    'self_purse',               // ÷ starting purse
    'self_slots_left',          // ÷ max squad size
    'self_overseas_left',       // ÷ max overseas
    'self_xi_strength',         // current best-XI strength ÷ 100
    'self_xi_gain',             // how much this player lifts my XI ÷ 10
    'self_need_batsman', 'self_need_bowler', 'self_need_all_rounder', 'self_need_keeper',
    'self_pacing',              // (share spent ÷ share of auction done) ÷ 3
    'self_affordability',       // spend limit ÷ fair value ÷ 4
    // the market
    'progress',
    'upcoming_same_role',       // ÷ 40
    'upcoming_similar',         // same role, rating ≥ this - 3, ÷ 10
    'upcoming_better',          // same role, higher rating, ÷ 10
    'rival_max_purse',          // ÷ starting purse
    'rival_mean_purse',
    'rivals_can_afford',        // share of rivals able to pay fair value
    'rivals_need_role',         // share of rivals still short of this role
    'team_count',               // ÷ 10
    // personality
    ...PERSONA_DIMS.map((d) => `persona_${d}`)
])

export const OBSERVATION_SIZE = OBSERVATION_FEATURES.length

const clip = (x, lo, hi) => Math.min(hi, Math.max(lo, x))

export const roleCounts = (squad) => {
    const counts = { BATSMAN: 0, BOWLER: 0, 'ALL ROUNDER': 0, 'WICKET KEEPER': 0 }
    for (const p of squad || []) if (p && counts[p.role] !== undefined) counts[p.role]++
    return counts
}

export const roleNeed = (squad, role) => {
    const target = SQUAD_TARGETS[role] ?? 0
    if (!target) return 0
    return Math.max(0, target - roleCounts(squad)[role]) / target
}

// Everything the rule bots and the observation share, computed once.
export const deriveLotFacts = (ctx) => {
    const rules = ctx.rules || DEFAULT_RULES
    const { lot, self } = ctx
    const fv = fairValue(lot, rules.pursePerTeam)
    const gain = xiGain(self.squad, lot)
    const spendLimit = botSpendLimit(self, cheapSlotPrice(ctx.upcoming))
    const upcomingSameRole = (ctx.upcoming || []).filter((p) => p.role === lot.role)
    return {
        rules,
        fairValue: fv,
        xiGain: gain,
        spendLimit,
        // Can this team bid on the lot at all (squad/overseas/base price)?
        eligible: bidBlocker({
            team: self,
            lot: { ...lot, currentBidderId: '' },
            rules,
            amount: lot.basePrice
        }) === null && spendLimit >= lot.basePrice,
        upcomingSameRole: upcomingSameRole.length,
        upcomingSimilar: upcomingSameRole.filter((p) => p.rating >= lot.rating - 3).length,
        upcomingBetter: upcomingSameRole.filter((p) => p.rating > lot.rating).length
    }
}

export const buildObservation = (ctx, persona = ZERO_PERSONA) => {
    const facts = deriveLotFacts(ctx)
    const { rules } = facts
    const { lot, self } = ctx
    const rivals = ctx.rivals || []
    const unit = slotBudget(rules)
    const stats = lot.stats || {}

    const spentShare = 1 - self.purseLeft / rules.pursePerTeam
    const progress = clip(ctx.progress ?? 0, 0, 1)
    const rivalPurses = rivals.map((r) => r.purseLeft / rules.pursePerTeam)
    const rivalsCanAfford = rivals.filter((r) =>
        bidBlocker({ team: r, lot: { ...lot, currentBidderId: '' }, rules, amount: facts.fairValue }) === null
    ).length
    const rivalsNeedRole = rivals.filter((r) => roleNeed(r.squad, lot.role) > 0).length

    const obs = [
        lot.role === 'BATSMAN' ? 1 : 0,
        lot.role === 'BOWLER' ? 1 : 0,
        lot.role === 'ALL ROUNDER' ? 1 : 0,
        lot.role === 'WICKET KEEPER' ? 1 : 0,
        lot.nationality === 'Overseas' ? 1 : 0,
        (lot.rating - 60) / 40,
        lot.basePrice / unit,
        facts.fairValue / unit,
        (stats.bat ?? 0) / 100,
        (stats.bwl ?? 0) / 100,
        (stats.pwr ?? 0) / 100,
        (stats.tec ?? 0) / 100,
        (stats.clt ?? 0) / 100,

        self.purseLeft / rules.pursePerTeam,
        (rules.maxPlayers - self.playerCount) / rules.maxPlayers,
        (rules.maxOverseas - self.overseasCount) / rules.maxOverseas,
        teamStrength(self.squad) / 100,
        clip(facts.xiGain / 10, 0, 1),
        ...ROLE_ORDER.map((role) => roleNeed(self.squad, role)),
        clip(spentShare / Math.max(progress, 0.05), 0, 3) / 3,
        clip(facts.spendLimit / facts.fairValue, 0, 4) / 4,

        progress,
        clip(facts.upcomingSameRole / 40, 0, 1),
        clip(facts.upcomingSimilar / 10, 0, 1),
        clip(facts.upcomingBetter / 10, 0, 1),
        rivalPurses.length ? Math.max(...rivalPurses) : 0,
        rivalPurses.length ? rivalPurses.reduce((s, x) => s + x, 0) / rivalPurses.length : 0,
        rivals.length ? rivalsCanAfford / rivals.length : 0,
        rivals.length ? rivalsNeedRole / rivals.length : 0,
        (rivals.length + 1) / 10,

        ...persona
    ]

    if (obs.length !== OBSERVATION_SIZE) {
        throw new Error(`Observation has ${obs.length} values, expected ${OBSERVATION_SIZE}`)
    }
    return obs
}

// The most a bot will pay if it takes `action`. 0 = pass.
export const capForAction = (ctx, action, facts = deriveLotFacts(ctx)) => {
    const multiplier = CAP_MULTIPLIERS[action] ?? 0
    if (multiplier === 0 || !facts.eligible) return 0
    const cap = Math.min(Math.floor(multiplier * facts.fairValue), facts.spendLimit)
    return cap >= ctx.lot.basePrice ? cap : 0
}

// 1 = legal. Pass is always legal; a bidding action is legal only if the
// team could actually place the opening bid and the cap reaches it (an
// action whose cap is below the base price would just be a pass in disguise).
export const actionMask = (ctx, facts = deriveLotFacts(ctx)) =>
    CAP_MULTIPLIERS.map((_, action) => (action === 0 ? 1 : capForAction(ctx, action, facts) > 0 ? 1 : 0))
