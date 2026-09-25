// The four hand-written bots. Each returns a price cap for the current lot
// (0 = not interested); the runtime bids up to it one increment at a time.
//
// All four share the logic real franchises describe (Hindustan Times,
// "IPL auction: the calm calculations hidden in the frenzied buying"):
//   - Money goes "to the position, not the player": value comes from how
//     much a player improves the playing XI (xiGain), not the name.
//   - Walk away once cheaper substitutes are still to come (KKR letting
//     David Hussey go past $400k because Morgan was available later).
//   - All-rounders earn a premium (already in fairValue).
// They differ only in appetite and timing, and add ±8% noise so a human
// can't learn the exact walk-away price.

import { deriveLotFacts, roleCounts, roleNeed } from './observation.js'
import { RULE_PERSONAS } from './personas.js'
import { XI_SIZE } from './rules.js'
import { XI_RULES } from './scoring.js'

const NOISE = 0.08

// Does this player fill a requirement the XI can't meet yet? A team with no
// keeper, or fewer than 5 bowling options, is playing with an empty slot.
const fillsMissingRequirement = (ctx) => {
    const counts = roleCounts(ctx.self.squad)
    const { role } = ctx.lot
    if (role === 'WICKET KEEPER') return counts['WICKET KEEPER'] < XI_RULES.minKeepers
    if (role === 'BOWLER' || role === 'ALL ROUNDER') {
        return counts.BOWLER + counts['ALL ROUNDER'] < XI_RULES.minBowlingOptions
    }
    return false
}

const appetite = {
    moneyball(ctx, f) {
        let value = f.fairValue * 0.9
        // Enough comparable players still to come → no need to fight now.
        if (f.upcomingSimilar >= 3) value *= 0.8
        return value
    },

    starChaser(ctx, f) {
        const { rating } = ctx.lot
        let value
        if (rating >= 90) value = f.fairValue * 1.9
        else if (rating >= 86) value = f.fairValue * 1.25
        else value = f.fairValue * 0.65
        // Can't keep splurging once the purse is nearly gone.
        if (ctx.self.purseLeft < ctx.rules.pursePerTeam * 0.25) value *= 0.6
        return value
    },

    balancedBuilder(ctx, f) {
        const need = roleNeed(ctx.self.squad, ctx.lot.role) // 0..1
        let value = f.fairValue * (0.55 + 0.9 * need)
        if (ctx.lot.role === 'ALL ROUNDER') value *= 1.1
        if (ctx.lot.role === 'WICKET KEEPER' && roleCounts(ctx.self.squad)['WICKET KEEPER'] === 0) {
            value *= 1.4
        }
        return value
    },

    opportunist(ctx, f) {
        const rivals = ctx.rivals || []
        const rivalMeanPurse = rivals.length
            ? rivals.reduce((s, r) => s + r.purseLeft, 0) / rivals.length / ctx.rules.pursePerTeam
            : 1
        let value = (ctx.progress ?? 0) < 0.35
            ? f.fairValue * 0.7
            : f.fairValue * (0.85 + 0.9 * (1 - rivalMeanPurse)) // rivals broke → push harder
        // Last chance at a real upgrade in this role.
        if (f.upcomingBetter === 0 && f.xiGain > 3) value *= 1.15
        return value
    }
}

export const RULE_BOT_IDS = Object.freeze(Object.keys(RULE_PERSONAS))

export const ruleBotCap = (personaId, ctx, rng = Math.random, { noise = NOISE } = {}) => {
    const decide = appetite[personaId]
    if (!decide) throw new Error(`Unknown rule persona: ${personaId}`)

    const f = deriveLotFacts(ctx)
    if (!f.eligible) return 0

    const squadSize = ctx.self.playerCount
    const useful = f.xiGain > 0.05

    let value
    if (!useful) {
        // Wouldn't make the XI: only a cheap depth signing, and only while
        // the squad is still thin.
        if (squadSize >= 15) return 0
        value = Math.min(ctx.lot.basePrice * 1.2, f.fairValue * 0.5)
    } else {
        if (squadSize >= 22 && f.xiGain < 0.5) return 0
        value = decide(ctx, f)

        // Pacing brake: spending far ahead of the auction's progress.
        const progress = ctx.progress ?? 0
        const spentShare = 1 - ctx.self.purseLeft / ctx.rules.pursePerTeam
        if (personaId !== 'starChaser' && spentShare > progress + 0.25) value *= 0.8

        // Every personality still has to field a legal XI.
        if (fillsMissingRequirement(ctx)) value = Math.max(value, f.fairValue * 1.1)

        // Endgame: money left at the end buys nothing. Past half-way, a bot
        // with plenty of money per open XI slot starts paying above fair value.
        if (progress > 0.5) {
            const openSlots = Math.max(1, XI_SIZE - squadSize)
            const richness = f.spendLimit / openSlots / f.fairValue
            if (richness > 1) {
                const push = Math.min(1 + (richness - 1) * (progress - 0.5) * 2, 2.2)
                value = Math.max(value, f.fairValue * push)
            }
        }
    }

    value *= 1 + (rng() * 2 - 1) * noise
    const cap = Math.min(Math.floor(value), f.spendLimit)
    return cap >= ctx.lot.basePrice ? cap : 0
}
