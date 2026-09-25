// The four hand-written bots. Each returns a price cap for the current lot
// (0 = not interested); the runtime bids up to it one increment at a time.
//
// Three layers, each with one job:
//   planning.js   — what is SAFE: may this team buy him at all, and the most
//                   it can pay while its best XI stays reachable (maxSafeBid).
//   botSignals.js — FACTS for valuation: how much of an XI place his gain
//                   is worth, how urgent/scarce the requirement he fills is,
//                   money per remaining opportunity, rival competition.
//   this file     — PERSONALITY: how strongly each bot reacts to those facts.
//
// All four share the logic real franchises describe (Hindustan Times,
// "IPL auction: the calm calculations hidden in the frenzied buying"):
// money goes "to the position, not the player" (value scales with XI gain);
// walk away while comparable players are still to come; all-rounders earn a
// premium (already in fairValue). ±8% noise so a human can't learn the exact
// walk-away price.

import { deriveLotFacts, roleCounts, roleNeed } from './observation.js'
import { classifyOpportunity, planBid, XI_GAIN } from './planning.js'
import { botSignals, overseasSlotContested } from './botSignals.js'
import { RULE_PERSONAS } from './personas.js'

const NOISE = 0.08

// ── Personalities ─────────────────────────────────────────────────────────
// desire(s, ctx, q)  fair-value multiple — each bot's identity. The existing
//                    formulas, with three changes: Moneyball's rises with gain
//                    quality q (it loves big XI gains per rupee), Balanced
//                    Builder's role need counts less while plenty of
//                    substitutes remain, and Star Chaser's old "purse < 25%"
//                    brake is replaced by the pacing below.
// gainSoftness       how fast value falls for small XI gains (bigger = stricter).
// requirementWeight  extra value per unit of requirement urgency (0..1).
// fillerQuantile     how good the players it reserves money for are (share of
//                    the remaining market's fair values) — Star Chaser plans
//                    cheap fillers around its stars, Balanced Builder doesn't.
// horizon            how many upgrades it expects to still make.
// concentration(s)   most it will put into one player, in units of "money
//                    per remaining opportunity" — the pacing discipline.
// surplus(s)         late money: how much above its desire it will go when
//                    it has more money per opportunity than the player costs
//                    ({ multiple: of fair value at most, share: of the gap }).
// paceTolerance      how far its spending may run ahead of the share of the
//                    auction's premium value already sold before it pulls
//                    back (negative = it deliberately runs behind).
// hoardResponse      how strongly it reaches further when it is behind that
//                    pace (sitting on money while good players are going).
// neutralDesire      its desire for an ordinary player; late money is scaled
//                    by desire ÷ this, so each bot's late spending follows its
//                    own priorities instead of one shared formula.
// finalShare         share of the safe maximum it commits when this is the
//                    last player who can fill a requirement.
const PERSONALITIES = {
    // "Most XI per rupee." Strict on small gains, favours players who are
    // cheap for their rating, walks away while substitutes remain, spends
    // surplus money only modestly.
    moneyball: {
        desire(s, ctx, q) {
            // Walks away the more comparable players are still to come.
            let m = 0.9 * (0.7 + 0.6 * q) * (1 - 0.35 * s.abundance)
            return m * Math.min(1.25, Math.max(0.8, Math.pow(s.efficiency, 0.3)))
        },
        gainSoftness: 0.8,
        requirementWeight: 0.6,
        fillerQuantile: 0.35,
        horizon: 4,
        concentration: () => 1.3,
        surplus: () => ({ multiple: 1.6, share: 0.7 }),
        neutralDesire: 0.9,
        paceTolerance: 0,
        hoardResponse: 3,
        finalShare: 0.5
    },

    // "Pay for genuine stars — but keep the rest of the XI buildable." Big
    // premium for 90+, will concentrate several opportunities' money in a
    // star, but that concentration halves with every star already owned and
    // it keeps a filler reserve for the rest of the XI.
    starChaser: {
        desire(s) {
            if (s.rating >= 90) return 1.9
            if (s.rating >= 86) return 1.25
            return 0.65
        },
        gainSoftness: 0.4,
        requirementWeight: 0.4,
        fillerQuantile: 0.2,
        horizon: 3,
        concentration: (s) => (s.isStar ? 3.5 / (1 + 0.75 * s.starsOwned) : 1),
        surplus: (s) => ({ multiple: s.isStar ? 3 : 1.3, share: 0.8 }),
        neutralDesire: 1,
        paceTolerance: 0.2,
        hoardResponse: 1.5,
        finalShare: 0.6
    },

    // "Complete, balanced squad." Values role need and requirement urgency
    // most, concentrates money on structural gaps, and spends late money on
    // what is still missing.
    balancedBuilder: {
        desire(s, ctx) {
            // Role need (0..1) matters less while plenty of substitutes remain.
            const need = roleNeed(ctx.self.squad, ctx.lot.role) * (1 - 0.5 * s.abundance)
            let m = 0.55 + 0.9 * need
            if (ctx.lot.role === 'ALL ROUNDER') m *= 1.1
            if (ctx.lot.role === 'WICKET KEEPER' && roleCounts(ctx.self.squad)['WICKET KEEPER'] === 0) m *= 1.4
            return m
        },
        gainSoftness: 0.6,
        requirementWeight: 1.2,
        fillerQuantile: 0.5,
        horizon: 3,
        concentration: (s, ctx) => 1 + s.urgency + 0.3 * roleNeed(ctx.self.squad, ctx.lot.role),
        surplus: () => ({ multiple: 2, share: 0.8 }),
        neutralDesire: 1,
        paceTolerance: 0.1,
        hoardResponse: 2,
        finalShare: 0.9
    },

    // "Exploit the market." Cautious early; later pushes harder the poorer
    // its rivals are and the fewer of them can afford the player, backs off
    // when the market is overheated or plenty of substitutes remain for
    // rivals to fight over.
    opportunist: {
        desire(s) {
            let m = s.progress < 0.35 ? 0.7 : 0.85 + 0.9 * (1 - s.rivalPurseShare)
            m *= 1 + 0.5 * (1 - s.competition) // few rivals can pay → pounce
            m *= 1 - 0.25 * Math.min(1, Math.max(0, s.heat - 1)) // overheated market → wait
            if (s.equivalentRemaining >= 5 && s.competition > 0.5) m *= 0.85 // let rivals fight
            if (s.betterRemaining === 0 && s.gain > 3) m *= 1.15 // last real upgrade in his role
            return m
        },
        gainSoftness: 0.6,
        requirementWeight: 0.8,
        fillerQuantile: 0.4,
        horizon: 3,
        concentration: (s) => 1 + (1 - s.competition),
        surplus: (s) => ({ multiple: 1.5 + (1 - s.competition), share: 0.4 + 0.5 * (1 - s.rivalPurseShare) }),
        neutralDesire: 1,
        paceTolerance: 0.05,
        hoardResponse: 3,
        finalShare: 0.7
    }
}

export const RULE_BOT_IDS = Object.freeze(Object.keys(RULE_PERSONAS))

// What a personality would pay for a player who improves its XI.
//   worth   = fair value × desire × gain quality × (1 + requirement urgency)
//   surplus = late money nudges worth up toward min(money per opportunity ×
//             his share of an opportunity, a personality multiple of fair
//             value × gain quality)
//   pacing  = never more than `concentration` × money per opportunity (less
//             when spending runs ahead of the premium value already sold)…
//   …except a critical requirement (at least (1 + requirementWeight) × fair
//             value) and a final opportunity (a share of the safe maximum).
// Everything is later capped by the planner's maxSafeBid.
export const personalityValue = (personaId, ctx, f, plan) => {
    const P = PERSONALITIES[personaId]
    const s = botSignals(ctx, plan, f)
    const quality = s.gainQuality(P.gainSoftness)
    if (quality <= 0) return 0

    const desire = P.desire(s, ctx, quality)
    const worth = s.fairValue * desire * quality * (1 + P.requirementWeight * s.urgency)
    // Late money follows the bot's own priorities: scaled by how much more (or
    // less) it wants this player than an ordinary one.
    let priority = Math.min(1.6, Math.max(0.6, desire / P.neutralDesire))
    // Behind the supply pace (money idle while good players go) → the bot's
    // plan is too spread out: put more into each real opportunity and reach
    // further above fair value. Small gains stay a small share of it.
    const hoard = Math.max(0, -s.paceGap)
    const perOpportunity = s.perOpportunity(P) * (1 + 3 * hoard)
    // Idle money also lifts a LOW priority back toward neutral (a player who
    // really improves the XI beats unspent purse); high priorities — a star
    // for Star Chaser, a gap for Balanced Builder — keep their premium.
    if (priority < 1) priority += (1 - priority) * Math.min(1, 2 * hoard)
    const surplus = P.surplus(s)
    // Plenty of comparable players still to come softens the normal late push
    // — but not the push from being behind pace: a bot that keeps losing
    // those substitutes has no reason to wait for the next one.
    const share = Math.min(1, surplus.share * (1 - 0.5 * s.abundance) + 2 * P.hoardResponse * hoard)
    const multiple = surplus.multiple * (1 + 2 * hoard)
    // Both limits scale with the size of the gain: a +0.1 never earns a big
    // premium, even when the bot has money and nothing better is left.
    const target = priority * Math.min(perOpportunity * s.opportunityShare(P.gainSoftness), s.fairValue * multiple * quality)
    let value = worth + share * Math.max(0, target - worth)

    // Pacing: at most `concentration` × money per opportunity — less once the
    // bot is running further ahead of the supply pace than its tolerance.
    const ahead = s.paceGap > P.paceTolerance
    if (ahead) value *= 0.8
    value = Math.min(value, perOpportunity * P.concentration(s, ctx) * (ahead ? 0.7 : 1))
    if (s.urgency >= 1) value = Math.max(value, s.fairValue * (1 + P.requirementWeight))
    if (s.finalOpportunity) value = Math.max(value, plan.budget.maxSafeBid * P.finalShare)
    return value
}

// The cheap price every personality pays for a squad player who won't make the XI.
const depthValue = (ctx, f) => Math.min(ctx.lot.basePrice * 1.2, f.fairValue * 0.5)

// Re-auction (Phase 1B.1): a second chance at players nobody bought. The
// category comes from planning.classifyOpportunity; prices go through the
// same personality valuation.
//   critical → at least 3× fair value (the largest cap multiplier bots use),
//              or everything the plan can safely spend if nobody else could
//              ever fill that requirement
//   useful / marginal → the personality's value (gain quality already
//              scales it down for small gains)
//   depth    → the usual cheap depth price
//   none     → pass
const CRITICAL_MULTIPLIER = 3
const reauctionValue = (personaId, ctx, f, plan) => {
    const opportunity = classifyOpportunity(ctx, plan)
    switch (opportunity.category) {
        case 'critical':
            return Math.max(
                personalityValue(personaId, ctx, f, plan),
                opportunity.finalOpportunity ? plan.budget.maxSafeBid : f.fairValue * CRITICAL_MULTIPLIER
            )
        case 'useful':
        case 'marginal':
            return personalityValue(personaId, ctx, f, plan)
        case 'depth':
            return depthValue(ctx, f)
        default:
            return 0
    }
}

// Personality desire capped by the shared planning layer: a bot may only bid
// if the purchase keeps the best reachable XI reachable, and never more than
// `maxSafeBid` (purse minus what completing that XI still costs). Pass `plan`
// in to avoid recomputing it.
export const ruleBotCap = (personaId, ctx, rng = Math.random, { noise = NOISE, plan = planBid(ctx) } = {}) => {
    if (!PERSONALITIES[personaId]) throw new Error(`Unknown rule persona: ${personaId}`)

    if (!plan.allowed) return 0
    const f = deriveLotFacts(ctx)

    let value
    if (plan.lotContext.phase === 'reauction') {
        value = reauctionValue(personaId, ctx, f, plan)
    } else if (f.xiGain < XI_GAIN.useful && overseasSlotContested(ctx)) {
        // A depth or marginal overseas player would take an overseas slot a
        // better overseas player still to come will need.
        return 0
    } else if (f.xiGain <= 0.05) {
        // Wouldn't make the XI: only a cheap depth signing, and only while
        // the squad is still thin.
        if (ctx.self.playerCount >= 15) return 0
        value = depthValue(ctx, f)
    } else {
        value = personalityValue(personaId, ctx, f, plan)
    }
    if (value <= 0) return 0

    value *= 1 + (rng() * 2 - 1) * noise
    const cap = Math.min(Math.floor(value), plan.budget.maxSafeBid)
    return cap >= ctx.lot.basePrice ? cap : 0
}
