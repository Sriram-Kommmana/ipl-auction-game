// Valuation signals for the rule bots — the bridge between the planning
// layer ("what is safe, what is missing, what is scarce") and the
// personalities ("how much do I want this player"). Every number here is a
// fact about the team, the market or the player; none of it is a decision.
// Personalities (ruleBots.js) weight these signals differently.

import { fairValue } from './valuation.js'
import { selectBestXI } from './scoring.js'
import { REQUIREMENT_STATUS, teamComposition } from './planning.js'

const EMPTY = Object.freeze([])
const STAR_RATING = 90

// ── Market summary (once per lot, shared by every team) ───────────────────
// Everyone still to come: ratings (for "how many upgrades are left"), fair
// values (for "what does a decent filler cost") and the market's typical
// rating-per-rupee (for Moneyball's efficiency). Cached on the same arrays
// the planning layer caches on, so a new lot / unsold player / phase gives
// fresh numbers.
const marketCache = new WeakMap()
const marketOf = (ctx) => {
    const upcoming = ctx.upcoming || EMPTY
    const returning = ctx.returning || EMPTY
    const purse = ctx.rules.pursePerTeam
    let inner = marketCache.get(upcoming)
    if (!inner) marketCache.set(upcoming, (inner = new WeakMap()))
    let entry = inner.get(returning)
    if (!entry || entry.purse !== purse) {
        const all = [...upcoming, ...returning]
        const values = all.map((p) => fairValue(p, purse))
        const efficiency = all.map((p, i) => p.rating / values[i]).sort((a, b) => a - b)
        entry = {
            purse,
            ratings: all.map((p) => p.rating).sort((a, b) => a - b),
            fairValues: values.sort((a, b) => a - b),
            medianEfficiency: efficiency.length ? efficiency[efficiency.length >> 1] : null,
            // Premium value (fair value above base price — the money-worthy
            // part of the market; weak players add almost nothing) still to
            // come in this round, and sitting in the unsold list. Players who
            // went unsold count as passed: nobody valued them at that price.
            premiumUpcoming: upcoming.reduce((sum, p, i) => sum + Math.max(0, values[i] - p.basePrice), 0),
            premiumReturning: returning.reduce((sum, p, i) => sum + Math.max(0, values[upcoming.length + i] - p.basePrice), 0),
            cheapestBase: all.length ? Math.min(...all.map((p) => p.basePrice)) : 0,
            overseasRatings: all.filter((p) => p.nationality === 'Overseas').map((p) => p.rating).sort((a, b) => a - b)
        }
        inner.set(returning, entry)
    }
    return entry
}

const countAbove = (sorted, x) => {
    let lo = 0
    let hi = sorted.length
    while (lo < hi) {
        const mid = (lo + hi) >> 1
        if (sorted[mid] <= x) lo = mid + 1
        else hi = mid
    }
    return sorted.length - lo
}
const quantile = (sorted, q) => (sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))] : 0)
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x))
const premiumOf = (p, purse) => Math.max(0, fairValue(p, purse) - p.basePrice)

// Premium value already bought by all teams — memoised per squad array.
const premiumCache = new WeakMap()
const squadPremium = (squad, purse) => {
    const hit = premiumCache.get(squad)
    if (hit && hit.length === squad.length && hit.purse === purse) return hit.premium
    const premium = squad.reduce((sum, p) => sum + premiumOf(p, purse), 0)
    premiumCache.set(squad, { length: squad.length, purse, premium })
    return premium
}

// Weakest player in the current Best XI (0 while the XI has empty slots),
// memoised per squad array like the other per-squad caches.
const floorCache = new WeakMap()
const xiFloor = (squad) => {
    const hit = floorCache.get(squad)
    if (hit && hit.length === squad.length) return hit.floor
    const xi = selectBestXI(squad)
    const floor = xi.emptySlots > 0 || xi.players.length === 0 ? 0 : Math.min(...xi.players.map((p) => p.rating))
    floorCache.set(squad, { length: squad.length, floor })
    return floor
}

// Overseas slots are the scarcest squad resource (8 per squad, every team
// fills them). Spending one on a player is a mistake while more overseas
// players better than him are still to come than this team would have
// slots left — one of them will need the slot. True = the slot is contested.
export const overseasSlotContested = (ctx) => {
    const { lot, self, rules } = ctx
    if (lot.nationality !== 'Overseas') return false
    const slotsAfter = rules.maxOverseas - self.overseasCount - 1
    return countAbove(marketOf(ctx).overseasRatings, lot.rating) > slotsAfter
}

// ── Signals for one team deciding on the player on the block ──────────────
export const botSignals = (ctx, plan, facts) => {
    const { lot, self } = ctx
    const rules = ctx.rules
    const market = marketOf(ctx)
    const gain = plan.playerImpact.xiGain
    const fullGain = lot.rating / 11 // what he adds by filling an empty XI place
    const progress = clamp(ctx.progress ?? 0, 0, 1)

    // Requirement urgency 0..1: 1 when the planner calls a requirement he
    // fills CRITICAL (or he unlocks one / is the last chance); otherwise how
    // thin the supply is — (this team + rivals needing it) ÷ what's left.
    let urgency = 0
    const fills = gain > 0.05 ? plan.playerImpact.fillsRequirement : []
    for (const r of fills) {
        const req = plan.requirements[r]
        const u = req.status === REQUIREMENT_STATUS.CRITICAL || req.finalOpportunity
            ? 1
            : clamp((req.need + req.rivalsNeeding) / Math.max(1, req.remainingAfterLot), 0, 1)
        urgency = Math.max(urgency, u)
    }
    if (plan.playerImpact.unlocksRequirement) urgency = 1

    // Rivals: how many could realistically pay his fair value (purse minus the
    // cheapest way to fill their own missing XI places), and how hot the
    // market has run (their spending relative to how far the auction is).
    const rivals = ctx.rivals || EMPTY
    const fv = facts.fairValue
    let able = 0
    let purseShare = 0
    for (const r of rivals) {
        const missing = teamComposition(r, rules).missing.players
        if (r.purseLeft - missing * market.cheapestBase >= fv) able++
        purseShare += r.purseLeft / rules.pursePerTeam
    }
    const rivalPurseShare = rivals.length ? purseShare / rivals.length : 1
    const competition = rivals.length ? able / rivals.length : 0
    const heat = (1 - rivalPurseShare) / Math.max(progress, 0.05)

    // Share of the auction's premium value already auctioned (sold or unsold).
    const passed = [self, ...rivals].reduce((sum, t) => sum + squadPremium(t.squad, rules.pursePerTeam), 0) + market.premiumReturning
    const left = market.premiumUpcoming + premiumOf(lot, rules.pursePerTeam)
    const premiumPassed = passed + left > 0 ? passed / (passed + left) : 0

    const openXI = plan.requirements.players.need
    const floor = xiFloor(self.squad)
    const upgradesLeft = countAbove(market.ratings, floor)
    // Typical XI gain still on offer once the XI is full: the average player
    // rated above this team's weakest XI member, replacing him.
    let typicalUpgrade = 0
    if (upgradesLeft > 0) {
        const above = market.ratings.slice(market.ratings.length - upgradesLeft)
        typicalUpgrade = (above.reduce((sum, r) => sum + r, 0) / above.length - floor) / 11
    }

    return {
        fairValue: fv,
        rating: lot.rating,
        isStar: lot.rating >= STAR_RATING,
        starsOwned: self.squad.filter((p) => p.rating >= STAR_RATING).length,
        gain,
        fullGain,
        progress,
        phase: plan.lotContext.phase,
        urgency,
        finalOpportunity: plan.lotContext.finalOpportunity && fills.length > 0,
        equivalentRemaining: plan.lotContext.equivalentRemaining,
        // 0..1: how plentiful comparable players (same role, rating within 3)
        // still are — 10 or more to come counts as fully abundant.
        abundance: clamp(plan.lotContext.equivalentRemaining / 10, 0, 1),
        betterRemaining: plan.lotContext.betterRemaining,
        openXI,
        upgradesLeft,
        rivalPurseShare,
        competition,
        heat,
        spentShare: 1 - self.purseLeft / rules.pursePerTeam,
        // Spending pace against SUPPLY: share of purse spent minus share of
        // the auction's premium value that has already been sold. > 0 =
        // running ahead of the market's value; < 0 = sitting on money while
        // the players worth it are going (money is worth nothing at the end).
        // Pace is a main-market signal: neutral in the re-auction, which keeps
        // its own (Phase 1B.1) discipline.
        paceGap: plan.lotContext.phase === 'reauction' ? 0 : (1 - self.purseLeft / rules.pursePerTeam) - premiumPassed,
        // Rating per rupee versus the rest of the market (>1 = cheaper per point).
        efficiency: market.medianEfficiency ? lot.rating / fv / market.medianEfficiency : 1,

        // Share of a full XI place his gain represents, 0..1, smooth. `softness`
        // is how quickly value falls for small gains (bigger = stricter):
        // filling an empty place → 1; a +0.1 upgrade → close to 0.
        gainQuality(softness) {
            if (gain <= 0) return 0
            return Math.min(1, (gain / (gain + softness)) / (fullGain / (fullGain + softness)))
        },

        // How big an opportunity he is, 0..1, for spending "money per
        // opportunity": while XI places are open that's gain quality (share
        // of a full place); once the XI is full every opportunity is an
        // upgrade, so he's measured against the typical upgrade still on
        // offer — a +0.2 is still a small fraction of one.
        opportunityShare(softness) {
            if (openXI > 0 || typicalUpgrade <= 0) return this.gainQuality(softness)
            return clamp(gain / typicalUpgrade, 0, 1)
        },

        // Money per remaining opportunity. Keeps back a realistic reserve —
        // the planner's floor, or open XI places × a filler price taken from
        // the players actually left (the personality chooses how good a
        // filler: `fillerQuantile` of their fair values) — and spreads the
        // rest over the open XI places plus the upgrades still available
        // (at most `horizon` of them). Late in the auction, when few upgrades
        // are left, this rises: that is the endgame, driven by supply rather
        // than by the clock.
        perOpportunity({ fillerQuantile, horizon }) {
            const openAfter = Math.max(0, openXI - (gain > 0.05 ? 1 : 0))
            const reserve = Math.max(plan.budget.requiredReserve ?? 0, openAfter * quantile(market.fairValues, fillerQuantile))
            const spare = Math.max(0, self.purseLeft - reserve)
            return spare / Math.max(1, openXI + Math.min(horizon, upgradesLeft))
        }
    }
}
