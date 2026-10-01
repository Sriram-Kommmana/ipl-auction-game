// Auction Awards — presentation metadata for the results page.
//
// Computed from the final squads only. Awards never feed back into the
// Squad Score or the ranking (squadRanking.js); they just read them. Every
// award breaks ties explicitly, so the same result always gives the same
// awards.
//
//   champion      every team ranked 1st by rankSquads (joint champions if
//                 the rank is shared)
//   bestValue     the purchase with the biggest saving against the game's
//                 own valuation: fair value − price paid (₹ lakhs), where
//                 fair value is valuation.js's fairValue curve (the anchor
//                 every bot prices against) for the room's purse. Ties:
//                 higher rating, then lower price, then lower slNo.
//   biggestSplurge  the highest price paid. Ties: higher rating, then
//                 lower slNo.
//   bargain       the cheapest purchase among high-rated players (rating
//                 ≥ BARGAIN_MIN_RATING — the top ~22% of the player pool),
//                 excluding the Best Value Buy winner so the two awards go
//                 to different players. Ties: higher rating, then lower slNo.
//   deepestSquad  the highest injury cover (the squad's expected Best XI
//                 strength with any one starter out). Ties: better rank.
//   highestRated  the highest-rated player bought. Ties: lower price, then
//                 lower slNo.
//
// An award with no eligible candidate (e.g. nobody bought a player rated
// 88+) is left out rather than given to someone who doesn't fit it.

import { fairValue } from './valuation.js'
import { rankSquads } from './squadRanking.js'

export const BARGAIN_MIN_RATING = 88

// fairValue's curve without its base-price floor: saved results don't keep
// each player's base price, and the floor never matters for 88+ players.
export const playerFairValue = (player, pursePerTeam) =>
    fairValue({ rating: player.rating, role: player.role, basePrice: 0 }, pursePerTeam)

// Every team starts with the same purse, so spent + left recovers it.
const roomPurse = (teams) =>
    teams.reduce((max, t) => Math.max(max, (Number(t.purseSpent) || 0) + (Number(t.purseLeft) || 0)), 0) || undefined

// First element after sorting by the comparator — a pure "best of" pick.
const best = (items, comparator) => items.slice().sort(comparator)[0] ?? null

const byRatingThenSl = (a, b) => (b.player.rating - a.player.rating) || (a.player.slNo - b.player.slNo)

export const computeAwards = (teams = []) => {
    const ranked = teams.length && teams.every((t) => t.ranking) ? teams : rankSquads(teams)
    if (!ranked.length) return {}
    const purse = roomPurse(ranked)

    // Purchases with a recorded price (a price of 0 means the sale wasn't
    // found in the history — never a real purchase).
    const buys = ranked.flatMap((team) => (team.squad || [])
        .filter((p) => Number(p.boughtFor) > 0)
        .map((p) => ({ team, player: p, price: Number(p.boughtFor), fair: playerFairValue(p, purse) })))

    const awards = {}

    awards.champion = { teams: ranked.filter((t) => t.ranking.rank === 1) }

    const value = best(buys.filter((b) => b.fair - b.price > 0),
        (a, b) => ((b.fair - b.price) - (a.fair - a.price)) || (b.player.rating - a.player.rating) ||
            (a.price - b.price) || (a.player.slNo - b.player.slNo))
    if (value) awards.bestValue = { ...value, saving: value.fair - value.price }

    const splurge = best(buys, (a, b) => (b.price - a.price) || byRatingThenSl(a, b))
    if (splurge) awards.biggestSplurge = splurge

    const bargain = best(
        buys.filter((b) => b.player.rating >= BARGAIN_MIN_RATING && b.player !== value?.player),
        (a, b) => (a.price - b.price) || byRatingThenSl(a, b))
    if (bargain) awards.bargain = bargain

    // Ranked order already breaks injury-cover ties by rank.
    const deepest = best(ranked, (a, b) => b.ranking.injuryCover - a.ranking.injuryCover)
    if (deepest) awards.deepestSquad = { team: deepest }

    const top = best(buys, (a, b) => (b.player.rating - a.player.rating) || (a.price - b.price) || (a.player.slNo - b.player.slNo))
    if (top) awards.highestRated = top

    return awards
}
