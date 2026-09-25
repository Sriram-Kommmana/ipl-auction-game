// Player "fair value" — the anchor every bot prices against.
//
// There's no historical auction-price data in this game to fit a real
// hedonic model (source: Malhotra 2022, JSA), so this is a hand-calibrated
// curve with the same shape: value grows steeply with quality, and
// all-rounders earn a multi-skill premium (Hindustan Times: franchises pay
// extra for players who fill two roles).
//
// Calibration (against the 323-player pool, 10 teams × ₹12,500L purse):
//   rating 96 → ~₹1,370L   rating 90 → ~₹820L   rating 85 → ~₹490L
//   rating 80 → ~₹265L     rating 75 → ~₹120L   (never below base price)
// The top 200 players sum to ~94% of the ten-team market, so the anchor is
// "what the whole market could plausibly pay", not a price anyone must pay.
// Bots multiply it by their own appetite (0.6× … 3×).

import { DEFAULT_RULES } from './rules.js'

export const FAIR_VALUE_CURVE = Object.freeze({
    k: 0.06,
    exponent: 2.8,
    pivotRating: 60,
    allRounderBonus: 1.5,
    referencePurse: 12500
})

export const fairValue = (player, pursePerTeam = DEFAULT_RULES.pursePerTeam) => {
    const { k, exponent, pivotRating, allRounderBonus, referencePurse } = FAIR_VALUE_CURVE
    const effective = player.rating + (player.role === 'ALL ROUNDER' ? allRounderBonus : 0)
    const raw = k * Math.pow(Math.max(0, effective - pivotRating), exponent)
    const scaled = Math.round(raw * (pursePerTeam / referencePurse))
    return Math.max(player.basePrice, scaled)
}

// Average money per squad slot — the natural unit for "is this expensive?"
export const slotBudget = (rules = DEFAULT_RULES) => rules.pursePerTeam / rules.maxPlayers
