// Reward for the RL bots.
//
// Main signal, at the end of the auction: how the bot's best XI compares with
// the other nine teams — both by how much (strength difference) and by
// place (rank). Rank on its own would make "second by 0.1" and "second by 5"
// look identical; the difference on its own would ignore that winning is what
// players see on the leaderboard.
//
// Persona terms are deliberately small (weight 0.3). They tilt a bot towards
// a style without letting it abandon winning to satisfy the style — the
// "tie-breaking secondary rewards" idea from d'Eon et al. 2024.
//
// Unspent purse earns nothing, so there is no reward for hoarding
// (cocoa-huang/rl-ad-bidding shows how easily budget agents learn to hoard).

import { selectBestXI, teamStrength } from './scoring.js'
import { PERSONA_DIMS } from './observation.js'

const PERSONA_WEIGHT = 0.3

const average = (xs) => (xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : 0)

const clip = (x, lo, hi) => Math.min(hi, Math.max(lo, x))

// Each term is roughly in -0.5..+0.5 so personas are comparable.
export const personaTerms = (team, rules) => {
    const { players: xi } = selectBestXI(team.squad)
    return {
        aggression: clip(team.purseSpent / rules.pursePerTeam - 0.5, -0.5, 0.5),
        bowlingFocus: clip((average(xi.map((p) => p.stats?.bwl ?? 0)) - 50) / 100, -0.5, 0.5),
        battingFocus: clip((average(xi.map((p) => p.stats?.bat ?? 0)) - 50) / 100, -0.5, 0.5),
        overseasFocus: clip(xi.filter((p) => p.nationality === 'Overseas').length / 4 - 0.5, -0.5, 0.5),
        starFocus: clip(team.squad.filter((p) => p.rating >= 90).length / 4 - 0.5, -0.5, 0.5)
    }
}

export const finalStanding = (teams) => {
    const strengths = teams.map((t) => teamStrength(t.squad))
    const order = strengths.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0])
    const rank = new Array(teams.length)
    order.forEach(([, i], place) => { rank[i] = place })
    return { strengths, rank }
}

export const episodeReward = (sim, learnerIndex, persona) => {
    const { strengths, rank } = finalStanding(sim.teams)
    const n = sim.teams.length
    const others = strengths.filter((_, i) => i !== learnerIndex)

    const relative = (strengths[learnerIndex] - average(others)) / 10
    const placement = n > 1 ? 0.5 * (1 - (2 * rank[learnerIndex]) / (n - 1)) : 0

    const terms = personaTerms(sim.teams[learnerIndex], sim.rules)
    const style = PERSONA_DIMS.reduce((s, dim, i) => s + (persona[i] || 0) * terms[dim], 0)

    return {
        reward: relative + placement + PERSONA_WEIGHT * style,
        strength: strengths[learnerIndex],
        rank: rank[learnerIndex],
        strengths,
        terms
    }
}

// Small per-lot nudge: buying someone who doesn't make the XI wastes money.
export const lotShaping = ({ won, price, xiGainAtOpen, slotBudget }) =>
    won && xiGainAtOpen <= 0.05 ? -0.02 * (price / slotBudget) : 0
