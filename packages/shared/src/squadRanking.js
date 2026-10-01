// Squad ranking for the results page — who won the auction, and why.
//
// How cricket analysts rate squads after an IPL auction comes down to the
// same few things: how strong the playing XI is, the Impact Player (a 12th
// man since 2023), how well the squad covers an injury, and whether the XI
// is balanced between batting and bowling. This module turns those into
// one Squad Score (0-100) built ON TOP of the existing Best XI (scoring.js,
// unchanged — it is also what the RL bots were trained to maximise):
//
//   Squad Score = 60% Matchday strength + 20% Injury cover + 20% Balance
//
//   Matchday strength — the best legal XI (scoring.js: max 4 overseas, a
//     keeper, 5 bowling options, empty slots count 0) plus the Impact
//     Player: the best bench player the IPL rule allows to come on (an
//     overseas player only if the XI has fewer than 4 overseas). Average
//     rating of those 12; no eligible bench player counts as 0.
//   Injury cover — the expected Best XI strength if one starter is
//     unavailable: re-pick the best legal XI without each starter in turn,
//     average the 11 results. A deep squad barely drops; a thin one loses
//     a whole slot.
//   Balance — the XI's batting unit and bowling unit from the player stats:
//     batting = average of the top 7 batting values in the XI (batting
//     value = mean of BAT and PWR), bowling = average of the top 5 BWL
//     values among the XI's bowling options. Balance = their mean. Missing
//     batters or bowlers count 0.
//
// Ranking order (rankSquads), each compared at the precision shown on
// screen (1 decimal) so what you see is what decided it:
//   1. Squad Score   2. XI strength   3. Injury cover   4. less purse spent
// Teams equal on all four share the rank ("=3").
//
// Pure functions, browser-safe — used by the server when results are saved
// and by the results page.

import { XI_SIZE } from './rules.js'
import { XI_RULES, selectBestXI, xiTotal } from './scoring.js'

export const SQUAD_SCORE_WEIGHTS = Object.freeze({ matchday: 0.6, injuryCover: 0.2, balance: 0.2 })
export const MATCHDAY_SIZE = XI_SIZE + 1
const BATTING_UNIT = 7
const BOWLING_UNIT = 5

const round1 = (x) => Math.round(x * 10) / 10
const isOverseas = (p) => p.nationality === 'Overseas'
const isBowlingOption = (p) => p.role === 'BOWLER' || p.role === 'ALL ROUNDER'
const stat = (p, k) => Number(p.stats?.[k]) || 0
export const battingValue = (p) => (stat(p, 'bat') + stat(p, 'pwr')) / 2
export const bowlingValue = (p) => stat(p, 'bwl')

// Average of the top `n` values; fewer than `n` values → the missing ones count 0.
const topAverage = (values, n) => values.sort((a, b) => b - a).slice(0, n).reduce((s, v) => s + v, 0) / n

// Best bench player the Impact Player rule allows onto the field.
const pickImpactPlayer = (squad, xi) => {
    const inXI = new Set(xi)
    const overseasInXI = xi.filter(isOverseas).length
    return squad
        .filter((p) => !inXI.has(p) && (overseasInXI < XI_RULES.maxOverseas || !isOverseas(p)))
        .sort((a, b) => (b.rating - a.rating) || ((a.slNo ?? 0) - (b.slNo ?? 0)))[0] ?? null
}

// Every number the leaderboard shows for one squad.
export const evaluateSquad = (squad = []) => {
    const players = (squad || []).filter(Boolean)
    const best = selectBestXI(players)
    const xi = best.players
    const impact = pickImpactPlayer(players, xi)

    const matchdayRaw = (best.total + (impact?.rating ?? 0)) / MATCHDAY_SIZE

    // One starter missing at a time; the squad re-picks its best legal XI.
    const injuryRaw = xi.length
        ? xi.reduce((s, out) => s + xiTotal(players.filter((p) => p !== out)), 0) / xi.length / XI_SIZE
        : 0

    const battingRaw = topAverage(xi.map(battingValue), BATTING_UNIT)
    const bowlingRaw = topAverage(xi.filter(isBowlingOption).map(bowlingValue), BOWLING_UNIT)
    const balanceRaw = (battingRaw + bowlingRaw) / 2

    const w = SQUAD_SCORE_WEIGHTS
    const scoreRaw = w.matchday * matchdayRaw + w.injuryCover * injuryRaw + w.balance * balanceRaw

    return {
        score: round1(scoreRaw),
        xiStrength: best.strength,
        matchday: round1(matchdayRaw),
        injuryCover: round1(injuryRaw),
        balance: round1(balanceRaw),
        batting: round1(battingRaw),
        bowling: round1(bowlingRaw),
        xi: xi.map((p) => p.slNo),
        emptySlots: best.emptySlots,
        impactPlayer: impact ? impact.slNo : null
    }
}

// Tie-break keys in order; `better` > 0 means a ranks above b.
export const TIEBREAKERS = Object.freeze([
    { key: 'score', label: 'Squad Score', better: (a, b) => a.score - b.score },
    { key: 'xiStrength', label: 'XI strength', better: (a, b) => a.xiStrength - b.xiStrength },
    { key: 'injuryCover', label: 'injury cover', better: (a, b) => a.injuryCover - b.injuryCover },
    { key: 'purseSpent', label: 'less purse spent', better: (a, b) => b.purseSpent - a.purseSpent }
])

const compare = (a, b) => {
    for (const t of TIEBREAKERS) {
        const d = t.better(a, b)
        if (d !== 0) return { order: -d, decidedBy: t.key }
    }
    return { order: 0, decidedBy: null }
}

// teams: [{ teamId, squad, purseSpent, ... }] → the same teams, best first,
// each with { ranking: { ...evaluateSquad, purseSpent, rank, tied, decidedBy } }.
//   rank      — competition ranking (1, 2, 2, 4): tied teams share a rank
//   tied      — true when another team has exactly the same rank
//   decidedBy — which tie-breaker put this team below the one directly
//               above it ('score' for a plain win; null at rank 1 or a tie)
export const rankSquads = (teams = []) => {
    const rows = teams.map((team) => ({
        team,
        r: { ...evaluateSquad(team.squad), purseSpent: Number(team.purseSpent) || 0 }
    }))
    rows.sort((a, b) => compare(a.r, b.r).order || String(a.team.teamId).localeCompare(String(b.team.teamId)))

    rows.forEach((row, i) => {
        if (i === 0) {
            row.r.rank = 1
            row.r.decidedBy = null
            return
        }
        const { order, decidedBy } = compare(rows[i - 1].r, row.r)
        row.r.rank = order === 0 ? rows[i - 1].r.rank : i + 1
        row.r.decidedBy = decidedBy
    })
    rows.forEach((row) => {
        row.r.tied = rows.some((o) => o !== row && o.r.rank === row.r.rank)
    })
    return rows.map(({ team, r }) => ({ ...team, ranking: r }))
}
