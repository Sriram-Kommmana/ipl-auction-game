// Safety invariants of a finished auction (Phase 2C, Step 7). Checked at the
// end of every RL episode — training, validation and evaluation — so a
// learner that finds a hole in the rules is caught the moment it happens.
// Read-only: it inspects the simulator's final state and never changes it.
//
//   purse     no negative purse; spent + left = purse; spent = sum of prices
//   squad     playerCount = squad size ≤ maxPlayers
//   overseas  overseasCount = overseas in squad ≤ maxOverseas
//   buys      each player sold at most once, never below base price; a
//             re-auction sale was unsold in the main round
//   XI        every team's Best XI is a legal selection from its own squad
//             (≤ 4 overseas; a full XI has a keeper and 5 bowling options)
//   progress  the auction finished within its lot budget (no deadlock)

import { XI_RULES, selectBestXI } from '../scoring.js'

const isOverseas = (p) => p.nationality === 'Overseas'
const isBowlingOption = (p) => p.role === 'BOWLER' || p.role === 'ALL ROUNDER'

export const auditAuction = (sim) => {
    const violations = []
    const add = (msg) => violations.push(msg)
    const { rules, teams } = sim
    if (!sim.done) add('auction not finished')
    if (sim.history.length > 2 * sim.mainLength) add(`${sim.history.length} lots for a ${sim.mainLength}-player pool`)

    const soldTo = new Map()
    const unsoldMain = new Set()
    const offered = { main: new Set(), reauction: new Set() }
    const spent = teams.map(() => 0)
    for (const h of sim.history) {
        if (offered[h.phase].has(h.slNo)) add(`player ${h.slNo} offered twice in the ${h.phase} round`)
        offered[h.phase].add(h.slNo)
        if (h.phase === 'reauction' && !unsoldMain.has(h.slNo)) add(`player ${h.slNo} re-auctioned without going unsold`)
        if (h.winner === null) {
            if (h.phase === 'main') unsoldMain.add(h.slNo)
            continue
        }
        if (soldTo.has(h.slNo)) add(`player ${h.slNo} sold twice`)
        soldTo.set(h.slNo, h.winner)
        const p = sim.players.get(h.slNo)
        if (!(Number.isInteger(h.price) && h.price >= p.basePrice)) add(`player ${h.slNo} sold for ${h.price} (base ${p.basePrice})`)
        spent[h.winner] += h.price
    }

    teams.forEach((t, i) => {
        const tag = `team ${i}`
        if (t.purseLeft < 0) add(`${tag}: purse ${t.purseLeft}`)
        if (t.purseSpent !== spent[i]) add(`${tag}: purseSpent ${t.purseSpent} ≠ prices ${spent[i]}`)
        if (t.purseLeft + t.purseSpent !== rules.pursePerTeam) add(`${tag}: left + spent ≠ ${rules.pursePerTeam}`)
        if (t.playerCount !== t.squad.length || t.playerCount > rules.maxPlayers) add(`${tag}: squad ${t.squad.length} / count ${t.playerCount}`)
        const os = t.squad.filter(isOverseas).length
        if (t.overseasCount !== os || os > rules.maxOverseas) add(`${tag}: overseas ${os} / count ${t.overseasCount}`)
        const ids = new Set(t.squad.map((p) => p.slNo))
        if (ids.size !== t.squad.length) add(`${tag}: duplicate player in squad`)
        for (const p of t.squad) if (soldTo.get(p.slNo) !== i) add(`${tag}: player ${p.slNo} in squad without a sale`)

        const xi = selectBestXI(t.squad)
        const xiIds = new Set(xi.players.map((p) => p.slNo))
        if (xiIds.size !== xi.players.length || xi.players.some((p) => !ids.has(p.slNo))) add(`${tag}: XI not drawn from the squad`)
        if (xi.players.length + xi.emptySlots !== XI_RULES.size) add(`${tag}: XI size`)
        if (xi.players.filter(isOverseas).length > XI_RULES.maxOverseas) add(`${tag}: XI overseas`)
        if (xi.emptySlots === 0) {
            if (xi.players.filter((p) => p.role === 'WICKET KEEPER').length < XI_RULES.minKeepers) add(`${tag}: full XI without a keeper`)
            if (xi.players.filter(isBowlingOption).length < XI_RULES.minBowlingOptions) add(`${tag}: full XI short of bowling`)
        }
    })
    return violations
}
