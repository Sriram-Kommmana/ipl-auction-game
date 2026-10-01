import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { parsePlayersCsv } from '../src/playersCsv.js'
import { rankSquads } from '../src/squadRanking.js'
import { fairValue } from '../src/valuation.js'
import { BARGAIN_MIN_RATING, computeAwards, playerFairValue } from '../src/auctionAwards.js'

const players = parsePlayersCsv(readFileSync(fileURLToPath(new URL('../../../apps/server/src/db/players.csv', import.meta.url)), 'utf8'))

let sl = 5000
const buy = (rating, boughtFor, role = 'BATSMAN', overseas = false) => ({
    slNo: sl++, playerName: `P${sl}`, role, rating, boughtFor,
    nationality: overseas ? 'Overseas' : 'Indian', stats: { bat: 60, pwr: 60, bwl: 60, tec: 80, clt: 80 }
})
const xi = (r, price) => [buy(r, price, 'WICKET KEEPER'), ...Array.from({ length: 5 }, () => buy(r, price, 'BOWLER')), ...Array.from({ length: 5 }, () => buy(r, price))]
const team = (teamId, squad, purseSpent, purse = 12500) => ({ teamId, teamName: teamId, ownerNickname: `${teamId} owner`, squad, purseSpent, purseLeft: purse - purseSpent })

const lcg = (seed) => () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32)
const randomRoom = (seed) => {
    const rng = lcg(seed)
    const pool = [...players]
    return Array.from({ length: 10 }, (_, i) => {
        const squad = Array.from({ length: 12 + Math.floor(rng() * 13) }, () => {
            const p = pool.splice(Math.floor(rng() * pool.length), 1)[0]
            return { ...p, boughtFor: p.basePrice + 10 * Math.floor(rng() * 150) }
        })
        const spent = squad.reduce((s, p) => s + p.boughtFor, 0)
        return team(`T${i}`, squad, spent, Math.max(12500, spent))
    })
}

test('fair value here is valuation.js fairValue without the base-price floor', () => {
    for (const p of players.slice(0, 60)) {
        assert.equal(playerFairValue(p, 12500), fairValue({ ...p, basePrice: 0 }, 12500))
        assert.ok(playerFairValue(p, 12500) <= fairValue(p, 12500))
    }
})

test('each award follows its documented formula', () => {
    const steal = buy(92, 300) // fair ≈ 1,000+ → saving ≈ 700+
    const pricey = buy(95, 2600)
    const cheapStar = buy(89, 120)
    const top = buy(97, 2000)
    const teams = [
        team('A', [...xi(80, 100), steal], 1400),
        team('B', [...xi(84, 150), pricey, top], 6250),
        team('C', [...xi(78, 50), cheapStar], 670)
    ]
    const a = computeAwards(teams)
    const ranked = rankSquads(teams)
    assert.deepEqual(a.champion.teams.map((t) => t.teamId), [ranked[0].teamId])
    assert.equal(a.biggestSplurge.player, pricey)
    assert.equal(a.highestRated.player, top)
    assert.equal(a.bestValue.player, steal)
    assert.equal(a.bestValue.saving, playerFairValue(steal, 12500) - 300)
    // Cheapest 88+ purchase that isn't the Best Value winner.
    assert.equal(a.bargain.player, cheapStar)
    assert.ok(cheapStar.rating >= BARGAIN_MIN_RATING)
    const deepest = [...ranked].sort((x, y) => y.ranking.injuryCover - x.ranking.injuryCover)[0]
    assert.equal(a.deepestSquad.team.teamId, deepest.teamId)
})

test('Best Value and Bargain go to different players', () => {
    const only88 = buy(90, 100)
    const second = buy(88, 400)
    const a = computeAwards([team('A', [...xi(80, 100), only88, second], 1600)])
    assert.equal(a.bestValue.player, only88)
    assert.equal(a.bargain.player, second)
})

test('awards with no eligible candidate are left out', () => {
    const a = computeAwards([team('A', xi(80, 100), 1100), team('B', xi(75, 50), 550)])
    assert.equal(a.bargain, undefined) // nobody rated 88+
    assert.ok(a.champion && a.biggestSplurge && a.highestRated && a.deepestSquad)
    assert.deepEqual(computeAwards([]), {})
    // A price of 0 (sale missing from history) is never treated as a purchase.
    const b = computeAwards([team('A', [buy(95, 0), ...xi(80, 100)], 1100)])
    assert.notEqual(b.highestRated.player.rating, 95)
})

test('ties break deterministically', () => {
    const x = buy(90, 500)
    const y = buy(90, 500) // same rating and price, higher slNo
    const a = computeAwards([team('A', [...xi(80, 100), y], 1600), team('B', [...xi(80, 100), x], 1600)])
    assert.equal(a.biggestSplurge.player, x)
    assert.equal(a.highestRated.player, x)
    // Identical squads → joint champions, both listed.
    const twins = computeAwards([team('A', xi(80, 100), 1100), team('B', xi(80, 100), 1100)])
    assert.deepEqual(twins.champion.teams.map((t) => t.teamId).sort(), ['A', 'B'])
})

test('same result → same awards, whatever the input order; ranking is untouched', () => {
    for (const seed of [1, 2, 3, 4, 5]) {
        const teams = randomRoom(seed)
        const before = JSON.stringify(rankSquads(teams).map((t) => [t.teamId, t.ranking]))
        const summary = (aw) => JSON.stringify(Object.fromEntries(Object.entries(aw).map(([k, v]) =>
            [k, v.teams ? v.teams.map((t) => t.teamId) : v.team && !v.player ? v.team.teamId : [v.team.teamId, v.player.slNo, v.price]])))
        const one = summary(computeAwards(teams))
        assert.equal(summary(computeAwards([...teams].reverse())), one)
        assert.equal(summary(computeAwards(rankSquads(teams))), one)
        assert.equal(JSON.stringify(rankSquads(teams).map((t) => [t.teamId, t.ranking])), before)
    }
})
