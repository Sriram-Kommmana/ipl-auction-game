import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { parsePlayersCsv } from '../src/playersCsv.js'
import { selectBestXI, teamStrength } from '../src/scoring.js'
import { battingValue, bowlingValue, evaluateSquad, rankSquads } from '../src/squadRanking.js'

const players = parsePlayersCsv(readFileSync(fileURLToPath(new URL('../../../apps/server/src/db/players.csv', import.meta.url)), 'utf8'))

let nextSl = 9000
const mk = (role, rating, { overseas = false, bat = 50, pwr = 50, bwl = 50 } = {}) => ({
    slNo: nextSl++, playerName: `P${nextSl}`, role, rating,
    nationality: overseas ? 'Overseas' : 'Indian',
    stats: { bat, pwr, bwl, tec: 80, clt: 80 }
})
// A legal XI: 1 keeper, 5 bowlers, 5 batters, all Indian.
const baseXI = (r = 80) => [
    mk('WICKET KEEPER', r),
    ...Array.from({ length: 5 }, () => mk('BOWLER', r)),
    ...Array.from({ length: 5 }, () => mk('BATSMAN', r))
]

const lcg = (seed) => () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32)
const sample = (rng, n) => {
    const pool = [...players]
    return Array.from({ length: n }, () => pool.splice(Math.floor(rng() * pool.length), 1)[0])
}

test('empty squad scores 0 everywhere', () => {
    const e = evaluateSquad([])
    assert.deepEqual([e.score, e.xiStrength, e.matchday, e.injuryCover, e.balance, e.emptySlots, e.impactPlayer], [0, 0, 0, 0, 0, 11, null])
})

test('XI strength and the XI itself are exactly the existing Best XI (scoring.js)', () => {
    const rng = lcg(7)
    for (let i = 0; i < 200; i++) {
        const squad = sample(rng, 1 + Math.floor(rng() * 25))
        const e = evaluateSquad(squad)
        assert.equal(e.xiStrength, teamStrength(squad))
        assert.deepEqual(e.xi, selectBestXI(squad).players.map((p) => p.slNo))
    }
})

test('Impact Player follows the IPL overseas rule', () => {
    const xi = baseXI(80)
    xi.slice(7, 11).forEach((p) => { p.nationality = 'Overseas' }) // 4 overseas batters start
    const overseasStar = mk('BATSMAN', 95, { overseas: true })
    const indianBench = mk('BATSMAN', 70)
    const e = evaluateSquad([...xi, overseasStar, indianBench])
    // The XI already has 4 overseas, so the overseas star may only replace one
    // of them in the XI; the impact player must be Indian.
    const chosen = [...xi, overseasStar, indianBench].find((p) => p.slNo === e.impactPlayer)
    assert.equal(chosen.nationality, 'Indian')

    // With 3 overseas in the XI an overseas impact player is allowed.
    const xi3 = baseXI(80)
    xi3.slice(8, 11).forEach((p) => { p.nationality = 'Overseas' })
    const e3 = evaluateSquad([...xi3, mk('BOWLER', 79, { overseas: true }), mk('BATSMAN', 60)])
    assert.equal([...xi3].some((p) => p.slNo === e3.impactPlayer), false)
    assert.equal(e3.matchday, Math.round(((80 * 11 + 79) / 12) * 10) / 10)
})

test('injury cover rewards bench depth', () => {
    const thin = baseXI(85)
    const deep = [...baseXI(85), mk('WICKET KEEPER', 84), mk('BOWLER', 84), mk('BATSMAN', 84)]
    const a = evaluateSquad(thin)
    const b = evaluateSquad(deep)
    assert.equal(a.xiStrength, b.xiStrength)
    assert.ok(b.injuryCover > a.injuryCover, `${b.injuryCover} > ${a.injuryCover}`)
    // Without any bench, losing a starter empties a slot: 10 × 85 / 11.
    assert.equal(a.injuryCover, Math.round(((85 * 10) / 11) * 10) / 10)
    assert.equal(rankSquads([{ teamId: 'A', squad: thin }, { teamId: 'B', squad: deep }])[0].teamId, 'B')
})

test('balance = mean of the top-7 batting and top-5 bowling units', () => {
    const squad = [
        mk('WICKET KEEPER', 80, { bat: 90, pwr: 70, bwl: 0 }),
        ...Array.from({ length: 5 }, () => mk('BATSMAN', 80, { bat: 80, pwr: 80, bwl: 10 })),
        ...Array.from({ length: 5 }, () => mk('BOWLER', 80, { bat: 20, pwr: 20, bwl: 90 }))
    ]
    const e = evaluateSquad(squad)
    const batting = (80 + 80 * 5 + 20) / 7 // keeper 80, five batters 80, best bowler 20
    assert.equal(e.batting, Math.round(batting * 10) / 10)
    assert.equal(e.bowling, 90)
    assert.equal(e.balance, Math.round(((batting + 90) / 2) * 10) / 10)
    assert.equal(battingValue(squad[0]), 80)
    assert.equal(bowlingValue(squad[6]), 90)
})

test('one superstar and nothing else never beats a complete squad', () => {
    const star = [mk('BATSMAN', 99, { bat: 99, pwr: 99 })]
    const ranked = rankSquads([{ teamId: 'STAR', squad: star }, { teamId: 'FULL', squad: baseXI(75) }])
    assert.equal(ranked[0].teamId, 'FULL')
})

test('tie-breakers: XI strength, then injury cover, then less purse spent, then a shared rank', () => {
    const same = () => baseXI(80)
    let r = rankSquads([
        { teamId: 'A', squad: same(), purseSpent: 9000 },
        { teamId: 'B', squad: same(), purseSpent: 8000 },
        { teamId: 'C', squad: same(), purseSpent: 9000 }
    ])
    assert.deepEqual(r.map((t) => [t.teamId, t.ranking.rank, t.ranking.tied]), [['B', 1, false], ['A', 2, true], ['C', 2, true]])
    assert.equal(r[1].ranking.decidedBy, 'purseSpent')
    assert.equal(r[2].ranking.decidedBy, null)

    // Same Squad Score at display precision, better XI wins.
    const hi = [...baseXI(80)]
    hi[1].rating = 81 // one bowler a point better → XI +0.1
    r = rankSquads([{ teamId: 'LO', squad: baseXI(80), purseSpent: 0 }, { teamId: 'HI', squad: hi, purseSpent: 99999 }])
    assert.equal(r[0].teamId, 'HI')
    assert.ok(['score', 'xiStrength'].includes(r[1].ranking.decidedBy))
})

test('ranking does not depend on input order and keeps every team field', () => {
    const rng = lcg(11)
    const teams = Array.from({ length: 10 }, (_, i) => ({ teamId: `T${i}`, squad: sample(rng, 15 + (i % 10)), purseSpent: 1000 * i, extra: i }))
    const a = rankSquads(teams).map((t) => [t.teamId, t.ranking.rank])
    const b = rankSquads([...teams].reverse()).map((t) => [t.teamId, t.ranking.rank])
    assert.deepEqual(a, b)
    assert.equal(rankSquads(teams).find((t) => t.teamId === 'T3').extra, 3)
    const ranks = rankSquads(teams).map((t) => t.ranking.rank)
    assert.deepEqual(ranks, [...ranks].sort((x, y) => x - y))
})
