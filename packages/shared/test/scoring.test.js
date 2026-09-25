import { test } from 'node:test'
import assert from 'node:assert/strict'
import { selectBestXI, teamStrength, xiGain, XI_RULES } from '../src/scoring.js'
import { createRng } from '../src/sim.js'

let nextSlNo = 1
const player = (role, rating, nationality = 'Indian') => ({ slNo: nextSlNo++, role, rating, nationality, stats: {} })

test('an empty squad scores 0', () => {
    assert.equal(teamStrength([]), 0)
    assert.equal(selectBestXI([]).emptySlots, 11)
})

test('one superstar no longer wins — the old average-rating exploit is gone', () => {
    // Old formula: average of one player = 96. New: 96 / 11 slots.
    assert.equal(teamStrength([player('BATSMAN', 96)]), 8.7)
})

test('a complete balanced XI scores its average rating', () => {
    const squad = [
        player('WICKET KEEPER', 80),
        ...Array.from({ length: 5 }, () => player('BATSMAN', 80)),
        ...Array.from({ length: 5 }, () => player('BOWLER', 80))
    ]
    const result = selectBestXI(squad)
    assert.equal(result.players.length, 11)
    assert.equal(result.emptySlots, 0)
    assert.equal(result.strength, 80)
})

test('no more than 4 overseas players make the XI', () => {
    const squad = [
        ...Array.from({ length: 6 }, () => player('BOWLER', 95, 'Overseas')),
        player('BOWLER', 70), // Indian bowling option — needed once only 4 overseas can play
        player('WICKET KEEPER', 70),
        ...Array.from({ length: 6 }, () => player('BATSMAN', 70))
    ]
    const { players } = selectBestXI(squad)
    assert.equal(players.filter((p) => p.nationality === 'Overseas').length, 4)
    assert.equal(players.length, 11)
})

test('overseas cap can leave a bowling slot empty when there is no Indian cover', () => {
    const squad = [
        ...Array.from({ length: 6 }, () => player('BOWLER', 95, 'Overseas')),
        player('WICKET KEEPER', 70),
        ...Array.from({ length: 6 }, () => player('BATSMAN', 70))
    ]
    const result = selectBestXI(squad)
    assert.equal(result.players.length, 10) // only 4 bowling options can play
    assert.equal(result.emptySlots, 1)
})

test('a missing keeper leaves a slot empty rather than being ignored', () => {
    const squad = [
        ...Array.from({ length: 6 }, () => player('BATSMAN', 85)),
        ...Array.from({ length: 6 }, () => player('BOWLER', 85))
    ]
    const result = selectBestXI(squad)
    assert.equal(result.players.length, 10)
    assert.equal(result.emptySlots, 1)
})

test('xiGain is zero for a player who would not make the XI', () => {
    const squad = [
        player('WICKET KEEPER', 90),
        ...Array.from({ length: 5 }, () => player('BATSMAN', 90)),
        ...Array.from({ length: 5 }, () => player('BOWLER', 90))
    ]
    assert.equal(xiGain(squad, player('BATSMAN', 70)), 0)
    assert.ok(xiGain(squad, player('BATSMAN', 95)) > 0)
})

// Exhaustive check: for small random squads, enumerate every legal XI and
// confirm the dynamic program finds the true maximum.
const bruteForceTotal = (squad) => {
    let best = 0
    const n = squad.length
    for (let mask = 0; mask < 1 << n; mask++) {
        let count = 0, wk = 0, bowl = 0, os = 0, sum = 0
        for (let i = 0; i < n; i++) {
            if (!(mask & (1 << i))) continue
            const p = squad[i]
            count++
            sum += p.rating
            if (p.role === 'WICKET KEEPER') wk++
            if (p.role === 'BOWLER' || p.role === 'ALL ROUNDER') bowl++
            if (p.nationality === 'Overseas') os++
        }
        if (count > XI_RULES.size || os > XI_RULES.maxOverseas) continue
        const empty = Math.max(0, XI_RULES.minKeepers - wk) + Math.max(0, XI_RULES.minBowlingOptions - bowl)
        if (count + empty > XI_RULES.size) continue
        best = Math.max(best, sum)
    }
    return best
}

test('the XI optimiser matches brute force on 300 random squads', () => {
    const rng = createRng(42)
    const roles = ['BATSMAN', 'BOWLER', 'ALL ROUNDER', 'WICKET KEEPER']
    for (let trial = 0; trial < 300; trial++) {
        const size = 1 + Math.floor(rng() * 14)
        const squad = Array.from({ length: size }, () =>
            player(roles[Math.floor(rng() * 4)], 67 + Math.floor(rng() * 30), rng() < 0.45 ? 'Overseas' : 'Indian')
        )
        const { total, players } = selectBestXI(squad)
        assert.equal(total, bruteForceTotal(squad), `trial ${trial}`)
        assert.equal(players.reduce((s, p) => s + p.rating, 0), total, `reconstructed XI must add up (trial ${trial})`)
    }
})
