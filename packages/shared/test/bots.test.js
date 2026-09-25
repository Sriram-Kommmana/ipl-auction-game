import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { parsePlayersCsv } from '../src/playersCsv.js'
import { fairValue } from '../src/valuation.js'
import { buildAuctionPool } from '../src/pool.js'
import {
    ACTION_COUNT, OBSERVATION_FEATURES, OBSERVATION_SIZE, actionMask, buildObservation, capForAction
} from '../src/observation.js'
import { ruleBotCap, RULE_BOT_IDS } from '../src/ruleBots.js'
import { forward, sampleAction, validateModel } from '../src/mlp.js'
import { createRng, playAuction } from '../src/sim.js'
import { finalStanding } from '../src/rewards.js'

const players = parsePlayersCsv(
    readFileSync(fileURLToPath(new URL('../../../apps/server/src/db/players.csv', import.meta.url)), 'utf8')
)
const bySlNo = new Map(players.map((p) => [p.slNo, p]))
const RULES = { pursePerTeam: 12500, maxPlayers: 25, maxOverseas: 8 }

const emptyTeam = (teamId) => ({ teamId, purseLeft: 12500, purseSpent: 0, playerCount: 0, overseasCount: 0, squad: [] })
const ctxFor = (lot, self = emptyTeam('A')) => ({
    rules: RULES, lot, self, rivals: [emptyTeam('B'), emptyTeam('C')], upcoming: players.slice(0, 50), progress: 0.2
})

test('the CSV loads all 323 players', () => {
    assert.equal(players.length, 323)
})

test('fair value rises with rating, never drops below base price, favours all-rounders', () => {
    const at = (rating, role = 'BATSMAN') => fairValue({ rating, role, basePrice: 20 })
    assert.ok(at(96) > at(90) && at(90) > at(80) && at(80) > at(70))
    assert.equal(fairValue({ rating: 67, role: 'BOWLER', basePrice: 50 }), 50)
    assert.ok(at(85, 'ALL ROUNDER') > at(85))
})

test('the pool auctions every player once, in set order', () => {
    const pool = buildAuctionPool(players, { rng: createRng(7) })
    assert.equal(pool.length, players.length)
    assert.equal(new Set(pool).size, players.length)
    const sets = pool.map((slNo) => bySlNo.get(slNo).setNo)
    assert.deepEqual(sets, [...sets].sort((a, b) => a - b))
})

test('observation has one value per named feature and the mask always allows pass', () => {
    const ctx = ctxFor(players[0])
    const obs = buildObservation(ctx)
    assert.equal(obs.length, OBSERVATION_SIZE)
    assert.equal(OBSERVATION_FEATURES.length, OBSERVATION_SIZE)
    assert.ok(obs.every(Number.isFinite))
    const mask = actionMask(ctx)
    assert.equal(mask.length, ACTION_COUNT)
    assert.equal(mask[0], 1)
})

test('a full squad can only pass', () => {
    const full = { ...emptyTeam('A'), playerCount: 25 }
    const mask = actionMask(ctxFor(players[0], full))
    assert.deepEqual(mask, [1, 0, 0, 0, 0, 0, 0, 0])
    for (const id of RULE_BOT_IDS) assert.equal(ruleBotCap(id, ctxFor(players[0], full), createRng(1)), 0)
})

test('caps never exceed the bot spend limit', () => {
    const poor = { ...emptyTeam('A'), purseLeft: 300, playerCount: 5 } // limit = 300 - 5×20 = 200
    const star = players.find((p) => p.rating >= 94)
    for (let a = 0; a < ACTION_COUNT; a++) assert.ok(capForAction(ctxFor(star, poor), a) <= 200)
    for (const id of RULE_BOT_IDS) assert.ok(ruleBotCap(id, ctxFor(star, poor), createRng(3)) <= 200)
})

test('10 rule bots play a whole auction without breaking any rule', () => {
    const agents = Array.from({ length: 10 }, (_, i) => ({ kind: 'rule', persona: RULE_BOT_IDS[i % 4] }))
    const { sim, capsLog } = playAuction({ players, agents, rules: RULES, seed: 11 })

    assert.ok(sim.done)
    for (const team of sim.teams) {
        assert.ok(team.purseLeft >= 0, 'purse never negative')
        assert.ok(team.playerCount <= 25, 'squad cap respected')
        assert.ok(team.overseasCount <= 8, 'overseas cap respected')
        assert.equal(team.squad.length, team.playerCount)
    }
    for (const lot of capsLog) {
        if (lot.winner === null) continue
        assert.ok(lot.price <= lot.caps[lot.winner], 'winner never pays above its cap')
        assert.ok(lot.price >= bySlNo.get(lot.slNo).basePrice)
    }
    const { strengths } = finalStanding(sim.teams)
    assert.ok(strengths.every((s) => s > 50), `every rule bot fields a real XI (got ${strengths.join(', ')})`)
})

test('the same seed replays the same auction', () => {
    const agents = Array.from({ length: 10 }, (_, i) => ({ kind: 'rule', persona: RULE_BOT_IDS[i % 4] }))
    const a = playAuction({ players, agents, rules: RULES, seed: 5 })
    const b = playAuction({ players, agents, rules: RULES, seed: 5 })
    assert.deepEqual(a.capsLog, b.capsLog)
})

test('MLP forward pass matches a hand calculation and masking is respected', () => {
    const W1 = Array.from({ length: 2 }, (_, r) => Array.from({ length: OBSERVATION_SIZE }, (_, c) => (c === r ? 1 : 0)))
    const W2 = Array.from({ length: ACTION_COUNT }, (_, r) => [r, -r])
    const model = validateModel({
        format: 'mlp-v1', observationSize: OBSERVATION_SIZE, actionCount: ACTION_COUNT, hiddenActivation: 'tanh',
        features: [...OBSERVATION_FEATURES],
        layers: [{ weight: W1, bias: [0, 0] }, { weight: W2, bias: new Array(ACTION_COUNT).fill(0) }]
    })
    const input = new Array(OBSERVATION_SIZE).fill(0)
    input[0] = 0.5
    input[1] = -0.25
    const h = [Math.tanh(0.5), Math.tanh(-0.25)]
    const logits = forward(model, input)
    logits.forEach((l, r) => assert.ok(Math.abs(l - (r * h[0] - r * h[1])) < 1e-12))

    const mask = [1, 0, 1, 0, 0, 0, 0, 0]
    const rng = createRng(9)
    for (let i = 0; i < 200; i++) assert.ok([0, 2].includes(sampleAction(logits, mask, { rng })))
})
