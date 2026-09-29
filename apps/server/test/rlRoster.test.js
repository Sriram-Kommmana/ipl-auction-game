// Production RL roster: registry loading, integrity fallbacks, the market
// window built from the room history, and complete auctions through the same
// seat runtime botManager uses. No Redis / MongoDB needed.
//
//   node --test test/rlRoster.test.js      (from apps/server)

import test from 'node:test'
import assert from 'node:assert/strict'
import { copyFileSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { DEFAULT_RULES, RL_PERSONAS, fairValue, planBid, selectBestXI } from '@ipl-auction/shared'
import { AuctionSim, createRng } from '@ipl-auction/shared/sim'
import { parsePlayersCsv } from '@ipl-auction/shared/players-csv'
import { ACT_SPEC_HASH, OBS_SPEC_HASH, RECENT_WINDOW, auditAuction, sampleEpisode } from '@ipl-auction/shared/rl'
import { agentCap } from '@ipl-auction/shared/sim'
import { buildExtras, createSeatRuntime, loadRoster } from '../src/bots/rlRoster.js'

const MODELS = fileURLToPath(new URL('../src/bots/models/', import.meta.url))
const players = parsePlayersCsv(readFileSync(new URL('../src/db/players.csv', import.meta.url), 'utf8'))
const byId = new Map(players.map((p) => [p.slNo, p]))
const roster = loadRoster()

// The room history list as timerManager / onSkip write it, from a simulator run.
const serverHistory = (sim) => sim.history.map((h) => JSON.stringify(h.winner === null
    ? { iplPlayerId: String(h.slNo), soldTo: null, soldFor: null, status: 'unsold' }
    : { iplPlayerId: String(h.slNo), soldTo: sim.teams[h.winner].teamId, soldFor: h.price, status: 'sold' }))

// What the training environment passes (RlEpisode.extras).
const envExtras = (sim) => ({
    poolSize: sim.mainLength,
    recent: sim.history.slice(-RECENT_WINDOW).map((h) => ({
        sold: h.winner !== null, price: h.price, fairValue: fairValue(sim.players.get(h.slNo), sim.rules.pursePerTeam),
        winnerTeamId: h.winner === null ? null : sim.teams[h.winner].teamId
    }))
})

test('R1. every RL personality has a verified model; specs match the frozen hashes', () => {
    assert.deepEqual(roster.errors, [])
    for (const id of Object.keys(RL_PERSONAS)) {
        const m = roster.byPersona[id]
        assert.ok(m, `${id} has no model`)
        assert.equal(m.policy.obsSpec.hash, OBS_SPEC_HASH)
        assert.equal(m.policy.actSpec.hash, ACT_SPEC_HASH)
        assert.equal(OBS_SPEC_HASH, '629b25783f833af7')
        assert.equal(ACT_SPEC_HASH, '5f72f510c48b1f46')
    }
})

test('R2. corrupted / missing / unassigned models → that seat falls back to its rule persona; the rest still load', () => {
    const dir = mkdtempSync(join(tmpdir(), 'rl-roster-'))
    try {
        const reg = JSON.parse(readFileSync(join(MODELS, 'registry.json'), 'utf8'))
        for (const m of Object.values(reg.models)) copyFileSync(join(MODELS, m.file), join(dir, m.file))
        // tamper one byte of the aggressor's model, delete the adaptive model's file, unassign one persona
        const tampered = reg.models[reg.roster.aggressor].file
        const bytes = readFileSync(join(dir, tampered))
        bytes[bytes.length - 3] = bytes[bytes.length - 3] === 0x31 ? 0x32 : 0x31
        writeFileSync(join(dir, tampered), bytes)
        rmSync(join(dir, reg.models[reg.roster.adaptive].file))
        delete reg.roster.overseasSpecialist
        writeFileSync(join(dir, 'registry.json'), JSON.stringify(reg))

        const r = loadRoster(dir)
        assert.equal(r.byPersona.aggressor, null)
        assert.equal(r.byPersona.adaptive, null)
        assert.equal(r.byPersona.overseasSpecialist, null)
        assert.ok(r.byPersona.paceFirst && r.byPersona.battingFirst)
        assert.ok(r.errors.some((e) => e.includes('sha256')))
        assert.equal(r.errors.length, 3)

        // a fallback seat plays its frozen rule persona
        const sim = new AuctionSim({ players, teamCount: 10, rng: createRng(7) })
        const ctx = sim.contextFor(0)
        const d = createSeatRuntime(r, 'aggressor').decide(ctx, { poolSize: sim.mainLength, recent: [] }, createRng(1))
        assert.equal(d.source, 'fallback')
        assert.equal(d.cap, agentCap({ kind: 'rule', persona: RL_PERSONAS.aggressor.fallback }, ctx, createRng(1)))
    } finally {
        rmSync(dir, { recursive: true, force: true })
    }
})

test('R3. no registry at all → every RL seat is a rule fallback (the game still runs)', () => {
    const dir = mkdtempSync(join(tmpdir(), 'rl-roster-empty-'))
    try {
        const r = loadRoster(dir)
        assert.ok(Object.values(r.byPersona).every((m) => m === null))
        assert.equal(createSeatRuntime(r, 'adaptive').state.disabled, true)
    } finally {
        rmSync(dir, { recursive: true, force: true })
    }
})

test('R4. the market window built from the room history equals the training environment\'s', () => {
    const sim = new AuctionSim({ players, teamCount: 10, rules: { ...DEFAULT_RULES, pursePerTeam: 15300 }, rng: createRng(11) })
    const rng = createRng(12)
    for (let lot = 0; lot < 60; lot++) {
        const caps = sim.teams.map((_, i) => agentCap({ kind: 'rule', persona: ['moneyball', 'starChaser', 'balancedBuilder', 'opportunist'][i % 4] }, sim.contextFor(i), rng))
        sim.resolveLot(caps)
        const got = buildExtras({ history: serverHistory(sim).slice(-RECENT_WINDOW), poolSize: sim.mainLength, players: byId, pursePerTeam: 15300 })
        assert.deepEqual(got, envExtras(sim))
    }
})

// A complete Full Pool auction (main round + re-auction) with the production
// roster in the five RL seats, the four rule bots and a human proxy — the
// way botManager plays: server-format history → buildExtras → runtime.decide.
const playProductRoom = (seed) => {
    const entry = sampleEpisode(seed)
    const rules = { ...DEFAULT_RULES, pursePerTeam: entry.purse }
    const rng = createRng(seed)
    const sim = new AuctionSim({ players, teamCount: 10, rules, rng })
    const seats = entry.seats.map((s) => {
        if (s.type === 'rule') return { agent: { kind: 'rule', persona: s.persona } }
        if (s.type === 'human') return s.proxy === 'passive' ? { passive: true } : { agent: { kind: 'rule', persona: s.persona, noise: s.noise } }
        return { rl: createSeatRuntime(roster, s.rlSeat), sources: {}, capOver: 0 }
    })
    while (!sim.done) {
        const extras = buildExtras({ history: serverHistory(sim).slice(-RECENT_WINDOW), poolSize: sim.mainLength, players: byId, pursePerTeam: rules.pursePerTeam })
        const caps = seats.map((s, i) => {
            const ctx = sim.contextFor(i)
            if (s.passive) return 0
            if (s.agent) return agentCap(s.agent, ctx, rng)
            const d = s.rl.decide(ctx, extras, rng)
            s.sources[d.source] = (s.sources[d.source] || 0) + 1
            const plan = planBid(ctx)
            if (d.cap > 0 && (!plan.allowed || d.cap > plan.budget.maxSafeBid)) s.capOver++
            return d.cap
        })
        sim.resolveLot(caps)
    }
    return { sim, seats }
}

test('R5. complete auctions with the production roster: legal, every RL squad completes a legal XI, no fallback, caps ≤ maxSafeBid', () => {
    for (const seed of [91001, 91002, 91003, 91004]) {
        const { sim, seats } = playProductRoom(seed)
        assert.deepEqual(auditAuction(sim), [], `seed ${seed}`)
        assert.ok(sim.history.some((h) => h.phase === 'reauction'), `seed ${seed}: no re-auction`)
        seats.forEach((s, i) => {
            if (!s.rl) return
            assert.equal(s.rl.state.disabled, false, `seed ${seed} seat ${i}: ${s.rl.state.disabledReason}`)
            assert.equal(s.sources.fallback ?? 0, 0)
            assert.ok((s.sources.rl ?? 0) > 50)
            assert.equal(s.capOver, 0)
            assert.equal(selectBestXI(sim.teams[i].squad).emptySlots, 0, `seed ${seed} seat ${i}: incomplete XI`)
        })
    }
})
