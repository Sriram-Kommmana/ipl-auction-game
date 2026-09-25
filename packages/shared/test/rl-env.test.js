// Phase 2B — IplAuctionEnv-v2 core: determinism (G), shape/masks, reward
// parity (K), episode structure, and proof that the environment is exactly
// the frozen AuctionSim + frozen agents (no second implementation).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { AuctionSim, agentCap, createRng, playAuction } from '../src/sim.js'
import { DEFAULT_RULES } from '../src/rules.js'
import { RULE_BOT_IDS } from '../src/ruleBots.js'
import { planBid } from '../src/planning.js'
import { BASE, OBS_SIZE, PASS, RlEpisode, episodeReturn, fnv1a64, rlActionMask, sampleEpisode } from '../src/rl/index.js'
import { players, randomPolicy } from './rlHelpers.js'

// Deterministic learner: a legal action picked from the learner stream.
const playEpisode = (entry, { snapshots = [], tremble = 0.01, pick = 'random' } = {}) => {
    const ep = new RlEpisode({ players, entry, snapshots, tremble })
    const rng = ep.learnerRng
    const transcript = []
    let step = ep.reset()
    transcript.push({ obs: step.obs, mask: step.mask, info: step.info })
    for (;;) {
        const legal = step.mask.flatMap((ok, a) => (ok ? [a] : []))
        const action = pick === 'pass' ? (step.mask[PASS] ? PASS : BASE) : legal[Math.floor(rng() * legal.length)]
        step = ep.step(action)
        transcript.push({ action, reward: step.reward, done: step.done, obs: step.obs, mask: step.mask, info: step.info })
        if (step.done) break
    }
    return { ep, transcript, summary: step.info.episode }
}

const entries = [1_000_010, 1_000_011, 1_000_012].map((s) => sampleEpisode(s))
const runs = entries.map((e) => playEpisode(e))

test('G1. same seed and actions → byte-identical episode transcript', () => {
    const again = playEpisode(entries[0])
    assert.equal(fnv1a64(JSON.stringify(again.transcript)), fnv1a64(JSON.stringify(runs[0].transcript)))
    assert.notEqual(fnv1a64(JSON.stringify(runs[1].transcript)), fnv1a64(JSON.stringify(runs[0].transcript)))
})

test('G2. every step: observation (80,), mask of 20 with at least one legal bid; masks respected', () => {
    for (const { transcript } of runs) {
        for (const t of transcript.filter((x) => !x.done)) {
            assert.equal(t.obs.length, OBS_SIZE)
            assert.equal(t.mask.length, 20)
            assert.ok(t.mask.some((ok, a) => ok && a !== PASS), 'a decision step always offers a bid')
            assert.equal(t.info.shieldActive, t.mask[PASS] === 0)
        }
    }
})

test('K4. reward parity: the rewards sum to final XI total / 110 (+ −2 if the XI has an empty slot); γ = 1', () => {
    for (const { ep, transcript, summary } of runs) {
        const sum = transcript.reduce((s, t) => s + (t.reward ?? 0), 0)
        const squad = ep.sim.teams[ep.learner].squad
        assert.ok(Math.abs(sum - episodeReturn(squad)) < 1e-9, `${sum} vs ${episodeReturn(squad)}`)
        assert.ok(Math.abs(sum - summary.return) < 1e-9)
        assert.ok(Math.abs(summary.xiTotal / 110 + (summary.emptySlots > 0 ? -2 : 0) - sum) < 1e-9)
        const last = transcript.at(-1)
        assert.equal(last.info.terminalReward, summary.emptySlots > 0 ? -2 : 0)
    }
})

test('G3. one episode = the full auction: every player once in the main round, then the re-auction; terminal state is final', () => {
    for (const { ep, transcript, summary } of runs) {
        const main = ep.sim.history.filter((h) => h.phase === 'main')
        assert.equal(main.length, ep.sim.mainLength)
        assert.ok(summary.reauctionLots > 0, 're-auction happened')
        assert.equal(summary.lots, ep.sim.history.length)
        assert.ok(ep.sim.done)
        const last = transcript.at(-1)
        assert.equal(last.obs, null)
        assert.equal(last.mask, null)
        assert.throws(() => ep.step(0), /over/)
    }
})

test('G4. the environment is exactly AuctionSim + the frozen agents (replayed independently, same random stream)', () => {
    for (const entry of entries) {
        const { ep } = playEpisode(entry, { pick: 'pass' })
        // Independent replay: bare AuctionSim, frozen rule agents via agentCap in seat order.
        const rng = createRng(entry.seed)
        const sim = new AuctionSim({ players, teamCount: 10, rules: { ...DEFAULT_RULES, pursePerTeam: entry.purse }, rng })
        while (!sim.done) {
            const ctxs = entry.seats.map((_, i) => sim.contextFor(i))
            const caps = entry.seats.map((seat, i) => {
                if (i === entry.learnerSeat) {
                    const m = rlActionMask(ctxs[i], planBid(ctxs[i]))
                    return m.shieldActive ? m.caps[BASE] : 0
                }
                if (seat.type === 'human' && seat.proxy === 'passive') return 0
                const agent = seat.type === 'human' ? { kind: 'rule', persona: seat.persona, noise: seat.noise } : { kind: 'rule', persona: seat.persona }
                return agentCap(agent, ctxs[i], rng)
            })
            sim.resolveLot(caps)
        }
        assert.deepEqual(ep.sim.history, sim.history, `seed ${entry.seed}`)
        assert.deepEqual(ep.sim.teams.map((t) => [t.purseLeft, t.playerCount]), sim.teams.map((t) => [t.purseLeft, t.playerCount]))
    }
})

test('G5. frozen rule-bot behaviour is byte-identical (fixed fingerprint of a full rule-bot auction)', () => {
    const agents = [...RULE_BOT_IDS, ...RULE_BOT_IDS, 'moneyball', 'starChaser'].map((persona) => ({ kind: 'rule', persona }))
    const { capsLog } = playAuction({ players, agents, seed: 4242 })
    // Recorded from the frozen Phase 1 code; any change to rule-bot decisions changes it.
    assert.equal(fnv1a64(JSON.stringify(capsLog)), FROZEN_FINGERPRINT)
})
const FROZEN_FINGERPRINT = 'ffe6a9da0430b279'

test('G6. Stage B: frozen random-init snapshot opponents play through the runtime, deterministically', () => {
    const snapshots = [randomPolicy('ppo', 31), randomPolicy('qrdqn', 32)]
    const entry = sampleEpisode(1_000_020, { league: { snapshotShare: 1, poolSize: 2 } })
    assert.equal(entry.seats.filter((s) => s.type === 'rlSnapshot').length, 4)
    const a = playEpisode(entry, { snapshots })
    const b = playEpisode(entry, { snapshots })
    assert.equal(fnv1a64(JSON.stringify(a.transcript)), fnv1a64(JSON.stringify(b.transcript)))
    for (const [i, seat] of a.ep.seats.entries()) if (seat?.kind === 'rl') assert.equal(seat.runtime.state.disabled, false, `snapshot seat ${i} healthy`)
    assert.throws(() => new RlEpisode({ players, entry, snapshots: [] }), /snapshot/)
})

test('G7. no player bought twice; purse, squad and overseas limits respected for every team', () => {
    for (const { ep } of runs) {
        const owners = new Set()
        for (const t of ep.sim.teams) {
            for (const p of t.squad) {
                assert.ok(!owners.has(p.slNo))
                owners.add(p.slNo)
            }
            assert.ok(t.purseLeft >= 0 && t.playerCount <= 25 && t.overseasCount <= 8)
        }
    }
})
