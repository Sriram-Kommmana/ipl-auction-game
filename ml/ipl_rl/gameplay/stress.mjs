// Multi-room inference stress for the production RL runtime (one Node
// process, like the game server): K complete auctions advanced lot by lot in
// round-robin, every RL seat on createRlSeat with the REAL clock, so the
// 20 ms room guard is live. Reports latency percentiles, the first (cold)
// decision per model, and every guard trip.
//
//   node stress.mjs --roster roster.json --rooms 20 [--warmup 1]

import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { performance } from 'node:perf_hooks'

const CP = new URL('../crossplay/', import.meta.url)
const { RL, ROOT, loadExports, loadPlayers, sha256 } = await import(new URL('common.mjs', CP))
const at = (rel) => fileURLToPath(new URL(rel, ROOT))
const { AuctionSim, agentCap, createRng } = await import(new URL('packages/shared/src/sim.js', ROOT))
const { DEFAULT_RULES } = await import(new URL('packages/shared/src/rules.js', ROOT))
const { fairValue } = await import(new URL('packages/shared/src/valuation.js', ROOT))
const { RL_PERSONAS } = await import(new URL('packages/shared/src/personas.js', ROOT))
const { sampleEpisode, createRlSeat, RECENT_WINDOW } = RL

const arg = (n, d) => (process.argv.includes(`--${n}`) ? process.argv[process.argv.indexOf(`--${n}`) + 1] : d)
const K = Number(arg('rooms', 20))
const WARMUP = Number(arg('warmup', 0))
const roster = JSON.parse(readFileSync(arg('roster'), 'utf8')).rl

const stageA = loadExports()
const repro = JSON.parse(readFileSync(at('ml/reports/phase2f/reproducibility.json'), 'utf8'))
const policyOf = (key) => {
    const [stage, algo, s] = key.split(':')
    if (stage === 'A') return stageA[`${algo}:${s}`].policy
    const v = repro.checkpointPolicies[`${algo}:${s}@c500`]
    const bytes = readFileSync(at(`ml/${v.policy}`))
    if (sha256(bytes) !== v.sha256) throw new Error(`${key}: sha256 mismatch`)
    return RL.loadPolicy(bytes.toString('utf8')).policy
}
const policies = Object.fromEntries(Object.values(roster).map((k) => [k, policyOf(k)]))
const players = loadPlayers()

const rooms = Array.from({ length: K }, (_, r) => {
    const entry = sampleEpisode(97000 + r)
    const rules = { ...DEFAULT_RULES, pursePerTeam: entry.purse }
    const rng = createRng(entry.seed)
    const sim = new AuctionSim({ players, teamCount: 10, rules, rng })
    const seats = entry.seats.map((s) => {
        if (s.type === 'rule') return { agent: { kind: 'rule', persona: s.persona } }
        if (s.type === 'human') return s.proxy === 'passive' ? { passive: true } : { agent: { kind: 'rule', persona: s.persona, noise: s.noise } }
        return { rl: createRlSeat({ policy: policies[roster[s.rlSeat]], fallbackPersona: RL_PERSONAS[s.rlSeat].fallback, completionGuard: true }), key: roster[s.rlSeat] }
    })
    return { sim, seats, rng, P: rules.pursePerTeam }
})

// Optional warm-up (what the server can do at startup): a few decisions per model on a throwaway room.
if (WARMUP) {
    const wr = rooms[0]
    const ctx = wr.sim.contextFor(0)
    const extras = { poolSize: wr.sim.mainLength, recent: [] }
    for (const [key, policy] of Object.entries(policies)) {
        const seat = createRlSeat({ policy, fallbackPersona: 'moneyball', completionGuard: true })
        for (let i = 0; i < 30; i++) seat.decide(ctx, extras)
    }
}

const lat = []
const first = {}
const trips = []
let active = K
const t0 = performance.now()
while (active > 0) {
    active = 0
    for (const room of rooms) {
        const { sim, seats, rng, P } = room
        if (sim.done) continue
        active++
        const extras = {
            poolSize: sim.mainLength,
            recent: sim.history.slice(-RECENT_WINDOW).map((h) => ({ sold: h.winner !== null, price: h.price, fairValue: fairValue(sim.players.get(h.slNo), P), winnerTeamId: h.winner === null ? null : sim.teams[h.winner].teamId }))
        }
        const caps = seats.map((s, i) => {
            const ctx = sim.contextFor(i)
            if (s.passive) return 0
            if (s.agent) return agentCap(s.agent, ctx, rng)
            const wasDisabled = s.rl.state.disabled
            const a = performance.now()
            const d = s.rl.decide(ctx, extras, rng)
            const ms = performance.now() - a
            if (d.reason !== 'no legal bid' && !wasDisabled) {
                lat.push(ms)
                if (first[s.key] === undefined) first[s.key] = ms
            }
            if (!wasDisabled && s.rl.state.disabled) trips.push({ key: s.key, reason: s.rl.state.disabledReason, lot: sim.history.length })
            return d.cap
        })
        sim.resolveLot(caps)
    }
}
const wall = performance.now() - t0
lat.sort((a, b) => a - b)
const q = (p) => lat[Math.min(lat.length - 1, Math.floor(p * lat.length))]
const out = {
    rooms: K, warmup: Boolean(WARMUP), decisions: lat.length, wallSeconds: +(wall / 1000).toFixed(1),
    ms: { mean: +(lat.reduce((a, b) => a + b, 0) / lat.length).toFixed(3), p50: +q(0.5).toFixed(3), p95: +q(0.95).toFixed(3), p99: +q(0.99).toFixed(3), p999: +q(0.999).toFixed(3), max: +lat.at(-1).toFixed(2) },
    over5ms: lat.filter((x) => x > 5).length, over20ms: lat.filter((x) => x > 20).length,
    firstDecisionMs: Object.fromEntries(Object.entries(first).map(([k, v]) => [k, +v.toFixed(2)])),
    guardTrips: trips
}
console.log(JSON.stringify(out, null, 1))
